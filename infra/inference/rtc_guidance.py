"""PI05 prefix guidance at sampling time; no checkpoint weights are changed.

The HTTP prefix is in physical absolute joint units. It must pass through the
checkpoint's action preprocessor before entering the padded flow latent space.
Imports stay lazy so transport-only mock mode does not require LeRobot/torch.
"""


def validate_prefix(value, chunk_size):
    import numpy as np

    if not isinstance(value, dict) or set(value) != {'previous_actions', 'overlap'}:
        raise ValueError('rtc requires exactly previous_actions and overlap')
    overlap = value['overlap']
    if type(overlap) is not int or not 1 <= overlap <= chunk_size:
        raise ValueError(f'rtc overlap must be an integer from 1 to {chunk_size}')
    actions = np.asarray(value['previous_actions'], dtype=np.float32)
    if actions.shape != (overlap, 14) or not np.isfinite(actions).all():
        raise ValueError('rtc previous_actions must contain overlap finite 14D absolute actions')
    return actions, overlap


def prepare_prefix(batch, actions, preprocessor, torch, max_action_dim):
    """Normalize observations and prefix together using the saved preprocessing.

    AddBatchDimensionActionStep only batches *1D* actions, so supply the prefix
    already shaped (1, overlap, 14). Padding follows normalization, as in PI05
    training; zero in a padded latent dimension is not physical zero radians.
    """
    batch = dict(batch)
    batch['action'] = torch.from_numpy(actions).unsqueeze(0)
    processed = preprocessor(batch)
    prefix = processed.pop('action', None)
    expected = (1, len(actions), 14)
    if prefix is None or tuple(prefix.shape) != expected or not torch.isfinite(prefix).all():
        raise RuntimeError('Checkpoint preprocessor did not produce a finite normalized RTC prefix')
    if max_action_dim < 14:
        raise RuntimeError('RTC model action dimension is smaller than the physical action dimension')
    prefix = torch.nn.functional.pad(prefix, (0, max_action_dim - 14))
    return processed, prefix


def guided_velocity(processor, x_t, prev_chunk_left_over, inference_delay, time,
                    original_denoise_step_partial, execution_horizon=None):
    """RTC vector-Jacobian guidance with gradients enabled before denoising.

    LeRobot's bundled implementation enables requires_grad after the denoiser
    call, losing its Jacobian. This local wrapper keeps that derivative and
    detaches each completed guidance step to avoid retaining the sampling graph.
    """
    import torch

    if prev_chunk_left_over is None:
        return original_denoise_step_partial(x_t)
    if x_t.ndim != 3 or prev_chunk_left_over.ndim != 3:
        raise RuntimeError('RTC expects batched action chunks')
    if inference_delay != 0:
        raise RuntimeError('This synchronous RTC protocol requires inference_delay=0')
    horizon = execution_horizon if execution_horizon is not None else processor.rtc_config.execution_horizon
    if not 1 <= horizon <= prev_chunk_left_over.shape[1] <= x_t.shape[1]:
        raise RuntimeError('Invalid RTC prefix horizon')
    if prev_chunk_left_over.shape[0] != x_t.shape[0] or prev_chunk_left_over.shape[2] != x_t.shape[2]:
        raise RuntimeError('RTC prefix does not match the model latent dimensions')
    with torch.inference_mode(False), torch.enable_grad():
        latent = x_t.detach().clone().requires_grad_(True)
        previous = prev_chunk_left_over.to(device=latent.device, dtype=latent.dtype)
        previous = torch.nn.functional.pad(previous, (0, 0, 0, latent.shape[1] - previous.shape[1]))
        weights = processor.get_prefix_weights(0, horizon, latent.shape[1]).to(latent).view(1, -1, 1)
        velocity = original_denoise_step_partial(latent)
        endpoint = latent - time * velocity
        error = ((previous - endpoint) * weights).detach()
        # Padded coordinates have no physical target. Do not ask the sampler
        # to match padding, or apply a prefix correction to padded latents.
        error[..., 14:] = 0
        correction = torch.autograd.grad(endpoint, latent, grad_outputs=error)[0]
        correction[..., 14:] = 0
        # Same schedule as LeRobot/Kinetix, expressed without a singularity at
        # tau=0 (the first flow step). Sampling never evaluates time=0.
        tau = 1.0 - float(time)
        maximum = processor.rtc_config.max_guidance_weight
        guidance = maximum if tau <= 0 else min(maximum, (time * time + tau * tau) / (time * tau))
        return (velocity - guidance * correction).detach()


def configure_rtc(policy):
    """Attach the supported PI05 RTC interface after strict checkpoint loading."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    if not callable(getattr(policy, 'init_rtc_processor', None)) or not callable(
            getattr(policy.model, '_rtc_enabled', None)):
        raise RuntimeError('Installed PI05 does not provide its RTC sampling interface')
    if getattr(policy.config, 'compile_model', False):
        raise RuntimeError('RTC guidance requires uncompiled PI05 sampling')

    class PrefixRTCProcessor(RTCProcessor):
        denoise_step = guided_velocity

    config = RTCConfig(enabled=True)
    policy.config.rtc_config = config
    policy.model.config.rtc_config = config
    processor = PrefixRTCProcessor(config)
    policy.rtc_processor = processor
    policy.model.rtc_processor = processor
    # Model weights stay frozen; the only gradient we need is d(endpoint)/d(x_t).
    policy.requires_grad_(False)
    return processor
