#!/usr/bin/env python3
"""Loopback-only pi0.5 chunk inference; robot devices never exist on this host."""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from action_limits import clamp_gripper_actions
from rtc_guidance import configure_rtc, prepare_prefix, validate_prefix

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / 'contract.json').read_text())
NAMES = [name for side in ('left', 'right') for name in
         [*[f'{side}.joint{i}' for i in range(1, 7)], f'{side}.gripper_rad']]


def cuda_memory_fraction(value):
    """Parse a fraction of device memory allowed to PyTorch's allocator."""
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError('CUDA memory fraction must be greater than 0 and at most 1')
    return fraction


def validate_deployment(meta, contract):
    for key in ('fps', 'task', 'dataset_repo', 'dataset_revision'):
        if meta.get(key) != contract[key]:
            raise ValueError(f'Checkpoint deployment mismatch: {key}')
    if meta.get('action_representation') != 'absolute':
        raise ValueError('Only absolute joint actions are supported')
    expected_crops = {f'observation.images.{name}': crop
                      for name, crop in contract['image_crops_xyxy'].items()}
    if meta.get('image_crops') != expected_crops:
        raise ValueError('Checkpoint deployment mismatch: image_crops')
    for key in ('dataset_manifest_sha256', 'dataset_mix'):
        if key in contract and meta.get(key) != contract[key]:
            raise ValueError(f'Checkpoint deployment mismatch: {key}')


def load_policy(checkpoint):
    """Load saved weights strictly; the upstream loader can swallow load errors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.gradient_checkpointing = False
    policy = PI05Policy(config)
    state = load_file(str(Path(checkpoint) / 'model.safetensors'))
    state = policy._fix_pytorch_state_dict_keys(state, config)
    state = {k if k.startswith('model.') else f'model.{k}': v for k, v in state.items()}
    state = policy._prepare_pretrained_state_dict(state)
    # Saved safetensors omit tied aliases. Restore only parameters actually tied
    # by this model, matching the training runtime's strict checkpoint loader.
    aliases = {}
    for name, param in policy.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    for names in aliases.values():
        saved = next((state[n] for n in names if n in state), None)
        if saved is not None:
            for name in names:
                state.setdefault(name, saved)
    policy.load_state_dict(state, strict=True)
    del state
    return policy.to('cuda').eval()


class Runtime:
    def __init__(self, checkpoint, mock=False, allow_training=False, cuda_memory_fraction=None):
        self.checkpoint = checkpoint
        self.mock = mock
        self.allow_training = allow_training
        self.cuda_memory_fraction = cuda_memory_fraction
        self.status = 'mock' if mock else 'waiting_for_training'
        self.error = None
        self.rtc = dict(supported=False, protocol='prefix-v1',
                        error='RTC is unavailable in transport mock mode' if mock else 'Policy is not loaded')
        self.lock = threading.Lock()
        if not mock:
            threading.Thread(target=self.load, daemon=True).start()

    def load(self):
        try:
            # Wait by default; concurrent startup is an explicit operator choice.
            # Resolve the checkpoint once so requests never follow a moving `last`.
            while not self.allow_training and subprocess.run(['pgrep', '-f', '[p]ython.*train_panthera.py'],
                                 stdout=subprocess.DEVNULL).returncode == 0:
                time.sleep(10)
            self.status = 'loading'
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            self.checkpoint = str(Path(self.checkpoint).resolve(strict=True))
            meta = json.loads((Path(self.checkpoint) / 'deployment.json').read_text())
            validate_deployment(meta, CONTRACT)
            if self.cuda_memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(self.cuda_memory_fraction)
            self.policy = load_policy(self.checkpoint)
            if (self.policy.config.action_feature_names != NAMES or
                    self.policy.config.use_relative_actions or
                    set(self.policy.config.image_features) !=
                    {f'observation.images.{x}' for x in CONTRACT['image_crops_xyxy']}):
                raise ValueError('Checkpoint feature contract mismatch')
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, pretrained_path=self.checkpoint)
            try:
                configure_rtc(self.policy)
                self.rtc = dict(supported=True, protocol='prefix-v1')
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                self.rtc = dict(supported=False, protocol='prefix-v1', error=str(exc))
            self.torch = torch
            self.status = 'ready'
        except Exception as exc:
            self.error = str(exc)
            self.status = 'failed'

    def infer(self, payload):
        import numpy as np
        from PIL import Image
        prefix = None
        overlap = 0
        if 'rtc' in payload:
            if not self.rtc['supported']:
                raise RuntimeError(f"RTC unsupported: {self.rtc.get('error', 'unavailable')}")
            prefix, overlap = validate_prefix(payload['rtc'], self.policy.config.chunk_size)
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected 14 finite joint/gripper positions')
        images = {}
        if set(payload['images']) != set(CONTRACT['image_crops_xyxy']):
            raise ValueError('Expected all three named cameras')
        for name, encoded in payload['images'].items():
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
                if im.size != (256, 256) or im.mode != 'RGB':
                    raise ValueError('Expected cropped 256x256 RGB JPEGs')
                images[name] = np.asarray(im).copy()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Inference busy')
        try:
            if self.mock:
                actions = np.tile(state, (50, 1))
            else:
                batch = {'observation.state': self.torch.from_numpy(state),
                         'task': CONTRACT['task']}
                batch.update({f'observation.images.{n}': self.torch.from_numpy(v)
                              .permute(2, 0, 1).float() / 255 for n, v in images.items()})
                # RTC locally enables input gradients for the denoiser VJP.
                # inference_mode disables those gradients even in enable_grad.
                with self.torch.no_grad() if prefix is not None else self.torch.inference_mode():
                    self.policy.reset()
                    kwargs = {}
                    if prefix is not None:
                        batch, normalized_prefix = prepare_prefix(
                            batch, prefix, self.pre, self.torch, self.policy.config.max_action_dim)
                        kwargs = dict(prev_chunk_left_over=normalized_prefix,
                                      inference_delay=0, execution_horizon=overlap)
                    else:
                        batch = self.pre(batch)
                    actions = self.post(self.policy.predict_action_chunk(batch, **kwargs))[0].cpu().numpy()
            if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                raise RuntimeError('Invalid model output')
            actions, corrections = clamp_gripper_actions(actions)
            result = dict(sequence=payload['sequence'], actions=actions.tolist(),
                          checkpoint=self.checkpoint, fps=20, mock=self.mock,
                          limit_corrections=corrections)
            if prefix is not None:
                result['rtc'] = dict(overlap=overlap)
            return result
        finally:
            self.lock.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--mock', action='store_true', help='Transport testing only; echoes state')
    p.add_argument('--allow-training', action='store_true',
                   help='Load alongside training after checking available GPU memory')
    p.add_argument('--cuda-memory-fraction', type=cuda_memory_fraction,
                   help="Limit PyTorch's GPU allocator to this fraction of device memory")
    args = p.parse_args()
    runtime = Runtime(args.checkpoint, args.mock, args.allow_training, args.cuda_memory_fraction)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {})
            self.send(200, dict(status=runtime.status, error=runtime.error,
                                checkpoint=runtime.checkpoint, contract=CONTRACT, names=NAMES,
                                rtc=runtime.rtc))

        def do_POST(self):
            if self.path != '/infer':
                return self.send(404, {})
            if runtime.status not in ('ready', 'mock'):
                return self.send(503, dict(error=runtime.status))
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid body size')
                self.connection.settimeout(5)
                payload = json.loads(self.rfile.read(size))
                self.send(200, runtime.infer(payload))
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, dict(error=str(exc)))
            except Exception as exc:
                self.send(503, dict(error=str(exc)))

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
