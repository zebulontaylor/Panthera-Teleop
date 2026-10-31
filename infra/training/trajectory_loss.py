"""Experimental timing-tolerant loss on predicted *action trajectories*.

This module does not change any trainer. Inputs are physical or consistently
normalized absolute poses, never pi0.5 flow velocities. For pi0.5's convention
x_t=(1-t)*action+t*noise, reconstruct action_hat=x_t-t*v_hat first; retain
the original flow objective and evaluate this as an auxiliary objective.

Uses the debiased soft-DTW construction of Blondel et al. (2021):
https://proceedings.mlr.press/v130/blondel21a.html
The bounded alignment and small synchronous term intentionally limit invariance.
This readable quadratic-time reference implementation is not GPU optimized.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _soft_dtw(x: Tensor, y: Tensor, gamma: float, band: int) -> Tensor:
    """One monotonic alignment shared across all action coordinates."""
    n = x.shape[0]
    costs = (x[:, None, :] - y[None, :, :]).square().mean(-1)
    # Only reachable cells are built: logsumexp of all -inf yields NaN grads.
    previous = {0: costs.new_zeros(())}
    for i in range(1, n + 1):
        current = {}
        for j in range(max(1, i - band), min(n, i + band) + 1):
            predecessors = []
            if j in previous:
                predecessors.append(previous[j])
            if j - 1 in previous:
                predecessors.append(previous[j - 1])
            if j - 1 in current:
                predecessors.append(current[j - 1])
            soft_min = -gamma * torch.logsumexp(-torch.stack(predecessors) / gamma, 0)
            current[j] = costs[i - 1, j - 1] + soft_min
        previous = current
    return previous[n]


def timing_tolerant_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    scale: Tensor,
    valid: Tensor | None = None,
    max_shift: int = 10,
    gamma: float = 0.01,
    timing_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Return scalar ``loss``, ``aligned``, and ``synchronous`` diagnostics.

    prediction/target: [batch, time, dimensions], equal shape and same units.
    scale: positive [dimensions] tolerances or fixed normalization scales.
        Include both arms and grippers in one vector; exclude padded model dims.
    valid: optional bool [batch,time], True for real targets. Each sequence must
        have a nonempty valid prefix. Padding is excluded before all arithmetic.
    max_shift: maximum alignment offset in samples (10 = 0.5 s at 20 Hz).
    gamma: soft-min temperature in squared scaled-error units.
    timing_weight: synchronous MSE fraction; defaults to 10%, with 90% alignment.

    Full endpoints and every valid sample participate in a monotonic path.
    Thus this tolerates timing changes within a chunk, but not arbitrary speed
    changes that move its endpoint beyond the chunk's covered motion. It is not
    a pure geometry metric or a measure of Cartesian insertion accuracy.
    Report both components; lower total loss alone cannot establish precision.
    Defaults are experimental, not tuned against Panthera rollouts.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape or min(prediction.shape) < 1:
        raise ValueError("prediction and target must share nonempty [B,T,D] shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must be floating point")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if isinstance(max_shift, bool) or not isinstance(max_shift, int) or max_shift < 0:
        raise ValueError("max_shift must be a nonnegative integer")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if not math.isfinite(timing_weight) or not 0 <= timing_weight <= 1:
        raise ValueError("timing_weight must be in [0,1]")
    # DP in float32 minimum, even when the policy uses mixed precision.
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    scales = torch.as_tensor(scale, device=prediction.device, dtype=dtype).detach()
    if scales.shape != prediction.shape[2:] or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("scale must contain one finite positive value per dimension")
    if valid is None:
        valid = torch.ones(prediction.shape[:2], device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,T]")
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("valid must contain a nonempty prefix for each sequence")
    aligned, synchronous = [], []
    for b, length in enumerate(valid.sum(1).tolist()):
        x = prediction[b, :length].to(dtype) / scales
        y = target[b, :length].detach().to(dtype) / scales
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("valid action values must be finite")
        cross = _soft_dtw(x, y, gamma, max_shift)
        self_x = _soft_dtw(x, x, gamma, max_shift)
        self_y = _soft_dtw(y, y, gamma, max_shift)
        aligned.append((cross - 0.5 * (self_x + self_y)) / length)
        synchronous.append((x - y).square().mean())
    aligned_mean = torch.stack(aligned).mean()
    synchronous_mean = torch.stack(synchronous).mean()
    return {
        "loss": (1 - timing_weight) * aligned_mean + timing_weight * synchronous_mean,
        "aligned": aligned_mean,
        "synchronous": synchronous_mean,
    }
