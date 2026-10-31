"""Slow-only, shared-clock trajectory retiming for offline dataset experiments.

The path is piecewise linear through the supplied pose knots. Limits are checked
on the exported, uniform-rate finite differences, NOT continuous derivatives.
Low-speed intervals retain their exact duration in the map; a uniform output
clock can shift their sample phase by less than one frame. Protected acceleration
violations (including unavoidable seams) are reported, never repaired by changing
protected timing. This is an offline data transform, not a robot safety planner.
"""
from __future__ import annotations

import numpy as np

JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = np.array([6, 13])


def trajectory_metrics(poses, fps=20.):
    """JSON-safe discrete speed/acceleration summaries, in rad/s and rad/s²."""
    q = np.asarray(poses, dtype=float)
    v = np.diff(q, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    out = {}
    for name, values, axes in [('joint_velocity', v, JOINTS),
                               ('gripper_velocity', v, GRIPPERS),
                               ('joint_acceleration', a, JOINTS),
                               ('gripper_acceleration', a, GRIPPERS)]:
        x = np.max(np.abs(values[:, axes]), axis=1) if len(values) else np.zeros(0)
        out[name] = {f'p{p}': float(np.percentile(x, p)) if len(x) else 0.
                     for p in [50, 90, 95, 99]}
        out[name]['max'] = float(x.max()) if len(x) else 0.
    out['duration_seconds'] = float(max(0, len(q) - 1) / fps)
    return out


def _supports(source_times, fps, n_intervals, order):
    """Inclusive interval ranges touched by each finite-difference stencil."""
    starts = np.floor(source_times[:-order] * fps + 1e-9).astype(int)
    ends = np.ceil(source_times[order:] * fps - 1e-9).astype(int) - 1
    return np.clip(starts, 0, n_intervals - 1), np.clip(ends, 0, n_intervals - 1)


def retime(poses, fps=20., joint_velocity=.6, joint_acceleration=3.,
           gripper_velocity=2., gripper_acceleration=10.,
           protect_joint_speed=.15, protect_gripper_speed=.3,
           protected_mask=None, max_iterations=80, max_scale=16.):
    """Return a retimed path, shared timing map, and explicit limit diagnostics.

    poses is N x 14, including a terminal next-action pose. A supplied boolean
    protected_mask has N-1 elements and ADDS protection to the automatic mask.
    Automatic protection requires both arm and gripper interval speeds below
    their thresholds; a moving gripper therefore prevents a joint pause from
    being treated as a whole-robot pause. No semantic contact inference is made.
    max_scale caps iterative interval dilation. Terminal uniform-grid rounding
    can add less than one further source-frame duration to one editable interval
    (its actual scale is then less than max_scale + 1). Exported samples lie on
    the source path, but connecting sparse output samples can skip source corners.
    Limits can be infeasible at protected seams. Inspect metrics['converged'] and
    metrics['violations']; the result is still useful for comparative exports.
    """
    q = np.asarray(poses, dtype=float)
    limits = np.array([joint_velocity, gripper_velocity,
                       joint_acceleration, gripper_acceleration], dtype=float)
    if (q.ndim != 2 or q.shape[1] != 14 or len(q) < 2 or
            not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0 or
            not np.isfinite(limits).all() or (limits <= 0).any() or
            not np.isfinite([protect_joint_speed, protect_gripper_speed]).all() or
            min(protect_joint_speed, protect_gripper_speed) < 0 or
            max_iterations < 0 or not np.isfinite(max_scale) or max_scale < 1):
        raise ValueError('Expected finite N x 14 poses, positive fps/limits and valid protection/iteration settings')
    dt = 1. / fps
    n = len(q) - 1
    source_knots = np.arange(n + 1) * dt
    speeds = np.abs(np.diff(q, axis=0)) * fps
    protected = ((speeds[:, JOINTS].max(axis=1) <= protect_joint_speed) &
                 (speeds[:, GRIPPERS].max(axis=1) <= protect_gripper_speed))
    if protected_mask is not None:
        extra = np.asarray(protected_mask)
        if extra.shape != (n,) or extra.dtype.kind != 'b':
            raise ValueError('protected_mask must be boolean with N-1 entries')
        protected |= extra
    editable = ~protected
    scales = np.maximum(1., np.maximum(speeds[:, JOINTS].max(axis=1) / joint_velocity,
                                      speeds[:, GRIPPERS].max(axis=1) / gripper_velocity))
    scales[protected] = 1.
    scales = np.minimum(scales, max_scale)
    editable_indices = np.flatnonzero(editable)

    def sample():
        durations = scales * dt
        # A terminal pose on the 20 Hz grid makes every action exactly one frame
        # ahead. Put sub-frame rounding in an editable interval only.
        total = durations.sum()
        ticks = int(np.ceil(total * fps - 1e-8))
        if len(editable_indices):
            durations[editable_indices[-1]] += max(0., ticks * dt - total)
        knots = np.r_[0., np.cumsum(durations)]
        output_times = np.arange(ticks + 1) * dt
        source_times = np.interp(output_times, knots, source_knots)
        source_times[-1] = source_knots[-1]
        sampled = np.column_stack([np.interp(source_times, source_knots, q[:, j]) for j in range(14)])
        return sampled, source_times, output_times, knots

    def diagnose(sampled, source_times):
        v = np.diff(sampled, axis=0) * fps
        a = np.diff(v, axis=0) * fps
        violations = {}
        updates = np.ones(n)
        for label, vals, axes, cap, order in [
                ('joint_velocity', v, JOINTS, joint_velocity, 1),
                ('gripper_velocity', v, GRIPPERS, gripper_velocity, 1),
                ('joint_acceleration', a, JOINTS, joint_acceleration, 2),
                ('gripper_acceleration', a, GRIPPERS, gripper_acceleration, 2)]:
            ratios = np.abs(vals[:, axes]).max(axis=1) / cap if len(vals) else np.zeros(0)
            start, end = _supports(source_times, fps, n, order)
            count_protected = count_editable = 0
            for k in np.flatnonzero(ratios > 1.00001):
                indices = np.arange(start[k], end[k] + 1)
                candidates = indices[editable[indices]]
                if not len(candidates):
                    count_protected += 1
                    continue
                count_editable += 1
                # Timing ramps affect neighboring editable intervals, while all
                # protected spans stay fixed. Bounded local growth is stable at
                # turns; uniform-grid validation catches phase-dependent seams.
                lo = max(0, start[k] - 2)
                hi = min(n, end[k] + 3)
                neighbors = np.arange(lo, hi)
                neighbors = neighbors[editable[neighbors]]
                factor = min(1.25, max(1.01, ratios[k] ** (1. / order) * 1.01))
                updates[neighbors] = np.maximum(updates[neighbors], factor)
            violations[label] = {'protected': count_protected, 'editable_or_seam': count_editable}
        return violations, updates

    iterations = 0
    for iteration in range(int(max_iterations) + 1):
        sampled, source_times, output_times, knots = sample()
        violations, updates = diagnose(sampled, source_times)
        if np.all(updates == 1.) or iteration == max_iterations:
            break
        updated = np.minimum(scales * updates, max_scale)
        updated[protected] = 1.
        if np.array_equal(updated, scales):
            break
        scales = updated
        iterations += 1
    actual_scales = np.diff(knots) / dt
    before, after = trajectory_metrics(q, fps), trajectory_metrics(sampled, fps)
    metrics = {
        'before': before, 'after': after,
        'duration_scale': after['duration_seconds'] / before['duration_seconds'],
        'added_duration_seconds': after['duration_seconds'] - before['duration_seconds'],
        'protected_intervals': int(protected.sum()), 'total_intervals': n,
        'protected_source_duration_seconds': float(protected.sum() * dt),
        'protected_duration_error_seconds': float(np.max(np.abs(np.diff(knots)[protected] - dt))) if protected.any() else 0.,
        'iterations': iterations, 'violations': violations,
        'max_interval_scale': float(actual_scales.max()),
        'intervals_at_scale_cap': int(np.count_nonzero(editable & (scales >= max_scale - 1e-9))),
        'scale_cap': float(max_scale),
        'converged': not any(v['editable_or_seam'] for v in violations.values()),
        'all_limits_satisfied': not any(sum(v.values()) for v in violations.values()),
        'limits': dict(zip(['joint_velocity', 'gripper_velocity', 'joint_acceleration', 'gripper_acceleration'], map(float, limits))),
        'constraint_scope': 'uniform-output finite differences; no continuous acceleration guarantee',
    }
    return dict(poses=sampled, source_times=source_times, output_times=output_times,
                knot_times=knots, protected_mask=protected, interval_scales=actual_scales,
                metrics=metrics)
