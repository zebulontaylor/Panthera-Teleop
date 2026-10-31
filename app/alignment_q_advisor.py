"""Freshness-checked advisory Q scoring. This module never executes actions."""
from __future__ import annotations

import math
from app.alignment_q_features import encode_observation, feature_spec, OBSERVATION_DIM

PHASES = ('pickup', 'insertion', 'gear_pickup', 'pin_pickup', 'gear_insertion', 'pin_insertion')
ACTION_CONVENTION = 'measured joint delta proxy; not commanded control action'
ACTION_SEMANTICS = '14 measured next-pose minus current-pose radians: left joints, right joints, left gripper, right gripper; not sent commands.'
HORIZON_S = .1


def _timestamp(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{label} must be a positive integer monotonic timestamp')
    return value


class AlignmentQAdvisor:
    """Score proposed measured-delta proxies; recommendations are advisory only.

    The caller must provide arm measurements captured at measured_state_ns, using
    the same monotonic clock as the prediction and now_ns. No hardware interface
    is accepted or invoked. Only trusted local checkpoints should be loaded.
    """
    def __init__(self, checkpoint, device='cpu', *, scorer=None):
        if scorer is None:
            from scripts.alignment_q_train import AlignmentQScorer
            scorer = AlignmentQScorer(checkpoint, device=device)
        self.scorer = scorer
        metadata = scorer.metadata
        audit = metadata.get('dataset_audit') or {}
        if audit.get('feature_spec') != feature_spec():
            raise ValueError('checkpoint feature schema or calibration identity mismatch')
        if audit.get('action_horizon_s') != HORIZON_S:
            raise ValueError('checkpoint must use a fixed 0.1 second action horizon')
        if metadata.get('observation_dim') != OBSERVATION_DIM or metadata.get('action_dim') != 14:
            raise ValueError('checkpoint observation/action dimensions mismatch')
        if metadata.get('action_convention') != ACTION_CONVENTION or audit.get('action_semantics') != ACTION_SEMANTICS:
            raise ValueError('checkpoint action convention mismatch')
        self._previous = None
        self._previous_ns = None
        self._phase = None
        self._last_seen_ns = None

    def reset(self):
        """Reset at episode boundaries, even when phase labels do not change."""
        self._previous = self._previous_ns = self._phase = self._last_seen_ns = None

    def score(self, prediction, arms, candidate_deltas, phase, *, now_ns,
              measured_state_ns, horizon_s=HORIZON_S):
        try:
            if phase not in PHASES:
                raise ValueError('unknown phase')
            if isinstance(horizon_s, bool) or not math.isfinite(float(horizon_s)) or float(horizon_s) != HORIZON_S:
                raise ValueError('only the fixed 0.1 second action horizon is supported')
            now = _timestamp(now_ns, 'now_ns')
            measured = _timestamp(measured_state_ns, 'measured_state_ns')
            timestamp = _timestamp(prediction['monotonic_ns'], 'prediction.monotonic_ns')
            if self._last_seen_ns is not None and timestamp <= self._last_seen_ns:
                raise ValueError('prediction timestamps must strictly increase')
            self._last_seen_ns = timestamp
            if not 0 <= now - timestamp <= 250_000_000:
                raise ValueError('prediction is stale or from the future')
            stamps = prediction['frame_timestamps_ns']
            camera_times = [_timestamp(stamps[name], name) for name in feature_spec()['camera_names']]
            if max(camera_times) - min(camera_times) > 40_000_000:
                raise ValueError('camera timestamps are not synchronized')
            if min(camera_times) != timestamp:
                raise ValueError('prediction timestamp does not match input frames')
            if max(camera_times) > now:
                raise ValueError('camera timestamp is from the future')
            if abs(measured - timestamp) > 10_000_000 or measured > now:
                raise ValueError('measured arm state is not synchronized with prediction')
            actions = []
            for candidate in candidate_deltas:
                if len(candidate) != 14:
                    raise ValueError('each candidate must contain 14 measured joint/gripper delta proxies')
                action = [float(v) for v in candidate]
                if not all(math.isfinite(v) for v in action):
                    raise ValueError('candidate deltas must be finite')
                actions.append(action)
            dt = None if self._previous_ns is None else (timestamp - self._previous_ns) / 1e9
            history = self._previous if phase == self._phase and dt is not None and .05 <= dt <= .15 else None
            observation = encode_observation(prediction, arms, history, dt if history is not None else None)
            scores = self.scorer.score_candidates(observation, actions, phase)
            if len(scores) != len(actions):
                raise ValueError('scorer returned wrong candidate count')
            eligible = [i for i, s in enumerate(scores) if s.get('supported') and s.get('q') is not None and math.isfinite(float(s['q']))]
            recommended = max(eligible, key=lambda i: scores[i]['q']) if eligible else None
            import copy
            self._previous, self._previous_ns, self._phase = copy.deepcopy(prediction), timestamp, phase
            return dict(status='ready', advisory_only=True, phase=phase, horizon_s=HORIZON_S,
                        scores=scores, q=[s.get('q') if i in eligible else None for i, s in enumerate(scores)],
                        recommended_index=recommended, history_valid=history is not None,
                        score_semantics='discounted annotated return; not success probability',
                        action_convention=ACTION_CONVENTION)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self._previous = self._previous_ns = self._phase = None
            return dict(status='rejected', advisory_only=True, reason=str(exc),
                        scores=[], q=[], recommended_index=None)
