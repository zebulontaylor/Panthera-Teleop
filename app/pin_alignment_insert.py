"""Three loaded button-style XY nudges, then one bounded toward-base sweep."""
import numpy as np

AUTO_NUDGE_STEP_M = .005
AUTO_NUDGE_PASSES = 3
AUTO_NUDGE_INTERVAL_S = 1.
AXIAL_RETRY_SPEED_M_S = .002
AXIAL_RETRY_LIMIT_M = .015
LOADED_XY_ENVELOPE_M = .03


class LoadedXYEnvelopeError(ValueError):
    pass


def loaded_xy_nudge_goal(position, delta, anchor):
    """Shared manual/automatic loaded nudge endpoint in native LEFT-base XY."""
    position = np.asarray(position, dtype=float)
    delta = np.asarray(delta, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if (any(value.shape != (3,) or not np.isfinite(value).all()
            for value in (position, delta, anchor))):
        raise ValueError('Invalid loaded XY nudge')
    if np.linalg.norm(delta[:2]) > AUTO_NUDGE_STEP_M + 1e-9:
        raise ValueError('Loaded XY nudge exceeds the 5 mm step limit')
    goal = position[:2] + delta[:2]
    if np.linalg.norm(goal-anchor[:2]) > LOADED_XY_ENVELOPE_M + 1e-12:
        raise LoadedXYEnvelopeError('Loaded XY target exceeds the 30 mm load envelope')
    return goal


class AutoInsertion:
    def __init__(self, start_xyz, config, started, generation=None):
        self.anchor = np.asarray(start_xyz, dtype=float).copy()
        if self.anchor.shape != (3,) or not np.isfinite(self.anchor).all() or not np.isfinite(started):
            raise ValueError('Invalid insertion origin')
        self.config = config
        self.started = self.last_time = float(started)
        self.generation = generation
        self.goal = self.anchor[:2].copy()
        self.ready_at = float(started)
        self.frames = set()
        # Count visual passes, including zero corrections inside the deadband.
        self.nudge_count = 0
        self.stop_reason = None
        self.waiting_reason = None
        self.unavailable = False
        radius = np.linalg.norm(self.anchor[:2])
        self.axial_direction = -self.anchor[:2]/radius if radius > 1e-6 else np.zeros(2)
        self.axial_distance = 0.
        self.axial_origin = None

    def update(self, now, position, proposal, validate_callback, baseline_armed=True):
        # The loaded-height monitor owns baseline/drop termination externally.
        # It does not add a second gating state machine to the button sequence.
        previous_time = self.last_time
        position = np.asarray(position, dtype=float)
        valid = (np.isfinite(now) and now >= self.last_time
                 and position.shape == (3,) and np.isfinite(position).all())
        if not valid:
            self.stop_reason = self.stop_reason or 'invalid_auto_measurement'
        else:
            self.last_time = float(now)
        phase = 'nudging'
        if self.stop_reason is None:
            if now-self.started >= self.config.max_duration_s:
                self.stop_reason = 'time_limit'
            elif self.nudge_count >= AUTO_NUDGE_PASSES and now >= self.ready_at:
                self.unavailable = False
                self.waiting_reason = None
                if not np.any(self.axial_direction):
                    self.stop_reason = 'invalid_axial_direction'
                else:
                    # Freeze the last accepted nudge endpoint; never reanchor to
                    # encoder drift. Detection and tracking lag do not pause or
                    # stop this bounded sweep once the three passes are done.
                    if self.axial_origin is None:
                        self.axial_origin = self.goal.copy()
                        axial_dt = 0.
                    else:
                        axial_dt = min(now-previous_time, self.config.max_sample_gap_s)
                    distance = min(AXIAL_RETRY_LIMIT_M,
                        self.axial_distance + AXIAL_RETRY_SPEED_M_S*axial_dt)
                    goal = self.axial_origin + self.axial_direction*distance
                    if np.linalg.norm(goal-self.anchor[:2]) > LOADED_XY_ENVELOPE_M+1e-12:
                        self.stop_reason = 'auto_xy_envelope'
                    else:
                        self.goal = goal
                        self.axial_distance = distance
                    phase = 'axial_limit' if distance >= AXIAL_RETRY_LIMIT_M else 'axial_retry'
            elif now >= self.ready_at:
                phase = 'waiting_for_preview'
                self.unavailable = True
                if not isinstance(proposal, dict):
                    self.waiting_reason = 'Alignment preview unavailable'
                elif proposal.get('frame_id') in self.frames:
                    self.waiting_reason = 'Waiting for a different nudge frame'
                else:
                    frame_id = proposal.get('frame_id')
                    try:
                        # Same validation and endpoint calculation as pressing
                        # the manual loaded nudge button at the current pose.
                        delta = np.asarray(validate_callback(proposal, frame_id, applying=True), dtype=float)
                        goal = loaded_xy_nudge_goal(position, delta, self.anchor)
                    except LoadedXYEnvelopeError:
                        self.stop_reason = 'auto_xy_envelope'
                    except Exception as exc:
                        self.waiting_reason = str(exc)
                    else:
                        if np.linalg.norm(delta[:2]) > 1e-12:
                            self.goal = goal
                        self.frames.add(frame_id)
                        self.nudge_count += 1
                        self.ready_at = now + AUTO_NUDGE_INTERVAL_S
                        self.unavailable = False
                        self.waiting_reason = None
                        phase = 'nudging'
        if self.stop_reason:
            phase = 'stopped'
        return dict(xy_goal=self.goal.copy(), dither_xy=np.zeros(2), phase=phase,
                    nudge_count=self.nudge_count, nudge_passes=AUTO_NUDGE_PASSES,
                    stop_reason=self.stop_reason, waiting_reason=self.waiting_reason,
                    preview_unavailable=self.unavailable,
                    axial_retry_m=self.axial_distance, axial_retry_limit_m=AXIAL_RETRY_LIMIT_M,
                    accumulation_remaining_s=0.)
