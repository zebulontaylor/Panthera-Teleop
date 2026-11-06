"""Pure, fail-closed Cartesian load calculation. No motor/device interfaces.

The downward Newton value is model-based feedforward, not measured force or
force regulation. Positive stiffness/damping and slew limits are design checks,
not physical stability validation. A caller must supervise measured freshness,
gravity compensation, policy exclusion, collision safety and actuator ownership.
"""
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
from pin_alignment_motion import NativeToolKinematics
from pin_regrasp_motion import rotation_log
from pin_contact_approach import (APPROACH_SPEED_M_S, APPROACH_SPEED_RAMP_S,
                                  APPROACH_XY_M, APPROACH_TIMEOUT_S, APPROACH_TRACKING_LAG_M)

ASSIST_CAP_NM = np.array([3., 6., 6., 1.5, 1.2, 1.2])
TOTAL_CAP_NM = np.array([5., 8., 8., 3., 2., 2.])
VELOCITY_WINDOW_S = .05
MIN_VELOCITY_WINDOW_S = .04
# Recorded combined encoder changes moved FK .544mm in one tick. This is
# a spatial quantization allowance, not an increase in allowed sustained speed.
POSITION_NOISE_MARGIN_M = .0006
ANGULAR_NOISE_MARGIN_RAD = .004


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('invalid_' + name)
    return result


@dataclass(frozen=True)
class CartesianLoadConfig:
    contact_approach: bool = False
    force_n: float = .5
    force_slew_n_s: float = 2.
    xy_stiffness_n_m: float = 600.
    xy_damping_ns_m: float = 10.
    xy_integral_gain_n_m_s: float = 9600.
    xy_integral_limit_n: float = 12.
    xy_integral_slew_n_s: float = 12.
    xy_integral_deadband_m: float = .00025
    vertical_damping_ns_m: float = 10.
    orientation_stiffness_nm_rad: float = 2.
    orientation_damping_nms_rad: float = .1
    joint_damping_nms_rad: float = .02
    max_lateral_travel_m: float = .03
    max_vertical_travel_m: float = .005
    max_orientation_error_rad: float = .1
    # Legacy configuration fields retained for caller compatibility; no speed trips.
    max_joint_speed_rad_s: float = .3
    max_tool_speed_m_s: float = .05
    max_down_speed_m_s: float = .02
    max_angular_speed_rad_s: float = .3
    joint_limit_margin_rad: float = .02
    max_duration_s: float = 30.
    max_update_gap_s: float = .1

    def __post_init__(self):
        if type(self.contact_approach) is not bool:
            raise ValueError('invalid_config_contact_approach')
        maxima = dict(force_n=3., force_slew_n_s=2., xy_stiffness_n_m=600.,
            xy_damping_ns_m=10., xy_integral_gain_n_m_s=9600.,
            xy_integral_limit_n=12., xy_integral_slew_n_s=12.,
            xy_integral_deadband_m=.00025, vertical_damping_ns_m=20.,
            orientation_stiffness_nm_rad=2., orientation_damping_nms_rad=.2,
            joint_damping_nms_rad=.05, max_lateral_travel_m=.03,
            max_vertical_travel_m=.005, max_orientation_error_rad=.1,
            max_joint_speed_rad_s=.3, max_tool_speed_m_s=.05,
            max_down_speed_m_s=.02, max_angular_speed_rad_s=.3,
            joint_limit_margin_rad=.1, max_duration_s=300., max_update_gap_s=.1)
        if self.contact_approach:
            maxima.update(force_n=.5, max_lateral_travel_m=APPROACH_XY_M,
                          max_vertical_travel_m=.020, max_duration_s=APPROACH_TIMEOUT_S)
        for key, value in asdict(self).items():
            if key == 'contact_approach':
                continue
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= maxima[key]:
                raise ValueError('invalid_config_' + key)


class CartesianLoadController:
    """Absolute XY targets in LEFT native base metres; fixed anchor rotation.

    Actual lower/upper joint limits and current gravity torques are mandatory.
    Any refused step latches a stop; construct a new explicitly anchored instance
    to rearm. Never clip an out-of-limit torque into an apparently valid command.
    """
    def __init__(self, anchor_state14, config=None, now=0., *, lower6=None,
                 upper6=None, kinematics=None):
        self.config = (config if isinstance(config, CartesianLoadConfig)
                       else CartesianLoadConfig(**(config or {})))
        anchor = _vector(anchor_state14, 14, 'anchor_state14')
        self.kinematics = kinematics or NativeToolKinematics()
        self.anchor_q = anchor[:6].copy()
        p, r = self.kinematics.pose(self.anchor_q)
        self.anchor_position = _vector(p, 3, 'anchor_position').copy()
        self.anchor_rotation = self._rotation(r)
        self.started = self.last_time = self._time(now)
        self.ramped_force_n = 0.
        self.xy_integral_n = np.zeros(2)
        self._previous_assist = np.zeros(6)
        self._transfer = None
        self.stopped_reason = None
        self.diagnostics = {}
        self._pose_history = deque([(self.started, self.anchor_position.copy(),
                                     self.anchor_rotation.copy(), self.anchor_q.copy())], maxlen=64)
        self.lower6 = None if lower6 is None else _vector(lower6, 6, 'lower6').copy()
        self.upper6 = None if upper6 is None else _vector(upper6, 6, 'upper6').copy()

    def begin_loaded(self, measured14, now, config):
        """One continuous approach-to-load transfer, retaining torque/history.

        The caller must atomically transfer the independent guard as well.
        Reanchoring is permitted once, after approach contact evidence only.
        """
        if not self.config.contact_approach or self.stopped_reason or self._transfer is not None:
            raise ValueError('loaded_transfer_requires_active_approach')
        now = self._time(now)
        if now < self.last_time or now-self.last_time > self.config.max_update_gap_s:
            raise ValueError('loaded_transfer_requires_fresh_feedback')
        config = config if isinstance(config, CartesianLoadConfig) else CartesianLoadConfig(**config)
        if config.contact_approach:
            raise ValueError('loaded_transfer_requires_loaded_config')
        state = _vector(measured14, 14, 'loaded_anchor_state14')
        p, r = self.kinematics.pose(state[:6])
        self._transfer = dict(started=now,
            z_target=float(self.diagnostics.get('approach_z_target_m', p[2])),
            damping=self.config.vertical_damping_ns_m)
        self.config = config
        self.anchor_q = state[:6].copy()
        self.anchor_position = _vector(p, 3, 'loaded_anchor_position').copy()
        self.anchor_rotation = self._rotation(r)
        self.started = now
        # Preserve measured velocity history, current force, integral and the
        # last emitted assistance. No gravity-only priming or zero-torque tick.

    @staticmethod
    def _time(value):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError('invalid_monotonic_time')
        return float(value)

    @staticmethod
    def _rotation(value):
        r = np.asarray(value, float)
        if r.shape != (3,3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .999:
            raise ValueError('invalid_rotation')
        return r.copy()

    def _invalid(self, reason):
        self.stopped_reason = self.stopped_reason or reason
        return dict(valid=False, stop=True, reason=self.stopped_reason,
            assist_torque_nm=[0.] * 6, full_torque_nm=None,
            commanded_force_n=0., physically_validated=False,
            diagnostics=dict(self.diagnostics),
            force_sensor_used=False, force_regulation=False, motor_command_sent=False)

    def compute(self, q, dq, gravity6, xy_target, now, *, limits=None):
        if self.stopped_reason:
            return self._invalid(self.stopped_reason)
        try:
            q = _vector(q, 6, 'q'); dq = _vector(dq, 6, 'dq')
            gravity = _vector(gravity6, 6, 'gravity6')
            target = _vector(xy_target, 2, 'absolute_xy_target_m')
            now = self._time(now); dt = now - self.last_time
            c = self.config
            if dt < 0: raise ValueError('time_regressed')
            if dt > c.max_update_gap_s + 1e-12: raise ValueError('update_gap_exceeded')
            if now - self.started >= c.max_duration_s: raise ValueError('duration_exceeded')
            if limits is None:
                lower, upper = self.lower6, self.upper6
            else:
                lower, upper = limits
            lower = _vector(lower, 6, 'actual_lower6')
            upper = _vector(upper, 6, 'actual_upper6')
            if np.any(lower >= upper): raise ValueError('invalid_actual_joint_limits')
            if np.any(q <= lower + c.joint_limit_margin_rad) or np.any(q >= upper - c.joint_limit_margin_rad):
                raise ValueError('joint_limit_margin')
            p, r = self.kinematics.pose(q)
            p = _vector(p, 3, 'tool_position'); r = self._rotation(r)
            j = np.asarray(self.kinematics.jacobian(q), float)
            if j.shape != (6,6) or not np.isfinite(j).all(): raise ValueError('invalid_jacobian')
            singular = np.linalg.svd(j, compute_uv=False)
            condition = float(singular[0] / max(singular[-1], 1e-30))
            if singular[-1] < 1e-4 or condition > 1000:
                raise ValueError('kinematic_singularity')
            raw_velocity = j @ dq
            displacement = p - self.anchor_position
            # Use measured pose changes for damping and speed diagnostics.
            # During the short startup window assistance remains zero;
            # absolute travel guards remain active from the first tick.
            candidates = [sample for sample in self._pose_history if
                          MIN_VELOCITY_WINDOW_S <= now-sample[0] <= c.max_update_gap_s+1e-12]
            baseline = (min(candidates, key=lambda sample: abs(now-sample[0]-VELOCITY_WINDOW_S))
                        if candidates else self._pose_history[0])
            window = now-baseline[0]
            priming = window < MIN_VELOCITY_WINDOW_S-1e-12
            delta_p = p-baseline[1]
            delta_r = rotation_log(r @ baseline[2].T)
            raw_pose_velocity = np.r_[delta_p, delta_r]/window if window > 0 else np.zeros(6)
            # The finite pose window is the damping low-pass filter. Do NOT
            # subtract guard quantization margins here: that would eliminate
            # damping during deliberate low-speed spiral tracking.
            velocity = raw_pose_velocity.copy()
            filtered_dq = (q-baseline[3])/window if window > 0 else np.zeros(6)
            smooth_load = c.contact_approach or self._transfer is not None
            if smooth_load and priming:
                # Do not release into an unassisted gravity-only window. Fresh
                # joint feedback supplies damping until pose history is ready.
                velocity = raw_velocity.copy()
                filtered_dq = dq.copy()
            self.diagnostics = dict(velocity_source='measured_pose_window_continuous_damping',
                speed_guards_enabled=False,
                velocity_window_s=window, priming_velocity=priming,
                tool_speed_m_s=float(max(0.,np.linalg.norm(delta_p)-POSITION_NOISE_MARGIN_M)/window) if window>0 else 0.,
                down_speed_m_s=float(max(0.,-delta_p[2]-POSITION_NOISE_MARGIN_M)/window) if window>0 else 0.,
                angular_speed_rad_s=float(max(0.,np.linalg.norm(delta_r)-ANGULAR_NOISE_MARGIN_RAD)/window) if window>0 else 0.,
                raw_pose_tool_speed_m_s=float(np.linalg.norm(raw_pose_velocity[:3])),
                raw_pose_tool_velocity_m_s=raw_pose_velocity[:3].tolist(),
                max_tool_speed_m_s=None,
                max_down_speed_m_s=None,
                max_angular_speed_rad_s=None,
                raw_sdk_tool_velocity_m_s=raw_velocity[:3].tolist(),
                raw_sdk_tool_speed_m_s=float(np.linalg.norm(raw_velocity[:3])),
                estimated_tool_velocity_m_s=velocity[:3].tolist(),
                position_noise_margin_m=POSITION_NOISE_MARGIN_M,
                speed_displacement_m=float(np.linalg.norm(delta_p)),
                speed_displacement_limit_m=None,
                lateral_travel_m=float(np.linalg.norm(displacement[:2])),
                vertical_travel_m=float(displacement[2]))
            # Absolute travel/orientation guards remain instantaneous and take
            # priority even while velocity history is being primed.
            if np.linalg.norm(displacement[:2]) > c.max_lateral_travel_m: raise ValueError('lateral_travel_exceeded')
            if np.linalg.norm(target - self.anchor_position[:2]) > c.max_lateral_travel_m: raise ValueError('xy_target_travel_exceeded')
            if abs(displacement[2]) > c.max_vertical_travel_m: raise ValueError('vertical_travel_exceeded')
            orientation_error = rotation_log(self.anchor_rotation @ r.T)
            if np.linalg.norm(orientation_error) > c.max_orientation_error_rad: raise ValueError('orientation_deviation_exceeded')
            force = (0. if priming and not smooth_load else
                     self.ramped_force_n + np.clip(c.force_n-self.ramped_force_n,
                         -c.force_slew_n_s*dt, c.force_slew_n_s*dt))
            # Project stored force for antiwindup; retain friction support inside
            # the radial encoder-scale deadband. Reversed errors unwind it.
            xy_error = target - p[:2]
            error_norm = np.linalg.norm(xy_error)
            integral = self.xy_integral_n.copy()
            if (not priming or smooth_load) and error_norm > c.xy_integral_deadband_m:
                effective_error = xy_error * (1. - c.xy_integral_deadband_m/error_norm)
                candidate = integral + c.xy_integral_gain_n_m_s * effective_error * dt
                candidate *= min(1., c.xy_integral_limit_n/max(np.linalg.norm(candidate), 1e-30))
                change = candidate-integral
                change *= min(1., c.xy_integral_slew_n_s*dt/max(np.linalg.norm(change), 1e-30))
                integral += change
            wrench = np.zeros(6)
            wrench[:2] = c.xy_stiffness_n_m * xy_error + integral - c.xy_damping_ns_m * velocity[:2]
            if c.contact_approach:
                # Smoothstep speed ramp; integrate analytically so the Z
                # reference begins with zero velocity and acceleration. The
                # load phase retains its existing compliant constant load.
                elapsed = now-self.started
                u = min(1., elapsed/APPROACH_SPEED_RAMP_S)
                distance = APPROACH_SPEED_M_S * (APPROACH_SPEED_RAMP_S*(u**3-.5*u**4)
                    + max(0., elapsed-APPROACH_SPEED_RAMP_S))
                z_target = self.anchor_position[2]-min(distance, c.max_vertical_travel_m)
                z_error = z_target-p[2]
                # Never pull a pin back upward toward an old airborne height.
                # Unexpected sag is stopped, not recovered through contact.
                spring = float(np.clip(600.*z_error, -force, 0.))
                wrench[2] = spring-c.vertical_damping_ns_m*velocity[2]
                self.diagnostics.update(approach_z_target_m=float(z_target),
                    approach_target_speed_m_s=APPROACH_SPEED_M_S*(3*u*u-2*u**3),
                    approach_z_error_m=float(z_error), approach_spring_force_n=spring,
                    # Saturated downward spring plus stable measured height
                    # distinguishes a blocked pin from normal slow tracking.
                    approach_load_established=bool(force>=c.force_n-1e-9
                        and -600.*z_error>=.95*c.force_n))
                if z_error > APPROACH_TRACKING_LAG_M:
                    raise ValueError('approach_descent_tracking_lost')
            elif self._transfer is not None:
                u = min(1., max(0., now-self._transfer['started']))
                blend = 3*u*u-2*u**3
                spring = float(np.clip(600.*(self._transfer['z_target']-p[2]), -force, 0.))
                damping = ((1-blend)*self._transfer['damping']+blend*c.vertical_damping_ns_m)
                wrench[2] = (1-blend)*spring-blend*force-damping*velocity[2]
                self.diagnostics['load_transfer_fraction'] = blend
            else:
                wrench[2] = -force - c.vertical_damping_ns_m * velocity[2]
            wrench[3:] = c.orientation_stiffness_nm_rad * orientation_error - c.orientation_damping_nms_rad * velocity[3:]
            assist = j.T @ wrench - c.joint_damping_nms_rad * filtered_dq
            xy_drive_scale = 1.
            if priming and not smooth_load:
                wrench[:] = 0.; assist[:] = 0.
                xy_drive_scale = 0.
            else:
                # Allocate both spring and integral XY drive to remaining
                # torque headroom. Keep downward load, damping and orientation
                # unchanged; store the applied integral to prevent windup.
                extra = j[:2].T @ (c.xy_stiffness_n_m*xy_error + integral)
                base = assist-extra
                if (np.all(np.abs(base) <= ASSIST_CAP_NM) and
                        np.all(np.abs(gravity+base) <= TOTAL_CAP_NM)):
                    scale = 1.
                    for origin, caps in ((base,ASSIST_CAP_NM),(gravity+base,TOTAL_CAP_NM)):
                        for value, change, cap in zip(origin,extra,caps):
                            if change > 0:
                                scale=min(scale,(cap-value)/change)
                            elif change < 0:
                                scale=min(scale,(-cap-value)/change)
                    scale=max(0.,min(1.,scale))
                    if scale < 1.:
                        scale *= 1.-1e-9
                        xy_drive_scale = scale
                        integral *= scale
                        wrench[:2] = scale*c.xy_stiffness_n_m*xy_error + integral - c.xy_damping_ns_m*velocity[:2]
                        assist = base + extra*scale
            if smooth_load:
                # Smooth all model assistance, including lateral recovery and
                # damping, rather than only ramping the downward force cap.
                requested_assist = assist.copy()
                assist = self._previous_assist + np.clip(assist-self._previous_assist,
                    -.5*dt, .5*dt)
                if c.contact_approach:
                    self.diagnostics['approach_load_established'] &= bool(
                        np.max(np.abs(requested_assist-assist)) <= .01)
                self.diagnostics['approach_assist_slew_nm_s'] = .5
            total = gravity + assist
            if not np.isfinite(total).all(): raise ValueError('nonfinite_torque')
            self.diagnostics.update(assist_torque_nm=assist.tolist(),
                assist_torque_limit_nm=ASSIST_CAP_NM.tolist(),
                full_torque_nm=total.tolist(), total_torque_limit_nm=TOTAL_CAP_NM.tolist())
            if np.any(np.abs(assist) > ASSIST_CAP_NM): raise ValueError('assist_torque_limit')
            if np.any(np.abs(total) > TOTAL_CAP_NM): raise ValueError('gravity_plus_assist_torque_limit')
            self.last_time = now; self.ramped_force_n = force
            self._previous_assist = assist.copy()
            self.xy_integral_n = integral
            self.diagnostics.update(xy_error_m=xy_error.tolist(),
                xy_drive_scale=xy_drive_scale,
                xy_requested_proportional_force_n=(c.xy_stiffness_n_m*xy_error).tolist(),
                xy_proportional_force_n=(xy_drive_scale*c.xy_stiffness_n_m*xy_error).tolist(),
                xy_integral_force_n=integral.tolist(), xy_commanded_force_n=wrench[:2].tolist())
            self._pose_history.append((now,p.copy(),r.copy(),q.copy()))
            while len(self._pose_history)>1 and now-self._pose_history[0][0]>.15:
                self._pose_history.popleft()
            return dict(valid=True, stop=False, reason='priming_velocity' if priming else 'ok', assist_torque_nm=assist.tolist(),
                full_torque_nm=total.tolist(), gravity_torque_nm=gravity.tolist(),
                wrench_left_base=wrench.tolist(), commanded_force_n=force,
                tool_position_left_m=p.tolist(), tool_velocity_left=velocity.tolist(),
                xy_target_left_m=target.tolist(), orientation_error_rad=orientation_error.tolist(),
                elapsed_s=now-self.started, physically_validated=False, force_sensor_used=False,
                jacobian_condition=condition,
                diagnostics=dict(self.diagnostics),
                force_regulation=False, stability_assessment='positive stiffness/damping and bounded feedforward; hardware stability unvalidated',
                wrench_semantics='J.T maps the commanded model wrench to joint assistance; no measured force, contact-force closure guarantee or force regulation',
                motor_command_sent=False)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            return self._invalid(str(exc))

    def step(self, state14, velocity14, xy_target, now, gravity6=None, limits=None):
        try:
            state = _vector(state14, 14, 'state14')
            velocity = _vector(velocity14, 14, 'velocity14')
        except (ValueError, TypeError) as exc:
            return self._invalid(str(exc))
        return self.compute(state[:6], velocity[:6], gravity6, xy_target, now, limits=limits)
