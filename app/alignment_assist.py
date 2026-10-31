"""Fail-closed, local image-based alignment assistance for hand guidance.

The calibrated Jacobian differentiates (moving landmark - target landmark),
stacked x/y in ``camera_names`` order, with respect to six joint radians.
It is valid only near the measured calibration pose. This controller creates
small spring torques about the *current measured pose*, never an integrated
position target. Camera calibration and jaw/contact evidence are mandatory.
"""
from copy import deepcopy

import numpy as np


class OperatorNudgeLease:
    """One episode-bound burst per press; delayed renewals cannot re-arm it."""
    def __init__(self):
        self.episode = self.press = self.started = self.renewed = None
        self.finished = set()

    def update(self, now, episode, press, held):
        if episode != self.episode:
            self.episode = episode
            self.press = self.started = self.renewed = None
        if not held:
            self.finished.add(press)
            if press == self.press:
                self.renewed = None
            return
        if press in self.finished:
            return
        if self.press != press:
            if self.press is not None:
                self.finished.add(self.press)
            self.press, self.started = press, now
        self.renewed = now

    def active(self, now, episode):
        if (self.renewed is not None and
                (now-self.renewed > .2 or now-self.started >= .3)):
            self.finished.add(self.press)
            self.renewed = None
        return (episode is not None and episode == self.episode and self.renewed is not None
                and 0 <= now-self.renewed <= .2 and 0 <= now-self.started < .3)


DEFAULT_CONFIG = dict(enabled=False, task='gear', max_age_seconds=.2,
    max_camera_skew_seconds=.06, min_visibility=.85, min_confidence=.5,
    max_pixel_std=8., max_error_px=60., deadband_px=2.,
    max_delta_rad=.006, max_torque_nm=.15, spring_nm_per_rad=15.,
    damping_nm_s_per_rad=.03, max_joint_speed_rad_s=.3,
    joint_limit_margin_rad=.03, damping_px_per_rad=5.,
    held_confirm_seconds=.2, max_nudge_seconds=.3, calibration=None)


class AlignmentAssist:
    def __init__(self, config=None):
        self.config = deepcopy(DEFAULT_CONFIG)
        self.config.update(config or {})
        self.enabled = bool(self.config['enabled'])
        self._held_since = None
        self._last_timestamp = None
        self._nudge_started = None
        self._burst_reference = None
        self._operator_press = None
        self._trial_stopped = False
        self._burst_gripper_position = None

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self._held_since = None
        self._last_timestamp = None
        self._nudge_started = None
        self._burst_reference = None

    def _off(self, reason):
        if reason != 'nudge_budget_exhausted':
            self._held_since = None
        if self.config.get('mode') != 'operator_supervised' and reason in ('disabled', 'not_collecting', 'gripper_not_holding', 'object_not_at_jaws', 'targets_too_far'):
            self._nudge_started = None
        return dict(active=False, reason=reason, torque_nm=[0.]*6,
                    delta_rad=[0.]*6, side=(self.config.get('calibration') or {}).get('side'),
                    task=self.config.get('task'))

    def evaluate(self, now, snapshot, arm, joint_limits, *, collecting=False, other_arm=None,
                 operator_confirmed=False, operator_press=None):
        """Return an additive torque command; invalid inputs always return zero.

        ``snapshot`` is CNN worker output; ``arm`` is hand_guide.feedback's
        measured arm dictionary. ``joint_limits`` is (lower, upper), radians.
        Enable only during recording and exclude return/hold/reset modes in the
        caller. ``checkpoint_id`` is the exact model identity supplied by the
        worker; camera_mapping must match its verified physical wrist mapping.
        """
        if not self.enabled:
            return self._off('disabled')
        if not collecting:
            return self._off('not_collecting')
        supervised = self.config.get('mode') == 'operator_supervised'
        if supervised:
            if not operator_confirmed or not operator_press:
                return self._off('hold_button_required')
            if operator_press != self._operator_press:
                self._operator_press = operator_press
                self._held_since = self._last_timestamp = self._nudge_started = None
                self._burst_reference = None
                self._trial_stopped = False
                self._burst_gripper_position = None
            if self._trial_stopped:
                return self._off('trial_press_exhausted')
        try:
            result = self._evaluate(float(now), snapshot, arm, joint_limits, other_arm, supervised)
            if supervised and not result['active']:
                self._trial_stopped = True
            return result
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, np.linalg.LinAlgError):
            if supervised:
                self._trial_stopped = True
            return self._off('invalid_evidence_or_calibration')

    def _evaluate(self, now, snapshot, arm, joint_limits, other_arm, supervised=False):
        c = self.config
        cal = c.get('calibration')
        acceptable = (isinstance(cal, dict) and (cal.get('verified') is True if not supervised else
            cal.get('scope') == 'operator_supervised' and cal.get('measured') is True and c['task'] == 'gear'))
        if not acceptable:
            return self._off('calibration_required')
        cameras = list(cal['camera_names'])
        if len(cameras) != 2 or set(cameras) != {'left_wrist', 'right_wrist'}:
            return self._off('invalid_camera_pair')
        mapping = cal['camera_mapping']
        if len(set(mapping[name] for name in cameras)) != 2:
            return self._off('invalid_camera_mapping')
        if snapshot.get('status') != 'ready':
            return self._off(snapshot.get('status', 'no_prediction'))
        if any(snapshot['camera_mapping'].get(name) != mapping[name] for name in cameras):
            return self._off('camera_mapping_mismatch')
        stamp = float(snapshot['monotonic_ns']) / 1e9
        age = now-stamp
        if not np.isfinite(age) or age < 0 or age > min(.25, float(c['max_age_seconds'])):
            return self._off('stale_prediction')
        stamps = [float(snapshot['frame_timestamps_ns'][mapping[name]]) / 1e9 for name in cameras]
        if (not np.all(np.isfinite(stamps)) or any(t > now or now-t > min(.25, float(c['max_age_seconds'])) for t in stamps)
                or max(stamps)-min(stamps) > min(.1, float(c['max_camera_skew_seconds']))):
            return self._off('unsynchronized_cameras')
        if self._last_timestamp is not None and stamp < self._last_timestamp:
            return self._off('out_of_order_prediction')
        task = c['task']
        if task not in ('gear', 'pin') or cal['side'] not in ('left', 'right'):
            return self._off('invalid_task_or_arm')
        prediction = snapshot['predictions'][task]
        if not cal['checkpoint_id'] or prediction['checkpoint_id'] != cal['checkpoint_id']:
            return self._off('checkpoint_mismatch')
        if list(prediction['image_size']) != list(cal['image_size']):
            return self._off('image_size_mismatch')
        q = self._vector(arm['position_rad'], 6)
        velocity = self._vector(arm['velocity_rad_s'], 6)
        limits = (joint_limits['lower'], joint_limits['upper']) if isinstance(joint_limits, dict) else joint_limits
        lower, upper = (self._vector(v, 6) for v in limits)
        margin = max(.01, float(c['joint_limit_margin_rad']))
        if np.any(q <= lower+margin) or np.any(q >= upper-margin):
            return self._off('joint_limit_margin')
        if np.max(np.abs(velocity)) > min(.5, float(c['max_joint_speed_rad_s'])):
            return self._off('operator_moving')
        reference = self._vector(cal['reference_joint_position_rad'], 6)
        envelope = float(cal['max_joint_distance_rad'])
        travel_bounded_servo = supervised and c.get('servo_travel_bounded') is True
        if not 0 < envelope <= (.4 if supervised else .3) or (not travel_bounded_servo and np.max(np.abs(q-reference)) > envelope):
            return self._off('outside_calibration_envelope')
        other_q = self._vector(other_arm['position_rad'], 6)
        other_reference = self._vector(cal['other_arm_reference_joint_position_rad'], 6)
        if not travel_bounded_servo and np.max(np.abs(other_q-other_reference)) > envelope:
            return self._off('other_arm_outside_calibration_envelope')
        if supervised:
            gp, gv = self._vector([arm['gripper']['position'], arm['gripper']['velocity']], 2)
            if self._burst_gripper_position is None:
                self._burst_gripper_position = gp
            if not 0 <= gp <= 2 or abs(gp-self._burst_gripper_position) > .04 or abs(gv) > .1:
                return self._off('gripper_changed')
        if supervised and self._burst_reference is not None and np.linalg.norm(q-self._burst_reference) >= .006:
            return self._off('measured_travel_limit')
        jacobian = np.asarray(cal['image_jacobian_px_per_rad'], dtype=float)
        if jacobian.shape != (4, 6) or not np.all(np.isfinite(jacobian)) or np.linalg.matrix_rank(jacobian) < 2:
            return self._off('invalid_image_jacobian')
        names = list(prediction['landmark_names'])
        moving = names.index('gear_center' if task == 'gear' else 'pin_tip')
        target_name = cal['target_landmark']
        if (task == 'gear' and target_name != 'slot_center') or (task == 'pin' and target_name != 'carrier_bore'):
            return self._off('invalid_target_landmark')
        target = names.index(target_name) if task == 'gear' else None
        indices = [list(prediction['camera_names']).index(name) for name in cameras]
        points = np.asarray(prediction['keypoints_2d'], dtype=float)
        desired_error = self._vector(cal['target_error_px'], 4)
        axes = list(cal['residual_axes'])
        if axes != ([1, 3] if task == 'gear' else [0, 1, 2, 3]):
            return self._off('invalid_residual_axes')
        errors = []
        for camera, index in zip(cameras, indices):
            if task == 'pin':
                roi = self._vector(cal['target_rois_px'][camera], 4)
                width, height = cal['image_size']
                if not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
                    return self._off('invalid_target_roi')
                candidates = [names.index(f'carrier_hole_{n}') for n in range(1, 5)]
                # Slot labels are unordered; choose physical bore by measured ROI,
                # independently in each view. Ambiguous ROIs never command motion.
                inside = [n for n in candidates if np.all(np.isfinite(points[index, n]))
                          and np.all(points[index, n] >= roi[:2])
                          and np.all(points[index, n] <= roi[2:])]
                if len(inside) != 1:
                    return self._off('pin_target_ambiguous_or_missing')
                target = inside[0]
            selected = points[index, [moving, target]]
            if selected.shape != (2, 2) or not np.all(np.isfinite(selected)):
                return self._off('invalid_landmarks')
            width, height = cal['image_size']
            if np.any(selected < 0) or np.any(selected >= [width, height]):
                return self._off('landmarks_out_of_frame')
            for field, threshold, comparison in (() if supervised else (
                    ('visibility_probability', max(.5, float(c['min_visibility'])), 'min'),
                    ('confidence', max(.1, float(c['min_confidence'])), 'min'),
                    ('pixel_std', min(20., float(c['max_pixel_std'])), 'max'))):
                values = np.asarray(prediction[field], dtype=float)[index, [moving, target]]
                if not np.all(np.isfinite(values)) or (np.any(values < threshold) if comparison == 'min' else np.any(values > threshold)):
                    return self._off('targets_not_visible_or_confident')
            offset = len(errors)
            error = selected[0]-selected[1]-desired_error[offset:offset+2]
            if np.linalg.norm(error) > min(100., float(c['max_error_px'])):
                result = self._off('targets_too_far')
                result.update(blocking_camera=camera, target_distance_px=float(np.linalg.norm(error)),
                              blocking_error_px=error.tolist(), max_error_px=min(100., float(c['max_error_px'])))
                return result
            errors.extend(error)
            if supervised:
                # Operator confirms retained gear and target while holding the
                # momentary control; uncalibrated heads/torque are not evidence.
                continue
            roi = self._vector(cal['jaw_rois_px'][camera], 4)
            if not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
                return self._off('invalid_jaw_roi')
            if not np.all((selected[0] >= roi[:2]) & (selected[0] <= roi[2:])):
                return self._off('object_not_at_jaws')
        if not supervised:
            gripper = arm['gripper']
            measured = self._vector([gripper['position'], gripper['torque'], gripper['velocity']], 3)
            contact = float(cal['gripper_contact_torque_nm'])
            closed = float(cal['gripper_closed_position_rad'])
            if not contact > 0 or not closed >= 0 or measured[0] > closed or abs(measured[1]) < contact or abs(measured[2]) > .1:
                return self._off('gripper_not_holding')
        if self._held_since is None:
            self._held_since = stamp
        self._last_timestamp = stamp
        if not supervised and stamp-self._held_since < max(.15, float(c['held_confirm_seconds'])):
            # Do not reset the dwell time while waiting for fresh confirming frames.
            return dict(active=False, reason='confirming_grasp', torque_nm=[0.]*6,
                        delta_rad=[0.]*6, side=cal['side'], task=task)
        error = np.asarray(errors)
        active_error = error[axes]
        if np.max(np.abs(active_error)) <= max(1., float(c['deadband_px'])):
            return dict(active=False, reason='aligned', torque_nm=[0.]*6,
                        delta_rad=[0.]*6, side=cal['side'], task=task, error_px=error.tolist())
        if self._nudge_started is None:
            self._nudge_started = now
            self._burst_reference = q.copy()
        if now-self._nudge_started >= min(.3, max(0., float(c['max_nudge_seconds']))):
            return self._off('nudge_budget_exhausted')
        jacobian = jacobian[axes]
        damping = max(1., float(c['damping_px_per_rad']))
        delta = -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping*damping*np.eye(len(axes)), active_error)
        cap = min(.006, max(0., float(c['max_delta_rad'])))
        delta *= min(1., cap / max(1e-12, float(np.linalg.norm(delta))))
        if np.any(q+delta <= lower+margin) or np.any(q+delta >= upper-margin):
            return self._off('joint_target_limit')
        limit = min(.15, max(0., float(c['max_torque_nm'])))
        if supervised:
            # Passive joints did not move at the weak spring's ~0.06 Nm.
            # Use the existing torque budget along the measured correction
            # direction; measured travel and the operator lease bound the pulse.
            torque = delta * (limit/max(1e-12, float(np.max(np.abs(delta)))))
        else:
            torque = min(20., max(0., float(c['spring_nm_per_rad'])))*delta
        torque -= min(.05, max(0., float(c['damping_nm_s_per_rad'])))*velocity
        torque = np.clip(torque, -limit, limit)
        return dict(active=True, reason='nudging', torque_nm=torque.tolist(),
                    delta_rad=delta.tolist(), side=cal['side'], task=task,
                    error_px=error.tolist(), prediction_monotonic_ns=snapshot['monotonic_ns'])

    @staticmethod
    def _vector(value, length):
        result = np.asarray(value, dtype=float)
        if result.shape != (length,) or not np.all(np.isfinite(result)):
            raise ValueError('invalid finite vector')
        return result
