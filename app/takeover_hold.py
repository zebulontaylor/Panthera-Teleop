"""Independent takeover by changes in handle or gripper load."""
import numpy as np

from neutral_hold import NeutralHold


class TakeoverHold(NeutralHold):
    def __init__(self, targets, settings, recording_settings, now):
        super().__init__(targets, settings, now)
        self.motion_settings = {key: recording_settings[key] for key in
            ('motion_gripper_rad', 'motion_velocity_rad_s', 'motion_confirm_seconds')}
        self.gripper_torque_threshold = float(settings.get('gripper_pull_torque_threshold_nm', 0.15))
        if not np.isfinite(self.gripper_torque_threshold) or self.gripper_torque_threshold <= 0:
            raise ValueError('Gripper torque change threshold must be finite and positive')
        self.gripper_baselines = {}
        self.gripper_torque_baselines = {}
        self.gripper_history = {side: [] for side in targets}
        self.gripper_since = {side: None for side in targets}

    def tick(self, now, arms):
        previous = dict(self.ready)
        super().tick(now, arms)
        for side, arm in arms.items():
            if previous[side]:
                continue
            gripper = arm['gripper']
            position, velocity = float(gripper['position']), float(gripper['velocity'])
            torque = float(gripper['torque'])
            if not np.isfinite([position, velocity, torque]).all():
                raise RuntimeError(f'{side}: invalid takeover gripper feedback')
            if side not in self.gripper_baselines:
                self.gripper_history[side].append((position, torque))
                if now-self.started < self.settings['baseline_seconds']:
                    continue
                baseline = np.median(self.gripper_history[side], axis=0)
                self.gripper_baselines[side] = float(baseline[0])
                self.gripper_torque_baselines[side] = float(baseline[1])
                self.gripper_history[side].clear()
            if self.ready[side]:
                self.diagnostics[side]['trigger'] = 'handle_pull'
                continue
            change = abs(position-self.gripper_baselines[side])
            torque_change = abs(torque-self.gripper_torque_baselines[side])
            squeezing = torque_change >= self.gripper_torque_threshold
            self.gripper_since[side] = ((now if self.gripper_since[side] is None else
                                        self.gripper_since[side]) if squeezing else None)
            self.diagnostics[side].update(gripper_baseline_rad=self.gripper_baselines[side],
                gripper_position_rad=position, gripper_change_rad=change,
                gripper_velocity_rad_s=velocity,
                gripper_torque_baseline_nm=self.gripper_torque_baselines[side],
                gripper_torque_nm=torque, gripper_torque_change_nm=torque_change)
            if (self.gripper_since[side] is not None and
                    now-self.gripper_since[side] >= self.motion_settings['motion_confirm_seconds']):
                self.ready[side] = True
                self.diagnostics[side]['trigger'] = 'gripper_force_change'
        return dict(self.ready)
