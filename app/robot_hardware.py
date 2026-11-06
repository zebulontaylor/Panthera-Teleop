"""Teleop/inference-side adapter for the independent motor backend and cameras.

This module cannot open motors; all arm objects are command-staging clients.
"""
import json
import base64
import time
from pathlib import Path
import numpy as np

from motor_client import MotorClient
from remote_inference import CONTRACT, ROOT, GRIPPER_ENCODER_TOLERANCE_RAD, image_payload


class GuardedLoadStopped(RuntimeError):
    """A verified motor-side guard stop, with the fresh pose already held there."""
    def __init__(self, guard_stop, measured_state14):
        super().__init__('Independent motor guard stopped load: '+str(guard_stop.get('reason')))
        self.guard_stop = dict(guard_stop)
        self.measured_state14 = np.asarray(measured_state14, dtype=float).copy()


class RobotHardware:
    """Frontend cameras and motor-client adapter; no SDK ownership."""
    def __init__(self):
        self.robots = {}
        self.cameras = None
        self.motor_client = None
        self.stale = {}
        self.hold_targets = {}
        self.check_active = lambda: None

    def open(self):
        from dashboard import Cameras
        self.config = json.loads((ROOT / 'config/recording.json').read_text())
        self.check_active()
        self.cameras = Cameras(type('Sink', (), {'submit': lambda *_: None})(), self.config)
        self.cameras.wait_ready()
        self.check_active()
        self.motor_client = MotorClient('inference')
        self.motor_client.open(check_active=self.check_active)
        self.robots = self.motor_client.robots
        self.configs = self.motor_client.configs
        self.lower = np.concatenate([np.r_[self.robots[s].joint_limits['lower'],
                                           self.robots[s].gripper_limits['lower']]
                                     for s in ('left', 'right')])
        self.upper = np.concatenate([np.r_[self.robots[s].joint_limits['upper'],
                                           self.robots[s].gripper_limits['upper']]
                                     for s in ('left', 'right')])

    def read(self):
        arms = {s: arm for s, (_, arm) in self.motor_client.read().items()}
        measured = np.asarray([v for s in ('left', 'right') for v in
                               arms[s]['position_rad'] + [arms[s]['gripper']['position']]])
        for i in (6, 13):
            if not self.lower[i] - GRIPPER_ENCODER_TOLERANCE_RAD <= measured[i] <= self.upper[i] + GRIPPER_ENCODER_TOLERANCE_RAD:
                raise RuntimeError('Gripper feedback outside encoder tolerance')
        camera_state = self.cameras.status()
        return measured, arms, camera_state

    def images(self):
        with self.cameras.lock:
            return {n: image_payload(self.cameras.latest[n]['jpeg'], n)
                    for n in CONTRACT['image_crops_xyxy']}

    def expo_images(self):
        """Return each exact inference JPEG together with its source frame receipt."""
        with self.cameras.lock:
            frames = {n: self.cameras.latest[n] for n in CONTRACT['image_crops_xyxy']}
            return ({n: image_payload(frame['jpeg'], n) for n, frame in frames.items()},
                    {n: {k: v for k, v in frame.items() if k != 'jpeg'}
                     for n, frame in frames.items()})

    def inference_images(self):
        """Policy crops and full-frame critic views from the same camera receipt."""
        with self.cameras.lock:
            frames = {n: self.cameras.latest[n] for n in CONTRACT['image_crops_xyxy']}
        return ({n: image_payload(frame['jpeg'], n) for n, frame in frames.items()},
                {n: {k: v for k, v in frame.items() if k != 'jpeg'} for n, frame in frames.items()},
                {n: base64.b64encode(frame['jpeg']).decode('ascii') for n, frame in frames.items()})

    def command(self, target):
        for offset, side in ((0, 'left'), (7, 'right')):
            self.command_arm(side, offset, target)
        if self.motor_client:
            self.motor_client.flush()

    @property
    def supports_guarded_load(self):
        return bool(self.motor_client and
                    getattr(self.motor_client, 'capabilities', {}).get('guarded_cartesian_load'))

    @property
    def guarded_load_limits(self):
        return dict(getattr(self.motor_client, 'guarded_load_limits', {}) or {}) if self.motor_client else {}

    @property
    def guarded_load_status(self):
        return getattr(self.motor_client, 'guard_stop', None) if self.motor_client else None

    def command_guarded_load(self, arms, target, assist_torque_nm, guard):
        """Stage LEFT guarded torque assistance and RIGHT hold atomically.

        The independent motor process adds gravity compensation and enforces
        the measured travel, velocity, torque and command-age guard. Both
        grippers retain their existing commanded squeeze.
        """
        if not self.supports_guarded_load:
            raise RuntimeError('Motor process must support guarded load before force control')
        assist = np.asarray(assist_torque_nm, dtype=float)
        target = np.asarray(target, dtype=float)
        if assist.shape != (6,) or target.shape != (14,) or not np.isfinite(assist).all() or not np.isfinite(target).all():
            raise ValueError('Finite six-joint assistance and fourteen-value hold target required')
        left = self.robots['left']
        if not left.guarded_gravity(assist.tolist(), guard):
            raise RuntimeError('Left guarded load command rejected')
        if not left.gripper_control(float(target[6]), CONTRACT['gripper_speed_rad_s'],
                                    CONTRACT['gripper_torque_limit_nm']):
            raise RuntimeError('Left gripper hold rejected')
        self.command_arm('right', 7, target)
        try:
            self.motor_client.flush()
        except RuntimeError as original:
            # A guard may trip between this tick's read and command submission.
            # Verify that exact session's stop; transport/other failures retain
            # the ordinary fault behavior rather than being silently swallowed.
            try:
                status = self.motor_client.rpc('status')
                stopped = status.get('guard_stop') or {}
                if status.get('error') or stopped.get('id') != guard['id']:
                    raise original
                age_ns = time.monotonic_ns() - int(status['sampled_monotonic_ns'])
                if not 0 <= age_ns <= 100_000_000:
                    raise original
                measured = np.asarray([v for side in ('left', 'right') for v in
                    status['arms'][side]['position_rad'] +
                    [status['arms'][side]['gripper']['position']]], dtype=float)
                if measured.shape != (14,) or not np.isfinite(measured).all():
                    raise original
            except (RuntimeError, ValueError, TypeError, KeyError):
                raise original
            raise GuardedLoadStopped(stopped, measured) from original

    def command_arm(self, side, offset, target, hold_gripper=True):
        robot = self.robots[side]
        if not robot.Joint_Pos_Vel(target[offset:offset + 6],
                                   np.full(6, CONTRACT['max_joint_speed_rad_s']),
                                   max_tqu=CONTRACT['torque_limits_nm'], iswait=False):
            raise RuntimeError(f'{side}: joint command rejected')
        if hold_gripper and not robot.gripper_control(float(target[offset + 6]),
                                     CONTRACT['gripper_speed_rad_s'],
                                     CONTRACT['gripper_torque_limit_nm']):
            raise RuntimeError(f'{side}: gripper command rejected')

    def recording_context(self, checkpoint):
        from motor_control import setup, TORQUE_LIMIT
        import yaml
        return dict(mode='human_in_the_loop', checkpoint=checkpoint, contract=CONTRACT,
                    torque_limits_nm=TORQUE_LIMIT.tolist(),
                    calibration=json.loads((ROOT / 'config/calibration/setup.json').read_text()),
                    arm_mapping=json.loads((setup.CONFIG / 'arm-mapping.json').read_text()),
                    robot_configs={s: yaml.safe_load(p.read_text()) for s, p in self.configs.items()},
                    motor_configs={s: yaml.safe_load((setup.CONFIG / f'{s}-motors.yaml').read_text())
                                   for s in self.configs},
                    urdf_sources={s: Path(yaml.safe_load(p.read_text())['urdf']['file_path']).read_text()
                                  for s, p in self.configs.items()})

    def guide(self, arms, target, released):
        zero = np.zeros(6)
        from reset_grippers import command_gripper
        for side, robot in self.robots.items():
            if not released[side]:
                self.command_arm(side, 0 if side == 'left' else 7, target, hold_gripper=False)
                command_gripper(robot, arms[side], dict(mode='position_velocity',
                    control_mode='position_hold', position_rad=float(target[6 if side == 'left' else 13]),
                    speed_limit_rad_s=CONTRACT['gripper_speed_rad_s'],
                    torque_limit_nm=CONTRACT['gripper_torque_limit_nm']))
                arms[side]['control_mode'] = 'position_hold'
                continue
            torque = np.asarray(arms[side]['gravity_model_torque_nm'])
            if not robot.pos_vel_tqe_kp_kd(zero, zero, torque, zero, zero):
                raise RuntimeError(f'{side}: gravity command rejected')
            command_gripper(robot, arms[side])
            arms[side]['control_mode'] = 'gravity_assist'
            arms[side]['gravity_torque_command_nm'] = torque.tolist()
            arms[side]['command'] = dict(mode='gravity_assist', torque_nm=torque.tolist())
        if self.motor_client:
            self.motor_client.flush()

    def validate_reset_path(self, initial, targets):
        for side, target in targets.items():
            offset = {'left': 0, 'right': 7}[side]
            self.motor_client.validate_path(side, initial[offset:offset+6], target)

    def hold_available(self):
        """Preserve even a partially opened session after a startup fault."""
        # A failed frontend releases control. The backend holds fresh measured poses
        # independently, including after a partial frontend startup.
        if self.motor_client:
            self.motor_client.release()
            state = self.motor_client.rpc('status')
            if state['error'] or state['mode'] != 'holding':
                raise RuntimeError(state['error'] or 'Backend hold is not established')

    def close(self):
        try:
            if self.motor_client:
                self.motor_client.close()
            if self.cameras:
                self.cameras.close()
        finally:
            self.robots = {}
