"""Independent motor owner. Teleop/inference connect over a local JSON socket.

Run with --run to open motors. Only this process imports and calls the SDK.
Loss of a client revokes its lease and holds measured joint positions.
"""
import argparse
from copy import deepcopy
from collections import deque
import fcntl
import json
from pathlib import Path
import secrets
import signal
import socketserver
import sys
import threading
import time

import numpy as np

from motor_control import (setup, ZERO, TORQUE_LIMIT, feedback, stop_robots,
                           validate_return_path)
from usb_stability import keep_usb_awake

ROOT = Path(__file__).resolve().parents[1]
SOCKET = ROOT / 'logs/motor-controller.sock'
SETTINGS = ROOT / 'config/motor-controller.json'
PROTOCOL_VERSION = 1
GUARDED_ASSIST_LIMIT = np.array([3., 6., 6., 1.5, 1.2, 1.2])
GUARD_CEILINGS = dict(max_xy_m=.030, max_down_m=.005, max_up_m=.002,
                      max_speed_m_s=.050, max_angular_rad=.1, max_duration_s=300.)
CONTACT_APPROACH_CEILINGS = {**GUARD_CEILINGS,
    'max_xy_m': .005, 'max_down_m': .020, 'max_duration_s': 30.}
LOADED_INSERTION_CEILINGS = {**GUARD_CEILINGS, 'max_up_m': .005}
GUARDED_FRESH_SECONDS = .1
GUARD_SPEED_WINDOW_SECONDS = .05
GUARD_POSITION_NOISE_M = .0003


def vector(value, size, name):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'{name} must be a numeric vector')
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'Invalid {name}')
    return result


def scalar(value, ceiling, name):
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= ceiling:
        raise ValueError(f'{name} exceeds backend limits')
    return value


class MotorEngine:
    """Stable controller boundary. SDK access is confined to tick/open/close."""
    def __init__(self, robots, settings, configs=None, clock=time.monotonic):
        self.robots = robots
        self.settings = settings
        self.configs = configs or {}
        self.clock = clock
        self.lock = threading.RLock()
        self.stale = {}
        self.arms = {}
        self.commands = None
        self.hold_targets = {}
        self.owner = None
        self.owner_name = None
        self.renewed = 0
        self.error = None
        self.sequence = 0
        self.sampled_ns = 0
        self.path_checks = []
        self.mode = 'holding'
        self.guard_stop = None
        self.active_guards = {}
        self.latched_guard_ids = set()
        self.retained_grippers = {}
        self.previous_tools = {}
        self.tool_history = {}
        self.sampled_clock = None
        self._has_commanded = False
        self._guard_rearm_required = False
        self._guard_cleanup_pending = False

    def description(self):
        return dict(version=PROTOCOL_VERSION,
            capabilities=dict(guarded_cartesian_load=True, guarded_contact_approach=True, guarded_continuous_insertion=True),
            guarded_load_limits=dict(assist_torque_nm=GUARDED_ASSIST_LIMIT.tolist(),
                **GUARD_CEILINGS, contact_approach=dict(CONTACT_APPROACH_CEILINGS),
                loaded_insertion=dict(LOADED_INSERTION_CEILINGS),
                command_fresh_seconds=GUARDED_FRESH_SECONDS,
                speed_guards_enabled=False,
                speed_window_seconds=GUARD_SPEED_WINDOW_SECONDS,
                position_noise_margin_m=GUARD_POSITION_NOISE_M), limits={side: dict(
            joint_limits={k: np.asarray(v).tolist() for k, v in robot.joint_limits.items()},
            gripper_limits={k: float(v) for k, v in robot.gripper_limits.items()})
            for side, robot in self.robots.items()},
            configs={s: str(p) for s, p in self.configs.items()}, settings=self.settings)

    @staticmethod
    def validate_guard(value):
        keys = {'id', 'anchor_tool_position_m', 'anchor_tool_rotation', *GUARD_CEILINGS}
        if not isinstance(value, dict) or set(value) not in (keys, keys | {'phase'}, keys | {'phase', 'transition_from'}):
            raise ValueError('Invalid guarded gravity guard fields')
        phase = value.get('phase')
        if 'phase' in value and phase not in ('contact_approach', 'loaded_insertion'):
            raise ValueError('Invalid guarded gravity phase')
        ceilings = (CONTACT_APPROACH_CEILINGS if phase == 'contact_approach' else
                    LOADED_INSERTION_CEILINGS if phase == 'loaded_insertion' else GUARD_CEILINGS)
        if 'transition_from' in value:
            previous = value['transition_from']
            if (phase != 'loaded_insertion' or not isinstance(previous, str) or
                    not previous.strip() or len(previous) > 128 or any(ord(c) < 32 for c in previous)):
                raise ValueError('Invalid guard transition_from')
        identity = value['id']
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 128 or any(ord(c) < 32 for c in identity):
            raise ValueError('Invalid guard id')
        position = vector(value['anchor_tool_position_m'], 3, 'guard anchor')
        rotation = np.asarray(value['anchor_tool_rotation'], dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError('Invalid guard anchor rotation')
        result = dict(id=identity, anchor_tool_position_m=position.tolist(),
                      anchor_tool_rotation=rotation.tolist())
        if phase is not None:
            result['phase'] = phase
        if 'transition_from' in value:
            result['transition_from'] = value['transition_from']
        for key, ceiling in ceilings.items():
            result[key] = scalar(value[key], ceiling, key)
        return result

    def validate_commands(self, commands):
        if not isinstance(commands, dict) or set(commands) != set(self.robots):
            raise ValueError('A complete command for both arms is required')
        result = deepcopy(commands)
        for side, command in result.items():
            if not isinstance(command, dict) or set(command) != {'joint', 'gripper'}:
                raise ValueError('Expected joint and gripper commands')
            robot = self.robots[side]
            joint = command['joint']
            if not isinstance(joint, dict):
                raise ValueError('Invalid joint command')
            if joint.get('mode') == 'guarded_gravity':
                if set(joint) != {'mode', 'assist_torque_nm', 'guard'}:
                    raise ValueError('Unknown guarded gravity fields')
                assist = vector(joint['assist_torque_nm'], 6, 'guarded assist torque')
                if np.any(np.abs(assist) > GUARDED_ASSIST_LIMIT):
                    raise ValueError('Guarded assist torque exceeds backend limits')
                joint['assist_torque_nm'] = assist.tolist()
                joint['guard'] = self.validate_guard(joint['guard'])
            elif joint.get('mode') == 'gravity':
                if set(joint) != {'mode', 'assist_torque_nm'}:
                    raise ValueError('Unknown gravity fields')
                assist = vector(joint['assist_torque_nm'], 6, 'assist torque')
                if np.any(np.abs(assist) > TORQUE_LIMIT):
                    raise ValueError('Assist torque exceeds backend limits')
            elif joint.get('mode') == 'position':
                if set(joint) != {'mode', 'position_rad', 'speed_rad_s', 'torque_nm'}:
                    raise ValueError('Unknown position fields')
                q = vector(joint['position_rad'], 6, 'joint target')
                if np.any(q < robot.joint_limits['lower']) or np.any(q > robot.joint_limits['upper']):
                    raise ValueError('Joint target outside backend limits')
                scalar(joint['speed_rad_s'], self.settings['max_joint_speed_rad_s'], 'Joint speed')
                torque = vector(joint['torque_nm'], 6, 'torque caps')
                if np.any(torque <= 0) or np.any(torque > self.settings['joint_torque_limits_nm']):
                    raise ValueError('Joint torque caps exceed backend limits')
            else:
                raise ValueError('Unknown joint mode')
            gripper = command['gripper']
            if not isinstance(gripper, dict):
                raise ValueError('Invalid gripper command')
            if gripper.get('mode') == 'passive':
                if set(gripper) != {'mode'}:
                    raise ValueError('Unknown passive gripper fields')
            elif gripper.get('mode') == 'position':
                if set(gripper) != {'mode', 'position_rad', 'speed_rad_s', 'torque_nm'}:
                    raise ValueError('Unknown gripper fields')
                q = gripper['position_rad']
                if type(q) not in (int, float) or not np.isfinite(q) or not robot.gripper_limits['lower'] <= q <= robot.gripper_limits['upper']:
                    raise ValueError('Gripper target outside backend limits')
                scalar(gripper['speed_rad_s'], self.settings['max_gripper_speed_rad_s'], 'Gripper speed')
                scalar(gripper['torque_nm'], self.settings['max_gripper_torque_nm'], 'Gripper torque')
            else:
                raise ValueError('Unknown gripper mode')
        if any(c['joint']['mode'] == 'guarded_gravity' for c in result.values()):
            if any(c['gripper']['mode'] != 'position' for c in result.values()):
                raise ValueError('Guarded gravity requires both commanded gripper holds')
            for command in result.values():
                if command['joint']['mode'] not in ('position', 'guarded_gravity'):
                    raise ValueError('Guarded gravity requires position hold on the other arm')
        return result

    def stop_guard(self, side, identity, reason):
        """Revoke load, not power. The next tick captures fresh measured holds."""
        for guarded_side, active in self.active_guards.items():
            self.latched_guard_ids.add(active['guard']['id'])
        self.latched_guard_ids.add(identity)
        if self.commands:
            self.retained_grippers = {s: deepcopy(c['gripper']) for s, c in self.commands.items()}
        self.guard_stop = dict(id=identity, side=side, reason=reason,
                               monotonic_ns=time.monotonic_ns())
        self.active_guards = {}
        self.commands = None
        self.hold_targets = {}
        self._guard_rearm_required = True
        self._guard_cleanup_pending = False
        self.mode = 'holding'

    def guard_temporal_reason(self, side, active):
        now = self.clock()
        if now - self.renewed > GUARDED_FRESH_SECONDS:
            return 'guard_command_stale'
        if now - active['started'] > active['guard']['max_duration_s']:
            return 'guard_duration_exceeded'
        if self.sampled_clock is None or now - self.sampled_clock > GUARDED_FRESH_SECONDS:
            return 'guard_feedback_stale'
        # The existing feedback watchdog records when each motor timestamp last
        # changed; use its host-monotonic clock, not device/wall timestamps.
        for index in range(7):
            previous = self.stale.get((side, index))
            if previous is not None and time.monotonic() - previous[1] > GUARDED_FRESH_SECONDS:
                return 'guard_motor_feedback_stale'
        return None

    def guard_motion_reason(self, side, active, now, check_deadline=True):
        arm = self.arms[side]
        pose = arm.get('tool_pose_base', {})
        try:
            position = np.asarray(pose.get('position_m'), dtype=float)
            rotation = np.asarray(pose.get('rotation'), dtype=float)
        except (TypeError, ValueError):
            return 'guard_invalid_tool_feedback'
        if position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            return 'guard_invalid_tool_feedback'
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
            return 'guard_invalid_tool_feedback'
        guard = active['guard']
        delta = position - np.asarray(guard['anchor_tool_position_m'])
        if np.linalg.norm(delta[:2]) > guard['max_xy_m']:
            return 'guard_xy_travel_exceeded'
        if -delta[2] > guard['max_down_m']:
            return 'guard_down_travel_exceeded'
        if delta[2] > guard['max_up_m']:
            return 'guard_up_travel_exceeded'
        relative = rotation @ np.asarray(guard['anchor_tool_rotation']).T
        angle = np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))
        if angle > guard['max_angular_rad']:
            return 'guard_orientation_exceeded'
        previous = self.previous_tools.get(side)
        # Guarded gravity loading does not stop on measured movement speed.
        # Keep the control deadline independent of velocity history.
        if check_deadline and previous is not None and not 0 < now - previous[0] <= GUARDED_FRESH_SECONDS:
            return 'guard_control_deadline'
        q = np.asarray(arm['position_rad'], dtype=float)
        lower, upper = (np.asarray(self.robots[side].joint_limits[key]) for key in ('lower', 'upper'))
        if np.any(q < lower + .01) or np.any(q > upper - .01):
            return 'guard_joint_limit_margin'
        return None

    def dispatch(self, request):
        if not isinstance(request, dict) or request.get('version') != PROTOCOL_VERSION:
            raise ValueError('Unsupported motor protocol version')
        operation = request.get('operation')
        with self.lock:
            self.expire_lease()
            if operation == 'status':
                return dict(self.description(), mode=self.mode, owner=self.owner_name,
                            error=self.error, sequence=self.sequence, sampled_monotonic_ns=self.sampled_ns,
                            arms=deepcopy(self.arms), guard_stop=deepcopy(self.guard_stop),
                            active_guards={s: a['guard']['id'] for s, a in self.active_guards.items()})
            if self.error:
                raise RuntimeError(self.error)
            if operation == 'acquire':
                if self.owner:
                    raise RuntimeError('Another frontend owns motor control')
                if not self.arms:
                    raise RuntimeError('Motor feedback is not ready')
                self.owner = secrets.token_urlsafe(32)
                self.owner_name = str(request.get('name', 'client'))[:80]
                self.commands = None
                self.renewed = self.clock()
                self.hold_targets = {}
                self._has_commanded = False
                return dict(self.description(), token=self.owner)
            if not self.owner or request.get('token') != self.owner:
                raise RuntimeError('Motor control lease expired or belongs to another frontend')
            if operation == 'release':
                self.revoke()
                return {'released': True}
            if operation == 'command':
                commands = self.validate_commands(request.get('commands'))
                proposed = {s: c['joint']['guard'] for s, c in commands.items()
                            if c['joint']['mode'] == 'guarded_gravity'}
                all_position = all(c['joint']['mode'] == 'position' for c in commands.values())
                if self._guard_rearm_required and not proposed and not all_position:
                    raise RuntimeError('Execute an explicit position hold after a guard stop')
                if proposed and self.path_checks:
                    raise RuntimeError('Wait for pending path checks before guarded gravity')
                if self.active_guards and proposed and set(proposed) != set(self.active_guards):
                    raise RuntimeError('End all active guards with an explicit position hold')
                transitions = set()
                for side, guard in proposed.items():
                    if guard['id'] in self.latched_guard_ids:
                        raise RuntimeError('Guard id is latched; explicitly hold and use a new guard id')
                    active = self.active_guards.get(side)
                    if active is not None:
                        if active['guard'] != guard:
                            old = active['guard']
                            if not (old.get('phase') == 'contact_approach' and
                                    guard.get('phase') == 'loaded_insertion' and
                                    guard.get('transition_from') == old['id'] and guard['id'] != old['id']):
                                raise RuntimeError('Active guard is immutable; invalid continuous transition')
                            reason = (self.guard_temporal_reason(side, active) or
                                      self.guard_motion_reason(side, active, self.clock(), check_deadline=False))
                            if reason:
                                raise RuntimeError('Continuous transition rejected: ' + reason)
                            pose = self.arms[side]['tool_pose_base']
                            distance = np.linalg.norm(np.asarray(pose['position_m']) - guard['anchor_tool_position_m'])
                            relative = np.asarray(pose['rotation']) @ np.asarray(guard['anchor_tool_rotation']).T
                            angle = np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))
                            if distance > .001 or angle > .01:
                                raise RuntimeError('Continuous transition anchor must match measured tool pose')
                            transitions.add(side)
                    elif 'transition_from' in guard:
                        raise RuntimeError('Continuous transition requires an active contact approach')
                    elif self.active_guards or self._guard_rearm_required:
                        raise RuntimeError('Execute an explicit position hold before a new guard')
                    elif self.sampled_clock is None or self.clock() - self.sampled_clock > GUARDED_FRESH_SECONDS:
                        raise RuntimeError('Fresh measured hold is required before guarded gravity')
                # An explicit position frame is the only ordinary way to end a
                # guard and acknowledge its stop before a different id starts.
                if self.active_guards and not proposed:
                    if not all_position:
                        raise RuntimeError('End guarded gravity with an explicit position hold')
                    for active in self.active_guards.values():
                        self.latched_guard_ids.add(active['guard']['id'])
                    self.active_guards = {}
                    self._guard_rearm_required = True
                for side, guard in proposed.items():
                    if side in transitions:
                        self.latched_guard_ids.add(self.active_guards[side]['guard']['id'])
                    if side not in self.active_guards or side in transitions:
                        self.active_guards[side] = dict(guard=deepcopy(guard), started=self.clock())
                if proposed:
                    self.retained_grippers = {s: deepcopy(c['gripper']) for s, c in commands.items()}
                    self.guard_stop = None
                elif self.retained_grippers:
                    self.retained_grippers = {s: deepcopy(c['gripper']) for s, c in commands.items()}
                if all_position:
                    self._guard_cleanup_pending = True
                # Validation of the entire frame precedes any change or renewal.
                self.commands = commands
                self.renewed = self.clock()
                self._has_commanded = True
                return dict(accepted=True, status='queued')
            if operation == 'validate_path':
                if self.active_guards:
                    raise RuntimeError('Path checks are unavailable during guarded gravity')
                side = request.get('side')
                if side not in self.robots:
                    raise ValueError('Unknown arm')
                initial = vector(request.get('initial'), 6, 'initial pose')
                target = vector(request.get('target'), 6, 'target pose')
                done = threading.Event()
                outcome = []
                self.path_checks.append((side, initial, target, done, outcome))
            else:
                raise ValueError('Unknown motor operation')
        # FK remains on the motor thread, never the socket handler.
        if not done.wait(2):
            raise RuntimeError('Motor path check timed out')
        if outcome:
            raise ValueError(outcome[0])
        return {'valid': True}

    def expire_lease(self):
        for side, active in list(self.active_guards.items()):
            reason = self.guard_temporal_reason(side, active)
            if reason:
                self.stop_guard(side, active['guard']['id'], reason)
                break
        lease = (self.settings['command_lease_seconds'] if self.commands or self._has_commanded else
                 self.settings.get('startup_lease_seconds', self.settings['command_lease_seconds']))
        if self.owner and self.clock() - self.renewed > lease:
            self.revoke()

    def revoke(self):
        if self.active_guards:
            side, active = next(iter(self.active_guards.items()))
            self.stop_guard(side, active['guard']['id'], 'guard_lease_released')
        self.owner = self.owner_name = self.commands = None
        self.hold_targets = {}
        self.mode = 'holding'

    def tick(self):
        with self.lock:
            if self.error:
                return
            try:
                samples = {s: feedback(r, s, self.stale) for s, r in self.robots.items()}
                self.arms = {s: arm for s, (_, arm) in samples.items()}
                self.sampled_ns = time.monotonic_ns()
                self.sampled_clock = self.clock()
                self.sequence += 1
                while self.path_checks:
                    side, initial, target, done, outcome = self.path_checks.pop(0)
                    try:
                        validate_return_path(self.robots[side], initial, target, side)
                    except Exception as exc:
                        outcome.append(str(exc))
                    finally:
                        done.set()
                self.expire_lease()
                # Preflight every active guard BEFORE commanding either arm.
                # A routine bounded-load stop holds fresh measured positions,
                # preserves both squeezes and does not power down the robot.
                for side, active in list(self.active_guards.items()):
                    reason = self.guard_motion_reason(side, active, self.sampled_clock)
                    if reason is None:
                        total = samples[side][0] + np.asarray(self.commands[side]['joint']['assist_torque_nm'])
                        if np.any(np.abs(total) > np.asarray(self.settings['joint_torque_limits_nm'])):
                            reason = 'guard_total_torque_exceeded'
                    if reason:
                        self.stop_guard(side, active['guard']['id'], reason)
                        break
                self.mode = 'client_control' if self.owner and self.commands else 'holding'
                for side, robot in self.robots.items():
                    torque, arm = samples[side]
                    frame = self.commands[side] if self.owner and self.commands else None
                    joint = frame['joint'] if frame else None
                    if joint and joint['mode'] == 'guarded_gravity':
                        # Validated above against the conservative TOTAL caps;
                        # never silently clip a Cartesian load into another one.
                        torque = torque + np.asarray(joint['assist_torque_nm'])
                        accepted = robot.pos_vel_tqe_kp_kd(ZERO, ZERO, torque, ZERO, ZERO)
                    elif joint and joint['mode'] == 'gravity':
                        torque = np.clip(torque + np.asarray(joint['assist_torque_nm']), -TORQUE_LIMIT, TORQUE_LIMIT)
                        accepted = robot.pos_vel_tqe_kp_kd(ZERO, ZERO, torque, ZERO, ZERO)
                    else:
                        if joint:
                            q, speed, caps = joint['position_rad'], joint['speed_rad_s'], joint['torque_nm']
                        else:
                            q = self.hold_targets.setdefault(side, arm['position_rad'])
                            speed = self.settings['disconnect_hold_speed_rad_s']
                            caps = self.settings['joint_torque_limits_nm']
                        accepted = robot.Joint_Pos_Vel(np.asarray(q), np.full(6, speed), max_tqu=caps, iswait=False)
                    if not accepted:
                        raise RuntimeError(f'{side}: backend joint command rejected')
                    gripper = frame['gripper'] if frame else self.retained_grippers.get(side, {'mode': 'passive'})
                    if gripper['mode'] == 'passive':
                        accepted = robot.gripper_control_MIT(0, 0, 0, 0, 0)
                    else:
                        accepted = robot.gripper_control(gripper['position_rad'], gripper['speed_rad_s'], gripper['torque_nm'])
                    if not accepted:
                        raise RuntimeError(f'{side}: backend gripper command rejected')
                if self._guard_cleanup_pending and self.commands and all(c['joint']['mode'] == 'position' for c in self.commands.values()):
                    self._guard_rearm_required = False
                    self._guard_cleanup_pending = False
                for side, arm in self.arms.items():
                    pose = arm.get('tool_pose_base', {})
                    position = np.asarray(pose.get('position_m'), dtype=float)
                    if position.shape == (3,) and np.isfinite(position).all():
                        self.previous_tools[side] = (self.sampled_clock, position.copy())
                        history = self.tool_history.setdefault(side, deque(maxlen=64))
                        history.append((self.sampled_clock, position.copy()))
                        while history and self.sampled_clock - history[0][0] > .15:
                            history.popleft()
            except Exception as exc:
                self.error = str(exc)
                self.mode = 'fault'
                self.revoke()
                self.mode = 'fault'
                stop_robots(self.robots)


class MotorServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            line = self.rfile.readline(65537)
            if not line or len(line) > 65536:
                return
            try:
                request = json.loads(line)
                if (isinstance(request, dict) and request.get('version') == PROTOCOL_VERSION
                        and request.get('operation') == 'shutdown'):
                    self.server.stopping.set()
                    result = {'stopping': True}
                else:
                    result = self.server.engine.dispatch(request)
                response = {'result': result}
            except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                response = {'error': str(exc)}
            try:
                self.wfile.write((json.dumps(response, allow_nan=False, default=lambda v: v.tolist()) + '\n').encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Open motors and run the backend')
    parser.add_argument('--socket', type=Path, default=SOCKET)
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='both')
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required to enable motor ownership')
    import yaml
    robots = {}
    server = None
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    (ROOT / 'logs').mkdir(exist_ok=True)
    with (ROOT / 'logs/control.lock').open('a') as control_lock:
        try:
            fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another motor controller owns the robot devices')
        try:
            settings = json.loads(SETTINGS.read_text())
            sides = ('left', 'right') if args.arm == 'both' else (args.arm,)
            configs = setup.prepare(sides)
            sys.path.insert(0, str(setup.SDK / 'scripts'))
            from Panthera_lib import Panthera
            mapping = json.loads((setup.CONFIG / 'arm-mapping.json').read_text())
            for side in sides:
                if stopping.is_set():
                    return
                expected = yaml.safe_load((setup.CONFIG / f'{side}-motors.yaml').read_text())['robot']['Serial_Type']
                if str(setup.resolve_board(mapping[side])) != expected:
                    raise RuntimeError('USB mapping changed during backend startup')
                keep_usb_awake(expected)
                robots[side] = Panthera(str(configs[side]))
                setup.wait_for_joint_feedback(robots[side], side)
            engine = MotorEngine(robots, settings, configs)
            engine.tick()
            if engine.error:
                raise RuntimeError(engine.error)
            args.socket.unlink(missing_ok=True)
            server = MotorServer(str(args.socket), Handler)
            args.socket.chmod(0o600)
            server.engine = engine
            server.stopping = stopping
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f'Motor backend ready: {args.socket}', flush=True)
            while not stopping.is_set():
                started = time.monotonic()
                engine.tick()
                stopping.wait(max(0, 1 / settings['control_hz'] - (time.monotonic() - started)))
        finally:
            if server:
                server.shutdown()
                server.server_close()
                args.socket.unlink(missing_ok=True)
            stop_robots(robots)


if __name__ == '__main__':
    main()
