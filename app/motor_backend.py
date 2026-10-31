"""Independent motor owner. Teleop/inference connect over a local JSON socket.

Run with --run to open motors. Only this process imports and calls the SDK.
Loss of a client revokes its lease and holds measured joint positions.
"""
import argparse
from copy import deepcopy
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

    def description(self):
        return dict(version=PROTOCOL_VERSION, limits={side: dict(
            joint_limits={k: np.asarray(v).tolist() for k, v in robot.joint_limits.items()},
            gripper_limits={k: float(v) for k, v in robot.gripper_limits.items()})
            for side, robot in self.robots.items()},
            configs={s: str(p) for s, p in self.configs.items()}, settings=self.settings)

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
            if joint.get('mode') == 'gravity':
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
        return result

    def dispatch(self, request):
        if not isinstance(request, dict) or request.get('version') != PROTOCOL_VERSION:
            raise ValueError('Unsupported motor protocol version')
        operation = request.get('operation')
        with self.lock:
            self.expire_lease()
            if operation == 'status':
                return dict(self.description(), mode=self.mode, owner=self.owner_name,
                            error=self.error, sequence=self.sequence, sampled_monotonic_ns=self.sampled_ns,
                            arms=deepcopy(self.arms))
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
                return dict(self.description(), token=self.owner)
            if not self.owner or request.get('token') != self.owner:
                raise RuntimeError('Motor control lease expired or belongs to another frontend')
            if operation == 'release':
                self.revoke()
                return {'released': True}
            if operation == 'command':
                commands = self.validate_commands(request.get('commands'))
                # Validation of the entire frame precedes any change or renewal.
                self.commands = commands
                self.renewed = self.clock()
                return dict(accepted=True, status='queued')
            if operation == 'validate_path':
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
        lease = (self.settings['command_lease_seconds'] if self.commands else
                 self.settings.get('startup_lease_seconds', self.settings['command_lease_seconds']))
        if self.owner and self.clock() - self.renewed > lease:
            self.revoke()

    def revoke(self):
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
                self.mode = 'client_control' if self.owner and self.commands else 'holding'
                for side, robot in self.robots.items():
                    torque, arm = samples[side]
                    frame = self.commands[side] if self.owner and self.commands else None
                    joint = frame['joint'] if frame else None
                    if joint and joint['mode'] == 'gravity':
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
                    gripper = frame['gripper'] if frame else {'mode': 'passive'}
                    if gripper['mode'] == 'passive':
                        accepted = robot.gripper_control_MIT(0, 0, 0, 0, 0)
                    else:
                        accepted = robot.gripper_control(gripper['position_rad'], gripper['speed_rad_s'], gripper['torque_nm'])
                    if not accepted:
                        raise RuntimeError(f'{side}: backend gripper command rejected')
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
