"""Motor-controller client. Contains no SDK imports or motor device access.

The SDK-shaped arm adapter only stages a complete command frame. flush() sends
both arms atomically; motor scheduling, feedback checks and lease expiry belong
to the independent backend.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOCKET = ROOT / 'logs/motor-controller.sock'
PROTOCOL_VERSION = 1


class MotorClient:
    def __init__(self, name, path=SOCKET):
        self.name = name
        self.path = Path(path)
        self.socket = self.stream = None
        self.token = None
        self.robots = {}
        self.frames = {}
        self.arms = {}
        self.lock = threading.RLock()
        self.configs = {}

    def connect(self):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(3)
        try:
            self.socket.connect(str(self.path))
        except OSError:
            self.socket.close()
            self.socket = None
            raise
        self.stream = self.socket.makefile('rwb')

    def open(self, autostart=True, check_active=lambda: None):
        try:
            self.connect()
        except (FileNotFoundError, ConnectionRefusedError):
            if not autostart:
                raise RuntimeError('Start app/motor_backend.py --run before connecting')
            (ROOT / 'logs').mkdir(exist_ok=True)
            # A separate user service also survives a frontend service's cgroup
            # cleanup. setsid alone would not provide that isolation.
            if not shutil.which('systemd-run'):
                raise RuntimeError('Start app/motor_backend.py --run in a separate terminal')
            log = ROOT / 'logs/motor-controller.log'
            process = subprocess.Popen([
                'systemd-run', '--user', '--unit=panthera-motor-controller', '--collect',
                '--property=Restart=no', '--property=TimeoutStopSec=10',
                f'--property=StandardOutput=append:{log}',
                f'--property=StandardError=append:{log}',
                sys.executable, '-u', str(ROOT / 'app/motor_backend.py'),
                '--run', '--socket', str(self.path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 25
            while True:
                check_active()
                try:
                    self.connect()
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline or process.poll() not in (None, 0):
                        raise RuntimeError('Motor backend startup failed; see logs/motor-controller.log or start app/motor_backend.py --run in a separate terminal')
                    time.sleep(.1)
        try:
            check_active()
            description = self.rpc('acquire', name=self.name)
            self.token = description['token']
            self.configs = {s: Path(p) for s, p in description['configs'].items()}
            self.robots = {s: MotorArm(self, s, limits) for s, limits in description['limits'].items()}
            self.read()
            return self
        except BaseException:
            self.close()
            raise

    def rpc(self, operation, **parameters):
        with self.lock:
            if not self.stream:
                raise RuntimeError('Motor backend is not connected')
            request = dict(version=PROTOCOL_VERSION, operation=operation, token=self.token, **parameters)
            try:
                self.stream.write((json.dumps(request, allow_nan=False, default=lambda x: x.tolist()) + '\n').encode())
                self.stream.flush()
                line = self.stream.readline(65537)
                if not line or len(line) > 65536:
                    raise RuntimeError('Motor backend disconnected')
                response = json.loads(line)
            except (OSError, ValueError) as exc:
                # Discard the stream: a late response cannot answer another RPC.
                self.stream.close()
                self.socket.close()
                self.stream = self.socket = None
                raise RuntimeError(f'Motor backend connection failed: {exc}') from exc
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response['result']

    def read(self):
        status = self.rpc('status')
        if status['error']:
            raise RuntimeError(status['error'])
        if time.monotonic_ns() - status['sampled_monotonic_ns'] > 500_000_000:
            raise RuntimeError('Motor backend feedback is stale')
        self.arms = status['arms']
        # Never reuse an old command for an omitted arm on the next frame.
        self.frames = {}
        return {s: (np.asarray(a['gravity_model_torque_nm']), deepcopy(a))
                for s, a in self.arms.items()}

    def flush(self):
        result = self.rpc('command', commands=self.frames)
        self.frames = {}
        return result

    def validate_path(self, side, initial, target):
        return self.rpc('validate_path', side=side, initial=initial, target=target)

    def release(self):
        """Revoke motion while keeping the status connection available."""
        try:
            if self.token and self.stream:
                self.rpc('release')
        except RuntimeError:
            pass  # An expired token is already revoked; status verifies the hold.
        finally:
            self.token = None
            self.frames = {}

    def close(self):
        with self.lock:
            try:
                self.release()
            finally:
                self.token = None
                if self.stream:
                    self.stream.close()
                if self.socket:
                    self.socket.close()
                self.stream = self.socket = None
                self.frames = {}


class MotorArm:
    """Compatibility adapter for existing trajectory/gripper producers."""
    def __init__(self, client, side, limits):
        self.client = client
        self.side = side
        self.joint_limits = {k: np.asarray(v) for k, v in limits['joint_limits'].items()}
        self.gripper_limits = limits['gripper_limits']

    def stage(self, part, command):
        self.client.frames.setdefault(self.side, {})[part] = command
        return True

    def Joint_Pos_Vel(self, position, velocity, max_tqu, iswait=False):
        if iswait:
            raise ValueError('Motor clients only support asynchronous targets')
        return self.stage('joint', dict(mode='position', position_rad=np.asarray(position).tolist(),
            speed_rad_s=float(np.max(velocity)), torque_nm=np.asarray(max_tqu).tolist()))

    def pos_vel_tqe_kp_kd(self, position, velocity, torque, kp, kd):
        if any(np.any(np.asarray(v) != 0) for v in (position, velocity, kp, kd)):
            raise ValueError('Only gravity assistance is supported by this adapter')
        gravity = np.asarray(self.client.arms[self.side]['gravity_model_torque_nm'])
        return self.stage('joint', dict(mode='gravity',
            assist_torque_nm=(np.asarray(torque) - gravity).tolist()))

    def gripper_control(self, position, speed, torque):
        return self.stage('gripper', dict(mode='position', position_rad=float(position),
            speed_rad_s=float(speed), torque_nm=float(torque)))

    def gripper_control_MIT(self, *values):
        if len(values) != 5 or any(v != 0 for v in values):
            raise ValueError('Only passive gripper MIT commands are supported')
        return self.stage('gripper', {'mode': 'passive'})

    def set_stop(self):
        # Stopping a frontend releases its lease; the backend retains the hold.
        self.client.close()
