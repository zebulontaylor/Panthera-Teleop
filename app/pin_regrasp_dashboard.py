"""Supervised autonomous regrasp collection, using the independent motor backend."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import secrets
import signal
import threading
import time

import numpy as np
import yaml

from cartesian_nudge import ToolKinematics
from dashboard_sequence import Cameras
from foot_pedal import FootPedal
from inference_pedal import InferencePedal
from motor_client import MotorClient
from pin_regrasp_motion import RegraspConfig, plan_regrasp, TRAINING_PHASES
from pin_regrasp_recording import PinRegraspEpisodeStore
from robot_hardware import RobotHardware

ROOT = Path(__file__).resolve().parents[1]
EDITABLE = {'retreat_m', 'cone_deg', 'angle_jitter_deg', 'tool_axis', 'speed_m_s',
            'lift_m', 'placement_jitter_m', 'base_elevation_deg',
            'joint_speed_rad_s', 'gripper_speed_rad_s', 'opening_speed_rad_s'}


class RegraspHardware(RobotHardware):
    def open(self):
        self.config = json.loads((ROOT / 'config/recording.json').read_text())
        name = self.config['alignment_assist']['camera_mapping']['left_wrist']
        self.camera_name = name
        self.config['cameras'] = {name: self.config['cameras'][name]}
        self.cameras = Cameras(type('Sink', (), {'submit': lambda *_: None})(), self.config)
        self.cameras.wait_ready()
        self.motor_client = MotorClient('pin-regrasp')
        self.motor_client.open(check_active=self.check_active)
        self.robots, self.configs = self.motor_client.robots, self.motor_client.configs
        self.lower = np.concatenate([np.r_[self.robots[s].joint_limits['lower'],
            self.robots[s].gripper_limits['lower']] for s in ('left', 'right')])
        self.upper = np.concatenate([np.r_[self.robots[s].joint_limits['upper'],
            self.robots[s].gripper_limits['upper']] for s in ('left', 'right')])

    def frame(self):
        with self.cameras.lock:
            return dict(self.cameras.latest[self.camera_name])


class Controller:
    def __init__(self, root, hardware_factory=RegraspHardware):
        self.root = Path(root)
        self.hardware_factory = hardware_factory
        self.lock = threading.RLock()
        self.commands = queue.Queue(maxsize=32)
        self.heartbeat_at = 0
        self.worker = None
        self.hardware = None
        self.recorder = None
        self.shutdown = threading.Event()
        self.settings = {k: v for k, v in asdict(RegraspConfig()).items() if k in EDITABLE}
        self.state = dict(mode='disconnected', error=None, phase=None, cycle=0,
                          anchor=None, pedal_connected=False)

    def snapshot(self):
        with self.lock:
            return dict(self.state, settings=dict(self.settings),
                        recording=self.recorder.snapshot() if self.recorder else None,
                        directory=str(self.root))

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def request(self, action, payload=None):
        payload = payload or {}
        if action == 'heartbeat':
            self.heartbeat_at = time.monotonic()
            return
        if action == 'settings':
            if not isinstance(payload, dict) or set(payload) - EDITABLE:
                raise ValueError('Unknown collection setting')
            with self.lock:
                values = dict(self.settings, **payload)
                try:
                    RegraspConfig(**values)  # Validate before changing live settings.
                except (TypeError, ValueError) as exc:
                    raise ValueError('Invalid collection settings: ' + str(exc)) from exc
                self.settings = values
            return
        if action == 'connect':
            with self.lock:
                if self.worker and self.worker.is_alive():
                    raise ValueError('Already connected')
                self.commands = queue.Queue(maxsize=32)
                self.shutdown.clear()
                self.heartbeat_at = time.monotonic()
                self.worker = threading.Thread(target=self.run, daemon=True)
                self.worker.start()
            return
        if action not in ('capture', 'adjust', 'pause', 'resume', 'disconnect'):
            raise ValueError('Unknown command')
        if action == 'disconnect':
            self.shutdown.set()
            return
        if not self.worker or not self.worker.is_alive():
            raise ValueError('Connect first')
        self.commands.put_nowait(action)

    def jpeg(self):
        hw = self.hardware
        return hw.cameras.jpeg(hw.camera_name) if hw and hw.cameras else None

    def run(self):
        hw, pedal, pool = self.hardware_factory(), None, ThreadPoolExecutor(max_workers=1)
        self.hardware = hw
        self.update(mode='connecting', error=None, anchor=None, cycle=0)
        anchor = target = None
        plan, future, index, cycle = None, None, 0, 0
        planned_start = None
        mode = 'holding'
        recording = False
        cycle_metadata = {}
        def finish(reason, complete=False):
            nonlocal recording
            if recording:
                self.recorder.finish(complete, reason)
                recording = False
        def bounded_pose(measured):
            pose = measured.copy()
            pose[[6, 13]] = np.clip(pose[[6, 13]], hw.lower[[6, 13]], hw.upper[[6, 13]])
            return pose
        def capture(measured):
            nonlocal anchor, plan, future, index
            anchor = bounded_pose(measured)
            plan, future, index = None, None, 0
            self.update(anchor=anchor.tolist())
        try:
            hw.check_active = lambda: self.check_active()
            hw.open()
            kin = ToolKinematics(yaml.safe_load(hw.configs['left'].read_text()))
            self.recorder = PinRegraspEpisodeStore(self.root, fps=30)
            pedal = FootPedal(hw.config.get('foot_pedal'))
            gestures = InferencePedal()
            measured, arms, cameras = hw.read()
            target = bounded_pose(measured)
            self.update(mode=mode, closed_gripper_rad=float(hw.lower[6]))
            while not self.shutdown.is_set():
                started = time.monotonic()
                measured, arms, cameras = hw.read()
                snapshot = pedal.snapshot()
                self.update(pedal_connected=snapshot['connected'])
                gesture = gestures.update(snapshot)
                action = None
                if gesture == 'single':
                    action = 'pause'
                elif gesture == 'double':
                    action = 'pause' if mode == 'running' else 'resume'
                elif gesture == 'review':
                    action = 'pause'
                if gesture in ('single', 'review'):
                    # A freeze cancels older queued controls, including resume.
                    while True:
                        try:
                            self.commands.get_nowait()
                        except queue.Empty:
                            break
                else:
                    try:
                        action = self.commands.get_nowait()
                    except queue.Empty:
                        pass
                if started - self.heartbeat_at > 2:
                    action = 'pause'
                if action in ('pause', 'adjust'):
                    finish(action)
                    previous_grippers = target[[6, 13]].copy()
                    target = bounded_pose(measured)
                    target[[6, 13]] = previous_grippers
                    mode = 'adjusting' if action == 'adjust' else 'paused'
                    if action == 'adjust':
                        plan, future, index = None, None, 0
                elif action == 'capture':
                    if mode == 'running':
                        raise ValueError('Pause or adjust before capturing the reference pose')
                    finish('capture')
                    capture(measured)
                    target, mode = anchor.copy(), 'holding'
                elif action == 'resume':
                    if mode == 'adjusting':
                        capture(measured)
                        target = anchor.copy()
                    if anchor is None:
                        raise ValueError('Capture the starting grasp position and angle first')
                    mode = 'running'
                sent = False
                if mode == 'running':
                    if not all(c['ok'] for c in cameras.values()):
                        raise RuntimeError('Left wrist camera is stale or unavailable')
                    if plan is None:
                        if future is None:
                            with self.lock:
                                cfg = RegraspConfig(**self.settings,
                                    open_gripper_rad=float(hw.upper[6]))
                            planned_start = target.copy()
                            planned_start[7:] = anchor[7:]
                            cycle_metadata = dict(seed=cycle, anchor=anchor.tolist(),
                                cycle_start=planned_start.tolist(), settings=asdict(cfg),
                                source_commit='0607a3c', closed_gripper_rad=float(hw.lower[6]))
                            future = pool.submit(plan_regrasp, anchor.copy(), kin, hw.lower.copy(),
                                                 hw.upper.copy(), seed=cycle, config=cfg,
                                                 current14=planned_start.copy())
                        if future.done():
                            plan, future, index = future.result(), None, 0
                            # Plan starts at the reference: no hidden jump after human intervention.
                            if np.max(np.abs(measured[:6] - planned_start[:6])) > .04:
                                raise RuntimeError('Arm moved while planning; adjust and capture again')
                    if plan is not None:
                        phase, next_target = plan[index]
                        joints = np.r_[0:6, 7:13]
                        if np.max(np.abs(measured[joints] - target[joints])) > .12:
                            raise RuntimeError('Joint tracking deviation')
                        if np.max(np.abs(next_target[joints] - target[joints])) > .025:
                            raise RuntimeError('Discontinuous trajectory; capture reference again')
                        # A measured-pose pause may leave a small tracking gap.
                        # Rejoin the pending waypoint at the same bounded speeds.
                        limits = np.full(14, cfg.joint_speed_rad_s / cfg.hz)
                        limits[[6, 13]] = (cfg.opening_speed_rad_s if phase == 'release'
                                            else cfg.gripper_speed_rad_s) / cfg.hz
                        emitted = target + np.clip(next_target-target, -limits, limits)
                        # Use measured FK too: tracking lag must not close fingers
                        # before the physical tool reaches the last 2 cm.
                        if phase in TRAINING_PHASES and emitted[6] < target[6]:
                            actual_p, _ = kin.pose(measured[:6])
                            grasp_p, _ = kin.pose(planned_start[:6])
                            if np.linalg.norm(actual_p-grasp_p) > cfg.closing_distance_m:
                                emitted[6] = target[6]
                        arrived = np.allclose(emitted, next_target, atol=1e-10, rtol=0)
                        if phase in TRAINING_PHASES:
                            if not recording:
                                self.recorder.begin(dict(cycle_metadata, cycle=cycle,
                                    resumed_mid_forward=index > 0 and plan[index-1][0] in TRAINING_PHASES))
                                recording = True
                            frame = hw.frame()
                            frame['camera'] = 'left_wrist'
                            hw.command(emitted)
                            sent = True
                            self.recorder.append(phase, measured.tolist(), emitted.tolist(),
                                                 frame.pop('jpeg'), frame)
                        else:
                            finish('forward_complete', complete=True)
                        target = emitted.copy()
                        self.update(phase=phase, cycle=cycle)
                        if arrived:
                            index += 1
                        if index == len(plan):
                            finish('forward_complete', complete=True)
                            plan, index = None, 0
                            cycle += 1
                if mode == 'adjusting':
                    hw.guide(arms, target, {'left': True, 'right': False})
                    target[:7] = bounded_pose(measured)[:7]
                elif not sent:
                    hw.command(target)
                self.update(mode=mode)
                self.shutdown.wait(max(0, 1/30 - (time.monotonic()-started)))
        except Exception as exc:
            self.update(mode='error', error=str(exc))
        finally:
            try:
                finish('session_stopped')
            finally:
                if pedal:
                    pedal.close()
                hw.close()
                self.hardware = None
                pool.shutdown(wait=False, cancel_futures=True)
                if self.state['mode'] != 'error':
                    self.update(mode='disconnected')

    def check_active(self):
        if self.shutdown.is_set() or time.monotonic() - self.heartbeat_at > 2:
            raise RuntimeError('Dashboard connection expired')


def serve(controller, port):
    token = secrets.token_urlsafe(32)
    html = (ROOT / 'app/web/pin-regrasp.html').read_text().replace('__TOKEN__', token).encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def send(self, code, body, mime='application/json'):
            data = json.dumps(body, allow_nan=False).encode() if mime == 'application/json' else body
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def trusted(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')
        def do_GET(self):
            if not self.trusted():
                return self.send(403, {'error': 'Invalid host'})
            path = self.path.split('?')[0]
            if path == '/':
                self.send(200, html, 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.send(200, controller.snapshot())
            elif path == '/camera.jpg':
                jpeg = controller.jpeg()
                self.send(200 if jpeg else 503, jpeg or b'', 'image/jpeg')
            else:
                self.send(404, {'error': 'Not found'})
        def do_POST(self):
            if not self.trusted() or self.headers.get('X-Control-Token') != token:
                return self.send(403, {'error': 'Invalid control token or host'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 <= length <= 8192:
                    raise ValueError('Request too large')
                payload = json.loads(self.rfile.read(length) or b'{}')
                if not self.path.startswith('/api/'):
                    raise ValueError('Unknown route')
                controller.request(self.path[5:], payload)
                self.send(200, controller.snapshot())
            except (ValueError, RuntimeError, queue.Full) as exc:
                self.send(400, {'error': str(exc)})
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Pin regrasp dashboard: http://127.0.0.1:{port}', flush=True)
    if threading.current_thread() is threading.main_thread():
        def stop_server(*_):
            controller.shutdown.set()
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop_server)
        signal.signal(signal.SIGINT, stop_server)
    try:
        server.serve_forever()
    finally:
        controller.shutdown.set()
        if controller.worker:
            controller.worker.join(timeout=5)
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8092)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/pin-regrasp-forward')
    args = parser.parse_args()
    serve(Controller(args.data_dir), args.port)
