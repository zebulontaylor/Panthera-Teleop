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
from urllib.parse import parse_qs, urlsplit

import numpy as np
import yaml

from cartesian_nudge import ToolKinematics
from dashboard_sequence import Cameras
from foot_pedal import FootPedal
from inference_pedal import InferencePedal
from motor_client import MotorClient
from pin_regrasp_motion import RegraspConfig, plan_regrasp, plan_alignment, TRAINING_PHASES
from pin_regrasp_recording import PinRegraspEpisodeStore
from robot_hardware import RobotHardware

ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROLES = ('left_wrist', 'right_wrist', 'overhead')

EDITABLE = {'task', 'retreat_m', 'cone_deg', 'angle_jitter_deg', 'tool_axis', 'speed_m_s',
            'lift_m', 'placement_jitter_m', 'carrier_jitter_m', 'base_elevation_deg', 'alignment_yaw_deg',
            'joint_speed_rad_s', 'gripper_speed_rad_s', 'opening_speed_rad_s'}


def camera_health(mapping, cameras):
    """Translate capture health to task roles without reading or opening devices."""
    health = {}
    for role, name in mapping.items():
        receipt = cameras.get(name)
        status = dict(receipt or {})
        status.update(camera=role, source_camera=name, ok=bool(receipt and receipt.get('ok')),
                      missing=receipt is None)
        health[role] = status
    return health


def camera_failure(health):
    failures = []
    for role, status in health.items():
        if status['ok']:
            continue
        details = []
        if status['missing']:
            details.append('missing stream status/receipt')
        elif status.get('age_s') is None:
            details.append('no captured frame receipt')
        else:
            details.append(f"frame age {status['age_s']:.3f}s (freshness limit 1.000s)")
        if status.get('error'):
            details.append('capture error: ' + str(status['error']))
        failures.append(f"{role} ({status['source_camera']}): " + '; '.join(details))
    return 'Collection camera is stale or unavailable: ' + ' | '.join(failures) if failures else None


class RegraspHardware(RobotHardware):
    def open(self):
        self.config = json.loads((ROOT / 'config/recording.json').read_text())
        roles = CAMERA_ROLES if getattr(self, 'task', 'regrasp') == 'alignment' else CAMERA_ROLES[:1]
        mapping = self.config.get('alignment_assist', {}).get('camera_mapping', {})
        self.camera_mapping = {role: mapping.get(role) for role in roles}
        names = list(self.camera_mapping.values())
        if (any(not isinstance(name, str) or not name or name not in self.config['cameras']
                for name in names) or len(set(names)) != len(names)):
            raise ValueError('Collection camera mapping requires distinct configured cameras for ' + ', '.join(roles))
        self.camera_name = self.camera_mapping['left_wrist']
        self.config['cameras'] = {name: self.config['cameras'][name] for name in names}
        self.cameras = Cameras(type('Sink', (), {'submit': lambda *_: None})(), self.config)
        self.cameras.wait_ready()
        self.motor_client = MotorClient('pin-regrasp')
        self.motor_client.open(check_active=self.check_active)
        self.robots, self.configs = self.motor_client.robots, self.motor_client.configs
        self.lower = np.concatenate([np.r_[self.robots[s].joint_limits['lower'],
            self.robots[s].gripper_limits['lower']] for s in ('left', 'right')])
        self.upper = np.concatenate([np.r_[self.robots[s].joint_limits['upper'],
            self.robots[s].gripper_limits['upper']] for s in ('left', 'right')])

    def frames(self):
        # Copy all receipts under one lock without mutating the capture cache.
        with self.cameras.lock:
            return {role: dict(self.cameras.latest[name], camera=role, source_camera=name)
                    for role, name in self.camera_mapping.items()}

    def frame(self):
        return self.frames()['left_wrist']


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
                          anchor=None, pedal_connected=False, camera_health={})

    def output_directory(self):
        if self.settings['task'] == 'alignment':
            return (self.root.with_name('pin-alignment-forward')
                    if self.root.name == 'pin-regrasp-forward' else self.root / 'alignment')
        return self.root

    def snapshot(self):
        with self.lock:
            return dict(self.state, settings=dict(self.settings),
                        camera_roles=list(CAMERA_ROLES if self.settings['task'] == 'alignment' else CAMERA_ROLES[:1]),
                        carrier_frame_mapping='parallel_bases',
                        recording=self.recorder.snapshot() if self.recorder else None,
                        directory=str(self.output_directory()))

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
                if values['task'] != self.settings['task']:
                    if self.worker and self.worker.is_alive():
                        raise ValueError('Disconnect before changing collection task')
                    self.state.update(anchor=None, phase=None, cycle=0)
                    self.recorder = None
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

    def jpeg(self, camera='left_wrist'):
        if camera not in CAMERA_ROLES:
            raise ValueError('Unknown camera role')
        hw = self.hardware
        name = getattr(hw, 'camera_mapping', {}).get(camera) if hw else None
        return hw.cameras.jpeg(name) if name and hw.cameras else None

    def run(self):
        hw, pedal, pool = self.hardware_factory(), None, ThreadPoolExecutor(max_workers=1)
        self.hardware = hw
        self.update(mode='connecting', error=None, anchor=None, cycle=0, camera_health={})
        anchor = target = None
        right_kin = None
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
        def read_hardware():
            measured, arms, cameras = hw.read()
            self.update(camera_health=camera_health(hw.camera_mapping, cameras))
            return measured, arms, cameras
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
            hw.task = self.settings['task']
            hw.open()
            kin = ToolKinematics(yaml.safe_load(hw.configs['left'].read_text()))
            self.recorder = PinRegraspEpisodeStore(self.output_directory(), fps=30)
            pedal = FootPedal(hw.config.get('foot_pedal'))
            gestures = InferencePedal()
            measured, arms, cameras = read_hardware()
            target = bounded_pose(measured)
            self.update(mode=mode, closed_gripper_rad=float(hw.lower[6]))
            while not self.shutdown.is_set():
                started = time.monotonic()
                measured, arms, cameras = read_hardware()
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
                        raise ValueError('Capture the task reference position and angle first')
                    mode = 'running'
                sent = False
                if mode == 'running':
                    failure = camera_failure(camera_health(hw.camera_mapping, cameras))
                    if failure:
                        raise RuntimeError(failure)
                    if plan is None:
                        if future is None:
                            with self.lock:
                                cfg = RegraspConfig(**self.settings,
                                    open_gripper_rad=float(hw.upper[6]))
                            planned_start = target.copy()
                            if cfg.task != 'alignment':
                                planned_start[7:] = anchor[7:]
                            planner_kwargs = {}
                            if cfg.task == 'alignment' and (cfg.carrier_jitter_m or
                                    not np.array_equal(planned_start[7:13], anchor[7:13])):
                                # Parallel bases share axes; only displacement rotation is needed.
                                transform = np.eye(4)
                                if right_kin is None:
                                    right_kin = ToolKinematics(yaml.safe_load(hw.configs['right'].read_text()))
                                planner_kwargs = dict(right_kinematics=right_kin,
                                    T_left_base_right_base=transform)
                            cycle_metadata = dict(seed=cycle, anchor=anchor.tolist(),
                                cycle_start=planned_start.tolist(), settings=asdict(cfg),
                                source_commit='67ce4d4' if cfg.task == 'alignment' else '0607a3c',
                                task=cfg.task, camera_mapping=dict(hw.camera_mapping), action_semantics='joint_position_targets_rad',
                                closed_gripper_rad=float(hw.lower[6]),
                                carrier_frame_mapping='parallel_bases',
                                carrier_mapping_displacement_only=True,
                                T_left_base_right_base=(planner_kwargs['T_left_base_right_base'].tolist()
                                    if planner_kwargs else None))
                            planner = plan_alignment if cfg.task == 'alignment' else plan_regrasp
                            future = pool.submit(planner, anchor.copy(), kin, hw.lower.copy(),
                                                 hw.upper.copy(), seed=cycle, config=cfg,
                                                 current14=planned_start.copy(), **planner_kwargs)
                        if future.done():
                            plan, future, index = future.result(), None, 0
                            # Plan starts at the reference: no hidden jump after human intervention.
                            planning_joints = np.r_[0:6, 7:13] if cfg.task == 'alignment' else np.arange(6)
                            if np.max(np.abs(measured[planning_joints] - planned_start[planning_joints])) > .04:
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
                        delta = next_target - target
                        if cfg.task == 'alignment':
                            # Keep both arms on the same segment while rejoining after a pause.
                            moving = np.abs(delta) > 0
                            scale = min(1., float(np.min(limits[moving] / np.abs(delta[moving])))) if np.any(moving) else 1.
                            emitted = target + scale * delta
                        else:
                            emitted = target + np.clip(delta, -limits, limits)
                        # Use measured FK too: tracking lag must not close fingers
                        # before the physical tool reaches the last 2 cm.
                        if cfg.task == 'regrasp' and phase in TRAINING_PHASES and emitted[6] < target[6]:
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
                            frames = hw.frames() if cfg.task == 'alignment' else None
                            frame = None if frames is not None else hw.frame()
                            hw.command(emitted)
                            sent = True
                            if frames is not None:
                                self.recorder.append_frames(phase, measured.tolist(), emitted.tolist(), frames)
                            else:
                                frame['camera'] = 'left_wrist'
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
                    alignment = self.settings['task'] == 'alignment'
                    hw.guide(arms, target, {'left': True, 'right': alignment})
                    if alignment:
                        target = bounded_pose(measured)
                    else:
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
            parsed = urlsplit(self.path)
            path = parsed.path
            if path == '/':
                self.send(200, html, 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.send(200, controller.snapshot())
            elif path == '/camera.jpg':
                camera = parse_qs(parsed.query).get('camera', ['left_wrist'])[0]
                if camera not in CAMERA_ROLES:
                    return self.send(400, {'error': 'Unknown camera role'})
                jpeg = controller.jpeg(camera)
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
