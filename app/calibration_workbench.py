#!/usr/bin/env python3
"""Separate local UI for four candidate calibration workflows.

Only the hold workflow can forward explicit leased sequence requests, and only
to a local collector exposing calibration.sequence. It never opens motor ports.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.error import URLError
import numpy as np

from calibration_workflow_common import CollectorClient, ROOT
from guided_hand_calibration import GuidedHandCalibration
from hold_run_calibration import HoldRunCalibration
from demonstrated_correction_calibration import DemonstratedCorrectionCalibration
from board_marker_calibration import BoardMarkerCalibration

WORKFLOWS = dict(guided=GuidedHandCalibration, hold=HoldRunCalibration,
    demonstration=DemonstratedCorrectionCalibration, board=BoardMarkerCalibration)


class Workbench:
    def __init__(self, workflow, client, port=8091):
        self.workflow, self.client = workflow, client
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.collector_error = 'Waiting for a healthy recording.'
        self.frames = {}
        self.prediction = None
        self.last_sample_at = None
        self.press_id = None
        self.press_episode = None
        self.browser_renewed = None
        html = (ROOT/'app/web/calibration-workbench.html').read_bytes()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def trusted(self):
                actual_port = owner.server.server_address[1]
                return self.headers.get('Host') in (f'127.0.0.1:{actual_port}', f'localhost:{actual_port}')

            def send(self, status, data, mime='application/json'):
                payload = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if not self.trusted():
                    return self.send(403, {})
                if self.path == '/':
                    return self.send(200, html, 'text/html; charset=utf-8')
                if self.path == '/api/status':
                    with owner.lock:
                        snapshot = dict(owner.workflow.status(), token=owner.token,
                            collector_error=owner.collector_error, frames=owner.frames,
                            prediction=owner.prediction, collector=owner.workflow.latest_status.get('active'),
                            fresh=owner.last_sample_at is not None and time.monotonic()-owner.last_sample_at < .3,
                            sequence=owner.workflow.latest_status.get('alignment_servo'),
                            sequence_gate=owner.workflow.latest_status.get('calibration_sequence', {'gate_ready': False, 'reason': 'sequence_collector_required'}),
                            holding=owner.browser_renewed is not None and time.monotonic()-owner.browser_renewed <= .2)
                    return self.send(200, snapshot)
                return self.send(404, {})

            def do_POST(self):
                actual_port = owner.server.server_address[1]
                if (not self.trusted() or self.headers.get('X-Session-Token') != owner.token
                        or self.headers.get('Origin') not in (None, f'http://127.0.0.1:{actual_port}', f'http://localhost:{actual_port}')):
                    return self.send(403, {'error': 'Invalid local session'})
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if self.path != '/api/action' or not 0 < size <= 4096:
                        raise ValueError('invalid_action_request')
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict) or not set(payload) <= {'action', 'settings', 'press_id'}:
                        raise ValueError('invalid_action_payload')
                    result = owner.action(payload)
                    self.send(200, result)
                except Exception as exc:
                    self.send(409, {'error': str(exc)})

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.worker = threading.Thread(target=self.poll, daemon=True)

    def action(self, payload):
        action = payload.get('action')
        with self.lock:
            if action in ('hold', 'release'):
                if self.workflow.name != 'hold':
                    raise ValueError('this_workflow_cannot_request_motion')
                press = payload.get('press_id')
                if not isinstance(press, str) or not 1 <= len(press) <= 80:
                    raise ValueError('explicit_press_id_required')
                if action == 'hold':
                    if self.workflow.phase in ('idle', 'complete', 'cancelled') or self.collector_error:
                        raise ValueError('begin_a_healthy_hold_session_first')
                    if not self.workflow.latest_status.get('calibration_sequence', {}).get('gate_ready'):
                        raise ValueError('independently_verified_sequence_collector_required')
                    axis = self.workflow.latest_status['calibration_sequence'].get('axis_base')
                    if axis is None or not np.allclose(axis, self.workflow.axis):
                        raise ValueError('workbench_and_verified_motion_axis_must_match')
                    if (self.press_id is not None and self.press_id != press and self.browser_renewed is not None
                            and time.monotonic()-self.browser_renewed <= .2):
                        raise ValueError('release_current_press_first')
                    self.press_id, self.press_episode = press, self.workflow.episode
                    result = self.client.sequence(True, press, self.press_episode)
                    self.browser_renewed = time.monotonic()
                    return result
                # Stop the heartbeat locally even if the collector is unreachable.
                if press == self.press_id:
                    self.browser_renewed = None
                return self.client.sequence(False, press, self.press_episode or self.workflow.episode)
            if self.browser_renewed is not None and time.monotonic()-self.browser_renewed <= .2:
                raise ValueError('release_sequence_before_changing_workflow')
            if action == 'begin' and (self.last_sample_at is None or time.monotonic()-self.last_sample_at > .3 or self.collector_error):
                raise ValueError('fresh_healthy_recording_required')
            self.workflow.command(action, payload.get('settings', {}))
            return dict(accepted=True, phase=self.workflow.phase, verified=False)

    def poll_once(self):
        sample = self.client.sample_board if self.workflow.name == 'board' else self.client.sample
        row, frames, collector = sample()
        with self.lock:
            self.frames = frames
            self.prediction = {key: row.get(key) for key in ('predictions', 'camera_mapping', 'monotonic_ns')}
            self.workflow.record_observation(row)
            self.workflow.observe(row, frames, collector)
            self.last_sample_at = time.monotonic()
            self.collector_error = None

    def poll(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self.poll_once()
            except Exception as exc:
                with self.lock:
                    self.collector_error = str(exc)
                    self.last_sample_at = None
            self.stop_event.wait(max(.01, .1-(time.monotonic()-started)))

    def close(self):
        self.stop_event.set()
        if self.press_id is not None:
            try:
                self.client.sequence(False, self.press_id, self.press_episode)
            except (OSError, ValueError, KeyError):
                pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.worker.is_alive():
            self.worker.join(timeout=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workflow', choices=WORKFLOWS, required=True)
    p.add_argument('--collector', default='http://127.0.0.1:8090')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path)
    p.add_argument('--port', type=int, default=8091)
    args = p.parse_args()
    output = args.output or ROOT/'logs/calibration-workflows'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+args.workflow+'-'+secrets.token_hex(3))
    workflow = WORKFLOWS[args.workflow](output)
    workbench = Workbench(workflow, CollectorClient(args.collector, args.data_root), args.port)
    workbench.worker.start()
    print(f'Calibration workbench: http://127.0.0.1:{workbench.server.server_address[1]}', flush=True)
    print(f'Output: {output}. Candidate reports never activate motion.', flush=True)
    try:
        while True:
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        workbench.close()


if __name__ == '__main__':
    main()
