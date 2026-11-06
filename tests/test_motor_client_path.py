"""Concurrent path preflights using fake socket handlers, never motors/SDK."""
from contextlib import contextmanager
import json
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from motor_client import MotorClient


@contextmanager
def blocked_path_server(reject=False):
    entered, finish = threading.Event(), threading.Event()
    state = dict(owner=None, acquire_count=0, release_count=0, operations=[], path_token=None)

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while True:
                line = self.rfile.readline()
                if not line:
                    return  # EOF closes only this connection, not its lease.
                request = json.loads(line)
                operation = request['operation']
                state['operations'].append(operation)
                response = None
                if operation == 'acquire':
                    state['acquire_count'] += 1
                    state['owner'] = 'owner-token'
                    result = dict(token=state['owner'], configs={}, limits={},
                                  guarded_load_limits=state.get('guarded_load_limits', {}))
                elif operation == 'status':
                    result = dict(error=None, sampled_monotonic_ns=time.monotonic_ns(), arms={},
                                  owner=state['owner'], guarded_load_limits=state.get('guarded_load_limits', {}))
                elif request.get('token') != state['owner']:
                    response = {'error': 'Lease expired'}
                elif operation == 'validate_path':
                    state['path_token'] = request['token']
                    entered.set()
                    finish.wait(3)
                    if reject:
                        response = {'error': 'left: return path dips below clearance'}
                    else:
                        result = {'valid': True}
                elif operation == 'command':
                    result = {'accepted': True}
                elif operation == 'release':
                    state['release_count'] += 1
                    state['owner'] = None
                    result = {'released': True}
                else:
                    response = {'error': 'Unknown operation'}
                self.wfile.write((json.dumps(response or {'result': result}) + '\n').encode())
                self.wfile.flush()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'fake-motor.sock'
        with Server(str(path), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                yield path, entered, finish, state
            finally:
                finish.set()
                server.shutdown()
                worker.join(2)


class MotorPathProbeTests(unittest.TestCase):
    def test_blocked_path_probe_does_not_block_primary_feedback_or_commands(self):
        with blocked_path_server() as (path, entered, finish, state):
            client = MotorClient('primary', path).open(autostart=False)
            result, errors = [], []

            def validate():
                try:
                    result.append(client.validate_path('left', [0]*6, [.1]*6))
                except Exception as exc:
                    errors.append(exc)

            probe_thread = threading.Thread(target=validate)
            probe_thread.start()
            try:
                self.assertTrue(entered.wait(1))
                primary_done = threading.Event()

                def control_tick():
                    try:
                        client.read()
                        self.assertTrue(client.rpc('command', commands={'held': True})['accepted'])
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        primary_done.set()

                control_thread = threading.Thread(target=control_tick)
                control_thread.start()
                self.assertTrue(primary_done.wait(.25), 'Primary RPC lock was held by path preflight')
                control_thread.join(1)
                self.assertTrue(probe_thread.is_alive(), 'Path preflight must still be blocked')
                finish.set()
                probe_thread.join(2)
                self.assertFalse(probe_thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(result, [{'valid': True}])
                self.assertEqual(state['path_token'], client.token)
                self.assertEqual(state['acquire_count'], 1)
                self.assertEqual(state['release_count'], 0)
                self.assertEqual(client.rpc('status')['owner'], 'owner-token')
            finally:
                finish.set()
                probe_thread.join(2)
                client.close()

    def test_rejected_probe_propagates_reason_without_releasing_primary_owner(self):
        with blocked_path_server(reject=True) as (path, entered, finish, state):
            client = MotorClient('primary', path).open(autostart=False)
            finish.set()
            try:
                with self.assertRaisesRegex(RuntimeError, 'dips below clearance'):
                    client.validate_path('left', [0]*6, [.1]*6)
                self.assertTrue(entered.is_set())
                self.assertEqual(state['acquire_count'], 1)
                self.assertEqual(state['release_count'], 0)
                self.assertEqual(client.rpc('status')['owner'], client.token)
                self.assertTrue(client.rpc('command', commands={'held': True})['accepted'])
            finally:
                client.close()

    def test_probe_refuses_without_existing_owner_and_never_autostarts(self):
        client = MotorClient('disconnected', '/does/not/exist')
        with self.assertRaisesRegex(RuntimeError, 'no control lease'):
            client.validate_path('left', [0]*6, [.1]*6)
        self.assertIsNone(client.socket)

    def test_client_exposes_loaded_limits_and_refreshes_without_inventing_new_caps(self):
        with blocked_path_server() as (path, entered, finish, state):
            state['guarded_load_limits'] = {'max_xy_m': .008, 'max_speed_m_s': .020}
            client = MotorClient('primary', path).open(autostart=False)
            try:
                self.assertEqual(client.guarded_load_limits['max_xy_m'], .008)
                self.assertEqual(client.guarded_load_limits['max_speed_m_s'], .020)
                state['guarded_load_limits'] = {'max_xy_m': .030, 'max_speed_m_s': .050}
                # Merely changing frontend code cannot make an older loaded
                # backend support a larger envelope. Only new status can do so.
                self.assertEqual(client.guarded_load_limits['max_xy_m'], .008)
                client.read()
                self.assertEqual(client.guarded_load_limits['max_xy_m'], .030)
                self.assertEqual(client.guarded_load_limits['max_speed_m_s'], .050)
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
