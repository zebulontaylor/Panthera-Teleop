"""Hardware-free intent routing, validation, and HTTP compatibility checks."""
import json
from pathlib import Path
import sys
import socket
import subprocess
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from robot_commands import RobotCommands
from dashboard import Dashboard


class RobotCommandTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.snapshot.return_value = dict(active='episode-1', phase='recording',
            control_active=True, alignment_assist={'mode': 'operator_supervised'},
            gear_grasp={'settled': True})
        self.commands = RobotCommands(self.store)

    def test_language_and_structured_commands_have_identical_effects(self):
        cases = [
            ('enable alignment assistance', 'alignment.assist', {'enabled': True}, {}),
            ('close the gear gripper', 'gear.grasp', {'action': 'close'}, {'episode_id': 'episode-1'}),
            ('keep this episode', 'episode.finish', {'label': 'kept'}, {'episode_id': 'episode-1'}),
            ('stop visual alignment', 'alignment.servo', {'enabled': False}, {}),
        ]
        for phrase, command, inferred, explicit in cases:
            with self.subTest(phrase=phrase):
                from unittest.mock import patch
                with patch('robot_commands.time.monotonic_ns', return_value=100):
                    self.commands.submit({'instruction': phrase, 'parameters': explicit})
                    language_call = self.store.submit.call_args
                    self.commands.submit({'command': command, 'parameters': dict(inferred, **explicit)})
                    self.assertEqual(language_call, self.store.submit.call_args)

    def test_rejects_ambiguous_unknown_and_overridden_instructions(self):
        requests = [[], {}, {'command': []}, {'instruction': 'move over there'},
            {'instruction': 'keep this episode and close the gripper'},
            {'instruction': 'keep this episode', 'command': 'episode.finish'},
            {'instruction': 'keep this episode', 'parameters': {'label': 'discarded'}},
            {'command': 'alignment.assist', 'parameters': {'enabled': True, 'torque': 99}},
            {'command': 'alignment.assist', 'parameters': None},
            {'command': 'alignment.assist', 'extra': True}]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.commands.submit(request)
        self.store.submit.assert_not_called()

    def test_language_preserves_episode_and_servo_gates(self):
        for request in [
            {'instruction': 'close the gear gripper'},
            {'instruction': 'keep this episode', 'parameters': {'episode_id': 'old'}},
            {'command': 'alignment.nudge', 'parameters': {'held': True, 'press_id': 'p', 'episode_id': 'old'}},
        ]:
            with self.assertRaises(ValueError):
                self.commands.submit(request)
        self.store.snapshot.return_value = {}
        with self.assertRaises(ValueError):
            self.commands.submit({'instruction': 'start visual alignment'})
        self.commands.submit({'instruction': 'stop visual alignment'})
        self.assertEqual(self.store.submit.call_args.args[0], 'alignment_servo')

    def test_http_catalog_authentication_and_both_input_forms(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = Dashboard(self.store, None, port, demo=True)
        self.addCleanup(server.close)
        with urlopen(server.url + '/api/commands') as response:
            catalog = json.load(response)
        self.assertIn('gear.grasp', catalog['commands'])
        self.assertIn('keep this episode', catalog['phrases'])
        for payload in ({'command': 'alignment.assist', 'parameters': {'enabled': True}},
                        {'instruction': 'Enable alignment assistance!'}):
            request = Request(server.url + '/api/commands', data=json.dumps(payload).encode())
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            request.add_header('X-Session-Token', server.token)
            with urlopen(request) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response)['status'], 'queued')
            self.store.submit.assert_called_with('alignment_assist', {'enabled': True})

        cli = Path(__file__).resolve().parents[1] / 'scripts/robot-command.py'
        for arguments in (['--list'], ['--instruction', 'enable alignment assistance']):
            result = subprocess.run([sys.executable, str(cli), '--port', str(port), *arguments],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsInstance(json.loads(result.stdout), dict)
        result = subprocess.run([sys.executable, str(cli), '--port', str(port),
                                 '--instruction', 'do something unknown'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Unsupported instruction', result.stderr)


if __name__ == '__main__':
    unittest.main()
