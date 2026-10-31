"""Tests for all four separate calibration interfaces; no motor device access."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
sys.path.insert(0, str(ROOT/'scripts'))
from calibration_workflow_common import CollectorClient
from calibration_workbench import Workbench
from guided_hand_calibration import GuidedHandCalibration
from demonstrated_correction_calibration import DemonstratedCorrectionCalibration
from board_marker_calibration import BoardMarkerCalibration
from hold_run_calibration import HoldRunCalibration
from calibration_sequence import HoldToRunSequence, load_motion_gate
from calibration_sequence_transport import SequenceCommands
from test_cartesian_nudge import fixture, LinearKinematics
from test_alignment_assist import advance
from test_nudge_calibration_protocols import measured_pairs
from audit_nudge_motion import audit


def endpoint(z, index, error=None):
    stamp = int((10+index*2)*1e9)
    return dict(position_m=[0., 0., z], rotation=np.eye(3).tolist(),
        error_px=error or [0., z*10000, 0., z*10000], other_q=[0.]*6,
        gripper_position=.5, camera_mapping={'left_wrist': 'port3', 'right_wrist': 'port2'},
        checkpoint_id='test', q=[0., 0., z, 0., 0., 0.], observations=7,
        first_monotonic_ns=stamp-700_000_000, last_monotonic_ns=stamp)


def begin(workflow, session='a', split='train'):
    workflow.last = dict(episode_id=session, monotonic_ns=int(time.monotonic_ns()))
    workflow.begin(dict(session=session, split=split, axis=[0, 0, 1]))


class WorkflowTests(unittest.TestCase):
    def test_guided_observations_detect_stable_holds_without_manual_markers(self):
        _, snapshot, arms, _, _ = fixture()
        with tempfile.TemporaryDirectory() as output:
            w = GuidedHandCalibration(output); begin(w)
            base = time.monotonic_ns()/1e9+1
            for i in range(24):
                t = base+i*.1
                z = 0 if i < 10 else .0005
                a = copy.deepcopy(arms)
                a['left']['tool_pose_base']['position_m'] = [0, 0, z]
                a['left']['position_rad'][2] = z
                p = copy.deepcopy(snapshot['predictions'])
                p['gear']['keypoints_2d'][1][0][1] = p['gear']['keypoints_2d'][2][0][1] = 112+10000*z
                stamp = int(t*1e9)
                row = dict(episode_id='a', time_s=t, monotonic_ns=stamp, state_monotonic_ns=stamp,
                    camera_mapping=snapshot['camera_mapping'], predictions=p,
                    frame_timestamps_ns={name: stamp for name in snapshot['camera_mapping'].values()},
                    arms=a, q_left=a['left']['position_rad'], q_right=a['right']['position_rad'])
                w.record_observation(row)
                w.observe(row)
            self.assertEqual(len(w.pairs), 1, w.observation_reason)
            self.assertEqual(w.counter, 1)
            self.assertTrue((Path(output)/'observation-cache.jsonl').exists())

    def test_guided_sessions_automatically_capture_and_fit(self):
        with tempfile.TemporaryDirectory() as output:
            workflow = GuidedHandCalibration(output)
            for index, (session, split) in enumerate([('a', 'train'), ('b', 'train'), ('c', 'validation')]):
                begin(workflow, session, split)
                workflow.stable(endpoint(0, index*10), {})
                for i, sign in enumerate(workflow.sequence):
                    workflow.stable(endpoint(sign*.0005, index*10+i+1), {})
                self.assertEqual(workflow.phase, 'complete')
            self.assertEqual(len(workflow.pairs), 24)
            workflow.command('analyze')
            self.assertTrue(workflow.report['statistical_candidate'], workflow.report)
            self.assertFalse(workflow.report['verified'])
            self.assertFalse(workflow.report['motion_authorized'])

    def test_guided_rejects_rotated_or_changed_reference(self):
        with tempfile.TemporaryDirectory() as output:
            w = GuidedHandCalibration(output); begin(w)
            w.stable(endpoint(0, 0), {})
            bad = endpoint(.0005, 1); bad['checkpoint_id'] = 'changed'
            w.stable(bad, {})
            self.assertEqual(len(w.pairs), 0)
            w.stable(endpoint(.0005, 2), {})
            w.stable(endpoint(0, 3, [0, 8, 0, 8]), {})
            self.assertEqual(len(w.pairs), 1)
            self.assertIn('Reference image', w.observation_reason)

    def test_demonstration_requires_improvement_and_saves_local_library(self):
        with tempfile.TemporaryDirectory() as output:
            w = DemonstratedCorrectionCalibration(output); begin(w)
            w.stable(endpoint(0, 0), {})
            w.command('capture'); w.stable(endpoint(.001, 1), {})
            self.assertEqual(w.phase, 'corrected_pose')
            w.command('capture'); w.stable(endpoint(.0015, 2), {})
            self.assertEqual(len(w.library), 0)
            w.command('capture'); w.stable(endpoint(0, 3), {})
            self.assertEqual(w.phase, 'pair_complete')
            self.assertEqual(len(w.library), 1)
            saved = json.loads((Path(output)/'demonstrated-library.json').read_text())
            self.assertFalse(saved['verified'])
            self.assertAlmostEqual(saved['examples'][0]['delta_xyz_m'][2], -.001)

    def test_board_and_marker_capture_known_rendered_targets(self):
        _, snapshot, arms, _, _ = fixture()
        for pattern in ('checkerboard', 'aruco'):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as output:
                w = BoardMarkerCalibration(output)
                w.last = dict(episode_id='a', monotonic_ns=1)
                w.begin(dict(session='a', split='train', axis=[0, 0, 1], pattern=pattern,
                    size_mm=20, columns=9, rows=6, camera='left_wrist', side='left', marker_id=3))
                image = np.full((480, 640), 255, np.uint8)
                if pattern == 'checkerboard':
                    for y in range(7):
                        for x in range(10):
                            image[100+y*40:100+(y+1)*40, 120+x*40:120+(x+1)*40] = 255 if (x+y)%2 else 0
                else:
                    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    image[140:340, 220:420] = cv2.aruco.generateImageMarker(dictionary, 3, 200)
                ok, encoded = cv2.imencode('.jpg', image); self.assertTrue(ok)
                physical = snapshot['camera_mapping']['left_wrist']
                frames = {physical: base64.b64encode(encoded).decode()}
                w.command('capture')
                for i in range(9):
                    stamp = int((10+i*.1)*1e9)
                    row = dict(episode_id='a', monotonic_ns=stamp, time_s=i*.1,
                        state_monotonic_ns=stamp, camera_mapping=snapshot['camera_mapping'],
                        frame_timestamps_ns={physical: stamp}, arms=copy.deepcopy(arms))
                    w.observe(row, frames)
                self.assertEqual(len(w.observations), 1, w.observation_reason)
                self.assertEqual(len(w.detection['corners']), 54 if pattern == 'checkerboard' else 4)
                w.command('analyze')
                self.assertFalse(w.report['statistical_candidate'])

    def test_collector_board_sampler_reads_exact_recorded_images_without_cnn(self):
        _, snapshot, arms, _, _ = fixture()
        snapshot['camera_mapping']['overhead'] = 'overhead'
        with tempfile.TemporaryDirectory() as output:
            directory = Path(output)/'pending'/'episode'; directory.mkdir(parents=True)
            stamp = time.monotonic_ns()
            metadata = dict(start_monotonic_ns=stamp-1_000_000_000,
                config={'alignment_assist': {'camera_mapping': snapshot['camera_mapping']}})
            (directory/'metadata.json').write_text(json.dumps(metadata))
            (directory/'states.jsonl').write_text(json.dumps(dict(monotonic_ns=stamp, arms=arms))+'\n')
            frames = []
            for physical in snapshot['camera_mapping'].values():
                path = directory/(physical+'.jpg'); path.write_bytes(b'exact-jpeg-bytes')
                frames.append(dict(camera=physical, monotonic_ns=stamp, path=path.name))
            (directory/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frames))
            client = CollectorClient('http://127.0.0.1:8090', output)
            state = dict(active='episode', phase='recording', cameras={name: {'ok': True} for name in snapshot['camera_mapping'].values()})
            client.get = Mock(return_value=state)
            row, images, _ = client.sample_board()
            self.assertEqual(row['monotonic_ns'], stamp)
            self.assertEqual(base64.b64decode(next(iter(images.values()))), b'exact-jpeg-bytes')
            self.assertTrue(all(call.args == ('/api/status',) for call in client.get.call_args_list))

    def test_http_token_and_workflow_motion_separation(self):
        for cls in (GuidedHandCalibration, DemonstratedCorrectionCalibration, BoardMarkerCalibration, HoldRunCalibration):
            with self.subTest(workflow=cls.name), tempfile.TemporaryDirectory() as output:
                client = Mock()
                workbench = Workbench(cls(output), client, 0)
                try:
                    url = 'http://127.0.0.1:'+str(workbench.server.server_address[1])
                    with urlopen(url) as response:
                        self.assertIn(b'Calibration workbench', response.read())
                    with self.assertRaises(HTTPError) as error:
                        urlopen(Request(url+'/api/action', data=b'{"action":"cancel"}'))
                    self.assertEqual(error.exception.code, 403); error.exception.close()
                    request = Request(url+'/api/action', data=b'{"action":"hold","press_id":"x"}',
                        headers={'X-Session-Token': workbench.token, 'Content-Type': 'application/json'})
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request)
                    self.assertEqual(error.exception.code, 409); error.exception.close()
                    client.sequence.assert_not_called()
                finally:
                    workbench.close()

    def test_output_directory_cannot_silently_overwrite_previous_run(self):
        with tempfile.TemporaryDirectory() as output:
            (Path(output)/'pairs.jsonl').write_text('existing evidence')
            with self.assertRaisesRegex(ValueError, 'new_empty_output'):
                GuidedHandCalibration(output)

    def test_hold_workbench_requires_matching_axis_and_forwards_only_explicit_press(self):
        with tempfile.TemporaryDirectory() as output:
            w = HoldRunCalibration(output); begin(w)
            w.stable(endpoint(0, 0), {})
            client = Mock()
            bench = Workbench(w, client, 0)
            bench.collector_error = None
            try:
                with self.assertRaisesRegex(ValueError, 'verified_sequence_collector'):
                    bench.action({'action': 'hold', 'press_id': 'a'})
                client.sequence.assert_not_called()
                w.latest_status = {'calibration_sequence': {'gate_ready': True, 'axis_base': [1, 0, 0]}}
                with self.assertRaisesRegex(ValueError, 'axis_must_match'):
                    bench.action({'action': 'hold', 'press_id': 'a'})
                client.sequence.assert_not_called()
                w.latest_status['calibration_sequence']['axis_base'] = [0, 0, 1]
                bench.action({'action': 'hold', 'press_id': 'a'})
                client.sequence.assert_called_once_with(True, 'a', 'a')
                with self.assertRaisesRegex(ValueError, 'release_sequence'):
                    bench.action({'action': 'cancel'})
                bench.action({'action': 'release', 'press_id': 'a'})
                self.assertIsNone(bench.browser_renewed)
            finally:
                bench.close()


def gated_sequence(output):
    config, snapshot, arms, limits, _ = fixture()
    rows, _ = measured_pairs()
    report = audit(rows)
    path = Path(output)/'audit.json'; path.write_text(json.dumps(report))
    gate = dict(operator_motion_verified=True, axis_base=[0, 0, 1], reference_q_rad=[0.]*6,
        camera_mapping=config['calibration']['camera_mapping'], checkpoint_id=config['calibration']['checkpoint_id'],
        motion_audit_path=str(path), motion_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    gate_path = Path(output)/'gate.json'; gate_path.write_text(json.dumps(gate))
    config['sequence_gate_path'] = str(gate_path)
    config['sequence_gate_sha256'] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    return config, snapshot, arms, limits, gate_path


class HoldSequenceTests(unittest.TestCase):
    def test_default_gate_blocks_and_changed_audit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'verified_motion_gate_required'):
            load_motion_gate(ROOT/'config/calibration-sequence-motion-gate.json')
        with tempfile.TemporaryDirectory() as output:
            config, snapshot, arms, limits, gate_path = gated_sequence(output)
            (Path(output)/'audit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'audit_hash_mismatch'):
                load_motion_gate(gate_path)

    def test_leased_sequence_is_bounded_and_finishes(self):
        with tempfile.TemporaryDirectory() as output:
            config, snapshot, arms, limits, _ = gated_sequence(output)
            controller = HoldToRunSequence(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
            for i in range(390):
                now = 10+i*.05
                controller.renew(now, 'press')
                advance(snapshot, now)
                target = controller.tick(now, snapshot, arms, allowed=True)
                self.assertLessEqual(np.linalg.norm(target[:3]), .000500001)
                arms['left']['position_rad'] = target.tolist()
                arms['left']['tool_pose_base']['position_m'] = target[:3].tolist()
                if not controller.running:
                    break
            self.assertEqual(controller.steps, 8)
            self.assertEqual(controller.reason, 'sequence_complete_release_button')
            np.testing.assert_allclose(target, np.zeros(6), atol=1e-12)

    def test_lost_heartbeat_and_carrier_motion_stop(self):
        with tempfile.TemporaryDirectory() as output:
            config, snapshot, arms, limits, _ = gated_sequence(output)
            c = HoldToRunSequence(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
            c.renew(10., 'press')
            with self.assertRaisesRegex(ValueError, 'expired'):
                c.tick(10.3, snapshot, arms, allowed=True)
            with self.assertRaisesRegex(ValueError, 'expired_press'):
                c.renew(10.3, 'press')
            c = HoldToRunSequence(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
            c.renew(10., 'next')
            arms['right']['position_rad'][0] = .004
            with self.assertRaisesRegex(ValueError, 'carrier_moved'):
                c.tick(10., snapshot, arms, allowed=True)

    def test_transport_is_explicit_and_disables_legacy_motion(self):
        store = Mock()
        store.snapshot.return_value = dict(active='trial', phase='recording', control_active=True,
            gear_grasp={'settled': True}, alignment_assist={'enabled': True})
        commands = SequenceCommands(store)
        commands.submit(dict(command='calibration.sequence', parameters=dict(held=True, press_id='x', episode_id='trial')))
        self.assertEqual(store.submit.call_args.args[0], 'calibration_sequence')
        with self.assertRaisesRegex(ValueError, 'legacy_motion'):
            commands.submit(dict(command='alignment.servo', parameters={'enabled': True}))
        with self.assertRaisesRegex(ValueError, 'current_real_recording'):
            commands.submit(dict(command='calibration.sequence', parameters=dict(held=True, press_id='x', episode_id='wrong')))


if __name__ == '__main__':
    unittest.main()
