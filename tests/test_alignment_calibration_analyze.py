"""Verify physical camera association and fail-closed offline analysis."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

def test_analyzer_uses_physical_ports_instead_of_model_slot_labels():
    import copy
    import json
    import tempfile
    import alignment_calibration_analyze as analyzer
    gear = dict(camera_names=['overhead', 'left_wrist', 'right_wrist'],
                landmark_names=['gear_center', 'slot_center'], checkpoint_id='model',
                keypoints_2d=[[[0, 0], [0, 0]], [[10, 10], [2, 2]], [[30, 30], [2, 2]]],
                visibility_probability=np.zeros((3, 2)).tolist(),
                confidence=np.ones((3, 2)).tolist(), pixel_std=np.ones((3, 2)).tolist())
    pin = copy.deepcopy(gear)
    pin['landmark_names'] = ['pin_tip', 'pin_top']+[f'carrier_hole_{n}' for n in range(1, 5)]
    pin['keypoints_2d'] = np.zeros((3, 6, 2)).tolist()
    for field in ('visibility_probability', 'confidence', 'pixel_std'):
        pin[field] = np.ones((3, 6)).tolist()
    row = dict(episode_id='trial', time_s=1.2, q_left=[0.]*6, q_right=[0.]*6,
               camera_mapping={'overhead': 'overhead', 'left_wrist': 'wrist_port2', 'right_wrist': 'wrist_port3'},
               predictions={'gear': gear, 'pin': pin},
               arms={'left': {'gripper': {'position': 1., 'torque': 0., 'velocity': 0.}}},
               frame_timestamps_ns={'wrist_port2': 1000000000, 'wrist_port3': 1000000000}, state_skew_s=0.)
    audit = dict(mapping={'overhead': 'overhead', 'wrist_port2': 'right', 'wrist_port3': 'left'},
                 moving_arm='left', episodes=[dict(id='trial', gear=dict(start_s=1., end_s=2., reference_s=2.),
                                                   pin=dict(start_s=1., end_s=2., reference_s=2.))])
    captured = []
    original_fit = analyzer._fit
    def fake_fit(samples, side, task):
        captured.append((task, samples))
        return dict(candidate_usable=False, reasons=[])
    analyzer._fit = fake_fit
    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            (root/'audit.json').write_text(json.dumps(audit))
            (root/'predictions.manifest.json').write_text(json.dumps({'complete': False}))
            report = analyzer.analyze(root/'predictions.jsonl', root/'audit.json')
            import hashlib
            identity = dict(complete=True, samples=1,
                            output_sha256=hashlib.sha256((root/'predictions.jsonl').read_bytes()).hexdigest())
            (root/'predictions.manifest.json').write_text(json.dumps(identity))
            complete = analyzer.analyze(root/'predictions.jsonl', root/'audit.json')
            assert complete['analysis_complete'] is True
            row['time_s'] = 1.3
            (root/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            tampered = analyzer.analyze(root/'predictions.jsonl', root/'audit.json')
            assert tampered['analysis_complete'] is False
            assert 'cache_output_sha256_mismatch_or_missing' in tampered['input_identity_errors']
            row['time_s'] = 1.2
            (root/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            identity['phase_selection'] = {'sha256': 'wrong-audit-hash'}
            (root/'predictions.manifest.json').write_text(json.dumps(identity))
            wrong_audit = analyzer.analyze(root/'predictions.jsonl', root/'audit.json')
            assert wrong_audit['analysis_complete'] is False
            assert 'phase_selection_audit_sha256_mismatch' in wrong_audit['input_identity_errors']
            identity.pop('phase_selection')
            (root/'predictions.manifest.json').write_text(json.dumps(identity))
            incomplete_audit = copy.deepcopy(audit)
            for index in range(16):
                missing = copy.deepcopy(audit['episodes'][0])
                missing['id'] = f'missing{index}'
                incomplete_audit['episodes'].append(missing)
            (root/'audit.json').write_text(json.dumps(incomplete_audit))
            missing_trials = analyzer.analyze(root/'predictions.jsonl', root/'audit.json')
            assert missing_trials['analysis_complete'] is False
            assert missing_trials['provenance']['episode_count'] == 1
        diagnostic = next(samples for task, samples in captured if task == 'gear' and samples)
        assert diagnostic[0]['error_px'] == [28., 28., 8., 8.]
        assert report['tasks']['gear']['counts']['strict_samples'] == 0
        assert report['tasks']['gear']['counts']['both_visible'] == 0
        assert report['tasks']['gear']['strict_fit']['candidate_usable'] is False
        assert report['tasks']['gear']['diagnostic_fit']['candidate_usable'] is False
        assert report['verified'] is False and report['motion_enabled'] is False
        assert report['analysis_complete'] is False
        assert report['provenance']['episode_count'] == 1
    finally:
        analyzer._fit = original_fit


def load_tests(loader, tests, pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in globals().items()
                              if name.startswith('test_') and callable(test))


if __name__ == '__main__':
    import unittest
    unittest.main()
