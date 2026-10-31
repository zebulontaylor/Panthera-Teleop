"""Synthetic image-Jacobian checks; no hardware or model dependencies."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from alignment_assist import AlignmentAssist


def fixture(task='gear'):
    names = ['gear_center', 'slot_center'] if task == 'gear' else ['pin_tip', 'pin_top', 'carrier_hole_1', 'carrier_hole_2', 'carrier_hole_3', 'carrier_hole_4']
    count = len(names)
    mapping = {'left_wrist': 'wrist_port2', 'right_wrist': 'wrist_port3'}
    calibration = dict(verified=True, side='right', camera_names=list(mapping),
        camera_mapping=mapping, checkpoint_id='sha256-model', image_size=[640, 480],
        reference_joint_position_rad=[0.]*6, other_arm_reference_joint_position_rad=[0.]*6, max_joint_distance_rad=.15,
        image_jacobian_px_per_rad=[[100, 0, 0, 0, 0, 0], [0, 100, 0, 0, 0, 0],
                                   [0, 0, 100, 0, 0, 0], [0, 0, 0, 100, 0, 0]],
        jaw_rois_px={name: [80, 80, 140, 140] for name in mapping},
        gripper_closed_position_rad=1.3, gripper_contact_torque_nm=.1,
        target_landmark='slot_center' if task == 'gear' else 'carrier_bore',
        target_rois_px={name: [95, 95, 105, 105] for name in mapping},
        target_error_px=[0.]*4, residual_axes=[1, 3] if task == 'gear' else [0, 1, 2, 3])
    points = np.full((3, count, 2), 100.)
    points[:, 0] = [108, 112]
    if task == 'pin':
        points[:, 3:] = [300, 300]
    snapshot = dict(status='ready', monotonic_ns=10_000_000_000,
        camera_mapping=mapping, frame_timestamps_ns={name: 10_000_000_000 for name in mapping.values()},
        predictions={task: dict(checkpoint_id='sha256-model', image_size=[640, 480],
            camera_names=['overhead', 'left_wrist', 'right_wrist'], landmark_names=names,
            keypoints_2d=points.tolist(), visibility_probability=np.ones((3, count)).tolist(),
            confidence=np.ones((3, count)).tolist(), pixel_std=np.ones((3, count)).tolist())})
    arm = dict(position_rad=[0.]*6, velocity_rad_s=[0.]*6,
               gripper=dict(position=1., torque=.2, velocity=0.))
    limits = dict(lower=[-2.]*6, upper=[2.]*6)
    return AlignmentAssist(dict(enabled=True, task=task, calibration=calibration)), snapshot, arm, limits


def advance(snapshot, seconds):
    snapshot['monotonic_ns'] = round(seconds*1e9)
    snapshot['frame_timestamps_ns'] = {k: round(seconds*1e9) for k in snapshot['frame_timestamps_ns']}


def confirmed(controller, snapshot, arm, limits):
    assert controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'confirming_grasp'
    advance(snapshot, 10.21)
    return controller.evaluate(10.21, snapshot, arm, limits, collecting=True, other_arm=arm)


def test_bounded_gear_height_correction_has_no_horizontal_torque():
    controller, snapshot, arm, limits = fixture()
    result = confirmed(controller, snapshot, arm, limits)
    assert result['active']
    assert np.linalg.norm(result['delta_rad']) <= .006000001
    assert max(abs(v) for v in result['torque_nm']) <= .15
    assert result['torque_nm'][1] < 0 and result['torque_nm'][3] < 0
    assert result['torque_nm'][0] == result['torque_nm'][2] == 0
    repeated = controller.evaluate(10.22, snapshot, arm, limits, collecting=True, other_arm=arm)
    assert repeated['delta_rad'] == result['delta_rad']


def test_disable_collection_and_calibration_fail_closed():
    controller, snapshot, arm, limits = fixture()
    assert controller.evaluate(10., snapshot, arm, limits)['reason'] == 'not_collecting'
    controller.set_enabled(False)
    assert controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'disabled'
    controller.set_enabled(True)
    controller.config['calibration']['verified'] = False
    assert controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'calibration_required'


def test_stale_and_swapped_camera_evidence_cannot_nudge():
    controller, snapshot, arm, limits = fixture()
    assert controller.evaluate(10.3, snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'stale_prediction'
    snapshot['camera_mapping'] = dict(left_wrist='wrist_port3', right_wrist='wrist_port2')
    assert controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'camera_mapping_mismatch'


def test_each_gate_returns_zero():
    changes = [
        lambda c, s, a: s['predictions']['gear']['visibility_probability'][1].__setitem__(1, .1),
        lambda c, s, a: s['predictions']['gear']['confidence'][2].__setitem__(0, .1),
        lambda c, s, a: a['gripper'].__setitem__('torque', 0.),
        lambda c, s, a: a['gripper'].__setitem__('position', 1.8),
        lambda c, s, a: a['position_rad'].__setitem__(0, .2),
        lambda c, s, a: a['position_rad'].__setitem__(0, 1.99),
        lambda c, s, a: a['velocity_rad_s'].__setitem__(0, .6),
        lambda c, s, a: s['predictions']['gear'].__setitem__('checkpoint_id', 'wrong'),
        lambda c, s, a: s['predictions']['gear']['keypoints_2d'][1].__setitem__(0, [300, 300]),
        lambda c, s, a: s['predictions']['gear']['keypoints_2d'][1].__setitem__(0, [float('nan'), 100]),
    ]
    for change in changes:
        controller, snapshot, arm, limits = fixture()
        change(controller.config, snapshot, arm)
        result = controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)
        assert not result['active'], result
        assert result['torque_nm'] == [0.]*6


def test_grasp_requires_distinct_fresh_frames_and_resets_when_released():
    controller, snapshot, arm, limits = fixture()
    assert controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'confirming_grasp'
    assert controller.evaluate(10.15, snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'confirming_grasp'
    arm['gripper']['torque'] = 0
    advance(snapshot, 10.16)
    assert controller.evaluate(10.16, snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'gripper_not_holding'
    arm['gripper']['torque'] = .2
    advance(snapshot, 10.3)
    assert controller.evaluate(10.3, snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'confirming_grasp'


def test_burst_budget_blocks_creep_even_with_fresh_predictions():
    controller, snapshot, arm, limits = fixture()
    assert confirmed(controller, snapshot, arm, limits)['active']
    for stamp in (10.52, 10.8):
        advance(snapshot, stamp)
        assert controller.evaluate(stamp, snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'nudge_budget_exhausted'
    controller.set_enabled(False)
    controller.set_enabled(True)
    assert controller.evaluate(10.8, snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'confirming_grasp'


def test_pin_explicit_shared_target_and_calibrated_alignment_offset():
    controller, snapshot, arm, limits = fixture('pin')
    assert confirmed(controller, snapshot, arm, limits)['active']
    controller, snapshot, arm, limits = fixture('pin')
    controller.config['calibration']['target_error_px'] = [8, 12, 8, 12]
    assert confirmed(controller, snapshot, arm, limits)['reason'] == 'aligned'


def test_malformed_evidence_and_jacobian_fail_closed():
    controller, snapshot, arm, limits = fixture()
    controller.config['calibration']['image_jacobian_px_per_rad'] = [[0]*6]*4
    assert controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)['reason'] == 'invalid_image_jacobian'
    assert controller.evaluate(10., {}, arm, limits, collecting=True, other_arm=arm)['torque_nm'] == [0.]*6



def test_pin_unordered_slots_select_bore_per_camera_and_reject_ambiguity():
    controller, snapshot, arm, limits = fixture('pin')
    points = snapshot['predictions']['pin']['keypoints_2d']
    points[2][2], points[2][4] = points[2][4], points[2][2]
    assert confirmed(controller, snapshot, arm, limits)['active']
    controller, snapshot, arm, limits = fixture('pin')
    snapshot['predictions']['pin']['keypoints_2d'][1][4] = [101, 101]
    result = controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=arm)
    assert result['reason'] == 'pin_target_ambiguous_or_missing'
    assert not result['active']


def test_other_wrist_arm_pose_is_required_and_inside_calibrated_envelope():
    controller, snapshot, arm, limits = fixture()
    result = controller.evaluate(10., snapshot, arm, limits, collecting=True)
    assert not result['active']
    other = dict(position_rad=[.2]*6)
    result = controller.evaluate(10., snapshot, arm, limits, collecting=True, other_arm=other)
    assert result['reason'] == 'other_arm_outside_calibration_envelope'


def load_tests(loader, tests, pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in globals().items()
                              if name.startswith('test_') and callable(test))


if __name__ == '__main__':
    import unittest
    unittest.main()
