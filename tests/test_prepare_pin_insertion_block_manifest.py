import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pin_insertion_block_manifest import prepare


def inputs(tmp_path):
    image = tmp_path / 'frame.png'
    image.write_bytes(b'original recorded image bytes')
    pose = {'position_m': [.1, .2, .3], 'rotation': np.eye(3).tolist()}
    arm = {'position_rad': [1., 2., 3., 4., 5., 6.],
           'gripper': {'position': 1.1}, 'tool_pose_base': pose}
    states = [{'monotonic_ns': 1_000_000_000 + i * 10_000_000,
               'arms': {'left': arm, 'right': arm}} for i in range(2)]
    frame = {'frame_index': 4, 'camera': 'wrist_port2', 'episode_time_s': 2.,
             'monotonic_ns': 1_009_000_000, 'path': 'frame.png', 'width': 640, 'height': 480}
    (tmp_path / 'states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states))
    (tmp_path / 'frames.jsonl').write_text(json.dumps(frame)+'\n')
    hashes = {n: hashlib.sha256((tmp_path/n).read_bytes()).hexdigest()
              for n in ('states.jsonl', 'frames.jsonl')}
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'source_path': str(tmp_path), 'source_recording_id': 'original',
                                'source_hashes': hashes, 'limitations': ['One recording'],
                                'blocks': [{'block_id': 'B1', 'split': 'train', 'start_s': 0., 'end_s': 3.}]}))
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'frames': [{'frame_index': 4, 'block_id': 'B1', 'split': 'train',
        'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'valid': [False, True],
        'keypoints_px': [None, None, 300, 280], 'uncertainty_px': [None, 6],
        'hole_reference_frame_index': 10, 'gate_label': None}]}))
    return split, labels


def test_original_clock_fk_and_annotation_only_reference_preserved(tmp_path):
    split, labels = inputs(tmp_path)
    out = tmp_path / 'out.json'
    prepare(split, [labels], out)
    result = json.loads(out.read_text()); row = result['frames'][0]
    assert row['source_state_monotonic_ns'] == 1_010_000_000
    assert row['camera_sample_skew_ms'] == -1.
    assert row['state14'] == [1., 2., 3., 4., 5., 6., 1.1] * 2
    assert np.allclose(np.array(row['T_left_base_tool'])[:3, 3], [.1, .2, .3])
    assert row['episode_id'] == 'original::B1'
    assert row['source_recording_id'] == row['session_id'] == 'original'
    assert row['valid'] == [False, True]
    assert row['keypoints_px'][:2] == [None, None]
    assert result['insertion_references_are_annotation_only']
    assert result['independent_recordings'] == 1 and not result['session_disjoint']


@pytest.mark.parametrize('change,match', [('split', 'crosses frozen'), ('image', 'hash differs'),
                                       ('source', 'Frozen source'), ('duplicate', 'Duplicate')])
def test_rejects_label_leakage_and_source_changes(tmp_path, change, match):
    split, labels = inputs(tmp_path)
    data = json.loads(labels.read_text())
    if change == 'split': data['frames'][0]['split'] = 'test'
    if change == 'image': (tmp_path / 'frame.png').write_bytes(b'changed')
    if change == 'source': (tmp_path / 'states.jsonl').write_text('{}\n')
    if change == 'duplicate': data['frames'].append(data['frames'][0].copy())
    labels.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        prepare(split, [labels], tmp_path / 'out.json')
