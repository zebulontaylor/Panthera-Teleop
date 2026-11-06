import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pin_insertion_contact_manifest import prepare


def fixture_files(tmp_path):
    image = tmp_path / 'frame.png'
    Image.new('RGB', (64, 48)).save(image)
    row = dict(id='e__0001', episode_id='e', session_id='g', split='train',
               video_frame=1, image_path=str(image), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
               state14=[0.] * 14, valid=[False, False], keypoints_px=[None] * 4,
               uncertainty_px=[None, None], gate_label=None)
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'frames': [row]}))
    audit = tmp_path / 'audit.json'
    row.update(valid=[False, True], keypoints_px=[None, None, 32, 24],
               uncertainty_px=[None, 7], target_hole_inferential=True)
    audit.write_text(json.dumps({'frames': [row]}))
    return base, audit


def test_replaces_exact_frame_without_promoting_hidden_pin_or_contact(tmp_path):
    base, audit = fixture_files(tmp_path)
    original = base.read_bytes()
    out = tmp_path / 'merged.json'
    prepare(base, [audit], out, tmp_path)
    result = json.loads(out.read_text())
    assert len(result['frames']) == 1
    row = result['frames'][0]
    assert row['valid'] == [False, True]
    assert row['keypoints_px'][:2] == [None, None]
    assert row['gate_label'] is None
    assert row['target_hole_inferential'] is True
    assert row['contact_audit'] is True
    assert base.read_bytes() == original


@pytest.mark.parametrize('field,value,message', [('split', 'test', 'frozen episode split'),
                                                  ('gate_label', 1, 'no verified outside-hole')])
def test_rejects_split_changes_and_unverified_contact_positives(tmp_path, field, value, message):
    base, audit = fixture_files(tmp_path)
    d = json.loads(audit.read_text())
    d['frames'][0][field] = value
    audit.write_text(json.dumps(d))
    with pytest.raises(ValueError, match=message):
        prepare(base, [audit], tmp_path / 'out.json', tmp_path)
