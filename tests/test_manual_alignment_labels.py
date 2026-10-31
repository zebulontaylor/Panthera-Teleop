import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('assemble_alignment_labels',
    Path(__file__).resolve().parents[1]/'scripts/assemble_alignment_labels.py')
labels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(labels)


class ManualLabelTests(unittest.TestCase):
    def point(self, visibility, xy=None):
        return dict(visibility=visibility, xy_px=xy, confidence='medium', uncertainty_px=5)

    def test_ambiguity_is_not_a_visibility_negative(self):
        ambiguous = labels.supervision(self.point('ambiguous'))
        self.assertFalse(ambiguous['position_supervision'])
        self.assertFalse(ambiguous['visibility_supervision'])
        self.assertIsNone(ambiguous['visibility_target'])
        hidden = labels.supervision(self.point('occluded'))
        self.assertFalse(hidden['position_supervision'])
        self.assertTrue(hidden['visibility_supervision'])
        self.assertIs(hidden['visibility_target'], False)
        visible = labels.supervision(self.point('visible', [300, 200]))
        self.assertTrue(visible['position_supervision'])
        self.assertIs(visible['visibility_target'], True)

    def test_no_hidden_coordinates_nonfinite_or_sheet_coordinates(self):
        for point in [self.point('occluded', [300, 200]), self.point('visible', [float('nan'), 0]),
                      self.point('visible', [975, 340]), self.point('visible', [0, 480])]:
            with self.assertRaises(ValueError):
                labels.validate_point(point)

    def test_episode_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(observations=[
                dict(observation_id='a', episode_id='same', split='train'),
                dict(observation_id='b', episode_id='same', split='validation')])))
            with self.assertRaisesRegex(ValueError, 'cannot cross'):
                labels.assemble(manifest, [], root/'out')

    def test_changed_source_image_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'source.jpg'
            image.write_bytes(b'changed')
            source = dict(observation_id='a', episode_id='trial', split='train', phase='empty',
                          images={camera: dict(path=str(image), sha256='originalhash')
                                  for camera in ('wrist_port2', 'wrist_port3')})
            observation = dict(observation_id='a', grasp={side:dict(object='empty',confidence='high')
                for side in ('left','right')}, views={camera:dict(landmarks={},carrier_bores=[])
                for camera in ('wrist_port2','wrist_port3')}, cross_view_target_identity='not_applicable')
            manifest, annotation = root/'manifest.json', root/'labels.json'
            manifest.write_text(json.dumps(dict(observations=[source])))
            annotation.write_text(json.dumps(dict(annotator='test',observations=[observation])))
            with self.assertRaisesRegex(ValueError, 'Source image changed'):
                labels.assemble(manifest, [annotation], root/'out')


if __name__ == '__main__':
    unittest.main()
