"""Deterministic checks of saved state/image clocks and omitted hold boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from hitl_recording import RolloutStore


class RecordingClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = json.loads((ROOT / 'config/recording.json').read_text())
        config.update(minimum_free_bytes=0, control_hz=100, start_delay_seconds=0)
        self.store = RolloutStore(self.temp.name, config, {'mode': 'human_in_the_loop'})
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.store.close, 'test')
        self.send('rollout_begin', dict(sample=dict(monotonic_ns=1_000_000_000), details={}))

    def send(self, kind, payload):
        self.store.submit(kind, payload)
        self.store.events.join()
        self.assertIsNone(self.store.failure)

    def sample(self, i, source='policy', origin=1_000_000_000, **extra):
        raw = origin + i*10_000_000
        frame = dict(camera='overhead', sequence=i, monotonic_ns=raw-5_000_000, jpeg=b'jpeg')
        self.send('sample', dict(dict(monotonic_ns=raw, sequence=i, control_source=source,
            policy_time_s=i*.005, policy_clock_rate=.5, inference_sequence=0,
            camera_frames={'overhead': frame}, arms={}), **extra))

    def saved(self, **outcome):
        self.send('rollout_end', dict(label='kept', **outcome))
        directory = next((Path(self.temp.name) / 'kept').iterdir())
        return {kind: [json.loads(l) for l in (directory / f'{kind}.jsonl').read_text().splitlines()]
                for kind in ('states', 'frames')}

    def test_episode_contract_records_selected_task_without_mutating_server_contract(self):
        self.send('rollout_end', dict(label='discarded'))
        contract = {'task': 'single gear', 'fps': 20}
        self.store.context['contract'] = contract
        self.send('rollout_begin', dict(sample=dict(monotonic_ns=2_000_000_000),
                                      details=dict(task='four gears')))
        self.assertEqual(self.store.active['metadata']['context']['contract']['task'], 'four gears')
        self.assertEqual(contract['task'], 'single gear')

    def test_frames_preserve_receive_time_and_map_to_the_policy_clock(self):
        for i in range(101):
            self.sample(i)
        rows = self.saved()
        self.assertEqual(len(rows['states']), 51)
        self.assertGreater(len(rows['frames']), 5)
        for frame in rows['frames']:
            elapsed = frame['monotonic_ns']-1_000_000_000
            self.assertEqual(frame['data_monotonic_ns'], 1_000_000_000 + elapsed//2)
        for a, b in zip(rows['frames'], rows['frames'][1:]):
            self.assertGreaterEqual(b['data_monotonic_ns'] - a['data_monotonic_ns'], 50_000_000)

    def test_holds_write_neither_states_nor_images_and_do_not_advance_time(self):
        for i in range(101):
            self.sample(i, 'hold')
        self.assertEqual(self.store.snapshot()['duration'], 0)
        self.assertFalse(self.store.snapshot()['collecting'])
        self.assertEqual(self.saved(), dict(states=[], frames=[]))

    def test_human_guidance_resumes_in_a_separate_segment_without_wait_time(self):
        for i in range(11):
            self.sample(i, 'human')
        for i in range(11, 101):
            self.sample(i, 'hold')
        for i in range(101, 112):
            self.sample(i, 'human')
        rows = self.saved()
        self.assertEqual({s['recording_segment'] for s in rows['states']}, {0, 1})
        self.assertAlmostEqual(rows['states'][-1]['data_time_s'], .21)
        self.assertAlmostEqual(rows['states'][-1]['episode_time_s'], 1.11)
        self.assertTrue(all(s['control_source'] == 'human' for s in rows['states']))

    def saved_metadata(self):
        directory = next((Path(self.temp.name) / 'kept').iterdir())
        return json.loads((directory / 'metadata.json').read_text())

    def test_operator_nudge_and_load_preserve_sources_host_clocks_and_segments(self):
        for i in range(11):
            self.sample(i)
        for i in range(11, 21):
            self.sample(i, 'hold')
        for i in range(21, 32):
            self.sample(i, 'operator_nudge', control_source_by_arm={'left': 'operator_nudge', 'right': 'hold'})
        for i in range(32, 43):
            self.sample(i, 'operator_load', control_source_by_arm={'left': 'operator_load', 'right': 'hold'})
        for i in range(43, 54):
            self.sample(i)
        rows = self.saved(success=True)
        metadata = self.saved_metadata()
        self.assertEqual({s['recording_segment'] for s in rows['states']}, {0, 1, 2, 3})
        self.assertAlmostEqual(rows['states'][-1]['data_time_s'], .33)
        for source, rate in [('policy', .5), ('operator_nudge', 1.), ('operator_load', 1.)]:
            samples = [s for s in rows['states'] if s['control_source'] == source]
            self.assertTrue(samples)
            for segment in {s['recording_segment'] for s in samples}:
                group = [s for s in samples if s['recording_segment'] == segment]
                self.assertEqual({s['data_clock_rate'] for s in group}, {rate})
                for a, b in zip(group, group[1:]):
                    self.assertEqual(b['data_monotonic_ns'] - a['data_monotonic_ns'],
                                     round((b['monotonic_ns'] - a['monotonic_ns']) * rate))
        self.assertTrue(any(f['recording_segment'] == 1 for f in rows['frames']))
        self.assertTrue(any(f['recording_segment'] == 2 for f in rows['frames']))
        self.assertEqual(metadata['human_interventions'], 0)
        self.assertEqual(metadata['operator_intervention_counts'],
                         dict(operator_nudge=1, operator_load=1, frozen_hold=0))
        self.assertEqual(metadata['assistance_interventions'], 2)
        self.assertTrue(metadata['review_recommended'])
        self.assertTrue(metadata['outcome']['human_assisted'])
        self.assertTrue(metadata['outcome']['operator_assisted'])
        self.assertFalse(metadata['outcome']['policy_success'])
        from scripts.prepare_hitl import select_intervals
        self.assertEqual(select_intervals(metadata, rows['states']), [])

    def test_frozen_right_marks_assistance_without_changing_policy_clock(self):
        for i in range(31):
            self.sample(i, control_source_by_arm={
                'left': 'policy', 'right': 'frozen_hold' if 10 <= i <= 20 else 'policy'})
        rows = self.saved(success=True)
        metadata = self.saved_metadata()
        self.assertEqual({s['control_source'] for s in rows['states']}, {'policy'})
        self.assertEqual({s['recording_segment'] for s in rows['states']}, {0})
        self.assertEqual({s['data_clock_rate'] for s in rows['states']}, {.5})
        self.assertAlmostEqual(rows['states'][-1]['data_time_s'], .15)
        self.assertTrue(any(s['control_source_by_arm']['right'] == 'frozen_hold' for s in rows['states']))
        self.assertEqual(metadata['human_interventions'], 0)
        self.assertEqual(metadata['operator_interventions'], 1)
        self.assertEqual(metadata['operator_intervention_counts']['frozen_hold'], 1)
        self.assertTrue(metadata['review_recommended'])
        self.assertFalse(metadata['outcome']['policy_success'])
        from scripts.prepare_hitl import select_intervals
        self.assertEqual(select_intervals(metadata, rows['states']), [])

    def test_human_counter_retains_guidance_meaning_and_policy_success_is_unassisted(self):
        for i in range(11):
            self.sample(i)
        rows = self.saved(success=True)
        metadata = self.saved_metadata()
        self.assertEqual(metadata['human_interventions'], 0)
        self.assertEqual(metadata['operator_interventions'], 0)
        self.assertFalse(metadata['review_recommended'])
        self.assertTrue(metadata['outcome']['policy_success'])
        from scripts.prepare_hitl import select_intervals
        self.assertEqual(select_intervals(metadata, rows['states'])[0][0], 'successful_unassisted')

    def test_unknown_control_source_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid control ownership'):
            self.store.collect(dict(control_source='made_up_operator', monotonic_ns=1_000_000_000, sequence=0))
        self.assertEqual(self.store.active['samples'], 0)
