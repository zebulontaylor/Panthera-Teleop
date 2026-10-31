"""Mixed demonstration transitions must preserve recorded source and ownership boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/action_critic_prepare.py'
spec = importlib.util.spec_from_file_location('action_critic_prepare', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def demonstration(episode='demo', split='train'):
    return [dict(episode_id=episode, session_id=episode, split=split,
                 source_category='kept', weak_demo=True, t=i / 5,
                 episode_end_s=1.6, terminal=i == 8,
                 images=[f'{episode}-{i}-{camera}.jpg' for camera in range(3)],
                 state=[i + coordinate / 100 for coordinate in range(14)],
                 state_time_s=i / 5, state_time_gap_s=0.,
                 control_source='teleop', recording_segment='segment-a') for i in range(9)]


class DemoPreparationTests(unittest.TestCase):
    def test_next_pose_chunks_preserve_frozen_identities_and_terminal_endpoint(self):
        rows = demonstration('frozen-demo', 'val')[:6]
        rows[-1]['terminal'] = True
        # Distracting command fields must never replace the measured BC targets.
        for sample in rows:
            sample['command_target_rad'] = [-999.] * 14
        transitions, rejected = m.prepare_episode(list(reversed(rows)))
        self.assertEqual(len(transitions), 1)
        transition = transitions[0]
        self.assertEqual(transition['state'], rows[0]['state'])
        self.assertEqual(transition['action_chunk'], [r['state'] for r in rows[1:]])
        self.assertEqual(transition['action_times_s'], [.2, .4, .6, .8, 1.])
        self.assertEqual(transition['future_images'], rows[-1]['images'])
        self.assertEqual(transition['dt'], 1.)
        for key in ('episode_id', 'session_id', 'split', 'recording_segment'):
            self.assertEqual(transition[key], rows[0][key])
        self.assertEqual(transition['action_provenance'], 'kinesthetic_demo_next_pose_proxy')
        self.assertIn('not sent motor commands', transition['action_semantics'])

    def test_rejects_gaps_segment_identity_changes_and_terminal_crossing(self):
        cases = [
            ('sample_gap', lambda rows: rows[3].update(t=.95)),
            ('state_time_gap', lambda rows: rows[2].update(state_time_gap_s=.2)),
            ('terminal_crossing', lambda rows: rows[2].update(terminal=True)),
            ('identity_or_segment_change', lambda rows: rows[2].update(recording_segment='b')),
            ('identity_or_segment_change', lambda rows: rows[2].update(split='test')),
            ('identity_or_segment_change', lambda rows: rows[2].update(session_id='other')),
            ('identity_or_segment_change', lambda rows: rows[2].update(episode_id='other')),
        ]
        for reason, mutate in cases:
            rows = demonstration()[:6]; mutate(rows)
            with self.subTest(reason=reason):
                transitions, rejected = m.prepare_episode(rows)
                self.assertEqual(transitions, [])
                self.assertEqual(rejected[reason], 1)

    def test_invalid_measured_state_cannot_become_action_target(self):
        for state in ([0.] * 13, [0.] * 15, [float('nan')] * 14,
                      [float('inf')] * 14, [False] * 14, ['0'] * 14, None):
            rows = demonstration()[:6]; rows[2]['state'] = state
            with self.subTest(state=state):
                self.assertFalse(m.valid_state(rows[2]))
                transitions, rejected = m.prepare_episode(rows)
                self.assertEqual(transitions, [])
                self.assertEqual(rejected['invalid_state'], 1)

    def test_hitl_requires_policy_ownership_through_the_entire_window(self):
        rows = demonstration('autonomous')[:6]
        for row in rows:
            row.update(source_group='hitl', source_category='discarded', control_source='policy',
                       control_source_by_arm=dict(left='policy', right='policy'))
        transitions, _ = m.prepare_episode(rows)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]['action_chunk'], [r['state'] for r in rows[1:]])
        for update in [dict(control_source='human'), dict(control_source='hold'),
                       dict(control_source='handoff'), dict(human_control_active=True),
                       dict(human_contact_observed=True),
                       dict(control_source_by_arm=dict(left='policy', right='human'))]:
            for index in (0, 3, 5):
                bad = [dict(row) for row in rows]; bad[index].update(update)
                with self.subTest(update=update, index=index):
                    self.assertFalse(m.autonomous(bad[index]))
                    transitions, rejected = m.prepare_episode(bad)
                    self.assertEqual(transitions, [])
                    self.assertEqual(rejected['hitl_intervention_or_hold'], 1)

    def test_cli_includes_discarded_and_policy_hitl_preserving_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = root / 'base'; base.mkdir()
            episodes, samples = [], []
            for episode, split, group, category in [
                ('normal-train', 'train', 'teleop', 'kept'),
                ('normal-val', 'val', 'teleop', 'kept'),
                ('normal-test', 'test', 'teleop', 'kept'),
                ('hitl', 'train', 'hitl', 'discarded'),
                ('hitl-human', 'train', 'hitl', 'discarded'),
                ('review', 'val', 'review', 'kept'),
                ('discarded', 'train', 'teleop', 'discarded')]:
                episodes.append(dict(episode_id=episode, source_group=group, source_category=category))
                rows = demonstration(episode, split)[:6]
                if group == 'hitl':
                    for row in rows:
                        row['control_source'] = 'policy'
                        row['control_source_by_arm'] = dict(left='policy', right='policy')
                    if episode == 'hitl-human': rows[3]['human_control_active'] = True
                samples.extend(rows)
                # Excluded episodes have no readable images, proving they are not visited.
                if episode not in ('review', 'hitl-human'):
                    for row in rows:
                        for image in row['images']: (base / image).touch()
            for name, rows in [('episodes', episodes), ('samples', samples)]:
                (base / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            out = root / 'out'
            subprocess.run([sys.executable, str(SCRIPT), '--base', str(base), '--output', str(out)],
                           check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in (out / 'transitions.jsonl').read_text().splitlines()]
            self.assertEqual({row['episode_id']: row['split'] for row in rows},
                             {'normal-train': 'train', 'normal-val': 'val', 'normal-test': 'test',
                              'discarded': 'train', 'hitl': 'train'})
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(Path(image).is_absolute() for row in rows for image in row['images']))
            by_episode = {row['episode_id']: row for row in rows}
            self.assertEqual(by_episode['discarded']['source_category'], 'discarded')
            self.assertEqual(by_episode['hitl']['source_group'], 'hitl')
            self.assertEqual(by_episode['hitl']['source_category'], 'discarded')
            # Curation and HITL provenance must not fabricate supervised negative values.
            for row in rows:
                self.assertNotIn('target', row)
                self.assertNotIn('value_target', row)
                self.assertNotIn('terminal_outcome', row)
                self.assertEqual(row['action_chunk'], [r['state'] for r in demonstration()[1:6]])
            report = json.loads((out / 'audit.json').read_text())
            self.assertEqual(report['action_shape'], [5, 14])
            self.assertEqual(report['counts'], {'train': 3, 'val': 1, 'test': 1})


if __name__ == '__main__':
    unittest.main()
