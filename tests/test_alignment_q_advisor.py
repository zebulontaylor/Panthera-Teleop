import copy
import json
from pathlib import Path
import unittest

from app.alignment_q_advisor import AlignmentQAdvisor, ACTION_CONVENTION, ACTION_SEMANTICS
from app.alignment_q_features import feature_spec, OBSERVATION_DIM


class FakeScorer:
    def __init__(self):
        self.metadata = dict(observation_dim=OBSERVATION_DIM, action_dim=14,
                             action_convention=ACTION_CONVENTION,
                             dataset_audit=dict(feature_spec=feature_spec(), action_semantics=ACTION_SEMANTICS, action_horizon_s=.1))
        self.calls = []
        self.results = [dict(supported=True, q=2.), dict(supported=False, q=None)]

    def score_candidates(self, observation, actions, phase):
        self.calls.append((observation, actions, phase))
        return self.results[:len(actions)]


class AlignmentQAdvisorTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'logs/alignment-calibration/predictions.phases10hz.corrected-order.jsonl'
        with path.open() as stream:
            self.row = json.loads(next(stream))
        self.scorer = FakeScorer()
        self.advisor = AlignmentQAdvisor(None, scorer=self.scorer)

    def score(self, timestamp=1_000_000_000, phase='gear_pickup', **kwargs):
        row = copy.deepcopy(self.row)
        row['monotonic_ns'] = timestamp
        row['frame_timestamps_ns'] = {name: timestamp for name in row['camera_names']}
        row.update(kwargs.pop('prediction_fields', {}))
        return self.advisor.score(row, row['arms'], [[0.] * 14, [.01] * 14], phase,
                                  now_ns=kwargs.pop('now_ns', timestamp + 5_000_000),
                                  measured_state_ns=kwargs.pop('measured_state_ns', timestamp), **kwargs)

    def test_advisory_support_and_history(self):
        first = self.score()
        self.assertTrue(first['advisory_only'])
        self.assertEqual(first['recommended_index'], 0)
        self.assertFalse(first['history_valid'])
        self.assertTrue(self.score(1_100_000_000)['history_valid'])
        self.assertFalse(self.score(1_200_000_000, phase='pin_pickup')['history_valid'])
        self.assertFalse(self.score(1_400_000_000, phase='pin_pickup')['history_valid'])
        self.advisor.reset()
        self.assertFalse(self.score()['history_valid'])

    def test_all_unsupported_has_no_recommendation(self):
        self.scorer.results = [dict(supported=False, q=None)] * 2
        self.assertIsNone(self.score()['recommended_index'])

    def test_scoring_failure_resets_history_and_consumes_timestamp(self):
        self.score()
        original = self.scorer.score_candidates
        def fail(*args):
            raise ValueError('candidate rejected')
        self.scorer.score_candidates = fail
        self.assertEqual(self.score(1_100_000_000)['status'], 'rejected')
        self.scorer.score_candidates = original
        self.assertEqual(self.score(1_100_000_000)['status'], 'rejected')
        self.assertFalse(self.score(1_200_000_000)['history_valid'])

    def test_checkpoint_contract(self):
        for key, value in [('action_dim', 12), ('action_convention', 'velocity')]:
            scorer = FakeScorer()
            scorer.metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                AlignmentQAdvisor(None, scorer=scorer)
        scorer = FakeScorer()
        scorer.metadata['dataset_audit']['action_horizon_s'] = .2
        with self.assertRaises(ValueError):
            AlignmentQAdvisor(None, scorer=scorer)
        scorer = FakeScorer()
        scorer.metadata['dataset_audit']['feature_spec']['checkpoint_ids']['gear'] = 'wrong'
        with self.assertRaises(ValueError):
            AlignmentQAdvisor(None, scorer=scorer)

    def test_reject_timing_phase_and_horizon(self):
        for kwargs in (dict(now_ns=1_300_000_000), dict(measured_state_ns=1_020_000_000),
                       dict(horizon_s=.2), dict(phase='gear'),
                       dict(prediction_fields={'monotonic_ns': 0}),
                       dict(prediction_fields={'frame_timestamps_ns': {'overhead': 1_000_000_000,
                            'wrist_port3': 1_050_000_000, 'wrist_port2': 1_000_000_000}})):
            self.advisor.reset()
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.score(**kwargs)['status'], 'rejected')
        self.advisor.reset()
        self.score()
        self.assertEqual(self.score()['status'], 'rejected')
        self.assertFalse(self.score(1_100_000_000)['history_valid'])
        self.assertEqual(len(self.scorer.calls), 2)


if __name__ == '__main__':
    unittest.main()
