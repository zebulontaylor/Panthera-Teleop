import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('alignment_q_train', Path(__file__).parents[1] / 'scripts/alignment_q_train.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def dataset(tmp_path, mutate=None):
    rows = []
    for split in ('train', 'val', 'test'):
        for i in range(16):
            rows.append(dict(episode_id=f'{split}-{i}', session_id=split, split=split,
                             observation=[float(i % 2)], next_observation=[float(i % 2)],
                             action=[float(i % 2)], reward=float(i % 2), terminal=True,
                             dt=1., phase='pickup'))
    if mutate:
        mutate(rows)
    path = tmp_path / 'data.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    return q.load_dataset(path)


def test_leakage_and_validation(tmp_path):
    with pytest.raises(ValueError, match='session_id split leakage'):
        dataset(tmp_path, lambda rows: rows[16].update(session_id='train'))
    with pytest.raises(ValueError, match='episode_id split leakage'):
        dataset(tmp_path, lambda rows: rows[16].update(episode_id='train-0'))
    with pytest.raises(ValueError, match='finite'):
        dataset(tmp_path, lambda rows: rows[0].update(observation=[float('nan')] ))


def test_bellman_terminal_and_elapsed_time():
    actual = q.bellman_target(torch.tensor([2., 2.]), torch.tensor([1., 0.]), torch.tensor([2., 2.]), torch.tensor([100., 100.]), .5)
    torch.testing.assert_close(actual, torch.tensor([2., 27.]))


def test_optimization_and_paired_support(tmp_path):
    torch.set_num_threads(1)
    data = dataset(tmp_path)
    output = tmp_path / 'critic.pt'
    result = q.train(data, output, epochs=60, hidden=16, lr=.01, batch_size=16, support_radius=.5)
    assert result['test']['overall']['mc_mse'] < .02
    assert result['history'][-1]['train_loss'] < result['history'][0]['train_loss']
    scorer = q.AlignmentQScorer(output)
    known = scorer.score_candidates([0.], [[0.], [1.], [100.]], 'pickup')
    assert known[0]['supported']
    assert known[1]['q'] is None  # Individually in range but never seen as this pair.
    assert known[2]['q'] is None
    assert scorer.score_candidates([0.], [[0.]], 'insertion')[0]['q'] is None


def test_train_only_normalization_and_monte_carlo(tmp_path):
    data = dataset(tmp_path, lambda rows: rows[16].update(observation=[1000.]))
    result = q.train(data, tmp_path / 'critic.pt', epochs=1, hidden=8)
    assert result['metadata']['observation_mean'] == [.5]
    rows = [dict(session_id='s', episode_id='e', phase='pickup', reward=1., dt=2., terminal=False),
            dict(session_id='s', episode_id='e', phase='pickup', reward=4., dt=1., terminal=True)]
    np.testing.assert_allclose(q.monte_carlo(rows, .5), [2., 4.])
    assert np.isnan(q.monte_carlo(rows[:1], .5)).all()


def test_explicit_episode_holdout(tmp_path):
    original = dataset(tmp_path)
    path = tmp_path / 'data.jsonl'
    rows = original['rows']
    for row in rows:
        row['session_id'] = 'same-session'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='session_id split leakage'):
        q.load_dataset(path)
    assert q.load_dataset(path, split_unit='episode')['split_unit'] == 'episode'


def test_mismatched_audit_is_rejected(tmp_path):
    dataset(tmp_path)
    (tmp_path / 'data.audit.json').write_text(json.dumps({'output_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='audit hash'):
        q.load_dataset(tmp_path / 'data.jsonl')


def test_monte_carlo_separates_phases():
    rows = [dict(session_id='s', episode_id='e', phase='pickup', reward=1., dt=1., terminal=False),
            dict(session_id='s', episode_id='e', phase='insertion', reward=10., dt=1., terminal=True)]
    actual = q.monte_carlo(rows, .9)
    assert np.isnan(actual[0])
    assert actual[1] == 10.


def test_finite_parameters_baseline_and_audit(tmp_path):
    data = dataset(tmp_path)
    for name in ('lr', 'gamma', 'tau', 'expectile', 'support_radius'):
        with pytest.raises(ValueError, match='finite'):
            q.train(data, tmp_path / 'bad.pt', **{name: float('nan')})
    audit = tmp_path / 'data.audit.json'
    audit.write_text(json.dumps({'reward': 'weak_demo_milestone'}))
    data = q.load_dataset(tmp_path / 'data.jsonl')
    result = q.train(data, tmp_path / 'critic.pt', epochs=1, hidden=8)
    assert result['metadata']['dataset_audit']['reward'] == 'weak_demo_milestone'
    metrics = result['test']['overall']
    assert metrics['constant_train_return_mc_mse'] == .25
    assert metrics['support_coverage'] == 1.
    assert metrics['support_reject_rate'] == 0.
    assert metrics['q_std'] >= 0


def test_ablation_removes_visual_prefix_at_inference(tmp_path):
    data = dataset(tmp_path)
    result = q.train(data, tmp_path / 'ablated.pt', epochs=1, hidden=8, ablate_first_n=1)
    assert result['metadata']['observation_mean'] == [0.]
    scorer = q.AlignmentQScorer(tmp_path / 'ablated.pt')
    low = scorer.score_candidates([0.], [[0.]], 'pickup')[0]
    high = scorer.score_candidates([1e5], [[0.]], 'pickup')[0]
    assert low == high
