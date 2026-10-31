#!/usr/bin/env python3
"""Offline IQL critic for visual-feature alignment transitions; never commands hardware.

Observations may contain frozen CNN features followed by robot state. The critic
is an MLP over those features, not an image encoder. Actions are measured joint
changes (a behavior proxy, not known commanded actions). Q predicts discounted
annotated return, never a calibrated success probability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import math

import numpy as np
import torch
from torch import nn

PHASES = ('pickup', 'insertion', 'gear_pickup', 'pin_pickup', 'gear_insertion', 'pin_insertion')
ACTION_CONVENTION = 'measured joint delta proxy; not commanded control action'


def load_dataset(path, split_unit="session"):
    if split_unit not in ('session', 'episode'):
        raise ValueError('invalid split unit')
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty dataset')
    dims = None
    owners = {'episode_id': {}, 'session_id': {}}
    for row in rows:
        for key in ('episode_id', 'session_id', 'split', 'observation', 'next_observation', 'action', 'reward', 'terminal', 'dt', 'phase'):
            if key not in row:
                raise ValueError(f'missing {key}')
        if row['split'] not in ('train', 'val', 'test') or row['phase'] not in PHASES:
            raise ValueError('invalid split or phase')
        for key in owners:
            ident = row[key]
            if not isinstance(ident, (str, int)) or isinstance(ident, bool) or (isinstance(ident, str) and not ident.strip()):
                raise ValueError(f'invalid {key}')
            previous = owners[key].setdefault(ident, row['split'])
            if previous != row['split'] and (key == 'episode_id' or split_unit == 'session'):
                raise ValueError(f'{key} split leakage: {ident}')
        arrays = [np.asarray(row[k], dtype=np.float32) for k in ('observation', 'next_observation', 'action')]
        if any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('vectors must be nonempty finite one-dimensional arrays')
        current = tuple(len(a) for a in arrays)
        dims = current if dims is None else dims
        if current != dims or current[0] != current[1]:
            raise ValueError('inconsistent vector dimensions')
        if not isinstance(row['terminal'], bool):
            raise ValueError('terminal must be boolean')
        if not np.isfinite(float(row['reward'])) or not np.isfinite(float(row['dt'])) or float(row['dt']) <= 0:
            raise ValueError('reward must be finite and dt positive')
    if any(not any(r['split'] == split for r in rows) for split in ('train', 'val', 'test')):
        raise ValueError('train, val and test must all be nonempty')
    audit_path = Path(path).with_suffix('.audit.json')
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    digest = hashlib.sha256(raw).hexdigest()
    if audit and ('output_sha256' in audit or 'feature_spec' in audit):
        if audit.get('output_sha256') != digest:
            raise ValueError('dataset audit hash does not match transition bytes')
    return {'audit': audit, 'rows': rows, 'sha256': digest, 'observation_dim': dims[0], 'action_dim': dims[2], 'split_unit': split_unit}


def bellman_target(reward, terminal, dt, next_value, gamma):
    return reward + (1 - terminal) * torch.pow(torch.as_tensor(gamma, device=dt.device), dt) * next_value


def network(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=128):
        super().__init__()
        self.q1 = network(observation_dim + len(PHASES) + action_dim, hidden)
        self.q2 = network(observation_dim + len(PHASES) + action_dim, hidden)
        self.v = network(observation_dim + len(PHASES), hidden)


def features(rows, metadata):
    obs = np.asarray([r['observation'] for r in rows], np.float32)
    nxt = np.asarray([r['next_observation'] for r in rows], np.float32)
    n = metadata['ablate_first_n']
    obs[:, :n] = 0
    nxt[:, :n] = 0
    mean, scale = np.asarray(metadata['observation_mean']), np.asarray(metadata['observation_scale'])
    phase = np.eye(len(PHASES), dtype=np.float32)[[PHASES.index(r['phase']) for r in rows]]
    state = np.concatenate(((obs - mean) / scale, phase), axis=1)
    next_state = np.concatenate(((nxt - mean) / scale, phase), axis=1)
    act = (np.asarray([r['action'] for r in rows]) - metadata['action_mean']) / metadata['action_scale']
    return state.astype(np.float32), next_state.astype(np.float32), act.astype(np.float32)


def monte_carlo(rows, gamma):
    """Returns for complete held-out episodes; truncated tails are excluded."""
    groups = {}
    result = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        groups.setdefault((row['session_id'], row.get('trajectory_id', (row['episode_id'], row['phase']))), []).append(i)
    for indices in groups.values():
        if all('t' in rows[i] for i in indices):
            indices.sort(key=lambda i: float(rows[i]['t']))
        value, complete = 0., False
        for i in reversed(indices):
            row = rows[i]
            if row['terminal']:
                value, complete = 0., True
            if complete:
                value = float(row['reward']) + gamma ** float(row['dt']) * value
                result[i] = value
    return result


def evaluate(model, rows, metadata, device, support=None):
    s, ns, a = [torch.tensor(x, device=device) for x in features(rows, metadata)]
    with torch.no_grad():
        q = torch.minimum(model.q1(torch.cat((s, a), 1)), model.q2(torch.cat((s, a), 1))).flatten()
        target = bellman_target(torch.tensor([r['reward'] for r in rows], device=device), torch.tensor([float(r['terminal']) for r in rows], device=device), torch.tensor([r['dt'] for r in rows], device=device), model.v(ns).flatten(), metadata['gamma'])
    q, target = q.cpu().numpy(), target.cpu().numpy()
    returns = monte_carlo(rows, metadata['gamma'])
    baseline = np.asarray([metadata.get('train_mc_phase_mean', {}).get(r['phase'], metadata.get('train_mc_mean', 0.)) for r in rows])
    distances = None
    if support is not None:
        paired = torch.cat((s, a), 1)
        distances = np.full(len(rows), np.inf)
        for phase in PHASES:
            ids = np.flatnonzero([r['phase'] == phase for r in rows])
            mask = support[:, metadata['observation_dim'] + PHASES.index(phase)] == 1
            if len(ids) and mask.any():
                distances[ids] = torch.cdist(paired[ids], support[mask]).min(1).values.cpu().numpy()
    def metric(mask):
        complete = mask & np.isfinite(returns)
        corr = float(np.corrcoef(q[complete], returns[complete])[0, 1]) if complete.sum() > 1 and np.std(q[complete]) > 1e-10 and np.std(returns[complete]) > 1e-10 else None
        diagnostics = {'q_std': float(np.std(q[mask])), 'q_mean': float(np.mean(q[mask])), 'mc_correlation': corr, 'constant_train_return_mc_mse': float(np.mean((baseline[complete] - returns[complete]) ** 2)) if complete.any() else None}
        if distances is not None:
            diagnostics.update(support_coverage=float(np.mean(distances[mask] <= metadata['support_radius'])), support_reject_rate=float(np.mean(distances[mask] > metadata['support_radius'])))
        return {**diagnostics, 'count': int(mask.sum()), 'td_mse': float(np.mean((q[mask] - target[mask]) ** 2)), 'mc_count': int(complete.sum()), 'mc_mse': float(np.mean((q[complete] - returns[complete]) ** 2)) if complete.any() else None}
    return {'overall': metric(np.ones(len(rows), bool)), 'phases': {p: metric(np.array([r['phase'] == p for r in rows])) for p in PHASES if any(r['phase'] == p for r in rows)}}


def train(dataset, output, epochs=300, batch_size=256, hidden=128, lr=3e-4, gamma=.9, expectile=.7, tau=.005, seed=0, device='cpu', ablate_first_n=0, support_radius=3., cpu_threads=2):
    if not all(math.isfinite(float(v)) for v in (gamma, expectile, tau, lr, support_radius)):
        raise ValueError('hyperparameters must be finite')
    if cpu_threads < 1:
        raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(cpu_threads)
    if not 0 < gamma <= 1 or not 0 < expectile < 1 or not 0 < tau <= 1 or epochs < 1 or batch_size < 1 or lr <= 0 or hidden < 1 or support_radius <= 0:
        raise ValueError('invalid training hyperparameters')
    if not 0 <= ablate_first_n <= dataset['observation_dim']:
        raise ValueError('invalid ablation prefix')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = dataset['rows']; train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    obs = np.asarray([r['observation'] for r in train_rows], np.float32); obs[:, :ablate_first_n] = 0
    act = np.asarray([r['action'] for r in train_rows], np.float32)
    metadata = {'split_unit': dataset.get('split_unit', 'session'), 'evaluation_scope': 'within_session_episode_holdout' if dataset.get('split_unit') == 'episode' else 'heldout_session', 'dataset_sha256': dataset['sha256'], 'observation_dim': dataset['observation_dim'], 'action_dim': dataset['action_dim'], 'hidden': hidden, 'gamma': gamma, 'expectile': expectile, 'tau': tau, 'seed': seed, 'ablate_first_n': ablate_first_n, 'support_radius': support_radius, 'action_convention': ACTION_CONVENTION, 'score_semantics': 'discounted annotated return; not success probability', 'observation_mean': obs.mean(0).tolist(), 'observation_scale': np.maximum(obs.std(0), 1e-3).tolist(), 'action_mean': act.mean(0).tolist(), 'action_scale': np.maximum(act.std(0), 1e-3).tolist()}
    train_returns = monte_carlo(train_rows, gamma)
    valid_returns = np.isfinite(train_returns)
    metadata.update(dataset_audit=dataset.get('audit'), cpu_threads=cpu_threads, train_mc_mean=float(train_returns[valid_returns].mean()) if valid_returns.any() else 0., train_mc_phase_mean={p: float(train_returns[m].mean()) for p in PHASES if (m := valid_returns & np.array([r['phase'] == p for r in train_rows])).any()})
    print(json.dumps({'evaluation_scope': metadata['evaluation_scope'], 'split_unit': metadata['split_unit'], 'score_semantics': metadata['score_semantics'], 'action_convention': ACTION_CONVENTION}), flush=True)
    model = Critic(dataset['observation_dim'], dataset['action_dim'], hidden).to(device)
    import copy
    target = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    s, ns, a = [torch.tensor(x, device=device) for x in features(train_rows, metadata)]
    reward = torch.tensor([r['reward'] for r in train_rows], device=device, dtype=torch.float32)
    terminal = torch.tensor([float(r['terminal']) for r in train_rows], device=device)
    dt = torch.tensor([r['dt'] for r in train_rows], device=device, dtype=torch.float32)
    best, best_state, history = float('inf'), None, []
    for epoch in range(epochs):
        losses = []
        for ids in torch.randperm(len(s), device=device).split(batch_size):
            sa = torch.cat((s[ids], a[ids]), 1)
            with torch.no_grad():
                q_min = torch.minimum(target.q1(sa), target.q2(sa)).flatten()
                y = bellman_target(reward[ids], terminal[ids], dt[ids], model.v(ns[ids]).flatten(), gamma)
            residual = q_min - model.v(s[ids]).flatten()
            v_loss = (torch.where(residual > 0, expectile, 1 - expectile) * residual.square()).mean()
            loss = v_loss + (model.q1(sa).flatten() - y).square().mean() + (model.q2(sa).flatten() - y).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.lerp_(p, tau)
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val_rows, metadata, device, torch.cat((s, a), 1))
        # Select on complete Monte Carlo returns when available, otherwise TD residual.
        criterion = metrics['overall']['mc_mse']
        criterion = metrics['overall']['td_mse'] if criterion is None else criterion
        if criterion < best:
            best = criterion; best_state = copy.deepcopy(model.state_dict()); metadata['best_epoch'] = epoch + 1
        report = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation': metrics}
        history.append(report); print(json.dumps(report), flush=True)
    model.load_state_dict(best_state)
    test = evaluate(model, [r for r in rows if r['split'] == 'test'], metadata, device, torch.cat((s, a), 1))
    checkpoint = {'metadata': metadata, 'model': {k: v.cpu() for k, v in best_state.items()}, 'support_state': s.cpu(), 'support_action': a.cpu()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, output)
    summary = {'checkpoint': str(output), 'metadata': metadata, 'validation': evaluate(model, val_rows, metadata, device, torch.cat((s, a), 1)), 'test': test, 'history': history}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(summary, indent=2))
    return summary


class AlignmentQScorer:
    def __init__(self, checkpoint, device='cpu'):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.metadata = data['metadata']; self.device = device
        self.model = Critic(self.metadata['observation_dim'], self.metadata['action_dim'], self.metadata['hidden']).to(device)
        self.model.load_state_dict(data['model']); self.model.eval()
        self.support = torch.cat((data['support_state'], data['support_action']), 1).to(device)

    def score_candidates(self, observation, actions, phase):
        if phase not in PHASES:
            raise ValueError('unknown phase')
        obs = np.asarray(observation, np.float32); acts = np.asarray(actions, np.float32)
        if obs.shape != (self.metadata['observation_dim'],) or acts.ndim != 2 or acts.shape[1] != self.metadata['action_dim'] or not np.isfinite(obs).all() or not np.isfinite(acts).all():
            raise ValueError('invalid observation/action dimensions or nonfinite values')
        if not len(acts):
            return []
        rows = [{'observation': obs, 'next_observation': obs, 'action': a, 'phase': phase} for a in acts]
        s, _, a = [torch.tensor(x, device=self.device) for x in features(rows, self.metadata)]
        sa = torch.cat((s, a), 1)
        # Require same phase and proximity to one actual paired training sample.
        phase_mask = self.support[:, self.metadata['observation_dim'] + PHASES.index(phase)] == 1
        with torch.no_grad():
            distances = torch.cdist(sa, self.support[phase_mask]).min(1).values if phase_mask.any() else torch.full((len(sa),), float('inf'), device=self.device)
            values = torch.minimum(self.model.q1(sa), self.model.q2(sa)).flatten()
        return [{'supported': bool(d <= self.metadata['support_radius']), 'q': float(q) if d <= self.metadata['support_radius'] else None, 'support_distance': float(d), 'reason': None if d <= self.metadata['support_radius'] else 'outside paired training support'} for q, d in zip(values.cpu(), distances.cpu())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset'); parser.add_argument('--output', required=True)
    for name, default, kind in [('epochs', 300, int), ('cpu-threads', 2, int), ('batch-size', 256, int), ('hidden', 128, int), ('lr', 3e-4, float), ('gamma', .9, float), ('expectile', .7, float), ('tau', .005, float), ('seed', 0, int), ('ablate-first-n', 0, int), ('support-radius', 3., float)]:
        parser.add_argument('--' + name, type=kind, default=default)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split-unit', choices=('session', 'episode'), default='session')
    args = vars(parser.parse_args()); dataset = load_dataset(args.pop('dataset'), split_unit=args.pop('split_unit'))
    result = train(dataset, **args)
    print(json.dumps({'final_test': result['test'], 'checkpoint': result['checkpoint']}))


if __name__ == '__main__':
    main()
