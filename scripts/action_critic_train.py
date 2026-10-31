#!/usr/bin/env python3
"""Fit an action-conditioned regressor to recorded future weak visual values.

Action chunks may be demonstrated next-pose targets (normal demo preparation)
or logged commands. Keep that convention fixed within a training dataset.

Transition JSONL rows: episode_id, session_id, split (train/val/test), images
(three current camera paths), state (D), action_chunk (H,A), future_images
(three recorded future camera paths), dt (elapsed seconds). Paths resolve against
--dataset-root or row image_root. Targets are gamma**dt * sigmoid(V(future)).
There is no dynamics model, reward learning, or calibrated task-success estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T

from state_value_train import VisualValue, atomic_save

WARNING = ('Uncalibrated weak demonstration-relative critic scores; not task-success '
           'probabilities or safe-control signals. Candidate actions outside recorded '
           'data may receive unreliable scores.')


def numeric_array(value, name, ndim):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if array.ndim != ndim or not array.size or any(d == 0 for d in array.shape):
        raise ValueError(f'{name} requires a nonempty {ndim}-dimensional array')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains nonfinite values')
    return array


def read_transitions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty transition manifest')
    episodes, sessions = {}, {}
    state_dim = action_shape = None
    for row in rows:
        split = row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('invalid split')
        for key, mapping in (('episode_id', episodes), ('session_id', sessions)):
            if row[key] is None:
                raise ValueError(f'missing {key}')
            identity = str(row[key])
            if not identity:
                raise ValueError(f'empty {key}')
            if identity in mapping and mapping[identity] != split:
                raise ValueError(f'{key} leakage: {identity}')
            mapping[identity] = split
        for field in ('images', 'future_images'):
            refs = row[field]
            if not isinstance(refs, list) or len(refs) != 3 or not all(isinstance(p, str) and p for p in refs):
                raise ValueError(f'{field} requires exactly three camera paths')
        state = numeric_array(row['state'], 'state', 1)
        actions = numeric_array(row['action_chunk'], 'action_chunk', 2)
        if state_dim is None:
            state_dim, action_shape = len(state), actions.shape
        if len(state) != state_dim or actions.shape != action_shape:
            raise ValueError('state dimensions and action chunk shape must be fixed across all splits')
        dt = float(row['dt'])
        if not math.isfinite(dt) or dt <= 0 or dt > np.finfo(np.float32).max:
            raise ValueError('dt must be finite and positive elapsed seconds')
        row['dt'] = dt
        # Explicit future identities, if supplied, must describe this recorded episode.
        for field, original in (('future_episode_id', 'episode_id'), ('future_session_id', 'session_id'), ('future_split', 'split')):
            if field in row and str(row[field]) != str(row[original]):
                raise ValueError(f'{field} crosses episode/session/split boundary')
    return rows, state_dim, tuple(action_shape)


def fit_normalization(rows):
    train = [r for r in rows if r['split'] == 'train']
    if not train:
        raise ValueError('training transitions required')
    result = {}
    for field, prefix in (('state', 'state'), ('action_chunk', 'action')):
        data = np.stack([np.asarray(r[field], dtype=np.float64) for r in train])
        result[prefix + '_mean'] = torch.tensor(data.mean(axis=0), dtype=torch.float32)
        result[prefix + '_std'] = torch.tensor(data.std(axis=0).clip(min=1e-6), dtype=torch.float32)
    return result


class ActionCritic(nn.Module):
    def __init__(self, state_dim, action_shape, hidden_dim=256):
        super().__init__()
        self.state_dim, self.action_shape = state_dim, tuple(action_shape)
        self.head = nn.Sequential(nn.Linear(256 + state_dim + math.prod(action_shape) + 1, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, z_current, state, action_chunk, dt):
        return torch.sigmoid(self.head(torch.cat((z_current, state, action_chunk.flatten(1), dt.reshape(-1, 1)), dim=-1))).squeeze(-1)


def load_teacher(bundle, device):
    config = bundle['config']
    model = VisualValue(pretrained=False, backbone=config.get('backbone', 'resnet50'))
    model.load_state_dict(bundle['model'])
    model.requires_grad_(False).to(device).eval()
    return model


def image_batch(refs, root, size, device):
    transform = T.Compose([T.Resize((size, size)), T.ToTensor(),
                           T.Normalize((.485, .456, .406), (.229, .224, .225))])
    views = []
    for ref in refs:
        with Image.open(Path(root) / ref) as image:
            views.append(transform(image.convert('RGB')))
    return torch.stack(views).unsqueeze(0).to(device)


def future_value_target(teacher, future_z, dt, gamma):
    """Detached weak future value, discounted per elapsed second."""
    if not math.isfinite(float(gamma)) or not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0,1]')
    elapsed = torch.as_tensor(dt, device=future_z.device, dtype=future_z.dtype)
    if not torch.isfinite(elapsed).all() or (elapsed <= 0).any():
        raise ValueError('dt must be finite and positive')
    with torch.no_grad():
        return torch.sigmoid(teacher.value(future_z)).flatten() * torch.pow(gamma, elapsed)


def encode_rows(rows, teacher, root, size, gamma, device, batch_size=8, progress_every=500):
    """Encode each distinct three-view observation once in bounded batches."""
    if batch_size < 1 or progress_every < 1:
        raise ValueError('encoding batch size and progress interval must be positive')
    observations, indices, current, future = [], {}, [], []
    for row in rows:
        image_root = str(Path(row.get('image_root', root)).resolve())
        for field, destination in (('images', current), ('future_images', future)):
            key = (image_root, tuple(row[field]))
            if key not in indices:
                indices[key] = len(observations)
                observations.append(key)
            destination.append(indices[key])
    encoded = []
    print(json.dumps({'encoding_unique_observations': len(observations),
                      'transition_rows': len(rows), 'encoding_batch_size': batch_size}), flush=True)
    next_progress = progress_every
    with torch.inference_mode():
        for offset in range(0, len(observations), batch_size):
            batch = observations[offset:offset + batch_size]
            images = torch.cat([image_batch(refs, image_root, size, 'cpu')
                                for image_root, refs in batch]).to(device)
            encoded.append(teacher.encode(images).cpu())
            done = offset + len(batch)
            if done >= next_progress or done == len(observations):
                print(json.dumps({'encoded_observations': done, 'total_observations': len(observations)}), flush=True)
                next_progress = (done // progress_every + 1) * progress_every
        features = torch.cat(encoded)
        z = features[current]
        targets = future_value_target(teacher, features[future].to(device),
                                      [r['dt'] for r in rows], gamma).cpu()
    return z.clone(), targets.clone()


def cached_features(path, identity, rows, teacher, root, size, gamma, device,
                    batch_size=8, progress_every=500):
    if path is not None and path.exists():
        try:
            cache = torch.load(path, map_location='cpu', weights_only=False)
            z, target = cache['z'], cache['target']
            if (cache['identity'] == identity and z.shape == (len(rows), 256)
                    and target.shape == (len(rows),) and torch.isfinite(z).all()
                    and torch.isfinite(target).all()):
                print(json.dumps({'feature_cache': str(path), 'status': 'hit'}), flush=True)
                return z.clone(), target.clone()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError):
            pass
        print(json.dumps({'feature_cache': str(path), 'status': 'rebuild'}), flush=True)
    z, target = encode_rows(rows, teacher, root, size, gamma, device, batch_size, progress_every)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save({'identity': identity, 'z': z, 'target': target}, path)
    return z, target


def normalized_inputs(rows, norm):
    state = torch.tensor(np.stack([r['state'] for r in rows]), dtype=torch.float32)
    action = torch.tensor(np.stack([r['action_chunk'] for r in rows]), dtype=torch.float32)
    return ((state - norm['state_mean']) / norm['state_std'],
            (action - norm['action_mean']) / norm['action_std'])


def choose_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device('cuda' if name == 'cuda' or (name == 'auto' and torch.cuda.is_available()) else 'cpu')


def train(args):
    if not 0 < args.gamma <= 1 or not math.isfinite(args.gamma):
        raise ValueError('gamma must be in (0,1], per elapsed second')
    if min(args.epochs, args.batch_size, args.hidden_dim) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('epochs, batch size, hidden dimension and learning rate must be positive')
    if args.cpu_threads < 1:
        raise ValueError('cpu threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rows, state_dim, action_shape = read_transitions(args.manifest)
    if not any(r['split'] == 'val' for r in rows):
        raise ValueError('heldout validation transitions required')
    norm = fit_normalization(rows)
    # Test rows are validated for leakage and shapes but never encoded or selected on.
    used = [r for r in rows if r['split'] in ('train', 'val')]
    device = choose_device(args.device)
    teacher_bundle = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
    teacher = load_teacher(teacher_bundle, device)
    size = int(teacher_bundle['config'].get('image_size', 224))
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    teacher_sha256 = hashlib.sha256(args.teacher_checkpoint.read_bytes()).hexdigest()
    identity = {'format': 'action_critic_features_v1', 'manifest_sha256': manifest_sha256,
                'teacher_sha256': teacher_sha256, 'dataset_root': str(args.dataset_root.resolve()),
                'image_roots': sorted({str(Path(r.get('image_root', args.dataset_root)).resolve()) for r in used}),
                'image_size': size, 'gamma': args.gamma, 'preprocessing': 'resize_imagenet_rgb_v1',
                'splits': ['train', 'val']}
    z, target = cached_features(getattr(args, 'feature_cache', None), identity, used, teacher,
                                args.dataset_root, size, args.gamma, device,
                                getattr(args, 'encode_batch_size', 8), getattr(args, 'encode_progress_every', 500))
    state, action = normalized_inputs(used, norm)
    elapsed = torch.tensor([r['dt'] for r in used], dtype=torch.float32)
    norm['dt_mean'] = elapsed[[r['split'] == 'train' for r in used]].mean()
    norm['dt_std'] = elapsed[[r['split'] == 'train' for r in used]].std(unbiased=False).clamp_min(1e-6)
    elapsed = (elapsed - norm['dt_mean']) / norm['dt_std']
    model = ActionCritic(state_dim, action_shape, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    masks = {sp: torch.tensor([i for i, r in enumerate(used) if r['split'] == sp]) for sp in ('train', 'val')}
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        current_value = torch.sigmoid(teacher.value(z[masks['val']].to(device))).flatten().cpu()
    baselines = {
        'val_current_teacher_mse': float((current_value - target[masks['val']]).square().mean()),
        'val_train_target_mean_mse': float((target[masks['train']].mean() - target[masks['val']]).square().mean())}
    print(json.dumps({'baselines': baselines}), flush=True)
    best = float('inf')
    for epoch in range(args.epochs):
        metrics = dict(baselines)
        for split in ('train', 'val'):
            model.train(split == 'train')
            indices = masks[split]
            if split == 'train':
                indices = indices[torch.randperm(len(indices))]
            squared_error = 0.
            for batch in indices.split(args.batch_size):
                with torch.set_grad_enabled(split == 'train'):
                    prediction = model(z[batch].to(device), state[batch].to(device), action[batch].to(device), elapsed[batch].to(device))
                    loss = (prediction - target[batch].to(device)).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite critic loss')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                squared_error += loss.item() * len(batch)
            metrics[split + '_mse'] = squared_error / len(indices)
        record = dict(epoch=epoch + 1, **metrics)
        bundle = {'format': 'recorded_action_critic_v1', 'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  'teacher': {'model': teacher_bundle['model'], 'config': teacher_bundle['config']},
                  'normalization': norm, 'config': {'state_dim': state_dim, 'action_shape': action_shape,
                  'hidden_dim': args.hidden_dim, 'image_size': size, 'gamma': args.gamma,
                  'action_provenance': sorted({r.get('action_provenance', 'unspecified') for r in used}),
                  'camera_order': ['overhead', 'wrist_port2', 'wrist_port3']},
                  'manifest_sha256': manifest_sha256,
                  'teacher_sha256': teacher_sha256,
                  'epoch': epoch + 1, 'metrics': metrics, 'warning': WARNING}
        atomic_save(bundle, args.output / 'last.pt')
        if metrics['val_mse'] < best:
            best = metrics['val_mse']
            atomic_save(bundle, args.output / 'best.pt')
        model.to(device)
        with (args.output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)


    selected = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'])
    model.eval()
    val_indices = masks['val']
    generator = torch.Generator().manual_seed(args.seed)
    shuffled = action[val_indices][torch.randperm(len(val_indices), generator=generator)]
    report = dict(selected_epoch=selected['epoch'], **selected['metrics'])
    for name, ablated_action in (('shuffled_action', shuffled),
                                 ('mean_action', torch.zeros_like(shuffled))):
        errors = []
        with torch.inference_mode():
            for offset in range(0, len(val_indices), args.batch_size):
                batch = val_indices[offset:offset + args.batch_size]
                prediction = model(z[batch].to(device), state[batch].to(device),
                                   ablated_action[offset:offset + len(batch)].to(device),
                                   elapsed[batch].to(device))
                errors.append((prediction.cpu() - target[batch]).square())
        report['val_' + name + '_mse'] = float(torch.cat(errors).mean())
    report['validation_rows'] = len(val_indices)
    report['warning'] = WARNING
    (args.output / 'validation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'validation_report': report}), flush=True)


def predict_candidates(args):
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if bundle.get('format') != 'recorded_action_critic_v1':
        raise ValueError('unsupported critic checkpoint')
    config, norm = bundle['config'], bundle['normalization']
    state = numeric_array(json.loads(args.state), 'state', 1)
    candidates = numeric_array(json.loads(args.candidates.read_text()), 'candidates', 3)
    if len(state) != config['state_dim'] or tuple(candidates.shape[1:]) != tuple(config['action_shape']):
        raise ValueError('state or candidate action shape differs from training')
    device = choose_device(args.device)
    teacher = load_teacher(bundle['teacher'], device)
    model = ActionCritic(config['state_dim'], config['action_shape'], config['hidden_dim']).to(device)
    model.load_state_dict(bundle['model'])
    model.eval()
    if not math.isfinite(args.dt) or args.dt <= 0:
        raise ValueError('dt must be finite and positive elapsed seconds')
    elapsed = (torch.full((len(candidates),), args.dt) - norm['dt_mean']) / norm['dt_std']
    rows = [{'state': state, 'action_chunk': candidate} for candidate in candidates]
    states, actions = normalized_inputs(rows, norm)
    with torch.inference_mode():
        z = teacher.encode(image_batch(args.images, '.', config['image_size'], device))
        scores = model(z.expand(len(candidates), -1), states.to(device), actions.to(device), elapsed.to(device)).cpu().tolist()
    result = {'scores': scores, 'ranking': sorted(range(len(scores)), key=lambda i: scores[i], reverse=True),
              'warning': WARNING, 'gamma_per_second': config['gamma']}
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--manifest', type=Path, required=True)
    training.add_argument('--teacher-checkpoint', type=Path, required=True)
    training.add_argument('--dataset-root', type=Path, default=Path('.'))
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--gamma', type=float, default=1., help='optional extra discount per elapsed second; teacher already predicts discounted weak value')
    training.add_argument('--feature-cache', type=Path, help='optional reusable feature/target .pt cache; input identities must match')
    training.add_argument('--encode-batch-size', type=int, default=8, help='three-view observations per teacher encoding batch')
    training.add_argument('--encode-progress-every', type=int, default=500)
    training.add_argument('--epochs', type=int, default=30)
    training.add_argument('--batch-size', type=int, default=64)
    training.add_argument('--hidden-dim', type=int, default=256)
    training.add_argument('--lr', type=float, default=1e-3)
    training.add_argument('--cpu-threads', type=int, default=4)
    training.add_argument('--seed', type=int, default=19)
    prediction = commands.add_parser('predict')
    prediction.add_argument('--checkpoint', type=Path, required=True)
    prediction.add_argument('--images', nargs=3, required=True)
    prediction.add_argument('--dt', type=float, required=True, help='candidate horizon in elapsed seconds')
    prediction.add_argument('--state', required=True, help='JSON numeric state vector')
    prediction.add_argument('--candidates', type=Path, required=True, help='JSON array [candidate,horizon,action_dim]')
    prediction.add_argument('--output', type=Path)
    for command in (training, prediction):
        command.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    (train if args.command == 'train' else predict_candidates)(args)


if __name__ == '__main__':
    main()
