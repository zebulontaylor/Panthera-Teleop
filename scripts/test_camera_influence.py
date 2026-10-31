#!/usr/bin/env python3
"""Prediction-only randomized camera ablation against the deployed model.

Never opens robot devices or posts to the controller. Frozen state, task and
images; repeated randomized requests separate image effects from sampler noise.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from remote_inference import CONTRACT, image_payload, request


def idle(status):
    if (status['mode'] != 'holding' or status.get('playing') or
            status.get('rollout_active') or status.get('simulation')):
        raise RuntimeError('Controller must be physically holding and idle')


def metrics(baseline, changed, seed):
    # Each chunk is one independent sample; timesteps are NOT replicates.
    a = baseline[:, :15, :][:, :, JOINTS].reshape(len(baseline), -1)
    b = changed[:, :15, :][:, :, JOINTS].reshape(len(changed), -1)
    points = np.concatenate([a, b])
    distances = np.sqrt(np.mean((points[:, None] - points[None, :]) ** 2, axis=2))
    n = len(a)

    def statistic(ix, iy):
        return (2 * distances[np.ix_(ix, iy)].mean()
                - distances[np.ix_(ix, ix)].mean()
                - distances[np.ix_(iy, iy)].mean())

    observed = statistic(np.arange(n), np.arange(n, 2*n))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(9999):
        # Preserve acquisition blocks: swap condition labels only within each
        # matched repeat, so a changing sampler/server does not confound time.
        swap = rng.integers(0, 2, size=n)
        ix = np.arange(n) + swap*n
        iy = np.arange(n) + (1-swap)*n
        exceed += statistic(ix, iy) >= observed
    mean_delta = changed.mean(axis=0) - baseline.mean(axis=0)
    return {
        'energy_statistic_rad': float(observed),
        'permutation_p': (exceed + 1) / 10000,
        'bonferroni_p_four_comparisons': min(1., 4 * (exceed + 1) / 10000),
        'first15_mean_trajectory_rms_difference_deg': float(np.rad2deg(np.sqrt(np.mean((b.mean(0)-a.mean(0))**2)))),
        'left_first15_mean_trajectory_rms_difference_deg': float(np.rad2deg(np.sqrt(np.mean(mean_delta[:15, :6]**2)))),
        'right_first15_mean_trajectory_rms_difference_deg': float(np.rad2deg(np.sqrt(np.mean(mean_delta[:15, 7:13]**2)))),
        'first15_max_mean_joint_difference_deg': float(np.rad2deg(np.max(np.abs(mean_delta[:15, JOINTS])))),
    }


JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--dashboard', default='http://127.0.0.1:8093')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 8:
        parser.error('Use at least eight independent chunks per condition')
    args.output.mkdir(parents=True, exist_ok=False)
    status = request(args.dashboard, '/api/status')
    idle(status)
    health = request(args.url, '/health')
    if health['status'] != 'ready' or health['contract'] != CONTRACT:
        raise RuntimeError('Expected ready real model with current contract')
    cameras = list(CONTRACT['image_crops_xyxy'])
    images = {}
    for name in cameras:
        with urlopen(args.dashboard + '/camera/' + name, timeout=5) as response:
            jpeg = response.read()
        (args.output / f'{name}.jpg').write_bytes(jpeg)
        images[name] = image_payload(jpeg, name)
    blank = io.BytesIO()
    Image.new('RGB', (256, 256), (0, 0, 0)).save(blank, format='JPEG', quality=95, subsampling=0)
    black = base64.b64encode(blank.getvalue()).decode()
    variants = {'real': images}
    variants.update({f'black_{name}': dict(images, **{name: black}) for name in cameras})
    variants['black_all'] = dict.fromkeys(cameras, black)
    payload = dict(sequence=0, state=status['measured'], images=images)
    (args.output / 'frozen-input.json').write_text(json.dumps(payload))
    # Strip dashboard session token from archived metadata.
    status.pop('token', None)
    results = {name: [] for name in variants}
    rng = np.random.default_rng(20261028)
    sequence = 0
    with (args.output / 'responses.jsonl').open('w') as stream:
        for repeat in range(args.repeats):
            for name in rng.permutation(list(variants)):
                idle(request(args.dashboard, '/api/status'))
                started = time.monotonic()
                response = request(args.url, '/infer', dict(payload, sequence=sequence, images=variants[name]), timeout=30)
                if response.get('mock') or response['checkpoint'] != health['checkpoint']:
                    raise RuntimeError('Mock model or checkpoint changed')
                if response['sequence'] != sequence or response['fps'] != CONTRACT['fps']:
                    raise RuntimeError('Response sequence/rate mismatch')
                actions = np.asarray(response['actions'])
                if actions.shape != (50, 14) or not np.isfinite(actions).all():
                    raise RuntimeError('Malformed response')
                results[name].append(actions)
                stream.write(json.dumps(dict(condition=name, repeat=repeat, latency_s=time.monotonic()-started, response=response))+'\n')
                stream.flush()
                sequence += 1
            print(f'Completed {repeat+1}/{args.repeats} randomized blocks', flush=True)
    baseline = np.asarray(results['real'])
    report = dict(
        checkpoint=health['checkpoint'], contract=health['contract'],
        physical_actions_executed=False, sampler_seed_controlled=False,
        design='Frozen live images/state/task; randomized blocked order; independent stochastic chunks; black image interventions',
        statistical_test='Energy distance on first15 arm joints; 9999 within-block label swaps; four-way Bonferroni',
        repeats_per_condition=args.repeats, action_horizon_tested=15,
        input_sha256={n: hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in images.items()},
        baseline_first15_joint_sampler_sd_deg=float(np.rad2deg(np.sqrt(np.mean(np.var(baseline[:, :15, JOINTS], axis=0, ddof=1))))),
        conditions={name: metrics(baseline, np.asarray(values), 20261028+i)
                    for i,(name,values) in enumerate(results.items()) if name != 'real'},
        initial_controller=status,
        final_controller={k:v for k,v in request(args.dashboard, '/api/status').items() if k != 'token'},
        limitations=['One scene and pose', 'Black images are outside normal camera distribution',
                     'Measures predicted joint targets, not successful object-responsive behavior or physical tracking',
                     'Sampling seed unavailable: randomized repeated statistical test rather than fixed-noise paired test'],
    )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('baseline_first15_joint_sampler_sd_deg', 'conditions')}, indent=2))


if __name__ == '__main__':
    main()
