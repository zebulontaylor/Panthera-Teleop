#!/usr/bin/env python3
"""Paired seeded CNN/proprioception ablation; retains every run, never picks on test."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.alignment_q_features import VISUAL_PREFIX_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        for variant, ablation in [('cnn_history', 0), ('proprio_only', VISUAL_PREFIX_DIM)]:
            checkpoint = args.output / f'{variant}.seed{seed}.pt'
            command = [sys.executable, str(ROOT / 'scripts/alignment_q_train.py'),
                       str(args.dataset), '--output', str(checkpoint), '--split-unit', 'episode',
                       '--seed', str(seed), '--epochs', str(args.epochs), '--hidden', '64',
                       '--ablate-first-n', str(ablation), '--cpu-threads', '2']
            started = time.monotonic()
            with checkpoint.with_suffix('.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(checkpoint.with_suffix('.pt.json').read_text())
            row = dict(variant=variant, seed=seed, checkpoint=str(checkpoint.resolve()),
                       elapsed_s=time.monotonic()-started, best_epoch=report['metadata']['best_epoch'],
                       validation=report['validation'], test=report['test'])
            results.append(row)
            print(json.dumps(row), flush=True)
    aggregate = {}
    for variant in ('cnn_history', 'proprio_only'):
        selected = [r for r in results if r['variant'] == variant]
        aggregate[variant] = {
            key: dict(mean=statistics.mean(values), min=min(values), max=max(values))
            for key in ('mc_mse', 'constant_train_return_mc_mse', 'support_coverage')
            if (values := [r['test']['overall'][key] for r in selected])}
    report = dict(runs=results, aggregate=aggregate,
                  selection='Each checkpoint selected on validation MC MSE; all preset seeds reported.',
                  interpretation='Same-session weak milestone return prediction, not physical task success or causal action-ranking validation.',
                  cnn_helped_return_prediction=aggregate['cnn_history']['mc_mse']['mean'] < aggregate['proprio_only']['mc_mse']['mean'])
    (args.output / 'experiment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'aggregate': aggregate}), flush=True)


if __name__ == '__main__':
    main()
