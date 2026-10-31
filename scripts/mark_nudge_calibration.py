#!/usr/bin/env python3
"""Annotate a held calibration endpoint; reads collector, never commands it."""
import argparse
import json
from pathlib import Path
from mark_alignment_calibration import mark

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8090')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--session', required=True)
    p.add_argument('--trial', required=True)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--label', choices=('start', 'end', 'reference'), required=True)
    p.add_argument('--requested-delta-mm', nargs=3, type=float)
    p.add_argument('--external-delta-mm', nargs=3, type=float,
        help='Independently measured tool displacement; do not copy encoder FK here')
    args = p.parse_args()
    row = mark(args.url, args.data_root, args.label)
    row.update(session_id=args.session, trial_id=args.trial, split=args.split)
    for source, key in ((args.requested_delta_mm, 'requested_delta_m'), (args.external_delta_mm, 'external_delta_m')):
        if source is not None:
            row[key] = [v/1000 for v in source]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps({key: row[key] for key in ('episode_id', 'session_id', 'trial_id', 'split', 'label', 'time_s')}))


if __name__ == '__main__':
    main()
