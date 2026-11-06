#!/usr/bin/env python3
"""Compare frozen held-pin checkpoints, with no fitting or checkpoint selection."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from app.held_pin_cnn import HeldPinCNN, SCHEMA
from scripts.train_held_pin_cnn import prepare, evaluate, summarize, file_sha256


def summary(records):
    result = {'count': len(records)}
    for name in ('top', 'bottom'):
        key = f'{name}_error_y_px_480'
        values = [r[key] for r in records if r[key] is not None]
        accepted = [r[key] for r in records if r[key] is not None and r['accepted']]
        result[name] = dict(summarize(values), misses_gt10px=sum(v > 10 for v in values),
                            accepted=summarize(accepted), accepted_misses_gt10px=sum(v > 10 for v in accepted))
    result.update(top_visible_count=sum(r['top_visible'] for r in records),
                  top_hidden_count=sum(not r['top_visible'] for r in records),
                  accepted_count=sum(r['accepted'] for r in records),
                  accepted_visible_count=sum(r['accepted'] and r['top_visible'] for r in records),
                  false_visible_count=sum(r['accepted'] and not r['top_visible'] for r in records))
    return result


def run(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    _, rows, axis = prepare(args.real_manifest, args.sim_dataset)
    with np.load(args.sim_dataset) as data:
        strata = {f'sim_{i}': {key: str(data[key][i]) for key in ('lighting_domain', 'variant') if key in data.files}
                  for i in range(len(data['images']))}
    splits = ['validation', 'test'] if args.split == 'both' else [args.split]
    output = {'protocol': 'Frozen comparison only; test scores must not select checkpoints.',
              'real_manifest': str(Path(args.real_manifest).resolve()),
              'real_manifest_sha256': file_sha256(args.real_manifest),
              'sim_dataset': str(Path(args.sim_dataset).resolve()),
              'sim_dataset_sha256': file_sha256(args.sim_dataset),
              'shadow_seeds': {'validation': 9721, 'test': 19721}, 'checkpoints': []}
    for checkpoint in args.checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        metadata = payload['metadata']
        if metadata.get('schema') != SCHEMA or metadata.get('camera') != 'left_wrist':
            raise ValueError(f'Wrong checkpoint schema or camera: {checkpoint}')
        if abs(float(metadata.get('axis_x_fraction', axis))-axis) > 1e-8:
            raise ValueError('Checkpoint and dataset axes differ')
        model = HeldPinCNN().to(device)
        model.load_state_dict(payload['model'])
        item = {'checkpoint': str(Path(checkpoint).resolve()), 'sha256': file_sha256(checkpoint),
                'selected_epoch': metadata.get('selected_epoch'), 'evaluations': {}}
        for split in splits:
            subset = [r for r in rows if r['split'] == split]
            if not subset:
                raise ValueError(f'No rows in requested split: {split}')
            for condition, seed in [('ordinary', None), ('extreme_shadows', output['shadow_seeds'][split])]:
                evaluation = evaluate(model, subset, device, axis,
                                      threshold=float(metadata.get('visibility_threshold', .6)), shadow_seed=seed)
                records = evaluation['records']
                for record in records:
                    record.update(strata.get(record['id'], {}))
                report = {'domains': {domain: summary([r for r in records if r['domain'] == domain])
                                      for domain in ('real', 'sim')}, 'strata': {}, 'records': records}
                for key in ('lighting_domain', 'variant', 'group_id'):
                    report['strata'][key] = {value: summary([r for r in records if r.get(key) == value])
                                             for value in sorted({r[key] for r in records if key in r})}
                item['evaluations'][f'{split}/{condition}'] = report
                print(json.dumps({'checkpoint': checkpoint, 'evaluation': f'{split}/{condition}',
                                  'domains': report['domains']}), flush=True)
        output['checkpoints'].append(item)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', action='extend', required=True)
    parser.add_argument('--real-manifest', required=True)
    parser.add_argument('--sim-dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['validation', 'test', 'both'], default='validation')
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
