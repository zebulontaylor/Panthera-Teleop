#!/usr/bin/env python3
"""Prepare demonstrations and autonomous HITL segments under the BC next-pose convention.

The five future measured poses are demonstration position-target proxies, not
executed motor commands. Teleop kept/discarded and autonomous HITL windows are included. Frozen
visual-teacher episode/session/split identities are retained exactly.
"""
import argparse
import collections
import json
import math
from pathlib import Path


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def valid_state(row):
    state = row.get('state')
    return (isinstance(state, list) and len(state) == 14
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in state))


def eligible_episode(row):
    return (row.get('source_group') == 'teleop' and row.get('source_category') in ('kept', 'discarded')) or row.get('source_group') == 'hitl'


def autonomous(row):
    ownership = row.get('control_source_by_arm')
    return (row.get('control_source') == 'policy'
            and row.get('human_control_active') is not True
            and row.get('human_contact_observed') is not True
            and (not ownership or all(ownership.get(side) == 'policy' for side in ('left', 'right'))))


def prepare_episode(rows, horizon=1.0, steps=5, max_gap=.30):
    """Return transitions and rejection counts for one frozen-split episode."""
    rows = sorted(rows, key=lambda row: row['t'])
    transitions, rejected = [], collections.Counter()
    period = horizon / steps
    for start in range(len(rows)):
        window = rows[start:start + steps + 1]
        if len(window) != steps + 1:
            rejected['insufficient_future'] += 1
            continue
        current, future = window[0], window[-1]
        if current.get('source_group') == 'hitl' and not all(autonomous(row) for row in window):
            rejected['hitl_intervention_or_hold'] += 1
            continue
        if any(not valid_state(row) for row in window):
            rejected['invalid_state'] += 1
            continue
        if any(row.get('terminal') for row in window[:-1]):
            rejected['terminal_crossing'] += 1
            continue
        if any(any(row.get(key) != current.get(key) for key in
                   ('episode_id', 'session_id', 'split', 'recording_segment')) for row in window):
            rejected['identity_or_segment_change'] += 1
            continue
        deltas = [b['t'] - a['t'] for a, b in zip(window, window[1:])]
        # Reject missing samples and materially irregular cadence, rather than
        # joining across pauses or treating a far-future state as the next target.
        if any(delta <= 0 or delta > max_gap for delta in deltas):
            rejected['sample_gap'] += 1
            continue
        if any(abs((row['t'] - current['t']) - k * period) > .12
               for k, row in enumerate(window)):
            rejected['target_time_gap'] += 1
            continue
        if any(float(row.get('state_time_gap_s', 0)) > .12 for row in window):
            rejected['state_time_gap'] += 1
            continue
        if any(not isinstance(row.get('images'), list) or len(row['images']) != 3 for row in window):
            rejected['invalid_images'] += 1
            continue
        transitions.append(dict(
            episode_id=current['episode_id'], session_id=current['session_id'], split=current['split'],
            images=current['images'], future_images=future['images'], state=current['state'],
            action_chunk=[row['state'] for row in window[1:]], dt=future['t'] - current['t'],
            t=current['t'], future_t=future['t'], action_times_s=[row['t'] for row in window[1:]],
            source_group=current.get('source_group', 'teleop'), source_category=current.get('source_category'),
            action_provenance='kinesthetic_demo_next_pose_proxy',
            action_semantics='Absolute future measured joint poses used as demonstration position targets; not sent motor commands',
            camera_order=['overhead', 'wrist_port2', 'wrist_port3'],
            recording_segment=current.get('recording_segment')))
    return transitions, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('work/state-value-20261029/full_trajectory'))
    parser.add_argument('--output', type=Path, default=Path('work/action-critic-mixed-20261029/dataset'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = {row['episode_id']: row for row in read_rows(args.base / 'episodes.jsonl') if eligible_episode(row)}
    grouped = collections.defaultdict(list)
    for row in read_rows(args.base / 'samples.jsonl'):
        if row['episode_id'] in episodes:
            row['source_group'] = episodes[row['episode_id']]['source_group']
            row['source_category'] = episodes[row['episode_id']]['source_category']
            grouped[row['episode_id']].append(row)
    counts, rejections, audit = collections.Counter(), collections.Counter(), []
    output = args.output / 'transitions.jsonl'
    with output.open('w') as stream:
        for eid, rows in sorted(grouped.items()):
            transitions, rejected = prepare_episode(rows)
            for row in transitions:
                # Absolute references allow the trainer to resolve original compact
                # camera images without copying or re-encoding this source dataset.
                for field in ('images', 'future_images'):
                    row[field] = [str((args.base / p).resolve()) for p in row[field]]
                    if not all(Path(p).is_file() for p in row[field]):
                        raise FileNotFoundError(f'Missing source image for {eid}')
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                counts[row['split']] += 1
            rejections.update(rejected)
            audit.append(dict(episode_id=eid, session_id=rows[0]['session_id'], split=rows[0]['split'],
                              source_group=episodes[eid]['source_group'], source_category=episodes[eid]['source_category'],
                              samples=len(rows), transitions=len(transitions)))
    report = dict(manifest=str(output.resolve()), counts=dict(counts), rejections=dict(rejections),
                  episodes=audit, action_shape=[5, 14], nominal_horizon_s=1.0,
                  action_provenance='kinesthetic_demo_next_pose_proxy',
                  action_semantics='Absolute future measured joint poses used as demonstration position targets; not sent motor commands',
                  inclusion='Teleop kept/discarded and HITL autonomous policy-only windows; exact frozen sample splits retained',
                  excluded='Teleop review; HITL intervention/hold/contact windows; gaps, invalid states, segment changes, terminal crossings; no fresh Oct29 sources')
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(counts=counts, episodes=len(audit), manifest=str(output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
