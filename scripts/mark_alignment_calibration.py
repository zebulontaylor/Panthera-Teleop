#!/usr/bin/env python3
"""Mark operator-demonstrated calibration poses in the running recording.

Read-only with respect to the collector and robot. Writes a separate session
manifest; never enables motion or marks a calibration as verified.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def status(url):
    with urlopen(url + '/api/status', timeout=3) as response:
        data = json.load(response)
    if data.get('demo') or data.get('error'):
        raise ValueError('A healthy real recording session is required')
    return data


def latest_sample(path):
    # The writer can be in the middle of its final line; use complete lines only.
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        offset = max(0, size - 262144)
        stream.seek(offset)
        raw = stream.read()
    lines = raw.split(b'\n')[:-1]
    if offset:
        lines = lines[1:]
    for line in reversed(lines):
        if line.strip():
            return json.loads(line)
    raise ValueError('No complete state sample yet')


def mark(url, data_root, label):
    before = status(url)
    episode = before.get('active')
    if (before.get('phase') != 'recording' or not episode
            or Path(episode).name != episode or episode in ('.', '..')):
        raise ValueError('Start a recording and hold the demonstrated pose first')
    directory = data_root / 'pending' / episode
    meta = json.loads((directory / 'metadata.json').read_text())
    sample = latest_sample(directory / 'states.jsonl')
    age = (time.monotonic_ns() - sample['monotonic_ns']) / 1e9
    if not 0 <= age <= 1:
        raise ValueError('Recording feedback is stale; marker was not saved')
    after = status(url)
    if after.get('active') != episode or after.get('phase') != 'recording':
        raise ValueError('Episode changed during measurement; marker was not saved')
    if not all(c.get('ok') for c in after.get('cameras', {}).values()) or len(after.get('cameras', {})) != 3:
        raise ValueError('All three cameras must be healthy')
    return dict(schema_version=1, episode_id=episode, label=label,
                monotonic_ns=sample['monotonic_ns'],
                time_s=(sample['monotonic_ns']-meta['start_monotonic_ns'])/1e9,
                arms=sample['arms'], camera_status=after['cameras'],
                camera_mapping=meta['config'].get('alignment_assist', {}).get('camera_mapping'),
                annotation_source='operator demonstration; not automatic success detection',
                verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8090')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/alignment-calibration/live-markers.jsonl')
    parser.add_argument('--label', required=True,
                        choices=('reference', 'motion-start', 'motion-end', 'empty'))
    args = parser.parse_args()
    try:
        row = mark(args.url, args.data_root, args.label)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Calibration marker not saved: {exc}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps({k: row[k] for k in ('episode_id', 'label', 'time_s', 'verified')}))


if __name__ == '__main__':
    main()
