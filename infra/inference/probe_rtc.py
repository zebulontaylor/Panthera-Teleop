#!/usr/bin/env python3
"""Inference-only RTC smoke test; never connects to robot hardware."""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with urlopen(args.url + '/health', timeout=5) as response:
        health = json.load(response)
    assert health['status'] == 'ready', health
    assert health['rtc']['supported'] is True, health['rtc']
    image = io.BytesIO()
    Image.new('RGB', (256, 256), (128, 128, 128)).save(image, format='JPEG')
    encoded = base64.b64encode(image.getvalue()).decode()
    body = dict(state=[0., .25, .25, 0., 0., 0., 1.2]*2,
                images={name: encoded for name in health['contract']['image_crops_xyxy']})
    records = []
    for sequence in range(4):
        payload = dict(body, sequence=sequence)
        if sequence:
            payload['rtc'] = dict(previous_actions=actions[1:11], overlap=10)
        started = time.monotonic()
        with urlopen(Request(args.url + '/infer', data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json'}), timeout=40) as response:
            result = json.load(response)
        elapsed = time.monotonic() - started
        actions = result['actions']
        assert result['checkpoint'] == health['checkpoint'] and result['sequence'] == sequence
        assert result['mock'] is False and len(actions) == 50
        assert all(len(row) == 14 and all(math.isfinite(v) for v in row) for row in actions)
        if sequence:
            assert result['rtc']['overlap'] == 10
        record = dict(sequence=sequence, guided=bool(sequence), latency_s=elapsed,
                      acknowledged_overlap=result.get('rtc', {}).get('overlap', 0),
                      within_request_age_limit=elapsed < health['contract']['max_observation_age_s'])
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(health=health, synthetic=True, robot_motion=False, probes=records)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
