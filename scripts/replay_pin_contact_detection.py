#!/usr/bin/env python3
"""Replay saved FK/gripper telemetry causally without connecting to hardware.

This measures heuristic trigger coverage, not contact accuracy. It assumes a
frozen right arm after the specified time; recordings are never modified.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pin_contact_detection import PinContactDetector


def replay(directory, limit=80, stride_s=.02, frozen_after_s=3.2):
    episodes=[]
    for path in sorted(Path(directory).glob('*/states.jsonl'))[-limit:]:
        detector=PinContactDetector(dict(enabled=True))
        first=None;last=-float('inf');triggers=[];phases={};skipped=0
        for line in path.open():
            sample=json.loads(line)
            stamp=sample['monotonic_ns']
            if first is None:first=stamp
            now=(stamp-first)/1e9
            if now-last < stride_s:continue
            last=now
            try:
                arms=sample['arms']
                state=[]
                for side in ('left','right'):
                    state+=arms[side]['position_rad']+[arms[side]['gripper']['position']]
                left=arms['left']['tool_pose_base']['position_m']
                right=arms['right']['tool_pose_base']['position_m']
            except KeyError:
                skipped+=1;continue
            result=detector.update(now,state,left,right,eligible=True,
                                   right_frozen=now>=frozen_after_s)
            phases.setdefault(result['phase'],now)
            if result['triggered']:
                triggers.append(dict(t=now,left_xyz=left,right_xyz=right,
                                     grip=state[6],evidence=result))
        episodes.append(dict(episode=path.parent.name,duration_s=last,
                             phases=phases,triggers=triggers,skipped_samples=skipped))
    return dict(schema='pin-contact-heuristic-replay-v1',
        selection=f'latest {limit} sorted episode IDs from {directory}',
        stride_s=stride_s,right_frozen_after_s=frozen_after_s,eligible=True,
        config=PinContactDetector(dict(enabled=True)).config,
        carrier_relative_region_m=PinContactDetector.REGION,
        limitation='trigger coverage only; no frame-level physical contact ground truth',
        episode_count=len(episodes),triggered_count=sum(bool(e['triggers']) for e in episodes),
        episodes=episodes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/kept')
    parser.add_argument('--limit',type=int,default=80)
    parser.add_argument('--stride-s',type=float,default=.02)
    parser.add_argument('--frozen-after-s',type=float,default=3.2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.limit<=0 or args.stride_s<=0 or args.frozen_after_s<0:
        parser.error('positive limit/stride and nonnegative freeze time required')
    report=replay(args.data,args.limit,args.stride_s,args.frozen_after_s)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(episode_count=report['episode_count'],
                          triggered_count=report['triggered_count'],output=str(args.output))))


if __name__=='__main__':main()
