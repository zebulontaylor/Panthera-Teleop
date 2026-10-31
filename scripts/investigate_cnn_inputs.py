#!/usr/bin/env python3
"""Offline paired camera-order experiment on exact manually annotated JPEGs.

Run using ../RobotArmTraining/.venv-cnns/bin/python. No robot imports or commands.
The target-bore score is an optimistic nearest-of-four candidate error, not a
claim that a carrier_hole_N head identifies the intended physical bore.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORDERS = {'native': ['overhead', 'wrist_port2', 'wrist_port3'],
          'physical_swap': ['overhead', 'wrist_port3', 'wrist_port2']}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearest(rows, timestamp):
    return min(rows, key=lambda row: abs(row['monotonic_ns'] - timestamp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT/'data/calibration/manual-labels/assembled/labels.json')
    parser.add_argument('--model-root', type=Path, default=ROOT.parent/'RobotArmTraining/outputs/panthera-assembly-cnns')
    parser.add_argument('--output-prefix', type=Path, default=ROOT/'logs/alignment-calibration/investigation-inputs')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.model_root))
    from tools.gear_alignment_model import GearAlignmentPredictor
    from tools.pin_perception_model import PinPerceptionPredictor
    checkpoints = {'gear': args.model_root/'gear_alignment/real_v4/adapted.pt',
                   'pin': args.model_root/'pin_perception/carrier_v4/adapted.pt'}
    models = {'gear': GearAlignmentPredictor(checkpoints['gear'], device=args.device),
              'pin': PinPerceptionPredictor(checkpoints['pin'], device=args.device)}
    observations = json.loads(args.labels.read_text())['observations']
    selected = [o for o in observations if any(l['visibility']=='visible' and k in ('gear_center','pin_tip','target_bore')
                for v in o['labels']['views'].values() for k,l in v['landmarks'].items())]
    episodes = {}
    for o in selected:
        if o['episode_id'] not in episodes:
            p=Path(o['images']['wrist_port2']['path']).parents[2]
            episodes[o['episode_id']] = (p,read_jsonl(p/'states.jsonl'),read_jsonl(p/'frames.jsonl'))
    results=[]; errors=[]; overlay_done=set()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for count,o in enumerate(selected):
        episode,states,frames=episodes[o['episode_id']]
        timestamp=round(sum(info['monotonic_ns'] for info in o['images'].values())/2)
        state=nearest(states,timestamp)
        if abs(state['monotonic_ns']-timestamp)>20e6:raise ValueError('state skew >20ms')
        overhead=nearest([f for f in frames if f['camera']=='overhead'], timestamp)
        if abs(overhead['monotonic_ns']-timestamp)>60e6:raise ValueError('overhead skew >60ms')
        paths={c:Path(info['path']) for c,info in o['images'].items()}
        paths['overhead']=episode/overhead['path']
        for camera, info in o['images'].items():
            if hashlib.sha256(paths[camera].read_bytes()).hexdigest() != info['sha256']:
                raise ValueError(f'Manual-label image changed: {paths[camera]}')
        images={}
        for c,p in paths.items():
            with Image.open(p) as im:
                if im.size!=(640,480):raise ValueError('Native JPEG dimensions required')
                images[c]=np.asarray(im.convert('RGB'))
        xyz=np.asarray([v for s in ('left','right') for v in state['arms'][s]['tool_pose_base']['position_m']],dtype=np.float32)
        row={'observation_id':o['observation_id'],'phase':o['phase'],'episode_id':o['episode_id'],
             'images':{c:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for c,p in paths.items()},
             'state_skew_s':(state['monotonic_ns']-timestamp)/1e9,
             'overhead_skew_s':(overhead['monotonic_ns']-timestamp)/1e9,'xyz_m':xyz.tolist(),'orders':{}}
        task='gear' if o['phase']=='gear' else 'pin'
        for order,cameras in ORDERS.items():
            p=models[task].predict(np.stack([images[c] for c in cameras]),**({'end_effector_positions':xyz} if task=='pin' else {}))
            pred={k:v.tolist() if hasattr(v,'tolist') else list(v) for k,v in p.items()
                  if k in ('keypoints_2d','landmark_names','camera_names','visibility_probability','confidence','pixel_std')}
            row['orders'][order]={'physical_input_cameras':cameras,'task':task,'prediction':pred}
            names=pred['landmark_names']; points=np.asarray(pred['keypoints_2d'])
            for camera,v in o['labels']['views'].items():
                slot=cameras.index(camera)
                for name,label in v['landmarks'].items():
                    if name not in ('gear_center','pin_tip','target_bore') or label['visibility']!='visible':continue
                    candidates=[j for j,n in enumerate(names) if n.startswith('carrier_hole_')] if name=='target_bore' else [names.index(name)]
                    truth=np.asarray(label['xy_px']); ds=np.linalg.norm(points[slot,candidates]-truth,axis=-1);j=candidates[int(np.argmin(ds))]
                    errors.append({'observation_id':o['observation_id'],'split':o['split'],'task':task,'camera':camera,'order':order,
                                   'landmark':name,'manual_x':truth[0].item(),'manual_y':truth[1].item(),
                                   'prediction_x':points[slot,j,0].item(),'prediction_y':points[slot,j,1].item(),
                                   'error_px':float(min(ds)),'candidate_head':names[j],
                                   'matching':'nearest_unordered_candidate_optimistic' if name=='target_bore' else 'named_landmark'})
        results.append(row)
        # Both order predictions rendered over the SAME original physical frames.
        if task not in overlay_done:
            sheet=Image.new('RGB',(1280,1040),'#161616');draw=ImageDraw.Draw(sheet)
            for oi,order in enumerate(ORDERS):
                pred=row['orders'][order]['prediction']; cameras=ORDERS[order]
                for ci,camera in enumerate(('wrist_port2','wrist_port3')):
                    offset=(ci*640,oi*520+40);sheet.paste(Image.fromarray(images[camera]),offset)
                    draw.text((offset[0]+8,offset[1]-30),f'{order} | {camera} | {o["observation_id"]}',fill='white')
                    slot=cameras.index(camera)
                    for j,name in enumerate(pred['landmark_names']):
                        if task=='gear' and name not in ('gear_center','slot_center'):continue
                        x,y=pred['keypoints_2d'][slot][j];x+=offset[0];y+=offset[1]
                        draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2);draw.text((x+5,y),name,fill='red')
                    for name,l in o['labels']['views'][camera]['landmarks'].items():
                        if l['visibility']=='visible':
                            x,y=l['xy_px'];x+=offset[0];y+=offset[1]
                            draw.line((x-7,y,x+7,y),fill='lime',width=2);draw.line((x,y-7,x,y+7),fill='lime',width=2);draw.text((x+7,y+10),'manual '+name,fill='lime')
            sheet.save(str(args.output_prefix)+f'-{task}-orders.jpg');overlay_done.add(task)
        print(json.dumps({'progress':count+1,'total':len(selected),'observation':o['observation_id']}),flush=True)
    def summary(rows):
        values=np.asarray([r['error_px'] for r in rows]);return {'n':len(values),'median_px':float(np.median(values)),'p90_px':float(np.quantile(values,.9)),'mean_px':float(np.mean(values))}
    groups=defaultdict(list)
    for r in errors:
        for keys in [('task','order'),('task','landmark','order'),('task','landmark','camera','order'),('task','split','order')]:
            groups['|'.join(f'{k}={r[k]}' for k in keys)].append(r)
    report={'experiment':'Exact original manual-label wrist JPEGs; paired order intervention; same weights/XYZ/RGB frames',
            'label_source':str(args.labels),'label_sha256':hashlib.sha256(args.labels.read_bytes()).hexdigest(),
            'checkpoints':{t:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for t,p in checkpoints.items()},
            'orders':ORDERS,'torch_version':torch.__version__,'device':str(models['gear'].device),
            'cpu_threads':2,'observations':len(results),
            'labeled_coordinates':len(errors)//2,'groups':{k:summary(v) for k,v in groups.items()},
            'caveats':['AI manual visual labels, not human certified','Bore matching is optimistic nearest unordered head, not target identity accuracy','Coordinate errors are evaluated without visibility gates','No recordings, model weights, runtime configuration, or robot state changed']}
    Path(str(args.output_prefix)+'.summary.json').write_text(json.dumps(report,indent=2)+'\n')
    Path(str(args.output_prefix)+'.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    with Path(str(args.output_prefix)+'.errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    print(json.dumps(report['groups'],indent=2))

if __name__=='__main__':main()
