#!/usr/bin/env python3
"""Frozen-checkpoint state-goodness evaluation. Use test once after validation selection."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from state_goodness_train import Goodness,load_frames,evaluate,shortcut_baselines


def read_evaluation_pairs(path,episodes,split):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for r in rows:
        if r.get('split')!=split:raise ValueError('evaluation pair split mismatch')
        for side in ('a','b'):
            ep=str(r['episode_id_'+side])
            if ep not in episodes or episodes[ep][0]['split']!=split:raise ValueError('evaluation pair episode/split mismatch')
            refs=r['observation_'+side]['current']
            if isinstance(refs,dict):r['observation_'+side]['current']=[refs[k] for k in ('overhead','wrist_port2','wrist_port3')]
        if r.get('preference')!='unknown' and not 0<float(r.get('confidence',1.))<=1:raise ValueError('evaluation preference confidence must be in (0,1]')
        if r['preference'] not in ('a_better','b_better','equal','unknown'):raise ValueError('invalid evaluation preference')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--dataset-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',choices=('val','test'),default='val');ap.add_argument('--preference-manifest',type=Path);ap.add_argument('--preference-image-root',type=Path,default=Path('.'));ap.add_argument('--device',choices=('cpu','cuda'),default='cuda');ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args();device=torch.device(args.device);ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];torch.set_num_threads(4)
    _,episodes=load_frames(args.manifest,cfg['clean_kept_confirmed']);pairs=read_evaluation_pairs(args.preference_manifest,episodes,args.split) if args.preference_manifest else []
    model=Goodness(cfg['backbone'],False,cfg['freeze_encoder'],cfg.get('camera_mode','all')).to(device);model.load_state_dict(ck['model']);model.eval()
    evaluation_args=SimpleNamespace(**{**cfg,**vars(args),'image_size':cfg['image_size']});report=evaluate(model,episodes,pairs,evaluation_args,device,split=args.split);report['shortcut_baselines']=shortcut_baselines(episodes,evaluation_args,split=args.split);report['checkpoint']=str(args.checkpoint);report['split']=args.split;report['warning']='Frozen post-selection evaluation only; uncalibrated demonstrated-state support, not current-policy success probability.'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
