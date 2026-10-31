#!/usr/bin/env python3
"""Rebuild every Q target for a selected teacher; safely reuse identical encodings.

Preparation only unless --train is explicitly supplied. The teacher must already
have passed its independent deployment gate; this utility does not select it.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from action_critic_train import (read_transitions, load_teacher, choose_device,
    image_batch, future_value_target, atomic_save, train)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(row, field, root):
    return tuple(str((Path(row.get('image_root', root)) / p).resolve()) for p in row[field])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--old-critic',type=Path,default=Path('work/action-critic-mixed-20261029/run/best.pt'))
    p.add_argument('--old-cache',type=Path,default=Path('work/action-critic-mixed-20261029/features.pt'))
    p.add_argument('--manifest',type=Path,default=Path('work/action-critic-mixed-20261029/dataset/transitions.jsonl'))
    p.add_argument('--teacher-manifest',type=Path,action='append',required=True,help='all training/validation teacher manifests, repeatable')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('.'))
    p.add_argument('--device',default='auto')
    p.add_argument('--train',action='store_true')
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('output must be a fresh versioned directory')
    rows,_,_=read_transitions(args.manifest)
    original_oct28_test={str(r['session_id']) for r in rows if r['split']=='test' and str(r['session_id']).startswith('20261028')}
    teacher_sessions={}
    for path in args.teacher_manifest:
        for line in path.read_text().splitlines():
            r=json.loads(line)
            session=str(r.get('session_id',str(r['episode_id'])[:8]))
            split=r['split']
            if session in teacher_sessions and teacher_sessions[session]!=split:
                raise ValueError(f'teacher session split conflict: {session}')
            teacher_sessions[session]=split
    # Conservative session-level alignment removes teacher-trained sessions from
    # Q validation/test, and teacher-development sessions from Q train/test.
    moved=[]
    for r in rows:
        split=teacher_sessions.get(str(r['session_id']))
        if split is not None and split != r['split']:
            moved.append({'session_id':r['session_id'],'old':r['split'],'new':split})
            r['split']=split
            if 'future_split' in r: r['future_split']=split
    test_sessions={str(r['session_id']) for r in rows if r['split']=='test'}
    if not original_oct28_test or not original_oct28_test.issubset(test_sessions):
        raise ValueError('Oct28 normal heldout session must remain test')
    args.output.mkdir(parents=True)
    manifest=args.output/'transitions.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rows,_,_=read_transitions(manifest)
    used=[r for r in rows if r['split'] in ('train','val')]
    old_rows,_,_=read_transitions(args.manifest)
    old_used=[r for r in old_rows if r['split'] in ('train','val')]
    bundle=torch.load(args.teacher,map_location='cpu',weights_only=False)
    old=torch.load(args.old_critic,map_location='cpu',weights_only=False)
    encoder_keys=[k for k in bundle['model'] if k.startswith(('backbone.','fuse.'))]
    identical=bool(encoder_keys) and all(k in old['teacher']['model'] and torch.equal(bundle['model'][k],old['teacher']['model'][k]) for k in encoder_keys)
    size=int(bundle['config'].get('image_size',224))
    identical &= size==old['config']['image_size']
    cache=torch.load(args.old_cache,map_location='cpu',weights_only=False)
    expected=cache['identity']
    safe=(identical and expected['manifest_sha256']==sha(args.manifest) and expected['teacher_sha256']==old['teacher_sha256'] and expected['image_size']==size and cache['z'].shape==(len(old_used),256) and torch.isfinite(cache['z']).all().item())
    features={}
    if safe:
        for r,z in zip(old_used,cache['z']):
            key=observation(r,'images',args.dataset_root)
            if key in features and not torch.allclose(features[key],z,atol=1e-5,rtol=1e-5):
                raise ValueError('duplicate cached observation has inconsistent features')
            features[key]=z
    required={observation(r,f,args.dataset_root) for r in used for f in ('images','future_images')}
    missing=sorted(required-features.keys())
    device=choose_device(args.device)
    torch.set_num_threads(4)
    teacher=load_teacher(bundle,device)
    print(json.dumps({'encoder_identical':identical,'reuse_safe':safe,'required_observations':len(required),'missing_to_encode':len(missing),'splits':dict(Counter(r['split'] for r in rows))}),flush=True)
    with torch.inference_mode():
        for offset in range(0,len(missing),32):
            keys=missing[offset:offset+32]
            images=torch.cat([image_batch(k,'.',size,'cpu') for k in keys]).to(device)
            for key,z in zip(keys,teacher.encode(images).cpu()): features[key]=z
        z=torch.stack([features[observation(r,'images',args.dataset_root)] for r in used])
        targets=[]
        for offset in range(0,len(used),512):
            batch=used[offset:offset+512]
            future=torch.stack([features[observation(r,'future_images',args.dataset_root)] for r in batch]).to(device)
            targets.append(future_value_target(teacher,future,[r['dt'] for r in batch],1.).cpu())
    identity={'format':'action_critic_features_v1','manifest_sha256':sha(manifest),'teacher_sha256':sha(args.teacher),'dataset_root':str(args.dataset_root.resolve()),'image_roots':sorted({str(Path(r.get('image_root',args.dataset_root)).resolve()) for r in used}),'image_size':size,'gamma':1.,'preprocessing':'resize_imagenet_rgb_v1','splits':['train','val']}
    feature_path=args.output/'features.pt'
    atomic_save({'identity':identity,'z':z,'target':torch.cat(targets)},feature_path)
    report={'teacher':str(args.teacher.resolve()),'teacher_sha256':sha(args.teacher),'encoder_identical':identical,'cache_reused':safe,'encoded_missing':len(missing),'all_targets_recomputed':True,'split_changes':[json.loads(x) for x in sorted({json.dumps(x,sort_keys=True) for x in moved})],'test_sessions':sorted(test_sessions),'teacher_manifests':[str(x) for x in args.teacher_manifest]}
    (args.output/'retarget-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.train:
        train(SimpleNamespace(manifest=manifest,teacher_checkpoint=args.teacher,dataset_root=args.dataset_root,output=args.output/'run',gamma=1.,feature_cache=feature_path,encode_batch_size=32,encode_progress_every=500,epochs=30,batch_size=128,hidden_dim=256,lr=.0003,cpu_threads=4,seed=19,device=args.device))

if __name__=='__main__': main()
