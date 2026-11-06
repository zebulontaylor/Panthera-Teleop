#!/usr/bin/env python3
"""Join independent block annotations to original saved images, clocks and FK.

Insertion references remain label provenance only. They are never inference
inputs or future observations supplied to a temporal cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(split_path, annotation_paths, output):
    split = json.loads(split_path.read_text())
    source = Path(split['source_path'])
    eid = split['source_recording_id']
    for name, expected in split['source_hashes'].items():
        if sha(source/name) != expected:
            raise ValueError('Frozen source recording changed: '+name)
    blocks = {b['block_id']: b for b in split['blocks']}
    frames = {r['frame_index']: r for r in map(json.loads, (source/'frames.jsonl').read_text().splitlines()) if r['camera']=='wrist_port2'}
    states = list(map(json.loads, (source/'states.jsonl').read_text().splitlines()))
    stamps = np.array([s['monotonic_ns'] for s in states], dtype=np.int64)
    if np.any(np.diff(stamps)<0):
        raise ValueError('Recorded state timestamps are not ordered')
    result = []; seen = set(); annotations = []
    for path in annotation_paths:
        doc = json.loads(path.read_text())
        annotations.append({'path':str(path.resolve()),'sha256':sha(path)})
        if doc.get('split_sha256') not in (None, sha(split_path)):
            raise ValueError('Annotation uses different frozen split')
        for label in doc['frames']:
            index = label['frame_index']; block_id = label['block_id']; b = blocks[block_id]
            if index in seen:
                raise ValueError('Duplicate annotated source frame')
            seen.add(index); frame = frames[index]
            if not b['start_s'] <= frame['episode_time_s'] < b['end_s'] or label['split'] != b['split']:
                raise ValueError('Annotation crosses frozen pose block/transition guard')
            image = (source/frame['path']).resolve()
            if sha(image) != label['image_sha256']:
                raise ValueError('Annotated image hash differs from saved source')
            timestamp = frame['monotonic_ns']; right = int(np.searchsorted(stamps,timestamp))
            candidates = [i for i in (right-1,right) if 0<=i<len(states)]
            nearest = min(candidates,key=lambda i:abs(int(stamps[i])-timestamp)); state = states[nearest]
            skew = (timestamp-int(stamps[nearest]))/1e6
            if abs(skew)>25:
                raise ValueError('No recorded state within 25 ms of image')
            q = []; transforms = {}
            for side in ('left','right'):
                arm = state['arms'][side]; q += arm['position_rad']+[arm['gripper']['position']]
                pose = arm['tool_pose_base']; t = np.eye(4); t[:3,:3] = pose['rotation']; t[:3,3] = pose['position_m']
                transforms['T_'+side+'_base_tool'] = t.tolist()
            r = {**label, **transforms, 'episode_id':eid+'::'+block_id,
                 'source_recording_id':eid, 'session_id':eid, 'source_frame_index':index,
                 'actual_episode_time_s':frame['episode_time_s'], 'image_path':str(image),
                 'image_width':frame['width'], 'image_height':frame['height'], 'width':frame['width'], 'height':frame['height'],
                 'state14':q, 'camera':'right_wrist','camera_source':'wrist_port2',
                 'camera_sample_skew_ms':skew,'source_state_monotonic_ns':int(stamps[nearest]),
                 'source_image_monotonic_ns':timestamp,'source_episode_path':str(source),
                 'contact_audit':True,'annotation_path':str(path.resolve())}
            r['uncertainty_px'] = [float(x or 0) for x in r['uncertainty_px']]
            result.append(r)
    summary = {}
    for name in ('train','validation','test'):
        rows = [r for r in result if r['split']==name]
        summary[name] = {'frames':len(rows),'blocks':sorted({r['block_id'] for r in rows}),
                         'pin_labels':sum(r['valid'][0] for r in rows),'hole_labels':sum(r['valid'][1] for r in rows),
                         'gate_positive':sum(r.get('gate_label')==1 for r in rows),
                         'gate_negative':sum(r.get('gate_label')==0 for r in rows),
                         'gate_unknown':sum(r.get('gate_label') is None for r in rows)}
    manifest = {'schema_version':3,'split_unit':'carrier_pose_block',
                'split_definition_path':str(split_path.resolve()),'split_definition_sha256':sha(split_path),
                'source_recording_id':eid,'independent_recordings':1,'session_disjoint':False,
                'annotation_sources':annotations,'frames':result,'summary':summary,
                'annotation_kind':'AI manual labels with insertion-reference provenance and uncertainty; user confirms recording protocol',
                'gate_definition':'Visual plus user-protocol contact labels; not force-certified contact or autonomous success',
                'insertion_references_are_annotation_only':True,
                'limitations':split['limitations'],
                'episodes':[{'episode_id':eid+'::'+b['block_id'],'source_recording_id':eid,'block_id':b['block_id'],'session_id':eid,'split':b['split']} for b in split['blocks']]}
    output.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--annotations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve() in {a.split.resolve(),*(x.resolve() for x in a.annotations)}:
        raise ValueError('Preserve original split and independent annotations')
    prepare(a.split,a.annotations,a.output)
