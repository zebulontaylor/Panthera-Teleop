#!/usr/bin/env python3
"""Explicit post-test continuous fixed-grasp replay; not an unseen benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_pin_hole_demo_cache import sha,read_jsonl,validate_bank,score_predictions


def validate_replay_protocol(protocol_path,references_path,config_path,manifest_path,checkpoint_path):
    protocol=json.loads(Path(protocol_path).read_text())
    if protocol.get('evaluation_protocol')!='continuous_demonstration_replay' or protocol.get('post_test_engineering_revision') is not True:
        raise ValueError('Explicit post-test engineering replay protocol required')
    for name in ('stationary_carrier_to_right_wrist','reset_on_recording_segment_change','fixed_grasp_assumption'):
        if protocol.get(name) is not True:raise ValueError('Replay requires explicit '+name)
    if protocol.get('independent_unseen_test') is not False or protocol.get('gt_or_future_test_reference_used_by_prediction') is not False:
        raise ValueError('Replay must disclose reused training history and reject future TEST references')
    for key,path in [('references_sha256',references_path),('config_sha256',config_path),('manifest_sha256',manifest_path),('checkpoint_sha256',checkpoint_path)]:
        if protocol.get(key)!=sha(path):raise ValueError('Replay artifact SHA mismatch: '+key)
    if protocol.get('implementation_sha256')!=sha(ROOT/'app/pin_hole_demo_cache.py'):raise ValueError('Frozen feature matcher changed')
    selected_path=Path(protocol['selected_config_path'])
    if sha(selected_path)!=protocol['selected_config_sha256']:raise ValueError('Original selected matcher config changed')
    selected=json.loads(selected_path.read_text());current=json.loads(Path(config_path).read_text())
    if {k:v for k,v in selected.items() if k!='max_hold_ns'}!={k:v for k,v in current.items() if k!='max_hold_ns'}:
        raise ValueError('Engineering replay may change only stationary hold horizon')
    if current.get('max_hold_ns')!=30_000_000_000:raise ValueError('Declared replay hold horizon must be30seconds')
    return protocol,current


def causal_context(frame,recording_id):
    """Carrier pose-block/world motion deliberately does not change this context."""
    segment=frame.get('recording_segment','unsegmented')
    return (f'{recording_id}/segment:{segment}',frame['camera'],f'{recording_id}/declared_same_carrier_grasp')


def update_inputs(frame,recording_id):
    sequence,camera,grasp=causal_context(frame,recording_id)
    stamp=int(frame['monotonic_ns'])
    return {'timestamp_ns':stamp,'now_ns':stamp,'sequence_id':sequence,'camera_id':camera,'grasp_id':grasp}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cnn-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    protocol,config=validate_replay_protocol(args.protocol,args.references,args.config,args.manifest,args.checkpoint)
    manifest=json.loads(args.manifest.read_text());bank=json.loads(args.references.read_text());validate_bank(bank,manifest)
    split_path=Path(manifest['split_definition_path'])
    if sha(split_path)!=manifest['split_definition_sha256']:raise ValueError('Frozen source split changed')
    split=json.loads(split_path.read_text());source=Path(split['source_path']);recording=split['source_recording_id']
    for name,digest in split['source_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('Source recording changed')
    # Prior report supplies ONLY frozen present-frame CNN and diagnostic first5
    # outputs for display. The feature cache never receives these predictions.
    prior=json.loads(args.cnn_report.read_text())
    if prior['checkpoint_sha256']!=sha(args.checkpoint) or prior['manifest_sha256']!=sha(args.manifest):raise ValueError('Frozen CNN comparison source mismatch')
    prior_lookup={(r['context_id'],r['frame_index']):r for r in prior['records']}
    frames=[r for r in read_jsonl(source/'frames.jsonl') if r['camera']=='wrist_port2']
    frames.sort(key=lambda r:r['frame_index'])
    if [r['frame_index'] for r in frames]!=list(range(frames[0]['frame_index'],frames[-1]['frame_index']+1)):raise ValueError('Missing native source frames')
    if any(b['monotonic_ns']<=a['monotonic_ns'] for a,b in zip(frames,frames[1:])):raise ValueError('Source timestamps regressed')
    from app.pin_hole_demo_cache import DemoHoleCache
    import cv2
    cv2.setNumThreads(4);cache=DemoHoleCache(args.references,**config)
    previous_context=None;previous_stamp=None;history=[];records=[];resets=[]
    for frame in frames:
        context=causal_context(frame,recording);reset=context!=previous_context
        if reset:
            cache.reset();resets.append({'frame_index':frame['frame_index'],'reason':'initialization' if previous_context is None else 'actual recording segment/camera/grasp context change','context':list(context)})
        image_path=source/frame['path'];image=cv2.imread(str(image_path))
        if image is None:raise ValueError('Cannot decode actual source frame')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result=cache.update(rgb,**update_inputs(frame,recording))
        matching=[b for b in split['blocks'] if b['first_frame']<=frame['frame_index']<=b['last_frame']]
        block=matching[0] if matching else None
        history.append({'frame_index':frame['frame_index'],'timestamp_ns':frame['monotonic_ns'],'recording_segment':frame.get('recording_segment'),
          'block_id':block['block_id'] if block else None,'block_split':block['split'] if block else 'guard_interval',
          'cache_context':list(context),'context_reset':reset,'input_image_sha256':sha(image_path),'demo_cache':result})
        if block and block['split']=='test':
            old=prior_lookup.get((block['block_id'],frame['frame_index']))
            if old is None or int(old['timestamp_ns'])!=int(frame['monotonic_ns']):raise ValueError('Missing exact frozen CNN comparison frame')
            records.append({**old,'demo_cache':result,'context_id':block['block_id'],'block_id':block['block_id'],
                'image_path':str(image_path.resolve()),'image_sha256':history[-1]['input_image_sha256'],
                'cache_context':list(context),'reset_before_frame':reset,'recording_segment':frame.get('recording_segment'),
                'raw_frame_gap_ms':(frame['monotonic_ns']-previous_stamp)/1e6 if previous_stamp else None})
        previous_context=context;previous_stamp=frame['monotonic_ns']
    expected={r['source_frame_index'] for r in manifest['frames'] if r['split']=='test'}
    if not expected<={r['frame_index'] for r in records}:raise ValueError('Missing scored TEST source frames')
    scores=score_predictions(records,manifest,'test','blocks')
    def modes(rows):return {name:sum(r['demo_cache']['mode']==name for r in rows) for name in ('tracked','held','lost')}
    report={'kind':'continuous_demonstration_replay','evaluation_protocol':'continuous_demonstration_replay',
       'post_test_engineering_revision':True,'independent_unseen_test':False,'evaluation_split':'test','domain':'blocks',
       'reused_test_benchmark':True,'prior_results_seen_context':True,'gt_or_future_reference_used_by_prediction':False,
       'references_path':str(args.references.resolve()),'references_sha256':sha(args.references),'config_sha256':sha(args.config),
       'manifest_sha256':sha(args.manifest),'checkpoint_sha256':sha(args.checkpoint),'implementation_sha256':sha(ROOT/'app/pin_hole_demo_cache.py'),
       'protocol_path':str(args.protocol.resolve()),'protocol_sha256':sha(args.protocol),'replay_protocol':protocol,
       'strict_comparison_report_sha256':sha(args.cnn_report),'processed_source_frames':len(history),'rendered_test_pose_frames':len(records),
       'history_mode_counts':modes(history),'mode_counts':modes(records),'causal_cache_coverage':sum(r['demo_cache']['hole_px'] is not None for r in records)/len(records),
       'context_resets':resets,'records':records,'causal_history_records':history,'evaluation':scores,
       'split_audit':{'independent_recordings':1,'training_reference_recording_reused':True,'session_disjoint':False,'context_persists_across_pose_blocks':True},
       'limitations':['Post-test engineering revision: same validation-selected feature matcher, hold horizon changed to30seconds AFTER cold-start benchmark dropout was observed.',
           'All1482nativeimages are consumed causally. Earlier TRAIN/VAL/guard frames can initialize/track memory; no future TEST image or evaluation coordinate seeds it.',
           'Offline TRAIN bank contains reference examples from the same recording; it is not an unseen demonstration test. Both TRAIN blocks precede scored B4/B5.',
           'Fixed carrier-to-right-wrist grasp is an explicit user assumption; stationary holds are remembered locations, not current detections.',
           'Actual source segment/camera/grasp changes reset memory. Right-arm world movement and pose-block boundaries do not.',
           'Original independent-start reports remain unchanged; no claim of calibrated contact/depth or hardware insertion success.']}
    def serial(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False,default=serial)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','causal_history_records','replay_protocol','evaluation')},indent=2));print(json.dumps(scores['summary'],indent=2))

if __name__=='__main__':main()
