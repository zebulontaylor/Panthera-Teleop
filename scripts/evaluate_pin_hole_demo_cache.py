#!/usr/bin/env python3
"""Causal TRAIN-reference hole-cache evaluation; annotations enter scoring only.

Validation may choose parameters. Reused test benchmarks require a previously
frozen reference/config selection, and are never described as unseen trials.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def validate_frozen_selection(selection_path,references_path,config_path,checkpoint=None):
    if not selection_path:raise ValueError('TEST requires a frozen validation selection artifact')
    selection=json.loads(Path(selection_path).read_text())
    if selection.get('selection_split')!='validation' or selection.get('test_used_for_selection') is not False:
        raise ValueError('Demo-cache parameters must be chosen using validation only')
    if selection.get('frozen_before_this_method_test',selection.get('frozen_before_test')) is not True:
        raise ValueError('Selection must be frozen before this method test execution')
    for name,path in [('references_sha256',references_path),('config_sha256',config_path)]:
        if selection.get(name)!=sha(path):raise ValueError('Frozen selection SHA mismatch: '+name)
    if checkpoint is not None and selection.get('checkpoint_sha256')!=sha(checkpoint):
        raise ValueError('Comparator checkpoint SHA mismatch with frozen selection')
    if selection.get('implementation_sha256') is not None and selection['implementation_sha256']!=sha(ROOT/'app/pin_hole_demo_cache.py'):
        raise ValueError('Frozen cache implementation changed')
    return selection


def validate_bank(bank,manifest=None):
    references=bank.get('references',[])
    if not references:raise ValueError('TRAIN reference bank is empty')
    seen=set();contexts=set()
    split=None
    if manifest and manifest.get('split_definition_path'):
        split_path=Path(manifest['split_definition_path'])
        if sha(split_path)!=manifest['split_definition_sha256']:raise ValueError('Frozen reference split SHA mismatch')
        split=json.loads(split_path.read_text())
    for ref in references:
        if ref.get('split')!='train':raise ValueError('Only TRAIN annotations/images may enter reference bank')
        if ref['id'] in seen:raise ValueError('Duplicate reference ID')
        seen.add(ref['id'])
        if np.asarray(ref['hole_px'],dtype=float).shape!=(2,) or not np.isfinite(np.asarray(ref['hole_px'],dtype=float)).all():raise ValueError('Invalid TRAIN hole annotation')
        source=ref.get('source_recording_id',ref.get('episode_id'))
        block=ref.get('block_id')
        contexts.add((source,block))
        if split and source==split['source_recording_id']:
            found=[b for b in split['blocks'] if b['block_id']==block and b['first_frame']<=ref['source_frame_index']<=b['last_frame']]
            if len(found)!=1 or found[0]['split']!='train':raise ValueError('Reference index crosses frozen TRAIN block boundary')
        if sha(ref['image_path'])!=ref['image_sha256']:raise ValueError('Reference image SHA mismatch')
    return contexts


def guard_reference_context(context,reference_contexts,domain):
    source,block=context
    if domain=='blocks':
        if context in reference_contexts:raise ValueError('Evaluation block overlaps TRAIN reference block')
    elif any(source==r[0] for r in reference_contexts):
        raise ValueError('Independent episode overlaps TRAIN reference recording')


def block_stream(manifest,split_name):
    from scripts.render_pin_insertion_block_video import chronological_stream
    split_path=Path(manifest['split_definition_path'])
    if sha(split_path)!=manifest['split_definition_sha256']:raise ValueError('Frozen block definition changed')
    split=json.loads(split_path.read_text())
    if split.get('split_frozen_before_coordinate_annotation_or_fitting') is not True:raise ValueError('Unfrozen carrier blocks')
    source=Path(split['source_path'])
    for name,digest in split['source_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('Source recording checksum mismatch')
    blocks=[b for b in split['blocks'] if b['split']==split_name]
    if not blocks:raise ValueError('No requested evaluation blocks')
    frames=read_jsonl(source/'frames.jsonl');states=read_jsonl(source/'states.jsonl')
    stream=chronological_stream(frames,states,blocks)
    output=[]
    for sample in stream:
        frame=sample['frame'];block=sample['block'];index=frame['frame_index']
        output.append({'context_id':block['block_id'],'block_id':block['block_id'],'source_recording_id':split['source_recording_id'],
            'frame_index':index,'timestamp_ns':frame['monotonic_ns'],'image_path':str((source/frame['path']).resolve()),
            'state14':sample['state14'],'state_record_receipt_skew_ms':sample['state_skew_ms'],'source_time_s':frame['episode_time_s'],
            'recording_segment':frame.get('recording_segment')})
    return output,{'split_unit':split['split_unit'],'independent_recordings':1,'session_disjoint':False,
                   'scope':'Carrier-pose-block holdout within one recording; context reset independently for each block',
                   'split_definition_sha256':sha(split_path)}


def episode_stream(manifest,split_name):
    from scripts.evaluate_pin_insertion_cnn import eligible_samples
    rows=[r for r in manifest['frames'] if r['split']==split_name]
    by_episode={r['episode_id']:r for r in rows};output=[]
    for eid,row in sorted(by_episode.items()):
        ep=Path(row.get('source_episode_path',ROOT/'data/pin-alignment-forward'/eid))
        meta,samples,_=eligible_samples(ep,stride=1,max_skew_ms=100)
        for sample,skew in samples:
            output.append({'context_id':eid,'episode_id':eid,'source_recording_id':eid,'frame_index':sample['video_frame'],
              'timestamp_ns':sample['monotonic_ns'],'state14':sample['state'],'video_path':str((ep/meta['videos']['right_wrist']).resolve()),
              'video_frame':sample['video_frame'],'camera_sample_skew_ms':skew,
              'source_time_s':(sample['monotonic_ns']-meta['started_monotonic_ns'])/1e9})
    if not output:raise ValueError('No independent evaluation episodes')
    return output,{'split_unit':'whole_saved_episode','independent_recordings':len(by_episode),'session_disjoint':None,
                   'scope':'Different source recordings from reference bank; grasp/session independence not inferred'}


def score_predictions(records,manifest,split_name,domain):
    # This function is called only AFTER all causal predictions have been saved
    # in memory. No coordinate/visibility/future-reference annotation is passed
    # to update(), reference selection, tracking or causal hold.
    index={(r['context_id'],r['frame_index']):r for r in records}
    errors=[]
    for label in manifest['frames']:
        if label['split']!=split_name:continue
        context=label['block_id'] if domain=='blocks' else label['episode_id']
        frame=label['source_frame_index'] if domain=='blocks' else label['video_frame']
        row=index.get((context,frame))
        if row is None:raise ValueError('Annotated frame absent from chronological stream')
        valid=label.get('valid',[False,False])[1]
        truth=label.get('target_hole_px')
        if not valid or truth is None:continue
        item={'context_id':context,'frame_index':frame,'label_uncertainty_px':label.get('uncertainty_px',[None,None])[1],
          'target_hole_inferential':label.get('hole_center_inferred',label.get('target_hole_inferential',label.get('target_hole_is_amodal'))),
          'hole_annotation_method':label.get('hole_center_method'),'hole_visibility':label.get('hole_visibility'),
          'annotation_pin_position_available':bool(label.get('valid',[False,False])[0]),'annotation_note':label.get('note'),
          'demo_cache_error_px':None,'current_cnn_error_px':None,'first5_cnn_cache_error_px':None,'demo_cache_mode':row['demo_cache'].get('mode')}
        for output,key in [(row['demo_cache'].get('hole_px'),'demo_cache_error_px'),(row.get('current_cnn_hole_px'),'current_cnn_error_px'),(row.get('first5_cnn_cache_hole_px'),'first5_cnn_cache_error_px')]:
            if output is not None:item[key]=float(np.linalg.norm(np.asarray(output,dtype=float)-np.asarray(truth,dtype=float)))
        errors.append(item)
    summaries={}
    for key in ('demo_cache_error_px','current_cnn_error_px','first5_cnn_cache_error_px'):
        values=[r[key] for r in errors if r[key] is not None]
        summaries[key]={'available_labels':len(values),'known_hole_labels':len(errors),'coverage':len(values)/len(errors) if errors else None,
                        'mean_px':float(np.mean(values)) if values else None,'median_px':float(np.median(values)) if values else None,
                        'p90_px':float(np.percentile(values,90)) if values else None,
                        'mean_with_unavailable_40px_penalty':float(np.mean([r[key] if r[key] is not None else 40. for r in errors])) if errors else None}
    return {'rows':errors,'summary':summaries,
       'label_strata':{name:{'labels':len(group),'available_demo_cache':sum(r['demo_cache_error_px'] is not None for r in group),
         'demo_cache_mean_px':float(np.mean(values)) if values else None}
         for name in ('direct','temporal_or_amodal')
         if (group := [r for r in errors if ('temporal_or_amodal' if r['target_hole_inferential'] else 'direct')==name])
         and (values := [r['demo_cache_error_px'] for r in group if r['demo_cache_error_px'] is not None])},
       'unavailable_predictions_not_counted_as_success':True}


def load_prediction_lookup(path,checkpoint,manifest_path):
    if not path:return {}
    report=json.loads(Path(path).read_text())
    if checkpoint is None or report.get('checkpoint_sha256')!=sha(checkpoint):raise ValueError('Prior CNN report checkpoint mismatch')
    manifest=json.loads(Path(manifest_path).read_text())
    permitted={sha(manifest_path)}
    for key in ('original_manifest_sha256','prediction_source_manifest_sha256'):
        if manifest.get(key):permitted.add(manifest[key])
    if report.get('manifest_sha256') not in permitted:raise ValueError('Prior CNN report manifest mismatch')
    lookup={}
    for r in report.get('frames',[]):
        context=r.get('block_id',r.get('episode_id'));frame=r.get('source_frame_index',r.get('video_frame'))
        if r.get('current_keypoints_normalized') is not None:lookup[(context,frame)]=r
    return lookup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--cnn-predictions',type=Path)
    p.add_argument('--domain',choices=['blocks','episodes'],default='blocks');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    selection=validate_frozen_selection(args.selection,args.references,args.config,args.checkpoint) if args.split=='test' else None
    # No evaluation images, cached test outputs, or reference images are opened
    # until the test selection guard above passes.
    manifest=json.loads(args.manifest.read_text());bank=json.loads(args.references.read_text());config=json.loads(args.config.read_text())
    bank_manifest=manifest
    if args.domain=='episodes' and bank.get('manifest_path'):
        bank_manifest=json.loads(Path(bank['manifest_path']).read_text())
    reference_contexts=validate_bank(bank,bank_manifest)
    samples,audit=block_stream(manifest,args.split) if args.domain=='blocks' else episode_stream(manifest,args.split)
    for row in samples:guard_reference_context((row['source_recording_id'],row.get('block_id')),reference_contexts,args.domain)
    from app.pin_hole_demo_cache import DemoHoleCache
    import cv2
    cv2.setNumThreads(4)
    cache=DemoHoleCache(args.references,**config)
    previous_context=None;previous_timestamp=None;seed_holes=[];frozen_cnn_hole=None;records=[];captures={};predictor=None
    lookup=load_prediction_lookup(args.cnn_predictions,args.checkpoint,args.manifest)
    try:
        for row in samples:
            context=row['context_id'];stamp=int(row['timestamp_ns'])
            if context!=previous_context:
                cache.reset();seed_holes=[];frozen_cnn_hole=None;previous_timestamp=None
            if previous_timestamp is not None and stamp<=previous_timestamp:raise ValueError('Chronology regressed or repeated')
            if 'image_path' in row:
                bgr=cv2.imread(row['image_path'])
            else:
                path=row['video_path']
                if path not in captures:captures[path]=cv2.VideoCapture(path)
                cap=captures[path];cap.set(cv2.CAP_PROP_POS_FRAMES,row['video_frame']);ok,bgr=cap.read()
                if not ok:bgr=None
            if bgr is None:raise ValueError('Cannot decode real source frame')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);wh=np.array([rgb.shape[1],rgb.shape[0]])
            # Core estimator receives ONLY the present RGB, causal timestamp and
            # opaque reset context. It receives no q, CNN prediction or labels.
            result=cache.update(rgb,timestamp_ns=stamp,now_ns=stamp,sequence_id=context,camera_id='right_wrist',grasp_id=context)
            current=None;prior=lookup.get((context,row['frame_index']))
            if prior is not None:
                prior_stamp=prior.get('source_image_monotonic_ns')
                if prior_stamp is not None and int(prior_stamp)!=stamp:raise ValueError('Prior CNN source image receipt timestamp mismatch')
                current=np.asarray(prior['current_keypoints_normalized'],dtype=float).reshape(2,2)*wh
            elif args.checkpoint:
                if predictor is None:
                    import torch
                    from app.pin_insertion_cnn import PinInsertionPredictor
                    torch.set_num_threads(4);predictor=PinInsertionPredictor(args.checkpoint)
                current=np.asarray(predictor.predict(rgb,row['state14'])['keypoints'],dtype=float).reshape(2,2)*wh
            if current is not None and len(seed_holes)<5:
                seed_holes.append(current[1].copy())
                if len(seed_holes)==5:frozen_cnn_hole=np.median(np.stack(seed_holes),axis=0).tolist()
            records.append({**row,'state14':row['state14'],'image_size_wh':wh.tolist(),'image_sha256':sha(row['image_path']) if 'image_path' in row else None,'demo_cache':result,
                            'current_cnn_pin_px':current[0].tolist() if current is not None else None,
                            'current_cnn_hole_px':current[1].tolist() if current is not None else None,
                            'first5_cnn_cache_hole_px':frozen_cnn_hole,'cnn_prediction_reused':prior is not None,
                            'reset_before_frame':context!=previous_context,'raw_frame_gap_ms':(stamp-previous_timestamp)/1e6 if previous_timestamp else None})
            previous_context=context;previous_timestamp=stamp
    finally:
        for cap in captures.values():cap.release()
    scores=score_predictions(records,manifest,args.split,args.domain)
    modes={name:sum(r['demo_cache'].get('mode')==name for r in records) for name in ('tracked','held','lost')}
    report={'kind':'train_demonstration_reference_hole_cache','evaluation_split':args.split,'domain':args.domain,'split_audit':audit,
      'implementation_sha256':sha(ROOT/'app/pin_hole_demo_cache.py'),'evaluator_sha256':sha(__file__),
      'references_path':str(args.references.resolve()),'references_sha256':sha(args.references),'config_path':str(args.config.resolve()),'config_sha256':sha(args.config),
      'validation_plan_sha256':sha(ROOT/'work/pin-hole-demo-cache/validation-plan.json') if (ROOT/'work/pin-hole-demo-cache/validation-plan.json').exists() else None,
      'manifest_path':str(args.manifest.resolve()),'manifest_sha256':sha(args.manifest),'checkpoint_path':str(args.checkpoint.resolve()) if args.checkpoint else None,
      'checkpoint_sha256':sha(args.checkpoint) if args.checkpoint else None,'selection':selection,'records':records,'evaluation':scores,'mode_counts':modes,
      'per_context':{context:{'frames':len(group),'cache_available':sum(r['demo_cache'].get('hole_px') is not None for r in group),
        'label_errors':[r for r in scores['rows'] if r['context_id']==context]} for context in sorted({r['context_id'] for r in records}) if (group := [r for r in records if r['context_id']==context])},
      'all_frame_quality':{key:{'count':len(values),'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None}
         for key in ('inliers','matches','reprojection_rmse_px','coverage') if (values := [float(r['demo_cache'][key]) for r in records if isinstance(r['demo_cache'].get(key),(int,float,np.number)) and np.isfinite(r['demo_cache'][key])])},
      'causal_cache_coverage':sum(r['demo_cache'].get('hole_px') is not None for r in records)/len(records),
      'reused_test_benchmark':args.split=='test','prior_results_seen_context':True,'test_used_for_parameter_selection':False,
      'gt_or_future_reference_used_by_prediction':False,'reset_policy':'Each evaluation block/episode starts independently from the same TRAIN-only bank; no cross-test-block cache.',
      'limitations':['One purposeful recording shares its grasp across frozen blocks; this is not independent-session validation.',
        'Demo reference hole annotations are TRAIN supervision; all evaluation annotation coordinates/visibility/future references are read only after causal inference for scoring.',
        'Old CNN first-five cache is a diagnostic baseline without an automated clear-view guarantee; no GT seed coordinates.',
        'Tracked/held hole and heuristic image-match quality are not calibrated metric/contact uncertainty or insertion success.',
        'B4/B5 and older episodes were evaluated previously with CNN methods; this method freezes parameters on VAL before its own reused benchmark evaluation.']}
    def serial(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False,default=serial)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','evaluation','selection')},indent=2));print(json.dumps(scores['summary'],indent=2))

if __name__=='__main__':main()
