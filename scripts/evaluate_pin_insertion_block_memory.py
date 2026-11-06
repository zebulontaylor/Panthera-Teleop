#!/usr/bin/env python3
"""Score a predeclared diagnostic CNN cache only AFTER frozen model selection.

This does not grant trusted visibility or runtime eligibility. It evaluates an
unverified first-five-frame cache, including the possibility of a bad seed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.adapt_pin_insertion_cnn import sha256
from scripts.evaluate_pin_insertion_adaptation import distribution


def evaluate(args):
    manifest=json.loads(Path(args.manifest).read_text());protocol=json.loads(Path(args.protocol).read_text());video=json.loads(Path(args.predictions).read_text())
    if protocol.get('manifest_sha256')!=sha256(args.manifest) or video.get('manifest_sha256')!=sha256(args.manifest):raise ValueError('memory manifest binding mismatch')
    if video.get('memory_protocol_sha256')!=sha256(args.protocol) or video.get('checkpoint_sha256')!=protocol.get('selected_checkpoint_sha256'):raise ValueError('memory protocol/checkpoint binding mismatch')
    if protocol.get('autonomous_memory_eligible') is not False or protocol.get('clear_visibility_verified') is not False:raise ValueError('this evaluator is only for explicitly unverified diagnostic cache')
    if any(protocol.get(key) is not False for key in ('future_insertion_reference_or_GT_seed','test_predictions_or_scores_seen_before_declaration')):
        raise ValueError('memory protocol must explicitly exclude GT/future/test-based seed selection')
    if video.get('test_labels_used') is not False or video.get('future_references_used') is not False:
        raise ValueError('video predictions must explicitly exclude labels and future references')
    frames={(r['block_id'],r['source_frame_index']):r for r in video['frames']};records=[]
    for block,seeds in protocol['seed_frame_indices'].items():
        median=np.median([frames[(block,i)]['current_keypoints_normalized'][2:] for i in seeds],axis=0)
        for prediction in video['frames']:
            if prediction['block_id']==block and prediction['source_frame_index']>max(seeds) and prediction.get('cached_hole_normalized') is not None:
                if not np.allclose(prediction['cached_hole_normalized'],median,rtol=0,atol=1e-7):raise ValueError('cache differs from fixed causal first-frame CNN median')
    for row in manifest['frames']:
        if row['split']!='test' or not row['valid'][1]:continue
        block=row['block_id'];index=row['source_frame_index'];seeds=protocol['seed_frame_indices'][block]
        if index<=max(seeds):continue
        prediction=frames[(block,index)];cached=prediction.get('cached_hole_normalized')
        if cached is None:continue
        size=np.array([row['image_width'],row['image_height']]);truth=np.asarray(row['keypoints_px'][2:],float)
        current=np.asarray(prediction['current_keypoints_normalized'][2:])*size;memory=np.asarray(cached)*size
        source=frames[(block,max(seeds))]['source_image_monotonic_ns']
        records.append({'block_id':block,'source_frame_index':index,
            'current_hole_error_px':float(np.linalg.norm(current-truth)),'cached_hole_error_px':float(np.linalg.norm(memory-truth)),
            'annotation_uncertainty_px':row['uncertainty_px'][1],'cache_age_raw_s':(prediction['source_image_monotonic_ns']-source)/1e9,
            'cache_source':'fixed first-five actual CNN predictions; unverified diagnostic only'})
    def stats(selected):
        return {'label_count':len(selected),'current_hole_error_px':distribution([r['current_hole_error_px'] for r in selected]),
            'cached_hole_error_px':distribution([r['cached_hole_error_px'] for r in selected]),
            'fraction_cache_improved':float(np.mean([r['cached_hole_error_px']<r['current_hole_error_px'] for r in selected])) if selected else None}
    report={'manifest_sha256':sha256(args.manifest),'protocol_sha256':sha256(args.protocol),'predictions_sha256':sha256(args.predictions),
        'checkpoint_sha256':video['checkpoint_sha256'],'autonomous_memory_eligible':False,'clear_visibility_verified':False,
        'test_used_for_model_or_seed_selection':False,'summary':stats(records),
        'per_block':{b:stats([r for r in records if r['block_id']==b]) for b in protocol['seed_frame_indices']},'records':records,
        'limitations':['First-five initialization was fixed before seeing selected-model test results, but actual clear visibility is unverified.',
            'Especially B5 may start inserted/occluded. A wrong cache is expected to persist; no memory trust/gate/motor eligibility is implied.',
            'Later insertion-reference coordinates are annotation-only scoring labels and never seed the cache.',
            'Held-out blocks share one recording/grasp; error and stated annotation uncertainty do not establish independent-session robustness.']}
    output=Path(args.output);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':report['summary'],'per_block':report['per_block']},indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','predictions','output'):p.add_argument('--'+name,required=True)
    evaluate(p.parse_args())

if __name__=='__main__':main()
