#!/usr/bin/env python3
"""Assemble the frozen-selection model audit without fitting or changing models."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.adapt_pin_insertion_cnn import sha256
from scripts.evaluate_pin_insertion_adaptation import validate_test_row
from scripts.evaluate_pin_insertion_frozen_challenge import image_plane_directions


def compact_landmarks(values):
    return {name:{'labels':value['valid_labels'],'mean_px':value['error_native_px']['mean'],
        'median_px':value['error_native_px']['median'],'p90_px':value['error_native_px']['p90'],
        'mean_stated_annotation_uncertainty_px':value['annotation_uncertainty_px']['mean']} for name,value in values.items()}


def compact_mode(mode):
    return {'landmarks':compact_landmarks(mode['landmarks']),'gate':mode['gate']}


def domain(report):
    result={name:{'normal':compact_mode(modes['normal']),
        'image_ablations':{mode:compact_landmarks(modes[mode]['landmarks']) for mode in ('black_images','shuffled_images')}} for name,modes in report['results'].items()}
    for name,modes in report['results'].items():
        if 'strata' in modes['normal']:result[name]['strata']={k:compact_mode(v) for k,v in modes['normal']['strata'].items()}
        if 'per_block' in modes['normal']:result[name]['per_block']={k:compact_mode(v) for k,v in modes['normal']['per_block'].items()}
    return result


def add_directions(result,report,manifest):
    rows={row['id']:row for row in manifest['frames']}
    for name in report['results']:
        predictions=report['predictions'][name]['normal'];selected=[rows[r['id']] for r in predictions]
        prepared=[validate_test_row(row) for row in selected]
        pred=np.asarray([row['keypoints_normalized'] for row in predictions]).reshape(-1,2,2)
        truth=np.stack([r[0] for r in prepared]);valid=np.stack([r[1] for r in prepared]);wh=np.stack([r[2] for r in prepared]);uncertainty=np.stack([r[3] for r in prepared])
        direction=image_plane_directions(pred,truth,valid,wh,uncertainty)
        result[name]['image_plane_direction']={k:v for k,v in direction.items() if k!='records'}


def summarize(directory,output):
    directory=Path(directory)
    def load(name):return json.loads((directory/name).read_text())
    selection=load('selection.json');manifest=load('manifest.json');test=load('test-evaluation.json');old=load('old-independent-evaluation.json')
    challenge=load('old-challenge-evaluation.json');memory=load('memory-evaluation.json');video=load('heldout-blocks-cnn-memory-diagnostic.json')
    if selection['selected_checkpoint_sha256']!=sha256(selection['selected_checkpoint']) or selection['manifest_sha256']!=sha256(directory/'manifest.json'):raise ValueError('frozen selection artifact changed')
    validation=load('proprio-validation.json');train=load('proprio.metrics.json')
    new_domain=domain(test);old_domain=domain(old)
    add_directions(new_domain,test,manifest)
    old_manifest=json.loads((ROOT/'work/pin-insertion-real/contact-manifest.json').read_text());add_directions(old_domain,old,old_manifest)
    gate=next(v['normal']['gate'] for k,v in new_domain.items() if k=='candidate')
    candidate_challenge=challenge['results']['candidate'];baseline_challenge=challenge['results']['baseline']
    report={'schema_version':1,'selected_checkpoint':selection['selected_checkpoint'],'selected_checkpoint_sha256':selection['selected_checkpoint_sha256'],
        'baseline_checkpoint':test['baseline_checkpoint'],'baseline_checkpoint_sha256':test['baseline_checkpoint_sha256'],
        'manifest_sha256':selection['manifest_sha256'],'selection_frozen_before_test':True,'selection_split':'B2 validation only',
        'selection_metric':selection['criterion'],'selection_candidates':selection['candidates'],
        'training':{'real_only':True,'synthetic_replay':False,'frames':sum(r['split']=='train' for r in manifest['frames']),
            'valid_pin_labels':sum(r['split']=='train' and r['valid'][0] for r in manifest['frames']),
            'valid_hole_labels':sum(r['split']=='train' and r['valid'][1] for r in manifest['frames']),
            'epochs':len(train['history']),'selected_epoch':train['metadata']['adaptation']['best_validation']['epoch'],
            'gate_exposure_per_epoch':train['history'][-1]['real_gate_exposure']},
        'within_recording_split':{'recordings':1,'session_disjoint':False,'train':['B1','B3'],'validation':['B2'],'test':['B4','B5'],
            'validation_pin_labels':1,'validation_hole_labels':8,'weak_selection':True},
        'validation':domain(validation),'new_same_recording_test':new_domain,
        'new_test_constant_train_mean':compact_landmarks(test['train_mean_baseline']['landmarks']),
        'old_independent_recording_test':old_domain,'old_independent_source_audit':old['independent_domain_audit'],
        'oct27_failed_challenge':{'recordings':1,'correlated_frames':3,
            'baseline':{'landmarks':compact_landmarks(baseline_challenge['landmarks']),'image_plane_direction':baseline_challenge['image_plane_direction']},
            'candidate':{'landmarks':compact_landmarks(candidate_challenge['landmarks']),'image_plane_direction':candidate_challenge['image_plane_direction']}},
        'causal_unverified_hole_cache':{'protocol_sha256':memory['protocol_sha256'],'autonomous_memory_eligible':False,
            'later_label_count':memory['summary']['label_count'],'current_mean_px':memory['summary']['current_hole_error_px']['mean'],
            'cached_mean_px':memory['summary']['cached_hole_error_px']['mean'],'fraction_cache_improved':memory['summary']['fraction_cache_improved'],
            'per_block':memory['per_block'],'conclusion':'fixed first-five CNN cache worsened mean hole error; does not fix memory initialization'},
        'control_eligibility':{'motor_commands':False,'xyz_disabled':True,'gate_enable_approved':False,
            'new_test_gate_false_positives_at_0_5':gate['0.5']['false_positive'],'new_test_gate_false_positives_at_0_8':gate['0.8']['false_positive'],
            'new_test_known_negative_count':gate['0.8']['known_negative_labels'],
            'reason':'3/4 and 2/4 known negatives falsely positive at thresholds0.5 and0.8; gate labels are not force-certified'},
        'video':{'path':str((directory/'heldout-blocks-cnn-memory-diagnostic.mp4').resolve()),
            'source_frames':len(video['frames']),'timing':video['timing'],'test_labels_used_for_rendering':video['test_labels_used'],
            'cached_hole_status':'unverified diagnostic ghost overlay; no trusted visibility fabricated'},
        'limitations':['B2 validation has only one pin label and one carrier pose; candidate selection is weak and not session-independent.',
            'B4/B5 improve within-recording localization; separate recordings and Oct27 challenge show different transfer behavior.',
            'Coordinate labels have uncalibrated AI annotation uncertainty; pixel errors below that scale do not certify precision.',
            'Image-plane direction cosine has no depth/base-frame/metric-XYZ or motor authority.',
            'All test results are post-selection reports and were not used for model, seed, or calibration tuning.']}
    report['artifact_sha256']={name:sha256(directory/name) for name in ('selection.json','manifest.json','test-evaluation.json','old-independent-evaluation.json',
        'old-challenge-evaluation.json','memory-protocol.json','memory-evaluation.json','heldout-blocks-cnn-memory-diagnostic.json')}
    output=Path(output)
    full=output.with_name('model-audit-summary.json')
    full.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def short_points(points):
        return {k:{'n':v['labels'],'mean_px':round(v['mean_px'],3) if v['mean_px'] is not None else None} for k,v in points.items()}
    def short_model(value):
        return {'landmarks':short_points(value['normal']['landmarks']),'gate':value['normal']['gate']}
    compact={'schema_version':1,'checkpoint':report['selected_checkpoint'],'checkpoint_sha256':report['selected_checkpoint_sha256'],
        'manifest_sha256':report['manifest_sha256'],'baseline_checkpoint':report['baseline_checkpoint'],
        'training':report['training'],'split':report['within_recording_split'],'selection_frozen_before_test':True,
        'selection_criterion':report['selection_metric'],
        'validation':{k:short_model(v) for k,v in report['validation'].items()},
        'new_test':{k:short_model(v) for k,v in new_domain.items()},
        'new_test_per_block':{b:short_points(v['landmarks']) for b,v in new_domain['candidate']['per_block'].items()},
        'new_test_constant_train_mean':short_points(report['new_test_constant_train_mean']),
        'old_independent_test':{k:short_model(v) for k,v in old_domain.items()},
        'old_independent_candidate_strata':{b:short_points(v['landmarks']) for b,v in old_domain['candidate']['strata'].items()},
        'oct27_challenge':{k:{'landmarks':short_points(v['landmarks']),
            'direction_cosine':round(v['image_plane_direction']['mean_cosine_all_nondegenerate'],3),
            'direction_valid_count':v['image_plane_direction']['nonzero_direction_count']} for k,v in report['oct27_failed_challenge'].items() if k in ('baseline','candidate')},
        'memory':{'later_labels':memory['summary']['label_count'],'current_mean_px':round(memory['summary']['current_hole_error_px']['mean'],3),
            'cache_mean_px':round(memory['summary']['cached_hole_error_px']['mean'],3),'cache_improved':int(sum(r['cached_hole_error_px']<r['current_hole_error_px'] for r in memory['records'])),'cache_worsened_or_equal':int(sum(r['cached_hole_error_px']>=r['current_hole_error_px'] for r in memory['records'])),
            'initialization_verified':False,'autonomous_eligible':False,'conclusion':'fixed first-five CNN cache worsens mean hole error'},
        'control_eligibility':report['control_eligibility'],'video':report['video']['path'],
        'full_audit':str(full.resolve()),'limitations':report['limitations']}
    output.write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n');print(output);return compact


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',default='work/pin-insertion-newdemo');p.add_argument('--output',default='work/pin-insertion-newdemo/model-summary.json')
    args=p.parse_args();summarize(args.directory,args.output)
