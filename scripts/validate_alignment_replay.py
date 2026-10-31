#!/usr/bin/env python3
"""Offline fixed-heldout-trial image-motion and recorded-direction support audit.

Never imports robot control or changes calibration. Commands are counterfactual
joint offsets, not replayed motor commands. Empirical interpolation below is
observational evidence only; it cannot establish closed-loop hardware behavior.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .alignment_calibration_fit import _pairs, _regress
except ImportError:
    from alignment_calibration_fit import _pairs, _regress

THRESHOLDS = dict(lag_s=.2, max_pair_s=.5, local_radius_rad=.3,
    min_excitation_rad=.002, damping_px_per_rad=5., max_delta_rad=.006,
    observed_direction_cosine=.9, observed_start_radius_rad=.05,
    max_observed_motion_rad=.05, min_observed_motion_rad=.002,
    min_observed_image_motion_px=2., min_predicted_image_motion_px=1.)


def dls(jacobian, error):
    j, e = np.asarray(jacobian), np.asarray(error)
    delta = -j.T @ np.linalg.solve(j @ j.T + 25*np.eye(len(e)), e)
    return delta*min(1., .006/max(np.linalg.norm(delta), 1e-12))


def training_reference(samples):
    """Both envelope and objective use training trials only."""
    grouped = defaultdict(list)
    for row in samples:
        grouped[row['episode_id']].append(row)
    endpoints = [max(rows, key=lambda r:r['time_s']) for rows in grouped.values()]
    target = np.median([r['error_px'] for r in endpoints], axis=0)
    q = np.asarray([r['q_left']+r['q_right'] for r in samples])
    candidates = [np.median(q, axis=0)]
    candidates += [np.median([r['q_left']+r['q_right'] for r in rows], axis=0)
                   for rows in grouped.values()]
    center = max(candidates, key=lambda c:np.sum(np.max(np.abs(q-c), axis=1)<=.3))
    return center, target


def samples_for(rows, audit, task, pin_heads=(1, 1)):
    eps = {e['id']:e for e in audit['episodes']}
    cameras = {arm:cam for cam, arm in audit['mapping'].items() if arm in ('left','right')}
    result=[]
    for row in rows:
        phase = eps[row['episode_id']][task]
        if not phase['start_s'] <= row['time_s'] <= phase['end_s']:
            continue
        if abs(row['state_skew_s']) > .01:
            continue
        times = [row['frame_timestamps_ns'][cameras[a]] for a in ('left','right')]
        if abs(times[0]-times[1]) > 60_000_000:
            continue
        p=row['predictions'][task]
        ids=[next(i for i,n in enumerate(p['camera_names']) if row['camera_mapping'][n]==cameras[a]) for a in ('left','right')]
        names=p['landmark_names']; points=np.asarray(p['keypoints_2d'])
        moving=names.index('gear_center' if task=='gear' else 'pin_tip')
        targets=([names.index('slot_center')]*2 if task=='gear' else
                 [names.index(f'carrier_hole_{h}') for h in pin_heads])
        error=(points[ids,moving]-points[ids,targets]).ravel()
        if np.isfinite(error).all():
            result.append(dict(episode_id=row['episode_id'], time_s=row['time_s'],
                q_left=row['q_left'], q_right=row['q_right'], error_px=error.tolist()))
    return result


def temporal_pair_rows(samples, lag_seconds=.2, max_pair_seconds=.5):
    """Keep frame identities when making the fitter's non-overlapping pairs.

    A joint pose is not a frame identifier: stationary frames can share q while
    their time, target image positions, and measured image errors differ.
    """
    episodes=defaultdict(list)
    for row in samples:
        try:
            t=float(row['time_s'])
            q=np.r_[row['q_left'],row['q_right']].astype(float)
            e=np.asarray(row['error_px'],dtype=float).reshape(-1)
            if q.shape!=(12,) or e.shape!=(4,) or not np.isfinite(np.r_[t,q,e]).all():
                continue
        except (KeyError,ValueError,TypeError):
            continue
        episodes[str(row['episode_id'])].append(row)
    result=[]
    for episode,rows in sorted(episodes.items()):
        rows.sort(key=lambda r:float(r['time_s']))
        i=0
        while i<len(rows)-1:
            j=i+1
            while j<len(rows) and float(rows[j]['time_s'])-float(rows[i]['time_s'])<lag_seconds:
                j+=1
            if j>=len(rows): break
            if float(rows[j]['time_s'])-float(rows[i]['time_s'])<=max_pair_seconds:
                result.append((rows[i],rows[j])); i=j+1
            else:
                i+=1
    return result


def evaluate(samples, task, split=None):
    axes=[1,3] if task=='gear' else list(range(4))
    folds=[]; all_actual=[]; all_prediction=[]; all_baseline=[]; conditional_actual=[]; conditional_prediction=[]
    test_ids = ({r['episode_id'] for r in samples} if split is None else {ep for ep,s in split.items() if s=='validation'})
    for episode in sorted(test_ids):
        train=[r for r in samples if (r['episode_id']!=episode if split is None else split[r['episode_id']]=='train')]
        test=[r for r in samples if r['episode_id']==episode]
        center,target=training_reference(train)
        def local(rows):
            return [r for r in rows if np.max(np.abs(np.r_[r['q_left'],r['q_right']]-center))<=.3]
        train_pairs,_=_pairs(local(train), .2, .5)
        test_pairs=temporal_pair_rows(local(test), .2, .5)
        fold=dict(episode_id=episode, training_episode_count=len({r['episode_id'] for r in train}),
            training_pair_count=len(train_pairs), test_pair_count=len(test_pairs),
            target_error_px_train_only=target.tolist(), reference_joint_positions_train_only=center.tolist(),
            evaluated_pairs=[], command_evidence=[])
        folds.append(fold)
        if len(train_pairs)<15 or not test_pairs:
            fold['status']='insufficient_local_pairs'; continue
        dx=np.asarray([p[2]-p[1] for p in train_pairs]); dy=np.asarray([p[3] for p in train_pairs])
        model=_regress(dx[:,:6],dx[:,6:],dy)
        j=model['coefficient'].T[axes]
        fold['train_identifiable_rank']=model['rank']
        fold['train_conditional_excitation_rms_rad']=model['excitation'].tolist()
        baseline_coef=np.linalg.lstsq(np.column_stack((dx[:,6:],np.ones(len(dx)))),dy,rcond=1e-6)[0]
        motions=[]
        for start,end in test_pairs:
            q0=np.r_[start['q_left'],start['q_right']]
            q1=np.r_[end['q_left'],end['q_right']]
            y=np.asarray(end['error_px'])-np.asarray(start['error_px'])
            dq=q1-q0; nuisance=np.r_[dq[6:],1.]
            prediction=dq[:6] @ model['coefficient']+nuisance @ model['nuisance']
            baseline=nuisance @ baseline_coef
            actual=y[axes]; predicted=prediction[axes]
            all_actual.append(actual); all_prediction.append(predicted); all_baseline.append(baseline[axes])
            conditional_actual.append(actual-(nuisance @ model['nuisance'])[axes])
            conditional_prediction.append((dq[:6] @ model['coefficient'])[axes])
            error=np.asarray(start['error_px'])[axes]-target[axes]
            observed_gain=float(np.dot(error,error)-np.dot(error+actual,error+actual))
            predicted_gain=float(np.dot(error,error)-np.dot(error+predicted,error+predicted))
            fold['evaluated_pairs'].append(dict(time_s=start['time_s'], end_time_s=end['time_s'],
                start_error_px=np.asarray(start['error_px'])[axes].tolist(),
                end_error_px=np.asarray(end['error_px'])[axes].tolist(), dq_rad=dq.tolist(),
                observed_image_change_px=actual.tolist(), predicted_image_change_px=predicted.tolist(),
                nuisance_only_image_change_px=baseline[axes].tolist(),
                observed_squared_error_reduction_px2=observed_gain,
                predicted_squared_error_reduction_px2=predicted_gain))
            motions.append((q0,dq,actual))
        for row in local(test):
            q=np.r_[row['q_left'],row['q_right']]
            error=np.asarray(row['error_px'])[axes]-target[axes]
            delta=dls(j,error); full=np.r_[delta,np.zeros(6)]
            norm=float(np.linalg.norm(full)); eligible=[]
            for q0,dq,actual in motions:
                motion_norm=float(np.linalg.norm(dq))
                if .002<=motion_norm<=.05 and np.max(np.abs(q-q0))<=.05 and norm<=motion_norm and norm>1e-9:
                    cosine=float(full @ dq/(norm*motion_norm))
                    if cosine>=.9:
                        eligible.append((cosine,motion_norm,actual))
            record=dict(time_s=row['time_s'], delta_rad=delta.tolist(),
                supported_by_recorded_direction=bool(eligible))
            if eligible:
                cosine,motion_norm,actual=max(eligible,key=lambda p:p[0])
                observed_scaled=actual*(norm/motion_norm)
                record.update(direction_cosine=cosine, scale=norm/motion_norm,
                    observed_interpolated_change_px=observed_scaled.tolist(),
                    interpolated_squared_error_reduction_px2=float(error @ error-(error+observed_scaled) @ (error+observed_scaled)))
            fold['command_evidence'].append(record)
        fold['status']='evaluated'
    actual,pred,baseline=map(np.asarray,(all_actual,all_prediction,all_baseline))
    summary=dict(trials=len(folds), evaluated_trials=sum(f['status']=='evaluated' for f in folds),
        heldout_pairs=len(actual))
    if len(actual):
        rmse=lambda x:float(np.sqrt(np.mean(x*x)))
        eligible=(np.linalg.norm(actual,axis=1)>=2)&(np.linalg.norm(pred,axis=1)>=1)
        summary.update(rmse_px=rmse(actual-pred), zero_motion_rmse_px=rmse(actual),
            nuisance_only_rmse_px=rmse(actual-baseline),
            direction_pairs=int(eligible.sum()),
            direction_accuracy=float(np.mean(np.sum(actual[eligible]*pred[eligible],axis=1)>0)) if eligible.any() else None)
    if len(actual):
        ca,cp=np.asarray(conditional_actual),np.asarray(conditional_prediction)
        eligible=(np.linalg.norm(ca,axis=1)>=2)&(np.linalg.norm(cp,axis=1)>=1)
        summary.update(improvement_over_zero=1-summary['rmse_px']/max(1e-12,summary['zero_motion_rmse_px']),
            improvement_over_nuisance=1-summary['rmse_px']/max(1e-12,summary['nuisance_only_rmse_px']),
            conditional_direction_pairs=int(eligible.sum()),
            conditional_direction_accuracy=float(np.mean(np.sum(ca[eligible]*cp[eligible],axis=1)>0)) if eligible.any() else None)
        for fold in folds:
            pairs=fold['evaluated_pairs']
            if pairs:
                yy=np.asarray([p['observed_image_change_px'] for p in pairs]); pp=np.asarray([p['predicted_image_change_px'] for p in pairs])
                fold['observed_prediction_rmse_px']=float(np.sqrt(np.mean((yy-pp)**2)))
                fold['zero_motion_rmse_px']=float(np.sqrt(np.mean(yy**2)))
    commands=[c for f in folds for c in f['command_evidence']]
    supported=[c for c in commands if c['supported_by_recorded_direction']]
    summary.update(proposed_offsets=len(commands), observed_direction_supported_offsets=len(supported),
        observed_support_fraction=len(supported)/max(1,len(commands)),
        interpolated_improving_supported_offsets=sum(c['interpolated_squared_error_reduction_px2']>0 for c in supported))
    reasons=[]
    if summary.get('improvement_over_zero',-1)<.1 or summary.get('improvement_over_nuisance',-1)<.1:
        reasons.append('heldout_prediction_does_not_beat_baselines_by_existing_10_percent_margin')
    if summary.get('conditional_direction_pairs',0)<20 or (summary.get('conditional_direction_accuracy') or 0)<.65:
        reasons.append('heldout_direction_support_below_existing_minimum')
    if summary['observed_support_fraction']<1.:
        reasons.append('proposed_correction_directions_not_covered_by_recorded_motion')
    if task=='pin': reasons.append('physical_bore_association_unverified')
    return dict(summary=summary, validation_reasons=reasons, correction_effect_validated=False, folds=folds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-manifest',type=Path,required=True)
    args=parser.parse_args(); data=args.predictions.read_bytes(); audit_bytes=args.audit.read_bytes()
    rows=[json.loads(l) for l in data.splitlines() if l.strip()]; audit=json.loads(audit_bytes)
    manifest=json.loads(args.predictions.with_suffix('.manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('output_sha256')!=hashlib.sha256(data).hexdigest() or manifest.get('samples')!=len(rows):
        raise ValueError('Prediction manifest completeness/hash/count check failed')
    if manifest.get('phase_selection',{}).get('sha256')!=hashlib.sha256(audit_bytes).hexdigest():
        raise ValueError('Prediction phase audit identity mismatch')
    if any(r['camera_mapping']!=manifest['camera_mapping'] for r in rows):
        raise ValueError('Inconsistent camera routing')
    split_data=json.loads(args.split_manifest.read_text()); split={}
    for r in split_data['observations']:
        if r['episode_id'] in split and split[r['episode_id']]!=r['split']:
            raise ValueError('Episode crosses split boundary')
        split[r['episode_id']]=r['split']
    if set(split.values())!={'train','validation'}: raise ValueError('Expected fixed train/validation split')
    report=dict(format_version=1, motion_enabled=False, hardware_validated=False,
        thresholds=THRESHOLDS, provenance=dict(predictions=str(args.predictions.resolve()),
        predictions_sha256=hashlib.sha256(data).hexdigest(), audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(), episode_split=split),
        method='Fixed existing13training/4validation trial split. Training-only envelope/objective; real heldout finite-motion prediction; capped DLS command direction compared to nearby heldout recorded 12-joint motions with cosine>=0.9. Image response scaled downward only.',
        limitations=['Diagnostic CNN coordinates: no per-frame visibility/grasp filtering.',
            'Training endpoint residual is a demonstrated approach objective, not certified seating geometry.',
            'Pin uses fixed carrier_hole_1 in each view; physical bore correspondence is UNVERIFIED. Pin results are exploratory only.',
            'Counterfactual response is observable only near recorded directions; interpolation assumes local linearity and cannot establish causality or closed-loop dynamics.',
            'No replay command has been sent to hardware. Torque response, contact safety, grasp retention and achieved motion require hardware validation.'],tasks={})
    for task in ('gear','pin'):
        samples=samples_for(rows,audit,task)
        report['tasks'][task]=evaluate(samples,task,split)
        covered=[]
        for ep in audit['episodes']:
            times=[s['time_s'] for s in samples if s['episode_id']==ep['id']]
            covered.append(dict(episode_id=ep['id'], reference_s=ep[task]['reference_s'],
                nearest_reference_distance_s=min((abs(t-ep[task]['reference_s']) for t in times),default=None)))
        report['tasks'][task]['reference_coverage']=covered
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['summary'] for k,v in report['tasks'].items()},indent=2))

if __name__=='__main__': main()
