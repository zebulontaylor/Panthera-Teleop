#!/usr/bin/env python3
"""Evaluate initial-pretrained and fine-tuned visual representations on heldout data.

All linear probes are fitted on TRAIN weak-demo labels only, with equal weight per
episode. Optional human pair labels use JSONL fields pair_id, split, episode_id,
time_a_s, time_b_s, preference (a_better/b_better/equal), and observation_a/b
objects. Each observation contains a
current triplet of overhead/wrist_port2/wrist_port3 paths. Pair labels are
evaluation-only and never used for fitting.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from state_value_train import VisualValue, read_manifest


def image_batch(rows, root, tf):
    images=[]
    for row in rows:
        views=[]
        image_root=row.get('image_root') or root
        for ref in row['images']:
            with Image.open(Path(image_root)/ref) as im:
                views.append(tf(im.convert('RGB')))
        images.append(torch.stack(views))
    return torch.stack(images)


def episode_regression_metrics(rows, prediction, target, mask):
    by_ep=collections.defaultdict(list)
    for i in np.flatnonzero(mask):
        by_ep[str(rows[i]['episode_id'])].append((float(prediction[i]),float(target[i])))
    if not by_ep:
        return {'episodes':0,'frames':0,'rmse':None,'mae':None}
    mse=[];mae=[];frames=0
    for pairs in by_ep.values():
        p=np.asarray([x[0] for x in pairs]);y=np.asarray([x[1] for x in pairs])
        mse.append(float(np.mean((p-y)**2)));mae.append(float(np.mean(np.abs(p-y))));frames+=len(pairs)
    return {'episodes':len(by_ep),'frames':frames,'rmse':float(np.sqrt(np.mean(mse))),'mae':float(np.mean(mae))}


def episode_weights(rows, indices):
    counts=collections.Counter(str(rows[i]['episode_id']) for i in indices)
    weights=np.asarray([1.0/counts[str(rows[i]['episode_id'])] for i in indices],dtype=np.float64)
    return weights*(len(weights)/weights.sum())


def fit_ridge(train_x, train_y, weights, all_x, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(train_x,sample_weight=weights)
    model=Ridge(alpha=alpha,solver='lsqr')
    model.fit(scaler.transform(train_x),train_y,sample_weight=weights)
    return model.predict(scaler.transform(all_x)),{'scaler':scaler,'model':model}


def select_eval_rows(rows, pair_rows, hz=1.0, fresh_test_hz=None):
    if hz<=0:raise ValueError('evaluation hz must be positive')
    by_ep=collections.defaultdict(list)
    for i,r in enumerate(rows):by_ep[str(r['episode_id'])].append(i)
    selected=set()
    for ep,indices in by_ep.items():
        indices.sort(key=lambda i:float(rows[i]['t']))
        times=np.asarray([float(rows[i]['t']) for i in indices]);start,end=float(times[0]),float(times[-1])
        # `hz` is a sampling frequency, so the interval is its reciprocal.
        row_hz=fresh_test_hz if ep and rows[indices[0]]['split']=='fresh_test' and fresh_test_hz is not None else hz
        if row_hz<=0:raise ValueError('evaluation hz must be positive')
        interval=1.0/row_hz
        targets=list(np.arange(start,end+1e-8,interval))
        if not targets or end-targets[-1]>1e-6:targets.append(end)
        for t in targets:selected.add(indices[int(np.argmin(np.abs(times-t)))])
        # Preserve both sides of observed policy/human ownership changes in an
        # external HITL holdout so the descriptive score snapshots are real
        # boundary observations, even when they fall between 1 Hz samples.
        for left,right in zip(indices,indices[1:]):
            a,b=rows[left],rows[right]
            def owner(r):
                arms=r.get('control_source_by_arm') or {}
                return (arms.get('left'),arms.get('right'),r.get('control_source'),r.get('human_control_active'))
            if (a.get('split')=='fresh_test' or b.get('split')=='fresh_test') and owner(a)!=owner(b):
                selected.add(left);selected.add(right)
    # Reviewed comparisons are encoded from the exact raw-frame paths in their
    # portable pair exports; they do not alter the deterministic manifest sample
    # used to fit probes, even when a comparison belongs to TRAIN.
    return [rows[i] for i in sorted(selected)]


def pair_current_image_paths(pair, side, image_root):
    obs=(pair.get(f'observation_{side}') or pair.get('observations',{}).get(side))['current']
    names=('overhead','wrist_port2','wrist_port3')
    paths=[]
    for name in names:
        item=obs[name]
        p=Path(item['path'] if isinstance(item,dict) else item)
        if not p.is_absolute():p=Path(image_root or pair.get('_pair_image_root') or '.')/p
        paths.append(p)
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f"pair {pair.get('pair_id')} {side} image(s) missing: {missing}")
    return paths


def load_supplemental_manifest(path, dataset_root):
    """Read a frozen external split without admitting it to train/val/test fitting."""
    raw=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not raw:raise ValueError(f'empty supplemental manifest: {path}')
    original_splits={str(r['episode_id']):str(r.get('split')) for r in raw}
    if set(original_splits.values()) not in ({'fresh_test'},{'test'}):
        raise ValueError('supplemental manifest must contain only fresh_test rows (or test rows from the same frozen external export)')
    normalized=[]
    for r in raw:
        r['split']='test'
        r['_evaluation_split']='fresh_test'
        if not r.get('image_root') and dataset_root is not None:r['image_root']=str(dataset_root)
        normalized.append(r)
    with tempfile.TemporaryDirectory(prefix='state-value-eval-') as tmp:
        p=Path(tmp)/'supplemental.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in normalized))
        rows,_=read_manifest(p)
    for r in rows:r['split']='fresh_test'
    return rows


def load_episode_metadata(paths):
    result={}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():continue
            item=json.loads(line);ep=str(item['episode_id'])
            if ep in result:raise ValueError(f'duplicate supplemental episode metadata: {ep}')
            result[ep]=item
    return result


def pair_metrics(pair_rows, progress_by_name, temporal_scores, bootstrap=2000, seed=19):
    pairs=[r for r in pair_rows if r['preference'] in ('a_better','b_better')]
    result={'n_pairs':len(pairs),'n_equal_pairs':sum(r['preference']=='equal' for r in pair_rows),
            'n_unknown_pairs':sum(r['preference']=='unknown' for r in pair_rows),
            'n_pair_clusters':len({x['cluster'] for x in pairs}),
            'definition':'Reviewer preference indicates which observation appears further along the named visual component. These labels are evaluation-only, not task completion or physical progress labels.',
            'uncertainty_note':'Any interval resamples the available session/episode pair clusters only; same-day small cohorts do not support independent-day uncertainty estimates.'}
    methods={**{k:('progress',v) for k,v in progress_by_name.items()},'finetuned_signed_temporal_distance':('distance',temporal_scores)}
    def calc(method_pairs,kind,score):
        by_ep=collections.defaultdict(list)
        for pair in method_pairs:
            margin=(float(score[pair['b']])-float(score[pair['a']])) if kind=='progress' else float(score[(pair['a'],pair['b'])])
            correct=(margin>0) if pair['preference']=='b_better' else (margin<0)
            by_ep[pair['cluster']].append(float(correct))
        vals=np.asarray([np.mean(x) for x in by_ep.values()],dtype=float)
        item={'pair_cluster_macro_accuracy':float(vals.mean()) if len(vals) else None,'pair_clusters':len(vals)}
        if len(vals)>1:
            rng=np.random.default_rng(seed);draws=[float(np.mean(rng.choice(vals,len(vals),replace=True))) for _ in range(bootstrap)]
            item['pair_cluster_bootstrap_95_ci']=[float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))]
        return item
    for name,(kind,score) in methods.items():
        item=calc(pairs,kind,score)
        equal=[]
        for pair in pair_rows:
            if pair['preference']!='equal':continue
            gap=(abs(float(score[pair['b']])-float(score[pair['a']]))) if kind=='progress' else abs(float(score[(pair['a'],pair['b'])]))
            equal.append((pair['cluster'],gap))
        grouped=collections.defaultdict(list)
        for cluster,gap in equal:grouped[cluster].append(gap)
        session_gaps=[float(np.mean(v)) for v in grouped.values()]
        item['equal_pair_absolute_margin']={'pair_clusters':len(session_gaps),'mean_session_gap':float(np.mean(session_gaps)) if session_gaps else None,
                                            'median_pair_gap':float(np.median([v for _,v in equal])) if equal else None}
        components=sorted({x['component'] for x in pairs});item['by_progress_component']={c:calc([x for x in pairs if x['component']==c],kind,score) for c in components}
        splits=sorted({x['split'] for x in pairs});item['by_pair_set_split']={s:calc([x for x in pairs if x['split']==s],kind,score) for s in splits}
        result[name]=item
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pair-labels',type=Path,action='append',default=[],help='reviewer JSONL; may be repeated. Each set is reported separately.')
    ap.add_argument('--pair-image-root',type=Path,help='override image root for all pair exports; otherwise each JSONL file directory is used')
    ap.add_argument('--evaluation-hz',type=float,default=1.0,help='deterministic maximum sampling rate per episode; exact reviewer pair observations are evaluated separately')
    ap.add_argument('--fresh-test-hz',type=float,help='optional separate sampling frequency for fresh_test rollouts (e.g. 5 for every observation)')
    ap.add_argument('--supplemental-manifest',type=Path,action='append',default=[],help='separate frozen fresh-test manifest; never used in fitting or candidate selection')
    ap.add_argument('--supplemental-dataset-root',type=Path,help='image root for supplemental manifest relative image paths')
    ap.add_argument('--supplemental-episodes',type=Path,action='append',default=[],help='episode metadata JSONL containing operator outcomes and assistance provenance')
    ap.add_argument('--supplemental-endstates',type=Path,action='append',default=[],help='optional descriptive visual end-state review JSONL; unknown remains unknown')
    ap.add_argument('--splits',nargs='+',choices=('train','val','test','fresh_test'),default=('train','val','test'),help='only encode and report these splits; use train val for candidate selection before the final test run')
    ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--ridge-alpha',type=float,default=10.)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=19)
    args=ap.parse_args()
    if args.batch_size<1 or args.ridge_alpha<=0:ap.error('batch-size and ridge-alpha must be positive')
    torch.set_num_threads(max(1,args.workers));random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows,episodes=read_manifest(args.manifest)
    base_episodes={str(r['episode_id']) for r in rows}
    for path in args.supplemental_manifest:
        extra=load_supplemental_manifest(path,args.supplemental_dataset_root)
        overlap=base_episodes & {str(r['episode_id']) for r in extra}
        if overlap:raise ValueError(f'supplemental fresh-test episode overlaps primary manifest: {sorted(overlap)[:3]}')
        rows.extend(extra)
    metadata=load_episode_metadata(args.supplemental_episodes)
    endstate_metadata=load_episode_metadata(args.supplemental_endstates)
    for r in rows:
        if str(r['episode_id']) in metadata:r['_operator_outcome']=metadata[str(r['episode_id'])].get('operator_outcome')
        if str(r['episode_id']) in endstate_metadata:r['_visual_terminal_review']=endstate_metadata[str(r['episode_id'])]
    if 'fresh_test' in args.splits and not args.supplemental_manifest:raise ValueError('fresh_test requested without --supplemental-manifest')
    available=set(r['split'] for r in rows)
    missing_splits=set(args.splits)-available
    if missing_splits:raise ValueError(f'requested splits absent from input: {sorted(missing_splits)}')
    rows=[r for r in rows if r['split'] in set(args.splits)]
    pair_sets={};pair_paths={}
    for path in args.pair_labels:
        set_name=f'{path.parent.parent.name}_{path.stem}'
        if set_name in pair_sets:raise ValueError(f'duplicate pair-set name: {set_name}')
        pair_sets[set_name]=[];pair_paths[set_name]=path
        for line in path.read_text().splitlines():
            if not line.strip():continue
            item=json.loads(line);item['_pair_image_root']=str(path.parent);pair_sets[set_name].append(item)
    all_pairs=[p for items in pair_sets.values() for p in items]
    if args.fresh_test_hz is not None and args.fresh_test_hz<=0:ap.error('--fresh-test-hz must be positive')
    rows=select_eval_rows(rows,all_pairs,args.evaluation_hz,args.fresh_test_hz)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    conf=ck['config'];backbone_name=conf.get('backbone','resnet50');size=int(conf.get('image_size',224));horizon=float(conf.get('value_horizon',30.))
    finetuned=VisualValue(pretrained=False,backbone=backbone_name);finetuned.load_state_dict(ck['model']);finetuned.to(device).eval()
    pretrained=VisualValue(pretrained=True,backbone=backbone_name).to(device).eval()
    tf=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    train_idx=np.asarray([i for i,r in enumerate(rows) if r['split']=='train' and r['_weak']],dtype=int)
    if not len(train_idx):raise ValueError('no weak-demo train rows for fitting frozen-feature probes')
    all_features={'pretrained':[],'finetuned':[],'finetuned_fused':[]};native_progress=[];native_value=[]
    # Encode all frames once per encoder. The same resize/normalization and camera order are used.
    for start in range(0,len(rows),args.batch_size):
        subset=rows[start:start+args.batch_size]
        batch=image_batch(subset,args.dataset_root,tf).to(device)
        with torch.inference_mode():
            z0=pretrained.backbone(batch.reshape(-1,3,size,size)).reshape(len(subset),-1)
            zf=finetuned.backbone(batch.reshape(-1,3,size,size)).reshape(len(subset),-1)
            fused=finetuned.fuse(zf)
            native_progress.extend(torch.sigmoid(finetuned.progress(fused)).flatten().cpu().numpy().tolist())
            native_value.extend(torch.sigmoid(finetuned.value(fused)).flatten().cpu().numpy().tolist())
            all_features['pretrained'].append(z0.cpu().numpy().astype(np.float32))
            all_features['finetuned'].append(zf.cpu().numpy().astype(np.float32))
            all_features['finetuned_fused'].append(fused.cpu().numpy().astype(np.float32))
        if start==0 or (start//args.batch_size)!=(max(start-args.batch_size,0)//args.batch_size):
            print(json.dumps({'encoded_rows':min(start+len(subset),len(rows)),'total_rows':len(rows)}),flush=True)
    x={k:np.concatenate(v,axis=0) for k,v in all_features.items()}
    phase=np.asarray([float(r['_phase']) for r in rows],dtype=np.float32)
    remaining=np.asarray([float(r['_remaining']) for r in rows],dtype=np.float32)
    value_target=np.exp(-remaining/horizon)
    target={'progress':phase,'value':value_target}
    native_progress=np.asarray(native_progress);native_value=np.asarray(native_value)

    predictions={'finetuned_native_progress':native_progress,'finetuned_native_value':native_value}
    models={}
    weights=episode_weights(rows,train_idx)
    for enc in ('pretrained','finetuned'):
        for task in ('progress','value'):
            pred,model=fit_ridge(x[enc][train_idx],target[task][train_idx],weights,x[enc],args.ridge_alpha)
            predictions[f'{enc}_linear_probe_{task}']=pred;models[f'{enc}_{task}']=model

    # Metadata diagnostics: raw time is a train-only baseline; normalized timestamp is a
    # deliberately privileged, non-deployable oracle because weak targets are defined from it.
    raw_t=np.asarray([float(r['t'])-float(r.get('episode_start_s',0.)) for r in rows],dtype=np.float32)[:,None]
    tpred,_=fit_ridge(raw_t[train_idx],phase[train_idx],weights,raw_t,args.ridge_alpha)
    state_dims={len(r['state']) for r in rows if r.get('state') is not None}
    state_pred=None
    if len(state_dims)==1 and all(r.get('state') is not None for r in rows):
        states=np.asarray([r['state'] for r in rows],dtype=np.float32)
        state_pred,_=fit_ridge(states[train_idx],phase[train_idx],weights,states,args.ridge_alpha)

    metric_report={'warning':'Weak targets reflect normalized demonstration time and discounted remaining recording time in kept demonstrations; they are not verified physical progress or task value. No heldout weak labels/pairs fit probes or model weights.',
        'evaluation_sampling':{'maximum_hz_per_episode':args.evaluation_hz,'fresh_test_hz':args.fresh_test_hz or args.evaluation_hz,'rows_after_sampling':len(rows),'pair_label_observations_are_encoded_at_their_exported_raw_frame_paths':True,'ownership_boundary_observations_preserved_for_fresh_test':True},
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'train_only_probe':{'type':'Ridge on frozen 3-view ResNet50 pooled embeddings','alpha':args.ridge_alpha,'episode_weighting':'equal total weight per training episode','fit_rows':len(train_idx)},'splits':{}}
    split_names=[s for s in args.splits if any(r['split']==s for r in rows)]
    for split in split_names:
        weak_mask=np.asarray([r['split']==split and r['_weak'] for r in rows])
        split_metrics={'weak_demo_rows':int(weak_mask.sum()),'weak_demo_episodes':len({str(rows[i]['episode_id']) for i in np.flatnonzero(weak_mask)}),'progress':{},'discounted_value':{}}
        for name,pred in predictions.items():
            task='value' if name.endswith('value') else 'progress'
            split_metrics['discounted_value' if task=='value' else 'progress'][name]=episode_regression_metrics(rows,pred,target[task],weak_mask)
        split_metrics['progress']['raw_elapsed_time_ridge']=episode_regression_metrics(rows,tpred,phase,weak_mask)
        split_metrics['progress']['normalized_timestamp_oracle_nondeployable']=episode_regression_metrics(rows,phase,phase,weak_mask)
        split_metrics['discounted_value']['remaining_time_oracle_nondeployable']=episode_regression_metrics(rows,value_target,value_target,weak_mask)
        if state_pred is not None:split_metrics['progress']['measured_state_ridge']=episode_regression_metrics(rows,state_pred,phase,weak_mask)
        metric_report['splits'][split]=split_metrics

    # The native value output mixes weak clock targets with sparse exact-endpoint
    # BCE labels. Report those endpoint rows separately as a diagnostic, not as
    # calibrated success probabilities.
    terminal_report={}
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    for split in split_names:
        indices=[i for i,r in enumerate(rows) if r['split']==split and r.get('terminal_outcome') in (0,1)]
        y=np.asarray([int(rows[i]['terminal_outcome']) for i in indices],dtype=int)
        score=np.asarray([native_value[i] for i in indices],dtype=float)
        item={'n':len(indices),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),
              'episode_ids':[str(rows[i]['episode_id']) for i in indices],
              'brier':float(brier_score_loss(y,score)) if len(set(y))==2 else None,
              'mean_model_score':float(score.mean()) if len(score) else None,
              'score_range':[float(score.min()),float(score.max())] if len(score) else None,
              'auroc':float(roc_auc_score(y,score)) if len(set(y))==2 else None,
              'average_precision':float(average_precision_score(y,score)) if len(set(y))==2 else None,
              'interpretation':'Sparse reviewed terminal visual outcome only. Score is a mixed-objective value output, not a calibrated task-success probability.'}
        terminal_report[split]=item
    metric_report['reviewed_terminal_output_diagnostic']=terminal_report

    # Operator review metadata is descriptive. Labels such as `review` and a
    # false success field are not synonymous with a task failure. Never compute
    # success/failure classification metrics from those fields.
    if any(r['split']=='fresh_test' for r in rows):
        groups=collections.defaultdict(list)
        for i,r in enumerate(rows):
            if r['split']=='fresh_test':groups[str(r['episode_id'])].append(i)
        review_rows=[];boundary_rows=[]
        def ownership(r):
            arms=r.get('control_source_by_arm') or {}
            return (str(arms.get('left','unknown')),str(arms.get('right','unknown')),
                    str(r.get('control_source','unknown')),bool(r.get('human_control_active',False)))
        for ep,indices in groups.items():
            indices.sort(key=lambda i:float(rows[i]['t']))
            terminal_i=indices[-1];terminal_row=rows[terminal_i]
            outcome=terminal_row.get('_operator_outcome')
            review_rows.append({'episode_id':ep,'session_id':terminal_row.get('session_id'),'frames_scored':len(indices),
                'terminal_t_s':float(terminal_row['t']),'terminal_visual_progress_score':float(native_progress[terminal_i]),
                'terminal_mixed_objective_value_score':float(native_value[terminal_i]),
                'operator_outcome':outcome,'human_assisted':None if outcome is None else outcome.get('human_assisted'),
                'policy_success':None if outcome is None else outcome.get('policy_success'),
                'operator_success_field':None if outcome is None else outcome.get('success'),
                'operator_label':None if outcome is None else outcome.get('label'),
                'visual_terminal_review':endstate_metadata.get(ep),
                'interpretation':'Uncalibrated model scores at the final recorded observation only. Operator review status is descriptive and is not converted to a task failure label.'})
            for prev_i,next_i in zip(indices,indices[1:]):
                before,after=rows[prev_i],rows[next_i]
                if ownership(before)!=ownership(after):
                    boundary_rows.append({'episode_id':ep,'time_before_s':float(before['t']),'time_after_s':float(after['t']),
                        'ownership_before':{'control_source_by_arm':before.get('control_source_by_arm'),'control_source':before.get('control_source'),'recording_segment':before.get('recording_segment')},
                        'ownership_after':{'control_source_by_arm':after.get('control_source_by_arm'),'control_source':after.get('control_source'),'recording_segment':after.get('recording_segment')},
                        'progress_score_before':float(native_progress[prev_i]),'progress_score_after':float(native_progress[next_i]),
                        'mixed_objective_value_before':float(native_value[prev_i]),'mixed_objective_value_after':float(native_value[next_i]),
                        'score_changes_are_descriptive_not_success_or_physical_progress_labels':True})
        metric_report['fresh_test_operator_review']={'episodes':len(review_rows),'label_counts':dict(collections.Counter(str(x['operator_label']) for x in review_rows)),
            'human_assisted_count':sum(bool(x['human_assisted']) for x in review_rows),'policy_success_count':sum(bool(x['policy_success']) for x in review_rows),
            'operator_success_field_true_count':sum(bool(x['operator_success_field']) for x in review_rows),
            'binary_task_outcome_metrics':'not computed: `review`/success=false/operator review does not establish a task failure label; no explicit successes were supplied',
            'fresh_test_model_selection':'excluded from training and all candidate selection; report only after checkpoint selection',
            'terminal_rows':review_rows,'ownership_boundary_events':len(boundary_rows),
            'visual_terminal_state_counts':dict(collections.Counter(str(v.get('visual_terminal_state')) for v in endstate_metadata.values())),
            'visual_terminal_review_scope':'Visual terminal completeness for pin-through-peripheral-bore only; incomplete/unknown are not success/failure probabilities or mechanical retention labels.'}
        (args.output/'fresh_test_terminal_operator_review.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in review_rows))
        (args.output/'fresh_test_ownership_boundaries.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in boundary_rows))

    # Evaluate fine-tuned signed temporal head on reviewed pairs, if supplied. The head
    # was trained on within-episode temporal pairs; cross-episode pair use remains unvalidated.
    if pair_sets:
        metric_report['reviewed_pair_metrics']={}
        pair_prediction_rows=[]
        pair_tf=tf
        for set_name,pairs in pair_sets.items():
            parsed=[];obs_features={};temporal={};clock_scores={};next_i=0
            for n,pair in enumerate(pairs):
                pref=str(pair.get('preference','')).lower()
                if pref not in ('a_better','b_better','equal','unknown'):raise ValueError(f'{set_name} pair {n} preference must be a_better/b_better/equal/unknown')
                if pair.get('split') not in ('train','val','test','external_new_cohort_holdout','fresh_test_external_only'):raise ValueError(f'{set_name} pair {n} has an unrecognized split')
                if pair.get('split') in ('external_new_cohort_holdout','fresh_test_external_only') and not (('test' in args.splits) or ('fresh_test' in args.splits)):raise ValueError('external holdout pair labels require explicitly selecting a heldout split')
                if pair.get('split') in ('train','val','test') and pair.get('split') not in args.splits:raise ValueError(f'{set_name} pair {n} split was excluded by --splits')
                parsed_pair={'pair_id':str(pair.get('pair_id',n)),'cluster':str(pair.get('session_id',pair.get('episode_id',n))),
                             'preference':pref,'component':str(pair.get('progress_component','unspecified')),
                             'split':str(pair.get('split')),'a':next_i,'b':next_i+1}
                for side,idx in (('a',next_i),('b',next_i+1)):
                    paths=pair_current_image_paths(pair,side,args.pair_image_root)
                    im=torch.stack([pair_tf(Image.open(p).convert('RGB')) for p in paths]).to(device)
                    with torch.inference_mode():
                        zz=finetuned.backbone(im).reshape(1,-1)
                        z0=pretrained.backbone(im).reshape(1,-1)
                        ff=finetuned.fuse(zz)
                        native_p=float(torch.sigmoid(finetuned.progress(ff)).item());native_v=float(torch.sigmoid(finetuned.value(ff)).item())
                    obs_features[idx]={'pretrained':z0.cpu().numpy()[0].astype(np.float32),'finetuned':zz.cpu().numpy()[0].astype(np.float32),
                                       'finetuned_fused':ff.cpu().numpy()[0].astype(np.float32),'finetuned_native_progress':native_p,'finetuned_native_value':native_v}
                with torch.inference_mode():
                    za=torch.as_tensor(obs_features[next_i]['finetuned_fused'],device=device)[None]
                    zb=torch.as_tensor(obs_features[next_i+1]['finetuned_fused'],device=device)[None]
                    temporal[(next_i,next_i+1)]=float(finetuned.distance(torch.cat([za,zb,zb-za],-1)).item())
                clock_scores[next_i]=float(pair.get('time_a_s',0.));clock_scores[next_i+1]=float(pair.get('time_b_s',0.))
                parsed.append(parsed_pair);next_i+=2
            pair_scores={'finetuned_native_progress':{i:v['finetuned_native_progress'] for i,v in obs_features.items()}}
            for enc in ('pretrained','finetuned'):
                for task in ('progress','value'):
                    pp=[]
                    for i in range(next_i):
                        xx=obs_features[i][enc][None]
                        pack=models[f'{enc}_{task}'];pp.append(float(pack['model'].predict(pack['scaler'].transform(xx))[0]))
                    if task=='progress':pair_scores[f'{enc}_linear_probe_progress']={i:p for i,p in enumerate(pp)}
            pair_scores['finetuned_native_progress']={i:v['finetuned_native_progress'] for i,v in obs_features.items()}
            pair_scores['within_episode_clock_oracle_nondeployable']=clock_scores
            report=pair_metrics(parsed,pair_scores,temporal,args.bootstrap,args.seed)
            for pair in parsed:
                a,b=pair['a'],pair['b'];row={k:pair[k] for k in ('pair_id','cluster','preference','component','split')}
                row['scores']={name:{'a':float(score[a]),'b':float(score[b]),'b_minus_a':float(score[b])-float(score[a])}
                               for name,score in pair_scores.items() if name!='finetuned_signed_temporal_distance'}
                row['scores']['finetuned_signed_temporal_distance']={'a_to_b':float(temporal[(a,b)])}
                pair_prediction_rows.append(row)
            path=pair_paths[set_name]
            report['label_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            report['n_rows_in_export']=len(pairs);report['pair_set']=set_name
            metric_report['reviewed_pair_metrics'][set_name]=report
        (args.output/'reviewed_pair_predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in pair_prediction_rows))
    else:
        metric_report['reviewed_pair_metrics']={'status':'not supplied'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(metric_report,indent=2,allow_nan=False)+'\n')
    with (args.output/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(rows):
            f.write(json.dumps({'episode_id':r['episode_id'],'split':r['split'],'t':r['t'],'weak_demo':r['_weak'],'phase_target':float(phase[i]) if r['_weak'] else None,'discounted_value_target':float(value_target[i]) if r['_weak'] else None,**{k:float(v[i]) for k,v in predictions.items()},'raw_elapsed_time_ridge':float(tpred[i]),'normalized_timestamp_oracle_nondeployable':float(phase[i]),'measured_state_ridge':None if state_pred is None else float(state_pred[i])},allow_nan=False)+'\n')
    if any(r['split']=='fresh_test' for r in rows):
        with (args.output/'fresh_test_rollout_scores.jsonl').open('w') as f:
            for i,r in enumerate(rows):
                if r['split']!='fresh_test':continue
                f.write(json.dumps({'episode_id':r['episode_id'],'session_id':r.get('session_id'),'t':float(r['t']),
                    'weak_demo_progress_uncalibrated':float(native_progress[i]),'weak_demo_discounted_value_uncalibrated':float(native_value[i]),
                    'control_source':r.get('control_source'),'control_source_by_arm':r.get('control_source_by_arm'),
                    'human_control_active':r.get('human_control_active'),'human_intervention_episode_observed':r.get('human_intervention_episode_observed'),
                    'recording_segment':r.get('recording_segment'),'policy_time_s':r.get('policy_time_s'),'data_time_s':r.get('data_time_s'),
                    'interpretation':'Raw model scores at sampled fresh-test observations; uncalibrated and not verified task progress/value.'},allow_nan=False)+'\n')
    print(json.dumps(metric_report,indent=2),flush=True)


if __name__=='__main__':main()
