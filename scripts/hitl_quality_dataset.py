#!/usr/bin/env python3
"""Offline HITL audit and explicitly reviewed LeRobot export. Never contacts hardware.

All motion thresholds are descriptive heuristics, not correction-success labels.
Export keeps measured observations and camera timestamps; only next-pose action
labels receive a symmetric three-tap low-pass filter. Grippers stay unchanged.
Short corrections are retained. Use MaskedActionDataset and an explicitly
mask-aware action loss; unmodified PI05 may ignore padding masks. The optional
valid_start_sampler restricts compatibility training to full 50-action starts.
"""
import argparse
import hashlib
import json
import math
import inspect
import shutil
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from hitl_recording import data_timestamp, data_origin

GRIPPERS = [6, 13]
JOINTS = [i for i in range(14) if i not in GRIPPERS]


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def poses(rows):
    return np.asarray([sum((r['arms'][a]['position_rad'] + [r['arms'][a]['gripper']['position']] for a in ('left', 'right')), []) for r in rows], dtype=np.float64)


def human_blocks(rows):
    """Inclusive row bounds, with no crossing handoffs, pauses, or timestamp gaps."""
    start = None
    for i, row in enumerate(rows):
        boundary = i and (row.get('recording_segment') != rows[i-1].get('recording_segment') or
                          data_timestamp(row) <= data_timestamp(rows[i-1]) or
                          data_timestamp(row) - data_timestamp(rows[i-1]) > 150_000_000)
        if start is not None and (boundary or row.get('control_source') != 'human'):
            yield start, i - 1
            start = None
        if row.get('control_source') == 'human' and start is None:
            start = i
    if start is not None:
        yield start, len(rows)-1


def regular_grid(rows, fps=20):
    times = np.asarray([data_timestamp(r) for r in rows], dtype=np.int64)
    ticks = np.arange(times[0], times[-1]+1, round(1e9/fps), dtype=np.int64)
    right = np.searchsorted(times, ticks).clip(0, len(times)-1)
    left = (right-1).clip(0)
    idx = np.where(abs(times[left]-ticks) <= abs(times[right]-ticks), left, right)
    return ticks, idx, poses(rows)[idx]


def metrics(q, fps=20):
    if len(q) < 2:
        return dict(frames=len(q), quiet_fraction=1., max_step_rad=0., rms_acceleration_rad_s2=0.)
    d = np.diff(q, axis=0)
    return dict(frames=len(q), quiet_fraction=float(np.mean(np.max(abs(d), axis=1) <= .004)),
                max_step_rad=float(abs(d).max()),max_joint_speed_rad_s=float(abs(d[:,JOINTS]).max()*fps),peak_joint_acceleration_rad_s2=float(abs(np.diff(q[:,JOINTS],n=2,axis=0)).max()*fps**2) if len(q)>2 else 0.,
                rms_acceleration_rad_s2=float(np.sqrt(np.mean(np.diff(q[:, JOINTS], n=2, axis=0)**2))*fps**2) if len(q)>2 else 0.)


def active_slices(q, quiet=.004, internal_frames=20, context=2):
    """Trim only extended quiet runs; keep short contact dwells and context.

    Split internal quiet spans >=1s, never join nonadjacent source positions.
    Visual review must confirm omitted quiet spans are not intentional contact.
    """
    if len(q)<2:
        return []
    active = np.max(abs(np.diff(q, axis=0)), axis=1)>quiet
    # A slow purposeful nudge can stay below the one-tick threshold. Mark every
    # real transition covered by a net displacement over up to0.25seconds.
    for lag in range(2,min(5,len(q)-1)+1):
        moving=np.max(abs(q[lag:]-q[:-lag]),axis=1)>quiet
        for shift in range(lag):
            active[shift:shift+len(moving)] |= moving
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices)>internal_frames)+1)
    return [(max(0,int(g[0])-context), min(len(q),int(g[-1])+2+context)) for g in groups]



def covered_slices(ticks, state_times, camera_times, state_limit=.01, camera_limit=.04, minimum_knots=3):
    """Contiguous aligned knots; never bridge an unusable observation."""
    good=np.abs(state_times-ticks)/1e9<=state_limit
    for ft in camera_times.values():
        if not len(ft):
            good[:]=False
            continue
        ft=np.asarray(ft,dtype=np.int64)
        right=np.searchsorted(ft,ticks).clip(0,len(ft)-1);left=(right-1).clip(0)
        errors=np.minimum(abs(ft[left]-ticks),abs(ft[right]-ticks))/1e9
        good &= errors<=camera_limit
    transitions=np.diff(np.r_[False,good,False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)) if b-a>=minimum_knots]


def smooth_actions(q, bound=.008):
    """No time warp, endpoint changes or gripper smoothing. Reject excess bias."""
    result = q.copy()
    if len(q)>2:
        result[1:-1, JOINTS] = (.25*q[:-2, JOINTS] + .5*q[1:-1, JOINTS] + .25*q[2:, JOINTS])
    delta = float(np.max(abs(result-q))) if len(q) else 0.
    if delta > bound:
        raise ValueError(f'Smoothing displacement {delta:.6f} exceeds bound {bound}')
    return result, delta


def valid_starts(length, horizon=50, offset=0):
    if horizon < 1:
        raise ValueError('horizon must be positive')
    return list(range(offset, offset+max(0,length-horizon+1)))


def valid_start_sampler(root, generator=None):
    """Pass as DataLoader(sampler=..., shuffle=False); use ordinary LeRobotDataset
    with delta_timestamps={'action':[i / fps for i in range(50)]}. Dataset must
    include all episodes in export order; episode subsets/reindexing are invalid.
    """
    import torch
    data = json.loads((Path(root)/'processing/valid_starts.json').read_text())
    return torch.utils.data.SubsetRandomSampler(data['global_indices'], generator=generator)



class MaskedActionDataset:
    """Wrap complete native LeRobotDataset configured with 50 action timestamps.

    Caller must apply a mask-aware per-timestep loss inside the training model.
    Returning action_is_pad alone DOES NOT make unmodified PI05 mask-aware.
    No episode subsetting or index remapping is supported.
    """
    def __init__(self, dataset, root):
        self.dataset=dataset
        data=json.loads((Path(root)/'processing/valid_horizons.json').read_text())
        self.valid_lengths=data['global_valid_lengths'];self.horizon=data['horizon']
        if len(dataset)!=len(self.valid_lengths):
            raise ValueError('Dataset length/order must match complete export')
    def __len__(self):
        return len(self.valid_lengths)
    def __getitem__(self,index):
        import torch
        sample=dict(self.dataset[index])
        action=sample['action'].clone()
        if action.ndim!=2 or action.shape[0]!=self.horizon:
            raise ValueError('Native loader must request exactly the exported action horizon')
        valid=torch.arange(self.horizon,device=action.device)<self.valid_lengths[index]
        action[~valid]=0
        sample.update(action=action,action_is_pad=~valid,action_valid_mask=valid)
        return sample


def masked_action_mean(per_element_loss, valid_mask):
    """Reduce BxTxD (or BxT) loss over valid elements only. Masks are boolean.

    Use inside model forward before reduction; a scalar PI05 loss has already
    discarded timestep information and cannot be corrected with this helper.
    """
    import torch
    mask=torch.as_tensor(valid_mask,device=per_element_loss.device,dtype=torch.bool)
    if per_element_loss.ndim<2 or tuple(per_element_loss.shape[:2])!=tuple(mask.shape):
        raise ValueError('Expected BxT loss prefix and BxT validity mask')
    while mask.ndim<per_element_loss.ndim: mask=mask.unsqueeze(-1)
    mask=mask.expand_as(per_element_loss)
    count=mask.sum()
    if not count.item(): raise ValueError('No valid supervised action elements')
    return torch.where(mask,per_element_loss,torch.zeros_like(per_element_loss)).sum()/count


def masked_action_mse(prediction,target,valid_mask):
    """Example mask-aware objective; PI05 flow loss needs its unreduced error."""
    return masked_action_mean((prediction-target).square(),valid_mask)



def masked_pi05_forward(policy,batch,reduction='mean'):
    """Opt-in local PI05 forward adapter, applying mask BEFORE loss reduction.

    Invoke instead of policy(batch) in a deliberately updated trainer. Keep
    action_valid_mask after device transfer/preprocessing. No installed package
    is patched. RTC/prefix loss masking requires separate integration.
    """
    import torch
    if getattr(policy.config,'rtc_config',None) is not None:
        raise ValueError('RTC requires separate verified mask integration')
    if reduction not in ('mean','none'): raise ValueError('Unsupported reduction')
    valid=batch['action_valid_mask'].bool()
    images,img_masks=policy._preprocess_images(batch)
    actions=policy.prepare_action(batch)
    losses=policy.model.forward(images,img_masks,batch['observation.language.tokens'],batch['observation.language.attention_mask'],actions)
    losses=losses[:,:,:policy.config.output_features['action'].shape[0]]
    if tuple(losses.shape[:2])!=tuple(valid.shape): raise ValueError('Mask/loss horizon mismatch')
    weights=valid.to(device=losses.device).unsqueeze(-1).expand_as(losses)
    counts=weights.sum(dim=(1,2))
    if bool((counts==0).any()): raise ValueError('Each sample needs valid action supervision')
    masked=torch.where(weights,losses,torch.zeros_like(losses))
    per_sample=masked.sum(dim=(1,2))/counts
    loss=per_sample if reduction=='none' else per_sample.mean()
    loss_dict={'loss':per_sample.mean().detach().item(),'loss_per_dim':(masked.sum(dim=(0,1))/weights.sum(dim=(0,1)).clamp_min(1)).detach().cpu().tolist()}
    return loss,loss_dict


def audit(source, output):
    intervals, sources = [], []
    for directory in sorted(source.iterdir()):
        if not (directory/'metadata.json').exists():
            continue
        meta = json.loads((directory/'metadata.json').read_text())
        rows = read_jsonl(directory/'states.jsonl')
        frames = read_jsonl(directory/'frames.jsonl')
        origin = data_origin(meta)
        count = 0
        for start, end in human_blocks(rows):
            chunk = rows[start:end+1]
            ticks, idx, q = regular_grid(chunk)
            first_s, last_s = (int(ticks[0])-origin)/1e9, (data_timestamp(chunk[-1])-origin)/1e9
            camera_metrics = {}
            for camera in ('overhead','wrist_port2','wrist_port3'):
                ft = np.asarray(sorted(data_timestamp(f) for f in frames if f['camera']==camera and data_timestamp(chunk[0])<=data_timestamp(f)<=data_timestamp(chunk[-1])), dtype=np.int64)
                if len(ft):
                    r = np.searchsorted(ft,ticks).clip(0,len(ft)-1); l=(r-1).clip(0)
                    selected=np.where(abs(ft[l]-ticks)<=abs(ft[r]-ticks),l,r)
                    camera_metrics[camera]=dict(frames=len(ft), repeated_selection_fraction=float(1-len(set(selected))/len(selected)),max_alignment_s=float(abs(ft[selected]-ticks).max()/1e9))
                else:
                    camera_metrics[camera]=dict(frames=0,missing=True)
            candidates=[]
            for a,b in active_slices(q):
                candidates.append(dict(start_s=(int(ticks[a])-origin)/1e9,end_s=(int(ticks[b-1])-origin)/1e9,grid_frames=b-a,valid_50_step_starts=max(0,b-a-50),metrics=metrics(q[a:b])))
            interval=dict(id=f'{directory.name}-human-{count:03d}',source_id=directory.name,recording_segment=chunk[0].get('recording_segment'),start_s=first_s,end_s=last_s,source_rows=len(chunk),metrics=metrics(q),cameras=camera_metrics,candidates=candidates,complete=meta.get('complete') is True,visual_decision='pending')
            intervals.append(interval); count+=1
        sources.append(dict(source_id=directory.name,human_intervals=count,complete=meta.get('complete') is True,states=len(rows)))
    result=dict(schema_version=1,thresholds=dict(quiet_step_rad=.004,internal_quiet_split_s=1.,activity_method='Mark transitions if any coordinate changes more than0.004rad across1..5grid ticks; include every transition spanned by qualifying net displacement',activity_max_window_s=.25,smoothing_max_deviation_rad=.008,horizon=50,fps=20),sources=sources,intervals=intervals,summary=dict(sources=len(sources),human_intervals=len(intervals),candidate_segments=sum(len(x['candidates']) for x in intervals),candidate_valid_starts=sum(c['valid_50_step_starts'] for x in intervals for c in x['candidates'])),limitations=['Kinematic motion does not establish useful correction or task success.','Quiet trimming requires visual approval; contact dwells can be intentional.','Camera freshness is flagged, not corrected by interpolating images.'])
    write_json(output,result)
    return result



def validate_interval_integrity(rows,all_rows,metadata,decision):
    """Source completeness is independent of local interval sensor integrity."""
    if not rows or not np.isfinite(poses(rows)).all(): raise ValueError('Nonfinite measured interval poses')
    for field in ('monotonic_ns','data_monotonic_ns'):
        stamps=[r.get(field,r['monotonic_ns']) for r in rows]
        if any(b<=a for a,b in zip(stamps,stamps[1:])):raise ValueError('Nonincreasing interval timestamps')
    for arm in ('left','right'):
        feedback=np.asarray([r['arms'][arm]['motor_feedback_time'] for r in rows],dtype=float)
        faults=np.asarray([r['arms'][arm]['motor_fault'] for r in rows])
        if not np.isfinite(feedback).all() or np.any(faults!=0): raise ValueError('Motor fault or nonfinite motor feedback')
        if np.any(np.diff(feedback,axis=0)<0):raise ValueError('Motor feedback timestamps move backward')
        ages=np.asarray([r['unix_ns']/1e9 for r in rows])[:,None]-feedback
        if np.any(ages<0) or np.any(ages>.05):raise ValueError('Motor feedback age outside0..50ms')
    if metadata.get('complete') is True: return None
    allowed={('20261028T183037-e26713b4',4),('20261028T153841-ae44a1c5',4),('20261028T185549-482189f4',9),('20261028T185829-c6fd1af5',0),('20261028T184300-6a14071f',0),('20261028T185407-3f48617a',0),('20261028T185549-482189f4',5),('20261028T185549-482189f4',27)}
    if (decision['source_id'],decision['recording_segment']) not in allowed:raise ValueError('Incomplete source interval is not in audited allowlist')
    approval=decision.get('incomplete_interval_approved')
    if not isinstance(approval,dict):raise ValueError('Incomplete interval requires explicit audit approval')
    path=Path(approval['audit_path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=approval['audit_sha256']:raise ValueError('Incomplete audit hash mismatch')
    audits=json.loads(path.read_text())
    matches=[a for a in audits if a['source_id']==decision['source_id'] and a['recording_segment']==decision['recording_segment'] and a['start_s']-1e-8<=decision['start_s']<=decision['end_s']<=a['end_s']+1e-8]
    if len(matches)!=1:raise ValueError('Accepted subinterval not covered by audited range')
    end=rows[-1]['monotonic_ns']
    terminal=metadata['end_monotonic_ns']
    later=[r for r in all_rows if end<r['monotonic_ns']<terminal and (r.get('control_source')!='human' or r.get('recording_segment')!=decision['recording_segment'])]
    later.extend(r for r in metadata.get('control_transitions',[]) if end<r['monotonic_ns']<terminal and r.get('source') in ('hold','policy'))
    later.sort(key=lambda r:r['monotonic_ns'])
    if not later:raise ValueError('Incomplete interval has no later closed ownership boundary')
    return dict(audit_sha256=digest,audited_interval=matches[0],closed_boundary_sequence=later[0]['sequence'])



def validate_export_contract(spec,source_contract):
    for key in ('fps','image_size','task'):
        if spec[key]!=source_contract[key]:raise ValueError(f'Contract mismatch: {key}')
    if spec['image_crops_xyxy']==source_contract['image_crops_xyxy']:return
    normalization=spec.get('image_crop_normalization',{})
    if normalization.get('approved') is not True or normalization.get('method')!='recrop_original_640x480':
        raise ValueError('Crop mismatch requires explicit original-image normalization approval')
    if normalization.get('target_crops_xyxy')!=spec['image_crops_xyxy']:
        raise ValueError('Declared crop normalization target differs from output crops')
    if source_contract['image_crops_xyxy'] not in normalization.get('allowed_source_crops_xyxy',[]):
        raise ValueError('Source crop geometry not covered by normalization approval')


def export(source, decisions_path, output, repo_id, template):
    """Only explicit visual accepts with evidence can supervise derived actions."""
    from postprocess_kept import JpegLeRobotDataset, NAMES, CAMERAS, prepare_image, vector, nearest_index
    if output.exists():
        raise FileExistsError(output)
    decisions=json.loads(decisions_path.read_text())
    if isinstance(decisions,dict): decisions=decisions.get('decisions', decisions.get('intervals',[]))
    accepted=[d for d in decisions if d.get('decision')=='accept']
    if not accepted: raise ValueError('No visually accepted decisions')
    spec=json.loads(template.read_text()); fps=spec['fps']
    if fps!=20: raise ValueError('This audit requires 20Hz')
    normalization=spec.get('image_crop_normalization')
    if normalization:
        target_path=Path(normalization['target_contract_path']);target_bytes=target_path.read_bytes()
        if hashlib.sha256(target_bytes).hexdigest()!=normalization['target_contract_sha256']:raise ValueError('Target inference contract changed since crop approval')
        if json.loads(target_bytes)['image_crops_xyxy']!=spec['image_crops_xyxy']:raise ValueError('Output crops differ from target inference contract')
    features={k:dict(dtype='float32',shape=(14,),names=NAMES) for k in ('observation.state','action','action.raw_next_state')}
    features.update({f'observation.images.{c}':dict(dtype='image',shape=(spec['image_size'],spec['image_size'],3),names=['height','width','channels']) for c in CAMERAS})
    dataset=JpegLeRobotDataset.create(repo_id=repo_id,root=output,fps=fps,robot_type='panthera_ht_bimanual_real',features=features,use_videos=False,metadata_buffer_size=1)
    reports=[]; rejected=[]; indices=[]; lengths=[]; coverage_reports=[]; offset=0; occupied=set()
    for decision in accepted:
        if not decision.get('evidence'): raise ValueError('Accept requires visual evidence')
        identifier=decision['source_id']
        if Path(identifier).name!=identifier or identifier in ('.','..'): raise ValueError('Invalid source_id')
        directory=source/identifier; meta=json.loads((directory/'metadata.json').read_text()); origin=data_origin(meta)
        contract=meta['context']['contract']
        validate_export_contract(spec,contract)
        all_rows=read_jsonl(directory/'states.jsonl')
        chosen=[r for r in all_rows if decision['start_s']-1e-8<=(data_timestamp(r)-origin)/1e9<=decision['end_s']+1e-8]
        if not chosen or any(r.get('control_source')!='human' or r.get('recording_segment')!=decision['recording_segment'] for r in chosen): raise ValueError('Decision crosses ownership/segment boundary')
        if len(list(human_blocks(chosen)))!=1: raise ValueError('Decision crosses timestamp gap')
        try:
            integrity_approval=validate_interval_integrity(chosen,all_rows,meta,decision)
        except (ValueError,KeyError) as error:
            rejected.append(dict(source_id=identifier,start_s=decision['start_s'],end_s=decision['end_s'],reason='Interval integrity: '+str(error)))
            continue
        ticks,idx,q=regular_grid(chosen,fps)
        all_frames=read_jsonl(directory/'frames.jsonl')
        groups={c:[f for f in all_frames if f['camera']==c and data_timestamp(chosen[0])<=data_timestamp(f)<=data_timestamp(chosen[-1])] for c in CAMERAS}
        coverage=covered_slices(ticks,np.asarray([data_timestamp(chosen[i]) for i in idx]),{c:np.asarray([data_timestamp(f) for f in groups[c]]) for c in CAMERAS},spec['max_state_alignment_s'],spec['max_camera_alignment_s'])
        state_aligned=np.abs(np.asarray([data_timestamp(chosen[i]) for i in idx])-ticks)/1e9<=spec['max_state_alignment_s']
        coverage_counts={}
        for camera,group in groups.items():
            ft=np.asarray([data_timestamp(f) for f in group],dtype=np.int64)
            if len(ft):
                right=np.searchsorted(ft,ticks).clip(0,len(ft)-1);left=(right-1).clip(0)
                errors=np.minimum(abs(ft[left]-ticks),abs(ft[right]-ticks))/1e9
                coverage_counts[camera]=dict(source_frames=len(ft),grid_knots=len(ticks),outside_40ms=int(np.sum(errors>.04)),outside_50ms=int(np.sum(errors>.05)),outside_60ms=int(np.sum(errors>.06)),outside_75ms=int(np.sum(errors>.075)),max_alignment_s=float(errors.max()),median_source_gap_s=float(np.median(np.diff(ft))/1e9) if len(ft)>1 else None)
            else: coverage_counts[camera]=dict(source_frames=0,grid_knots=len(ticks),outside_40ms=len(ticks))
        coverage_reports.append(dict(source_id=identifier,recording_segment=decision['recording_segment'],start_s=decision['start_s'],end_s=decision['end_s'],grid_knots=len(ticks),state_unaligned_knots=int(np.sum(~state_aligned)),cameras=coverage_counts,aligned_runs=coverage,aligned_knots_in_minimum_runs=sum(b-a for a,b in coverage)))
        segments=[]
        for a,b in active_slices(q):
            for c,d in coverage:
                first,last=max(a,c),min(b,d)
                if last-first>=3: segments.append((first,last))
        if not segments:
            rejected.append(dict(source_id=identifier,start_s=decision['start_s'],end_s=decision['end_s'],reason='No contiguous active aligned run with 3 state knots',coverage_runs=coverage))
        for a,b in segments:
            local=q[a:b]
            try:
                smooth,deviation=smooth_actions(local)
            except ValueError as error:
                rejected.append(dict(source_id=identifier,start_s=(int(ticks[a])-origin)/1e9,end_s=(int(ticks[b-1])-origin)/1e9,reason=str(error)))
                continue
            quality=metrics(smooth)
            if quality['max_joint_speed_rad_s']>1.0 or quality['peak_joint_acceleration_rad_s2']>8.0:
                rejected.append(dict(source_id=identifier,start_s=(int(ticks[a])-origin)/1e9,end_s=(int(ticks[b-1])-origin)/1e9,reason='Derived joint target exceeds conservative 1 rad/s or 8 rad/s2 gate',metrics=quality))
                continue
            failure=None
            rows_out=[]; alignment=[]
            for j in range(a,b-1):
                tick=int(ticks[j]); row={'task':spec['task'],'observation.state':q[j].astype('float32'),'action':smooth[j-a+1].astype('float32'),'action.raw_next_state':q[j+1].astype('float32')}
                info=dict(source_time_s=(tick-origin)/1e9,action_time_s=(int(ticks[j+1])-origin)/1e9,state_data_monotonic_ns=data_timestamp(chosen[idx[j]]),action_data_monotonic_ns=data_timestamp(chosen[idx[j+1]]),state_sequence=chosen[idx[j]]['sequence'],action_sequence=chosen[idx[j+1]]['sequence'],cameras={})
                for c in CAMERAS:
                    group=groups[c]
                    if not group:
                        failure=f'Missing camera {c}';break
                    ft=np.asarray([data_timestamp(f) for f in group]); f=group[nearest_index(ft,tick)]
                    error=abs(data_timestamp(f)-tick)/1e9
                    if error>spec['max_camera_alignment_s']:
                        failure=f'Stale camera {c}: {error}';break
                    row[f'observation.images.{c}']=prepare_image(directory/f['path'],spec['image_crops_xyxy'][c],spec['image_size'])
                    info['cameras'][c]=dict(path=f['path'],offset_s=(data_timestamp(f)-tick)/1e9,source_sha256=hashlib.sha256((directory/f['path']).read_bytes()).hexdigest())
                if failure: break
                if max(abs(data_timestamp(chosen[idx[j]])-tick),abs(data_timestamp(chosen[idx[j+1]])-int(ticks[j+1])))/1e9>spec['max_state_alignment_s']: failure='State alignment exceeds contract';break
                rows_out.append(row);alignment.append(info)
            if failure:
                rejected.append(dict(source_id=identifier,start_s=(int(ticks[a])-origin)/1e9,end_s=(int(ticks[b-1])-origin)/1e9,reason=failure))
                continue
            source_keys={(identifier,item['state_sequence']) for item in alignment}
            if occupied & source_keys:
                rejected.append(dict(source_id=identifier,start_s=(int(ticks[a])-origin)/1e9,end_s=(int(ticks[b-1])-origin)/1e9,reason='Overlapping accepted decisions duplicate measured source states'))
                continue
            occupied.update(source_keys)
            for row in rows_out: dataset.add_frame(row)
            dataset.save_episode()
            starts=valid_starts(len(rows_out),50,offset);indices.extend(starts)
            lengths.extend(min(50,len(rows_out)-i) for i in range(len(rows_out)))
            reports.append(dict(episode_index=len(reports),source_id=identifier,source_image_crops_xyxy=contract['image_crops_xyxy'],output_image_crops_xyxy=spec['image_crops_xyxy'],image_crop_normalization=normalization,source_complete=meta.get('complete') is True,source_end_reason=meta.get('end_reason'),incomplete_interval_integrity_approval=integrity_approval,recording_segment=decision['recording_segment'],visual_decision=decision,start_s=(int(ticks[a])-origin)/1e9,end_s=(int(ticks[b-1])-origin)/1e9,source_hashes={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('metadata.json','states.jsonl','frames.jsonl')},before=metrics(local),after=metrics(smooth),repeated_camera_frames={c:len(alignment)-len({item['cameras'][c]['path'] for item in alignment}) for c in CAMERAS},max_smoothing_deviation_rad=deviation,valid_starts=len(starts),alignment=alignment))
            offset+=len(rows_out)
    dataset.finalize()
    write_json(output/'processing/rejected_candidates.json',rejected)
    if not reports: raise ValueError('No accepted interval has 2 aligned actions after cleanup; see rejected_candidates.json')
    write_json(output/'processing/valid_starts.json',dict(horizon=50,fps=fps,global_indices=indices,total_frames=offset,required_sampler='scripts.hitl_quality_dataset.valid_start_sampler'))
    write_json(output/'processing/valid_horizons.json',dict(horizon=50,fps=fps,global_valid_lengths=lengths,total_frames=offset))
    write_json(output/'processing/camera_coverage.json',dict(camera_alignment_limit_s=spec['max_camera_alignment_s'],reports=coverage_reports))
    (output/'processing/masked_training.py').write_text('import json\nfrom pathlib import Path\n\n'+'\n\n'.join(inspect.getsource(item) for item in (MaskedActionDataset,masked_action_mean,masked_action_mse,masked_pi05_forward)))
    audit_paths={item['visual_decision']['incomplete_interval_approved']['audit_path'] for item in reports if item['incomplete_interval_integrity_approval'] is not None}
    if audit_paths:
        digests={hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in audit_paths}
        if len(digests)!=1:raise ValueError('Incomplete intervals must use one canonical audit artifact')
        shutil.copy2(next(iter(audit_paths)),output/'processing/incomplete_interval_audit.json')
    shutil.copy2(decisions_path,output/'processing/visual_decisions.json')
    write_json(output/'processing/export_contract.json', spec)
    if normalization:shutil.copy2(normalization['target_contract_path'],output/'processing/target_inference_contract.json')
    (output/'processing/valid_start_sampler.py').write_text('''"""Sampler for this complete, unreindexed LeRobot export; never subset episodes."""
import json
from pathlib import Path
def valid_start_sampler(root, generator=None):
    from torch.utils.data import SubsetRandomSampler
    spec=json.loads((Path(root)/"processing/valid_starts.json").read_text())
    return SubsetRandomSampler(spec["global_indices"],generator=generator)
''')
    write_json(output/'processing/quality_manifest.json',dict(repo_id=repo_id,fps=fps,activity_detection=dict(net_displacement_rad=.004,window_ticks=[1,2,3,4,5],window_max_s=.25,method='OR all covered transitions for qualifying multiscale net displacement; trim/split quiet spans, no minimum duration'),quality_gates=dict(max_joint_speed_rad_s=1.0,peak_joint_acceleration_rad_s2=8.0,threshold_basis='Conservative offline descriptive acceptance gates; not a physical safety or task-success certification'),smoothing=dict(max_deviation_rad=.008,kernel=[.25,.5,.25],grippers_preserved=True,measured_observations_preserved=True),episodes=reports,action_semantics='Next measured pose with bounded three-tap smoothing of 12 joints only; measured observations and grippers preserved',training_requirement='Short episodes require MaskedActionDataset and mask-aware per-timestep loss; unmodified PI05 is not compatible with short starts. Optional valid_start_sampler selects full50 starts only.',decisions_sha256=hashlib.sha256(decisions_path.read_bytes()).hexdigest()))
    return dict(episodes=len(reports),frames=offset,valid_starts=len(indices))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['audit','export']);p.add_argument('--source',type=Path,default=ROOT/'data/hitl/review');p.add_argument('--output',type=Path,required=True);p.add_argument('--decisions',type=Path);p.add_argument('--repo-id');p.add_argument('--template',type=Path,default=ROOT/'config/hitl-quality-20261029.json');a=p.parse_args()
    result=audit(a.source,a.output) if a.command=='audit' else export(a.source,a.decisions,a.output,a.repo_id,a.template)
    print(json.dumps(result.get('summary',result)))
