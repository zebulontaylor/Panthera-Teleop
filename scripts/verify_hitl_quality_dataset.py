#!/usr/bin/env python3
"""Independent source, supervision, image and native-chunk audit of cleaned HITL."""
import argparse
import io
import runpy
import sys
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from hitl_quality_dataset import valid_start_sampler
from postprocess_kept import CAMERAS, data_origin, data_timestamp, prepare_image, read_rows, sha256, source_path, vector, write_json


def trajectory_metrics(actions, states, fps):
    actions = np.asarray(actions, dtype=float)
    states = np.asarray(states, dtype=float)
    quiet = np.max(np.abs(actions-states), axis=1) <= .004
    tail = 0
    for value in quiet[::-1]:
        if not value:
            break
        tail += 1
    return dict(frames=len(actions), quiet_fraction=float(quiet.mean()), quiet_tail_seconds=tail/fps,
                maximum_acceleration_rad_s2=float(np.max(np.abs(np.diff(actions, n=2, axis=0)))*fps**2) if len(actions)>2 else 0.,
                maximum_jerk_rad_s3=float(np.max(np.abs(np.diff(actions, n=3, axis=0)))*fps**3) if len(actions)>3 else 0.)


def precise_pose(row):
    return np.asarray([v for arm in ('left','right') for v in row['arms'][arm]['position_rad']+[row['arms'][arm]['gripper']['position']]],dtype=float)


def knot_metrics(q, fps=20):
    joints=[i for i in range(14) if i not in (6,13)]
    d=np.diff(q,axis=0); acc=np.diff(q[:,joints],n=2,axis=0)
    return dict(frames=len(q),quiet_fraction=float(np.mean(np.max(abs(d),axis=1)<=.004)),
                max_step_rad=float(abs(d).max()),max_joint_speed_rad_s=float(abs(d[:,joints]).max()*fps),
                peak_joint_acceleration_rad_s2=float(abs(acc).max()*fps**2),
                rms_acceleration_rad_s2=float(np.sqrt(np.mean(acc**2))*fps**2))


def verify(root, source):
    # Delete no prior receipts: write a new receipt only after every assertion passes.
    manifest = json.loads((root/'processing/quality_manifest.json').read_text())
    contract = json.loads((root/'processing/export_contract.json').read_text())
    decisions_path=root/'processing/visual_decisions.json'
    assert sha256(decisions_path)==manifest['decisions_sha256']
    decisions=json.loads(decisions_path.read_text())
    if isinstance(decisions,dict): decisions=decisions.get('decisions',decisions.get('intervals',[]))
    accepted=[d for d in decisions if d.get('decision')=='accept']
    saved_starts=json.loads((root/'processing/valid_starts.json').read_text())
    assert saved_starts['horizon']==50 and saved_starts['fps']==20
    saved_horizons=json.loads((root/'processing/valid_horizons.json').read_text())
    assert saved_horizons['horizon']==50 and saved_horizons['fps']==20
    masked_module=runpy.run_path(str(root/'processing/masked_training.py'))
    info = json.loads((root/'meta/info.json').read_text())
    assert info['fps'] == contract['fps'] == 20
    rows = []
    for path in sorted((root/'data').rglob('*.parquet')):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: row['index'])
    assert [r['index'] for r in rows] == list(range(len(rows)))
    assert len(rows) == info['total_frames']
    statistics=json.loads((root/'meta/stats.json').read_text())
    for feature,values in statistics.items():
        for name,value in values.items():
            assert np.isfinite(np.asarray(value,dtype=float)).all(), (feature,name)
    for feature,description in info['features'].items():
        if description['dtype'] in ('image','video','string'): continue
        values=np.asarray([r[feature] for r in rows],dtype=float)
        if values.ndim==1: values=values[:,None]
        expected_stats={'mean':values.mean(axis=0),'std':values.std(axis=0),'min':values.min(axis=0),'max':values.max(axis=0),'count':[len(rows)]}
        expected_stats.update({f'q{int(q*100):02d}':np.quantile(values,q,axis=0) for q in (.01,.1,.5,.9,.99)})
        for name,value in expected_stats.items():
            np.testing.assert_allclose(statistics[feature][name],value,atol=1e-12,rtol=1e-10)

    assert len(manifest['episodes']) == info['total_episodes']
    native = LeRobotDataset(manifest['repo_id'], root=root, video_backend='pyav',
        delta_timestamps={'action': [i/info['fps'] for i in range(50)]})
    # Native action queries otherwise decode up to150 unrelated JPEGs per start
    # in the installed loader's row-query fallback. Verify full-loader parity
    # before narrowing its HF view; native horizon/boundary code is unchanged.
    parity_indices=sorted({0,len(rows)//2,len(rows)-1})
    full_samples={i:native[i] for i in parity_indices}
    native.hf_dataset=native.hf_dataset.remove_columns(list(native.meta.camera_keys))
    for i,full in full_samples.items():
        numeric=native[i]
        np.testing.assert_array_equal(numeric['action'].numpy(),full['action'].numpy())
        np.testing.assert_array_equal(numeric['action_is_pad'].numpy(),full['action_is_pad'].numpy())
        for camera in CAMERAS:
            assert tuple(full[f'observation.images.{camera}'].shape)==(3,contract['image_size'],contract['image_size'])
    wrapped=masked_module['MaskedActionDataset'](native,root)
    report = []
    checked_images = checked_starts = 0
    all_valid = []
    all_horizons=[]
    native_rows_checked=0
    occupied = {}
    source_states_seen=set()
    for ep in manifest['episodes']:
        index = ep['episode_index']
        directory = source/ep['source_id']
        metadata = json.loads((directory/'metadata.json').read_text())
        assert metadata['label'] == 'review'
        assert metadata['context']['mode']=='human_in_the_loop'
        for key in ('fps','image_size','task'):
            assert metadata['context']['contract'][key]==contract[key]
        normalization=contract['image_crop_normalization']
        assert normalization['approved'] is True and normalization['method']=='recrop_original_640x480'
        target_path=root/'processing/target_inference_contract.json'
        assert sha256(target_path)==normalization['target_contract_sha256']
        target_contract=json.loads(target_path.read_text())
        desired={'overhead':[120,80,520,480],'wrist_port2':[80,0,560,480],'wrist_port3':[80,0,560,480]}
        assert contract['image_crops_xyxy']==normalization['target_crops_xyxy']==target_contract['image_crops_xyxy']==desired
        assert metadata['context']['contract']['image_crops_xyxy'] in normalization['allowed_source_crops_xyxy']
        assert ep['source_image_crops_xyxy']==metadata['context']['contract']['image_crops_xyxy']
        assert ep['output_image_crops_xyxy']==desired
        assert ep['image_crop_normalization']==normalization
        assert ep['source_complete']==metadata['complete']
        assert ep['source_end_reason']==metadata.get('end_reason')
        decision=ep['visual_decision']
        assert decision in accepted and decision.get('evidence')
        assert decision['source_id']==ep['source_id']
        assert decision['recording_segment']==ep['recording_segment']
        assert decision['start_s']-1e-8 <= ep['start_s'] <= ep['end_s'] <= decision['end_s']+1e-8
        if metadata['complete'] is not True:
            allowlist={('20261028T183037-e26713b4',4),('20261028T153841-ae44a1c5',4),('20261028T185549-482189f4',9),('20261028T185829-c6fd1af5',0),('20261028T184300-6a14071f',0),('20261028T185407-3f48617a',0),('20261028T185549-482189f4',5),('20261028T185549-482189f4',27)}
            assert (ep['source_id'],ep['recording_segment']) in allowlist
            approval=decision['incomplete_interval_approved']
            provenance=root/'processing/incomplete_interval_audit.json'
            assert sha256(provenance)==approval['audit_sha256']
            audit=json.loads(provenance.read_text())
            assert any(a['source_id']==decision['source_id'] and a['recording_segment']==decision['recording_segment'] and a['start_s']-1e-8<=decision['start_s'] and a['end_s']+1e-8>=decision['end_s'] for a in audit)

        for name in ('metadata.json', 'states.jsonl', 'frames.jsonl'):
            assert sha256(directory/name) == ep['source_hashes'][name]
        original = read_rows(directory/'states.jsonl')
        accepted_rows=[r for r in original if decision['start_s']-1e-8<=r['data_time_s']<=decision['end_s']+1e-8]
        assert accepted_rows
        for clock in ('monotonic_ns','data_monotonic_ns'):
            assert all(b[clock]>a[clock] for a,b in zip(accepted_rows,accepted_rows[1:]))
        for arm in ('left','right'):
            previous=None
            for r in accepted_rows:
                a=r['arms'][arm]
                assert np.isfinite(precise_pose(r)).all()
                assert all(v==0 for v in a['motor_fault'])
                feedback=np.asarray(a['motor_feedback_time'])
                ages=r['unix_ns']/1e9-feedback
                assert np.isfinite(feedback).all() and ((ages>=0)&(ages<=.05)).all()
                if previous is not None: assert (feedback>=previous).all()
                previous=feedback
        if metadata['complete'] is not True:
            last=accepted_rows[-1]
            boundaries=[t for t in metadata['control_transitions'] if last['monotonic_ns']<t['monotonic_ns']<metadata['end_monotonic_ns'] and t['source']!='human']
            assert boundaries and boundaries[0]['monotonic_ns']<metadata['end_monotonic_ns']

        origin = data_origin(metadata)
        start = origin + round(ep['start_s']*1e9)
        end = origin + round(ep['end_s']*1e9)
        interval = [r for r in original if start-10 <= data_timestamp(r) <= end+10]
        assert interval and all(r['control_source']=='human' for r in interval)
        assert len({r['recording_segment'] for r in interval}) == 1
        times=[data_timestamp(r) for r in interval]
        assert all(0 < b-a <= 150_000_000 for a,b in zip(times,times[1:]))
        source_id = ep['source_id']
        for a,b in occupied.setdefault(source_id, []):
            assert end <= a or start >= b, 'Overlapping source intervals duplicate supervision'
        occupied[source_id].append((start,end))
        state_map = {r['sequence']:r for r in original}
        frames = read_rows(directory/'frames.jsonl')
        image_map = {(r['camera'],r['path']):r for r in frames}
        alignment = ep['alignment']
        keys={(source_id,a['state_sequence']) for a in alignment}
        assert not (keys & source_states_seen), 'Source measured observations repeated across episodes'
        source_states_seen.update(keys)
        exported = [r for r in rows if r['episode_index']==index]
        assert len(exported)==len(alignment)>0
        raw_actions=[]; actions=[]; states=[]
        camera_paths={camera:[] for camera in CAMERAS}
        for i,(row,a) in enumerate(zip(exported,alignment)):
            assert row['frame_index']==i
            assert abs(row['timestamp']-i/20)<1e-4
            state=state_map[a['state_sequence']]; target=state_map[a['action_sequence']]
            assert start-10_000_000 <= data_timestamp(state) < data_timestamp(target) <= end+10_000_000
            assert state['control_source']==target['control_source']=='human'
            for owned in (state,target):
                owners=owned['control_source_by_arm']
                assert set(owners)=={'left','right'} and 'human' in owners.values()
                assert all(v in ('human','hold') for v in owners.values())
            assert state['recording_segment']==target['recording_segment']==ep['recording_segment']
            assert origin+round(decision['start_s']*1e9)-10 <= data_timestamp(state) < data_timestamp(target) <= origin+round(decision['end_s']*1e9)+10
            np.testing.assert_array_equal(np.asarray(row['observation.state'],dtype=np.float32),vector(state))
            raw=vector(target); action=np.asarray(row['action'],dtype=np.float32)
            assert np.isfinite(action).all()
            np.testing.assert_array_equal(action[[6,13]],raw[[6,13]])
            np.testing.assert_array_equal(np.asarray(row['action.raw_next_state'],dtype=np.float32),raw)
            bound=manifest['smoothing']['max_deviation_rad']
            assert bound <= .008
            assert np.max(np.abs(action-raw)) <= bound+1e-6
            tick=origin+round(a['source_time_s']*1e9)
            assert abs(a['action_time_s']-a['source_time_s']-.05)<1e-8
            assert a['state_data_monotonic_ns']==data_timestamp(state)
            assert a['action_data_monotonic_ns']==data_timestamp(target)
            if i:
                assert abs(a['source_time_s']-alignment[i-1]['source_time_s']-.05)<1e-8
            assert abs(data_timestamp(state)-tick)/1e9 <= contract['max_state_alignment_s']
            assert abs(data_timestamp(target)-tick-50_000_000)/1e9 <= contract['max_state_alignment_s']
            for camera in CAMERAS:
                path=a['cameras'][camera]['path']; frame=image_map[camera,path]
                assert origin+round(decision['start_s']*1e9)-10 <= data_timestamp(frame) <= origin+round(decision['end_s']*1e9)+10
                assert abs((data_timestamp(frame)-tick)/1e9) <= contract['max_camera_alignment_s']
                assert abs(a['cameras'][camera]['offset_s']-(data_timestamp(frame)-tick)/1e9)<1e-8
                assert sha256(directory/path)==a['cameras'][camera]['source_sha256']
                image=prepare_image(directory/path,contract['image_crops_xyxy'][camera],contract['image_size'])
                buffer=io.BytesIO(); Image.fromarray(image).save(buffer,format='JPEG',quality=95,subsampling=0)
                assert row[f'observation.images.{camera}']['bytes']==buffer.getvalue()
                camera_paths[camera].append(path)
                checked_images+=1
            states.append(row['observation.state']); actions.append(action); raw_actions.append(raw)
        assert ep['repeated_camera_frames']=={c:len(v)-len(set(v)) for c,v in camera_paths.items()}
        # Independently reproduce the declared symmetric three-tap target filter.
        q=np.asarray([precise_pose(state_map[alignment[0]['state_sequence']])]+[precise_pose(state_map[a['action_sequence']]) for a in alignment],dtype=float)
        np.testing.assert_array_equal(np.asarray(states[1:],dtype=np.float32),np.asarray(raw_actions[:-1],dtype=np.float32))
        expected=q.copy(); joints=[j for j in range(14) if j not in (6,13)]
        expected[1:-1,joints]=.25*q[:-2,joints]+.5*q[1:-1,joints]+.25*q[2:,joints]
        np.testing.assert_array_equal(np.asarray(actions,dtype=np.float32),expected[1:].astype(np.float32))
        before=knot_metrics(q); after=knot_metrics(expected)
        for name,actual in (('before',before),('after',after)):
            for key,value in actual.items():
                assert np.isclose(ep[name][key],value,atol=1e-10,rtol=1e-9), (name,key,ep[name][key],value)
        deviation=float(np.max(abs(expected-q)))
        assert np.isclose(ep['max_smoothing_deviation_rad'],deviation,atol=1e-12)
        assert deviation <= .008
        assert manifest['quality_gates']['max_joint_speed_rad_s'] <= 1.0
        assert manifest['quality_gates']['peak_joint_acceleration_rad_s2'] <= 8.0
        assert after['max_joint_speed_rad_s'] <= manifest['quality_gates']['max_joint_speed_rad_s']
        assert after['peak_joint_acceleration_rad_s2'] <= manifest['quality_gates']['peak_joint_acceleration_rad_s2']
        assert len(exported)>=2
        for local_index,row in enumerate(exported):
            horizon=min(50,len(exported)-local_index)
            all_horizons.append(horizon)
            item=native[row['index']]
            expected_pad=np.arange(50)>=horizon
            np.testing.assert_array_equal(item['action_is_pad'].numpy(),expected_pad)
            targets=np.asarray([exported[min(local_index+k,len(exported)-1)]['action'] for k in range(50)],dtype=np.float32)
            np.testing.assert_array_equal(item['action'].numpy(),targets)
            masked=wrapped[row['index']]
            np.testing.assert_array_equal(masked['action_is_pad'].numpy(),expected_pad)
            np.testing.assert_array_equal(masked['action_valid_mask'].numpy(),~expected_pad)
            np.testing.assert_array_equal(masked['action'][:horizon].numpy(),targets[:horizon])
            assert (masked['action'][horizon:]==0).all().item()
            native_rows_checked+=1
        valid=[r['index'] for r in exported[:max(0,len(exported)-49)]]
        for global_index in valid:
            item=native[global_index]
            assert not item['action_is_pad'].any().item()
            np.testing.assert_array_equal(item['action'].numpy(),np.asarray([r['action'] for r in rows[global_index:global_index+50]],dtype=np.float32))
            checked_starts+=1
        assert ep['valid_starts']==len(valid)
        all_valid.extend(valid)
        report.append(dict(episode_id=f'{source_id}:{index}',valid_50_step_starts=len(valid),raw=trajectory_metrics(raw_actions,states,20),smoothed=trajectory_metrics(actions,states,20)))
        print(f'Checked episode {index+1}/{len(manifest["episodes"])}; {native_rows_checked} native rows, {checked_images} exact images',file=sys.stderr,flush=True)
    assert native_rows_checked==len(rows)
    assert saved_horizons['global_valid_lengths']==all_horizons
    assert saved_horizons['total_frames']==len(rows)
    # Adversarial objective: enormous padded errors must change neither loss nor gradient.
    import torch
    mask=torch.tensor([[True,True,False,False],[True,False,False,False]])
    prediction=torch.ones((2,4,14),requires_grad=True)
    prediction.data[~mask]=1e8
    objective=masked_module['masked_action_mse'](prediction,torch.zeros_like(prediction),mask)
    assert objective.item()==1.0
    objective.backward()
    assert (prediction.grad[~mask]==0).all().item()
    assert (prediction.grad[mask]!=0).all().item()
    for bad_loss,bad_mask in ((torch.ones(2,4,14),torch.zeros(2,4,dtype=torch.bool)),(torch.tensor(1.),mask)):
        try:
            masked_module['masked_action_mean'](bad_loss,bad_mask)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid mask/loss shape was accepted')
    # Exercise the shipped PI05 entry point with unreduced, adversarial padded loss.
    from types import SimpleNamespace
    errors=torch.ones((2,4,16),requires_grad=True)
    errors.data[~mask]=1e8
    fake=SimpleNamespace(config=SimpleNamespace(rtc_config=None,output_features={'action':SimpleNamespace(shape=(14,))}),
                         _preprocess_images=lambda batch:([],[]),prepare_action=lambda batch:batch['action'],
                         model=SimpleNamespace(forward=lambda *args:errors))
    fake_batch={'action_valid_mask':mask,'action':torch.zeros(2,4,14),
                'observation.language.tokens':torch.zeros(2,1,dtype=torch.long),
                'observation.language.attention_mask':torch.ones(2,1,dtype=torch.bool)}
    pi_loss,_=masked_module['masked_pi05_forward'](fake,fake_batch)
    assert pi_loss.item()==1.
    pi_loss.backward()
    assert (errors.grad[~mask]==0).all().item()
    assert (errors.grad[mask,:14]!=0).all().item()
    fake.config.rtc_config={}
    try:
        masked_module['masked_pi05_forward'](fake,fake_batch)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported RTC was accepted')
    assert saved_starts['total_frames']==len(rows)
    assert saved_starts['global_indices']==all_valid
    assert sorted(list(valid_start_sampler(root)))==all_valid
    packaged=runpy.run_path(str(root/'processing/valid_start_sampler.py'))
    assert sorted(list(packaged['valid_start_sampler'](root)))==all_valid
    result=dict(passed=True,checked_observations=len(rows),decoded_and_exact_source_images=checked_images,
                native_action_only_view_full_loader_parity_indices=parity_indices,native_media_checks='All exported images independently recreated from raw source; native action-only HF view retains original numeric rows and native query/padding logic, with full-loader action/mask parity at start/mid/end.',
                native_50_step_unpadded_starts=checked_starts,native_all_start_padding_masks_checked=native_rows_checked,episode_metrics=report,
                verified_artifacts_sha256={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and (p.suffix=='.parquet' or p.parent==root/'meta' or p.name in ('quality_manifest.json','export_contract.json','visual_decisions.json','valid_starts.json','valid_horizons.json','masked_training.py','valid_start_sampler.py','target_inference_contract.json','incomplete_interval_audit.json'))},
                visual_approval_scope='Matches saved accept decisions and evidence; verifier does not independently infer task success',
                training_or_robot_execution_performed=False)
    write_json(root/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'data/hitl/review')
    args=parser.parse_args()
    print(json.dumps(verify(args.root.resolve(),args.source.resolve()),indent=2))
