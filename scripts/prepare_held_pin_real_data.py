#!/usr/bin/env python3
"""Freeze conservative LEFT held-pin identity audit; never project RIGHT/FK labels.

Run with work/pin-insertion-cnn/runtime/bin/python. The resulting candidates
are excluded from supervision until a direct distal-endpoint review is available.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/held-pin-cnn'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    frames=[]
    source=ROOT/'work/pin-insertion-newdemo/manifest.json'
    original=json.loads(source.read_text())
    for r in original['frames']:
        p=Path(r['image_path'].replace('wrist_port2','wrist_port3'))
        if not p.exists(): continue
        frames.append(dict(id='left_'+r['id'],image_path=str(p),source_image_path=str(p),image_sha256=sha(p),camera='left_wrist',camera_source='wrist_port3',width=640,height=480,source_recording_id=r['source_recording_id'],source_frame_index=r['frame_index'],source_timestamp_basis='Counterpart frame index selected from RIGHT candidate manifest; LEFT timestamp not independently resolved',time_s=r.get('time_s'),timestamp_is_exact=False,group_id=r['episode_id'],split=r['split'],tip_px=None,visible=False,visibility_supervision_valid=False,localization_supervision_valid=False,training_eligible=False,annotation_uncertainty_px=None,annotation_status='identity_unconfirmed_exclude',notes='AI visual review, not human certified. Jaw-tip circular feature may be proximal pin cap; no confidently identified physical distal free endpoint visible in LEFT. Do not use this row as a visibility-negative or label the cap. RIGHT coordinates and FK were not transferred.'))
    older=OUT/'older-candidates.json'
    if older.exists():
        for i,(image,video,idx) in enumerate(json.loads(older.read_text())):
            p=ROOT/image;v=ROOT/video;episode=v.parent.name
            metadata=json.loads((v.parent/'metadata.json').read_text())
            anchor=metadata.get('anchor',[])
            group=hashlib.sha256(json.dumps(anchor).encode()).hexdigest()[:12]
            samples=[json.loads(s) for s in (v.parent/'samples.jsonl').read_text().splitlines()]
            matches=[s for s in samples if s.get('video_frame')==idx]
            sample=matches[0] if matches else {}
            cam=sample.get('cameras',{}).get('left_wrist',{})
            frames.append(dict(id=f'left_{episode}_{idx}',image_path=str(p),image_sha256=sha(p),source_video_path=str(v),source_video_sha256=sha(v),source_frame_index=idx,source_recording_id=episode,source_monotonic_ns=cam.get('monotonic_ns'),source_unix_ns=cam.get('unix_ns'),camera='left_wrist',camera_source='wrist_port3',width=640,height=480,group_id='anchor_'+group,split=['train','validation','test'][int(group,16)%3],tip_px=None,visible=False,visibility_supervision_valid=False,localization_supervision_valid=False,training_eligible=False,annotation_uncertainty_px=None,annotation_status='identity_unconfirmed_exclude',notes='AI visual review, not human certified. Circular jaw-tip cap/ring is not a confirmed distal endpoint. Paired RIGHT frames were inspected for identity only, without coordinate transfer. Excluded entirely, not a visibility-negative.'))
    manifest=dict(schema_version=1,purpose='Frozen real LEFT distal held-pin audit; no usable localization supervision established',annotation_method='Direct RGB visual review only; no FK/grasp projection, no RIGHT coordinate reuse',source_candidate_manifest=str(source),source_candidate_manifest_sha256=sha(source),frames=frames)
    (OUT/'real-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit=dict(total_candidates=len(frames),reviewed_contact_sheets=['newdemo-left-review.jpg','older-left-review.jpg'],paired_identity_evidence='paired-identity.jpg',visible_confirmed_distal_endpoints=0,training_eligible=0,excluded_identity_unconfirmed=len(frames),annotation_certification='AI reviewed, not human certified',limitations=['Candidate selection is not exhaustive: 62 newdemo counterparts and 48 frames from 12 of 278 older episodes.','LEFT camera observes proximal cap/jaw-tip region; no confidently identified distal free endpoint found.','Do not equate null coordinates with physical occlusion or train a visibility head on these excluded rows.','Newdemo LEFT frame timestamp is not independently resolved; same-index RIGHT time is candidate context only.','More direct distal-visible LEFT acquisitions or physical identity review are required before training a real LEFT tip model.'])
    (OUT/'real-data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

# Approved TOP objective is a separate artifact, never a mutation of distal audit.
def prepare_approved_top():
    import statistics
    body=json.loads((OUT/'body-candidate-manifest.json').read_text())
    extra=json.loads((OUT/'top-reviewed-coordinates.json').read_text())
    source=ROOT/'data/hitl/review/20261104T193211-4db4b17f'
    source_frames={r['frame_index']:r for r in map(json.loads,(source/'frames.jsonl').read_text().splitlines()) if r['camera']=='wrist_port3'}
    rows=[]
    labels={r['source_frame_index']:r['cap_center_px'] for r in body['frames']}
    labels.update({int(n):xy for n,xy in extra.items()})
    for n,xy in sorted(labels.items()):
        f=source_frames[n];block='B1' if n<=266 else 'B2' if n<=481 else 'B3';split='validation' if block=='B2' else 'train';p=source/f['path']
        rows.append(dict(id=f'left_top_20261104T193211_{n:07d}',image_path=str(p),image_sha256=sha(p),source_recording_id=source.name,source_frame_index=n,source_monotonic_ns=f['monotonic_ns'],source_unix_ns=f['unix_ns'],time_s=f['episode_time_s'],camera='left_wrist',camera_source='wrist_port3',width=640,height=480,group_id=source.name+'::'+block,block_id=block,split=split,top_px=xy,top_y_px=xy[1],top_visible=True,top_supervision_valid=True,bottom_px=None,bottom_y_px=None,bottom_supervision_valid=False,tip_px=None,visibility_supervision_valid=False,training_eligible=True,annotation_uncertainty_px=3,annotation_status='approved_top_visible_cap_face_center',shaft_axis_px=None,notes='Approved TOP landmark: directly visible proximal cap-face CENTER, not upper silhouette edge and never distal bottom. Only top Y is supervised. X records visual annotation provenance, not an inferred output. Physical distal BOTTOM remains unobservable and unsupervised. AI visual annotation, not human certified.'))
    med=statistics.median(r['top_px'][0] for r in rows)
    result=dict(schema_version=1,frozen=True,objective='Approved TOP cap-face center Y only; BOTTOM inferred downstream without real GT',landmark_definition='TOP is visible proximal cap-face center; NOT uppermost silhouette point. BOTTOM is physical distal endpoint and has no real LEFT labels.',image_size=[640,480],axis_x_fraction=med/640,axis_x_px=med,axis_x_fraction_convention='native x / width, same native-pixel normalization as model; use fixed visual gripper centerline, not network x',axis_x_provenance={'method':'Median manually visually reviewed proximal cap center x across 60 approved real LEFT frames','candidate_expected_centerline_px':333,'observed_median_px':med,'uncertainty_px':3,'not_network_prediction':True},split_unit='fixed B1/B3 train, B2 validation from preexisting frozen source split; one recording/grasp',counts={'total':len(rows),'train':sum(r['split']=='train' for r in rows),'validation':sum(r['split']=='validation' for r in rows),'test':0,'real_bottom_labels':0},limitations=['No independent real TOP test set: B4/B5 do not expose clear cap face.','One recording/grasp; added nearby frames improve visual coverage, not independent sample count.','TOP y spans only a small range near jaw tips; no real BOTTOM Y accuracy can be measured.','No shaft axis labels; visible jaw edges are not pin shaft evidence.'],source_frames_path=str(source/'frames.jsonl'),source_frames_sha256=sha(source/'frames.jsonl'),frames=rows)
    (OUT/'top-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts']))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-top', action='store_true', help='Build approved TOP-only manifest; preserve distal audit')
    args = parser.parse_args()
    prepare_approved_top() if args.approved_top else main()
