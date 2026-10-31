#!/usr/bin/env python3
"""Export candidate before/after human-ownership observations, never raw edits."""
from __future__ import annotations
import argparse, bisect, collections, hashlib, json, math, shutil
from pathlib import Path

CAMS=('overhead','wrist_port2','wrist_port3')
SPLIT_MAP='work/state-value-v2-20261029/split-map.json'

def read(path):
 with open(path,encoding='utf8') as f:
  for s in f:
   if s.strip(): yield json.loads(s)
def arm_is_human(row):
 return row.get('control_source')=='human' or any(v=='human' for v in (row.get('control_source_by_arm') or {}).values())
def state_vector(row):
 arms=row.get('arms') or {};out=[]
 for side in ('left','right'):
  arm=arms.get(side) or {}; pos=arm.get('position_rad') or []
  grip=(arm.get('gripper') or {}).get('position')
  if len(pos)!=6 or not isinstance(grip,(int,float)):return None
  out.extend(float(x) for x in pos);out.append(float(grip))
 return out
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def nearest(frames,ts,t,lower=None,upper=None):
 lo=0 if lower is None else bisect.bisect_left(ts,lower)
 hi=len(ts) if upper is None else bisect.bisect_left(ts,upper)
 if hi<=lo:return None
 i=bisect.bisect_left(ts,t);cand=[j for j in (i-1,i,lo,hi-1) if lo<=j<hi]
 return min(set(cand),key=lambda j:abs(ts[j]-t)) if cand else None
def canonical_session(eid):
 if eid.startswith('20261027'):return '20261027'
 if eid.startswith('20261028'):
  try: hh=int(eid[9:11]);mm=int(eid[11:13]);return f'{eid[:8]}T{hh:02d}:{30 if mm>=30 else 0:02d}'
  except: return eid[:8]
 return eid[:8]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('data/hitl'));ap.add_argument('--out',type=Path,default=Path('work/state-value-v2-20261029/hitl-pair-candidates'));ap.add_argument('--min-duration-s',type=float,default=.25);ap.add_argument('--min-state-delta-rad',type=float,default=.01);ap.add_argument('--max-camera-gap-s',type=float,default=.20);ap.add_argument('--postrelease-pose-tolerance-rad',type=float,default=.01);ap.add_argument('--postrelease-endpoint',choices=('stable_hold','first_nonhuman'),default='stable_hold',help='Endpoint selection policy; first_nonhuman uses the first policy state within 0.5 s after the first post-release policy transition and records its pose drift.');ap.add_argument('--evaluation-holdout',type=Path,help='Optional frozen evaluation ID list or samples.jsonl. Restricts export to listed episode IDs; must use an evaluation-only output path.');ap.add_argument('--evaluation-split',choices=('fresh_test','test'),default='fresh_test',help='Output split for --evaluation-holdout.');a=ap.parse_args()
 out=a.out.resolve(); imgroot=out/'frames';imgroot.mkdir(parents=True,exist_ok=True)
 evaluation_ids=None
 if a.evaluation_holdout:
  evaluation_ids={r['episode_id'] for r in read(a.evaluation_holdout)}
  if not evaluation_ids: ap.error('--evaluation-holdout contains no episode IDs')
  if 'evaluation' not in str(out).lower() or 'intent-pairs' not in str(out): ap.error('evaluation exports must use a separate evaluation/*intent-pairs* output path')
  id_split={eid:a.evaluation_split for eid in evaluation_ids}
 else:
  split_data=json.loads(Path(SPLIT_MAP).read_text());id_split={eid:v['split'] for session,v in split_data['session_splits'].items() for eid in v['episode_ids']}
 candidates=[];reasons=collections.Counter();by_split=collections.Counter();state_total=0;episodes_seen=0
 for category in ('kept','review','discarded','pending'):
  for folder in sorted((a.root/category).glob('*')):
   if not folder.is_dir():continue
   mp=folder/'metadata.json';fp=folder/'frames.jsonl';sp=folder/'states.jsonl'
   if not (mp.exists() and fp.exists() and sp.exists()):reasons['missing_files']+=1;continue
   try:meta=json.loads(mp.read_text())
   except Exception:reasons['bad_metadata']+=1;continue
   eid=str(meta.get('id') or folder.name);episodes_seen+=1
   session=canonical_session(eid);split=id_split.get(eid)
   if evaluation_ids is not None:
    if eid not in evaluation_ids:reasons['not_in_frozen_evaluation_holdout']+=1;continue
   elif eid.startswith('20261029') or split=='excluded_fresh_day':reasons['frozen_or_oct29']+=1;continue
   if split not in ((a.evaluation_split,) if evaluation_ids is not None else ('train','val')):reasons['not_known_train_val']+=1;continue
   if meta.get('complete') is not True:reasons['incomplete_reviewed_for_segments']+=1
   states=sorted(read(sp),key=lambda r:int(r.get('monotonic_ns') or 0));frames=defaultdict(list)
   for r in read(fp):
    if r.get('camera') in CAMS and isinstance(r.get('monotonic_ns'),int):frames[r['camera']].append(r)
   for c in CAMS:frames[c].sort(key=lambda r:r['monotonic_ns'])
   if any(not frames[c] for c in CAMS):reasons['camera_stream_missing']+=1;continue
   fts={c:[int(x['monotonic_ns']) for x in frames[c]] for c in CAMS}; sts=[int(r.get('monotonic_ns') or 0) for r in states]
   trans=sorted((x for x in (meta.get('control_transitions') or []) if isinstance(x.get('monotonic_ns'),int)),key=lambda x:x['monotonic_ns'])
   starts=[(i,x) for i,x in enumerate(trans) if x.get('source')=='human']
   if not starts:reasons['no_human_transition']+=1;continue
   for human_idx,start in starts:
    start_ns=int(start['monotonic_ns']); release=next((x for x in trans[human_idx+1:] if x.get('source')!='human'),None)
    if release is None:reasons['no_release_event']+=1;continue
    end_ns=int(release['monotonic_ns']); interval=[r for r in states if start_ns<=int(r.get('monotonic_ns') or 0)<end_ns and arm_is_human(r)]
    if not interval:reasons['no_owned_states']+=1;continue
    pre_i=bisect.bisect_left(sts,start_ns)-1
    if pre_i<0:reasons['no_pre_state']+=1;continue
    pre=states[pre_i];human_end=interval[-1];t_pre=int(pre['monotonic_ns']);t_human_end=int(human_end['monotonic_ns']);duration=(t_human_end-start_ns)/1e9
    if duration<a.min_duration_s:reasons['short_human_segment']+=1;continue
    va=state_vector(pre);human_vec=state_vector(human_end)
    if va is None or human_vec is None:reasons['invalid_state']+=1;continue
    diffs=[abs(y-x) for x,y in zip(va,human_vec)];max_delta=max(diffs);rms=math.sqrt(sum(d*d for d in diffs)/len(diffs))
    if max_delta<a.min_state_delta_rad:reasons['no_measurable_motion']+=1;continue
    # Prefer a post-release state to reduce human-hand appearance leakage.
    release_i=next((j for j,x in enumerate(trans) if x is release),None)
    later_trans=trans[(release_i+1 if release_i is not None else len(trans)):]
    next_human=next((x for x in later_trans if x.get('source')=='human'),None)
    post_window_start=end_ns
    policy_start=next((x for x in later_trans if x.get('source')=='policy'),None)
    if a.postrelease_endpoint=='first_nonhuman' and policy_start is not None:post_window_start=int(policy_start['monotonic_ns'])
    post_window_end=min(post_window_start+500_000_000,int(next_human['monotonic_ns']) if next_human else (sts[-1]+1 if sts else post_window_start))
    hold_end_ns=post_window_end
    hold_candidates=[]
    if a.postrelease_endpoint=='first_nonhuman':
     for hrow in states:
      htime=int(hrow.get('monotonic_ns') or 0)
      if not post_window_start<=htime<=post_window_end or arm_is_human(hrow):continue
      if hrow.get('control_source')!='policy':continue
      delta=state_vector(hrow)
      if delta is None:continue
      drift=max(abs(y-x) for x,y in zip(human_vec,delta))
      hold_candidates.append((htime,hrow,drift))
    else:
     for hrow in states:
      htime=int(hrow.get('monotonic_ns') or 0)
      if not end_ns<=htime<post_window_end or hrow.get('control_source')!='hold':continue
      delta=state_vector(hrow)
      if delta is None:continue
      drift=max(abs(y-x) for x,y in zip(human_vec,delta));rel=(htime-end_ns)/1e9
      if .30<=rel<=.50 and drift<=a.postrelease_pose_tolerance_rad:hold_candidates.append((abs(rel-.40),hrow,drift))
    hold_selected=min(hold_candidates,key=lambda x:x[0]) if a.postrelease_endpoint=='stable_hold' and hold_candidates else (min(hold_candidates,key=lambda x:x[0]) if hold_candidates else None)
    if a.postrelease_endpoint=='first_nonhuman' and hold_selected is None:
     reasons['no_postrelease_policy_endpoint']+=1;continue
    post=(hold_selected[1] if hold_selected else human_end);vb=state_vector(post);t_post=int(post['monotonic_ns'])
    if vb is None:reasons['invalid_state']+=1;continue
    used_postrelease=hold_selected is not None;hold_drift=hold_selected[2] if hold_selected else None
    selected={}; failed=False
    for cam in CAMS:
     camrefs=[]
     targets=[('a',t_pre,None,start_ns)]
     if used_postrelease:targets.append(('human_endpoint',t_human_end,start_ns,end_ns))
     targets.append(('b',t_post,post_window_start,hold_end_ns) if used_postrelease else ('b',t_post,start_ns,end_ns))
     for tag,target,lower,upper in targets:
      ix=nearest(frames[cam],fts[cam],target,lower,upper)
      if ix is None:failed=True;break
      fr=frames[cam][ix];gap=abs(int(fr['monotonic_ns'])-target)/1e9
      src=folder/fr['path']
      if gap>a.max_camera_gap_s or not src.is_file():failed=True;break
      rel=f'frames/{eid}/human_{human_idx+1:03d}/{tag}/{cam}.jpg';dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
      shutil.copy2(src,dest)
      camrefs.append({'image':rel,'source_path':fr['path'],'frame_index':fr.get('frame_index'),'episode_time_s':fr.get('episode_time_s'),'data_time_s':fr.get('data_time_s'),'monotonic_ns':fr['monotonic_ns'],'offset_from_state_s':(int(fr['monotonic_ns'])-target)/1e9,'source_frame_sha256':sha(src)})
     if failed:break
     selected[cam]={ref['image'].rsplit('/',2)[-2]:ref for ref in camrefs}
    if failed:reasons['camera_gap_or_missing']+=1;continue
    pair_id=f'{eid}_human_{human_idx+1:03d}'
    start_rel=(start_ns-int(meta.get('start_monotonic_ns') or start_ns))/1e9;pre_t=float(pre.get('episode_time_s') if pre.get('episode_time_s') is not None else (t_pre-int(meta.get('start_monotonic_ns') or t_pre))/1e9);post_t=float(post.get('episode_time_s') if post.get('episode_time_s') is not None else (t_post-int(meta.get('start_monotonic_ns') or t_post))/1e9);human_end_t=float(human_end.get('episode_time_s') if human_end.get('episode_time_s') is not None else (t_human_end-int(meta.get('start_monotonic_ns') or t_human_end))/1e9)
    arm_modes={side:any((r.get('control_source_by_arm') or {}).get(side)=='human' for r in interval) for side in ('left','right')}
    candidates.append({'pair_id':pair_id,'episode_id':eid,'session_id':session,'split':split,'observation_a':{'current':{cam:selected[cam]['a']['image'] for cam in CAMS},'state':va},'observation_b':{'current':{cam:selected[cam]['b']['image'] for cam in CAMS},'state':vb},'time_a_s':pre_t,'time_b_s':post_t,'time_reference':'episode_time_s / host monotonic elapsed relative to metadata.start_monotonic_ns','source_state_timing_a':{'monotonic_ns':t_pre,'data_time_s':pre.get('data_time_s'),'episode_time_s':pre.get('episode_time_s')},'source_state_timing_b':{'monotonic_ns':t_post,'data_time_s':post.get('data_time_s'),'episode_time_s':post.get('episode_time_s')},'human_endpoint_timing':{'monotonic_ns':t_human_end,'data_time_s':human_end.get('data_time_s'),'episode_time_s':human_end.get('episode_time_s')},'selected_b_phase':('first_policy_state_after_release' if a.postrelease_endpoint=='first_nonhuman' else 'post_release_pose_stable_hold') if used_postrelease else 'last_human_owned_state_fallback','human_visible_risk':True,'pixel_hand_presence_verified_absent':False,'hand_presence_status':'unverified_pending_visual_review','postrelease_state_drift_max_rad':hold_drift,'postrelease_endpoint_policy':a.postrelease_endpoint,'postrelease_pose_tolerance_rad':a.postrelease_pose_tolerance_rad,'postrelease_delay_s':(t_post-end_ns)/1e9 if used_postrelease else None,'policy_start_delay_after_release_s':(post_window_start-end_ns)/1e9 if used_postrelease else None,'time_human_endpoint_s':human_end_t,'time_human_start_s':start_rel,'time_human_end_s':(end_ns-int(meta.get('start_monotonic_ns') or end_ns))/1e9,'human_segment_duration_s':duration,'human_arms':arm_modes,'max_abs_state_delta_rad':max_delta,'rms_state_delta_rad':rms,'pre_to_human_endpoint_state_delta_rad':[y-x for x,y in zip(va,human_vec)],'pre_to_selected_b_state_delta_rad':[y-x for x,y in zip(va,vb)],'candidate_preference':'b_better_weak_intent','confidence':0.5,'review_status':'pending_visual_review','label_source':'user_workflow_weak_intent_prior','rationale':'User reports human intervention is triggered when policy is expected to fail; measurable motion during human ownership toward task targets is weak-intent improvement evidence. This pair is not a verified task-success label. Reviewer must exclude visibly known failed states and confirm direction of meaningful progress. B is the first policy state in the first 0.5 s of the post-release policy segment, or a recorded nonhuman post-release state in stable_hold mode; exact release-to-state delay and 14D pose drift from the human endpoint are recorded. Pixel-level hand absence is unverified and requires visual review.','episode_metadata_complete':meta.get('complete') is True,'source_metadata_units':meta.get('units'),'source_metadata_sha256':sha(mp),'source_frames_jsonl_sha256':sha(fp),'source_states_jsonl_sha256':sha(sp),'transition_start':start,'transition_release':release,'source_frame_provenance':selected})
    reasons['candidate_pairs']+=1;by_split[split]+=1
   state_total+=len(states)
 out.mkdir(parents=True,exist_ok=True)
 with open(out/'pairs_candidates.jsonl','w') as f:
  for r in candidates:f.write(json.dumps(r,separators=(',',':'))+'\n')
 provenance=[{'pair_id':r['pair_id'],'episode_id':r['episode_id'],'source_category':next((c for c in ('kept','review','discarded','pending') if (a.root/c/r['episode_id']/'metadata.json').exists()),None),'source_episode_dir':str(next((a.root/c/r['episode_id'] for c in ('kept','review','discarded','pending') if (a.root/c/r['episode_id']/'metadata.json').exists()),Path('unknown')).resolve()),'source_metadata_sha256':r['source_metadata_sha256'],'source_frames_jsonl_sha256':r['source_frames_jsonl_sha256'],'source_states_jsonl_sha256':r['source_states_jsonl_sha256']} for r in candidates]
 (out/'source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 summary={'status':'evaluation_only_pending_visual_review' if evaluation_ids is not None else 'candidate_manifest_pending_visual_review','evaluation_only':evaluation_ids is not None,'evaluation_holdout_samples':str(a.evaluation_holdout.resolve()) if a.evaluation_holdout else None,'evaluation_split':a.evaluation_split if evaluation_ids is not None else None,'pairs':len(candidates),'episodes':len({r['episode_id'] for r in candidates}),'pairs_by_split':dict(by_split),'episodes_scanned':episodes_seen,'source_roots':[str(a.root.resolve())],'excluded_ids_frozen_holdout':'In evaluation mode, only IDs in the provided frozen holdout list are included and all rows use the explicitly selected evaluation split; this artifact is evaluation-only and not for training/model selection. Otherwise, split mapping loaded from work/state-value-v2-20261029/split-map.json; all test sessions and Oct29 records excluded. Incomplete records eligible only when the individual human segment has a recorded release and synchronized endpoint frames.','filter':{'complete_required_for_episode_outcome':False,'episode_must_have_recorded_human_segment_with_release':True,'eligible_splits':[a.evaluation_split] if evaluation_ids is not None else ['train','val'],'min_human_owned_duration_s':a.min_duration_s,'min_max_abs_measured_state_delta_rad':a.min_state_delta_rad,'require_three_cameras':list(CAMS),'max_nearest_camera_gap_s':a.max_camera_gap_s,'postrelease_endpoint_policy':a.postrelease_endpoint,'postrelease_state_search_window_s':{'seconds':0.5,'reference':'first subsequent policy transition'} if a.postrelease_endpoint=='first_nonhuman' else {'seconds':[0.30,0.50],'reference':'human release'},'postrelease_pose_tolerance_rad':a.postrelease_pose_tolerance_rad,'image_alignment':'nearest raw camera frame by host monotonic_ns; A frames before human takeover; in first_nonhuman mode B is strictly the first sampled policy state within 0.5s after the first post-release policy transition and before any new human transition; pairs without this B endpoint are excluded. The delay from human release and 14D drift from the final human state are recorded. Frames are constrained to the policy segment and exact offsets retained. Pixel-level hand absence remains unverified and requires visual review.'},'candidate_target':'b_better_weak_intent, confidence 0.5 from user workflow explanation; pending visual review, not task success. Position and gripper dimensions are radians per source metadata; filter uses maximum absolute measured 14D joint/gripper displacement, not a success criterion.','counts_by_filter_reason':dict(reasons),'outputs':{'manifest':str((out/'pairs_candidates.jsonl').resolve()),'copied_images_root':str(imgroot),'source_provenance_sidecar':str((out/'source_provenance.json').resolve())}}
 (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 from collections import defaultdict
 main()
