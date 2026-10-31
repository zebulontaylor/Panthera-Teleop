#!/usr/bin/env python3
"""Freeze and export a fresh, external HITL holdout without training use."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from PIL import Image
from state_value_prepare import CAMERAS, make_plan, encode_episode_samples

CATEGORIES=("kept","review","discarded","pending")

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_jsonl(path:Path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
def count_lines(path:Path):
 try:return sum(1 for line in path.open(encoding='utf8') if line.strip())
 except (OSError,UnicodeError):return None
def safe_write_json(path:Path,data:Any):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(temp,path)
def write_jsonl(path:Path,rows):
 temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w',encoding='utf8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 os.replace(temp,path)
def now_utc():return dt.datetime.now(dt.timezone.utc)
def parse_dt(value):
 if not value:return None
 try:return dt.datetime.fromisoformat(str(value).replace('Z','+00:00')).astimezone(dt.timezone.utc)
 except Exception:return None

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--source-root',type=Path,default=Path('data/hitl'))
 ap.add_argument('--output',type=Path,default=Path('work/state-value-20261029/fresh-hitl-holdout'))
 ap.add_argument('--date-prefix',default='20261029')
 ap.add_argument('--sample-hz',type=float,default=5.0)
 ap.add_argument('--max-gap-s',type=float,default=.15)
 ap.add_argument('--image-size',type=int,default=320)
 a=ap.parse_args(); source_root=a.source_root.resolve(); out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit(f'refusing to overwrite existing holdout: {out}')
 out.mkdir(parents=True,exist_ok=True); cutoff=now_utc(); cutoff_iso=cutoff.isoformat().replace('+00:00','Z')
 candidates=[]
 for category in CATEGORIES:
  folder=source_root/category
  if not folder.is_dir():continue
  for epdir in sorted(folder.iterdir()):
   if not epdir.is_dir() or not epdir.name.startswith(a.date_prefix):continue
   # Read metadata once; capture the initial inventory cutoff before any export work.
   meta_path=epdir/'metadata.json'; frames_path=epdir/'frames.jsonl'; states_path=epdir/'states.jsonl'
   try:meta=json.loads(meta_path.read_text(encoding='utf8'))
   except Exception as exc:meta={};metadata_error=str(exc)
   else:metadata_error=None
   created=parse_dt(meta.get('created_utc')); ended=parse_dt(meta.get('ended_utc'))
   file_times={}
   for f in (meta_path,frames_path,states_path):
    try:file_times[f.name]=dt.datetime.fromtimestamp(f.stat().st_mtime,dt.timezone.utc).isoformat().replace('+00:00','Z')
    except OSError:file_times[f.name]=None
   is_complete=meta.get('complete') is True and meta.get('id',epdir.name)==epdir.name and metadata_error is None
   reason=None
   if metadata_error:reason='metadata_unreadable'
   elif not is_complete:reason='active_or_incomplete_at_cutoff'
   elif created and created>cutoff:reason='created_after_cutoff'
   elif ended and ended>cutoff:reason='ended_after_cutoff'
   elif any(parse_dt(value) and parse_dt(value)>cutoff for value in file_times.values() if value):reason='source_file_modified_after_cutoff'
   frames_count=count_lines(frames_path);states_count=count_lines(states_path)
   frame_rows=[];state_rows=[]
   if reason is None:
    try:
     frame_rows=read_jsonl(frames_path);state_rows=read_jsonl(states_path)
    except Exception as exc:reason='frames_or_states_unreadable';metadata_error=str(exc)
   camera_counts=Counter(r.get('camera','unknown') for r in frame_rows)
   ownership=Counter()
   for row in state_rows:
    source=row.get('control_source')
    if source is not None:ownership[f'control_source:{source}']+=1
    by_arm=row.get('control_source_by_arm')
    if isinstance(by_arm,dict):
     for arm,value in by_arm.items():
      if value is not None:ownership[f'{arm}:{value}']+=1
    if row.get('recording_segment') is not None:ownership[f'recording_segment:{row["recording_segment"]}']+=1
   record={
    'episode_id':epdir.name,'source_category':category,'source_dir':str(epdir),'created_utc':meta.get('created_utc'),
    'ended_utc':meta.get('ended_utc'),'metadata_complete':meta.get('complete'),'metadata_label':meta.get('label'),
    'included_in_holdout':reason is None,'exclusion_reason':reason,'metadata_error':metadata_error,
    'metadata_sha256':sha(meta_path) if meta_path.is_file() else None,
    'frames_sha256':sha(frames_path) if frames_path.is_file() else None,
    'states_sha256':sha(states_path) if states_path.is_file() else None,
    'frames_jsonl_count':frames_count,'states_jsonl_count':states_count,
    'frames_parsed_count':len(frame_rows) if reason is None else None,'states_parsed_count':len(state_rows) if reason is None else None,
    'frame_counts_by_camera':dict(camera_counts),'state_control_ownership_counts':dict(ownership),
    'operator_outcome':meta.get('outcome'),'human_interventions':meta.get('human_interventions'),
    'handoff_events':meta.get('handoff_events'),'control_transitions':meta.get('control_transitions'),
    'data_duration_s':meta.get('data_duration_s'),'end_reason':meta.get('end_reason'),
    'source_file_mtimes_utc':file_times,
   }
   candidates.append((record,meta,frame_rows,state_rows))
 inventory={'schema_version':1,'dataset':'fresh HITL external holdout inventory','inventory_cutoff_utc':cutoff_iso,
  'date_prefix':a.date_prefix,'source_root':str(source_root),'category_dirs_scanned':list(CATEGORIES),
  'selection_policy':'Only directories present by cutoff with metadata.complete=true and metadata ID matching folder name; pending/incomplete/active excluded. No raw source changed.',
  'candidate_count':len(candidates),'included_count':sum(r['included_in_holdout'] for r,_,_,_ in candidates),
  'excluded_count':sum(not r['included_in_holdout'] for r,_,_,_ in candidates),
  'candidate_inventory':[r for r,_,_,_ in candidates],
  'holdout_role':'fresh_test external evaluation only; must not enter training, hyperparameter tuning, threshold selection, or model selection.'}
 safe_write_json(out/'inventory.json',inventory)
 eligible=[(r,m,fr,sr) for r,m,fr,sr in candidates if r['included_in_holdout']]
 samples=[];episode_records=[];global_stats=Counter();owner_counts=defaultdict(Counter);operator_outcomes=Counter();episode_counts=Counter()
 image_size=(a.image_size,int(a.image_size*3/4))
 for rec,meta,frame_rows,state_rows in eligible:
  eid=rec['episode_id'];before=(rec['metadata_sha256'],rec['frames_sha256'],rec['states_sha256'])
  duration=float(meta.get('data_duration_s') or 0)
  # HITL episode_time_s is process-clock-like, while data_time_s is the bounded
  # recorded trajectory clock. Use the latter for sampling, but retain original
  # times in manifest provenance and frame paths in each row.
  normalized=[]
  for row in frame_rows:
   copy=dict(row)
   if copy.get('data_time_s') is not None:copy['episode_time_s']=float(copy['data_time_s'])
   normalized.append(copy)
  planned,_,coverage=make_plan(normalized,duration,a.sample_hz,a.max_gap_s)
  ep={'episode_id':eid,'session_id':eid,'split':'fresh_test','source_category':rec['source_category'],
      'weak_demo':False,'raw_source_kind':'directory','source_dir':Path(rec['source_dir']),
      'episode_end_s':duration,'source_group':'human_in_the_loop','human_intervention_observed':
       (meta.get('outcome') or {}).get('human_assisted'),
      'task_attempt_observed':None}
  rows,stats=encode_episode_samples(ep,planned,state_rows,
      int(meta.get('start_data_monotonic_ns') or meta.get('trigger_monotonic_ns') or 0),
      out,image_size,a.max_gap_s)
  # All timing in model-facing samples uses data_time_s, including measured state.
  for row in rows:
   if isinstance(row.get('data_time_s'),(int,float)):row['state_time_s']=float(row['data_time_s'])
   row['external_holdout']=True
  # Only attach a terminal recorded-completion marker when operator success is
  # explicitly true. Do not infer autonomous success from a human-assisted run.
  outcome=meta.get('outcome') if isinstance(meta.get('outcome'),dict) else {}
  if rows and outcome.get('success') is True:
   terminal=rows[-1]
   terminal['terminal_recorded_success']=1
   terminal['terminal_success_provenance']={'source':'metadata.outcome.success','label':outcome.get('label'),
     'human_assisted':outcome.get('human_assisted'),'policy_success':outcome.get('policy_success'),
     'scope':'recorded behavior only; not autonomously attributable if human assistance occurred'}
   if outcome.get('human_assisted') is True:terminal['terminal_autonomous_success']=None
   elif isinstance(outcome.get('policy_success'),bool):terminal['terminal_autonomous_success']=int(outcome['policy_success'])
  # Preserve ownership event boundaries from metadata and per-state source fields.
  episode_record={**rec,'dataset_role':'fresh_test_external_only','session_id':eid,'split':'fresh_test',
   'source_group':(meta.get('context') or {}).get('mode'),'context':meta.get('context'),
   'raw_metadata_outcome':outcome,'human_interventions':meta.get('human_interventions'),
   'handoff_events':meta.get('handoff_events'),'control_transitions':meta.get('control_transitions'),
   'start_data_monotonic_ns':meta.get('start_data_monotonic_ns'),'start_monotonic_ns':meta.get('start_monotonic_ns'),
   'data_duration_s':duration,'end_reason':meta.get('end_reason'),'sampling_coverage':coverage,
   'sample_count':len(rows),'sample_hz':a.sample_hz,'camera_order':list(CAMERAS),
   'timing_basis':'frame data_time_s; state chosen by nearest host monotonic_ns to median three-camera timestamp'}
  after=(sha(Path(rec['source_dir'])/'metadata.json'),sha(Path(rec['source_dir'])/'frames.jsonl'),sha(Path(rec['source_dir'])/'states.jsonl'))
  if before!=after:raise RuntimeError(f'completed source changed during holdout export: {eid}')
  episode_record['source_hashes_reverified_after_export']=True
  samples.extend(rows);episode_records.append(episode_record);global_stats.update(stats)
  episode_counts[rec['source_category']]+=1
  if outcome.get('success') is True:operator_outcomes['explicit_success']+=1
  elif outcome.get('success') is False:operator_outcomes['explicit_not_success_or_review']+=1
  else:operator_outcomes['no_explicit_operator_success']+=1
  for row in rows:
   for field in ('human_control_active','human_contact_signal_available'):
    if row.get(field) is not None:owner_counts['per_sample'][f'{field}:{row[field]}']+=1
   if row.get('control_source') is not None:owner_counts['per_sample'][f'control_source:{row["control_source"]}']+=1
   if row.get('control_source_by_arm'):
    for arm,value in row['control_source_by_arm'].items():owner_counts['per_sample'][f'{arm}:{value}']+=1
   if row.get('recording_segment') is not None:owner_counts['per_sample'][f'recording_segment:{row["recording_segment"]}']+=1
 samples.sort(key=lambda r:(r['episode_id'],r['t']))
 episode_records.sort(key=lambda r:r['episode_id'])
 write_jsonl(out/'episodes.jsonl',episode_records);write_jsonl(out/'samples.jsonl',samples)
 # Inventory rows hold before/after source hashes and raw outcomes; episode manifest
 # embeds the same frozen provenance with sample coverage and output paths.
 audit={'schema_version':1,'dataset':'fresh HITL external holdout','inventory_cutoff_utc':cutoff_iso,
  'role':'fresh_test external evaluation only; excluded from all training and hyperparameter/model selection',
  'episodes':len(episode_records),'samples':len(samples),'split':'fresh_test','sample_hz':a.sample_hz,
  'image_size':list(image_size),'camera_order':list(CAMERAS),
  'state_order':[f'left_joint_{i+1}' for i in range(6)]+['left_gripper']+[f'right_joint_{i+1}' for i in range(6)]+['right_gripper'],
  'source_category_episode_counts':dict(episode_counts),'operator_success_summary':dict(operator_outcomes),
  'explicit_terminal_recorded_success_rows':sum(1 for r in samples if 'terminal_recorded_success' in r),
  'human_ownership_sample_counts':{k:dict(v) for k,v in owner_counts.items()},
  'source_annotation_semantics':'Raw metadata.outcome is preserved. label=review or success=false is not mapped to task failure. A positive terminal recorded-success marker is attached only for an explicit metadata.outcome.success=true; human-assisted examples are not autonomous-success labels.',
  'human_intervention_semantics':'Per-frame control_source/control_source_by_arm/recording_segment are retained; metadata handoff_events, control_transitions and human_interventions remain episode provenance. Do not treat control ownership as physical contact.',
  'terminal_label_scope':'Only explicit terminal recorded completion; no label propagation to earlier states.',
  'inventory_file_sha256':sha(out/'inventory.json'),'episodes_sha256':sha(out/'episodes.jsonl'),'samples_sha256':sha(out/'samples.jsonl'),
  'unique_referenced_images':len({p for r in samples for p in r['images']}),'stats':dict(global_stats),
  'no_training_or_tuning_use':True}
 safe_write_json(out/'audit.json',audit)
 print(json.dumps({'output':str(out),'cutoff_utc':cutoff_iso,'candidate_count':inventory['candidate_count'],'included_completed':len(eligible),
  'excluded':[(r['episode_id'],r['exclusion_reason']) for r,_,_,_ in candidates if not r['included_in_holdout']],
  'samples':len(samples),'operator_success_summary':dict(operator_outcomes),'human_assisted':sum(1 for _,m,_,_ in eligible if (m.get('outcome') or {}).get('human_assisted') is True),
  'row_counts_by_episode':{e['episode_id']:e['sample_count'] for e in episode_records}},indent=2))
if __name__=='__main__':main()
