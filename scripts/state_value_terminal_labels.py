#!/usr/bin/env python3
"""Attach sparse, explicitly reviewed terminal HF failure supervision only."""
import argparse, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();root=a.dataset
 ep=read(root/'episodes.jsonl');rows=read(root/'samples.jsonl');idx=defaultdict(list)
 for r in rows:idx[r['episode_id']].append(r)
 for r in rows:
  r.pop('terminal_outcome',None);r.pop('terminal_outcome_confidence',None);r.pop('terminal_outcome_provenance',None)
 conf={'high':.9,'medium':.7,'low':.5};counts=Counter();excluded=Counter()
 for e in ep:
  if e.get('dataset_cohort')!='hf_new_discarded_161' or e.get('outcome')!='failed':continue
  if e.get('subtype')=='gear_drop_then_human_recovery':excluded[e['split']]+=1;continue
  duration=float(e.get('episode_end_s') or 0); evidence=[float(t) for t in e.get('evidence_times_s',[]) if isinstance(t,(int,float))]
  candidates=idx.get(e['episode_id'],[])
  last=next((r for r in reversed(candidates) if r.get('terminal')),None)
  if last is None or not evidence or duration-max(evidence)>.5:continue
  last['terminal_outcome']=0;last['terminal_outcome_confidence']=conf.get(str(e.get('confidence','')).lower(),.5)
  last['terminal_outcome_provenance']={'source':'reviewed_hf_episode_annotation','annotation_outcome':e.get('outcome'),'annotation_confidence':e.get('confidence'),'subtype':e.get('subtype'),'reviewer_model':e.get('reviewer_model'),'review_method':e.get('review_method'),'rationale':e.get('rationale'),'evidence_times_s':evidence,'evidence_to_recording_end_s':round(duration-max(evidence),6),'scope':'recorded episode terminal observation only; not policy-only failure'}
  counts[e['split']]+=1
 target=root/'samples.jsonl';tmp=target.with_suffix('.jsonl.tmp')
 with tmp.open('w',encoding='utf8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 os.replace(tmp,target)
 auditpath=root/'audit.json';audit=json.loads(auditpath.read_text());audit['samples_sha256']=sha(target)
 audit['label_semantics']='No dense frame-level outcome/reward is assigned. source_category and weak_demo are curation provenance only. Sparse terminal_outcome labels are separately attached only where reviewed terminal failure evidence qualifies.'
 audit.pop('target_field_present',None);audit['dense_target_field_present']=False;audit['sparse_terminal_outcome_present']=True
 audit['terminal_supervision']={'definition':'Sparse recorded-behavior failure at the exact terminal observation, only for reviewed HF failed annotations with reviewed evidence within 0.5 s of recording end; no frame propagation.','negative_rows_by_split':dict(counts),'positive_rows_by_split':{},'excluded_recovery_subtypes':['gear_drop_then_human_recovery'],'excluded_recovery_episodes_by_split':dict(excluded),'apparent_success_labels_used':False}
 audit['terminal_supervision_rows']=sum(counts.values());tmp=auditpath.with_suffix('.json.tmp');tmp.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,auditpath)
 print(json.dumps({'sparse_terminal_negatives_by_split':dict(counts),'recovery_excluded_by_split':dict(excluded),'sample_rows':len(rows),'samples_sha256':audit['samples_sha256']},indent=2))
if __name__=='__main__':main()
