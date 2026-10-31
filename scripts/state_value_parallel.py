#!/usr/bin/env python3
"""Parallel 5 Hz trajectory export wrapper; runs independent ID shards then merges."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "scripts/state_value_prepare.py"
CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
NEW_DATE_SPLITS = {
    "20260930": "train", "20261002": "train", "20261006": "train",
    "20261007": "val", "20261008": "val", "20261005": "test", "20261011": "test",
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def read_jsonl(path:Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--old-manifest",type=Path,default=ROOT/"work/state-critic-20261028/episodes.json")
    p.add_argument("--old-samples",type=Path,default=ROOT/"work/state-critic-20261028/dataset/samples.jsonl")
    p.add_argument("--hf-root",type=Path,default=ROOT/"work/state-critic-20261029/hf-new-discarded")
    p.add_argument("--output",type=Path,default=ROOT/"work/state-value-20261029/full_trajectory")
    p.add_argument("--sample-hz",type=float,default=5.0)
    p.add_argument("--max-gap-s",type=float,default=.15)
    p.add_argument("--image-size",type=int,default=320)
    a=p.parse_args(); output=a.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    if any((output/x).exists() for x in ("episodes.jsonl","samples.jsonl","audit.json")):
        raise SystemExit(f"refusing to overwrite existing complete export: {output}")
    old=json.loads(a.old_manifest.read_text()); annotations=read_jsonl(a.hf_root/"episodes.jsonl")
    ids=sorted([x["episode_id"] for x in old["episodes"]]+[x["episode_id"] for x in annotations])
    if len(ids)!=527 or len(set(ids))!=527: raise ValueError("expected 527 unique source episodes")
    n=max(1,min(a.workers,len(ids))); chunks=[ids[i::n] for i in range(n)]
    # Put chunks beside the final dataset so moving completed image trees is atomic.
    temp=Path(tempfile.mkdtemp(prefix=".state-value-chunks-",dir=output.parent))
    def run(item):
        i,chunk=item; part=temp/f"part-{i:02d}"
        cmd=[sys.executable,str(PREP),"--old-manifest",str(a.old_manifest.resolve()),
             "--old-samples",str(a.old_samples.resolve()),"--hf-root",str(a.hf_root.resolve()),
             "--output",str(part),"--sample-hz",str(a.sample_hz),"--max-gap-s",str(a.max_gap_s),
             "--image-size",str(a.image_size),"--episode-ids",*chunk]
        result=subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError(f"chunk {i} failed ({result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        print(f"completed worker chunk {i+1}/{n} ({len(chunk)} episodes)",flush=True)
        return part
    parts=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures=[pool.submit(run,(i,c)) for i,c in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures): parts.append(future.result())
        samples=[]; episodes=[]; stats=Counter()
        for part in sorted(parts):
            samples.extend(read_jsonl(part/"samples.jsonl")); episodes.extend(read_jsonl(part/"episodes.jsonl"))
            stats.update(json.loads((part/"audit.json").read_text()).get("stats",{}))
            # Each worker owns disjoint episode IDs; move only its generated image trees.
            src=part/"images"; dst=output/"images"
            if src.exists():
                for tree in src.iterdir():
                    target=dst/tree.name
                    if target.exists(): raise ValueError(f"duplicate output image tree: {target}")
                    target.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(tree),str(target))
        samples.sort(key=lambda r:(r["episode_id"],r["t"]))
        episodes.sort(key=lambda r:r["episode_id"])
        # Optional sparse supervision: only explicitly reviewed HF terminal
        # failures whose final evidence covers recording end. Never propagate it.
        confidence_scale={"high":0.9,"medium":0.7,"low":0.5}
        terminal_supervision=Counter()
        rows_by_episode=defaultdict(list)
        for row in samples: rows_by_episode[row["episode_id"]].append(row)
        for ep in episodes:
            if ep.get("dataset_cohort")!="hf_new_discarded_161" or ep.get("outcome")!="failed": continue
            if ep.get("subtype")=="gear_drop_then_human_recovery": continue
            rows=rows_by_episode.get(ep["episode_id"],[])
            duration=float(ep.get("episode_end_s") or 0)
            evidence=[float(t) for t in ep.get("evidence_times_s",[]) if isinstance(t,(int,float))]
            if not rows or not evidence or duration-max(evidence)>0.5: continue
            end_row=next((r for r in reversed(rows) if r.get("terminal")),None)
            if end_row is None: continue
            raw_conf=str(ep.get("confidence","")).lower()
            end_row["terminal_outcome"]=0
            end_row["terminal_outcome_confidence"]=confidence_scale.get(raw_conf,0.5)
            end_row["terminal_outcome_provenance"]={
                "source":"reviewed_hf_episode_annotation",
                "annotation_outcome":ep.get("outcome"),"annotation_confidence":ep.get("confidence"),
                "subtype":ep.get("subtype"),"reviewer_model":ep.get("reviewer_model"),
                "review_method":ep.get("review_method"),"rationale":ep.get("rationale"),
                "evidence_times_s":evidence,"evidence_to_recording_end_s":round(duration-max(evidence),6),
                "scope":"recorded episode terminal observation only; not policy-only failure",
            }
            terminal_supervision[ep["split"]]+=1
        for name,records in (("samples.jsonl",samples),("episodes.jsonl",episodes)):
            tmp=output/(name+".tmp")
            with tmp.open("w",encoding="utf-8") as f:
                for row in records:f.write(json.dumps(row,ensure_ascii=False)+"\n")
            os.replace(tmp,output/name)
        ep_by_id={e["episode_id"]:e for e in episodes}
        by_split=Counter(e["split"] for e in episodes); samples_split=Counter(r["split"] for r in samples)
        cats=defaultdict(Counter); owner=defaultdict(Counter); cohorts=defaultdict(Counter)
        for e in episodes:
            cats[e["split"]][e["source_category"]]+=1
            cats[e["split"]][f"weak_demo_{e['weak_demo']}"]+=1
            cohorts[e["dataset_cohort"]]["episodes"]+=1; cohorts[e["dataset_cohort"]]["samples"]+=e["sample_count"]
        for r in samples:
            cohort=ep_by_id[r["episode_id"]]["dataset_cohort"]
            if r.get("human_control_active") is not None:owner[cohort][f"human_control_{r['human_control_active']}"]+=1
            if r.get("control_source") is not None:owner[cohort]["rows_with_control_source"]+=1
            if r.get("recording_segment") is not None:owner[cohort]["rows_with_recording_segment"]+=1
        old_samples=a.old_samples.resolve(); old_manifest=a.old_manifest.resolve(); annot=a.hf_root/"episodes.jsonl"
        audit={
            "schema_version":1,"dataset":"full-trajectory state-value inputs","task":"put gear in carrier and pin in it",
            "label_semantics":"No dense frame-level outcome/reward is assigned. source_category and weak_demo are curation provenance only. Sparse terminal_outcome labels are separately attached only where reviewed terminal failure evidence qualifies.",
            "source_cohorts":{"frozen_manual":366,"hf_discarded_pinned":161,"total_episodes":len(episodes)},
            "source_repo":"ZebTaylor/panthera-discarded-critic-audit-20261028","source_revision":"a03503bab8ccb3f377fc360e8186a5bd71097eb5",
            "old_episode_manifest_sha256":sha256(old_manifest),"old_frozen_samples_sha256":sha256(old_samples),"hf_annotation_file_sha256":sha256(annot),
            "split_policy_old":"Frozen per-episode split/session mapping from state-critic dataset; whole old validation/test remain validation/test.",
            "split_policy_new":NEW_DATE_SPLITS,"episode_counts_by_split":dict(by_split),"sample_counts_by_split":dict(samples_split),
            "source_category_by_episode_and_split":{s:dict(c) for s,c in cats.items()},"cohort_counts":{s:dict(c) for s,c in cohorts.items()},
            "recorded_control_provenance_counts":{s:dict(c) for s,c in owner.items()},"sampling_hz":a.sample_hz,
            "image_size":[a.image_size,int(a.image_size*3/4)],"camera_order":list(CAMERAS),
            "state_order":[f"left_joint_{i+1}" for i in range(6)]+["left_gripper"]+[f"right_joint_{i+1}" for i in range(6)]+["right_gripper"],
            "max_camera_state_gap_s":a.max_gap_s,"sample_rows":len(samples),"unique_referenced_images":len({p for r in samples for p in r["images"]}),
            "dense_target_field_present":False,"sparse_terminal_outcome_present":True,"human_contact_limitation":"Per-frame ownership fields are retained where recorded. Physical human contact is not sensor-recorded; episode-level intervention provenance is not time-propagated.",
            "terminal_supervision":{"definition":"Sparse recorded-behavior failure at the exact terminal observation, only for reviewed HF `failed` annotations with reviewed evidence within 0.5 s of recording end; no frame propagation.","negative_rows_by_split":dict(terminal_supervision),"positive_rows_by_split":{},"excluded_recovery_subtypes":["gear_drop_then_human_recovery"],"apparent_success_labels_used":False},
            "hitl_limitations":"HITL ownership/policy/recording segment fields are preserved where available; observations only, no synthesized transitions.",
            "archive_hashes_by_episode":{e["episode_id"]:e.get("source_archive_sha256_verified") for e in episodes if e.get("source_archive_sha256_verified")},
            "episodes_sha256":sha256(output/"episodes.jsonl"),"samples_sha256":sha256(output/"samples.jsonl"),"stats":dict(stats),
            "timestamp_semantics":"t is median of selected camera capture times; state is the nearest measured 14D row in host monotonic time.",
            "parallel_workers":n,"parallel_chunks":len(parts),
        }
        tmp=output/"audit.json.tmp";tmp.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n");os.replace(tmp,output/"audit.json")
        print(json.dumps({"output":str(output),"episodes":len(episodes),"samples":len(samples),"splits":dict(by_split),"sample_counts":dict(samples_split),"images":audit["unique_referenced_images"],"audit":str(output/"audit.json")},indent=2))
    finally:
        shutil.rmtree(temp,ignore_errors=True)

if __name__=="__main__":main()
