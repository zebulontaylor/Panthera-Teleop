#!/usr/bin/env python3
"""Export conservative terminal-window observations from the pinned HF cohort.

This creates a separate dataset for visual state-critic experiments. Labels are
attached only to the last three 1 Hz observations of each annotated episode,
and only when the annotation's final evidence time is within 0.5 s of the
recorded episode duration. The source archive and annotation hashes remain in
row provenance; unknown/non-attempt episodes are not assigned objective labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
REVISION = "a03503bab8ccb3f377fc360e8186a5bd71097eb5"
REPO = "ZebTaylor/panthera-discarded-critic-audit-20261028"
DATE_SPLITS = {
    "20260930": "train", "20261002": "train", "20261006": "train",
    "20261007": "val", "20261008": "val",
    "20261005": "test", "20261011": "test",
}

# These narrow labels are grounded in annotation rationales, not inferred from
# the episode-level `failed` / `apparent_success` category. Unknowns remain -1.
PIN_VISIBLE_CANDIDATE_IDS = {
    "20260930T011804-7d74ea9d", "20260930T203328-d578f0dd",
    "20261002T183443-aa9a0d75", "20261002T184528-e0b12e37",
    "20261005T151753-88a4c88a", "20261005T225826-9192085f",
}
PIN_VISIBLE_POSITIVE_IDS = {
    "20260930T203328-d578f0dd", "20261002T184528-e0b12e37",
    "20261005T151753-88a4c88a",
}
# Filled from an explicit review of the terminal rationale: pin visibly lies
# outside the bore after an insertion attempt (tabletop or crosswise carrier).
PIN_PLACEMENT_FAILURE_IDS: set[str] = set()
PIN_PLACEMENT_FAILURE_IDS.update({
    "20260930T015326-5203c283", "20260930T021027-c0f37bf9",
    "20260930T202406-d5274f24", "20260930T202515-94c13be9",
    "20260930T202609-90f80771", "20260930T202638-b02cde5b",
    "20260930T202705-4910c986", "20260930T202842-7c90cc4d",
    "20260930T204055-99decb5a", "20260930T213820-b9d64df7",
    "20260930T213845-be0755a1", "20260930T214118-4761e23f",
    "20260930T214620-189c0975", "20261002T175538-373a1176",
    "20261002T180342-659a577d", "20261002T184449-3e681b4d",
    "20261002T195853-d55429a2", "20261002T200045-5a5540b7",
    "20261002T200448-ed6bf2f9", "20261005T154026-fb2653b9",
    "20261005T225005-3daca7d9", "20261005T225143-c7ec0ad2",
})
TARGET_DEFINITION = (
    "At the final visually supported observation, pin remains visibly at the intended peripheral carrier bore after release (1), "
    "or pin is visibly on the tabletop/crosswise on the carrier after an insertion attempt (0). "
    "Episode-level outcomes alone are not used; all other examples are unknown (-1)."
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nearest(rows: list[dict[str, Any]], key: str, target: float, tolerance: float) -> dict[str, Any]:
    row = min(rows, key=lambda x: abs(float(x[key]) - target))
    delta = abs(float(row[key]) - target)
    if delta > tolerance:
        raise ValueError(f"nearest {key} gap {delta:.3f}s exceeds {tolerance}s")
    return row


def extract_jpeg(tar: tarfile.TarFile, member: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    source = tar.extractfile(member)
    if source is None:
        raise FileNotFoundError(member)
    with Image.open(source) as image:
        image.convert("RGB").resize((320, 240), Image.Resampling.LANCZOS).save(
            out, "JPEG", quality=88, optimize=True
        )


def state_vector(row: dict[str, Any]) -> list[float]:
    arms = row.get("arms", {})
    left, right = arms.get("left", {}), arms.get("right", {})
    values = (left.get("position_rad"), left.get("gripper", {}).get("position"),
              right.get("position_rad"), right.get("gripper", {}).get("position"))
    if values[0] is None or values[2] is None or values[1] is None or values[3] is None:
        raise ValueError("missing measured joint/gripper state")
    result = [float(v) for v in values[0]] + [float(values[1])] + [float(v) for v in values[2]] + [float(values[3])]
    if len(result) != 14:
        raise ValueError(f"expected 14 state dimensions, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("work/state-critic-20261029/hf-new-discarded"))
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = root / "dataset"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    annotations_path = root / "episodes.jsonl"
    annotations = jsonl(annotations_path)
    selected = [x for x in annotations if x.get("outcome") in ("failed", "apparent_success")]
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    episode_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    image_cache: set[str] = set()

    for annotation in selected:
        eid = annotation["episode_id"]
        split = DATE_SPLITS.get(eid[:8])
        if split is None:
            skipped.append({"episode_id": eid, "reason": "date_without_predeclared_split"})
            continue
        stats = annotation.get("recording_statistics") or annotation.get("recording_stats") or {}
        duration = stats.get("duration_s")
        if duration is None:
            duration = annotation.get("raw_recording", {}).get("duration_s")
        evidence = [float(t) for t in annotation.get("evidence_times_s", [])]
        if duration is None or not evidence or max(evidence) < float(duration) - 0.5:
            skipped.append({"episode_id": eid, "reason": "terminal_evidence_not_supported"})
            continue
        duration = float(duration)
        archive_rel = annotation["source_archive"]
        archive_path = root / archive_rel
        if not archive_path.is_file():
            skipped.append({"episode_id": eid, "reason": "source_archive_not_downloaded"})
            continue
        actual_hash = sha256(archive_path)
        expected_hash = annotation.get("source_archive_sha256") or annotation.get("raw_recording", {}).get("archive_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"archive SHA-256 mismatch for {eid}: {actual_hash} != {expected_hash}")
        with tarfile.open(archive_path, "r:gz") as tar:
            names = set(tar.getnames())
            if not {"metadata.json", "frames.jsonl", "states.jsonl"}.issubset(names):
                raise ValueError(f"incomplete archive schema: {eid}")
            metadata = json.load(tar.extractfile("metadata.json"))
            start_ns = int(metadata.get("start_monotonic_ns", metadata.get("trigger_monotonic_ns")))
            frames_by_camera: dict[str, list[dict[str, Any]]] = {cam: [] for cam in CAMERAS}
            for line in tar.extractfile("frames.jsonl"):
                frame = json.loads(line)
                if frame.get("camera") in frames_by_camera and frame.get("episode_time_s") is not None:
                    frames_by_camera[frame["camera"]].append(frame)
            for frames in frames_by_camera.values():
                frames.sort(key=lambda f: float(f["episode_time_s"]))
            state_rows = [json.loads(line) for line in tar.extractfile("states.jsonl")]
            state_rows.sort(key=lambda s: int(s["monotonic_ns"]))
            if not state_rows or any(not frames_by_camera[c] for c in CAMERAS):
                skipped.append({"episode_id": eid, "reason": "missing_states_or_camera"})
                continue

            # Use one observation nearest final evidence, never earlier frames
            # that might precede a later failure. Clamp to shared video extent.
            final_evidence_t = max(evidence)
            common_video_end = min(float(frames_by_camera[c][-1]["episode_time_s"]) for c in CAMERAS)
            t = min(final_evidence_t, common_video_end)
            current_paths, history_paths, frame_provenance = [], [], {}
            for cam in CAMERAS:
                current = nearest(frames_by_camera[cam], "episode_time_s", t, 0.15)
                hist_target = max(float(frames_by_camera[cam][0]["episode_time_s"]), t - 1.0)
                hist = nearest(frames_by_camera[cam], "episode_time_s", hist_target, 0.15)
                for frame, dest_list in ((current, current_paths), (hist, history_paths)):
                    frame_member = frame["path"]
                    relative = Path("images") / eid / cam / f"{int(frame.get('frame_index',0)):07d}.jpg"
                    local = dataset / relative
                    extract_jpeg(tar, frame_member, local)
                    local_rel = str(relative)
                    dest_list.append(local_rel)
                    image_cache.add(local_rel)
                frame_provenance[cam] = {
                    "current_episode_time_s": float(current["episode_time_s"]),
                    "history_episode_time_s": float(hist["episode_time_s"]),
                    "current_source_path": current["path"],
                    "history_source_path": hist["path"],
                }
            state_target_ns = start_ns + int(round(t * 1e9))
            state = min(state_rows, key=lambda s: abs(int(s["monotonic_ns"]) - state_target_ns))
            state_gap_s = abs(int(state["monotonic_ns"]) - state_target_ns) / 1e9
            if state_gap_s > 0.15:
                skipped.append({"episode_id": eid, "reason": f"state_gap_{state_gap_s:.3f}s"})
                continue
            target_outcome = 1 if eid in PIN_VISIBLE_POSITIVE_IDS else 0 if eid in PIN_PLACEMENT_FAILURE_IDS else -1
            target_evidence_code = ("pin_visible_at_peripheral_bore_after_release" if target_outcome == 1 else
                                    "pin_outside_bore_after_insertion_attempt" if target_outcome == 0 else "unknown")
            rows.append({
                    "episode_id": eid,
                    "session_id": eid[:8],
                    "split": split,
                    "t": round(t, 3),
                    "images": current_paths,
                    "history_images": history_paths,
                    "state": state_vector(state),
                    "outcome": target_outcome,
                    "terminal": True,
                    "episode_end_s": duration,
                    "recording_end_s": duration,
                    "terminal_window_index": 0,
                    "label_source": "reviewed",
                    "confidence": 0.5 if annotation.get("confidence") == "medium" else 1.0,
                    "label_status": "rationale_coded" if target_outcome >= 0 else "target_unknown",
                    "target_definition": TARGET_DEFINITION,
                    "target_evidence_code": target_evidence_code,
                    "target_candidate_pin_visible_positive": eid in PIN_VISIBLE_CANDIDATE_IDS,
                    "target_visual_review_status": (
                        "peripheral_bore_pin_visibly_retained_after_release" if target_outcome == 1 else
                        "peripheral_bore_retention_ambiguous_unknown" if eid in PIN_VISIBLE_CANDIDATE_IDS else
                        "explicit_pin_outside_after_insertion_attempt" if target_outcome == 0 else "not_eligible"
                    ),
                    "source_group": annotation.get("source_group"),
                    "control_mode": annotation.get("control_mode"),
                    "human_intervention_observed": annotation.get("human_intervention_observed"),
                    "task_attempt_observed": annotation.get("task_attempt_observed"),
                    "source_repo": REPO,
                    "source_revision": REVISION,
                    "source_archive": archive_rel,
                    "source_archive_sha256": actual_hash,
                    "source_category": annotation.get("primary_category"),
                    "source_subtype": annotation.get("subtype"),
                    "annotation_outcome": annotation.get("outcome"),
                    "annotation_confidence": annotation.get("confidence"),
                    "annotation_secondary_review_status": annotation.get("secondary_review_status"),
                    "annotation_rationale": annotation.get("rationale"),
                    "annotation_evidence_times_s": evidence,
                    "annotation_terminal_evidence_time_s": max(evidence),
                    "terminal_evidence_to_recording_end_s": round(duration - max(evidence), 6),
                    "annotation_evidence_time_reference": annotation.get("evidence_time_reference"),
                    "annotation_label_origin": annotation.get("label_origin"),
                    "source_requested_t_s": final_evidence_t,
                    "source_actual_t_s": t,
                    "source_end_time_delta_s": round(duration - max(evidence), 6),
                    "source_terminal_frame_provenance": frame_provenance,
                    "state_time_gap_s": round(state_gap_s, 6),
            })
            episode_counts[split] += 1
            outcome_counts[f"{split}:{target_outcome}"] += 1

    rows.sort(key=lambda x: (x["episode_id"], x["t"]))
    out_path = dataset / "samples.jsonl"
    temp = out_path.with_suffix(".jsonl.tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(out_path)
    counts_by_episode: dict[str, Counter[str]] = {s: Counter() for s in ("train", "val", "test")}
    seen: dict[str, set[str]] = {s: set() for s in counts_by_episode}
    for row in rows:
        seen[row["split"]].add(row["episode_id"])
        name = {1: "pin_visible_positive", 0: "pin_placement_failure", -1: "unknown"}[row["outcome"]]
        counts_by_episode[row["split"]][name] += 1
    for split in seen:
        counts_by_episode[split]["episodes"] = len(seen[split])
    by_source: dict[str, Counter[str]] = {s: Counter() for s in seen}
    by_control: dict[str, Counter[str]] = {s: Counter() for s in seen}
    annotation_outcomes: dict[str, Counter[str]] = {s: Counter() for s in seen}
    archive_inventory: dict[str, str] = {}
    for row in rows:
        split = row["split"]
        by_source[split][str(row.get("source_group") or "unknown")] += 1
        by_control[split][str(row.get("control_mode") or "unknown")] += 1
        annotation_outcomes[split][str(row.get("annotation_outcome") or "unknown")] += 1
        archive_inventory[row["episode_id"]] = row["source_archive_sha256"]
    audit = {
        "source_repo": REPO, "source_revision": REVISION,
        "annotation_file_sha256": sha256(annotations_path),
        "superseded_candidate_export": {
            "path": "samples.jsonl.pre-peripheral-review",
            "sha256": sha256(dataset / "samples.jsonl.pre-peripheral-review") if (dataset / "samples.jsonl.pre-peripheral-review").is_file() else None,
            "reason_superseded": "Initial rationale-only candidate mapping used six positives; original terminal frames were rechecked against peripheral, not central, carrier bores and three candidates were made unknown.",
        },
        "label_semantics": "narrow terminal pin-placement visual predicate; no whole-trajectory propagation",
        "task": "put gear in carrier and pin in it",
        "target_definition": TARGET_DEFINITION,
        "positive_episode_ids": sorted(PIN_VISIBLE_POSITIVE_IDS),
        "source_annotation_positive_candidates_reviewed": sorted(PIN_VISIBLE_CANDIDATE_IDS),
        "candidate_positive_ids_reclassified_unknown_after_peripheral_bore_recheck": sorted(PIN_VISIBLE_CANDIDATE_IDS - PIN_VISIBLE_POSITIVE_IDS),
        "negative_episode_ids": sorted(PIN_PLACEMENT_FAILURE_IDS),
        "target_coding_method": "Rationale review nominated six candidates. Original 640x480 terminal wrist frames were then rechecked against the peripheral carrier bores (not the central bearing): three show a pin visibly retained at a peripheral bore after release; three remain unknown. Twenty-two negatives require an explicit post-insertion pin outside the bore. No episode-level outcome was converted into this target.",
        "unknown_episode_count": sum(1 for row in rows if row["outcome"] == -1),
        "terminal_evidence_rule": "max(evidence_times_s) >= recorded_duration_s - 0.5; final evidence time clamped to shared available camera extent",
        "window_policy": "one final observation nearest final annotation evidence per episode, with 1 s history; 14D measured state nearest host monotonic frame time within 0.15s; no earlier labeled observations",
        "camera_order": list(CAMERAS), "image_size": [320, 240],
        "split_policy": DATE_SPLITS,
        "split_limitations": "Whole-date split reduces episode leakage, but same task/setup and temporal/category shifts remain. The narrow target has no eligible validation examples and only 5 eligible examples in the new held-out test split (2 positive, 3 negative), all from Oct 5. New test cohort is distinct from frozen prior cohort; source annotations and rationale-coded target are model-reviewed, not operator-verified.",
        "candidate_episode_count": len(selected),
        "included_episode_count": sum(len(ids) for ids in seen.values()),
        "row_count": len(rows),
        "per_split": {split: dict(counts) for split, counts in counts_by_episode.items()},
        "per_split_source_group": {split: dict(counts) for split, counts in by_source.items()},
        "per_split_control_mode": {split: dict(counts) for split, counts in by_control.items()},
        "per_split_original_annotation_outcome": {split: dict(counts) for split, counts in annotation_outcomes.items()},
        "source_archive_sha256_by_episode": archive_inventory,
        "skipped": skipped,
        "image_count_referenced_unique": len(image_cache),
        "samples_sha256": sha256(out_path),
        "apparent_success_caveat": "Episode-level apparent_success labels are preserved as provenance only. They do not imply full task completion. Three of six source candidates were reclassified unknown after terminal-frame recheck against the peripheral bore; the three retained positives indicate visible pin placement only, not verified mechanical seating or full task completion.",
    }
    audit_path = dataset / "audit.json"
    temp_audit = audit_path.with_suffix(".json.tmp")
    temp_audit.write_text(json.dumps(audit, indent=2) + "\n")
    temp_audit.replace(audit_path)
    print(json.dumps({"dataset": str(dataset), "rows": len(rows), "per_split": audit["per_split"],
                      "skipped_count": len(skipped), "unique_images": len(image_cache)}, indent=2))


if __name__ == "__main__":
    main()
