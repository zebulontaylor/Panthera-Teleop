#!/usr/bin/env python3
"""Build a sampled visual/state dataset for offline state-critic training.

Only manual operator demos are read. Source category is retained in audit
provenance and a separate curated-proxy file, never in trainer sample inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
DATE_SPLITS = {
    "20261022": "train",
    "20261023": "train",
    "20261025": "val",
    "20261027": "val",
    "20261028": "test",
}
BLOCK_SPLITS = {
    # UTC half-hour recording blocks, predeclared after inspecting source
    # category distribution. Old dates stay grouped as whole dates.
    ("20261028", "00:00"): "test",
    ("20261028", "17:00"): "train",
    ("20261028", "17:30"): "val",
    ("20261028", "18:00"): "test",
    ("20261028", "18:30"): "test",
    ("20261028", "19:00"): "train",
    ("20261028", "19:30"): "val",
    ("20261028", "20:00"): "val",
    ("20261028", "21:00"): "train",
    ("20261028", "21:30"): "train",
    ("20261028", "22:00"): "test",
    ("20261028", "22:30"): "test",
}
ALLOWED_CONFIDENCE = {"high", "medium"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_outcome(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) == value and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"success", "succeeded", "complete", "completed", "1", "true"}:
            return 1
        if normalized in {"failure", "failed", "incomplete", "0", "false"}:
            return 0
        if normalized in {"unknown", "uncertain", "ambiguous", "-1", ""}:
            return None
    return None


def read_labels(work_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, str]]:
    paths: set[Path] = set()
    for pattern in (
        "reviewer*.jsonl",
        "labels/reviewer*.jsonl",
        "labels/**/reviewer*.jsonl",
    ):
        paths.update(work_root.glob(pattern))
    label_root = work_root / "state-critic-20261028"
    if label_root.exists():
        for pattern in ("reviewer*.jsonl", "labels/reviewer*.jsonl", "labels/**/reviewer*.jsonl"):
            paths.update(label_root.glob(pattern))

    by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    read_paths = []
    raw_snapshots: dict[Path, bytes] = {}
    for path in sorted(paths):
        read_paths.append(str(path.resolve()))
        raw = path.read_bytes()
        raw_snapshots[path] = raw
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = row.get("episode_id")
            if not episode_id:
                continue
            record = dict(row)
            key = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            provenance = {"file": str(path.resolve()), "line": line_number}
            if key in by_episode[str(episode_id)]:
                by_episode[str(episode_id)][key]["_provenance"].append(provenance)
                continue
            record["_provenance"] = [provenance]
            record["_normalized_outcome"] = normalize_outcome(row.get("outcome"))
            record["_confidence"] = str(row.get("confidence", "unknown")).strip().lower()
            by_episode[str(episode_id)][key] = record
    changed = [str(path) for path, raw in raw_snapshots.items() if path.read_bytes() != raw]
    if changed:
        raise RuntimeError(f"review label files changed during snapshot: {changed}")
    hashes = {str(path.resolve()): hashlib.sha256(raw).hexdigest()
              for path, raw in raw_snapshots.items()}
    return ({episode_id: list(records.values()) for episode_id, records in by_episode.items()},
            read_paths, hashes)


def resolve_label(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"outcome": -1, "confidence": 0.0, "label_source": "reviewed",
                "status": "unreviewed"}
    records = [record for record in records
               if str(record.get("review_status", "")).strip().lower() != "pending"]
    if not records:
        return {"outcome": -1, "confidence": 0.0, "label_source": "reviewed",
                "status": "pending_review"}
    outcomes = {record["_normalized_outcome"] for record in records}
    if len(outcomes) > 1:
        return {"outcome": -1, "confidence": 0.0, "label_source": "reviewed",
                "status": "conflicting_outcomes"}
    outcome = next(iter(outcomes))
    if outcome is None:
        return {"outcome": -1, "confidence": 0.0, "label_source": "reviewed",
                "status": "reviewer_unknown"}
    if any(record["_confidence"] not in ALLOWED_CONFIDENCE for record in records):
        return {"outcome": -1, "confidence": 0.0, "label_source": "reviewed",
                "status": "confidence_below_threshold"}
    confidences = {record["_confidence"] for record in records}
    confidence = 0.5 if "medium" in confidences else 1.0
    return {"outcome": outcome, "confidence": confidence, "label_source": "reviewed",
            "status": "resolved"}


def split_info(episode_id: str) -> tuple[str | None, str, str | None, str]:
    date = episode_id[:8]
    old_split = DATE_SPLITS.get(date)
    if date != "20261028":
        return old_split, date, old_split, date
    hour = episode_id[9:11]
    minute = int(episode_id[11:13])
    block = f"{hour}:{'00' if minute < 30 else '30'}"
    block_id = f"{date}T{block}"
    return BLOCK_SPLITS.get((date, block)), block_id, old_split, block_id


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
    temporary.replace(path)


def write_label_snapshot(output_root: Path, label_records: dict[str, list[dict[str, Any]]]) -> None:
    rows = []
    for episode_id in sorted(label_records):
        for record in label_records[episode_id]:
            clean = {key: value for key, value in record.items() if not key.startswith("_")}
            clean["provenance"] = record["_provenance"]
            rows.append(clean)
    atomic_jsonl(output_root / "label-records.jsonl", rows)


def write_curated_proxy(output_root: Path, episodes: list[dict[str, Any]]) -> None:
    rows = []
    for episode in episodes:
        category = episode.get("source_category")
        if category not in ("kept", "discarded"):
            continue
        split, session_id, original_date_split, _ = split_info(episode["episode_id"])
        rows.append({
            "episode_id": episode["episode_id"],
            "session_id": session_id,
            "split": split,
            "original_date_split": original_date_split,
            "outcome": 1 if category == "kept" else 0,
            "label_source": "curated_proxy",
            "confidence": 0.0,
            "diagnostic_only": True,
            "source_category": category,
        })
    atomic_jsonl(output_root / "curated-proxy-diagnostic-only.jsonl", rows)


def relabel_only(manifest: dict[str, Any], manifest_path: Path, output_root: Path,
                 label_records: dict[str, list[dict[str, Any]]], label_paths: list[str],
                 label_hashes: dict[str, str]) -> None:
    samples_path = output_root / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(f"--relabel-only requires existing {samples_path}")
    resolved = {episode["episode_id"]: resolve_label(label_records.get(episode["episode_id"], []))
                for episode in manifest["episodes"]}
    by_id = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    sample_rows = read_jsonl(samples_path)
    sample_counts = defaultdict(Counter)
    valid_episodes = defaultdict(set)
    for row in sample_rows:
        episode_id = row["episode_id"]
        split, session_id, original_date_split, block_id = split_info(episode_id)
        if split is None:
            raise ValueError(f"No approved split for {episode_id} ({block_id})")
        label = resolved[episode_id]
        row.update(split=split, session_id=session_id, original_date_split=original_date_split,
                   outcome=label["outcome"], confidence=label["confidence"],
                   label_source="reviewed", label_status=label["status"])
        sample_counts[split][label["status"]] += 1
        sample_counts[split][f"outcome_{label['outcome']}"] += 1
        valid_episodes[split].add(episode_id)

    episode_label_counts = defaultdict(Counter)
    category_counts = defaultdict(Counter)
    date_counts = Counter()
    block_counts = Counter()
    terminal_time_audit = []
    for episode in manifest["episodes"]:
        episode_id = episode["episode_id"]
        split, session_id, original_date_split, block_id = split_info(episode_id)
        if split is None:
            raise ValueError(f"No approved split for {episode_id} ({block_id})")
        date_counts[split] += 1
        if episode_id.startswith("20261028"):
            block_counts[split] += 1
        category_counts[split][episode.get("source_category", "unknown")] += 1
        label = resolved[episode_id]
        episode_label_counts[split][label["status"]] += 1
        episode_label_counts[split][f"outcome_{label['outcome']}"] += 1
        review_times = [float(record["terminal_time_s"]) for record in label_records.get(episode_id, [])
                        if record.get("terminal_time_s") is not None
                        and str(record.get("review_status", "")).strip().lower() != "pending"]
        if review_times:
            duration_s = float(episode["duration_s"])
            selected_time_s = min(review_times)
            if selected_time_s < duration_s - 0.5:
                terminal_time_audit.append({
                    "episode_id": episode_id,
                    "reviewer_terminal_time_s": selected_time_s,
                    "recorded_episode_end_s": duration_s,
                    "delta_s": round(duration_s - selected_time_s, 6),
                    "sampling_end_policy": "retain samples through recorded episode duration; do not truncate on label terminal_time_s",
                })

    # Replace dataset rows and provenance files atomically, with audit.json as
    # the final commit marker for this metadata generation.
    atomic_jsonl(samples_path, sample_rows)
    write_label_snapshot(output_root, label_records)
    write_curated_proxy(output_root, manifest["episodes"])
    audit_path = output_root / "audit.json"
    previous = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    manifest_bytes = manifest_path.read_bytes()
    sample_counts_total = Counter(row["split"] for row in sample_rows)
    original_preparation = previous.get("original_preparation_stats",
                                       previous.get("preparation_stats", {}))
    preparation_stats = dict(original_preparation)
    preparation_stats.update({
        "samples_written": len(sample_rows),
        "samples_train": sample_counts_total.get("train", 0),
        "samples_val": sample_counts_total.get("val", 0),
        "samples_test": sample_counts_total.get("test", 0),
        "review_terminal_earlier_than_recorded_end": len(terminal_time_audit),
    })
    original_date_counts = Counter(DATE_SPLITS.get(episode["episode_id"][:8], "unassigned")
                                   for episode in manifest["episodes"])
    previous.update({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "label_records_snapshot_sha256": hashlib.sha256(
            (output_root / "label-records.jsonl").read_bytes()).hexdigest(),
        "label_files_read": label_paths,
        "label_file_sha256": label_hashes,
        "label_records_snapshot": str(output_root / "label-records.jsonl"),
        "split_policy": "Whole UTC dates 2026-10-22/23/25/27; 2026-10-28 split by UTC half-hour block to reduce curation-category collapse across splits.",
        "date_split_policy_original": DATE_SPLITS,
        "block_split_policy_20261028": {f"{date}T{block}": split
                                         for (date, block), split in BLOCK_SPLITS.items()},
        "split_episode_counts": dict(date_counts),
        "original_date_split_episode_counts": dict(original_date_counts),
        "split_sample_counts": dict(sample_counts_total),
        "split_20261028_block_episode_counts": dict(block_counts),
        "source_category_counts_by_split_audit_only": {key: dict(value)
                                                        for key, value in category_counts.items()},
        "label_episode_counts_by_split": {key: dict(value) for key, value in episode_label_counts.items()},
        "label_sample_counts_by_split": {key: dict(value) for key, value in sample_counts.items()},
        "session_id_policy": "UTC date for 2026-10-22/23/25/27; UTC date+half-hour block for 2026-10-28.",
        "split_limitations": [
            "The 2026-10-28 assignments are half-hour time blocks within one day, not independent recording sessions.",
            "Source categories remain imbalanced within splits, and category/time effects may be confounded.",
            "Original whole-date split is retained per sample as original_date_split for a separate date-generalization diagnostic.",
        ],
        "session_is_utc_date_from_episode_id": False,
        "label_episode_count": len(manifest["episodes"]),
        "duplicate_label_episodes": sum(len(records) > 1 for records in label_records.values()),
        "conflicting_label_episodes": sum(label["status"] == "conflicting_outcomes"
                                           for label in resolved.values()),
        "label_ids_not_in_frozen_manifest": sorted(set(label_records) - set(by_id)),
        "reviewer_terminal_time_discrepancies": terminal_time_audit,
        "valid_sample_episodes_by_split": {key: len(value) for key, value in valid_episodes.items()},
        "relabel_only": True,
        "curated_proxy_file": str(output_root / "curated-proxy-diagnostic-only.jsonl"),
        "original_preparation_stats": original_preparation,
        "preparation_stats": preparation_stats,
    })
    previous.pop("date_split_policy", None)
    previous.pop("split_date_counts", None)
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(previous, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(audit_path)
    print(json.dumps({
        "samples": len(sample_rows),
        "split_episode_counts": dict(date_counts),
        "source_category_counts_by_split_audit_only": {key: dict(value)
                                                        for key, value in category_counts.items()},
        "label_episode_counts_by_split": {key: dict(value) for key, value in episode_label_counts.items()},
        "label_sample_counts_by_split": {key: dict(value) for key, value in sample_counts.items()},
        "duplicate_label_episodes": previous["duplicate_label_episodes"],
        "conflicting_label_episodes": previous["conflicting_label_episodes"],
    }, indent=2), flush=True)


def flat_state(row: dict[str, Any]) -> list[float] | None:
    arms = row.get("arms") or {}
    result: list[float] = []
    for side in ("left", "right"):
        arm = arms.get(side) or {}
        joints = arm.get("position_rad")
        gripper = (arm.get("gripper") or {}).get("position")
        if not isinstance(joints, list) or len(joints) != 6 or gripper is None:
            return None
        result.extend(float(value) for value in joints)
        result.append(float(gripper))
    return result


def nearest_by_time(rows: list[dict[str, Any]], target_ns: int) -> dict[str, Any]:
    return min(rows, key=lambda row: (abs(int(row["monotonic_ns"]) - target_ns),
                                      int(row["monotonic_ns"])))


def nearest_by_episode_s(rows: list[dict[str, Any]], target_s: float) -> dict[str, Any]:
    return min(rows, key=lambda row: (abs(float(row["episode_time_s"]) - target_s),
                                      float(row["episode_time_s"])))


def save_compact_image(source_path: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    with Image.open(source_path) as image:
        image.convert("RGB").resize((320, 240), Image.Resampling.LANCZOS).save(
            temporary, format="JPEG", quality=88, optimize=True)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("work"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("work/state-critic-20261028/episodes.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("work/state-critic-20261028/dataset"))
    parser.add_argument("--max-gap-s", type=float, default=0.15)
    parser.add_argument("--history-seconds", type=float, default=1.0)
    parser.add_argument("--relabel-only", action="store_true",
                        help="update labels and splits on an existing sampled dataset without re-reading episode data or rewriting images")
    args = parser.parse_args()
    work_root = args.work_root.resolve()
    manifest_path = args.manifest.resolve()
    output_root = args.output.resolve()
    image_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    episodes = manifest["episodes"]
    label_records, label_paths, label_hashes = read_labels(work_root)
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}

    if args.relabel_only:
        relabel_only(manifest, manifest_path, output_root, label_records, label_paths, label_hashes)
        return

    unknown_label_ids = sorted(set(label_records) - set(episode_by_id))
    resolved = {episode_id: resolve_label(label_records.get(episode_id, []))
                for episode_id in episode_by_id}

    # Retain every supplied reviewer record (including duplicate/conflicting
    # records) verbatim, plus its source file and line for auditability.
    records_path = output_root / "label-records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for episode_id in sorted(label_records):
            for record in label_records[episode_id]:
                clean = {key: value for key, value in record.items()
                         if not key.startswith("_")}
                clean["provenance"] = record["_provenance"]
                stream.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # Create a separate explicitly weak curation proxy. It is never copied to
    # samples.jsonl and must not be used as a verified task outcome.
    proxy_path = output_root / "curated-proxy-diagnostic-only.jsonl"
    with proxy_path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            category = episode.get("source_category")
            if category not in ("kept", "discarded"):
                continue
            stream.write(json.dumps({
                "episode_id": episode["episode_id"],
                "session_id": episode["episode_id"][:8],
                "split": DATE_SPLITS.get(episode["episode_id"][:8], "unassigned"),
                "outcome": 1 if category == "kept" else 0,
                "label_source": "curated_proxy",
                "confidence": "unknown",
                "diagnostic_only": True,
                "source_category": category,
            }, ensure_ascii=False) + "\n")

    sample_path = output_root / "samples.jsonl"
    stats = Counter()
    per_split = defaultdict(Counter)
    valid_sample_episodes = defaultdict(set)
    warnings = []
    with sample_path.open("w", encoding="utf-8") as stream:
        for episode_number, episode in enumerate(episodes, start=1):
            episode_id = episode["episode_id"]
            source_dir = Path(episode["source_dir"])
            split, session_id, original_date_split, block_id = split_info(episode_id)
            if split is None:
                warnings.append({"episode_id": episode_id, "reason": "date_block_missing_from_split_policy",
                                 "block_id": block_id})
                stats["unassigned_date_episodes"] += 1
                continue
            metadata_path = source_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            episode_start_ns = int(metadata.get("start_monotonic_ns", 0))
            if episode_start_ns <= 0:
                warnings.append({"episode_id": episode_id, "reason": "missing_start_monotonic_ns"})
                stats["episodes_missing_start"] += 1
                continue

            frame_rows = read_jsonl(source_dir / "frames.jsonl")
            state_rows = read_jsonl(source_dir / "states.jsonl")
            frames = {camera: sorted(
                [row for row in frame_rows if row.get("camera") == camera],
                key=lambda row: int(row["monotonic_ns"])) for camera in CAMERAS}
            states = [row for row in state_rows if row.get("monotonic_ns") is not None
                      and flat_state(row) is not None]
            states.sort(key=lambda row: int(row["monotonic_ns"]))
            if not states or any(not frames[camera] for camera in CAMERAS):
                warnings.append({"episode_id": episode_id, "reason": "missing_usable_state_or_camera_frames"})
                stats["episodes_missing_sources"] += 1
                continue

            duration_s = float(episode["duration_s"])
            last_integer_s = int(duration_s)
            label = resolved[episode_id]
            records = label_records.get(episode_id, [])
            review_times = [float(record["terminal_time_s"]) for record in records
                            if record.get("terminal_time_s") is not None]
            reviewed_terminal_s = min(review_times) if review_times else None
            if reviewed_terminal_s is not None and reviewed_terminal_s < duration_s - 0.5:
                stats["review_terminal_earlier_than_recorded_end"] += 1

            for second in range(int(args.history_seconds), last_integer_s + 1):
                current_target_ns = episode_start_ns + int(second * 1e9)
                history_target_ns = episode_start_ns + int((second - args.history_seconds) * 1e9)
                current_frames = {}
                history_frames = {}
                frame_quality = True
                for camera in CAMERAS:
                    current = nearest_by_time(frames[camera], current_target_ns)
                    history = nearest_by_time(frames[camera], history_target_ns)
                    if (abs(int(current["monotonic_ns"]) - current_target_ns) > args.max_gap_s * 1e9
                            or abs(int(history["monotonic_ns"]) - history_target_ns) > args.max_gap_s * 1e9):
                        frame_quality = False
                        break
                    current_frames[camera] = current
                    history_frames[camera] = history
                if not frame_quality:
                    stats["samples_rejected_camera_gap"] += 1
                    continue

                # Pick the state closest in monotonic time to the current camera
                # capture midpoint and reject if state/camera skew exceeds 150ms.
                current_frame_ns = sorted(int(current_frames[c]["monotonic_ns"]) for c in CAMERAS)[1]
                state_row = nearest_by_time(states, current_frame_ns)
                state_gap_s = abs(int(state_row["monotonic_ns"]) - current_frame_ns) / 1e9
                if state_gap_s > args.max_gap_s:
                    stats["samples_rejected_state_gap"] += 1
                    continue
                state_value = flat_state(state_row)
                if state_value is None:
                    stats["samples_rejected_invalid_state"] += 1
                    continue

                images = []
                history_images = []
                for camera in CAMERAS:
                    for selected, mapping in ((current_frames[camera], images),
                                               (history_frames[camera], history_images)):
                        frame_id = f"{int(selected.get('frame_index', 0)):07d}.jpg"
                        relative_image = Path("images") / episode_id / camera / frame_id
                        save_compact_image(source_dir / selected["path"], output_root / relative_image)
                        mapping.append(relative_image.as_posix())

                last_sample_s = float(last_integer_s)
                sample = {
                    "episode_id": episode_id,
                    "session_id": session_id,
                    "split": split,
                    "original_date_split": original_date_split,
                    "t": float(second),
                    "images": images,
                    "history_images": history_images,
                    "state": state_value,
                    "outcome": label["outcome"],
                    "terminal": second == last_integer_s,
                    "episode_end_s": duration_s,
                    "label_source": "reviewed",
                    "confidence": label["confidence"],
                    "label_status": label["status"],
                }
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["samples_written"] += 1
                stats[f"samples_{split}"] += 1
                per_split[split][label["status"]] += 1
                per_split[split][f"outcome_{label['outcome']}"] += 1
                valid_sample_episodes[split].add(episode_id)
            if episode_number % 25 == 0 or episode_number == len(episodes):
                print(f"dataset episodes processed: {episode_number}/{len(episodes)}", flush=True)

    episode_label_counts = defaultdict(Counter)
    for episode_id, label in resolved.items():
        split = DATE_SPLITS.get(episode_id[:8], "unassigned")
        episode_label_counts[split][label["status"]] += 1
        episode_label_counts[split][f"outcome_{label['outcome']}"] += 1

    audit = {
        "schema_version": 1,
        "task": "put the gear in the carrier and put a pin in it",
        "source": "manual operator demos only; HITL is excluded",
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "label_files_read": label_paths,
        "label_file_sha256": label_hashes,
        "episode_count": len(episodes),
        "date_split_policy_original": DATE_SPLITS,
        "block_split_policy_20261028": {f"{date}T{block}": split
                                         for (date, block), split in BLOCK_SPLITS.items()},
        "split_date_counts": dict(Counter(split_info(e["episode_id"])[0] or "unassigned"
                                           for e in episodes)),
        "label_episode_counts_by_split": {key: dict(value) for key, value in episode_label_counts.items()},
        "label_sample_counts_by_split": {key: dict(value) for key, value in per_split.items()},
        "session_is_utc_date_from_episode_id": True,
        "samples_file": str(sample_path),
        "image_root": str(image_root),
        "sample_rate_hz": 1,
        "history_seconds": args.history_seconds,
        "camera_order": list(CAMERAS),
        "image_size": [320, 240],
        "state_order": [f"left.joint{i}" for i in range(1, 7)] + ["left.gripper"]
                       + [f"right.joint{i}" for i in range(1, 7)] + ["right.gripper"],
        "state_units": "radians",
        "state_alignment": "Measured 14D state nearest to median current-camera monotonic_ns; sample rejected if any camera is >150ms from target or state is >150ms from camera midpoint.",
        "terminal_semantics": "Only the final 1Hz sample at or before recorded episode_end_s is terminal. Reviewer terminal_time_s is preserved in label-records.jsonl and does not truncate motion samples.",
        "unknown_semantics": "Outcome -1 includes unreviewed episodes, explicit reviewer unknowns, conflicting duplicate outcomes, and any duplicate set containing confidence below medium. All source records are preserved in label-records.jsonl.",
        "curated_proxy_file": str(proxy_path),
        "curated_proxy_warning": "Diagnostic-only kept=1/discarded=0 proxy; excluded from trainer samples and not a verified success label.",
        "duplicate_label_episodes": sum(len(records) > 1 for records in label_records.values()),
        "conflicting_label_episodes": sum(label["status"] == "conflicting_outcomes" for label in resolved.values()),
        "label_ids_not_in_frozen_manifest": unknown_label_ids,
        "preparation_stats": dict(stats),
        "valid_sample_episodes_by_split": {key: len(value) for key, value in valid_sample_episodes.items()},
        "warnings": warnings,
    }
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "episode_count", "split_date_counts", "label_episode_counts_by_split",
        "label_sample_counts_by_split", "preparation_stats", "valid_sample_episodes_by_split",
        "duplicate_label_episodes", "conflicting_label_episodes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
