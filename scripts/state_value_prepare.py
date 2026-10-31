#!/usr/bin/env python3
"""Build full-trajectory, 5 Hz state-value training inputs.

The export contains observations and provenance, not synthesized task rewards.
Source curation is retained only as a weak-demo flag; episode annotations are
stored as provenance and are never propagated into frame-level task labels.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
HF_REPO = "ZebTaylor/panthera-discarded-critic-audit-20261028"
HF_REVISION = "a03503bab8ccb3f377fc360e8186a5bd71097eb5"
NEW_DATE_SPLITS = {
    "20260930": "train", "20261002": "train", "20261006": "train",
    "20261007": "val", "20261008": "val",
    "20261005": "test", "20261011": "test",
}


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if len(result) != 14 or not all(math.isfinite(v) for v in result):
        return None
    return result


def nearest_sorted(rows: list[dict[str, Any]], times: list[float], target: float,
                   key: str) -> dict[str, Any]:
    i = bisect.bisect_left(times, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(rows)]
    best = min(candidates, key=lambda j: (abs(times[j] - target), times[j]))
    return rows[best]


def state_at_ns(states: list[dict[str, Any]], times_ns: list[int], target_ns: int) -> tuple[dict[str, Any], float]:
    i = bisect.bisect_left(times_ns, target_ns)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(states)]
    best = min(candidates, key=lambda j: (abs(times_ns[j] - target_ns), times_ns[j]))
    return states[best], abs(times_ns[best] - target_ns) / 1e9


def make_plan(frame_rows: list[dict[str, Any]], duration_hint_s: float | None,
              sample_hz: float, max_gap_s: float) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    by_camera = {camera: [] for camera in CAMERAS}
    for row in frame_rows:
        if row.get("camera") in by_camera and row.get("episode_time_s") is not None:
            by_camera[row["camera"]].append(row)
    for rows in by_camera.values():
        rows.sort(key=lambda row: (float(row["episode_time_s"]), int(row.get("frame_index", 0))))
    times_by_camera = {
        camera: [float(row["episode_time_s"]) for row in rows]
        for camera, rows in by_camera.items()
    }
    missing_cameras = [camera for camera in CAMERAS if not by_camera[camera]]
    if missing_cameras:
        return [], [], {"skip_reason": "missing_camera_stream", "missing_cameras": missing_cameras}
    first_t = max(float(by_camera[c][0]["episode_time_s"]) for c in CAMERAS)
    camera_end = min(float(by_camera[c][-1]["episode_time_s"]) for c in CAMERAS)
    end_t = min(camera_end, float(duration_hint_s)) if duration_hint_s is not None and duration_hint_s > 0 else camera_end
    if end_t < first_t:
        raise ValueError("camera duration precedes common start")
    period = 1.0 / sample_hz
    count = int(math.floor((end_t - first_t) / period + 1e-9)) + 1
    targets = [first_t + i * period for i in range(count)]
    if targets and end_t - targets[-1] > period * 0.5:
        targets.append(end_t)
    planned: list[dict[str, Any]] = []
    selected_members: set[str] = set()
    rejected_gap = 0
    for target in targets:
        selected: dict[str, dict[str, Any]] = {}
        gaps = {}
        for camera in CAMERAS:
            frame = nearest_sorted(by_camera[camera], times_by_camera[camera], target,
                                   "episode_time_s")
            gap = abs(float(frame["episode_time_s"]) - target)
            gaps[camera] = gap
            selected[camera] = frame
        if max(gaps.values()) > max_gap_s:
            rejected_gap += 1
            continue
        planned.append({"target_t_s": target, "frames": selected, "camera_gaps_s": gaps})
        selected_members.update(frame["path"] for frame in selected.values())
    return planned, sorted(selected_members), {
        "camera_first_t_s": {c: float(by_camera[c][0]["episode_time_s"]) for c in CAMERAS},
        "camera_last_t_s": {c: float(by_camera[c][-1]["episode_time_s"]) for c in CAMERAS},
        "common_start_t_s": first_t, "common_end_t_s": end_t,
        "targets_before_gap_filter": len(targets), "samples_rejected_camera_gap": rejected_gap,
    }


def save_compact_bytes(raw: bytes, destination: Path, size: tuple[int, int]) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    with Image.open(__import__("io").BytesIO(raw)) as image:
        image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
            temporary, "JPEG", quality=88, optimize=True
        )
    temporary.replace(destination)


def save_compact_path(source: Path, destination: Path, size: tuple[int, int]) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    with Image.open(source) as image:
        image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
            temporary, "JPEG", quality=88, optimize=True
        )
    temporary.replace(destination)


def recorded_ownership(state: dict[str, Any]) -> dict[str, Any]:
    control_source = state.get("control_source")
    by_arm = state.get("control_source_by_arm")
    sources = []
    if control_source is not None:
        sources.append(control_source)
    if isinstance(by_arm, dict):
        sources.extend(value for value in by_arm.values() if value is not None)
    normalized = " ".join(str(value).lower() for value in sources)
    explicit = bool(sources)
    human_active = True if any(token in normalized for token in ("human", "operator", "teleop")) else (
        False if explicit else None
    )
    contact_keys = ("human_contact_observed", "human_contact", "contact_observed",
                    "human_touch_observed", "human_intervention")
    contact = next((state[key] for key in contact_keys if key in state), None)
    return {
        "control_mode": state.get("control_mode"),
        "control_source": control_source,
        "control_source_by_arm": by_arm,
        "recording_segment": state.get("recording_segment"),
        "policy_time_s": state.get("policy_time_s"),
        "data_time_s": state.get("data_time_s"),
        "human_control_active": human_active,
        "human_contact_observed": contact,
        "human_contact_signal_available": contact is not None,
    }


def encode_episode_samples(episode: dict[str, Any], planned: list[dict[str, Any]],
                           states: list[dict[str, Any]], start_ns: int | None,
                           output_root: Path, image_size: tuple[int, int],
                           max_gap_s: float) -> tuple[list[dict[str, Any]], Counter[str]]:
    valid_states = [row for row in states if row.get("monotonic_ns") is not None and flat_state(row) is not None]
    valid_states.sort(key=lambda row: int(row["monotonic_ns"]))
    if not valid_states:
        raise ValueError("no usable measured state rows")
    times_ns = [int(row["monotonic_ns"]) for row in valid_states]
    rows: list[dict[str, Any]] = []
    stat: Counter[str] = Counter()
    kept_plans = []
    for plan in planned:
        timestamps_ns = [int(plan["frames"][camera]["monotonic_ns"]) for camera in CAMERAS]
        current_ns = sorted(timestamps_ns)[1]
        state_row, state_gap_s = state_at_ns(valid_states, times_ns, current_ns)
        if state_gap_s > max_gap_s:
            stat["samples_rejected_state_gap"] += 1
            continue
        state_value = flat_state(state_row)
        if state_value is None:
            stat["samples_rejected_invalid_state"] += 1
            continue
        frame_timestamps = {camera: float(plan["frames"][camera]["episode_time_s"]) for camera in CAMERAS}
        actual_t = sorted(frame_timestamps.values())[1]
        ownership = recorded_ownership(state_row)
        images = []
        for camera in CAMERAS:
            frame = plan["frames"][camera]
            source_index = int(frame.get("frame_index", 0))
            relative = Path("images") / episode["episode_id"] / camera / f"{source_index:07d}.jpg"
            images.append(relative.as_posix())
        row = {
            "episode_id": episode["episode_id"],
            "session_id": episode["session_id"],
            "split": episode["split"],
            "source_category": episode["source_category"],
            "weak_demo": episode["weak_demo"],
            "t": round(actual_t, 6),
            "target_t_s": round(float(plan["target_t_s"]), 6),
            "episode_end_s": episode["episode_end_s"],
            "terminal": False,
            "images": images,
            "camera_times_s": frame_timestamps,
            "source_frame_indices": {camera: int(plan["frames"][camera].get("frame_index", 0)) for camera in CAMERAS},
            "source_frame_paths": {camera: plan["frames"][camera]["path"] for camera in CAMERAS},
            "state": state_value,
            "state_time_s": float(state_row.get("episode_time_s", actual_t)),
            "state_time_gap_s": round(state_gap_s, 6),
            "control_mode": ownership["control_mode"],
            "control_source": ownership["control_source"],
            "control_source_by_arm": ownership["control_source_by_arm"],
            "recording_segment": ownership["recording_segment"],
            "policy_time_s": ownership["policy_time_s"],
            "data_time_s": ownership["data_time_s"],
            "human_control_active": ownership["human_control_active"],
            "human_contact_observed": ownership["human_contact_observed"],
            "human_contact_signal_available": ownership["human_contact_signal_available"],
            "human_intervention_episode_observed": episode.get("human_intervention_observed"),
            "task_attempt_observed_episode": episode.get("task_attempt_observed"),
        }
        rows.append(row)
        kept_plans.append(plan)
    if rows:
        rows[-1]["terminal"] = True
    stat["samples_written"] = len(rows)
    # Save only referenced frame files, preserving source frame IDs in path names.
    for plan in kept_plans:
        for camera in CAMERAS:
            frame = plan["frames"][camera]
            relative = Path("images") / episode["episode_id"] / camera / f"{int(frame.get('frame_index',0)):07d}.jpg"
            if episode.get("raw_source_kind") == "directory":
                save_compact_path(episode["source_dir"] / frame["path"], output_root / relative, image_size)
            else:
                raw = episode["extracted_selected_frames"].get(frame["path"])
                if raw is None:
                    raise ValueError(f"selected archive image was not extracted: {frame['path']}")
                save_compact_bytes(raw, output_root / relative, image_size)
    return rows, stat


def old_splits(samples_path: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for row in read_jsonl(samples_path):
        pair = (str(row["split"]), str(row["session_id"]))
        previous = mapping.setdefault(row["episode_id"], pair)
        if previous != pair:
            raise ValueError(f"frozen old split/session inconsistency: {row['episode_id']}")
    return mapping


def iter_stream_archive(archive_path: Path, frame_callback: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    """Read a tar.gz once, extracting only frame members selected by callback."""
    frames: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    states: list[dict[str, Any]] | None = None
    selected: dict[str, bytes] = {}
    names_to_extract: set[str] = set()
    frame_lines: bytes | None = None
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name == "frames.jsonl":
                stream = tar.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(member.name)
                frame_lines = stream.read()
                frames = read_jsonl_bytes(frame_lines)
                names_to_extract = set(frame_callback(frames))
            elif member.name == "metadata.json":
                stream = tar.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(member.name)
                metadata = json.load(stream)
            elif member.name == "states.jsonl":
                stream = tar.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(member.name)
                states = read_jsonl_bytes(stream.read())
            elif member.name in names_to_extract:
                stream = tar.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(member.name)
                selected[member.name] = stream.read()
    # Keep empty/partial archived recordings in the episode audit; they contribute
    # no training observations when a required stream is absent.
    return metadata or {}, frames or [], states or [], selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-manifest", type=Path, default=Path("work/state-critic-20261028/episodes.json"))
    parser.add_argument("--old-samples", type=Path, default=Path("work/state-critic-20261028/dataset/samples.jsonl"))
    parser.add_argument("--hf-root", type=Path, default=Path("work/state-critic-20261029/hf-new-discarded"))
    parser.add_argument("--output", type=Path, default=Path("work/state-value-20261029/full_trajectory"))
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--max-gap-s", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--episode-ids", nargs="*", help="optional smoke subset; full manifest validation still runs")
    args = parser.parse_args()
    old_manifest_path, old_samples_path = args.old_manifest.resolve(), args.old_samples.resolve()
    hf_root, output_root = args.hf_root.resolve(), args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_size = (args.image_size, int(args.image_size * 3 / 4))

    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_map = old_splits(old_samples_path)
    if len(old_map) != int(old_manifest["episode_count"]):
        raise ValueError(f"frozen old split manifest has {len(old_map)} episodes, expected {old_manifest['episode_count']}")
    old_episodes = []
    for item in old_manifest["episodes"]:
        split_session = old_map.get(item["episode_id"])
        if split_session is None:
            raise ValueError(f"old episode lacks frozen split: {item['episode_id']}")
        split, session_id = split_session
        category = item.get("source_category", "unknown")
        source_dir = Path(item["source_dir"])
        duration = float(item["duration_s"])
        old_episodes.append({
            **item, "session_id": session_id, "split": split,
            "dataset_cohort": "frozen_manual_366", "source_category": category,
            "weak_demo": category == "kept", "raw_source_kind": "directory",
            "source_dir": source_dir, "episode_end_s": duration,
            "source_group": "teleop", "human_intervention_observed": None,
            "task_attempt_observed": None,
        })

    annotation_path = hf_root / "episodes.jsonl"
    hf_annotations = read_jsonl(annotation_path)
    requested_ids = set(args.episode_ids or [])
    hf_ids = {x["episode_id"] for x in hf_annotations}
    old_ids = {x["episode_id"] for x in old_episodes}
    overlap = sorted(hf_ids & old_ids)
    if overlap:
        raise ValueError(f"old and HF cohort episode ID overlap: {overlap[:10]}")
    new_episodes = []
    for annotation in hf_annotations:
        eid = annotation["episode_id"]
        date = eid[:8]
        if date not in NEW_DATE_SPLITS:
            raise ValueError(f"new HF date missing predeclared split: {eid}")
        archive_rel = annotation["source_archive"]
        archive_path = hf_root / archive_rel
        expected_hash = annotation.get("source_archive_sha256") or annotation.get("raw_recording", {}).get("archive_sha256")
        requested_for_run = not requested_ids or eid in requested_ids
        if requested_for_run and not archive_path.is_file():
            raise FileNotFoundError(f"HF archive missing: {archive_path}")
        archive_hash = sha256(archive_path) if requested_for_run else None
        if requested_for_run and expected_hash and archive_hash != expected_hash:
            raise ValueError(f"HF archive hash mismatch for {eid}")
        record = dict(annotation)
        record.update({
            "dataset_cohort": "hf_new_discarded_161", "split": NEW_DATE_SPLITS[date],
            "session_id": date, "source_category": "discarded", "weak_demo": False,
            "raw_source_kind": "archive", "source_archive_path": archive_path,
            "source_archive_relative": archive_rel, "source_archive_sha256_verified": archive_hash,
            "episode_end_s": float((annotation.get("raw_recording") or {}).get("duration_s")
                                    or (annotation.get("recording_statistics") or annotation.get("recording_stats") or {}).get("duration_s") or 0.0),
            "human_intervention_observed": annotation.get("human_intervention_observed"),
            "task_attempt_observed": annotation.get("task_attempt_observed"),
        })
        new_episodes.append(record)

    episodes = sorted([*old_episodes, *new_episodes], key=lambda x: x["episode_id"])
    if len(episodes) != 527:
        raise ValueError(f"expected 366+161=527 episodes, got {len(episodes)}")
    if args.episode_ids:
        wanted = requested_ids
        missing = wanted - {episode["episode_id"] for episode in episodes}
        if missing:
            raise ValueError(f"requested episode IDs not found: {sorted(missing)}")
        episodes = [episode for episode in episodes if episode["episode_id"] in wanted]
    samples: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    per_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    ownership_counts: dict[str, Counter[str]] = defaultdict(Counter)
    duration_by_episode: dict[str, float] = {}
    for index, episode in enumerate(episodes, start=1):
        if episode["raw_source_kind"] == "directory":
            source_dir: Path = episode["source_dir"]
            frame_rows = read_jsonl(source_dir / "frames.jsonl")
            state_rows = read_jsonl(source_dir / "states.jsonl")
            duration_hint = float(episode["episode_end_s"])
            planned, _, coverage = make_plan(frame_rows, duration_hint, args.sample_hz, args.max_gap_s)
            metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
            rows, ep_stats = encode_episode_samples(episode, planned, state_rows,
                int(metadata.get("start_monotonic_ns") or metadata.get("trigger_monotonic_ns") or 0),
                output_root, image_size, args.max_gap_s)
            source_frames = sum(1 for row in frame_rows if row.get("camera") in CAMERAS)
            frame_counts = {camera: sum(1 for row in frame_rows if row.get("camera") == camera) for camera in CAMERAS}
            source_kind = "directory"
            archive_hash = None
        else:
            archive_path: Path = episode["source_archive_path"]
            duration_hint = float(episode["episode_end_s"])
            plan_holder: dict[str, Any] = {}
            def select_members(frames: list[dict[str, Any]]) -> list[str]:
                planned, members, coverage = make_plan(frames, duration_hint, args.sample_hz, args.max_gap_s)
                plan_holder.update(planned=planned, coverage=coverage, frames=frames)
                return members
            metadata, frames, state_rows, selected_bytes = iter_stream_archive(archive_path, select_members)
            episode["extracted_selected_frames"] = selected_bytes
            if not frames:
                planned = []
                rows = []
                ep_stats = Counter({"episodes_skipped_missing_frames": 1})
                source_frames = 0
                frame_counts = {camera: 0 for camera in CAMERAS}
                coverage = {"skip_reason": "missing_frames"}
            elif not state_rows:
                planned = []
                rows = []
                ep_stats = Counter({"episodes_skipped_missing_states": 1})
                source_frames = len(frames)
                frame_counts = {camera: sum(1 for row in frames if row.get("camera") == camera) for camera in CAMERAS}
                coverage = {"skip_reason": "missing_states", "source_frame_count": len(frames)}
            elif not plan_holder.get("planned"):
                planned = []
                rows = []
                ep_stats = Counter({"episodes_skipped_camera_coverage": 1})
                source_frames = len(frames)
                frame_counts = {camera: sum(1 for row in frames if row.get("camera") == camera) for camera in CAMERAS}
                coverage = plan_holder.get("coverage", {"skip_reason": "camera_coverage"})
            else:
                planned = plan_holder["planned"]
                rows, ep_stats = encode_episode_samples(episode, planned, state_rows,
                    int(metadata.get("start_monotonic_ns") or metadata.get("trigger_monotonic_ns") or 0),
                    output_root, image_size, args.max_gap_s)
                frames = plan_holder["frames"]
                source_frames = len(frames)
                frame_counts = {camera: sum(1 for row in frames if row.get("camera") == camera) for camera in CAMERAS}
                coverage = plan_holder["coverage"]
            source_kind = "hf_archive"
            archive_hash = episode["source_archive_sha256_verified"]
            episode.pop("extracted_selected_frames", None)
        # Mark one final row terminal; do not assign any task outcome/reward here.
        if rows:
            rows[-1]["terminal"] = True
        samples.extend(rows)
        stats.update(ep_stats)
        stats["episodes_processed"] += 1
        stats[f"episodes_{episode['split']}"] += 1
        stats[f"samples_{episode['split']}"] += len(rows)
        per_split[episode["split"]][episode["source_category"]] += 1
        per_split[episode["split"]][f"weak_demo_{episode['weak_demo']}"] += 1
        per_cohort[episode["dataset_cohort"]]["episodes"] += 1
        per_cohort[episode["dataset_cohort"]]["samples"] += len(rows)
        if "common_end_t_s" in coverage and "common_start_t_s" in coverage:
            duration_by_episode[episode["episode_id"]] = float(coverage["common_end_t_s"] - coverage["common_start_t_s"])
        for row in rows:
            if row["human_control_active"] is not None:
                ownership_counts[episode["dataset_cohort"]][f"human_control_{row['human_control_active']}"] += 1
            if row["control_source"] is not None:
                ownership_counts[episode["dataset_cohort"]]["rows_with_control_source"] += 1
            if row["recording_segment"] is not None:
                ownership_counts[episode["dataset_cohort"]]["rows_with_recording_segment"] += 1
        episode_record = {key: value for key, value in episode.items()
                          if key not in {"source_dir", "source_archive_path", "raw_source_kind", "source_archive_path"}}
        episode_record["source_dir"] = str(episode.get("source_dir")) if episode.get("source_dir") else None
        episode_record["source_archive_path"] = str(episode.get("source_archive_path")) if episode.get("source_archive_path") else None
        episode_record["coverage"] = coverage
        episode_record["source_image_count_by_camera"] = frame_counts
        episode_record["source_image_count_total"] = source_frames
        episode_record["sample_count"] = len(rows)
        episode_record["recorded_owner_fields"] = [name for name in
            ("control_mode", "control_source", "control_source_by_arm", "recording_segment")
            if any(row.get(name) is not None for row in rows)]
        episode_record["source_archive_sha256_verified"] = archive_hash
        episode_rows.append(episode_record)
        if index % 10 == 0 or index == len(episodes):
            print(f"trajectory export episodes: {index}/{len(episodes)}; samples={len(samples)}", flush=True)

    # Separate atomic artifacts; no `outcome` or per-frame task reward field.
    samples.sort(key=lambda row: (row["episode_id"], row["t"]))
    episodes_path = output_root / "episodes.jsonl"
    samples_path = output_root / "samples.jsonl"
    for path, records in ((episodes_path, episode_rows), (samples_path, samples)):
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp.replace(path)
    old_manifest_hash = sha256(old_manifest_path)
    old_samples_hash = sha256(old_samples_path)
    annotation_hash = sha256(annotation_path)
    label_count_by_split = {}
    # Every row is self-supervised/temporal input; weak source curation is not a task outcome.
    for split in ("train", "val", "test"):
        label_count_by_split[split] = {"rows": sum(1 for x in samples if x["split"] == split),
                                       "weak_demo_rows": sum(1 for x in samples if x["split"] == split and x["weak_demo"])}
    audit = {
        "schema_version": 1,
        "dataset": "full-trajectory state-value inputs",
        "task": "put gear in carrier and pin in it",
        "label_semantics": "No task outcome/reward is assigned to trajectory frames. source_category and weak_demo are curation provenance only. The HF visual annotations remain episode-level provenance and are not propagated.",
        "source_cohorts": {"frozen_manual": 366, "hf_discarded_pinned": 161, "total_episodes": len(episode_rows)},
        "source_repo": HF_REPO, "source_revision": HF_REVISION,
        "old_episode_manifest_sha256": old_manifest_hash,
        "old_frozen_samples_sha256": old_samples_hash,
        "hf_annotation_file_sha256": annotation_hash,
        "split_policy_old": "Frozen per-episode split/session mapping from work/state-critic-20261028/dataset/samples.jsonl; whole old validation/test remain validation/test.",
        "split_policy_new": NEW_DATE_SPLITS,
        "split_counts_by_episode": {s: dict(Counter(x["split"] for x in episode_rows if x["split"] == s)) for s in ()},
        "episode_counts_by_split": dict(Counter(x["split"] for x in episode_rows)),
        "sample_counts_by_split": dict(Counter(x["split"] for x in samples)),
        "source_category_by_episode_and_split": {s: dict(per_split[s]) for s in per_split},
        "cohort_counts": {s: dict(c) for s, c in per_cohort.items()},
        "recorded_control_provenance_counts": {s: dict(c) for s, c in ownership_counts.items()},
        "sampling_hz": args.sample_hz,
        "image_size": list(image_size),
        "camera_order": list(CAMERAS),
        "state_order": [f"left_joint_{i+1}" for i in range(6)] + ["left_gripper"] + [f"right_joint_{i+1}" for i in range(6)] + ["right_gripper"],
        "max_camera_state_gap_s": args.max_gap_s,
        "sample_rows": len(samples),
        "unique_referenced_images": len({p for row in samples for p in row["images"]}),
        "target_field_present": False,
        "per_split_row_counts": label_count_by_split,
        "human_contact_limitation": "Raw per-frame ownership/control-source fields are retained where available. Physical human contact is not available as a sensor signal; only reviewed episode-level human_intervention_observed flags are preserved, and are not time-propagated as contact labels.",
        "hitl_limitations": "HITL policy/human/hold source and recording_segment are retained per sampled state when present. No transitions are synthesized across segments; the export is observations only.",
        "archive_hashes_by_episode": {x["episode_id"]: x.get("source_archive_sha256_verified")
                                       for x in episode_rows if x.get("source_archive_sha256_verified")},
        "episodes_sha256": sha256(episodes_path),
        "samples_sha256": sha256(samples_path),
        "stats": dict(stats),
        "timestamp_semantics": "t is median of the three selected camera episode_time_s values; state is nearest measured 14D row by host monotonic_ns to the median camera capture timestamp.",
    }
    # Fix keys that are empty by split when a category has no rows.
    audit["split_counts_by_episode"] = dict(Counter(x["split"] for x in episode_rows))
    audit_path = output_root / "audit.json"
    temp_audit = audit_path.with_suffix(".json.tmp")
    temp_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_audit.replace(audit_path)
    print(json.dumps({"output": str(output_root), "episodes": len(episode_rows), "samples": len(samples),
                      "splits": audit["split_counts_by_episode"], "rows": audit["sample_counts_by_split"],
                      "images": audit["unique_referenced_images"], "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
