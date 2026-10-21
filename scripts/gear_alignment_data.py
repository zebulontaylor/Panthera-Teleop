#!/usr/bin/env python3
"""Extract reviewed gear approaches and their weak demonstration-relative Z labels.

Run from the repository root with ``python scripts/gear_alignment_data.py``.
The output JSONL references original JPEGs and states without modifying recordings.
Labels describe tool motion relative to a reviewed pose; they are not calibrated
gear-to-opening distances. Use episode splits before fitting a model.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearest(rows: list[dict], times: list[int], target: int, max_skew_s: float) -> dict:
    """Reject stale receive timestamps; ties select the earlier sample."""
    if not times:
        raise ValueError("The observation stream is empty")
    right = bisect_left(times, target)
    index = min((i for i in (right - 1, right) if 0 <= i < len(times)),
                key=lambda i: (abs(times[i] - target), times[i]))
    if abs(times[index] - target) / 1e9 > max_skew_s:
        raise ValueError(f"Observation exceeds allowed timestamp skew at {target}")
    return rows[index]


def relative_tool_z(state: dict, gear_arm: str, carrier_arm: str) -> float:
    z = [float(state["arms"][arm]["tool_pose_base"]["position_m"][2])
         for arm in (gear_arm, carrier_arm)]
    if not all(math.isfinite(value) for value in z):
        raise ValueError("Recorded tool position contains a non-finite Z coordinate")
    return z[0] - z[1]


def extract_episode(path: Path, selection: dict, manifest: dict, fps: float = 10.0,
                    camera_max_skew_s: float = 0.04, state_max_skew_s: float = 0.01) -> list[dict]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    start, end, reference = (float(selection[key]) for key in ("start_s", "end_s", "reference_s"))
    if not all(math.isfinite(value) for value in (start, end, reference)) or not 0 <= start <= reference <= end:
        raise ValueError("Require 0 <= start_s <= reference_s <= end_s")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("id") != selection["id"] or metadata.get("label") != "kept" or not metadata.get("complete"):
        raise ValueError("Selection must identify a complete operator-kept episode")
    if metadata.get("source_kind") == "hitl" or metadata.get("context", {}).get("mode") == "hitl":
        raise ValueError("HITL clocks require a separate explicit selection contract")
    origin = int(metadata["start_monotonic_ns"])
    states = sorted(read_jsonl(path / "states.jsonl"), key=lambda row: row["monotonic_ns"])
    state_times = [row["monotonic_ns"] for row in states]
    frames = read_jsonl(path / "frames.jsonl")
    cameras = {camera: sorted((row for row in frames if row["camera"] == camera),
                             key=lambda row: row["monotonic_ns"]) for camera in manifest["cameras"]}
    camera_times = {camera: [row["monotonic_ns"] for row in rows] for camera, rows in cameras.items()}
    ref_ns = origin + round(reference * 1e9)
    ref_state = nearest(states, state_times, ref_ns, state_max_skew_s)
    reference_z = relative_tool_z(ref_state, manifest["gear_arm"], manifest["carrier_arm"])
    output = []
    for tick in range(math.floor((end - start) * fps + 1e-8) + 1):
        time_s = start + tick / fps
        timestamp = origin + round(time_s * 1e9)
        state = nearest(states, state_times, timestamp, state_max_skew_s)
        selected = {camera: nearest(rows, camera_times[camera], timestamp, camera_max_skew_s)
                    for camera, rows in cameras.items()}
        z_error = relative_tool_z(state, manifest["gear_arm"], manifest["carrier_arm"]) - reference_z
        output.append({
            "episode_id": selection["id"], "split": selection["split"], "phase": selection["phase"],
            "episode_path": str(path.resolve()), "time_s": time_s, "monotonic_ns": timestamp,
            "images": {camera: {"path": str((path / row["path"]).resolve()),
                                  "monotonic_ns": row["monotonic_ns"],
                                  "skew_s": (row["monotonic_ns"] - timestamp) / 1e9,
                                  "image_size_wh": [row["width"], row["height"]]}
                       for camera, row in selected.items()},
            "state_monotonic_ns": state["monotonic_ns"],
            "state_skew_s": (state["monotonic_ns"] - timestamp) / 1e9,
            "joints_rad": {arm: state["arms"][arm]["position_rad"] for arm in ("left", "right")},
            "gripper_rad": {arm: state["arms"][arm]["gripper"]["position"] for arm in ("left", "right")},
            "tool_pose_base": {arm: state["arms"][arm]["tool_pose_base"] for arm in ("left", "right")},
            "demonstration_relative_tool_z_error_m": z_error,
            "demonstration_left_tool_z_correction_m": -z_error,
            "reference_s": reference, "reference_state_monotonic_ns": ref_state["monotonic_ns"],
            "reference_relative_tool_z_m": reference_z,
            "label_kind": "weak_demo_reference", "physical_object_height_ground_truth": False,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/gear-alignment-demos.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kept")
    parser.add_argument("--output", type=Path, default=ROOT / "work/gear-alignment/samples.jsonl")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    ids = [row["id"] for row in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Episode IDs must appear once so a recording cannot cross splits")
    rows = [sample for selection in manifest["episodes"]
            for sample in extract_episode(args.data_root / selection["id"], selection, manifest, args.fps)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    report = {"samples": len(rows), "samples_by_split": dict(Counter(row["split"] for row in rows)),
              "episodes_by_split": dict(Counter(row["split"] for row in manifest["episodes"])),
              "max_camera_skew_s": max(abs(frame["skew_s"]) for row in rows for frame in row["images"].values()),
              "max_state_skew_s": max(abs(row["state_skew_s"]) for row in rows),
              "target": manifest["target_name"], "physical_object_height_ground_truth": False,
              "manifest": str(args.manifest.resolve()), "output": str(args.output.resolve())}
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
