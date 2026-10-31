#!/usr/bin/env python3
"""Prepare blind visual-review contact sheets for state-critic labels.

Reads complete raw episodes from data/{kept,review,discarded}, aligns camera
frames by episode time, and writes reviewer-facing contact sheets and manifests
under work/state-critic-20261028 by default. It never edits source recordings.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
SHARD_COUNT = 4
CELL_W, CELL_H = 320, 240
PANEL_GAP = 8
ROW_LABEL_H = 22
CAMERA_LABEL_H = 20
ROW_PAD = 6


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def nearest(rows: list[dict[str, Any]], target_s: float) -> dict[str, Any]:
    return min(rows, key=lambda row: (abs(float(row["episode_time_s"]) - target_s),
                                      float(row["episode_time_s"])))


def choose_times(duration_s: float) -> list[tuple[str, float]]:
    # Three internal trajectory quartiles plus four terminal windows give seven
    # views, spanning the attempt while emphasizing its outcome.
    choices = [("q25", duration_s * 0.25),
               ("q50", duration_s * 0.50),
               ("q75", duration_s * 0.75),
               ("last10", max(0.0, duration_s - 10.0)),
               ("last5", max(0.0, duration_s - 5.0)),
               ("last2", max(0.0, duration_s - 2.0)),
               ("last", duration_s)]
    return choices


def make_sheet(episode: dict[str, Any], output_root: Path) -> None:
    episode_dir = Path(episode["source_dir"])
    frames = read_jsonl(episode_dir / "frames.jsonl")
    by_camera = {camera: sorted(
        [row for row in frames if row.get("camera") == camera],
        key=lambda row: float(row["episode_time_s"])) for camera in CAMERAS}
    missing = [camera for camera, rows in by_camera.items() if not rows]
    if missing:
        raise ValueError(f"{episode['episode_id']}: missing camera frames {missing}")

    max_s = max(float(row["episode_time_s"]) for rows in by_camera.values() for row in rows)
    duration_s = max_s
    selections = choose_times(duration_s)
    font = ImageFont.load_default()
    row_h = ROW_LABEL_H + CAMERA_LABEL_H + CELL_H + ROW_PAD
    width = PANEL_GAP + len(CAMERAS) * (CELL_W + PANEL_GAP)
    height = PANEL_GAP + len(selections) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for r, (tag, target_s) in enumerate(selections):
        y = PANEL_GAP + r * row_h
        draw.text((PANEL_GAP, y), f"{tag}  target {target_s:.2f}s", fill="black", font=font)
        panel_y = y + ROW_LABEL_H
        for col, camera in enumerate(CAMERAS):
            x = PANEL_GAP + col * (CELL_W + PANEL_GAP)
            row = nearest(by_camera[camera], target_s)
            image_path = episode_dir / row["path"]
            with Image.open(image_path) as source:
                image = source.convert("RGB").resize((CELL_W, CELL_H), Image.Resampling.LANCZOS)
            camera_time = float(row["episode_time_s"])
            draw.text((x, panel_y), f"{camera}  t={camera_time:.2f}s", fill="black", font=font)
            sheet.paste(image, (x, panel_y + CAMERA_LABEL_H))

    destination = output_root / episode["sheet_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=90, optimize=True)


def inventory(data_root: Path) -> list[dict[str, Any]]:
    episodes = []
    for category in ("kept", "review", "discarded"):
        category_root = data_root / category
        if not category_root.exists():
            continue
        for episode_dir in sorted(p for p in category_root.iterdir() if p.is_dir()):
            metadata_path = episode_dir / "metadata.json"
            frames_path = episode_dir / "frames.jsonl"
            if not metadata_path.is_file() or not frames_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("complete") is not True:
                continue
            frames = read_jsonl(frames_path)
            times = [float(row["episode_time_s"]) for row in frames if "episode_time_s" in row]
            if not times:
                continue
            episode_id = str(metadata.get("id", episode_dir.name))
            episodes.append({
                "episode_id": episode_id,
                "source_category": category,
                "source_dir": str(episode_dir.resolve()),
                "duration_s": round(max(times), 6),
                "sheet_path": f"contact_sheets/{episode_id}.jpg",
            })
    return sorted(episodes, key=lambda episode: episode["episode_id"])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("work/state-critic-20261028"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    episodes = inventory(data_root)
    manifest = {
        "schema_version": 1,
        "source_root": str(data_root),
        "episode_count": len(episodes),
        "selection": "complete episodes only; sorted by episode_id; source categories are provenance and are not rendered on contact sheets",
        "camera_order": list(CAMERAS),
        "contact_sheet_rows": ["q25", "q50", "q75", "last10", "last5", "last2", "last"],
        "episodes": episodes,
    }
    write_json(output_root / "episodes.json", manifest)

    # Stratify by source category so each reviewer shard receives a near-equal
    # share of each available source category. Shards themselves omit category.
    shards: list[list[dict[str, str]]] = [[] for _ in range(SHARD_COUNT)]
    for category in ("kept", "review", "discarded"):
        members = [episode for episode in episodes if episode["source_category"] == category]
        for index, episode in enumerate(members):
            shard_index = index % SHARD_COUNT
            shards[shard_index].append({
                "episode_id": episode["episode_id"],
                "sheet_path": episode["sheet_path"],
            })
    for index, shard in enumerate(shards):
        shard.sort(key=lambda item: item["episode_id"])
        write_json(output_root / f"label-shard-{index + 1}.json", {
            "schema_version": 1,
            "shard": index + 1,
            "episode_count": len(shard),
            "episodes": shard,
        })

    failures = []
    finished = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(make_sheet, episode, output_root): episode for episode in episodes}
        for future in as_completed(futures):
            episode = futures[future]
            try:
                future.result()
                finished += 1
                if finished == 1 or finished % 25 == 0 or finished == len(episodes):
                    print(f"contact sheets ready: {finished}/{len(episodes)} ({episode['episode_id']})", flush=True)
            except Exception as exc:  # Keep going so the manifest exposes all failures.
                failures.append({"episode_id": episode["episode_id"], "error": str(exc)})
                print(f"failed {episode['episode_id']}: {exc}", flush=True)
    write_json(output_root / "preparation-errors.json", {"errors": failures})
    print(json.dumps({
        "episodes": len(episodes),
        "sheets_written": finished,
        "errors": len(failures),
        "shard_sizes": [len(shard) for shard in shards],
        "output_root": str(output_root),
    }, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
