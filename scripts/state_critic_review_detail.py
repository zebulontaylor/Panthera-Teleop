#!/usr/bin/env python3
"""Render dense synchronized wrist-camera pages for manual state-critic review."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work/state-critic-20261028/episodes.json"
OUTROOT = ROOT / "work/state-critic-20261028/detail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_id")
    ap.add_argument("--start", type=float, help="window start in seconds (default: max(0,duration-10))")
    ap.add_argument("--end", type=float, help="window end in seconds (default: episode duration)")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    ep = next((e for e in manifest["episodes"] if e["episode_id"] == args.episode_id), None)
    if ep is None:
        ap.error(f"episode not present in frozen manifest: {args.episode_id}")
    source = Path(ep["source_dir"])
    duration = float(ep["duration_s"])
    end = duration if args.end is None else min(float(args.end), duration)
    start = max(0.0, end - 10.0) if args.start is None else max(0.0, float(args.start))
    if end <= start:
        ap.error("--end must be greater than --start")

    frames = [json.loads(line) for line in (source / "frames.jsonl").open() if line.strip()]
    cams = {c: sorted((f for f in frames if f.get("camera") == c), key=lambda f: f["episode_time_s"])
            for c in ("wrist_port2", "wrist_port3")}
    times = []
    t = start
    while t <= end + 1e-8:
        times.append(round(t, 3))
        t += .5

    outdir = OUTROOT / args.episode_id
    outdir.mkdir(parents=True, exist_ok=True)
    provenance = []
    font = ImageFont.load_default()
    for page_num, offset in enumerate(range(0, len(times), 3), 1):
        batch = times[offset:offset + 3]
        canvas = Image.new("RGB", (1280, 1500), "white")
        draw = ImageDraw.Draw(canvas)
        for row, target in enumerate(batch):
            for col, cam in enumerate(("wrist_port2", "wrist_port3")):
                nearest = min(cams[cam], key=lambda f: abs(float(f["episode_time_s"]) - target))
                img_path = source / nearest["path"]
                img = Image.open(img_path).convert("RGB")
                x, y = col * 640, row * 500
                canvas.paste(img, (x, y + 20))
                label = f"target {target:.2f}s | {cam} @ {float(nearest['episode_time_s']):.3f}s"
                draw.text((x + 4, y + 4), label, fill="black", font=font)
                provenance.append({"page": page_num, "row": row + 1, "target_time_s": target,
                                   "camera": cam, "actual_time_s": float(nearest["episode_time_s"]),
                                   "path": str(img_path.relative_to(ROOT))})
        page_path = outdir / f"page-{page_num}.jpg"
        canvas.save(page_path, quality=92)
    (outdir / "frame-provenance.json").write_text(json.dumps({"episode_id": args.episode_id,
        "window_start_s": start, "window_end_s": end, "spacing_s": .5,
        "pages": math.ceil(len(times) / 3), "frames": provenance}, indent=2) + "\n")
    print(outdir)


if __name__ == "__main__":
    main()
