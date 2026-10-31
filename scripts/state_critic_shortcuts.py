#!/usr/bin/env python3
"""Fixed-split diagnostic baselines for the frozen curation-proxy critic run.

These are confound checks, not task-success estimates. The proxy labels encode
curation (kept/discarded) and may reflect recording date, setup, or stationary
behavior. Test rows are evaluated once; no tuning is performed on them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "work/state-critic-20261028/runs/curation-proxy-seed19"
CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
SEED = 19


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metric_block(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auroc": float(roc_auc_score(y, p)) if len(y) and np.unique(y).size == 2 else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
    }


def score_splits(rows: list[dict], probabilities: np.ndarray) -> dict:
    result = {}
    for split in ("train", "val", "test"):
        indexes = [i for i, row in enumerate(rows)
                   if row.get("split") == split and row.get("outcome") in (0, 1)]
        y = np.asarray([int(rows[i]["outcome"]) for i in indexes], dtype=np.int64)
        p = probabilities[indexes]
        frame = metric_block(y, p)
        episodes: dict[str, list[int]] = {}
        for local, index in enumerate(indexes):
            episodes.setdefault(str(rows[index]["episode_id"]), []).append(local)
        ey, ep = [], []
        for local_indexes in episodes.values():
            labels = {int(y[j]) for j in local_indexes}
            if len(labels) == 1:
                ey.append(labels.pop())
                ep.append(float(np.mean(p[local_indexes])))
        aggregate = metric_block(np.asarray(ey, dtype=np.int64), np.asarray(ep, dtype=float))
        result[split] = {"n_frames": frame["n"], "positive_rate": frame["positive_rate"],
                         "frame_auroc": frame["auroc"], "frame_brier": frame["brier"],
                         "episodes": len(ey), "episode_aggregate": aggregate}
    return result


def session_blocks(rows: list[dict]) -> dict:
    blocks: dict[str, dict] = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        session = str(row.get("session_id", "unknown"))
        block = blocks.setdefault(session, {"frames": 0, "labeled_frames": 0,
                                             "frame_outcomes": {"0": 0, "1": 0, "unknown": 0},
                                             "episodes": {}})
        block["frames"] += 1
        outcome = int(row["outcome"])
        label_key = str(outcome) if outcome in (0, 1) else "unknown"
        block["frame_outcomes"][label_key] += 1
        if outcome in (0, 1):
            block["labeled_frames"] += 1
        episode = block["episodes"].setdefault(str(row["episode_id"]), outcome)
        if episode != outcome:
            raise ValueError(f"episode outcome changes in test session {session}")
    result = {}
    for session, block in sorted(blocks.items()):
        labels = list(block["episodes"].values())
        block["episode_outcomes"] = {
            "0": sum(value == 0 for value in labels),
            "1": sum(value == 1 for value in labels),
            "unknown": sum(value not in (0, 1) for value in labels),
        }
        block["episode_count"] = len(labels)
        del block["episodes"]
        result[session] = block
    return result


def fit_predict(features: np.ndarray, rows: list[dict]) -> np.ndarray:
    labels = np.asarray([int(row["outcome"]) for row in rows], dtype=np.int64)
    train = np.asarray([i for i, row in enumerate(rows)
                        if row.get("split") == "train" and labels[i] >= 0], dtype=int)
    known = labels[train]
    if len(train) == 0 or np.unique(known).size != 2:
        raise ValueError("train split must have labeled examples from both classes")
    scaler = StandardScaler().fit(features[train])
    train_x = scaler.transform(features[train])
    all_x = scaler.transform(features)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    model.fit(train_x, known)
    return model.predict_proba(all_x)[:, 1]


def rgb_mean_features(rows: list[dict], root: Path) -> np.ndarray:
    result = np.empty((len(rows), len(CAMERAS) * 3), dtype=np.float32)
    for i, row in enumerate(rows):
        paths = row.get("images")
        if not isinstance(paths, list) or len(paths) != len(CAMERAS):
            raise ValueError(f"row {i} must contain three ordered image paths")
        for camera_index, relpath in enumerate(paths):
            with Image.open(root / relpath) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            result[i, camera_index * 3:(camera_index + 1) * 3] = pixels.mean(axis=(0, 1)) / 255.0
    return result


def validate_splits(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("manifest is empty")
    episode_split: dict[str, str] = {}
    session_split: dict[str, str] = {}
    episode_label: dict[str, int] = {}
    for row in rows:
        episode = str(row["episode_id"])
        split = row["split"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"invalid split {split!r}")
        if episode in episode_split and episode_split[episode] != split:
            raise ValueError(f"episode crosses splits: {episode}")
        episode_split[episode] = split
        session = str(row.get("session_id", ""))
        if session and session in session_split and session_split[session] != split:
            raise ValueError(f"session crosses splits: {session}")
        if session:
            session_split[session] = split
        outcome = int(row["outcome"])
        if outcome in (0, 1):
            if episode in episode_label and episode_label[episode] != outcome:
                raise ValueError(f"episode label changes within episode: {episode}")
            episode_label[episode] = outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "work/state-critic-20261028/dataset")
    parser.add_argument("--feature-cache", type=Path,
                        default=ROOT / "work/state-critic-20261028/features.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "work/state-critic-20261028/diagnostics/shortcut-audit")
    parser.add_argument("--skip-visual-embeddings", action="store_true",
                        help="skip the visual-only logistic model on cached ResNet features")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.curated_proxy_snapshot.jsonl"
    provenance_path = run_dir / "snapshot-provenance.json"
    reference_path = run_dir / "metrics.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if manifest_hash != provenance["manifest_sha256"]:
        raise ValueError("frozen proxy manifest hash does not match snapshot provenance")
    rows = read_jsonl(manifest_path)
    validate_splits(rows)
    if any(row.get("label_source") != "curated_proxy" or not row.get("diagnostic_only")
           for row in rows):
        raise ValueError("expected diagnostic-only curated_proxy rows")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("manifest_sha256") != manifest_hash:
        raise ValueError("reference critic metrics were produced from a different manifest")
    cache_file = np.load(args.feature_cache, allow_pickle=False) if not args.skip_visual_embeddings else None
    if cache_file is not None:
        if str(cache_file["key"].item()) != reference.get("feature_cache_key"):
            raise ValueError("feature cache key does not match the frozen critic run")
        if cache_file["features"].shape[0] != len(rows):
            raise ValueError("feature cache row count does not match frozen manifest")

    results = {
        "warning": "Diagnostic only: targets are kept/discarded curation proxies, not verified task outcomes. High performance can reflect environment, recording date, or stationary/reset confounds and does not establish causality or shortcut absence.",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "feature_cache": None if args.skip_visual_embeddings else str(args.feature_cache.resolve()),
        "feature_cache_key": None if cache_file is None else str(cache_file["key"].item()),
        "split_policy": "Frozen train/val/test assignments from the curation-proxy run manifest; fit only on labeled train frames, evaluate once on val/test.",
        "fit": {"model": "StandardScaler + LogisticRegression", "C": 1.0,
                "max_iter": 1000, "random_state": SEED, "hyperparameter_selection": "none"},
        "rows": len(rows),
        "labeled_frames": int(sum(row["outcome"] in (0, 1) for row in rows)),
        "test_session_blocks": session_blocks(rows),
        "uncertainty_note": "Episode aggregation and episode bootstrap do not address correlation between episodes recorded in the same session block. Test has few blocks; two blocks may be single-class, so block-bootstrap AUROC intervals would be unstable. Treat these as descriptive diagnostics, not independent-sample confidence bounds.",
        "features": {},
    }

    rgb_features = rgb_mean_features(rows, args.dataset_root.resolve())
    rgb_probability = fit_predict(rgb_features, rows)
    results["features"]["per_view_mean_rgb"] = {
        "description": "Current image mean R,G,B per camera in overhead, wrist_port2, wrist_port3 order.",
        "dimension": int(rgb_features.shape[1]),
        "metrics": score_splits(rows, rgb_probability),
    }

    state_features = np.asarray([row["state"] for row in rows], dtype=np.float32)
    state_probability = fit_predict(state_features, rows)
    state_scores = score_splits(rows, state_probability)
    results["features"]["state_only_C1_recomputed"] = {
        "description": "Same 14D state-only logistic baseline as the original report; recomputed for episode aggregation.",
        "dimension": int(state_features.shape[1]),
        "metrics": state_scores,
        "existing_report_frame_metrics": reference["diagnostic_baselines"]["state_only"],
    }

    if cache_file is not None:
        visual = np.asarray(cache_file["features"][:, :3 * 512], dtype=np.float32)
        visual_probability = fit_predict(visual, rows)
        results["features"]["resnet_current_visual_only"] = {
            "description": "Frozen ResNet18 current-image embeddings only (first 3 x 512 features); excludes state, history, and current-minus-history features.",
            "dimension": int(visual.shape[1]),
            "metrics": score_splits(rows, visual_probability),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "features": {
        name: {split: {"auroc": values["frame_auroc"], "brier": values["frame_brier"],
                       "episode_auroc": values["episode_aggregate"]["auroc"],
                       "episode_brier": values["episode_aggregate"]["brier"]}
               for split, values in item["metrics"].items()}
        for name, item in results["features"].items()}}, indent=2))


if __name__ == "__main__":
    main()
