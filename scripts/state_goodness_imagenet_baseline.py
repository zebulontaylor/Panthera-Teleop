#!/usr/bin/env python3
"""Train-only pooled ImageNet-feature baseline for audited state goodness labels.

The classifier is fit only on explicit train support/mistake labels. It has no
timestamp, episode identity, state vector, source-category or ownership input.
Use after the evidence manifest and evaluation pairs are frozen; do not select a
model from test metrics. Outputs are diagnostic relative-support scores, not
calibrated task-success probabilities.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50

from state_goodness_train import evaluation_rows, load_frames


def parse_triplet(refs):
    if isinstance(refs, dict):
        return [refs[k] for k in ("overhead", "wrist_port2", "wrist_port3")]
    if len(refs) != 3:
        raise ValueError("every observation must contain three ordered camera images")
    return refs


def encode_observations(rows, root, model, device, image_size, batch_size):
    tf = T.Compose([
        T.Resize((image_size, image_size)), T.ToTensor(),
        T.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    root = Path(root)
    def get_image(ref, row_root=None):
        path = (Path(row_root) if row_root else root) / ref
        with Image.open(path) as im:
            return tf(im.convert("RGB"))

    feats = []
    for start in range(0, len(rows), batch_size):
        sub = rows[start:start + batch_size]
        batch = torch.stack([
            torch.stack([get_image(ref, row.get("image_root")) for ref in parse_triplet(row["images"])])
            for row in sub
        ]).to(device)
        with torch.inference_mode():
            z = model(batch.reshape(-1, 3, image_size, image_size))
            # Preserve ordered camera channels as a fixed concat (6144-D); no learned fusion.
            z = z.reshape(len(sub), -1)
        feats.append(z.cpu().numpy().astype(np.float32))
        if start == 0 or start + len(sub) == len(rows) or start % (batch_size * 50) == 0:
            print(json.dumps({"encoded_rows": start + len(sub), "total_rows": len(rows)}), flush=True)
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float32)


def sample_weights(rows):
    """Equal total weight per class, then session, episode, and observation."""
    groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for i, row in enumerate(rows):
        groups[int(row["_label"])][str(row.get("session_id", row["episode_id"]))][str(row["episode_id"])].append(i)
    if len(groups) != 2:
        raise ValueError("train support/mistake labels must include both classes")
    result = np.zeros(len(rows), dtype=np.float64)
    n_classes = len(groups)
    for label, sessions in groups.items():
        for episodes in sessions.values():
            for indices in episodes.values():
                weight = 1.0 / (n_classes * len(sessions) * len(episodes) * len(indices))
                result[indices] = weight
    return result


def score_metrics(y, p, rows):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    result = {
        "n": int(len(y)),
        "class_counts": {str(k): int((y == k).sum()) for k in (0, 1)},
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, p >= .5)) if len(set(y)) == 2 else None,
        "bce": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "interpretation": "Diagnostic classifier of demonstrated-state support versus reviewed mistake evidence; not a calibrated success probability.",
    }
    session_metrics = []
    for session in sorted({str(r.get("session_id", r["episode_id"])) for r in rows}):
        ix = [i for i, r in enumerate(rows) if str(r.get("session_id", r["episode_id"])) == session]
        yy, pp = y[ix], p[ix]
        if len(set(yy)) == 2:
            session_metrics.append({"session_id": session, "n": len(ix), "roc_auc": float(roc_auc_score(yy, pp))})
    result["two_class_session_macro_roc_auc"] = float(np.mean([m["roc_auc"] for m in session_metrics])) if session_metrics else None
    result["two_class_sessions"] = session_metrics
    return result


def read_pair_rows(paths, split):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError(f"pair split must be exactly {split}: {path}")
            for side in ("a", "b"):
                refs = row["observation_" + side]["current"]
                row["observation_" + side]["current"] = parse_triplet(refs)
            rows.append(row)
    return rows


def pair_metrics(rows, classifier, scaler, encoder, device, root, image_size, batch_size):
    known = [r for r in rows if r["preference"] in ("a_better", "b_better", "equal")]
    unknown = len(rows) - len(known)
    observations = []
    for row in known:
        for side in ("a", "b"):
            observations.append({"images": row["observation_" + side]["current"], "image_root": str(root)})
    x = encode_observations(observations, root, encoder, device, image_size, batch_size)
    p = classifier.predict_proba(scaler.transform(x))[:, 1] if len(x) else np.zeros((0,), dtype=float)
    details = []
    for i, row in enumerate(known):
        delta = float(p[2 * i] - p[2 * i + 1])
        preference = row["preference"]
        correct = None if preference == "equal" else (delta > 0 if preference == "a_better" else delta < 0)
        details.append({
            "pair_id": row.get("pair_id"),
            "episode_id_a": row.get("episode_id_a"),
            "episode_id_b": row.get("episode_id_b"),
            "session_id": row.get("session_id"),
            "preference": preference,
            "progress_component": row.get("progress_component"),
            "within_episode": row.get("episode_id_a") == row.get("episode_id_b"),
            "score_delta_a_minus_b": delta,
            "directional_correct": correct,
            "equal_absolute_gap": abs(delta) if preference == "equal" else None,
        })
    directional = [r for r in details if r["directional_correct"] is not None]
    equal = [r["equal_absolute_gap"] for r in details if r["equal_absolute_gap"] is not None]
    sessions = collections.defaultdict(list)
    for row in directional:
        sessions[str(row.get("session_id"))].append(float(row["directional_correct"]))
    return {
        "all_rows": len(rows), "known_rows": len(known), "unknown_rows": unknown,
        "directional_pairs": len(directional),
        "directional_accuracy": float(np.mean([r["directional_correct"] for r in directional])) if directional else None,
        "session_macro_directional_accuracy": float(np.mean([np.mean(v) for v in sessions.values()])) if sessions else None,
        "directional_sessions": len(sessions),
        "equal_pairs": len(equal), "equal_mean_absolute_score_gap": float(np.mean(equal)) if equal else None,
        "within_episode_directional_accuracy": float(np.mean([r["directional_correct"] for r in directional if r["within_episode"]])) if any(r["within_episode"] for r in directional) else None,
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--clean-kept-confirmed", action="store_true")
    ap.add_argument("--train-preferences", type=Path, action="append", default=[])
    ap.add_argument("--val-preferences", type=Path, action="append", default=[])
    ap.add_argument("--test-preferences", type=Path, action="append", default=[])
    ap.add_argument("--preference-image-root", type=Path, default=Path("."))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--frames-per-episode", type=int, default=8)
    ap.add_argument("--logistic-c", type=float, default=1.0)
    args = ap.parse_args()
    if args.image_size < 32 or args.batch_size < 1 or args.frames_per_episode < 1:
        ap.error("image size, batch size and frames per episode must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    rows, episodes = load_frames(args.manifest, args.clean_kept_confirmed)
    all_frames = []
    indices = {}
    for split in ("train", "val", "test"):
        selected = evaluation_rows(episodes, split, args.frames_per_episode)
        indices[split] = selected
        all_frames.extend(selected)
    if not indices["train"]:
        raise ValueError("no train labels available")
    # Freeze ImageNet only; the three-view mean is the sole feature transform.
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.to(device).eval()
    unique_rows = []
    seen = set()
    for row in all_frames:
        key = (str(row.get("image_root", args.dataset_root)), tuple(parse_triplet(row["images"])))
        if key not in seen:
            seen.add(key); unique_rows.append(row)
    x_unique = encode_observations(unique_rows, args.dataset_root, net, device, args.image_size, args.batch_size)
    xmap = {(str(r.get("image_root", args.dataset_root)), tuple(parse_triplet(r["images"]))): x_unique[i] for i, r in enumerate(unique_rows)}
    features = {split: np.stack([xmap[(str(r.get("image_root", args.dataset_root)), tuple(parse_triplet(r["images"]))) ] for r in subset]) for split, subset in indices.items()}
    train_rows = indices["train"]
    y_train = np.asarray([int(r["_label"]) for r in train_rows])
    if len(set(y_train)) != 2:
        raise ValueError("train labels must contain demonstrated support and reviewed mistake examples")
    scaler = StandardScaler().fit(features["train"])
    model = LogisticRegression(C=args.logistic_c, max_iter=2000, random_state=19)
    model.fit(scaler.transform(features["train"]), y_train, sample_weight=sample_weights(train_rows))
    report = {
        "baseline": "frozen ImageNet ResNet-50 (IMAGENET1K_V2), ordered concatenation of three per-camera global features, train-only logistic regression",
        "feature_definition": {"backbone": "ResNet-50 IMAGENET1K_V2", "per_camera_dim": 2048, "camera_order": ["overhead", "wrist_port2", "wrist_port3"], "combined_dim": 6144, "fusion": "ordered concatenation; no learned fusion", "standardization": "StandardScaler fitted only on training features"},
        "objective": "Demonstrated clean support versus explicitly reviewed mistake terminal states; score is not calibrated task-success probability.",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selection_usage": "Diagnostic baseline only; features/classifier fit with train labels only. Validation/test metrics must not be used for candidate selection.",
        "counts": {split: {"examples": len(subset), "episodes": len({str(r['episode_id']) for r in subset}), "sessions": len({str(r.get('session_id', r['episode_id'])) for r in subset}), "class_counts": {str(k): sum(r['_label'] == k for r in subset) for k in (0., 1.)}} for split, subset in indices.items()},
        "scores": {},
    }
    for split, subset in indices.items():
        if not subset:
            report["scores"][split] = {"n": 0, "message": "no labeled examples"}
            continue
        prob = model.predict_proba(scaler.transform(features[split]))[:, 1]
        report["scores"][split] = score_metrics([int(r["_label"]) for r in subset], prob, subset)
    all_pair_details = {}
    for split, pair_paths in (("train", args.train_preferences), ("val", args.val_preferences), ("test", args.test_preferences)):
        if pair_paths:
            pair_rows = read_pair_rows(pair_paths, split)
            all_pair_details[split] = pair_metrics(pair_rows, model, scaler, net, device, args.preference_image_root, args.image_size, args.batch_size)
    if all_pair_details:
        report["preference_diagnostics"] = all_pair_details
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
