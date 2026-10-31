"""Small contract tests for the offline state critic data pipeline."""

import importlib.util
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_state(t_ns):
    arm = {"position_rad": [0.1] * 6, "gripper": {"position": 0.2}}
    return {"monotonic_ns": t_ns, "arms": {"left": arm, "right": arm}}


def test_dataset_emits_ordered_trainer_manifest_with_date_session_split(tmp_path, monkeypatch):
    dataset = load_script("state_critic_dataset")
    episode_id = "20261025-demo-001"
    source = tmp_path / "episode"
    source.mkdir()
    start_ns = 10_000_000_000
    (source / "metadata.json").write_text(json.dumps({"start_monotonic_ns": start_ns}))
    frame_rows = []
    for camera_index, camera in enumerate(dataset.CAMERAS):
        for second in (0, 1, 2):
            rel = f"frames/{camera}/{second}.jpg"
            frame_path = source / rel
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 8), (camera_index * 50, second * 20, 5)).save(frame_path)
            frame_rows.append({"camera": camera, "monotonic_ns": start_ns + second * 1_000_000_000,
                               "frame_index": second, "path": rel})
    write_jsonl(source / "frames.jsonl", frame_rows)
    write_jsonl(source / "states.jsonl", [make_state(start_ns + t * 1_000_000_000) for t in (1, 2)])

    work_root = tmp_path / "work"
    work_root.mkdir()
    write_jsonl(work_root / "reviewer.jsonl", [{"episode_id": episode_id, "outcome": "success",
                                                 "confidence": "high"}])
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps({"episodes": [{"episode_id": episode_id,
                                                  "source_dir": str(source), "duration_s": 2.2}]}))
    output = tmp_path / "dataset"
    monkeypatch.setattr("sys.argv", ["state_critic_dataset", "--work-root", str(work_root),
                                      "--manifest", str(manifest), "--output", str(output)])
    dataset.main()

    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["split"] == "val" and row["session_id"] == "20261025" for row in rows)
    assert all(row["label_source"] == "reviewed" and row["outcome"] == 1 for row in rows)
    assert rows[0]["images"] == [f"images/{episode_id}/{camera}/0000001.jpg"
                                  for camera in dataset.CAMERAS]
    assert rows[0]["history_images"] == [f"images/{episode_id}/{camera}/0000000.jpg"
                                          for camera in dataset.CAMERAS]
    assert len(rows[0]["images"]) == len(rows[0]["history_images"]) == 3


def test_conflicting_and_pending_labels_remain_unknown():
    dataset = load_script("state_critic_dataset")
    def record(outcome, status="complete", confidence="high"):
        return {"_normalized_outcome": outcome, "_confidence": confidence,
                "review_status": status}

    assert dataset.resolve_label([record(1), record(0)]) == {
        "outcome": -1, "confidence": 0.0, "label_source": "reviewed",
        "status": "conflicting_outcomes"}
    assert dataset.resolve_label([record(1, status="pending")]) == {
        "outcome": -1, "confidence": 0.0, "label_source": "reviewed",
        "status": "pending_review"}


def test_one_class_split_still_reports_brier_score():
    trainer = load_script("state_critic_train")
    result = trainer.metrics([1, 1], [0.8, 0.6])
    assert result["n"] == 2
    assert result["auroc"] is None
    assert result["average_precision"] is None
    assert abs(result["brier"] - (0.2**2 + 0.4**2) / 2) < 1e-12


def test_feature_cache_recomputes_when_identity_changes(tmp_path):
    import torch

    trainer = load_script("state_critic_train")
    image_paths = []
    for index in range(3):
        path = tmp_path / f"view{index}.jpg"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
        image_paths.append(path.name)

    class CountingBackbone:
        calls = 0

        def __call__(self, batch):
            self.calls += 1
            return torch.full((len(batch), 512), float(self.calls))

    backbone = CountingBackbone()
    row = {"images": image_paths}
    cache = tmp_path / "features.npz"
    transform = lambda image: torch.zeros((3, 2, 2))
    first = trainer.image_features([row], tmp_path, backbone, transform, "cpu", 1,
                                   cache, "identity-before-image-change")
    repeated = trainer.image_features([row], tmp_path, backbone, transform, "cpu", 1,
                                      cache, "identity-before-image-change")
    assert backbone.calls == 1
    assert (first == repeated).all()

    Image.new("RGB", (8, 8), (255, 0, 0)).save(tmp_path / image_paths[0])
    trainer.image_features([row], tmp_path, backbone, transform, "cpu", 1,
                           cache, "identity-after-image-change")
    assert backbone.calls == 2
