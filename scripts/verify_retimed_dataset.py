#!/usr/bin/env python3
"""Independently check a derived retimed LeRobot export against its source dataset.

Does not call the retiming solver or reuse its reported trajectory metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
NUMERIC = ("observation.state", "action", "observation.velocity", "observation.effort")


def json_read(path):
    return json.loads(Path(path).read_text())


def episode_rows(root):
    """Stream full episode rows without retaining the image corpus in memory."""
    current, rows = None, []
    for path in sorted((Path(root) / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                ep = int(row["episode_index"])
                if current is not None and ep != current:
                    yield current, rows
                    rows = []
                current = ep
                rows.append(row)
    if rows:
        yield current, rows


def interpolate(path, frames):
    """Reproduce one common linear path coordinate for every channel."""
    frames = np.asarray(frames, dtype=np.float64)
    assert np.isfinite(frames).all()
    assert frames.min() >= -1e-9 and frames.max() <= len(path) - 1 + 1e-9
    frames = np.clip(frames, 0, len(path) - 1)
    left = np.floor(frames).astype(int)
    right = np.minimum(left + 1, len(path) - 1)
    weight = frames - left
    return path[left] * (1 - weight[:, None]) + path[right] * weight[:, None]


def check_clock(source_frames, action_frames, interval_scales, protected, fps):
    """Validate the actual map, including exact protected interval durations."""
    scales = np.asarray(interval_scales, dtype=np.float64)
    protected = np.asarray(protected, dtype=bool)
    assert scales.shape == protected.shape
    assert np.isfinite(scales).all() and np.all(scales >= 1 - 1e-9)
    np.testing.assert_allclose(scales[protected], 1, atol=1e-10, rtol=0)
    output_knots = np.r_[0., np.cumsum(scales)] / fps
    knots = np.arange(len(scales) + 1, dtype=float)
    assert np.all(np.diff(output_knots) > 0)
    sf, af = np.asarray(source_frames), np.asarray(action_frames)
    assert np.all(np.diff(sf) > 0) and np.all(af > sf)
    count = len(sf)
    # The export can discard less than one final tick, but must never hold the
    # endpoint artificially or extend beyond its selected source trajectory.
    np.testing.assert_allclose(sf, np.interp(np.arange(count) / fps, output_knots, knots), atol=2e-7, rtol=0)
    np.testing.assert_allclose(af, np.interp((np.arange(count) + 1) / fps, output_knots, knots), atol=2e-7, rtol=0)
    assert count / fps <= output_knots[-1] + 1e-8
    assert output_knots[-1] - count / fps < 1 / fps + 1e-8
    return float(output_knots[-1]), float(np.sum(protected) / fps)


def check_camera_provenance(mapping, original, camera, fps, source_count=None):
    for key, value in original.items():
        if key.startswith(camera + ".") and key != camera + ".offset_s":
            assert mapping[key] == value
    original_offset = float(original[camera + ".offset_s"])
    assert abs(mapping[camera + ".original_offset_s"] - original_offset) < 1e-10
    expected = float(original["source_episode_time_s"]) + original_offset - mapping["source_episode_time_s"]
    assert abs(mapping[camera + ".offset_s"] - expected) < 1e-10
    extra_bound = .5 / fps
    if source_count is not None:
        clamped = int(np.floor(mapping["source_state_frame"] + .5)) >= source_count
        assert mapping["image_source_terminal_clamped"] == clamped
        extra_bound = 1 / fps if clamped else .5 / fps
        assert abs(mapping["camera_alignment_extra_bound_s"] - extra_bound) < 1e-10
    assert abs(expected) <= abs(original_offset) + extra_bound + 1e-7


def expected_precision_label(phases, source_time, multiplier):
    active = [p for p in phases if p["start_s"] <= source_time < p["end_s"]]
    weighted = False
    for phase in active:
        confidence = str(phase.get("confidence", "")).lower()
        if confidence == "visual":
            confidence = str(phase.get("visual_reviewer_confidence", "medium")).lower()
        weighted |= confidence in {"high", "medium", "medium-high"}
    return dict(gear_insertion=any(p["phase"] == "gear_insertion" for p in active),
                pin_insertion=any(p["phase"] == "pin_insertion" for p in active),
                loss_weight=float(multiplier if weighted else 1.),
                annotation_scope=";".join(sorted({p.get("scope", "visual_phase_window") for p in active})),
                confidence=";".join(sorted({str(p.get("confidence", "unknown")) for p in active})))


def derivative_summary(path, fps):
    velocity = np.diff(np.asarray(path, dtype=np.float64), axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    summary = {}
    for kind, values in (("velocity", velocity), ("acceleration", acceleration)):
        for name, dimensions in (("joint", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]), ("gripper", [6, 13])):
            maxima = np.max(np.abs(values[:, dimensions]), axis=1) if len(values) else np.zeros(0)
            summary[f"{name}_{kind}"] = {"max": float(maxima.max()) if len(maxima) else 0.,
                "p95": float(np.percentile(maxima, 95)) if len(maxima) else 0.,
                "p99": float(np.percentile(maxima, 99)) if len(maxima) else 0.}
    return summary


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source, dataset, output=None, native=True):
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    assert source != dataset
    output = Path(output) if output else dataset / "processing/verification.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace verification report: {output}")
    manifest = json_read(dataset / "processing/manifest.json")
    source_manifest = json_read(source / "processing/manifest.json")
    info = json_read(dataset / "meta/info.json")
    source_info = json_read(source / "meta/info.json")
    cameras = tuple(k.removeprefix("observation.images.") for k, feature in info["features"].items() if feature["dtype"] == "image")
    assert cameras
    fps = float(info["fps"])
    assert fps == source_info["fps"] == 20
    assert info["total_episodes"] == source_info["total_episodes"]
    assert len(manifest["episodes"]) == len(source_manifest["episodes"]) == info["total_episodes"]
    for key in NUMERIC:
        assert info["features"][key]["shape"] == [14]
    precision_rows = pq.read_table(dataset / "meta/precision_weights.parquet").to_pylist()
    source_precision_path = source / "meta/precision.json"
    source_precision = json_read(source_precision_path) if source_precision_path.exists() else {}
    precision_multiplier = float(source_precision.get("suggested_precision_multiplier", 3.))
    totals, images, episodes = 0, 0, []
    sums = {key: [] for key in NUMERIC}
    native_samples = []
    src_iterator, out_iterator = iter(episode_rows(source)), iter(episode_rows(dataset))
    for expected_episode in range(info["total_episodes"]):
        se, sr = next(src_iterator)
        oe, rows = next(out_iterator)
        assert se == oe == expected_episode
        ep = manifest["episodes"][oe]
        src_ep = source_manifest["episodes"][se]
        assert ep["id"] == src_ep["id"]
        assert ep["source_hashes"] == src_ep["source_hashes"]
        sidecar = dataset / f"processing/retiming/episode_{oe:06d}"
        mapping = pq.read_table(sidecar.with_suffix(".parquet")).to_pylist()
        detail = json_read(sidecar.with_suffix(".json"))
        assert detail["source_episode_index"] == se
        assert len(mapping) == len(rows) == ep["exported_observations"]
        sf = np.array([m["source_state_frame"] for m in mapping])
        af = np.array([m["source_action_frame"] for m in mapping])
        duration, protected = check_clock(sf, af, detail["interval_scales"], detail["protected_mask"], fps)
        assert len(detail["interval_scales"]) == len(sr)
        source_path = np.vstack([np.asarray([r["observation.state"] for r in sr]), sr[-1]["action"]])
        state = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
        action = np.asarray([r["action"] for r in rows], dtype=np.float32)
        np.testing.assert_allclose(state, interpolate(source_path, sf), atol=2e-7, rtol=1e-6)
        np.testing.assert_allclose(action, interpolate(source_path, af), atol=2e-7, rtol=1e-6)
        # A 14D future target is constructed through exactly the same clock.
        np.testing.assert_allclose(action[:-1], state[1:], atol=2e-7, rtol=1e-6)
        np.testing.assert_array_equal(action[-1], np.asarray(sr[-1]["action"], dtype=np.float32))
        np.testing.assert_array_equal(state[0], np.asarray(sr[0]["observation.state"], dtype=np.float32))
        velocity = np.asarray([r["observation.velocity"] for r in rows], dtype=np.float32)
        expected_velocity = (action - state) * np.float32(fps)
        np.testing.assert_allclose(velocity, expected_velocity, atol=2e-7, rtol=1e-6)
        effort_path = np.asarray([r["observation.effort"] for r in sr] + [sr[-1]["observation.effort"]])
        actual_effort = np.asarray([r["observation.effort"] for r in rows])
        np.testing.assert_allclose(actual_effort, interpolate(effort_path, sf), atol=2e-7, rtol=1e-6)
        knots = np.asarray(detail["source_knot_episode_times_s"])
        assert knots.shape == (len(sr) + 1,) and np.all(np.diff(knots) > 0)
        source_alignment = None
        source_alignment_path = source / f"processing/alignment/episode_{se:06d}.parquet"
        if source_alignment_path.exists():
            assert sha256(source_alignment_path) == detail["source_alignment_sha256"]
            source_alignment = pq.read_table(source_alignment_path).to_pylist()
            np.testing.assert_allclose(knots[:-1], [a["source_episode_time_s"] for a in source_alignment], atol=1e-10, rtol=0)
        np.testing.assert_allclose([m["source_episode_time_s"] for m in mapping], np.interp(sf, np.arange(len(knots)), knots), atol=1e-9, rtol=0)
        np.testing.assert_allclose([m["action_source_episode_time_s"] for m in mapping], np.interp(af, np.arange(len(knots)), knots), atol=1e-9, rtol=0)
        speeds = np.abs(np.diff(source_path, axis=0)) * fps
        configuration = manifest["retiming"]["config"]
        protected_expected = ((speeds[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].max(1) <= configuration.get("protect_joint_speed", .15)) &
                              (speeds[:, [6, 13]].max(1) <= configuration.get("protect_gripper_speed", .3)))
        precision_expected = np.zeros(len(sr), dtype=bool)
        for phase in src_ep.get("precision_phases", []):
            precision_expected |= (knots[:-1] < phase["end_s"]) & (knots[1:] > phase["start_s"])
        np.testing.assert_array_equal(detail["precision_protected_mask"], precision_expected)
        protected_expected |= precision_expected
        np.testing.assert_array_equal(detail["protected_mask"], protected_expected)
        for local, (r, m) in enumerate(zip(rows, mapping)):
            assert r["index"] == m["index"] == totals + local
            assert r["episode_index"] == m["episode_index"] == oe
            assert r["frame_index"] == m["frame_index"] == local
            assert abs(r["timestamp"] - local / fps) < 2e-5
            assert r["task_index"] == 0
            weight = precision_rows[totals + local]
            for key in ("index", "episode_index", "frame_index"):
                assert weight[key] == r[key]
            assert abs(weight["timestamp"] - local / fps) < 1e-10
            assert abs(weight["action_source_time_s"] - m["action_source_episode_time_s"]) < 1e-10
            assert abs(weight["observation_source_time_s"] - m["source_episode_time_s"]) < 1e-10
            expected_label = expected_precision_label(src_ep.get("precision_phases", []),
                m["action_source_episode_time_s"], precision_multiplier)
            for key, value in expected_label.items():
                assert weight[key] == value, f"Incorrect precision label {oe}/{local}: {key}"
            image_frame = int(m["source_image_frame"])
            assert 0 <= image_frame < len(sr)
            expected_image_frame = int(np.clip(np.floor(sf[local] + .5), 0, len(sr) - 1))
            assert image_frame == expected_image_frame
            for camera in cameras:
                key = f"observation.images.{camera}"
                encoded = r[key]["bytes"]
                assert encoded and encoded[:2] == b"\xff\xd8"
                assert hashlib.sha256(encoded).digest() == hashlib.sha256(sr[image_frame][key]["bytes"]).digest()
                if source_alignment is not None:
                    check_camera_provenance(m, source_alignment[image_frame], camera, fps, len(sr))
                images += 1
        for key in NUMERIC:
            values = np.asarray([r[key] for r in rows], dtype=np.float64)
            assert values.shape == (len(rows), 14) and np.isfinite(values).all()
            sums[key].append(values)
        native_samples.append((totals, len(rows), action.copy()))
        totals += len(rows)
        episodes.append(dict(episode_index=oe, source_frames=len(sr), output_frames=len(rows),
                             source_duration_s=len(sr) / fps, continuous_retimed_duration_s=duration,
                             output_duration_s=len(rows) / fps, protected_source_duration_s=protected,
                             source_kinematics=derivative_summary(source_path, fps),
                             exported_kinematics=derivative_summary(np.vstack([state, action[-1]]), fps)))
        if (oe + 1) % 25 == 0 or oe + 1 == info['total_episodes']:
            print(f'Verified rows/images/provenance: {oe + 1}/{info["total_episodes"]} episodes', flush=True)
    assert next(src_iterator, None) is None and next(out_iterator, None) is None
    assert totals == info["total_frames"] == manifest["totals"]["observations"]
    assert len(precision_rows) == totals
    if source_precision:
        precision = json_read(dataset / "meta/precision.json")
        coverage = dict(total_action_targets=totals,
                        weighted_action_targets=sum(row["loss_weight"] > 1 for row in precision_rows),
                        gear_action_targets=sum(row["gear_insertion"] for row in precision_rows),
                        pin_action_targets=sum(row["pin_insertion"] for row in precision_rows),
                        episodes=len(episodes))
        assert precision["coverage"] == coverage
        assert manifest["precision_loss_metadata"]["coverage"] == coverage
        assert precision["table"] == "meta/precision_weights.parquet" and precision["join_key"] == "index"
        assert info["precision_loss_metadata"]["table"] == precision["table"]
    # Recompute the numeric metadata from the exported rows (rather than source
    # metadata or the solver's diagnostics). Quantiles use LeRobot approximation.
    stats = json_read(dataset / "meta/stats.json")
    for key, arrays in sums.items():
        values = np.concatenate(arrays)
        assert stats[key]["count"] == [totals]
        for name, expected in (("min", values.min(0)), ("max", values.max(0)),
                               ("mean", values.mean(0)), ("std", values.std(0))):
            np.testing.assert_allclose(stats[key][name], expected, atol=2e-6, rtol=1e-5, err_msg=f"{key} {name}")
    source_hash_checks = {}
    source_hash_checks.update(manifest["retiming"]["source_hashes"])
    for ep in manifest["episodes"]:
        detail = json_read(dataset / ep["retiming_report"])
        source_hash_checks.update(detail["source_data_files_sha256"])
    for relative, expected in source_hash_checks.items():
        assert sha256(source / relative) == expected, f"Source modified: {relative}"
    source_tasks = {int(row["task_index"]): row["__index_level_0__"]
                    for row in pq.read_table(source / "meta/tasks.parquet").to_pylist()}
    native_checks = 0
    if native:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch
        import datasets

        class BoundedNativeDataset(LeRobotDataset):
            """Use inherited item/chunk behavior, with only one image episode in RAM."""
            def load_hf_dataset(self):
                assert len(self.episodes) == 1
                episode = self.episodes[0]
                path = self.root / self.meta.get_data_file_path(episode)
                table = pq.read_table(path, filters=[("episode_index", "=", episode)])
                assert table["frame_index"].to_pylist() == list(range(len(table)))
                bounded = datasets.Dataset(table, info=datasets.DatasetInfo(
                    features=get_hf_features_from_features(self.features)))
                bounded.set_transform(hf_transform_to_torch)
                return bounded

        for episode_number, (start, length, actions) in enumerate(native_samples):
            loader = BoundedNativeDataset(manifest["repo_id"], root=dataset, episodes=[episode_number],
                video_backend="pyav", delta_timestamps={"action": [i / fps for i in range(50)]})
            assert len(loader) == length and loader.num_episodes == 1
            for local in sorted({0, length // 2, length - 1}):
                item = loader[local]
                assert item["task"] == source_tasks[0]
                assert tuple(item["observation.state"].shape) == (14,)
                assert tuple(item["action"].shape) == (50, 14)
                expected = actions[np.minimum(np.arange(50) + local, length - 1)]
                np.testing.assert_array_equal(item["action"].numpy(), expected)
                for camera in cameras:
                    image_key = f"observation.images.{camera}"
                    height, width, channels = source_info["features"][image_key]["shape"]
                    assert tuple(item[image_key].shape) == (channels, height, width)
                np.testing.assert_array_equal(item["action_is_pad"].numpy(), np.arange(50) + local >= length)
                native_checks += 1
            del loader
            if (episode_number + 1) % 25 == 0 or episode_number + 1 == len(native_samples):
                print(f'Verified native 50-step chunks: {episode_number + 1}/{len(native_samples)} episodes', flush=True)
    result = dict(passed=True, source=str(source), dataset=str(dataset), checked_observations=totals,
                  identical_source_jpeg_checks=images, native_loader_50_action_chunk_checks=native_checks,
                  independent_interpolation_verified=True, exact_protected_interval_durations_verified=True,
                  numeric_metadata_verified=True, camera_offset_provenance_verified=True, precision_weights_and_coverage_verified=True,
                  native_loader_backend="Inherited LeRobot item/action-chunk logic with one episode in-memory Arrow table; no full dataset HF disk cache",
                  episodes=episodes,
                  training_or_robot_execution_performed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-native", action="store_true", help="Skip native loader checks for development only")
    args = parser.parse_args()
    verify(args.source, args.dataset, args.output, native=not args.skip_native)
