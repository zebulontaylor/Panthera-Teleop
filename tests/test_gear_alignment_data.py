"""Check label sign and host-clock alignment without opening robot connections."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "gear_alignment_data", Path(__file__).resolve().parents[1] / "scripts/gear_alignment_data.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def episode(tmp_path):
    origin = 10_000_000_000
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"id": "fixture", "label": "kept", "complete": True, "start_monotonic_ns": origin}))
    states, frames = [], []
    for tick, gear_z in enumerate([0.10, 0.11, 0.12]):
        timestamp = origin + tick * 100_000_000
        arms = {arm: {"position_rad": [0] * 6, "gripper": {"position": 0.5},
                      "tool_pose_base": {"position_m": [0, 0, z]}}
                for arm, z in [("left", gear_z), ("right", 0.04)]}
        states.append({"monotonic_ns": timestamp, "arms": arms})
        for camera in ["wrist_port2", "wrist_port3"]:
            frames.append({"camera": camera, "monotonic_ns": timestamp,
                           "episode_time_s": 999, "path": f"{camera}/{tick}.jpg",
                           "width": 640, "height": 480})
    for name, rows in [("states", states), ("frames", frames)]:
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"cameras": ["wrist_port2", "wrist_port3"], "gear_arm": "left", "carrier_arm": "right"}
    selection = {"id": "fixture", "split": "heldout", "phase": "approach",
                 "start_s": 0, "end_s": 0.2, "reference_s": 0.2}
    return manifest, selection


class GearAlignmentDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.manifest, self.selection = episode(self.path)

    def test_relative_height_sign_and_monotonic_clock(self):
        rows = data.extract_episode(self.path, self.selection, self.manifest)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["demonstration_relative_tool_z_error_m"], -0.02)
        self.assertAlmostEqual(rows[0]["demonstration_left_tool_z_correction_m"], 0.02)
        self.assertAlmostEqual(rows[-1]["demonstration_relative_tool_z_error_m"], 0)
        self.assertEqual(rows[0]["images"]["wrist_port2"]["skew_s"], 0)
        self.assertTrue(all(row["split"] == "heldout" and not row["physical_object_height_ground_truth"]
                            for row in rows))

    def test_missing_camera_and_stale_observations_are_rejected(self):
        self.manifest["cameras"].append("overhead")
        with self.assertRaisesRegex(ValueError, "empty"):
            data.extract_episode(self.path, self.selection, self.manifest)
        with self.assertRaisesRegex(ValueError, "timestamp skew"):
            data.nearest([{"monotonic_ns": 0}], [0], 50_000_000, 0.04)

    def test_timestamp_tie_prefers_earlier_record(self):
        early, late = {"monotonic_ns": 0}, {"monotonic_ns": 2}
        self.assertIs(data.nearest([early, late], [0, 2], 1, 0.04), early)


if __name__ == "__main__":
    unittest.main()
