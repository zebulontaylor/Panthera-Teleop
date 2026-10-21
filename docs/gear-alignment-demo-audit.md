# Gear insertion demonstration contract

The gear alignment helper matches the height of the held planet gear to the
carrier's side opening. In the reviewed recordings the left gripper carries the
gear toward the gap between the two carrier plates. The right gripper holds the
carrier. The top pin bore and central bearing are separate features.

`wrist_port2` shows the approach from the gear gripper, with the carrier rim
directly ahead. `wrist_port3` shows the approach from the carrier gripper, with
the gear approaching the upper-left perimeter. Both views help distinguish the
held gear from the tabletop pin. These assignments follow the visible motions
and left gripper state changes; the saved camera calibration still marks arm
assignments unverified.

## Reviewed alignment intervals

The [selection manifest](../config/gear-alignment-demos.json) reserves whole
episodes for training, development, and held-out evaluation. Times are seconds
from `metadata.start_monotonic_ns`. Each interval follows the same grasp until
a near-contact reference before opening the gear gripper. Pickup, release,
withdrawal, and subsequent pin insertion are outside the selected intervals.

| Episode | Split | Interval, s | Reference, s |
| --- | --- | --- | --- |
| `20260930T020311-ee8fe583` | Train | 3–5 | 5 |
| `20260930T020432-eacee9a3` | Train | 3–4.5 | 4.5 |
| `20260930T020558-ca008d0e` | Train | 4–6 | 6 |
| `20261005T150837-d76f2a97` | Development | 5–9 | 9 |
| `20261011T191045-ee339dd5` | Held out | 5–7.5 | 7.5 |

## Available training targets

The recordings contain RGB JPEGs, host receive timestamps, measured joints,
gripper states, and `tool_pose_base` for each arm. They contain no calibrated
depth, camera-to-robot transforms, object-center poses, or contact event labels.
The visible side gap is partially covered near insertion, so these images do
not establish exact seating height or successful insertion.

[`gear_alignment_data.py`](../scripts/gear_alignment_data.py) extracts a weak
reference target:

```text
relative_z = left_tool_base_z - right_tool_base_z
error = relative_z_at_observation - relative_z_at_reference
left_only_correction = -error
```

This target describes the demonstrated tool motion. Its interpretation as a
vertical correction assumes both arm base Z axes point vertically. Fixed base
height offsets cancel within an episode. Tool rotation changes and different
object grasp offsets can change object height without the corresponding tool
Z change, so this target cannot replace physical gear and gap labels from
calibrated geometry. Keep its `weak_demo_reference` provenance when using it
for domain adaptation or comparison with simulation labels.

From the repository root, run:

```bash
python scripts/gear_alignment_data.py
```

The command writes `work/gear-alignment/samples.jsonl` and
`work/gear-alignment/samples.summary.json`. Rows reference original JPEGs, retain
camera and state timestamp offsets, and include the episode split and weak Z
target. The extractor rejects camera offsets above 40 ms and state offsets
above 10 ms. It leaves recordings unchanged. Split by episode when training;
neighboring frames from one approach share the same grasp and reference.

Five demonstrations support an initial development comparison. They do not
measure hardware insertion success or establish reliable behavior across new
grasp offsets and carrier configurations.
