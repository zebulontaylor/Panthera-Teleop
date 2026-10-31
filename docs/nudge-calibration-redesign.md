# Calibrate intended motion before calibrating nudging

For the current gear-height task, the recommended first route is a **motion-first
audit followed by bidirectional task-axis calibration**. A small directly measured
model is easier to identify than six joint sensitivities from general hand-guided
movement. Move to a Cartesian or geometric model only when the task requires it.

All new scripts are separate offline/read-only tools. They never import a robot
controller, start a movement, replace weights, update live configuration, or set
calibration verified. Existing controllers and the three experimental launchers
are unchanged. Reports always contain `verified: false` and
`motion_authorized: false`. A `statistical_candidate` means only that the specified
offline checks passed; it does not enable assistance or establish seating accuracy.

## Routes provided

| Route | Purpose | New scripts |
| --- | --- | --- |
| Motion first | Check requested movement against encoder FK and independently measured physical tool displacement | `audit_nudge_motion.py` |
| Bidirectional task axis — recommended first | Learn pixels per meter along one measured physical axis; check sign, reserved sessions and direction dependence | `calibrate_nudge_axis.py` |
| Cartesian translation calibration | Learn a 4 by 3 pixels-per-meter mapping from independent XYZ movements, holding orientation/carrier fixed | `calibrate_nudge_cartesian.py` |
| Geometric camera route | Estimate intrinsics and camera-to-tool transform with metric checkerboard geometry and reserved observations | `calibrate_nudge_camera.py`, `calibrate_nudge_hand_eye.py` |

Collection/preparation tools are `mark_nudge_calibration.py` and
`build_nudge_calibration_pairs.py`. `audit_recorded_nudge_coverage.py` checks what
the old data can support without relabeling it as a controlled experiment.

Adaptive Jacobian updates can be a later extension, but they are deliberately
not installed here: updating from a wrong movement can reinforce a bad model.
First establish a trustworthy physical direction, local operating region and
separate validation data. Online estimation is a recognized visual-servo method;
see [Jacobian learning research](https://flowers.inria.fr/mlopes/myrefs/06-iros-vs.pdf).

## Proposed collection process

1. Verify a physical axis in the left arm base frame. Check actual tool movement
   with an independent measurement, not only the same FK model that generated
   the command. Resolve joint order, signs, units and mounting errors before fitting.
2. Keep the carrier stationary and retain the same gear/grasp. Hold a demonstrated
   good-height reference and mark it. A demonstration supplies a reference offset;
   it does not certify seating or insertion geometry.
3. Collect deliberate small movements in both directions along that axis. Hold
   each endpoint still for at least a second before marking. The offline filters
   accept measured displacements from 0.2 to 3 mm; this is a data-selection range,
   not an instruction to command a particular displacement or move near contact.
4. Return to the same reference periodically and at session end. Inspect position,
   orientation and image-offset return differences for drift, backlash, slip or
   changes in target association.
5. Collect at least two training sessions and a separate reserved session in
   distinct episodes. Include both directions within every session. For example,
   eight marked movement pairs per session provide sixteen training and eight
   validation pairs. Keep the same local camera/grasp/target configuration.
6. Fit the training sessions only; evaluate reserved measured image changes. Then
   assess an independently supervised bounded correction. A good predictive fit
   alone is not evidence that a proposed correction succeeds physically.

The marking tool reads the active collector; it does not perform step 3. Move
manually or through a separately evaluated movement controller. No new hardware
collection has been performed by these scripts.

## Marking and preparing a dataset

During a healthy active recording, hold an endpoint still and annotate it:

```bash
work/teleop-venv/bin/python scripts/mark_nudge_calibration.py \
  --session train-a --trial z-up-01 --split train --label start \
  --output logs/nudge-protocol/markers.jsonl
```

After moving and settling, use the same session/trial and `--label end`.
Annotate a return reference with a distinct trial name and `--label reference`.
Use a different session and episode for validation and `--split validation`.
Do not assign neighboring observations from one recording to both sets.

Optional end-marker fields are `--requested-delta-mm X Y Z` and
`--external-delta-mm X Y Z`. The latter is an independent physical measurement;
copying the encoder-derived result into it defeats the motion audit.

Generate a corrected-order CNN cache for these completed recordings using the
existing `alignment_calibration_infer.py`, with `--fps 10`, explicit `--episode`
arguments and `--camera-names overhead wrist_port3 wrist_port2`. Audit camera
contents again if connections change. Use a new output cache path.

Then assemble stable endpoint pairs:

```bash
work/teleop-venv/bin/python scripts/build_nudge_calibration_pairs.py \
  --predictions logs/nudge-protocol/predictions.jsonl \
  --markers logs/nudge-protocol/markers.jsonl \
  --output logs/nudge-protocol/pairs.jsonl
```

Each endpoint uses the preceding 0.75 seconds, requiring at least five observations
with distinct frames in both wrists. It checks frame/state skew, joint speed,
faults, in-frame landmarks, tool/carrier/gripper stability and perception identity.
Endpoint windows may not overlap. The companion manifest records source hashes,
rejected trials, reference plateaus and reference-return differences.
Stable CNN output does not itself establish visibility or retained-object truth.

## Fit and audit

For a physically checked base-Z task axis:

```bash
work/teleop-venv/bin/python scripts/audit_nudge_motion.py \
  --pairs logs/nudge-protocol/pairs.jsonl --output logs/nudge-protocol/motion-audit.json
work/teleop-venv/bin/python scripts/calibrate_nudge_axis.py \
  --pairs logs/nudge-protocol/pairs.jsonl --axis 0 0 1 \
  --output logs/nudge-protocol/axis-candidate.json
work/teleop-venv/bin/python scripts/calibrate_nudge_cartesian.py \
  --pairs logs/nudge-protocol/pairs-xyz.jsonl \
  --output logs/nudge-protocol/cartesian-candidate.json
```

The Cartesian input requires independent XYZ excitation, not the same vertical-only
dataset. Both fits difference the endpoint residuals and use robust regression
through the origin. The response ordering is left wrist x/y, right wrist x/y;
the reported response is pixels per **meter**, not per joint radian.

Pairs with carrier movement above 0.003 rad, tool rotation above 0.003 rad or
gripper movement above 0.02 rad are rejected. Task-axis fitting additionally checks
off-axis motion. Across sessions, endpoints must share a 10 mm local neighborhood,
tool orientation within 0.02 rad, carrier joint position within 0.03 rad of its
median, and gripper range within 0.04 rad. These experimental thresholds should
be reviewed against real measurement noise and the intended task.

Axis fitting needs at least eight training and four reserved pairs, with at least
two pairs in each direction in every session. Cartesian fitting needs at least
twelve training and six reserved pairs and well-conditioned three-axis excitation.
Both require at least two training sessions and a separate reserved session.
Session and episode overlap is rejected. Reported model checks require a 20%
improvement over predicting no image movement, at least four eligible reserved
direction checks and 80% directional agreement. Axis slopes fitted separately in
each direction must agree within 30%. These thresholds are screening criteria,
not hardware certification. Candidate files are not automatically loaded by any
existing launcher, and do not include a certified desired alignment target.

## Camera and hand-eye route

Use this route when metric 3D alignment is needed. OpenCV supplies the underlying
[camera and hand-eye calibration APIs](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).
First collect checkerboard views with a measured square size, broad field coverage
and varied tilt, keeping the camera/lens settings and resolution fixed.

The camera manifest is a JSON list of objects containing `path`, `session_id` and
`split`. Image paths are relative to that manifest. At least ten detected training
images and four reserved images from separate sessions are required. For a board
with nine by six inner corners:

```bash
work/teleop-venv/bin/python scripts/calibrate_nudge_camera.py \
  --manifest /absolute/path/to/camera-manifest.json --corners 9 6 \
  --square-size-mm "$MEASURED_SQUARE_MM" \
  --output logs/nudge-protocol/camera-candidate.json
```

Set the shell variable to the actual measured square size. The output reports
intrinsics, distortion and reserved reprojection errors. It checks board-normal
diversity and a 1 px reprojection threshold. Reprojection alone cannot validate
metric scale or hand-eye accuracy.

For hand-eye calibration, obtain synchronized tool poses and board poses from
that camera using the measured intrinsics and board geometry. Its input JSONL
rows have `session_id`, unique `pose_id`, `split`, `T_base_tool` and
`T_camera_target`, each transform a 4 by 4 matrix with translation in meters.
`T_A_B` maps coordinates in B into A. Camera target poses can be obtained with
OpenCV `solvePnP`; this script consumes poses rather than automatically collecting
or synchronizing a new board-image dataset.

```bash
work/teleop-venv/bin/python scripts/calibrate_nudge_hand_eye.py \
  --poses /absolute/path/to/board-and-tool-poses.jsonl \
  --output logs/nudge-protocol/hand-eye-candidate.json
```

The same calibration board must remain fixed in the same robot base frame across
all poses. At least eight training and four reserved poses are required, with
independent rotation axes and disjoint sessions. The candidate camera-to-tool
transform is checked by reconstructing the stationary board in reserved poses;
the current screening bounds are 2 mm translation and 0.5 degrees rotation.
Translation-only motion is rejected as insufficient excitation.

Repeat separately for the other camera. The inter-arm base transform and the
demonstrated 3D assembly offset remain additional measured prerequisites for the
stereo controller. These tools do not invent or certify those values.

## Existing-data result and verification

The old 443-observation session provides 128 non-overlapping temporal pairs.
Only five satisfy the new geometric base-Z screening: three positive and two
negative. Ninety-eight have insufficient/out-of-range axis displacement and
twenty-five have excessive off-axis motion. It is one episode without the new
protocol annotations or independent reserved sessions, so no new fit is claimed.
The coverage report is `logs/alignment-calibration/new-calibration-coverage.json`.

Tests cover recovery of known signed axis/XYZ responses, rejection of leakage,
off-axis movement, rotation, hysteresis and wrong reserved response; duplicate and
overlapping plateaus; physical-motion disagreement; and recovery of known camera
intrinsics and hand-eye transforms. Original and previous experimental controllers
are included in the regression run:

```bash
PYTHONPATH=tests work/teleop-venv/bin/python -m unittest \
  test_nudge_calibration_protocols test_additional_nudges test_cartesian_nudge \
  test_alignment_servo test_supervised_alignment test_robot_commands test_alignment_dashboard
```

These checks use synthetic measurements and existing recordings. No new motors
were commanded and no fresh physical calibration has been collected.
