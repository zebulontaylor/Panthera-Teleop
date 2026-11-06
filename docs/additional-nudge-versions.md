# Additional separate nudge experiments

These are additional implementations, not hardware-validated fixes. The original
controller, its configuration, `start.sh`, and the earlier vertical-only experiment
remain unchanged. Both new versions execute one bounded position step per start,
then hold; neither automatically chains steps or sends legacy torque pulses.

| Version | Launcher | Main implementation | Required inputs |
| --- | --- | --- | --- |
| Original | `start.sh` | `app/alignment_assist.py`, `app/alignment_servo.py` | Existing profile |
| Vertical-only experiment | `start-cartesian-nudge.sh` | `app/cartesian_nudge.py` | Existing visual profile and URDF |
| Constrained image-based experiment | `start-constrained-nudge.sh` | `app/constrained_visual_nudge.py` | Existing visual profile, URDF, measured forward axis |
| Stereo 3D position experiment | `start-stereo-nudge.sh` | `app/stereo_pose_nudge.py` | URDF, measured forward axis, calibrated cameras/transforms and demonstrated 3D offset |

## Constrained image-based version

The solver minimizes the two gear-height pixel errors over three-dimensional
tool translation while fixing tool orientation. Constraints limit nominal step
length to 0.5 mm, joint-step norm to 0.003 rad, and displacement opposite a measured
forward axis. Neither camera's predicted absolute height error may increase by
more than 0.01 px; the predicted combined reduction must be at least 0.25 px.
The numerical solver's result is checked again through joint limits and 11
interpolated robot-model poses. The nonlinear modeled step must remain within
0.501 mm, with at most 5 micrometers of modeled retreat tolerance.

Unlike the vertical-only version, it can use a limited horizontal translation
when that satisfies all constraints. It still relies on the existing local
image Jacobian, which has not independently established corrective performance.

Settings are separate in `config/alignment-constrained-experiment.json`.
`forward_axis_base` is a measured unit vector expressed in the **left robot's
base frame**, pointing in the direction that should count as forward. This is
not assumed to be +X. The default is null and `axis_verified` is false, so the
controller reports `measured_forward_axis_required` and issues no correction.
Only set this flag after physically checking the axis and robot kinematics.

## Stereo 3D position version

This is translation-only position-based visual servoing. The current gear CNN
provides a center and slot-reference point; these do not determine full object
orientation. Tool orientation remains fixed at its initial value. The version
does not implement full six-degree-of-freedom object-pose alignment or insertion.

Each landmark is triangulated from the two calibrated wrist cameras. The 3D
residual is:

```
gear_center_in_left_base - slot_reference_in_left_base - demonstrated_offset
```

The controller requests a translation opposite that residual, at most 0.5 mm,
with the same joint, orientation, path and no-retreat checks. It blocks rather
than inventing an alternative path if the requested translation would retreat.
Residuals above 10 mm are outside its local working neighborhood; the alignment
deadband is 0.3 mm per coordinate. These are experimental settings, not measured
assembly tolerances.

The separate `config/alignment-stereo-experiment.json` intentionally contains
null geometry and `verified: false`. It needs all of the following measured data:

- `intrinsics`: 3 by 3 camera matrix at the actual inference image resolution.
- `distortion`: OpenCV distortion coefficients for each camera.
- `T_tool_camera`: calibrated camera-to-tool rigid transform for each wrist.
- `side`: verified physical arm assignment for each camera.
- `T_left_base_right_base`: transform from the right base frame into the left.
- `target_offset_left_base_m`: demonstrated gear-minus-slot 3D offset at the
  intended alignment, with the same grasp, reference geometry and orientation.
- Matching camera mapping, image size and checkpoint identity.
- A physically checked `forward_axis_base` and `axis_verified` as above.

For every transform, `T_A_B` maps points expressed in B into A. Translations are
meters. The example has no identity substitutes for missing measurements.
Do not mark geometry verified based on synthetic tests alone.

Measured tool poses are associated with the exact inference input frame stamps;
camera-to-pose timestamp mismatch above 30 ms blocks reconstruction. Triangulation
also requires at least 2 degrees of ray parallax, positive depth between 0.03 and
2 meters, and reprojection error within 2 pixels in both views. Camera mapping,
checkpoint, calibration and image dimensions must match. Amodal CNN references
may still fail physical triangulation assumptions even if reprojection passes.

## Runtime behavior shared by both

Separate entry points are `app/hand_guide_constrained.py` and
`app/hand_guide_stereo.py`, with separate dashboard modules and HTML files.
`app/bounded_alignment_trial.py` supplies their shared bounded trial lifecycle.
The original camera/checkpoint, local-pose, freshness, proximity and joint-limit
gates remain active. As in the supervised original, the operator is responsible
for retained gear and a clear target; no new automatic grasp detector is installed.

After an explicit start, each version settles and gathers three distinct fresh
observations, computes at most one step, settles again and checks response. It
reports unconfirmed physical movement, worsening residual, inconclusive response,
or an observed step. None of these results certifies assembly success.

Total tool displacement above 1 mm, joint travel above 0.006 rad, measured retreat
above 0.1 mm, orientation change above 0.005 rad, changed gripper, tracking error,
model/feedback disagreement, or ended collection stops or aborts correction.
Carrier-arm movement above 0.005 rad ends correction. An unfinished trial has a
five-second budget. Position hold remains until **Release position hold** unless
an abort returns control to gravity assistance. Tracking and workspace checks
continue during hold. Position commands use 0.02 rad/s and the existing limits
capped by the starting modeled gravity magnitude plus 0.3 Nm.

These are encoder/URDF constraints, not externally measured tool bounds or a
collision checker. The robot-command interface, kinematics and actual tool path
still need physical verification before judging whether either solves retreat.

## Offline checks and findings

```bash
PYTHONPATH=tests work/teleop-venv/bin/python -m unittest \
  test_additional_nudges test_cartesian_nudge test_alignment_servo \
  test_supervised_alignment test_robot_commands test_alignment_dashboard
work/teleop-venv/bin/python scripts/validate_additional_nudges.py
```

The constrained solver produced 144 proposals from 443 recorded observations
using **hypothetical +X forward**, preserving the original pose/proximity bounds;
287 were already inside the pixel deadband and 12 were outside those bounds.
Maximum modeled step was 0.500204 mm. Maximum modeled retreat was about 0.000504
mm, within the 0.005 mm path tolerance. This is geometric feasibility using the
fitted visual model, not a prediction validated against actual corrective motion.
It does not populate or verify the live forward-axis setting.

Synthetic stereo checks recover known metric offsets and reject insufficient
parallax, mismatched checkpoint/calibration, inconsistent image points and stale
camera poses. Real-recording 3D evaluation is blocked by missing measured geometry.
Reports and source hashes are saved in
`logs/alignment-calibration/additional-nudges-offline.json`.

## Selecting a version

These launchers start real hardware control, like the existing launcher. Neither
has been started as part of implementation. Stop the current
collector before switching; all versions share the original exclusive controller
lock. Each uses the existing recording directory and recording prerequisites.

```bash
./start-constrained-nudge.sh
./start-stereo-nudge.sh
```

Run only one launcher. Each dashboard labels its own experiment and has its own
one-step action. The legacy torque-nudge route is rejected in both. To use a
separate measured settings file without changing the supplied defaults:

```bash
./start-stereo-nudge.sh --experiment-config /absolute/path/to/measured-settings.json
```

Neither default settings file allows correction. The next physical prerequisite
for the constrained version is a verified forward axis and local motion mapping;
the stereo version additionally needs the camera and target calibration above.
`./start.sh` and `./start-cartesian-nudge.sh` retain their prior behavior.
