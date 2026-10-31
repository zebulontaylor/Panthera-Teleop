# Optional CNN alignment assistance during collection

## Current supervised gear trial

The October 22 live calibration recording `20261022T214847-732880cf` supplies
a measured gear Jacobian and an operator-demonstrated height reference in
`config/calibration/gear-supervised-trial.json`. The recording configuration now
selects `mode: "operator_supervised"` and loads this file on startup. It does
not mark automatic calibration as verified. Pin is not supported in this mode.

The dashboard shows two inference previews: green marks the gear center, yellow
marks the slot reference, and blue marks the demonstrated gear height. Each
overlay uses the exact JPEG that produced its predictions. The operator checks
these marks, confirms a retained gear and clear carrier, then holds **Hold for
one test nudge** during an active recording. This explicit operator confirmation
replaces the automatic jaw/visibility/confidence/contact-torque predicates for
that press; those predicates have not been validated for this passive-gripper
setup. Gripper position must remain within 0.04 rad of its position at that press,
with speed at most 0.1 rad/s. This is not automatic retained-object detection.

Each unique press has a 0.3-second total window and a heartbeat lease of at most
0.2 seconds. Release, lost heartbeat, or exhausted time cannot be renewed using
the same press ID. A controller fault or measured joint travel of 0.006 rad
latches that press off. Fresh/skew-bounded camera evidence, checkpoint/camera
identity, finite in-frame landmarks, proximity, both-arm local poses, slow
controlled-arm motion, joint limits, 0.006-rad requested offsets, and 0.15-Nm
additive torque caps still apply. Both arms must remain within 0.3 rad per joint
of their calibration poses; keep the carrier still during each trial. Return,
hold, reset, simulation and non-recording modes
exclude assistance. The operator must release and press again for another test.

Supervised pulses use the permitted torque magnitude along the measured DLS
direction, capped at 0.15 Nm per joint, with velocity damping. Automatic mode
retains the original weak spring. The first hardware attempts used the weak
spring and sent about 0.06 Nm for 0.10–0.13 seconds with no encoder movement;
they did not establish physical correction. The stronger supervised pulse
still needs hardware evaluation. The dashboard retains the last press reason
and peak requested additive torque after the button is released.

The 443 synchronized calibration observations had no timestamp rejects. A
diagnostic temporal split used 71 motion pairs for fitting and 35 later pairs
for evaluation: height-motion RMSE was 5.05 px versus a 12.28 px no-motion
baseline (58.8% improvement). Five independent joint directions were retained.
The returned reference differed by 10.46 and -3.02 vertical pixels between the
two views. This single demonstration does not establish independent correction
performance or seating accuracy. The profile uses the first demonstrated
reference and an experimental 0.3-rad local joint envelope. This envelope was
widened from 0.15 rad after a hand-returned carrier pose blocked the first trial;
the wider region has not been independently calibrated. Its report and source hashes are
stored with the profile. Physical response must be checked in the supervised
trial before considering automatic activation.

Offline controller calls on the recorded observations produce bounded proposals
and aligned results; other samples are blocked by movement and proximity gates.
The original weak-spring replay's peak proposed torque was below 0.072 Nm. These were
calculations only, not motor commands. See
`logs/alignment-calibration/supervised-replay.20261022T214847.json`.

To mark a demonstrated pose while recording:

```bash
python3 scripts/mark_alignment_calibration.py --label reference
```

The same tool accepts `motion-start` and `motion-end`. Markers are separate
operator annotations; the source recording is unchanged. Analyze an inferred
cache using `scripts/analyze_marked_alignment.py --help`. Changes to the
controller or calibration profile require a collector restart; support the
arms before stopping the active controller.

## Automatic assistance

Start the normal collector with `./start.sh`. The episode dashboard has an
**Enable CNN alignment nudges** checkbox. The current saved configuration enables
inference in supervised gear mode; the controller's automatic-mode default is
off. To explicitly start inference, use `./start.sh --alignment-assist`; select pin perception
with `./start.sh --alignment-assist --alignment-task pin` after restoring automatic
mode and supplying a pin calibration. The configured task
is otherwise `gear`. Restart to change task/calibration.

Both downloaded adapted models run in a separate process. The current recording
configuration selects `RobotArmTraining/.venv-cnns-cuda/bin/python` and
`device: "auto"`, which uses the RTX 3070. The checkpoint bundle remains
`RobotArmTraining/outputs/panthera-assembly-cnns`. The original CPU environment
`RobotArmTraining/.venv-cnns` remains available. `alignment_assist.python` and
`model_root` configure these paths.
The training runtime needs the bundle's numpy/torch requirements plus Pillow
(installed in the existing local runtime). Recording does not import torch.
All three native 640×480 RGB views are network inputs, including overhead;
the two wrist views supply motion evidence. `camera_mapping` maps model slots
to actual camera contents. **For the current 17-trial setup, use `overhead,
wrist_port3, wrist_port2`.** The training recordings called the gear/pin camera
port 2 and carrier camera port 3; the current connections reverse those camera
contents. Copying those historical USB names caused the poor overlays. The
recording configuration and live/offline defaults now correct the order. Left
and right tool XYZ ordering remains unchanged. See the
[input-routing investigation](cnn-input-investigation.md) for the controlled
comparison and corrected overlays.

## Current setup requires calibration before motion

The supplied setting `alignment_assist.calibration.verified` is **false**.
Enabling the checkbox starts inference, but the controller reports
`calibration_required` and adds no torque. Existing camera calibration and
wrist-to-arm assignments are unverified. The checkpoints output image landmarks;
they do not supply a measured real-robot displacement or a grasp classifier.
Synthetic rigid-grasp IK results cannot supply this missing real mapping.

A measured local image Jacobian is required for each task and grasp/view setup.
It differentiates the moving-minus-target landmark residual in each wrist view
with respect to the selected arm's six joint angles. This avoids inventing
camera-to-base transforms. Establish the matrix and sign using measured small
joint displacements around a supported, known pose, then verify that predicted
corrections reduce the residual. No calibration motion is launched by this
feature. Do not substitute an identity matrix or simulation calibration.

The calibration object requires:

- `verified: true`, `side: "left"` or `"right"`, `camera_names:
  ["left_wrist", "right_wrist"]`, and `camera_mapping` matching the verified
  logical-to-physical wrist mapping used for inference.
- `checkpoint_id`: the selected model file's SHA-256, and `image_size: [640,480]`.
  The installed gear hash is
  `6c62fdb0dc069eb89eb27fbbfa300c3915fc2d5ab395e135f7ff9599c9e39273`;
  pin is `7fb9ddd55dc006c6c50d1e77fdf62726005d45d812916a54951de9689c1cfe7f`.
- `reference_joint_position_rad`: six measured joint positions, and
  `max_joint_distance_rad`: the measured local validity range (at most 0.3 rad).
  `other_arm_reference_joint_position_rad` records the other arm's six positions;
  both arms must stay within that range because both wrist views move.
- `image_jacobian_px_per_rad`: four rows by six columns, in
  left-wrist x/y then right-wrist x/y residual order.
- `target_error_px`: four moving-minus-target differences at the intended good
  alignment pose. They need not be zero. `residual_axes: [1,3]` for gear height
  correction; `[0,1,2,3]` for pin alignment. Gear assistance does not try to
  drive the gear center laterally into the slot.
- `target_landmark: "slot_center"` for gear; see pin target association below.
- `jaw_rois_px`: a rectangle `[xmin,ymin,xmax,ymax]` in each logical wrist view
  where the held gear center or pin tip must appear; and measured
  `gripper_closed_position_rad` and positive `gripper_contact_torque_nm`.

For pin, set `target_landmark: "carrier_bore"` and `target_rois_px` to a measured
rectangle in each logical wrist view enclosing the same intended physical bore.
The four detected bore slots are unordered. The controller selects each view's
candidate independently and requires exactly one candidate in each ROI. Missing
or ambiguous candidates block motion; matching slot numbers across views is
insufficient to establish a shared target.

## Motion gates and recorded commands

Nudges require an active recording, verified matching calibration, fresh
predictions (default maximum age 0.2 seconds), wrist timestamp skew at most
0.06 seconds, visible/confident landmarks in both wrist views, residuals within
60 pixels of calibrated alignment in each view, slow arm motion, and joint
limit clearance. Grasp evidence combines the object at the calibrated jaws with
closed gripper position and measured contact torque, sustained across at least
0.2 seconds of fresh camera evidence. This is a contact/vision heuristic;
model confidence itself is uncalibrated.

The image residual gives a damped least-squares joint correction capped at
0.006 rad in norm. A weak spring adds at most 0.15 Nm per joint to gravity
compensation. Each nudge lasts at most 0.3 seconds. The burst remains exhausted
until the grasp/proximity gate clears, collection ends, or the option is toggled.
Targets do not accumulate between control ticks. Loss of valid evidence removes
the assist torque. Return, neutral hold, gripper reset, and simulation exclude
assistance. Gripper commands remain passive during assistance.

Every state sample includes `alignment_assist` with enabled/active state,
waiting reason, selected task, and command diagnostics when available. During a
nudge, the sample and selected arm use `control_mode: "alignment_assist"`;
the arm stores the additive torque and actual final sent torque. The episode
configuration snapshots the task, limits, camera mapping, and calibration.
Inference failures appear beside the checkbox and leave ordinary guidance
available. Actual hardware alignment accuracy has not been validated.
