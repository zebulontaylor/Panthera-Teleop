# Four separate candidate calibration workflows

These implementations add four local browser workflows without changing the
original collector, prior calibration tools, or the vertical, constrained and
stereo nudge experiments. All candidate reports stay unverified and are never
automatically installed. No new motor collector has been run as part of this work.

## Launchers

| Workflow | Separate UI launcher | Workflow implementation | Operator action |
| --- | --- | --- | --- |
| Guided hand calibration | `start-guided-calibration.sh` | `app/guided_hand_calibration.py` | Save a good reference automatically, then hand-guide small movements following prompts |
| Hold-to-run sequence | `start-hold-calibration.sh` | `app/hold_run_calibration.py`, `app/calibration_sequence.py` | Hold the button while a separately gated collector executes bounded axis steps |
| Demonstrate a correction | `start-demonstration-calibration.sh` | `app/demonstrated_correction_calibration.py` | Capture a nearby misalignment and a manually improved pose |
| Board or ArUco marker | `start-board-calibration.sh` | `app/board_marker_calibration.py` | Capture stable images at varied wrist poses with a fixed metric target |

The UI launchers open a local HTTP service, not robot motor ports. Each prints
its URL, normally `http://127.0.0.1:8091`, and a fresh output directory under
`logs/calibration-workflows/`. Visit that URL in the browser. Use one workbench
at a time, or choose distinct `--port` values. Existing output directories with
data are rejected so an earlier run cannot silently be overwritten.

Common options:

```bash
./start-guided-calibration.sh --collector http://127.0.0.1:8090 --port 8091
./start-demonstration-calibration.sh --output /absolute/path/to/new-run-directory
```

The existing real collector must already have a healthy active recording for
capture to begin. Choose a session name and training or reserved-validation set.
A new session requires a distinct recording episode. The UI suggests subsequent
session names/sets after a completed session; these remain editable. Captures,
fits, and source images are saved automatically. The page reports missing evidence
and exposes detailed measurements in a collapsible results panel.

For guided, demonstration and board workflows, start the normal collector through
the existing procedure. These workbenches read data; they do not disable or alter
the original collector's buttons. Keep unrelated nudge controls unused while
collecting calibration data. A separate motor collector is required for hold-to-run,
as described below.

## 1. Guided hand calibration — recommended starting point

Choose the physically checked task axis, such as base Z. Select **Begin session**
while holding the demonstrated good alignment still. The wizard automatically
captures the reference, then prompts:

```
+axis → reference → −axis → reference → +axis → reference → −axis → reference
```

The operator supplies these movements by hand. This version sends no movement
requests. It accepts small axis displacements from 0.2 to 1 mm, checks off-axis
drift and orientation, and checks that reference returns reproduce the image
offset. It requires at least 0.6 seconds of stable observations from a 0.75-second
window, using distinct frames in both wrists. Endpoint selection and trial marking
are automatic. The sequence yields eight pairs per session.

Collect two training sessions and a reserved session in separate episodes, then
select **Analyze captured evidence**. The existing new axis-fitting tool handles
the numerical analysis, direction dependence, identity checks and split leakage.
Results are written to `response-candidate.json`; pairs and reference events are
saved in `pairs.jsonl` and `events.jsonl`. A timestamped `observation-cache.jsonl` preserves sampled predictions and measured states for review.

This is the closest replacement for the original general hand-guided calibration,
but it deliberately asks for useful movement directions rather than accepting
arbitrary movement. It still depends on an operator-selected good reference,
retained grasp and physically meaningful robot kinematics.

## 2. Hold-to-run bounded sequence

This design has two separate launchers:

```bash
./start-calibration-sequence-collector.sh
./start-hold-calibration.sh
```

**The collector launcher starts real robot control**, like `start.sh`; it has not
been run during implementation. Support the arms and stop the existing collector
before changing versions. It shares the normal frontend lease/lock and independent motor backend. The
normal recording dashboard continues to provide recording and gear-grasp controls.
The workbench is a separate UI; neither launcher replaces the original script.

The default `config/calibration-sequence-motion-gate.json` blocks automatic motion.
The button remains disabled until the special collector reports a configured
gate. The gate requires:

- Explicit operator attestation that position-command motion, axis, signs and
  physical tool displacement were independently checked. Manual hand movement
  alone does not verify the position-command interface.
- A measured unit `axis_base` and local measured `reference_q_rad`.
- Matching inference camera mapping and checkpoint identity.
- A motion-first audit report with at least six three-way agreements, including
  at least three positive and three negative near-axis movements, with no missing
  or failed measurement outcomes.
- The exact SHA-256 of that audit file. The collector also pins its gate-file
  hash at startup; a changed gate requires restart.

The audit report checks recorded numeric agreement; the operator attestation is
still needed to establish that those measurements came from the actual commanded
movement, correct physical axis and independent external measurement. Software
does not automatically grant that attestation.

To use a separate measured gate file without modifying the supplied default:

```bash
./start-calibration-sequence-collector.sh --motion-gate /absolute/path/to/measured-gate.json
```

Begin a workbench session and hold the demonstrated reference still until captured.
Then hold **Hold to run bounded calibration sequence**. The collector independently
checks recording, settled gear grasp, camera evidence, calibration identity, local
joint region, feedback, joint limits and lease freshness. It computes physical
axis movements from the URDF; the old visual Jacobian does not choose directions.

Each requested offset is +0.5 mm, reference, −0.5 mm, reference, repeated twice.
Each step must be at most 0.003 rad in joint norm. The reference region is at most
0.1 rad per joint from the verified pose; total measured joint travel is at most
0.006 rad in norm. Measured tool displacement above 0.8 mm, off-axis travel above
0.15 mm, orientation change above 0.003 rad, changed gripper or carrier movement
stops the sequence. Eleven interpolated model poses check each proposed step.
Position commands use 0.02 rad/s and the existing torque limits capped by the
starting gravity model magnitude plus 0.3 Nm.

Settled endpoints are held at least 1.2 seconds for the independent workbench to
observe. The whole press has a 20-second budget and a 0.2-second heartbeat lease.
Release, lost focus, missing heartbeat, excluded collection modes or faults end
position commands. Expired/released press IDs cannot rearm the same run. On stop,
the running frontend returns to its ordinary gravity-assistance path, so the operator
must continue to support the arms. No automatic return movement is attempted. If the entire frontend stops or disconnects, the independent motor backend retains powered position hold; stopping the frontend does not shut down that backend.

The workbench does not renew a press in the background. Browser heartbeats are
forwarded individually to the motor-loop request queue. Analysis and session
changes are rejected while the press is active. Sequence motion and automatic
endpoint capture are separate; missed/unstable endpoint evidence does not become
a successful calibration. Legacy torque nudges, visual-servo starts and envelope
widening are disabled in this collector variant.

## 3. Teach a correction

Begin while holding a known good alignment; the wizard saves that reference.
Then show a nearby misalignment and select **Capture endpoint**. After manually
correcting it, select **Capture endpoint** again. The software captures stable
windows after each request and accepts an example only when both-view height
error improves clearly, without material orientation, carrier or grasp change.

The accepted translation range is 0.2–3 mm, with at least 2 px combined height-error
reduction and no individual height-error increase above 1 px. This captures a
demonstrated improvement, not seating or insertion success. Select **Next demonstrated
correction** for another example, or finish and collect a distinct session.

`demonstrated-library.json` stores local start poses, image offsets, model/camera
identity, measured joint deltas and tool translations. It is an unverified library
of examples, **not an executable replay policy**. No nudge controller consumes it
automatically. Axis analysis can still be requested, but may reject demonstrations
with mixed directions or insufficient independent validation. That is expected:
teaching corrections and identifying a general response mapping are different tasks.

## 4. Board and marker calibration

Choose a wrist camera and explicitly identify the physical arm carrying it.
Choose a checkerboard with measured square size and inner-corner count, or an
ArUco `DICT_4X4_50` marker with measured side length and marker ID. Checkerboard
capture is the recommended initial intrinsic-calibration option; a single small
marker supplies fewer geometric features and needs especially diverse views.

The target must remain stationary in the same workspace/base frame throughout
all training and reserved poses. Move the wrist to provide varied tilt, distance
and image coverage. Hold still and select **Capture endpoint**. Duplicate frames
and near-duplicate wrist poses are rejected. The preview overlays detected corners
on the exact captured image. Changing camera, target geometry or resolution
requires a separate workbench run.

This workflow reads exact JPEGs and frame timestamps from the collector's recording
files and matches them to recorded measured tool poses. It does not require gear
landmark predictions or held-gear evidence. At least five observations spanning
0.6 seconds establish a stable wrist; camera-to-state skew must be within 30 ms.
The board's physical stability and arm assignment remain operator responsibilities.

Collect at least ten detected training poses and four reserved poses in distinct
sessions/episodes. Select **Analyze captured evidence** to run camera intrinsic
calibration, estimate board poses, and fit/check a camera-to-tool transform.
The reserved board's reconstructed constancy checks the hand-eye candidate.
The minimum counts alone do not guarantee enough geometric diversity.

Saved files include exact captured JPEGs, `board-observations.json`,
`camera-candidate.json`, `board-and-tool-poses.jsonl`, and `hand-eye-candidate.json`.
Image hashes and observation timestamps are retained. Inter-arm base geometry,
the demonstrated 3D assembly offset, actual retained-object detection and physical
correction verification remain separate requirements for the stereo controller.

## Shared implementation and verification

`app/calibration_workbench.py` supplies the local authenticated server and sampling
worker. `app/calibration_workflow_common.py` handles timestamp matching and stable
endpoint preparation. `app/web/calibration-workbench.html` renders the mode-specific
controls and results. Only the hold mode has a movement-request transport, and it
requires an explicit current press and the separate collector capability/gate.

The special collector consists of `app/hand_guide_sequence.py`,
`app/dashboard_sequence.py`, `app/calibration_sequence_transport.py` and
`app/calibration_sequence.py`. It stages commands through the independent motor client; the motor backend owns
all SDK access. HTTP handlers never call motor methods. Fitting and image detection run in the workbench, away from the
motor loop.

The tests use synthetic motion/geometry and rendered targets. They check complete
guided sessions, reserved fitting, demonstrated improvements, image/frame identity,
checkerboard and marker capture, HTTP authorization, movement-transport separation,
motion-gate hashes, bounded sequence completion, expired heartbeats and carrier
movement. The initial combined regression run passes 74 checks:

```bash
PYTHONPATH=tests work/teleop-venv/bin/python -m unittest \
  test_calibration_workflows test_nudge_calibration_protocols test_additional_nudges \
  test_cartesian_nudge test_alignment_servo test_supervised_alignment \
  test_robot_commands test_alignment_dashboard test_motor_backend
```

These are experimental software implementations. Browser rendering and read-only
local services were inspected; no fresh physical calibration or new motor sequence
was performed. Screens and tests do not validate the robot's actual commanded path.

The main collector and several other existing files were updated elsewhere during implementation to introduce an independent motor backend. Those edits were left untouched; the new sequence frontend was adapted to the updated motor-client interface. The earlier experimental collectors remain as they were.
