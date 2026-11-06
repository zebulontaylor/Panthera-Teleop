# Panthera gravity assist and recording

## Getting started

Switch between dashboards from the repository root:

```bash
./start-teleop.sh
./start-inference.sh
./start-regrasp.sh
```

Each launcher stops this checkout's running teleop, inference, or regrasp dashboard, lets
recording cleanup finish, and starts the selected dashboard at
http://127.0.0.1:8093. The independent motor backend keeps running. Launchers run
in the foreground; Ctrl+C stops the dashboard. Extra arguments are forwarded,
for example `./start-teleop.sh --port 8090`. Inference and regrasp start disconnected; use
its Connect button when ready. `--help` leaves the current session running.

On an already configured machine, run `./start.sh` from the repository root. A fresh clone also needs the manufacturer's [Panthera-HT SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK) and a Python environment: the local `work/` folder is excluded from Git.

### Autonomous pin regrasp collection

`./start-regrasp.sh` opens the pin collection dashboard on port 8093, switching
away from the current dashboard while retaining the independent backend hold.
The controller adapts RobotArmTraining commit `0607a3c` to real hardware.
Connect, select **Adjust by hand**, place the left tool at the table-supported
pin grasp position and angle, then **Capture reference** and **Start / resume**.
The right arm holds its captured pose. Reference capture does not move the robot.

The inference pedal classifier is reused: **single tap** always freezes/holds the
measured arm pose, **double tap** pauses/resumes, and **hold** pauses. Use the
dashboard’s **Adjust by hand** button for gravity assistance. A single tap also
ends hand adjustment and holds position; it never starts or resumes motion. Resuming
from hand adjustment captures the new measured reference automatically. Pausing
holds the measured pose and resumes the remaining trajectory; a browser heartbeat
loss also pauses. Settings apply to the next cycle. The default **Toward base
(upward)** retreat points toward the arm base at **40° elevation above horizontal**,
with an **8° cone half-angle** (adjustable from 0° to 45°). The forward approach comes from the base side and
down toward the grasp. The fixed captured grasp position sets the horizontal
bearing independently of wrist angle. Elevation is adjustable from 0° to 90° for
this base-relative mode. Legacy ±X/±Y/±Z tool retreat axes remain available and use
the captured tool orientation. Wrist jitter uses the captured tool orientation;
placement jitter stays centered on the initial position across cycles. The axis
selector specifies the **retreat** direction; the forward approach runs back
toward the pin.

Each cycle releases with a bounded linear opening ramp, with no minimum release
dwell; retreat begins on the next tick after the full-open command. Opening speed
is independent of closing speed and defaults to 4.0 rad/s (adjustable 0.1–5.0 rad/s).
A full 2 rad opening stroke takes 0.5 s at this default, versus the prior 1.875 s.
Closing behavior is unchanged. The arm approaches with the gripper open until the
remaining distance is at most 0.02 m, checked against both planned and measured
tool positions. Closing is linear and timed to reach the calibrated lower limit
at arrival; if the selected gripper speed is too low, it continues steadily after
arrival. The fingers do not close fully before reaching the pin. This avoids restarting an easing curve
or adding a minimum dwell, before lifting, placing, and settling. Cartesian speed
remains 0.015 m/s by default with a 0.15 m/s ceiling; placement jitter remains zero
by default with a 0.10 m maximum. Joint speed defaults to 0.15 rad/s (adjustable
0.01–0.6 rad/s), and gripper closing speed defaults to 2.0 rad/s (adjustable
0.1–5.0 rad/s). Speed settings are ceilings, not guaranteed achieved rates: IK
bounds can keep motion below them. The
existing gripper torque limit remains in effect. Local IK and joint/speed limits
are checked before execution. This supervised hardware mode has no simulated
contact, support, or lift-success sensor; monitor the left wrist view and pause
if a grasp fails. Saved episodes are labeled `unverified`, never automatically
successful.

Only autonomous **approach (`return`) and close** are saved, as separate episodes
under `data/pin-regrasp-forward/<episode-id>/`: `left_wrist.avi`, timestamped
`state`/`action` samples in `samples.jsonl`, and `metadata.json`. Only the configured
left wrist camera is opened. Human adjustment, release, retreat, lift, and placement
are excluded. Pause/intervention finishes a partial episode; resumed forward motion
starts a new episode marked `resumed_mid_forward`. Original camera timestamps and
sequence numbers distinguish reused frames when capture is slower than 30 Hz.
These files use their own schema and are not automatically added to HITL datasets.

For a separate disconnected preview without switching dashboards:
`scripts/start-pin-regrasp-dashboard.sh --port 8092`. Use the switching launcher
before connecting if another dashboard owns the cameras or motor lease.

### Autonomous pin alignment collection

The same `./start-regrasp.sh` dashboard now offers **Pin alignment** under
**Collection task**. While disconnected, select it and Apply settings, then
connect and choose **Adjust by hand**. Both arms enter gravity assistance in
this mode: hold the pin and carrier in their grippers and position the pin tip
about 1 mm above the hole. **Capture reference**, clear the motion area, then
**Start / resume**. Capturing does not infer hole position or verify clearance.

Alignment retreats along the selected angled path and returns with fixed wrist
orientation and grippers. In Toward base mode, **Cone clockwise rotation** defaults
to 40° clockwise viewed from above (+Z), toward the pickup side. This rotates the
horizontal bearing without changing elevation or cone width. Set it to 0° to
restore the direct baseward bearing; negative values rotate counterclockwise.
Regrasp and explicit tool-axis directions are unaffected. **Carrier placement jitter** adds sideways carrier
dragging during the pin retreat, centered on the original reference (0–0.10 m radius).
Both arms share the same eased progress: the pin follows the carrier displacement
while retreating relative to the hole, then returns while the carrier holds. Both
arms are preflighted together and the shared duration expands if either reaches a
speed limit. This does not establish whole-arm collision clearance.
The carrier holds still during the recorded return. Dragging uses the selected
**Motion speed**, subject to the **Joint speed ceiling** and trajectory easing. There is no insertion, release, close, or lift phase.

To enable carrier jitter, capture the aligned reference, set **Carrier placement
jitter**, Apply settings, and start. No inter-arm calibration is required for
this installation: the arm bases have matching parallel axes, so carrier and pin
displacements use the same XYZ directions. Base spacing cancels because targets
are offsets from the captured reference, rather than absolute points transferred
between bases. Setting jitter back to zero returns to the original reference.

The tabletop must be horizontal in the left arm base frame. This geometric mapping
does not sense object slip, contact, or hole clearance; outcomes remain unverified.

Only the forward `return` is recorded, with left wrist, right wrist, and overhead
videos plus 14 joint-position targets in radians. Alignment opens all three
configured cameras and shows their previews. Each sample preserves the capture
timestamps and sequence numbers for each stream; cameras are not hardware
synchronized. All three streams must remain healthy. Camera faults identify the logical and
physical camera, last frame age, and capture error; the last health snapshot is
retained after a fault. Alignment episodes use
schema version 2 with a video map and per-camera receipts; regrasp keeps its
existing single-camera format. Episodes remain `unverified` and are saved separately under
`data/pin-alignment-forward/` (or `<custom --data-dir>/alignment/`). Pause and
intervention terminate partial episodes, as in regrasp mode. The existing
regrasp exporter excludes alignment episodes to avoid labeling them as grasps.
Task changes require disconnecting and capturing a new reference after reconnecting.

### Pin alignment in simulation

The companion [RobotArmTraining repository](https://github.com/zebulontaylor/RobotArmTraining)
provides a simulation version of the local retreat/return collection pattern.
From that checkout, with its actuator dependencies installed, run:

```bash
MUJOCO_GL=egl .venv-assembly/bin/python tools/collect_pin_regrasps.py \
  --task alignment --output outputs/pin_alignment/first_run \
  --episodes 3 --max-attempts 10 --image-size 128 --save-failures
```

Setup establishes a held-pin reference about 1 mm above the carrier hole.
Recorded motion approaches that reference along the pickup's angled path from
the arm-base side, then backs away and repeats. Both horizontal and vertical
motion follow the same eased path. The grippers stay fixed, and the final return
ends at the aligned reference. Between approaches, the right arm drags the
carrier sideways on the table, then holds still while the pin approaches its
new hole location. The alignment setup places the carrier on bare tabletop,
clear of the fixed assembly fixture. Each reset offsets the pin and carrier
together; the carrier grasp keeps its modeled contact position and recorded wrist
orientation, with reachability and contact checks at each reset.
Successful attempts label forward return motion; repositioning, retreat, and
failed attempts are excluded from training. The controller uses privileged simulator geometry.
Its actions are 14 normalized increments for both arms, distinct from the regrasp
simulator's 13-value Cartesian targets and this dashboard's joint targets. See
`sim/actuator/pin_regrasp.md` in the companion checkout for cycle settings,
the dataset contract, and the preview command.

### Exporting pin regrasp episodes

The retained collection has a separate LeRobot v3 export at
`data/processed/panthera-pin-regrasp-100-20hz`, published at
[ZebTaylor/panthera-pin-regrasp-100-20hz](https://huggingface.co/datasets/ZebTaylor/panthera-pin-regrasp-100-20hz)
(100 episodes, 4,222 samples). It uses real left wrist images
(`wrist_port3`) and real measured state/next-state actions on a 20 Hz grid.
Overhead and right-wrist (`wrist_port2`) images are paired random donor views
from heuristic pin-pickup windows in the single-bolt insertion dataset. These
auxiliary views are deliberately unrelated to the new episode's motion; per-frame
source identities and image hashes are in `processing/alignment/`. Grasp success
is unverified. Original source recordings and discarded episodes are unchanged.

Use `work/postprocess-venv/bin/python` with these scripts:

- `scripts/select_pin_grasp_donors.py` selects and previews pickup-stage donor frames.
- `scripts/export_pin_regrasp_dataset.py --source data/pin-regrasp-forward --donor-manifest logs/pin-regrasp-donor-manifest.json --output <new-directory>` creates a new export and refuses an existing destination.
- `scripts/verify_pin_regrasp_dataset.py --dataset <export-directory>` checks every state, next-state action and image against source files, verifies native LeRobot loading, and creates a preview.

Camera captures must align within 60 ms. Stale heads/tails are trimmed only in the
export and disclosed in its report; internal camera gaps reject export. Discarded,
incomplete, or erroneous episodes are excluded. The export's dataset card explains
the exact action, camera-substitution, and sampling conventions.

### Single-arm hand guidance

For a connected right arm only, start the motor backend in one terminal and
the hand-guidance client in another:

```bash
work/teleop-venv/bin/python app/motor_backend.py --run --arm right
work/teleop-venv/bin/python app/single_arm_hand_guide.py --run --arm right
```

This mode provides gravity assistance and a passive gripper, without recording
or automatic trajectories. Ctrl+C in the client leaves the backend holding the
measured pose.
The backend defaults to both arms when `--arm` is omitted.

### 1. Install the software

The commands below use Linux x86_64, Python 3.10, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The vendor's prebuilt motor wheel targets Linux x86_64 and was built on Ubuntu 22. For other platforms or binary compatibility issues, follow the [SDK's source-build instructions](https://github.com/HighTorque-Robotics/Panthera-HT_SDK/blob/4a267c148cffa56b369222110bcd0b925809e365/panthera_python/README.md).

```bash
git clone https://github.com/zebulontaylor/Panthera-Teleop.git
cd Panthera-Teleop
mkdir -p work
git clone https://github.com/HighTorque-Robotics/Panthera-HT_SDK.git work/Panthera-HT_SDK
git -C work/Panthera-HT_SDK checkout 4a267c148cffa56b369222110bcd0b925809e365

uv venv --python 3.10 work/teleop-venv
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python work/teleop-venv/bin/python \
  work/Panthera-HT_SDK/panthera_python/motor_whl/hightorque_robot-1.2.0-cp310-cp310-linux_x86_64.whl \
  -r work/Panthera-HT_SDK/panthera_python/requirements.txt \
  -r requirements-recording.txt 'pin==4.1.0'
```

The SDK revision and Pinocchio version above match this setup. `UV_SKIP_WHEEL_FILENAME_CHECK=1` accommodates the vendor wheel's filename/metadata version mismatch. The launchers use `work/teleop-venv/bin/python` directly, so activating the environment is optional.

Check imports, then try the dashboard in simulation:

```bash
work/teleop-venv/bin/python -c 'import hightorque_robot, pinocchio, numpy, yaml, cv2'
./start.sh --demo --data-dir /tmp/panthera-demo
```

Simulation opens the dashboard at http://127.0.0.1:8090 without opening motor devices. Stop it with Ctrl+C before starting live control.

### 2. Configure your hardware

Live recording uses two Panthera arms and three USB cameras. Connect the arms through their communication sockets, power their motor supplies, and enable motor power with the round buttons. Your user must have access to the robot serial devices and camera devices.

The optional PCsensor USB FootSwitch finishes and classifies an active episode:
one tap keeps, two flag for review, and three delete to recoverable trash.
For first-time Linux access, run `sudo ./scripts/enable-foot-pedal.sh`.
Connect the pedal before starting `./start.sh`; the dashboard displays its
connection and classification status. The pedal is used only for episode
keep/review/delete decisions; it does not annotate force or start episodes.
The recorder exclusively captures pedal keystrokes so they cannot activate
dashboard shortcuts. Ordinary keyboard input is unaffected.
If a connected pedal is unplugged or loses events, the session stops and preserves
the unfinished episode for review. If absent at launch (or disabled through
`foot_pedal.enabled` in `config/recording.json`), use K/R/D on the dashboard.
Hardware-free demo mode does not open the pedal. To check the pedal independently
while control is stopped, run `work/teleop-venv/bin/python app/foot_pedal.py` and
press/release it.

The checked-in USB mapping and camera paths describe the original laptop. On another machine, update:

- `config/robot-configs/arm-mapping.json`: set each arm's `usb_path` and `pci_controller` from its Linux USB topology. Inspect `readlink -f /sys/class/tty/ttyACM*/device` and identify which physical arm is left/right. The tested boards share a serial number, so serial-number links cannot distinguish them.
- `config/recording.json`: set the three `cameras` paths using `ls -l /dev/v4l/by-path/` and confirm each view. Close other applications using the cameras.

Check the mappings and URDFs without opening motor devices:

```bash
./scripts/start-hand-guide.sh
```

This regenerates the left/right robot YAMLs, including absolute SDK/model paths, for your checkout. Keep the robot cables in the configured hub ports afterward.

### 3. Start recording

Use the **Trial mode** selector on the Episode desk to switch between
**Single bolt insertion · left arm** and **Regular testing** between trials,
after reset. Switching takes effect immediately for the next trial without
restarting or moving the arms. The current held pose is unchanged. Regular
testing restores both-arm triggering and resets, with pre-roll and no setup
delay. The selection lasts for the current session; startup uses the settings
in `config/recording.json`.

The current startup configuration is for **left-arm bolt insertion trials**.
Only left-arm motion starts a take. After motion triggers recording, the first
2 seconds are a setup window: move the left arm from neutral to the pose that
represents having just inserted the gear, then pick up and insert the bolt.
A short rising three-note chime sounds when the first state sample is saved
after the setup window. State samples and camera images from that window
(including pre-roll) are not saved. Saved timestamps start at the end of the window. Both arms' subsequent
state data remain in the usual dataset format. Takes ended before any saved
state sample are sent to review instead of being kept as empty demonstrations.

Finishing a take returns only the left arm to neutral and opens only its gripper;
the right arm continues its existing control without a reset. Pull only the left
handle to resume. These settings are `motion_arms: ["left"]`,
`start_delay_seconds: 2.0`, and `neutral_return.arms: ["left"]` in
`config/recording.json`. Restart the recording frontend to apply changes. Set
both arm lists to `["left", "right"]` and the delay to `0` to restore dual-arm
recording with pre-roll.

Optional CNN gear/pin alignment assistance is available from the recording dashboard.
The current configuration enables a supervised gear trial using the October 22
hand-guided calibration. Hold the dashboard's test-nudge button to authorize one
bounded pulse; release stops assistance. Automatic nudges still require verified
calibration and validated grasp/visibility evidence. Pin is not enabled for the
supervised trial.
See the [alignment assist setup](docs/alignment-assist.md).
The [recorded calibration analysis](docs/recorded-alignment-calibration.md) uses
the 17 kept October 22 trials and reports held-out validation separately from
the CNN visibility and grasp checks.

For repeated, feedback-controlled vertical corrections using the existing
demonstrated reference, use `./start-incremental-nudge.sh` and the dashboard's
**Align vertically with repeated nudges** action. This separate supervised experiment checks fresh
images between timed moves and holds when both views are within tolerance.
The launcher now selects the user-requested [relaxed supervised trial](docs/relaxed-nudge.md):
up to 5 mm per move and 50 mm total travel, with fresh feedback between moves.
Supervised mode replans from measured joint positions without requiring exact
settling or confirmed visual progress; hardware limits and stop controls remain.
See [incremental height alignment](docs/incremental-nudge.md) for the underlying
controller and strict profile. The revised mode has not yet demonstrated
convergence on hardware.

With the return path clear, run:

```bash
./start.sh
```

The dashboard opens at http://127.0.0.1:8090. Guide the arms by their physical handles; K keeps an episode, R flags it for review, and D discards it. Each decision triggers a raised-neutral return and gripper opening. Ctrl+C ends the frontend session; the independent motor backend retains powered position hold. Recordings are saved under `data/` and are excluded from Git. The recorder reserves 2 GiB of free disk space.

For optional Quest controller operation, also clone [Quest_controller_stream](https://github.com/HighTorque-Robotics/Quest_controller_stream) into `work/Quest_controller_stream`, install its Python/UI dependencies into the same environment, and install Android `adb`. Follow [the Quest guide](docs/quest-quick-start.md); the preserved passthrough changes are in `archive/setup/quest-passthrough.patch`. Physical handle mode uses the Panthera SDK without a headset or Quest bridge.

## Independent motor backend

Teleop and inference now connect to a separate motor-controller process. The
backend owns the SDK, motor loop, feedback checks, command caps, and control lease.
It starts as the independent `panthera-motor-controller` user service on the first
live connection. Frontend exit or lease expiry leaves the arms in powered position
hold with passive grippers. Explicitly stop powered control with `python3 scripts/motor-controller.py stop`.
See [the architecture and lifecycle guide](docs/motor-controller.md).

## Robot command interface

The backend accepts structured commands and supported plain-English instructions
through `/api/commands`. Use `python3 scripts/robot-command.py --list` to inspect
available actions. See [the command API guide](docs/robot-commands.md) for examples,
authentication, and execution semantics.

## Recording and control

Run `./start.sh` from this workspace. It checks all three cameras and both arm configurations, enables independent gravity assistance, and opens http://127.0.0.1:8090. Grippers are passive during demonstrations and open to 60% during each reset. There is no starting-pose or exit trajectory. Episode decisions return the configured reset arms to a raised neutral pose. The session started by Codex runs as the user service `panthera-gravity-assist`; inspect it with `systemctl --user status panthera-gravity-assist` and use `./scripts/stop-gravity-assist.sh` to stop that frontend. Run `python3 scripts/motor-controller.py stop` to stop the motor backend and end powered control. A running session must be stopped before launching another.

Move either arm to start an episode automatically. The recorder includes 0.75 seconds of pre-roll and requires sustained motion for 0.12 seconds to reject encoder jitter. Keep the browser page focused:

During recording, the pedal finishes and classifies the active episode at any
time: **one short tap keeps, two flag for review, three delete to recoverable
trash**. No gripper gesture or automatic end cue is required. Tap length is
0.04–0.5 seconds; pause 0.65 seconds after the last release to confirm.
Release the handles and clear the return path first: the **first press immediately
ends capture and starts the neutral return**, while further taps select the label
during the return. Only saving the final label waits for the tap window. Long
holds and groups above three taps cancel the group and save the take for review,
marked incomplete, with a low cancellation tone. K/R/D remain available while an
episode is recording. Interrupted classification also preserves the take for review.
The pedal does not start episodes or record downward-force requests.

Every classification press clicks, and a confirmation tone announces the saved
category (one high tone keep, two review, three descending delete). Local audio
uses `pw-play`, `paplay`, or `aplay`, runs without browser focus, and reports
playback failures on the dashboard. Set `audio_cues.enabled` to false to disable it.

| Key | Action |
| --- | --- |
| K | Finish and keep the episode |
| R | Finish and flag the episode for review |
| D | Finish and move the episode to recoverable trash |

Release the handles and clear the return path before pressing K/R/D. Each decision immediately starts returning both arms to `[0, 0.25, 0.25, 0, 0, 0]` radians. The model places the tool about 6.4 cm above the zero pose. Joint speed is capped at 0.60 rad/s (2× the preceding speed setting), with a 0.6-second minimum trajectory duration, a 2.0 rad/s² acceleration cap, and return torque caps are `[5, 8, 8, 3, 2, 2]` Nm. Recording stays paused throughout the return and clears its pre-roll afterward. Arrival requires both arms within 0.015 rad of target and no more than 0.004 rad of encoder motion across a 0.25-second window. Reported velocity spikes cannot reset arrival while the encoder pose is stable. Real positional error or vibration still prevents arrival. Return logs include joint error, reported speed, encoder motion, and planned duration. Both arms remain in powered position hold at neutral. Grab a handle and gently pull: a sustained change in measured motor torque releases that arm into gravity assistance, independently of the other arm. A 0.5-second hold baseline and 0.15-second confirmation reject brief torque spikes. Pull thresholds are configured per joint as `[0.6, 1.2, 1.2, 0.6, 0.25, 0.25]` Nm. Recording remains blocked while both arms are held; the first released arm rearms it and movement starts the next episode. The other arm stays held until its own handle is pulled. During the return, both grippers move to 60% of their configured travel (currently 1.2 rad) at 0.8 rad/s with a 0.5 Nm torque cap. Each becomes passive once it reaches the reset opening, so it can be closed freely. Pulling an arm handle also cancels any remaining gripper opening for that arm. An obstructed gripper is released to passive control after 5 seconds and its error is shown on the dashboard; the arms keep holding. Gripper reset outcomes are included in the return log. Return results are saved in `data/return-events.jsonl`. A rejected path, tracking deviation, or arrival timeout cancels the return, shows an error, and holds the measured poses until handle pulls release them. Joint/motor/camera faults still stop the session. The tool path is checked against dipping below its endpoint heights; this is not obstacle or collision detection. Idle pauses during a demonstration do not end it. Ctrl+C in the teleop terminal releases its control lease; the motor backend retains powered position hold. `python3 scripts/motor-controller.py stop` ends powered control. Closing the dashboard does not stop the controller.

Episodes live under `data/kept`, `data/review`, or `data/discarded`. Discarded episodes remain recoverable and are excluded from the kept dataset. An unfinished episode goes to review on shutdown. After an abrupt interruption, pending episodes are recovered to review on the next launch. No test recordings enter these directories; simulation defaults should be directed to a temporary directory.

Each episode contains `metadata.json`, `states.jsonl`, `frames.jsonl`, and timestamped JPEGs in `images/<camera>/`. States include both arms' positions, velocities, measured torques, gripper states, motor modes/faults and SDK feedback times; sent gravity torques and zero-stiffness commands; URDF-derived tool poses; host wall-clock and monotonic timestamps; sequence numbers and control compute time. Metadata snapshots the URDFs, robot and motor configurations, USB mapping, thresholds, camera settings, and calibration status. Camera frames carry individual timestamps, indices, actual image sizes, and file paths. Match camera and joint observations by their shared host monotonic clock. These are receive timestamps, not hardware exposure synchronization.

Hand-guidance states preserve the pedal connection, physical switch state, and
sample timestamp under `foot_pedal`. `foot_pedal.events` preserves each
press/release since the previous state sample, including short taps between
control ticks, with Linux input key codes and host monotonic event timestamps.
Pedal signal semantics and device paths are saved in metadata under
`context.foot_pedal`; pedal input is used only for episode classification.
Metadata records `end_reason=foot_pedal` for pedal decisions. Subsequent taps
during the return are saved separately in `pedal-decision-events.jsonl`;
motor-driven return states stay out of the episode.

Control targets 150 Hz; actual achieved rate can be measured from the state timestamps. Cameras target 20 Hz at 640 × 480 with JPEG quality 85. Recording I/O runs on a separate thread with a bounded queue. The session stops and preserves its active episode for review on recording failure, lost camera frames, invalid/stale motor feedback, motor faults, or joint-limit violations. Two GiB of free disk space is reserved. Joint limits and the existing gravity torque caps are retained. Camera intrinsics, hand-eye transforms, and wrist-to-arm assignments remain unverified; each tool pose is expressed in that arm's model base frame.

The folders are organized by purpose:

- `app/`: controller, recorder, and local browser dashboard.
- `config/`: USB mapping, active robot YAMLs, camera and recording settings, calibration materials.
- `scripts/`: alternate Quest launcher, headset setup, configuration-only hand-guide launcher, and diagnostic probe.
- `data/`: classified episodes; `logs/`: runtime logs and the controller lock.
- `tests/`: hardware-free episode and control checks.
- `docs/`: setup history and Quest instructions.
- `archive/`: earlier experiments, result JSONs, camera captures, old board configs, and setup downloads.
- `work/`: vendor SDK, Quest bridge, Python environment, and Android platform tools. Keep these dependencies in place.

Check configuration without enabling motors: `./scripts/start-hand-guide.sh`.

Test without motors: `./start.sh --demo --data-dir /tmp/panthera-demo`; add `--demo-cameras` to include real camera capture. Simulation is clearly marked in the dashboard and metadata.

Run hardware-free checks: `work/teleop-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`.

Never run a separate probe or legacy experiment while control is active. The motor backend and legacy Quest launcher share an exclusive controller lock; stop the backend before using a legacy controller. Keep robot USB cables in the saved hub ports.

If the serial connection drops during an idle hold, install the targeted USB protection once, with the controller stopped:

```bash
sudo ./scripts/enable-robot-usb.sh
```

Reconnect both robot USB cables afterward, then launch normally. This disables autosuspend for the Livelybot boards and the tested Terminus robot hub, and excludes the boards' serial ports from ModemManager probing. At startup the hand-guidance controller also keeps each board's upstream USB ancestry awake when permissions allow; otherwise it prints the setup command. These settings mitigate host power management and probing; they cannot prevent cable, supply, or firmware resets. The September 30 failure coincided with kernel USB disconnect/re-enumeration events for both boards, while control commands were still being sent.

A recognized serial disconnect now ends control cleanly with exit status 1, attempts to stop both arms, and finalizes any unfinished episode for review, without a Python traceback. Restart after checking the USB connection. Control does not automatically resume through the SDK's background reconnect path, which clears motor objects referenced by the Python wrapper.

If a board appears in `lsusb` but no `/dev/ttyACM*` port appears, startup distinguishes an unconfigured USB device from a missing board. Kernel messages such as `rejected 1 configuration due to insufficient available bus power` mean Linux refused to configure the device. Check the hub's external power connection or use a powered hub, then reconnect it. Autosuspend settings do not fix a missing USB power budget.

## Postprocessed gear/carrier/pin demonstrations

**Camera setup disclaimer (October 2, 2026, America/Chicago):** The overhead camera was repositioned on this date. Recordings collected after the change have a different overhead viewpoint from earlier recordings, so the visual data are not identical across collection sessions. Future Hugging Face releases that include these recordings will contain both camera setups. The existing September 29–30 export predates this change. Use the original episode collection timestamps in `processing/source_metadata/` to identify the collection date; the date alone does not identify whether an October 2 episode was recorded before or after the camera move.

The September 29 and 30 sessions contain 102 kept demonstrations. Their combined
export is under `data/processed/panthera-gear-carrier-pin-real-20hz-102/`.
The first 30 episodes were reused from
`data/processed/panthera-gear-carrier-pin-real-20hz/`; only the 72 new episodes
were transformed, under `data/processed/panthera-gear-carrier-pin-sept30-20hz/`.
Full-rate trimmed copies are under `data/processed/trimmed-full-rate/` and
`data/processed/trimmed-full-rate-sept30/`, and every episode also has an archive
in the combined export. Original kept recordings are preserved.
The export uses LeRobot v3.0, 20 Hz, three 256 × 256 RGB views,
and native 14D bimanual joint/gripper states and next-pose actions. Images use
fixed square crops without padding; the overhead view keeps a tight 256 × 256
work area at native pixel resolution. `config/postprocessing-gear-pin.json`
records the original 30 visual release cutoffs, crop coordinates and timing/action
contract; `config/postprocessing-gear-pin-sept30.json` annotates the new 72.

Rebuild into new output directories with `scripts/postprocess_kept.py` and audit
with `scripts/verify_processed.py` and `scripts/verify_full_rate.py`, using
`work/postprocess-venv/bin/python`. `scripts/merge_processed.py` combines compatible
processed batches while checking source identity and preserving embedded JPEG
bytes. `scripts/publish_processed.py --update-existing <combined-export>` updates
the existing Hugging Face repo after verification, preserving a snapshot tag of
the previous release.
The dataset README contains the full commands, source limitations and inference
crop requirements. These scripts only process saved files and never send motor
commands. The Hugging Face destination is
https://huggingface.co/datasets/FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz.

## Trim recovery clips from flagged runs

Run `python3 scripts/review_runs.py`, then open http://127.0.0.1:8091.
This standalone tool reads saved recordings only. It does not connect to the
arms or cameras, and can run alongside the recording dashboard.

The queue includes all Kept recordings as well as Review recordings, while
preserving their categories. Use the category filter to show only Kept runs.
Choose a run from the review queue and drag across the selection timeline to
keep the recovery. Drag either handle to refine the start or end; the camera
views follow the boundary as you drag. Click the timeline to seek without changing
the selection. Focus a handle and use arrow keys for 0.05-second adjustments
(Shift for 1 second). You can also set **Keep from** and **Keep until** using
the playhead buttons or exact seconds. The three camera
views use the shared host recording clock. Arrow keys step overhead camera frames;
Space toggles playback, I/O mark boundaries, and S saves. Slow playback and
**Preview kept range** help check the cut. One contiguous interval is retained
per run. Use **Save & next unedited** to work through the queue. Saved badges,
a progress count, and the Edited / Not edited filter track work across restarts.
Switch runs freely without a warning; unfinished selections and note text stay
in memory while the page is open. Use **Save notes only** to persist notes without
approving a crop or marking the run edited. Notes live in the companion
`config/postprocessing-review.review-state.json` file. **Change category** moves
the original recording between Review, Kept, and Discarded and updates its label.
Moved runs remain accessible in the queue, including through category filters.
Discarding removes the run from the crop export list; moving it back does not
automatically approve it again.

Edits are atomically saved to `config/postprocessing-review.json` in the same
annotation format as the regular postprocessor. Only saved runs appear in that
file; crop saves leave original recordings and their labels unchanged. Category
changes update their folder and label; saved crops track that source category. The default
processing settings and image crops come from
`config/postprocessing-gear-pin-sept30.json`. Override `--template`, `--source`,
`--annotations`, or `--port` for a different task or recording folder.

Export the saved selections using new output directories:

```bash
work/postprocess-venv/bin/python scripts/postprocess_kept.py \
  --source data/review \
  --annotations config/postprocessing-review.json \
  --output data/processed/review-recoveries-20hz \
  --trimmed-output data/processed/review-recoveries-full-rate \
  --repo-id FoxNerdSaysMoo/panthera-review-recoveries
```

The exporter requires explicit `review_approved: true` annotations for review
runs, which the editor adds on save. It retains the normal complete-recording,
alignment, camera-coverage, and minimum-sample checks. An interrupted recording
is shown with a warning in the editor and is not eligible for this exporter
without separate source validation. Exporting does not upload or publish data.

## Human-in-the-loop policy corrections

The policy dashboard now supports recorded rollouts with **H** to take over into
hand guidance and automatically start recording at takeover, **P** to end a
correction and hold, then **Play** to resume with a fresh prediction. Recording a
full rollout before Play is optional. Keep/review/discard outcomes include quality, notes, operator
and scene. Export selects human correction segments plus successful unassisted
rollouts, with separate episode boundaries for every correction.

Try `./scripts/start-inference-dashboard.sh --demo --port 8094` without hardware.
For live collection, run the existing inference tunnel and launch the same command
without `--demo`; recordings go to `data/hitl/`. See the
[collection and export guide](docs/human-in-the-loop.md) for the full workflow.

## Experimental EXPO-FT reinforcement learning

The [EXPO-FT guide](docs/expo-ft.md) covers the new lightweight edit actor/Q
learner, exact policy-decision recording, genuine pi0.5 candidate caches, and
separate guarded inference/training commands. The laptop passed a batch-256
synthetic training check. Real Panthera RL still needs fresh eligible rollouts;
the existing SFT exports cannot substitute for executed-action replay.
