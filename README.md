# Panthera gravity assist and recording

Run `./start.sh` from this workspace. It checks all three cameras and both arm configurations, enables independent gravity assistance, and opens http://127.0.0.1:8090. Grippers are passive during demonstrations and open to 60% during each reset. There is no starting-pose or exit trajectory. Episode decisions return both arms to a raised neutral pose. The session started by Codex runs as the user service `panthera-gravity-assist`; inspect it with `systemctl --user status panthera-gravity-assist` and support both arms before running `./scripts/stop-gravity-assist.sh` to stop it. A running session must be stopped before launching another.

Move either arm to start an episode automatically. The recorder includes 0.75 seconds of pre-roll and requires sustained motion for 0.12 seconds to reject encoder jitter. Keep the browser page focused:

| Key | Action |
| --- | --- |
| K | Finish and keep the episode |
| R | Finish and flag the episode for review |
| D | Finish and move the episode to recoverable trash |

After K/R/D, release the handles and clear the return path. After a 2-second countdown and a brief stillness check, both arms return to `[0, 0.25, 0.25, 0, 0, 0]` radians. The model places the tool about 6.4 cm above the zero pose. Joint speed is capped at 0.60 rad/s (2× the preceding speed setting), with a 0.6-second minimum trajectory duration, a 2.0 rad/s² acceleration cap, and return torque caps are `[5, 8, 8, 3, 2, 2]` Nm. Recording stays paused throughout the return and clears its pre-roll afterward. Arrival requires both arms within 0.015 rad of target and no more than 0.004 rad of encoder motion across a 0.25-second window. Reported velocity spikes cannot reset arrival while the encoder pose is stable. Real positional error or vibration still prevents arrival. Return logs include joint error, reported speed, encoder motion, and planned duration. Both arms remain in powered position hold at neutral. Grab a handle and gently pull: a sustained change in measured motor torque releases that arm into gravity assistance, independently of the other arm. A 0.5-second hold baseline and 0.15-second confirmation reject brief torque spikes. Pull thresholds are configured per joint as `[0.6, 1.2, 1.2, 0.6, 0.25, 0.25]` Nm. Recording remains blocked while both arms are held; the first released arm rearms it and movement starts the next episode. The other arm stays held until its own handle is pulled. During the return, both grippers move to 60% of their configured travel (currently 1.2 rad) at 0.8 rad/s with a 0.5 Nm torque cap. Each becomes passive once it reaches the reset opening, so it can be closed freely. Pulling an arm handle also cancels any remaining gripper opening for that arm. An obstructed gripper is released to passive control after 5 seconds and its error is shown on the dashboard; the arms keep holding. Gripper reset outcomes are included in the return log. Return results are saved in `data/return-events.jsonl`. A rejected path, tracking deviation, or arrival timeout cancels the return, shows an error, and holds the measured poses until handle pulls release them. Joint/motor/camera faults still stop the session. The tool path is checked against dipping below its endpoint heights; this is not obstacle or collision detection. Idle pauses during a demonstration do not end it. Support both arms before stopping: Ctrl+C in the controller terminal ends assistance and sends the SDK stop command. Closing the dashboard does not stop the controller.

Episodes live under `data/kept`, `data/review`, or `data/discarded`. Discarded episodes remain recoverable and are excluded from the kept dataset. An unfinished episode goes to review on shutdown. After an abrupt interruption, pending episodes are recovered to review on the next launch. No test recordings enter these directories; simulation defaults should be directed to a temporary directory.

Each episode contains `metadata.json`, `states.jsonl`, `frames.jsonl`, and timestamped JPEGs in `images/<camera>/`. States include both arms' positions, velocities, measured torques, gripper states, motor modes/faults and SDK feedback times; sent gravity torques and zero-stiffness commands; URDF-derived tool poses; host wall-clock and monotonic timestamps; sequence numbers and control compute time. Metadata snapshots the URDFs, robot and motor configurations, USB mapping, thresholds, camera settings, and calibration status. Camera frames carry individual timestamps, indices, actual image sizes, and file paths. Match camera and joint observations by their shared host monotonic clock. These are receive timestamps, not hardware exposure synchronization.

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

Never run a separate probe or legacy experiment while control is active. The current gravity and Quest launchers share an exclusive controller lock. Keep robot USB cables in the saved hub ports.

## Postprocessed gear/carrier/pin demonstrations

The September 29 kept-session export is under
`data/processed/panthera-gear-carrier-pin-real-20hz/`, with a separate full-rate
trimmed copy under `data/processed/trimmed-full-rate/`. Original kept recordings
are preserved. The export uses LeRobot v3.0, 20 Hz, three 256 × 256 RGB views,
and native 14D bimanual joint/gripper states and next-pose actions. Images use
fixed square crops without padding; the overhead view keeps a tight 256 × 256
work area at native pixel resolution. `config/postprocessing-gear-pin.json`
records the visual release cutoffs, crop coordinates and timing/action contract.

Rebuild into new output directories with `scripts/postprocess_kept.py` and audit
with `scripts/verify_processed.py`, using `work/postprocess-venv/bin/python`.
The dataset README contains the full commands, source limitations and inference
crop requirements. These scripts only process saved files and never send motor
commands. The Hugging Face destination is
https://huggingface.co/datasets/FoxNerdSaysMoo/panthera-gear-carrier-pin-real-20hz.
