# Incremental height alignment

The launcher now selects the [relaxed supervised trial](relaxed-nudge.md), as
requested for an observed hardware experiment. The limits below describe the
original strict profile; run `app/hand_guide.py --run --incremental --alignment-assist`
without `--relaxed-nudge` to select it explicitly.

This supervised experiment uses the existing hand-guided demonstration as a
starting estimate and closes the loop with fresh images after each small vertical
step. It does not require a new checkerboard or custom calibration recording.
The first hardware attempts stopped before any visual correction. The revised
controller has not yet demonstrated physical convergence; see the
[live diagnosis](incremental-nudge-live-diagnosis.md).

## Operation

Stop the current collector before changing launchers; its motor backend remains
running in powered hold. The incremental launcher connects through the same
backend client, with exclusive frontend ownership. Start:

```bash
./start-incremental-nudge.sh
```

During an active recording, retain the gear with the existing grasp control,
check the gear and intended carrier opening in both views, and select **Align
vertically with repeated nudges**. This authorizes a bounded sequence of
corrections. **Stop nudges / release position hold** ends the sequence and returns
to gravity assistance. The sequence
also stops when recording/grasp eligibility ends. The original and one-step
launchers retain their existing behavior.

The controller holds when both vertical residuals are within 3 pixels of the
demonstrated reference. This is not necessarily zero raw pixel separation between
the gear and slot landmarks. The reference includes their expected offset at the
demonstrated alignment. Holding does not automatically restart correction.

## Feedback and limits

The existing four-by-six image-motion matrix is projected onto a pure left-base-Z
translation using the robot model. Each proposal applies a fraction of the
estimated correction, capped at 0.5 mm on hardware and 0.003 rad joint norm. Smaller residuals
produce smaller corrections. The other arm stays stationary and tool orientation
and horizontal position are constrained.

Before the first correction, three fresh stable observations are collected while
remaining in gravity assistance. A rejected start does not switch into position
hold. After a commanded correction settles, three fresh camera observations determine the next error. The
controller checks the measured response before requesting another step. It stops
on inconsistent directions, worsening error, insufficient progress, or exhausted
budgets. A view already within tolerance is allowed only if the predicted step
keeps it within tolerance. Both active views must request the same Z direction;
their estimated movement magnitudes need not match exactly.

Each start allows at most 20 steps and 30 seconds, with a 2 mm total movement
budget. Commanded path and measured settled-endpoint path are counted separately;
high-rate encoder jitter is not accumulated as travel. Every feedback update still
checks displacement and excessive step excursion. Horizontal drift is limited to
0.25 mm and orientation change to 0.005 rad.
The existing camera/checkpoint identity, freshness, landmark proximity, joint
limits, and both-arm local calibration envelopes remain enforced. The mode cannot
widen the calibration envelope through dashboard settings or send legacy torque
pulses. Physical bounds use encoders and the robot model; they are not independent
measurements of tool location.

The hardware mode uses the SDK position resolution (approximately 0.0006283 rad)
and a 0.5 mm maximum step. It rounds intended endpoints to representable motor
positions, searches nearby representable candidates when rounding exceeds a
limit, checks the resulting modeled displacement and independent joint
completion corners, and constructs commands that survive the SDK's truncating
position conversion. Endpoint verification uses stable target encoder counts
instead of requiring arbitrary continuous targets to settle within 25 micrometers.
Stable encoder counts determine hardware settling; noisy reported velocities
cannot reset the dwell while every position count remains unchanged. Each image
must still pass the existing freshness gate when accepted, while the three-image
aggregation window allows up to 0.5 seconds for inference latency. The dashboard
reports the current waiting stage and accepted image count.
If no acceptable representable step exists, it reports `motor_resolution_limited`.
The non-hardware test mode retains the original 0.25 mm cap.

The initial mapping remains fixed during a run. Feedback handles moderate gain
error through repeated corrections; this version does not train or persist a new
mapping from its own movements. A wrong response direction or incompatible
two-view target still causes a stop.

## Demonstration provenance

The runtime loads `config/calibration/gear-supervised-trial.json` through the
existing recording configuration. It preserves its measured, operator-supervised,
unverified provenance. The October 22 hand-guided session supplies the matrix,
reference joint poses, checkpoint/camera identity, and vertical target offsets
of approximately +49.39 and -68.13 pixels.

The session's temporal holdout is useful evidence for an initial estimate, but
does not establish repeated corrective performance. The stricter multi-episode
[final-approach replay](offline-alignment-validation.md) also did not validate
the proposed correction directions. Neither those results nor synthetic
closed-loop tests establish physical insertion success.

## Offline verification

Run without opening motor devices:

```bash
PYTHONPATH=tests work/teleop-venv/bin/python -m unittest \
  test_incremental_nudge test_incremental_integration test_cartesian_nudge \
  test_alignment_servo test_supervised_alignment test_alignment_dashboard \
  test_robot_commands
work/teleop-venv/bin/python scripts/validate_incremental_nudge.py
```

The recorded-pose audit checks whether bounded proposals can be constructed from
existing observations. It does not simulate camera responses to unrecorded moves
or claim convergence from replay. Closed-loop synthetic tests separately exercise
the feedback logic with deliberately inaccurate response gains and injected
failures.

The recorded-pose audit examined 443 observations. Within the 431 that met the
partial pose/proximity bounds, 287 were already within image tolerance, 11
produced a hardware-representable proposal, 116 had no accepted representable
step, eight lacked observable height response, and nine had conflicting camera
directions. These counts do not include every runtime gate
and are not success rates. Results and source hashes are saved in
`logs/alignment-calibration/incremental-nudge-offline.json`.

The revised controller, integration, backend, and related regression checks pass
73 tests. On this recorded dataset, the representable-target search took at most
21 ms per evaluated proposal. These are software checks; no physical convergence
test was performed as part of the revision.
