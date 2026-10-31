# Calibration from the October 22 kept trials

The stricter fixed-split final-approach checks and closed-empty grasp challenge
are now documented in [offline validation](offline-alignment-validation.md).
They do not validate automatic nudges; the broader statistical result below
must not be used as motion authorization.

Correcting the wrist-camera routing changed the result substantially. The gear
mapping now passes the existing offline statistical checks. The pin mapping
improves but narrowly fails a stability check. Neither is yet verified for robot
motion because real grasp/visibility evidence and pin-bore association remain
unresolved. The earlier wrong-order rejection does not demonstrate that the
17 videos are insufficient or that the CNNs require retraining.

The routing investigation and corrected prediction overlays are described in
[the input investigation](cnn-input-investigation.md). Historical training port
names were mistaken for stable camera identities: the current physical
gear/pin and carrier views are reversed relative to those names. Current model
inputs must be `overhead, wrist_port3, wrist_port2`, with left tool XYZ followed
by right tool XYZ unchanged.

## Corrected results

All 17 kept trials contain 367.56 seconds, 54,523 state samples, and 22,004
images. The corrected 10 Hz approach-phase pass processed 636 synchronized
three-camera observations with no timestamp rejects. Cache hashes, observation
counts, episode identities, and phase-selection hashes verify.

| Task | Local motion pairs | Held-out RMSE | No-motion baseline | Direction accuracy | Statistical checks |
| --- | ---: | ---: | ---: | ---: | --- |
| Gear | 85 | 17.42 px | 24.68 px | 89.2% | Pass |
| Pin | 88 | 46.46 px | 51.75 px | 87.7% | Fails bootstrap stability |

Gear prediction improves over the no-motion baseline by 29.4%, and over the
other-arm-only baseline by 53.9%; minimum bootstrap direction consistency is
98.4%. Pin improves by 10.2% and 10.7%, respectively, but minimum bootstrap
direction consistency is 89.06%, below the unchanged 90% acceptance threshold.
These errors measure image-motion prediction, not the point-localization
errors in the routing experiment or physical insertion accuracy.

Candidate matrices, reference poses, confidence intervals, and trial-held-out
folds are saved in
`logs/alignment-calibration/calibration-report.corrected-order.json`, with a
readable summary at `calibration-report.corrected-order.md` in the same folder.
The gear `statistical_candidate_usable` flag passes; actual `candidate_usable`
and `verified` remain false because the fit uses ungated image coordinates.

## Evidence and limits

The fitter uses short within-trial differences of landmark error and measured
joint angles. It controls for both-arm motion, retains independently observed
joint directions, and limits the candidate to a local region within 0.3 rad of
both reference poses. Validation leaves out whole trials and bootstraps whole
trials. Adjacent images are not randomly split into training and validation.
Matrices use physical left-wrist x/y followed by physical right-wrist x/y.

The phase selections in `config/calibration/recorded-alignment-trials.json` are
coarse reviewed intervals, not per-frame insertion-success truth. A separately
reviewed final gear approach is preserved in
`config/calibration/recorded-alignment-near-trials.json`. The old selection's
historical input-camera-order note is superseded by the current routing fix;
the episode windows and image/joint timestamps are unchanged.

Real adaptation trained positions without real visibility loss. Low raw
visibility outputs are therefore not evidence of poor real localization.
The pin-tip visibility head saw zero synthetic visible examples in one wrist
slot; the gear entrance is an amodal reference. The original blanket real
visibility gate is not a validated real-world detector. Passive gripper torque
also does not reliably distinguish a retained grasp. Manual labels now provide
visible/hidden/ambiguous points and held-object examples without converting
ambiguous points into false visibility negatives.

Pin bore slots are unordered. Choosing a nearby predicted bore independently
in each camera does not verify the same physical bore. The diagnostic pin fit
retains that limitation explicitly. Nothing in the offline analysis establishes
collision-free motion or insertion success.

## Reproduce and preserve provenance

```bash
../RobotArmTraining/.venv-cnns/bin/python scripts/alignment_calibration_infer.py \
  --camera-names overhead wrist_port3 wrist_port2 \
  --phase-manifest config/calibration/recorded-alignment-trials.json \
  --output logs/alignment-calibration/predictions.phases10hz.corrected-order.jsonl \
  --fps 10 --resume

../RobotArmTraining/.venv-cnns/bin/python scripts/alignment_calibration_analyze.py \
  --predictions logs/alignment-calibration/predictions.phases10hz.corrected-order.jsonl \
  --audit config/calibration/recorded-alignment-trials.json \
  --output logs/alignment-calibration/calibration-report.corrected-order.json
```

The source recordings remain unchanged. The scripts import no robot-control
interface and send no motor commands. Source JPEGs, metadata, state streams,
frame indexes, and checkpoint weights are hashed. Analysis rejects mismatched
cache bytes/counts/episode identities or phase-selection hashes.

The earlier files `predictions.jsonl`, `predictions.phases10hz.jsonl`,
`calibration-report.json`, and `calibration-report.near-alignment3hz.json`
preserve the wrong-order investigation. They are superseded for assessing the
current models and must not be used to activate assistance.
