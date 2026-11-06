# Held pin CNN: varied grasps and extreme shadows

Latest retraining: 2026-11-05. The default checkpoint is `work/held-pin-cnn/best.pt`, schema `held-pin-left-center-axis-y-v2`.

The model estimates proximal cap-center TOP Y and inferred distal BOTTOM Y on the configured gripper centerline, native X=335 at width 640. Physical endpoint identities are preserved, including cases where their image-Y order reverses. RGB enters a coarse stage and a native-detail ROI stage. BOTTOM remains inferred; these outputs are not verified real distal-tip measurements.

## Data and training

The synthetic dataset increased from 849 to 10,188 rendered images: 12 deterministic grasp variants for each of 849 source anchors, across 33 acquisition/pose groups. Splits are 6,480 train (22 groups), 1,380 validation (5 groups), and 2,328 test (6 groups). Variants retain the source group split, including across real and simulated domains.

Grasp hypotheses vary pin midpoint optical Y/Z by ±5 mm, add independent optical-Z depth by ±5 mm, and tilt optical X by ±25 degrees. Recorded arms, camera, mesh and pin dimensions remain fixed. These are kinematic grasp hypotheses; physical jaw contact and real grasp distributions are not certified.

Each split includes ordinary, hard-shadow and extreme-shadow rendered lighting. Training also uses hard/soft cast shadows, local and stripe shadows, exposure, gradients, vignettes, glare, noise, blur, affine framing and visibility-aware cutouts. Combined shadow attenuation reaches 97%; 25% of shadow draws retain ordinary illumination. Photometric shadows preserve labels; opaque cutouts mask covered visible targets. Lighting fields are computed at up to 120×160 then interpolated.

The real labels remain 37 training and 23 validation TOP coordinates from one acquisition and likely one grasp. No additional ambiguous gear-like images were relabeled as pins. There are no verified real BOTTOM labels or real test positives.

Training fine-tuned the prior model for 60 epochs at AdamW learning rate 0.0002, batch 24, real-repeat 96, seed 721. This produced 601,920 stochastic sample presentations; repeats are not independent new labels. Selected epoch: 16. Selection used validation only: ordinary real TOP + 0.2 × sim BOTTOM + 0.5 × shadow real TOP + 0.1 × shadow sim BOTTOM. Final test scores did not select weights.

## Frozen before/after comparison

Errors are absolute Y pixels at native 480-pixel image height. Real validation is a narrow, correlated domain. Extreme shadows here include deterministic image-level stress on top of the rendered or real image.

| Split / target / condition | Before mean / p90 | After mean / p90 | Before / after misses >10px |
|---|---:|---:|---:|
| validation real TOP ordinary | 2.11 / 3.25 | 1.75 / 2.99 | 0 / 0 |
| validation real TOP extreme shadows | 48.33 / 102.71 | 1.90 / 3.36 | 18 / 0 |
| validation sim BOTTOM ordinary | 22.76 / 68.29 | 5.07 / 10.91 | 614 / 191 |
| validation sim BOTTOM extreme shadows | 94.68 / 139.13 | 7.27 / 14.70 | 1329 / 337 |
| test sim BOTTOM ordinary | 24.42 / 64.70 | 6.52 / 13.96 | 1236 / 525 |
| test sim BOTTOM extreme shadows | 91.14 / 135.60 | 8.99 / 17.69 | 2218 / 796 |

The constant simulation training-mean BOTTOM baseline has validation mean 7.03 px / p90 14.29 px, and test mean 7.09 px / p90 14.48 px. It is photometrically invariant. Under extreme shadows, the learned BOTTOM remains worse than that constant baseline. The narrow real validation constant TOP baseline is also competitive. These results do not establish precise real endpoint localization or real grasp generalization.

## Visibility and limits

Confidence remains uncalibrated. The runtime keeps `top_observed:false`, `top_visibility_verified:false`, `confidence_calibrated:false` and `automatic_nudge_eligible:false`. An accepted coordinate is an estimate, and the real distal endpoint has no accuracy ground truth.

- Test ordinary: accepts 687/714 visible TOPs and falsely accepts 1407/1614 hidden TOPs.
- Test extreme shadows: accepts 691/714 visible TOPs and falsely accepts 1421/1614 hidden TOPs.

Fresh real recordings with independent grasps and reviewed endpoint labels are still needed to validate deployment robustness. Fixed-X inference cannot represent lateral displacement away from the configured centerline. Synthetic accuracy is not physical calibration or contact validation.

## Artifacts and reproduction

- Run directory: `work/held-pin-cnn/varied-grasps-shadow-20261105/`.
- Dataset, generator audit and grasp/lighting review: `work/held-pin-cnn/varied-grasps-sim-20261105/`.
- `comparison.json`: frozen hashes, all predictions, strata and errors; `metrics.json`: training metadata and selected-model evaluations.
- `comparison-preview.jpg`: validation context/detail before-after review; `augmentation-preview.jpg`: lighting examples.
- `previous-best.pt` and `previous-*.json`: preserved earlier weights and reports.
- `run-training.sh`: exact training command. Generator: `scripts/generate_held_pin_axis_sim.py --generate --variants-per-anchor 12`.
- Evaluation: `scripts/evaluate_held_pin_robustness.py`; rendering: `scripts/render_held_pin_robustness.py`.

The running dashboard was using the previous weights during active force-guided motion and recording. The installed weights are used on the next perception-worker/dashboard restart; no live motion was interrupted to reload them.

Checks: 26 model, augmentation and integration tests passed; three-image generator smoke passed; 36 training crops across 12 groups and all lighting domains were visually reviewed. Training and evaluation issued no robot motor commands.

Installed checkpoint SHA256: `9d435b706f5100b8020304dd806a712e6428881912615686018189e94321c6a9`. Previous checkpoint SHA256: `3e44d773678483b6994cfa9b26d65c26549b11a116efa49aafb17adb26de2b76`.
