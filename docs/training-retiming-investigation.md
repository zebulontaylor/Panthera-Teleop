# Training trajectory retiming investigation

The current data contains fast motion segments. A useful next experiment is to slow only fast segments during a new offline dataset export, with a shared clock for both arms, grippers, measured state, future action targets, and cameras. Preserve the originals and current training run. This investigation does not establish that demonstration speed causes training loss or rollout failures, and it does not establish safe robot limits.

Aim for approximately constant speed during free-motion cruise, low acceleration there, and smooth acceleration/deceleration ramps at transitions. Constant nonzero acceleration and constant velocity cannot hold together. Initially preserve slow contact segments and pauses; do not speed them up to force uniform motion. Treat grippers separately when choosing thresholds, while keeping their timing synchronized with the arms.

## Current pipeline and data

The current curated dataset is `data/processed/panthera-oct22-onward-curated-20hz`: **227 episodes, 73,074 rows, 20 Hz**. It includes alignment sidecars, source metadata, and trimmed full-rate archives. Recording is configured for nominal 150 Hz state and 20 Hz cameras in `config/recording.json`. The paired restyled dataset has the same episode/row counts but lacks its own alignment and full-rate directories; preserve and verify its source lineage when creating paired timing variants.

`scripts/postprocess_kept.py` builds a uniform source-time grid. Each output row contains the nearest measured pose at time `t`, an absolute 14D action equal to the measured pose at `t + 50 ms`, and the nearest image from each of three cameras at `t`. Actions include native gripper radians and are measured-next-pose imitation proxies, not demonstrated actuator commands. Current state/camera alignment tolerances are 10/40 ms for the curated data. Slowing footage will repeat source images; using full-rate source states avoids unnecessarily stretching the existing 20 Hz pose staircase.

Retiming fits before this exporter's uniform sampling: define a monotonic source-time map `s(tau)` and export state/images at `s(tau)`, action at `s(tau + 50 ms)`, while retaining the 20 Hz output contract. Changing FPS or timestamps alone is insufficient. HITL's `data_monotonic_ns` already distinguishes training time from physical host time; preserve control-ownership and recording-segment boundaries and do not interpolate across excluded holds.

## Current measurements

`scripts/analyze_training_timing.py` and `docs/training-timing-analysis.json` provide the numeric audit. Joint statistics below use the maximum absolute value across the twelve joints per frame. These are discrete 20 Hz trajectory estimates, and acceleration in particular is noisy; they are **not measurements of physical robot acceleration**.

| Quantity | p50 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Joint velocity, rad/s | 0.16336 | 0.95504 | 1.33204 | 2.84000 |
| Joint interior finite-difference acceleration, rad/s² | — | 7.037 | 11.812 | 55.543 |
| Gripper velocity, rad/s | — | 2.56354 | — | 7.46442 |
| Gripper interior finite-difference acceleration, rad/s² | — | 21.865 | — | 135.465 |

The following are illustrative timing comparisons, not recommended hardware limits:

| Joint velocity / gripper velocity, rad/s | Velocity-only local stretch lower bound on total duration |
| --- | ---: |
| 0.6 / 2 | 1.10900× |
| 0.8 / 2 | 1.06735× |
| 1.0 / 3 | 1.01910× |

These local bounds account for velocity alone. They omit acceleration-limited ramps, so a feasible smooth local solution can take longer.

| Joint velocity / acceleration; gripper velocity / acceleration | Whole-episode uniform scale reference |
| --- | ---: |
| 0.6 / 2; 2 / 8 | 3.46055× |
| 0.8 / 3; 2 / 8 | 3.24643× |
| 1.0 / 4; 3 / 12 | 2.66444× |

Velocity units are rad/s and acceleration units rad/s². The second table conservatively slows each entire episode enough to satisfy the discrete estimates under uniform scaling. It is a feasible reference for those estimates, not a prediction of an optimized local method. No local acceleration-constrained solver has been implemented or evaluated yet.

Reproduce to a fresh report path:

```bash
work/postprocess-venv/bin/python scripts/analyze_training_timing.py data/processed/panthera-oct22-onward-curated-20hz --output /tmp/panthera-training-timing-analysis.json
```

## Implementation and experiment plan

1. Add a separate derived export mode with an explicit timing-map sidecar. Preserve source hashes/timestamps, output timestamps, interpolation provenance, original episode identity, and all control-segment boundaries. Start from full-rate real source data and derive paired real/restyled samples using the same map.
2. Estimate a usable geometric path and select smooth timing. A piecewise-linear pose path has corners with infinite continuous acceleration when traversed without stopping; satisfying finite differences alone does not fix those corners. Consider stops, suitable path interpolation, or carefully justified smoothing. Do not blindly smooth through contact, insertion, or gripper events. [TOPPRA's primary quickstart](https://hungpham2511.github.io/toppra/quickstart.html) illustrates fixed-path velocity/acceleration-constrained parameterization and is a candidate to evaluate, not an existing integration.
3. Resample all modalities through the same map. Scale or recompute velocity consistently; measured effort remains source effort and does not become a physical prediction for slower contact. Check camera/source offsets in source time and preserve camera identities. Rebuild future action chunks, dataset statistics, manifests, indexed precision weights, and real/restyled lineage.
4. Extend verification beyond the current exact-source-row assertions in `scripts/verify_processed.py`: verify monotonic maps, shared timing, no boundary crossing, interpolation reproduction, finite values, velocity/acceleration estimates, repeated-image rates, duration inflation, unchanged endpoint/path tolerances, and native loader action-chunk padding.
5. Run a paired original-versus-retimed ablation grouped by original source episode, so crops and restyled versions cannot leak across splits. Hold sample/update budgets and real/restyled sampling policy equal; stretching otherwise increases the sampling weight of those motion regions. Evaluate both on a fixed original-timing holdout, report phase-specific results and training loss, and require controlled robot task-success/tracking evaluation before attributing practical benefit to retiming.

Keep the first experiment narrow: slow selected fast free-motion regions, retain contact/pause timing, report added duration and repeated images, and compare against both the unmodified data and a simple uniform-slowing baseline.
