# Separate retiming export against the October 28 reference

The new October 28 demonstrations provide the timing reference. The experiment stretches fast parts of the previous curated dataset toward that reference, preserves slow movements and pauses in the source clock, and writes separate LeRobot datasets. Training settings and active training data remain unchanged. Nothing is published.

## Reference cohort and comparison

`config/postprocessing-oct28-retiming-reference.json` selects **28 complete operator-kept recordings** from October 28, 2026, 12:41–13:12 America/Chicago. The UTC `20261028T001512` recording belongs to October 27 locally and is excluded. The annotations cut terminal pedal-classification tails at the earliest terminal-opening cue or classification-state timestamp: **35.483 seconds** from cue to metadata end, or **34.996 seconds** using the export manifest's logged-state endpoint accounting. They make no earlier motion or pause cuts and do not certify task success. The completed reference export has **11,171 next-pose rows / 558.55 seconds** at 20 Hz. Its actual numeric audit, `docs/retiming-new-reference-export-timing.json`, exactly matches the preparation baseline's episode count, frame count, velocity, and acceleration summaries.

The previous curated set contains **227 episodes / 73,074 rows / 3,653.70 seconds**. The table compares pooled frame maxima: maximum absolute joint value across twelve joints, or across two grippers, followed by the 95th percentile across frames. Differences never cross episode boundaries.

| Quantity, frame-max p95 | Previous original | New reference | Actual saved retimed export |
| --- | ---: | ---: | ---: |
| Joint velocity, rad/s | 0.9550 | 0.7665 | 0.8000 |
| Joint acceleration, rad/s² | 7.0372 | 5.7805 | 4.9823 |
| Gripper velocity, rad/s | 2.5635 | 2.4127 | 2.4466 |
| Gripper acceleration, rad/s² | 21.8655 | 18.8496 | 17.2462 |
| Total transition duration, s | 3,653.70 | 558.55 | 3,813.65 |

The saved retimed export contains **227 episodes / 76,273 rows**, adding **159.95 seconds / 4.38%** to the previous dataset. The new reference remains at its original timing. Its different episode and task-phase composition means that comparable pooled percentiles do not establish equivalent behavior or training benefit.

The table uses actual saved-export measurements from `docs/retiming-old-export-actual-timing.json`; they agree with the selected candidate predictions in `docs/retiming-candidate-comparison.json` within 0.0001. Both exports are complete and independently verified. The consolidated measured comparison is `docs/retiming-export-comparison.json`. The retimed dataset passed checks on all **76,273 rows / 228,819 camera images**, plus **681 native 50-step action chunks** with exact boundary padding. The new reference passed checks on all **33,513 camera images** and **84 native 50-step action chunks**. The scoped test suite has **24 passing tests**.

## Selected transform

`config/retiming-oct28-reference.json` selects candidate index **2** from six fixed exploratory candidates:

- Joint velocity shaping target: **0.8 rad/s**; acceleration shaping target: **20 rad/s²**.
- Gripper velocity shaping target: **3 rad/s**; acceleration shaping target: **40 rad/s²**.
- Preserve intervals with both joint speed at most **0.15 rad/s** and gripper speed at most **0.3 rad/s**.
- Also preserve annotated precision-phase windows, including gear/pin insertion windows, regardless of speed.
- Run at most **40 iterations**, with an iterative interval stretch bound of **4×**; the final output-grid rounding can add less than one 50 ms tick to one editable interval per episode.

The caps tune the resulting distribution; they are not the desired pooled p95 values or robot safety limits. Directly using the new reference's rounded p95 values as caps (joints 0.8/6, grippers 2.5/19) reduced acceleration too much: predicted joint/gripper acceleration p95 became 4.1258/12.7462, with **17.01%** added duration. The selected 20/40 acceleration targets give closer resulting percentiles with less stretching. Selection minimizes the mean absolute log ratio of four resulting p95 values to the new reference, plus a small duration penalty. No training loss or robot rollout informed the choice.

The transform uses the existing exported **20 Hz pose knots** plus the terminal next-pose action. It slows traversal of their piecewise-linear path, adds no spatial smoothing, and cannot recover source motion between the original samples. One monotonic map drives both arms and grippers. Cameras use one shared nearest original observation row for all three views; original encoded JPEG bytes are retained. Resampling can repeat frames. Camera selection adds at most half a source tick to the original alignment error ordinarily, or one full tick where the terminal image is clamped to the final source observation. Every action is recomputed as the pose at the following output tick.

Protected intervals keep their exact duration in the map; output sampling can shift their phase by less than one frame. Across the previous dataset, **1,409.85 source seconds** are protected. Only **8,863 of 73,074 source intervals (12.13%)** receive stretching; the other intervals keep their original duration. Existing precision weights are rebuilt using the mapped action time. Low speed conservatively protects likely contact and pauses, but unannotated contact is not semantically identified; fast contact outside saved precision windows can still be stretched.

The selected candidate reports residual threshold exceedances rather than claiming all constraints hold:

| Finite-difference quantity | Protected stencils | Editable or seam stencils |
| --- | ---: | ---: |
| Joint velocity | 31 | 2 |
| Joint acceleration | 0 | 1 |
| Gripper velocity | 234 | 7 |
| Gripper acceleration | 75 | 15 |

There are **33 intervals at the stretch cap**. Counts are stencil exceedances and can overlap in time or across quantities. Protected motion is deliberately retained even if it exceeds a shaping target; seams and the bounded iteration/stretch budget can leave other residuals. This is a comparative offline transform, not a hard-limit trajectory planner. Finite-difference acceleration includes quantization and resampling effects; piecewise-linear corners do not establish continuous acceleration bounds. Interpolated measured effort is historical context, not a prediction of slower physical dynamics.

## Outputs and provenance

Completed separate datasets:

- `data/processed/panthera-oct28-reference-20hz`: completed new-method export at its original timing; actual numeric baseline verified.
- `data/processed/panthera-oct22-onward-retimed-oct28-reference-20hz`: retimed previous curated dataset.

The retimed export records its configuration, source hashes, source/export timing maps, interpolation weights, original episode IDs, protected intervals, per-episode metrics, and row mappings in `processing/retiming/`, with aggregate provenance in `processing/manifest.json`. Numeric statistics are recomputed from saved values in float64, including exact per-episode/global quantiles; native image statistics are preserved. This avoids float32 variance cancellation observed in the native writer. Precision weights are rebuilt from mapped source action time. Original cohort summaries describe source rows and are retained under the retiming source contract rather than presented as current export counts. Original datasets and raw recordings remain available.

Verification independently checks interpolation, monotonic shared timing, exact protected-interval durations, unchanged encoded camera bytes, episode/frame counts, numeric metadata, and native LeRobot loading with 50-step future-action chunks including terminal padding. Compare the actual exported timing audit and verification report before selecting data for training.

## Reproduce

Run from the repository root with `work/postprocess-venv/bin/python`. Commands refuse existing output paths; select fresh paths when repeating an experiment.

Prepare the numeric reference and annotations without exporting images:

```bash
work/postprocess-venv/bin/python scripts/prepare_retiming_reference.py \
  --annotations /tmp/oct28-reference-annotations.json \
  --report /tmp/oct28-reference-baseline.json
```

The preparation uses the same nearest measured-state, common-camera availability, 20 Hz grid, and next-pose semantics as the original exporter. Image crops follow the current curated manifest: overhead `[120,80,520,480]`, wrists `[80,0,560,480]`. All retained image files are checked for existence; state/camera alignment is checked against 10/40 ms tolerances.

Export the new reference using the saved annotations:

```bash
work/postprocess-venv/bin/python scripts/postprocess_kept.py \
  --source data/kept \
  --annotations config/postprocessing-oct28-retiming-reference.json \
  --output data/processed/panthera-oct28-reference-20hz \
  --trimmed-output data/processed/panthera-oct28-reference-full-rate \
  --repo-id local/panthera-oct28-reference-20hz
```

Repeat the numeric candidate comparison to fresh report/config paths:

```bash
work/postprocess-venv/bin/python scripts/compare_retiming_candidates.py \
  --source data/processed/panthera-oct22-onward-curated-20hz \
  --reference docs/retiming-new-cohort-baseline.json \
  --output /tmp/retiming-candidate-comparison.json \
  --selected-config /tmp/retiming-selected-config.json
```

Export the selected retiming and independently verify it:

```bash
work/postprocess-venv/bin/python scripts/export_retimed_dataset.py \
  --source data/processed/panthera-oct22-onward-curated-20hz \
  --output data/processed/panthera-oct22-onward-retimed-oct28-reference-20hz \
  --repo-id local/panthera-oct22-onward-retimed-oct28-reference-20hz \
  --config config/retiming-oct28-reference.json

work/postprocess-venv/bin/python scripts/verify_retimed_dataset.py \
  --source data/processed/panthera-oct22-onward-curated-20hz \
  --dataset data/processed/panthera-oct22-onward-retimed-oct28-reference-20hz

work/postprocess-venv/bin/python scripts/analyze_training_timing.py \
  data/processed/panthera-oct22-onward-retimed-oct28-reference-20hz \
  --output /tmp/retiming-export-actual-timing.json
```

The raw new-cohort comparison is in `docs/retiming-new-cohort-baseline.json`; candidate scores, settings, percentiles, duration, and residual counts are in `docs/retiming-candidate-comparison.json`. Future training comparisons should group all derived versions by original episode to prevent split leakage and control sample/update budgets because stretching changes frame counts and sampling weights.

The final comparison can be regenerated against these verified datasets to a fresh path:

```bash
work/postprocess-venv/bin/python scripts/summarize_retiming_exports.py \
  --previous data/processed/panthera-oct22-onward-curated-20hz \
  --reference data/processed/panthera-oct28-reference-20hz \
  --retimed data/processed/panthera-oct22-onward-retimed-oct28-reference-20hz \
  --output /tmp/retiming-export-comparison.json
```

The retimed exporter refreshes numeric statistics automatically. For a fresh unretimed reference created with the original exporter, use `refresh_numeric_statistics(Path(reference_root))` from `scripts/export_retimed_dataset.py` before using its normalization statistics; the delivered reference already has this refresh and an independent 81-statistic verification in `processing/numeric-statistics-independent-verification.json`.
