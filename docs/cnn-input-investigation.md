# CNN input routing investigation

The poor overlays were caused primarily by reversed wrist-camera inputs in the
collector integration. The same checkpoint weights and images produce much
better coordinates when current inputs are ordered `overhead, wrist_port3,
wrist_port2`. The earlier conclusion that the model needed retraining based on
those overlays was premature.

The historical adaptation data called the gear/pin camera `wrist_port2` and
the carrier camera `wrist_port3`. The current 17 trials have the opposite
physical camera contents. Copying the historical USB names into the current
input order therefore reversed the learned views. Original native development
frames were recovered and visually compared, and the same installed gear
checkpoint still localizes the gear on those original frames.

## Controlled comparison

Both conditions use identical original JPEGs, measured left/right tool XYZ,
RGB decoding, 640×480 dimensions, normalization, coordinate decoder, and model
weights. Predictions are mapped back to the actual recorded camera before
comparison with the manually reviewed visible points. Only wrist-camera order
changes. Tool XYZ ordering does not change.

| Visible point | Count | Previous order median / p90 error | Corrected order median / p90 error |
| --- | ---: | ---: | ---: |
| Gear axial center | 28 | 6.49 / 89.78 px | 3.73 / 6.85 px |
| Pin insertion endpoint | 12 | 125.73 / 182.60 px | 6.53 / 14.25 px |
| Bore, nearest unordered candidate | 5 | 59.17 / 77.77 px | 2.82 / 4.31 px |

These 45 eligible coordinates occur in 27 observations from 16 episodes. The
manual review set covers all 17 episodes; one has no eligible visible point.
Labels are AI visual annotations with pixel uncertainty, not human-certified
truth. The bore score is an optimistic nearest-candidate comparison and does
not establish the same physical target's identity across camera views.

The four held-out episodes also improve: gear median 7.55 to 4.77 pixels,
and pin-task points median 108.54 to 5.75 pixels. These subsets are small
(eight gear points and three pin-task points), so they do not establish full
hardware insertion accuracy.

## Corrections

The recording configuration, live inference default, and offline inference
default now route the current physical left/gear-pin camera (`wrist_port3`)
into the model's `left_wrist` slot, and the physical right/carrier camera
(`wrist_port2`) into `right_wrist`. Left XYZ followed by right XYZ is unchanged.
Camera cables or capture mappings must be checked again if their identities
change. The checkpoint metadata's historical port names are not stable camera
identities.

The initial two-view visibility rejection was also an inappropriate test of
real localization. Real adaptation trained image positions without real
visibility loss. One pin-tip head saw zero synthetic visible examples, and the
gear entrance is an amodal reference. A near-zero visibility output does not
prove a bad predicted coordinate. Correcting that interpretation does not
provide an independently validated real grasp detector or motion calibration.

Wrong-order calibration reports are retained for provenance and superseded.
The corrected-order phase inference and recalibration are saved separately as
`logs/alignment-calibration/predictions.phases10hz.corrected-order.jsonl` and
`logs/alignment-calibration/calibration-report.corrected-order.json`.
The corrected gear mapping now passes the existing offline statistical checks:
held-out image-motion RMSE is 17.42 pixels versus a 24.68-pixel no-motion baseline,
with 89.2% directional accuracy. Pin improves but its bootstrap consistency is
89.06%, below the unchanged 90% acceptance threshold. Neither is verified for
physical motion while real grasp evidence and pin-bore association are unresolved.

## Artifacts

- `data/calibration/manual-labels/assembled/labels.json`: 68 observations,
  136 wrist views, with source hashes and annotations from three subagents.
- `data/calibration/manual-labels/primary-review.json`: independent primary-agent
  spot checks of nine observations; three ambiguous endpoint labels removed.
- `data/calibration/manual-labels/assembled/training.jsonl`: explicit supervision
  masks. Hidden points have no coordinates; ambiguous points are not visibility
  negatives. Splits keep 13 training and four validation episodes separate.
- `logs/alignment-calibration/investigation-inputs.summary.json`, `.errors.csv`,
  `.predictions.jsonl`: complete paired experiment and hashes.
- `logs/alignment-calibration/investigation-inputs-gear-orders.jpg` and
  `investigation-inputs-pin-orders.jpg`: old inputs on top, corrected inputs
  below; model points in red and manual visible points in green.
- `logs/alignment-calibration/investigation-preprocessing.md`: RGB, scaling,
  normalization, stride/offset decoder and checkpoint hash audit.
- `logs/alignment-calibration/investigation-domain.md`: historical reference
  image recovery, camera-content comparison, and original-domain reproduction.

Reproduce the paired comparison with
`../RobotArmTraining/.venv-cnns/bin/python scripts/investigate_cnn_inputs.py`.
No robot motion or model retraining was used for this diagnosis.

## CUDA runtime

The NVIDIA RTX 3070 is available. The original CNN environment contained CPU-only
PyTorch, so a separate runtime was installed at
`/home/robotics/Desktop/RobotArmTraining/.venv-cnns-cuda/bin/python` with
PyTorch `2.14.1+cu130`, numpy `2.5.4`, and Pillow `13.0.0`. The recording
configuration uses this Python with `device: "auto"`. Offline inference and the
paired-input investigation also accept `--device` and default to `auto`; run
them with this GPU Python to select CUDA automatically.

`logs/alignment-calibration/cuda-verification.json` records unchanged checkpoint
weights, real-frame inputs, actual `cuda:0` model devices, and CPU/GPU coordinate
agreement within 0.0025 pixels. Mean inference was 4.37 ms for gear and 4.49 ms
for pin, versus 87.10 and 90.94 ms on CPU in the same ten-repeat test. The live
worker subprocess also returned both predictions on CUDA; warm end-to-end
worker time was 17.42 ms. Initial model loading takes longer.

A cloned gear network completed CUDA forward/backward with 44 finite gradient
tensors and nonzero gradients. No optimizer update or checkpoint save occurred.
The environment dependency snapshot is `cuda-runtime-freeze.txt` beside the
verification report. Reproduce the hardware checks with:

```bash
../RobotArmTraining/.venv-cnns-cuda/bin/python scripts/verify_alignment_cuda.py
```
