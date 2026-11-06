# Final pin alignment CNN — November 4, 2026

**Superseded simulator:** this initial experiment used legacy opposed-base XML,
not the existing photo setup. Its synthetic results do not validate the real
bench geometry. See [the scene correction](pin-insertion-photo-setup.md).

Follow-up: [real adaptation, calibration and contact-stage audit](pin-insertion-sim2real.md).
The results below preserve the initial sim-only experiment.

This experiment trains a small CNN on MuJoCo CAD renders and evaluates it on
saved real demonstrations. The **left arm moves the pin**, the right arm holds
the carrier, and the input is the right wrist camera. It is an offline prototype,
not a validated automatic insertion controller. **Real-demo validation failed;
automatic nudging remains unfinished and is not enabled.**

## Measured results

The selected checkpoint is `work/pin-insertion-cnn/heatmap.pt`. Training used
3,072 CAD-rendered examples from six measured demonstration pose streams:
2,560 training examples and 512 validation examples from a separate source
episode. There are 966 active labels; the remainder cover alignment, distance,
hovering, and occlusion. The heatmap CNN ran for 100 GPU epochs; epoch 65 was
selected using simulated visible-landmark validation error. Real data was not
used for training or checkpoint selection.

| Simulated validation, 256×192 | Pin-tip mean error | Hole mean error |
| --- | ---: | ---: |
| Direct regression, 40 epochs | 24.19 px | 19.36 px |
| Direct regression, stronger landmark loss, 100 epochs | 21.32 px | 17.47 px |
| Spatial heatmap, 100 epochs | **5.34 px** | **10.25 px** |
| Heatmap with shuffled images | 37.91 px | 34.25 px |

The heatmap medians are 3.37 px for the pin and 4.79 px for the hole. Image
shuffling substantially degrades performance, which supports learned visual
localization in simulation. It does not establish real-image transfer.

At gate score 0.8, synthetic precision is 83.6% and recall is 80.9%, with
25 false activations among 355 inactive validation examples. All 128 synthetic
hover examples were inactive. These are single-frame metrics, not metrics of
the adapter's persistence rule. The 3D delta head is not useful enough:
per-axis mean absolute error is 1.112 mm versus 1.117 mm for predicting zero,
and mean direction cosine is only 0.095.

Real evaluation covers all 162 saved frames across six separate forward
recordings, plus existing independent annotations from four older validation
episodes. The two annotated pin tips have mean error **91.86 px** and the single
annotated bore has error **104.30 px**, at the original 640×480 resolution.
This is far too little annotation to estimate general accuracy, but these
errors and the inspected overlays reject deployment. The CNN sometimes tracks
gripper features as the pin; the gate exceeds 0.8 in visibly separated approach
frames. Real XYZ and gate accuracy remain unmeasured.

All 3,072 measured arm states are within simulated arm limits. Maximum FK
disagreement against saved app poses is 3.58e-16 m and 5.17e-8 rad. No arm
angles were clipped. **32 tests and 11 additional subtests passed.**

Compact results are saved in [pin-insertion-cnn-results.json](pin-insertion-cnn-results.json).
Full predictions and overlays are in
`work/pin-insertion-cnn/heatmap-real-evaluation.json` and
`work/pin-insertion-cnn/heatmap-real-evaluation.jpg`; synthetic metrics and
image ablations are in `work/pin-insertion-cnn/heatmap-sim-evaluation.json`.
The two regression runs are preserved under `baseline/` and `regression/`.

## Model and correction contract

`app/pin_insertion_cnn.py` accepts a full RGB camera image and native measured
state14: left joints 1–6, left gripper, right joints 1–6, right gripper, in radians.
Both arms' joint sine/cosine features condition the CNN. The selected variant
uses spatial landmark heatmaps at stride four, Gaussian supervision, and a
softargmax coordinate readout. A direct-regression variant remains available
for comparison. Three heads predict:

- Pin contact-tip and nearest carrier-bore image coordinates, normalized to the
  full image. The contact tip is used instead of the opposite end of the pin.
- A translation of the **left-held pin**, in metres in the **right native base
  frame**, toward the hole. This is not a right-arm joint command.
- An uncalibrated sigmoid score for whether a correction is needed.

Training uses episode-disjoint validation, masked landmark/translation losses,
class-balanced gate loss, brightness/color/noise/blur augmentation, and small
occlusions that avoid labeled landmarks. Render randomization changes field of
view, assembly orientation, material colors, and lighting. Images retain a 4:3
render aspect before the same model resize used on real images. Inference
rejects a square crop passed in place of a full camera frame.

`BoundedPinCorrection` returns advisory proposals only. Its default shadow mode
returns zero executable translation. It requires three fresh consecutive frames,
entry/exit score thresholds of 0.8/0.6, timestamps from one monotonic clock,
at most 150 ms age and 20 ms image/state skew, finite complete state14, and limits
the proposed translation norm to 0.5 mm. It has no motor or camera access.
Setting `enabled=True` only exposes the bounded vector; it does not establish
calibration or send a command.

To execute later, transform the vector with a measured rotation
`delta_left = R_left_from_right @ delta_right`. Do not add the base translation
to a displacement. The existing `parallel_bases` displacement assumption is not
a measured calibration. A motion supervisor must additionally check the left
arm Jacobian/IK, joint limits, contact load, carrier stability, cumulative travel,
and whether the observed image error decreases after each step.

## What the simulation establishes

`scripts/pin_insertion_sim.py` replays measured joint states without inventing
joint perturbations. It compares each selected state's MuJoCo TCP against the
saved app FK for both arms and fails on disagreement greater than 10 micrometres
or 0.0001 rad. This establishes model/frame consistency, not independent physical
robot calibration. Gripper encoder-to-opening conversion remains the existing
approximation in `Real2SimBridge`.

The large carrier and four seated gears use the sibling simulation's CAD and
assembled origins. The bore-center pitch radius is 17.9625 mm (individual bore
diameter approximately 8.34 mm), and the mouth plane is 17.25 mm above
the carrier origin; pin tip is at local z=0. Target selection always uses the
nearest of the four bores, avoiding an invisible arbitrary target identity.
The table is lowered 50 mm to the previously tested base-height hypothesis.

The synthetic gate requires supported landmarks, tip height within
1 mm of the carrier plane, and a 0.2–12 mm lateral error. Near-hole offsets are
sampled logarithmically; pin tilt is randomized from 0–15 degrees. Translation
alone does not correct pin tilt, and the 0.2 mm deadband is not an insertion
success criterion. Aligned, distant,
hovering 8–35 mm above the carrier, and occluded samples are negatives. Depth
visibility allows at most 1 mm of external foreground tolerance. Exact render
segmentation supports the amodal pin-tip center behind its own surface. For
gating, expected pin-over-hole overlap is allowed when the carrier remains
visible; the hole position loss still requires direct visibility. These are geometric
labels: touching a particular surface, friction, slip, and insertion success
are not established by them.

**Remaining registration gaps:** real wrist intrinsics/distortion/extrinsics,
relative robot base rotation, and object-to-jaw transforms are unverified.
Parts are placed in plausible randomized wrist-camera views rather than fitted
to the actual jaws. Arm joint states are accurate to the saved model; the
complete physical assembly scene is not calibrated. No dynamics are stepped
and no physical insertion success is claimed.

## Real evaluation

`scripts/evaluate_pin_insertion_cnn.py` reads recordings only. It enforces the
saved right-wrist mapping (`wrist_port2`) and rejects overlap with any source,
training, or validation episode named in the checkpoint. It samples across each
episode and writes a prediction JSON and review montage.

The dedicated forward recordings have `label=unverified`; `forward_complete`
does not mean insertion succeeded. Their image/sample timestamp difference is
a receipt-time proxy, not a verified measured-state age. The existing independent
AI visual annotations provide only two pin-tip and one bore position labels in
the right-wrist validation set. They are not human-certified. Real nudge-vector
accuracy and activation precision/recall cannot be measured from these labels.

## Reproduce

Create a Python 3.12 environment and install `requirements-pin-insertion.txt`
(use a compatible CUDA PyTorch build for GPU training). The existing local run
uses `work/pin-insertion-cnn/runtime/bin/python`, which reuses the installed
PyTorch packages from the sibling checkout. The sibling scene and its meshes
must be available; `--scene` can override the default.

```bash
MUJOCO_GL=egl python scripts/pin_insertion_sim.py \
  --episode data/kept/20261022T222502-6321638f \
  --episode data/kept/20261022T223034-fa96590a \
  --episode data/kept/20261022T224407-e6499944 \
  --episode data/kept/20261023T180718-47e74c70 \
  --episode data/kept/20261023T181449-320a66bd \
  --episode data/kept/20261023T182037-65ab8000 \
  --samples-per-episode 512 --image-size 192 --seed 43 \
  --output work/pin-insertion-cnn/sim_dataset_v6.npz

python scripts/train_pin_insertion_cnn.py \
  --dataset work/pin-insertion-cnn/sim_dataset_v6.npz \
  --output work/pin-insertion-cnn/heatmap.pt \
  --epochs 100 --image-size 160 --batch-size 64 --device cuda --seed 42 \
  --model-variant heatmap --heatmap-weight 10 \
  --keypoint-weight 1000 --delta-weight 1 --select-by keypoint

python scripts/evaluate_pin_insertion_sim.py \
  --checkpoint work/pin-insertion-cnn/heatmap.pt \
  --dataset work/pin-insertion-cnn/sim_dataset_v6.npz \
  --output work/pin-insertion-cnn/heatmap-sim-evaluation.json --device cuda

python scripts/evaluate_pin_insertion_cnn.py \
  --checkpoint work/pin-insertion-cnn/heatmap.pt \
  --episodes 001e02d4fa414832b033b544edf8a349 \
    285fdd7db4ba418e8a8f2594c52128d8 6804bd9642a84a12802af28a9de5817c \
    97bd7a271371430daa0cb7e4ba213a41 c2ba77d46efa4577827b7434a2f75a34 \
    eb22572759ff4ab785ffeb4c112231ec \
  --manual-labels data/calibration/manual-labels/assembled/training.jsonl \
  --output work/pin-insertion-cnn/heatmap-real-evaluation.json --stride 1

python -m pytest tests/test_pin_insertion_cnn.py tests/test_pin_insertion_sim.py \
  tests/test_evaluate_pin_insertion_cnn.py tests/test_real2sim_bridge.py \
  tests/test_real2sim_data.py -q
```

Large generated datasets, checkpoints and images live under ignored `work/`.
The checkpoint and reports preserve dataset hashes, source IDs, explicit
coordinate semantics, random seed, split, and simulation assumptions.

## What is needed before automatic correction

Fit and independently check the right-wrist intrinsics/distortion and mount,
the base-to-base rotation, and both held-object offsets. The synthetic carrier
and pin should then project onto the real parts across separate recorded
frames, with their held poses coupled to the measured joints. Training on
accurate joint values alone cannot establish this scene registration.

Add independently reviewed real pin/hole positions and on-carrier/hover/empty/
inserted labels, split by complete recording session. Measure keypoint error,
false activations, and correct nudge directions on those held-out labels.
Only after that should a supervised bounded motion trial test contact response
and whether the pin actually enters the bore. A high gate score alone is not
evidence of contact or successful insertion.
