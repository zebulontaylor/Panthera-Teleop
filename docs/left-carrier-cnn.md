# LEFT-wrist carrier CNN and real adaptation

The initial experiments train simulated RGB at recorded approach/insertion poses. After real-image transfer failed, the user explicitly authorized OpenCV-assisted real-demo training. The later real-adaptation stage uses reviewed visible-bore labels and conservative OpenCV tracks, with simulated replay. No joint state enters either CNN and no hardware commands are issued. Real accuracy remains limited by the annotation/audit evidence below.

## Reproduce

Use `/home/robotics/Desktop/RobotArmTraining/.venv-cnns-cuda/bin/python` (PyTorch CUDA, MuJoCo 3.14.0, gymnasium, scipy, trimesh, manifold3d, Pillow, opencv):

```bash
python scripts/generate_left_carrier_data.py
python scripts/generate_left_carrier_data.py --generate-dataset --size 224
python scripts/train_left_carrier_cnn.py --epochs 100 --augment --resume work/left-carrier-cnn/development/measured_pose_mild_augmentation/best.pt
```

The first command creates a real LEFT/same-measured-q SIM LEFT contact sheet for visual review. Actual LEFT is physical wrist_port3. The generator uses shared `NativeQPhotoReconstruction`/`PhotoSetupBridge` native measured-state forward kinematics, corrected side-by-side photo scene and reproduced carrier visual mesh. The left camera's position, rotation and FOV remain constant and unchanged, verified after rendering. Native renders 320×240 preserve4:3 and are bilinearly resized 224×224, matching runtime full-frame preprocessing.

The held pin follows the actual nominal-reset left-tool grasp prior. Carrier follows the existing TRAIN-only right-tool/carrier PnP hypothesis. Complete carrier/gear geometry remains anchored to that carrier transform. This is explicit kinematic reconstruction, not physically settled contact or verified rollout. Pin position is particularly dependent on its grasp prior.

## Data and split

1,012 measured trajectory anchors produce 2,683 simulated frames. Original 34 approach recording groups use samples across their full saved trajectories, not just sparse annotated endpoints. Newdemo `20261104T193211-4db4b17f` uses actual LEFT frame timestamps with nearest measured state: B1/B3 train, B2 validation, B4/B5 test, excluding supplied guard intervals. Newdemo blocks share one recording and likely grasp; they are not independent sessions. Original recording groups are distinct, but multiple recordings may share a session/grasp.

The frozen supplied group splits yield 2,228 train / 115 validation / 340 test frames, across 22/5/12 disjoint recording-or-block groups. No augmented source anchor crosses a split. Each train anchor has four variants: three exact measured joint states (two with mild light/material variation), and one with joint jitter bounded 0.003 rad plus mild light/material changes. Validation and test are exact nominal anchors with no augmentation. `source-poses.json` records source frame, state, split, group and variation for every render. `split.npz` records group membership. Training stores uint8 images on CPU and transfers minibatches to CUDA.

The ten channels are face center, four cardinal insertion bores (pitch radius 17.9625 mm), four cardinal rim points (radius 26.49 mm), and pin tip. Face plane is 17.25 mm in carrier local coordinates. Cardinal bore offsets agree with original CAD assembled pin origins. A location contributes to spatial/coordinate loss only if in frame, front-facing and depth visible; pin's own rendered segmentation may validate its endpoint when its surface occludes its axis. Visibility BCE uses visible/hidden labels. Hidden coordinates are retained only for geometric evaluation, never spatial training loss.

The final run adds extensive stochastic augmentation on every training minibatch, following the earlier CNN mechanisms in `scripts/train_pin_insertion_cnn.py::augment` and `scripts/adapt_pin_insertion_cnn.py::affine_sample`. The generalized 10-landmark implementation is `app/carrier_face_augmentation.py`. It retains exposure/noise/blur/framing and extends white balance, color/saturation/grayscale, gamma/contrast, degradation/downsampling/quantization/sharpness, a Gaussian specular glare streak and rectangular/stripe occlusion. Small affine transforms update all landmark positions exactly; occluded and out-of-frame points are marked invisible for visibility BCE and excluded from location loss. Validation/test are untouched. Exact ranges/probabilities and preview provenance are saved in `augmentation.json`; `augmentation-preview.jpg` shows original+four variants per pose. Dedicated RNG seed 4401 makes training repeatable.

The final 100 epochs warm-start the 100-epoch mild-augmentation measured-pose baseline and reset validation selection. This creates 222,800 fresh stochastic training presentations across 4,700 augmented minibatches; it does not create additional independent poses or recordings. Best checkpoint selection uses validation loss only. Visibility threshold selection uses validation F1 only. The mild baseline had already evaluated the heldout poses before user-directed augmentation was added; its exact outputs are archived under `development/measured_pose_mild_augmentation/`. Those baseline test scores did not guide augmentation or tuning. Final test is evaluated after validation selection, with errors in network pixels and native 640×480 pixels. Explicit planar geometry fits sufficiently distributed confident visible landmarks and extrapolates hidden bores; unsupported fits withhold hidden locations. Hidden-bore error and support coverage are reported separately, restricted to true-hidden bores actually inferred at runtime (below predicted visibility threshold, so excluded from homography support). Six original test recordings held gears rather than bare pins; their carrier metrics remain but their artificial pin channel is excluded from location/visibility metrics using predeclared source notes. Overall synthetic loss is labeled separately. The final report also includes TRAIN constant-mean and shuffled-image bore baselines in native pixels; neither is used for selection.

## Artifacts and limits

`work/left-carrier-cnn/best.pt` contains actual trained weights and schema/provenance metadata, including dataset SHA256. `metrics.json`, `history.json`, `heldout_predictions.npz` and `heldout.jpg` capture actual training/evaluation. `samples.jpg` shows visible supervision. `measured-demo-preview/measured-demo-contact-sheet.jpg` compares real and rendered LEFT views before training. Green circles in heldout sheet are true visible labels; red crosses are confident predictions.

Earlier arbitrary random-pose/aim/FOV runs were rejected because they did not represent approach/insertion. Their weights, dataset and metrics exist only under `development/rejected_pose_domain/`; none are deployment results. The accepted measured-domain checkpoint uses unchanged optics and distinct pose data.

Known gaps: left optical model uncalibrated against a camera calibration target; approximate carrier geometry/materials; fixed held-pin prior and TRAIN-only carrier grasp hypothesis; no real-image label accuracy, uncertainty calibration, motion accuracy or control-success validation. Synthetic test errors establish performance only within this reconstruction domain. Real visual framing was compared before launch; real perception accuracy must still be evaluated.

## Frozen synthetic result (earlier)

The final augmented run completed 100 epochs on RTX 3070 in 205.5 seconds; validation selected epoch 80. Published checkpoint SHA256:

`46e85494b97eb535029a5712360ed1c21ed70d6908f555b50975e19d1ce6037a`

On the 340 synthetic heldout frames, mean visible insertion-bore error is **2.85 pixels at 640×480**, versus **53.60 pixels** for the TRAIN constant-coordinate mean and **61.40 pixels** with shuffled heldout images. This supports image dependence within the synthetic reconstruction domain. Overall visible-landmark error is 3.03 native pixels; visibility precision is 0.967 and recall 0.998 at the validation-selected threshold 0.35.

Homography support exists for 340/340 frames, but **none of the 23 truly hidden bores was classified hidden and inferred**. Therefore hidden-bore recovery accuracy is not demonstrated: the model's confidence can incorrectly call hidden bores observed. No post-test threshold tuning was performed. Geometric extrapolation code exists, but these results do not establish reliable occlusion handling.

Pin evaluation is especially limited: after excluding 146 frames from six real gear-holding groups, only five visible simulated pin labels remain (mean 1.45 native pixels). This is insufficient evidence of pin-tip accuracy for insertion. The fixed pin-grasp prior and synthetic appearance still require real validation. Final metrics and eligibility masks are in `metrics.json` and `heldout_metric_mask.npz`; `checkpoint.json` records all final training/augmentation provenance. The epoch-80 selected weights saw 178,240 augmented sample presentations; the executed 100-epoch run saw 222,800.

## Follow-up visibility audit

The original validation poses contain no hidden bores: all four bore channels are visible in all 115 frames. Thus validation F1 cannot calibrate true bore occlusion. Rim 2 has 24 hidden examples, rim 3 only 2, while the pin has no visible validation examples. These counts and confidence ranges are saved in `validation-visibility-audit.json`.

A validation-only controlled stress audit placed a 13×13 dark patch over each known bore. At the existing 0.35 threshold, every patched bore remained falsely visible. Thresholds very close to 1 can separate many of these artificial patches, but this does not establish calibration on real occlusion. `validation-occlusion-stress.json` records that evidence; no weights or operational thresholds were changed from it. Future calibration must include visible and hidden examples and reserve new final audit poses.

## Original-CNN reuse and local visibility variant

`app/carrier_face_legacy_cnn.py` defines the follow-up `legacy-adapted-left-local-visibility-v1` variant. It reuses the original sibling `outputs/panthera-assembly-cnns/pin_perception/carrier_v4/adapted.pt` shared stem, quarter/eighth/sixteenth encoders, decoders and learned context bias. The stem retains original RGB normalization, XY channels and a constant LEFT camera identity (index 1). Context uses only the original LEFT feature-column block. No overhead/right images or proprioception are accepted. Four old unordered bore filters initialize the canonical bore heads; the simulator labels then establish the experiment's canonical ordering. This does not resolve physical rotational ambiguity on an unmarked carrier.

Visibility uses local 3×3 feature samples at each predicted landmark plus context and landmark identity. Sampling coordinates are detached so visibility cannot move a location to an easier patch. Per-landmark class-balanced BCE separates visible/hidden examples. Training additionally targets known face points with synthetic occluders, masking every covered location while retaining invisible BCE labels. Existing heavy training augmentation remains active. Original visual encoder learning rate is 0.0001, reused decoder/context 0.0003, new heads 0.001. An initial uniform-LR attempt was stopped before evaluation and archived under `legacy/development_uniform_lr/`.

The legacy variant has a separate `legacy/` artifact directory and preserves the first small-CNN experiment. Checkpoint metadata selects its runtime model factory; absent variant metadata remains compatible with earlier checkpoints. Unit tests verify exact shared weight reuse, correct LEFT context slice, constant camera identity and absence of non-RGB inputs.

A fresh visibility audit reserves 80 exact B4/B5 recorded frames absent from prior rendering anchors, frozen before legacy inference. These share existing heldout pose blocks/grasp and are not independent sessions. Natural simulator labels include 16 hidden bores. Visibility calibration uses validation only, including clearly identified controlled occlusion stress; the new audit is evaluated only after thresholds are frozen. Controlled synthetic occlusion results must be reported separately from natural depth-visibility results and real-image validation.

## OpenCV-assisted real adaptation

The user authorized this stage after rejecting synthetic-only real-image performance. `scripts/prepare_left_carrier_opencv.py` produces 378 eligible actual LEFT frames with 648 verified visible-bore points, using B1/B3 reviewed seeds, short-window forward/backward-consistent optical flow and per-frame local rim/ellipse evidence. Missing points remain UNKNOWN; hidden coordinates are not generated from FK. Training source is `work/left-carrier-opencv/manifest.json`. The separate 15-frame AI-reviewed manual evaluation manifest retains B2 validation and B4/B5 test. For this new real-training stage B2 is repurposed explicitly for checkpoint selection; B4/B5 never train or select weights. All are from one recording/grasp, not independent-session evidence. Native label uncertainty is approximately 5–6 pixels and is not human-certified.

```bash
python scripts/adapt_left_carrier_cnn.py --epochs 100
```

The model starts from the original-CNN-reuse simulated checkpoint. Each minibatch contains actual real images plus randomly sampled simulated replay. Unordered actually-visible real bore points are assigned to four predicted bore slots with Hungarian matching. Unassigned bores, face center, rims and pin have no real location loss or negative visibility label. Only labeled visible points enter real location loss. Synthetic training occluders on previously observed points create explicit invisible BCE labels and exclude location loss; source-unlabeled real points stay UNKNOWN. Unit tests verify zero location/visibility gradient on unknown points.

Training retains extensive stochastic augmentation, with real/sim loss weights 3/0.5 and encoder/decoder/head learning rates 0.00001/0.00005/0.0003. Actual image hashes are checked before training. The 100-epoch schedule is fixed; checkpoints are selected by mean native-pixel B2 visible-bore error, then B4/B5 are evaluated once. Artifacts are under `work/left-carrier-cnn/real-adapted/`. Runtime uses a fixed 0.5 binary decision floor, explicitly uncalibrated for real occlusion. Model choice must compare real error, visibility coverage and qualitative behavior against the independent OpenCV runtime baseline; the CNN is not assumed superior.

### Frozen real-adaptation outcome

The actual 100-epoch real-adaptation run completed in 98.4 seconds; B2 selected epoch 11. It processed 37,800 augmented real presentations plus 37,800 simulated replay presentations. Checkpoint SHA256:

`a04aab1b007ed6a8016d325084c36b4e2ba9b7f7f42d0270cc412f4ade7bc442`

On B2, mean unordered visible-bore error improved from 8.25 to 3.42 native pixels across 21 annotated bores in seven frames; matched visibility coverage improved from 13/21 to 19/21. On the eight B4/B5 frames, mean raw matched-bore error is **9.08 pixels**, median 8.16 and p90 13.47, with 19/23 matched annotations classified visible. These raw-slot assignment metrics are not a false-positive-aware detector benchmark; compare against OpenCV with the same visible/coverage matching protocol before selecting a backend. The small, correlated AI-annotated set does not certify insertion accuracy.

The current-photo qualitative preview improves the north-bore prediction, but its inferred right hole lies outside the carrier despite the homography's support flag. This demonstrates that layout self-consistency alone does not validate a real face fit. **The real-adapted CNN is not approved as the runtime default on this evidence.** No test-driven retraining followed. The direct OpenCV runtime is evaluated as an alternative, as the user requested. See `real-adapted/metrics.json`, `real-adapted/real-evaluation.jpg` and `real-adapted/attachment-preview.jpg`.
