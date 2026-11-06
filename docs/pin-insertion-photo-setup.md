# Photo setup correction and contact recording

The previous synthetic pipeline loaded `scene_bimanual.xml` directly. That is
the legacy opposed-base scene: its table, camera mounting, and arm layout did
not represent the recorded bench. The replacement uses the existing
`RobotArmTraining/sim/pin_insertion_env.py` environment with the
`side_by_side` layout and `photo_20261027` setup profile. The older synthetic
generators now require an explicit `--legacy-prototype` flag for reproduction.

## Environment and registration

`scripts/pin_insertion_environment.py` binds the original task model and its
physical reset/controller. MuJoCo **3.14.0** and Gymnasium **1.2.3** match the
training repository; the previous runtime's MuJoCo 3.3.7 could not establish
the original grasp reliably. Native arm joint angles are passed through
unchanged. Native gripper encoder conversion remains an approximation, distinct
from the task simulator's metric jaw motion.

The original finger STL contains separate white-digit and black-mount connected
components. `photo_finger_visual_partition.py` splits their original facets
without moving vertices, changing collisions, or changing inertia. Both black
and white geometry participate in camera occlusion. A rejected initial fit
that omitted black occluders is retained only as an audit artifact.

Selected registration artifacts under `work/pin-insertion-real/`:

| Artifact | Evidence and scope |
|---|---|
| `own-finger-camera-fixed-grip.json` | Own-right-finger silhouettes, three train and two validation images. Fixed jaw conversion. Validation IoU 0.839 versus nominal 0.458; boundary error 1.68 versus 8.33 px at 320-pixel width. |
| `physical-grasp-planar-base-registration.json` | Recorded arm angles and TRAIN pin annotations, fixed camera and physically observed pin grasp. Both bases remain upright at the same tabletop height. Relative XY separation 13.94 / 621.72 mm and yaw −0.169°. Validation pin projection error 18.56 px. |
| `photo-carrier-grasp.json` | TRAIN bore annotations fit the carrier pose relative to the right tool. Translation −24.25 / 0.67 / 9.74 mm. Four-bore projection error 1.60 px, but the annotations contain only two distinct bore layouts; this is limited pose coverage. |

These are conditional image-registration hypotheses, not independent metric
calibration or demonstrated insertion accuracy. In particular, carrier yaw has
a fourfold bore symmetry. The planar base constraint is selected for physical
consistency, despite a slightly worse validation projection error than allowing
the bases to tilt and float.

The matched replay reconstructs the supplied full CAD assembly, including all
four gears and the central `small_carrier` unit. Recorded arm angles remain
exact. Parts are attached kinematically in this replay: it is a visual comparison,
not a physical insertion rollout. The visible spare pin on the table is distinct
from the left-held `pin_2`; the held pin retains the original physical-reset
tool-relative transform.

### Carrier face reproduction

An existing checkpoint sidecar preserves a newer `carrier_real_face_v4`
photograph-based visual design. Its original mesh and builder are unavailable
on this machine. At the user's request, the visual face was reproduced
from that saved specification, with source provenance retained. It is an
approximate visual reconstruction, not recovered manufacturing CAD.

The specified face has a 52.98 mm outside diameter, a 22.95 mm central upper
opening, four 8.24 mm insertion bores at 17.9625 mm radius, four 2.6 mm inner
holes, and four raised mounting bosses with 3 mm holes. The 4 mm boss height
was assumed in the original visual profile; it is not a measured dimension.

An independent horizontal cross-section audit of the original visual STL
confirms the insertion bores have approximately 4.119 mm radius, consistent
with the 4.12 mm design value. The original decomposed collision boundary is
approximately 4.111 mm because of faceting. The 0.17 mm threshold in the task
success metric is **not** bore radial clearance or a measured chamfer; earlier
diagnostic aperture/category code incorrectly inferred 4.17 mm from it.
Correcting this label geometry does not alter the simulated collision meshes.

The reproducible builder is `scripts/photo_carrier_visual_profile.py`, with
the recovered specification versioned under `config/pin-insertion-photo/`.
It preserves the original lower CAD, recovers the upper plate bottom at
14.29 mm from original vertices, and rebuilds the upper face using mesh boolean
operations. The resulting visual STL is watertight with consistent winding.
Export verification reloads the actual float32 STL; independent section and
ray tests confirm the bore centers, open apertures, mouth height, boss height,
and absence of accidental caps. The final STL contains 16,386 triangles. The
adapter verifies that collision geometry, mass, inertia, joints and target
coordinates are unchanged. Versioned camera/base/profile files make the new
registered setup the default; `--setup nominal` selects the original baseline.
For seeds 0, 1 and 2, original-visual and reproduced-visual registered models
also produce bit-identical full physical states after reset and three hold
steps, with both grasps and carrier support valid. This specifically verifies
that changing appearance did not change those physical trajectories.

## New contact recording

`data/hitl/review/20261104T193211-4db4b17f` is a 94.9-second recording with five
stationary carrier poses. The user moved the pin around the hole and inserted
it at each pose. The insertion moments provide annotation references; a visible
proximal pin cap must not be substituted for the hole or buried distal tip.

The frozen split in `work/pin-insertion-newdemo/split.json` was defined before
coordinate annotation or fitting:

| Pose block | Time interval, seconds | Use |
|---|---|---|
| B1 | 0–18 | Train |
| B2 | 21–32 | Validation |
| B3 | 34–56 | Train |
| B4 | 60–82 | Test |
| B5 | 84–94.9 | Test |

Movement between blocks is excluded. These are pose holdouts **within one
recording and likely one grasp**, not independent sessions. Older independent
recordings remain separate tests. `prepare_pin_insertion_block_manifest.py`
checks original source hashes, frozen block boundaries, image hashes, and
recorded state/image clock alignment; it preserves actual source recording
identity and saved FK. Future insertion references are annotation provenance
only, never a seed supplied to the online hole cache.

The final annotation manifest contains 39 train / 8 validation / 16 test frames,
with 16 / 1 / 6 visible pin-tip labels and 39 / 8 / 16 hole labels. Hidden pin
tips remain null. Independent review removed six ambiguous TRAIN pin labels
that could have taught the model to follow a static small bore. The single
visible validation pin is a serious limitation for checkpoint selection.
Positive gate labels use the user's contact-exploration protocol plus visual
evidence; they are not force-certified contact labels.

Recorded encoder FK was independently checked on all 63 annotated images:
timestamp-matched native joint angles reproduce the saved tool poses in each
arm's own base to numerical precision (maximum position difference
2.99e-16 m, maximum rotation matrix entry difference 6.11e-16). This checks the
joint/FK convention, not real camera calibration or gripper encoder travel.

## Real-image retraining and held-out results

Two 120-epoch candidates were adapted from the earlier `adapted.pt` checkpoint,
using only new B1/B3 images, photometric/affine augmentation, and recorded
states. **No legacy synthetic replay was used in this adaptation.** An image-only
candidate and a joint-conditioned candidate were compared on B2 using equal
weight for the pin and hole mean errors. The joint-conditioned checkpoint at
epoch 86 was selected and its SHA frozen before test evaluation. Its balanced
validation error was 9.40 px, versus 9.94 px for the image-only candidate. B2
contains only one valid pin label, so this selection has limited support.

All errors below are native 640×480 image pixels:

| Evaluation | Earlier checkpoint pin / hole | Selected checkpoint pin / hole | Label counts pin / hole |
|---|---:|---:|---:|
| B4/B5 held-out poses, same new recording | 86.07 / 20.79 | 13.76 / 7.29 | 6 / 16 |
| Older independent recordings, approach plus contact audit | 20.59 / 12.42 | 9.35 / 10.21 | 16 / 44 |

The constant TRAIN-mean baseline on B4/B5 is 54.13 px for the pin and 23.19 px
for the hole. All six visible pin labels in the new test are in B4; B5 supplies
hole labels but no independently visible distal pin tips. Typical annotation
uncertainty is approximately 6 px; sub-uncertainty differences are not metrology.

At activation threshold 0.8, the new test's 12 protocol-positive frames are all
detected, but **2 of its 4 negative frames activate falsely**. At threshold 0.5,
3 of 4 negatives activate falsely. These small, correlated counts do not support
autonomous enabling. The older test has 20 known negatives and no verified
positives: zero false activations there does not establish positive recall.
The model remains localization-only; direct XYZ execution is disabled.

Frozen selection, model hashes, checkpoint ancestry, predictions, ablations and
split audits are saved in `work/pin-insertion-newdemo/selection.json`,
`test-evaluation.json`, and `old-independent-evaluation.json`. Checkpoint
ancestry retains the earlier optimization/selection sources so the new block
IDs cannot disguise overlap with earlier training data.

The separate October 27 failed-alignment challenge remains difficult: three
correlated labeled frames improve from 34.03 to 20.42 px pin error, while hole
error worsens from 11.98 to 16.23 px. Image-plane correction-direction cosine
improves from 0.03 to 0.55, but this is not a calibrated three-dimensional nudge
or an executed insertion test.

### Causal hole retention

Before viewing selected-model test results, an offline memory protocol was
frozen: take the median CNN hole prediction from the first five frames in each
held-out block, then retain it. No future insertion reference or annotation
coordinate enters the cache. Those initial views are explicitly **unverified**,
so this diagnostic does not satisfy the runtime's trusted-visible initialization
requirement and cannot enable motion.

On 14 later labeled frames, fresh predictions average 7.21 px hole error and
the retained predictions 9.00 px. Retention helps 4 of 14 frames and worsens the
mean. Remembering an initially biased location does not solve localization;
reliable initialization is still required. The video distinguishes current
predictions from the retained diagnostic location.

## Physical diagnostic

`generate_pin_insertion_photo_data.py` uses the original environment reset and
bounded controller steps. The `descend-to-contact` policy stops descent after
observed contact and then holds. It records full simulation state, native arm
angles, actual contact forces, projected landmarks, depth visibility, and bore
aperture visibility. It does not teleport parts during a rollout. These outputs
remain diagnostic-only: a contact snapshot does not establish stable alignment,
a verified enable gate, or successful autonomous insertion.

The physical task's nominal grasp and the image-fitted kinematic reconstruction
are separate modes. Registering a camera changes rendering only. Changing a
physical grasp requires a successful contact-based reset and subsequent physics
validation; a good-looking kinematic attachment is insufficient.

With the registered camera and planar bases, the first seed-42 descent at action
0.2 reaches rim contact but subsequently exceeds the original environment's
35 N force limit (35.88 N) and terminates at step 167. The guard was not bypassed.
A separate, gentler action-0.1 diagnostic completes 400 steps, latches initial
contact at step 294, and retains both grasps and carrier support. It remains
misaligned and does not insert; contact is intermittent after the latch. The
failed and gentler runs are retained separately. Neither is a CNN-controlled
robot trial or training-approved physical-success dataset.

The final gentler run uses the verified v4 visual STL, registered camera and
planar bases, and the frozen selected real-adapted checkpoint. **CNN transfer
to this physical simulator is still poor:** hole error is 87.34 px over 16
directly visible hole samples; no pin sample passes the external-visibility
validity check, so pin accuracy is unmeasured. The final held grasp differs
from the image-fitted native-joint replay. This diagnostic is shown explicitly
and was not used to retune the frozen real-test checkpoint.

## Verification

The final combined offline test run passed **127 tests**, covering model and
adaptation contracts, split/source integrity, causal memory, native FK and
simulation adapters, camera/base registration, carrier export and aperture
geometry, physics invariance, and video rendering contracts. `git diff --check`
also passes. Tests do not certify hardware alignment accuracy. No live robot
commands were sent.

Artifact paths, SHA256 hashes and the combined outcome are recorded in
`docs/pin-insertion-photo-setup-results.json`.

## Demo-reference hole cache

`app/pin_hole_demo_cache.py` implements a separate hole estimator using six
clear TRAIN demonstration images. Four references come from B1/B3 of the new
recording, and two from older TRAIN recordings. Each reference contains a
manually annotated upper-mouth center, carrier-face polygon, target-neighborhood
exclusion mask, source identity, and image checksum. The v2 reference audit
traces the complete outer lip and assigns 4 px annotation uncertainty.

At inference, masked reference features are matched to the current carrier.
A robust image transform transfers the annotated hole center into the current
image. Carrier optical flow bridges short matching gaps; a brief stationary
hold is available if visual tracking drops out. Matching does not require a
fresh CNN hole prediction. Sequence/camera/grasp changes reset temporal state;
world motion of the right arm does not itself trigger a reset. Each held-out
test block nevertheless starts independently from the same TRAIN reference bank
for evaluation, rather than receiving a cached test coordinate.

Four configurations were compared on the original B2 validation labels. The
selected affine configuration uses eight geometric inliers, a 0.72 descriptor
ratio, a 3 px robust-fit threshold, and a 2-second maximum hold. It produced
167 tracked / 3 unavailable frames out of 170. On the seven labeled frames
with available output, error was 7.35 px versus 14.98 px for the CNN over all
eight labels. Model selection penalizes unavailable labels by 40 px, avoiding
an apparent improvement obtained by dropping difficult frames. The selected
score is 11.44 px. Original labels and their uncertainty remain preserved.

Use `work/pin-hole-demo-cache/references-v2.json` with
`work/pin-hole-demo-cache/selected-config.json`. The frozen selection artifact
binds both files, the implementation and comparator CNN by SHA256. The pure
`merge_with_prediction(prediction, cache_result, image_size_wh)` helper keeps
the current CNN pin unchanged and substitutes only an available cached hole.
It retains the original CNN hole for comparison. When the cache is unavailable,
the returned source explicitly identifies the unverified CNN fallback.
Gate and XYZ outputs are not modified or authorized by this helper.

```python
import json
from app.pin_hole_demo_cache import DemoHoleCache, merge_with_prediction

bank = "work/pin-hole-demo-cache/references-v2.json"
config = json.load(open("work/pin-hole-demo-cache/stationary-config.json"))
cache = DemoHoleCache(bank, **config)

# In the camera loop: preserve grasp_id until the carrier is released/regrasped.
prediction = predictor.predict(rgb, measured_state14)
hole = cache.update(
    rgb, timestamp_ns=image_receipt_ns, now_ns=now_ns,
    sequence_id=rollout_id, camera_id="right_wrist", grasp_id=carrier_grasp_id,
)
prediction = merge_with_prediction(prediction, hole, (rgb.shape[1], rgb.shape[0]))
```

The original B4/B5 and older separate recordings are reused benchmarks from
earlier experiments. Parameters are frozen on validation before testing this
new method, but these are not newly collected unseen test recordings. A second
annotator independently audited the full upper-mouth rim from raw evaluation
images without accessing old labels or predictions. Those audit labels are
supplementary and are not used to tune the cache or replace earlier reports.

### Independent-start results and continuous memory

The frozen independent-start benchmark gives:

| Evaluation | Available frames | Available hole labels | Demo-cache mean error | Current CNN mean error |
|---|---:|---:|---:|---:|
| Older separate recordings, 10 episodes | 238 / 239 | 43 / 44 | 2.02 px | 10.21 px, 44 labels |
| New B4/B5, reset independently at each pose | 224 / 505 | 10 / 16 | 3.70 px | 7.29 px, 16 labels |

Available-only error is not an overall success rate. In B4 the cache cannot
initialize until frame 1016, then loses visual support and expires its 2-second
hold after frame 1082. B5 remains available throughout. The B4/B5 score including
the predefined 40 px penalty for missing labels is 17.31 px, worse than the CNN.
Those original results remain in `test-blocks.json` and `test-independent.json`.

`stationary-config.json` is a subsequent engineering profile: it preserves the
validation-selected matcher and extends only the stationary hold to 30 seconds.
It assumes the carrier remains fixed relative to the right wrist camera until
the caller changes `grasp_id`. The continuous replay processes every source
frame, including observations before B4/B5, and resets on actual recording
segment changes rather than evaluation pose boundaries. Held estimates are
explicitly marked as held, not current detections. This corresponds to the
user's intended continuous grasp, but is a post-benchmark replay of a recording
used to build references, not a new independent validation result.

The blind full-rim annotation audit supports a label-plane ambiguity in the
earlier validation scores. Against that separate audit, the available demo
cache averages 1.41 px on B2 and 3.55 px on B4/B5. Reference/evaluation annotation
uncertainty is several pixels, so these are consistency measurements rather
than metrology. Neither the bank nor parameter selection uses those new
evaluation coordinates. Original-score results are retained alongside them.

On this workstation, the six-reference cache alone takes a median 37 ms per
640×480 frame (mean 41 ms, p95 55 ms; four OpenCV threads, 90 validation frames
after warmup). Image decoding and pin CNN inference are excluded from that
timing. Runtime and source hashes are in `runtime-benchmark.json`.

The completed continuous replay processes all **1,482** images with a single
initial reset. Across scored B4/B5 it retains output on **505/505 frames**:
222 are freshly tracked and 283 use the held position. Mean error over all
16 original hole labels is **3.52 px**, compared with **7.29 px** for the CNN.
Against the independent blind rim audit it is **3.60 px**, so the improvement
does not depend on replacing the original annotation convention. These remain
small, correlated samples with several-pixel annotation uncertainty.

This replay establishes useful continuous memory under the fixed-grasp
assumption. It does not erase the separate-start failure: beginning B4 without
earlier observations still delays acquisition, and the conservative 2-second
benchmark cache expires during occlusion. No tracker parameters were tuned on
the replay; the only engineering change was longer hold retention and removal
of artificial pose-block resets. Regrasping must change the caller's grasp ID.

The cache, evaluator, continuous replay, existing memory and predictor tests
pass **44 checks** together. The demonstration videos use purple for the
demo-cache hole, cyan for the current CNN pin, yellow for the CNN hole and gray
for the earlier first-five-frame cache. Tracked and held estimates are labeled
separately; evaluation coordinates are never used to draw the tracker output.
