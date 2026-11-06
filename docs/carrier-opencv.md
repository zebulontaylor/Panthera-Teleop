# Left-wrist carrier face mapping

The dashboard defaults to `--alignment-backend opencv`. The trained pin model
continues to estimate TOP and BOTTOM Y on the configured gripper centerline.
Changing the hole backend does not change the nudge controller, step limits,
right-arm freeze, height lock, or operator-triggered execution.

`app/left_carrier_opencv.py` uses real TRAIN references, fresh dark-aperture/rim
verification, and geometric face registration. Every prediction uses the current
RGB image. Missing observations are not retained at old pixel coordinates as the
left camera moves. The four large insertion bores are the only automatic nudge
targets; auxiliary threaded/mounting holes and the bearing supply face landmarks.
Automatic selection rechecks the bore nearest the current pin estimate on each
supported face map. Only an explicit operator selection keeps a chosen bore.

The complete face layout comes from
`config/pin-insertion-photo/carrier-real-face-v4-profile.json`. It is an approximate
photograph/CAD reconstruction. Three matched non-collinear landmarks can fit an
affine face map; additional verified landmarks can check/refine registration.
Projected holes and the face outline remain hypotheses, distinct from current
rim detections. Raised mounting bosses are about 4 mm above the plate in that
profile and are not treated as exact coplanar metric correspondences. Absolute
physical bore numbering and the layout's image orientation are not established
by the four rotationally symmetric insertion bores alone.

The image overlay draws the actual fitted ellipse for fresh insertion bores in
green, with a small center cross. Teal diamonds mark retained bore centers and
orange crosses mark inferred centers; neither implies a freshly fitted rim.
Blue crosses show eligible supporting landmarks. Approximate projected opening
outlines and unverified auxiliary candidates are omitted from the default overlay;
the face outline remains gray. Purple squares mark the selected insertion target.
The old fixed seven-pixel bore circles were center markers, not measured sizes;
the detector's radius limits are unchanged. Pin TOP
is estimated and BOTTOM is inferred; neither is certified real metrology.

The original supplied frame can initialize without a reference match: two
adjacent large-bore rims and a verified concentric bearing center form three
non-collinear anchors. The resulting map projects the plate outline and all 13
profile openings (four insertion, four inner threaded, four mounting, one center).
Detected small holes check the two possible inner-hole phase orientations. A
phase is marked verified only with sufficient fresh image support; physical
quarter-turn bore identity remains unverified. Opposite bores plus their center
are collinear and do not establish this three-point map.

## Real-data labels and evaluation

`scripts/prepare_left_carrier_opencv.py` generated 378 real training frames with
648 visible-bore pseudo labels, using ten reviewed B1/B3 seeds. Bidirectional LK
tracks the rims and every frame must pass independent local ellipse, contrast,
rim-coverage and scale checks. Occluded tracks terminate. No FK, simulator
projection, hidden coordinates, or heldout frames supply these real labels.

The reference bank is
`work/left-carrier-opencv/runtime/reference-bank.json`. Its source images are
TRAIN-only. Fifteen manually reviewed B2/B4/B5 frames are kept separately in
`work/left-carrier-opencv/evaluation-manifest.json`; B2 is validation, B4/B5 test.
These blocks share a recording and likely a grasp. Labels have about 5–6 px
uncertainty, and unordered matching does not verify physical bore identity.

Frozen evaluation results, per-frame outputs and native examples are under
`work/left-carrier-opencv/runtime/`. The initial reference/rim baseline, later
auxiliary/global face mapping, and trained CNN comparisons are separate artifacts.
Coverage and abstentions matter alongside pixel error; simulation accuracy alone
does not establish real performance.

The original frozen reference/rim baseline found 29/44 manual bore centers within
10 px, with no unmatched observed prediction. The full-face extension reuses
those 15 frames as a development audit: 33/44 matches, one unmatched observation,
and 12/15 supported face maps. Its matched error is about 1.82 px mean, below the
manual annotation uncertainty; this is not an independent precision measurement.
The reused B4/B5 subset has 13/23 matches and 5/8 supported maps. No claim is made
that occluded holes have been directly detected or that every real view works.

The CNN was also trained on the OpenCV real labels with synthetic replay and
substantial stochastic augmentation. Its artifact remains at
`work/left-carrier-cnn/real-adapted/best.pt` for comparison, available using
`--alignment-backend cnn`. See `docs/left-carrier-cnn.md` and
`docs/held-pin-cnn.md` for the actual training runs and limitations.
