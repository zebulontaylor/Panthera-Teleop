# Object-aware reconstruction follow-up

The motion-only experiment does **not** meet the real2sim evaluation goal.
It did not extract carrier, gear or pin poses, and its static simulation object
locations were not reconstructed from the demonstrations.

The [gripper-anchored reconstruction](real2sim-grasp-reconstruction.md) is a
stronger alternative to the image-only approach below: use measured pickup
positions and a rigid object-in-jaw transform to reconstruct held object motion.
It can start without camera calibration; images refine the grasp offsets.
The calibration limitations below apply to independent image-based metric
reconstruction, not to metric information already available in recorded FK.

The follow-up tested actual image landmark extraction on
`20261023T180718-47e74c70`. Saved diagnostics are under
`data/experiments/real2sim-objects/`:

- `predictions.jsonl`: 45 synchronized samples at 3 Hz from the existing adapted
  gear and pin/carrier models, using overhead / port3 left / port2 right.
- `object-landmarks-contact-sheet.jpg`: inspected wrist-image overlays.
- `object-extraction-review.json`: visual and numeric rejection findings.
- `focal_sensitivity.json`: camera-local carrier pose hypotheses from approximate
  manually corrected bore centres at 8 s, across assumed focal lengths.

No metric world object pose has been accepted, and no simulation object has
been moved on the basis of these uncalibrated predictions.

## Findings

Carrier bore locations are plausible in several approach frames, but the raw
detector sometimes maps two bore slots to the same physical hole. For example,
two right-view outputs at 5.33 s are only 5.1 pixels apart. Start/end frames
hallucinate objects on background. Gear and pin predictions likewise need
phase selection and correction. Detector confidence alone is insufficient.

The carrier CAD provides a metric square of four bore centres at radius
17.9625 mm, so corrected observations can constrain pose. However, these are
unordered planar points; carrier yaw has 90-degree symmetry and normal sign
needs a visible-face prior. The four bore mouths lie 17.25 mm above the CAD
body origin. `scripts/real2sim_object_pose.py` fits all acceptable camera-local
pose hypotheses with explicit supplied intrinsics/distortion, and separately
supports plane homography queries under an explicit same-plane assumption.
Neither API silently invents camera/world calibration.

Metric location is currently ambiguous. In the left wrist's same 8 s image,
assumed focal lengths 250–550 pixels yield carrier depth **110–242 mm**, while
all best-fit reprojection RMS errors remain below 1.8 pixels. In the right
wrist the corresponding range is 74–163 mm. These are a sensitivity experiment,
not measured depth intervals or reconstructed world locations. Lens distortion
was set to zero for this demonstration and is visibly substantial in the real
views, so even these assumptions are incomplete.

The saved camera calibration has null intrinsics and extrinsics. A metric fit
must estimate/measure those parameters, resolve the relative bases, and check
corrected object landmarks across distinct frames. A single planar observation
with a guessed focal length cannot establish correct simulation positions.

## Success criterion

Accept an object reconstruction only when its fitted pose projects onto the
same carrier/gear/pin features in multiple real views, remains consistent on
unused frames, and has bounded sensitivity to calibration/annotation errors.
Then initialize the simulator with those object poses in the same frame as the
arms and check demonstration replay contacts before evaluating the learned
policy. Gear orientation needs silhouette/CAD evidence beyond its single centre
landmark; two pin endpoints establish its axis but not axial roll.
