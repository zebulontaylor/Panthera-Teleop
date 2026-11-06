# Stationary carrier-face observation accumulation

`app/stationary_face_accumulation.py` provides `StationaryFaceAccumulator.update(prediction, state14, timestamp_ns, config=None, *, eligible=True)` and `reset(reason='reset')`. It reads data only and has no camera, motor or hardware interfaces. Root worker integration owns frame/state freshness checks and passes only eligible measurements. An optional `now_ns` configuration enables local stale/future checks as well.

Accumulation begins after at least two frames and 0.4 seconds within one measured-state anchor. Both arms must remain within 0.003 radians per joint; both grippers within 0.01 radians. Native-chain FK additionally limits each tool's anchored translation to 0.5 mm and rotation to 0.005 radians, including the existing 0.165 m tool offset. Instantaneous joint/gripper velocity limits supplement these anchored tests. Slowly accumulating drift therefore clears history even when consecutive changes are small. This FK is a motion gate, not a new camera or metric calibration.

A movement, invalid/nonmonotonic timestamp, ineligible frame, camera/context/cache-epoch change, or input gap over 2.5 seconds clears the pool immediately and returns the current raw prediction during renewed qualification. Per-frame SIFT reference IDs do not define identity and may switch or disappear without resetting the pool. Worker integration resets actual grasp/camera/context changes explicitly. Three consistent fresh bore displacements over 4 pixels, or a bearing shift corroborated by a second coplanar landmark, also reset history despite unchanged joints. An isolated bearing jump is discarded without clearing the face pool or accepting its current homography.

Only current `observed=True`, non-inferred large-bore and auxiliary observations enter history. The four main slots use the existing image-cyclic canonical ring of radius 0.0179625 m; their phase is not verified physical CAD yaw. Auxiliary observations associate by kind/plane and a nearest-neighbor distance of at most 6 pixels. Positions use per-landmark medians over at most 31 samples. Each unique large bore has geometry weight 1, bearing 0.5, verified coplanar inner point 0.25; repeated frames do not increase a landmark's fitting weight. Raised bosses can be retained for display but never enter the homography fit. A later verified inner-point identity upgrades an earlier unclassified pool record; temporary loss of identity does not erase prior verification.

Before pooling, cyclic bore labels associate with historical image positions using
an unambiguous phase match within the visual shift limit. The homography and
auxiliary canonical coordinates follow that same phase change. This prevents
reference changes from pooling opposite bores into a false center position.
Isolated main-bore jumps are discarded after checking for coherent face motion.

At least three noncollinear pooled main/bearing anchors can fit an image-affine correction to the previous mapping, preserving its perspective. Verified small points may refine it through the existing bounded weighted helper. When pooled anchor support is insufficient, a currently supported mapping is smoothed using medians of its projected canonical corners, never by averaging homography coefficients. A last good complete face mapping may be retained within the fixed-carrier hold interval. Outline and layout geometry propagate through `H_new @ inverse(H_old)` when the mapping changes.

`config.fixed_carrier_hold=False` disables retention and clears memory. `hold_s` sets per-observation retention, capped at 30 seconds. Old observations and their mapping expire; repeatedly returning a retained mapping does not refresh its supporting observation timestamps. The input stream must continue delivering fresh sensor state, even while RGB hole detection is intermittent.

The merged output includes `accumulation.stationary`, `frames`, `point_count`, `retained_count`, `age_s` and `reset_reason`. Main holes expose `sample_count`, `last_seen_age_s` and `accumulated`. Retained observations are marked `observed=False, accumulated=True`; layout projections never observed remain `inferred=True`. `visible` reports fresh detections only. Accepted fresh bore detections use their current measurement for target coordinates, normalized coordinates and layout centers; `fresh_observation_px` records that same measurement. Historical medians supply retained bore coordinates and pooled geometry fitting, without overriding a fresh target. Historical auxiliary layout glyphs are also explicitly historical rather than fresh observations.

Unmatched auxiliary candidates without eligible geometry correspondence are removed
instead of leaving historical guesses in the pool. Eligible supporting anchors
still retain the normal bounded history.

The worker also calls `accumulate_pin()` after predicting pin endpoints. At the
same qualified stationary pose, matched TOP/BOTTOM estimates use medians over at
most 31 pairs within the same hold timeout (capped at 30 seconds). Missing pin
estimates can retain that position until expiry; `pin_accumulation` reports sample
count, age and whether the estimate is retained. Retention never establishes
visibility or automatic-motion eligibility, and BOTTOM remains inferred. Manual
pin selection overrides pooling. Movement and context changes clear both pools.

Losing current face registration alone does not clear either qualified pool.
Input freshness is checked at worker receipt for accumulation; completed-result
freshness still independently blocks stale motion proposals. A slow search cannot
erase valid history solely because the result finishes late. Incoming stale or
skewed data, pose changes, long input gaps and expired observations still clear
or expire memory. A stationary, pooled estimate is evidence of accumulated image
support, not certified insertion accuracy or permission to move.
