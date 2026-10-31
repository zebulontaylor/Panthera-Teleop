# Offline alignment validation from the kept recordings

The offline checks do not yet validate automatic alignment nudges. They use the
original 17 recordings, the existing visual annotations, and the unchanged
13-training / 4-validation episode split. No motors were commanded, runtime
weights were not replaced, and alignment assistance remains disabled with
calibration unverified.

Inference uses CUDA with the corrected model input order
`overhead, wrist_port3, wrist_port2`. The final-approach cache contains 748
synchronized observations with no rejected timestamps. Cache, source image,
checkpoint, phase-selection, and label identities are recorded in the reports.
An additional kept recording appeared during this work; it is excluded from
these original-17 results.

## Correction prediction on reserved trials

Training alone defines the local reference pose, conditional image Jacobian,
and demonstrated target offset. Reserved trials supply actual measured
joint/image changes. The demonstrated offset is not certified seating geometry.

| Task | Reserved motion pairs | Image-motion RMSE | No-motion baseline | Improvement | Proposed directions supported by recorded motion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gear | 27 | 17.06 px | 17.86 px | 4.48% | 0 / 92 |
| Pin, exploratory | 20 | 61.03 px | 66.57 px | 8.32% | 1 / 84 |

Both improvements fall below the existing 10% margin. One reserved gear trial
is worse than no motion (23.61 versus 11.83 px). Pin results also lack verified
cross-camera identity of the physical bore. Its one supported interpolation
cannot establish correction reliability or isolate other-arm effects.

Recorded-direction support requires a nearby starting pose, a 12-joint motion
direction cosine of at least 0.9, and downward-only scaling of the observed
response. The 92/84 offsets are diagnostic DLS proposals, not runtime-approved
commands. The videos contain too little matching motion to establish their
effect. Aggregate joint excitation alone does not supply this local coverage.

The earlier gear statistical pass used leave-one-episode-out fitting across
all 17 trials and a broader gear window. This stricter test reserves four whole
trials and examines the final approach. Those results answer different questions;
the earlier pass is not evidence that these proposed nudges work.

See [the detailed replay report](../logs/alignment-validation/replay-near.md)
and [numeric evidence](../logs/alignment-validation/replay-near.json).

## Grasp recognition

A frozen CNN encoder with an offline visual classifier correctly classified
15/15 original reserved labels: four empty, four gear, and seven pin. However,
all four reserved empty examples have open/neutral grippers. A classifier using
only gripper position also gets 15/15 by argmax, and masking the jaw-tip region
does not remove the visual classifier's accuracy. This set cannot establish
closed-empty rejection, slip detection, or actual object retention. Its softmax
scores are not calibrated confidence estimates. Labels are AI visual annotations.

The supplementary visual audit found closed-empty examples in all 17 existing
recordings, using both wrist views and preceding/following frames. The unchanged
original classifier misclassified all four reserved examples as pin; three
passed its 0.85 acceptance threshold. Thus the original 15/15 result concealed
a real closed-empty failure.

A second offline candidate adds only the 13 training-trial closed-empty labels,
with the same features, regularization, and threshold. It correctly labels all
four reserved closed-empty examples. On the original 15 reserved examples it
gets 14/15 by argmax; the one pin-to-empty prediction abstains at the acceptance
threshold. Across the combined reserved set it has 18 correct accepted results
and one abstention, with no false held-object accepts among eight empty examples.
That small, correlated set still does not establish slip/release reliability.
These validation episodes have now been inspected repeatedly; this is an
exploratory candidate evaluation, not a fresh independent certification.

The original result and diagnostics remain in
[the grasp report](../logs/alignment-validation/grasp-summary.md); the separate
[supplemental numeric report](../logs/alignment-validation/grasp-supplement-report.json)
preserves the original classifier failure and training-only refit. All labels
are AI visual annotations, including a second-agent check of the four reserved
closed-empty examples. No runtime grasp detector was installed.
See the [supplemental grasp summary](../logs/alignment-validation/grasp-supplement-summary.md)
for per-set metrics and uncertainty bounds.

## Controller gates and software checks

Replaying the actual controller on 636 corrected-order approach observations
produces no activation. Enabling the option alone stops at the unverified
calibration gate. An isolated diagnostic fixture exposes further blockers:
the original two-view visibility/confidence checks and passive-gripper torque
predicate each pass zero recorded observations. This does not imply poor CNN
point localization; those heads and torque criteria do not establish the real
visibility/grasp conditions required here. Missing seating/target geometry is
explicitly marked as test placeholders, never saved as verified calibration.

All 30 synthetic injected-fault cases start from a demonstrated active baseline
and stop with zero requested torque. They cover stale/skewed/missing/invalid
evidence, visibility loss, checkpoint/camera changes, absent grasp, joint limits,
excluded collection modes, disabling, and the 0.3-second burst limit. Additional
tests enforce the 0.006-rad offset and 0.15-Nm component caps. These establish
software behavior; they do not validate perception, contact dynamics, or achieved
physical movement. See [the gate report](../logs/alignment-validation/gates.md).

Verification completed with 47 passing unittest checks across the alignment,
manual-label, and visual-grasp suites, including the 30-case fault matrix and
the repeated-pose frame-identity regression. An independent subagent recomputed
all 47 reserved motion-pair deltas and error reductions after that fix.

Remaining evidence gaps are real retained-object detection, appropriate
visibility/target association, measured seating references, and independently
supported corrective movement. The offline candidate files do not authorize
turning on robot motion.
