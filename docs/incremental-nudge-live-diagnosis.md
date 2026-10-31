# Initial incremental-nudge attempts

## Subsequent zero-step timeout

The `20261022T234245-7f1e72e9` recording contains a later `trial_timeout`
before the first correction. In the inspected 7,591 active-controller samples,
all six left-joint positions were identical and vision reported ready throughout.
The maximum absolute reported joint velocity had median 0.0141 rad/s,
90th percentile 0.0267 rad/s, and maximum 0.0723 rad/s. The old 0.02 rad/s
settling gate therefore reset the dwell despite unchanged position counts.
The fix uses stable hardware position counts for settling, retains the separate
perception evaluator's movement/freshness gates, and exposes the wait stage and
sample count. Three individually fresh images may now span a 0.5-second
aggregation window, avoiding starvation from normal inference latency.

## Initial position-hold aborts

Read-only analysis of
`data/review/20261022T233102-1a4804f8/states.jsonl` found 83 completed starts
and 304 active incremental-controller samples. All attempts remained at
`steps=0`; every active sample commanded the unchanged initial joint position.
All stopped with `measured_cartesian_bound`. No visual correction was issued.

| Measured quantity, start to first stopped sample | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Elapsed time | 19.66 ms | 26.79 ms | 34.37 ms |
| Encoder/FK horizontal displacement | 0.283 mm | 0.729 mm | 1.363 mm |
| Encoder/FK downward displacement | 0.314 mm | 0.806 mm | 1.791 mm |

Every horizontal displacement exceeded the 0.25 mm bound before the required
150 ms settling interval. These movements occurred during the transition from
gravity assistance to initial position hold, before image-based nudging.
In the first attempt, joint 2/3 measured torque changed from approximately
−1.84/+2.92 Nm to −0.115/+0.135 Nm while the original position target remained
commanded; the next state sample recorded the abort. This identifies the failing
stage but does not establish a unique physical cause of the torque change.

Recorded joint position increments were approximately 0.0006283 rad. At these
poses, one count on joints 1–5 corresponded to median modeled tool translations
of 0.201, 0.196, 0.306, 0.156, and 0.118 mm respectively. The initial controller's
0.0002 rad and 0.025 mm settling tolerances were smaller than these increments.
This is a separate resolution mismatch; it is not evidence of a camera
calibration error. Cartesian values here come from joint encoders and FK, not
independent optical measurements of physical displacement.

The revised proposal audit in `scripts/validate_incremental_nudge.py` evaluates
the hardware path: a 0.5 mm continuous proposal cap followed by SDK-grid target
encoding and checks of independently completing joints. Its report is
`logs/alignment-calibration/incremental-nudge-offline.json`, including source
hashes. Earlier counts from the 0.25 mm unquantized proposal audit describe a
different implementation and must not be reported as current hardware-path
coverage. Neither audit measures convergence, verifies physical alignment, nor
commands hardware. SDK encoding/decoding checks can establish a software
round-trip error; they cannot alone explain all observed holding behavior.

## Later relaxed-profile trial

Read-only analysis of
`data/pending/20261022T235057-d45ff3fb/states.jsonl` found two attempts, each
issuing one correction. The first issued its target at 7.329120581 s and stopped
at 7.436701769 s with `collection_or_grasp_ended`: the recorded grasp remained
active but `settled` changed to false. Encoder/FK displacement during those
107.6 ms was `[-1.034, +1.391, -2.225] mm`. No post-step visual response was
evaluated before that stop.

The second issued its target at 12.942005299 s and stopped at 14.500796366 s
with `image_response_wrong_direction`, then held position. Encoder/FK
displacement was `[-0.159, +0.169, -1.671] mm`. The two vertical residuals changed
from `[35.77391, -7.29890] px` to `[33.69102, -8.42626] px`: actual change
`[-2.08289, -1.12737] px`, versus predicted `[-1.22931, +3.56613] px`. One view
improved while the other moved opposite to the prediction. Thus this episode's
failure to continue was a response check after an actual movement, unlike the
earlier zero-step hold-transition failures.

Fifteen distinct readiness predictions between 10 and 12.94 s had exactly
unchanged recorded left-arm joint positions. Their vertical residual standard
deviations were `[2.906, 0.447] px`, median absolute deviations
`[2.025, 0.267] px`, and ranges `[28.353, 39.645]` and `[-9.089, -7.377] px`.
The last 0.94 s contained five distinct predictions; the second-view standard
deviation was 0.542 px and range was 1.612 px. The observed opposite 1.127 px
shift is comparable to raw-frame scatter, but these samples do not prove it
was noise. Post-step raw readiness residuals were not stored, so the available
post-step evidence is the controller's three-frame median. Reobserving at the
settled endpoint can resolve ambiguous response evidence; ignoring a persistent
opposite response is not supported by these measurements. These statistics are
not an independent calibration or physical alignment validation.

## Physical-step confirmation timeout

In `data/pending/20261022T235530-6c7961fb/states.jsonl`, one correction was
issued at 5.876072325 s and stopped at 11.879992280 s with
`physical_step_not_confirmed` after 6.004 s. Requested encoder/FK vertical
displacement was −1.334 mm; achieved displacement was −1.613 mm, in the
requested direction. Achieved horizontal displacement was `[+0.199, +0.238] mm`.
This was a settling-confirmation failure, not absence of motion.

Target encoder counts were `[-1296, 2459, 730, 345, 758, 824]`; observed counts
at the stop were `[-1295, 2459, 729, 345, 756, 822]`. Joints 5 and 6 remained
two counts from target, exceeding the configured one-count settling tolerance.
Over the final three seconds, per-joint count ranges were `[1, 0, 1, 1, 1, 0]`.
The last count change occurred at 11.826380285 s, only 54 ms before timeout,
also preventing the required 150 ms exact-count stability interval.

The measured endpoint differed from the requested endpoint by
`[+0.231, +0.358, -0.278] mm`, with rotation difference 0.00260 rad. These
Cartesian measurements are encoder/FK estimates. They demonstrate movement
and explain the settling predicates; they do not establish visual convergence
or independently measured physical alignment.
