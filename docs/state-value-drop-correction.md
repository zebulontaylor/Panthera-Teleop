# State-value correction — 2026-10-29

**Not fixed reliably; candidate rejected for deployment.** The original inference teacher and action critic remain unchanged. No robot commands were issued.

The deployed value head learned a weak discounted recording-end target. Discarded/HITL frames fed through that teacher do not become negative examples. This correction uses reviewed physical before/after preferences, never intervention timing as a failure label.

## What ran

- 23 older reviewed comparisons:10 gear releases across4 episodes,6 carrier losses across2 episodes,7 pin regressions. Repeated pairs are correlated.
- Initial head, fusion, reset-head fusion, and final-backbone adaptation failed the original four held-out pin-readiness comparisons (at best1/4). Those readiness comparisons differ from pin-placement regressions.
- The confirmed user HITL drop was then explicitly moved from a reserved regression case into training:2 comparisons from one episode. This was disclosed before training and before any test evaluation.
- Two gear and two pin regressions from Oct5/11 were reverified and reused for development validation. They were previously evaluated with the old model and are not pristine holdouts. Entire sessions were excluded from normal anchors.
- Targeted head correction won the validation-only selection at epoch173, score0.02593 versus0.17081 for fusion/reset-head. Normal-demo teacher anchors constrained score drift, but not ranking.

## Results and rejection

On the now-trained user clip, value changed from **0.648 →0.673** before/after release to **0.861 →0.197**. This is training performance, not evidence of generalization.

The two separate development gear regressions changed to **0.109 →0.0009** and **0.387 →0.033**. All4 development comparisons ranked correctly.

After selection,10 reserved Oct28 normal-progress comparisons fell from **7/10 correct to3/10**. Mean signed margin changed from+0.0195 to−0.0606. This fails deployment. Small validation pointwise anchor MSE(0.0030) did not prevent damaging normal-progress ordering.

No later candidate was selected or tuned against that test result. The failing candidate remains versioned under `work/state-value-dropfix-20261029/targeted-head/best.pt`; do not copy it over the active bundle or use it for RL. Q retargeting was prepared but not run.

## Reproduction

Training code: `scripts/state_value_correct.py`, `scripts/state_value_correct_representation.py`, `scripts/state_value_correct_backbone.py`. Seven tests cover preference gradients, split/image leakage guards, and frozen-layer behavior. Each run's `report.json` records arguments, source hashes, per-pair values and validation history.

Primary artifacts under `work/state-value-dropfix-20261029/`: `targeted-protocol.json`, `selection.json`, `deployment-decision.json`, `original-normal-test.json`, `corrected-normal-test.json`, reviewed manifests and visual audit files.

Further correction needs balanced, reviewed physical-progress and drop/recovery supervision from the same modern HITL scenes. Simply fitting negative pairs more strongly while distilling old scores was insufficient. Future validation must test progress ordering and regressions together; these exposed test cases are now regression checks, not a fresh benchmark.
