# CNN-conditioned alignment Q research

This implements and locally trains a real Bellman Q-function for four phases:
gear pickup, pin pickup, gear insertion approach, and pin insertion approach.
It is an offline research critic, not a validated controller. No robot control
configuration is changed and no motion is executed.

The existing gear and pin/carrier CNNs are frozen. Their coordinates, relative
landmark offsets, uncertainty, previous outputs and temporal differences are
combined with measured joints, grippers, FK and velocity. The model can learn
sensorimotor relationships without converting image pixels to metres. That does
not make unobserved depth identifiable: changes in camera placement, occlusion,
contact and target-hole identity still require representative data and feedback.

The algorithm follows [Implicit Q-Learning](https://arxiv.org/abs/2110.06169):
twin action-conditioned Q heads, an expectile state-value head, detached Bellman
targets, and Polyak target updates. Discounting is per elapsed second. No
maximization over invented actions is used in training. This implementation
provides the critic and candidate scoring, not an advantage-weighted actor.

## What the training data actually means

The verified cohort is 17 October 22 recordings. CNN inference uses original
640×480 RGB frames, measured six-dimensional tool XYZ for the pin model, and the
audited camera order `overhead, wrist_port3, wrist_port2`. The model weights and
input schema are pinned by hash. Other recording dates have not been assumed to
share this camera mapping.

There are 3,675 cached observations and 1,350 transitions at 10 Hz:
320 gear pickup, 316 pin pickup, 367 gear insertion approach, and 347 pin
insertion approach. A gear pickup window is omitted for missing stable closure
evidence. The split is 11 training, 3 validation, and 3 test episodes, yielding
890/234/226 transitions. All share one recording day; the explicit
`--split-unit episode` option acknowledges same-session evaluation. The trainer
defaults to stricter session separation for future datasets.

Pickup milestones are the last stable downward gripper crossing below 0.7 rad,
with a prior opening above 0.9 rad. Insertion milestones are the existing
reviewed approach endpoints. A reward of 1 marks the final transition to each
milestone; other rewards are 0. These are **not verified grasp or seating
outcomes**. Gaps, incomplete endpoints, incorrect CNN contracts and irregular
action cadence are rejected. Phase histories reset, and neither episode time nor
future observations are supplied as model inputs. Future closure evidence is
used only to establish the offline label.

An action is a 0.1-second **measured pose delta proxy**, in this exact order:
six left joints, six right joints, left gripper, right gripper (radians).
These kinesthetic demonstrations do not establish the effect of commanding
those deltas. Q values estimate discounted weak milestone return, not success
probabilities. Predominantly positive demonstrations also provide little evidence
about alternative corrective actions or failure recovery.

## Files and reproduction

Use the existing CUDA runtime:

```bash
PY=/home/robotics/Desktop/RobotArmTraining/.venv-cnns-cuda/bin/python
python3 scripts/alignment_q_prepare.py
$PY scripts/alignment_q_experiment.py \
  --dataset work/alignment-q-20261030/transitions.jsonl \
  --output work/alignment-q-20261030/reproduction --epochs 3000
$PY -m pytest -q tests/test_alignment_q_features.py \
  tests/test_alignment_q_prepare.py tests/test_alignment_q_train.py \
  tests/test_alignment_q_advisor.py tests/test_alignment_vision.py
```

The preparation command consumes the completed native CNN cache. To regenerate
that cache at a new path, invoke `scripts/alignment_calibration_infer.py` with
`--fps 10 --device cuda --output PATH` and one `--episode ID` for each episode in
`config/calibration/recorded-alignment-near-trials.json`. Omit `--phase-manifest`
because full recordings are needed to find pickup milestones. Then supply that
new path to `alignment_q_prepare.py --predictions PATH`.

Training runs on CPU with two threads and small 64-unit heads; only extraction
uses the local 8 GB RTX 3070. No AWS resources were launched. The paired
experiment trains three preset seeds with CNN/history inputs and three with that
entire prefix removed. Every run is reported; checkpoint epochs are selected
using validation Monte Carlo return error. The constant baseline uses only
training returns. Monte Carlo and TD errors measure return fitting, not causal
action-ranking quality or physical task success.

`app/alignment_q_features.py` contains the shared, versioned feature encoder.
`scripts/alignment_q_train.py` contains training and `AlignmentQScorer`.
`app/alignment_q_advisor.py` adds the inference contract. For example, from the
repository root:

```python
from app.alignment_q_advisor import AlignmentQAdvisor

advisor = AlignmentQAdvisor(
    'work/alignment-q-20261030/extended/cnn_history.seed0.pt')
# prediction is an AlignmentVision snapshot; arms must be the measured state
# synchronized to that snapshot, not arbitrarily newer robot state.
result = advisor.score(prediction, arms, candidate_deltas, 'gear_pickup',
                       now_ns=now_ns, measured_state_ns=measured_state_ns)
```

The advisor checks checkpoint/schema identity, 0.1-second horizon, 250 ms
freshness, camera synchronization and 10 ms measured-state synchronization.
History resets on phase changes, gaps or invalid inputs. Call `reset()` at each
episode boundary. Candidates must be near an actual paired observation/action
from the same training phase; individually plausible joint ranges are not enough.
Unsupported scores are `None`, and no supported candidate means no recommendation.
This support test is a conservative heuristic, not a calibrated safety guarantee.
The advisor has no actuator API and is not connected to the live dashboard.

## Local experiment results (October 30, 2026)

Three fixed seeds per variant, 3,000 epochs per run. Lower return MSE is better.

| Model | Test return MSE, mean | Seed range | Supported test pairs |
| --- | ---: | ---: | ---: |
| CNN + history + robot state | 0.01220 | 0.00982–0.01545 | 0.0% |
| Robot state only | 0.00806 | 0.00690–0.00868 | 29.6% |
| Training phase-mean constant baseline | 0.00320 | — | — |

CNN features did not improve held-out return fitting in this experiment. Both learned
models trail the constant baseline. The CNN support gate accepts none of the held-out
recorded pairs at the unchanged radius of 3; that threshold is not a calibrated
risk bound. Do not enable robot control from these results.

An initial 600-epoch experiment was undertrained: validation was still improving
at its cap. The same preset seeds were extended to 3,000 epochs. All short and
extended reports are retained. Because test results were visible during this
research iteration, treat these comparisons as exploratory, not a pristine final
benchmark. No hyperparameter or checkpoint was chosen to optimize test scores.

Artifacts: `work/alignment-q-20261030/extended/experiment.json` (all runs),
`extended/cnn_history.seed0.pt` (fixed primary research checkpoint),
`transitions.audit.json` (data contract), and `inference-smoke.json` (real checkpoint
loading and support rejection). Paths are relative to `work/alignment-q-20261030`
unless already prefixed. All 34 tests and 20 unittest subtests passed.

## What is needed to establish alignment benefit

Collect synchronized, actually commanded small corrections with verified
pickup/retained-grasp and seated/not-seated outcomes, including failed attempts
and recoveries. Include multiple camera distances and independent sessions,
while recording camera roles and target bore identity. Those data distinguish
actions that improve alignment from states that merely resemble a successful
demonstration. A useful next experiment is a held-out-session comparison of
candidate rankings followed by bounded, operator-supervised physical trials.
The present checkpoints should not be treated as evidence that those tests pass.
