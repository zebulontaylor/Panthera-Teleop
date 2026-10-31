# Action-conditioned future-value critic

`scripts/action_critic_train.py` learns a score for a current three-camera observation,
current joint state, proposed action chunk, and time horizon. The existing `VisualValue`
checkpoint supplies a frozen visual encoder and training targets:

```text
target = gamma_per_second ** dt * sigmoid(teacher.value(teacher.encode(future_images)))
Q(current_images, state, action_chunk, dt) ≈ target
```

This is supervised prediction of future value under the recorded actions. Future images
are used only to produce training/evaluation labels. Candidate scoring needs only the
current observation and proposed actions; it does not render or predict future images.
There is no policy maximization, reward learning, or robot control in this script.
The teacher's existing scores represent weak demonstration-relative value, not calibrated
task success. Candidate rankings outside the recorded action distribution need validation.

## Data

Provide JSONL transitions with these fields:

```json
{"episode_id":"demo-001","session_id":"session-001","split":"train","images":["now/overhead.jpg","now/wrist_port2.jpg","now/wrist_port3.jpg"],"state":[0.1,0.2],"action_chunk":[[0.15,0.25],[0.2,0.3]],"future_images":["later/overhead.jpg","later/wrist_port2.jpg","later/wrist_port3.jpg"],"dt":0.4}
```

The short vectors above illustrate the format. Use the actual controller dimensions,
joint order, units, and action convention consistently for training and inference.
`action_chunk` is a fixed-length sequence of position targets spanning `dt` seconds
from the current observation to the future observation. For normal kinesthetic demos,
these targets are the demonstrated future joint positions, matching the repository's
next-pose imitation-learning action convention. They are action proxies, not logged
motor commands. Keep camera order overhead,
wrist_port2, wrist_port3. Resolve image paths relative to `--dataset-root`.

Use whole episodes and sessions for splits. Also preserve the teacher's original
held-out split assignments: a critic validation episode should not have trained the
teacher. The transition validator cannot reconstruct the teacher's training membership
from its checkpoint. Do not join observations across episode boundaries, recording gaps,
or control handoffs. A terminal observation may be the future endpoint; do not create
transitions beyond it. This model has no separate terminal reward or terminal mask.

`scripts/action_critic_prepare.py` builds transitions from kept and discarded normal
demonstrations plus autonomous HITL windows in the existing state-value export. HITL
windows must retain policy ownership throughout all sampled frames, with no human control,
contact, holds, or segment changes. These checks use the available 5 Hz samples.
It preserves the teacher's episode/session splits. Discarded/HITL categories are provenance,
not automatically assigned negative targets: the frozen teacher scores every future frame.
The critic learns the value of demonstrated position-target
trajectories; this does not establish outcomes for arbitrary commanded motions.

## Training and inference

The local environment with PyTorch is `work/state-critic-20261028/venv/bin/python`.
The existing teacher is
`work/state-value-20261029/model-bundle/value_model.pt`. Train with:

```bash
work/state-critic-20261028/venv/bin/python scripts/action_critic_train.py train \
  --manifest /path/to/transitions.jsonl --dataset-root /path/to/images \
  --teacher-checkpoint work/state-value-20261029/model-bundle/value_model.pt \
  --output work/action-critic --epochs 30
```

The mixed dataset prepared on October 29 is
`work/action-critic-mixed-20261029/dataset/transitions.jsonl`: 25,072 train,
12,115 validation, and 12,032 untouched test transitions. Each action chunk is
five 14D future position targets over approximately one second. Its `audit.json`
records inclusion counts and rejected windows. The training run is stored under
`work/action-critic-mixed-20261029/run` with reusable frozen features in
`work/action-critic-mixed-20261029/features.pt`.

The completed 30-epoch run selected epoch 5 as `run/best.pt`. Validation MSE was
0.005088, versus 0.006748 for predicting future value with the current teacher
score (24.6% lower error). Shuffling the action chunks increased MSE to 0.012038;
substituting the mean training action gave 0.008004. These are validation-set
diagnostics, not a blind test or proof of reliable counterfactual action ranking.
The test split was not encoded or evaluated. `run/validation_report.json`,
`run/provenance.json`, and `run/train.log` retain metrics and training details.

Score candidate chunks stored as a JSON array of shape `[candidates, steps, action_dim]`:

```bash
work/state-critic-20261028/venv/bin/python scripts/action_critic_train.py predict \
  --checkpoint work/action-critic/best.pt \
  --images /path/to/overhead.jpg /path/to/wrist_port2.jpg /path/to/wrist_port3.jpg \
  --state '[0.1, 0.2]' --dt 0.4 --candidates /path/to/candidates.json
```

Replace the example state with a vector matching the training dimensions. All candidates
in a scoring call share `--dt`; the output gives scores in input order and descending
ranked indices. The default `--gamma 1` directly distills future teacher value;
use a smaller gamma to apply an additional discount per second. The teacher already
learned discounted demonstration-relative value, so choose this extra discount deliberately.

Training writes `best.pt`, `last.pt`, and `history.jsonl`. It selects the best model by
held-out validation MSE and excludes test transitions from fitting and model selection.
Use a fresh output directory for each run. Training writes a self-contained critic
checkpoint with the frozen teacher and normalization statistics. No teacher fine tuning
is performed. Normalization is fitted only on critic training transitions.

Synthetic tests verify implementation behavior and do not establish robot performance.

## Inference dashboard

The dashboard enables the local critic by default. Its separate persistent worker
uses `work/state-critic-20261028/venv/bin/python`, so the robot environment does not
need PyTorch. Launch normally with `./start-inference.sh`; options include
`--no-critic`, `--critic-device cpu`, `--critic-python`, and `--critic-checkpoint`.
The model loads lazily on the first preview or accepted prediction. On the local
RTX 3070, warm model scoring took about 8–9 ms; CPU scoring took about 105 ms.
Loading the CUDA worker takes roughly two seconds. Control never waits for a score.

Policy crops and full-frame critic images share the same camera receipts. The critic
uses the original request state and the validated, blended proposal. The first second
of nominal policy targets is interpolated to the five future poses used in training.
Proposals shorter than one second display unavailable instead of being padded.
The score describes a proposed trajectory, not necessarily the shorter executed prefix
or its speed-limited physical duration.

The Action value panel displays current teacher value, predicted future value and their
difference. The rollout chart plots current and predicted values at each proposal time;
predicted values are not measured future outcomes. It survives Stop and Finish and resets
at the next rollout. Long charts are downsampled for display; accepted score records with
sequence, rollout key, timing, and model hashes are appended to
`logs/inference-critic.jsonl`. Scores from superseded observations are discarded.

This is an informational monitor, separate from the EXPO replay/reward pipeline.
The existing EXPO implementation uses a fixed executed prefix, terminal rewards and
bootstrapped critic targets (see `docs/expo-ft.md`). This one-second demonstration-proxy
critic does not directly match that contract, particularly C=17 at 20 Hz. An EXPO-FT
integration must align action horizon and physical timing, validate outcome targets,
and explicitly choose how to initialize or train its critics; dashboard deltas are
not automatically fed back as rewards.
