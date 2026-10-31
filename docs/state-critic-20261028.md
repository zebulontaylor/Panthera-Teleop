# Offline state critic experiment — October 28, 2026

## Status

October 29 follow-up: the overnight heartbeat was removed after the user requested
a new Vast RTX PRO 6000 rental and latest-policy inference. The original experiment
below is preserved as a diagnostic record. Current deployment and probe results:
`work/state-critic-20261029/vast/README.md`. The additional public discarded-data
cohort and corrected visual labels are under
`work/state-critic-20261029/hf-new-discarded/dataset/`. The new narrowly supervised
terminal pin-placement experiment also failed to establish a trustworthy critic;
do not interpret either experiment as a validated whole-task value model.

Work is in progress. No critic has been deployed to the robot. The user authorized
autonomous labeling and AWS training, with an overnight result requested for
October 29. An hourly heartbeat named `finish-and-evaluate-offline-robot-critic`
continues this chat until 08:00 America/Chicago October 29.

The AWS host `admin@54.166.156.98` is currently unreachable on port 22. Repeated
read-only SSH attempts timed out; the existing inference tunnel is also stuck
connecting. Do not stop or replace existing AWS services. The local RTX 3070 has
8 GiB total memory and was approximately 7 GiB free at initial inspection.
An isolated local environment and pretrained training/prediction smoke test work.

The first **curation-proxy** run has completed locally:
`work/state-critic-20261028/runs/curation-proxy-seed19/`.
It includes `critic.pt`, `metrics.json`, `predictions.jsonl`, a frozen training
manifest, provenance, and the log. The shared frozen-feature cache is
`work/state-critic-20261028/features.npz`.

Held-out frame AUROC is 0.995 for the visual/state model and **0.994 for the
state-only baseline**; Brier scores are 0.025 and 0.006 respectively. Time-only
AUROC is 0.757. These metrics concern kept/discarded curation decisions only.
The very similar state-only result and observed stationary/reset recordings
mean this is NOT evidence of visual assembly-success recognition. No trustworthy
task-success or action-ranking critic has yet been validated.

Four new dataset/cache/metric tests pass. Checkpoint inference on 64 held-out
observations reproduces saved predictions to maximum absolute difference
3.28e-7. The run includes a frozen trainer, bundled encoder cache, version/hashes
manifest, and `model-card.md` with offline inference instructions.

All 366 overview sheets have now received a first-pass visual review. The
conservative merged snapshot currently has 360 unknown outcomes, five incomplete
outcomes, and one medium-confidence apparent success. The sole positive is in
the test partition: do not move it into training to manufacture class coverage.
Some negative labels remain provisional as well. Targeted higher-resolution
checks are continuing; the initial dense six-episode re-review still could not
resolve seating. No success-model training is justified by this label snapshot.

### Overnight diagnostic update (first follow-up)

AWS SSH still timed out at the October 28 evening follow-up. A new fixed-C=1
logistic diagnostic used the frozen proxy manifest, with preprocessing fit on
training rows only. Mean RGB from each camera (only nine features) reaches
test frame AUROC **0.9524** and episode AUROC **0.9829**; current ResNet visual
embeddings reach **0.9522 / 0.9829**. This supports caution about the proxy's
environment/recording correlations, though it does not prove a causal shortcut.

The 57 labeled test episodes represent only **four recording blocks**: one
positive at 00:00, six positives/one negative at 18:00, 29 negatives at 22:00,
and 20 negatives at 22:30 UTC. The original episode-bootstrap intervals do not
account for correlation within these blocks. Treat them as descriptive and do
not interpret the narrow intervals as strong evidence of generalization.

Reproduction: run `scripts/state_critic_shortcuts.py` with the critic venv Python.
Results and limitations are saved under
`work/state-critic-20261028/diagnostics/shortcut-audit/` (`metrics.json`,
`findings.md`). No original model weights, labels, or dataset splits changed.

### Label reliability follow-up

An independent follow-up inspected the six provisional negative labels using
contact sheets and dense endpoint frames. Five retained support as apparent
incomplete attempts. Episode `20261028T195448-e0452552` was downgraded to unknown:
it is a short stationary/reset-like recording without a clear task attempt.
The original reviewer file is archived, and the correction cites
`work/state-critic-20261028/negative-label-audit.json`. This does not add positive
training labels, so success-model training remains unjustified.

Annotation file integrity was also checked: all 366 frozen episode IDs occur
exactly once, with no missing referenced images. These checks establish file
coverage, not label correctness. The AWS retry again timed out. No training or
deployed services were started or modified during this follow-up.

## Data and annotation

The frozen inventory is `work/state-critic-20261028/episodes.json`: 366 complete
manual episodes (236 kept, 106 discarded, 24 review). New recordings appeared
during inventory construction; do not silently change this experimental cohort.
The raw data must remain untouched. Folder categories are curation decisions,
not verified task outcomes.

The task, from the existing postprocessing configuration, is **put the gear in
the carrier and put a pin in it**. The large circular silver fixture is the
carrier. The target gear fits a peripheral pocket, not the central bearing.
There are multiple spare gears and pins on the table. A loose spare is not
evidence of failure. End-of-recording gripper reset/release is also not failure.

Four Luna reviewer agents own shards 1 through 4, respectively. Reviewer files
are under the work directory and/or its `labels` subdirectory. Early provisional
labels incorrectly treated spare parts as failures; reviewers were instructed to
revise them. Do not use uncorrected early snapshots for supervised training.
Uninspected examples must have pending review status and no claimed image
evidence. Consolidate duplicate records conservatively and retain provenance.

Labels describe **apparent visual completion**, not force-verified mechanical
seating: 1 if the observed gear placement and pin insertion/release support
completion, 0 for a clearly failed/incomplete attempt, -1 when unknown.
Use original frames around insertion when the seven-row overview misses it.
`calibration-detail.jpg` in the work directory provides an apparent-success
reference from `20261028T181240-48890010`: pin cap retained in the upper-left
carrier pocket after gripper release around 16.5–17 seconds, a further press,
then carrier release. It is a medium-confidence visual inference, not physical
verification. Sparse contact sheets often cannot resolve the critical detail.

The initial whole-date split was rejected **before training**: every discarded
episode is from October 28, leaving no negative curation examples in training.
The revised pre-training split holds out whole episodes and half-hour UTC blocks
on October 28:

- Train: October 22/23 plus October 28 blocks 19:00 and 21:30 (202 episodes;
  153 kept, 29 discarded, 20 review).
- Validation: October 25/27 plus October 28 blocks 17:30, 19:30, 20:00
  (107 episodes; 76 kept, 27 discarded, 4 review).
- Test: October 28 blocks 00:00, 18:00, 22:00, 22:30
  (57 episodes; 7 kept, 50 discarded).

These blocks are NOT independent recording days or proven independent sessions.
Time/category confounding and test imbalance substantially limit the evaluation.
The initial export has 8,590 observations: 4,071 train, 2,451 validation, 2,068 test.
Evaluate actual reviewed outcome coverage before interpreting any metric. Manual demonstrations
support value under demonstrated behavior, not calibrated autonomous value.
HITL data is audited separately and excluded from this first training cohort
because human assistance and truncated recording semantics need separate handling.

## Reproduction

Preparation: `python3 scripts/state_critic_prepare.py` builds overview sheets and
the frozen inventory. Do not rerun inventory unnecessarily.

Dataset: `python3 scripts/state_critic_dataset.py` exports aligned 1 Hz observations,
three full-frame camera views, one-second image history, and measured 14D joint/
gripper state. Check its audit and trainer schema before launching. Camera order
is overhead, wrist_port2, wrist_port3. Paths resolve relative to the dataset root.
Unknown outcomes remain -1. Curation proxies live in a separate diagnostic label
manifest and must never be represented as independently verified outcomes.

Environment: `work/state-critic-20261028/venv/bin/python`. It exposes existing
PyTorch through a .pth file and installs torchvision/sklearn into this isolated
environment. Existing robot/training environments are not modified.

Trainer: `scripts/state_critic_train.py` extracts frozen pretrained ResNet18
features, concatenates current/history/difference features and robot state, and
fits outcome and discounted terminal-outcome heads. Feature extraction is batched;
use `--feature-batch-rows 8` for the local GPU, and a shared `.npz` feature cache.
Training labels and splits are not features. Remaining time is only used for the
Monte Carlo target and MUST NOT be added as a model input (future information).

The curation-proxy run is a diagnostic: it predicts kept/discarded decisions,
not task success. Preserve this distinction in checkpoint names and reports.
Training on reviewed labels requires supported positive AND negative training
examples; a one-class dataset is not evidence of a useful success critic.

AWS helper: `scripts/state_critic_aws.sh` prints its plan by default. Review its
arguments and use `--launch` only after connectivity/resources are verified.
It uses a dedicated `/mnt/pi05/state-critic-20261028` directory, copies only the
compact dataset/trainer, and does not touch inference services or robot hardware.

## Remaining work

1. Finish targeted full-resolution re-review; audit label
   provenance and remove provisional false-failure labels.
2. Dataset export, schema validation, and the first frozen run are complete. Use
   `scripts/state_critic_dataset.py --relabel-only` after label updates.
3. The curation-proxy diagnostic is complete. Train a reviewed-outcome baseline
   only if supported labels cover both classes. Do not select hyperparameters on
   held-out test results or reinterpret unknown labels as successes/failures.
4. Evaluate episode-grouped uncertainty, terminal/early/late performance, and
   state-only/time-only shortcuts. Save exact checkpoints, predictions, and failure
   cases. Assess that these are behavior-outcome models, not tested Q critics.
5. Retry AWS access during the overnight heartbeat. Continue locally if unavailable.
6. Deliver an honest morning report: provide usable artifacts, and explicitly say
   if insufficient visual labels or held-out performance prevent a trustworthy
   critic claim. Disable the heartbeat after that final report.
