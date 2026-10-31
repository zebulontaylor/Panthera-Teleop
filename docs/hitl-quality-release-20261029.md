# HITL correction quality release — 2026-10-29

Reviewed all 89 review-marked rollouts: 60 Zeb and 29 Lisa. Visual subagents reviewed correction behavior, with dedicated Lisa review and additional examination of short nudges. Accepted corrections show acquisition, reduced part/fixture separation, or improved local alignment. Full insertion success is not required or asserted.

Final export: **91 subclips, 4,606 action rows at 20 Hz, 230.3 seconds, 47 original rollouts**. Operator contribution: 70 Zeb clips and 21 Lisa clips. 63 clips are shorter than 2.5 seconds; the shortest is 0.35 seconds. The 97 approved visual spans were independently filtered and split by sensor and motion integrity.

## Cleaning and verification

- Preserve every raw recording. Each clip stays within one human-controlled recording segment. The opposite held arm is retained because it may stabilize the fixture.
- Trim quiet edges and split extended inactivity without concatenating across omitted source time. Multiscale motion detection retains slow nudges; no model-horizon minimum duration is imposed.
- Preserve measured observations and grippers. Smooth only joint action targets with a three-tap filter, bounded to 0.008 rad from the raw next measured pose. Retain raw targets for comparison. Actual maximum deviation is 0.006911 rad.
- Frame-weighted joint acceleration RMS decreased 36.1%, from 0.5890 to 0.3765 rad/s². Retained trajectories meet the declared offline 1 rad/s joint-speed and 8 rad/s² acceleration gates.
- Require finite positions, zero motor fault flags, monotonically ordered timestamps and feedback age no greater than 50 ms. Audited clean intervals can precede a later fault in an incomplete rollout; eight exact ranges have explicit integrity provenance.
- Use original camera images with recorded alignment offsets bounded to 50 ms; never fabricate intermediate images. The bound accommodates measured ~100 ms saved-camera gaps. Normalize raw images to the current inference crop contract, preserving source crop provenance.
- Independent verification compares every exported image against its source transform, raw observations/actions against measured states, derived smoothing against its formula, numeric statistics against stored rows, and all true action horizons against native LeRobot loading. Adversarial checks reject corrupted statistics and camera provenance.

## Training contract

**Use the bundled mask-aware dataset wrapper and PI05 loss adapter.** Short corrections retain their true length. Mask invalid future positions before loss reduction; the old PI05 scalar loss cannot be fixed afterward. The adapter is opt-in and does not modify installed software, training jobs, or deployment. Unit/adversarial tests establish zero direct loss gradient from padded targets. The optional full-horizon sampler has 1,480 starts and excludes short nudges; it is a compatibility option, not the main dataset selection rule.

Do not stitch policy continuations onto human clips: the inspected handoffs omit 1.743–7.668 seconds of wall time and change recording clock rates. Masking missing future targets preserves the actual boundary. Separate progressive policy episodes would need their own review.

No training or robot evaluation was run. The earlier padding/quiet-tail mechanism is a plausible contributor to the reported regression, not a proven sole cause. Visible corrective progress and better motion statistics do not establish improved model performance. Any later evaluation should group by original rollout and compare against a no-HITL control.

## Publication

Published and verified [Hugging Face release](https://huggingface.co/datasets/ZebTaylor/panthera-hitl-reviewed-clean-20261029-20hz) at commit `cc4ee5b9ada3444b932704ffaa857681914febc0`. Verified 157 remote files: 118 LFS SHA-256 blobs and 39 regular files. Native LeRobot `v3.0` tag points to this commit. No entire exported episode has a pose range at or below 0.004 rad across all coordinates.

## Artifacts

- Local dataset: `data/processed/hitl-reviewed-clean-20261029-20hz`.
- Detailed decisions and review evidence: dataset `processing/audit/` and `processing/review_evidence/`.
- Source alignment, per-clip metrics and provenance: `processing/quality_manifest.json`.
- Training integration: `processing/masked_training.py`.
- Build log and working review material: `work/hitl-quality-20261029/`.
- Publication receipt: `docs/hitl-quality-release-20261029.json` (remote verification passed).
