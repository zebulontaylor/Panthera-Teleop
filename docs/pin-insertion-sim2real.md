# Pin alignment: real adaptation and contact-stage audit

**Scene correction, 2026-11-04:** the synthetic scenes described below used the
legacy opposed-base XML, bypassing the existing photo setup. They are historical
experiments, not a faithful reconstruction of the bench. See
[the corrected photo setup and new recording audit](pin-insertion-photo-setup.md).
The reported real-image metrics remain historical measurements; they do not
validate that legacy simulator or establish insertion success.

This continues the [initial sim-only experiment](pin-insertion-cnn.md). The left
arm moves the pin, the right arm holds the carrier, and inference uses the full
right wrist image plus native measured state14. All work is offline.

The important operating condition is **the pin resting on the carrier outside
the bore**, with a visible bore or a bore partly obscured by the pin. Approach
localization is a preliminary test, not evidence that this condition works.
Physical contact, failed alignment, insertion, and occlusion must remain
separate labels. Image proximity alone does not establish contact.

## Real adaptation completed

The frozen `work/pin-insertion-real/manifest.json` contains 102 independently
visually annotated frames from 34 episodes, split by saved grasp/anchor groups:

| Split | Episodes | Frames | Valid pin labels | Valid hole labels |
|---|---:|---:|---:|---:|
| Train | 20 | 60 | 40 | 40 |
| Validation | 4 | 12 | 8 | 8 |
| Test | 10 | 30 | 8 | 26 |

Exact saved anchor hashes are a grouping proxy, not verified acquisition-session
identities. Earlier evaluation episodes are excluded. Labels are AI visual
annotations with explicit uncertainty, not human-certified metrology. Most
usable pin labels show separated approach views. Six test episodes hold gears,
not bare pins; their hole and negative-gate labels are retained but must not be
counted as pin-contact tests. No recorded positive activation labels were verified.

`adapted.pt` was trained for 120 epochs, with real photometric/affine augmentation
and replay of the original synthetic training split. Epoch 116 was selected
using only validation labels. Camera/grasp projections were **not** substituted
for independent real labels. Test pixels and labels were excluded from fitting
and model selection.

| Separate real test, original 640×480 pixels | Pin mean error (8) | Hole mean error (26) |
|---|---:|---:|
| Original sim-only CNN | 71.60 | 58.55 |
| Real-adapted CNN | 6.65 | 11.32 |
| Constant mean of training labels | 36.00 | 7.90 |
| Adapted CNN, shuffled images | 52.17 | 12.27 |

The pin result supports improved visual localization. The hole model still loses
to the constant-position baseline in this limited-motion test. At gate threshold
0.8, false activations fall from 3/26 known negatives to 0/26; positive recall
is unknown, and an always-off gate would pass this test. These are not insertion
success rates. See `test-evaluation.json` for medians, tails, counts, image
ablations, provenance and all predictions.

The adapted checkpoint declares `localization_only`, `direct_xyz_disabled` and
`gate_unverified`. `BoundedPinCorrection` refuses its XYZ output even when an
adapter caller requests enabled operation. No motor command was executed.

## Camera and grasp calibration

`table-camera.json` fits the right wrist mount, intrinsics and radial distortion
using stationary wood features and recorded FK. Visually audited table-only
regions exclude wall/arm correspondences. An independent full demo gives
2.79 px mean transfer error across 624 correspondences; combined held-out pairs
give 2.89 px across 812. This is conditional on the table-plane/FK assumptions,
not independent physical camera metrology. Nominal mount uncertainty of several
millimetres is optimistic because feature errors are correlated.

Twenty additional held-pin labels span fuller wrist motions: 14 training and six
validation frames. `calibration-expanded.json` holds the table camera fixed and
fits relative bases plus shared/per-grasp pin offsets. Selection compares all
candidates on the same validation observations:

| Expanded fit | Pin reprojection mean |
|---|---:|
| Train, 54 observations | 9.42 px |
| Validation, 14 observations | 18.74 px |
| Separate test, 8 observations | 29.72 px |

The wider motion improves base/grasp identifiability approximately 6–18×, but
linearized uncertainty remains roughly 15–47 mm. The recorded image/state timing
proxy can differ by 50 ms. Carrier symmetry, unknown cylinder axis, joint-model
error and grip compliance remain unresolved. **This fit is not insertion-grade
metric calibration.** The pin-axis direction opposite the fitted tip offset is
an explicit geometric prior, not an observed orientation measurement.

A subsequent CAD feasibility check **rejects the fitted grasp**: its pin tip is
52.4 mm from the modeled jaw-pad envelope, exceeding the entire 17.88 mm pin
length. No axis choice makes that offset an accurately held pin in this model.
Good image reprojection alone did not resolve the coupled base/grasp ambiguity.
`contact-geometry-audit.json` retains this result and the failed render examples.

`calibrated-hypothesis-sim.mp4` replays actual held-out joint states with this
candidate geometry. It checks model FK agreement, but performs no contact
dynamics and establishes no physical success. It exposes localization failures
in the reconstructed appearance. The original `adapted-sim.mp4` instead shows
validation snapshots from the original randomized simulation; these are also
not closed-loop insertion trials.

## Contact-stage evaluation

The dense contact audit is stored separately in `contact-labels-a/b/c.json`;
the original manifest and checkpoint remain immutable. It distinguishes full,
crescent and hidden bore views, pin versus jaw occlusion, temporal amodal hole
estimates, uncertain contact and object identity. A shiny surface inside the
empty bore can resemble the pin end; before/after views must rule this out.
Invisible distal endpoints are not replaced with gripper tips or proximal caps.

Successful insertion/release clips do not establish failed outside-hole contact.
The archived October 27 failed attempts are reserved as an additional whole-
episode challenge set, never a training or calibration source. Three independently
reviewed frames from one failed attempt show the pin beside the intended bore;
physical contact is likely from the sequence, not force-confirmed.

Generic partial-hole augmentation preserves independently known amodal centers,
protects the visible pin endpoint and masks gate supervision. It is an occlusion
robustness experiment, not a physical contact simulation or a source of positive
real activation labels.

The merged `contact-manifest.json` preserves the original split identities:
74 train frames, 24 validation frames and 60 test frames. Of the 40 audited test
end frames, 16 show bare pins and 24 show gears. Only eight bare-pin endpoints
survive independent review, and none of their hole centers is reliably visible.
The 24 gear-held hole labels are reported separately, never as bare-pin contact
performance. Train additions provide 29 amodal hole labels but no reliable new
distal pin endpoints. Reflections and uncertain endpoints were explicitly rejected.

`contact-real.pt` retrains from the original sim checkpoint for 120 epochs with
real affine/photometric augmentation, original simulation replay and partial-hole
occlusion (3,366 applied augmentations). Epoch 18 wins the predeclared balanced
contact-validation criterion: 24.79 px pin / 2.37 px hole, using eight/four labels.
The earlier `adapted.pt` scores 21.51 / 15.08 on those same validation labels.
Test results do **not** support promotion:

| Held-out condition | Earlier adapted pin / hole | Contact candidate pin / hole |
|---|---:|---:|
| Bare-pin near-contact, 8 pin labels; hole hidden | 34.54 / unknown px | 43.91 / unknown px |
| Failed off-hole challenge, 3 paired labels in one trial | 34.03 / 11.98 px | 38.62 / 10.85 px |

Challenge image-vector direction is also unreliable (contact candidate mean
cosine −0.10). This is an image-plane diagnostic, not a robot-frame nudge score.
No real positive gate labels or physical insertion success were established.
`contact-test-evaluation.json` preserves the failures and object-specific counts.

The separate contact simulator generates 40 CAD-supported diagnostic cases,
eight each for outside/rim contact, aligned bore, hover and distant positions.
It samples the actual mesh under a tilted pin cap, solves hypothetical arm IK,
enforces joint limits and rejects large inter-arm overlaps. A physically plausible
fixed CAD grasp replaces the rejected fitted offset. Resulting joints are
**simulation IK states**, not unchanged measured joints. The renders use real
training-source seeds and are not an independent session test. There is no
contact dynamics or success trial. Their `diagnostic_only` metadata prevents
accidental training replay; none was used to train the reported checkpoints.

## Retaining an early hole observation

`app/pin_insertion_memory.py` implements the right-wrist stationary-carrier
assumption explicitly. It takes a robust median of designated early visible CNN
hole observations, then retains that location while updating the pin each frame.
It resets on declared episode, grasp, camera, sequence or image geometry changes,
invalid timestamps, expired retention or conflicting trusted observations. It
does not infer contact, issue motion, or turn the gate on. Seed dispersion is not
treated as calibrated localization uncertainty.

The offline sequence protocol manually identifies a clear initialization window;
it supplies **no labeled hole coordinates** to the cache. Automatic selection of
a trustworthy window remains unvalidated. The failed-attempt test uses frames
330–350 for initialization and evaluates independent labels at 370/379/388.
Raw receipt-time gaps and the longer offline retention horizon are explicitly
recorded, not silently replaced by compressed video time.

| Retention evaluation | Fresh CNN hole mean | Early CNN hole retained |
|---|---:|---:|
| Failed attempt, 3 correlated labels | 10.85 px | 23.96 px |
| Four Oct31 held-out pin clips, 4 visible post-init labels | 8.06 px | 8.55 px |

The initial detector is wrong in the challenge, so memory preserves the error.
Retention is implemented but these experiments show **no accuracy improvement**.
The next blocker is a reliable early hole lock plus usable contact-stage pin
supervision, rather than repeatedly guessing a completely hidden hole.

Videos in `work/pin-insertion-real/`:

- `videos/early-hole-memory-real.mp4`: actual failed attempt; current pin heatmap
  and explicitly cached initial hole heatmap, with independent labels where available.
- `videos/failed-contact-real.mp4`: earlier adapted model on the same failure,
  without memory; source timestamps and holds are disclosed.
- `contact-diagnostic-heatmaps.mp4`: CAD contact hypotheses with current CNN
  heatmaps, clearly marked diagnostic/no dynamics/training-source seeds.
- `videos/adapted-real.mp4` and `videos/adapted-sim.mp4`: broader real test and
  original synthetic validation snapshots from the earlier adaptation.

## Geometry erratum

17.9625 mm is the radius of the **circle of bore centers**, not an individual
bore radius. The CAD pin is 8 mm diameter; the intended individual bore is about
8.34 mm diameter. This makes a partially covered crescent plausible. Earlier
frozen artifact descriptions mistakenly called the pitch radius a bore radius;
their hashes are preserved, with this correction recorded separately. Target
labels refer to the four outer pin bores, not the smaller inner hardware holes.

## Reproduction and artifacts

Use `work/pin-insertion-cnn/runtime/bin/python`, or install
`requirements-pin-insertion.txt` in a compatible Python environment. Large
artifacts remain under ignored `work/pin-insertion-real/`.

```bash
python scripts/adapt_pin_insertion_cnn.py \
  --checkpoint work/pin-insertion-cnn/heatmap.pt \
  --manifest work/pin-insertion-real/manifest.json \
  --output work/pin-insertion-real/adapted.pt \
  --device cuda --epochs 120 --image-size 160 --batch-size 16 --affine \
  --sim-dataset work/pin-insertion-cnn/sim_dataset_v6.npz --sim-weight .25

python scripts/calibrate_pin_insertion_geometry.py \
  --labels work/pin-insertion-real/geometry-manifest.json \
  --prior work/pin-insertion-real/table-camera.json --fix-camera --starts 4 \
  --output work/pin-insertion-real/calibration-expanded.json

python scripts/prepare_pin_insertion_contact_manifest.py \
  --base work/pin-insertion-real/manifest.json \
  --audits work/pin-insertion-real/contact-labels-a.json \
    work/pin-insertion-real/contact-labels-b.json \
    work/pin-insertion-real/contact-labels-c.json \
  --output work/pin-insertion-real/contact-manifest.json

python scripts/adapt_pin_insertion_cnn.py \
  --checkpoint work/pin-insertion-cnn/heatmap.pt \
  --manifest work/pin-insertion-real/contact-manifest.json \
  --output work/pin-insertion-real/contact-real.pt \
  --device cuda --epochs 120 --image-size 160 --batch-size 16 --affine \
  --sim-dataset work/pin-insertion-cnn/sim_dataset_v6.npz --sim-weight .25 \
  --hole-occlusion-probability .4 --select-contact-only

python scripts/evaluate_pin_insertion_contact.py \
  --manifest work/pin-insertion-real/contact-manifest.json \
  --challenge work/pin-insertion-real/challenge-reviewed.json \
  --checkpoints work/pin-insertion-real/adapted.pt work/pin-insertion-real/contact-real.pt \
  --output work/pin-insertion-real/contact-test-evaluation.json

python scripts/evaluate_pin_hole_memory.py \
  --checkpoint work/pin-insertion-real/contact-real.pt \
  --labels work/pin-insertion-real/challenge-reviewed.json \
  --episode data/hitl/review/20261027T155906-04975fdb \
  --output work/pin-insertion-real/videos/early-hole-memory-real.mp4
```

The selected model, manifest hashes, run arguments, split identities, training
curves, calibration assumptions and render metadata are retained in their JSON
sidecars. Videos display actual spatial CNN heatmaps, normalized independently
per landmark for visibility; the overlay intensity is not calibrated confidence.

[Compact results and artifact hashes](pin-insertion-sim2real-results.json) retain
both improvements and failed contact/memory tests. Final verification passed
85 focused tests and 11 additional subtests, including split exclusion, rejected
diagnostic training data, hidden-label masking, contact support geometry, temporal
memory resets and motion-disabled checkpoint behavior. `git diff --check` passed.
