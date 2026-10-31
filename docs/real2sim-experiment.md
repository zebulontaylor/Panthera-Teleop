# Real demonstration mapping experiment — October 23, 2026

The arm motion maps successfully into MuJoCo. Six real kept demonstrations
produced 2,237 observation/action pairs at 20 Hz (111.85 seconds). Correcting the
table height reduced pooled arm tracking RMS error from **0.07009 rad to
0.003798 rad** (about 18.5× lower). This is a useful foundation for simulation
evaluation, but the current scene cannot yet measure the learned policy's
assembly success.

## What was tested

The experiment reads saved files only. It does not connect to motors, cameras,
the live recorder, or the inference service. Original recordings and the sibling
simulation project's assets are preserved.

The recorded six joint coordinates, axes, limits and tool offset match the
simulator: joint order/sign mapping is identity. The right arm's simulated
world pose is transformed back into its own base frame for comparison with the
recorded tool pose. All mapped tool positions match the recorded URDF-derived
positions within 4e-16 metres; rotation error is below 5e-8 rad. This checks
coordinate/model compatibility, **not independently measured robot accuracy**.
None of these frames exceeds a simulated arm joint limit. Maximum nearest-state
sampling skew is 4.05 ms.

States retain native 14D radians: left joints 1–6, left gripper, right joints
1–6, right gripper. Each action is the next measured grid pose at t+50 ms,
consistent with the real training export. It is a kinesthetic target proxy,
not a recorded position-control command. The last observation is omitted when
its next pose would leave the recording. Actions and states are sampled without
interpolation; exact selected measurement timestamps and skews are retained.

The dynamic tests initialize at the first measured pose and execute the entire
sequence through MuJoCo's position servos with 2 ms physics steps and interpolated
targets. Each replay was allowed up to 60 seconds; all six finished in full with
no rejected arm targets. Reported RMS pools all 12 arm joints and all executed
frames. A second replay disables contacts to isolate controller tracking.
This is **oracle demonstration tracking**, not learned policy inference.

## Table registration ablation

The synthetic scene has tabletop z=+0.05 m relative to the arm bases. In the real
images the bases appear mounted directly on the tabletop. Testing tabletop z=0
eliminates the artificial intersections across all six demonstrations. This is
a supported registration hypothesis; tabletop height was not independently
surveyed.

Only the table-top collision/rendering geom is lowered in memory. Object, fixture,
base and camera poses remain synthetic. No source XML is rewritten.

| Episode | Pairs | Default table RMS, rad | Table at base RMS, rad | Default intersecting frames |
|---|---:|---:|---:|---:|
| 20261022T222502-6321638f | 396 | 0.04872 | 0.003773 | 88.4% |
| 20261022T223034-fa96590a | 416 | 0.06923 | 0.003566 | 88.5% |
| 20261022T224407-e6499944 | 465 | 0.09195 | 0.003642 | 92.0% |
| 20261023T180718-47e74c70 | 300 | 0.06350 | 0.004292 | 84.3% |
| 20261023T181449-320a66bd | 315 | 0.06948 | 0.003955 | 84.4% |
| 20261023T182037-65ab8000 | 345 | 0.06354 | 0.003698 | 85.5% |
| **Pooled** | **2,237** | **0.07009** | **0.003798** | **87.6%** |

With the table at base height, there are **zero** arm/table intersecting frames;
the worst original penetration was 44.33 mm. Corrected contact-enabled tracking
matches the contact-disabled baseline for these trajectories. That also means
these tests do not establish grasp/contact fidelity: the task objects have not
been registered to the real demonstrations. Maximum corrected absolute arm
error is 0.04009 rad; low pooled RMS does not mean every joint tracks exactly.

## What remains before policy evaluation is meaningful

1. **Register the two bases.** The synthetic right base is at [.72,.16,0], yaw
   pi. Mapped tool centres remain at least 224–261 mm apart across these clips,
   while the real videos show both grippers working on the same assembly.
   Estimate the relative base transform from shared object/robot landmarks and
   check separate frames. Joint sign changes would not fix this world alignment.
2. **Register cameras and image preprocessing.** The real wrist USB-to-arm
   assignments are unverified. Comparison columns keep USB names and explicit
   synthetic arm names; placing them alongside one another does not establish
   correspondence. Current videos show full views at preview resolution, not
   the policy's three 256×256 training crops. Intrinsics/distortion, wrist
   transforms, overhead pose, crops and appearance need validation.
3. **Recover object poses and grasp state.** Kinematic videos replay only arm
   motion; parts remain at the synthetic initial poses. Dynamic parts are free
   to move under physics, but are initialized at synthetic stations. Neither
   reconstructs real carrier/gear/pin trajectories or real grasp geometry.
4. **Measure jaw opening.** The bridge explicitly assumes 0–2 motor radians
   maps linearly to 0–40 mm per finger (80 mm total). Actual linkage and zero
   need calibration. Raw mapped states/actions preserve out-of-range encoder
   values for diagnostics. Dynamic targets alone saturate grippers to [0,2],
   counting each changed channel; arm targets are never silently clipped.

The next useful experiment is to register bases and initial part poses, then
require a real demonstration replay to reproduce its grasp and assembly
contacts. Once that works, connect the existing policy through the native-radian
bridge and evaluate closed-loop success on separate initial scenes. Current
artifacts are motion-reconstruction diagnostics, not a task-valid synthetic
training dataset or an estimate of real-world policy success.

## Saved artifacts and implementation

- Compact measured comparison: [real2sim-results.json](real2sim-results.json).
- Default scene run: `data/experiments/real2sim-default-table/`.
- Corrected table run: `data/experiments/real2sim-table-at-base/`.
- Each episode contains `mapped.npz`, `report.json`, `source_metadata.json`.
  The corrected run also contains `comparison.mp4` and `comparison.jpg` with
  three real views above three kinematically rendered views. Videos sample
  playback at 5 fps; source state/action pairs remain 20 Hz.
- NPZ preserves native and simulated state/action arrays, base-local TCPs,
  original timestamps/skews, and both dynamic trajectories with prediction
  timestamps. Sim grippers use **per-finger metres**, native grippers radians.
- Summary files include source recording/scene/tool hashes, dependency versions,
  explicit assumptions and `model_evaluated=false`, `task_success_evaluated=false`.
- `scripts/real2sim_data.py`: saved-demo adapter.
- `scripts/real2sim_bridge.py`: reversible coordinate conversion and MuJoCo runtime.
- `scripts/experiment_real2sim.py`: experiment and visual comparisons.
- `scripts/summarize_real2sim.py`: array-based aggregation across runs.

## Reproduce

Use an isolated Python 3.10+ environment; no motor SDK is required. The checked-in
scene and meshes from the sibling `RobotArmTraining` checkout must be available.
Use `--scene /path/to/scene_bimanual.xml` to override its default location.

```bash
python3 -m venv work/real2sim-venv
work/real2sim-venv/bin/python -m pip install -r requirements-real2sim.txt
work/real2sim-venv/bin/python -m unittest discover -s tests -p 'test_real2sim*.py' -v

MUJOCO_GL=egl work/real2sim-venv/bin/python scripts/experiment_real2sim.py \
  --episode 20261022T222502-6321638f \
  --episode 20261022T223034-fa96590a \
  --episode 20261022T224407-e6499944 \
  --episode 20261023T180718-47e74c70 \
  --episode 20261023T181449-320a66bd \
  --episode 20261023T182037-65ab8000 \
  --dynamic-seconds 60 --table-offset-m 0.05 --render \
  --output data/experiments/real2sim-table-at-base-new
```

Positive `--table-offset-m` lowers the table. Omit it for the default scene.
Outputs must be fresh directories. `--dynamic-seconds 0` performs kinematic
mapping only. The tests cover native/next-state pairing, timestamp ties, invalid
inputs, base-frame FK, actual integration, exact target sequencing, flag
restoration and short recordings. **13 tests passed** in the experiment runtime.
