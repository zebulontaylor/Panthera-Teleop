# Gripper-anchored real2sim reconstruction

Grasp events provide metric object-location constraints directly from recorded
joint positions. Camera calibration is not a prerequisite for this approach.
For a non-slipping grasp:

`T_base_object(t) = T_base_gripper(t) @ T_gripper_object`

Estimate the fixed object-in-jaw offset from geometry and images; recorded FK
then supplies the held trajectory. The pickup pose also estimates where the
object started. Images identify the object, check slip and refine the offset.

## Applied to one demo

All three views and gripper channels were inspected for
`20261023T180718-47e74c70`. These measured tool origins are pickup anchors,
**not fitted object centres**:

| Object | Arm / frame | Pickup | Tool XYZ, metres |
|---|---|---:|---|
| Carrier | Right native base | 2.70 s | [.20211, .26721, .01658] |
| Gear | Left native base | 3.80 s | [.18630, -.25480, .01333] |
| Pin | Left native base | 10.15 s | [.24733, -.26687, .01871] |

Pickup windows: carrier 2.3–2.8 s, gear 2.9–4.0 s, probable pin 9.4–10.9 s.
Gear release is approximately 6.05–6.4 s; pin release 12.55–12.95 s. Empty jaw
reclosures were rejected as pickups. Contact sheets and detailed evidence are
under `data/experiments/real2sim-grasp/`.

The pilot reconstructs object trajectories with an **explicit starting prior**:
upright bottom-centre at table z=0, centre XY equal to pickup tool XY, yaw zero.
It derives the constant tool-to-object transform from this assumed pickup pose
and propagates it with measured FK. These centre/yaw priors have not yet been
fitted to jaw images/CAD, so this is approximate. A fitted `tool_to_object`
transform can replace each prior directly in the configuration.

Object pose is withheld after release (NaN in the arrays). Parts can fall,
slide, be pushed or move with their support; subsequent poses require physics
or further observations. Parts are not snapped into successful assembly poses.

## Remaining registration

Each FK stream uses its own robot base. A common scene needs their relative
transform. Repeated assembly contacts provide constraints; measured base
distance and orientation would also resolve it. A single nearly stationary
carrier contact does not determine every base-transform degree of freedom.

Part centres and orientations within the jaws need refinement. Carrier rim
grasps and ambiguous gear yaw mean a pickup tool position is a metric anchor
with an offset, rather than an exact object centre. The next fit combines
those offsets with common-frame contact consistency and checks the result
against real jaw images.

## Artifacts and reproduction

- `scripts/real2sim_grasp.py`: rigid-grasp propagation, tabletop prior and planar
  inter-base fitting with separate validation points.
- `scripts/experiment_real2sim_grasp.py`: applies inspected grasps to saved data.
- `config/real2sim-grasp-pilot.json`: event windows and replaceable pose priors.
- `data/experiments/real2sim-grasp/trajectory-pilot/`: object trajectory NPZ,
  native FK, held masks, offsets and provenance report.

```bash
work/real2sim-venv/bin/python scripts/experiment_real2sim_grasp.py \
  --episode data/kept/20261023T180718-47e74c70 \
  --config config/real2sim-grasp-pilot.json \
  --output data/experiments/real2sim-grasp/trajectory-pilot-new
```

All 22 focused real2sim tests pass, including grasp propagation and base-fit
validation. Saved held poses were checked against measured TCP × grasp offset.
