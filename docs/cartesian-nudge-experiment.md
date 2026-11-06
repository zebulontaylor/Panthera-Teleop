# Experimental Cartesian gear nudge

This is a separate alternative to the existing torque nudge and repeated visual
servo. It has been checked offline, not evaluated on hardware. The normal
`start.sh`, original controllers, dashboard and saved configuration are unchanged.
Their exact pre-experiment copies and SHA-256 manifest are under
`archive/pre-cartesian-nudge-20261022/`.

## Behavior

The alternative dashboard offers **One vertical nudge and hold**. Each start
allows at most one tool displacement of 0.5 mm along the left robot's base Z
axis, with a joint-step norm of at most 0.003 rad. Both upward and downward
corrections are possible. Base Z must be verified against the physical setup;
the kinematic model does not independently measure gravity or tool location.

The controller solves the six-dimensional robot Jacobian for pure translation
along Z with unchanged X/Y and orientation. It then projects the existing
four-by-six image Jacobian onto that direction. It requires both wrist views
to request the same direction and compatible displacement magnitudes. If either
view is already inside the 3-pixel deadband while the other is not, it blocks
rather than guessing. It checks 11 samples of the predicted joint path.

The existing checkpoint, camera identity, freshness, proximity, joint-limit and
both-arm calibration-envelope gates remain in effect. Supervised operation
continues to rely on the operator's retained-object and target check; it does
not install an automatic grasp detector. Legacy torque pulses and runtime
calibration-envelope widening are rejected in this alternative dashboard.

After a step, the arm settles and three distinct fresh predictions are compared
with its pre-step errors. It reports physical movement not confirmed, worsening
image error, inconclusive response, or observed step. It never automatically
takes a second step. Position hold remains until **Release position hold**.
An observed step is not an independent calibration certification.

Measured horizontal drift above 0.25 mm, orientation change above 0.005 rad,
total tool displacement above 1 mm, excessive joint travel, tracking deviation,
model/feedback disagreement, changed gripper, or ended collection aborts or
ends correction. Limits use encoders and the URDF, not external metrology.
Right-arm movement blocks correction. A five-second budget ends an unfinished
trial. The position-command speed limit is 0.02 rad/s and motor torque limits
are capped at the starting modeled gravity magnitude plus 0.3 Nm, within the
existing configured limits. These settings still require physical evaluation.

## Offline findings

Run without connecting to robot devices:

```bash
work/teleop-venv/bin/python scripts/validate_cartesian_nudge.py
PYTHONPATH=tests work/teleop-venv/bin/python -m unittest \
  test_cartesian_nudge test_alignment_servo test_supervised_alignment \
  test_robot_commands test_alignment_dashboard
```

The 443-observation calibration cache yielded just one geometric proposal:
324 were blocked for camera direction/deadband disagreement, 108 for magnitude
disagreement, and 10 for insufficient predicted height response. The one proposal
was 0.480 mm vertically, with 0.000583 mm modeled horizontal displacement.
An independent finite-difference calculation confirms the URDF Jacobian's Z
direction. This checks software/model consistency, not physical accuracy.
Replay proposal counts do not include all runtime camera and grasp gates.
Results and source hashes are in
`logs/alignment-calibration/cartesian-nudge-offline.json`.

The low proposal count is a reason to obtain new calibration data. It does not
prove the CNN is inaccurate or establish which failure caused the reported
backward event. Record measured small positive/negative Z movements at fixed
X/Y and orientation with the carrier stationary. Verify actual tool movement
independently and estimate each view's pixel response per meter. Use separate
movements to test the direction and response before permitting repeated steps.
The current alternative still uses the original local visual fit, so it can
correctly refuse to move frequently.

## Separate launcher

The alternative is available through:

```bash
./start-cartesian-nudge.sh
```

This command starts real hardware control just as `start.sh` does. It has not
been run during this experiment. Use it only in a supervised session after
stopping the existing collector; both launchers share
the existing exclusive controller lock. Closing the settled gear grasp and
starting an active recording remain prerequisites for the dashboard action.
Release the hold before requesting another one-step trial. The standard
`./start.sh` continues to select the original implementation.

Implementation: `app/cartesian_nudge.py`, `app/hand_guide_cartesian.py`,
`app/dashboard_cartesian.py`, `app/cartesian_nudge_mode.py`, and
`app/web/cartesian.html`.
