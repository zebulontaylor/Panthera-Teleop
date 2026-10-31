# Supervised relaxed incremental nudging

The `supervised_relaxed` profile uses distance and speed limits with fresh visual
feedback. Its policy is `distance_speed_only`. Calibration remains unverified;
the strict profile is unchanged and can be selected without `--relaxed-nudge`.

| Limit | Supervised relaxed |
| --- | ---: |
| Maximum 3D displacement per commanded step | 5 mm |
| Cumulative Cartesian path / displacement from start | 50 mm |
| Commanded joint speed | 0.02 rad/s |
| Visual alignment tolerance | 3 px in both views |

Each correction changes at least one joint by one encoder tick, about 0.000628
rad (0.036 degrees). This is a joint resolution, not a fixed Cartesian minimum.
The distance ceilings still apply to the encoded endpoint and the checked joint
completion positions. Smaller moves are possible. An already aligned view pair
does not force a movement.

After the commanded movement interval plus 0.2 seconds, the controller collects
three fresh observations and plans again from measured joint positions. It does
not require a fixed fraction of physical progress or encoder settling before
replanning. It can continue through worsening or oscillating image errors within
the remaining distance budget. There is no separate step-count or total-duration
limit in this profile.

Experimental calibration-envelope, image-proximity, gripper/carrier-change,
reported-velocity noise, Cartesian orientation/sideways-drift, joint-norm travel,
and response-improvement stops are removed for this profile. The preflight and
controller share the same evidence validation. They retain camera/checkpoint
identity, fresh synchronized finite image evidence, valid finite hardware state,
physical joint limits, backend torque limits and explicit operator control.
Recording, grasp authorization, reset/return exclusion, alignment disable, and
Stop/release remain in force. The saved calibration and the strict profile are
not edited.

## Historical offline evidence

The earlier offline audit, before the current distance/speed-only policy, used actual encoder-grid targets and all 64 combinations of
independently completed joint moves. Across 443 recorded observations, 287 were
already within image tolerance, 135 produced proposals, 10 lacked an observable
height response, and 11 had conflicting camera directions. Of 431 observations
within partial pose/proximity bounds, 127 produced proposals. These historical counts are
not current-policy eligibility or measured convergence.

The exact failed live pose at episode time 6.606824395 s in
`data/pending/20261022T234534-07928832/states.jsonl` previously stopped at zero
steps with `motor_resolution_limited`. In that earlier-policy calculation using the same joint pose and image error,
the relaxed proposal predicted an endpoint displacement of
`[-0.1036, -0.1853, -1.3857] mm` and rotation of 0.00144 rad. Across completion
corners the maximum horizontal displacement is 1.859 mm and rotation is
0.01039 rad. Predicted vertical image residuals change from
`[32.116, -8.678] px` to `[30.794, -5.245] px`.

That isolated calculation took approximately 28 ms; at the configured joint
speed the largest joint change requires at least 0.503 s, before settling and
fresh-image acquisition. These are an offline compute measurement and a nominal
speed-based lower bound, not measured hardware latency.

Rerun the current policy without hardware (this does not reproduce the historical
counts above):

```bash
work/teleop-venv/bin/python scripts/validate_incremental_nudge.py \
  --controller-profile supervised_relaxed \
  --failed-states data/pending/20261022T234534-07928832/states.jsonl \
  --output logs/alignment-calibration/distance-speed-offline.json
```

The new report retains current source hashes and the exact failed pose/error.
It uses a separate filename to preserve the historical relaxed and strict reports. If the recording is moved to review, use its new
path. No audit commands motor hardware, and no physical convergence is claimed.
