# Pin carrier-region handoff replay

The current v2 heuristic recognizes causal pin pickup, lift, transport and a
sustained carrier-region entry with RIGHT frozen. Entry must persist for 0.15
seconds. It does not require the approaching pin to stop or descend before
handoff: the guarded approach phase then lowers under a bounded model-based
load and infers contact from settled height before loaded insertion begins.
Neither phase measures physical contact or establishes successful insertion.

The latest 80 sorted `data/kept` episodes were replayed with v2 at at least
20 ms sample spacing, with observation eligibility assumed true and RIGHT
freeze assumed after 3.2 seconds. **80/80 episodes triggered once.** This is
trigger coverage on demonstration telemetry, not sensitivity or false-positive
accuracy against contact labels. The replay does not reproduce live policy
pauses, inference transitions, or browser/context changes. The live controller
observes causal pickup across valid paused/predicting intervals and separately
gates the handoff on active policy execution.

Report: `work/pin-contact-detection/latest80-region-v2-replay.json`. The latest
three v2 handoffs occurred at 11.97, 12.04 and 12.56 seconds.
The previous v1 report (`latest80-causal-replay.json`) also had 80/80 coverage,
but required measured descent and a near-carrier vertical stall. Its trigger
times below are historical v1 results and do not describe v2 handoff timing.

The pickup lower gripper threshold is -0.02 rad because native encoder feedback
can be slightly negative: 55/76 episodes visiting the low pickup zone had minimum
grip below +0.005 rad; the minimum observed was -0.01445 rad. The pickup region
requires native LEFT TCP x > 0.21 m and z < 0.05 m, followed by narrow-grip
settling, lift and transport. Settling may finish during early lift, preserving
the causal table pickup anchor. Requiring the hand to remain motionless at the
table missed smooth human pickups.

A separate visual spot check of negative-grip examples `20261031T152311-a72b8cd1`
and `20261031T152355-8ce20208` showed expected pickup/transport and a small metal
protrusion near the closed jaws. Historical v1 candidate triggers were 11.40 s and 13.00 s;
the sampled images placed the tip near the carrier around 11.5 s and 13.0 s.
These coarse image checks do not establish exact contact timing.

In the previous v1 replay, the latest three episodes triggered at 12.19, 12.26 and 12.72 seconds with relative
native TCP z of approximately 0.030, 0.029 and 0.030 m. The final carrier region
is LEFT minus RIGHT native XYZ: x [-0.07, 0.02], y [-0.64, -0.55],
z [0.015, 0.04] m. These bounds depend on the fixed two-base installation.
A pin, empty grasp or another thin object can share gripper coordinates; the
sequence remains a probable pin stage rather than object identity proof.

Reproduce without hardware access:

```sh
work/teleop-venv/bin/python scripts/replay_pin_contact_detection.py --output work/pin-contact-detection/latest80-region-v2-replay.json
```

Pickup audit: `work/pin-contact-detection/pickup-distribution.json`. Tests:
`work/teleop-venv/bin/python -m pytest -q tests/test_pin_contact_detection.py` (24 passed).

The historical pickup-zone audit found six true pickup paths shifted to x=0.222–0.249 m,
so the x threshold was broadened to 0.21 m. A trial y restriction [-0.30,-0.20]
rejected five demonstrations; their actual low narrow-grip paths extended beyond
that interval (approximately -0.38 to -0.18 m). That extra y restriction was
removed because the causal pickup/lift/transport sequence supplies the stage
evidence. All 80 triggered with those pickup thresholds; thresholds were tuned on this same cohort, so this
is not held-out accuracy or proof that other object/empty-grasp motions cannot
trigger. In v1, shifted pickup example 155247 triggered at 11.89 s, near its sampled
carrier approach at approximately 12 s.
