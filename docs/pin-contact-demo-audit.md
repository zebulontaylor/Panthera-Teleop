# Pin push/contact stage audit

AI visually inspected actual wrist_port3 images from three initial operator-kept demonstrations: 20261031T162906-0055bc5e, 20261031T162845-8467e8e9, and 20261031T153335-a7c3015c. The contact sheet is `pin-contact-audit-contactsheet.jpg`; phase telemetry and gravity-subtracted torque summaries are in `pin-contact-demo-audit.json`. Physical contact timing and straight-in versus nudge-required outcomes are not human-certified.

The pin approaches the carrier face at approximately 12.5, 12.0, and 11.5 seconds respectively. These observations support a causal sequence detector: distinguish gear handling first, identify a later table pickup with narrow grip, observe lifting, observe transport over the carrier, then observe descent and a local stall. A stall is a probable contact candidate, not proof of mechanical loading or evidence a nudge is required.

| Signal | Observed example values | Implication |
|---|---|---|
| Gear at carrier around 5s | left grip .53–.57; left-minus-right TCP height .009–.010m | Mere carrier proximity or low velocity confuses gear and pin. |
| Pin table pickup | left TCP x .287–.297m, y -.243 to -.282m, z .020–.021m | Require pickup outside the carrier region; These three examples fit x>.25m and z<.05m; later shifted pickups need the broader runtime pickup region reported in pin-contact-detector-replay.md. |
| Pin held grip | .011–.055rad after closure | A .015rad lower bound rejects a genuine example. Further 80-demo audit found most candidate pin pickups also near -.010rad; a positive lower bound is not justified. |
| Pin lift | TCP reaches z .068–.080m | Observing at least .015m rise from pickup supports transport. |
| Pin face approach/contact | left-minus-right TCP x -.020 to -.027m, y -.589 to -.600m, z .024–.033m | Runtime region x[-.07,.02], y[-.64,-.55], z[.015,.04] contains these examples. |
| Descent | left TCP descends from .068–.080m toward .048–.056m | A .010m descent requirement has support. |

TCP positions are in each arm's own recorded base frame. Subtracting them assumes the existing fixed base arrangement; y near -.6 is not a world-space separation. Tool frames do not measure pin tip or carrier surface positions, so these are stage features specific to the present grasp/setup.

The left measured-minus-gravity-model torque is a poor standalone contact gate. During the contact plateau, joint2/3 median absolute residuals were about .005–.018Nm. During moving approach the same channels had medians around .05–.08Nm and p90 around .19–.22Nm. Gear contact also has residual activity. A threshold of .1Nm would detect motion/transients while missing much sustained pin contact. Operator gravity-assist handling, friction, payload, calibration errors and acceleration confound these signals. Avoid presenting this residual as measured force.

Recommended candidate: latch the observed pin pickup/lift sequence, require the selected carrier region and downward progression, then a short .15–.25s stall. Replay and report coverage before activating. Current raw-image evidence cannot establish that a .5mm vertical-range stall over .15s equals first contact in all demos. A fresh visual pin/face overlap or supported target observation would strengthen the stage gate; independent force sensing or validated torque-to-contact estimation is needed for true contact confirmation.

Automatic nudging should additionally depend on a reliable current off-target observation. Straight-in demonstrations can reach the same contact candidate; contact alone must not mandate a lateral correction. Missing localization should remain unknown. Repeated supported nonzero error outside an uncertainty-aware deadband distinguishes a proposed nudge from merely having reached the contact stage.

## Additional pickup coverage audit

A separate latest-80 scan found narrow table visits (after6s, left x>.25m, z<.05m, gripper<.25rad) in74/80 recordings. Final narrow-table gripper was <=.015rad in60/74; median minimum near the last table visit was -.01005rad and minimum -.01131rad. All74 had more than .15s between first and last narrow-table samples, but this does not establish a continuous stable dwell. Do not infer pin presence or absence from positive grip gap. A broad -.02rad lower bound plus pickup/lift/transport sequence preserves these examples while retaining ambiguity about missed grasps. Detailed results: `pin-contact-pickup80-audit.json`.

Two additional negative-grip episodes,20261031T152311-a72b8cd1 and20261031T152355-8ce20208, were visually inspected in `pin-contact-negative-grip-contactsheet.jpg`. Both show expected pickup/transport and a small metallic protrusion near closed jaws; face approaches are around11–11.5s and12–13s, compatible with candidate stage detections at11.4s and13.0s. These are approximate image-based observations, not independent contact labels.
