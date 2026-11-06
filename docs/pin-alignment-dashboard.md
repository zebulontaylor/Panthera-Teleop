# Pin alignment preview

The inference dashboard replaces its action-value chart with a left-wrist
alignment preview. Viewing the preview never moves the robot. Operator buttons
request individual corrections separately. The left arm carries the pin; the
right arm holds the carrier.

The default hole backend is OpenCV (`--alignment-backend opencv`). It registers
TRAIN-only real left-camera references, then verifies each visible insertion
bore against fresh dark-ellipse and rim evidence. Hidden bore locations are
extrapolated only from a supported registration of the known face layout.
The overlay distinguishes observed holes (green), inferred holes (orange),
supporting face landmarks (blue), and the selected insertion target (purple).
It belongs to the exact analyzed image. The logical left camera is resolved
through the configured camera mapping, defaulting to `wrist_port3`.

The reference bank is `work/left-carrier-opencv/runtime/reference-bank.json`.
The trained CNN remains available with `--alignment-backend cnn` for comparison;
it is not the default because of poor real-image generalization. OpenCV also
abstains on views it cannot register or verify; it never holds a stale image
coordinate through camera motion.

`work/held-pin-cnn/best.pt` supplies separate TOP and BOTTOM Y estimates on the
fixed gripper centerline (native X=335 at width 640). Real proximal-cap labels
supervise TOP; simulation supplies the signed projected offset to the physical
distal BOTTOM. BOTTOM is explicitly inferred, not observed. Both endpoints are
estimates; visibility confidence and real BOTTOM accuracy are unverified. The
model does not predict X or use encoder/FK coordinates as real target labels.

With OpenCV fixed-carrier hold enabled, stationary pin endpoint estimates are
also median-pooled over at most 31 matched pairs within the hold timeout (capped
at 30 seconds). The dashboard distinguishes pooled and retained inferred tips.
Either-arm motion or a new camera/grasp context clears both pin and face history.
Registration loss alone preserves qualified history until expiry. Processing
delay can make a result stale and block nudges without discarding valid history.

Capture hole selects a bore in the current face layout. Its image position is
recomputed on each frame as the left camera moves. Arbitrary manual points use
the current face transform when available; otherwise they expire when either
arm moves. Missing visual/geometry support never falls back to a held image
coordinate. Clear the selection after carrier slip or a new grasp. The symmetric
face does not establish absolute physical bore numbering across arbitrary
quarter-turn rotations or grasps.

To override the target, pause the policy, freeze the right arm, and use **Set
hole manually** in the analyzed image. This stores an explicit operator reference.
The CNN continues tracking the pin and registering the face.

**Set pin manually** overrides the current pin coordinate until either arm moves
more than 0.01 rad in a measured joint. Recapture after nudging, or use **Clear
pin** to resume CNN tracking.
**Clear hole** preserves the selected pin; **Clear pin** preserves the hole.
Changing the grasp or resetting the robot session clears both references.

The existing gain, pixel deadband, step cap, fixed left height, right-arm freeze,
fresh-frame/pose checks, IK/path validation, and operator-triggered motion logic
remain in place. Pixel-to-plane geometry now uses the left camera and its current
joint pose, rather than right-camera optics. The left mount and carrier grasp
remain registration hypotheses, not validated metric calibration. See
`docs/left-carrier-cnn.md` and `docs/held-pin-cnn.md` for training/evaluation
provenance and limitations.

The monitor retains recent analyzed frames for the configured age window.
Buttons use the exact displayed sample while a newer image is loading.
The default age limit is 5 seconds, independent of analysis frequency, with a
configurable maximum of 10 seconds. Motion still checks the current arm pose.
Images remain visible until their replacement is ready. Coordinate changes
invalidate old proposals while retaining the last image on screen.

## Operator corrections

Pause the policy and freeze the right arm before applying a nudge. The Apply
button requests one XY correction, capped at 5 mm; smaller configured caps and
shorter estimated corrections remain possible. Only the four insertion bores
are automatic targets. Choose the intended numbered bore explicitly; automatic
selection waits for a supported full-face map and a pin estimate.

Normal nudges hold the starting native-left TCP height and wrist orientation.
The planner creates dense Cartesian XY waypoints at that height, independently
preflights their paths, then checks the actual retimed piecewise-quintic path.
Predicted height error must stay within 10 micrometres, and every emitted command
gets another FK height check. This is a model-based software bound, not a claim
of independently measured hardware height accuracy.

If feedback changes during asynchronous planning, the old endpoint is not reused
from a new joint origin: the controller rebuilds the relative XY path at the
fresh held height. Larger pose changes, stale proposals, unsupported holes,
unreachable targets and excessive joint motion still reject the request. Both
grippers retain their commanded squeeze. A preview alone never starts motion.

## Auto insertion

With automatic contact assist armed, causal pin pickup and lift followed by
0.15 seconds inside the carrier approach region pauses the policy mid-action.
The detector keeps observing between single-step actions and only hands off
while policy motion is active. A pre-contact vertical stall is no longer needed.

The automatic handoff first holds XY and uses a model-based downward spring
effort capped at 0.5 N, with a separate fixed 20 mm descent / 5 mm XY /
30 second guard.
The Z reference accelerates smoothly over one second to 1 mm/s, with vertical
downward-only position effort and damping active during startup. It never
commands an upward position spring to recover a pin that has already descended.
Measured descent more than 2.6 mm below the slow reference terminates the
approach in measured position hold, rather than attempting to lift back toward
the reference. This catches the startup support/gravity-model mismatch seen in
the November 5 recording: 8.8 mm initial sag followed by an upward spring pull.
That transition still needs hardware verification before automatic approach use.
The effort cap ramps
at 0.15 N/s and all approach joint assistance slews at 0.5 Nm/s. Gentle XY
integral recovery holds the fixed lateral target. These are reference and model
limits; actual speed depends on robot response. Once the downward spring is
loaded against blocked motion, a half-second stable-height window
(0.6 mm range to tolerate native encoder quantization) allows contact to be inferred.
The controller transfers directly to the loaded insertion guard at the settled
pose, preserving torque support, measured velocity history and XY integral
recovery. There is no intervening position hold or zero-assistance priming
window. Z stabilization blends into compliant insertion load over one second;
XY nudges wait 1.5 seconds after the transfer to allow loaded settling.
Automatic loaded insertion allows up to 5 mm upward movement for rebound;
ordinary load sessions retain their 2 mm upward limit. The motor owner permits
this guard transfer only once, from an unexpired approach within its limits,
and requires the new anchor to match fresh measured pose within 1 mm / 0.01 rad.
Repeated loaded reanchoring is rejected. The normal 4 mm descent allowance
therefore excludes the initial approach. Missing contact stops at
the approach travel/time limit. This remains inferred contact, not a force
measurement. Restart both the motor backend and inference controller for the
continuous guarded transfer capability; older motor backends cannot arm auto contact.

Pause the policy, freeze RIGHT, choose the intended hole, and confirm that the pin
is resting on the carrier. **Auto insert · load + nudges** starts one guarded
load session using the displayed downward load, duration, drop threshold and
debounce. Defaults are 0.5 N commanded downward load and a 120 second attempt.

The controller makes three visual nudge passes, spaced one second apart. Each
pass uses the same preview validation, XY goal calculation and 10 mm/s target
slew as the manual **Apply XY nudge** button, with the existing gain and hard
5 mm radial step cap. A zero correction counts as a completed pass without
changing the target. Each pass requires a distinct fresh supported frame.
Missing or refused previews show the actual validation reason while downward
load and drop monitoring continue. An accepted correction keeps its fixed
endpoint through detection loss. The next pass does not wait for measured
settlement, a quiet interval, or visual alignment.

After three passes, the XY target moves toward the LEFT arm base at 2 mm/s
while retaining downward load. Its horizontal direction is fixed from the
original tool position toward the native base origin. This movement no longer
requires visual detections or measured target convergence. The cumulative axial
target shift is capped at 15 mm for the attempt. At that cap the target holds
until a drop or another existing stop. The original 30 mm XY envelope includes
both visual corrections and axial travel. These are commanded target distances;
actual tool motion depends on the load controller and robot response.
Auto insertion does not apply XY dither; the existing dither fields remain for
configuration compatibility.

Context/settings changes, the overall timeout, Stop load, operator disconnect,
and existing motor/load guards terminate the attempt and remove loading. All
passes share the original load timer and travel envelope. Manual nudges are
unavailable during auto insertion. A sustained measured downward drop stops as
an **insertion candidate**; the software cannot confirm seating. This feature
has hardware-free lifecycle tests and has not been validated on robot contact.
Restart the inference controller to load new control code and capability.

## Downward load and manual nudging

The load mode is separate from policy playback. Downward means negative Z in
the left-arm base frame. The requested value is **commanded load in newtons**,
implemented through gravity compensation and Jacobian-transposed torque; there
is no wrist force/torque sensor or claim of measured force regulation. Lateral
position and wrist orientation are held compliantly; Z has no position spring.
The operator must confirm that the pin is resting on the carrier before loading.

XY uses a 600 N/m proportional spring plus a bounded integral force to build
assistance against static friction, with 10 Ns/m XY damping. The integral gain
is 9600 N/m/s, its vector magnitude is capped at 12 N, and its vector change
is limited to 12 N/s. These gains apply to manual loaded nudges and all
auto-insert stages, including the axial sweep. Only
the error outside a radial 0.25 mm deadband is integrated; inside that deadband
the bounded accumulated assistance is retained. Reversed error unwinds it.
The stored integral itself is capped to prevent hidden windup, and it remains
zero during velocity priming. Starting a new load session resets it. Target
slewing does not reset it. The proportional and integral XY drive are scaled
together to available joint-assistance and total-torque headroom. Downward load,
orientation assistance and damping are retained; the stored integral is reduced to
the applied value. The per-joint assistance torque caps are
`[3, 6, 6, 1.5, 1.2, 1.2]` Nm in both the load controller and independent
motor backend. Total torque caps including gravity remain
`[5, 8, 8, 3, 2, 2]` Nm. Torque-stop diagnostics include the requested torques
and caps. Restart both processes after changing these limits.
Existing joint-assistance, total-torque, travel and
orientation refusals still apply; the integral changes neither Z nor orientation
control. Diagnostics expose XY error, proportional force, integral force and
commanded XY force. A software static-friction plant test verifies bounded force
buildup, but does not calibrate real robot friction or validate hardware stability.

A loaded nudge changes the XY target once while keeping the downward load.
Manual loaded nudges retain the preview step cap (default/maximum 5 mm) and
the existing nominal target speed of 10 mm/s. Auto insertion always caps each step at
5 mm. The force controller continues controlling Z;
the normal fixed-height position trajectory is not used during downward load.
For a manual loaded nudge, completion requires measured XY error within 0.4 mm for 0.2 seconds, rather than
completion of target slew. After four seconds without convergence the dashboard
reports that the target was not reached while retaining the active load. It
displays measured tracking error and logs measured displacement separately from
the requested correction.
Changing preview gain or step size cancels queued XY requests without stopping
the existing load.

Auto insertion leaves the legacy XY dither controls unused. Its motion sequence
is three visual nudge passes followed by bounded toward-base movement.

Maximum load duration is 300 seconds (default 120 seconds). The duration ceiling
is shared by auto insertion, the Cartesian controller, and the independent motor
process. Existing torque limits and load guards remain in place, including the
30 mm load envelope. Load duration and drop-monitor settings are established when
the load session starts; manual nudges do not reset them.

Downward movement during initial load settling does not stop the attempt.
Once a stable loaded-height baseline is acquired, a later sustained drop from
that baseline is reported as an insertion candidate, not proof of a successful
insertion. The absolute hard downward-travel limit still includes initial
settling displacement. Timeout, Stop, and guard violations
remove the load and hold the measured pose. The independent motor process also
checks command age, measured travel, orientation, and torque bounds. Downward-load mode does not
stop on joint, tool, downward, or angular speed; measured velocity remains in
use for damping and diagnostics. General position-command speed limits remain active.
Load mode requires that process's guarded-load capability; an older motor
process must not fall back to unguarded torque commands.

## Direction and scale

`app/pin_alignment_geometry.py` intersects the pin and hole image rays with the
estimated carrier face. Their difference is a translation tangent to that
face. Measured right-arm joints determine the wrist-camera orientation. The
nearly parallel arm bases require only a small rotation for a direction vector:
`delta_left = R_right_from_left.T @ delta_right`. Base translation is irrelevant
for this vector. Gain, maximum step, and a pixel deadband bound the proposal.

The geometry uses the existing camera, carrier grasp, and planar base profiles
under `config/pin-insertion-photo/`. Native forward kinematics was checked
against all 63 labeled demo poses: maximum right-tool position discrepancy was
1.31e-16 m and maximum rotation-matrix discrepancy was 2.22e-16. This checks the
joint convention and transform implementation; it does not validate physical
camera scale or pin contact. Metric estimates assume the pin base is on the
carrier face. The CNN's learned XYZ output and activation score do not authorize
movement.

## Right-arm freeze

The freeze toggle is separate from the alignment preview. During policy
execution it holds the measured right-arm joint pose and preserves the
commanded gripper hold, while the left arm continues. It applies to subsequent
policy chunks as well as the current trajectory. Pause/resume retains the
freeze. Reset, manual takeover, and a new robot session clear it. Resuming the
right arm rebuilds the bounded trajectory from the held pose.

The UI requires an explicit freeze capability from the running controller.
An older process can serve a newer HTML file without loading its Python control
code; a controller restart is then required. Restarting only the browser does
not update the robot control loop.

## Local validation

Use `app/inference_dashboard.py --demo` for simulated arms and cameras. The
preview model runs in a separate process using the pin-insertion runtime, so
model loading and image processing do not run in the motor control loop.
Recorded-image UI checks are diagnostics, not additional held-out model tests.

Hardware-free checks include active recordings of normal nudges and loaded XY
nudges. A scripted measured downward step verifies
load removal; this is a control-lifecycle test, not a contact-physics or physical
force-control validation. Operator corrections are recorded as `operator_nudge`
or `operator_load`; they do not count as autonomous policy success.

## Automatic settling and nudge advice

**Automatically freeze after settling** defaults off. During policy playback,
right-arm motion must first exceed 0.1 rad from its initial observed pose. The
recent-task carrier gate is native right joint 1 > 0.35 rad, joint 2 > 1.2 rad,
and measured gripper < 1.3 rad. The six joints and measured gripper must then
remain within a 0.01 rad peak-to-peak range for one continuous second, with
right-joint target error at most 0.05 rad. Feedback gaps over 0.1 seconds,
policy pauses, and ineligible control modes clear that stability window.

Automatic freeze holds the measured right-arm joints and tightens the right
gripper once by reducing its existing commanded position by 0.05 rad, clamped
to the configured gripper limits. The existing speed and torque limits still
apply. The tighter command is held without repeated tightening. This adds a
small squeeze; it does not verify grasp force. Manual freeze preserves the
existing gripper command. The left arm continues. Manual unfreeze inhibits
another automatic freeze until the checkbox is toggled or the collection/session
resets. Disabling detection leaves an existing freeze held; use Unfreeze right
arm to release it. Reset and takeover clear the hold and detector history.
The gate is specific to the recent carrier task. The
[right-arm demo audit](right-arm-settle-demo-audit.md) found candidates in 79/80
recent kept demonstrations, with two later small corrections. That audit used
six-joint stability; runtime additionally requires stable measured gripper,
tracking, fresh feedback and active playback. It is not hardware validation.

**Detect persistent visual misalignment** is a separate advisory toggle, also
off by default. With RIGHT frozen, it requires at least three distinct supported
fresh frames spanning a second, a consistent selected target and correction
direction, and both arms stationary within 0.01 rad of their observation poses.
Stale images, detection loss, target changes, motion, or unsupported coordinates
clear the evidence. Repeated polls of one image do not add evidence.

The readout distinguishes Nudge suggested, Visually aligned, and unknown.
It never starts motion or establishes contact or successful insertion. The
current held-pin model's inferred distal endpoint (`center_axis_inferred`) is
not accepted as verified evidence: use a fresh manual pin reference for this
advisory mode until real distal-tip localization is validated. After moving an
arm, recapture the manual pin point. The most reliable future insertion gate
would combine calibrated tip/hole tracking with measured contact and insertion
progress; pixel coincidence alone cannot provide that evidence.

## Automatic handoff when the pin presses the carrier

**Automatically assist when pin presses carrier** arms actual automatic motion,
separately from the visual-misalignment advisory. It uses the displayed insertion
load, duration and drop settings as captured when armed. With RIGHT
frozen and policy playback active, the controller observes the later narrow-grip
pin pickup, lift, transport, descent and brief slowdown near the carrier. It then
cancels the remaining policy trajectory and in-flight prediction, holds the
measured pose, and starts the existing guarded load plus automatic XY corrections.
This trigger does not require visual misalignment: straight-in attempts enter
the same three-pass routine. A zero correction counts as a pass without
commanding a lateral displacement.

Arm it before pickup. The state machine needs the pickup/lift history to avoid
mistaking earlier gear handling for pin insertion. Enable automatic right-arm
freeze too, or freeze RIGHT manually. It waits if the guarded motor backend or
alignment monitor is unavailable. The automatic trigger uses a task-specific
native TCP/gripper path heuristic for this fixed installation, not a contact
sensor. Motor-torque residuals were not reliable standalone contact indicators
in the inspected demonstrations. See the [contact demo audit](pin-contact-demo-audit.md).

After handoff, each of the three visual nudge passes requires supported fresh
visual geometry. The following axial movement proceeds independently of detection. The advisory-only detector's
separate verdict does not gate this handoff. Missing target localization waits
without inventing a correction. Existing travel, duration, joint, torque,
heartbeat, takeover and motor-backend guards remain active. Logs distinguish an
inferred automatic contact candidate from explicit manual contact confirmation.
The loaded-height monitor likewise records inferred baseline authority rather
than an operator acknowledgment.

Turning the toggle off stops an active automatic assist and leaves policy
paused. Stop load, pedal pause and Escape also stop it. An attempt is latched;
rearm with the toggle for another attempt, or finish/reset the episode. A new
collection/session clears the observed phase history while retaining the enabled
setting within the running controller. No automatic policy resumption occurs.
