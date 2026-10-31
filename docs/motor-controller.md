# Motor backend and teleop/inference clients

The robot controller is a separate process from the hand-guidance and inference
frontends. Frontend edits or restarts do not reload its motor SDK objects or its
control loop.

```text
Teleop / recording / alignment       Policy inference / browser controls
                \                    /
                 MotorClient (local JSON protocol v1)
                              |
                    logs/motor-controller.sock
                              |
             motor_backend.py — independent user service
             feedback checks, gravity model, command caps,
             control timing, owner lease, disconnect hold
                              |
                      Panthera SDK → motors
```

## Ownership

`app/motor_backend.py` owns both arms, the exclusive `logs/control.lock`, feedback
validation, the motor command loop, and the SDK lifetime. It imports no dashboard,
recording, alignment, or inference modules. `app/motor_control.py` contains shared
feedback/gravity/path checks. `app/robot_setup.py` contains hardware configuration
and USB preflight. Backend ceilings live in `config/motor-controller.json`, separate
from frontend configuration and inference model contracts.

`app/motor_client.py` contains the versioned socket client and arm adapters. The
adapters stage commands; they do not call the SDK. `flush()` sends a complete
frame containing both arms' joint and gripper commands. The backend validates
all fields before replacing its current frame. Acceptance means queued; an SDK
failure appears in backend status on a subsequent tick.

`app/hand_guide.py`, `app/inference_dashboard.py` (through
`app/robot_hardware.py`), and the physical path in `app/remote_inference.py` use
this client. Cameras, recording, policy requests, and task decisions remain on the
frontend side. The incremental-nudge launcher uses this same backend client and
can replace the collector without stopping powered hold. Other experimental controllers and the official Quest launcher
still use their legacy direct-SDK path and exclusive controller lock; stop the
motor backend before running those alternate controllers.

## Lifetime and switching frontends

On the first live connection, a frontend starts the independent
`panthera-motor-controller` user service with `systemd-run --user`. Its process
and systemd control group are separate from the frontend's, so stopping a
frontend service does not also kill the motor backend. Backend output goes to
`logs/motor-controller.log`. No backend is started in hand-guidance demo mode.

If a user systemd manager is unavailable, start the backend explicitly in a
separate terminal before starting a frontend:

```bash
work/teleop-venv/bin/python -u app/motor_backend.py --run
```

The backend initially holds the measured joint positions with passive grippers.
Only one frontend may acquire control at a time. A frontend can finish startup
while the backend holds for up to 30 seconds. Once command streaming starts,
every accepted frame renews a 0.5-second lease. Status reads and rejected commands
do not renew that lease. On release or expiry, the backend captures fresh measured
joint positions and holds those positions, makes the grippers passive, and revokes
the old token. An expired client must reconnect explicitly; delayed old requests
cannot resume motion. Closing a frontend leaves the backend and its motor lock
running. Another frontend can then acquire control.

The loop targets 150 Hz independently of frontend updates. It re-evaluates gravity
using current feedback on each tick, retaining any accepted bounded assistance
offset. Position commands retain their backend-validated speed and torque caps.
Actual rate and USB behavior still need physical validation. Hardware feedback
faults or SDK command rejection latch a backend fault and attempt the supported
SDK stop for both arms; these faults do not automatically restart motor control.

Inspect or explicitly stop the backend:

```bash
python3 scripts/motor-controller.py status
# Support both arms before ending powered control:
python3 scripts/motor-controller.py stop
```

`systemctl --user stop panthera-motor-controller` also stops a service-started
backend. Stopping the backend ends powered control; stopping teleop or inference
alone now retains position hold. The optional English/structured `/api/commands`
interface remains a frontend task API and is separate from this motor protocol.

## Checks without motors

`tests/test_motor_backend.py` uses fake motors to verify exclusive ownership,
frame rejection, torque/speed limits, lease expiry, measured-pose hold, gravity
updates, hardware fault handling, and a real socket connection from a frontend
process that exits abruptly. None of those tests imports the vendor motor SDK
or opens a serial device.
