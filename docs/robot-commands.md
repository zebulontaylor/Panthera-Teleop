# Communicating robot intent

The hand-guidance backend accepts structured commands and a bounded plain-English
vocabulary through `POST /api/commands`. Both use `RobotCommands` in
`app/robot_commands.py`, which owns validation and routes requests into the existing
recorder/controller queues. HTTP threads do not call the motor SDK. The existing
control loop still checks episode identity, request freshness, grasp readiness,
calibration, and motion limits where applicable.

The original dashboard routes remain available and use the same command service.
This task-command API belongs to the teleop frontend. Teleop and policy inference
both use the independent [motor backend](motor-controller.md) for motor control.

## From a terminal

With a dashboard already running:

```bash
python3 scripts/robot-command.py --list
python3 scripts/robot-command.py --command alignment.assist --parameters '{"enabled":true}'
python3 scripts/robot-command.py --instruction 'enable alignment assistance'
python3 scripts/robot-command.py --instruction 'close the gear gripper' --parameters '{"episode_id":"CURRENT_EPISODE_ID"}'
```

The CLI reads the current session token from the loopback dashboard. Use
`--port` for a different local port. Obtain the current episode ID from
`GET /api/status`; episode-bound commands require that explicit ID so an old
instruction cannot silently apply to a new episode.

## HTTP contract

`GET /api/commands` lists command fields and accepted phrases. Send JSON to
`POST /api/commands` with the current `X-Session-Token` from `/api/status`:

```json
{"command":"gear.grasp","parameters":{"action":"close","episode_id":"CURRENT_EPISODE_ID"}}
```

Or send a supported instruction:

```json
{"instruction":"close the gear gripper","parameters":{"episode_id":"CURRENT_EPISODE_ID"}}
```

Provide exactly one of `command` and `instruction`. Unknown fields, unsupported
phrases, conflicting parameters, and invalid state are rejected with HTTP 409.
Session, Host, and Origin checks remain enforced (invalid sessions return 403).
HTTP 202 means the intent was validated and queued, **not that the robot executed
or completed it**. Continue monitoring `/api/status` for controller state.

| Command | Parameters |
| --- | --- |
| `alignment.assist` | `enabled`: boolean |
| `alignment.nudge` | `held`: boolean, `press_id`: string, `episode_id`: string |
| `gear.grasp` | `action`: `close` or `release`, `episode_id`: string |
| `alignment.settings` | `settings`: bounded `max_joint_distance_rad` and/or `max_error_px` |
| `alignment.servo` | `enabled`: boolean |
| `episode.finish` | `label`: `kept`, `review`, or `discarded`; `episode_id`: string |

Natural-language phrases include enable/disable alignment assistance, close/release
the gear gripper, start/stop visual alignment, keep this episode, flag this episode
for review, and discard this episode. Capitalization, whitespace, and trailing
periods/exclamation marks are normalized. Other wording is rejected. Gripper
release means cancelling the grasp assistance, allowing passive hand guidance;
it does not command the gripper to open. Episode decisions retain the existing
configured neutral-return behavior. Visual alignment retains its existing
supervised recording and settled-grasp requirements. Manual nudge remains a
structured hold/renew/release command with the existing operator lease.

This vocabulary does not plan arbitrary tasks like “assemble the gear,” infer
coordinates, or translate free text into joint trajectories. Extend the explicit
command service and its tests when adding another supported robot skill.

## Hardware-free validation

```bash
PYTHONPATH=app work/teleop-venv/bin/python -m unittest discover -s tests -p 'test_robot_commands.py' -v
```
