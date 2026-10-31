"""Motor feedback, gravity limits, and path checks. No teleop or inference imports."""
import sys
import time

import numpy as np

import robot_setup as setup
TORQUE_LIMIT = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
ZERO = np.zeros(6)


def compensation(robot, label='arm', states=None):
    states = robot.get_current_state() if states is None else states
    positions = np.array([s.position for s in states], dtype=float)
    velocities = np.array([s.velocity for s in states], dtype=float)
    if positions.shape != (6,) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise RuntimeError(f'{label}: invalid joint feedback; stopping hand guidance')
    if any(m.get_current_motor_state().fault != 0 for m in robot.Motors):
        raise RuntimeError(f'{label}: motor fault; stopping hand guidance')
    lower = np.asarray(robot.joint_limits['lower'])
    upper = np.asarray(robot.joint_limits['upper'])
    outside = np.flatnonzero((positions < lower) | (positions > upper))
    if len(outside):
        detail = '; '.join(f'joint {i+1} = {positions[i]:.5f} rad, allowed [{lower[i]:.3f}, {upper[i]:.3f}]' for i in outside)
        raise RuntimeError(f'{label}: {detail}; stopping hand guidance')
    torque = np.asarray(robot.get_Gravity(positions), dtype=float)
    if torque.shape != (6,) or not np.all(np.isfinite(torque)):
        raise RuntimeError(f'{label}: invalid gravity torque; stopping hand guidance')
    return np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT), positions


def stop_robots(robots):
    # set_brake exists in the C++ source but is not exposed by this Python wheel.
    # Attempt every arm even if one board has disconnected.
    for side, robot in robots.items():
        try:
            robot.set_stop()
        except Exception as exc:
            print(f'{side}: SDK stop failed: {exc}', file=sys.stderr, flush=True)


def feedback(robot, side, stale):
    states = robot.get_current_state()
    torque, positions = compensation(robot, side, states)
    gripper = robot.get_current_state_gripper()
    # The passive gripper encoder can read slightly negative at its closed stop.
    # YAML limits constrain commanded targets; retain raw measured positions.
    # 999 is the SDK's uninitialized feedback sentinel, not an encoder offset.
    if gripper.position == 999.0:
        raise RuntimeError(f'{side}: gripper position is the SDK feedback placeholder')
    all_states = states + [gripper]
    now = time.monotonic()
    for index, state in enumerate(all_states):
        if not all(np.isfinite(getattr(state, field)) for field in ('position', 'velocity', 'torque', 'time')):
            raise RuntimeError(f'{side}: invalid motor {index+1} feedback')
        if state.fault:
            raise RuntimeError(f'{side}: motor {index+1} fault {state.fault}')
        key = (side, index)
        previous, changed = stale.get(key, (None, now))
        if previous != state.time:
            changed = now
        elif now-changed > 0.5:
            raise RuntimeError(f'{side}: motor {index+1} feedback stale for >0.5 seconds')
        stale[key] = (state.time, changed)
    pose = robot.forward_kinematics(positions)
    arm = dict(position_rad=positions.tolist(), velocity_rad_s=[float(s.velocity) for s in states],
               measured_torque_nm=[float(s.torque) for s in states],
               motor_feedback_time=[float(s.time) for s in all_states],
               motor_ids=[int(m.get_motor_id()) for m in robot.Motors],
               motor_fault=[int(s.fault) for s in all_states], motor_mode=[int(s.mode) for s in all_states],
               gravity_torque_command_nm=torque.tolist(),
               gravity_model_torque_nm=torque.tolist(),
               command=dict(position_rad=[0.0]*6, velocity_rad_s=[0.0]*6,
                            kp=[0.0]*6, kd=[0.0]*6, torque_nm=torque.tolist()),
               gripper=dict(position=float(gripper.position), velocity=float(gripper.velocity),
                            torque=float(gripper.torque), command=[0.0]*5),
               tool_pose_base=dict(position_m=pose['position'], rotation=pose['rotation'].tolist()))
    return torque, arm


def validate_return_path(robot, initial, target, side):
    points = [np.asarray(robot.forward_kinematics(initial*(1-a)+target*a)['position'])
              for a in np.linspace(0, 1, 101)]
    floor = min(points[0][2], points[-1][2])-0.01
    if not all(np.all(np.isfinite(p)) and p[2] >= floor for p in points):
        raise ValueError(f'{side}: return path dips below its endpoint clearance; return canceled')
