"""Independent hand guidance using the manufacturer's gravity-only control law.

No startup trajectory, Quest input, or arm-to-arm following.
Episode decisions trigger a raised-neutral return and gripper opening, then hold.
On exit gravity assistance stops using the SDK's supported stop method.
The SDK destructor subsequently applies its configured exit behavior.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import fcntl
import signal
import subprocess
import queue
import sys
import time

import numpy as np
import yaml

from recording import EpisodeStore
from foot_pedal import FootPedal, DESCRIPTION as PEDAL_DESCRIPTION
from dashboard_stereo import Cameras, Dashboard
from neutral_return import NeutralReturn
from neutral_hold import NeutralHold
from reset_grippers import ResetGrippers, command_gripper
from usb_stability import keep_usb_awake, is_serial_disconnect
from alignment_assist import OperatorNudgeLease
from cartesian_nudge_mode import PreviewOnlyAlignmentAssist as AlignmentAssist
from stereo_pose_nudge import StereoAlignmentVision as AlignmentVision
from gear_grasp import GearGrasp
from cartesian_nudge import ToolKinematics
from bounded_alignment_trial import StereoPoseNudge as CartesianNudge

spec = importlib.util.spec_from_file_location('teleop_setup', Path(__file__).with_name('launch-teleop.py'))
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
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


def simulated_sample(started):
    age = time.monotonic()-started
    phase = age % 12
    q = 0.06*np.sin((phase-3)*2) if phase > 3 else 0.0
    v = 0.12*np.cos((phase-3)*2) if phase > 3 else 0.0
    arms = {side: dict(position_rad=[q]*6, velocity_rad_s=[v]*6,
                       measured_torque_nm=[0.0]*6, gravity_torque_command_nm=[0.0]*6,
                       gripper=dict(position=0.0, velocity=0.0, torque=0.0),
                       simulated=True) for side in ('left', 'right')}
    return dict(monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns(), arms=arms, simulated=True)


def validate_return_path(robot, initial, target, side):
    points = [np.asarray(robot.forward_kinematics(initial*(1-a)+target*a)['position'])
              for a in np.linspace(0, 1, 101)]
    floor = min(points[0][2], points[-1][2])-0.01
    if not all(np.all(np.isfinite(p)) and p[2] >= floor for p in points):
        raise ValueError(f'{side}: return path dips below its endpoint clearance; return canceled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--demo', action='store_true', help='Simulated arms; never open motor devices')
    parser.add_argument('--demo-cameras', action='store_true', help='Include real cameras in a simulation')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--alignment-assist', action='store_true',
                        help='Enable optional CNN alignment assist (requires verified local calibration for motion)')
    parser.add_argument('--alignment-task', choices=('gear', 'pin'),
                        help='Select the alignment task; defaults to recording configuration')
    parser.add_argument('--data-dir', type=Path, default=setup.ROOT/'data')
    parser.add_argument('--experiment-config', type=Path, default=setup.ROOT/'config/alignment-stereo-experiment.json')
    args = parser.parse_args()
    try:
        configs = {} if args.demo else setup.prepare()
    except setup.RobotConnectionError as exc:
        print(f'Robot startup unavailable: {exc} No motor devices opened.',
              file=sys.stderr, flush=True)
        return 1
    if not args.run and not args.demo:
        print('Hand-guidance configurations ready. No motor devices opened.')
        return
    # A shared lock also protects the Quest launcher from competing for motor ports.
    lock_path = setup.ROOT/('logs/simulation.lock' if args.demo else 'logs/control.lock')
    lock_path.parent.mkdir(exist_ok=True)
    lock_fd = lock_path.open('w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another controller owns this workspace; stop it before starting a new mode')
    config = json.loads((setup.ROOT/'config/recording.json').read_text())
    if args.port:
        config['port'] = args.port
    alignment_config = dict(config.get('alignment_assist', {}))
    alignment_config['experiment'] = json.loads(args.experiment_config.read_text())
    if alignment_config.get('calibration_file'):
        alignment_config['calibration'] = json.loads((setup.ROOT/alignment_config['calibration_file']).read_text())
    if args.alignment_assist:
        alignment_config['enabled'] = True
    if args.alignment_task:
        alignment_config['task'] = args.alignment_task
    config['alignment_assist'] = alignment_config
    context = dict(mode='simulation' if args.demo else 'independent_gravity_assist',
                   torque_limits_nm=TORQUE_LIMIT.tolist(),
                   calibration=json.loads((setup.ROOT/'config/calibration/setup.json').read_text()),
                   arm_mapping=json.loads((setup.CONFIG/'arm-mapping.json').read_text()),
                   robot_configs={side: yaml.safe_load(path.read_text()) for side, path in configs.items()},
                   motor_configs={side: yaml.safe_load((setup.CONFIG/f'{side}-motors.yaml').read_text()) for side in configs},
                   urdf_sources={side: Path(yaml.safe_load(path.read_text())['urdf']['file_path']).read_text()
                                 for side, path in configs.items()})
    robots = {}
    store = cameras = dashboard = pedal = vision = None
    reason = 'operator_stop'
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        pedal = FootPedal(config.get('foot_pedal'), enabled=not args.demo)
        context['foot_pedal'] = dict(PEDAL_DESCRIPTION, devices=pedal.paths)
        store = EpisodeStore(args.data_dir, config, context)
        alignment = AlignmentAssist(alignment_config)
        operator_lease = OperatorNudgeLease()
        gear_grasp = GearGrasp()
        alignment_servo = None
        alignment_servo_status = dict(active=False, reason='not_started')
        last_alignment_trial = None
        alignment_readiness = None
        alignment_readiness_updated = -float('inf')
        vision = AlignmentVision(alignment_config)
        alignment_enabled = bool(alignment_config.get('enabled', False))
        alignment.set_enabled(alignment_enabled)
        vision.set_enabled(alignment_enabled)
        store.update(alignment_assist=dict(enabled=alignment_enabled, active=False,
                     reason='waiting_for_cameras' if alignment_enabled else 'disabled'))
        store.update(foot_pedal=pedal.snapshot())
        print(f'Foot pedal: {pedal.paths or "not connected (force request recorded as null)"}', flush=True)
        if not args.demo or args.demo_cameras:
            cameras = Cameras(store, config)
            cameras.wait_ready()
            print('All three cameras passed capture preflight.', flush=True)
        dashboard = Dashboard(store, cameras, config['port'], demo=args.demo, vision=vision)
        print(f'Episode dashboard: {dashboard.url}', flush=True)
        if not args.no_browser:
            subprocess.Popen(['xdg-open', dashboard.url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not args.demo:
            sys.path.insert(0, str(setup.SDK/'scripts'))
            from Panthera_lib import Panthera
            for side in ('left', 'right'):
                if stopping:
                    return
                mapping = json.loads((setup.CONFIG/'arm-mapping.json').read_text())[side]
                expected = yaml.safe_load((setup.CONFIG/f'{side}-motors.yaml').read_text())['robot']['Serial_Type']
                if str(setup.resolve_board(mapping)) != expected:
                    raise RuntimeError('USB mapping changed; restart the launcher')
                keep_usb_awake(expected)
                robots[side] = Panthera(str(configs[side]))
                setup.wait_for_joint_feedback(robots[side], side)
        # Validate both models/readings before either arm receives assist torque.
        stale = {}
        for side, robot in robots.items():
            feedback(robot, side, stale)
        if config.get('neutral_return', {}).get('enabled') and not args.demo:
            NeutralReturn(config['neutral_return'], {side: (r.joint_limits['lower'], r.joint_limits['upper'])
                                                     for side, r in robots.items()}, time.monotonic())
        print('SIMULATION ACTIVE; motors unopened.' if args.demo else
              'HAND GUIDANCE ACTIVE: both arms independently gravity assisted; grippers passive. Support handles.', flush=True)
        store.update(control_active=True)
        last_print = 0.0
        started = time.monotonic()
        sequence = 0
        neutral = None
        hold = None
        reset_grippers = None
        handoff_recording_released = False
        return_request = None
        return_path_checked = False
        demo_return_pose = None
        demo_gripper_position = {side: 0.0 for side in ('left', 'right')}
        demo_limits = {side: (np.array([-2.4, -.1, -.1, -1.6, -1.7, -2.5]),
                              np.array([2.4, 3.2, 4, 1.6, 1.7, 2.5])) for side in ('left', 'right')}
        while not stopping:
            tick = time.monotonic()
            if store.failure:
                raise RuntimeError(store.failure)
            if cameras and not all(c['ok'] for c in cameras.status().values()):
                raise RuntimeError(f'Camera stream lost: {cameras.status()}')
            if args.demo:
                sample = simulated_sample(started)
                for side, position in demo_gripper_position.items():
                    sample['arms'][side]['gripper']['position'] = position
                if demo_return_pose is not None:
                    for side, q in demo_return_pose.items():
                        sample['arms'][side]['position_rad'] = q.tolist()
                        sample['arms'][side]['velocity_rad_s'] = [0.0]*6
            else:
                samples = {side: feedback(robot, side, stale) for side, robot in robots.items()}
                sample = dict(monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns(),
                              arms={side: arm for side, (_, arm) in samples.items()})
            sample['foot_pedal'] = pedal.snapshot()
            store.update(foot_pedal=sample['foot_pedal'])
            while True:
                try:
                    alignment_enabled = store.alignment_requests.get_nowait()
                except queue.Empty:
                    break
                alignment.set_enabled(alignment_enabled)
                vision.set_enabled(alignment_enabled)
            while True:
                try:
                    hold_request = store.alignment_operator_requests.get_nowait()
                except queue.Empty:
                    break
                operator_lease.update(hold_request['received_monotonic_ns']/1e9,
                    hold_request['episode_id'], hold_request['press_id'], hold_request['held'])
            while True:
                try:
                    trial_settings = store.alignment_settings_requests.get_nowait()
                except queue.Empty:
                    break
                if alignment_config.get('mode') == 'operator_supervised' and not operator_lease.active(tick, store.snapshot()['active']):
                    if 'max_joint_distance_rad' in trial_settings:
                        alignment.config['calibration']['max_joint_distance_rad'] = trial_settings['max_joint_distance_rad']
                    if 'max_error_px' in trial_settings:
                        alignment.config['max_error_px'] = trial_settings['max_error_px']
                    store.submit('handoff_event', dict(event='supervised_alignment_settings', settings=trial_settings))
                    alignment_readiness_updated = -float('inf')
            try:
                request = store.control_requests.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:
                limits = demo_limits if args.demo else {side: (r.joint_limits['lower'], r.joint_limits['upper'])
                                                        for side, r in robots.items()}
                neutral = NeutralReturn(config['neutral_return'], limits, tick)
                hold = None
                reset_grippers = None
                store.update(gripper_reset=None, return_gripper_error=None)
                handoff_recording_released = False
                return_request = request
                return_path_checked = False
                demo_return_pose = {s: np.asarray(a['position_rad']) for s, a in sample['arms'].items()} if args.demo else None
                print(f"Episode {request['episode_id']} labeled {request['label']}; returning to raised neutral after release countdown.", flush=True)
            return_targets = None
            released = {}
            if neutral is not None:
                try:
                    return_targets, complete, stage = neutral.tick(tick, sample['arms'])
                    if neutral.initial is not None and not return_path_checked:
                        for side, r in robots.items():
                            validate_return_path(r, neutral.initial[side], neutral.targets[side], side)
                        return_path_checked = True
                    remaining = max(0, config['neutral_return']['release_delay_seconds']-(tick-neutral.requested_at))
                    store.update(return_stage=stage, return_countdown_s=round(remaining, 1))
                    if args.demo and return_targets is not None:
                        demo_return_pose = return_targets
                    if return_targets is not None and reset_grippers is None:
                        gripper_limits = {s: dict(lower=0.0, upper=2.0) for s in sample['arms']} if args.demo else {
                            s: r.gripper_limits for s, r in robots.items()}
                        reset_grippers = ResetGrippers(config['neutral_return']['gripper_open'], gripper_limits, tick)
                    if complete:
                        hold = NeutralHold(neutral.targets, config['neutral_return'], tick)
                        return_targets = hold.targets
                        store.submit('return_complete', dict(request=return_request,
                            hold_for_ready=True,
                            measured_position_rad={s: a['position_rad'] for s, a in sample['arms'].items()},
                            diagnostics=neutral.diagnostics,
                            target_position_rad=config['neutral_return']['position_rad']))
                        neutral = None
                        print('Raised neutral reached; HOLDING both arms. Pull each handle to release that arm into gravity assistance.', flush=True)
                except (ValueError, RuntimeError) as exc:
                    hold = NeutralHold({s:a['position_rad'] for s,a in sample['arms'].items()},
                                       config['neutral_return'], tick)
                    store.submit('return_complete', dict(request=return_request, error=str(exc),
                        hold_for_ready=True,
                        diagnostics=neutral.diagnostics,
                        measured_position_rad={s: a['position_rad'] for s, a in sample['arms'].items()}))
                    neutral = None
                    return_targets = hold.targets
                    print(f'Neutral return canceled: {exc}. HOLDING measured poses; pull handles to release.', flush=True)
            if hold is not None:
                if args.demo:
                    # Exercise independent handoffs in simulation without motor devices.
                    age = tick-hold.started
                    for side, delay in [('left', 1.5), ('right', 3.0)]:
                        if age >= delay:
                            sample['arms'][side]['measured_torque_nm'][0] = 0.9
                previous_ready = dict(hold.ready)
                ready = hold.tick(tick, sample['arms'])
                released = ready
                return_targets = {s:q for s,q in hold.targets.items() if not ready[s]}
                store.update(handoff_ready=ready)
                if any(ready.values()) and not handoff_recording_released:
                    store.submit('handoff_ready', dict(sample=sample, ready=ready))
                    handoff_recording_released = True
                for side in ready:
                    if ready[side] and not previous_ready[side]:
                        store.submit('handoff_event', dict(event='handle_pull_release', side=side,
                                     request=return_request, diagnostics=hold.diagnostics[side]))
                        print(f'{side}: sustained handle load detected; gravity assistance enabled.', flush=True)
                if all(ready.values()):
                    hold = None
                    return_targets = None
                    if args.demo:
                        demo_return_pose = None
                        started = tick
            gripper_commands = {}
            if reset_grippers is not None:
                previous_status = dict(reset_grippers.status)
                gripper_commands = reset_grippers.tick(tick, sample['arms'], released)
                store.update(gripper_reset=dict(reset_grippers.status))
                for side, status in reset_grippers.status.items():
                    if status != previous_status[side]:
                        store.submit('handoff_event', dict(event='reset_gripper_'+status, side=side,
                            request=return_request, measured_position_rad=sample['arms'][side]['gripper']['position'],
                            target_position_rad=reset_grippers.targets[side]))
                        if status == 'timeout':
                            store.update(return_gripper_error=f'{side}: gripper did not reach reset opening; released to passive control')
                if args.demo:
                    for side, command in gripper_commands.items():
                        demo_gripper_position[side] = min(command['position_rad'],
                            demo_gripper_position[side]+command['speed_limit_rad_s']/config['control_hz'])
            alignment_status = dict(enabled=alignment_enabled, active=False, reason='disabled',
                                    mode=alignment_config.get('mode', 'automatic'))
            grasp_episode = store.snapshot()['active']
            grasp_allowed = (store.snapshot()['phase'] == 'recording' and neutral is None
                             and hold is None and not gripper_commands and alignment_config.get('mode') == 'operator_supervised')
            while True:
                try:
                    grasp_request = store.gear_grasp_requests.get_nowait()
                except queue.Empty:
                    break
                if grasp_request['action'] == 'release':
                    if grasp_request['episode_id'] == gear_grasp.episode:
                        gear_grasp.cancel()
                elif (grasp_allowed and grasp_request['episode_id'] == grasp_episode
                      and 0 <= time.monotonic()-grasp_request['received_monotonic_ns']/1e9 <= .5):
                    try:
                        gear_grasp.start(tick, grasp_episode, sample['arms']['left']['gripper'],
                            robots['left'].gripper_limits if 'left' in robots else dict(lower=0., upper=2.))
                        store.update(gear_grasp_error=None)
                    except ValueError as exc:
                        store.update(gear_grasp_error=str(exc))
            gear_grasp_command = gear_grasp.tick(tick, grasp_episode, sample['arms']['left']['gripper'], grasp_allowed)
            store.update(gear_grasp=gear_grasp.status())
            sample['gear_grasp'] = gear_grasp.status()
            if alignment_config.get('mode') == 'operator_supervised':
                alignment_status['target_error_px'] = alignment_config['calibration']['target_error_px']
                alignment_status['trial_limits'] = dict(max_joint_distance_rad=alignment.config['calibration']['max_joint_distance_rad'],
                                                       max_error_px=alignment.config['max_error_px'])
            servo_target = None
            if alignment_enabled:
                if cameras:
                    vision.submit(cameras.recording_frames(), sample['arms'], sample['monotonic_ns'])
                prediction = vision.snapshot()
                side = (alignment_config.get('calibration') or {}).get('side')
                episode_id = store.snapshot()['active']
                while True:
                    try:
                        servo_request = store.alignment_servo_requests.get_nowait()
                    except queue.Empty:
                        break
                    if not servo_request['enabled']:
                        alignment_servo = None
                        alignment_servo_status = dict(active=False, reason='released')
                    elif (not args.demo and alignment_servo is None and grasp_allowed and gear_grasp.settled
                          and servo_request['episode_id'] == episode_id
                          and 0 <= time.monotonic()-servo_request['received_monotonic_ns']/1e9 <= .5):
                        alignment_servo = CartesianNudge(alignment.config, tick, sample['arms'],
                            robots['left'].joint_limits, config['neutral_return']['torque_limits_nm'],
                            kinematics=ToolKinematics(yaml.safe_load((setup.CONFIG/'left.yaml').read_text())))
                if alignment_servo is not None:
                    try:
                        servo_target = alignment_servo.tick(tick, prediction, sample['arms'],
                            allowed=grasp_allowed and gear_grasp.settled)
                        alignment_servo_status = alignment_servo.status()
                    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                        alignment_servo = None
                        alignment_servo_status = dict(active=False, reason=str(exc))
                if side in robots:
                    collecting_alignment = (store.snapshot()['phase'] == 'recording' and neutral is None and
                        hold is None and not gripper_commands and
                        (gear_grasp_command is None or gear_grasp.settled) and servo_target is None)
                    alignment_status.update(alignment.evaluate(tick, prediction, sample['arms'][side],
                        robots[side].joint_limits,
                        other_arm=sample['arms']['left' if side == 'right' else 'right'], collecting=collecting_alignment,
                        operator_confirmed=operator_lease.active(tick, episode_id),
                        operator_press=operator_lease.press))
                    if alignment_config.get('mode') == 'operator_supervised' and tick-alignment_readiness_updated >= .2:
                        # Separate controller instance: diagnostics cannot arm or
                        # modify the real burst, and this torque is never sent.
                        alignment_readiness = AlignmentAssist(alignment.config).evaluate(tick, prediction,
                            sample['arms'][side], robots[side].joint_limits,
                            other_arm=sample['arms']['left' if side == 'right' else 'right'],
                            collecting=collecting_alignment, operator_confirmed=True, operator_press='diagnostic-only')
                        alignment_readiness_updated = tick
                    if alignment_readiness is not None:
                        alignment_status['readiness'] = alignment_readiness
                    if operator_lease.active(tick, episode_id) and alignment_status['reason'] != 'trial_press_exhausted':
                        previous_peak = ((last_alignment_trial or {}).get('peak_torque_nm', 0.)
                            if (last_alignment_trial or {}).get('press_id') == operator_lease.press else 0.)
                        last_alignment_trial = dict(press_id=operator_lease.press,
                            reason=alignment_status['reason'], active=alignment_status['active'],
                            peak_torque_nm=max(previous_peak, max(abs(v) for v in alignment_status['torque_nm'])))
                else:
                    alignment_status['reason'] = 'simulation_no_motion' if args.demo else 'calibration_required'
                alignment_status['vision_status'] = prediction.get('status')
                alignment_status['error'] = prediction.get('error') or prediction.get('errors', {}).get(
                    alignment_config.get('task', 'gear'))
            if last_alignment_trial is not None:
                alignment_status['last_trial'] = last_alignment_trial
            if not alignment_enabled:
                alignment_servo = None
                alignment_servo_status = dict(active=False, reason='disabled')
            sample['alignment_servo'] = alignment_servo_status
            store.update(alignment_servo=alignment_servo_status)
            sample['alignment_assist'] = alignment_status
            store.update(alignment_assist=alignment_status)
            for side, robot in robots.items():
                torque, arm = samples[side]
                if alignment_status['active'] and alignment_status.get('side') == side:
                    assist_torque = np.asarray(alignment_status['torque_nm'], dtype=float)
                    torque = np.clip(torque + assist_torque, -TORQUE_LIMIT, TORQUE_LIMIT)
                    arm['alignment_assist_torque_nm'] = assist_torque.tolist()
                    arm['gravity_torque_command_nm'] = torque.tolist()
                    arm['command']['torque_nm'] = torque.tolist()
                if return_targets is not None and side in return_targets:
                    settings = config['neutral_return']
                    if not robot.Joint_Pos_Vel(return_targets[side], np.full(6, settings['max_joint_speed_rad_s']),
                                               max_tqu=settings['torque_limits_nm'], iswait=False):
                        raise RuntimeError(f'{side}: neutral position command rejected')
                    arm['command'] = dict(mode='joint_position_velocity', position_rad=return_targets[side].tolist(),
                                          speed_limit_rad_s=settings['max_joint_speed_rad_s'],
                                          torque_limit_nm=settings['torque_limits_nm'])
                    arm['gravity_torque_command_nm'] = None
                elif servo_target is not None and side == 'left':
                    if not robot.Joint_Pos_Vel(servo_target, np.full(6, .02),
                            max_tqu=alignment_servo.torque_limits, iswait=False):
                        raise RuntimeError('Left visual-servo position command rejected')
                    arm['command'] = dict(mode='joint_position_velocity', position_rad=servo_target.tolist(),
                        speed_limit_rad_s=.02, torque_limit_nm=alignment_servo.torque_limits.tolist())
                    arm['gravity_torque_command_nm'] = None
                elif not robot.pos_vel_tqe_kp_kd(ZERO, ZERO, torque, ZERO, ZERO):
                    raise RuntimeError(f'{side}: gravity command rejected')
                try:
                    command_gripper(robot, arm, gripper_commands.get(side) or
                                    (gear_grasp_command if side == 'left' else None))
                except RuntimeError as exc:
                    raise RuntimeError(f'{side}: {exc}') from exc
                arm['control_mode'] = 'neutral_return' if neutral is not None else (
                    'neutral_hold' if return_targets is not None and side in return_targets else (
                    'alignment_hover' if servo_target is not None and side == 'left' else (
                    'alignment_assist' if alignment_status['active'] and alignment_status.get('side') == side
                    else 'gravity_assist')))
            sample['control_mode'] = 'neutral_return' if neutral is not None else (
                'partial_neutral_hold' if hold is not None else (
                'alignment_hover' if servo_target is not None else (
                'alignment_assist' if alignment_status['active'] else 'gravity_assist')))
            sample['sequence'] = sequence
            sample['command_sent_monotonic_ns'] = time.monotonic_ns()
            sample['control_compute_s'] = time.monotonic()-tick
            store.submit('sample', sample)
            sequence += 1
            now = time.monotonic()
            if now - last_print >= 5:
                status = store.snapshot()
                print(f"phase={status['phase']} episode={status['active']} samples={status['samples']} " +
                      ' | '.join(f"{side}: q={np.round(arm['position_rad'], 3).tolist()}"
                                 for side, arm in sample['arms'].items()), flush=True)
                last_print = now
            time.sleep(max(0, 1/config['control_hz']-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        print('Hand guidance stopped. Gravity assistance ending; support the arms.', flush=True)
    except Exception as exc:
        reason = f'fault: {exc}'
        if store:
            store.update(error=str(exc), control_active=False)
        message = f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} HAND GUIDANCE STOPPED: {exc}'
        print(message, file=sys.stderr, flush=True)
        try:
            with (setup.ROOT/'logs/hand-guide-events.log').open('a') as log:
                log.write(message + '\n')
        except OSError as log_error:
            print(f'Could not save event: {log_error}', file=sys.stderr, flush=True)
        if is_serial_disconnect(exc):
            print('Robot USB connection lost. Ending control and saving any unfinished '
                  'episode for review. Support the arms, check USB/power, and restart '
                  'the launcher. For idle USB protection run '
                  'sudo ./scripts/enable-robot-usb.sh with control stopped.',
                  file=sys.stderr, flush=True)
            return 1
        raise
    finally:
        if store:
            store.update(control_active=False)
        stop_robots(robots)
        if vision:
            vision.close()
        if cameras:
            cameras.close()
        if store:
            store.close(reason)
        if dashboard:
            dashboard.close()
        if pedal:
            pedal.close()
        lock_fd.close()


if __name__ == '__main__':
    sys.exit(main())
