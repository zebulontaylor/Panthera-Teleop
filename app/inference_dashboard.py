"""Loopback web controls for bounded, operator-stepped physical policy inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import queue
import secrets
import signal
import sys
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from remote_inference import (CONTRACT, ROOT, GRIPPER_ENCODER_TOLERANCE_RAD,
                              SmoothActionTrajectory, check_arm_tracking, image_payload, request, validate_chunk)

from hitl_recording import RolloutStore
from expo_recording import ExplorationRolloutStore
from takeover_hold import TakeoverHold
from rtc_chunking import supports_rtc, prepare_rtc, confirm_rtc
from foot_pedal import FootPedal
from inference_pedal import InferencePedal

LEASE_SECONDS = 2.0
DEFAULT_ACTIONS_PER_STEP = 15


def validate_server_contract(contract):
    """Require runtime compatibility while allowing training provenance to vary."""
    if not isinstance(contract, dict):
        raise RuntimeError('Inference server returned an invalid robot contract')
    provenance = {'dataset_manifest_sha256', 'dataset_mix'}
    mismatched = sorted(key for key in (CONTRACT.keys() | contract.keys()) - provenance
                        if key not in CONTRACT or key not in contract
                        or CONTRACT[key] != contract[key])
    if mismatched:
        raise RuntimeError('Inference server robot contract differs: ' + ', '.join(mismatched))


def measured_hold_target(measured, target, lower, upper, released):
    """Hold arms where they are, preserving squeeze on grippers still under control."""
    result = np.clip(measured, lower, upper)
    for side, index in (('left', 6), ('right', 13)):
        if not released.get(side, False):
            result[index] = target[index]
    return result


class CollectionReset:
    """Collection's measured-start trajectory, capped to inference speeds."""
    def __init__(self, settings, lower, upper, initial, now):
        from neutral_return import NeutralReturn
        # Teleop's selected-arm toggle does not apply to inference resets.
        self.settings = dict(settings, arms=['left', 'right'], release_delay_seconds=0,
                             max_joint_speed_rad_s=min(settings['max_joint_speed_rad_s'],
                                                       CONTRACT['max_joint_speed_rad_s']))
        limits = {s: (lower[o:o+6], upper[o:o+6]) for s, o in [('left', 0), ('right', 7)]}
        self.neutral = NeutralReturn(self.settings, limits, now)
        self.gripper_settings = settings['gripper_open']
        fraction = self.gripper_settings['travel_fraction']
        self.gripper_indices = [6 if side == 'left' else 13 for side in self.neutral.targets]
        indices = self.gripper_indices
        self.gripper_targets = lower[indices] + fraction * (upper[indices] - lower[indices])
        self.started = now

    def tick(self, now, dt, measured, target):
        arms = {s: dict(position_rad=measured[o:o+6], velocity_rad_s=[0]*6)
                for s, o in [('left', 0), ('right', 7)]}
        joints, settled, _ = self.neutral.tick(now, arms)
        # A reset created mid-loop can be newer than this tick's timestamp.
        if joints is None:
            return target.copy(), False
        result = target.copy()
        for side, positions in joints.items():
            offset = 0 if side == 'left' else 7
            result[offset:offset+6] = positions
        speed = min(self.gripper_settings['speed_rad_s'], CONTRACT['gripper_speed_rad_s'])
        indices = self.gripper_indices
        result[indices] += np.clip(self.gripper_targets-target[indices], -speed*dt, speed*dt)
        opened = np.all(np.abs(measured[indices]-self.gripper_targets) <=
                        self.gripper_settings['arrival_tolerance_rad'])
        if not opened and now-self.started > self.gripper_settings['timeout_seconds']:
            raise RuntimeError('Collection reset grippers did not reach their opening; reset stopped')
        commanded_open = np.allclose(result[indices], self.gripper_targets, atol=1e-8, rtol=0)
        return result, bool(settled and opened and commanded_open)


def action_count(value):
    if type(value) is not int or not 1 <= value <= 50:
        raise ValueError('Actions per step must be an integer from 1 to 50')
    return value


def blend_count(value):
    if type(value) is not int or not 0 <= value <= 50:
        raise ValueError('Blend actions must be an integer from 0 to 50')
    return value


def blend_predictions(actions, previous_tail, count):
    """Crossfade aligned absolute targets, then let the bounded path retime them.

    Observations are taken at stopped step boundaries, so old row N aligns with
    new row 0. No wall-clock latency offset applies while the robot is holding.
    This is client-side overlap blending, not model-side RTC guidance.
    """
    count = blend_count(count)
    result = np.asarray(actions, dtype=float).copy()
    if previous_tail is None:
        return result, 0
    overlap = min(count, len(previous_tail), len(result))
    if overlap:
        u = np.arange(1, overlap + 1, dtype=float) / overlap
        weight = (u*u*(3 - 2*u))[:, None]
        result[:overlap] = (1 - weight) * previous_tail[:overlap] + weight * result[:overlap]
    return result, overlap


class ActionStep(SmoothActionTrajectory):
    """Blend exactly N policy waypoints, then stop and hold the last target."""
    def __init__(self, actions, count, started):
        selected = actions[:action_count(count)]
        if len(selected) != count:
            raise ValueError('Prediction contains fewer actions than requested')
        super().__init__(selected, started)


from robot_hardware import RobotHardware

class Controller:
    def __init__(self, url, hardware_factory=RobotHardware, log_path=None, data_dir=None, expo_record_dir=None,
                 expo_objective='gear_and_pin', expo_prefix_length=1, critic_options=None):
        self.url = url
        self.data_dir = Path(data_dir) if data_dir else ROOT / 'data/hitl'
        self.recorder = None
        self.expo_record_dir = Path(expo_record_dir) if expo_record_dir else None
        self.expo_recorder = None
        if expo_objective not in ('gear_insertion', 'gear_and_pin'):
            raise ValueError('Unknown EXPO training objective')
        self.expo_objective = expo_objective
        self.expo_prefix_length = action_count(expo_prefix_length)
        self.collection_commands = queue.Queue(maxsize=1)
        self.hardware_factory = hardware_factory
        self.log_path = log_path or ROOT / 'logs/inference-steps.jsonl'
        self.lock = threading.Lock()
        self.critic_options = critic_options
        self.critic = None
        self._critic_observation = None
        self._critic_counter = 0
        self._critic_history_start = None
        self._critic_rollout_key = secrets.token_hex(8)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.commands = queue.Queue(maxsize=1)
        self.worker = None
        self.hardware = None
        self.heartbeat_at = 0
        self.pedal_context = {}
        self.task_generation = 0
        self.state = dict(task=CONTRACT['task'], supported_tasks=[CONTRACT['task']], mode='stopped', error=None, checkpoint=None, measured=None,
                          target=None, cameras={}, prediction=None, action_index=0,
                          step_count=DEFAULT_ACTIONS_PER_STEP, steps_completed=0, latency_s=None,
                          playing=False, actions_per_step=DEFAULT_ACTIONS_PER_STEP, blend_actions=10,
                          blended_actions=0, foot_pedal_connected=False, foot_pedal_error=None,
                          hold_status='unverified', control_source='hold', rollout_active=False,
                          recording=None, collection_pending=False, handoff_ready={}, handoff_baseline_ready=False, handoff_gripper_enabled=True)
        self.state['critic'] = dict(status='idle' if critic_options is not None else 'disabled')
        self.state['critic_history'] = []
        self.state['critic_samples'] = 0
        if self.expo_record_dir is not None:
            self.state.update(actions_per_step=self.expo_prefix_length, step_count=self.expo_prefix_length,
                              blend_actions=0, expo_recording_enabled=True, expo_prefix_length=self.expo_prefix_length,
                              expo_objective=expo_objective)

    def check_expo_configuration(self, count, blend):
        if self.expo_record_dir is not None and (count != self.expo_prefix_length or blend != 0):
            raise ValueError(f'EXPO collection requires exactly {self.expo_prefix_length} actions per step and 0 blended actions')

    @staticmethod
    def expo_images(hardware, cameras):
        if hasattr(hardware, 'expo_images'):
            return hardware.expo_images()
        return hardware.images(), dict(camera_frame_metadata_exact=False, status=cameras)

    def update(self, **values):
        if 'prediction' in values and values['prediction'] is None:
            self.clear_critic()
        with self.lock:
            self.state.update(values)

    def clear_critic(self):
        with self.lock:
            self._critic_observation = None
            self.state['critic'] = dict(status='idle' if self.critic_options is not None else 'disabled')
        if self.critic is not None:
            self.critic.invalidate()

    def reset_critic_history(self, started_at=None):
        self.clear_critic()
        with self.lock:
            self._critic_history_start = started_at
            self._critic_rollout_key = secrets.token_hex(8)
            self.state['critic_history'] = []
            self.state['critic_samples'] = 0

    def capture_observation(self, hardware, cameras):
        if self.critic_options is not None and hasattr(hardware, 'inference_images'):
            return hardware.inference_images()
        images, receipts = self.expo_images(hardware, cameras)
        return images, receipts, None

    def _critic_result(self, result):
        with self.lock:
            if result.get('observation_key') != self._critic_observation or self._critic_observation is None:
                return
            self.state['critic'] = dict(result, received_at=time.time())
            if result.get('status') == 'ok' and all(
                    isinstance(result.get(key), (int, float)) and np.isfinite(result[key])
                    for key in ('q', 'current_v', 'delta')):
                sampled_at = result.get('sampled_at', time.time())
                if self._critic_history_start is None:
                    self._critic_history_start = sampled_at
                point = dict(t=max(0., sampled_at - self._critic_history_start),
                             q=result['q'], current_v=result['current_v'], delta=result['delta'],
                             sequence=result.get('sequence'))
                self.state['critic_history'].append(point)
                self.state['critic_samples'] += 1
                if len(self.state['critic_history']) > 1200:
                    self.state['critic_history'] = self.state['critic_history'][::2]
        try:
            path = Path(self.log_path).with_name('inference-critic.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(dict(result, unix_ns=time.time_ns()), allow_nan=False) + '\n')
        except (OSError, ValueError):
            pass  # Diagnostic storage must not affect control.

    def score_prediction(self, images, state, actions, fps, sequence, executed_count):
        if self.critic_options is None:
            return
        with self.lock:
            self._critic_counter += 1
            observation_key = f'{os.getpid()}:{self._critic_counter}'
            self._critic_observation = observation_key
            self.state['critic'] = dict(status='scoring', sequence=sequence)
        try:
            if images is None:
                raise ValueError('Full-frame camera observation unavailable')
            if self.critic is None:
                from action_critic_monitor import ActionCriticMonitor
                self.critic = ActionCriticMonitor(callback=self._critic_result, **self.critic_options)
            self.critic.submit(images, state, actions, fps=fps, observation_key=observation_key,
                               rollout_key=self._critic_rollout_key,
                               sequence=sequence, executed_count=executed_count,
                               scope='proposed_one_second_policy_trajectory',
                               sampled_at=time.time())
        except Exception as exc:
            self._critic_result(dict(status='error', reason=str(exc), observation_key=observation_key))

    def snapshot(self):
        with self.lock:
            state = json.loads(json.dumps(self.state))
            state['connected'] = self.worker is not None and self.worker.is_alive()
            if self.recorder:
                state['recording'] = self.recorder.snapshot()
            if self.expo_recorder:
                state['expo_recording'] = self.expo_recorder.snapshot()
            return state

    def cue(self, name):
        if self.recorder:
            self.recorder.audio.play(name)

    def heartbeat(self, pedal_context=None):
        with self.lock:
            self.heartbeat_at = time.monotonic()
            if isinstance(pedal_context, dict):
                self.pedal_context = dict(pedal_context)

    def pedal_gesture(self, gesture):
        """Route pedal gestures through the same guards as dashboard controls."""
        with self.lock:
            mode = self.state['mode']
            playing = self.state['playing']
            active = self.state['rollout_active']
            pending = self.state['collection_pending']
            context = dict(self.pedal_context)
            count = context.get('count', self.state['actions_per_step'])
            blend = context.get('blend', self.state['blend_actions'])
            stale = time.monotonic() - self.heartbeat_at > LEASE_SECONDS
        if stale or self.pause_event.is_set() or pending:
            return
        names = {key: context.get(key, '') for key in ('operator', 'scene')}
        if gesture == 'review':
            if not active:
                raise RuntimeError('No active rollout to save for Review')
            outcome = dict(label='review', success=False, quality=context.get('quality', 2),
                           notes=context.get('notes', ''), milestones=context.get('milestones', {}),
                           reason='foot_pedal_review', reset_after_finish=True)
            if self.expo_record_dir is not None:
                outcome['review_approved'] = True
            self.collection('finish', outcome)
        elif gesture == 'single':
            self.collection('resume' if mode in ('takeover_hold', 'human') else 'intervene', names)
        elif gesture == 'double':
            if playing:
                self.stop()
            elif mode == 'holding':
                count, blend = action_count(count), blend_count(blend)
                self.check_expo_configuration(count, blend)
                if active:
                    self.submit('play', count, blend)
                else:
                    # Begin recording before playback, just as browser Play does.
                    self.collection('begin', dict(names, pedal_play=dict(count=count, blend=blend)))


    def connect(self, reset_pose=True):
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise RuntimeError('A robot session is already active')
            self.start_session(reset_pose=reset_pose)

    def start_session(self, reset_pose):
        """Called with self.lock held, after the previous worker has exited."""
        self.stop_event.clear()
        self.pause_event.clear()
        self.commands = queue.Queue(maxsize=1)
        self.collection_commands = queue.Queue(maxsize=1)
        self.heartbeat_at = time.monotonic()
        self.state.update(mode='connecting', error=None, prediction=None,
                          measured=None, target=None, cameras={}, action_index=0,
                          playing=False, blended_actions=0,
                          hold_status='unverified', control_source='hold', rollout_active=False,
                          collection_pending=False, handoff_ready={}, handoff_baseline_ready=False)
        self.worker = threading.Thread(target=self.run, args=(reset_pose,),
                                       daemon=True, name='policy-control')
        self.worker.start()

    def configure(self, count, blend):
        count, blend = action_count(count), blend_count(blend)
        self.check_expo_configuration(count, blend)
        with self.lock:
            if self.state['mode'] not in ('holding', 'queued', 'predicting', 'executing'):
                raise RuntimeError('Connect and wait for the arms to hold before configuring playback')
            self.state.update(actions_per_step=count, blend_actions=blend)

    def select_task(self, task):
        with self.lock:
            if not isinstance(task, str) or task not in self.state['supported_tasks']:
                raise ValueError('Choose a task supported by the connected checkpoint')
            if (self.state['mode'] != 'holding' or self.state['playing'] or
                    self.state['rollout_active'] or self.state['collection_pending'] or
                    not self.commands.empty() or not self.collection_commands.empty()):
                raise RuntimeError('Pause in hold and finish the rollout before changing tasks')
            if task == self.state['task']:
                return
            self.state.update(task=task, prediction=None, action_index=0, blended_actions=0,
                              latency_s=None)
            self.task_generation += 1
        self.reset_critic_history()

    def submit(self, kind, count=None, blend=None):
        if kind not in ('step', 'play', 'predict', 'reset'):
            raise ValueError('Unknown inference command')
        with self.lock:
            # A resume without new settings keeps the operator's last selection.
            count = action_count(self.state['actions_per_step'] if count is None else count)
            blend = blend_count(self.state['blend_actions'] if blend is None else blend)
            if kind in ('step', 'play'):
                self.check_expo_configuration(count, blend)
            if kind == 'reset' and self.state['rollout_active']:
                raise RuntimeError('Finish the rollout before resetting')
            if (self.state['mode'] != 'holding' or self.state['playing'] or self.stop_event.is_set() or
                    self.pause_event.is_set()):
                raise RuntimeError('Connect and wait for the arms to hold before taking a step')
            if time.monotonic() - self.heartbeat_at > LEASE_SECONDS:
                raise RuntimeError('Browser connection is stale; motion remains paused')
            if (self.expo_record_dir is not None and kind in ('step', 'play') and
                    not self.state['rollout_active']):
                raise RuntimeError('Record an EXPO trial before playing or stepping the policy')
            try:
                self.commands.put_nowait((kind, count))
            except queue.Full:
                raise RuntimeError('A command is already pending')
            self.state['mode'] = 'queued'
            self.state['error'] = None
            self.state['playing'] = kind == 'play'
            if kind in ('step', 'play'):
                self.state.update(actions_per_step=count, blend_actions=blend)
        if kind == 'play':
            self.cue('policy_play')

    def collection(self, action, payload=None):
        payload = payload or {}
        if self.expo_record_dir is not None and any(payload.get(key, self.expo_objective) != self.expo_objective
                for key in ('objective', 'training_objective')):
            raise ValueError('Trial objective must match the configured EXPO collection objective')
        if action not in ('begin', 'intervene', 'resume', 'finish'):
            raise ValueError('Unknown collection action')
        if action == 'begin':
            for key in ('operator', 'scene'):
                if not isinstance(payload.get(key), str) or not 1 <= len(payload[key].strip()) <= 120:
                    raise ValueError(f'Enter an {key} name (1–120 characters)')
        if action == 'finish':
            if type(payload.get('reset_after_finish', False)) is not bool:
                raise ValueError('reset_after_finish must be a boolean')
            milestones = payload.get('milestones', {})
            if (not isinstance(milestones, dict) or
                    any(key not in ('gear_pickup', 'gear_transport', 'gear_insertion', 'gear_seating',
                                    'pin_pickup', 'pin_insertion', 'pin_pushing') or value not in ('success', 'failure')
                        for key, value in milestones.items())):
                raise ValueError('Choose success or failure for known task milestones')
            if payload.get('label') not in ('kept', 'review', 'discarded'):
                raise ValueError('Choose keep, review, or discard')
            if type(payload.get('success')) is not bool:
                raise ValueError('Mark success or failure')
            if type(payload.get('quality')) is not int or not 1 <= payload['quality'] <= 5:
                raise ValueError('Quality must be an integer from 1 to 5')
            if not isinstance(payload.get('notes', ''), str) or len(payload.get('notes', '')) > 1000:
                raise ValueError('Notes must be at most 1000 characters')
            if 'review_approved' in payload and not (self.expo_record_dir is not None and
                    payload.get('label') == 'review' and payload.get('success') is False and
                    payload['review_approved'] is True):
                raise ValueError('Review approval is available only for an explicitly failed EXPO trial')
        with self.lock:
            if self.state['collection_pending']:
                raise RuntimeError('A collection change is already pending')
            if self.state['mode'] not in ('holding', 'queued', 'predicting', 'executing', 'takeover_hold', 'human'):
                raise RuntimeError('Connect and wait for the collection pose')
            if time.monotonic() - self.heartbeat_at > LEASE_SECONDS or self.pause_event.is_set():
                raise RuntimeError('Browser connection is stale or motion is stopping')
            if action == 'begin' and (self.state['mode'] != 'holding' or self.state['rollout_active']):
                raise RuntimeError('Begin from hold with no active rollout')
            if action == 'finish' and not self.state['rollout_active']:
                raise RuntimeError('No correction or rollout is recording')
            if action == 'resume' and self.state['mode'] not in ('takeover_hold', 'human'):
                raise RuntimeError('Resume is available during human correction')
            if action == 'intervene' and self.state['mode'] in ('takeover_hold', 'human'):
                raise RuntimeError('Human correction is already active')
            if action in ('begin', 'finish') and (self.recorder is None or self.recorder.failure):
                raise RuntimeError('Rollout recorder is unavailable')
            try:
                self.collection_commands.put_nowait((action, payload))
                self.state['collection_pending'] = True
            except queue.Full:
                raise RuntimeError('A collection change is already pending')

    def stop(self):
        """Cancel motion, retaining the connection and powered position hold."""
        with self.lock:
            audible = self.state['playing'] or self.state['mode'] in ('queued', 'predicting', 'executing')
            self.state['playing'] = False
            self.pause_event.set()
        if audible:
            self.cue('policy_pause')
        self.clear_critic()

    def shutdown(self):
        """Release resources only for a deliberate controller shutdown."""
        self.stop_event.set()
        self.clear_critic()
        if self.critic is not None:
            self.critic.close()

    def check_active(self):
        if self.stop_event.is_set():
            raise RuntimeError('Operator stopped the session')

    def clear_commands(self):
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                return

    def jpeg(self, name):
        hardware = self.hardware
        return hardware.cameras.jpeg(name) if hardware and hardware.cameras else None

    def collection_reset(self, hardware, measured):
        if 'neutral_return' not in hardware.config:
            return None
        reset = CollectionReset(hardware.config['neutral_return'], hardware.lower,
                                hardware.upper, measured, time.monotonic())
        hardware.validate_reset_path(measured, reset.neutral.targets)
        self.update(mode='resetting', prediction=None, action_index=0)
        return reset

    def run(self, reset_pose=True):
        hardware = self.hardware_factory()
        hardware.check_active = self.check_active
        self.hardware = hardware
        pool = ThreadPoolExecutor(max_workers=1)
        future = step = None
        previous_tail = None
        rtc_body = None
        target = None
        sequence = 0
        sample_sequence = 0
        last_policy_clock = None
        expo_feedback_pending = False
        pending_finish_reset = None
        human = rollout = False
        handoff = None
        released = {}
        takeover_details = {}
        completed = self.state['steps_completed']
        pedal = None
        gestures = InferencePedal()
        try:
            health = request(self.url, '/health')
            if health['status'] != 'ready':
                raise RuntimeError(health.get('error') or 'AWS inference is not ready')
            validate_server_contract(health['contract'])
            checkpoint = health['checkpoint']
            rtc_enabled = supports_rtc(health)
            task_echo_required = 'supported_tasks' in health
            supported_tasks = health.get('supported_tasks', [CONTRACT['task']])
            if (not isinstance(supported_tasks, list) or not supported_tasks or
                    any(not isinstance(task, str) or not task.strip() for task in supported_tasks) or
                    CONTRACT['task'] not in supported_tasks):
                raise RuntimeError('Inference server returned invalid supported tasks')
            self.update(task=CONTRACT['task'], supported_tasks=list(dict.fromkeys(supported_tasks)))
            task_generation = self.task_generation
            self.update(blending_method='rtc' if rtc_enabled else 'crossfade')
            if (self.expo_record_dir is not None and health.get('chunk_size') is not None and
                    health['chunk_size'] != self.expo_prefix_length):
                raise RuntimeError('EXPO server chunk size differs from the fixed collection prefix length')
            if self.expo_record_dir is not None and (
                    health.get('training_objective') is not None and
                    health['training_objective'] != self.expo_objective or
                    health.get('rl_model_id') is not None and
                    health.get('training_objective') != self.expo_objective):
                raise RuntimeError('EXPO server training objective differs from the collection objective')
            expo_policy_identity = {key: health[key] for key in
                                    ('checkpoint_id', 'processor_id', 'normalizer_hash', 'rl_model_id', 'training_objective')
                                    if health.get(key) is not None}
            if self.expo_record_dir is not None:
                self.expo_recorder = ExplorationRolloutStore(self.expo_record_dir,
                    dict(checkpoint=checkpoint, checkpoint_id=health.get('checkpoint_id'),
                         processor_id=health.get('processor_id'), contract=health['contract'],
                         policy_bundle_id=health.get('policy_bundle_id'), server_url=self.url,
                         policy_identity=expo_policy_identity, server_health=health,
                         training_objective=self.expo_objective,
                         fixed_prefix_length=self.expo_prefix_length,
                         simulated=getattr(hardware, 'simulated', False)))
            self.update(checkpoint=checkpoint)
            self.check_active()
            hardware.open()
            try:
                pedal = FootPedal(hardware.config.get('foot_pedal'),
                                  enabled=not getattr(hardware, 'simulated', False))
                self.update(foot_pedal_connected=bool(pedal.devices), foot_pedal_error=None)
            except RuntimeError as exc:
                self.update(foot_pedal_connected=False, foot_pedal_error=str(exc))
            self.check_active()
            if hardware.cameras is not None:
                recording_context = hardware.recording_context(checkpoint)
                recording_context['contract'] = health['contract']
                self.recorder = RolloutStore(self.data_dir, hardware.config, recording_context)
                hardware.cameras.store = self.recorder
            measured, arms, cameras = hardware.read()
            target = measured.copy()
            target[[6, 13]] = np.clip(target[[6, 13]], hardware.lower[[6, 13]], hardware.upper[[6, 13]])
            self.update(measured=measured.tolist(), target=target.tolist(), cameras=cameras,
                        simulation=getattr(hardware, 'simulated', False))
            reset = self.collection_reset(hardware, measured) if reset_pose else None
            if reset is None:
                self.update(mode='holding')
            last_tick = last_publish = time.monotonic()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a') as log:
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    with self.lock:
                        if task_generation != self.task_generation:
                            previous_tail = None
                            task_generation = self.task_generation
                    dt = now - last_tick
                    last_tick = now
                    self.check_active()
                    measured, arms, cameras = hardware.read()
                    measured_at = time.monotonic_ns()
                    gesture = None
                    if pedal is not None:
                        try:
                            gesture = gestures.update(pedal.snapshot())
                        except RuntimeError as exc:
                            self.stop()
                            pedal.close()
                            pedal = None
                            gestures.reset()
                            self.update(foot_pedal_connected=False, foot_pedal_error=str(exc))
                    expo_next_images = None
                    expo_next_observation_id = None
                    # This feedback reflects the preceding command, including
                    # the final waypoint read on the tick after completion.
                    policy_clock = last_policy_clock
                    last_policy_clock = None
                    with self.lock:
                        browser_lost = time.monotonic() - self.heartbeat_at > LEASE_SECONDS
                    if gesture and not browser_lost:
                        try:
                            self.pedal_gesture(gesture)
                        except (RuntimeError, ValueError) as exc:
                            self.update(foot_pedal_error=str(exc))
                            self.cue('cancel')
                        else:
                            self.update(foot_pedal_error=None)
                    if self.pause_event.is_set() or browser_lost:
                        gestures.reset()
                        gesture = None
                        pending_finish_reset = None
                        if self.expo_recorder:
                            self.expo_recorder.boundary('browser_connection_lost' if browser_lost else 'operator_stop')
                            if browser_lost:
                                self.expo_recorder.end(dict(label='review', success=None, complete=False,
                                                            reason='browser_connection_lost'))
                        expo_feedback_pending = False
                        policy_clock = None
                        if future is not None:
                            future.cancel()
                            sequence += 1
                        if human:
                            target = measured_hold_target(measured, target, hardware.lower, hardware.upper, released)
                        human = False
                        handoff = None
                        released = {}
                        self.update(handoff_ready={}, handoff_baseline_ready=False)
                        if browser_lost and rollout:
                            self.recorder.submit('rollout_end', dict(label='review', success=False,
                                quality=1, complete=False, reason='browser_connection_lost'))
                            rollout = False
                        self.update(rollout_active=rollout)
                        while not self.collection_commands.empty():
                            self.collection_commands.get_nowait()
                        future = step = reset = None
                        previous_tail = None
                        self.clear_commands()
                        self.pause_event.clear()
                        self.update(mode='holding', playing=False, blended_actions=0,
                                    prediction=None, action_index=0, control_source='hold', collection_pending=False,
                                    error='Browser connection lost; motion canceled, arms remain held'
                                    if browser_lost else 'Motion stopped; arms remain held')
                    try:
                        if dt > 0.1:
                            raise RuntimeError('Local control loop missed its deadline; motion canceled')
                        if (rollout or human or reset is not None or step is not None or future is not None or not self.commands.empty()) and not all(
                                c['ok'] and c['age_s'] < CONTRACT['max_camera_age_s'] for c in cameras.values()):
                            raise RuntimeError('Camera stream missing or stale; motion canceled')
                        if not human:
                            try:
                                check_arm_tracking(target, measured)
                            except RuntimeError as exc:
                                target = np.clip(measured, hardware.lower, hardware.upper)
                                raise RuntimeError(f'{exc}; holding measured pose') from exc
                        if self.recorder and self.recorder.failure:
                            raise RuntimeError(f'Recording failed: {self.recorder.failure}')
                        if self.expo_recorder and self.expo_recorder.failure:
                            raise RuntimeError(f'EXPO recording failed: {self.expo_recorder.failure}')
                        if expo_feedback_pending:
                            next_images, next_cameras, next_critic = self.capture_observation(hardware, cameras)
                            expo_next_images = (next_images, next_cameras, next_critic)
                            expo_next_observation_id = self.expo_recorder.completed(
                                measured.tolist(), next_images, measured_at, next_cameras)
                            expo_feedback_pending = False
                        change = None
                        try:
                            change = self.collection_commands.get_nowait()
                        except queue.Empty:
                            pass
                        if change:
                            policy_clock = None
                            action, details = change
                            details = dict(details, task=self.state['task'])
                            if action == 'finish' and details.get('reset_after_finish', False):
                                details = dict(details, decision_id=secrets.token_hex(12))
                            if self.expo_recorder:
                                if action == 'begin':
                                    self.expo_recorder.begin(details, measured_at)
                                elif action == 'finish':
                                    self.expo_recorder.end(dict(details, complete=True))
                                else:
                                    self.expo_recorder.boundary(action)
                            expo_feedback_pending = False
                            if future is not None:
                                future.cancel()
                                sequence += 1
                            future = step = None
                            previous_tail = None
                            self.clear_commands()
                            target = measured_hold_target(measured, target, hardware.lower, hardware.upper, released)
                            if action == 'begin':
                                self.reset_critic_history(started_at=time.time())
                                if details.get('pedal_play'):
                                    self.update(actions_per_step=details['pedal_play']['count'],
                                                blend_actions=details['pedal_play']['blend'])
                                self.recorder.submit('rollout_begin', dict(
                                    sample=dict(monotonic_ns=measured_at),
                                    details=dict(details, task=self.state['task'], actions_per_step=self.state['actions_per_step'],
                                                 blend_actions=self.state['blend_actions'])))
                                rollout = True
                            elif action == 'intervene':
                                settings = hardware.config.get('takeover_hold', hardware.config.get('neutral_return'))
                                if settings is None:
                                    raise RuntimeError('Handle-pull takeover is not configured')
                                handoff = TakeoverHold({side: target[offset:offset+6]
                                    for side, offset in (('left', 0), ('right', 7))}, settings, hardware.config, now)
                                released = dict(handoff.ready)
                                takeover_details = details
                                if hasattr(hardware, 'prepare_handoff'):
                                    hardware.prepare_handoff(now)
                                human = True
                            elif action == 'resume':
                                human = False
                                # Explicit Play after regripping avoids restarting while hands are on the arms.
                            elif action == 'finish':
                                human = rollout = False
                                self.recorder.submit('rollout_end', details)
                            if not human:
                                handoff = None
                                released = {}
                            self.update(mode='takeover_hold' if human else 'holding', playing=False,
                                        handoff_ready=released, handoff_baseline_ready=False,
                                        rollout_active=rollout, control_source='hold',
                                        prediction=None, action_index=0, error=None, collection_pending=False)
                            if action in ('intervene', 'resume'):
                                self.cue('intervention_start' if action == 'intervene' else 'intervention_end')
                            if action == 'begin' and details.get('pedal_play'):
                                self.submit('play', **details['pedal_play'])
                            if (action == 'finish' and details.get('reset_after_finish', False)
                                    and not self.pause_event.is_set()):
                                # Let the writer confirm the outcome is saved before moving.
                                pending_finish_reset = details['decision_id']
                                self.update(mode='saving')
                        if pending_finish_reset is not None:
                            saved = self.recorder.snapshot().get('last_decision') or {}
                            if saved.get('decision_id') == pending_finish_reset:
                                pending_finish_reset = None
                                self.check_active()
                                if not self.pause_event.is_set():
                                    reset = self.collection_reset(hardware, measured)
                                    if reset is None:
                                        raise RuntimeError('Recording finished, but collection reset is not configured')
                        if human:
                            previous_ready = released.copy()
                            released = handoff.tick(now, arms)
                            for side, offset in (('left', 0), ('right', 7)):
                                # Keep the policy's grip until this arm is explicitly released.
                                if released[side]:
                                    target[offset:offset+7] = np.clip(measured[offset:offset+7],
                                        hardware.lower[offset:offset+7], hardware.upper[offset:offset+7])
                            if any(released.values()) and not rollout and self.recorder:
                                self.reset_critic_history(started_at=time.time())
                                names = {key: (takeover_details.get(key, '').strip()[:120] or 'unspecified')
                                         if isinstance(takeover_details.get(key, ''), str) else 'unspecified'
                                         for key in ('operator', 'scene')}
                                self.recorder.submit('rollout_begin', dict(sample=dict(monotonic_ns=measured_at),
                                    details=dict(names, task=self.state['task'], recording_start='human_takeover',
                                        handoff='handle_pull_or_gripper_torque_change', handoff_settings=dict(handoff.settings, **handoff.motion_settings),
                                        actions_per_step=self.state['actions_per_step'],
                                        blend_actions=self.state['blend_actions'])))
                                rollout = True
                            for side in released:
                                if released[side] and not previous_ready[side]:
                                    self.cue('handoff')
                                    event = dict(event=handoff.diagnostics[side]['trigger']+'_release', side=side, monotonic_ns=measured_at,
                                                 **handoff.diagnostics[side])
                                    log.write(json.dumps(event) + '\n')
                                    log.flush()
                                    if rollout:
                                        self.recorder.submit('handoff_event', event)
                            self.update(mode='human' if any(released.values()) else 'takeover_hold',
                                rollout_active=rollout, handoff_ready=released,
                                handoff_baseline_ready=len(handoff.baselines) == len(released))
                        if reset is not None:
                            target, reset_done = reset.tick(now, dt, measured, target)
                            if reset_done:
                                log.write(json.dumps(dict(unix_ns=time.time_ns(), event='collection_reset_completed',
                                                          target=target.tolist())) + '\n')
                                log.flush()
                                reset = None
                                self.update(mode='holding')
                                self.cue('end')
                        if not human and reset is None and future is None and step is None:
                            command = None
                            try:
                                command = self.commands.get_nowait()
                            except queue.Empty:
                                with self.lock:
                                    if self.state['playing'] and not self.pause_event.is_set():
                                        command = ('play', self.state['actions_per_step'])
                            if command is not None:
                                kind, count = command
                                with self.lock:
                                    blend = self.state['blend_actions']
                                    if task_generation != self.task_generation:
                                        previous_tail = None
                                        task_generation = self.task_generation
                                    if kind == 'play':
                                        count = self.state['actions_per_step']
                                if kind != 'play':
                                    previous_tail = None
                                if kind == 'reset':
                                    reset = self.collection_reset(hardware, measured)
                                    if reset is None:
                                        raise RuntimeError('Collection reset is not configured')
                                else:
                                    if self.expo_recorder and kind in ('step', 'play'):
                                        images, request_cameras, critic_images = (expo_next_images if expo_next_images is not None
                                                                  else self.capture_observation(hardware, cameras))
                                    else:
                                        images, request_cameras, critic_images = self.capture_observation(hardware, cameras)
                                    observed_at = time.monotonic()
                                    inference_body = dict(sequence=sequence, state=measured.tolist(), images=images,
                                                          task=self.state['task'])
                                    rtc_body = None
                                    if rtc_enabled and kind == 'play' and not self.expo_recorder:
                                        rtc_body = prepare_rtc(previous_tail, blend)
                                        if rtc_body is not None:
                                            inference_body['rtc'] = rtc_body
                                    if self.expo_recorder and kind in ('step', 'play'):
                                        observation_id = self.expo_recorder.requested(sequence, measured.tolist(), images,
                                            measured_at, request_cameras, expo_next_observation_id)
                                        if observation_id is not None:
                                            inference_body['observation_id'] = observation_id
                                    future = pool.submit(request, self.url, '/infer',
                                                         inference_body, 3)
                                    self.update(mode='predicting', step_count=count, prediction=None, action_index=0)
                        if future is not None:
                            if now - observed_at > CONTRACT['max_observation_age_s']:
                                raise RuntimeError('Inference observation expired')
                            if future.done():
                                result = future.result()
                                if self.expo_recorder and kind in ('step', 'play'):
                                    self.expo_recorder.responded(result)
                                if ((task_echo_required or 'task' in result) and
                                        result.get('task') != inference_body['task']):
                                    raise RuntimeError('Inference response task differs from the selected task')
                                if result['checkpoint'] != checkpoint:
                                    raise RuntimeError('AWS checkpoint changed; reconnect before moving')
                                if self.expo_recorder and any(result.get(key) != value
                                        for key, value in expo_policy_identity.items()):
                                    raise RuntimeError('EXPO policy identity changed; reconnect before moving')
                                latency = time.monotonic() - observed_at
                                log.write(json.dumps(dict(unix_ns=time.time_ns(), kind=kind, count=count,
                                                          latency_s=latency, task=inference_body['task'], measured=measured.tolist(),
                                                          response=result)) + '\n')
                                log.flush()
                                raw_actions = validate_chunk(result, sequence, target, hardware.lower, hardware.upper,
                                                         action_count=count if kind == 'step' or self.expo_recorder else None)
                                if (self.expo_recorder and result.get('chunk_size') is not None and
                                        result['chunk_size'] != self.expo_prefix_length):
                                    raise RuntimeError('EXPO response chunk size differs from the fixed collection prefix length')
                                tail = None
                                if kind == 'play' and previous_tail is not None:
                                    old_tail, old_observed_at = previous_tail
                                    if now - old_observed_at <= CONTRACT['max_observation_age_s']:
                                        tail = old_tail
                                if rtc_enabled:
                                    overlap = confirm_rtc(result, rtc_body)
                                    actions = raw_actions.copy()
                                else:
                                    actions, overlap = blend_predictions(raw_actions, tail, blend if kind == 'play' else 0)
                                self.update(prediction=actions.tolist(), latency_s=latency,
                                            blended_actions=overlap)
                                self.score_prediction(critic_images, inference_body['state'], actions.tolist(),
                                                      result['fps'], sequence, count if kind in ('step', 'play') else 0)
                                if kind == 'play':
                                    log.write(json.dumps(dict(unix_ns=time.time_ns(), event='play_blend',
                                                              sequence=sequence, count=count,
                                                              blend_actions=blend, blended_actions=overlap,
                                                              method='rtc' if rtc_enabled else 'crossfade',
                                                              actions=actions.tolist())) + '\n')
                                    log.flush()
                                sequence += 1
                                future = None
                                if kind in ('step', 'play'):
                                    step = ActionStep(actions, count, now)
                                    if self.expo_recorder:
                                        self.expo_recorder.accepted(result, actions[:count].tolist())
                                    self.update(mode='executing')
                                else:
                                    self.update(mode='holding')
                        if step is not None:
                            target, done = step.tick(now, dt, target)
                            progress, rate = step.policy_clock(now)
                            last_policy_clock = (sequence - 1, progress, rate)
                            if done:
                                if self.expo_recorder and self.expo_recorder.active:
                                    expo_feedback_pending = True
                                completed += 1
                                self.update(mode='holding', action_index=count, steps_completed=completed)
                                log.write(json.dumps(dict(unix_ns=time.time_ns(), event='step_completed',
                                                          count=count, sequence=sequence - 1)) + '\n')
                                log.flush()
                                step = None
                                previous_tail = (actions[count:].copy(), observed_at) if kind == 'play' else None
                            else:
                                self.update(action_index=step.index + 1)
                    except Exception as exc:
                        gestures.reset()
                        pending_finish_reset = None
                        if self.expo_recorder:
                            try:
                                self.expo_recorder.boundary('control_fault: ' + str(exc))
                                self.expo_recorder.end(dict(label='review', success=None, complete=False,
                                                            reason='control_fault: ' + str(exc)))
                            except RuntimeError:
                                pass
                        expo_feedback_pending = False
                        policy_clock = last_policy_clock = None
                        if future is not None:
                            future.cancel()
                        if human:
                            target = measured_hold_target(measured, target, hardware.lower, hardware.upper, released)
                        human = False
                        handoff = None
                        released = {}
                        self.update(handoff_ready={}, handoff_baseline_ready=False)
                        if rollout and self.recorder and not self.recorder.failure:
                            self.recorder.submit('rollout_end', dict(label='review', success=False,
                                quality=1, complete=False, reason='control_fault: ' + str(exc)))
                        rollout = False
                        self.update(rollout_active=False, control_source='hold', collection_pending=False)
                        future = step = reset = None
                        previous_tail = None
                        self.clear_commands()
                        while not self.collection_commands.empty():
                            self.collection_commands.get_nowait()
                        sequence += 1
                        self.update(mode='holding', playing=False, blended_actions=0,
                                    error=str(exc) + '; command rejected, arms remain held')
                    if self.stop_event.is_set():
                        break
                    if human:
                        hardware.guide(arms, target, released)
                    else:
                        hardware.command(target)
                        if self.expo_recorder and last_policy_clock is not None:
                            self.expo_recorder.command(target.tolist(), time.monotonic_ns(), last_policy_clock[1])
                    source = 'human' if human and any(released.values()) else 'policy' if policy_clock else 'hold'
                    self.update(hold_status=('gravity_assist' if all(released.values()) else 'partial_hold')
                                if source == 'human' else 'commanded', control_source=source)
                    if self.recorder:
                        for side, offset in (('left', 0), ('right', 7)):
                            if not released.get(side, False):
                                arms[side]['gravity_torque_command_nm'] = None
                                arms[side]['command'] = dict(mode='joint_position_velocity',
                                    position_rad=target[offset:offset+6].tolist(),
                                    speed_limit_rad_s=CONTRACT['max_joint_speed_rad_s'],
                                    torque_limit_nm=CONTRACT['torque_limits_nm'])
                                arms[side]['gripper']['command'] = dict(mode='position_velocity',
                                    position_rad=float(target[offset+6]),
                                    speed_limit_rad_s=CONTRACT['gripper_speed_rad_s'],
                                    torque_limit_nm=CONTRACT['gripper_torque_limit_nm'])
                        self.recorder.submit('sample', dict(monotonic_ns=measured_at,
                            unix_ns=time.time_ns(), sequence=sample_sequence, arms=arms,
                            control_source=source, command_target_rad=None if source == 'human' else target.tolist(),
                            control_source_by_arm={side: 'human' if released.get(side, False) else
                                'policy' if source == 'policy' else 'hold' for side in ('left', 'right')},
                            camera_frames=hardware.cameras.recording_frames(),
                            policy_time_s=policy_clock[1] if policy_clock else None,
                            policy_clock_rate=policy_clock[2] if policy_clock else None,
                            control_compute_s=time.monotonic()-now,
                            inference_sequence=policy_clock[0] if policy_clock else sequence-1,
                            actions_per_step=self.state['actions_per_step'], blend_actions=self.state['blend_actions']))
                        sample_sequence += 1
                    if now - last_publish >= 0.05:
                        self.update(measured=measured.tolist(), target=target.tolist(), cameras=cameras)
                        last_publish = now
                    time.sleep(max(0, 1 / hardware.config['control_hz'] - (time.monotonic() - now)))
        except BaseException as exc:
            if self.expo_recorder:
                try:
                    self.expo_recorder.boundary('control_fault: ' + str(exc))
                    self.expo_recorder.end(dict(label='review', success=None, complete=False,
                                                reason='control_fault: ' + str(exc)))
                except RuntimeError:
                    pass
            self.update(mode='fault', playing=False, error=str(exc), rollout_active=False,
                        control_source='hold', handoff_ready={}, handoff_baseline_ready=False)
            if self.recorder:
                try:
                    # Never wait for disk flush before establishing powered fault hold.
                    self.recorder.submit('rollout_end', dict(label='review', success=False,
                        quality=1, complete=False, reason='control_fault: ' + str(exc)))
                except RuntimeError:
                    pass  # Writer failure preserves Review/pending for recovery.
            print(f'Policy session fault: {exc}', flush=True)
            # The backend retains motor ownership when the frontend faults.
            # Legacy test/hardware adapters retain their existing hold path.
            if target is not None or getattr(hardware, 'robots', {}):
                self.update(mode='fault_hold', hold_status='unverified',
                            error=f'{exc}; motion blocked. Reconnect after supporting both arms.')
                while not self.stop_event.is_set():
                    try:
                        if target is None or getattr(hardware, 'motor_client', None) is not None:
                            hardware.hold_available()
                        else:
                            hardware.command(target)
                        self.update(hold_status='commanded')
                    except BaseException as hold_error:
                        self.update(hold_status='failed', error=f'{exc}; hold command failed: {hold_error}. '
                                    'Powered hold is not reliable. Support both arms, check USB/power, '
                                    'then use Reconnect.')
                    self.stop_event.wait(1 / hardware.config['control_hz'])
        finally:
            if pedal is not None:
                pedal.close()
            self.update(foot_pedal_connected=False)
            try:
                if self.expo_recorder:
                    self.expo_recorder.close('session_stopped')
                if self.recorder:
                    self.recorder.close('session_stopped')
                hardware.close()
            except BaseException as exc:
                self.update(mode='fault', error=f'Robot cleanup failed: {exc}')
            finally:
                self.hardware = None
                pool.shutdown(wait=False, cancel_futures=True)
                if self.snapshot()['mode'] != 'fault':
                    self.update(mode='stopped', hold_status='unverified')


def session_process(pipe, url, hardware_factory, health_request=None, data_dir=None, expo_record_dir=None,
                    expo_objective='gear_and_pin', expo_prefix_length=1, critic_options=None):
    """SDK faults, native aborts and blocked calls stay outside the web process."""
    # Ctrl-C reaches the whole foreground process group. The web process requests
    # orderly cleanup; the independent motor backend retains powered hold.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if health_request is not None:
        global request
        request = health_request
    controller = Controller(url, hardware_factory, data_dir=data_dir, expo_record_dir=expo_record_dir,
                            expo_objective=expo_objective, expo_prefix_length=expo_prefix_length,
                            critic_options=critic_options)
    try:
        while True:
            method, args = pipe.recv()
            if method == 'shutdown':
                controller.shutdown()
                if controller.worker:
                    # Recording stores can take up to 16 seconds to flush.
                    controller.worker.join(timeout=20)
                # Native destructors can send on invalid serial descriptors.
                # Process exit closes all descriptors without running them again.
                os._exit(0)
            try:
                result = getattr(controller, method)(*args)
                pipe.send((True, result))
            except Exception as exc:
                pipe.send((False, str(exc)))
    except (EOFError, BrokenPipeError):
        # Parent disappeared: do not silently release an existing powered hold.
        controller.stop()
        if controller.worker:
            controller.worker.join()


class SessionSupervisor:
    def __init__(self, url, hardware_factory=RobotHardware, health_request=None, data_dir=None, expo_record_dir=None,
                 expo_objective='gear_and_pin', expo_prefix_length=1, critic_options=None):
        self.url = url
        self.hardware_factory = hardware_factory
        self.health_request = health_request
        self.data_dir = data_dir
        self.expo_record_dir = expo_record_dir
        self.expo_objective = expo_objective
        self.expo_prefix_length = expo_prefix_length
        self.critic_options = critic_options
        self.lock = threading.RLock()
        self.recovery_cancel = threading.Event()
        self.recovery_worker = None
        self.process = self.pipe = None
        self.state = Controller(url, expo_record_dir=expo_record_dir, expo_objective=expo_objective,
                                expo_prefix_length=expo_prefix_length, critic_options=critic_options).snapshot()
        self.spawn()

    def spawn(self):
        context = multiprocessing.get_context('spawn')
        self.pipe, child_pipe = context.Pipe()
        self.process = context.Process(target=session_process,
                                       args=(child_pipe, self.url, self.hardware_factory,
                                             self.health_request, self.data_dir, self.expo_record_dir,
                                             self.expo_objective, self.expo_prefix_length, self.critic_options), daemon=True)
        self.process.start()
        child_pipe.close()

    def rpc(self, method, *args):
        with self.lock:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError('Robot session exited. Support both arms and use Reconnect.')
            try:
                self.pipe.send((method, args))
                if not self.pipe.poll(1):
                    # Never consume a late reply as the answer to another command.
                    self.pipe.close()
                    raise RuntimeError('Robot session is unresponsive. Support both arms and use Reconnect.')
                ok, result = self.pipe.recv()
            except (OSError, EOFError, ValueError) as exc:
                raise RuntimeError('Robot session connection failed. Support both arms and use Reconnect.') from exc
            if not ok:
                raise RuntimeError(result)
            return result

    def snapshot(self):
        with self.lock:
            recovering = self.recovery_worker is not None and self.recovery_worker.is_alive()
            if not recovering:
                try:
                    self.state = self.rpc('snapshot')
                except RuntimeError as exc:
                    self.state.update(mode='fault', playing=False, hold_status='unverified', error=str(exc),
                                      connected=self.process is not None and self.process.is_alive())
            state = json.loads(json.dumps(self.state))
            state['recovering'] = recovering
            return state

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def connect(self):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Connection recovery is already running')
            self.rpc('connect')

    def heartbeat(self, pedal_context=None):
        with self.lock:
            if not (self.recovery_worker and self.recovery_worker.is_alive()):
                self.rpc('heartbeat', pedal_context)

    def configure(self, count, blend):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before configuring playback')
            self.rpc('configure', count, blend)

    def select_task(self, task):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before changing tasks')
            self.rpc('select_task', task)

    def submit(self, kind, count=None, blend=None):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before taking a step')
            self.rpc('submit', kind, count, blend)

    def collection(self, action, payload=None):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before collecting')
            self.rpc('collection', action, payload)

    def jpeg(self, name):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                return None
            try:
                return self.rpc('jpeg', name)
            except RuntimeError:
                return None

    def stop(self):
        self.recovery_cancel.set()
        with self.lock:
            if not (self.recovery_worker and self.recovery_worker.is_alive()):
                self.rpc('stop')

    def recover(self, arms_supported=False):
        if arms_supported is not True:
            raise ValueError('Support both arms before reconnecting: powered hold will be released')
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Connection recovery is already running')
            if self.snapshot()['mode'] not in ('fault', 'fault_hold', 'holding', 'stopped'):
                raise RuntimeError('Stop motion before reconnecting')
            self.recovery_cancel.clear()
            self.state.update(mode='recovering', prediction=None, hold_status='unverified',
                              error='Releasing old connection; keep both arms supported')
            self.recovery_worker = threading.Thread(target=self.finish_recovery, daemon=True)
            self.recovery_worker.start()

    def release_process(self, timeout=3):
        # Frontend cleanup releases its motion lease. The independent motor
        # backend continues holding the arms throughout a dashboard switch.
        if self.process and self.process.is_alive():
            try:
                self.pipe.send(('shutdown', ()))
            except (OSError, ValueError):
                pass
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            if self.process.is_alive():
                raise RuntimeError('Old robot process did not exit; reconnection blocked')
        if self.pipe:
            self.pipe.close()

    def finish_recovery(self):
        try:
            # HTTP Stop remains responsive while waiting for native cleanup.
            self.release_process()
            with self.lock:
                self.spawn()
                if self.recovery_cancel.is_set():
                    self.state.update(mode='stopped', connected=False,
                                      error='Recovery canceled; keep both arms supported')
                    return
                self.rpc('connect', False)
        except Exception as exc:
            self.update(mode='fault', error=f'Reconnection failed: {exc}', hold_status='unverified')

    def shutdown(self):
        self.recovery_cancel.set()
        if self.recovery_worker:
            self.recovery_worker.join(timeout=8)
        self.release_process(timeout=23)


def make_server(controller, port):
    token = secrets.token_urlsafe(32)
    html_path = Path(__file__).parent / 'web/inference.html'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, mime='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def do_GET(self):
            if not self.trusted():
                return self.send(403, {})
            path = urlsplit(self.path).path
            if path == '/':
                return self.send(200, html_path.read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/status':
                return self.send(200, dict(controller.snapshot(), token=token, fps=CONTRACT['fps'],
                                           speed_limit=CONTRACT['max_joint_speed_rad_s'],
                                           gripper_speed_limit=CONTRACT['gripper_speed_rad_s']))
            if path == '/reference/gear-pin-overhead.jpg':
                return self.send(200, (ROOT / 'app/web/reference/gear-pin-overhead.jpg').read_bytes(),
                                 'image/jpeg')
            if path.startswith('/camera/'):
                name = path.rsplit('/', 1)[-1]
                if name not in CONTRACT['image_crops_xyxy']:
                    return self.send(404, {})
                jpeg = controller.jpeg(name)
                return self.send(200 if jpeg else 503, jpeg or b'', 'image/jpeg')
            self.send(404, {})

        def do_POST(self):
            if (not self.trusted() or self.headers.get('X-Session-Token') != token or
                    self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}')):
                return self.send(403, {'error': 'Invalid session'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2048:
                    raise ValueError('Invalid request size')
                self.connection.settimeout(2)
                payload = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path in ('/api/stop', '/api/pause'):
                    controller.stop()
                elif path == '/api/heartbeat':
                    controller.heartbeat(payload)
                elif path == '/api/connect':
                    controller.connect()
                elif path == '/api/recover':
                    controller.recover(payload.get('arms_supported', False))
                elif path == '/api/reset':
                    controller.submit('reset')
                elif path.startswith('/api/collection/'):
                    controller.collection(path.rsplit('/', 1)[-1], payload)
                elif path == '/api/task':
                    controller.select_task(payload.get('task'))
                elif path == '/api/configure':
                    controller.configure(payload.get('count'), payload.get('blend'))
                elif path in ('/api/predict', '/api/step', '/api/play'):
                    controller.submit(path.rsplit('/', 1)[-1], payload.get('count'), payload.get('blend'))
                else:
                    return self.send(404, {})
                self.send(202, {'accepted': True})
            except (ValueError, TypeError, AttributeError, RuntimeError, queue.Full) as exc:
                self.send(409, {'error': str(exc)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--data-dir', type=Path, help='Raw rollout directory (default data/hitl)')
    parser.add_argument('--expo-record-dir', type=Path,
                        help='Optional separate EXPO decision ledger; fixed action prefix and 0 blending')
    parser.add_argument('--expo-prefix-length', type=int, choices=range(1, 51), default=17,
                        help='Fixed EXPO action chunk length for this session (default 17); cannot change during a session')
    parser.add_argument('--expo-objective', choices=('gear_and_pin', 'gear_insertion'), default='gear_and_pin',
                        help='Fixed EXPO outcome objective; base pi0.5 task prompt is unchanged')
    parser.add_argument('--demo', action='store_true', help='Simulated policy, arms and cameras; no devices opened')
    parser.add_argument('--critic', action=argparse.BooleanOptionalAction, default=True,
                        help='Asynchronous local action-value diagnostic (default enabled)')
    parser.add_argument('--critic-checkpoint', type=Path, default=ROOT/'work/action-critic-mixed-20261029/run/best.pt')
    parser.add_argument('--critic-python', type=Path, default=ROOT/'work/state-critic-20261028/venv/bin/python')
    parser.add_argument('--critic-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    critic_options = dict(checkpoint=args.critic_checkpoint, python=args.critic_python,
                          device=args.critic_device) if args.critic else None
    if args.demo:
        from hitl_demo import DemoHardware, demo_request
        import tempfile
        data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix='panthera-hitl-demo-'))
        controller = SessionSupervisor(args.url, DemoHardware, demo_request, data_dir, args.expo_record_dir,
                                       args.expo_objective, args.expo_prefix_length, critic_options)
        print(f'SIMULATION — recordings: {data_dir}', flush=True)
    else:
        controller = SessionSupervisor(args.url, data_dir=args.data_dir, expo_record_dir=args.expo_record_dir,
                                       expo_objective=args.expo_objective, expo_prefix_length=args.expo_prefix_length,
                                       critic_options=critic_options)
    server = make_server(controller, args.port)
    shutting_down = threading.Event()

    def shutdown(*_):
        if shutting_down.is_set():
            return
        shutting_down.set()

        def finish_shutdown():
            try:
                controller.shutdown()
            finally:
                server.shutdown()

        # Keep blocking RPCs and recording flushes out of the signal handler.
        threading.Thread(target=finish_shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f'Policy step controls: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        controller.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
