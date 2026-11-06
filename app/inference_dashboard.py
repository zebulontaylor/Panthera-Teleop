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
from urllib.parse import urlsplit, parse_qs

import numpy as np

from remote_inference import (CONTRACT, ROOT, GRIPPER_ENCODER_TOLERANCE_RAD,
                              SmoothActionTrajectory, check_arm_tracking, image_payload, request, validate_chunk)

from hitl_recording import RolloutStore
from expo_recording import ExplorationRolloutStore
from takeover_hold import TakeoverHold
from rtc_chunking import supports_rtc, prepare_rtc, confirm_rtc
from foot_pedal import FootPedal
from inference_pedal import InferencePedal
from right_arm_settle import RightArmSettleDetector
from pin_nudge_detection import PinNudgeDetector
from pin_contact_detection import PinContactDetector
from pin_contact_approach import (ContactApproach, APPROACH_DOWN_M, APPROACH_XY_M,
                                  APPROACH_TIMEOUT_S, APPROACH_FORCE_N, APPROACH_FORCE_SLEW_N_S)
from pin_alignment_insert import loaded_xy_nudge_goal

LEASE_SECONDS = 2.0
DEFAULT_ACTIONS_PER_STEP = 10
AUTO_FREEZE_GRIPPER_CLOSE_RAD = .05
AUTO_NUDGE_SPEED_M_S = .010
LOADED_NUDGE_SPEED_M_S = .010


class NudgeReanchorRequired(RuntimeError):
    """A safe held-pose change requires rebuilding the height-locked path."""


def loaded_nudge_progress(job, position, xy_target, now):
    """Report measured convergence, rather than completion of target slew."""
    position = np.asarray(position, dtype=float)
    error = float(np.linalg.norm(job['goal_xy'] - position[:2]))
    slewing = np.linalg.norm(job['goal_xy'] - xy_target) > 1e-10
    if not slewing and error <= .0004:
        if job.get('settled_since') is None:
            job['settled_since'] = now
    else:
        job['settled_since'] = None
    settled = job.get('settled_since')
    done = settled is not None and now - settled >= .2
    status = 'done' if done else 'stalled' if now-job['started'] >= 4. else 'executing'
    reason = ('Measured XY reached target; load remains active' if done else
              'XY target not reached; load remains active' if status == 'stalled' else
              'Slewing XY target; load remains active' if slewing else
              'Tracking XY target; load remains active')
    return dict(status=status, reason=reason, frame_id=job['frame_id'], xy_only=True,
                measured_delta_xyz_mm=((position-job['start_position'])*1000).tolist(),
                tracking_error_mm=error*1000)


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


from robot_hardware import RobotHardware, GuardedLoadStopped

class Controller:
    def __init__(self, url, hardware_factory=RobotHardware, log_path=None, data_dir=None, expo_record_dir=None,
                 expo_objective='gear_and_pin', expo_prefix_length=1, critic_options=None, alignment_options=None):
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
        self.alignment_options = alignment_options
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
        self.state = dict(task=CONTRACT['task'], supported_tasks=[CONTRACT['task']], mode='stopped', error=None, checkpoint=None, policy_ready=False, policy_error=None, measured=None,
                          target=None, cameras={}, prediction=None, action_index=0,
                          step_count=DEFAULT_ACTIONS_PER_STEP, steps_completed=0, latency_s=None,
                          playing=False, actions_per_step=DEFAULT_ACTIONS_PER_STEP, blend_actions=10,
                          blended_actions=0, foot_pedal_connected=False, foot_pedal_error=None,
                          hold_status='unverified', control_source='hold', rollout_active=False,
                          recording=None, collection_pending=False, handoff_ready={}, handoff_baseline_ready=False, handoff_gripper_enabled=True)
        self.alignment = None
        self._alignment_starting = False
        self._alignment_context = secrets.token_hex(8)
        self._freeze_requested = None
        self._nudge_requested = None
        self._nudge_generation = 0
        self._last_force_stop = None
        self._force_requested = None
        self._force_stop_requested = False
        self._force_xy_requested = None
        self._right_freeze_target = None
        self.right_settle = RightArmSettleDetector()
        self.pin_nudge_detector = PinNudgeDetector()
        self.pin_contact_detector = PinContactDetector()
        self._auto_contact_load = None
        self._auto_contact_stop = False
        self.state.update(auto_pin_contact_supported=True,
                          auto_pin_contact=dict(self.pin_contact_detector.result))
        self.state.update(pin_nudge_detection_supported=True,
                          pin_nudge_detection=self.pin_nudge_detector.update(None, time.monotonic_ns(), False))
        self.state.update(right_arm_auto_freeze_supported=True,
                          right_arm_auto_freeze=dict(self.right_settle.result))
        self.state.update(right_arm_freeze_supported=True, right_arm_frozen=False, right_arm_freeze_pending=False,
                          right_arm_freeze_requested=None, right_arm_freeze_target=None,
                          right_arm_freeze_reason=None)
        from pin_alignment_insert import AUTO_NUDGE_STEP_M
        self.state['alignment_limits'] = dict(max_auto_nudge_mm=AUTO_NUDGE_STEP_M*1000, auto_nudge_speed_mm_s=AUTO_NUDGE_SPEED_M_S*1000, max_nudge_mm=5.,max_load_xy_mm=30.,max_load_duration_s=300.)
        self.state.update(alignment_insert_supported=True, alignment_force_supported=False, alignment_force=dict(active=False, phase='off', reason=None, model_based_load=True, force_sensor=False))
        self.state.update(alignment_nudge_supported=True, alignment_capture_supported=True, alignment_manual_capture_supported=True, alignment_manual_pin_supported=True,
                          alignment_nudge=dict(status='idle', reason=None))
        self.state['alignment'] = dict(status='idle' if alignment_options is not None else 'disabled',
                                       preview_only=True, motor_permission=False)
        self.state['critic'] = dict(status='idle' if critic_options is not None else 'disabled')
        self.state['critic_history'] = []
        self.state['critic_samples'] = 0
        if self.expo_record_dir is not None:
            self.state.update(actions_per_step=self.expo_prefix_length, step_count=self.expo_prefix_length,
                              blend_actions=0, expo_recording_enabled=True, expo_prefix_length=self.expo_prefix_length,
                              expo_objective=expo_objective)

    def _start_alignment(self):
        if self.alignment_options is None or self.alignment is not None or self._alignment_starting:
            return
        self._alignment_starting = True
        def start():
            try:
                from pin_alignment_monitor import PinAlignmentMonitor
                monitor = PinAlignmentMonitor(**self.alignment_options)
                with self.lock:
                    self.alignment = monitor
                if self.stop_event.is_set():
                    monitor.close()
            except Exception as exc:
                with self.lock:
                    self.state['alignment'] = dict(status='error', reason=str(exc), preview_only=True, motor_permission=False)
            finally:
                self._alignment_starting = False
        threading.Thread(target=start, name='alignment-start', daemon=True).start()

    def capture_alignment(self, hardware, measured, measured_at):
        monitor = self.alignment
        if monitor is None:
            return
        try:
            frames = hardware.cameras.recording_frames() if hardware.cameras else {}
            # Alignment consumes the logical LEFT wrist, independently of USB naming.
            config = getattr(hardware, 'config', {})
            mapping = config.get('alignment_assist', {}).get('camera_mapping', {}) if isinstance(config, dict) else {}
            frame = frames.get(mapping.get('left_wrist', 'wrist_port3'))
            if isinstance(frame, dict) and frame.get('jpeg'):
                monitor.submit(jpeg=frame['jpeg'], state=measured.tolist(),
                               timestamp_ns=frame.get('monotonic_ns'), measured_timestamp_ns=measured_at,
                               camera_id='left_wrist', grasp_id=self._alignment_context,
                               context_id=self._alignment_context)
        except Exception as exc:
            with self.lock:
                self.state['alignment'] = dict(status='error', reason=str(exc), preview_only=True, motor_permission=False)

    def configure_alignment(self, values):
        from pin_alignment_monitor import validate_config
        if self.alignment_options is None:
            raise RuntimeError('Alignment preview is disabled')
        current = self.alignment_options.get('config', {})
        config = validate_config(values, current=current)
        if config['max_step_mm'] > 5:
            raise ValueError('Operator nudge max_step_mm must be at most 5 mm')
        self._nudge_generation += 1
        self._nudge_requested = None
        self._force_xy_requested = None
        if self.state['mode'] == 'force_auto' or (self._force_requested or {}).get('auto_insert'):
            self._force_stop_requested = True
        with self.lock:
            if self.state['mode'] == 'nudge_queued':
                self.state.update(mode='holding', alignment_nudge=dict(status='idle', reason='Configuration changed'))
        self.alignment_options = dict(self.alignment_options, config=config)
        if self.alignment is not None:
            self.alignment.configure(config)
        return config

    def reset_alignment(self):
        self._nudge_generation += 1
        self._nudge_requested = None
        self._force_stop_requested = True
        with self.lock:
            if self.state['mode'] == 'nudge_queued':
                self.state.update(mode='holding', alignment_nudge=dict(status='idle', reason='Alignment reset'))
        self._alignment_context = secrets.token_hex(8)
        if self.alignment is not None:
            self.alignment.reset(grasp_id=self._alignment_context, context_id=self._alignment_context)

    @staticmethod
    def alignment_frame_id(value):
        if type(value) is int and value > 0:
            return value
        if isinstance(value,str) and value.isascii() and value.isdigit() and len(value) <= 20 and int(value) > 0:
            return int(value)
        raise ValueError('A positive integer analyzed frame_id is required')

    def alignment_preview(self, expected_id):
        expected_id = self.alignment_frame_id(expected_id)
        if self.alignment is None:
            raise RuntimeError('Alignment preview unavailable')
        lookup = getattr(self.alignment,'status_for_frame',None)
        proposal = lookup(expected_id) if callable(lookup) else None
        if not isinstance(proposal,dict):
            # Compatibility with diagnostic adapters; never substitute another ID.
            proposal = self.alignment.status()
        if not isinstance(proposal,dict) or str(proposal.get('frame_id')) != str(expected_id):
            raise RuntimeError('Analyzed frame expired; review the current image and retry')
        return proposal

    def alignment_frame(self, expected_id):
        try:
            expected_id = self.alignment_frame_id(expected_id)
        except ValueError:
            return None
        if self.alignment is None:
            return None
        frame_id, jpeg = self.alignment.analyzed_frame(expected_id)
        return (frame_id, jpeg) if jpeg and str(frame_id) == str(expected_id) else None

    @staticmethod
    def force_measurement_diagnostics(kernel, measured, velocity6, position, now, output):
        """Retain the rejected measurement; held stop ticks are omitted by recording."""
        diagnostic = dict(output.get('diagnostics') or {})
        raw_velocity = kernel.kinematics.jacobian(np.asarray(measured)[:6])@np.asarray(velocity6)
        travel = np.asarray(position)-kernel.anchor_position
        diagnostic.update(sdk_joint_velocity_rad_s=np.asarray(velocity6).tolist(),
                          sdk_tool_speed_m_s=float(np.linalg.norm(raw_velocity[:3])),
                          sdk_down_speed_m_s=float(-raw_velocity[2]),
                          sdk_angular_speed_rad_s=float(np.linalg.norm(raw_velocity[3:])),
                          max_tool_speed_m_s=kernel.config.max_tool_speed_m_s,
                          max_down_speed_m_s=kernel.config.max_down_speed_m_s,
                          max_angular_speed_rad_s=kernel.config.max_angular_speed_rad_s,
                          max_joint_speed_rad_s=kernel.config.max_joint_speed_rad_s,
                          lateral_travel_m=float(np.linalg.norm(travel[:2])),
                          vertical_travel_m=float(travel[2]),elapsed_s=float(now-kernel.started))
        # Older diagnostic kernels guard SDK J*dq directly. New kernels supply
        # their actual windowed decision metric and retain SDK values separately.
        diagnostic.setdefault('tool_speed_m_s',diagnostic['sdk_tool_speed_m_s'])
        diagnostic.setdefault('down_speed_m_s',diagnostic['sdk_down_speed_m_s'])
        diagnostic.setdefault('velocity_source','sdk_joint_velocity')
        return diagnostic

    @staticmethod
    def force_stop_description(reason, diagnostic):
        if reason == 'approach_descent_tracking_lost':
            return ('Carrier approach stopped: arm dropped below its slow descent reference; '
                    'holding measured pose. Gravity compensation and startup support need checking.')
        speed_fields = {'tool_speed_exceeded':('tool_speed_m_s','max_tool_speed_m_s'),
                        'down_speed_exceeded':('down_speed_m_s','max_down_speed_m_s')}
        if reason in speed_fields:
            value_key,limit_key=speed_fields[reason]
            value=diagnostic.get(value_key);limit=diagnostic.get(limit_key)
            if isinstance(value,(int,float)) and isinstance(limit,(int,float)):
                return (f'{reason}: {value*1000:.2f} mm/s > {limit*1000:.2f} mm/s '
                        f'({diagnostic.get("velocity_source","unknown")}; '
                        f'XY travel {diagnostic.get("lateral_travel_m",0)*1000:.3f} mm, '
                        f'Z travel {diagnostic.get("vertical_travel_m",0)*1000:.3f} mm)')
        return reason

    @staticmethod
    def force_hardware_supported(hardware):
        limits = getattr(hardware,'guarded_load_limits',{})
        return bool(getattr(hardware,'supports_guarded_load',False) and isinstance(limits,dict) and
                    limits.get('max_xy_m',0) >= .03 and limits.get('max_speed_m_s',0) >= .05)

    @staticmethod
    def contact_approach_supported(hardware):
        limits = getattr(hardware, 'guarded_load_limits', {}).get('contact_approach', {})
        loaded = getattr(hardware, 'guarded_load_limits', {}).get('loaded_insertion', {})
        return (isinstance(limits, dict) and limits.get('max_down_m', 0) >= APPROACH_DOWN_M
                and limits.get('max_xy_m', 0) >= APPROACH_XY_M
                and limits.get('max_duration_s', 0) >= APPROACH_TIMEOUT_S
                and isinstance(loaded, dict) and loaded.get('max_up_m', 0) >= .005)

    @staticmethod
    def force_load_config(values, *, require_ack=True):
        from pin_alignment_search import SearchConfig
        allowed = {'force_n','pin_on_carrier','frame_id',
                   'max_duration_s','drop_threshold_mm','drop_debounce_s',
                   'oscillation_amplitude_mm','oscillation_frequency_hz'}
        if not isinstance(values, dict) or set(values)-allowed:
            raise ValueError('Unknown load setting')
        force_n = values.get('force_n', .5)
        if isinstance(force_n, bool) or not np.isfinite(force_n) or not 0 < force_n <= 3:
            raise ValueError('force_n must be greater than zero and at most 3 N')
        if require_ack and values.get('pin_on_carrier') is not True:
            raise ValueError('Explicit pin_on_carrier operator acknowledgment is required')
        config = SearchConfig(max_duration_s=values.get('max_duration_s',120.),
                              oscillation_amplitude_m=values.get('oscillation_amplitude_mm',.2)/1000,
                              oscillation_frequency_hz=values.get('oscillation_frequency_hz',15.),
                              insertion_drop_m=values.get('drop_threshold_mm',3.)/1000,
                              insertion_debounce_s=values.get('drop_debounce_s',.2))
        return float(force_n), config

    def configure_auto_pin_contact(self, enabled, load=None):
        if type(enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        # Arming authorizes a detected-contact handoff, rather than asserting
        # that the operator has already observed physical contact.
        settings = self.force_load_config(load or {}, require_ack=False) if enabled else None
        with self.lock:
            self._auto_contact_load = settings
            self.pin_contact_detector.configure(enabled)
            self.state['auto_pin_contact'] = dict(self.pin_contact_detector.result)
            if not enabled:
                self._auto_contact_stop = True

    def finish_auto_pin_contact(self):
        with self.lock:
            if self.pin_contact_detector.latched:
                self.state['auto_pin_contact'] = dict(self.pin_contact_detector.complete())

    def request_alignment_force(self, values, *, auto_insert=False):
        force_n, config = self.force_load_config(values)
        with self.lock:
            if (not self.state['alignment_force_supported'] or self.state['mode'] != 'holding' or
                    self.state['playing'] or not self.state['right_arm_frozen'] or
                    self.state['right_arm_freeze_pending'] or self.state['collection_pending'] or
                    self.pause_event.is_set() or not self.commands.empty() or not self.collection_commands.empty() or
                    time.monotonic()-self.heartbeat_at > LEASE_SECONDS):
                raise RuntimeError('Guarded motor backend, paused powered hold, frozen RIGHT and live operator are required')
            self._force_requested = dict(force_n=force_n, load_config=config, auto_insert=auto_insert, generation=self._nudge_generation)
            self._force_stop_requested = False
            self.state.update(mode='force_queued', alignment_force=dict(active=False, phase='queued',
                              requested_force_n=force_n, reason=None, model_based_load=True, force_sensor=False))

    def request_alignment_insert(self, values):
        if self.alignment is None:
            raise RuntimeError('Alignment preview is required for auto insertion')
        # Detection can be warming up. The loaded loop waits for a fresh,
        # supported proposal before authorizing any XY correction.
        self.request_alignment_force(values, auto_insert=True)

    def request_alignment_force_stop(self):
        self._force_requested = None
        self._force_stop_requested = True

    def _operator_alignment_guard(self):
        with self.lock:
            if (self.state['mode'] != 'holding' or self.state['playing'] or
                    not self.state['right_arm_frozen'] or self.state['right_arm_freeze_pending'] or
                    self.state['collection_pending'] or not self.commands.empty() or
                    not self.collection_commands.empty() or self.pause_event.is_set() or
                    time.monotonic()-self.heartbeat_at > LEASE_SECONDS):
                raise RuntimeError('Pause in powered hold, freeze RIGHT, and keep the operator connected first')
            if self._nudge_requested is not None:
                raise RuntimeError('A nudge is already queued')
        if self.alignment is None:
            raise RuntimeError('Alignment preview is not running')

    @staticmethod
    def validate_nudge_preview(proposal, measured, expected_frame_id, now_ns=None, *, allow_zero=False):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if str(proposal.get('frame_id')) != str(expected_frame_id):
            raise RuntimeError('Preview frame changed; review the current image and retry')
        stamp = proposal.get('analyzed_timestamp_ns')
        config = proposal.get('config') or {}
        frequency = config.get('frequency_hz',20.)
        if isinstance(frequency,bool) or not np.isfinite(frequency) or frequency <= 0:
            raise RuntimeError('Invalid preview cadence')
        # Permit slower analysis/operator response independently of camera
        # cadence. Current-pose and context checks still apply.
        configured_window = config.get('max_frame_age_ms',5000.)/1000
        age_window = min(configured_window,10.)
        if not np.isfinite(age_window) or age_window <= 0 or not isinstance(stamp,int) or not 0 <= now_ns-stamp <= age_window*1e9:
            raise RuntimeError('Nudge preview is stale or has an invalid timestamp')
        if not proposal.get('nudge_valid') or proposal.get('hole_source') not in (
                'demo_reference_tracked', 'demo_reference_held', 'captured_demo_reference', 'captured_operator_reference',
                'left_carrier_cnn_observed', 'left_carrier_opencv_observed', 'left_carrier_layout_inferred',
                'left_carrier_stationary_accumulated'):
            raise RuntimeError('A supported carrier detection or explicitly captured operator hole is required')
        if proposal.get('hole_source')=='left_carrier_stationary_accumulated' and not proposal.get('accumulation',{}).get('stationary'):
            raise RuntimeError('Retained hole requires both arms to remain stationary')
        source = np.asarray(proposal.get('measured_state14', []), dtype=float)
        measured = np.asarray(measured, dtype=float)
        if source.shape != (14,) or measured.shape != (14,) or not np.isfinite(source).all() or not np.isfinite(measured).all():
            raise RuntimeError('Finite measured proposal state14 is required')
        if np.max(np.abs(source[[0,1,2,3,4,5,7,8,9,10,11,12]]-measured[[0,1,2,3,4,5,7,8,9,10,11,12]])) > .02:
            raise RuntimeError('Arm pose moved since the analyzed frame; wait for a fresh preview')
        delta = np.asarray(proposal.get('nudge_left_mm', []), dtype=float)/1000
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise RuntimeError('Nudge requires a finite XYZ estimate')
        # Hold pin height even when the estimated carrier plane is tilted.
        delta = delta.copy()
        delta[2] = 0.
        if np.linalg.norm(delta) > .005000001 or (not allow_zero and np.linalg.norm(delta) <= 0):
            raise RuntimeError('Nudge must be nonzero and at most 5 mm; adjust preview settings')
        if not allow_zero and np.linalg.norm(delta[:2]) <= 1e-12:
            raise RuntimeError('Nudge requires a nonzero XY correction')
        return delta

    def request_alignment_nudge(self, frame_id):
        with self.lock:
            loaded = self.state['mode'] == 'force_holding' and self.state['alignment_force'].get('active')
        if loaded:
            if self.alignment is None or time.monotonic()-self.heartbeat_at > LEASE_SECONDS:
                raise RuntimeError('Fresh preview and connected operator required')
            proposal = self.alignment_preview(frame_id)
            delta_xy = self.validate_nudge_preview(proposal, proposal.get('measured_state14', []), frame_id)
            if np.linalg.norm(delta_xy[:2]) <= 1e-12:
                raise RuntimeError('Under-load nudge requires a nonzero XY correction')
            with self.lock:
                if self.state['mode'] != 'force_holding':
                    raise RuntimeError('Manual corrections require load hold')
                self._force_xy_requested = dict(proposal=proposal, frame_id=frame_id)
                self.state['alignment_nudge'] = dict(status='queued', frame_id=frame_id, xy_only=True)
            return
        self._operator_alignment_guard()
        proposal = self.alignment_preview(frame_id)
        self.validate_nudge_preview(proposal, proposal.get('measured_state14', []), frame_id)
        with self.lock:
            # Atomically reserve powered hold against policy commands.
            if (self.state['mode'] != 'holding' or self.state['playing'] or not self.commands.empty() or
                    not self.state['right_arm_frozen'] or self.state['right_arm_freeze_pending']):
                raise RuntimeError('Controller state changed; pause and retry')
            self._nudge_requested = dict(proposal=proposal, frame_id=frame_id, generation=self._nudge_generation)
            self.state.update(mode='nudge_queued', alignment_nudge=dict(status='queued', frame_id=frame_id))

    def operator_alignment_point(self, frame_id, point_px):
        if (not isinstance(point_px,(list,tuple)) or len(point_px) != 2 or
                any(type(value) not in (int,float) for value in point_px)):
            raise ValueError('Operator point requires two numeric pixel coordinates')
        point = np.asarray(point_px,dtype=float)
        status = self.alignment_preview(frame_id)
        size = np.asarray(status.get('image_size_wh',[]),dtype=float)
        if (size.shape != (2,) or not np.isfinite(point).all() or
                not np.isfinite(size).all() or np.any(size <= 0) or
                np.any(point < 0) or np.any(point >= size)):
            raise ValueError('Operator point must lie within the analyzed native image')
        return point.tolist()

    def capture_alignment_hole(self, frame_id, hole_px=None):
        frame_id = self.alignment_frame_id(frame_id)
        self._operator_alignment_guard()
        if hole_px is not None:
            return self.alignment.capture_hole(frame_id,hole_px=self.operator_alignment_point(frame_id,hole_px))
        return self.alignment.capture_hole(frame_id)

    def clear_alignment_hole(self):
        self._nudge_generation += 1
        self._nudge_requested = None
        self._force_stop_requested = True
        with self.lock:
            if self.state['mode'] == 'nudge_queued':
                self.state.update(mode='holding',alignment_nudge=dict(status='idle',reason='Hole cleared'))
        if self.alignment is not None:
            return self.alignment.clear_hole()

    def capture_alignment_pin(self, frame_id, pin_px):
        frame_id = self.alignment_frame_id(frame_id)
        self._operator_alignment_guard()
        return self.alignment.capture_pin(frame_id,self.operator_alignment_point(frame_id,pin_px))

    def clear_alignment_pin(self):
        self._operator_alignment_guard()
        return self.alignment.clear_pin()

    @staticmethod
    def validate_nudge_commit(planned, measured):
        """Commit only a path anchored at the fresh held pose.

        Ordinary encoder drift asks for asynchronous replanning rather than
        reusing an old endpoint at a different height.
        """
        from pin_regrasp_motion import rotation_log
        encoder_lsb = 2*np.pi/10000
        joint_tolerance = .01  # physical fresh-pose drift bound, not an encoder-count gate
        initial_q = np.asarray(planned['initial_state14'],dtype=float)[:6]
        fresh_q = np.asarray(measured,dtype=float)[:6]
        endpoint_q = np.asarray(planned['target14'],dtype=float)[:6]
        kin = planned['kinematics']
        current_position,current_rotation = kin.pose(fresh_q)
        joint_drift = float(np.max(np.abs(fresh_q-initial_q)))
        position_drift = float(np.linalg.norm(current_position-planned['initial_position']))
        orientation_drift = float(np.linalg.norm(rotation_log(current_rotation@np.asarray(planned['initial_rotation']).T)))
        diagnostic = (f'joint drift {joint_drift:.9f} rad (limit {joint_tolerance:.9f}); '
                      f'tool drift {position_drift*1000:.4f} mm; orientation drift {orientation_drift:.7f} rad')
        if (not np.isfinite(fresh_q).all() or joint_drift > joint_tolerance or
                position_drift > .001 or orientation_drift > .01):
            raise RuntimeError('Arm moved beyond the fresh-pose safety envelope (1 mm / 0.01 rad) during nudge preflight; '+diagnostic+'; retry with a fresh preview')
        # Reusing an old endpoint after resetting the joint origin to a fresh
        # measurement creates an unintended vertical correction. Rebuild the
        # entire relative XY path from the fresh pose instead.
        if position_drift > .00001 or orientation_drift > .0001:
            raise NudgeReanchorRequired('Nudge pose anchor changed; rebuild the height-locked path from fresh feedback')
        endpoint_position,_ = kin.pose(endpoint_q)
        displacement = float(np.linalg.norm(endpoint_position-current_position))
        if displacement > .005+.0001:
            raise RuntimeError(f'Fresh-pose nudge exceeds the 5 mm bound plus 0.1 mm FK tolerance ({displacement*1000:.4f} mm); '+diagnostic)
        if np.linalg.norm(endpoint_q-fresh_q) > .2:
            raise RuntimeError('Fresh nudge joint path exceeds its 0.2 rad bound; '+diagnostic)
        # Match the independent motor preflight's 101-point height clearance,
        # now starting at this tick's measured joints rather than the old read.
        floor = min(current_position[2],endpoint_position[2])-.01
        max_translation_error = max_orientation_error = 0.
        max_height_error = 0.
        path = np.asarray(planned.get('waypoints14',[planned['initial_state14'],planned['target14']]),float)
        if path.ndim!=2 or path.shape[1]!=14 or len(path)<2 or not np.isfinite(path).all():
            raise RuntimeError('Invalid Cartesian nudge waypoint path')
        if (not np.allclose(path[-1],planned['target14'],rtol=0,atol=1e-10)
                or not np.allclose(path[0,:6],initial_q,rtol=0,atol=1e-10)):
            raise RuntimeError('Fresh nudge path fails Cartesian/orientation bounds: endpoint or origin differs from checked path')
        for index,alpha in enumerate(np.linspace(0.,1.,101)):
            phase=alpha*(len(path)-1);segment=min(int(phase),len(path)-2);fraction=phase-segment
            sample=path[segment,:6]*(1-fraction)+path[segment+1,:6]*fraction
            point,rotation = kin.pose(sample)
            if not np.isfinite(point).all() or point[2] < floor:
                raise RuntimeError('Fresh nudge path fails endpoint height clearance; '+diagnostic)
            max_height_error=max(max_height_error,abs(float(point[2]-current_position[2])))
            if max_height_error>.00001:
                raise RuntimeError('Fresh nudge path changes the held TCP height; '+diagnostic)
            if index % 5 == 0:
                expected = current_position*(1-alpha)+endpoint_position*alpha
                translation_error = float(np.linalg.norm(point-expected))
                orientation_error = float(np.linalg.norm(rotation_log(rotation@current_rotation.T)))
                max_translation_error = max(max_translation_error,translation_error)
                max_orientation_error = max(max_orientation_error,orientation_error)
                # Physical path envelopes; tight IK numerical solver precision
                # remains independent from these movement-safety tolerances.
                if translation_error > .00001 or max_height_error > .00001 or orientation_error > .01:
                    raise RuntimeError('Fresh nudge path fails Cartesian/orientation bounds; '+diagnostic)
        return dict(encoder_lsb_rad=float(encoder_lsb),encoder_joint_tolerance_rad=float(joint_tolerance),
                    preflight_joint_drift_rad=joint_drift,preflight_tool_drift_m=position_drift,
                    preflight_orientation_drift_rad=orientation_drift,fresh_translation_norm_m=displacement,fresh_translation_limit_m=.0051,
                    endpoint_numerical_tolerance_m=.0001,
                    fresh_max_path_translation_error_m=max_translation_error,
                    fresh_max_path_orientation_error_rad=max_orientation_error,
                    fresh_max_path_height_error_m=max_height_error,fresh_path_samples=101)

    @staticmethod
    def plan_operator_nudge(hardware, measured, target, delta, right_hold):
        from pin_alignment_motion import plan_left_nudge, NativeToolKinematics
        plan = plan_left_nudge(measured, delta, hardware.lower, hardware.upper, max_step_m=.005)
        if not plan.get('valid'):
            raise RuntimeError(plan.get('reason') or 'Cartesian nudge could not be solved')
        endpoint = np.asarray(plan['target14'], dtype=float)
        if endpoint.shape != (14,) or not np.isfinite(endpoint).all():
            raise RuntimeError('Nudge planner returned an invalid joint target')
        endpoint[6] = target[6]  # Preserve the commanded LEFT grasp preload too.
        endpoint[7:14] = right_hold
        if np.any(endpoint < hardware.lower) or np.any(endpoint > hardware.upper):
            raise RuntimeError('Nudge target exceeds joint limits')
        path=np.asarray(plan['waypoints14'],dtype=float)
        path[:,6]=target[6];path[:,7:14]=right_hold
        if np.any(path<hardware.lower) or np.any(path>hardware.upper):
            raise RuntimeError('Nudge waypoint exceeds joint limits')
        client = getattr(hardware, 'motor_client', None)
        if client is not None:
            for start,end in zip(path,path[1:]):
                client.validate_path('left',start[:6].tolist(),end[:6].tolist())
        kinematics = NativeToolKinematics()
        initial_position, initial_rotation = kinematics.pose(measured[:6])
        from pin_xy_trajectory import HeightLockedNudgeTrajectory
        trajectory=HeightLockedNudgeTrajectory(path,0.,kinematics,initial_position,initial_rotation)
        return dict(target14=endpoint, waypoints14=path, trajectory=trajectory,
                    metadata=dict(plan.get('metadata',{}),**trajectory.preflight_metadata),initial_state14=measured.copy(),
                    kinematics=kinematics, initial_position=initial_position, initial_rotation=initial_rotation)

    def request_right_arm_freeze(self, frozen):
        if not isinstance(frozen, bool):
            raise ValueError('frozen must be a boolean')
        with self.lock:
            if self.state['mode'] not in ('holding', 'queued', 'predicting', 'executing') or self.state.get('control_source') == 'human':
                raise RuntimeError('Connect and leave manual takeover before freezing')
            self._freeze_requested = frozen
            if not frozen:
                self.right_settle.inhibited = True
            self.state.update(right_arm_freeze_pending=True, right_arm_freeze_requested=frozen)

    def configure_right_auto_freeze(self, enabled):
        with self.lock:
            self.state['right_arm_auto_freeze'] = dict(self.right_settle.configure(enabled))

    def configure_pin_nudge_detection(self, enabled):
        with self.lock:
            self.pin_nudge_detector.set_enabled(enabled)
            self.state['pin_nudge_detection'] = self.pin_nudge_detector.update(None, time.monotonic_ns(), False)

    def clear_right_freeze(self, reason):
        with self.lock:
            self.pin_contact_detector.reset()
            self.state['auto_pin_contact'] = dict(self.pin_contact_detector.result)
            self.right_settle.reset()
            self.pin_nudge_detector.reset()
            self.state['pin_nudge_detection'] = self.pin_nudge_detector.update(None, time.monotonic_ns(), False)
            self.state['right_arm_auto_freeze'] = dict(self.right_settle.result)
            self._freeze_requested = None
            self._right_freeze_target = None
            self.state.update(right_arm_frozen=False, right_arm_freeze_pending=False,
                              right_arm_freeze_requested=None, right_arm_freeze_target=None,
                              right_arm_freeze_reason=reason)

    def project_right_hold(self, actions):
        result = np.asarray(actions).copy()
        if self._right_freeze_target is not None:
            result[..., 7:14] = self._right_freeze_target
        return result

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
            if 'alignment_force' in values:
                force = dict(values['alignment_force'])
                if not force.get('active') and force.get('reason_code') and force.get('diagnostics'):
                    self._last_force_stop = dict(reason=force.get('reason'),reason_code=force['reason_code'],
                                                  diagnostics=dict(force['diagnostics']),unix_ns=time.time_ns())
                if self._last_force_stop is not None:
                    force['last_stop'] = self._last_force_stop
                    if not force.get('active') and not force.get('diagnostics'):
                        force['diagnostics'] = self._last_force_stop['diagnostics']
                values['alignment_force'] = force
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
        if self.alignment is not None:
            alignment = self.alignment.status()
            with self.lock:
                self.state['alignment'] = alignment
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
        self._freeze_requested = None
        self.right_settle.reset()
        self.pin_contact_detector.reset()
        self._auto_contact_stop = False
        self.state['auto_pin_contact'] = dict(self.pin_contact_detector.result)
        self.state['right_arm_auto_freeze'] = dict(self.right_settle.result)
        self.pin_nudge_detector.reset()
        self.state['pin_nudge_detection'] = self.pin_nudge_detector.update(None, time.monotonic_ns(), False)
        self._nudge_requested = None
        self._nudge_generation += 1
        self._right_freeze_target = None
        self.state.update(right_arm_frozen=False, right_arm_freeze_pending=False, right_arm_freeze_requested=None, right_arm_freeze_target=None)
        self._force_requested = self._force_xy_requested = None
        self._force_stop_requested = False
        self.state['alignment_force'] = dict(active=False, phase='off',reason='New session',model_based_load=True,force_sensor=False)
        self._alignment_context = secrets.token_hex(8)
        self.stop_event.clear()
        self.pause_event.clear()
        self.commands = queue.Queue(maxsize=1)
        self.collection_commands = queue.Queue(maxsize=1)
        self.heartbeat_at = time.monotonic()
        self.expo_recorder = None
        self.state.update(mode='connecting', error=None, prediction=None,
                          policy_ready=False, policy_error=None, checkpoint=None,
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
            if kind in ('step', 'play', 'predict') and not self.state['policy_ready']:
                raise RuntimeError('Policy unavailable: ' + (self.state['policy_error'] or 'connect to a ready inference server first'))
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
            if action == 'begin' and (self.expo_record_dir is not None or payload.get('pedal_play')) and not self.state['policy_ready']:
                raise RuntimeError('Policy unavailable; connect to a ready inference server before recording policy trials')
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
            self._freeze_requested = None
            self._nudge_requested = None
            self._nudge_generation += 1
            self.state.update(right_arm_freeze_pending=False, right_arm_freeze_requested=None)
            self.pause_event.set()
        if audible:
            self.cue('policy_pause')
        self.clear_critic()

    def shutdown(self):
        """Release resources only for a deliberate controller shutdown."""
        self.stop_event.set()
        self.clear_right_freeze('Controller shutdown')
        self.clear_critic()
        if self.critic is not None:
            self.critic.close()
        if self.alignment is not None:
            self.alignment.close()

    def check_active(self):
        if self.stop_event.is_set():
            raise RuntimeError('Operator stopped the session')

    def clear_commands(self):
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                return

    def set_camera_gain(self, name, value):
        hardware = self.hardware
        if not hardware or not hardware.cameras:
            raise RuntimeError('Connect cameras before adjusting gain')
        if not isinstance(name, str) or name not in CONTRACT['image_crops_xyxy']:
            raise ValueError('Unknown camera')
        setter = getattr(hardware.cameras, 'set_gain', None)
        if setter is None:
            raise RuntimeError('Manual camera gain is unavailable in this session')
        return setter(name, value)

    def jpeg(self, name):
        hardware = self.hardware
        return hardware.cameras.jpeg(name) if hardware and hardware.cameras else None

    def collection_reset(self, hardware, measured):
        self.clear_right_freeze('Collection reset')
        self.reset_alignment()
        if 'neutral_return' not in hardware.config:
            return None
        reset = CollectionReset(hardware.config['neutral_return'], hardware.lower,
                                hardware.upper, measured, time.monotonic())
        hardware.validate_reset_path(measured, reset.neutral.targets)
        self.update(mode='resetting', prediction=None, action_index=0)
        return reset

    def run(self, reset_pose=True):
        self._start_alignment()
        hardware = self.hardware_factory()
        hardware.check_active = self.check_active
        self.hardware = hardware
        pool = ThreadPoolExecutor(max_workers=1)
        future = step = None
        nudge_future = nudge_step = nudge_job = None
        force_session = None
        force_command = None
        last_contact_tick = -float('inf')
        from pin_alignment_motion import NativeToolKinematics
        contact_kinematics = NativeToolKinematics()
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
            health = dict(contract=CONTRACT)
            checkpoint = None
            rtc_enabled = task_echo_required = False
            expo_policy_identity = {}
            task_generation = self.task_generation
            self.update(task=CONTRACT['task'], supported_tasks=[CONTRACT['task']],
                        blending_method='crossfade')
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
                self.update(policy_ready=True, policy_error=None)
            except Exception as exc:
                # Local powered hold and operator alignment do not need AWS.
                # Never allow policy commands or invent EXPO policy provenance.
                health = dict(contract=CONTRACT)
                checkpoint = None
                rtc_enabled = task_echo_required = False
                expo_policy_identity = {}
                self.update(policy_ready=False, policy_error=str(exc), checkpoint=None,
                            task=CONTRACT['task'], supported_tasks=[CONTRACT['task']],
                            blending_method='crossfade')
            self.check_active()
            hardware.open()
            self.update(alignment_force_supported=self.force_hardware_supported(hardware),
                        auto_pin_contact_supported=self.force_hardware_supported(hardware)
                            and self.contact_approach_supported(hardware))
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
                    self.capture_alignment(hardware, measured, measured_at)
                    with self.lock:
                        auto_freeze = self.right_settle.update(now, measured, target,
                            eligible=(not human and reset is None and not self.pause_event.is_set()
                                      and now-self.heartbeat_at <= LEASE_SECONDS
                                      and (self.state['playing'] or step is not None)
                                      and self.state['mode'] in ('holding','queued','predicting','executing')),
                            frozen=self._right_freeze_target is not None)
                        self.state['right_arm_auto_freeze'] = dict(self.right_settle.result)
                        freeze_request = self._freeze_requested
                        auto_freeze = bool(freeze_request is None and auto_freeze and self._right_freeze_target is None)
                        if auto_freeze:
                            freeze_request = True
                        self._freeze_requested = None
                    if freeze_request is not None:
                        # Capture on the control thread, from this tick's fresh measured pose.
                        if human or reset is not None:
                            self.clear_right_freeze('Manual takeover or reset')
                        else:
                            if freeze_request:
                                held = measured[7:14].copy()
                                # Tighten once from the existing squeeze command, not
                                # the object-blocked encoder position. The frozen
                                # target then preserves this grip without ratcheting.
                                grip_target = target[13] - (AUTO_FREEZE_GRIPPER_CLOSE_RAD if auto_freeze else 0.)
                                held[6] = np.clip(grip_target, hardware.lower[13], hardware.upper[13])
                                self._right_freeze_target = held
                                target[7:14] = held
                            else:
                                self._right_freeze_target = None
                            if future is not None:
                                future.cancel()
                                sequence += 1
                            future = None
                            previous_tail = rtc_body = None
                            last_policy_clock = None
                            self.clear_commands()
                            if step is not None:
                                originals = getattr(step, 'policy_actions', step.actions)
                                remaining = originals[min(step.index, len(originals)-1):].copy()
                                step = ActionStep(self.project_right_hold(remaining), len(remaining), now)
                                step.policy_actions = remaining
                                actions = step.actions.copy()
                                count = len(remaining)
                            if step is None:
                                self.update(mode='holding')
                            if self.expo_recorder:
                                self.expo_recorder.boundary('right_arm_freeze' if freeze_request else 'right_arm_unfreeze')
                            with self.lock:
                                self.state.update(right_arm_frozen=freeze_request, right_arm_freeze_pending=False,
                                                  right_arm_freeze_requested=None,
                                                  right_arm_freeze_target=None if not freeze_request else held.tolist(),
                                                  right_arm_freeze_reason=('Measured right joints; right gripper tightened by up to 0.05 rad within limits' if auto_freeze else
                                                                           'Measured right joints; existing gripper squeeze command retained') if freeze_request else 'Released with bounded trajectory',
                                                  right_control_source='frozen_hold' if freeze_request else 'policy_or_hold',
                                                  prediction=None)
                            log.write(json.dumps(dict(event='right_arm_freeze' if freeze_request else 'right_arm_unfreeze',
                                automatic=bool(auto_freeze), monotonic_ns=measured_at,
                                target=target[7:14].tolist(), detector=dict(self.right_settle.result)))+'\n')
                            log.flush()

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
                    force_command = None
                    nudge_command_tick = False
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
                        with self.lock:
                            if self.pin_contact_detector.latched:
                                self.pin_contact_detector.complete()
                            else:
                                self.pin_contact_detector.reset()
                            self.state['auto_pin_contact'] = dict(self.pin_contact_detector.result)
                        if force_session is not None:
                            target[:6] = measured[:6]
                        force_session = None
                        self._force_requested = self._force_xy_requested = None
                        self._force_stop_requested = False
                        self.update(alignment_force=dict(active=False, phase='off', reason='Operator pause or connection loss', model_based_load=True, force_sensor=False))
                        if nudge_future is not None:
                            nudge_future.cancel()
                        nudge_future = nudge_step = nudge_job = None
                        self._nudge_requested = None
                        self.update(alignment_nudge=dict(status='idle', reason='Operator pause or connection loss'))
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
                            self.clear_right_freeze('Collection lifecycle transition')
                            self.reset_alignment()
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
                                tracking_target = target.copy()
                                if force_session is not None:
                                    tracking_target[:6] = measured[:6]
                                check_arm_tracking(tracking_target, measured)
                            except RuntimeError as exc:
                                target = np.clip(measured, hardware.lower, hardware.upper)
                                raise RuntimeError(f'{exc}; holding measured pose') from exc
                        if self.recorder and self.recorder.failure:
                            raise RuntimeError(f'Recording failed: {self.recorder.failure}')
                        if self.expo_recorder and self.expo_recorder.failure:
                            raise RuntimeError(f'EXPO recording failed: {self.expo_recorder.failure}')
                        with self.lock:
                            if self._auto_contact_stop:
                                if force_session is not None and force_session.get('automatic_pin_contact'):
                                    self._force_stop_requested = True
                                if (self._force_requested or {}).get('automatic_pin_contact'):
                                    self._force_requested = None
                                    self.state.update(mode='holding', playing=False,
                                        alignment_force=dict(active=False, phase='off', reason='Automatic contact assistance disabled',
                                                             model_based_load=True, force_sensor=False))
                                self._auto_contact_stop = False
                            if self.pin_contact_detector.enabled and force_session is None and now-last_contact_tick >= .02:
                                last_contact_tick = now
                                left_position, _ = contact_kinematics.pose(measured[:6])
                                right_position, _ = contact_kinematics.pose(measured[7:13])
                                command_position, _ = contact_kinematics.pose(target[:6])
                                contact = self.pin_contact_detector.update(now, measured, left_position, right_position,
                                    eligible=bool(not human and reset is None and not self.state['collection_pending']
                                        and self.collection_commands.empty() and nudge_future is None and nudge_step is None
                                        and self.state['mode'] in ('holding','queued','predicting','executing')
                                        and not self.pause_event.is_set() and not browser_lost
                                        and self.force_hardware_supported(hardware)
                                        and self.contact_approach_supported(hardware) and self.alignment is not None),
                                    right_frozen=self._right_freeze_target is not None,
                                    allow_trigger=bool(self.state['playing'] or step is not None),
                                    commanded_left_xyz=command_position)
                                self.state['auto_pin_contact'] = dict(contact)
                                if contact['triggered'] and self._auto_contact_load is not None:
                                    if future is not None:
                                        future.cancel()
                                    future = step = None
                                    sequence += 1
                                    previous_tail = rtc_body = None
                                    last_policy_clock = policy_clock = None
                                    expo_feedback_pending = False
                                    self.clear_commands()
                                    target = measured_hold_target(measured, target, hardware.lower, hardware.upper, {})
                                    force_n, load_config = self._auto_contact_load
                                    self._force_requested = dict(force_n=force_n, load_config=load_config,
                                        auto_insert=True, generation=self._nudge_generation,
                                        automatic_pin_contact=True, contact_evidence=dict(contact))
                                    self._force_stop_requested = False
                                    self.state.update(mode='force_queued', playing=False, prediction=None,
                                        action_index=0, blended_actions=0,
                                        alignment_force=dict(active=False, phase='queued', requested_force_n=force_n,
                                            automatic_pin_contact=True, reason='Carrier approach detected: policy paused for guarded descent',
                                            model_based_load=True, force_sensor=False))
                                    if self.expo_recorder:
                                        self.expo_recorder.boundary('auto_pin_contact')
                                    log.write(json.dumps(dict(event='auto_pin_contact_handoff',monotonic_ns=measured_at,
                                        measured_state14=measured.tolist(),contact_evidence=contact,
                                        requested_force_n=force_n,physical_contact_confirmed=False))+'\n')
                                    log.flush()
                        if self._force_stop_requested:
                            force_was_active = force_session is not None or self.state['mode'].startswith('force_')
                            if force_session is not None:
                                target[:6] = measured[:6]
                                if force_session.get('automatic_pin_contact'):
                                    self.finish_auto_pin_contact()
                            force_session = None
                            self._force_requested = self._force_xy_requested = None
                            self._force_stop_requested = False
                            if force_was_active:
                                self.update(mode='holding', alignment_force=dict(active=False, phase='off', reason='Operator stop or context reset', model_based_load=True, force_sensor=False))
                        if self._force_requested is not None:
                            command = self._force_requested
                            self._force_requested = None
                            if (human or reset is not None or future is not None or step is not None or
                                    nudge_future is not None or nudge_step is not None or self.state['playing'] or
                                    not self.state['right_arm_frozen'] or not self.force_hardware_supported(hardware)):
                                raise RuntimeError('Force requires policy paused, RIGHT frozen and guarded motor backend')
                            from pin_alignment_force import CartesianLoadController
                            from pin_alignment_search import LoadedDropMonitor
                            approaching = bool(command.get('automatic_pin_contact'))
                            if approaching and not self.contact_approach_supported(hardware):
                                raise RuntimeError('Restart motor backend for guarded carrier approach support')
                            kernel_config = dict(force_n=command['force_n'],max_lateral_travel_m=.03,
                                max_tool_speed_m_s=.05,max_duration_s=command['load_config'].max_duration_s)
                            if command.get('automatic_pin_contact'):
                                kernel_config['force_slew_n_s'] = APPROACH_FORCE_SLEW_N_S
                            if approaching:
                                kernel_config.update(contact_approach=True,
                                    force_n=min(command['force_n'], APPROACH_FORCE_N),
                                    max_lateral_travel_m=APPROACH_XY_M,
                                    max_vertical_travel_m=APPROACH_DOWN_M,
                                    vertical_damping_ns_m=20., max_duration_s=APPROACH_TIMEOUT_S,
                                    xy_integral_gain_n_m_s=600., xy_integral_limit_n=1.5,
                                    xy_integral_slew_n_s=.5)
                            kernel = CartesianLoadController(measured, kernel_config, now=now,
                                    lower6=hardware.lower[:6], upper6=hardware.upper[:6])
                            position = kernel.anchor_position.copy()
                            identity = secrets.token_hex(12)
                            guard = dict(id=identity, anchor_tool_position_m=position.tolist(),
                                         anchor_tool_rotation=kernel.anchor_rotation.tolist(), max_xy_m=.03,
                                         max_down_m=.004, max_up_m=.002, max_speed_m_s=.05,
                                         max_angular_rad=.1, max_duration_s=command['load_config'].max_duration_s)
                            if approaching:
                                guard.update(phase='contact_approach', max_xy_m=APPROACH_XY_M,
                                    max_down_m=APPROACH_DOWN_M, max_duration_s=APPROACH_TIMEOUT_S)
                            force_session = dict(kernel=kernel, drop=LoadedDropMonitor(now,position,command['load_config']),
                                config=command['load_config'], force_n=command['force_n'], started=now,
                                automatic_pin_contact=bool(command.get('automatic_pin_contact')),
                                xy_target=position[:2].copy(), xy_goal=position[:2].copy(), guard=guard)
                            force_session['command'] = command
                            if approaching:
                                force_session['approach'] = ContactApproach(now, position)
                            if command.get('auto_insert'):
                                from pin_alignment_insert import AutoInsertion
                                if command['generation'] != self._nudge_generation or self.alignment is None:
                                    raise RuntimeError('Alignment changed before auto insertion started')
                                if not approaching:
                                    force_session['auto'] = AutoInsertion(position, command['load_config'], now)
                                force_session['generation'] = command['generation']
                            target[:6] = measured[:6]
                            if self.expo_recorder:
                                self.expo_recorder.boundary('operator_load')
                            self.update(mode='force_auto' if command.get('auto_insert') else 'force_holding', alignment_force=dict(active=True, phase='approaching_contact' if approaching else 'loading', requested_force_n=kernel.config.force_n, reason=None, model_based_load=True, force_sensor=False))
                            log.write(json.dumps(dict(event='auto_pin_contact_force_start' if command.get('automatic_pin_contact') else 'operator_force_start',
                                unix_ns=time.time_ns(),guard=guard, requested_force_n=command['force_n'],
                                operator_pin_on_carrier=not command.get('automatic_pin_contact',False),
                                contact_evidence=command.get('contact_evidence'))) + '\n')
                            log.flush()
                        if force_session is not None:
                            if not self.state['right_arm_frozen'] or human or reset is not None or self.state['playing']:
                                raise RuntimeError('Force ownership changed; load stopped')
                            guard_stop = getattr(hardware,'guarded_load_status',None)
                            if guard_stop and guard_stop.get('id') == force_session['guard']['id']:
                                raise RuntimeError('Independent motor guard stopped load: '+str(guard_stop.get('reason')))
                            kernel = force_session['kernel']
                            position, rotation = kernel.kinematics.pose(measured[:6])
                            automatic_contact = force_session.get('automatic_pin_contact',False)
                            approach = force_session.get('approach')
                            drop = (approach.update(now, position,
                                        load_established=bool(kernel.diagnostics.get('approach_load_established')))
                                    if approach is not None else force_session['drop'].update(now,position,
                                        contact_confirmed=not automatic_contact, contact_candidate=automatic_contact))
                            stop_reason = drop.get('stop_reason')
                            if (force_session.get('generation') is not None and
                                    (force_session['generation'] != self._nudge_generation or self.alignment is None)):
                                stop_reason = 'alignment_context_changed'
                            auto_sample = None
                            settling = now < force_session.get('loaded_ready_at', 0)
                            if (not settling and 'loaded_ready_at' in force_session
                                    and force_session.get('auto') is None and not stop_reason):
                                from pin_alignment_insert import AutoInsertion
                                force_session['auto'] = AutoInsertion(kernel.anchor_position,
                                    force_session['config'], now)
                            if force_session.get('auto') is not None and not stop_reason:
                                if force_session['generation'] != self._nudge_generation or self.alignment is None:
                                    stop_reason = 'alignment_context_changed'
                                else:
                                    proposal = self.alignment.status()
                                    auto_sample = force_session['auto'].update(now, position, proposal,
                                        lambda preview, frame_id, applying=False: self.validate_nudge_preview(
                                            preview, measured if applying else preview.get('measured_state14', []),
                                            frame_id, allow_zero=True),
                                        baseline_armed=drop['phase']=='armed')
                                    stop_reason = auto_sample.get('stop_reason')
                                    force_session['xy_goal'] = auto_sample['xy_goal']
                                    count = auto_sample['nudge_count']
                                    if count != force_session.get('logged_auto_count', 0):
                                        force_session['logged_auto_count'] = count
                                        log.write(json.dumps(dict(event='operator_auto_insert_nudge', unix_ns=time.time_ns(),
                                            nudge_count=count, frame_id=proposal.get('frame_id'),
                                            goal_xy_m=force_session['xy_goal'].tolist(), measured_xyz_m=position.tolist()))+'\n');log.flush()
                            if self._force_xy_requested is not None:
                                request_xy = self._force_xy_requested
                                self._force_xy_requested = None
                                delta = self.validate_nudge_preview(request_xy['proposal'],measured,request_xy['frame_id'])
                                xy_goal = loaded_xy_nudge_goal(position, delta, kernel.anchor_position)
                                force_session['xy_goal'] = xy_goal
                                force_session['manual_nudge'] = dict(frame_id=request_xy['frame_id'],
                                    started=now, start_position=position.copy(), goal_xy=xy_goal.copy())
                                self.update(alignment_nudge=dict(status='executing',frame_id=request_xy['frame_id'],xy_only=True))
                                log.write(json.dumps(dict(event='operator_loaded_xy_nudge',unix_ns=time.time_ns(),frame_id=request_xy['frame_id'],proposal_delta_xyz_m=delta.tolist(),requested_delta_xyz_m=[float(delta[0]),float(delta[1]),0.],start_measured_xyz_m=position.tolist(),goal_xy_m=xy_goal.tolist()))+'\n');log.flush()
                            phase = ('approaching_contact' if approach is not None else
                                     'settling_contact' if settling else 'holding')
                            difference = force_session['xy_goal']-force_session['xy_target']
                            distance = np.linalg.norm(difference)
                            if distance:
                                nudge_speed = (AUTO_NUDGE_SPEED_M_S if force_session.get('auto') is not None
                                               else LOADED_NUDGE_SPEED_M_S)
                                force_session['xy_target'] += difference*min(1.,nudge_speed*dt/distance)
                            if force_session.get('manual_nudge') is not None:
                                progress=loaded_nudge_progress(force_session['manual_nudge'],
                                    position, force_session['xy_target'], now)
                                if (progress['status'] in ('done','stalled') and
                                        force_session['manual_nudge'].get('logged_status') != progress['status']):
                                    log.write(json.dumps(dict(event='operator_loaded_xy_result',unix_ns=time.time_ns(),
                                        measured_xyz_m=position.tolist(),**progress))+'\n');log.flush()
                                    force_session['manual_nudge']['logged_status']=progress['status']
                                self.update(alignment_nudge=progress)
                            commanded_xy = force_session['xy_target'].copy()
                            if auto_sample is not None:
                                commanded_xy += auto_sample['dither_xy']
                                phase = auto_sample['phase']
                            left_arm = arms.get('left',{})
                            velocity6 = np.asarray(left_arm.get('velocity_rad_s',[]),dtype=float)
                            gravity6 = left_arm.get('gravity_model_torque_nm')
                            if velocity6.shape != (6,) or not np.isfinite(velocity6).all():
                                raise RuntimeError('Measured LEFT joint velocity unavailable; force stopped')
                            velocity14 = np.zeros(14);velocity14[:6] = velocity6
                            output = kernel.step(measured,velocity14,commanded_xy,now,gravity6=gravity6,
                                                 limits=(hardware.lower[:6],hardware.upper[:6]))
                            if not output['valid']:
                                stop_reason = output.get('reason') or 'Force controller refused command'
                            if stop_reason:
                                diagnostics = self.force_measurement_diagnostics(kernel,measured,velocity6,position,now,output)
                                description = self.force_stop_description(stop_reason,diagnostics)
                                stopped_guard_id = force_session['guard']['id']
                                target[:6] = measured[:6]
                                if force_session.get('automatic_pin_contact'):
                                    self.finish_auto_pin_contact()
                                force_session = None
                                self._force_xy_requested = None
                                self.update(mode='holding', alignment_force=dict(active=False,phase='off',reason=description,
                                            reason_code=stop_reason,diagnostics=diagnostics,measured_xyz=position.tolist(),
                                            downward_travel_mm=-diagnostics['vertical_travel_m']*1000,
                                            insertion_confirmed=False,model_based_load=True,force_sensor=False))
                                log.write(json.dumps(dict(event='operator_force_stop',unix_ns=time.time_ns(),
                                          measured_timestamp_ns=measured_at,reason=description,reason_code=stop_reason,
                                          diagnostics=diagnostics,measured_state14=measured.tolist(),guard_id=stopped_guard_id))+'\n');log.flush()
                            elif approach is not None and drop['contact_ready']:
                                # Keep the emitted torque and measured velocity
                                # history intact across the guarded handoff.
                                # Blend Z support for 1s, then allow another .5s
                                # of loaded settling before visual XY motion.
                                from pin_alignment_search import LoadedDropMonitor
                                previous_guard = force_session['guard']
                                kernel.begin_loaded(measured, now, dict(force_n=force_session['force_n'],
                                    force_slew_n_s=APPROACH_FORCE_SLEW_N_S,
                                    max_lateral_travel_m=.03, max_tool_speed_m_s=.05,
                                    max_duration_s=force_session['config'].max_duration_s,
                                    vertical_damping_ns_m=20., xy_integral_gain_n_m_s=600.,
                                    xy_integral_limit_n=1.5, xy_integral_slew_n_s=.5))
                                guard = dict(previous_guard, id=secrets.token_hex(12),
                                    phase='loaded_insertion', transition_from=previous_guard['id'],
                                    anchor_tool_position_m=kernel.anchor_position.tolist(),
                                    anchor_tool_rotation=kernel.anchor_rotation.tolist(),
                                    max_xy_m=.03, max_down_m=.004, max_up_m=.005,
                                    max_duration_s=force_session['config'].max_duration_s)
                                force_session.update(guard=guard,
                                    drop=LoadedDropMonitor(now, position, force_session['config']),
                                    loaded_ready_at=now+1.5, xy_target=position[:2].copy(),
                                    xy_goal=position[:2].copy())
                                del force_session['approach']
                                target[:6] = measured[:6]
                                force_command = (output['assist_torque_nm'], guard)
                                log.write(json.dumps(dict(event='auto_pin_contact_approach_complete',
                                    unix_ns=time.time_ns(), measured_xyz_m=position.tolist(),
                                    approach_downward_travel_m=drop['downward_from_start_m'],
                                    physical_contact_confirmed=False))+'\n');log.flush()
                                log.write(json.dumps(dict(event='auto_pin_contact_loaded_start',
                                    unix_ns=time.time_ns(), guard=guard, continuous_transfer=True,
                                    requested_force_n=force_session['force_n'], physical_contact_confirmed=False))+'\n');log.flush()
                                self.update(mode='force_auto', alignment_force=dict(active=True,
                                    phase='settling_contact', automatic_pin_contact=True, auto_insert=True,
                                    reason='Maintaining load while contact settles before XY nudges',
                                    model_based_load=True, force_sensor=False))
                            else:
                                target[:6] = measured[:6]
                                force_command = (output['assist_torque_nm'],force_session['guard'])
                                active_diagnostics = dict(output.get('diagnostics') or {})
                                if 'raw_sdk_tool_speed_m_s' in active_diagnostics:
                                    active_diagnostics.setdefault('sdk_tool_speed_m_s',active_diagnostics['raw_sdk_tool_speed_m_s'])
                                display_config = force_session['config']
                                self.update(alignment_force=dict(active=True,phase=phase if settling or drop['phase']=='armed' else drop['phase'],
                                    automatic_pin_contact=force_session.get('automatic_pin_contact',False),
                                    requested_force_n=kernel.config.force_n,ramped_force_n=kernel.ramped_force_n,
                                    auto_insert=bool(force_session['command'].get('auto_insert')),
                                    baseline_authority=drop['baseline_authority'],
                                    nudge_count=auto_sample['nudge_count'] if auto_sample else 0,
                                    axial_retry_mm=auto_sample['axial_retry_m']*1000 if auto_sample else 0.,
                                    axial_retry_limit_mm=auto_sample['axial_retry_limit_m']*1000 if auto_sample else 0.,
                                    accumulation_remaining_s=auto_sample.get('accumulation_remaining_s', 0.) if auto_sample else 0.,
                                    waiting_reason=auto_sample.get('waiting_reason') if auto_sample else None,
                                    preview_unavailable=auto_sample.get('preview_unavailable', False) if auto_sample else False,
                                    reason=None,diagnostics=active_diagnostics,
                                    measured_xyz=position.tolist(),baseline_z_m=drop['baseline_z_m'],
                                    downward_travel_mm=drop['downward_from_start_m']*1000,
                                    xy_target=force_session['xy_target'].tolist(),model_based_load=True,force_sensor=False,
                                    config=dict(oscillation_amplitude_mm=display_config.oscillation_amplitude_m*1000,
                                                oscillation_frequency_hz=display_config.oscillation_frequency_hz,max_duration_s=force_session['config'].max_duration_s,
                                                drop_threshold_mm=force_session['config'].insertion_drop_m*1000,
                                                drop_debounce_s=force_session['config'].insertion_debounce_s)))
                        if nudge_job is not None and nudge_job['generation'] != self._nudge_generation:
                            if nudge_future is not None:
                                nudge_future.cancel()
                            nudge_future = nudge_step = nudge_job = None
                            self.update(mode='holding', alignment_nudge=dict(status='idle', reason='Alignment context reset; nudge stopped'))
                        if self._nudge_requested is not None:
                            job = self._nudge_requested
                            self._nudge_requested = None
                            if (future is not None or step is not None or reset is not None or human or
                                    not self.state['right_arm_frozen'] or self.state['playing']):
                                raise RuntimeError('Nudge requires paused hold with RIGHT frozen')
                            delta = self.validate_nudge_preview(job['proposal'], measured, job['frame_id'])
                            nudge_job = job
                            if self.expo_recorder:
                                self.expo_recorder.boundary('operator_nudge')
                            nudge_future = pool.submit(self.plan_operator_nudge, hardware, measured.copy(),
                                                       target.copy(), delta.copy(), self._right_freeze_target.copy())
                            self.update(mode='nudge_planning', alignment_nudge=dict(status='planning', frame_id=job['frame_id']))
                        if nudge_future is not None and nudge_future.done():
                            planned = nudge_future.result()
                            nudge_future = None
                            if nudge_job['generation'] != self._nudge_generation or not self.state['right_arm_frozen']:
                                raise RuntimeError('Nudge canceled by context change')
                            self.validate_nudge_preview(nudge_job['proposal'], measured, nudge_job['frame_id'])
                            try:
                                commit=self.validate_nudge_commit(planned,measured)
                            except NudgeReanchorRequired:
                                attempts=nudge_job.get('height_reanchors',0)+1
                                if attempts>4:
                                    raise RuntimeError('Held pose kept changing during height-locked preflight; wait for a steady pose and retry')
                                nudge_job['height_reanchors']=attempts
                                delta=self.validate_nudge_preview(nudge_job['proposal'],measured,nudge_job['frame_id'])
                                nudge_future=pool.submit(self.plan_operator_nudge,hardware,measured.copy(),
                                                         target.copy(),delta.copy(),self._right_freeze_target.copy())
                                self.update(mode='nudge_planning',alignment_nudge=dict(status='planning',frame_id=nudge_job['frame_id'],
                                    reason='Reanchoring XY path to the fresh held height'))
                            else:
                                planned['metadata']=dict(planned['metadata'],**commit,
                                                         height_reanchors=nudge_job.get('height_reanchors',0))
                                target=np.asarray(planned['waypoints14'][0],dtype=float).copy()
                                nudge_step=planned['trajectory']
                                nudge_step.reset_started(now)
                                self.update(mode='nudging', alignment_nudge=dict(status='executing', frame_id=nudge_job['frame_id'], metadata=planned['metadata']))
                                log.write(json.dumps(dict(event='operator_nudge', unix_ns=time.time_ns(),
                                                          frame_id=nudge_job['frame_id'], measured=measured.tolist(),
                                                          target=np.asarray(planned['target14']).tolist(),
                                                          nudge_left_mm=nudge_job['proposal']['nudge_left_mm'],commit_validation=planned['metadata']))+'\n')
                                log.flush()
                        if nudge_step is not None:
                            nudge_command_tick = True
                            target, nudge_done = nudge_step.tick(now, dt, target)
                            if nudge_done:
                                nudge_step = None
                                self.update(mode='holding', alignment_nudge=dict(status='done', frame_id=nudge_job['frame_id']))
                                nudge_job = None
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
                            self.clear_right_freeze('Collection lifecycle transition')
                            self.reset_alignment()
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
                        if not human and reset is None and future is None and step is None and nudge_future is None and nudge_step is None and force_session is None and force_command is None:
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
                                policy_actions = actions.copy()
                                actions = self.project_right_hold(actions)
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
                                    step.policy_actions = policy_actions[:count].copy()
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
                        if nudge_future is not None:
                            nudge_future.cancel()
                        nudge_future = nudge_step = nudge_job = None
                        self._nudge_requested = None
                        if force_session is not None:
                            target[:6] = measured[:6]
                            if force_session.get('automatic_pin_contact'):
                                self.finish_auto_pin_contact()
                        force_session = None
                        force_command = None
                        self._force_requested = self._force_xy_requested = None
                        self.update(alignment_force=dict(active=False,phase='off',reason=str(exc),model_based_load=True,force_sensor=False))
                        self.update(alignment_nudge=dict(status='rejected', reason=str(exc)))
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
                            self.clear_right_freeze('Collection lifecycle transition')
                            self.reset_alignment()
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
                        target = self.project_right_hold(target)
                        if force_command is not None:
                            try:
                                hardware.command_guarded_load(arms,target,*force_command)
                            except GuardedLoadStopped as stopped:
                                fresh = np.asarray(stopped.measured_state14,dtype=float)
                                if fresh.shape != (14,) or not np.isfinite(fresh).all():
                                    raise RuntimeError('Motor guard stop did not provide finite measured joints') from stopped
                                # The motor guard already holds this pose. Remove load,
                                # preserve both commanded grips and the RIGHT freeze target.
                                measured = fresh.copy()
                                target[:6] = fresh[:6]
                                if force_session is not None and force_session.get('automatic_pin_contact'):
                                    self.finish_auto_pin_contact()
                                force_session = None
                                force_command = None
                                self._force_requested = self._force_xy_requested = None
                                self.update(mode='holding',alignment_force=dict(active=False,phase='off',
                                            reason=str(stopped),guard_stop=stopped.guard_stop,
                                            model_based_load=True,force_sensor=False))
                                hardware.command(target)
                                log.write(json.dumps(dict(event='operator_force_guard_stop',unix_ns=time.time_ns(),
                                                          guard_stop=stopped.guard_stop,held_target=target.tolist()))+'\n')
                                log.flush()
                        else:
                            hardware.command(target)
                        if self.expo_recorder and last_policy_clock is not None:
                            self.expo_recorder.command(target.tolist(), time.monotonic_ns(), last_policy_clock[1])
                    source = 'human' if human and any(released.values()) else 'operator_load' if force_command is not None else 'operator_nudge' if nudge_command_tick else 'policy' if policy_clock else 'hold'
                    self.update(hold_status=('gravity_assist' if all(released.values()) else 'partial_hold')
                                if source == 'human' else 'commanded', control_source=source)
                    if self.recorder:
                        for side, offset in (('left', 0), ('right', 7)):
                            if not released.get(side, False):
                                if side == 'left' and force_command is not None:
                                    assist, active_guard = force_command
                                    arms[side]['gravity_torque_command_nm'] = (np.asarray(arms[side]['gravity_model_torque_nm'])+np.asarray(assist)).tolist()
                                    arms[side]['command'] = dict(mode='guarded_gravity', assist_torque_nm=list(assist), guard=active_guard)
                                else:
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
                            alignment_force=dict(self.state['alignment_force']) if source == 'operator_load' else None,
                            control_source_by_arm={side: 'human' if released.get(side, False) else
                                'frozen_hold' if side == 'right' and self._right_freeze_target is not None else
                                source if side == 'left' and source in ('operator_nudge','operator_load') else
                                'policy' if source == 'policy' else 'hold' for side in ('left', 'right')},
                            camera_frames=hardware.cameras.recording_frames(),
                            policy_time_s=policy_clock[1] if policy_clock else None,
                            policy_clock_rate=policy_clock[2] if policy_clock else None,
                            control_compute_s=time.monotonic()-now,
                            inference_sequence=policy_clock[0] if policy_clock else sequence-1,
                            actions_per_step=self.state['actions_per_step'], blend_actions=self.state['blend_actions']))
                        sample_sequence += 1
                    if now - last_publish >= 0.05:
                        proposal = self.alignment.status() if self.alignment is not None else None
                        proposal_pose = np.asarray((proposal or {}).get('measured_state14', []))
                        pose_current = (proposal_pose.shape == (14,) and
                            np.isfinite(proposal_pose).all() and
                            np.max(np.abs(proposal_pose[[0,1,2,3,4,5,7,8,9,10,11,12]] -
                                          measured[[0,1,2,3,4,5,7,8,9,10,11,12]])) <= .01)
                        with self.lock:
                            self.state['pin_nudge_detection'] = self.pin_nudge_detector.update(
                                proposal, time.monotonic_ns(), self._right_freeze_target is not None,
                                eligible=bool(pose_current and not human and reset is None
                                    and now-self.heartbeat_at <= LEASE_SECONDS
                                    and self.state['mode'] in ('holding','queued','predicting','executing')))
                        self.update(measured=measured.tolist(), target=target.tolist(), cameras=cameras)
                        last_publish = now
                    time.sleep(max(0, 1 / hardware.config['control_hz'] - (time.monotonic() - now)))
        except BaseException as exc:
            self.update(alignment_force=dict(active=False,phase='off',reason='Control fault: '+str(exc),model_based_load=True,force_sensor=False))
            self.clear_right_freeze('Control fault')
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
                            error=f'{exc}; motion blocked. Use Reconnect.')
                while not self.stop_event.is_set():
                    try:
                        if target is None or getattr(hardware, 'motor_client', None) is not None:
                            hardware.hold_available()
                        else:
                            hardware.command(target)
                        self.update(hold_status='commanded')
                    except BaseException as hold_error:
                        self.update(hold_status='failed', error=f'{exc}; hold command failed: {hold_error}. '
                                    'Powered hold is not reliable. Check USB/power, '
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
                    expo_objective='gear_and_pin', expo_prefix_length=1, critic_options=None, alignment_options=None):
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
                            critic_options=critic_options, alignment_options=alignment_options)
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
                 expo_objective='gear_and_pin', expo_prefix_length=1, critic_options=None, alignment_options=None):
        self.url = url
        self.hardware_factory = hardware_factory
        self.health_request = health_request
        self.data_dir = data_dir
        self.expo_record_dir = expo_record_dir
        self.expo_objective = expo_objective
        self.expo_prefix_length = expo_prefix_length
        self.alignment_options = alignment_options
        self.critic_options = critic_options
        self.lock = threading.RLock()
        self.recovery_cancel = threading.Event()
        self.recovery_worker = None
        self.process = self.pipe = None
        self.state = Controller(url, expo_record_dir=expo_record_dir, expo_objective=expo_objective,
                                expo_prefix_length=expo_prefix_length, critic_options=critic_options, alignment_options=alignment_options).snapshot()
        self.spawn()

    def spawn(self):
        context = multiprocessing.get_context('spawn')
        self.pipe, child_pipe = context.Pipe()
        self.process = context.Process(target=session_process,
                                       args=(child_pipe, self.url, self.hardware_factory,
                                             self.health_request, self.data_dir, self.expo_record_dir,
                                             self.expo_objective, self.expo_prefix_length, self.critic_options, self.alignment_options), daemon=True)
        self.process.start()
        child_pipe.close()

    def rpc(self, method, *args):
        with self.lock:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError('Robot session exited. Use Reconnect.')
            try:
                self.pipe.send((method, args))
                if not self.pipe.poll(1):
                    # Never consume a late reply as the answer to another command.
                    self.pipe.close()
                    raise RuntimeError('Robot session is unresponsive. Use Reconnect.')
                ok, result = self.pipe.recv()
            except (OSError, EOFError, ValueError) as exc:
                raise RuntimeError('Robot session connection failed. Use Reconnect.') from exc
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

    def request_alignment_force(self, values):
        return self.rpc('request_alignment_force', values)

    def request_alignment_insert(self, values):
        return self.rpc('request_alignment_insert', values)


    def request_alignment_force_stop(self):
        return self.rpc('request_alignment_force_stop')

    def request_alignment_nudge(self, frame_id):
        return self.rpc('request_alignment_nudge', frame_id)

    def capture_alignment_pin(self, frame_id, pin_px):
        return self.rpc('capture_alignment_pin',frame_id,pin_px)

    def clear_alignment_pin(self):
        return self.rpc('clear_alignment_pin')

    def clear_alignment_hole(self):
        return self.rpc('clear_alignment_hole')

    def capture_alignment_hole(self, frame_id, hole_px=None):
        return self.rpc('capture_alignment_hole', frame_id, hole_px)

    def configure_alignment(self, values):
        return self.rpc('configure_alignment', values)

    def reset_alignment(self):
        return self.rpc('reset_alignment')

    def alignment_frame(self, expected_id):
        return self.rpc('alignment_frame', expected_id)

    def request_right_arm_freeze(self, frozen):
        return self.rpc('request_right_arm_freeze', frozen)

    def configure_right_auto_freeze(self, enabled):
        return self.rpc('configure_right_auto_freeze', enabled)

    def configure_pin_nudge_detection(self, enabled):
        return self.rpc('configure_pin_nudge_detection', enabled)

    def configure_auto_pin_contact(self, enabled, load=None):
        return self.rpc('configure_auto_pin_contact', enabled, load)

    def set_camera_gain(self, name, value):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Wait for recovery before adjusting camera gain')
            return self.rpc('set_camera_gain', name, value)

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

    def recover(self):
        with self.lock:
            if self.recovery_worker and self.recovery_worker.is_alive():
                raise RuntimeError('Connection recovery is already running')
            if self.snapshot()['mode'] not in ('fault', 'fault_hold', 'holding', 'stopped'):
                raise RuntimeError('Stop motion before reconnecting')
            self.recovery_cancel.clear()
            self.state.update(mode='recovering', prediction=None, hold_status='unverified',
                              error='Reconnecting arms')
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
                                      error='Recovery canceled')
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

        def send(self, status, body, mime='application/json', headers=None):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
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
            if path == '/api/alignment/image':
                expected = parse_qs(urlsplit(self.path).query).get('frame_id', [None])[0]
                frame = controller.alignment_frame(expected)
                if frame is None:
                    return self.send(409, {'error': 'Analyzed frame unavailable or expired'})
                return self.send(200, frame[1], 'image/jpeg', {'X-Alignment-Frame-Id': frame[0]})
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
                if path == '/api/alignment/force/start':
                    controller.request_alignment_force(payload)
                elif path == '/api/alignment/force/insert':
                    controller.request_alignment_insert(payload)
                elif path == '/api/alignment/force/stop':
                    controller.request_alignment_force_stop()
                elif path == '/api/alignment/nudge':
                    controller.request_alignment_nudge(payload.get('frame_id'))
                elif path == '/api/alignment/pin':
                    controller.capture_alignment_pin(payload.get('frame_id'),payload.get('pin_px'))
                elif path == '/api/alignment/pin/clear':
                    controller.clear_alignment_pin()
                elif path == '/api/alignment/capture':
                    controller.capture_alignment_hole(payload.get('frame_id'),payload.get('hole_px'))
                elif path == '/api/alignment/config':
                    controller.configure_alignment(payload)
                elif path == '/api/alignment/reset':
                    controller.clear_alignment_hole()
                elif path == '/api/freeze-right':
                    controller.request_right_arm_freeze(payload.get('frozen'))
                elif path == '/api/auto-freeze-right':
                    controller.configure_right_auto_freeze(payload.get('enabled'))
                elif path == '/api/pin-nudge-detection':
                    controller.configure_pin_nudge_detection(payload.get('enabled'))
                elif path == '/api/auto-pin-contact':
                    controller.configure_auto_pin_contact(payload.get('enabled'), payload.get('load'))
                elif path in ('/api/stop', '/api/pause'):
                    controller.stop()
                elif path == '/api/heartbeat':
                    controller.heartbeat(payload)
                elif path == '/api/connect':
                    controller.connect()
                elif path == '/api/recover':
                    controller.recover()
                elif path == '/api/reset':
                    controller.submit('reset')
                elif path.startswith('/api/collection/'):
                    controller.collection(path.rsplit('/', 1)[-1], payload)
                elif path == '/api/task':
                    controller.select_task(payload.get('task'))
                elif path == '/api/camera/gain':
                    controller.set_camera_gain(payload.get('camera'), payload.get('gain'))
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
    parser.add_argument('--critic', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional asynchronous local action-value diagnostic')
    parser.add_argument('--critic-checkpoint', type=Path, default=ROOT/'work/action-critic-mixed-20261029/run/best.pt')
    parser.add_argument('--critic-python', type=Path, default=ROOT/'work/state-critic-20261028/venv/bin/python')
    parser.add_argument('--critic-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--alignment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--alignment-backend', choices=('opencv','cnn'), default='opencv',
                        help='Left-wrist hole localization backend (default OpenCV real-rim verification)')
    args = parser.parse_args()
    alignment_options = dict(backend=args.alignment_backend) if args.alignment else None
    critic_options = dict(checkpoint=args.critic_checkpoint, python=args.critic_python,
                          device=args.critic_device) if args.critic else None
    if args.demo:
        from hitl_demo import DemoHardware, demo_request
        import tempfile
        data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix='panthera-hitl-demo-'))
        controller = SessionSupervisor(args.url, DemoHardware, demo_request, data_dir, args.expo_record_dir,
                                       args.expo_objective, args.expo_prefix_length, critic_options, alignment_options)
        print(f'SIMULATION — recordings: {data_dir}', flush=True)
    else:
        controller = SessionSupervisor(args.url, data_dir=args.data_dir, expo_record_dir=args.expo_record_dir,
                                       expo_objective=args.expo_objective, expo_prefix_length=args.expo_prefix_length,
                                       critic_options=critic_options, alignment_options=alignment_options)
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
