"""Single-writer episode store. Disk I/O never runs on the motor control thread."""
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid

from audio_cues import AudioCues
from episode_pedal import EpisodePedal


EPISODE_MODES = {
    'regular': dict(arms=['left', 'right'], start_delay_seconds=0),
    'single_bolt': dict(arms=['left'], start_delay_seconds=2),
}


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        import os
        os.fsync(f.fileno())
    temporary.replace(path)


class EpisodeStore:
    def __init__(self, root, config, context):
        self.root = Path(root)
        self.config = config
        self.context = context
        for label in ('pending', 'kept', 'review', 'discarded'):
            (self.root / label).mkdir(parents=True, exist_ok=True)
        # Interrupted sessions never silently enter the training set.
        for path in (self.root / 'pending').iterdir():
            if path.is_dir():
                meta_path = path / 'metadata.json'
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'id': path.name}
                meta.update(label='review', complete=False, end_reason='recovered_after_interruption')
                atomic_json(meta_path, meta)
                path.rename(self.root / 'review' / path.name)
        self.events = queue.Queue(maxsize=1024)
        self.control_requests = queue.SimpleQueue()
        self.alignment_requests = queue.SimpleQueue()
        self.alignment_operator_requests = queue.SimpleQueue()
        self.gear_grasp_requests = queue.SimpleQueue()
        self.alignment_settings_requests = queue.SimpleQueue()
        self.alignment_servo_requests = queue.SimpleQueue()
        self.return_pending = False
        self.pending_pedal_decision = None
        self.lock = threading.Lock()
        self.status = dict(phase='starting', active=None, samples=0, duration=0,
                           error=None, last_decision=None, counts={}, control_active=False,
                           return_stage=None, return_error=None)
        self.pedal_classifier = EpisodePedal(config.get('episode_pedal'))
        self.audio = AudioCues(config.get('audio_cues', {}).get('enabled', False), self.update)
        self.update(motion_arms=config.get("motion_arms", ["left", "right"]),
                    reset_arms=config.get("neutral_return", {}).get("arms", ["left", "right"]),
                    start_delay_seconds=config.get("start_delay_seconds", 0), setup_remaining_s=0)
        self.update(episode_mode=next((name for name, settings in EPISODE_MODES.items()
                    if config.get('motion_arms', ['left', 'right']) == settings['arms']
                    and config.get('neutral_return', {}).get('arms', ['left', 'right']) == settings['arms']
                    and config.get('start_delay_seconds', 0) == settings['start_delay_seconds']), 'custom'))
        self.pre_roll = deque()
        self.active = None
        self.baseline = None
        self.trigger_since = None
        self.still_since = None
        self.wait_still = False
        self.last_flush = time.monotonic()
        self.failure = None
        self.refresh_counts()
        self.thread = threading.Thread(target=self.worker, name='episode-writer', daemon=True)
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.status)

    def refresh_counts(self):
        self.update(counts={label: sum(p.is_dir() for p in (self.root / label).iterdir())
                           for label in ('kept', 'review', 'discarded')})

    def submit(self, kind, payload):
        if self.failure:
            raise RuntimeError(f'Recording failed: {self.failure}')
        try:
            self.events.put_nowait((kind, payload))
        except queue.Full:
            self.failure = 'Recorder queue full; session stopped to avoid silent data loss'
            self.update(error=self.failure)
            raise RuntimeError(self.failure)

    def motion_arms(self, sample):
        return [arm for side, arm in sample["arms"].items()
                if side in self.config.get("motion_arms", ["left", "right"])]

    def set_episode_mode(self, mode):
        if not isinstance(mode, str) or mode not in EPISODE_MODES:
            raise ValueError('Episode mode must be regular or single_bolt')
        if not self.thread.is_alive():
            raise RuntimeError('Recorder is stopped')
        request = dict(mode=mode, done=threading.Event())
        self.submit('episode_mode', request)
        while not request['done'].wait(0.2):
            if not self.thread.is_alive():
                raise RuntimeError(self.failure or 'Recorder is stopped')
        if 'error' in request:
            raise RuntimeError(request['error'])
        return request['result']

    def apply_episode_mode(self, mode):
        phase = self.snapshot()['phase']
        if (self.active or self.pending_pedal_decision
                or (self.return_pending and phase != 'holding_neutral')
                or phase in ('returning_to_neutral', 'stopped', 'error')):
            raise RuntimeError('Change episode mode between takes, after reset and pedal classification finish')
        settings = EPISODE_MODES[mode]
        # Shared with the control loop. Existing holds retain their own targets;
        # the selected arms take effect on the next neutral return.
        self.config['motion_arms'] = list(settings['arms'])
        self.config.setdefault('neutral_return', {})['arms'] = list(settings['arms'])
        self.config['start_delay_seconds'] = settings['start_delay_seconds']
        self.pre_roll.clear()
        self.baseline = self.trigger_since = self.still_since = None
        self.update(episode_mode=mode, motion_arms=list(settings['arms']),
                    reset_arms=list(settings['arms']),
                    start_delay_seconds=settings['start_delay_seconds'], setup_remaining_s=0)
        return dict(episode_mode=mode)

    def pose(self, sample):
        return [x for arm in self.motion_arms(sample) for x in arm['position_rad']] + [
            arm['gripper']['position'] for arm in self.motion_arms(sample)]

    def motion(self, sample):
        pose = self.pose(sample)
        if self.baseline is None:
            self.baseline = pose
        joint_count = sum(len(a['position_rad']) for a in self.motion_arms(sample))
        changed = any(abs(p-b) > (self.config['motion_position_rad'] if i < joint_count
                                  else self.config['motion_gripper_rad'])
                      for i, (p, b) in enumerate(zip(pose, self.baseline)))
        speed = max(abs(v) for a in self.motion_arms(sample) for v in
                    a['velocity_rad_s'] + [a['gripper']['velocity']])
        return changed or speed > self.config['motion_velocity_rad_s'], speed

    def start(self, sample):
        if shutil.disk_usage(self.root).free < self.config['minimum_free_bytes']:
            raise RuntimeError('Less than configured free disk reserve; cannot start episode')
        identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        path = self.root / 'pending' / identifier
        path.mkdir()
        self.pedal_classifier.reset()
        self.pedal_classifier.update(sample, sample['monotonic_ns'])
        self.update(episode_end_detected=False, pedal_taps=0,
                    pedal_mode='classification' if self.pedal_classifier.settings.get('enabled', True) else 'idle',
                    pedal_notice=None)
        self.active = dict(path=path, id=identifier, trigger_ns=sample['monotonic_ns'],
                           start_ns=min([e[1]['monotonic_ns'] for e in self.pre_roll] or [sample['monotonic_ns']]),
                           samples=0, frames={name: 0 for name in self.config['cameras']})
        delay = max(0.0, float(self.config.get('start_delay_seconds', 0)))
        original_start_ns = self.active['start_ns']
        if delay:
            self.active['start_ns'] = sample['monotonic_ns'] + int(delay * 1e9)
        self.update(setup_remaining_s=delay)
        self.active['metadata'] = dict(schema_version=1, id=identifier, label='pending', complete=False,
                                      created_utc=datetime.now(timezone.utc).isoformat(),
                                      trigger_monotonic_ns=sample['monotonic_ns'],
                                      start_monotonic_ns=self.active['start_ns'],
                                      original_start_monotonic_ns=original_start_ns,
                                      start_delay_seconds=delay,
                                      config=deepcopy(self.config), context=deepcopy(self.context),
                                      episode_mode=self.snapshot()['episode_mode'],
                                      units=dict(position='rad', velocity='rad/s', torque='Nm', tool_position='m'),
                                      timing='Host monotonic receive timestamps; cameras are not hardware synchronized',
                                      calibration='Uncalibrated cameras; tool poses are URDF-derived in each arm base frame')
        atomic_json(path / 'metadata.json', self.active['metadata'])
        self.active['states_fd'] = (path / 'states.jsonl').open('w')
        self.active['frames_fd'] = (path / 'frames.jsonl').open('w')
        for kind, payload in self.pre_roll:
            self.write(kind, payload)
        self.pre_roll.clear()
        self.update(phase='recording', active=identifier, samples=self.active['samples'])

    def write(self, kind, payload):
        active = self.active
        if kind in ('sample', 'frame'):
            remaining = max(0.0, (active['start_ns'] - payload['monotonic_ns']) / 1e9)
            if kind == 'sample':
                self.update(setup_remaining_s=remaining)
            if payload['monotonic_ns'] < active['start_ns']:
                return
        if kind == 'sample':
            data = dict(payload, episode_time_s=(payload['monotonic_ns']-active['start_ns']) / 1e9)
            active['states_fd'].write(json.dumps(data, allow_nan=False) + '\n')
            active['samples'] += 1
            if active['samples'] == 1 and active['metadata']['episode_mode'] == 'single_bolt':
                self.audio.play('recording_start')
            self.update(samples=active['samples'], duration=data['episode_time_s'])
        elif kind == 'frame':
            name = payload['camera']
            directory = active['path'] / 'images' / name
            directory.mkdir(parents=True, exist_ok=True)
            index = active['frames'][name]
            relative = Path('images') / name / f'{index:07d}.jpg'
            (active['path'] / relative).write_bytes(payload['jpeg'])
            data = {k: v for k, v in payload.items() if k != 'jpeg'}
            data.update(path=str(relative), frame_index=index,
                        episode_time_s=(payload['monotonic_ns']-active['start_ns']) / 1e9)
            active['frames_fd'].write(json.dumps(data) + '\n')
            active['frames'][name] += 1

    def finish(self, label, reason='operator', complete=True, preserve_pedal=False):
        if not self.active:
            return
        active = self.active
        for key in ('states_fd', 'frames_fd'):
            fd = active[key]
            fd.flush()
            import os
            os.fsync(fd.fileno())
            fd.close()
        meta = active['metadata']
        if label != 'discarded' and not active['samples'] and self.config.get('start_delay_seconds', 0):
            label, complete, reason = 'review', False, 'ended_before_setup_complete'
        meta.update(label=label, complete=complete, end_reason=reason,
                    ended_utc=datetime.now(timezone.utc).isoformat(),
                    samples=active['samples'], camera_frames=active['frames'],
                    end_monotonic_ns=time.monotonic_ns())
        atomic_json(active['path'] / 'metadata.json', meta)
        active['path'].rename(self.root / label / active['id'])
        self.active = None
        self.wait_still = True
        self.baseline = None
        self.still_since = None
        self.trigger_since = None
        self.pre_roll.clear()
        if not preserve_pedal:
            self.pedal_classifier.reset()
            self.update(episode_end_detected=False, pedal_taps=0, pedal_mode='idle')
        self.update(active=None, phase='waiting_for_stillness', samples=0, duration=0, setup_remaining_s=0,
                    last_decision=self.snapshot()['last_decision'] if preserve_pedal else dict(id=active['id'], label=label))
        self.refresh_counts()

    def resolve_pedal_decision(self, payload, label, cues):
        pending = self.pending_pedal_decision
        canceled = 'cancel' in cues
        if label or canceled:
            # The first press already finished the take and started reset.
            # A rejected gesture must also release the recording gate.
            label = label or 'review'
            meta = json.loads((pending['path'] / 'metadata.json').read_text())
            meta.update(label=label, complete=not canceled,
                        end_reason='foot_pedal_invalid_gesture' if canceled else 'foot_pedal',
                        classified_monotonic_ns=payload['monotonic_ns'])
            if label != 'discarded' and not meta['samples'] and meta.get('start_delay_seconds', 0):
                label = 'review'
                meta.update(label=label, complete=False, end_reason='ended_before_setup_complete')
            atomic_json(pending['path'] / 'metadata.json', meta)
            destination = self.root / label / pending['id']
            if destination != pending['path']:
                pending['path'].rename(destination)
            self.pending_pedal_decision = None
            self.pedal_classifier.reset()
            self.update(pedal_decision_pending=False, episode_end_detected=False,
                        pedal_mode='idle', pedal_taps=0,
                        pedal_notice=('Pedal gesture not recognized. Take saved for review; '
                                      'recording can resume after reset. Use short taps (0.04–0.5 seconds).'
                                      if canceled else None),
                        last_decision=dict(id=pending['id'], label=label))
            self.refresh_counts()
            if not canceled:
                self.audio.play(label)

    def handle(self, kind, payload):
        if kind == 'episode_mode':
            try:
                payload['result'] = self.apply_episode_mode(payload['mode'])
            except (ValueError, RuntimeError) as exc:
                payload['error'] = str(exc)
            finally:
                payload['done'].set()
            return
        if kind == 'alignment_servo':
            self.alignment_servo_requests.put(payload)
            return
        if kind == 'alignment_trial_settings':
            self.alignment_settings_requests.put(payload)
            return
        if kind == 'gear_grasp':
            self.gear_grasp_requests.put(payload)
            return
        if kind == 'alignment_operator_hold':
            self.alignment_operator_requests.put(payload)
            return
        if kind == 'alignment_assist':
            if type(payload.get('enabled')) is not bool:
                raise ValueError('Alignment assist enabled must be a boolean')
            self.alignment_requests.put(payload['enabled'])
            return
        if kind == 'decision':
            if self.active and payload['episode_id'] == self.active['id']:
                provisional = payload.get('provisional', False)
                self.finish(payload['label'], payload.get('reason', 'operator'),
                            complete=not provisional, preserve_pedal=provisional)
                if not provisional:
                    self.audio.play(payload['label'])
                if self.config.get('neutral_return', {}).get('enabled'):
                    self.return_pending = True
                    self.update(phase='returning_to_neutral', return_stage='countdown', return_error=None)
                    self.control_requests.put(dict(episode_id=payload['episode_id'], label='pending' if provisional else payload['label']))
            return
        if kind == 'return_complete':
            if not self.return_pending:
                return
            with (self.root/'return-events.jsonl').open('a') as log:
                log.write(json.dumps(dict(payload, unix_ns=time.time_ns()), allow_nan=False)+'\n')
            if payload.get('hold_for_ready'):
                self.pre_roll.clear()
                self.update(phase='holding_neutral', return_stage=None,
                            return_error=payload.get('error'),
                            handoff_ready={side: False for side in self.config.get('neutral_return', {}).get('arms', ['left', 'right'])})
                return
            self.return_pending = False
            self.wait_still = True
            self.baseline = None
            self.still_since = None
            self.trigger_since = None
            self.pre_roll.clear()
            self.update(phase='waiting_for_stillness', return_stage=None,
                        return_error=payload.get('error'))
            return
        if kind == 'handoff_ready':
            if not self.return_pending:
                return
            self.return_pending = False
            self.wait_still = False
            self.baseline = self.pose(payload['sample'])
            self.still_since = None
            self.trigger_since = None
            self.pre_roll.clear()
            self.update(phase='armed', return_stage=None, handoff_ready=payload['ready'])
            return
        if kind == 'handoff_event':
            with (self.root/'return-events.jsonl').open('a') as log:
                log.write(json.dumps(dict(payload, unix_ns=time.time_ns()), allow_nan=False)+'\n')
            return
        if kind == 'close':
            self.finish('review', payload.get('reason', 'session_stopped'), complete=False)
            self.update(phase='stopped', control_active=False)
            return
        if kind == 'sample':
            for event in payload.get('foot_pedal', {}).get('events', []):
                if event['pressed']:
                    self.audio.play('press')
        if kind == 'sample' and self.pending_pedal_decision:
            pending = self.pending_pedal_decision
            _, label, cues = self.pedal_classifier.update(payload, pending['trigger_ns'])
            for cue in cues:
                if cue != 'decision_start':
                    self.audio.play(cue)
            if payload.get('foot_pedal', {}).get('events'):
                with (pending['path'] / 'pedal-decision-events.jsonl').open('a') as log:
                    log.write(json.dumps(dict(monotonic_ns=payload['monotonic_ns'],
                                              foot_pedal=payload['foot_pedal'])) + '\n')
            self.update(pedal_taps=self.pedal_classifier.taps)
            self.resolve_pedal_decision(payload, label, cues)
            return
        if self.pending_pedal_decision:
            return
        if self.return_pending:
            # Motor-driven resets are never demonstrations or the next take's pre-roll.
            return
        if self.active:
            label = None
            if kind == 'sample':
                mode, label, cues = self.pedal_classifier.update(payload, self.active['trigger_ns'])
                for cue in cues:
                    if cue != 'decision_start':
                        self.audio.play(cue)
                if 'decision_start' in cues:
                    if 'episode_end_detection' not in self.active['metadata']:
                        self.active['metadata']['episode_end_detection'] = dict(
                            monotonic_ns=self.pedal_classifier.ready_ns,
                            method='foot_pedal',
                            settings=self.config.get('episode_pedal', {}))
                self.update(episode_end_detected=self.pedal_classifier.ready_ns is not None,
                            pedal_taps=self.pedal_classifier.taps,
                            pedal_mode='classification' if mode else 'idle')
            self.write(kind, payload)
            if kind == 'sample' and 'decision_start' in cues:
                pending = dict(id=self.active['id'], trigger_ns=self.active['trigger_ns'],
                               path=self.root / 'review' / self.active['id'])
                self.handle('decision', dict(label='review', episode_id=self.active['id'],
                                            reason='foot_pedal_pending', provisional=True))
                self.pending_pedal_decision = pending
                self.update(pedal_decision_pending=True)
                self.resolve_pedal_decision(payload, label, cues)
            elif label:
                self.handle('decision', dict(label=label, episode_id=self.active['id'], reason='foot_pedal'))
            return
        if kind == 'sample':
            moving, speed = self.motion(payload)
            now = payload['monotonic_ns'] / 1e9
            if self.wait_still:
                # Movement in position is measured against the preceding sample here.
                still = not moving and speed < self.config['motion_velocity_rad_s'] / 2
                self.still_since = (self.still_since or now) if still else None
                self.baseline = self.pose(payload)
                if self.still_since and now-self.still_since >= self.config['rearm_still_seconds']:
                    self.wait_still = False
                    self.pre_roll.clear()
                    self.update(phase='armed')
                return
            self.update(phase='armed')
            self.trigger_since = (self.trigger_since or now) if moving else None
        self.pre_roll.append((kind, payload))
        cutoff = payload['monotonic_ns'] - int(self.config['pre_roll_seconds']*1e9)
        # Camera threads may arrive out of order by a few milliseconds.
        while self.pre_roll and self.pre_roll[0][1]['monotonic_ns'] < cutoff:
            self.pre_roll.popleft()
        if kind == 'sample' and self.trigger_since and now-self.trigger_since >= self.config['motion_confirm_seconds']:
            self.start(payload)

    def worker(self):
        try:
            while True:
                try:
                    kind, payload = self.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    self.handle(kind, payload)
                    now = time.monotonic()
                    if self.active and now-self.last_flush > 0.5:
                        self.active['states_fd'].flush()
                        self.active['frames_fd'].flush()
                        if shutil.disk_usage(self.root).free < self.config['minimum_free_bytes']:
                            raise RuntimeError('Free disk space fell below recording reserve')
                        self.last_flush = now
                finally:
                    self.events.task_done()
                if kind == 'close':
                    break
        except BaseException as exc:
            self.failure = str(exc)
            self.update(error=self.failure, phase='error')
            try:
                self.finish('review', 'recording_error: ' + self.failure, complete=False)
            except Exception:
                pass  # pending metadata will be recovered on the next launch.
            self.update(error=self.failure, phase='error')

    def close(self, reason):
        if self.thread.is_alive():
            self.events.put(('close', {'reason': reason}), timeout=3)
            self.thread.join(timeout=10)
        self.audio.close()
