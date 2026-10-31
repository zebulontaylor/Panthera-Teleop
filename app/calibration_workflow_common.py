"""Read-only collector sampling and automatic stable-endpoint collection."""
import copy
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_nudge_calibration_pairs import plateau
from nudge_calibration_common import angle, fit_response, write_report


class CollectorClient:
    def __init__(self, url, data_root):
        from urllib.parse import urlsplit
        address = urlsplit(url)
        if address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost') or address.path not in ('', '/'):
            raise ValueError('local_collector_url_required')
        self.url = url.rstrip('/')
        self.data_root = Path(data_root)

    def get(self, route):
        with urlopen(self.url+route, timeout=1) as response:
            return json.load(response)

    def sequence(self, held, press_id, episode_id):
        state = self.get('/api/status')
        if held and state.get('active') != episode_id:
            raise ValueError('recording_changed')
        payload = dict(command='calibration.sequence', parameters=dict(held=held,
            press_id=press_id, episode_id=episode_id))
        request = Request(self.url+'/api/commands', data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'X-Session-Token': state['token']})
        with urlopen(request, timeout=.5) as response:
            return json.load(response)

    def sample(self):
        before = self.get('/api/status')
        episode = before.get('active')
        if (before.get('demo') or before.get('phase') != 'recording' or not episode
                or Path(episode).name != episode or episode in ('.', '..')):
            raise ValueError('healthy_real_recording_required')
        if before.get('error'):
            raise ValueError('collector_error')
        preview = self.get('/api/alignment-preview')
        prediction = preview['prediction']
        if prediction.get('status') != 'ready':
            raise ValueError('waiting_for_predictions')
        stamp = int(prediction['monotonic_ns'])
        if not 0 <= (time.monotonic_ns()-stamp)/1e9 <= .25:
            raise ValueError('prediction_stale')
        directory = self.data_root/'pending'/episode
        metadata = json.loads((directory/'metadata.json').read_text())
        with (directory/'states.jsonl').open('rb') as stream:
            stream.seek(0, 2)
            offset = max(0, stream.tell()-1048576)
            stream.seek(offset)
            lines = stream.read().split(b'\n')[:-1]
        if offset:
            lines = lines[1:]
        states = [json.loads(line) for line in lines if line.strip()]
        if not states:
            raise ValueError('waiting_for_recorded_states')
        state = min(states, key=lambda r: abs(r['monotonic_ns']-stamp))
        if abs(state['monotonic_ns']-stamp) > 10_000_000:
            raise ValueError('state_prediction_timestamp_skew')
        after = self.get('/api/status')
        if after.get('active') != episode or after.get('phase') != 'recording':
            raise ValueError('recording_changed')
        if not all(c.get('ok') for c in after.get('cameras', {}).values()) or len(after.get('cameras', {})) != 3:
            raise ValueError('healthy_three_camera_stream_required')
        arms = state['arms']
        row = dict(prediction, episode_id=episode,
            time_s=(stamp-metadata['start_monotonic_ns'])/1e9,
            state_monotonic_ns=state['monotonic_ns'], arms=arms,
            q_left=arms['left']['position_rad'], q_right=arms['right']['position_rad'])
        return row, preview.get('frames', {}), after

    @staticmethod
    def tail(path):
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            offset = max(0, stream.tell()-1048576)
            stream.seek(offset)
            lines = stream.read().split(b'\n')[:-1]
        if offset:
            lines = lines[1:]
        return [json.loads(line) for line in lines if line.strip()]

    def sample_board(self):
        """Read exact recorded JPEGs and matched states; no CNN dependency."""
        import base64
        before = self.get('/api/status')
        episode = before.get('active')
        if (before.get('demo') or before.get('phase') != 'recording' or before.get('error')
                or not episode or Path(episode).name != episode or episode in ('.', '..')):
            raise ValueError('healthy_real_recording_required')
        directory = (self.data_root/'pending'/episode).resolve()
        metadata = json.loads((directory/'metadata.json').read_text())
        mapping = metadata['config']['alignment_assist']['camera_mapping']
        all_frames = self.tail(directory/'frames.jsonl')
        groups = {name: [r for r in all_frames if r['camera'] == name] for name in mapping.values()}
        if any(not v for v in groups.values()):
            raise ValueError('waiting_for_recorded_camera_frames')
        stamp = min(max(r['monotonic_ns'] for r in rows) for rows in groups.values())
        selected = {name: min(rows, key=lambda r: abs(r['monotonic_ns']-stamp)) for name, rows in groups.items()}
        if not 0 <= (time.monotonic_ns()-stamp)/1e9 <= .25:
            raise ValueError('recorded_frames_stale')
        states = self.tail(directory/'states.jsonl')
        if not states:
            raise ValueError('waiting_for_recorded_states')
        state = min(states, key=lambda r: abs(r['monotonic_ns']-stamp))
        if abs(state['monotonic_ns']-stamp) > 10_000_000:
            raise ValueError('state_frame_timestamp_skew')
        frames = {}
        for name, frame in selected.items():
            path = (directory/frame['path']).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('image_path_outside_recording')
            frames[name] = base64.b64encode(path.read_bytes()).decode('ascii')
        after = self.get('/api/status')
        if after.get('active') != episode or after.get('phase') != 'recording':
            raise ValueError('recording_changed')
        if len(after.get('cameras', {})) != 3 or not all(c.get('ok') for c in after['cameras'].values()):
            raise ValueError('healthy_three_camera_stream_required')
        row = dict(status='ready', episode_id=episode, monotonic_ns=stamp,
            time_s=(stamp-metadata['start_monotonic_ns'])/1e9,
            state_monotonic_ns=state['monotonic_ns'], arms=state['arms'],
            q_left=state['arms']['left']['position_rad'], q_right=state['arms']['right']['position_rad'],
            camera_mapping=mapping, frame_timestamps_ns={name: r['monotonic_ns'] for name, r in selected.items()}, predictions={})
        return row, frames, after


class Workflow:
    name = 'base'

    def __init__(self, output):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError('choose_a_new_empty_output_directory')
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.pairs = []
        self.events = []
        self.references = []
        self.last = None
        self.session = None
        self.split = None
        self.episode = None
        self.axis = None
        self.phase = 'idle'
        self.message = 'Start a real recording, then begin a session.'
        self.report = None
        self.observation_reason = None
        self.latest_status = {}
        self.session_episodes = {}
        self.before = None
        self.reference = None
        self.counter = 0
        self.last_recorded_stamp = None

    def begin(self, settings):
        if self.last is None:
            raise ValueError('fresh_recording_observation_required')
        if self.phase not in ('idle', 'complete', 'cancelled'):
            raise ValueError('cancel_or_finish_current_session_first')
        session = settings.get('session', '').strip()
        split = settings.get('split')
        if not session or len(session) > 64 or split not in ('train', 'validation'):
            raise ValueError('session_name_and_explicit_split_required')
        if session in self.session_episodes:
            raise ValueError('use_a_new_session_name')
        episode = self.last['episode_id']
        if episode in {v[0] for v in self.session_episodes.values()}:
            raise ValueError('new_session_requires_a_distinct_recording_episode')
        axis = np.asarray(settings.get('axis', [0, 0, 1]), dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-6:
            raise ValueError('unit_task_axis_required')
        if self.axis is not None and not np.allclose(axis, self.axis):
            raise ValueError('keep_same_axis_or_start_a_separate_workbench_run')
        self.axis, self.session, self.split, self.episode = axis, session, split, episode
        self.session_episodes[session] = (episode, split)
        self.phase = 'reference'
        self.message = 'Hold the demonstrated good alignment still for one second.'
        self.rows = []
        self.before = self.reference = None
        self.counter = 0
        self.persist()

    def command(self, action, settings=None):
        if action == 'begin':
            return self.begin(settings or {})
        if action == 'cancel':
            self.phase = 'cancelled'
            self.message = 'Session cancelled; previously captured evidence is retained.'
        elif action == 'finish':
            self.phase = 'complete'
            self.message = 'Session finished. Begin a new recording/session for more evidence.'
        elif action == 'analyze':
            self.report = fit_response(self.pairs, self.axis)
            sources = [self.output/'pairs.jsonl']
            if (self.output/'observation-cache.jsonl').exists():
                sources.append(self.output/'observation-cache.jsonl')
            write_report(self.output/'response-candidate.json', self.report, sources)
        else:
            raise ValueError('unsupported_workflow_action')
        self.persist()

    def observe(self, row, frames=None, collector_status=None):
        self.latest_status = collector_status or {}
        if self.last is not None and row['monotonic_ns'] <= self.last['monotonic_ns']:
            return
        self.last = row
        if self.phase in ('idle', 'complete', 'cancelled'):
            return
        if row['episode_id'] != self.episode:
            self.phase = 'cancelled'
            self.message = 'Recording changed. Start a new calibration session.'
            self.persist()
            return
        self.rows.append(row)
        self.rows = [r for r in self.rows if r['monotonic_ns'] >= row['monotonic_ns']-1_200_000_000]
        try:
            endpoint = plateau(self.rows, dict(episode_id=self.episode, time_s=row['time_s']))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.observation_reason = str(exc)
            return
        if endpoint['last_monotonic_ns']-endpoint['first_monotonic_ns'] < 600_000_000:
            self.observation_reason = 'Hold still for at least 0.6 seconds of distinct observations.'
            return
        self.observation_reason = None
        # Include the controlled-arm joint pose for demonstrated primitives.
        endpoint['q'] = np.median([r['q_left'] for r in self.rows if r['time_s'] >= row['time_s']-.75], axis=0).tolist()
        endpoint['time_s'] = row['time_s']
        self.stable(endpoint, frames or {})

    def record_observation(self, row):
        if (self.phase in ('idle', 'complete', 'cancelled') or row['episode_id'] != self.episode
                or self.last_recorded_stamp is not None and row['monotonic_ns'] <= self.last_recorded_stamp):
            return
        with (self.output/'observation-cache.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(row, calibration_session=self.session, calibration_split=self.split), allow_nan=False)+'\n')
        self.last_recorded_stamp = row['monotonic_ns']

    def stable(self, endpoint, frames):
        raise NotImplementedError

    def pair(self, start, end, **extra):
        if start['last_monotonic_ns'] >= end['first_monotonic_ns']:
            return False
        row = dict(session_id=self.session, episode_id=self.episode, trial_id=str(len(self.pairs)+1),
            split=self.split, start=copy.deepcopy(start), end=copy.deepcopy(end), **extra)
        self.pairs.append(row)
        self.persist()
        return True

    def event(self, kind, endpoint):
        self.events.append(dict(kind=kind, session_id=self.session, episode_id=self.episode,
            split=self.split, endpoint=copy.deepcopy(endpoint)))
        self.persist()

    def persist(self):
        for name, rows in (('pairs.jsonl', self.pairs), ('events.jsonl', self.events)):
            temporary = self.output/(name+'.tmp')
            temporary.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
            temporary.replace(self.output/name)

    def status(self):
        return dict(workflow=self.name, phase=self.phase, message=self.message,
            observation_reason=self.observation_reason, session=self.session, split=self.split,
            pairs=len(self.pairs), sessions=len(self.session_episodes), report=self.report,
            output=str(self.output.resolve()), verified=False, motion_authorized=False)
