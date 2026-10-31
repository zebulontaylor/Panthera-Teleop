"""Synchronous forward-only pin regrasp evidence, separate from training episodes."""
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np


class PinRegraspEpisodeStore:
    def __init__(self, root, fps=30, max_camera_age_s=1.0):
        self.root = Path(root)
        self.fps = float(fps)
        self.max_camera_age_ns = int(max_camera_age_s * 1e9)
        if not math.isfinite(self.fps) or self.fps <= 0 or self.max_camera_age_ns <= 0:
            raise ValueError('Positive recording FPS and camera age required')
        self.active = False
        self.path = None
        self.episode_id = None
        self.samples = 0
        self.error = None
        self.complete = False
        self._writer = self._stream = None

    def snapshot(self):
        return dict(active=self.active, path=str(self.path) if self.path else None,
                    episode_id=self.episode_id, samples=self.samples,
                    error=self.error, complete=self.complete)

    def begin(self, metadata):
        if self.active:
            raise RuntimeError('Forward episode already active')
        # Validate serialization before creating an episode.
        json.dumps(metadata, allow_nan=False)
        self.episode_id = uuid.uuid4().hex
        self.path = self.root / self.episode_id
        self.path.mkdir(parents=True)
        self.samples = 0
        self.error = None
        self.complete = False
        self._last_sequence = self._last_camera_ns = None
        self._size = None
        self._metadata = dict(metadata, schema_version=1, episode_id=self.episode_id,
                              label='unverified', camera='left_wrist', phases=['return', 'close'],
                              started_unix_ns=time.time_ns(), started_monotonic_ns=time.monotonic_ns(),
                              complete=False, reason='recording', samples=0,
                              video='left_wrist.avi', samples_file='samples.jsonl', fps=self.fps)
        self._write_metadata()
        self._stream = (self.path / 'samples.jsonl').open('w')
        self.active = True
        return self.snapshot()

    def _write_metadata(self):
        temporary = self.path / 'metadata.json.tmp'
        temporary.write_text(json.dumps(self._metadata, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.path / 'metadata.json')

    def append(self, phase, measured14, target14, jpeg, camera_metadata):
        if not self.active:
            raise RuntimeError('No active forward episode')
        try:
            if phase not in ('return', 'close'):
                raise ValueError('Only return and close phases may be recorded')
            measured, target = list(measured14), list(target14)
            if any(len(v) != 14 or any(isinstance(x, bool) or not math.isfinite(float(x)) for x in v)
                   for v in (measured, target)):
                raise ValueError('Measured state and target must contain 14 finite values')
            camera = dict(camera_metadata)
            if camera.get('camera') != 'left_wrist' or camera.get('error') or camera.get('ok') is False:
                raise ValueError('Healthy left wrist camera required')
            sequence, captured, unix_ns = (camera.get(k) for k in ('sequence', 'monotonic_ns', 'unix_ns'))
            if any(type(v) is not int or v < 0 for v in (sequence, captured, unix_ns)):
                raise ValueError('Camera sequence and capture timestamps required')
            now = time.monotonic_ns()
            if not 0 <= now - captured <= self.max_camera_age_ns:
                raise ValueError('Fresh left wrist camera required')
            if self._last_sequence is not None and (sequence < self._last_sequence or captured < self._last_camera_ns):
                raise ValueError('Camera sequence or timestamp moved backwards')
            if not isinstance(jpeg, bytes) or not jpeg.startswith(b'\xff\xd8'):
                raise ValueError('Left wrist JPEG bytes required')
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid left wrist JPEG')
            size = (frame.shape[1], frame.shape[0])
            if any(camera.get(k, expected) != expected for k, expected in zip(('width', 'height'), size)):
                raise ValueError('Camera dimensions do not match JPEG')
            if self._size is not None and size != self._size:
                raise ValueError('Camera dimensions changed within episode')
            if self._writer is None:
                self._writer = cv2.VideoWriter(str(self.path / 'left_wrist.avi'),
                                              cv2.VideoWriter_fourcc(*'MJPG'), self.fps, size)
                if not self._writer.isOpened():
                    raise RuntimeError('Cannot open left wrist video writer')
                self._size = size
            # Do not serialize image payloads or unrelated camera metadata.
            row = dict(index=self.samples, phase=phase, monotonic_ns=now, unix_ns=time.time_ns(),
                       state=[float(x) for x in measured], action=[float(x) for x in target],
                       camera=dict(camera='left_wrist', sequence=sequence, monotonic_ns=captured,
                                   unix_ns=unix_ns, width=size[0], height=size[1]), video_frame=self.samples)
            self._writer.write(frame)
            self._stream.write(json.dumps(row, allow_nan=False) + '\n')
            self._stream.flush()
            self.samples += 1
            self._last_sequence, self._last_camera_ns = sequence, captured
            return self.snapshot()
        except Exception as exc:
            self.error = str(exc)
            self.finish(False, 'recording_error: ' + self.error)
            raise

    def finish(self, complete=False, reason='interrupted'):
        if not self.active:
            return self.snapshot()
        self.active = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._stream.close()
        self._stream = None
        self.complete = bool(complete) and self.samples > 0 and not self.error
        self._metadata.update(complete=self.complete, reason=str(reason), samples=self.samples,
                              label='unverified', finished_unix_ns=time.time_ns(),
                              finished_monotonic_ns=time.monotonic_ns(), error=self.error)
        self._write_metadata()
        return self.snapshot()
