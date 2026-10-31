"""Nonblocking, opt-in assembly CNN inference in the training Python runtime.

Only JPEGs and measured tool XYZ cross the process boundary. No torch import or
robot command runs in the recording process. Camera order must be verified when
calibrating: port numbers do not establish left/right wrist identity.
The configured worker Python needs the checkpoint bundle's torch/numpy
requirements and Pillow for native JPEG decoding.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


# Model slots describe learned camera roles, not USB port names. The October 22
# setup has the left gear/pin camera on port3 and right carrier camera on port2;
# historical real-adaptation recordings used the opposite port assignments.
DEFAULT_CAMERA_NAMES = ('overhead', 'wrist_port3', 'wrist_port2')


class AlignmentVision:
    def __init__(self, config):
        self.config = dict(config)
        training = Path(__file__).resolve().parents[2] / 'RobotArmTraining'
        self.model_root = Path(config.get('model_root', training / 'outputs/panthera-assembly-cnns'))
        self.python = str(config.get('python', training / '.venv-cnns/bin/python'))
        mapping = config.get('camera_mapping')
        self.camera_names = (tuple(mapping[name] for name in ('overhead', 'left_wrist', 'right_wrist'))
                             if mapping else tuple(config.get('camera_names', DEFAULT_CAMERA_NAMES)))
        if len(self.camera_names) != 3 or len(set(self.camera_names)) != 3:
            raise ValueError('CNN camera_names must contain three distinct names in overhead, left wrist, right wrist order')
        self._lock = threading.Lock()
        self._pending = queue.Queue(maxsize=1)
        self._enabled = False
        self._closed = False
        self._generation = 0
        self._process = None
        self._last = {'status': 'disabled', 'enabled': False}
        self._preview_frames = {}
        self._last_submitted = None
        self._thread = threading.Thread(target=self._run, daemon=True, name='alignment-cnn')
        self._thread.start()

    def set_enabled(self, enabled):
        with self._lock:
            if self._closed:
                return
            enabled = bool(enabled)
            if enabled == self._enabled:
                return
            self._enabled = enabled
            self._generation += 1
            self._last_submitted = None
            self._last = {'status': 'loading' if enabled else 'disabled', 'enabled': enabled}
            self._preview_frames = {}

    def submit(self, frames, arms):
        """Replace queued input; never wait for inference or import model libraries."""
        with self._lock:
            if not self._enabled or self._closed:
                return False
            try:
                selected = [frames[name] for name in self.camera_names]
                timestamps = tuple(int(frame['monotonic_ns']) for frame in selected)
                if timestamps == self._last_submitted:
                    return False
                request = dict(generation=self._generation, camera_names=list(self.camera_names),
                               frame_timestamps_ns=dict(zip(self.camera_names, timestamps)),
                               monotonic_ns=min(timestamps),
                               jpeg=[frame['jpeg'] for frame in selected])
                # Pin context must be real measured FK; do not invent zeros.
                try:
                    xyz = [float(value) for side in ('left', 'right')
                           for value in arms[side]['tool_pose_base']['position_m']]
                    request['tool_xyz_m'] = xyz if len(xyz) == 6 else None
                except (KeyError, TypeError, ValueError):
                    request['tool_xyz_m'] = None
            except (KeyError, TypeError, ValueError) as exc:
                self._last = {'status': 'error', 'enabled': True, 'error': f'Missing camera frames: {exc}'}
                return False
            self._last_submitted = timestamps
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            self._pending.put_nowait(request)
            return True

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._last)

    def preview(self):
        """Return predictions with the exact input JPEGs, never latest-camera overlays."""
        with self._lock:
            return dict(prediction=copy.deepcopy(self._last), frames=dict(self._preview_frames))

    def _start(self):
        return subprocess.Popen([self.python, '-u', str(Path(__file__).resolve()), '--worker',
                                 str(self.model_root), self.config.get('device', 'auto'),
                                 str(self.config.get('cpu_threads', 2))],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def _run(self):
        while not self._closed:
            try:
                request = self._pending.get(timeout=.1)
            except queue.Empty:
                continue
            with self._lock:
                if not self._enabled or request['generation'] != self._generation:
                    continue
            try:
                if self._process is None or self._process.poll() is not None:
                    self._process = self._start()
                if self._closed:
                    self._process.terminate()
                    return
                wire = {key: value for key, value in request.items() if key != 'jpeg'}
                wire['jpeg'] = [base64.b64encode(data).decode('ascii') for data in request['jpeg']]
                self._process.stdin.write(json.dumps(wire, allow_nan=False) + '\n')
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError('CNN worker exited; check configured Python, Pillow, torch and model_root')
                result = json.loads(line)
                result.update(enabled=True, completed_monotonic_ns=time.monotonic_ns())
            except Exception as exc:
                result = dict(status='error', enabled=True, error=str(exc))
                process = self._process
                if process is not None and process.poll() is None:
                    process.terminate()
            with self._lock:
                if self._enabled and request['generation'] == self._generation:
                    self._last = result
                    self._preview_frames = ({name: data for name, data in zip(request['camera_names'], wire['jpeg'])}
                                            if result.get('status') == 'ready' else {})

    def close(self):
        with self._lock:
            self._closed = True
            self._enabled = False
            self._generation += 1
            self._last = {'status': 'disabled', 'enabled': False}
        process = self._process
        if process is not None:
            process.terminate()
        self._thread.join(timeout=2)
        if process is not None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def _worker(model_root, device='auto', cpu_threads=2):
    """Worker JSON-lines protocol. All heavyweight dependencies stay here."""
    try:
        import io
        import hashlib
        import numpy as np
        from PIL import Image
        import torch
        torch.set_num_threads(max(1, int(cpu_threads)))
        root = Path(model_root).resolve()
        sys.path.insert(0, str(root))
        from tools.gear_alignment_model import GearAlignmentPredictor
        from tools.pin_perception_model import PinPerceptionPredictor
        checkpoints = dict(gear='gear_alignment/real_v4/adapted.pt',
                           pin='pin_perception/carrier_v4/adapted.pt')
        models = dict(gear=GearAlignmentPredictor(root / checkpoints['gear'], device=device),
                      pin=PinPerceptionPredictor(root / checkpoints['pin'], device=device))
        identities = {task: hashlib.sha256((root / path).read_bytes()).hexdigest()
                      for task, path in checkpoints.items()}
        initialization_error = None
    except Exception as exc:
        initialization_error = f'CNN initialization failed: {exc}'
    for line in sys.stdin:
        try:
            if initialization_error:
                raise RuntimeError(initialization_error)
            request = json.loads(line)
            images = []
            for encoded in request['jpeg']:
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    if image.size != (640, 480):
                        raise ValueError('Assembly CNNs require native 640x480 RGB; resizing invalidates calibration')
                    images.append(np.asarray(image.convert('RGB')))
            images = np.stack(images)
            if images.shape != (3, 480, 640, 3):
                raise ValueError('Expected three native RGB camera frames')
            predictions, errors = {}, {}
            for task, model in models.items():
                kwargs = {}
                if task == 'pin':
                    xyz = np.asarray(request.get('tool_xyz_m'), dtype=np.float32)
                    if xyz.shape != (6,) or not np.isfinite(xyz).all():
                        errors[task] = 'Pin CNN needs measured left and right tool XYZ in metres'
                        continue
                    kwargs['end_effector_positions'] = xyz
                prediction = model.predict(images, **kwargs)
                keys = ('keypoints_2d', 'visibility_probability', 'confidence', 'pixel_std',
                        'peak_probability', 'camera_names', 'landmark_names')
                predictions[task] = {key: prediction[key].tolist() if hasattr(prediction[key], 'tolist')
                                     else list(prediction[key]) for key in keys}
                predictions[task]['checkpoint'] = str(root / checkpoints[task])
                predictions[task]['checkpoint_id'] = identities[task]
                predictions[task]['image_size'] = [640, 480]
            result = dict(status='ready', monotonic_ns=request['monotonic_ns'],
                          frame_timestamps_ns=request['frame_timestamps_ns'],
                          camera_names=request['camera_names'], predictions=predictions,
                          camera_mapping=dict(zip(('overhead', 'left_wrist', 'right_wrist'),
                                                  request['camera_names'])),
                          errors=errors, model_root=str(root), device=str(models['gear'].device))
        except Exception as exc:
            result = dict(status='error', error=str(exc))
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[1] == '--worker':
    _worker(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'auto',
            int(sys.argv[4]) if len(sys.argv) > 4 else 2)
