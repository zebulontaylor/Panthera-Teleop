"""Read-only adapter for operator-kept real recordings (stdlib + NumPy).

Actions are the next measured 20 Hz state, not recorded motor commands. Native
coordinates are radians, ordered left joints 1..6, left gripper, right joints
1..6, right gripper. No interpolation, normalization or actuator remapping occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    try:
        result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in result):
        raise ValueError(f"Expected JSON objects in {path}")
    return result


def _times(rows: list[dict], description: str) -> np.ndarray:
    times = [row.get('monotonic_ns') for row in rows]
    if any(type(t) is not int or not 0 <= t <= np.iinfo(np.int64).max for t in times):
        raise ValueError(f"{description}: monotonic_ns must be nonnegative int64 timestamps")
    result = np.asarray(times, dtype=np.int64)
    if np.any(result[1:] <= result[:-1]):
        raise ValueError(f"{description}: timestamps must be strictly increasing")
    return result


def _nearest(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    return np.where(targets - times[left] <= times[right] - targets, left, right)


@dataclass
class RealEpisode:
    path: Path
    metadata: dict
    fps: float
    grid_monotonic_ns: np.ndarray
    state: np.ndarray
    action: np.ndarray
    state_monotonic_ns: np.ndarray
    action_monotonic_ns: np.ndarray
    state_skew_s: np.ndarray
    action_skew_s: np.ndarray
    frames: dict[str, list[dict]]

    @property
    def time_s(self) -> np.ndarray:
        """Observation grid times relative to the saved episode origin."""
        return (self.grid_monotonic_ns - self.metadata['start_monotonic_ns']) / 1e9

    def nearest_image(self, camera: str, monotonic_ns: int,
                      max_skew_s: float | None = None) -> dict | None:
        """Return original frame metadata plus absolute path/skew; never extrapolate."""
        if max_skew_s is not None and (not math.isfinite(max_skew_s) or max_skew_s < 0):
            raise ValueError('max_skew_s must be finite and nonnegative')
        rows = self.frames.get(camera, [])
        if not rows or not rows[0]['monotonic_ns'] <= monotonic_ns <= rows[-1]['monotonic_ns']:
            return None
        times = np.asarray([row['monotonic_ns'] for row in rows], dtype=np.int64)
        row = rows[int(_nearest(times, np.asarray([monotonic_ns]))[0])]
        skew = (row['monotonic_ns'] - monotonic_ns) / 1e9
        if max_skew_s is not None and abs(skew) > max_skew_s:
            return None
        return {**row, 'path': str((self.path / row['path']).resolve()), 'skew_s': skew}


def load_episode(path: Path, fps: float = 20.0) -> RealEpisode:
    """Load a complete kept real episode and pair each grid state with its successor.

    The regular grid starts at the first measured state and ends at or before the
    last. The final grid point supplies only the preceding action. Signed skews
    compare each selected measurement with its own grid target, including actions.
    Camera timestamps are validated per camera because their JSONL is interleaved.
    """
    path = Path(path).resolve()
    if not math.isfinite(fps) or fps <= 0 or fps > 1e9:
        raise ValueError('fps must be finite and in (0, 1e9]')
    try:
        metadata = json.loads((path / 'metadata.json').read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read episode metadata: {exc}') from exc
    if not isinstance(metadata, dict) or metadata.get('label') != 'kept' or metadata.get('complete') is not True:
        raise ValueError('Require a complete operator-kept episode')
    context = metadata.get('context', {})
    if not isinstance(context, dict):
        raise ValueError('metadata.context must be an object')
    if metadata.get('simulated') or context.get('mode') in ('simulation', 'sim', 'demo') or metadata.get('source_kind') in ('simulation', 'sim'):
        raise ValueError('Simulation recordings are not real demonstrations')
    rows = _rows(path / 'states.jsonl')
    times = _times(rows, 'states')
    if len(times) < 2:
        raise ValueError('Require at least two measured states')
    origin = metadata.get('start_monotonic_ns')
    if type(origin) is not int or not 0 <= origin <= int(times[0]):
        raise ValueError('metadata.start_monotonic_ns must be nonnegative and no later than the first state')
    vectors = []
    for row in rows:
        try:
            if row.get('simulated'):
                raise ValueError('Simulation sample')
            vector = []
            for side in ('left', 'right'):
                arm = row['arms'][side]
                if arm.get('simulated'):
                    raise ValueError('Simulation arm')
                q = np.asarray(arm['position_rad'], dtype=np.float64)
                if q.shape != (6,):
                    raise ValueError('Expected six joint positions per arm')
                vector.extend(q)
                vector.append(float(arm['gripper']['position']))
            if not np.isfinite(vector).all():
                raise ValueError('Nonfinite measured state')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid real state at {row["monotonic_ns"]}: {exc}') from exc
        vectors.append(vector)
    duration = int(times[-1]) - int(times[0])
    count = math.floor(duration * fps / 1e9) + 1
    offsets = np.rint(np.arange(count, dtype=np.float64) * (1e9 / fps)).astype(np.int64)
    grid = times[0] + offsets[offsets <= duration]
    if len(grid) < 2:
        raise ValueError('Episode is shorter than one sampling interval')
    selected = _nearest(times, grid)
    values = np.asarray(vectors, dtype=np.float64)[selected]
    measured = times[selected]
    skew = (measured - grid) / 1e9
    frames: dict[str, list[dict]] = {}
    for frame in _rows(path / 'frames.jsonl'):
        if not isinstance(frame.get('camera'), str) or not isinstance(frame.get('path'), str):
            raise ValueError('Frame requires camera and path strings')
        image = (path / frame['path']).resolve()
        if not image.is_relative_to(path):
            raise ValueError('Frame path escapes episode directory')
        frames.setdefault(frame['camera'], []).append(frame)
    for camera, camera_rows in frames.items():
        _times(camera_rows, f'camera {camera}')
    return RealEpisode(path, metadata, fps, grid[:-1], values[:-1], values[1:],
                       measured[:-1], measured[1:], skew[:-1], skew[1:], frames)
