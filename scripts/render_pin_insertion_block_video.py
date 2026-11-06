#!/usr/bin/env python3
"""Chronological, selection-gated video of whole held-out carrier blocks.

No annotation coordinates or future insertion references enter inference/cache.
Video durations use original image host receipt timestamps, not data timestamps.
"""
from __future__ import annotations
import argparse
import bisect
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.adapt_pin_insertion_cnn import sha256
from scripts.evaluate_pin_insertion_block_model import validate_selection


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_contract(manifest_path, checkpoint, selection_path):
    selection = validate_selection(selection_path, checkpoint, manifest_path)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('split_unit') not in ('carrier_pose_block', 'stationary_carrier_pose_block_within_one_recording'):
        raise ValueError('explicit within-recording carrier block manifest required')
    split_path = Path(manifest['split_definition_path'])
    if sha256(split_path) != manifest['split_definition_sha256']:
        raise ValueError('frozen split SHA mismatch')
    split = json.loads(split_path.read_text())
    if split.get('split_frozen_before_coordinate_annotation_or_fitting') is not True:
        raise ValueError('split must be frozen before coordinate annotation')
    blocks = [b for b in split['blocks'] if b['split'] == 'test']
    if {b['block_id'] for b in blocks} != {'B4', 'B5'}:
        raise ValueError('renderer requires entire frozen B4 and B5 test blocks')
    source = Path(split['source_path'])
    for name, digest in split['source_hashes'].items():
        if sha256(source / name) != digest:
            raise ValueError('source recording SHA mismatch: ' + name)
    # Deliberately do not read manifest frames/coordinates into the rendering path.
    return split, sorted(blocks, key=lambda b: b['first_frame']), source, selection


def state14(state):
    result = []
    for side in ('left', 'right'):
        arm = state['arms'][side]
        q = arm['position_rad']
        if len(q) != 6:
            raise ValueError('six measured joints per arm required')
        result.extend(q + [arm['gripper']['position']])
    if not np.isfinite(result).all():
        raise ValueError('nonfinite measured state')
    return result


def chronological_stream(frames, states, blocks, max_skew_ns=25_000_000):
    """Every source frame exactly once, original ordering; never synthetic state."""
    stamps = [s['monotonic_ns'] for s in states]
    if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('state receipt timestamps must strictly increase')
    previous = None
    seen = set()
    selected = []
    for frame in frames:
        if frame['camera'] != 'wrist_port2':
            continue
        matches = [b for b in blocks if b['first_frame'] <= frame['frame_index'] <= b['last_frame']]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError('overlapping frozen blocks')
        block = matches[0]
        stamp = frame['monotonic_ns']
        if previous is not None and stamp <= previous:
            raise ValueError('image receipt timestamps must strictly increase')
        previous = stamp
        if frame['frame_index'] in seen:
            raise ValueError('duplicate image frame index')
        seen.add(frame['frame_index'])
        if not block['start_s'] <= frame['episode_time_s'] < block['end_s']:
            raise ValueError('frame violates frozen time boundary')
        right = bisect.bisect_left(stamps, stamp)
        choices = [i for i in (right - 1, right) if 0 <= i < len(stamps)]
        i = min(choices, key=lambda i: abs(stamps[i] - stamp))
        if abs(stamps[i] - stamp) > max_skew_ns:
            raise ValueError('no actual recorded state within 25 ms of image receipt')
        selected.append({'frame': frame, 'block': block, 'state14': state14(states[i]),
                         'state_monotonic_ns': stamps[i], 'state_skew_ms': (stamps[i] - stamp) / 1e6})
    for block in blocks:
        expected = set(range(block['first_frame'], block['last_frame'] + 1))
        actual = {r['frame']['frame_index'] for r in selected if r['block']['block_id'] == block['block_id']}
        if expected != actual:
            raise ValueError('source stream missing frozen block frames')
    return selected


def memory_ranges(values, protocol_path, manifest_path, blocks, checkpoint=None):
    if not values and not protocol_path:
        return {}, None
    if not protocol_path:
        raise ValueError('memory requires explicit predeclared clear/locked-grasp protocol')
    protocol = json.loads(Path(protocol_path).read_text())
    diagnostic = protocol.get('protocol_kind') == 'diagnostic causal unverified initialization baseline'
    if diagnostic:
        if checkpoint is None or protocol.get('selected_checkpoint_sha256') != sha256(checkpoint):
            raise ValueError('diagnostic memory checkpoint SHA mismatch')
        if protocol.get('manifest_sha256') != sha256(manifest_path) or any(protocol.get(k) is not False for k in
                ('clear_visibility_verified', 'autonomous_memory_eligible', 'future_insertion_reference_or_GT_seed',
                 'test_predictions_or_scores_seen_before_declaration')):
            raise ValueError('diagnostic memory must remain unverified, ineligible, causal and predeclared')
        expected = {b['block_id']: list(range(b['first_frame'], b['first_frame'] + 5)) for b in blocks}
        if protocol.get('seed_frame_indices') != expected:
            raise ValueError('diagnostic seed must be fixed first five source frames per block')
        ranges = {b: (indices[0], indices[-1]) for b, indices in expected.items()}
        if values and set(values) != {f'{b}:{a}:{z}' for b, (a, z) in ranges.items()}:
            raise ValueError('CLI seed ranges disagree with frozen diagnostic protocol')
        return ranges, protocol
    if protocol.get('manifest_sha256') != sha256(manifest_path) or protocol.get('declared_before_test') is not True:
        raise ValueError('memory protocol must bind manifest and precede test inspection')
    ranges = {}
    by_id = {b['block_id']: b for b in blocks}
    for value in values:
        name, first, last = value.split(':'); first, last = int(first), int(last)
        b = by_id[name]
        if name in ranges or not b['first_frame'] <= first <= last <= b['last_frame']:
            raise ValueError('invalid memory initialization range')
        declaration = protocol.get('blocks', {}).get(name, {})
        if declaration.get('frame_range') != [first, last] or any(declaration.get(k) is not True for k in
                ('clear_visible', 'carrier_pose_locked', 'grasp_locked')):
            raise ValueError('memory range needs externally declared clear view and locked carrier/grasp')
        # Only early source observations may seed memory, never late insertion frames.
        if first != b['first_frame'] or last > b['first_frame'] + 90:
            raise ValueError('memory seed must start at first block frame and end within early 90 frames')
        ranges[name] = (first, last)
    return ranges, protocol


class CausalHoleCache:
    def __init__(self, ranges):
        self.ranges = ranges
        self.block = None
        self.samples = []
        self.value = None

    def observe(self, block, index, current_hole):
        if self.block != block:
            self.block, self.samples, self.value = block, [], None
        bounds = self.ranges.get(block)
        if bounds is None:
            return None
        first, last = bounds
        point = np.asarray(current_hole, float)
        if point.shape != (2,) or not np.isfinite(point).all() or (point < 0).any() or (point > 1).any():
            raise ValueError('invalid current CNN hole prediction')
        if first <= index <= last:
            self.samples.append(point.copy())
        # Finalize only after observing final seed frame; never use future predictions.
        if index >= last and self.value is None:
            if len(self.samples) != last - first + 1:
                raise ValueError('incomplete causal memory seed')
            self.value = np.median(self.samples, axis=0)
        return None if self.value is None else self.value.copy()


def compose(rgb, prediction, cached_hole, text, roi):
    import cv2
    from scripts.render_pin_insertion_heatmaps import spatial_probability
    h, w = rgb.shape[:2]
    if w * 3 != h * 4:
        raise ValueError('native full-frame aspect must be 4:3')
    x, y, rw, rh = roi
    if rw * 3 != rh * 4 or min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > w or y + rh > h:
        raise ValueError('ROI must be in-frame with native 4:3 pixel aspect')
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kp = np.asarray(prediction['keypoints'], float).reshape(2, 2)
    if not np.isfinite(kp).all() or (kp < 0).any() or (kp > 1).any():
        raise ValueError('invalid normalized current CNN keypoints')
    probs = spatial_probability(prediction['keypoint_logits'])
    colors = [(255, 200, 0), (0, 220, 255)]
    panels = [base.copy()]
    for i in range(2):
        strength = cv2.resize((probs[i] / probs[i].max()).astype(np.float32), (w, h))
        alpha = (.65 * strength)[..., None]
        panels.append((base * (1 - alpha) + np.array(colors[i]) * alpha).astype(np.uint8))
    for panel in panels:
        for point, color in zip(kp, colors):
            pixel = np.minimum(np.rint(point * [w, h]).astype(int), [w - 1, h - 1])
            cv2.drawMarker(panel, tuple(pixel), color, cv2.MARKER_CROSS, 15, 2)
        if cached_hole is not None:
            pixel = np.minimum(np.rint(cached_hole * [w, h]).astype(int), [w - 1, h - 1])
            cv2.drawMarker(panel, tuple(pixel), (220, 50, 220), cv2.MARKER_DIAMOND, 20, 2)
    roi_panel = cv2.resize(panels[0][y:y + rh, x:x + rw], (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panels[0], (x, y), (x + rw, y + rh), (180, 180, 180), 1)
    canvas = np.zeros((2 * h + 186, 2 * w, 3), np.uint8)
    for panel, (cx, cy) in zip((panels[0], roi_panel, panels[1], panels[2]), ((0, 110), (w, 110), (0, 110 + h), (w, 110 + h))):
        canvas[cy:cy + h, cx:cx + w] = panel
    def line(value, yy, scale=.56):
        cv2.putText(canvas, value, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)
    line('REAL B4/B5 | held-out carrier blocks, SAME recording and likely SAME grasp', 24, .7)
    line(text, 48)
    line('Top: full camera frame | fixed ROI. Bottom: PIN likelihood | HOLE likelihood.', 71)
    line('Cyan/yellow crosses: CURRENT CNN. Magenta diamond: CACHED early CNN hole; UNVERIFIED DIAGNOSTIC.', 94)
    logit = float(np.asarray(prediction['active_logit']).flat[0])
    if not np.isfinite(logit):
        raise ValueError('invalid gate logit')
    gate = 1 / (1 + np.exp(-np.clip(logit, -60, 60)))
    line(f'Gate score {gate:.3f}: unverified, not force/contact certification. No motion executed.', 2 * h + 137)
    line('Spatial maps independently normalized for display. No GT overlays or future reference coordinates.', 2 * h + 161)
    return canvas


def render(args):
    split, blocks, source, selection = load_contract(args.manifest, args.checkpoint, args.selection)
    ranges, protocol = memory_ranges(args.memory_init_frame_range, args.memory_protocol, args.manifest, blocks, args.checkpoint)
    stream = chronological_stream(read_jsonl(source / 'frames.jsonl'), read_jsonl(source / 'states.jsonl'), blocks)
    # Loading a selected checkpoint and test pixels occurs only after all contracts pass.
    import cv2
    import torch
    from PIL import Image
    from app.pin_insertion_cnn import PinInsertionPredictor
    from scripts.render_pin_insertion_heatmaps import infer
    torch.set_num_threads(4)
    predictor = PinInsertionPredictor(args.checkpoint, args.device)
    if predictor.metadata.get('model_variant') != 'heatmap':
        raise ValueError('actual spatial-logit heatmap model required')
    cache = CausalHoleCache(ranges)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    audit = []
    with tempfile.TemporaryDirectory(prefix='pin-block-video-') as temporary:
        temp = Path(temporary); names = []; durations = []
        for i, row in enumerate(stream):
            f, b = row['frame'], row['block']
            with Image.open(source / f['path']) as image:
                rgb = np.asarray(image.convert('RGB')).copy()
            if tuple(rgb.shape[1::-1]) != (f['width'], f['height']):
                raise ValueError('decoded source dimensions mismatch')
            meta = predictor.metadata.get('dataset_metadata', {})
            aspect = meta.get('image_width', f['width']) / meta.get('image_height', f['height'])
            if abs(rgb.shape[1] / rgb.shape[0] / aspect - 1) > .02:
                raise ValueError('full source aspect disagrees with checkpoint')
            prediction = infer(predictor, rgb, row['state14'])
            current_hole = np.asarray(prediction['keypoints']).reshape(2, 2)[1]
            remembered = cache.observe(b['block_id'], f['frame_index'], current_hole)
            text = f'{b["block_id"]} source frame {f["frame_index"]} | t={f["episode_time_s"]:.3f}s | actual state skew {row["state_skew_ms"]:+.2f}ms'
            canvas = compose(rgb, prediction, remembered, text, args.roi)
            name = temp / f'{i:06d}.jpg'
            if not cv2.imwrite(str(name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise ValueError('failed to write video frame')
            names.append(name)
            # Preserve measured deltas inside each block; omit excluded transition gaps.
            if i + 1 < len(stream) and stream[i + 1]['block']['block_id'] == b['block_id']:
                duration = (stream[i + 1]['frame']['monotonic_ns'] - f['monotonic_ns']) / 1e9
            else:
                previous = [r['frame']['monotonic_ns'] for r in stream[max(0, i - 1):i] if r['block']['block_id'] == b['block_id']]
                duration = (f['monotonic_ns'] - previous[0]) / 1e9 if previous else 1 / 15
            durations.append(duration)
            audit.append({'block_id':b['block_id'], 'source_frame_index':f['frame_index'], 'source_image_monotonic_ns':f['monotonic_ns'],
                          'source_state_monotonic_ns':row['state_monotonic_ns'], 'state_skew_ms':row['state_skew_ms'],
                          'duration_s':duration, 'current_keypoints_normalized':np.asarray(prediction['keypoints']).tolist(),
                          'cached_hole_normalized':None if remembered is None else remembered.tolist()})
        concat = temp / 'frames.txt'
        concat.write_text(''.join(f"file '{name}'\noption framerate 1000\nduration {duration:.9f}\n" for name, duration in zip(names, durations)) + f"file '{names[-1]}'\noption framerate 1000\n")
        subprocess.run([ffmpeg_executable(), '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat),
                        '-fps_mode', 'vfr', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    report = {'checkpoint_sha256':sha256(args.checkpoint), 'manifest_sha256':sha256(args.manifest), 'selection':selection,
              'source_recording_id':split['source_recording_id'], 'blocks':blocks, 'frames':audit,
              'within_recording_holdout':True, 'independent_recordings':1, 'same_grasp_likely':True,
              'test_labels_used':False, 'future_references_used':False, 'state_clock':'original host receipt monotonic_ns',
              'memory_protocol':protocol, 'memory_protocol_sha256':sha256(args.memory_protocol) if args.memory_protocol else None, 'autonomous_memory_eligible':False, 'memory_init_frame_ranges':ranges, 'roi':args.roi,
              'timing':'VFR original receipt deltas within blocks; excluded transition gap omitted; no frame sampling',
              'checkpoint_image_size':predictor.metadata.get('image_size', 96)}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--roi', nargs=4, type=int, default=[160, 160, 320, 240], metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--memory-init-frame-range', action='append', default=[], help='predeclared BLOCK:FIRST:LAST; default no memory')
    parser.add_argument('--memory-protocol', help='SHA-bound external clear-view/locked-grasp declaration JSON')
    render(parser.parse_args())

if __name__ == '__main__':
    main()
