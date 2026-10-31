#!/usr/bin/env python3
"""Offline real-demo -> MuJoCo experiment; never imports hardware control.

Produces paired trajectories and visual comparisons, not task-success scores.
The existing simulator's unregistered objects/bases/cameras are left explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from real2sim_data import load_episode
from real2sim_bridge import Real2SimBridge

ARM = np.array([*range(6), *range(7, 13)])


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max()))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_dynamic(bridge, episode, seconds, contacts):
    original_flags = bridge.model.opt.disableflags
    contact_flag = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    bridge.model.opt.disableflags = ((original_flags & ~contact_flag) if contacts
                                     else (original_flags | contact_flag))
    predicted, errors = [], []
    clamped_gripper_channels = 0
    rejection = None
    count = min(len(episode.action), int(seconds * episode.fps))
    try:
        bridge.reset_dynamic(episode.state[0])
        for action in episode.action[:count]:
            # Native exported targets stay exact; only dynamic grippers saturate.
            bounded = action.copy()
            bounded[[6, 13]] = np.clip(bounded[[6, 13]], 0, 2)
            clamped_gripper_channels += int(np.count_nonzero(bounded != action))
            violations = bridge.violations(bounded)
            if violations:
                rejection = violations
                break
            bridge.step_action(bounded, dt=1 / episode.fps)
            measured = bridge.native_state()
            if not np.isfinite(measured).all():
                raise RuntimeError('Nonfinite dynamic replay state')
            predicted.append(measured)
            errors.append(measured[ARM] - action[ARM])
    finally:
        bridge.model.opt.disableflags = original_flags
    error = np.asarray(errors, dtype=float).reshape(-1, 12)
    return np.asarray(predicted, dtype=float).reshape(-1, 14), dict(
        frames=len(predicted), requested_frames=count, seconds=len(predicted) / episode.fps,
        rejected_action=rejection, clamped_gripper_channels=clamped_gripper_channels,
        arm_abs_error_rad=stats(np.abs(error)),
        arm_rmse_rad=float(np.sqrt(np.mean(error ** 2))) if error.size else None,
        arm_per_joint_rmse_rad=np.sqrt(np.mean(error ** 2, axis=0)).reshape(2, 6).tolist() if error.size else None,
        arm_order=['left', 'right'], joint_order=[1, 2, 3, 4, 5, 6],
        interpretation='oracle next measured poses used as servo targets; pooled tracking error, not policy or task-success evaluation',
        contacts_enabled=contacts)


def comparison_indices(count):
    if count < 1:
        raise ValueError('Require at least one observation')
    return np.unique(np.linspace(0, count - 1, min(3, count), dtype=int))


def comparison(bridge, ep, indices, output):
    """Real overhead / sim overhead / real wrists / sim wrists. No wrist assignment claim."""
    import cv2
    renderer = mujoco.Renderer(bridge.model, height=240, width=320)
    width, height = 960, 530
    sheet = Image.new('RGB', (width, height * len(indices)), '#171c24')
    movie = cv2.VideoWriter(str(output / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                            5, (width, height))
    if not movie.isOpened():
        raise RuntimeError('Could not open MP4 writer')

    def frame_at(i):
        bridge.set_pose(ep.state[i])
        canvas = Image.new('RGB', (width, height), '#171c24')
        draw = ImageDraw.Draw(canvas)
        real_names = ['overhead', 'wrist_port2', 'wrist_port3']
        sim_names = ['overhead', 'left_wrist', 'right_wrist']
        for col, (real, sim) in enumerate(zip(real_names, sim_names)):
            row = ep.nearest_image(real, int(ep.grid_monotonic_ns[i]), max_skew_s=.04)
            if row:
                with Image.open(row['path']) as image:
                    canvas.paste(image.convert('RGB').resize((320, 240)), (col * 320, 25))
            draw.text((col * 320 + 5, 6), f'REAL {real}', fill='white')
            renderer.update_scene(bridge.data, camera=sim)
            canvas.paste(Image.fromarray(renderer.render()), (col * 320, 290))
            draw.text((col * 320 + 5, 270), f'SIM {sim} | t={ep.time_s[i]:.2f}s', fill='white')
        return canvas

    try:
        for k, index in enumerate(indices):
            sheet.paste(frame_at(int(index)), (0, k * height))
        # Kinematic playback at real time, sampled to 5fps to keep artifacts small.
        for i in range(0, len(ep.state), max(1, round(ep.fps / 5))):
            canvas = frame_at(i)
            movie.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        sheet.save(output / 'comparison.jpg', quality=90)
    finally:
        movie.release()
        renderer.close()


def run_episode(path, args):
    ep = load_episode(path)
    output = args.output / path.name
    output.mkdir()
    bridge = Real2SimBridge(args.scene)
    bridge.model.geom_pos[bridge.model.geom('table_top').id, 2] -= args.table_offset_m
    raw = {r['monotonic_ns']: r for r in
           (json.loads(line) for line in (path / 'states.jsonl').read_text().splitlines() if line.strip())}
    positions, rotations, pos_error, rot_error, contacts = [], [], [], [], []
    for state, ns in zip(ep.state, ep.state_monotonic_ns):
        bridge.set_pose(state)
        p, r = bridge.local_tcp()
        positions.append(p.copy()); rotations.append(r.copy())
        reference = raw[int(ns)]['arms']
        rp = np.array([reference[s]['tool_pose_base']['position_m'] for s in ('left', 'right')])
        rr = np.array([reference[s]['tool_pose_base']['rotation'] for s in ('left', 'right')])
        if rp.shape != (2, 3) or rr.shape != (2, 3, 3) or not np.isfinite(rp).all() or not np.isfinite(rr).all():
            raise ValueError(f'Invalid recorded FK at {ns}')
        pos_error.append(np.linalg.norm(p - rp, axis=-1))
        relative = np.einsum('aij,akj->aik', r, rr)
        rot_error.append(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        contacts.append(bridge.contact_summary())
    q = ep.state[:, ARM].reshape(-1, 2, 6)
    ranges = bridge.arm_ranges
    violation = np.maximum(ranges[..., 0] - q, q - ranges[..., 1]).clip(0)
    mapped_state = np.array([bridge.map_state(s) for s in ep.state])
    mapped_action = np.array([bridge.map_state(s) for s in ep.action])
    arrays = dict(native_state_rad=ep.state, native_action_rad=ep.action,
                  sim_state=mapped_state, sim_action=mapped_action,
                  grid_monotonic_ns=ep.grid_monotonic_ns,
                  measured_state_monotonic_ns=ep.state_monotonic_ns,
                  measured_action_monotonic_ns=ep.action_monotonic_ns,
                  state_skew_s=ep.state_skew_s, action_skew_s=ep.action_skew_s,
                  tcp_base_position_m=positions, tcp_base_rotation=rotations)
    report = dict(episode=path.name, source=str(path.resolve()),
                  frames=len(q), duration_s=len(q) / ep.fps,
                  table_offset_m=args.table_offset_m,
                  table_offset_interpretation='table-only registration ablation; positive offset lowers table; objects remain at synthetic poses',
                  state_alignment_s=stats(np.abs(ep.state_skew_s)),
                  action_alignment_s=stats(np.abs(ep.action_skew_s)),
                  fk_position_error_m=stats(pos_error), fk_rotation_error_rad=stats(rot_error),
                  arm_limit_violation_frame_fraction=float(np.any(violation > 1e-6, axis=(1, 2)).mean()),
                  arm_limit_violation_max_rad=float(violation.max()),
                  gripper_outside_assumed_range_fraction=float(np.mean((ep.state[:, [6, 13]] < 0) | (ep.state[:, [6, 13]] > 2))),
                  kinematic_contacts=contacts,
                  source_sha256={n: digest(path / n) for n in ('metadata.json', 'states.jsonl', 'frames.jsonl')})
    for enabled, name in ((False, 'no_contact'), (True, 'scene_contact')):
        if args.dynamic_seconds:
            trajectory, metrics = replay_dynamic(bridge, ep, args.dynamic_seconds, enabled)
            arrays['dynamic_' + name + '_native'] = trajectory
            arrays['dynamic_' + name + '_grid_monotonic_ns'] = (ep.grid_monotonic_ns[:len(trajectory)] + round(1e9 / ep.fps))
            report['dynamic_' + name] = metrics
    np.savez_compressed(output / 'mapped.npz', **arrays)
    if args.render:
        comparison(bridge, ep, comparison_indices(len(q)), output)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'source_metadata.json').write_text(json.dumps(ep.metadata, indent=2) + '\n')
    brief = {k: v for k, v in report.items() if k not in ('kinematic_contacts', 'source_sha256')}
    print(json.dumps(brief), flush=True)
    return brief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/kept'))
    parser.add_argument('--episode', action='append', required=True)
    parser.add_argument('--scene', type=Path, default=Path('/home/robotics/Desktop/RobotArmTraining/sim/actuator/scene_bimanual.xml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dynamic-seconds', type=float, default=5.)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--table-offset-m', type=float, default=0., help='Lower table only by this many metres; objects remain synthetic')
    args = parser.parse_args()
    if not np.isfinite(args.dynamic_seconds) or args.dynamic_seconds < 0:
        parser.error('--dynamic-seconds must be finite and nonnegative')
    if not np.isfinite(args.table_offset_m):
        parser.error('--table-offset-m must be finite')
    args.output.mkdir(parents=True, exist_ok=False)
    reports = [run_episode(args.source / episode, args) for episode in args.episode]
    summary = dict(table_offset_m=args.table_offset_m,
                   table_offset_interpretation='table-only registration ablation; objects remain at synthetic poses',
                   schema_version=1, episodes=reports, scene=str(args.scene.resolve()),
                   scene_sha256=digest(args.scene), mujoco_version=mujoco.__version__, numpy_version=np.__version__,
                   tool_source_sha256={name: digest(Path(__file__).parent / name) for name in
                                       ('experiment_real2sim.py', 'real2sim_bridge.py', 'real2sim_data.py')},
                   model_evaluated=False, task_success_evaluated=False,
                   action_contract='20Hz next measured pose; joints radians; simulated grippers per-finger metres',
                   limitations=['Synthetic base placement and cameras are unregistered.',
                                'Gripper uses assumed linear 0..2 radians -> 0..0.04m per finger.',
                                'Object poses stay at synthetic defaults in kinematic replay; no tracked real object poses.',
                                'Physical replay uses next measured pose proxies, not demonstrated actuator commands.',
                                'Real wrist USB-to-arm assignment is unverified.',
                                'Full raw episodes, no manual task trimming or task-success labels.'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
