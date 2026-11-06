#!/usr/bin/env python3
"""Offline CNN diagnostic of saved physical rollout images and full states.

Never runs controls or modifies physics. Images are those captured by the
physical generator, rather than kinematic registrations of real carrier poses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render(args):
    import cv2
    import torch
    from app.pin_insertion_cnn import PinInsertionPredictor
    from scripts.render_pin_insertion_heatmaps import infer, compose_frame
    torch.set_num_threads(4)
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data['metadata_json']))
    if not metadata.get('physics_simulated') or not all(key in data for key in ('qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart')):
        raise ValueError('Require physical generator full-state archive')
    predictor = PinInsertionPredictor(args.checkpoint, device=args.device)
    if predictor.metadata.get('model_variant') != 'heatmap':
        raise ValueError('Actual spatial-logit heatmap checkpoint required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix('.mp4v.mp4')
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (1440, 506))
    if not writer.isOpened():
        raise RuntimeError('Cannot create video')
    audit = metadata['rollout_audits'][0]
    provenance = metadata.get('bridge_provenance', {})
    registered = provenance.get('registered_setup')
    setup_label = ('physical reset grasp + registered camera/planar base' if registered
                   else 'nominal physical grasp/base + registered wrist camera')
    steps = {int(row['step']): row for row in audit['steps']}
    rows = []
    try:
        for i, rgb in enumerate(data['images']):
            step = int(data['step'][i])
            physical = steps.get(step, {})
            contacts = physical.get('pin_carrier_gear_contacts_after_step', [])
            current_force = sum(float(c['normal_force_n']) for c in contacts)
            peak = float(physical.get('info', {}).get('pin_contact_load_n', 0.))
            out = infer(predictor, rgb, data['state'][i])
            truth = data['keypoints'][i].reshape(2, 2).astype(float)
            valid = data['keypoints_valid'][i].reshape(2, 2).all(axis=1)
            truth[~valid] = np.nan
            subtitle = ('step %d | sim %.3fs | %s | force %.3fN peak %.3fN | aperture %.2f' %
                        (step, data['sim_time_s'][i], data['regime'][i], current_force, peak,
                         float(data['hole_aperture_visible_fraction'][i, 0])))
            canvas = compose_frame(rgb, out,
                'PHYSICAL SIM | ' + setup_label,
                subtitle, truth, xyz_disabled=True)
            canvas[480:] = 0
            cv2.putText(canvas, 'Green: visible simulator geometry; crosses/heatmaps: CNN. Offline diagnostic; no CNN control.',
                        (12, 495), cv2.FONT_HERSHEY_SIMPLEX, .59, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(canvas)
            if i in {0, len(data['images']) // 2, len(data['images']) - 1}:
                cv2.imwrite(str(args.output.with_name(args.output.stem + '-frame-%03d.png' % i)), canvas)
            prediction = np.asarray(out['keypoints']).reshape(2, 2)
            error = np.linalg.norm((prediction - data['keypoints'][i].reshape(2, 2)) * [rgb.shape[1], rgb.shape[0]], axis=1)
            rows.append({'step': step, 'sim_time_s': float(data['sim_time_s'][i]), 'regime': str(data['regime'][i]),
                'pin_carrier_contact_normal_force_sum_n': current_force, 'control_interval_peak_pin_external_load_n': peak,
                'prediction_px': (prediction * [rgb.shape[1], rgb.shape[0]]).tolist(),
                'geometric_projection_px': (data['keypoints'][i].reshape(2, 2) * [rgb.shape[1], rgb.shape[0]]).tolist(),
                'projection_position_valid': valid.tolist(), 'direct_visible': data['direct_visible'][i].tolist(),
                'error_to_geometry_px': error.tolist(), 'state14': data['state'][i].tolist()})
    finally:
        writer.release()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = None
    if ffmpeg:
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(args.output)], check=True)
        raw.unlink()
    else:
        raw.replace(args.output)
    report = {
        'dataset': str(args.dataset.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'camera_registration': metadata.get('camera_registration'),
        'semantics': setup_label + '; original contact grasps retained. No real carrier/grasp kinematic registration. No model controls, gate validity false, XYZ disabled.',
        'registered_setup': registered,
        'source_and_asset_provenance': provenance,
        'geometric_bore_radius_m': metadata.get('individual_bore_radius_m'),
        'geometry_erratum': 'Archives with radius0.00417m predate measured0.00412m radius correction. Regime/aperture proxy labels only; collision physics unchanged.',
        'full_state_capture': metadata['full_state_capture'], 'render_origin': 'Exact RGB snapshots captured during physical integration; no qpos replay or teleportation.',
        'state14_peak_to_peak_rad': np.ptp(data['state'], axis=0).tolist(),
        'qpos_peak_to_peak': np.ptp(data['qpos'], axis=0).tolist(),
        'contact_events': audit['contact_events'], 'regime_counts': metadata['category_counts'],
        'final_step_info': audit['steps'][-1]['info'], 'snapshots': rows,
    }
    for landmark, name in enumerate(('pin', 'hole')):
        errors = [r['error_to_geometry_px'][landmark] for r in rows if r['projection_position_valid'][landmark]]
        report[name + '_position_valid_error_px'] = {'count': len(errors), 'mean': float(np.mean(errors)) if errors else None,
            'median': float(np.median(errors)) if errors else None}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=5.)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    report = render(args)
    print(json.dumps({key: value for key, value in report.items() if key not in ('snapshots', 'camera_registration', 'final_step_info')}, indent=2))


if __name__ == '__main__':
    main()
