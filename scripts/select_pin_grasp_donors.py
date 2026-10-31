#!/usr/bin/env python3
"""Select auditable, heuristic pin pickup donor frames; never changes sources."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

CAMERAS = ('overhead', 'wrist_port2', 'wrist_port3')
THRESHOLDS = dict(open_min_rad=.5, closed_max_rad=.25, minimum_drop_rad=.3,
                  opening_lookback_s=1.5, approach_before_close_s=1., after_close_s=.2,
                  close_x_m=[.27, .34], close_y_m=[-.31, -.23], close_z_m=[-.01, .045],
                  window_x_m=[.26, .35], window_y_m=[-.32, -.22], window_z_m=[-.01, .065],
                  minimum_candidate_frames=5)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region(p, prefix):
    return all(THRESHOLDS[f'{prefix}_{a}_m'][0] <= v <= THRESHOLDS[f'{prefix}_{a}_m'][1]
               for a, v in zip('xyz', p))


def select(dataset, raw_root):
    manifest_path = dataset/'processing/manifest.json'
    provenance = json.loads(manifest_path.read_text())
    result = dict(dataset_root=str(dataset), raw_root=str(raw_root),
        selection_method='Unverified heuristic, not a human stage annotation or proof of grasp. Select the FIRST closing transition in the low loose-pin FK region after substantial recent opening. Window begins no earlier than its recent maximum opening and 1s before <=0.25rad closure; ends 0.2s after closure. Retain only low pickup-region frames. All times/indexes and camera receipts come from existing processed alignment parquet.',
        thresholds=THRESHOLDS, source_manifest_sha256=sha(manifest_path),
        camera_mapping={'left_wrist': 'wrist_port3', 'right_wrist': 'wrist_port2', 'overhead': 'overhead'},
        candidates=[], windows=[], excluded=[])
    for episode in provenance['episodes']:
        eid, index = episode['id'], episode['episode_index']
        source = raw_root/eid
        try:
            states_path = source/'states.jsonl'
            states = [json.loads(line) for line in states_path.open()]
            times = [s['monotonic_ns'] for s in states]
            grips = [s['arms']['left']['gripper']['position'] for s in states]
            metadata = json.loads((source/'metadata.json').read_text())
            if metadata.get('episode_mode') != 'single_bolt':
                raise ValueError('Source mode is not single_bolt')
            origin = metadata['start_monotonic_ns']
            transition = None
            for i, s in enumerate(states):
                pos = s['arms']['left']['tool_pose_base']['position_m']
                if grips[i] > THRESHOLDS['closed_max_rad'] or not region(pos, 'close'):
                    continue
                a = bisect.bisect_left(times, times[i]-int(1.5e9))
                peak = max(range(a, i+1), key=lambda k: grips[k])
                if grips[peak] >= .5 and grips[peak]-grips[i] >= .3:
                    transition = (i, peak)
                    break
            if transition is None:
                raise ValueError('No first substantial open-to-close transition in low pickup region')
            close, peak = transition
            start_ns, end_ns = max(times[peak], times[close]-int(1e9)), times[close]+int(.2e9)
            alignment_path = dataset/f'processing/alignment/episode_{index:06d}.parquet'
            receipts = pq.read_table(alignment_path).to_pylist()
            candidates = []
            for row in receipts:
                stamp = row['state_monotonic_ns']
                if not start_ns <= stamp <= end_ns:
                    continue
                nearest = min((max(0, bisect.bisect_left(times, stamp)-1),
                               min(len(times)-1, bisect.bisect_left(times, stamp))), key=lambda k: abs(times[k]-stamp))
                arm = states[nearest]['arms']['left']
                position = arm['tool_pose_base']['position_m']
                if not region(position, 'window'):
                    continue
                candidates.append(dict(episode_index=index, frame_index=row['frame_index'],
                    dataset_index=row['index'], stage='pin_grasp_heuristic', source_id=eid,
                    source_time_s=row['source_episode_time_s'],
                    state_monotonic_ns=stamp, position_m=position,
                    gripper_rad=arm['gripper']['position'],
                    cameras={c: dict(source_path=row[c+'.source_path'],
                                    sha256=row[c+'.source_sha256'],
                                    offset_s=row[c+'.offset_s']) for c in CAMERAS}))
            if len(candidates) < 5:
                raise ValueError('Fewer than five conservative pickup-region frames')
            result['candidates'].extend(candidates)
            result['windows'].append(dict(episode_index=index, source_id=eid,
                first_frame_index=candidates[0]['frame_index'], last_frame_index=candidates[-1]['frame_index'],
                start_s=(start_ns-origin)/1e9, end_s=(end_ns-origin)/1e9,
                closing_threshold_time_s=(times[close]-origin)/1e9,
                peak_open_rad=grips[peak], closing_threshold_rad=grips[close],
                candidate_count=len(candidates), states_sha256=sha(states_path),
                alignment_sha256=sha(alignment_path)))
        except (ValueError, OSError, KeyError) as exc:
            result['excluded'].append(dict(episode_index=index, source_id=eid, reason=str(exc)))
    return result


def preview(result, output):
    windows = result['windows']
    selected = [windows[round(i*(len(windows)-1)/min(8,len(windows)-1))] for i in range(min(9,len(windows)))] if len(windows)>1 else windows
    sheet = Image.new('RGB', (3*256, len(selected)*286), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, window in enumerate(selected):
        candidates = [c for c in result['candidates'] if c['episode_index'] == window['episode_index']]
        # Show a late-approach frame just before threshold crossing, not transfer.
        candidate = min(candidates, key=lambda c: abs(c['source_time_s']-(window['closing_threshold_time_s']-.15)))
        for j, camera in enumerate(CAMERAS):
            receipt = candidate['cameras'][camera]
            source = Path(result['raw_root'])/candidate['source_id']/receipt['source_path']
            if sha(source) != receipt['sha256']:
                raise ValueError('Source camera hash mismatch: '+str(source))
            crop = (120,80,520,480) if camera == 'overhead' else (80,0,560,480)
            with Image.open(source) as im:
                tile = im.convert('RGB').crop(crop).resize((256,256),Image.Resampling.LANCZOS)
            sheet.paste(tile,(j*256,i*286+30))
            draw.text((j*256+3,i*286+17),camera,fill='black')
        draw.text((3,i*286+2),f"ep{candidate['episode_index']} {candidate['source_id']} t={candidate['source_time_s']:.2f}s grip={candidate['gripper_rad']:.2f} heuristic",fill='black')
    sheet.save(output)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=root/'data/processed/panthera-single-bolt-insertion-20261029-20hz')
    parser.add_argument('--raw-root',type=Path,default=root/'data/kept')
    parser.add_argument('--output',type=Path,default=root/'logs/pin-regrasp-donor-manifest.json')
    parser.add_argument('--preview',type=Path,default=root/'logs/pin-grasp-donor-preview.png')
    args = parser.parse_args()
    result = select(args.dataset.resolve(), args.raw_root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    preview(result,args.preview)
    print(json.dumps(dict(windows=len(result['windows']),candidates=len(result['candidates']),excluded=len(result['excluded']),manifest=str(args.output),preview=str(args.preview))))

if __name__ == '__main__':
    main()
