#!/usr/bin/env python3
"""Offline early-CNN hole retention audit. No labeled coordinates seed memory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def evaluate_forward_memory(predictor,manifest_path,output):
    import cv2
    from app.pin_insertion_memory import PinInsertionHoleMemory
    from scripts.render_pin_insertion_heatmaps import infer,assert_render_disjoint
    from scripts.evaluate_pin_insertion_cnn import eligible_samples
    manifest=json.loads(Path(manifest_path).read_text())
    selected=sorted({r['episode_id'] for r in manifest['frames'] if r['split']=='test' and r.get('pin_tip_px') is not None})
    assert_render_disjoint(selected,predictor.metadata)
    results=[]
    for eid in selected:
        labels={r['video_frame']:r for r in manifest['frames'] if r['episode_id']==eid}
        ep=ROOT/'data/pin-alignment-forward'/eid;meta,samples,_=eligible_samples(ep,stride=1,max_skew_ms=100)
        cap=cv2.VideoCapture(str(ep/meta['videos']['right_wrist']))
        memory=PinInsertionHoleMemory(min_seed_frames=3,max_memory_age_ns=60_000_000_000,max_frame_gap_ns=None)
        records=[];errors=[]
        try:
            for row,skew in samples:
                index=row['video_frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
                if not ok:raise ValueError('Cannot decode forward frame')
                out=infer(predictor,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),row['state']);initializing=index<3
                result=memory.observe(out['keypoints'].tolist(),observation_ns=row['monotonic_ns'],now_ns=row['monotonic_ns'],
                   episode_id=eid,grasp_id='fixed_initial_carrier_grasp',camera_id='right_wrist',sequence_id=eid,
                   image_size=(640,480),grasp_locked=True,hole_visible=initializing,hole_trusted=initializing,initialize=initializing)
                label=labels.get(index)
                if not initializing and label and label['valid'][1] and label.get('target_hole_px') is not None:
                    current=np.asarray(out['keypoints']).reshape(2,2)[1]*[640,480];truth=np.asarray(label['target_hole_px'])
                    retained=np.asarray(result['keypoints'][2:])*[640,480] if result['memory_available'] else None
                    errors.append({'episode_id':eid,'video_frame':index,'single_frame_hole_error_px':float(np.linalg.norm(current-truth)),
                                   'retained_hole_error_px':float(np.linalg.norm(retained-truth)) if retained is not None else None})
                records.append({'video_frame':index,'initialization':initializing,'memory':result})
        finally:cap.release()
        results.append({'episode_id':eid,'evaluation':errors,'records':records})
    all_errors=[r for e in results for r in e['evaluation']]
    report={'checkpoint_metadata':predictor.metadata,'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
       'initialization':'Fixed first three source frames0,1,2; CNN-only coordinates. Manual clear-window/stationary carrier assumption, not automated visibility.',
       'test_episodes':selected,'post_initialization_label_count':len(all_errors),'episodes':results,
       'limitations':['Only independent visible labels after initial three frames evaluated; hidden final endpoints have no invented ground truth.',
                      'No coordinate labels seed memory. No test-based choice of seed window, checkpoint, or cache thresholds.',
                      'Static carrier in right-wrist frame is an explicit prior. No contact inference or metric correction.']}
    for key in ('single_frame_hole_error_px','retained_hole_error_px'):
        values=[r[key] for r in all_errors if r[key] is not None];report[key+'_mean']=float(np.mean(values)) if values else None
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','checkpoint_metadata')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--labels',type=Path)
    p.add_argument('--episode',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forward-manifest',type=Path,help='Evaluate fixed first-three-frame memory on four heldout pin episodes; output JSON')
    p.add_argument('--init-start',type=int,default=330);p.add_argument('--init-end',type=int,default=350)
    p.add_argument('--end-frame',type=int,default=388);p.add_argument('--fps',type=float,default=5)
    args=p.parse_args()
    import cv2,torch,imageio_ffmpeg
    from app.pin_insertion_cnn import PinInsertionPredictor
    from app.pin_insertion_memory import PinInsertionHoleMemory
    from scripts.render_pin_insertion_heatmaps import infer,compose_frame,spatial_probability,assert_render_disjoint
    torch.set_num_threads(4)
    predictor=PinInsertionPredictor(args.checkpoint)
    if args.forward_manifest:
        evaluate_forward_memory(predictor,args.forward_manifest,args.output);return
    if args.episode is None or args.labels is None:raise ValueError('Challenge mode requires --episode and --labels')
    meta=json.loads((args.episode/'metadata.json').read_text());eid=meta['id'];assert_render_disjoint([eid],predictor.metadata)
    frame_rows=[json.loads(l) for l in (args.episode/'frames.jsonl').read_text().splitlines()]
    frames=[r for r in frame_rows if r['camera']=='wrist_port2' and args.init_start<=r['frame_index']<=args.end_frame]
    frames.sort(key=lambda r:r['frame_index'])
    states=[json.loads(l) for l in (args.episode/'states.jsonl').read_text().splitlines()]
    # The raw wall gaps are declared, never silently replaced with compressed data time.
    timestamps=np.array([r['monotonic_ns'] for r in frames],dtype=np.int64)
    largest_gap=int(np.diff(timestamps).max(initial=0));horizon=int(timestamps[-1]-timestamps[0])+int(1e9)
    memory=PinInsertionHoleMemory(min_seed_frames=3,max_memory_age_ns=horizon,max_frame_gap_ns=largest_gap+1,
                                 max_input_age_ns=250_000_000,max_disagreement_px=32.)
    labels_raw=json.loads(args.labels.read_text());labels={r['frame_index']:r for r in labels_raw['frames'] if r['episode_id']==eid}
    args.output.parent.mkdir(parents=True,exist_ok=True);raw=args.output.with_suffix('.raw.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(1440,506))
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    seed_maps=[];records=[];errors=[]
    try:
        for frame in frames:
            index=frame['frame_index'];timestamp=frame['monotonic_ns']
            state=min(states,key=lambda s:abs(s['monotonic_ns']-timestamp));skew=(state['monotonic_ns']-timestamp)/1e6
            if abs(skew)>40:raise ValueError('State/image receipt skew too large')
            q=[]
            for arm in ('left','right'):q.extend(state['arms'][arm]['position_rad']);q.append(state['arms'][arm]['gripper']['position'])
            image=cv2.imread(str(args.episode/frame['path']))
            if image is None:raise ValueError('Missing raw frame')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);out=infer(predictor,rgb,q)
            initializing=index<=args.init_end
            result=memory.observe(out['keypoints'].tolist(),observation_ns=timestamp,now_ns=timestamp,
                episode_id=eid,grasp_id='visually_reviewed_same_right_carrier_grasp',camera_id='wrist_port2',
                sequence_id='offline_declared_continuity_across_source_holds',image_size=(640,480),grasp_locked=True,
                hole_visible=initializing,hole_trusted=initializing,initialize=initializing)
            if initializing:seed_maps.append(spatial_probability(out['keypoint_logits'])[1])
            current=np.asarray(out['keypoints']).reshape(2,2)
            available=bool(result.get('memory_available'))
            held=np.asarray(result['keypoints'] if available else out['keypoints'],dtype=float).reshape(2,2)
            render={**out,'keypoints':np.r_[current[0],held[1]]}
            if seed_maps:
                logits=np.asarray(out['keypoint_logits']).copy();logits[1]=np.log(np.mean(seed_maps,axis=0).clip(1e-30));render['keypoint_logits']=logits
            gt=None
            if index in labels:
                label=labels[index];gt=np.asarray(label['keypoints_px'],dtype=float).reshape(2,2)/[640,480]
                if label['valid'][1]:
                    truth=np.asarray(label['target_hole_px']);errors.append({'frame_index':index,
                       'single_frame_hole_error_px':float(np.linalg.norm(current[1]*[640,480]-truth)),
                       'retained_hole_error_px':float(np.linalg.norm(held[1]*[640,480]-truth)) if available else None})
            status='INITIALIZING early CNN; visible-window selection, no coordinate seed' if initializing else 'RETAINED early CNN hole; stationary carrier/wrist prior, not current detection'
            subtitle=f'frame {index} source t={(timestamp-meta["start_monotonic_ns"])/1e9:.3f}s | '+status
            canvas=compose_frame(rgb,render,'REAL CHALLENGE | current pin + cached early hole; source holds removed',subtitle,gt,xyz_disabled=True)
            cv2.rectangle(canvas,(960,52),(1439,83),(0,0,0),-1)
            cv2.putText(canvas,'CACHED INITIAL CNN HOLE heatmap',(970,75),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            writer.write(canvas)
            records.append({'frame_index':index,'monotonic_ns':timestamp,'data_time_s':frame.get('data_time_s'),
                'recording_segment':frame.get('recording_segment'),'initialization':initializing,
                'state_record_receipt_skew_ms':skew,'single_frame_keypoints':out['keypoints'].tolist(),'memory':result})
    finally:writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],check=True);raw.unlink()
    summary={'output':str(args.output.resolve()),'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      'episode_id':eid,'initialization_frames':[args.init_start,args.init_end],'initialization_source':'Actual early CNN predictions only; manually reviewed visible pre-occlusion window is a prototype assumption, not an automated visibility classifier. No GT coordinate seeds.',
      'labels_sha256':hashlib.sha256(args.labels.read_bytes()).hexdigest(),
      'frames':len(records),'largest_raw_frame_gap_s':largest_gap/1e9,'raw_sequence_span_s':(timestamps[-1]-timestamps[0])/1e9,
      'memory_gap_override_s':(largest_gap+1)/1e9,'memory_horizon_override_s':horizon/1e9,'fps':args.fps,
      'evaluation':errors,'records':records,'limitations':['Offline declared same carrier grasp/right wrist frame; no geometric warp or optical-flow tracking.',
         'Raw timestamp gap overrides are explicit offline review hypotheses, not live-controller defaults.',
         'Cached yellow map is mean early CNN spatial softmax; retained marker follows memory API consensus readout, so their maxima need not coincide.',
         'Contact likely, force unconfirmed; gate unknown, XYZ disabled. Only later independent labels evaluate memory.',
         'One failure episode with three correlated label frames; no generalization or insertion-success estimate.']}
    for key in ('single_frame_hole_error_px','retained_hole_error_px'):
        values=[r[key] for r in errors if r[key] is not None];summary[key+'_mean']=float(np.mean(values)) if values else None
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))

if __name__=='__main__':main()
