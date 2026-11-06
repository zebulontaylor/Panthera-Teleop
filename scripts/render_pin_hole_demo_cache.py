#!/usr/bin/env python3
"""Render recorded causal demo-cache predictions; never rerun or seed from labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_marker(image,point,color,label=None,radius=7):
    import cv2
    if point is None:return
    p=np.asarray(point,dtype=float)
    if p.shape!=(2,) or not np.isfinite(p).all():return
    xy=tuple(np.rint(p).astype(int));cv2.circle(image,xy,radius,color,2,cv2.LINE_AA)
    if label:cv2.putText(image,label,(xy[0]+9,xy[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)


def display_repetitions(records,fps,playback_speed=1.):
    """Cumulative clock rounding avoids independent per-frame duration drift."""
    result=[];context=None;elapsed=0.;emitted=0;previous=None
    for i,row in enumerate(records):
        if row['context_id']!=context:
            context=row['context_id'];elapsed=0.;emitted=0;previous=None
        stamp=int(row['timestamp_ns'])
        if previous is not None and stamp<=previous:raise ValueError('Renderer source clock regressed')
        if i+1<len(records) and records[i+1]['context_id']==context:
            duration=(records[i+1]['timestamp_ns']-stamp)/1e9
            if duration<=0:raise ValueError('Renderer source clock regressed')
        else:duration=1/fps
        elapsed+=duration/playback_speed
        count=max(1,round(elapsed*fps)-emitted)
        emitted+=count;result.append(count);previous=stamp
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=float,default=30);p.add_argument('--playback-speed',type=float,default=1.)
    p.add_argument('--roi',type=int,nargs=4,default=[180,230,440,460],help='Fixed native x0 y0 x1 y1, no GT centering')
    p.add_argument('--max-frames',type=int);args=p.parse_args()
    if not 0<args.fps<=60 or not 0<args.playback_speed<=4:raise ValueError('Invalid display timing')
    report=json.loads(args.report.read_text());records=report['records']
    if report.get('gt_or_future_reference_used_by_prediction') is not False:raise ValueError('Report does not certify label-free prediction path')
    if sha(report['references_path'])!=report['references_sha256']:raise ValueError('Reference bank changed')
    replay=report.get('kind')=='continuous_demonstration_replay'
    if replay:
        protocol=report.get('replay_protocol',{})
        if report.get('post_test_engineering_revision') is not True or report.get('independent_unseen_test') is not False or protocol.get('stationary_carrier_to_right_wrist') is not True:
            raise ValueError('Replay requires explicit engineering-revision/fixed-grasp disclosure')
    if report['evaluation_split']=='test' and not replay:
        selection=report.get('selection')
        if not selection or selection.get('selection_split')!='validation' or selection.get('test_used_for_selection') is not False:raise ValueError('No frozen VAL-only method selection in test report')
    bank=json.loads(Path(report['references_path']).read_text());references={r['id']:r for r in bank['references']}
    import cv2,imageio_ffmpeg
    cv2.setNumThreads(4)
    purple=(255,70,220);cyan=(255,255,0);yellow=(0,220,255);gray=(160,160,160)
    captures={};reference_images={};rendered=[];x0,y0,x1,y1=args.roi
    if not 0<=x0<x1<=640 or not 0<=y0<y1<=480:raise ValueError('ROI outside native640x480')
    args.output.parent.mkdir(parents=True,exist_ok=True);raw=args.output.with_suffix('.raw.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(1280,720))
    if not writer.isOpened():raise RuntimeError('Cannot open video writer')
    repetition_counts=display_repetitions(records,args.fps,args.playback_speed)
    try:
        for index,row in enumerate(records):
            if args.max_frames is not None and len(rendered)>=args.max_frames:break
            if row.get('image_path'):
                if row.get('image_sha256') and sha(row['image_path'])!=row['image_sha256']:raise ValueError('Recorded evaluation image changed')
                image=cv2.imread(row['image_path'])
            else:
                path=row['video_path']
                if path not in captures:captures[path]=cv2.VideoCapture(path)
                cap=captures[path];cap.set(cv2.CAP_PROP_POS_FRAMES,row['video_frame']);ok,image=cap.read()
                if not ok:image=None
            if image is None:raise ValueError('Cannot decode source RGB')
            display=image.copy();result=row['demo_cache']
            polygon=result.get('carrier_polygon_px')
            if polygon is not None:
                poly=np.asarray(polygon,dtype=float)
                if poly.ndim==2 and poly.shape[1]==2 and np.isfinite(poly).all():cv2.polylines(display,[np.rint(poly).astype(np.int32)],True,purple,1,cv2.LINE_AA)
            draw_marker(display,row.get('current_cnn_pin_px'),cyan)
            draw_marker(display,row.get('current_cnn_hole_px'),yellow)
            draw_marker(display,row.get('first5_cnn_cache_hole_px'),gray,radius=11)
            draw_marker(display,result.get('hole_px'),purple,radius=8)
            canvas=np.zeros((720,1280,3),dtype=np.uint8);canvas[80:560,:640]=display
            # Fixed ROI is selected once, independent of labeled/predicted hole.
            crop=display[y0:y1,x0:x1];scale=min(600/crop.shape[1],440/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
            canvas[85:85+crop.shape[0],660:660+crop.shape[1]]=crop
            def text(value,y,color=(240,240,240),scale=.57,x=10):cv2.putText(canvas,str(value),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
            title='VAL B2 | TRAIN-demo feature cache' if report['evaluation_split']=='validation' else 'REUSED TEST | TRAIN-demo feature cache; within-recording blocks' if report['domain']=='blocks' else 'REUSED TEST | separate recording; TRAIN-demo feature cache'
            if replay:title='CONTINUOUS REPLAY | fixed-grasp memory; post-test engineering revision'
            text(title,26,scale=.76)
            text(f'{row["context_id"]} frame {row["frame_index"]} source t={row.get("source_time_s",0):.3f}s | {result.get("mode")} / {result.get("method")}',55,scale=.6)
            text('Purple: demo-cache hole',590,purple);text('Cyan: current CNN pin',614,cyan)
            text('Yellow: current CNN hole | gray: first-five CNN cache',638,yellow)
            age=result.get('age_ns');age_text=f'{age/1e9:.2f}s' if isinstance(age,(int,float)) else 'n/a'
            quality=f'inliers={result.get("inliers",0)} matches={result.get("matches",0)} area={result.get("coverage",0):.3f} age={age_text}'
            text(quality,665,scale=.53)
            uncertainty=result.get('uncertainty_px');uncertainty_text=f'{uncertainty:.1f}px heuristic' if isinstance(uncertainty,(int,float)) else 'unknown'
            if replay:
                footer=f'Last visual support age {age_text} | '+('STATIONARY HOLD, not current detection' if result.get('mode')=='held' else 'visual support; no metric/contact certainty')
                text(footer,690,purple if result.get('mode')=='held' else (240,240,240),scale=.5)
            else:text(f'uncertainty: {uncertainty_text} | no contact, depth or success inference',690,scale=.5)
            reference_id=result.get('reference_id')
            if reference_id in references:
                ref=references[reference_id]
                if reference_id not in reference_images:
                    if sha(ref['image_path'])!=ref['image_sha256']:raise ValueError('TRAIN reference image changed')
                    thumb=cv2.imread(ref['image_path']);draw_marker(thumb,ref['hole_px'],purple,radius=8);reference_images[reference_id]=cv2.resize(thumb,(192,144))
                canvas[565:709,1078:1270]=reference_images[reference_id]
                text('TRAIN ref',553,purple,scale=.5,x=1080)
            if result.get('mode')=='held':text('HELD stationary prior, not a current detection',74,purple,scale=.5,x=660)
            if result.get('mode')=='lost':text('LOST: no output; no substitute ground truth',74,purple,scale=.5,x=660)
            repeats=repetition_counts[index]
            for _ in range(repeats):writer.write(canvas)
            rendered.append({'context_id':row['context_id'],'frame_index':row['frame_index'],'source_timestamp_ns':row['timestamp_ns'],'display_repeats':repeats})
    finally:
        writer.release()
        for cap in captures.values():cap.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],check=True);raw.unlink()
    metadata={'evaluation_protocol':report.get('evaluation_protocol','independent_context_start_benchmark'),'post_test_engineering_revision':replay,
       'processed_inference_frames':report.get('processed_source_frames',len(records)),'output':str(args.output.resolve()),'report_sha256':sha(args.report),'renderer_sha256':sha(__file__),
       'source_frames':len(rendered),'display_frames':sum(r['display_repeats'] for r in rendered),'fps':args.fps,
       'duration_s':sum(r['display_repeats'] for r in rendered)/args.fps,'fixed_roi_xyxy':args.roi,'playback_speed':args.playback_speed,
       'rendered_frames':rendered,'annotations_used_by_renderer':'TRAIN reference thumbnail only. No evaluation hole coordinates are read/rendered.',
       'timing':('Rendered B4/B5segments use raw native-frame clocks; unrendered gaps removed. Inference consumed all earlier frames continuously, WITHOUT pose-block resets.' if replay else 'Full chronological native frames; cumulative-clock frame repetition approximates raw host-receipt intervals within each block. Inter-context gaps removed and contexts independently reset.'),
       'limitations':['Purpletracked means feature registration/opticalflow estimate; purpleheld is causal stationary-prior memory, not current visibility.',
          'Reference thumbnail is TRAIN supervision only. Reference exclusion regions and actual match quality are documented in report.',
          'Reused benchmark and within-recording limitations remain; no contact/depth/hardware success claim.']}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({k:v for k,v in metadata.items() if k!='rendered_frames'},indent=2))

if __name__=='__main__':main()
