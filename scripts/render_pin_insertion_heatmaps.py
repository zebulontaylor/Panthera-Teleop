#!/usr/bin/env python3
"""Render actual model spatial likelihoods over offline synthetic/real test images."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def spatial_probability(logits):
    values=np.asarray(logits,dtype=np.float64)
    if values.ndim!=3 or values.shape[0]!=2 or not np.isfinite(values).all():
        raise ValueError('Require finite two-channel spatial logits')
    values=np.exp(values-values.max(axis=(1,2),keepdims=True))
    return values/values.sum(axis=(1,2),keepdims=True)


def load_manifest(path):
    text=Path(path).read_text()
    try:
        data=json.loads(text); rows=data.get('frames',data.get('samples',[])) if isinstance(data,dict) else data
    except json.JSONDecodeError:
        rows=[json.loads(line) for line in text.splitlines() if line.strip()]
    selected=[r for r in rows if r.get('split') in ('test','heldout','challenge_test')]
    if not selected: raise ValueError('Manifest must contain explicit test/heldout frames')
    for r in selected:
        if not r.get('image_path',r.get('image')): raise ValueError('Frame image_path required')
        q=np.asarray(r.get('state14',r.get('state')),dtype=float)
        if q.shape!=(14,) or not np.isfinite(q).all(): raise ValueError('Frame needs finite native state14')
    return selected


def infer(predictor,rgb,state):
    import torch
    import torch.nn.functional as F
    from app.pin_insertion_cnn import joint_features
    tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None].to(predictor.device)/255
    size=predictor.metadata.get('image_size',96)
    tensor=F.interpolate(tensor,size=(size,size),mode='bilinear',align_corners=False)
    joints=torch.from_numpy(joint_features(state).copy())[None].to(predictor.device)
    with torch.inference_mode(): out=predictor.model(tensor,joints,return_logits=True)
    return {k:v[0].cpu().numpy() for k,v in out.items()}


def compose_frame(rgb,out,title,subtitle,truth=None,xyz_disabled=False):
    import cv2
    width,height=480,360
    base=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(width,height))
    probs=spatial_probability(out['keypoint_logits'])
    panels=[base.copy()]
    colors=[(255,100,0),(0,200,255)]
    for i in range(2):
        strength=cv2.resize((probs[i]/max(probs[i].max(),1e-20)).astype(np.float32),(width,height))
        alpha=(strength*.65)[...,None]
        panel=(base*(1-alpha)+np.asarray(colors[i])*alpha).astype(np.uint8)
        panels.append(panel)
    kp=np.asarray(out['keypoints']).reshape(2,2)
    for panel in panels:
        for i,xy in enumerate(kp):
            x,y=np.rint(xy*[width,height]).astype(int)
            cv2.drawMarker(panel,(x,y),colors[i],cv2.MARKER_CROSS,15,2)
        if truth is not None:
            for xy in np.asarray(truth,dtype=float).reshape(2,2):
                if np.isfinite(xy).all(): cv2.circle(panel,tuple(np.rint(xy*[width,height]).astype(int)),7,(0,255,0),2)
    canvas=np.zeros((height+146,width*3,3),dtype=np.uint8)
    for i,panel in enumerate(panels): canvas[85:85+height,i*width:(i+1)*width]=panel
    def line(text,y,color=(240,240,240),scale=.65): cv2.putText(canvas,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    line(title,24,scale=.8);line(subtitle[:155],49,scale=.55)
    for i,label in enumerate(('Original + prediction crosses','PIN likelihood (blue)','HOLE likelihood (yellow)')):
        cv2.putText(canvas,label,(i*width+10,75),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    gate=float(1/(1+np.exp(-np.clip(float(np.asarray(out['active_logit']).flat[0]),-700,700))))
    delta=np.asarray(out['delta_xyz_m'])*1000
    if xyz_disabled:
        line('Gate score %.3f (unverified) | XYZ disabled; localization-only'%gate,height+110)
    else:
        line('Gate score %.3f | advisory LEFT pin delta in RIGHT base [mm]: %+.2f %+.2f %+.2f'%(gate,*delta),height+110)
    line('Green circles: independent labels when available. Heatmaps normalized per channel for display; no motion executed.',height+135,scale=.55)
    return canvas


def assert_render_disjoint(ids,metadata):
    from scripts.evaluate_pin_insertion_cnn import assert_disjoint
    # Embedded annotation manifests include reserved test rows as provenance;
    # only adaptation train/selection episodes are forbidden.
    base={k:v for k,v in metadata.items() if k!='adaptation'}
    adaptation=metadata.get('adaptation',{})
    forbidden=set(adaptation.get('train_episode_ids',[]))|set(adaptation.get('validation_episode_ids',[]))
    overlap=set(ids)&forbidden
    if overlap: raise ValueError('Adaptation training/validation overlap: '+', '.join(sorted(overlap)))
    return assert_disjoint(ids,base)


def manifest_truth(row):
    if 'keypoints_px' not in row: return None
    points=np.asarray([np.nan if v is None else v for v in row['keypoints_px']],dtype=float).reshape(2,2)
    valid=np.asarray(row.get('valid',[True,True]),dtype=bool)
    points[~valid]=np.nan
    return points/np.array([row.get('image_width',row.get('width')),row.get('image_height',row.get('height'))])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--manifest',type=Path);sources.add_argument('--dataset',type=Path);sources.add_argument('--episodes',nargs='+')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/pin-alignment-forward')
    p.add_argument('--intervals',type=Path,help='Reviewed real intervals JSON: intervals [{episode_id,start_frame,end_frame,note}]')
    p.add_argument('--labels-manifest',type=Path,help='Exact episode/video-frame manual labels only; no interpolation')
    p.add_argument('--groups',nargs='+');p.add_argument('--fps',type=float,default=6)
    p.add_argument('--playback-speed',type=float,default=1.0,help='Saved-video time / display time; .25 shows actual frames at quarter speed')
    p.add_argument('--max-frames',type=int,default=240);p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if not 0<args.fps<=60 or args.max_frames<1 or not 0<args.playback_speed<=4: raise ValueError('Invalid fps/max frames')
    import cv2
    import torch
    from app.pin_insertion_cnn import PinInsertionPredictor
    from scripts.evaluate_pin_insertion_cnn import assert_disjoint, validate_camera
    torch.set_num_threads(4)
    predictor=PinInsertionPredictor(args.checkpoint,device=args.device)
    validate_camera('right_wrist',predictor.metadata)
    if predictor.metadata.get('model_variant')!='heatmap': raise ValueError('Heatmap checkpoint required; regression has no spatial maps')
    frames=[];captures=[]
    if args.dataset:
        with np.load(args.dataset,allow_pickle=False) as archive:
            data={key:archive[key] for key in ('images','state','states','groups','group_ids','keypoints','keypoint_valid','keypoints_valid') if key in archive.files}
        groups=data['group_ids'] if 'group_ids' in data else data['groups']
        selected=args.groups or predictor.metadata.get('test_groups') or predictor.metadata.get('val_groups',[])
        if not selected: raise ValueError('Explicit held-out/validation groups required')
        if set(map(str,selected)) & set(map(str,predictor.metadata.get('train_groups',[]))): raise ValueError('Training groups cannot be rendered as tests')
        ids=np.flatnonzero(np.isin(groups.astype(str),list(map(str,selected))))
        if not len(ids): raise ValueError('No matching dataset groups')
        ids=ids[np.linspace(0,len(ids)-1,min(len(ids),args.max_frames),dtype=int)]
        validity=data.get('keypoints_valid',data.get('keypoint_valid'))
        for i in ids:
            frames.append({'rgb':data['images'][i],'state14':data['state'][i] if 'state' in data else data['states'][i],
                           'keypoints':np.where(validity[i].reshape(2,2)>0,data['keypoints'][i].reshape(2,2),np.nan) if validity is not None else (data['keypoints'][i] if 'keypoints' in data else None),'label':f'group {groups[i]} sample {i}'})
        title='SIM SYNTHETIC | original-scene prototype validation frames, not rollout'
    elif args.manifest:
        rows=load_manifest(args.manifest)
        assert_render_disjoint([r['episode_id'] for r in rows if r.get('episode_id')],predictor.metadata)
        for r in rows[:args.max_frames]:
            image=r.get('image_path',r.get('image')); image=image['path'] if isinstance(image,dict) else image
            bgr=cv2.imread(str(image))
            if bgr is None: raise ValueError(f'Cannot read {image}')
            frames.append(dict(r,rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),state14=r.get('state14',r.get('state')),keypoints=manifest_truth(r),
                               label=r.get('caption',f'{r.get("episode_id","")} frame {r.get("video_frame",r.get("frame_index",""))}')))
        title='REAL HELDOUT | offline saved test frames'
        if any(r.get('split')=='challenge_test' for r in rows):
            title='REAL CHALLENGE | failed attempt; contact likely, unconfirmed; source holds removed'
    else:
        assert_render_disjoint(args.episodes,predictor.metadata)
        from scripts.evaluate_pin_insertion_cnn import eligible_samples
        labels={}
        intervals=[]
        if args.intervals:
            interval_data=json.loads(args.intervals.read_text())
            intervals=interval_data.get('intervals',[]) if isinstance(interval_data,dict) else interval_data
            if not intervals:raise ValueError('No reviewed intervals supplied')
        if args.labels_manifest:
            labels={(r['episode_id'],r['video_frame']):r for r in load_manifest(args.labels_manifest)}
        for episode in args.episodes:
            ep=args.data_root/episode
            meta,samples,_=eligible_samples(ep,stride=1,max_skew_ms=100)
            cap=cv2.VideoCapture(str(ep/meta['videos']['right_wrist']));captures.append(cap)
            source_fps=float(meta.get('fps',30))
            source_stride=max(1,round(source_fps*args.playback_speed/args.fps))
            last_video_frame=-source_stride
            for row,skew in samples:
                matches=[r for r in intervals if r['episode_id']==episode and r['start_frame']<=row['video_frame']<=r['end_frame']]
                if intervals and not matches:continue
                time=row['monotonic_ns']/1e9
                if row['video_frame']-last_video_frame<source_stride: continue
                last_video_frame=row['video_frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,row['video_frame']);ok,bgr=cap.read()
                if not ok: raise ValueError('Cannot decode saved test frame')
                frames.append({'rgb':cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),'state14':row['state'],
                               'keypoints':manifest_truth(labels[(episode,row['video_frame'])]) if (episode,row['video_frame']) in labels else None,
                               'label':f'{episode[:12]} frame {row["video_frame"]} t={time-meta['started_monotonic_ns']/1e9:.3f}s '+(str(matches[0].get('note','Contact/visibility uncertain')) if matches else f'camera/sample skew {skew:.1f} ms')})
                if len(frames)>=args.max_frames: break
            if len(frames)>=args.max_frames: break
        title=f'REAL HELDOUT | saved right-wrist video | {args.playback_speed:g}x playback'
        if intervals:title='REAL HELDOUT | reviewed near-contact intervals; contact/visibility uncertain'
    if not frames: raise ValueError('No renderable frames')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw=args.output.with_suffix('.mp4v.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(1440,506))
    if not writer.isOpened(): raise RuntimeError('Cannot open MP4 writer')
    try:
        for row in frames:
            out=infer(predictor,row['rgb'],row['state14'])
            writer.write(compose_frame(row['rgb'],out,title,row['label'],row.get('keypoints'),predictor.metadata.get('direct_xyz_disabled',False)))
    finally:
        writer.release()
        for cap in captures: cap.release()
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ffmpeg:
        subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],check=True)
        raw.unlink()
    else: raw.replace(args.output)
    report={'output':str(args.output.resolve()),'frames':len(frames),'fps':args.fps,'duration_s':len(frames)/args.fps,
            'xyz_disabled':bool(predictor.metadata.get('direct_xyz_disabled',False)),
            'labels_manifest':str(args.labels_manifest.resolve()) if args.labels_manifest else None,
            'playback_speed':args.playback_speed,'title':title,'checkpoint':str(args.checkpoint.resolve()),
            'limitations':['Offline inference; no insertion success or motor execution demonstrated.',
                           'Display heatmaps are actual model spatial softmax, scaled per channel; intensity is not calibrated confidence.',
                           'Frame sequence plays at chosen display FPS; saved frame sampling may omit time gaps.']}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
