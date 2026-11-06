#!/usr/bin/env python3
"""Persistent localization/cache/geometry preview worker. No motor APIs."""
import argparse,base64,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class AlignmentWorker:
    def __init__(self,checkpoint,references,device='cpu'):
        import torch,cv2
        cv2.setNumThreads(4)
        from app.pin_insertion_cnn import PinInsertionPredictor
        from app.pin_hole_demo_cache import DemoHoleCache
        torch.set_num_threads(2)
        self.predictor=PinInsertionPredictor(checkpoint,device=device)
        stationary=ROOT/'work/pin-hole-demo-cache/stationary-config.json'
        self.cache=DemoHoleCache.from_references(references,json.loads(stationary.read_text()) if stationary.exists() else dict(max_hold_ns=30_000_000_000))
        self.epoch=None;self.camera_id=None
        self.checkpoint=str(checkpoint);self.references=str(references)

    def score(self,request):
        import cv2,numpy as np
        from app.pin_alignment_monitor import validated_config
        from app.pin_alignment_geometry import estimate_preview_nudge
        config=validated_config(request['config']);started=time.monotonic_ns();stamp=int(request['timestamp_ns'])
        image=request['image'];raw=base64.b64decode(image['data'],validate=True)
        if image['kind']=='rgb':
            if len(raw)!=640*480*3:raise ValueError('raw RGB must be640x480x3')
            rgb=np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()
        elif image['kind']=='jpeg':
            bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError('invalid JPEG')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        else:raise ValueError('unknown image encoding')
        if rgb.shape!=(480,640,3):raise ValueError('full uncropped640x480 right-wrist image required')
        if stamp<=0 or stamp>time.monotonic_ns():raise ValueError('invalid/future monotonic image timestamp')
        if request['cache_epoch']!=self.epoch:self.cache.reset();self.epoch=request['cache_epoch']
        self.cache.max_hold=int(config['hold_s']*1e9) if config['fixed_carrier_hold'] else 0
        cache=self.cache.update(rgb,timestamp_ns=stamp,now_ns=stamp,sequence_id=request['context_id'],camera_id=request['camera_id'],grasp_id=request['grasp_id'])
        visual_cache=cache
        captured=request.get('captured_hole')
        if captured is not None:
            if captured.get('context')!={k:request[k] for k in ('camera_id','grasp_id','context_id')}:raise ValueError('captured hole context mismatch')
            xy=np.asarray(captured['hole_px'],float)
            if xy.shape!=(2,) or not np.isfinite(xy).all() or not (0<=xy[0]<640 and 0<=xy[1]<480):raise ValueError('invalid captured hole')
            cache=dict(mode='captured',hole_px=xy.tolist(),source_ns=captured['captured_timestamp_ns'],assumption=captured['assumption'])
        prediction=self.predictor.predict(rgb,request['state']);uv=np.asarray(prediction['keypoints'],float).reshape(2,2)
        pin=uv[0];hole=uv[1];pin_source='cnn'
        captured_pin=request.get('captured_pin')
        if captured_pin is not None:
            if captured_pin.get('context')!={k:request[k] for k in ('camera_id','grasp_id','context_id')}:raise ValueError('captured pin context mismatch')
            point=np.asarray(captured_pin['pin_px'],float)
            if point.shape!=(2,) or not np.isfinite(point).all() or not (0<=point[0]<640 and 0<=point[1]<480):raise ValueError('invalid captured pin')
            pin=point/[640,480];pin_source='operator_selected'

        if not np.isfinite(pin).all() or not ((pin>=0).all() and (pin<=1).all()):raise ValueError('current CNN pin localization invalid')
        hole_source='cnn_unverified_fallback';cache_valid=cache['mode'] in ('tracked','held','captured') and cache.get('hole_px') is not None
        if cache_valid:hole=np.asarray(cache['hole_px'])/[640,480];hole_source=captured.get('source','captured_demo_reference') if captured is not None else 'demo_reference_'+cache['mode']
        pin_px=pin*[640,480];hole_px=hole*[640,480]
        geometry=dict(valid=False,reason='No supported carrier reference; CNN hole is unverified',delta_left_m=[0,0,0],delta_right_m=[0,0,0])
        if cache_valid:geometry=estimate_preview_nudge(request['state'],pin_px.tolist(),hole_px.tolist(),gain=config['gain'],max_step_m=config['max_step_mm']/1000,deadband_px=config['deadband_px'],frame_yaw_deg=config['frame_yaw_deg'])
        measured=request.get('measured_timestamp_ns');measure_type_valid=isinstance(measured,int) and not isinstance(measured,bool) and measured>0
        skew_ms=abs(measured-stamp)/1e6 if measure_type_valid else None
        if skew_ms is None or skew_ms>config['max_state_skew_ms']:geometry.update(valid=False,reason='measured-state timestamp unavailable or image/state skew exceeds limit')
        now=time.monotonic_ns();age_ms=(now-stamp)/1e6;stale=age_ms>config['max_frame_age_ms']
        if cache['mode']=='held' and cache.get('source_ns') and now-cache['source_ns']>int(config['hold_s']*1e9):
            cache_valid=False;geometry.update(valid=False,reason='hole memory hold expired in wall time')
        if measure_type_valid and measured>now:geometry.update(valid=False,reason='measured-state timestamp is in the future')
        if stale:geometry.update(valid=False,reason='analyzed frame is stale; no current proposal')
        pin_px=pin*[640,480];hole_px=hole*[640,480];error=hole_px-pin_px
        overlay=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.circle(overlay,tuple(np.rint(pin_px).astype(int)),7,(255,255,0),2)
        cv2.circle(overlay,tuple(np.rint(hole_px).astype(int)),8,(255,0,200),2)
        valid=bool(geometry.get('valid'))
        has_step=valid and np.linalg.norm(geometry.get('delta_left_m',[0,0,0]))>1e-12
        if has_step:cv2.arrowedLine(overlay,tuple(np.rint(pin_px).astype(int)),tuple(np.rint(hole_px).astype(int)),(0,220,255),2,tipLength=.2)
        pin_label_offset=[9,20] if pin_px[1]>=hole_px[1] else [9,-12]
        hole_label_offset=[10,-12] if pin_px[1]>=hole_px[1] else [10,20]
        cv2.putText(overlay,'pin manual' if captured_pin is not None else 'pin',tuple(np.clip(np.rint(pin_px+pin_label_offset),[0,15],[600,470]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1,cv2.LINE_AA)
        cv2.putText(overlay,'hole' if cache_valid else 'hole unverified',tuple(np.clip(np.rint(hole_px+hole_label_offset),[0,15],[510,470]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,0,200),1,cv2.LINE_AA)
        cv2.putText(overlay,'PIN cyan / HOLE purple / PREVIEW ONLY',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(overlay,f"hole: {hole_source} | {cache['mode']}",(8,46),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(overlay,'Alignment direction (preview)' if has_step else 'Aligned / zero preview step' if valid else 'Nudge unavailable',(8,69),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,220,255) if has_step else (230,230,230),1,cv2.LINE_AA)
        okay,jpeg=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,88])
        if not okay:raise RuntimeError('overlay JPEG encoding failed')
        valid=bool(geometry.get('valid'))
        return dict(status='stale' if stale else 'ready' if cache_valid else 'waiting_for_hole',ready=not stale and cache_valid,localization_ready=not stale and cache_valid,nudge_blocked_reason=None if valid else geometry.get('reason'),reason=geometry.get('reason'),stale_result=stale,
            analyzed_timestamp_ns=stamp,measured_timestamp_ns=measured,measured_state14=list(request['state']),captured_hole=captured,visual_cache=visual_cache,completed_timestamp_ns=now,frame_age_ms=age_ms,analyzed_age_s=age_ms/1000,state_image_skew_ms=skew_ms,queue_age_ms=(started-stamp)/1e6,
            image_size_wh=[640,480],pin_source=pin_source,captured_pin=captured_pin,pin_uv=pin.tolist(),hole_uv=hole.tolist(),pin_px=pin_px.tolist(),hole_px=hole_px.tolist(),error_px=error.tolist(),pixel_error_px=float(np.linalg.norm(error)),hole_source=hole_source,
            cache={**cache,'age_s':(now-cache['source_ns'])/1e9 if cache.get('source_ns') else None,'clock':'causal observation timestamp for update; actual wall age reported separately'},
            nudge_valid=valid,nudge_left_mm=(np.asarray(geometry.get('delta_left_m',[0,0,0]))*1000).tolist() if valid else [0,0,0],nudge_right_mm=(np.asarray(geometry.get('delta_right_m',[0,0,0]))*1000).tolist() if valid else [0,0,0],
            geometry=geometry,arrow_kind='alignment_direction' if has_step else 'none',preview_only=True,motor_permission=False,cnn_gate_permission=False,cnn_xyz_head_used=False,checkpoint=self.checkpoint,reference_bank=self.references,
            analyzed_jpeg_b64=base64.b64encode(jpeg.tobytes()).decode('ascii'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--references',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--backend',choices=('auto','cnn','opencv'),default='auto');args=parser.parse_args()
    worker=None
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            if worker is None:
                if args.backend=='opencv':
                    from app.left_carrier_alignment import LeftCarrierAlignmentWorker
                    worker=LeftCarrierAlignmentWorker(None,args.device,backend='opencv',references=args.references)
                else:
                    # The checkpoint contract selects its adapter, never the USB
                    # port or filename. Explicit legacy checkpoints remain usable.
                    import torch
                    metadata=torch.load(args.checkpoint,map_location='cpu',weights_only=True).get('metadata',{})
                    if metadata.get('schema')=='carrier-face-left-rgb-v1':
                        from app.left_carrier_alignment import LeftCarrierAlignmentWorker
                        worker=LeftCarrierAlignmentWorker(args.checkpoint,args.device)
                    else:
                        worker=AlignmentWorker(args.checkpoint,args.references,args.device)
            result=worker.score(request)
        except Exception as exc:result=dict(status='error',ready=False,nudge_valid=False,reason=str(exc)[:500],preview_only=True,motor_permission=False)
        result.update(request_id=request.get('request_id'),generation=request.get('generation'))
        print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':main()
