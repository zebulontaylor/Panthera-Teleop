"""Stdlib-only latest-frame advisory alignment sidecar; never issues motor commands."""
from __future__ import annotations
import base64,copy,json,math,os,selectors,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=dict(enabled=True,frequency_hz=20.,gain=.5,max_step_mm=5.,deadband_px=3.,hold_s=30.,frame_yaw_deg=0.,fixed_carrier_hold=True,max_frame_age_ms=5000.,max_state_skew_ms=100.)


def validated_config(values=None,base=None):
    result=dict(DEFAULT_CONFIG)
    if base is not None:
        result.update(validated_config(base))
    for key,value in (values or {}).items():
        if key not in result:raise ValueError('unknown alignment config: '+key)
        if key in ('enabled','fixed_carrier_hold'):
            if type(value) is not bool:raise ValueError(key+' must be boolean')
        else:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError(key+' must be finite numeric')
            lo,hi={'frequency_hz':(.2,20),'gain':(0,2),'max_step_mm':(0,5),'deadband_px':(0,100),'hold_s':(0,120),'frame_yaw_deg':(-180,180),'max_frame_age_ms':(100,10000),'max_state_skew_ms':(0,1000)}[key]
            if not lo<=value<=hi:raise ValueError(key+' outside allowed range')
            value=float(value)
        result[key]=value
    return result


def validate_config(config=None,*,current=None):return validated_config(config,current)


class PinAlignmentMonitor:
    def __init__(self,callback=None,python=None,checkpoint=None,references=None,device='cpu',timeout_s=15,config=None,backend='opencv'):
        self.callback=callback or (lambda result:None);self.timeout_s=float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s<=0:raise ValueError('positive timeout required')
        if backend not in ('auto','cnn','opencv'):raise ValueError('unknown carrier backend')
        default_references=ROOT/('work/left-carrier-opencv/runtime/reference-bank.json' if backend=='opencv' else 'work/pin-hole-demo-cache/references-v2.json')
        self.command=[str(python or ROOT/'work/pin-insertion-cnn/runtime/bin/python'),'-u',str(ROOT/'app/pin_alignment_worker.py'),'--checkpoint',str(checkpoint or ROOT/'work/left-carrier-cnn/real-adapted/best.pt'),'--references',str(references or default_references),'--device',device]
        self.command+=['--backend',backend]
        self._condition=threading.Condition();self._pending=None;self._process=None;self._closed=False
        self._generation=0;self._cache_epoch=0;self._captured_hole=None;self._captured_pin=None;self._request_id=0;self._last_submit=0.;self._config=validated_config(config)
        self._context=dict(camera_id='left_wrist',grasp_id='current',context_id='default');self._jpeg=None;self._jpeg_frame_id=None;self._snapshots={}
        self._status=self._empty('pending' if self._config['enabled'] else 'disabled','waiting for first measured left-wrist frame' if self._config['enabled'] else 'alignment preview disabled')
        self._thread=threading.Thread(target=self._run,name='pin-alignment-monitor',daemon=True);self._thread.start()

    def _empty(self,status,reason):return dict(status=status,reason=reason,ready=False,preview_only=True,motor_permission=False,config=dict(self._config),generation=self._generation,frame_id=None,analyzed_image_available=False,nudge_valid=False,nudge_left_mm=[0.,0.,0.],nudge_right_mm=[0.,0.,0.])

    def _invalidate_locked(self,reason,reset_cache=False,retain_visual=False):
        self._generation+=1;self._pending=None
        if not retain_visual:self._jpeg=None;self._jpeg_frame_id=None
        self._snapshots.clear();self._last_submit=0.
        if reset_cache:
            self._cache_epoch+=1;self._captured_hole=None;self._captured_pin=None
        self._status=self._empty('pending' if self._config['enabled'] else 'disabled',reason)

    def configure(self,config=None,**values):
        with self._condition:
            updated=validated_config({**(config or {}),**values},self._config)
            if updated!=self._config:self._config=updated;self._invalidate_locked('configuration changed; waiting for fresh frame')
            return dict(self._config)

    def reset(self,*,grasp_id=None,context_id=None):
        with self._condition:
            for key,value in [('grasp_id',grasp_id),('context_id',context_id)]:
                if value is not None:
                    if not isinstance(value,str) or not value:raise ValueError(key+' requires nonempty string')
                    self._context[key]=value
            self._invalidate_locked('explicit hole memory reset; waiting for fresh frame',True)
    invalidate=reset

    def capture_hole(self,expected_frame_id,hole_px=None):
        """Latch a fresh, explicit demo-supported hole; never a CNN fallback."""
        with self._condition:
            status=self.status_for_frame(expected_frame_id)
            if status is None:raise ValueError('analyzed frame unavailable or expired; refresh before capture')
            if self._captured_hole is not None:raise ValueError('clear captured hole before recapturing')
            if type(expected_frame_id) is not int or status.get('frame_id')!=expected_frame_id:raise ValueError('analyzed frame changed; refresh before capture')
            manual=hole_px is not None
            if status.get('stale_result') or not status.get('analyzed_timestamp_ns') or status.get('status') in ('error','pending','disabled'):raise ValueError('fresh analyzed frame required')
            if not manual and not status.get('localization_ready'):raise ValueError('fresh supported localization required')
            if not manual and status.get('hole_source') not in ('demo_reference_tracked','demo_reference_held','left_carrier_cnn_observed','left_carrier_opencv_observed','left_carrier_layout_inferred','left_carrier_stationary_accumulated'):raise ValueError('capture requires a supported reference, not CNN fallback')
            if status.get('hole_source')=='left_carrier_stationary_accumulated' and not status.get('accumulation',{}).get('stationary'):
                raise ValueError('retained hole requires a stationary arm pose')
            hole=hole_px if manual else status.get('hole_px')
            if not isinstance(hole,(list,tuple)) or len(hole)!=2 or not all(not isinstance(x,bool) and isinstance(x,(int,float)) and math.isfinite(x) for x in hole) or not (0<=hole[0]<640 and 0<=hole[1]<480):raise ValueError('invalid capture coordinates')
            self._captured_hole=dict(hole_px=list(hole),source='captured_operator_reference' if manual else 'captured_demo_reference',operator_selected=manual,captured_timestamp_ns=time.monotonic_ns(),source_frame_id=expected_frame_id,source_timestamp_ns=status.get('analyzed_timestamp_ns'),source_cache=copy.deepcopy(status.get('cache',{})),context=dict(self._context),assumption='fixed right-wrist coordinate; rigid carrier grasp unchanged; clear after manual takeover or new grasp')
            if status.get('camera_role')=='left_wrist':
                self._captured_hole.update(camera_role='left_wrist',source='captured_operator_reference',
                    measured_state14=copy.deepcopy(status.get('measured_state14',[])),
                    assumption='selected carrier face location re-localized in every left-wrist frame; clear after new grasp')
                candidates=[h for h in status.get('carrier_holes',[]) if h.get('hole_px') is not None]
                if candidates and status.get('carrier_geometry_supported'):
                    nearest=min(candidates,key=lambda h:sum((a-b)**2 for a,b in zip(h['hole_px'],hole)))
                    if not manual or sum((a-b)**2 for a,b in zip(nearest['hole_px'],hole))<=12**2:
                        self._captured_hole['hole_index']=nearest['index']
                if 'hole_index' not in self._captured_hole and status.get('carrier_homography') is not None:
                    # Keep this stdlib monitor free of NumPy/Torch. Invert the
                    # projective map with a two-equation plane solve.
                    h=status['carrier_homography'];u,v=hole
                    a,b=h[0][0]-u*h[2][0],h[0][1]-u*h[2][1]
                    c,d=h[1][0]-v*h[2][0],h[1][1]-v*h[2][1]
                    e,f=u*h[2][2]-h[0][2],v*h[2][2]-h[1][2]
                    determinant=a*d-b*c
                    if abs(determinant)>1e-12:
                        self._captured_hole['face_point_m']=[(e*d-b*f)/determinant,(a*f-e*c)/determinant]
            captured=copy.deepcopy(self._captured_hole)
            self._invalidate_locked('hole captured; waiting for a fresh pin observation',retain_visual=True)
            return captured

    def capture_pin(self,expected_frame_id,pin_px):
        with self._condition:
            status=self.status_for_frame(expected_frame_id)
            if status is None:raise ValueError('analyzed frame unavailable or expired; refresh before capture')
            point=pin_px
            if not isinstance(point,(list,tuple)) or len(point)!=2 or not all(not isinstance(x,bool) and isinstance(x,(int,float)) and math.isfinite(x) for x in point) or not (0<=point[0]<640 and 0<=point[1]<480):raise ValueError('invalid pin coordinates')
            self._captured_pin=dict(pin_px=list(point),source='captured_operator_reference',operator_selected=True,captured_timestamp_ns=time.monotonic_ns(),source_frame_id=expected_frame_id,source_timestamp_ns=status.get('analyzed_timestamp_ns'),context=dict(self._context),assumption='operator-selected fixed image pin coordinate; clear when pin moves')
            if status.get('camera_role')=='left_wrist':
                self._captured_pin.update(camera_role='left_wrist',measured_state14=copy.deepcopy(status.get('measured_state14',[])),
                    assumption='operator-selected pixel expires when either arm moves; recapture after nudge')
            captured=copy.deepcopy(self._captured_pin)
            self._invalidate_locked('pin captured; waiting for fresh measured state',retain_visual=True)
            return captured

    def clear_hole(self):
        with self._condition:
            self._captured_hole=None
            self._cache_epoch+=1
            self._invalidate_locked('hole memory cleared; manual pin retained',retain_visual=True)

    def clear_pin(self):
        with self._condition:
            self._captured_pin=None
            self._invalidate_locked('operator pin cleared; waiting for fresh localization',retain_visual=True)

    def submit(self,rgb=None,state=None,timestamp_ns=None,*,jpeg=None,measured_timestamp_ns=None,camera_id=None,grasp_id=None,context_id=None,**identity):
        if not isinstance(timestamp_ns,int) or isinstance(timestamp_ns,bool) or timestamp_ns<=0:raise ValueError('actual positive monotonic image timestamp_ns required')
        state=[float(v) for v in state]
        if len(state)!=14 or not all(math.isfinite(v) for v in state):raise ValueError('finite measured native state14 required')
        if measured_timestamp_ns is not None and (not isinstance(measured_timestamp_ns,int) or isinstance(measured_timestamp_ns,bool) or measured_timestamp_ns<=0):raise ValueError('measured timestamp must be positive monotonic integer')
        if (rgb is None)==(jpeg is None):raise ValueError('provide exactly one rgb or encoded jpeg')
        with self._condition:
            if self._closed:return None
            context=dict(self._context)
            for key,value in [('camera_id',camera_id),('grasp_id',grasp_id),('context_id',context_id)]:
                if value is not None:
                    if not isinstance(value,str) or not value:raise ValueError(key+' requires nonempty string')
                    context[key]=value
            if context!=self._context:self._context=context;self._invalidate_locked('camera/grasp context changed',True)
            if not self._config['enabled']:return None
            now=time.monotonic()
            if now-self._last_submit<1/self._config['frequency_hz']:return None
            self._last_submit=now;self._request_id+=1
            image=('rgb',rgb.copy()) if rgb is not None else ('jpeg',jpeg if isinstance(jpeg,str) else bytes(jpeg))
            request=dict(state=state,timestamp_ns=timestamp_ns,measured_timestamp_ns=measured_timestamp_ns,request_id=self._request_id,generation=self._generation,cache_epoch=self._cache_epoch,config=dict(self._config),captured_hole=copy.deepcopy(self._captured_hole),captured_pin=copy.deepcopy(self._captured_pin),**context)
            self._pending=(request,image,dict(identity),time.monotonic());self._condition.notify()
            return self._request_id

    def status(self):
        with self._condition:
            result=copy.deepcopy(self._status);result['captured_hole']=copy.deepcopy(self._captured_hole);result['captured_pin']=copy.deepcopy(self._captured_pin)
            result['analyzed_image_available']=self._jpeg is not None;result['display_frame_id']=self._jpeg_frame_id
        if result['captured_pin'] is not None:result['captured_pin']['age_s']=(time.monotonic_ns()-result['captured_pin']['captured_timestamp_ns'])/1e9
        if result['captured_hole'] is not None:result['captured_hole']['age_s']=(time.monotonic_ns()-result['captured_hole']['captured_timestamp_ns'])/1e9
        cache_source=result.get('cache',{}).get('source_ns')
        if cache_source:result['cache']['age_s']=(time.monotonic_ns()-cache_source)/1e9
        stamp=result.get('analyzed_timestamp_ns')
        if stamp:
            age=(time.monotonic_ns()-stamp)/1e6;result['frame_age_ms']=age;result['analyzed_age_s']=age/1000
            if age<0 or age>result['config']['max_frame_age_ms']:
                result.update(status='stale',ready=False,localization_ready=False,stale_result=True,nudge_valid=False,nudge_left_mm=[0.,0.,0.],nudge_right_mm=[0.,0.,0.],reason='analyzed frame is stale; waiting for a current result')
        return result

    def status_for_frame(self,frame_id):
        """Return only the exact retained fresh current-generation snapshot."""
        with self._condition:
            snapshot=self._snapshots.get(frame_id)
            if snapshot is None:return None
            result=copy.deepcopy(snapshot[0])
            stamp=result.get('analyzed_timestamp_ns')
            age=(time.monotonic_ns()-stamp)/1e6 if stamp else float('inf')
            if age<0 or age>self._config['max_frame_age_ms']:return None
            result['frame_age_ms']=age;result['analyzed_age_s']=age/1000
            return result

    def analyzed_frame(self,expected_id=None):
        with self._condition:
            if expected_id is None:return self._jpeg_frame_id,self._jpeg
            snapshot=self._snapshots.get(expected_id)
            if snapshot is None or self.status_for_frame(expected_id) is None:return None,None
            return expected_id,snapshot[1]
    def analyzed_jpeg(self):return self.analyzed_frame()[1]

    def _publish(self,result,request,identity,started):
        encoded=result.pop('analyzed_jpeg_b64',None)
        try:jpeg=base64.b64decode(encoded,validate=True) if encoded else None
        except (ValueError,TypeError):result=self._empty('error','invalid analyzed JPEG from worker');jpeg=None
        with self._condition:
            # Publish completed current-generation frames even when newer input
            # awaits inference; suppressing superseded requests would starve UI.
            if self._closed or request['generation']!=self._generation:return
            normalized=self._empty(result.get('status','error'),result.get('reason'));normalized.update(result);result=normalized
            result.update(identity);result.update(frame_id=request['request_id'],request_id=request['request_id'],generation=request['generation'],config=dict(self._config),preview_only=True,motor_permission=False,total_latency_ms=(time.monotonic()-started)*1000,analyzed_image_available=jpeg is not None)
            self._status=result;self._jpeg=jpeg;self._jpeg_frame_id=request['request_id'] if jpeg else None
            self._snapshots[request['request_id']]=(copy.deepcopy(result),jpeg)
            retained_frames=math.ceil(self._config['frequency_hz']*self._config['max_frame_age_ms']/1000)+3
            while len(self._snapshots)>retained_frames:del self._snapshots[next(iter(self._snapshots))]
        try:self.callback(self.status())
        except Exception:pass

    def _run(self):
        process=None
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:self._condition.wait()
                    if self._closed:return
                    request,image,identity,started=self._pending;self._pending=None
                try:
                    kind,payload=image
                    if kind=='rgb':
                        if tuple(payload.shape)!=(480,640,3) or str(payload.dtype)!='uint8':raise ValueError('full uncropped640x480 RGB uint8 required')
                        payload=payload.tobytes()
                    request['image']=dict(kind=kind,data=payload if isinstance(payload,str) else base64.b64encode(payload).decode('ascii'))
                    if process is None:
                        process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
                        with self._condition:self._process=process
                    process.stdin.write(json.dumps(request,allow_nan=False)+'\n');process.stdin.flush()
                    result=json.loads(self._read_response(process))
                    if result.get('request_id')!=request['request_id'] or result.get('generation')!=request['generation']:raise RuntimeError('alignment worker response identity mismatch')
                except Exception as exc:
                    result=self._empty('error',str(exc)[:500]);self._stop(process);process=None
                    with self._condition:self._process=None
                self._publish(result,request,identity,started)
        finally:self._stop(process)

    def _read_response(self,process):
        deadline=time.monotonic()+self.timeout_s;data=bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not selector.select(remaining):raise TimeoutError('alignment worker timed out')
                chunk=os.read(process.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('alignment worker exited')
                data.extend(chunk)
                if len(data)>3*1024*1024:raise RuntimeError('alignment response exceeds limit')
                if b'\n' in data:
                    line,extra=bytes(data).split(b'\n',1)
                    if extra:raise RuntimeError('unsolicited alignment output')
                    return line.decode()

    @staticmethod
    def _stop(process):
        if process is None:return
        try:process.terminate();process.wait(timeout=.5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        except ProcessLookupError:process.wait()
        finally:
            for stream in (process.stdin,process.stdout):
                if stream:stream.close()

    def close(self):
        with self._condition:
            self._closed=True;self._generation+=1;self._pending=None;process=self._process;self._condition.notify_all()
        if process is not None:
            try:process.terminate()
            except ProcessLookupError:pass
