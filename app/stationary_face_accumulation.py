"""Measured-state-gated temporal face observations; no robot or camera control.

Only raw, actually observed image landmarks enter history. Pin data always comes
from the current input and is never retained. Coordinates remain image pixels.
"""
from copy import deepcopy
from collections import deque
import numpy as np
from app.pin_alignment_geometry import right_link6_transform

R=.0179625
CANONICAL=np.array([[R,0],[0,R],[-R,0],[0,-R]],float)
FACE_FIELDS=('homography','face_homography','face_outline_px','profile_circle_outline_px','layout_holes','face_landmarks','layout_orientation_verified','inner_layout_phase_deg','geometry_kind','registration_kind','point_fit','geometry_source','profile_source','reference_id')

def _point(value):
 try:
  p=np.asarray(value,float)
  return p if p.shape==(2,) and np.isfinite(p).all() else None
 except (TypeError,ValueError):return None

def _H(value):
 try:
  h=np.asarray(value,float)
  return h/h[2,2] if h.shape==(3,3) and np.isfinite(h).all() and abs(np.linalg.det(h))>1e-12 and abs(h[2,2])>1e-12 else None
 except (TypeError,ValueError):return None

def _project(points,H):
 p=np.c_[np.asarray(points,float).reshape(-1,2),np.ones(len(points))]@H.T
 if (np.abs(p[:,2])<1e-10).any():raise ValueError('projective horizon')
 return p[:,:2]/p[:,2,None]

def _tool(q):
 t=right_link6_transform(q);t=t.copy();t[:3,3]+=t[:3,:3]@np.array([.165,0,0]);return t

def _warp_face(pred,D):
 def warp(value):
  try:
   p=np.asarray(value,float)
   if p.ndim==1 and p.shape==(2,):return _project([p],D)[0].tolist()
   if p.ndim==2 and p.shape[1]==2:return _project(p,D).tolist()
  except (ValueError,TypeError):pass
  return value
 for key in ('face_outline_px','profile_circle_outline_px'):
  if pred.get(key) is not None:pred[key]=warp(pred[key])
 for key in ('layout_holes','face_landmarks'):
  for item in pred.get(key,[]) or []:
   for name in ('px','center','profile_px','outline_px'):
    if item.get(name) is not None:item[name]=warp(item[name])

class StationaryFaceAccumulator:
 def __init__(self):self.reset('initial')
 def reset(self,reason='reset'):
  self.anchor=None;self.previous=None;self.previous_ns=None;self.anchor_ns=None;self.frames=0;self.points={};self.last_complete=None;self.last_H=None;self.last_good_ns=None;self.corner_history=deque(maxlen=11);self.context=None;self.rejected_bearing=False;self.reset_reason=str(reason);self.pin_samples=deque(maxlen=31);self.pin_last_ns=None
 def accumulate_pin(self,endpoints,timestamp_ns,config=None):
  """Pool matched endpoint estimates only at the same qualified stationary pose."""
  result=deepcopy(endpoints or {});c=dict(config or {})
  ttl=min(30.,max(0.,float(c.get('hold_s',30.))))
  qualified=(type(timestamp_ns) is int and timestamp_ns==self.previous_ns
   and self.anchor_ns is not None and self.frames>=2
   and (timestamp_ns-self.anchor_ns)/1e9>=float(c.get('stable_s',.4))
   and c.get('fixed_carrier_hold',True) and ttl>0
   and (self.pin_last_ns is None or timestamp_ns>self.pin_last_ns))
  if not qualified:
   self.pin_samples.clear();self.pin_last_ns=None
   result['accumulation']=dict(stationary=False,sample_count=0,last_seen_age_s=None,retained=False)
   return result
  self.pin_last_ns=timestamp_ns;cutoff=timestamp_ns-int(ttl*1e9)
  self.pin_samples=deque((s for s in self.pin_samples if s[0]>=cutoff),maxlen=31)
  top=_point(result.get('top_px'));bottom=_point(result.get('bottom_px'))
  fresh=top is not None and bottom is not None
  if fresh:self.pin_samples.append((timestamp_ns,np.stack((top,bottom))))
  if self.pin_samples:
   top,bottom=np.median(np.array([s[1] for s in self.pin_samples]),axis=0)
   result.update(top_px=top.tolist(),bottom_px=bottom.tolist(),pin_px=bottom.tolist())
   if not fresh:
    # A retained estimate never establishes physical visibility or permission.
    result.update(visible=False,top_detected=False,top_observed=False,top_visibility_verified=False,
     bottom_observed=False,bottom_inferred=True,automatic_nudge_eligible=False)
  result['accumulation']=dict(stationary=True,sample_count=len(self.pin_samples),
   last_seen_age_s=(timestamp_ns-self.pin_samples[-1][0])/1e9 if self.pin_samples else None,
   retained=bool(self.pin_samples and not fresh))
  return result
 def _median(self,record):return np.median(np.array([s[1] for s in record['samples']]),axis=0)
 def _context(self,p,c):return (c.get('context_id'),c.get('context_key'),c.get('cache_epoch'),p.get('camera',self.context[-1] if self.context else None))
 def _observations(self,p,c):
  result=[];width=float(c.get('image_width',p.get('image_width',640)));height=float(c.get('image_height',p.get('image_height',480)));holes=p.get('holes',[]) or [];pixels=p.get('holes_px',[]) or [];visible=p.get('visible',[]) or []
  for i in range(4):
   h=holes[i] if i<len(holes) else {};observed=bool(h.get('observed',visible[i] if i<len(visible) else False)) and not h.get('inferred',False)
   if not observed:continue
   px=_point(pixels[i] if i<len(pixels) else h.get('px'))
   if px is None and _point(h.get('uv')) is not None:px=_point(h['uv'])*[width,height]
   if px is not None:result.append(dict(key=f'main:{i}',slot=i,px=px,xy=CANONICAL[i],kind='insertion_bore',plane='main_face',eligible=True,weight=1.,template=h))
  for a in p.get('face_landmarks',[]) or []:
   if not a.get('observed',False) or a.get('inferred',False):continue
   px=_point(a.get('px',a.get('center')))
   if px is None:continue
   kind=a.get('kind',a.get('type','auxiliary'));plane=a.get('plane','main_face');xy=_point(a.get('canonical_xy_m'));bearing=kind in ('central_bearing','bearing_center')
   if bearing:xy=np.zeros(2)
   eligible=bool(a.get('metric_homography_eligible',bearing)) and plane not in ('raised_boss','boss') and kind!='boss_mounting' and xy is not None
   candidates=[(float(np.linalg.norm(self._median(r)-px)),key) for key,r in self.points.items() if key.startswith('aux:') and r['kind']==kind and r['plane']==plane]
   nearby=[item for item in candidates if item[0]<=6]
   key=min(nearby)[1] if nearby else 'aux:'+str(a.get('id',kind))+':'+str(len(self.points))
   result.append(dict(key=key,slot=None,px=px,xy=xy,kind=kind,plane=plane,eligible=eligible,weight=.5 if bearing else .25 if eligible else 0.,template=a))
  return result
 def _associate_slots(self,raw,c):
  # Reference and cold-start cyclic labels need not share the same phase.
  # Match image positions before assigning observations to temporal records.
  pixels=raw.get('holes_px',[]) or [];visible=raw.get('visible',[]) or []
  historical={i:self._median(self.points[f'main:{i}']) for i in range(4) if f'main:{i}' in self.points}
  if not historical or len(pixels)!=4:return
  limit=float(c.get('visual_shift_px',4.))
  options=[]
  for phase in range(4):
   matches=[np.linalg.norm(_point(pixels[j])-historical[(j+phase)%4]) for j in range(4) if j<len(visible) and visible[j] and _point(pixels[j]) is not None and (j+phase)%4 in historical]
   close=[d for d in matches if d<=limit]
   options.append((len(close),-sum(close),phase))
  ranked=sorted(options,reverse=True);count,_,phase=ranked[0]
  # Require unambiguous, spatially gated evidence; translation remains motion.
  if not phase or not count or (len(ranked)>1 and ranked[1][0]==count):return
  for key in ('holes_px','visible','holes'):
   values=raw.get(key)
   if values is not None and len(values)==4:
    raw[key]=[deepcopy(values[(i-phase)%4]) for i in range(4)]
  for i,item in enumerate(raw.get('holes',[]) or []):
   item.update(index=i,id=f'insertion_bore_{i}',canonical_xy_m=CANONICAL[i].tolist())
  # T maps the stable coordinate basis into this frame's original basis.
  basis=np.linalg.lstsq(CANONICAL,np.roll(CANONICAL,phase,axis=0),rcond=None)[0].T
  T=np.eye(3);T[:2,:2]=basis
  for key in ('homography','face_homography'):
   h=_H(raw.get(key))
   if h is not None:raw[key]=(h@T).tolist()
  for item in raw.get('face_landmarks',[]) or []:
   xy=_point(item.get('canonical_xy_m'))
   if xy is not None:item['canonical_xy_m']=(basis.T@xy).tolist()
  for item in raw.get('layout_holes',[]) or []:
   if item.get('kind')=='insertion_bore':
    suffix=str(item.get('id','')).rsplit('_',1)[-1]
    if suffix.isdigit():item['id']=f'insertion_bore_{(int(suffix)+phase)%4}'
 def _stationary_motion(self,q,now,c):
  arms=np.r_[0:6,7:13];grips=[6,13]
  if np.max(abs(q[arms]-self.anchor[arms]))>float(c.get('joint_tolerance_rad',.003)) or np.max(abs(q[grips]-self.anchor[grips]))>float(c.get('gripper_tolerance_rad',.01)):return 'joint_motion'
  for offset in (0,7):
   a=_tool(self.anchor[offset:offset+6]);b=_tool(q[offset:offset+6]);angle=np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1))
   if np.linalg.norm(a[:3,3]-b[:3,3])>float(c.get('tool_translation_tolerance_m',.0005)) or angle>float(c.get('tool_orientation_tolerance_rad',.005)):return 'tool_motion'
  dt=(now-self.previous_ns)/1e9
  if dt>0 and (np.max(abs(q[arms]-self.previous[arms]))/dt>float(c.get('joint_velocity_rad_s',.05)) or np.max(abs(q[grips]-self.previous[grips]))/dt>float(c.get('gripper_velocity_rad_s',.15))):return 'joint_velocity'
  return None
 def _visual_motion(self,obs,raw,c):
  limit=float(c.get('visual_shift_px',4.));deltas=[];matches=[];self.rejected_bearing=False
  for a in obs:
   record=self.points.get(a['key'])
   if record is None and a['kind'] in ('central_bearing','bearing_center'):
    candidates=[r for r in self.points.values() if r['kind'] in ('central_bearing','bearing_center')]
    record=min(candidates,key=lambda r:np.linalg.norm(a['px']-self._median(r))) if candidates else None
   if record is None and a['eligible'] and a['xy'] is not None:
    record=next((r for r in self.points.values() if r['eligible'] and r['kind']==a['kind'] and r['xy'] is not None and np.linalg.norm(r['xy']-a['xy'])<1e-8),None)
   if record is not None:matches.append((a,a['px']-self._median(record)))
  rejected=set()
  for a,delta in matches:
   if a['kind'] not in ('central_bearing','bearing_center') or np.linalg.norm(delta)<=limit:continue
   coherent=any(other['eligible'] and other['kind'] not in ('central_bearing','bearing_center') and np.linalg.norm(change)>limit and np.linalg.norm(change-delta)<=3 for other,change in matches if other is not a)
   if coherent:return 'bearing_motion'
   rejected.add(id(a));self.rejected_bearing=True
  obs[:]=[a for a in obs if id(a) not in rejected]
  deltas=[delta for a,delta in matches if a['slot'] is not None]
  if len(deltas)>=3:
   d=np.asarray(deltas);median=np.median(d,axis=0)
   if np.linalg.norm(median)>limit and np.median(np.linalg.norm(d-median,axis=1))<=3:return 'visual_motion'
  h=_H(raw.get('homography'))
  if self.last_H is not None and h is not None and sum(a['slot'] is not None for a in obs)>=3:
   d=_project(CANONICAL,h)-_project(CANONICAL,self.last_H)
   if np.median(np.linalg.norm(d,axis=1))>limit and np.median(np.linalg.norm(d-np.median(d,axis=0),axis=1))<=3:return 'visual_motion'
  return None

 def _fit(self):
  main=[];aux=[]
  for record in self.points.values():
   if not record['eligible'] or record['xy'] is None:continue
   target=(record['xy'],self._median(record),record['weight'],record['samples'][-1][0])
   (main if record['slot'] is not None or record['kind'] in ('central_bearing','bearing_center') else aux).append(target)
  if len(main)<3:return None,None
  xy=np.array([r[0] for r in main]);target=np.array([r[1] for r in main]);weights=np.array([r[2] for r in main])
  source=_project(xy,self.last_H) if self.last_H is not None else xy
  X=np.c_[source,np.ones(len(source))]
  if np.linalg.matrix_rank(X)<3:return None,None
  beta=np.linalg.lstsq(X*np.sqrt(weights[:,None]),target*np.sqrt(weights[:,None]),rcond=None)[0];base=np.vstack((beta.T,[0,0,1]));base=base@self.last_H if self.last_H is not None else base
  # Optional inner points have bounded PER-LANDMARK weight, never per-frame votes.
  if aux:
   from app.left_carrier_opencv import weighted_affine_refine
   mains=[r for r in main if not np.allclose(r[0],[0,0])];bearing=next((r[1] for r in main if np.allclose(r[0],[0,0])),None)
   base,_=weighted_affine_refine(base,[r[0] for r in mains],[r[1] for r in mains],[r[0] for r in aux],[r[1] for r in aux],bearing_px=bearing,aux_weight=.25)
  residual=np.linalg.norm(_project(xy,base)-target,axis=1)
  if not np.isfinite(base).all() or np.max(residual)>2.:return None,None
  return base,min(r[3] for r in main)
 def update(self,prediction,state14,timestamp_ns,config=None,*,eligible=True):
  raw=deepcopy(prediction);c=dict(config or {})
  try:now=int(timestamp_ns);q=np.asarray(state14,float)
  except (TypeError,ValueError,OverflowError):self.reset('ineligible_frame');return self._raw(raw,False,0)
  ttl=min(30.,max(0.,float(c.get('hold_s',30.))));context=self._context(raw,c)
  invalid=not eligible or q.shape!=(14,) or not np.isfinite(q).all() or now<0
  if c.get('now_ns') is not None:
   age=(int(c['now_ns'])-now)/1e9;invalid|=age>float(c.get('max_input_age_s',.25)) or age < -float(c.get('future_tolerance_s',.02))
  if invalid or not c.get('fixed_carrier_hold',True) or ttl<=0:
   self.reset('ineligible_frame' if invalid else 'hold_disabled');return self._raw(raw,False,now)
  if self.context is not None and context!=self.context:self.reset('context_changed')
  if self.previous_ns is not None and now<=self.previous_ns:self.reset('nonmonotonic_timestamp');return self._raw(raw,False,now)
  if self.previous_ns is not None and (now-self.previous_ns)/1e9>float(c.get('max_gap_s',2.5)):self.reset('input_gap')
  motion=self._stationary_motion(q,now,c) if self.anchor is not None else None
  if motion:self.reset(motion)
  self.context=context
  if self.anchor is None:self.anchor=q.copy();self.anchor_ns=now
  self.previous=q.copy();self.previous_ns=now;self.frames+=1
  cutoff=now-int(ttl*1e9);had=bool(self.points)
  for key in list(self.points):
   record=self.points[key];record['samples']=deque((s for s in record['samples'] if s[0]>=cutoff),maxlen=31)
   if not record['samples']:del self.points[key]
  if self.last_good_ns is not None and self.last_good_ns<cutoff:self.last_H=None;self.last_complete=None;self.last_good_ns=None;self.corner_history.clear()
  if had and not self.points:self.reset_reason='ttl_expired'
  self._associate_slots(raw,c)
  obs=self._observations(raw,c);visual=self._visual_motion(obs,raw,c)
  if visual:
   self.reset(visual);self.context=context;self.anchor=q.copy();self.previous=q.copy();self.anchor_ns=now;self.previous_ns=now;self.frames=1;obs=self._observations(raw,c)
  # Reject isolated jumps after coherent-motion detection has had its chance.
  obs[:]=[a for a in obs if a['slot'] is None or a['key'] not in self.points or np.linalg.norm(a['px']-self._median(self.points[a['key']]))<=float(c.get('visual_shift_px',4.))]
  # Unverified auxiliary IDs can change with contour ordering. Keep their
  # consecutive spatial tracks, but never retain unmatched guesses as openings.
  # Verified coplanar anchors keep the normal bounded temporal memory.
  current_keys={a['key'] for a in obs}
  for key in list(self.points):
   record=self.points[key]
   if record['slot'] is None and not record['eligible'] and key not in current_keys:
    del self.points[key]
  for a in obs:
   key=a['key']
   if key not in self.points:self.points[key]={**a,'samples':deque(maxlen=31)}
   elif a['eligible']:
    for field in ('xy','eligible','weight','template'):self.points[key][field]=deepcopy(a[field])
   self.points[key]['samples'].append((now,a['px'].copy()))
  stationary=(now-self.anchor_ns)/1e9>=float(c.get('stable_s',.4)) and self.frames>=2
  if not stationary:return self._raw(raw,False,now)
  h,good_at=self._fit();current_h=_H(raw.get('homography')) if raw.get('supported',raw.get('geometry_valid',False)) and not self.rejected_bearing else None
  if h is None and current_h is not None:
   self.corner_history.append((now,_project(CANONICAL,current_h)));self.corner_history=deque((r for r in self.corner_history if r[0]>=cutoff),maxlen=11)
   import cv2
   corners=np.median(np.array([r[1] for r in self.corner_history]),axis=0);h=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),corners.astype(np.float32));good_at=now
  if h is not None:
   self.last_H=h;self.last_good_ns=good_at
  elif self.last_H is not None:h=self.last_H
  result=deepcopy(raw)
  if self.last_complete:
   for key in FACE_FIELDS:
    if key in self.last_complete and (not raw.get('supported') or raw.get(key) is None or (isinstance(raw.get(key),list) and not raw.get(key))):result[key]=deepcopy(self.last_complete[key])
  old_h=_H(result.get('homography'))
  if h is not None:
   if old_h is not None:
    try:_warp_face(result,h@np.linalg.inv(old_h))
    except (ValueError,np.linalg.LinAlgError):pass
   result['homography']=h.tolist();result['supported']=True;result['geometry_valid']=True;result['geometry_kind']='stationary_observed_landmark_pool';result['method']='left_carrier_stationary_accumulated'
  else:result['supported']=False;result['geometry_valid']=False
  fresh={a['slot']:a for a in obs if a['slot'] is not None};holes=[];pixels=[];retained=0;ages=[];width=float(c.get('image_width',raw.get('image_width',640)));height=float(c.get('image_height',raw.get('image_height',480)))
  for i in range(4):
   template=deepcopy((raw.get('holes',[]) or [])[i]) if i<len(raw.get('holes',[]) or []) else {'index':i};record=self.points.get(f'main:{i}');observed=i in fresh
   # A fresh target must describe this frame, not the pooled fitting median.
   # History supplies coordinates only while that bore is not observed.
   px=fresh[i]['px'].copy() if observed else self._median(record) if record else _project([CANONICAL[i]],h)[0] if h is not None else None
   age=(now-record['samples'][-1][0])/1e9 if record else None;accumulated=bool(record is not None and not observed);retained+=int(accumulated)
   if age is not None:ages.append(age)
   template.update(index=i,observed=observed,accumulated=accumulated,inferred=bool(px is not None and record is None),sample_count=len(record['samples']) if record else 0,last_seen_age_s=age,uv=(px/[width,height]).tolist() if px is not None else None,metric_homography_eligible=observed)
   if observed:template['fresh_observation_px']=fresh[i]['px'].tolist()
   holes.append(template);pixels.append(px.tolist() if px is not None else None)
  result['holes']=holes;result['holes_px']=pixels;result['visible']=[i in fresh for i in range(4)]
  # All retained auxiliary observations remain explicitly historical, never fresh.
  landmarks=[];retained_aux=0
  for a in obs:
   if a['slot'] is None:
    item=deepcopy(a['template']);item.update(sample_count=len(self.points[a['key']]['samples']),last_seen_age_s=0.,accumulated=False);landmarks.append(item)
  for record in self.points.values():
   if record['slot'] is not None or any(a['key']==record['key'] for a in obs):continue
   retained_aux+=1;a=deepcopy(record['template']);a.update(px=self._median(record).tolist(),observed=False,accumulated=True,inferred=False,last_seen_age_s=(now-record['samples'][-1][0])/1e9,sample_count=len(record['samples']));landmarks.append(a)
  result['face_landmarks']=landmarks
  # Keep layout glyph metadata consistent with the merged four main slots.
  for item in result.get('layout_holes',[]) or []:
   if item.get('kind')=='insertion_bore' and str(item.get('id','')).rsplit('_',1)[-1].isdigit():
    i=int(item['id'].rsplit('_',1)[-1])
    if 0<=i<4:item.update(px=pixels[i],observed=holes[i]['observed'],accumulated=holes[i]['accumulated'],inferred=holes[i]['inferred'])
   elif item.get('kind')!='insertion_bore':
    kind=item.get('kind');px=_point(item.get('px'));actual=[a for a in obs if a['slot'] is None and a['kind']==kind and px is not None and np.linalg.norm(a['px']-px)<=6]
    historical=[r for r in self.points.values() if r['slot'] is None and r['kind']==kind and px is not None and np.linalg.norm(self._median(r)-px)<=6]
    item['observed']=bool(actual);item['accumulated']=bool(not actual and historical);item['inferred']=not bool(actual or historical)
    if actual:item['px']=actual[0]['px'].tolist();item['last_seen_age_s']=0.
    elif historical:item['last_seen_age_s']=(now-historical[0]['samples'][-1][0])/1e9
  ages.extend((now-r['samples'][-1][0])/1e9 for r in self.points.values())
  if self.last_good_ns is not None:ages.append((now-self.last_good_ns)/1e9)
  result['accumulation']=dict(stationary=True,frames=self.frames,point_count=len(self.points),retained_count=retained+retained_aux,retained_hole_count=retained,age_s=max(ages,default=0.),reset_reason=self.reset_reason)
  result['source']='left_carrier_stationary_accumulated';result['pixel_hold']=bool(h is not None and (retained+retained_aux or not raw.get('supported')))
  if h is not None:self.last_complete=deepcopy(result)
  return result
 def _raw(self,raw,stationary,now):
  raw['accumulation']=dict(stationary=stationary,frames=self.frames,point_count=len(self.points),retained_count=0,age_s=0.,reset_reason=self.reset_reason);return raw
