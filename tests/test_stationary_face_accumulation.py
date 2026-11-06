import numpy as np
from app.stationary_face_accumulation import StationaryFaceAccumulator,CANONICAL

H=np.array([[2000.,0,320],[0,2000.,240],[0,0,1]])
def point(i):return (CANONICAL[i]*2000+[320,240]).tolist()
def raw(slots=(),supported=False,shift=(0,0),aux=(),pin=None):
 pixels=[(np.array(point(i))+shift).tolist() if supported or i in slots else None for i in range(4)]
 return {'camera':'left_wrist','reference_id':'same','supported':supported,'geometry_valid':supported,'homography':(np.array([[1,0,shift[0]],[0,1,shift[1]],[0,0,1]])@H).tolist() if supported else None,'holes_px':pixels,'visible':[i in slots for i in range(4)],'holes':[dict(index=i,observed=i in slots,inferred=i not in slots and supported,uv=np.array(pixels[i]).tolist() if pixels[i] else None) for i in range(4)],'face_landmarks':list(aux),'face_outline_px':[[300,220],[340,220],[340,260],[300,260]] if supported else None,'layout_holes':[dict(id='insertion_bore_0',kind='insertion_bore',px=pixels[0],outline_px=[[350,235],[355,240]])] if supported else [],'pin_px':pin}
def update(a,p,t,q=None,c=None,**kw):return a.update(p,np.zeros(14) if q is None else q,int(t*1e9),c,**kw)
def warm(a):
 update(a,raw([0]),0);update(a,raw([1]),.2);return update(a,raw([2]),.4)

def test_intermittent_union_qualifies_and_never_promotes_retained_to_observed():
 a=StationaryFaceAccumulator();r=warm(a);assert r['supported'];assert r['accumulation']['stationary'];assert r['accumulation']['point_count']==3;assert r['accumulation']['retained_count']==2;assert r['visible']==[False,False,True,False];assert r['holes'][0]['accumulated'] and not r['holes'][0]['observed'];assert r['holes'][3]['inferred'] and not r['holes'][3]['accumulated']

def test_movement_first_frame_clears_history_and_cumulative_creep_uses_anchor():
 a=StationaryFaceAccumulator();warm(a);q=np.zeros(14);q[7]=.01;r=update(a,raw([3]),.5,q);assert not r['accumulation']['stationary'];assert r['holes_px'][0] is None;assert not a.points.get('main:0')
 b=StationaryFaceAccumulator();warm(b)
 for i in range(1,12):
  q=np.zeros(14);q[0]=i*.0003;r=update(b,raw([2]),.4+i*.2,q)
  if r['accumulation']['reset_reason'] in ('joint_motion','tool_motion'):break
 assert not r['accumulation']['stationary'];assert r['accumulation']['reset_reason'] in ('joint_motion','tool_motion');assert 'main:0' not in b.points

def test_context_hold_disable_gap_and_eligibility_reset():
 a=StationaryFaceAccumulator();warm(a);r=update(a,raw(),.5,c={'context_id':'new'});assert not r['accumulation']['stationary'];assert not r['supported']
 warm(a);r=update(a,raw(),1.,c={'fixed_carrier_hold':False});assert r['accumulation']['reset_reason']=='hold_disabled';assert not a.points
 b=StationaryFaceAccumulator();warm(b);r=update(b,raw(),4.);assert r['accumulation']['reset_reason']=='input_gap';assert not r['supported'];assert not r['accumulation']['stationary']
 warm(b);r=update(b,raw(),5.,eligible=False);assert not b.points;assert r['accumulation']['reset_reason']=='ineligible_frame'

def test_ttl_expiry_does_not_pool_inferred_and_pin_is_always_fresh():
 a=StationaryFaceAccumulator();update(a,raw([0,1,2],True,pin=[1,2]),0,c={'hold_s':1});r=update(a,raw([],True,pin=None),.5,c={'hold_s':1});assert r['pin_px'] is None;assert len(a.points)==3
 r=update(a,raw(),1.6,c={'hold_s':1});assert not a.points;assert not r['supported'];assert all(p is None for p in r['holes_px'])

def test_median_rejects_single_main_outlier_and_boss_cannot_move_homography():
 a=StationaryFaceAccumulator();update(a,raw([0,1,2]),0);update(a,raw([0,1,2]),.2);warm_r=update(a,raw([0,1,2]),.4);bad=raw([0]);bad['holes_px'][0]=[point(0)[0]+30,point(0)[1]+20];r=update(a,bad,.6);assert np.linalg.norm(np.array(r['holes_px'][0])-point(0))<1e-6
 boss=dict(id='boss',kind='boss_mounting',plane='raised_boss',px=[50,50],canonical_xy_m=[0,0],observed=True,metric_homography_eligible=True)
 r=update(a,raw([],aux=[boss]),.8);assert np.allclose(np.array(r['homography']),np.array(warm_r['homography']));assert r['accumulation']['point_count']==4

def test_fresh_bore_uses_current_pixels_while_retained_bore_uses_history():
 a=StationaryFaceAccumulator()
 for t in (0,.2,.4):update(a,raw([0,1,2],True),t)
 current=raw([0],True)
 current['holes_px'][0]=(np.array(point(0))+[3.,0.]).tolist()
 r=update(a,current,.6)
 assert r['holes_px'][0]==current['holes_px'][0]
 assert r['holes'][0]['fresh_observation_px']==current['holes_px'][0]
 assert r['holes'][0]['observed'] and not r['holes'][0]['accumulated']
 assert np.allclose(r['holes'][0]['uv'],np.array(current['holes_px'][0])/[640,480])
 assert r['layout_holes'][0]['px']==current['holes_px'][0]
 # Geometry can remain pooled without moving a fresh target back to its median.
 assert np.allclose(r['homography'],H)
 retained=update(a,raw(),.8)
 assert np.allclose(retained['holes_px'][0],point(0))
 assert retained['holes'][0]['accumulated'] and not retained['holes'][0]['observed']

def test_fresh_visual_shift_and_bearing_shift_reset_without_joint_motion():
 a=StationaryFaceAccumulator();warm(a);r=update(a,raw([0,1,2],shift=(6,0)),.6);assert r['accumulation']['reset_reason']=='visual_motion';assert not r['accumulation']['stationary']
 b=StationaryFaceAccumulator();bearing=lambda p:dict(kind='central_bearing',id='center',plane='main_face',px=p,observed=True,metric_homography_eligible=True,canonical_xy_m=[0,0]);update(b,raw([0,1],aux=[bearing([320,240])]),0);update(b,raw([0,1],aux=[bearing([320,240])]),.5);r=update(b,raw([0,1],aux=[bearing([330,240])]),.7);assert r['accumulation']['stationary'];assert r['accumulation']['reset_reason']!='bearing_motion';assert np.allclose(next(v for v in b.points.values() if v['kind']=='central_bearing')['samples'][-1][1],[320,240])

def test_fk_tool_displacement_can_reset_below_joint_tolerance():
 a=StationaryFaceAccumulator();warm(a);q=np.zeros(14);q[1:4]=.002;r=update(a,raw(),.8,q);assert r['accumulation']['reset_reason']=='tool_motion';assert not r['accumulation']['stationary']

def test_nonmonotonic_stale_future_and_ttl_cap():
 a=StationaryFaceAccumulator();warm(a);r=update(a,raw(),.3);assert r['accumulation']['reset_reason']=='nonmonotonic_timestamp'
 r=update(a,raw(),1,c={'now_ns':int(2e9)});assert r['accumulation']['reset_reason']=='ineligible_frame'
 r=update(a,raw(),3,c={'now_ns':int(2e9)});assert r['accumulation']['reset_reason']=='ineligible_frame'


def test_reference_switch_and_reference_loss_do_not_reset_real_context():
 a=StationaryFaceAccumulator();warm(a);p=raw();p['reference_id']='different_sift_reference';r=update(a,p,.6);assert r['accumulation']['stationary'];assert r['holes'][0]['accumulated'];p['reference_id']=None;r=update(a,p,.8);assert r['accumulation']['stationary'];assert r['accumulation']['retained_count']==3

def test_retained_auxiliary_layout_is_not_current_observation():
 a=StationaryFaceAccumulator();bearing=dict(id='center',kind='central_bearing',plane='main_face',px=[320,240],observed=True,canonical_xy_m=[0,0],metric_homography_eligible=True);p=raw([0,1,2],True,aux=[bearing]);p['layout_holes'].append(dict(id='central_bearing',kind='central_bearing',px=[320,240],observed=True,inferred=False));update(a,p,0);update(a,p,.5);r=update(a,raw(),.7);item=next(v for v in r['layout_holes'] if v['kind']=='central_bearing');assert not item['observed'];assert item['accumulated'];assert not item['inferred'];assert not r['face_landmarks'][0]['observed']


def test_late_inner_confirmation_upgrades_pool_geometry_metadata_without_downgrade():
 a=StationaryFaceAccumulator();unknown=dict(id='variable_inner',kind='inner_threaded',plane='main_face',px=[330,250],observed=True,metric_homography_eligible=False,canonical_xy_m=None);update(a,raw([0,1,2],aux=[unknown]),0);verified={**unknown,'metric_homography_eligible':True,'canonical_xy_m':[.005,.005]};update(a,raw([0,1,2],aux=[verified]),.3);record=next(r for r in a.points.values() if r['kind']=='inner_threaded');assert record['eligible'];assert np.allclose(record['xy'],[.005,.005]);assert record['weight']==.25;update(a,raw([0,1,2],aux=[unknown]),.6);record=next(r for r in a.points.values() if r['kind']=='inner_threaded');assert record['eligible'];assert np.allclose(record['xy'],[.005,.005]);assert len(record['samples'])==3


def test_bearing_motion_needs_second_coherent_coplanar_observation():
 a=StationaryFaceAccumulator();bearing=lambda p:dict(kind='central_bearing',id='center',plane='main_face',px=p,observed=True,metric_homography_eligible=True,canonical_xy_m=[0,0]);update(a,raw([0,1],aux=[bearing([320,240])]),0);update(a,raw([0,1],aux=[bearing([320,240])]),.5);r=update(a,raw([0,1],shift=[6,0],aux=[bearing([326,240])]),.7);assert r['accumulation']['reset_reason']=='bearing_motion';assert not r['accumulation']['stationary']


def relabel(p,phase):
 from copy import deepcopy
 p=deepcopy(p)
 for key in ('holes_px','visible','holes'):p[key]=p[key][phase:]+p[key][:phase]
 rotation=np.linalg.lstsq(CANONICAL,np.roll(CANONICAL,-phase,axis=0),rcond=None)[0].T
 T=np.eye(3);T[:2,:2]=rotation
 if p['homography'] is not None:p['homography']=(np.array(p['homography'])@T).tolist()
 return p


def test_cyclic_reference_switch_preserves_bore_positions_and_homography_basis():
 for phase in (1,2,3):
  a=StationaryFaceAccumulator();update(a,raw(range(4),True),0)
  r=update(a,relabel(raw(range(4),True),phase),.5)
  assert r['supported']
  assert np.allclose(r['holes_px'],[point(i) for i in range(4)])
  assert np.allclose(r['homography'],H)
  assert all(h['sample_count']==2 for h in r['holes'])
  assert r['accumulation']['reset_reason']!='visual_motion'


def test_partial_cyclic_switch_associates_by_image_and_isolated_outlier_is_not_fresh():
 a=StationaryFaceAccumulator();update(a,raw(range(4),True),0)
 r=update(a,relabel(raw([0,2],True),2),.5)
 assert np.allclose(r['holes_px'],[point(i) for i in range(4)])
 assert r['visible']==[True,False,True,False]
 bad=raw([0]);bad['holes_px'][0]=[320,240]
 r=update(a,bad,.7)
 assert np.allclose(r['holes_px'][0],point(0))
 assert not r['holes'][0]['observed']
 assert r['holes'][0]['sample_count']==2


def test_unverified_auxiliary_tracks_do_not_leave_historical_duplicates():
 a=StationaryFaceAccumulator()
 def auxiliary(index,px,verified=False):
  return dict(id=f'current_inner_threaded_{index}',kind='inner_threaded',plane='main_face',
   px=px,observed=True,metric_homography_eligible=verified,
   canonical_xy_m=[.005,.005] if verified else None)
 update(a,raw([0,1,2],aux=[auxiliary(0,[330,250])]),0)
 # Contour order changes, but spatial association preserves the track.
 r=update(a,raw([0,1,2],aux=[auxiliary(7,[331,250])]),.5)
 assert len(r['face_landmarks'])==1
 assert r['face_landmarks'][0]['sample_count']==2
 # An unmatched, unverified candidate replaces the old guess rather than
 # growing a stationary cloud of remembered openings.
 r=update(a,raw([0,1,2],aux=[auxiliary(8,[340,250])]),.7)
 assert len(r['face_landmarks'])==1
 assert not r['face_landmarks'][0]['accumulated']
 assert r['accumulation']['point_count']==4
 r=update(a,raw([0,1,2]),.9)
 assert r['face_landmarks']==[]
 assert r['accumulation']['point_count']==3
 # A geometrically eligible anchor still survives a missed observation.
 update(a,raw([0,1,2],aux=[auxiliary(9,[330,250],True)]),1.1)
 r=update(a,raw([0,1,2]),1.3)
 assert len(r['face_landmarks'])==1
 assert r['face_landmarks'][0]['accumulated']
 assert r['face_landmarks'][0]['metric_homography_eligible']


def test_pin_endpoint_pool_uses_same_stationary_frame_and_preserves_flags():
 a=StationaryFaceAccumulator();warm(a)
 def pin(top,bottom):
  return dict(top_px=top,bottom_px=bottom,bottom_inferred=True,top_detected=True,
   visible=True,confidence=.3,automatic_nudge_eligible=False,top_observed=False)
 first=pin([320,100],[320,150]);r=a.accumulate_pin(first,int(.4e9))
 assert r['accumulation']==dict(stationary=True,sample_count=1,last_seen_age_s=0.,retained=False)
 assert r['confidence']==.3 and r['bottom_inferred'] and not r['automatic_nudge_eligible']
 update(a,raw([0,1,2]),.6)
 r=a.accumulate_pin(pin([324,104],[324,154]),int(.6e9))
 assert r['top_px']==[322,102] and r['bottom_px']==[322,152]
 assert r['pin_px']==r['bottom_px'] and r['accumulation']['sample_count']==2
 assert first['top_px']==[320,100]
 update(a,raw([0,1,2]),.8)
 r=a.accumulate_pin(dict(top_px=None,bottom_px=None,visible=True,top_detected=True),int(.8e9))
 assert r['top_px']==[322,102] and r['accumulation']['retained']
 assert abs(r['accumulation']['last_seen_age_s']-.2)<1e-8
 assert not r['visible'] and not r['top_detected'] and not r['automatic_nudge_eligible']
 assert r['bottom_inferred'] and not r['bottom_observed']


def test_pin_endpoint_pool_resets_ttl_and_rejects_unqualified_frames():
 a=StationaryFaceAccumulator();p=dict(top_px=[320,100],bottom_px=[320,150],bottom_inferred=True)
 r=a.accumulate_pin(p,0);assert not r['accumulation']['stationary']
 warm(a);a.accumulate_pin(p,int(.4e9),{'hold_s':.5})
 update(a,raw([0,1,2]),1.)
 r=a.accumulate_pin({},int(1e9),{'hold_s':.5})
 assert r['accumulation']['sample_count']==0 and not r['accumulation']['retained']
 assert r.get('bottom_px') is None
 a.accumulate_pin(p,int(1e9)+1)
 assert not a.pin_samples  # Image identity must match the face update exactly.
 update(a,raw([0,1,2]),1.2);a.accumulate_pin(p,int(1.2e9))
 r=a.accumulate_pin(p,int(1.2e9));assert not r['accumulation']['stationary']
 assert not a.pin_samples  # Repeated/nonincreasing pin frames cannot add votes.
 update(a,raw([0,1,2]),1.4);a.accumulate_pin(p,int(1.4e9))
 a.reset('test');assert not a.pin_samples
 warm(a);a.accumulate_pin(p,int(.4e9))
 q=np.zeros(14);q[0]=.01;update(a,raw([0,1,2]),.6,q)
 r=a.accumulate_pin(p,int(.6e9));assert not r['accumulation']['stationary']
 warm(a)
 r=a.accumulate_pin(p,int(.4e9),{'fixed_carrier_hold':False})
 assert not r['accumulation']['stationary'] and not a.pin_samples


def test_complete_registration_dropout_retains_face_and_pin_until_ttl():
 a=StationaryFaceAccumulator();c={'hold_s':1.}
 update(a,raw(range(4),True),0,c=c)
 r=update(a,raw(range(4),True),.5,c=c)
 a.accumulate_pin(dict(top_px=[320,100],bottom_px=[320,150],bottom_inferred=True),int(.5e9),c)
 original=np.array(r['holes_px'])
 # No registration, no visible holes, no homography and no pin estimate.
 for t in (.7,.9,1.2):
  r=update(a,raw(),t,c=c)
  p=a.accumulate_pin(dict(top_px=None,bottom_px=None),int(t*1e9),c)
  assert r['supported'] and r['accumulation']['stationary']
  assert np.allclose(r['holes_px'],original)
  assert all(h['accumulated'] and not h['observed'] for h in r['holes'])
  assert r['accumulation']['reset_reason']=='initial'
  assert p['bottom_px']==[320,150] and p['accumulation']['retained']
  assert not p['visible'] and not p['automatic_nudge_eligible']
 r=update(a,raw(),1.6,c=c)
 p=a.accumulate_pin({},int(1.6e9),c)
 assert not r['supported'] and all(x is None for x in r['holes_px'])
 assert r['accumulation']['reset_reason']=='ttl_expired'
 assert p['accumulation']['sample_count']==0 and p.get('bottom_px') is None


def test_registration_dropout_does_not_bypass_ineligible_pose_or_input_gap():
 for cause in ('stale','motion','gap'):
  a=StationaryFaceAccumulator();warm(a)
  a.accumulate_pin(dict(top_px=[320,100],bottom_px=[320,150]),int(.4e9))
  q=np.zeros(14)
  if cause=='motion':q[0]=.01
  t=3. if cause=='gap' else .6
  r=update(a,raw(),t,q=q,eligible=cause!='stale')
  p=a.accumulate_pin({},int(t*1e9))
  assert not r['supported'] and not r['accumulation']['stationary']
  assert not p['accumulation']['stationary'] and not a.pin_samples
  assert r['accumulation']['reset_reason'] in ('ineligible_frame','joint_motion','tool_motion','input_gap')
