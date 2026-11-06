"""Visible carrier silhouette evidence; never a precision CAD or pose constraint.

A TRAIN-only radial appearance template can replace the approximate circular
profile. Current image edges only correct supported visible arcs; hidden and
ambiguous arcs retain the projected template and are explicitly inferred.
"""
import json
from pathlib import Path
import cv2
import numpy as np

DEFAULT_TEMPLATE=Path(__file__).resolve().parents[1]/'work/left-carrier-opencv/edgefit/train-edge-template.json'


def _project(xy,H):
 p=np.c_[xy,np.ones(len(xy))]@np.asarray(H,float).T
 return p[:,:2]/p[:,2,None]


def _sample(image,p):
 return cv2.remap(image,np.asarray(p[:,0],np.float32)[None],np.asarray(p[:,1],np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]


def _edge_observations(gray,points,face_polygon=None,search_px=6):
 smooth=cv2.GaussianBlur(np.asarray(gray),(3,3),.7);gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
 tangent=np.roll(points,-1,0)-np.roll(points,1,0);normal=np.c_[-tangent[:,1],tangent[:,0]];normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-6)
 mask=None
 if face_polygon is not None:
  mask=np.zeros(gray.shape,np.uint8);cv2.fillPoly(mask,[np.rint(face_polygon).astype(np.int32)],255)
  # Conservative TRAIN polygons mark visible interior, never an exact contour.
  mask=cv2.dilate(mask,np.ones((11,11),np.uint8))
 offsets=np.arange(-search_px,search_px+.1,.5);scores=[]
 for offset in offsets:
  p=points+normal*offset;dx=_sample(gx,p);dy=_sample(gy,p);magnitude=np.hypot(dx,dy);alignment=np.abs(dx*normal[:,0]+dy*normal[:,1])/np.maximum(magnitude,1)
  score=np.where(alignment>.75,magnitude*alignment,0)
  if mask is not None:score=np.where(_sample(mask,p)>250,score,0)
  scores.append(score)
 scores=np.array(scores);best=np.argmax(scores,axis=0);strength=scores[best,np.arange(len(points))];shift=offsets[best]
 # Distinct separated competing edges are ambiguous (jaw/face/boss boundaries).
 competing=scores.copy()
 for j,b in enumerate(best):competing[max(0,b-4):min(len(offsets),b+5),j]=0
 second=competing.max(0)
 valid=(strength>85)&(strength>second*1.25)&(abs(shift)<search_px-.5)
 # Require a coherent arc; isolated glints and abrupt shifts never bend outline.
 coherent=np.zeros(len(points),bool)
 for i in range(len(points)):
  ids=np.mod(np.arange(i-2,i+3),len(points));coherent[i]=valid[ids].sum()>=4 and np.ptp(shift[ids][valid[ids]])<3
 valid &=coherent
 corrected=points.copy();corrected[valid]+=normal[valid]*shift[valid,None]
 return corrected,valid,strength,shift


def refine_face_outline(gray,H,face_polygon=None,template=None,search_px=6):
 """Return composite outline and observed flags; optional low-weight edge evidence.

 H maps image-cyclic face XY metres to native image pixels. Edge observations
 have weight .05 and are not sufficient to authorize an H or motor update.
 """
 if template is None and DEFAULT_TEMPLATE.exists():template=json.loads(DEFAULT_TEMPLATE.read_text())
 n=144;theta=np.linspace(0,2*np.pi,n,endpoint=False);r=np.full(n,.02649)
 source='approximate_photo_profile_circle'
 if template is not None:
  tr=np.asarray(template['radial_distance_m'],float);r=np.interp(theta,np.linspace(0,2*np.pi,len(tr),endpoint=False),tr,period=2*np.pi);source=template.get('source','TRAIN_visible_edge_template')
 xy=np.c_[np.cos(theta),np.sin(theta)]*r[:,None];projected=_project(xy,H)
 corrected,valid,strength,shift=_edge_observations(gray,projected,face_polygon,search_px)
 if template is not None and 'support_per_angle' in template:
  support=np.asarray(template['support_per_angle']);supported=np.interp(theta,np.linspace(0,2*np.pi,len(support),endpoint=False),support,period=2*np.pi)>.5
  valid &=supported;corrected[~valid]=projected[~valid]
 observations=[dict(index=int(i),point_px=corrected[i].tolist(),projected_px=projected[i].tolist(),observed=bool(valid[i]),inferred=not bool(valid[i]),gradient_strength=float(strength[i]),normal_correction_px=float(shift[i]) if valid[i] else 0.,secondary_fit_weight=.05 if valid[i] else 0.,metric_homography_eligible=False) for i in range(n)]
 return dict(face_outline_px=corrected.tolist(),projected_face_outline_px=projected.tolist(),edge_observations=observations,shape_template_source=source,observed_arc_fraction=float(valid.mean()),visible_edge_mean_correction_px=float(np.mean(abs(shift[valid]))) if valid.any() else None,edge_fit_is_pose_constraint=False,limitations='Visible appearance edge only; boss height/scallops and own-jaw occlusion prevent exact coplanar silhouette geometry.')


def build_train_edge_template(reference_bank,out_path=DEFAULT_TEMPLATE):
 """Aggregate independently visible TRAIN arcs; never use evaluation images."""
 from app.left_carrier_opencv import CANONICAL,cyclic_layout
 from scripts.prepare_left_carrier_opencv import refine_bore
 bank=json.loads(Path(reference_bank).read_text());theta=np.linspace(0,2*np.pi,144,endpoint=False);unit=np.c_[np.cos(theta),np.sin(theta)];measurements=[];sources=[]
 for f in bank.get('frames',bank.get('references',[])):
  if f.get('split')!='train':raise ValueError('TRAIN references required')
  if not f.get('carrier_polygon_px'):continue
  gray=cv2.imread(f['image_path'],0);raw=np.array(f['unordered_visible_bores_px'],float);points=np.array([(refine_bore(gray,p,8.5,max_shift=6) or {'center':p})['center'] for p in raw]);full=cyclic_layout(points);H=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),full.astype(np.float32));projected=_project(unit*.02649,H)
  corrected,valid,strength,shift=_edge_observations(gray,projected,f['carrier_polygon_px'],7)
  canonical=_project(corrected,np.linalg.inv(H));radial=np.linalg.norm(canonical,axis=1);radial[~valid]=np.nan;measurements.append(radial);sources.append(dict(id=f['id'],image_path=f['image_path'],observed_arc_fraction=float(valid.mean())))
 if not measurements:raise ValueError('No visible-face TRAIN masks')
 values=np.array(measurements);support=np.isfinite(values).sum(0);r=np.full(144,.02649)
 # Retain a shape change only with agreement in at least two TRAIN views.
 for i in range(144):
  v=values[:,i];v=v[np.isfinite(v)]
  if len(v)>=2 and np.ptp(v)<.003:r[i]=np.median(v)
 # Mild local smoothing prevents learned edge noise from becoming a scallop.
 r=(np.roll(r,1)+2*r+np.roll(r,-1))/4
 result=dict(source='TRAIN_only_real_visible_radial_silhouette_v1',radial_distance_m=r.tolist(),support_per_angle=support.tolist(),references=sources,source_bank=str(Path(reference_bank).resolve()),limitations=['Appearance template, not manufacturing CAD.','Unsupported arcs retain26.49mm approximate profile.','Raised bosses and hidden contour are not metric planar constraints.'])
 out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));return result


def build_manual_edge_template(edge_seeds,reference_bank,out_path=DEFAULT_TEMPLATE):
 """Map directly reviewed TRAIN perimeter arcs to an appearance template."""
 import hashlib
 from app.left_carrier_opencv import CANONICAL,cyclic_layout
 from scripts.prepare_left_carrier_opencv import refine_bore
 bank=json.loads(Path(reference_bank).read_text());rows=bank.get('frames',bank.get('references',[]));by_index={r['source_frame_index']:r for r in rows};seeds=json.loads(Path(edge_seeds).read_text());measurements=[];sources=[]
 for f in seeds['frames']:
  if f.get('split')!='train':raise ValueError('Manual edge template requires TRAIN labels')
  row=by_index[f['source_frame_index']];gray=cv2.imread(row['image_path'],0);raw=np.array(row['unordered_visible_bores_px'],float);points=np.array([(refine_bore(gray,p,8.5,max_shift=6) or {'center':p})['center'] for p in raw]);full=cyclic_layout(points);H=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),full.astype(np.float32));r=np.full(144,np.nan);bins={}
  for arc in f['visible_edge_arcs_px']:
   arc=np.asarray(arc,float)
   for start,end in zip(arc[:-1],arc[1:]):
    dense=start+(end-start)*np.linspace(0,1,max(2,int(np.linalg.norm(end-start)*2)))[:,None];xy=_project(dense,np.linalg.inv(H));angles=np.mod(np.arctan2(xy[:,1],xy[:,0]),2*np.pi);radii=np.linalg.norm(xy,axis=1)
    for i,value in zip(np.rint(angles/(2*np.pi)*144).astype(int)%144,radii):
     if .019<value<.032:bins.setdefault(int(i),[]).append(float(value))
  for i,values in bins.items():r[i]=np.median(values)
  measurements.append(r);sources.append(dict(id=f.get('id',row['id']),image_path=row['image_path'],source_frame_index=f['source_frame_index'],manual_observed_arc_fraction=float(np.isfinite(r).mean())))
 values=np.array(measurements);support=np.isfinite(values).sum(0);radial=np.full(144,.02649)
 for i in range(144):
  v=values[:,i];v=v[np.isfinite(v)]
  if len(v):radial[i]=np.median(v)
 # Only smooth supported samples: unknown arcs retain an honest approximate prior.
 smoothed=radial.copy()
 for i in range(144):
  ids=np.mod([i-1,i,i+1],144)
  if np.all(support[ids]>0):smoothed[i]=np.dot(radial[ids],[.25,.5,.25])
 result=dict(source='TRAIN_direct_reviewed_visible_perimeter_arcs_v1',radial_distance_m=smoothed.tolist(),support_per_angle=support.tolist(),references=sources,source_bank=str(Path(reference_bank).resolve()),source_bank_sha256=hashlib.sha256(Path(reference_bank).read_bytes()).hexdigest(),manual_edge_source=str(Path(edge_seeds).resolve()),manual_edge_sha256=hashlib.sha256(Path(edge_seeds).read_bytes()).hexdigest(),limitations=['Direct RGB AI-reviewed TRAIN edge arcs, not human-certified silhouette.','Mapping depends on approximate bore homography; appearance shape, not metric CAD.','Unsupported arcs retain26.49mm approximate profile.','Raised bosses and own-jaw-hidden edges cannot constrain planar face pose.'])
 out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));return result
