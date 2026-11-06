#!/usr/bin/env python3
"""Real LEFT visible-bore adaptation with unordered matching and simulated replay.

Real labels are reviewed OpenCV tracks, not FK. Unlabeled real points are UNKNOWN.
B2 selects weights; B4/B5 are read only after selection. No hardware interfaces.
"""
import os,sys,json,argparse,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from app.carrier_face_legacy_cnn import LegacyLeftCarrierCNN,targeted_occlusion,VARIANT
from app.carrier_face_augmentation import augment_batch,AUGMENTATION_CONFIG
from app.carrier_face_cnn import carrier_face_loss

def atomic(payload,path):
 temp=path.with_name(path.name+'.tmp');torch.save(payload,temp);os.replace(temp,path)
def load_rows(path,training=False):
 data=json.loads(Path(path).read_text());rows=data['frames'];result=[]
 for row in rows:
  if training and not(row.get('training_eligible') and row['split']=='train'):continue
  if row.get('camera_source')!='wrist_port3':raise ValueError('Only actual LEFT wrist_port3 accepted')
  raw=Path(row['image_path']).read_bytes()
  if row.get('image_sha256') and hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Image hash changed')
  im=Image.open(row['image_path']).convert('RGB');w,h=im.size;points=np.asarray(row['unordered_visible_bores_px'],np.float32)
  if not 1<=len(points)<=4 or not np.isfinite(points).all():raise ValueError('One to four actually visible points required')
  result.append(dict(row=row,image=np.asarray(im.resize((224,224),Image.Resampling.BILINEAR)),points=points/[w,h]))
 return result

def masked_loss(output,points,visible,known):
 total,_=carrier_face_loss(output,points,visible);total-=F.binary_cross_entropy_with_logits(output['visibility_logits'],visible.float());raw=F.binary_cross_entropy_with_logits(output['visibility_logits'],visible.float(),reduction='none');pos=(visible&known).float();neg=(~visible&known).float();pc=pos.sum(0);nc=neg.sum(0);classes=(pc>0).float()+(nc>0).float();balanced=((raw*pos).sum(0)/pc.clamp_min(1)+(raw*neg).sum(0)/nc.clamp_min(1))/classes.clamp_min(1);return total+balanced[classes>0].mean()

def assign(pred,items):
 target=torch.zeros_like(pred);known=torch.zeros(pred.shape[:2],device=pred.device,dtype=torch.bool)
 for i,item in enumerate(items):
  pts=item['points'];cost=np.linalg.norm((pred[i,1:5].detach().cpu().numpy()[:,None]-pts[None])*[640,480],axis=-1);slots,labels=linear_sum_assignment(cost)
  target[i,torch.tensor(slots+1,device=pred.device)]=torch.tensor(pts[labels],device=pred.device,dtype=pred.dtype);known[i,torch.tensor(slots+1,device=pred.device)]=True
 return target,known

def real_occlusion(images,points,visible,generator):
 x=images.clone();v=visible.clone();b,_,h,w=x.shape;weights=v[:,:9].float();has=weights.sum(1)>0;weights[~has,0]=1;slot=torch.multinomial(weights,1,generator=generator).squeeze(1);center=points[torch.arange(b,device=x.device),slot];extent=.025+.07*torch.rand(b,2,device=x.device,generator=generator);lo=center-extent/2;hi=center+extent/2;enabled=(torch.rand(b,device=x.device,generator=generator)<.65)&has;yy,xx=torch.meshgrid((torch.arange(h,device=x.device)+.5)/h,(torch.arange(w,device=x.device)+.5)/w,indexing='ij');cover=(xx[None]>=lo[:,0,None,None])&(xx[None]<=hi[:,0,None,None])&(yy[None]>=lo[:,1,None,None])&(yy[None]<=hi[:,1,None,None])&enabled[:,None,None];fill=.03+.65*torch.rand(b,3,1,1,device=x.device,generator=generator);x=torch.where(cover[:,None],fill,x);hit=(points[:,:,0]>=lo[:,0,None])&(points[:,:,0]<=hi[:,0,None])&(points[:,:,1]>=lo[:,1,None])&(points[:,:,1]<=hi[:,1,None])&enabled[:,None];return x,points,v&~hit

def train(args):
 torch.set_num_threads(4);torch.manual_seed(177);rng=np.random.default_rng(177);device='cuda' if torch.cuda.is_available() else 'cpu';gen=torch.Generator(device=device).manual_seed(5511);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 train_rows=load_rows(args.train_manifest,True);eval_rows=load_rows(args.evaluation_manifest);val=[r for r in eval_rows if r['row']['split']=='validation'];test=[r for r in eval_rows if r['row']['split']=='test'];assert train_rows and val and test;train_groups={r['row']['group_id'] for r in train_rows};val_groups={r['row']['group_id'] for r in val};test_groups={r['row']['group_id'] for r in test};assert not(train_groups&val_groups or train_groups&test_groups or val_groups&test_groups)
 sim=np.load(args.sim_dataset);sim_ids=np.flatnonzero(sim['splits']=='train');sim_images=torch.tensor(sim['images'].transpose(0,3,1,2));sim_points=torch.tensor(sim['keypoints'],device=device);sim_visible=torch.tensor(sim['visible'],device=device)
 model=LegacyLeftCarrierCNN().to(device);payload=torch.load(args.source,map_location=device,weights_only=True);model.load_state_dict(payload['model']);backbone=[];decoder=[];heads=[]
 for name,param in model.named_parameters():(backbone if name.startswith(('stem.','encoder_')) else decoder if name.startswith(('decoder_','cross_view.')) else heads).append(param)
 optimizer=torch.optim.AdamW([{'params':backbone,'lr':.00001},{'params':decoder,'lr':.00005},{'params':heads,'lr':.0003}],weight_decay=.0001)
 def batch_image(items):return torch.tensor(np.stack([r['image'] for r in items]).transpose(0,3,1,2),device=device,dtype=torch.float32)/255
 def evaluate(items):
  model.eval();errors=[];visible_errors=[];preds=[];confidence=[];matched=0;visible_matched=0
  with torch.inference_mode():
   for chunk in np.array_split(np.arange(len(items)),max(1,int(np.ceil(len(items)/24)))):
    group=[items[j] for j in chunk];result=model(batch_image(group));p=result['keypoints'].cpu().numpy();c=result['visibility_logits'].sigmoid().cpu().numpy();preds.extend(p.tolist());confidence.extend(c.tolist())
    for i,row in enumerate(group):
     cost=np.linalg.norm((p[i,1:5,None]-row['points'][None])*[640,480],axis=-1);slots,labels=linear_sum_assignment(cost);err=cost[slots,labels];errors.extend(err.tolist());matched+=len(err);visible_matched+=int((c[i,slots+1]>=.5).sum());visible_errors.extend(err[c[i,slots+1]>=.5].tolist())
  return dict(mean_native640x480_px=float(np.mean(errors)),median_native640x480_px=float(np.median(errors)),p90_native640x480_px=float(np.quantile(errors,.9)),visible_annotated_bores=matched,predicted_visible_matched=visible_matched,matched_visibility_coverage=visible_matched/max(1,matched),visible_matched_mean_px=float(np.mean(visible_errors)) if visible_errors else None,frame_count=len(items),keypoints=preds,confidence=confidence)
 initial=evaluate(val);best=float('inf');history=[];start=time.time();presentations=0
 for epoch in range(args.epochs):
  model.train();losses=[]
  for indices in np.array_split(rng.permutation(len(train_rows)),int(np.ceil(len(train_rows)/24))):
   items=[train_rows[j] for j in indices];real_image=batch_image(items)
   with torch.no_grad():base=model(real_image);rt,rk=assign(base['keypoints'],items)
   real_image,rt,rv=augment_batch(real_image,rt,rk,generator=gen);real_image,rt,rv=real_occlusion(real_image,rt,rv,gen);si=rng.choice(sim_ids,len(items),replace=True);sx=sim_images[si].to(device=device,dtype=torch.float32)/255;st=sim_points[si];sv=sim_visible[si];sx,st,sv=augment_batch(sx,st,sv,generator=gen);sx,st,sv=targeted_occlusion(sx,st,sv,generator=gen);all_images=torch.cat((real_image,sx));optimizer.zero_grad(set_to_none=True);prediction=model(all_images,return_logits=True);ro={k:v[:len(items)] for k,v in prediction.items()};so={k:v[len(items):] for k,v in prediction.items()};loss=3*masked_loss(ro,rt,rv,rk)+.5*masked_loss(so,st,sv,torch.ones_like(sv));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()));presentations+=len(items)
  validation=evaluate(val);score=validation['mean_native640x480_px'];history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_mean_native_px=score));print(history[-1],flush=True)
  if score<best:
   best=score;metadata=dict(payload['metadata']);metadata.update(schema='carrier-face-left-rgb-v1',model_variant=VARIANT,camera='left_wrist',image_size=224,real_adapted=True,real_training_source=str(Path(args.train_manifest).resolve()),real_training_sha256=hashlib.sha256(Path(args.train_manifest).read_bytes()).hexdigest(),validation_source=str(Path(args.evaluation_manifest).resolve()),validation_selection='B2 manually reviewed visible unordered bores; repurposed at real-adaptation stage',selected_real_epoch=epoch+1,visibility_threshold=.5,visibility_threshold_floor=.5,visibility_thresholds=[.5]*9+[1.000001],visibility_calibration='Fixed class-balanced binary decision .5; real confidence/occlusion NOT calibrated',source_checkpoint_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),real_unlabeled_visibility='UNKNOWN; never negative',real_location_loss='Hungarian match actual visible bore coordinates only; no FK real labels',augmentation=AUGMENTATION_CONFIG,real_encoder_lr=.00001,decoder_lr=.00005,head_lr=.0003,real_weight=3.,sim_replay_weight=.5,real_perception_validated=False);metadata.pop('visibility_calibration',None);metadata['visibility_policy']='Fixed .5 floor, uncalibrated on real occlusion';atomic(dict(model=model.state_dict(),metadata=metadata),out/'best.pt')
  model.train()
 selected=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(selected['model']);final_val=evaluate(val);final_test=evaluate(test);report=dict(initial_validation={k:v for k,v in initial.items() if k not in ('keypoints','confidence')},validation={k:v for k,v in final_val.items() if k not in ('keypoints','confidence')},test={k:v for k,v in final_test.items() if k not in ('keypoints','confidence')},selected_epoch=selected['metadata']['selected_real_epoch'],epochs=args.epochs,seconds=time.time()-start,real_train_frames=len(train_rows),real_bore_labels=sum(len(r['points']) for r in train_rows),real_augmented_presentations=presentations,sim_replay_presentations=presentations,train_groups=sorted(train_groups),validation_groups=sorted(val_groups),test_groups=sorted(test_groups),test_used_for_selection=False,annotation='AI-reviewed visible bore seeds plus conservative OpenCV tracks; uncertainty~5-6nativepixels; nothuman-certified',real_unlabeled_visibility='UNKNOWN',source_manifest_sha256=selected['metadata']['real_training_sha256'],checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest());(out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'history.json').write_text(json.dumps(history,indent=2));(out/'evaluation-predictions.json').write_text(json.dumps(dict(validation=final_val,test=final_test),indent=2));print(json.dumps(report,indent=2),flush=True)
 # One panel per evaluation frame: true reviewed visible bore labels green; raw slots red.
 all_eval=val+test;all_prediction=final_val['keypoints']+final_test['keypoints'];sheet=Image.new('RGB',(640*3,505*int(np.ceil(len(all_eval)/3))))
 for i,item in enumerate(all_eval):
  im=Image.open(item['row']['image_path']).convert('RGB');draw=ImageDraw.Draw(im)
  for point in item['points']*[640,480]:draw.ellipse((point[0]-5,point[1]-5,point[0]+5,point[1]+5),outline='lime',width=2)
  for point in np.asarray(all_prediction[i])[1:5]*[640,480]:draw.line((point[0]-5,point[1],point[0]+5,point[1]),fill='red',width=2);draw.line((point[0],point[1]-5,point[0],point[1]+5),fill='red',width=2)
  x=(i%3)*640;y=(i//3)*505;sheet.paste(im,(x,y+25));ImageDraw.Draw(sheet).text((x+5,y+5),item['row']['split']+' '+item['row']['id'],fill='white')
 sheet.save(out/'real-evaluation.jpg')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--train-manifest',default='work/left-carrier-opencv/manifest.json');p.add_argument('--evaluation-manifest',default='work/left-carrier-opencv/evaluation-manifest.json');p.add_argument('--sim-dataset',default='work/left-carrier-cnn/dataset.npz');p.add_argument('--source',default='work/left-carrier-cnn/legacy/best.pt');p.add_argument('--output',default='work/left-carrier-cnn/real-adapted');p.add_argument('--epochs',type=int,default=100);train(p.parse_args())
