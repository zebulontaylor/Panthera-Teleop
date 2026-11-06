#!/usr/bin/env python3
"""Train actual RGB-only visible-landmark CNN and report disjoint heldout seeds."""
import sys,json,time,argparse,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from app.carrier_face_cnn import CarrierFaceCNN,carrier_face_loss,SCHEMA,LANDMARK_NAMES,fit_face_geometry,project_face

def atomic_checkpoint(payload,path):
 temporary=path.with_name(path.name+".tmp");torch.save(payload,temporary);os.replace(temporary,path)

def train(directory,epochs=100,augment=False,resume=None,legacy=False):
 out=Path(directory);a=np.load(out/'dataset.npz');torch.set_num_threads(4);torch.manual_seed(73);rng=np.random.default_rng(73)
 device='cuda' if torch.cuda.is_available() else 'cpu';n=len(a['images'])
 if 'splits' not in a:raise ValueError('Measured source-group split required; random-seed split rejected')
 tr=np.flatnonzero(a['splits']=='train');va=np.flatnonzero(a['splits']=='validation');te=np.flatnonzero(a['splits']=='test')
 groups=[set(a['groups'][idx]) for idx in (tr,va,te)]
 assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]),'source group leakage'
 np.savez(out/'split.npz',train_seeds=a['seeds'][tr],validation_seeds=a['seeds'][va],test_seeds=a['seeds'][te],train_groups=np.array(sorted(groups[0])),validation_groups=np.array(sorted(groups[1])),test_groups=np.array(sorted(groups[2])))
 x=torch.tensor(a['images'].transpose(0,3,1,2),dtype=torch.uint8)
 def inputs(batch):return x[batch].to(device=device,dtype=torch.float32)/255
 y=torch.tensor(a['keypoints'],device=device);v=torch.tensor(a['visible'],device=device)
 provenance={}
 if legacy:
  from app.carrier_face_legacy_cnn import LegacyLeftCarrierCNN,balanced_visibility_loss,targeted_occlusion,VARIANT
  model=LegacyLeftCarrierCNN().to(device);provenance=model.load_original_adapted()
 else:model=CarrierFaceCNN().to(device)
 def loss_for(output,target,mask):
  total,metrics=carrier_face_loss(output,target,mask)
  if legacy:total=total-torch.nn.functional.binary_cross_entropy_with_logits(output['visibility_logits'],mask.float())+balanced_visibility_loss(output['visibility_logits'],mask)
  return total,metrics
 if resume:model.load_state_dict(torch.load(resume,map_location=device,weights_only=True)['model'])
 augmentation_rng=torch.Generator(device=device).manual_seed(4401)
 if augment:
  from app.carrier_face_augmentation import augment_batch
 if legacy:
  backbone=[];decoder=[];heads=[]
  for name,param in model.named_parameters():
   (backbone if name.startswith(('stem.','encoder_')) else decoder if name.startswith(('decoder_','cross_view.')) else heads).append(param)
  opt=torch.optim.AdamW([{'params':backbone,'lr':.0001},{'params':decoder,'lr':.0003},{'params':heads,'lr':.001}],weight_decay=.0001)
 else:opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
 best=float('inf');history=[];start=time.time()
 def evaluate(indices):
  model.eval();ls=[];pred=[];conf=[]
  with torch.inference_mode():
   for batch in np.array_split(indices,max(1,int(np.ceil(len(indices)/64)))):
    p=model(inputs(batch),return_logits=True);loss,_=loss_for(p,y[batch],v[batch]);ls.append(float(loss));pred.append(p['keypoints'].cpu().numpy());conf.append(p['visibility_logits'].sigmoid().cpu().numpy())
  return float(np.mean(ls)),np.concatenate(pred),np.concatenate(conf)
 for epoch in range(epochs):
  model.train();ls=[]
  for batch in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/48))):
   opt.zero_grad(set_to_none=True);image=inputs(batch);target=y[batch];mask=v[batch]
   if augment:image,target,mask=augment_batch(image,target,mask,generator=augmentation_rng)
   if legacy:image,target,mask=targeted_occlusion(image,target,mask,generator=augmentation_rng)
   p=model(image,return_logits=True);loss,_=loss_for(p,target,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append(float(loss.detach()))
  val,_,_=evaluate(va);history.append(dict(epoch=epoch+1,train=float(np.mean(ls)),validation=val));print(history[-1],flush=True)
  if val<best:
   best=val;atomic_checkpoint(dict(model=model.state_dict(),metadata=dict(schema=SCHEMA,model_variant=VARIANT if legacy else 'small-left-rgb-v1',legacy_provenance=provenance,local_visibility=legacy,balanced_visibility=legacy,targeted_occlusion=legacy,camera='left_wrist',image_size=int(x.shape[-1]),visibility_threshold=.6,synthetic_only=True,real_calibrated=False,stochastic_train_augmentation=augment,augmentation_seed=4401,warm_start=str(resume) if resume else None,training_seed=73,epochs=epoch+1,landmark_names=list(LANDMARK_NAMES),optics='constant unchanged left_wrist native optics; measured-state replay, calibration hypothesis')),out/'best.pt')
 payload=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(payload['model']);val_loss,val_p,val_c=evaluate(va);val_vis=a['visible'][va];thresholds=np.arange(.35,.81,.025);f1=[]
 for threshold in thresholds:
  yes=val_c>=threshold;tp=(yes&val_vis).sum();f1.append(float(2*tp/max(1,yes.sum()+val_vis.sum())))
 threshold=float(thresholds[int(np.argmax(f1))]);payload['metadata']['visibility_threshold']=threshold;payload['metadata']['threshold_source']='validation visibility F1';payload['metadata']['dataset_sha256']=hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest();payload['metadata']['dataset_provenance']=json.loads((out/'dataset.json').read_text());atomic_checkpoint(payload,out/'best.pt')
 loss,p,c=evaluate(te);truth=a['keypoints'][te];vis=a['visible'][te];size=x.shape[-1];err=np.linalg.norm(p-truth,axis=-1)*size;observed=c>=threshold
 report=dict(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,seconds=time.time()-start,epochs=epochs,best_epoch=payload['metadata']['epochs'],test_loss=loss,split=dict(train=len(tr),validation=len(va),test=len(te),seed_disjoint=True,source_groups_disjoint=True,train_groups=len(groups[0]),validation_groups=len(groups[1]),test_groups=len(groups[2]),newdemo_same_recording_grasp=True),visible_mean_px=float(err[vis].mean()),visible_median_px=float(np.median(err[vis])),visible_p90_px=float(np.quantile(err[vis],.9)),visibility_precision=float((observed&vis).sum()/max(1,observed.sum())),visibility_recall=float((observed&vis).sum()/max(1,vis.sum())),per_landmark={name:dict(visible_count=int(vis[:,k].sum()),mean_px=float(err[:,k][vis[:,k]].mean()) if vis[:,k].any() else None) for k,name in enumerate(LANDMARK_NAMES)},real_validation=False,visibility_threshold=threshold,threshold_source='validation F1 only',visible_mean_native640x480_px=float(np.linalg.norm((p-truth)*[640,480],axis=-1)[vis].mean()))
 # Geometry extrapolation assessed ONLY on heldout hidden labels, never in training loss.
 inferred=[];inferred_native=[];support=0
 for i in range(len(te)):
  geo=fit_face_geometry(p[i,:9],c[i,:9],threshold=threshold);support+=geo is not None
  if geo:
   for k in range(1,5):
    if not vis[i,k]:
     delta=np.array(geo['projected'][k])-truth[i,k];inferred.append(float(np.linalg.norm(delta)*size));inferred_native.append(float(np.linalg.norm(delta*[640,480])))
 for k,name in enumerate(LANDMARK_NAMES):
  report['per_landmark'][name]['mean_native640x480_px']=float(np.linalg.norm((p[:,k]-truth[:,k])*[640,480],axis=-1)[vis[:,k]].mean()) if vis[:,k].any() else None
 report['confident_visible_mean_native640x480_px']=float(np.linalg.norm((p-truth)*[640,480],axis=-1)[vis&observed].mean()) if (vis&observed).any() else None
 report['geometry']=dict(supported_frames=support,total_frames=len(te),hidden_bore_count=len(inferred),hidden_bore_mean_px=float(np.mean(inferred)) if inferred else None,hidden_bore_mean_native640x480_px=float(np.mean(inferred_native)) if inferred_native else None)
 (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'history.json').write_text(json.dumps(history,indent=2));np.savez(out/'heldout_predictions.npz',indices=te,keypoints=p,confidence=c)
 from PIL import Image,ImageDraw
 sheet=Image.new('RGB',(size*6,size*3))
 for j,idx in enumerate(te[:18]):
  im=Image.fromarray(x[idx].permute(1,2,0).numpy());draw=ImageDraw.Draw(im)
  for k in range(10):
   if vis[j,k]:
    xx,yy=truth[j,k]*size;draw.ellipse((xx-2,yy-2,xx+2,yy+2),outline='lime')
   if observed[j,k]:
    xx,yy=p[j,k]*size;draw.line((xx-3,yy,xx+3,yy),fill='red');draw.line((xx,yy-3,xx,yy+3),fill='red')
  sheet.paste(im,((j%6)*size,(j//6)*size))
 sheet.save(out/'heldout.jpg');print(json.dumps(report,indent=2),flush=True)
def finalize_declared_heldout_mask(directory):
 out=Path(directory);a=np.load(out/'dataset.npz');pred=np.load(out/'heldout_predictions.npz');report=json.loads((out/'metrics.json').read_text());indices=pred['indices'];p=pred['keypoints'];confidence=pred['confidence'];truth=a['keypoints'][indices];visible=a['visible'][indices].copy();eligible=np.ones_like(visible)
 manifest=json.loads((ROOT/'work/pin-insertion-real/manifest.json').read_text());gear_groups=sorted(set(f['episode_id'] for f in manifest['frames'] if 'Gear-like held object' in f.get('note','')))
 gear=np.isin(a['groups'][indices],gear_groups);eligible[gear,9]=False;visible &=eligible;observed=(confidence>=report['visibility_threshold'])&eligible;err=np.linalg.norm(p-truth,axis=-1)*a['images'].shape[1];native=np.linalg.norm((p-truth)*[640,480],axis=-1)
 train_ids=np.flatnonzero(a['splits']=='train');train_y=a['keypoints'][train_ids];train_v=a['visible'][train_ids];baseline=np.stack([train_y[:,k][train_v[:,k]].mean(0) for k in range(10)]);bore_visible=visible[:,1:5];permutation=np.random.default_rng(9041).permutation(len(indices));shuffled=np.linalg.norm((p[permutation,1:5]-truth[:,1:5])*[640,480],axis=-1);constant=np.linalg.norm((baseline[None,1:5]-truth[:,1:5])*[640,480],axis=-1)
 report['carrier_bores']={'visible_count':int(bore_visible.sum()),'mean_native640x480_px':float(native[:,1:5][bore_visible].mean()),'train_constant_mean_baseline_native640x480_px':float(constant[bore_visible].mean()),'shuffled_image_error_native640x480_px':float(shuffled[bore_visible].mean()),'shuffle_seed':9041,'comparison_used_for_selection':False}
 supported=0;inferred_errors=[];hidden_total=int((~a['visible'][indices,1:5]).sum())
 for i in range(len(indices)):
  geometry=fit_face_geometry(p[i,:9],confidence[i,:9],threshold=report['visibility_threshold']);supported+=geometry is not None
  if geometry:
   for k in range(1,5):
    if not visible[i,k] and confidence[i,k]<report['visibility_threshold']:inferred_errors.append(float(np.linalg.norm((np.asarray(geometry['projected'][k])-truth[i,k])*[640,480])))
 report['geometry']={'supported_frames':supported,'total_frames':len(indices),'support_coverage':supported/len(indices),'true_hidden_bores':hidden_total,'runtime_inferred_true_hidden_bores':len(inferred_errors),'hidden_inference_coverage':len(inferred_errors)/max(1,hidden_total),'hidden_bore_mean_native640x480_px':float(np.mean(inferred_errors)) if inferred_errors else None,'definition':'Only true-hidden bores below model visibility threshold; withheld landmark excluded from homography support.'}
 report['pin_metric_exclusion']={'declared_before_final_evaluation':True,'reason':'Real recording held gear, not bare pin; synthetic pin prior is not pin-insertion replay','groups':gear_groups,'frames':int(gear.sum()),'carrier_metrics_retained':True}
 report['synthetic_all_channels_test_loss']=report.pop('test_loss',report.get('synthetic_all_channels_test_loss'))
 report.update(visible_mean_px=float(err[visible].mean()),visible_median_px=float(np.median(err[visible])),visible_p90_px=float(np.quantile(err[visible],.9)),visible_mean_native640x480_px=float(native[visible].mean()),visibility_precision=float((observed&visible).sum()/max(1,observed.sum())),visibility_recall=float((observed&visible).sum()/max(1,visible.sum())),confident_visible_mean_native640x480_px=float(native[visible&observed].mean()))
 for k,name in enumerate(LANDMARK_NAMES):report['per_landmark'][name]=dict(visible_count=int(visible[:,k].sum()),mean_px=float(err[:,k][visible[:,k]].mean()) if visible[:,k].any() else None,mean_native640x480_px=float(native[:,k][visible[:,k]].mean()) if visible[:,k].any() else None)
 (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'heldout_metric_mask.npz',indices=indices,eligible=eligible,visible=visible)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',default='work/left-carrier-cnn');p.add_argument('--epochs',type=int,default=100);p.add_argument('--augment',action='store_true');p.add_argument('--legacy',action='store_true');p.add_argument('--resume');a=p.parse_args();train(a.directory,a.epochs,a.augment,a.resume,a.legacy);finalize_declared_heldout_mask(a.directory)
