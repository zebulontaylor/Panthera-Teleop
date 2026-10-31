#!/usr/bin/env python3
"""Offline-only visual grasp candidate; no robot imports or runtime writes.

A frozen existing image encoder plus fixed L2 multinomial classifier uses only
physical-left wrist_port3 jaw pixels. The four validation episodes never select
features, parameters or thresholds. Labels are AI visual annotations, not truth.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('empty', 'gear', 'pin')
CROP = (240, 260, 440, 410)  # Native jaw region, fixed before training/evaluation.
THRESHOLD = .85
L2 = .01


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return [max(0., center-radius), min(1., center+radius)]


def load_rows(path, verify=True):
    data = json.loads(Path(path).read_text())
    rows, excluded, episodes, hashes, seen = [], [], {}, {}, set()
    for source in data['label_provenance']:
        if verify and sha256(source['path']) != source['sha256']:
            raise ValueError('Label provenance hash mismatch')
    for row in data['observations']:
        ep, split, oid = row['episode_id'], row['split'], row['observation_id']
        if split not in {'train', 'validation'} or (ep in episodes and episodes[ep] != split):
            raise ValueError('Episode split leakage or invalid split')
        if oid in seen:
            raise ValueError('Duplicate observation')
        seen.add(oid); episodes[ep] = split
        for camera, image in row['images'].items():
            h = image['sha256']
            if verify and sha256(image['path']) != h:
                raise ValueError('Image source hash mismatch')
            if h in hashes and hashes[h] != split:
                raise ValueError('Duplicate image across splits')
            hashes[h] = split
        label = row['labels']['grasp']['left']
        if label['object'] not in CLASSES or label['confidence'] not in {'high','medium'}:
            excluded.append({'observation_id':oid,'label':label}); continue
        rows.append(dict(row, target=CLASSES.index(label['object'])))
    if Counter(episodes.values()) != {'train':13, 'validation':4}:
        raise ValueError('Expected original 13 train / 4 validation episode split')
    return rows, excluded, episodes


def fit_classifier(x, y):
    """Fixed regularization; fit scaling and weights on supplied train rows only."""
    import torch
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-6] = 1
    xt = torch.tensor((x-mean)/scale, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    w = torch.zeros((x.shape[1],3), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w,b], max_iter=100, line_search_fn='strong_wolfe',
                          tolerance_grad=1e-8, tolerance_change=1e-10)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt@w+b, yt) + L2*w.square().sum()/2
        loss.backward(); return loss
    opt.step(closure)
    return {'mean':mean, 'scale':scale, 'weights':w.detach().numpy(), 'bias':b.detach().numpy()}


def predict(model,x):
    logits = ((x-model['mean'])/model['scale'])@model['weights']+model['bias']
    logits -= logits.max(1,keepdims=True)
    exp = np.exp(logits)
    return exp/exp.sum(1,keepdims=True)


def metrics(y,prob):
    pred=prob.argmax(1); accepted=prob.max(1)>=THRESHOLD
    cm=np.zeros((3,3),dtype=int)
    for a,b in zip(y,pred): cm[a,b]+=1
    gate=np.zeros((3,4),dtype=int)
    for a,b,keep in zip(y,pred,accepted): gate[a,b if keep else 3]+=1
    out={'n':len(y), 'confusion_rows_truth_columns_prediction':cm.tolist(),
         'gate_confusion_columns_empty_gear_pin_abstain':gate.tolist(),
         'accuracy':float((pred==y).mean()) if len(y) else None,
         'accuracy_wilson95':wilson(int((pred==y).sum()),len(y)),
         'accepted':int(accepted.sum()), 'classes':{}}
    for i,name in enumerate(CLASSES):
        actual=y==i; guessed=(pred==i)&accepted
        tp=int((actual&guessed).sum()); n=int(actual.sum()); npred=int(guessed.sum())
        out['classes'][name]={'actual':n,'accepted_correct':tp,'accepted_predictions':npred,
          'gated_sensitivity':tp/n if n else None,'sensitivity_wilson95':wilson(tp,n),
          'gated_precision':tp/npred if npred else None,'precision_wilson95':wilson(tp,npred)}
    empty=y==0; false=int((empty&accepted&(pred!=0)).sum()); n=int(empty.sum())
    out['empty_false_grasp']={'false_positives':false,'empty_frames':n,
      'rate':false/n if n else None,'wilson95':wilson(false,n)}
    out['gear_pin_confusions_argmax']=int((((y==1)&(pred==2))|((y==2)&(pred==1))).sum())
    return out


def extract_features(rows, checkpoint, model_root, device, mask=None):
    import torch
    from PIL import Image
    import torch.nn.functional as F
    sys.path.insert(0,str(model_root))
    from tools.multiview_keypoints import MultiviewKeypointNetwork
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=MultiviewKeypointNetwork(**state['model_config']).to(device).eval()
    model.load_state_dict(state['model_state'])
    for p in model.parameters(): p.requires_grad_(False)
    features=[]
    with torch.inference_mode():
        for row in rows:
            # Only this fixed jaw crop enters the CNN. No phase, time, arm pose,
            # filenames, other cameras or gripper scalars enter visual features.
            image=Image.open(row['images']['wrist_port3']['path']).convert('RGB')
            if image.size != (640,480): raise ValueError('Expected native 640x480')
            pixels=np.asarray(image.crop(CROP)).copy()
            tip=np.zeros(pixels.shape[:2],dtype=bool); tip[0:75,55:140]=True
            if mask=='jaw_tip_removed': pixels[tip]=128
            if mask=='outside_tip_removed': pixels[~tip]=128
            x=torch.from_numpy(pixels).to(device).permute(2,0,1)[None].float()/255
            h,w=x.shape[-2:]
            yy,xx=torch.meshgrid(torch.linspace(2*CROP[1]/479-1,2*(CROP[3]-1)/479-1,h,device=device),
              torch.linspace(2*CROP[0]/639-1,2*(CROP[2]-1)/639-1,w,device=device), indexing='ij')
            camera=torch.zeros(1,3,h,w,device=device); camera[:,1]=1
            x=torch.cat(((x-.5)/.25,torch.stack((xx,yy))[None],camera),1)
            quarter=model.encoder_quarter(model.stem(x))
            eighth=model.encoder_eighth(quarter); deep=model.encoder_sixteenth(eighth)
            feature=torch.cat([F.adaptive_avg_pool2d(layer,(2,2)).flatten() for layer in (eighth,deep)])
            features.append(feature.cpu().numpy())
    return np.asarray(features),state['metadata'].get('real_adaptation_provenance',{})


def run(labels,output,model_root,device):
    import torch
    torch.set_num_threads(2); torch.manual_seed(17)
    rows,excluded,episodes=load_rows(labels)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint=model_root/'gear_alignment/real_v4/adapted.pt'
    features,encoder_provenance=extract_features(rows,checkpoint,model_root,device)
    pretrained_eps=set(sum(encoder_provenance.get('episode_splits',{}).values(),[]))
    if pretrained_eps & set(episodes): raise ValueError('Frozen encoder episode overlap')
    train=np.asarray([r['split']=='train' for r in rows]); y=np.asarray([r['target'] for r in rows])
    scalar=np.asarray([[r['measured_grippers']['left']['position']] for r in rows])
    models={}; results={}; predictions=[]
    for name,x in [('visual',features),('encoder_position_baseline',scalar)]:
        model=fit_classifier(x[train],y[train]); models[name]=model
        prob=predict(model,x)
        # Leave-one-TRAIN-episode-out diagnostics; no hyperparameter selection.
        inner=np.zeros((int(train.sum()),3)); train_ids=np.flatnonzero(train)
        for ep in sorted({rows[i]['episode_id'] for i in train_ids}):
            inner_test=np.asarray([rows[i]['episode_id']==ep for i in train_ids])
            inner[inner_test]=predict(fit_classifier(x[train][~inner_test],y[train][~inner_test]),x[train][inner_test])
        results[name]={'train_leave_one_episode_out':metrics(y[train],inner),
          'validation':metrics(y[~train],prob[~train]),'by_validation_episode':{}}
        for ep in sorted(ep for ep,s in episodes.items() if s=='validation'):
            mask=np.asarray([r['episode_id']==ep for r in rows])
            results[name]['by_validation_episode'][ep]=metrics(y[mask],prob[mask])
        for i,row in enumerate(rows):
            predictions.append({'model':name,'observation_id':row['observation_id'],'episode_id':row['episode_id'],
              'split':row['split'],'truth':CLASSES[y[i]],'probabilities':dict(zip(CLASSES,prob[i].tolist())),
              'prediction':CLASSES[int(prob[i].argmax())], 'accepted':bool(prob[i].max()>=THRESHOLD),
              'grasp_label_confidence':row['labels']['grasp']['left']['confidence']})
    ablations={}
    for mask in ('jaw_tip_removed','outside_tip_removed'):
        masked,_=extract_features([r for r in rows if r['split']=='validation'],checkpoint,model_root,device,mask=mask)
        masked_prob=predict(models['visual'],masked)
        ablations[mask]=metrics(y[~train],masked_prob)
    coverage={}
    for split in ('train','validation'):
        empty=[r for r in rows if r['split']==split and r['target']==0]
        coverage[split]={'empty_count':len(empty),'empty_phase_counts':dict(Counter(r['phase'] for r in empty)),
          'empty_gripper_positions_rad':[r['measured_grippers']['left']['position'] for r in empty],
          'empty_at_or_below_0_6rad_gear_like_closure':sum(r['measured_grippers']['left']['position']<=.6 for r in empty),
          'empty_at_or_below_0_2rad_pin_like_closure':sum(r['measured_grippers']['left']['position']<=.2 for r in empty)}
    provenance={'labels_path':str(labels.resolve()),'labels_sha256':sha256(labels),
      'label_sources_verified':True,'all_labeled_image_sha256_verified':True,
      'episode_split':episodes,'pretrained_encoder_episode_overlap':[],
      'encoder_checkpoint':str(checkpoint),'encoder_sha256':sha256(checkpoint),
      'encoder_real_adaptation_provenance':encoder_provenance,
      'recording_metadata_hashes':{ep:{f:sha256(ROOT/'data/kept'/ep/f) for f in ('metadata.json','states.jsonl','frames.jsonl')} for ep in episodes},
      'script_sha256':sha256(__file__)}
    report={'schema_version':1,'offline_only':True,'runtime_motion_authorized':False,'grasp_safety_validation_passed':False,
      'validation_verdict':'Insufficient negative/control coverage; candidate not validated for retained-object gating.',
      'method':'Frozen gear CNN stem/eighth/sixteenth learned features pooled 2x2 from fixed native left jaw crop; L2 multinomial logistic regression. No augmentation or threshold/model selection on validation.',
      'classes':list(CLASSES),'crop_native_xyxy':list(CROP),'camera':'wrist_port3',
      'classifier_l2':L2,'acceptance_probability_threshold':THRESHOLD,
      'probabilities_calibrated':False,'labeled_counts':dict(Counter(r['split'] for r in rows)),
      'excluded':excluded,'provenance':provenance,'results':results,'negative_coverage':coverage,
      'occlusion_ablation_diagnostic_only':{'tip_native_xyxy':[295,260,380,335],
        'interpretation':'Fixed classifier tested under artificial 128-grey occlusion; distribution shift is not a causal proof. Not used for selection or training.',
        'results':ablations},
      'limitations':['AI visual labels are not independent human-certified ground truth.',
        'Only 17 same-session episodes and 68 selected stills, with sparse empty/release negatives; frames within episode are correlated.',
        'Wilson intervals treat frames independently and are optimistic for clustered trials; only four validation episodes.',
        'No established temporal persistence, slip detection, closed-but-empty controls, contact-force evidence, or sustained grasp test.',
        'Fixed jaw crop can learn jaw opening/object appearance; it does not establish retention under motion.',
        'Softmax probabilities are uncalibrated; 0.85 is a fixed exploratory abstention threshold and not a validated runtime gate.',
        'Classifier and frozen checkpoint never altered runtime configuration or base weights.']}
    (output/'grasp-report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'grasp-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    np.savez(output/'grasp-candidate.npz',**models['visual'],classes=np.asarray(CLASSES),crop=np.asarray(CROP),threshold=THRESHOLD)
    print(json.dumps({'counts':report['labeled_counts'],'visual':results['visual']['validation'],
      'baseline':results['encoder_position_baseline']['validation']},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,default=ROOT/'data/calibration/manual-labels/assembled/labels.json')
    p.add_argument('--output',type=Path,default=ROOT/'logs/alignment-validation')
    p.add_argument('--model-root',type=Path,default=ROOT.parent/'RobotArmTraining/outputs/panthera-assembly-cnns')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();run(a.labels,a.output,a.model_root,a.device)
