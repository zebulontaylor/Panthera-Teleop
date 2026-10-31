#!/usr/bin/env python3
"""Small reviewed-pair correction of layer4/fusion/value, selected on val only."""
import argparse,json,random,math
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from state_value_correct import anchors,read_pairs,load_model,digest,objective,metrics
from state_value_train import atomic_save


def trainable_layers(model):
    for p in model.parameters():p.requires_grad_(False)
    for block in (model.backbone.layer4,model.fuse,model.value):
        for p in block.parameters():p.requires_grad_(True)
    model.eval()  # Freeze all running BatchNorm statistics throughout adaptation.


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('pairs','teacher-checkpoint','output','anchors'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--anchor-batch-size',type=int,default=4);ap.add_argument('--anchors-per-episode',type=int,default=12)
    ap.add_argument('--lr',type=float,default=1e-5);ap.add_argument('--value-lr',type=float,default=1e-4);ap.add_argument('--anchor-weight',type=float,default=2.);ap.add_argument('--anchor-val-weight',type=float,default=.2);ap.add_argument('--margin',type=float,default=.15);ap.add_argument('--seed',type=int,default=19);ap.add_argument('--device',default='cuda');ap.add_argument('--reset-value-head',action='store_true')
    args=ap.parse_args()
    if min(args.epochs,args.batch_size,args.anchor_batch_size,args.anchors_per_episode)<1:raise ValueError('positive counts required')
    for k in ('lr','value_lr','anchor_weight','anchor_val_weight','margin'):
        if not math.isfinite(getattr(args,k)) or getattr(args,k)<0:raise ValueError('invalid '+k)
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device);model,ck=load_model(args.teacher_checkpoint,device)
    rows=read_pairs(args.pairs);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    if not train or not val:raise ValueError('train/val required')
    ar=anchors(args.anchors,rows,args.anchors_per_episode);atr=[r for r in ar if r['split']=='train'];ava=[r for r in ar if r['split']=='val']
    if not atr:raise ValueError('normal train anchors required')
    args.output.mkdir(parents=True,exist_ok=True)
    size=ck['config']['image_size'];tf=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def images(refs):
        result=[]
        for views in refs:
            ims=[]
            for p in views:
                with Image.open(p) as im:ims.append(tf(im.convert('RGB')))
            result.append(torch.stack(ims))
        return torch.stack(result).to(device)
    def encode_cache(selected):
        refs=sorted(set(tuple(r[s]) for r in selected for s in ('a','b')));cache={}
        with torch.no_grad():
            for start in range(0,len(refs),4):
                sub=refs[start:start+4];cache.update(zip(sub,model.encode(images(sub)).detach().cpu()))
        return cache
    def anchor_logits(selected):
        result=[]
        with torch.no_grad():
            for start in range(0,len(selected),8):result.extend(model.value(model.encode(images([r['images'] for r in selected[start:start+8]]))).sigmoid().flatten().cpu().tolist())
        return torch.tensor(result,device=device)
    print('Caching original teacher anchor targets (train/val only)',flush=True)
    targets=anchor_logits(atr);vtargets=anchor_logits(ava)
    teacher={s:metrics(model.value,seq,encode_cache(seq),device,args.margin,1.) for s,seq in [('train',train),('val',val)]}
    if args.reset_value_head:
        for module in model.value.modules():
            if hasattr(module,'reset_parameters'):module.reset_parameters()
    trainable_layers(model)
    opt=torch.optim.AdamW([{'params':model.backbone.layer4.parameters(),'lr':args.lr},{'params':model.fuse.parameters(),'lr':args.lr},{'params':model.value.parameters(),'lr':args.value_lr}],weight_decay=1e-4)
    report={'teacher':teacher,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'teacher_checkpoint_sha256':digest(args.teacher_checkpoint),'pairs_sha256':digest(args.pairs),'anchors_sha256':digest(args.anchors),'anchor_counts':{'train':len(atr),'val':len(ava)},'selection':'Reviewed val preference loss plus normal val anchor drift; all test images excluded. Layer4+fuse+value train with frozen BatchNorm statistics.','history':[]};best=float('inf')
    for epoch in range(args.epochs):
        random.shuffle(train)
        for start in range(0,len(train),args.batch_size):
            sub=train[start:start+args.batch_size];ix=random.sample(range(len(atr)),min(args.anchor_batch_size,len(atr)))
            refs=[r[s] for r in sub for s in ('a','b')]+[atr[i]['images'] for i in ix]
            z=model.encode(images(refs));v=model.value(z).flatten();vp=v[:2*len(sub)].reshape(-1,2);labels=torch.tensor([r['label'] for r in sub],dtype=torch.float32,device=device)
            loss=objective(vp[:,0],vp[:,1],labels,args.margin,1.)+args.anchor_weight*F.mse_loss(v[2*len(sub):].sigmoid(),targets[ix])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);opt.step()
        result={s:metrics(model.value,seq,encode_cache(seq),device,args.margin,1.) for s,seq in [('train',train),('val',val)]}
        pred=anchor_logits(ava);drift=float(F.mse_loss(pred,vtargets)) if ava else 0.;score=result['val']['loss']+args.anchor_val_weight*drift
        entry={'epoch':epoch+1,**result,'anchor_val_mse':drift,'selection_score':score};report['history'].append(entry)
        if score<best:
            best=score;state={k:v for k,v in ck.items() if k not in ('optimizer','scheduler','scaler')};state.update(model=model.state_dict(),best=best,epoch=epoch+1,correction_provenance={k:v for k,v in report.items() if k!='history'},selection_objective=report['selection']);atomic_save(state,args.output/'best.pt');report['best_epoch']=epoch+1;report['corrected']=result;report['best_anchor_val_mse']=drift
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'epoch':epoch+1,'train_acc':result['train']['directional_accuracy'],'val_acc':result['val']['directional_accuracy'],'val_loss':result['val']['loss'],'anchor_drift':drift}),flush=True)

if __name__=='__main__':main()
