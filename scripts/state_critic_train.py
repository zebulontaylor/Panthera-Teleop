#!/usr/bin/env python3
"""Offline multi-view outcome critic; estimates recorded behavior outcomes, never robot actions.

JSONL rows require episode_id, split (train/val/test or predict), images (3 relative paths in
fixed camera order), and outcome (1/0/-1 unknown). Optional fields: session_id, t seconds,
episode_end_s, terminal bool, history_images (3 paths at t-1), state (finite numeric vector),
label_source, confidence. Images resolve under --dataset-root. Only reviewed labels are included
by default; proxy data must be explicitly selected with --label-sources.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np

CAMERAS = ("overhead", "wrist_port2", "wrist_port3")

def read_rows(path):
    with Path(path).open() as f: return [json.loads(x) for x in f if x.strip()]

def metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None,
        "auroc": float(roc_auc_score(y,p)) if len(y) and len(np.unique(y))>1 else None,
        "average_precision": float(average_precision_score(y,p)) if len(y) and len(np.unique(y))>1 else None,
        "brier": float(brier_score_loss(y,p)) if len(y) else None}

def episode_bootstrap(rows, pred, n_boot=500, seed=19):
    from collections import defaultdict
    groups=defaultdict(list)
    for r,p in zip(rows,pred):
        if r["outcome"] in (0,1): groups[str(r["episode_id"])].append((int(r["outcome"]),float(p)))
    ids=list(groups)
    if len(ids)<2: return {"episodes":len(ids),"intervals":{}}
    rng=np.random.default_rng(seed); draws={k:[] for k in ("auroc","average_precision","brier")}
    for _ in range(n_boot):
        vals=[item for ep in rng.choice(ids,len(ids),replace=True) for item in groups[ep]]
        m=metrics([a for a,_ in vals],[b for _,b in vals])
        for k in draws:
            if m[k] is not None: draws[k].append(m[k])
    return {"episodes":len(ids),"method":"episode bootstrap percentile 95% CI",
        "intervals":{k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in draws.items() if v}}

def image_features(rows, root, backbone, transform, device, batch_size, cache_path, cache_key):
    import torch
    from PIL import Image
    if cache_path and Path(cache_path).exists():
        try:
            z=np.load(cache_path,allow_pickle=False)
            if str(z["key"].item())==cache_key: return z["features"].astype(np.float32)
        except Exception: pass
    features=[]
    with torch.inference_mode():
        for start in range(0,len(rows),max(1,batch_size)):
            tensors=[]; end=min(len(rows),start+batch_size)
            for r in rows[start:end]:
                paths=[r["images"], r.get("history_images") or r["images"]]
                for group in paths:
                    for rel in group:
                        path=Path(root)/rel
                        if not path.is_file(): raise FileNotFoundError(path)
                        with Image.open(path) as im: tensors.append(transform(im.convert("RGB")))
            z=backbone(torch.stack(tensors).to(device)).reshape(end-start,-1,512).cpu().numpy()
            # Keep each camera's current image embedding. History contributes both its embedding
            # and a current-minus-history feature, while missing history contributes zeros.
            for j,r in enumerate(rows[start:end]):
                cur=z[j,:3]
                hist=z[j,3:6] if r.get("history_images") is not None else np.zeros_like(cur)
                delta=(cur-hist) if r.get("history_images") is not None else np.zeros_like(cur)
                features.append(np.concatenate([cur.flatten(),hist.flatten(),delta.flatten()]))
            if end // 250 != start // 250 or end == len(rows):
                print(json.dumps({"feature_rows_done":end,"feature_rows_total":len(rows)}),flush=True)
    features=np.asarray(features,dtype=np.float32)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache_path,key=np.asarray(cache_key),features=features)
    return features

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,required=True,help="JSONL rows; arbitrary rows are allowed with --predict")
    ap.add_argument("--dataset-root",type=Path,default=Path(".")); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--predict",action="store_true",help="inference on all rows; requires --checkpoint")
    ap.add_argument("--checkpoint",type=Path); ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--feature-batch-rows",type=int,default=2,help="at most six images per row; bounds GPU image batch")
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-4); ap.add_argument("--gamma",type=float,default=.99)
    ap.add_argument("--cpu-threads",type=int,default=4); ap.add_argument("--seed",type=int,default=19)
    ap.add_argument("--label-sources",default="reviewed",help="comma-separated accepted sources, or *")
    ap.add_argument("--pretrained",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--feature-cache",type=Path,help="default: OUTPUT/features.npz")
    a=ap.parse_args()
    if a.epochs < 1: raise ValueError("--epochs must be >= 1")
    if a.batch_size < 1: raise ValueError("--batch-size must be >= 1")
    if a.feature_batch_rows < 1: raise ValueError("--feature-batch-rows must be >= 1")
    if not 0 < a.gamma <= 1: raise ValueError("--gamma must be in (0, 1]")
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights,resnet18
    from torchvision import transforms
    torch.set_num_threads(max(1,a.cpu_threads)); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=read_rows(a.manifest)
    if a.predict:
        if not a.checkpoint: raise ValueError("--predict needs --checkpoint")
        allowed=None
    else:
        allowed=None if a.label_sources.strip()=="*" else set(a.label_sources.split(","))
        rows=[r for r in rows if allowed is None or r.get("label_source","reviewed") in allowed]
    if not rows: raise ValueError("No manifest rows remain")
    for i,r in enumerate(rows):
        if not r.get("episode_id") or len(r.get("images",[]))!=3: raise ValueError(f"row {i}: requires episode_id and 3 images")
        if not a.predict and r.get("outcome",-1) not in (-1,0,1): raise ValueError(f"row {i}: invalid outcome")
        if not a.predict and r.get("split") not in ("train","val","test"): raise ValueError(f"row {i}: split must be train/val/test")
        if r.get("history_images") is not None and len(r["history_images"])!=3: raise ValueError(f"row {i}: history_images needs 3 paths")
    splits={s:[i for i,r in enumerate(rows) if r.get("split")==s] for s in ("train","val","test")}
    if not a.predict:
        if not splits["train"]: raise ValueError("Need train rows")
        ep_split={}; sess_split={}; ep_label={}
        for r in rows:
            ep=str(r["episode_id"]); s=r["split"]
            if ep in ep_split and ep_split[ep]!=s: raise ValueError(f"episode leakage across splits: {ep}")
            ep_split[ep]=s
            if r.get("session_id") is not None:
                session=str(r["session_id"])
                if session in sess_split and sess_split[session]!=s: raise ValueError(f"session leakage across splits: {session}")
                sess_split[session]=s
            if r.get("outcome") in (0,1):
                if ep in ep_label and ep_label[ep]!=r["outcome"]: raise ValueError(f"inconsistent outcomes within {ep}")
                ep_label[ep]=r["outcome"]
    a.output.mkdir(parents=True,exist_ok=True)
    ck=None
    if a.predict:
        ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); pretrained=ck["pretrained"]
    else: pretrained=a.pretrained
    weights=(ResNet18_Weights[ck["weights_id"]] if ck and pretrained else ResNet18_Weights.DEFAULT) if pretrained else None
    model_backbone=resnet18(weights=weights); weights_id=weights.name if weights else "random-resnet18"
    model_backbone.fc=nn.Identity();model_backbone.eval().to(device)
    # Full-frame resize preserves the scene around the insertion site (no ImageNet center crop).
    weight_transform=weights.transforms() if weights else None
    image_mean=weight_transform.mean if weights else (.485,.456,.406)
    image_std=weight_transform.std if weights else (.229,.224,.225)
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(image_mean,image_std)])
    image_refs=[]
    for r in rows:
        refs=[]
        for rel in [*r["images"], *(r.get("history_images") or [])]:
            p=Path(a.dataset_root)/rel
            try: st=p.stat(); refs.append((rel,st.st_size,st.st_mtime_ns))
            except FileNotFoundError: refs.append((rel,None,None))
        image_refs.append(refs)
    cache_identity=json.dumps({"dataset_root":str(a.dataset_root.resolve()),"image_refs":image_refs,"weights_id":weights_id,"transform":"RGB resize 224x224, ImageNet normalization, full frame","camera_order":CAMERAS},sort_keys=True,separators=(",",":"))
    cache_key=hashlib.sha256(cache_identity.encode()).hexdigest()
    manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    cache_path=a.feature_cache or (a.output/"features.npz")
    if a.feature_batch_rows < 1: raise ValueError("--feature-batch-rows must be positive")
    vis=image_features(rows,a.dataset_root,model_backbone,transform,device,a.feature_batch_rows,cache_path,cache_key)
    state_dim=ck["state_dim"] if ck else 0
    if ck: state_dim=int(state_dim)
    present=[r.get("state") for r in rows if r.get("state") is not None]
    if present:
        dims={len(s) for s in present}
        if len(dims)!=1 or (state_dim and next(iter(dims))!=state_dim): raise ValueError("Inconsistent state-vector dimensions")
        state_dim=next(iter(dims))
        state=np.asarray([r.get("state",[0.]*state_dim) for r in rows],np.float32)
        if not np.isfinite(state).all(): raise ValueError("state vectors must be finite")
    else: state=np.zeros((len(rows),state_dim),np.float32)
    times=np.asarray([float(r.get("t",0)) for r in rows],np.float32).reshape(-1,1)
    raw=np.concatenate([vis,state],axis=1)
    y=np.asarray([int(r.get("outcome",-1)) for r in rows],np.float32)
    # Per-episode explicit endpoint takes precedence. Otherwise use marked terminal timestamp,
    # then last observed timestamp as a documented fallback.
    ends={}; explicit_ends={}; terminal_ends={}; maxima={}; starts={}
    for r in rows:
        ep=str(r["episode_id"]); t=float(r.get("t",0)); maxima[ep]=max(maxima.get(ep,float("-inf")),t);starts[ep]=min(starts.get(ep,float("inf")),t)
        if r.get("episode_end_s") is not None:
            prev=explicit_ends.setdefault(ep,float(r["episode_end_s"]))
            if prev != float(r["episode_end_s"]): raise ValueError(f"inconsistent episode_end_s in {ep}")
        if r.get("terminal") is True: terminal_ends[ep]=max(terminal_ends.get(ep,float("-inf")),t)
    ends={ep:explicit_ends.get(ep,terminal_ends.get(ep,maxima[ep])) for ep in maxima}
    gamma=ck["gamma"] if ck else a.gamma
    value=np.asarray([float(y[i] if y[i]==0 else gamma**max(0,ends[str(r["episode_id"])]-float(r.get("t",0)))) if y[i]>=0 else 0 for i,r in enumerate(rows)],np.float32)
    class Head(nn.Module):
        def __init__(self,n):
            super().__init__();self.body=nn.Sequential(nn.Linear(n,512),nn.ReLU(),nn.Dropout(.2),nn.Linear(512,128),nn.ReLU());self.outcome=nn.Linear(128,1);self.value=nn.Linear(128,1)
        def forward(self,x):
            h=self.body(x);return self.outcome(h).squeeze(-1),self.value(h).squeeze(-1)
    head=Head(raw.shape[1]).to(device)
    if ck:
        head.load_state_dict(ck["model"]);mean=ck["mean"];scale=ck["scale"]
    else:
        train=np.asarray([i for i in splits["train"] if y[i]>=0],int)
        if not len(train): raise ValueError("No supervised training rows")
        if np.unique(y[train]).size != 2:
            raise ValueError("Training split must contain both outcome classes (0 and 1); refusing a one-class fit")
        mean=raw[train].mean(0);scale=raw[train].std(0);scale[scale<1e-6]=1
    x=((raw-mean)/scale).astype(np.float32)
    if not a.predict:
        from torch.utils.data import WeightedRandomSampler
        train=np.asarray([i for i in splits["train"] if y[i]>=0],int)
        ep_counts={}; ep_class={}
        for i in train:
            ep=str(rows[i]["episode_id"]); ep_counts[ep]=ep_counts.get(ep,0)+1; ep_class[ep]=int(y[i])
        episode_class_counts={c:max(1,sum(1 for label in ep_class.values() if label==c)) for c in (0,1)}
        w=np.asarray([1/episode_class_counts[int(y[i])]/ep_counts[str(rows[i]["episode_id"])] for i in train],np.float64)
        w/=w.sum()
        sampler=WeightedRandomSampler(torch.as_tensor(w),num_samples=len(train),replacement=True)
        xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(y).to(device);vt=torch.from_numpy(value).to(device)
        opt=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=a.weight_decay);val=np.asarray([i for i in splits["val"] if y[i]>=0],int)
        best=float("inf");best_state=None;validation_source="val" if len(val) else "train_fallback_no_validation_split"
        # Draw balanced episode/class mini-batches with replacement.
        for epoch in range(a.epochs):
            head.train();order=list(sampler)
            for st in range(0,len(order),a.batch_size):
                ix=train[np.asarray(order[st:st+a.batch_size])];lo,v=head(xt[ix]);loss=nn.functional.binary_cross_entropy_with_logits(lo,yt[ix])+nn.functional.mse_loss(v,vt[ix]);opt.zero_grad();loss.backward();opt.step()
            head.eval();ix=val if len(val) else train
            with torch.no_grad(): lo,v=head(xt[ix]);score=float(nn.functional.binary_cross_entropy_with_logits(lo,yt[ix])+nn.functional.mse_loss(v,vt[ix]))
            print(json.dumps({"epoch":epoch+1,"validation_joint_loss":score,"validation_rows":len(ix)}))
            if score<best: best=score;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        head.load_state_dict(best_state)
        torch.save({"model":head.state_dict(),"mean":mean,"scale":scale,"state_dim":state_dim,"gamma":gamma,"pretrained":pretrained,"weights_id":weights_id,"camera_order":CAMERAS,"image_transform":"RGB resize 224x224, ImageNet normalization, full frame","image_mean":list(image_mean),"image_std":list(image_std),"visual_features":"ResNet18 per view; current, optional history, and current-minus-history embeddings","feature_dim":512,"label_sources":a.label_sources,"training_manifest_sha256":manifest_hash,"feature_cache_key":cache_key},a.output/"critic.pt")
    head.eval(); report={"warning":"Behavior outcome/value estimates from recorded data; not calibrated, not an autonomous progress signal, and not for robot control.","label_sources":"inference" if a.predict else a.label_sources,"gamma_per_second":gamma,"camera_order":CAMERAS,"weights_id":weights_id,"manifest_sha256":manifest_hash,"feature_cache_key":cache_key,"splits":{}}
    if not a.predict:
        report["model_selection_source"]=validation_source
        if not len(val): report["model_selection_warning"]="No validation rows; best epoch selected on training loss, so reported test metrics remain held out but checkpoint selection may overfit."
    if not a.predict:
        from collections import Counter
        report["label_source_counts"]=dict(Counter(str(r.get("label_source","reviewed")) for r in rows))
        report["source_category_counts"]=dict(Counter(str(r.get("source_category","unknown")) for r in rows))
        report["diagnostic_only"]=bool(rows) and all(r.get("diagnostic_only") is True for r in rows)
        if report["label_source_counts"] == {"curated_proxy": len(rows)}:
            report["interpretation_limit"]="Kept/discarded curation proxy only; not verified task success. Split is temporal blocks from the same date range and remains confounded with time and stationary/reset behavior."
        confidences=[r["confidence"] for r in rows if r.get("confidence") is not None]
        numeric_conf=[float(x) for x in confidences if isinstance(x,(int,float))]
        report["label_confidence"]={"n":len(confidences),"levels":dict(Counter(str(x) for x in confidences if not isinstance(x,(int,float)))),"numeric_mean":float(np.mean(numeric_conf)) if numeric_conf else None,"numeric_min":float(np.min(numeric_conf)) if numeric_conf else None,"numeric_max":float(np.max(numeric_conf)) if numeric_conf else None}
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        diagnostics={}
        for name,features in (("state_only",state),("time_only",times)):
            d={}; diagnostics[name]=d
            if not features.shape[1]: d["warning"]="No state vectors";continue
            tr=np.asarray([i for i in splits["train"] if y[i]>=0],int)
            if np.unique(y[tr]).size<2: d["warning"]="Training needs both classes";continue
            sc=StandardScaler().fit(features[tr]);xx=sc.transform(features);clf=LogisticRegression(max_iter=1000,random_state=a.seed).fit(xx[tr],y[tr].astype(int))
            for s,idx in splits.items():
                known=[i for i in idx if y[i]>=0];p=clf.predict_proba(xx[known])[:,1] if known else []
                d[s]=metrics(y[known],p)
                if known:d[s]["episode_bootstrap"]=episode_bootstrap([rows[i] for i in known],p)
        report["diagnostic_baselines"]=diagnostics
    predictions=[]
    with torch.no_grad():
        eval_splits={"predict":list(range(len(rows)))} if a.predict else splits
        for s,idx in eval_splits.items():
            if not idx: continue
            lo,v=head(torch.from_numpy(x[idx]).to(device));p=torch.sigmoid(lo).cpu().numpy();v=v.cpu().numpy()
            known=[j for j,i in enumerate(idx) if y[i]>=0]
            if not a.predict:
                mm=metrics(y[np.asarray(idx)[known]],p[known]) if known else metrics([],[])
                mm["value_rmse"]=float(np.sqrt(np.mean((v[known]-value[np.asarray(idx)[known]])**2))) if known else None
                mm["episode_bootstrap"]=episode_bootstrap([rows[idx[j]] for j in known],p[known]) if known else None
                groups={}
                for j in known:groups.setdefault(str(rows[idx[j]]["episode_id"]),[]).append(j)
                ey=[];ep=[]
                for js in groups.values():
                    labs={int(y[idx[j]]) for j in js}
                    if len(labs)==1:ey.append(labs.pop());ep.append(float(np.mean(p[js])))
                mm["episode_aggregate"]=metrics(ey,ep)
                # Early/late and explicitly terminal slices reveal temporal shortcut patterns.
                phases={"terminal":[],"early":[],"late":[]}
                for j in known:
                    r=rows[idx[j]]; ep_id=str(r["episode_id"]); duration=max(1e-6,ends[ep_id]-starts[ep_id])
                    frac=(float(r.get("t",0))-starts[ep_id])/duration
                    if r.get("terminal") is True: phases["terminal"].append(j)
                    phases["early" if frac<.5 else "late"].append(j)
                mm["temporal_slices"]={k:metrics([y[idx[j]] for j in js],[p[j] for j in js]) for k,js in phases.items()}
                report["splits"][s]=mm
            for j,i in enumerate(idx):
                predictions.append({"episode_id":rows[i]["episode_id"],"session_id":rows[i].get("session_id"),"split":s,"t":rows[i].get("t"),"outcome":int(y[i]),"label_source":rows[i].get("label_source"),"confidence":rows[i].get("confidence"),"outcome_probability":float(p[j]),"discounted_outcome_estimate":float(v[j]),"target_discounted_outcome":None if y[i]<0 else float(value[i])})
    (a.output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (a.output/"predictions.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in predictions))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
