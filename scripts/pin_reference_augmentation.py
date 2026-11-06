"""Paired photometric nuisance augmentation; never moves or invents landmarks."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def augment_pair(current_rgb, reference_rgb, keypoints, valid, rng):
    """Return uint8 paired RGB, copied normalized (2,2) labels and bool (2,) mask.

    Shadows retain supervised coordinates. Saturating synthetic bloom masks a
    covered current-frame landmark rather than supervising an invisible fake tip.
    Reference and current share lighting structure with modest exposure mismatch.
    No spatial transforms, object labels, or background pin impostors are added.
    """
    a=np.asarray(current_rgb); b=np.asarray(reference_rgb)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=3:
        raise ValueError('current/reference must have matching HWC RGB shapes')
    k=np.asarray(keypoints).copy(); v=np.asarray(valid,dtype=bool).copy()
    if k.shape!=(2,2) or v.shape!=(2,):raise ValueError('expected keypoints (2,2), valid (2,)')
    h,w=a.shape[:2]; yy,xx=np.mgrid[:h,:w]
    # Carrier ROI is weakly centered on known hole, with wide support. Labels
    # control nuisance placement only and retain exactly their original values.
    center=k[1]*[w,h] if v[1] and np.isfinite(k[1]).all() else np.array([w*.5,h*.65])
    cx,cy=center+rng.normal(0,[w*.10,h*.10]); radius=rng.uniform(.16,.36)*min(w,h)
    roi=np.exp(-(((xx-cx)/(radius*1.5))**2+((yy-cy)/radius)**2)/2)
    angle=rng.uniform(-np.pi,np.pi)
    grad=((xx/w-.5)*np.cos(angle)+(yy/h-.5)*np.sin(angle))
    shared=rng.uniform(.75,1.22)*(1+rng.uniform(-.30,.30)*grad)
    shadow=np.zeros((h,w),np.float32)
    if rng.random()<.85:
        # A projected edge/cast polygon with variable hard/soft penumbra.
        nx,ny=np.cos(angle),np.sin(angle); tx,ty=-ny,nx
        edge=np.array([cx,cy])+rng.uniform(-radius,radius)*np.array([nx,ny])
        span=max(w,h)*2
        points=[edge-span*np.array([tx,ty]),edge+span*np.array([tx,ty]),
                edge+span*np.array([tx,ty])+span*np.array([nx,ny]),
                edge-span*np.array([tx,ty])+span*np.array([nx,ny])]
        mask=Image.new('L',(w,h));ImageDraw.Draw(mask).polygon([tuple(p) for p in points],fill=255)
        blur=float(rng.choice([0.,.5,2.,5.,10.]))*min(w,h)/240
        mask=mask.filter(ImageFilter.GaussianBlur(blur))
        shadow=np.asarray(mask,dtype=np.float32)/255*roi*rng.uniform(.25,.78)
    glint=np.zeros((h,w),np.float32)
    if rng.random()<.65:
        gx,gy=np.array([cx,cy])+rng.normal(0,radius*.55,2)
        ga=rng.uniform(-np.pi,np.pi); dx=xx-gx;dy=yy-gy
        u=dx*np.cos(ga)+dy*np.sin(ga);z=-dx*np.sin(ga)+dy*np.cos(ga)
        sx=rng.uniform(.02,.22)*radius;sy=rng.uniform(.008,.04)*radius
        glint=np.exp(-.5*((u/max(sx,1))**2+(z/max(sy,1))**2))*rng.uniform(55,230)*roi
    out=[]
    for i,img in enumerate((a,b)):
        mismatch=rng.uniform(.90,1.10) if i else 1.
        f=img.astype(np.float32)*shared[...,None]*(1-shadow[...,None]*rng.uniform(.85,1.10))*mismatch
        f+=glint[...,None]*rng.uniform(.7,1.15)
        f+=rng.normal(0,rng.uniform(.0,2.),f.shape)
        out.append(np.clip(f,0,255).astype(np.uint8))
    # Mask only strong opaque synthetic bloom over current landmark evidence.
    for j in range(2):
        if v[j] and np.isfinite(k[j]).all() and np.all((k[j]>=0)&(k[j]<1)):
            x,y=np.minimum((k[j]*[w,h]).astype(int),[w-1,h-1])
            patch=out[0][max(0,y-1):y+2,max(0,x-1):x+2]
            if glint[y,x]>120 and np.mean(patch>247)>.7:v[j]=False
    return out[0],out[1],k,v
