"""Training-only RGB variation while retaining measured demonstration poses.

Extends scripts/train_pin_insertion_cnn.py::augment and
scripts/adapt_pin_insertion_cnn.py::affine_sample. Their two-landmark and
amodal-cutout contracts cannot directly handle ten visible-only landmarks.
The exposure/noise/blur/framing mechanisms are retained and expanded below.

Affine changes model small image framing uncertainty, never synthetic 3-D robot
poses. Call only on training batches; validation and test inputs stay untouched.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def transform_points(points, forward):
    """Apply normalized-image forward 2x3 affine matrices (not sampling inverses)."""
    return torch.bmm(points, forward[:,:,:2].transpose(1,2))+forward[:,:,2][:,None,:]


def sampling_transform(forward):
    """Convert [0,1] forward image transform into [-1,1] grid-sampling inverse."""
    A=forward[:,:,:2];offset=forward[:,:,2]
    inverse=torch.linalg.inv(A)
    inverse_offset=torch.bmm(inverse,(.5-offset)[:,:,None]).squeeze(-1)*2-1
    return torch.cat((inverse,inverse_offset[:,:,None]),-1)


def augment_batch(images, points, visible, generator=None):
    if images.ndim!=4 or images.shape[1]!=3 or points.shape!=(len(images),10,2) or visible.shape!=(len(images),10):
        raise ValueError('BCHW RGB, Bx10x2 points and Bx10 visibility required')
    if not images.is_floating_point() or not points.is_floating_point():
        raise ValueError('floating images and normalized points required')
    b,_,h,w=images.shape;device=images.device;dtype=images.dtype
    def rand(*shape):return torch.rand(shape,device=device,dtype=dtype,generator=generator)
    def uniform(low,high,*shape):return low+(high-low)*rand(*shape)
    # Coordinates use [0,1], centered at .5; adjust aspect for true image rotation.
    angle=uniform(-6,6,b)*math.pi/180;scale=uniform(.9,1.1,b)
    c=angle.cos()*scale;s=angle.sin()*scale
    A=torch.zeros(b,2,2,device=device,dtype=dtype)
    A[:,0,0]=c;A[:,1,1]=c;A[:,0,1]=-s*h/w;A[:,1,0]=s*w/h
    translation=uniform(-.04,.04,b,2)
    offset=.5-A.sum(-1)*.5+translation
    forward=torch.cat((A,offset[:,:,None]),-1)
    theta=sampling_transform(forward)
    grid=F.affine_grid(theta,images.shape,align_corners=False)
    x=F.grid_sample(images,grid,mode='bilinear',padding_mode='reflection',align_corners=False)
    p=transform_points(points,forward)
    valid=visible.bool().clone() & torch.isfinite(p).all(-1) & (p>=0).all(-1) & (p<=1).all(-1)
    # Illumination, exposure, gamma, white balance, contrast, color and grayscale.
    x=x*uniform(.65,1.35,b,1,1,1)*uniform(.80,1.20,b,3,1,1)
    gray=(x*torch.tensor([.299,.587,.114],device=device,dtype=dtype)[None,:,None,None]).sum(1,keepdim=True)
    saturation=uniform(.45,1.5,b,1,1,1)
    saturation=torch.where(rand(b,1,1,1)<.08,torch.zeros_like(saturation),saturation)
    x=gray+(x-gray)*saturation
    mean=x.mean((1,2,3),keepdim=True)
    x=(x-mean)*uniform(.65,1.45,b,1,1,1)+mean+uniform(-.07,.07,b,1,1,1)
    x=x.clamp(0,1).pow(uniform(.75,1.35,b,1,1,1))
    blurred=F.avg_pool2d(F.pad(x,(1,1,1,1),mode='reflect'),3,stride=1)
    # Both blur and mild unsharp masking, selected independently per frame.
    sharp=uniform(-.65,.45,b,1,1,1)
    x=x+sharp*(x-blurred)
    low=F.interpolate(F.interpolate(x,size=(max(4,h//2),max(4,w//2)),mode='bilinear',align_corners=False),size=(h,w),mode='bilinear',align_corners=False)
    x=torch.where(rand(b,1,1,1)<.35,low,x)
    noise=torch.randn(x.shape,device=device,dtype=dtype,generator=generator)*uniform(0,.035,b,1,1,1)
    x=x+noise
    # Compression-like quantization (not a claim of exact JPEG encoding).
    levels=uniform(45,160,b,1,1,1)
    compressed=torch.round(x*levels)/levels
    x=torch.where(rand(b,1,1,1)<.35,compressed,x)
    yy,xx=torch.meshgrid((torch.arange(h,device=device,dtype=dtype)+.5)/h,
                         (torch.arange(w,device=device,dtype=dtype)+.5)/w,indexing='ij')
    # Smooth off-axis specular streak; photometric glare keeps labels intact.
    gx=uniform(0,1,b,1,1);gy=uniform(0,1,b,1,1);ga=uniform(-math.pi,math.pi,b,1,1)
    along=(xx-gx)*ga.cos()+(yy-gy)*ga.sin()
    across=-(xx-gx)*ga.sin()+(yy-gy)*ga.cos()
    glare=torch.exp(-across.square()/uniform(.0003,.003,b,1,1)-along.square()/uniform(.025,.15,b,1,1))
    x=x+glare[:,None]*uniform(.05,.30,b,1,1,1)*(rand(b,1,1,1)<.45)
    # Occluders are the only synthetic effects that remove visibility labels.
    for stripe in (False,True):
        center=uniform(.05,.95,b,2)
        extent=uniform(.05,.22,b,2)
        if stripe:extent[:,0]=uniform(.015,.055,b);extent[:,1]=uniform(.25,.65,b)
        enabled=rand(b)<(.40 if not stripe else .20)
        lo=center-extent/2;hi=center+extent/2
        mask=(xx[None]>=lo[:,0,None,None])&(xx[None]<=hi[:,0,None,None])&(yy[None]>=lo[:,1,None,None])&(yy[None]<=hi[:,1,None,None])&enabled[:,None,None]
        fill=uniform(.03,.6,b,3,1,1)
        x=torch.where(mask[:,None],fill,x)
        covered=(p[:,:,0]>=lo[:,0,None])&(p[:,:,0]<=hi[:,0,None])&(p[:,:,1]>=lo[:,1,None])&(p[:,:,1]<=hi[:,1,None])&enabled[:,None]
        valid=valid & ~covered
    return x.clamp(0,1),p,valid

AUGMENTATION_CONFIG = {
    'training_only': True, 'implementation': 'GPU PyTorch generalized from repository two-landmark pipeline',
    'source_functions': ['scripts/train_pin_insertion_cnn.py::augment',
                         'scripts/adapt_pin_insertion_cnn.py::affine_sample'],
    'affine': {'rotation_degrees': [-6,6], 'scale': [.9,1.1], 'translation_fraction': [-.04,.04],
               'keypoints': 'same forward affine as image; inverse used only for grid sampling'},
    'exposure': [.65,1.35], 'white_balance': [.80,1.20], 'saturation': [.45,1.5],
    'grayscale_probability': .08, 'contrast': [.65,1.45], 'brightness_offset': [-.07,.07],
    'gamma': [.75,1.35], 'sharpness': [-.65,.45], 'downsample_probability': .35,
    'sensor_noise_std': [0,.035], 'quantization_probability': .35,
    'quantization_levels': [45,160], 'specular_streak_probability': .45,
    'rectangle_occlusion_probability': .40, 'stripe_occlusion_probability': .20,
    'occlusion_semantics': 'newly covered keypoints become invisible; never supervise hidden coordinates',
    'three_dimensional_pose_augmentation': False,
}


def export_augmentation_preview(images,points,visible,path,*,seed=4401,variants=4):
    """Save original plus independent variants for a small training-only RGB audit."""
    from PIL import Image,ImageDraw
    generator=torch.Generator(device=images.device).manual_seed(seed)
    rows=[]
    for i in range(min(len(images),4)):
        row=[(images[i],points[i],visible[i])]
        for _ in range(variants):
            x,p,v=augment_batch(images[i:i+1],points[i:i+1],visible[i:i+1],generator)
            row.append((x[0],p[0],v[0]))
        rows.append(row)
    tile=224;canvas=Image.new('RGB',((variants+1)*tile,len(rows)*(tile+20)),'#222222')
    draw=ImageDraw.Draw(canvas)
    for r,row in enumerate(rows):
        for c,(x,p,v) in enumerate(row):
            im=Image.fromarray((x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')).resize((tile,tile))
            pen=ImageDraw.Draw(im)
            for xy,ok in zip(p.detach().cpu().tolist(),v.detach().cpu().tolist()):
                if ok:
                    xx,yy=xy[0]*tile,xy[1]*tile;pen.ellipse((xx-2,yy-2,xx+2,yy+2),outline='lime')
            canvas.paste(im,(c*tile,r*(tile+20)))
            draw.text((c*tile+3,r*(tile+20)+tile+2),'original' if c==0 else f'stochastic variant {c}',fill='white')
    canvas.save(path)
    return {'seed':seed,'source_frames':len(rows),'augmented_variants':len(rows)*variants,
            'visible_labels_removed_in_variants':sum(int((row[0][2].bool() & ~item[2].bool()).sum()) for row in rows for item in row[1:])}
