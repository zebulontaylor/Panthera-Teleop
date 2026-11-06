"""Training-only held-pin lighting variation layered on carrier augmentation.

This changes illumination, not the measured grasp or 3-D pose. Shadows retain
labels; only the shared augmentation's geometric bounds/occluders remove them.
"""
from __future__ import annotations

from copy import deepcopy
import math
import torch
from torch.nn import functional as F
from app.carrier_face_augmentation import augment_batch as carrier_augment_batch
from app.carrier_face_augmentation import AUGMENTATION_CONFIG as CARRIER_CONFIG


def apply_shadows(images, generator=None, *, extreme=False):
    """Apply varied multiplicative lighting to BCHW RGB in [0,1].

    A device-matching torch.Generator makes every draw reproducible. This helper
    is photometric only and never reads or modifies landmark labels.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError('floating BCHW RGB images required')
    b, _, h, w = images.shape
    device, dtype = images.device, images.dtype
    # Evaluate the smooth lighting field cheaply, retaining native-resolution RGB.
    field_h, field_w = min(h,120), min(w,160)
    def rand(*shape):
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)
    def uniform(lo, hi, *shape):
        return lo + (hi-lo)*rand(*shape)
    yy, xx = torch.meshgrid((torch.arange(field_h, device=device, dtype=dtype)+.5)/field_h,
                            (torch.arange(field_w, device=device, dtype=dtype)+.5)/field_w, indexing='ij')
    light = torch.ones((b,field_h,field_w), device=device, dtype=dtype)
    # Broad directional cast shadows: hard edges and wide penumbras both occur.
    for _ in range(2):
        angle = uniform(-math.pi, math.pi, b,1,1)
        distance = (xx-.5)*angle.cos() + (yy-.5)*angle.sin()
        edge = uniform(-.4,.4,b,1,1)
        softness = uniform(.003,.13,b,1,1)
        mask = torch.sigmoid((distance-edge)/softness)
        strength = uniform(.35,.97,b,1,1)*(rand(b,1,1)<.75)
        light = light*(1-strength*mask)
    # Several arbitrarily oriented finger/tool-like stripes and local patches.
    for _ in range(4):
        angle = uniform(-math.pi,math.pi,b,1,1)
        dx, dy = xx-uniform(0,1,b,1,1), yy-uniform(0,1,b,1,1)
        along = dx*angle.cos()+dy*angle.sin()
        across = -dx*angle.sin()+dy*angle.cos()
        width = uniform(.015,.20,b,1,1)
        softness = uniform(.002,.06,b,1,1)
        stripe = torch.sigmoid((width-across.abs())/softness)
        local = torch.exp(-.5*((along/uniform(.08,.4,b,1,1)).square() +
                                 (across/uniform(.04,.25,b,1,1)).square()))
        mask = torch.where(rand(b,1,1)<.55,stripe,local)
        light = light*(1-uniform(.3,.97,b,1,1)*mask*(rand(b,1,1)<.7))
    # Uneven ambient light, off-center vignette, and occasional severe low light.
    angle = uniform(-math.pi,math.pi,b,1,1)
    gradient = ((xx-.5)*angle.cos()+(yy-.5)*angle.sin()+.71)/1.42
    light = light*(1-uniform(0,.85,b,1,1)*gradient)
    radius = ((xx-uniform(.25,.75,b,1,1)).square()+
              (yy-uniform(.25,.75,b,1,1)).square())/.5
    light = light*(1-uniform(0,.9,b,1,1)*radius.clamp(0,1))
    exposure = torch.where(rand(b,1,1)<.3, uniform(.03,.25,b,1,1), uniform(.55,1,b,1,1))
    if extreme:
        # Darken the central cap region severely while retaining peripheral context.
        central = torch.exp(-.5*(((xx-.523)/.22).square()+((yy-.646)/.22).square()))
        light = light*(1-.97*central)
        exposure = exposure.clamp(.3,.65)
    # Bound the combined attenuation at 97%, even where effects overlap.
    attenuation = (light*exposure).clamp(.03,1)[:,None]
    if (field_h,field_w) != (h,w):
        attenuation = F.interpolate(attenuation,size=(h,w),mode='bilinear',align_corners=False)
    result = (images*attenuation).clamp(0,1)
    if not extreme:
        result = torch.where((rand(b,1,1,1)<.25), images, result)
    return result


def augment_batch(images, points, visible, generator=None):
    """Carrier geometry/occlusion plus held-pin severe lighting, same API."""
    images, points, visible = carrier_augment_batch(images, points, visible, generator)
    return apply_shadows(images, generator), points, visible


AUGMENTATION_CONFIG = deepcopy(CARRIER_CONFIG)
AUGMENTATION_CONFIG.update({
    'implementation': 'held-pin training wrapper around carrier augmentation',
    'ordinary_lighting_probability': .25, 'extreme_cap_center': [.523,.646],
    'lighting_field_max_height_width': [120,160],
    'lighting_field_upsampling': 'bilinear align_corners=False',
    'shadow_cast_count': 2, 'shadow_cast_probability': .75,
    'shadow_edge_softness_fraction': [.003,.13],
    'shadow_stripe_or_local_count': 4, 'shadow_stripe_or_local_probability': .7,
    'shadow_strength': [.3,.97], 'gradient_strength': [0,.85],
    'vignette_strength': [0,.9], 'low_light_probability': .3,
    'low_light_exposure': [.03,.25], 'ordinary_exposure': [.55,1],
    'maximum_combined_lighting_attenuation': .97,
    'shadow_label_semantics': 'photometric lighting preserves all shared augmentation labels',
    'grasp_semantics': 'retains measured grasps; no synthetic 3-D poses',
})


def export_augmentation_preview(images, points, visible, path, *, seed=4401, variants=4):
    """Export original and held-pin variants with visible landmark overlays."""
    from PIL import Image, ImageDraw
    generator = torch.Generator(device=images.device).manual_seed(seed)
    rows = min(len(images),4)
    tile = 224
    canvas = Image.new('RGB',((variants+1)*tile,rows*(tile+20)),'#222222')
    removed = 0
    for r in range(rows):
        items = [(images[r],points[r],visible[r])]
        for _ in range(variants):
            x,p,v = augment_batch(images[r:r+1],points[r:r+1],visible[r:r+1],generator)
            items.append((x[0],p[0],v[0]))
            removed += int((visible[r].bool() & ~v[0]).sum())
        for c,(x,p,v) in enumerate(items):
            im = Image.fromarray((x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')).resize((tile,tile))
            draw = ImageDraw.Draw(im)
            for xy,ok in zip(p.detach().cpu().tolist(),v.detach().cpu().tolist()):
                if ok:
                    px,py = xy[0]*tile,xy[1]*tile
                    draw.ellipse((px-2,py-2,px+2,py+2),outline='lime')
            canvas.paste(im,(c*tile,r*(tile+20)))
            ImageDraw.Draw(canvas).text((c*tile+3,r*(tile+20)+tile+2), 'original' if c==0 else f'held-pin variant {c}',fill='white')
    canvas.save(path)
    return {'seed':seed,'source_frames':rows,'augmented_variants':rows*variants,
            'visible_labels_removed_in_variants':removed}

# Descriptive backwards-compatible helper name for photometric-only callers.
apply_shadow_lighting = apply_shadows
