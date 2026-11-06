#!/usr/bin/env python3
"""Render a deterministic validation-only before/after endpoint audit as JPEG."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from app.held_pin_cnn import HeldPinCNN, SCHEMA
from app.held_pin_augmentation import apply_shadows
from scripts.train_held_pin_cnn import prepare, batch


def select_diverse(rows, count):
    """Greedily balance groups, lighting domains and variants, deterministically."""
    selected, used = [], Counter()
    remaining = sorted(rows,key=lambda r:r['id'])
    while remaining and len(selected)<count:
        def score(r):
            return (used['g',r['group_id']],used['l',r.get('lighting_domain','real')],
                    used['v',r.get('variant','real')],r['id'])
        row = min(remaining,key=score)
        remaining.remove(row)
        selected.append(row)
        for key,value in [('g',row['group_id']),('l',row.get('lighting_domain','real')),
                          ('v',row.get('variant','real'))]:
            used[key,value] += 1
    return selected


def load_model(path,device,axis):
    payload = torch.load(path,map_location=device,weights_only=True)
    metadata = payload['metadata']
    if metadata.get('schema')!=SCHEMA or metadata.get('camera')!='left_wrist':
        raise ValueError(f'Unsupported checkpoint: {path}')
    if abs(metadata.get('axis_x_fraction',axis)-axis)>1e-6:
        raise ValueError('Checkpoint and manifest center axes differ')
    model = HeldPinCNN().to(device)
    model.load_state_dict(payload['model'])
    return model.eval()


def predict(model,x,axis):
    with torch.inference_mode():
        out = model(x,axis_x=axis,roi_size_px=160*x.shape[-1]/640)
    confidence = torch.minimum(out['coarse']['visibility_logits'].sigmoid(),
                               out['fine']['visibility_logits'].sigmoid())[0]
    return {'top':float(out['top_y'][0]),'bottom':float(out['bottom_y'][0]),
            'confidence':float(confidence)}


def overlay(image,row,predictions,axis,box=None):
    """Draw only endpoint Y on configured axis, never predict image-dependent X."""
    image = image.copy()
    w,h = image.size
    draw = ImageDraw.Draw(image)
    if box is None: box=(0,0,w,h)
    left,top,right,bottom = box
    def position(y):
        return (axis*w-left,y*h-top)
    crop = image.crop(box)
    pen = ImageDraw.Draw(crop)
    for key,color in [('top','#60ff60'),('bottom','#ff70ed')]:
        valid = row['top_visible'] if key=='top' else row['bottom_valid']
        if valid:
            px,py=position(float(row[key][1]))
            pen.line((px-10,py,px+10,py),fill=color,width=2)
            pen.rectangle((px-3,py-3,px+3,py+3),outline=color,width=2)
    for index,(prediction,color) in enumerate(zip(predictions,['#45ddff','#ffb52e'])):
        for key in ('top','bottom'):
            y=prediction[key]
            if not np.isfinite(y):continue
            px,py=position(y)
            radius=5+index*3
            if key=='top':pen.ellipse((px-radius,py-radius,px+radius,py+radius),outline=color,width=2)
            else:
                pen.line((px-radius,py-radius,px+radius,py+radius),fill=color,width=2)
                pen.line((px-radius,py+radius,px+radius,py-radius),fill=color,width=2)
    return crop


def error_text(row,pred):
    top = f'{abs(pred["top"]-float(row["top"][1]))*480:.1f}px' if row['top_visible'] else 'no visible GT'
    bottom = f'{abs(pred["bottom"]-float(row["bottom"][1]))*480:.1f}px' if row['bottom_valid'] else 'no GT'
    return f'T {top} / B {bottom}; conf {pred["confidence"]:.2f}'


def render(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    _,rows,axis = prepare(args.real_manifest,args.sim_dataset)
    # Read strata only for selection; held-out test frames are never selected.
    with np.load(args.sim_dataset) as data:
        for row in rows:
            if row['domain']=='sim':
                i=int(row['id'].split('_')[1])
                for key in ('variant','lighting_domain'):
                    if key in data.files:row[key]=str(data[key][i])
    eligible=[r for r in rows if r['split']==args.split]
    chosen=select_diverse([r for r in eligible if r['domain']=='real'],4)
    chosen+=select_diverse([r for r in eligible if r['domain']=='sim'],8)
    if not chosen:raise ValueError('No validation samples available')
    models=[load_model(path,device,axis) for path in (args.before,args.after)]
    # Each row contains original and fixed stress; each has context plus detail.
    width,row_height,header=1040,230,100
    canvas=Image.new('RGB',(width,header+row_height*len(chosen)),'#171b22')
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    draw.text((10,7),'HELD PIN / validation only / fixed center axis / original + extreme shadow',fill='white',font=font)
    draw.text((10,28),'Before = cyan | After = orange | TOP = circle | BOTTOM = X | GT TOP = green square',fill='white',font=font)
    draw.text((10,49),'GT BOTTOM = magenta square (SIM AMODAL, including hidden); real BOTTOM has no ground truth.',fill='white',font=font)
    draw.text((10,70),'Error is absolute endpoint Y at 480px reference height. Predictions shown regardless of confidence.',fill='white',font=font)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    for index,row in enumerate(chosen):
        x,*_=batch([row],device)
        stress=apply_shadows(x,generator,extreme=True)
        base_y=header+index*row_height
        label=f'{row["domain"].upper()} {row["id"]} | {row["group_id"]} | variant {row.get("variant","-")} | {row.get("lighting_domain","camera")}'
        draw.text((8,base_y+2),label,fill='white',font=font)
        for condition,tensor in enumerate((x,stress)):
            preds=[predict(model,tensor,axis) for model in models]
            rgb=(tensor[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            im=Image.fromarray(rgb)
            w,h=im.size
            center_y=float(row['top'][1]) if np.isfinite(row['top'][1]) else .65
            # Truth-centered fixed 160px@640 crop, shared by before/after and stress.
            crop_size=min(w,h,round(180*w/640))
            left=max(0,min(w-crop_size,round(axis*w-crop_size/2)))
            top=max(0,min(h-crop_size,round(center_y*h-crop_size/2)))
            box=(left,top,left+crop_size,top+crop_size)
            context=overlay(im,row,preds,axis).resize((200,150))
            detail=overlay(im,row,preds,axis,box).resize((180,180))
            origin=condition*520
            canvas.paste(context,(origin+8,base_y+40))
            canvas.paste(detail,(origin+216,base_y+25))
            draw.text((origin+8,base_y+23),'ORIGINAL' if condition==0 else f'EXTREME SHADOW seed {args.seed}',fill='white',font=font)
            # Label and error text sit below both panels and do not hide pixels.
            draw.text((origin+8,base_y+193),'Before '+error_text(row,preds[0]),fill='#45ddff',font=font)
            draw.text((origin+8,base_y+211),'After   '+error_text(row,preds[1]),fill='#ffb52e',font=font)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output,format='JPEG',quality=94,subsampling=0)
    print(f'Saved {output}: {len(chosen)} validation samples, {canvas.width}x{canvas.height}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('real-manifest','sim-dataset','before','after','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--split',choices=['validation'],default='validation')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',type=int,default=9721)
    render(parser.parse_args())


if __name__=='__main__':main()
