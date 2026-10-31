#!/usr/bin/env python3
"""Score visual state goodness from three images; no recording metadata required.

Use a trusted goodness checkpoint. The score measures learned demonstration support
and reviewed relative visual quality, not calibrated current-policy completion odds.
"""
import argparse,json
from pathlib import Path
import torch
from state_goodness_train import Goodness,Images


def load_model(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('kind')!='visual_state_goodness':raise ValueError('expected a visual_state_goodness checkpoint')
    config=checkpoint['config'];model=Goodness(config['backbone'],pretrained=False,freeze_encoder=config.get('freeze_encoder',False),camera_mode=config.get('camera_mode','all')).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint


def score_images(model,checkpoint,image_groups,device,image_root=Path('.')):
    reader=Images(image_root,int(checkpoint['config']['image_size']),False)
    for refs in image_groups:
        if len(refs)!=3:raise ValueError('exactly three ordered camera images required: overhead,wrist_port2,wrist_port3')
    with torch.inference_mode():scores=torch.sigmoid(model(torch.stack([reader(refs) for refs in image_groups]).to(device))).cpu().tolist()
    return [{'state_goodness_uncalibrated':float(score),'interpretation':'Learned demonstrated-state support / relative visual quality; not calibrated current-policy completion probability.'} for score in scores]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);source=ap.add_mutually_exclusive_group(required=True);source.add_argument('--images',nargs=3,type=Path,metavar=('OVERHEAD','WRIST_PORT2','WRIST_PORT3'));source.add_argument('--manifest',type=Path,help='JSONL rows need only images triplet; optional id is echoed')
    ap.add_argument('--image-root',type=Path,default=Path('.'));ap.add_argument('--output',type=Path);ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');ap.add_argument('--gpu-memory-gb',type=float,default=4.);ap.add_argument('--cpu-threads',type=int,default=4)
    args=ap.parse_args()
    if args.batch_size<1 or args.gpu_memory_gb<=0:ap.error('positive batch size and memory limit required')
    torch.set_num_threads(args.cpu_threads);device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(.95,args.gpu_memory_gb*1024**3/torch.cuda.get_device_properties(device).total_memory))
    model,checkpoint=load_model(args.checkpoint,device);rows=[{'images':args.images}] if args.images else [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    if not rows:ap.error('empty manifest')
    results=[]
    for start in range(0,len(rows),args.batch_size):
        selected=rows[start:start+args.batch_size];scores=score_images(model,checkpoint,[r['images'] for r in selected],device,args.image_root)
        for row,score in zip(selected,scores):
            if 'id' in row:score['id']=row['id']
            results.append(score)
    content=''.join(json.dumps(r,allow_nan=False)+'\n' for r in results)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(content)
    else:print(content,end='')


if __name__=='__main__':main()
