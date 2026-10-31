#!/usr/bin/env python3
"""Persistent JSONL informational critic. Never sends robot commands."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

CAMERA_ORDER = ('overhead', 'wrist_port2', 'wrist_port3')


def resample_actions(state, actions, fps=20):
    """Interpolate next-pose targets from measured t=0 to 5 future samples."""
    import numpy as np
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if state.shape != (14,) or actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError('expected state14 and actions Nx14')
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError('nonfinite state/actions')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if len(actions) / fps < 1 - 1e-9:
        return None
    times = np.arange(len(actions) + 1) / fps
    values = np.concatenate((state[None], actions))
    return np.stack([np.interp(np.arange(1, 6) / 5, times, values[:, i]) for i in range(14)], axis=1).astype(np.float32)


class Scorer:
    def __init__(self, checkpoint, device):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from action_critic_train import ActionCritic, choose_device, load_teacher
        from torchvision import transforms as T
        torch.set_num_threads(2)
        self.device = choose_device(device)
        path = Path(checkpoint).resolve()
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        config = bundle['config']
        if config['state_dim'] != 14 or tuple(config['action_shape']) != (5, 14):
            raise ValueError('checkpoint requires incompatible state/action shape')
        if tuple(config['camera_order']) != CAMERA_ORDER:
            raise ValueError('checkpoint camera order mismatch')
        self.teacher = load_teacher(bundle['teacher'], self.device)
        self.model = ActionCritic(14, (5,14), config['hidden_dim']).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(bundle['model'])
        self.norm = {key: value.to(self.device) for key,value in bundle['normalization'].items()}
        self.transform = T.Compose([T.Resize((config['image_size'],config['image_size'])), T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.metadata = {'checkpoint':str(path), 'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'teacher_sha256':bundle.get('teacher_sha256'), 'manifest_sha256':bundle.get('manifest_sha256'), 'action_provenance':config.get('action_provenance'), 'device':str(self.device), 'uncalibrated':True, 'warning':bundle.get('warning'), 'scope':'whole proposed first 1s prediction; not executed prefix', 'horizon_s':1.0, 'timing_convention':'nominal policy seconds; physical execution may be slower', 'image_preprocessing':'full uncropped 4:3 RGB resized to teacher input', 'camera_order':list(CAMERA_ORDER)}

    def score(self, request):
        import torch
        from PIL import Image
        started = time.perf_counter()
        actions = resample_actions(request['state'], request['actions'], float(request.get('fps',20)))
        if actions is None:
            return {'status':'unavailable','reason':'proposal has less than 1s horizon', **self.metadata}
        views=[]
        for name in CAMERA_ORDER:
            with Image.open(io.BytesIO(base64.b64decode(request['images'][name], validate=True))) as image:
                if image.size not in ((640, 480), (320, 240)):
                    raise ValueError(f'{name}: critic requires full uncropped 640x480 or 320x240 frame')
                views.append(self.transform(image.convert('RGB')))
        norm=self.norm
        with torch.inference_mode():
            images=torch.stack(views)[None].to(self.device)
            z=self.teacher.encode(images)
            v=torch.sigmoid(self.teacher.value(z)).flatten().item()
            state=torch.tensor(request['state'],dtype=torch.float32,device=self.device)[None]
            action=torch.tensor(actions,device=self.device)[None]
            dt=torch.tensor([1.],device=self.device)
            q=self.model(z,(state-norm['state_mean'])/norm['state_std'],(action-norm['action_mean'])/norm['action_std'],(dt-norm['dt_mean'])/norm['dt_std']).item()
        if not all(math.isfinite(x) for x in (q,v)):
            raise ValueError('nonfinite critic output')
        return {'status':'ok','q':q,'current_v':v,'delta':q-v,'latency_ms':(time.perf_counter()-started)*1000, **self.metadata}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    args=parser.parse_args()
    scorer=Scorer(args.checkpoint,args.device)
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            result=scorer.score(request)
        except Exception as exc:
            print(f'critic request failed: {exc}',file=sys.stderr,flush=True)
            result={'status':'error','reason':str(exc)[:500]}
        result.update({key:request.get(key) for key in ('request_id','generation')})
        print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':
    main()
