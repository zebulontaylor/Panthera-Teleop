#!/usr/bin/env python3
"""Verify CUDA inference/backward on saved frames; never command robots or save weights."""
import argparse
from bisect import bisect_left
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--output', type=Path, default=root/'logs/alignment-calibration/cuda-verification.json')
    parser.add_argument('--model-root', type=Path, default=root.parent/'RobotArmTraining/outputs/panthera-assembly-cnns')
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.model_root.resolve()))
    from tools.gear_alignment_model import GearAlignmentPredictor
    from tools.pin_perception_model import PinPerceptionPredictor
    manifest = json.loads((root/'data/calibration/manual-labels/manifest.json').read_text())
    reference = next(row for row in manifest['observations'] if row['phase'] == 'pin_approach')
    episode = root/'data/kept'/reference['episode_id']
    frame_rows = [json.loads(line) for line in (episode/'frames.jsonl').read_text().splitlines()]
    states = [json.loads(line) for line in (episode/'states.jsonl').read_text().splitlines()]
    stamp = reference['images']['wrist_port2']['monotonic_ns']
    state = min(states, key=lambda row: abs(row['monotonic_ns']-stamp))
    overhead = min((row for row in frame_rows if row['camera'] == 'overhead'),
                   key=lambda row: abs(row['monotonic_ns']-stamp))
    paths = [episode/overhead['path'], Path(reference['images']['wrist_port3']['path']),
             Path(reference['images']['wrist_port2']['path'])]
    images = np.stack([np.asarray(Image.open(path).convert('RGB')) for path in paths])
    xyz = np.asarray([v for side in ('left','right') for v in state['arms'][side]['tool_pose_base']['position_m']], dtype=np.float32)
    report = dict(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_available=True, gpu=torch.cuda.get_device_name(0),
                  camera_order=['overhead','wrist_port3','wrist_port2'],
                  image_paths=[str(path.resolve()) for path in paths],
                  image_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                  python=sys.executable, tasks={}, model_weights_modified=False, robot_commands_sent=False)
    for task, cls, relative in [
            ('gear', GearAlignmentPredictor, 'gear_alignment/real_v4/adapted.pt'),
            ('pin', PinPerceptionPredictor, 'pin_perception/carrier_v4/adapted.pt')]:
        checkpoint = args.model_root/relative
        cpu = cls(checkpoint, device='cpu')
        gpu = cls(checkpoint, device='cuda')
        kwargs = dict(end_effector_positions=xyz) if task == 'pin' else {}
        expected = cpu.predict(images, **kwargs)
        actual = gpu.predict(images, **kwargs)
        maximum_difference = float(np.max(np.abs(expected['keypoints_2d']-actual['keypoints_2d'])))
        timings = {}
        for name, model in [('cpu',cpu),('cuda',gpu)]:
            for _ in range(2): model.predict(images, **kwargs)
            start = time.perf_counter()
            for _ in range(args.repeats): model.predict(images, **kwargs)
            timings[name] = (time.perf_counter()-start)*1000/args.repeats
        report['tasks'][task] = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            model_device=str(next(gpu.model.parameters()).device), mean_inference_ms=timings,
            max_cpu_cuda_coordinate_difference_px=maximum_difference,
            finite_outputs=bool(np.isfinite(actual['keypoints_2d']).all()))
        if maximum_difference > 2 or not report['tasks'][task]['finite_outputs']:
            raise RuntimeError(f'{task}: CUDA coordinate comparison failed: {maximum_difference}px')
        if task == 'gear':
            # Clone protects the loaded predictor and its buffers from train-mode changes.
            train_model = copy.deepcopy(gpu.model).train()
            output = train_model(torch.from_numpy(images.copy()).unsqueeze(0).to('cuda'))
            loss = sum(output[key].float().square().mean() for key in ('heatmap_logits','offsets','visibility_logits'))
            loss.backward()
            gradients = [p.grad for p in train_model.parameters() if p.grad is not None]
            report['cuda_backward'] = dict(loss=float(loss.detach()),
                gradient_tensors=len(gradients), finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                nonzero=any(bool(g.abs().max()>0) for g in gradients), weights_saved=False)
            if not report['cuda_backward']['finite'] or not report['cuda_backward']['nonzero']:
                raise RuntimeError('CUDA backward failed')
            del train_model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
