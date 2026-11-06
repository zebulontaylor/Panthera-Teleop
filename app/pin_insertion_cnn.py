"""RGB/proprioceptive pin alignment model and purely advisory bounded adapter.

State14 convention: six left joints, left gripper, six right joints, right gripper. No motor
interface is imported or called. Checkpoints must come from a trusted source.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = 'pin-insertion-rgb-both-arms-v2'


def joint_features(state):
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError('finite native state14 required; both measured arms must be present')
    return np.concatenate((value[:6], value[7:13]))


class SpatialPool(nn.Module):
    """Adaptive 4x4 averages without nondeterministic CUDA pooling backward."""
    def forward(self, value):
        h, w = value.shape[-2:]
        return torch.stack([
            torch.stack([value[..., i*h//4:((i+1)*h+3)//4,
                                    j*w//4:((j+1)*w+3)//4].mean(dim=(-2, -1))
                         for j in range(4)], dim=-1)
            for i in range(4)], dim=-2)


class PinInsertionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 48, 3, 2, 1),
            nn.ReLU(), nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
            SpatialPool(), nn.Flatten())
        self.fusion = nn.Sequential(nn.Linear(1024 + 24, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.keypoints = nn.Linear(64, 4)
        self.delta = nn.Linear(64, 3)
        self.gate = nn.Linear(64, 1)

    def forward(self, image, joints):
        features = self.fusion(torch.cat((self.vision(image), joints.sin(), joints.cos()), dim=1))
        return {'keypoints': self.keypoints(features).sigmoid(),
                'delta_xyz_m': self.delta(features) * .001, 'active_logit': self.gate(features)}


class HeatmapPinInsertionCNN(nn.Module):
    """Stride-four landmark likelihoods with spatial expectation readout.

    The convolutional score head retains image position; proprioception conditions
    the visual channels and the global gate/delta heads. Heatmap logits are opt-in
    for training so runtime outputs remain the original compact three fields.
    """
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 24, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 2, dilation=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 2, dilation=2), nn.ReLU())
        self.condition = nn.Linear(24, 128)
        self.score = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 2, 1))
        self.fusion = nn.Sequential(nn.Linear(1024+24,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.pool = SpatialPool()
        self.delta = nn.Linear(64,3)
        self.gate = nn.Linear(64,1)

    def forward(self, image, joints, *, return_logits=False):
        proprio = torch.cat((joints.sin(), joints.cos()),dim=1)
        visual = self.vision(image)
        scale,bias = self.condition(proprio).chunk(2,dim=1)
        visual = visual*(1+.1*scale.tanh()[:,:,None,None])+.1*bias.tanh()[:,:,None,None]
        logits = self.score(visual)
        n,_,h,w = logits.shape
        probability = logits.flatten(2).softmax(dim=-1).reshape(n,2,h,w)
        xs = (torch.arange(w,device=image.device,dtype=image.dtype)+.5)/w
        ys = (torch.arange(h,device=image.device,dtype=image.dtype)+.5)/h
        u = (probability.sum(dim=2)*xs).sum(dim=-1)
        v = (probability.sum(dim=3)*ys).sum(dim=-1)
        keypoints = torch.stack((u,v),dim=-1).flatten(1)
        global_features = self.fusion(torch.cat((self.pool(visual).flatten(1),proprio),dim=1))
        out = {'keypoints':keypoints,'delta_xyz_m':self.delta(global_features)*.001,
               'active_logit':self.gate(global_features)}
        if return_logits:
            out['keypoint_logits'] = logits
        return out


def build_model(variant='regression'):
    if variant == 'regression': return PinInsertionCNN()
    if variant == 'heatmap': return HeatmapPinInsertionCNN()
    raise ValueError(f'unknown model variant: {variant}')


class PinInsertionPredictor:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.metadata = payload['metadata']
        if self.metadata.get('schema') != SCHEMA or self.metadata.get('controlled_arm') != 'left' or self.metadata.get('coordinate_frame') != 'right_base':
            raise ValueError('checkpoint schema mismatch')
        self.model = build_model(self.metadata.get('model_variant','regression')).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval()

    def predict(self, image, state):
        rgb = np.asarray(image)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16:
            raise ValueError('image must be RGB uint8 HWC, at least 16x16')
        source = self.metadata.get('dataset_metadata', {})
        if source.get('image_width') and source.get('image_height'):
            expected_aspect = source['image_width'] / source['image_height']
            if abs((rgb.shape[1] / rgb.shape[0]) / expected_aspect - 1) > .02:
                raise ValueError('use the full uncropped camera frame with the training aspect ratio')
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(self.device) / 255
        size = self.metadata.get('image_size', 96)
        tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)
        joints = torch.from_numpy(joint_features(state).copy())[None].to(self.device)
        with torch.inference_mode():
            out = self.model(tensor, joints)
        return {key: out[key][0].cpu().numpy().tolist() for key in ('keypoints','delta_xyz_m','active_logit')}


class BoundedPinCorrection:
    """Return proposals only; even enabled mode never sends motor commands.

    Default shadow mode returns zero executable delta. Three consecutive fresh
    frames above the entry gate are required; the exit gate provides hysteresis.
    Timestamps use one monotonic clock. Invalid frames clear persistence.
    """
    def __init__(self, predictor, *, enabled=False, max_delta_m=.0005,
                 enter_probability=.8, exit_probability=.6, persistence=3,
                 max_age_ns=150_000_000, max_skew_ns=20_000_000):
        if not math.isfinite(max_delta_m) or not 0 < max_delta_m <= .0005:
            raise ValueError('delta cap must be positive and at most 0.5 mm')
        if not 0 <= exit_probability <= enter_probability <= 1 or persistence < 1:
            raise ValueError('invalid gate configuration')
        if max_age_ns <= 0 or max_skew_ns < 0:
            raise ValueError('invalid timestamp limits')
        self.predictor, self.enabled, self.cap = predictor, bool(enabled), max_delta_m
        self.enter, self.exit, self.persistence = enter_probability, exit_probability, persistence
        self.max_age, self.max_skew = max_age_ns, max_skew_ns
        self.reset()

    def reset(self):
        self._count, self._open, self._last_ns = 0, False, None

    def propose(self, image, state, *, frame_ns, measured_state_ns, now_ns):
        result = {'eligible': False, 'enabled': self.enabled, 'delta_xyz_m': [0., 0., 0.],
                  'proposed_delta_xyz_m': [0., 0., 0.], 'controlled_arm': 'left', 'coordinate_frame': 'right_base', 'requires_calibrated_left_base_transform': True}
        try:
            stamps = (frame_ns, measured_state_ns, now_ns)
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in stamps):
                raise ValueError('timestamps must be positive monotonic integers')
            if self._last_ns is not None and frame_ns <= self._last_ns:
                raise ValueError('frame timestamp must increase')
            if not 0 <= now_ns - frame_ns <= self.max_age or not 0 <= now_ns - measured_state_ns <= self.max_age:
                raise ValueError('stale or future input')
            if abs(frame_ns - measured_state_ns) > self.max_skew:
                raise ValueError('image/state timestamp skew')
            if self._last_ns is not None and frame_ns - self._last_ns > self.max_age:
                self._count, self._open = 0, False
            self._last_ns = frame_ns
            joint_features(state)
            prediction = self.predictor.predict(image, state)
            kp = np.asarray(prediction['keypoints'], dtype=float)
            delta = np.asarray(prediction['delta_xyz_m'], dtype=float)
            logits = np.asarray(prediction['active_logit'], dtype=float)
            if kp.shape != (4,) or delta.shape != (3,) or logits.size != 1 or not all(np.isfinite(v).all() for v in (kp, delta, logits)) or (kp < 0).any() or (kp > 1).any():
                raise ValueError('invalid model output')
            probability = 1 / (1 + math.exp(-float(np.clip(logits.item(), -60, 60))))
            if probability >= (self.exit if self._open else self.enter):
                self._count += 1
                self._open = self._count >= self.persistence
            else:
                self._count, self._open = 0, False
            delta *= min(1., self.cap / max(float(np.linalg.norm(delta)), 1e-30))
            result.update(eligible=self._open, keypoints=kp.tolist(), active_probability=probability,
                          proposed_delta_xyz_m=delta.tolist(), reason='eligible' if self._open else 'gate/persistence')
            if getattr(self.predictor, 'metadata', {}).get('direct_xyz_disabled', False):
                self._count, self._open = 0, False
                result.update(eligible=False, proposed_delta_xyz_m=[0., 0., 0.],
                              reason='localization_only_checkpoint_xyz_disabled')
                return result
            if self.enabled and self._open:
                result['delta_xyz_m'] = delta.tolist()
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
            self._count, self._open = 0, False
            result['reason'] = str(exc)
        return result
