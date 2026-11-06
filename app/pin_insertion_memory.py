"""Explicit stationary-carrier hole memory; no motor or contact decisions.

A fixed right-wrist/carrier relationship is a caller-declared assumption, never
inferred from gripper encoders or an uncalibrated CNN score. Only designated,
externally trusted visible initialization observations seed the memory. The pin
always comes from the current observation. No optical flow or metric warp is used.
"""
from __future__ import annotations
import math
import numpy as np


class PinInsertionHoleMemory:
    def __init__(self, *, min_seed_frames=3, max_seed_frames=32,
                 max_memory_age_ns=30_000_000_000,max_input_age_ns=250_000_000,
                 max_frame_gap_ns=None,max_disagreement_px=32.,uncertainty_growth_px_per_s=.25):
        if type(min_seed_frames) is not int or type(max_seed_frames) is not int or not 1<=min_seed_frames<=max_seed_frames:
            raise ValueError('invalid seed frame counts')
        for value in (max_memory_age_ns,max_input_age_ns):
            if type(value) is not int or value<=0:raise ValueError('time limits must be positive integer nanoseconds')
        if max_frame_gap_ns is not None and (type(max_frame_gap_ns) is not int or max_frame_gap_ns<=0):raise ValueError('frame gap must be None or positive integer nanoseconds')
        if not math.isfinite(max_disagreement_px) or max_disagreement_px<=0 or not math.isfinite(uncertainty_growth_px_per_s) or uncertainty_growth_px_per_s<0:raise ValueError('invalid uncertainty/disagreement limits')
        self.min_seed=min_seed_frames;self.max_seed=max_seed_frames;self.horizon=max_memory_age_ns
        self.max_input_age=max_input_age_ns;self.max_gap=max_frame_gap_ns;self.disagreement=max_disagreement_px
        self.growth=uncertainty_growth_px_per_s
        self.reset()

    def reset(self):
        self._context=None;self._last_ns=None;self._seeds=[];self._hole=None;self._source_ns=None
        self._uncertainty=None;self._dispersion=None;self._initializing=False

    def observe(self,keypoints,*,observation_ns,now_ns,episode_id,grasp_id,camera_id,sequence_id,
                image_size,grasp_locked,hole_visible,hole_trusted,initialize=False,hole_uncertainty_px=None):
        result={'keypoints':[None]*4,'memory_available':False,'hole_source':'unavailable',
            'source_ns':None,'age_ns':None,'uncertainty_px':None,'uncertainty_kind':'uncalibrated stationary-prior heuristic',
            'seed_count':0,'seed_dispersion_px':None,'stationary_drift_allowance_px':None,'reset_reason':None,'gate_positive_from_memory':False,'motor_action':False,
            'stationary_carrier_prior':True,'current_pin_source':'current CNN observation'}
        try:
            for value in (observation_ns,now_ns):
                if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<=0:raise ValueError('positive monotonic integer timestamps required')
            if not 0<=now_ns-observation_ns<=self.max_input_age:raise ValueError('observation stale or from future')
            identifiers=(episode_id,grasp_id,camera_id,sequence_id)
            if any(not isinstance(value,str) or not value for value in identifiers):raise ValueError('explicit episode/grasp/camera/sequence IDs required')
            if any(type(value) is not bool for value in (grasp_locked,hole_visible,hole_trusted,initialize)):raise ValueError('explicit boolean trust/visibility/grasp declarations required')
            wh=np.asarray(image_size,dtype=float)
            if wh.shape!=(2,) or not np.isfinite(wh).all() or (wh<16).any() or not np.equal(wh,np.floor(wh)).all():raise ValueError('image_size must be full-frame width,height')
            kp=np.asarray(keypoints,dtype=float)
            if kp.shape!=(4,) or not np.isfinite(kp[:2]).all() or (kp[:2]<0).any() or (kp[:2]>1).any():raise ValueError('finite normalized current pin coordinates required')
            result['keypoints'][:2]=kp[:2].tolist()
            if hole_uncertainty_px is not None and (isinstance(hole_uncertainty_px,bool) or not math.isfinite(hole_uncertainty_px) or hole_uncertainty_px<0):raise ValueError('hole uncertainty must be finite nonnegative pixels')
            if (hole_visible or hole_trusted) and (not np.isfinite(kp[2:]).all() or (kp[2:]<0).any() or (kp[2:]>1).any()):raise ValueError('trusted/visible hole coordinates must be finite and normalized')
            if hole_trusted and not hole_visible:raise ValueError('hole trust requires explicit visible evidence')
            context=identifiers+tuple(wh.astype(int).tolist())
            reason=None
            if self._context is not None and context!=self._context:reason='episode/grasp/camera/sequence/image geometry changed'
            elif self._last_ns is not None and observation_ns<=self._last_ns:raise ValueError('timestamp regressed or repeated')
            elif self._last_ns is not None and self.max_gap is not None and observation_ns-self._last_ns>self.max_gap:reason='configured observation gap exceeded'
            if reason:self.reset();result['reset_reason']=reason
            self._context=context;self._last_ns=int(observation_ns)
            if not grasp_locked:
                self.reset();result['reset_reason']='stationary right-wrist/carrier grasp assumption not declared';return result
            if self._source_ns is not None and now_ns-self._source_ns>self.horizon:
                self._seeds=[];self._hole=None;self._source_ns=None;self._uncertainty=None
                result['reset_reason']='memory horizon exceeded'
            # A new explicit initialization window replaces an old reference rather
            # than accumulating observations across unrelated seed windows.
            if initialize and not self._initializing:
                self._seeds=[];self._hole=None;self._source_ns=None;self._uncertainty=None
            self._initializing=initialize
            if initialize and hole_visible and hole_trusted:
                self._seeds.append((kp[2:].copy(),int(observation_ns),hole_uncertainty_px))
                self._seeds=self._seeds[-self.max_seed:]
                if len(self._seeds)>=self.min_seed:
                    points=np.stack([entry[0] for entry in self._seeds]);center=np.median(points,axis=0)
                    spread=float(np.median(np.linalg.norm((points-center)*wh,axis=1)))
                    supplied=[float(entry[2]) for entry in self._seeds if entry[2] is not None]
                    self._hole=center;self._source_ns=max(entry[1] for entry in self._seeds)
                    self._dispersion=spread
                    self._uncertainty=max(spread,float(np.median(supplied))) if supplied else None
            elif self._hole is not None and hole_visible and hole_trusted:
                if np.linalg.norm((kp[2:]-self._hole)*wh)>self.disagreement:
                    self._seeds=[];self._hole=None;self._source_ns=None;self._uncertainty=None
                    result['reset_reason']='trusted visible hole disagrees with stationary reference'
            result['seed_count']=len(self._seeds)
            if self._hole is not None:
                age=int(now_ns-self._source_ns)
                result.update(memory_available=True,hole_source='remembered_stationary_prior',source_ns=self._source_ns,
                    age_ns=age,uncertainty_px=self._uncertainty+self.growth*age/1e9 if self._uncertainty is not None else None,
                    seed_dispersion_px=self._dispersion,stationary_drift_allowance_px=self.growth*age/1e9)
                result['keypoints'][2:]=self._hole.tolist()
            else:result['reason']='explicit trusted initialization window required'
        except (ValueError,TypeError,OverflowError) as exc:
            self.reset();result['reset_reason']=str(exc)
        return result


HoleMemory=PinInsertionHoleMemory
