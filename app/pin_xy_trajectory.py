"""Preflight and supervise a height-locked nudge using the existing retimed spline.

Pure math only: no controller, device, transport, or motor command is created.
Construct this wrapper off the control thread. The caller must use the exact
prevalidated origin and independently check fresh state and collision safety.
"""
from __future__ import annotations
import numpy as np
from remote_inference import SmoothActionTrajectory
from pin_regrasp_motion import rotation_log

MAX_HEIGHT_ERROR_M = .00001
MAX_ORIENTATION_ERROR_RAD = .01
MAX_TRAVEL_M = .005
FK_TRAVEL_MARGIN_M = .0001
PREFLIGHT_SAMPLES_PER_SEGMENT = 21


def _vector(value,size,name):
    array=np.asarray(value,dtype=float)
    if array.shape!=(size,) or not np.isfinite(array).all():raise ValueError('invalid_'+name)
    return array.copy()


def _bezier(controls,u):
    work=np.asarray(controls,dtype=float).copy()
    for size in range(len(work)-1,0,-1):work[:size]=(1-u)*work[:size]+u*work[1:size+1]
    return work[0]


class HeightLockedNudgeTrajectory:
    """Same bounded quintic trajectory, checked before planning and every emission.

    The checked path includes the actual quintic Bezier segments, not merely
    joint-linear waypoint connections. The emitted command also needs an FK
    check because per-joint deadline/speed clipping can leave the spline.
    """
    def __init__(self,waypoints14,started,kinematics,anchor_position,anchor_rotation):
        path=np.asarray(waypoints14,dtype=float)
        if path.ndim!=2 or path.shape[1]!=14 or not 2<=len(path)<=51 or not np.isfinite(path).all():
            raise ValueError('expected_2_to_51_finite_14D_waypoints')
        if not np.isfinite(started):raise ValueError('invalid_trajectory_start')
        if not np.array_equal(path[:,6:],np.broadcast_to(path[0,6:],path[:,6:].shape)):
            raise ValueError('height_locked_nudge_changes_held_channels')
        self.kinematics=kinematics
        self.anchor_position=_vector(anchor_position,3,'anchor_position')
        self.anchor_rotation=np.asarray(anchor_rotation,dtype=float).copy()
        if (self.anchor_rotation.shape!=(3,3) or not np.isfinite(self.anchor_rotation).all()
                or not np.allclose(self.anchor_rotation.T@self.anchor_rotation,np.eye(3),atol=1e-6,rtol=0)
                or abs(np.linalg.det(self.anchor_rotation)-1)>1e-6):
            raise ValueError('invalid_anchor_rotation')
        self.waypoints14=path.copy();self._held=path[0,6:].copy()
        self._smooth=SmoothActionTrajectory(path[1:],float(started))
        # Never let tick implicitly choose a different origin after preflight.
        self._smooth._plan(path[0].copy())
        self._last_emitted=path[0].copy();self._last_now=None;self._emitted_travel=0.
        self._last_position=self.anchor_position.copy()
        self._preflight()

    def _pose_errors(self,command):
        position,rotation=self.kinematics.pose(command[:6])
        position=_vector(position,3,'predicted_position');rotation=np.asarray(rotation,float)
        if rotation.shape!=(3,3) or not np.isfinite(rotation).all():raise ValueError('invalid_predicted_rotation')
        height=abs(float(position[2]-self.anchor_position[2]))
        angle=float(np.linalg.norm(rotation_log(rotation@self.anchor_rotation.T)))
        travel=float(np.linalg.norm(position-self.anchor_position))
        if not np.isfinite([height,angle,travel]).all():raise ValueError('invalid_predicted_pose')
        return position,height,angle,travel

    def _check_command(self,command,*,runtime=False):
        exception=RuntimeError if runtime else ValueError
        if command.shape!=(14,) or not np.isfinite(command).all():raise exception('invalid_height_locked_command')
        if not np.allclose(command[6:],self._held,atol=1e-12,rtol=0):raise exception('height_locked_nudge_changes_held_channels')
        # De Casteljau can perturb a mathematically constant decimal channel by
        # one ulp. Normalize that roundoff, never an actual held-joint change.
        command[6:]=self._held
        try:position,height,angle,travel=self._pose_errors(command)
        except (ValueError,TypeError,np.linalg.LinAlgError) as exc:raise exception(str(exc)) from exc
        if height>MAX_HEIGHT_ERROR_M+1e-12:raise exception('height_locked_nudge_height_violation')
        if angle>MAX_ORIENTATION_ERROR_RAD+1e-12:raise exception('height_locked_nudge_orientation_violation')
        if travel>MAX_TRAVEL_M+FK_TRAVEL_MARGIN_M:raise exception('height_locked_nudge_travel_violation')
        return position,height,angle,travel

    def _preflight(self):
        origin,_,origin_angle,_=self._check_command(self.waypoints14[0])
        if np.linalg.norm(origin-self.anchor_position)>1e-6 or origin_angle>1e-6:
            raise ValueError('height_locked_nudge_anchor_origin_mismatch')
        endpoint_positions=[]
        for command in self.waypoints14:
            position,_,_,_=self._check_command(command);endpoint_positions.append(position)
        waypoint_travel=float(np.linalg.norm(np.diff(endpoint_positions,axis=0),axis=1).sum())
        if waypoint_travel>MAX_TRAVEL_M+FK_TRAVEL_MARGIN_M:raise ValueError('height_locked_nudge_path_travel_violation')
        if not np.array_equal(self._smooth.controls[:,:,6:],np.broadcast_to(self._held,self._smooth.controls[:,:,6:].shape)):
            raise ValueError('height_locked_nudge_changes_held_channels')
        previous=origin;length=0.;max_height=max_angle=max_radius=0.;samples=0
        # Controls are the exact controls generated by the existing speed and
        # acceleration-retimed planner. De Casteljau matches its tick evaluator.
        for controls in self._smooth.controls:
            for u in np.linspace(0,1,PREFLIGHT_SAMPLES_PER_SEGMENT):
                command=_bezier(controls,float(u))
                if not np.allclose(command[6:],self._held,atol=1e-12,rtol=0):raise ValueError('height_locked_nudge_changes_held_channels')
                command[6:]=self._held
                position,height,angle,radius=self._check_command(command)
                length+=float(np.linalg.norm(position-previous));previous=position
                max_height=max(max_height,height);max_angle=max(max_angle,angle);max_radius=max(max_radius,radius);samples+=1
        if length>MAX_TRAVEL_M+FK_TRAVEL_MARGIN_M:raise ValueError('height_locked_nudge_spline_travel_violation')
        self.preflight_metadata={'height_locked':True,'trajectory_kind':'existing_retimed_piecewise_quintic_bezier',
            'exact_origin_preplanned':True,'waypoint_count':len(self.waypoints14),'segment_count':len(self._smooth.controls),
            'samples_per_segment':PREFLIGHT_SAMPLES_PER_SEGMENT,'spline_samples':samples,
            'max_spline_height_error_m':max_height,'height_error_limit_m':MAX_HEIGHT_ERROR_M,
            'max_spline_orientation_error_rad':max_angle,'orientation_error_limit_rad':MAX_ORIENTATION_ERROR_RAD,
            'max_spline_radius_m':max_radius,'spline_path_length_m':length,'waypoint_path_length_m':waypoint_travel,
            'max_travel_m':MAX_TRAVEL_M,'travel_fk_margin_m':FK_TRAVEL_MARGIN_M,
            'held_channels_unchanged':True,'duration_s':self._smooth.duration,
            'runtime_emitted_command_fk_watchdog':True,'sampled_preflight_not_continuous_cartesian_proof':True,
            'motor_commands_issued':False}

    @property
    def started(self):return self._smooth.started

    @started.setter
    def started(self,value):
        if not np.isfinite(value):raise ValueError('invalid_trajectory_start')
        self._smooth.started=float(value)

    def reset_started(self,now):
        self.started=now;self._smooth.index=0;self._last_now=None
        self._last_emitted=self.waypoints14[0].copy();self._last_position=self.anchor_position.copy();self._emitted_travel=0.

    @property
    def duration(self):return self._smooth.duration

    @property
    def speed(self):return self._smooth.speed

    @property
    def acceleration(self):return self._smooth.acceleration

    def remaining(self,now):return self._smooth.remaining(now)

    def policy_clock(self,now):return self._smooth.policy_clock(now)

    def tick(self,now,dt,target):
        if not np.isfinite(now) or (self._last_now is not None and now<self._last_now):
            raise RuntimeError('height_locked_nudge_clock_regressed')
        current=np.asarray(target,dtype=float)
        if current.shape!=(14,) or not np.isfinite(current).all():raise RuntimeError('invalid_height_locked_command')
        if not np.allclose(current,self._last_emitted,rtol=0,atol=1e-10):
            raise RuntimeError('height_locked_nudge_origin_or_command_changed')
        emitted,done=self._smooth.tick(now,dt,current)
        if not np.allclose(emitted[6:],self._held,atol=1e-12,rtol=0):raise RuntimeError('height_locked_nudge_changes_held_channels')
        emitted[6:]=self._held
        position,_,_,_=self._check_command(emitted,runtime=True)
        distance=self._emitted_travel+float(np.linalg.norm(position-self._last_position))
        if distance>MAX_TRAVEL_M+FK_TRAVEL_MARGIN_M:raise RuntimeError('height_locked_nudge_emitted_travel_violation')
        self._emitted_travel=distance;self._last_position=position;self._last_emitted=emitted.copy();self._last_now=float(now)
        return emitted,done
