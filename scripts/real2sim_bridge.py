"""Native 14-channel real/demo bridge; no hardware or simulator project imports.

Gripper conversion is an explicit endpoint approximation, not jaw calibration.
Arm channels remain unchanged. Out-of-range poses are measurable without clipping.
"""
from pathlib import Path
import numpy as np
import mujoco

DEFAULT_SCENE = Path('/home/robotics/Desktop/RobotArmTraining/sim/actuator/scene_bimanual.xml')
ARM_COLUMNS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP_COLUMNS = np.array([6, 13])


def vector14(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError('Expected a finite vector with shape (14,)')
    return value.copy()


class Real2SimBridge:
    def __init__(self, scene=DEFAULT_SCENE, gripper_closed_rad=0., gripper_open_rad=2., gripper_open_m=.04):
        endpoints = np.asarray((gripper_closed_rad, gripper_open_rad), dtype=float)
        if endpoints.shape == (2,):
            endpoints = np.tile(endpoints, (2, 1))
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all() or np.any(endpoints[:, 1] <= endpoints[:, 0]):
            raise ValueError('Gripper range must be increasing finite (2,) or (2,2) endpoints')
        if not np.isfinite(gripper_open_m) or not 0 < gripper_open_m <= .04:
            raise ValueError('Per-finger maximum must be in (0,.04] metres')
        self.gripper_rad_range, self.gripper_open_m = endpoints, float(gripper_open_m)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.qadr, self.ctrl_ids, self.finger_other, self.sites = [], [], [], []
        for prefix in ('', 'right_'):
            names = [prefix + f'joint{i}' for i in range(1, 7)] + [prefix + 'L_finger_joint']
            self.qadr.extend(self.model.jnt_qposadr[[self.model.joint(n).id for n in names]])
            self.ctrl_ids.extend([self.model.actuator(n).id for n in names[:6]] + [self.model.actuator(prefix+'gripper').id])
            self.finger_other.append(self.model.jnt_qposadr[self.model.joint(prefix+'R_finger_joint').id])
            self.sites.append(self.model.site(prefix+'grip_site').id)
        self.qadr, self.ctrl_ids = np.array(self.qadr), np.array(self.ctrl_ids)
        self.arm_qadr = self.qadr.reshape(2, 7)[:, :6].copy()
        self.finger_qadr = np.column_stack((self.qadr[GRIP_COLUMNS], self.finger_other))
        self.base_ids = (0, self.model.body('right_base').id)
        self.limits = self.model.actuator_ctrlrange[self.ctrl_ids].copy()
        self.limits[GRIP_COLUMNS, 1] = self.gripper_open_m
        self.arm_ranges = self.limits.reshape(2, 7, 2)[:, :6].copy()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        mujoco.mj_forward(self.model, self.data)

    def real_to_sim(self, value):
        value = vector14(value)
        low, high = self.gripper_rad_range.T
        value[GRIP_COLUMNS] = (value[GRIP_COLUMNS]-low)/(high-low)*self.gripper_open_m
        return value

    def sim_to_real(self, value):
        value = vector14(value)
        low, high = self.gripper_rad_range.T
        value[GRIP_COLUMNS] = low+value[GRIP_COLUMNS]/self.gripper_open_m*(high-low)
        return value

    def violations(self, value, *, real=True):
        value = self.real_to_sim(value) if real else vector14(value)
        lower, upper = self.limits.T
        delta = np.maximum(lower-value, 0)+np.maximum(value-upper, 0)
        return [dict(index=int(i), value=float(value[i]), lower=float(lower[i]), upper=float(upper[i]), excess=float(delta[i])) for i in np.flatnonzero(delta)]

    def state(self, *, real=False):
        value = self.data.qpos[self.qadr].copy()
        return self.sim_to_real(value) if real else value

    def set_kinematic_pose(self, value, *, real=True):
        value = self.real_to_sim(value) if real else vector14(value)
        self.data.qpos[self.qadr] = value
        self.data.qpos[self.finger_other] = -value[GRIP_COLUMNS]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.tcp_base()

    def tcp_base(self):
        result = {}
        for side, site, base in zip(('left', 'right'), self.sites, self.base_ids):
            rotation = self.data.xmat[base].reshape(3, 3)
            result[side] = dict(position=rotation.T@(self.data.site_xpos[site]-self.data.xpos[base]),
                                rotation=rotation.T@self.data.site_xmat[site].reshape(3, 3))
        return result

    def initialize_dynamic(self, value, *, real=True):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        self.set_kinematic_pose(value, real=real)
        self.data.ctrl[self.ctrl_ids] = self.state()
        self.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, value, *, real=True, duration=.05):
        target = self.real_to_sim(value) if real else vector14(value)
        violations = self.violations(target, real=False)
        if violations:
            raise ValueError(f'Action exceeds simulation limits: {violations}')
        if not np.isfinite(duration):
            raise ValueError("Duration must be finite")
        ticks = round(duration/self.model.opt.timestep)
        if not np.isfinite(duration) or ticks < 1 or not np.isclose(ticks*self.model.opt.timestep, duration):
            raise ValueError('Duration must be a positive integer number of physics ticks')
        start = self.data.ctrl[self.ctrl_ids].copy()
        for tick in range(1, ticks+1):
            self.data.ctrl[self.ctrl_ids] = start+(target-start)*(tick/ticks)
            mujoco.mj_step(self.model, self.data)
            if not all(np.isfinite(v).all() for v in (self.data.qpos, self.data.qvel, self.data.qacc)):
                raise RuntimeError('Nonfinite MuJoCo state')
        return self.state()

    def contact_summary(self):
        def kind(geom):
            body = self.model.geom_bodyid[geom]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
            while body:
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
                if bname == 'right_base': return 'right'
                if bname == 'link1': return 'left'
                body = self.model.body_parentid[body]
            if name == 'table_top': return 'table'
            return 'object'
        result = []
        for contact in self.data.contact:
            if contact.dist >= 0: continue
            a, b = kind(contact.geom1), kind(contact.geom2)
            robots = {'left', 'right'}
            category = ('inter_arm' if {a,b} == robots else 'robot_table' if 'table' in (a,b) and (a in robots or b in robots) else 'robot_object' if (a in robots) != (b in robots) else 'robot_self' if a in robots and b in robots else 'object')
            result.append(dict(category=category, distance_m=float(contact.dist), geom1=int(contact.geom1), geom2=int(contact.geom2)))
        return dict(count=len(result), maximum_penetration_m=max((-c['distance_m'] for c in result), default=0.),
                    counts={category: sum(c['category'] == category for c in result) for category in ('inter_arm','robot_table','robot_object','robot_self','object')}, contacts=result)

    map_state = real_to_sim
    inverse_state = sim_to_real
    set_pose = set_kinematic_pose
    reset_dynamic = initialize_dynamic

    def native_state(self):
        return self.state(real=True)

    def local_tcp(self):
        poses = self.tcp_base()
        return (np.stack([poses[s]['position'] for s in ('left','right')]),
                np.stack([poses[s]['rotation'] for s in ('left','right')]))

    def step_action(self, native14, dt=.05):
        return self.step(native14, duration=dt)
