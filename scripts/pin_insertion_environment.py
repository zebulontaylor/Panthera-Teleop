"""Adapter to the EXISTING photo-profile pin insertion environment.

Never constructs the legacy opposed XML directly. Camera/layout/materials and
physical grasps belong to RobotArmTraining's existing environment. Native arm
angles pass through unchanged; native gripper radians remain an endpoint-only
approximation and are not jaw calibration.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import threading
import numpy as np

WORKSPACE_ROOT=Path(__file__).resolve().parents[1]
TRAINING_ROOT=WORKSPACE_ROOT.parent/'RobotArmTraining'
SETUP_PROFILE='photo_20261027'
EXPECTED_MUJOCO_VERSION='3.14.0'
DEFAULT_REGISTERED_PROFILE=WORKSPACE_ROOT/'config/pin-insertion-photo/registered-setup.json'
_VISUAL_BUILD_LOCK=threading.RLock()


def _activate_training_imports():
    if not TRAINING_ROOT.is_dir():raise FileNotFoundError(TRAINING_ROOT)
    if str(TRAINING_ROOT) not in sys.path:sys.path.insert(0,str(TRAINING_ROOT))


def create_photo_environment(*,seed=None,config=None,render_mode=None,visual_partition=True,carrier_visual_profile=None,setup='registered',registration_profile=None):
    """Return registered photo task; setup='nominal' is an explicit baseline.

    Accepted camera/planar-base registration is applied BEFORE original task
    reset. Original contact grasps and servos remain genuine; no fitted carrier
    object pose or native-radian gripper scaling is injected into physics.
    """
    if setup not in ('registered','nominal'):raise ValueError('setup must be registered or nominal')
    if carrier_visual_profile is None:carrier_visual_profile=(setup=='registered')
    _activate_training_imports()
    import mujoco
    if mujoco.__version__ != EXPECTED_MUJOCO_VERSION:
        raise RuntimeError(f'Existing photo physics requires mujoco=={EXPECTED_MUJOCO_VERSION}; got{mujoco.__version__}. See{TRAINING_ROOT}/requirements-actuator.txt')
    from sim.pin_insertion_env import BimanualPinInsertionEnv
    if visual_partition or carrier_visual_profile:
        import sim.bimanual_actuator_env as bimanual
        if __package__:
            from .photo_finger_visual_partition import apply_finger_visual_partition
        else:
            from photo_finger_visual_partition import apply_finger_visual_partition
        with _VISUAL_BUILD_LOCK:
            original=bimanual.apply_photo_setup;metadata=[];carrier_metadata=[]
            def photo_with_partition(root):
                profile=original(root)
                if visual_partition:metadata.append(apply_finger_visual_partition(root))
                if carrier_visual_profile:
                    if __package__:
                        from .photo_carrier_visual_profile import apply_carrier_visual_profile
                    else:
                        from photo_carrier_visual_profile import apply_carrier_visual_profile
                    carrier_metadata.append(apply_carrier_visual_profile(root))
                return profile
            bimanual.apply_photo_setup=photo_with_partition
            try:env=BimanualPinInsertionEnv(config=config,render_mode=render_mode,setup_profile=SETUP_PROFILE)
            finally:bimanual.apply_photo_setup=original
        env.photo_visual_partition=metadata[0] if metadata else None
        env.photo_carrier_visual_profile=carrier_metadata[0] if carrier_metadata else None
    else:
        with _VISUAL_BUILD_LOCK:env=BimanualPinInsertionEnv(config=config,render_mode=render_mode,setup_profile=SETUP_PROFILE)
        env.photo_visual_partition=None
        env.photo_carrier_visual_profile=None
    env.photo_registration=None
    if setup=='registered':apply_registered_photo_setup(env,registration_profile)
    if seed is not None:env.reset(seed=int(seed))
    return env


if __package__:
    from .real2sim_bridge import Real2SimBridge
else:
    from real2sim_bridge import Real2SimBridge


class PhotoSetupBridge(Real2SimBridge):
    """Native14 kinematic view of a live existing photo-profile task model/data."""
    def __init__(self,environment=None,*,seed=None,setup='registered',registration_profile=None,gripper_closed_rad=0.,gripper_open_rad=2.,gripper_open_m=.04):
        self.environment=environment or create_photo_environment(seed=seed,setup=setup,registration_profile=registration_profile)
        self.sim=self.environment.sim
        if self.sim.layout!='side_by_side' or self.sim.setup_profile!=SETUP_PROFILE:
            raise ValueError('PhotoSetupBridge requires existing side_by_side/photo_20261027 environment')
        super().__init__(model=self.sim.model,data=self.sim.data,gripper_closed_rad=gripper_closed_rad,
                         gripper_open_rad=gripper_open_rad,gripper_open_m=gripper_open_m)

    def set_kinematic_pose(self,value,*,real=True):
        result=super().set_kinematic_pose(value,real=real)
        self.data.ctrl[self.ctrl_ids]=self.state()
        self.sim.sync_control_state()
        return result

    def initialize_dynamic(self,*args,**kwargs):
        raise NotImplementedError('Use existing environment.reset(seed=...) and environment.step(action) for physical task control')

    def step(self,*args,**kwargs):
        raise NotImplementedError('Use existing environment.step(action) to preserve its task controller and physical action semantics')

    def grasp_transform(self,side,body_name):
        """Snapshot observed SIM tool<-object transform after actual task reset."""
        if side not in ('left','right'):raise ValueError('side must be left or right')
        index=('left','right').index(side);site=self.sites[index];body=self.model.body(body_name).id
        tool=np.eye(4);tool[:3,:3]=self.data.site_xmat[site].reshape(3,3);tool[:3,3]=self.data.site_xpos[site]
        obj=np.eye(4);obj[:3,:3]=self.data.xmat[body].reshape(3,3);obj[:3,3]=self.data.xpos[body]
        return np.linalg.inv(tool)@obj

    def provenance(self):
        _activate_training_imports()
        from sim.real_setup import PROFILE_PATH
        from sim.actuator_env import TABLE_Z,ASSETS
        import mujoco,gymnasium
        contract_path=WORKSPACE_ROOT/'config/pin-insertion-photo/geometry-contract.json'
        contract=json.loads(contract_path.read_text()) if contract_path.exists() else None
        profile=json.loads(PROFILE_PATH.read_text());texture=TRAINING_ROOT/profile['texture']['file']
        paths=[TRAINING_ROOT/'sim/bimanual_actuator_env.py',TRAINING_ROOT/'sim/pin_insertion_env.py',
               TRAINING_ROOT/'sim/real_setup.py',PROFILE_PATH,texture,TRAINING_ROOT/'requirements-actuator.txt',ASSETS/f'scene_bimanual_{SETUP_PROFILE}.xml']
        return dict(environment='sim.pin_insertion_env.BimanualPinInsertionEnv',setup_profile=SETUP_PROFILE,
            runtime_versions=dict(mujoco=mujoco.__version__,gymnasium=gymnasium.__version__,numpy=np.__version__),required_mujoco_version=EXPECTED_MUJOCO_VERSION,layout=self.sim.layout,source_root=str(TRAINING_ROOT),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            base_frames={s:dict(name=self.model.body(b).name,position_world_m=self.data.xpos[b].tolist(),rotation_world=self.data.xmat[b].reshape(3,3).tolist()) for s,b in zip(('left','right'),self.base_ids)},
            cameras={name:dict(parent=self.model.body(int(self.model.cam_bodyid[self.model.camera(name).id])).name,
                position_parent_m=self.model.cam_pos[self.model.camera(name).id].tolist(),fovy_deg=float(self.model.cam_fovy[self.model.camera(name).id])) for name in self.sim.cameras},
            tabletop_world_z_m=float(TABLE_Z),native_state_order='left joints1..6,gripper,right joints1..6,gripper (radians)',
            arm_mapping='exact joint angles, no offsets or retargeting',gripper_mapping='endpoint approximation only; not measured jaw calibration',
            geometry_contract=contract,carrier_visual_profile=getattr(self.environment,'photo_carrier_visual_profile',None),registered_setup=getattr(self.environment,'photo_registration',None),visual_partition=getattr(self.environment,'photo_visual_partition',None),physical_reset='existing task reset/contact implementation; adapter does not replace grasps',
            real_calibrated=False,camera_source='accepted own-finger image registration on existing photo environment' if getattr(self.environment,'photo_registration',None) else 'nominal photograph setup profile; approximate visual dimensions')

    def close(self):self.environment.close()

    # Override aliases defined on the parent so they dispatch to this adapter.
    set_pose=set_kinematic_pose
    reset_dynamic=initialize_dynamic


def apply_photo_camera_registration(environment,profile_path):
    """Apply selected own-finger CAMERA registration once; never scale physics jaws.

    The profile's gripper scaling applies only to native-radian replay mapping.
    Original task physics already uses metric finger travel/contact and retains it.
    """
    import mujoco
    path=Path(profile_path);profile=json.loads(path.read_text())
    if profile.get('rejected_as_final'):raise ValueError('Registration profile explicitly superseded/rejected')
    if profile.get('kind')!='own_right_finger_silhouette_registration_hypothesis':
        raise ValueError('Expected own-right-finger registration profile; old pin/table fits are not accepted')
    sim=environment.sim
    if sim.layout!='side_by_side' or sim.setup_profile!=SETUP_PROFILE:raise ValueError('Existing photo environment required')
    position=np.asarray(profile['model_camera_local_position_m'],float)
    rotation=np.asarray(profile['model_camera_local_rotation_mujoco'],float);fovy=float(profile['fovy_deg'])
    if (position.shape!=(3,) or not np.isfinite(position).all() or rotation.shape!=(3,3) or
        not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5) or not 30<fovy<140):
        raise ValueError('Invalid camera registration geometry')
    physical_arrays=(sim.data.qpos,sim.data.qvel,sim.data.ctrl,sim.data.act,sim.data.qacc_warmstart)
    before=[a.copy() for a in physical_arrays]
    cid=sim.model.camera('right_wrist').id;sim.model.cam_pos[cid]=position
    mujoco.mju_mat2Quat(sim.model.cam_quat[cid],rotation.ravel());sim.model.cam_fovy[cid]=fovy
    mujoco.mj_camlight(sim.model,sim.data)
    if not all(np.array_equal(a,b) for a,b in zip(before,physical_arrays)):raise RuntimeError('Camera registration altered physical state')
    return dict(kind=profile['kind'],profile_path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera='right_wrist',
        position_parent_m=position.tolist(),rotation_parent_mujoco=rotation.tolist(),fovy_deg=fovy,
        physics_gripper_scaling_applied=False,native_replay_gripper_scale=profile.get('right_gripper_opening_scale',1.),
        note='Camera-only conditional image registration; no object/joint/control/mass/mesh changes and no claim of metric calibration')


def _rigid(position,rotation):
    value=np.eye(4);value[:3,:3]=rotation;value[:3,3]=position;return value


def _validate_rigid(value):
    value=np.asarray(value,dtype=float)
    if (value.shape!=(4,4) or not np.isfinite(value).all() or not np.allclose(value[3],[0,0,0,1]) or
        not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(value[:3,:3]),1,atol=1e-5)):
        raise ValueError('Expected proper finite rigid transform')
    return value


def load_registered_setup_profile(profile_path=None,*,include_diagnostic_carrier=False):
    """Load immutable component references; reject changed/mismatched profiles."""
    path=Path(profile_path) if profile_path is not None else DEFAULT_REGISTERED_PROFILE
    if path.exists():
        descriptor=json.loads(path.read_text())
    elif profile_path is not None:
        raise FileNotFoundError(path)
    else:
        directory=WORKSPACE_ROOT/'config/pin-insertion-photo'
        descriptor=dict(schema_version=1,name='photo_20261027_registered_v1',kind='registered_photo_setup',
            components={name:dict(path=str(directory/filename),sha256=hashlib.sha256((directory/filename).read_bytes()).hexdigest())
                for name,filename in [('camera','camera.json'),('planar_base','planar-base.json')]},
            physical_scope='Camera and planar base applied before actual original task reset; actual original contact grasps retained.',
            diagnostic_scope='TRAIN-only carrier/tool transform and complete CAD relationships may be used only for explicit native-q kinematic reconstruction.',
            diagnostic_carrier_candidate=dict(path=str(directory/'carrier-grasp-training.json'),status='provisional_pending_carrier_CAD_audit_not_part_of_physical_profile'),carrier_grasp_physically_validated=False,real_calibrated=False)
    if descriptor.get('kind')!='registered_photo_setup' or descriptor.get('schema_version')!=1:raise ValueError('Invalid registered setup descriptor')
    result=json.loads(json.dumps(descriptor))
    for name,reference in result['components'].items():
        component=Path(reference['path']);component=component if component.is_absolute() else path.parent/component
        if hashlib.sha256(component.read_bytes()).hexdigest()!=reference['sha256']:raise ValueError(f'Registered {name} profile hash mismatch')
        reference['path']=str(component.resolve())
    camera=json.loads(Path(result['components']['camera']['path']).read_text())
    base=json.loads(Path(result['components']['planar_base']['path']).read_text())
    if camera.get('rejected_as_final') or camera.get('kind')!='own_right_finger_silhouette_registration_hypothesis':raise ValueError('Unaccepted camera profile')
    if not base.get('fixed_parameters',{}).get('base_planar'):raise ValueError('Physical registered setup requires upright/common-height planar base fit')
    if not np.allclose(base['T_right_tool_camera_opencv'],camera['T_right_tool_camera_opencv'],atol=1e-8):raise ValueError('Base/camera registration mismatch')
    if include_diagnostic_carrier:
        reference=result.get('diagnostic_carrier_candidate')
        if not reference:raise ValueError('No explicit diagnostic carrier candidate')
        component=Path(reference['path']);component=component if component.is_absolute() else path.parent/component
        carrier=json.loads(component.read_text())
        if carrier.get('camera_sha256')!=result['components']['camera']['sha256']:raise ValueError('Carrier/camera registration provenance mismatch')
        result['components']['diagnostic_carrier']=dict(path=str(component.resolve()),sha256=hashlib.sha256(component.read_bytes()).hexdigest(),scope='provisional_diagnostic_only',geometry_final=False)
    result['descriptor_path']=str(path.resolve()) if path.exists() else None
    return result


def write_registered_setup_profile(path=DEFAULT_REGISTERED_PROFILE):
    profile=load_registered_setup_profile();profile.pop('descriptor_path',None)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(profile,indent=2)+'\n');return path


def apply_planar_base_registration(environment,profile_path,*,diagnostic_only=False):
    """Update fixed base geometry before reset; never teleport an active rollout."""
    import mujoco
    if getattr(environment,'_ready',False) and not diagnostic_only:raise ValueError('Apply base registration before physical reset, never during an active rollout')
    path=Path(profile_path);profile=json.loads(path.read_text());base=_validate_rigid(profile['T_right_base_left_base'])
    if not np.allclose(base[2],[0,0,1,0],atol=1e-8) or not np.allclose(base[:2,2],0,atol=1e-8):raise ValueError('Base fit must preserve common height and upright axes')
    model,data=environment.model,environment.data
    left=model.body('left_base').id;right=model.body('right_base').id
    left_world=_rigid(data.xpos[left],data.xmat[left].reshape(3,3));right_world=left_world@np.linalg.inv(base)
    arrays=(data.qpos,data.qvel,data.ctrl,data.act,data.qacc_warmstart);saved=[a.copy() for a in arrays]
    model.body_pos[right]=right_world[:3,3];mujoco.mju_mat2Quat(model.body_quat[right],right_world[:3,:3].ravel())
    mujoco.mj_setConst(model,data)
    for target,value in zip(arrays,saved):target[:]=value
    mujoco.mj_forward(model,data)
    if not all(np.array_equal(a,b) for a,b in zip(arrays,saved)):raise RuntimeError('Base registration changed generalized state')
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),T_right_base_left_base=base.tolist(),
        mode='native_q_diagnostic' if diagnostic_only else 'physical_setup_before_reset',common_height_and_upright=True,
        note='Only fixed base pose registered; no pin/carrier pose assignment, joint retargeting or grasp replacement')


def apply_registered_photo_setup(environment,profile_path=None,*,diagnostic_only=False):
    if getattr(environment,'_ready',False) and not diagnostic_only:raise ValueError('Apply registered setup before physical reset, never during an active rollout')
    descriptor=load_registered_setup_profile(profile_path,include_diagnostic_carrier=diagnostic_only)
    camera=apply_photo_camera_registration(environment,descriptor['components']['camera']['path'])
    base=apply_planar_base_registration(environment,descriptor['components']['planar_base']['path'],diagnostic_only=diagnostic_only)
    environment.photo_registration=dict(name=descriptor['name'],scope='native_q_diagnostic' if diagnostic_only else 'physical_task',
        descriptor=descriptor,camera=camera,planar_base=base,carrier_grasp_applied_to_physics=False,
        diagnostic_carrier_profile=descriptor['components'].get('diagnostic_carrier',descriptor.get('diagnostic_carrier_candidate')),real_calibrated=False)
    return environment.photo_registration


class NativeQPhotoReconstruction:
    """Explicit kinematic reconstruction; object assignment never enters task physics."""
    diagnostic_only=True
    def __init__(self,*,profile_path=None,seed=0):
        self.environment=create_photo_environment(setup='nominal')
        self.environment.reset(seed=int(seed),options={'randomize':False})
        self.bridge=PhotoSetupBridge(environment=self.environment)
        self.pin_tool_transform=self.bridge.grasp_transform('left','pin_2')
        self.registration=apply_registered_photo_setup(self.environment,profile_path,diagnostic_only=True)
        carrier=json.loads(Path(self.registration['diagnostic_carrier_profile']['path']).read_text())
        self.carrier_tool_transform=_validate_rigid(carrier['T_right_tool_carrier'])
        self.camera_profile=json.loads(Path(self.registration['descriptor']['components']['camera']['path']).read_text())
        self.cad=json.loads((TRAINING_ROOT/'sim/actuator/manifest.json').read_text())['parts']

    def set_native_pose(self,native14,*,complete_assembly=True):
        import mujoco
        self.bridge.set_kinematic_pose(native14);model,data=self.bridge.model,self.bridge.data
        opening=data.qpos[self.bridge.finger_qadr[1,0]]*self.camera_profile['right_gripper_opening_scale']
        data.qpos[self.bridge.finger_qadr[1]]=[opening,-opening];mujoco.mj_forward(model,data)
        def tool(side):
            index=('left','right').index(side);site=self.bridge.sites[index]
            return _rigid(data.site_xpos[site],data.site_xmat[site].reshape(3,3))
        def place(name,pose):
            body=model.body(name).id;quat=np.empty(4);mujoco.mju_mat2Quat(quat,pose[:3,:3].ravel())
            if model.body_jntnum[body]:
                adr=model.jnt_qposadr[model.body_jntadr[body]];data.qpos[adr:adr+7]=np.r_[pose[:3,3],quat]
            else:model.body_pos[body]=pose[:3,3];model.body_quat[body]=quat
        pin=tool('left')@self.pin_tool_transform;carrier=tool('right')@self.carrier_tool_transform
        place('pin_2',pin);place('large_carrier',carrier)
        if complete_assembly:
            for name in ('small_carrier','gear_1','gear_2','gear_3','gear_4'):
                offset=np.asarray(self.cad[name]['assembled_origin'])-self.cad['large_carrier']['assembled_origin']
                pose=carrier.copy();pose[:3,3]+=carrier[:3,:3]@offset;place(name,pose)
            for gid in range(model.ngeom):
                name=model.geom(gid).name or ''
                if name=='fixture' or name.endswith('_depth_stop'):model.geom_group[gid]=3
        mujoco.mj_forward(model,data)
        return dict(mode='native_q_diagnostic',physical_rollout=False,T_world_pin=pin.tolist(),T_world_carrier=carrier.tolist(),
            grasp_source='Pin from actual nominal physical reset; carrier TRAIN-only PnP hypothesis',native_arm_angles_changed=False)

    def step(self,*args,**kwargs):raise RuntimeError('Kinematic reconstruction cannot step physics; use create_photo_environment() for genuine physical reset/rollout')
    def close(self):self.bridge.close()


def create_native_q_reconstruction(*,profile_path=None,seed=0):return NativeQPhotoReconstruction(profile_path=profile_path,seed=seed)
