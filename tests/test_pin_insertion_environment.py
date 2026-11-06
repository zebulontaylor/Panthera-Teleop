"""Verify binding preserves existing photo environment and native arm frames."""
import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('mujoco');pytest.importorskip('gymnasium')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pin_insertion_environment import create_photo_environment,PhotoSetupBridge,SETUP_PROFILE
from real2sim_bridge import ARM_COLUMNS


@pytest.fixture(scope='module')
def adapter():
    env=create_photo_environment(setup="nominal")
    qpos=env.data.qpos.copy();bridge=PhotoSetupBridge(environment=env)
    np.testing.assert_array_equal(bridge.data.qpos,qpos)
    yield bridge
    bridge.close()


def test_native14_arm_passthrough_and_correct_bases(adapter):
    state=adapter.native_state();state[0]=.1;state[7]=-.1
    poses=adapter.set_kinematic_pose(state)
    np.testing.assert_array_equal(adapter.native_state()[ARM_COLUMNS],state[ARM_COLUMNS])
    assert [adapter.model.body(i).name for i in adapter.base_ids]==['left_base','right_base']
    for side in ('left','right'):
        arm=adapter.sim.arms[side];base=adapter.model.body(side+'_base').id
        R=adapter.data.xmat[base].reshape(3,3)
        expected=R.T@(adapter.data.site_xpos[arm.ee_site]-adapter.data.xpos[base])
        np.testing.assert_allclose(poses[side]['position'],expected)


def test_existing_material_optics_not_legacy(adapter):
    p=adapter.provenance()
    assert p['setup_profile']==SETUP_PROFILE
    assert p['layout']=='side_by_side'
    np.testing.assert_allclose(p['cameras']['right_wrist']['position_parent_m'],[.045,0,.1])
    assert adapter.model.mat('photo_bench').id>=0
    assert adapter.model.body('center_tower').id>=0
    assert p['base_frames']['left']['position_world_m'][1]==.29
    assert p['base_frames']['right']['position_world_m'][1]==-.29


def test_physics_not_routed_through_legacy_bridge(adapter):
    with pytest.raises(NotImplementedError,match='environment.step'):adapter.step(np.zeros(14))
    with pytest.raises(NotImplementedError,match='environment.reset'):adapter.initialize_dynamic(np.zeros(14))


def test_exact_runtime_existing_physical_reset():
    env=create_photo_environment(setup="nominal")
    try:
        _,info=env.reset(seed=0,options={'randomize':False})
        assert info['pinched'] and info['holder_pinched'] and info['carrier_supported']
    finally:env.close()


def test_camera_registration_preserves_physical_state(adapter,tmp_path):
    import json
    from pin_insertion_environment import apply_photo_camera_registration
    cid=adapter.model.camera('right_wrist').id
    import mujoco
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,adapter.model.cam_quat[cid])
    profile=dict(kind='own_right_finger_silhouette_registration_hypothesis',model_camera_local_position_m=adapter.model.cam_pos[cid].tolist(),
        model_camera_local_rotation_mujoco=rotation.reshape(3,3).tolist(),fovy_deg=72.,right_gripper_opening_scale=1.4)
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    before=[a.copy() for a in (adapter.data.qpos,adapter.data.qvel,adapter.data.ctrl,adapter.data.act,adapter.data.qacc_warmstart)]
    report=apply_photo_camera_registration(adapter.environment,path)
    for a,b in zip(before,(adapter.data.qpos,adapter.data.qvel,adapter.data.ctrl,adapter.data.act,adapter.data.qacc_warmstart)):np.testing.assert_array_equal(a,b)
    assert not report['physics_gripper_scaling_applied']


def test_exact_visual_partition_preserves_physics_and_native_fk():
    import sim.bimanual_actuator_env as bimanual
    original_hook=bimanual.apply_photo_setup
    baseline=create_photo_environment(visual_partition=False,setup="nominal")
    partitioned=create_photo_environment(visual_partition=True,setup="nominal")
    try:
        assert bimanual.apply_photo_setup is original_hook
        a,b=baseline.model,partitioned.model
        assert (a.nq,a.nv)==(b.nq,b.nv)
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_pos','jnt_axis','jnt_range'):
            np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
        ga=np.flatnonzero(a.geom_group==3);gb=np.flatnonzero(b.geom_group==3)
        assert len(ga)==len(gb)
        for field in ('geom_bodyid','geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction'):
            np.testing.assert_array_equal(getattr(a,field)[ga],getattr(b,field)[gb])
        ba,bb=PhotoSetupBridge(environment=baseline),PhotoSetupBridge(environment=partitioned)
        q=ba.native_state();q[0]=.12;q[7]=-.2
        pa,pb=ba.set_kinematic_pose(q),bb.set_kinematic_pose(q)
        for side in ('left','right'):
            np.testing.assert_array_equal(pa[side]['position'],pb[side]['position'])
            np.testing.assert_array_equal(pa[side]['rotation'],pb[side]['rotation'])
        assert b.mesh('photo_L_white_digit').id>=0
        assert partitioned.photo_visual_partition is not None
    finally:baseline.close();partitioned.close()


def test_registered_is_default_and_physical_reset_retains_true_contacts():
    import json
    from pin_insertion_environment import load_registered_setup_profile,apply_planar_base_registration
    env=create_photo_environment()
    try:
        bridge=PhotoSetupBridge(environment=env);profile=load_registered_setup_profile()
        camera=json.loads(Path(profile['components']['camera']['path']).read_text())
        cid=env.model.camera('right_wrist').id
        np.testing.assert_allclose(env.model.cam_pos[cid],camera['model_camera_local_position_m'])
        assert env.photo_registration['scope']=='physical_task'
        assert not env.photo_registration['carrier_grasp_applied_to_physics']
        for seed in range(3):
            _,info=env.reset(seed=seed)
            assert info['pinched'] and info['holder_pinched'] and info['carrier_supported']
        left,right=bridge.base_ids
        assert abs(env.data.xpos[left,2]-env.data.xpos[right,2])<1e-12
        np.testing.assert_allclose(env.data.xmat[right].reshape(3,3)[:,2],[0,0,1],atol=1e-12)
        before=env.data.qpos.copy()
        with pytest.raises(ValueError,match='before physical reset'):apply_planar_base_registration(env,profile['components']['planar_base']['path'])
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()


def test_registered_descriptor_rejects_changed_components(tmp_path):
    import json
    from pin_insertion_environment import load_registered_setup_profile
    profile=load_registered_setup_profile();profile['components']['camera']['sha256']='0'*64
    path=tmp_path/'bad.json';path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='hash mismatch'):load_registered_setup_profile(path)


def test_native_q_reconstruction_is_separate_from_physical_rollout():
    from pin_insertion_environment import create_native_q_reconstruction
    reconstruction=create_native_q_reconstruction(seed=0)
    try:
        bridge=reconstruction.bridge;q=bridge.native_state();q[0]+=.01;q[7]-=.01
        report=reconstruction.set_native_pose(q)
        np.testing.assert_array_equal(bridge.native_state()[ARM_COLUMNS],q[ARM_COLUMNS])
        np.testing.assert_allclose(bridge.grasp_transform('left','pin_2'),reconstruction.pin_tool_transform,atol=1e-10)
        np.testing.assert_allclose(bridge.grasp_transform('right','large_carrier'),reconstruction.carrier_tool_transform,atol=1e-10)
        assert report['mode']=='native_q_diagnostic' and not report['physical_rollout']
        assert reconstruction.environment.photo_registration['scope']=='native_q_diagnostic'
        with pytest.raises(RuntimeError,match='cannot step physics'):reconstruction.step(np.zeros(14))
    finally:reconstruction.close()


def test_reproduced_carrier_visual_preserves_original_physics_and_targets():
    old=create_photo_environment(setup='nominal',carrier_visual_profile=False)
    new=create_photo_environment(setup='nominal',carrier_visual_profile=True)
    try:
        a,b=old.model,new.model
        assert (a.nq,a.nv,a.nu)==(b.nq,b.nv,b.nu)
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_pos','jnt_axis','jnt_range','site_pos'):
            np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
        ga,gb=np.flatnonzero(a.geom_group==3),np.flatnonzero(b.geom_group==3)
        for field in ('geom_bodyid','geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction','geom_dataid'):
            np.testing.assert_array_equal(getattr(a,field)[ga],getattr(b,field)[gb])
        _,ia=old.reset(seed=0,options={'randomize':False});_,ib=new.reset(seed=0,options={'randomize':False})
        np.testing.assert_array_equal(old.data.qpos,new.data.qpos)
        assert ia['pinched'] and ib['pinched'] and ib['holder_pinched'] and ib['carrier_supported']
        assert new.photo_carrier_visual_profile['newly_reproduced']
        assert not new.photo_carrier_visual_profile['original_recovered_mesh']
    finally:old.close();new.close()


def test_default_registration_references_versioned_profiles():
    import json
    from pin_insertion_environment import DEFAULT_REGISTERED_PROFILE,load_registered_setup_profile
    assert 'config/pin-insertion-photo' in str(DEFAULT_REGISTERED_PROFILE)
    raw=json.loads(DEFAULT_REGISTERED_PROFILE.read_text())
    for reference in raw['components'].values():assert not Path(reference['path']).is_absolute()
    loaded=load_registered_setup_profile()
    for reference in loaded['components'].values():assert 'config/pin-insertion-photo' in reference['path']
