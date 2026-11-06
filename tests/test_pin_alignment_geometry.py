import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from pin_alignment_geometry import (preview_from_geometry, ray_plane_point,
                                    estimate_preview_nudge, right_link6_transform)


def preview(pin=(0,0), hole=(1,0), **kwargs):
    return preview_from_geometry(pin,hole,np.eye(3),np.eye(4),
        np.array([0.,0.,1.]),np.array([0.,0.,1.]),
        kwargs.pop('rotation',np.eye(3)),gain=kwargs.pop('gain',1.),
        max_step_m=kwargs.pop('max_step_m',10.),deadband_px=kwargs.pop('deadband_px',0.),**kwargs)


def test_identity_direction_and_yaw_conversion():
    out = preview()
    np.testing.assert_allclose(out['delta_left_m'],[1,0,0])
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    np.testing.assert_allclose(preview(rotation=rotation)['delta_left_m'],[0,-1,0])
    np.testing.assert_allclose(preview(frame_yaw_deg=90)['delta_left_m'],[0,1,0],atol=1e-12)


def test_deadband_gain_and_norm_cap():
    np.testing.assert_allclose(preview(deadband_px=1)['delta_left_m'],0)
    np.testing.assert_allclose(preview(gain=.2)['delta_left_m'],[.2,0,0])
    out = preview(hole=(3,4),max_step_m=.01)
    np.testing.assert_allclose(out['delta_left_m'],[.006,.008,0])


def test_reject_parallel_and_behind_rays():
    import pytest
    with pytest.raises(ValueError,match='parallel'):
        ray_plane_point([0,0],np.eye(3),np.eye(4),np.ones(3),np.array([1,0,0]))
    with pytest.raises(ValueError,match='behind'):
        ray_plane_point([0,0],np.eye(3),np.eye(4),np.array([0,0,-1]),np.array([0,0,1]))
    assert not preview(pin=(float('nan'),0))['valid']
    assert not preview(gain=-1)['valid']


def test_tilted_plane_tangency():
    normal = np.array([.2,.3,1.])
    out = preview_from_geometry([0,0],[.1,.2],np.eye(3),np.eye(4),
        np.array([0,0,1.]),normal,np.eye(3),max_step_m=1,deadband_px=0)
    assert out['valid']
    assert abs(np.dot(normal,out['delta_right_m'])) < 1e-12


def test_native_fk_and_profiles_are_device_free():
    t = right_link6_transform(np.zeros(6))
    np.testing.assert_allclose(t[:3,:3],np.eye(3))
    np.testing.assert_allclose(t[:3,3],[.081649,0,.168719])
    assert not estimate_preview_nudge([0]*13,[0,0],[1,0])['valid']
    out = estimate_preview_nudge([0]*14,[282,306],[290,306])
    assert out['valid'],out
    assert out['label'] == 'preview estimated'
    assert not out['metric_accuracy_validated']
    assert np.linalg.norm(out['delta_left_m']) <= .00100000001


def test_full_center_delta_ignores_gain_cap_and_deadband_but_keeps_frames():
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    out=preview(hole=(3,4),rotation=rotation,gain=.1,max_step_m=.01)
    np.testing.assert_allclose(out['unclamped_full_delta_right_m'],[3,4,0])
    np.testing.assert_allclose(out['unclamped_full_delta_left_m'],[4,-3,0])
    np.testing.assert_allclose(out['delta_left_m'],[.008,-.006,0])
    quiet=preview(hole=(3,4),rotation=rotation,deadband_px=10,frame_yaw_deg=90)
    np.testing.assert_allclose(quiet['delta_left_m'],0,atol=1e-12)
    np.testing.assert_allclose(quiet['unclamped_full_delta_left_m'],[3,4,0],atol=1e-12)
    np.testing.assert_allclose(quiet['unclamped_full_delta_right_m'],[3,4,0])


def test_left_camera_motion_changes_projection_and_preserves_face():
    from pin_alignment_geometry import left_preview_geometry
    state = np.zeros(14)
    K, camera, point, normal, _ = left_preview_geometry(state)
    moved = state.copy(); moved[0] = .2
    K2, camera2, point2, normal2, _ = left_preview_geometry(moved)
    assert not np.allclose(camera, camera2)
    np.testing.assert_allclose(point, point2)
    np.testing.assert_allclose(normal, normal2)
    world = camera @ [0, 0, .3, 1]
    def project(t):
        optical = np.linalg.solve(t, world)[:3]
        uv = K @ optical
        return uv[:2]/uv[2]
    assert np.linalg.norm(project(camera)-project(camera2)) > 1
    np.testing.assert_allclose(K, K2)


def test_left_estimator_analytic_mapping_and_existing_bounds(tmp_path):
    import json
    from pin_alignment_geometry import estimate_left_preview_nudge, right_link6_transform
    # Cancel FK to create a known OpenCV camera at right-base origin. The face
    # is z=1, so 10 pixels at f=100 is +0.1m along native-left X.
    state = np.zeros(14)
    link = right_link6_transform(state[:6])
    mount = np.linalg.inv(link)
    face = np.eye(4); face[2,3] = 1-.01725
    tool = link.copy(); tool[:3,3] += tool[:3,:3] @ [.165,0,0]
    profiles = {
        'left-camera.json': dict(camera='left_wrist', image_width=200, image_height=200,
            fovy_deg=90, model_camera_local_position_m=mount[:3,3].tolist(),
            model_camera_local_rotation_mujoco=(mount[:3,:3] @ np.diag([1,-1,-1])).tolist()),
        'carrier-grasp-training.json': dict(T_right_tool_carrier=(np.linalg.inv(tool) @ face).tolist()),
        'planar-base.json': dict(T_right_base_left_base=np.eye(4).tolist())}
    for name, value in profiles.items():
        (tmp_path/name).write_text(json.dumps(value))
    out = estimate_left_preview_nudge(state,[100,100],[110,100],profile_dir=tmp_path)
    assert out['valid'],out
    np.testing.assert_allclose(out['delta_left_m'],[.001,0,0],atol=1e-12)
    np.testing.assert_allclose(out['unclamped_full_delta_left_m'],[.1,0,0],atol=1e-12)
    quiet = estimate_left_preview_nudge(state,[100,100],[110,100],profile_dir=tmp_path,
                                       deadband_px=20,frame_yaw_deg=90)
    np.testing.assert_allclose(quiet['delta_left_m'],0)
    np.testing.assert_allclose(quiet['unclamped_full_delta_left_m'],[0,.1,0],atol=1e-12)
    assert not out['real_calibrated']
    assert not estimate_left_preview_nudge([0]*13,[0,0],[1,0])['valid']
