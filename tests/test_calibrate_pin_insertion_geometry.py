import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_pin_insertion_geometry import rigid,pin_camera_points,project_opencv,fit,jacobian_report,load_observations,teacher_projections
from scipy.spatial.transform import Rotation


def test_pose_chain_right_and_left_frames_are_distinct():
    left=np.array([rigid(translation=[1,0,0])]);right=np.array([rigid(translation=[0,2,0])])
    camera=rigid(translation=[0,0,.2]);base=rigid(translation=[0,0,1])
    points=pin_camera_points(left,right,camera,base,[[0,0,.1]])
    np.testing.assert_allclose(points,[[1,-2,.9]])


def test_opencv_projection_matches_cv2_distortion():
    import cv2
    points=np.array([[.01,.02,.2],[-.02,.01,.3]])
    K=np.array([[400,0,320],[0,410,240],[0,0,1.]])
    distortion=np.array([.1,-.02,.001,.002,.003])
    expected=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,distortion)[0].reshape(-1,2)
    np.testing.assert_allclose(project_opencv(points,K,distortion),expected)


def synthetic_rows():
    K=np.array([[400,0,320],[0,400,240],[0,0,1.]])
    prior=dict(T_right_tool_camera_opencv=np.eye(4).tolist(),T_right_base_left_base=np.eye(4).tolist(),
        pin_tip_left_tool_m=[.01,.005,.02],K=K.tolist(),distortion=[0]*5)
    rows=[]
    rng=np.random.default_rng(3)
    for i in range(18):
        left=rigid(Rotation.from_rotvec(rng.normal(0,.2,3)).as_matrix(),rng.normal(0,.04,3)+[0,0,.35])
        right=rigid(Rotation.from_rotvec(rng.normal(0,.1,3)).as_matrix(),rng.normal(0,.01,3))
        point=pin_camera_points([left],[right],np.eye(4),np.eye(4),[prior['pin_tip_left_tool_m']])
        split='train' if i<12 else 'validation' if i<15 else 'test'
        rows.append(dict(episode_id=split,grasp_group_id=split,split=split,observation_id=str(i),left=left,right=right,
            xy=project_opencv(point,K,np.zeros(5))[0],sigma=1.,width=640,height=480,carrier_bores_px=None))
    return rows,prior


def test_fit_exact_prior_and_no_test_label_leakage():
    rows,prior=synthetic_rows()
    result=fit(rows,prior,starts=1)
    assert result['training']['mean_px']<1e-7
    assert result['test']['mean_px']<1e-7
    changed=[dict(r,xy=r['xy']+[100,50]) if r['split']=='test' else r for r in rows]
    shifted=fit(changed,prior,starts=1)
    np.testing.assert_allclose(result['T_right_tool_camera_opencv'],shifted['T_right_tool_camera_opencv'])
    assert shifted['test']['mean_px']>100
    assert not result['real_calibrated']


def test_duplicate_episode_split_rejected():
    rows,prior=synthetic_rows();rows[-1]['episode_id']='train'
    with pytest.raises(ValueError,match='leakage'):fit(rows,prior,starts=1)


def test_rank_report_excludes_prior_and_detects_nullspace():
    report=jacobian_report(np.array([[1.,0,0],[0,1,0]]),['a','b','c'])
    assert report['rank']==2
    assert report['columns']==3
    assert report['null_or_weak_directions'][0]['c']==1
    assert report['prior_rows_excluded']


def test_root_frames_schema_uses_session_grasp_and_ready_fk(tmp_path):
    import json
    path=tmp_path/'labels.json'
    path.write_text(json.dumps({'frames':[dict(episode_id='ep',session_id='anchor',split='val',id='frame',
        T_left_base_tool=np.eye(4).tolist(),T_right_base_tool=np.eye(4).tolist(),pin_tip_px=[200,300],camera_sample_skew_ms=-23.2)]}))
    rows,rejected=load_observations(path,tmp_path)
    assert not rejected
    assert rows[0]['grasp_group_id']=='anchor'
    assert rows[0]['split']=='validation'
    assert rows[0]['state_image_skew_s']==.0232


def test_teacher_projects_only_unlabeled_training_frames():
    rows,prior=synthetic_rows();result=fit(rows,prior,starts=1)
    frame=dict(id='hidden',episode_id='train',session_id='train',split='train',pin_tip_px=None,
        T_left_base_tool=rows[0]['left'].tolist(),T_right_base_tool=rows[0]['right'].tolist(),width=640,height=480)
    out=teacher_projections({'frames':[frame,dict(frame,id='test',split='test'),dict(frame,id='labeled',pin_tip_px=[1,2])]},result)
    assert len(out)==1
    assert out[0]['id']=='hidden'
    assert out[0]['source_kind']=='calibration_teacher_hypothesis_not_ground_truth'
    assert out[0]['visibility']=='UNOBSERVED'


def test_fixed_camera_remains_exact_prior():
    rows,prior=synthetic_rows();result=fit(rows,prior,fit_camera=False,fit_base=True,starts=1)
    assert result['fixed_parameters']['camera']
    np.testing.assert_array_equal(result['T_right_tool_camera_opencv'],prior['T_right_tool_camera_opencv'])
    assert not any(n.startswith('camera_') for n in result['parameter_names'])


def test_fixed_grasp_and_camera_with_bounded_base_only():
    rows,prior=synthetic_rows();result=fit(rows,prior,fit_camera=False,fit_grasp=False,fit_base=True,base_translation_bound_m=.1,base_rotation_bound_deg=15,starts=1)
    np.testing.assert_array_equal(result['pin_tip_left_tool_m'],prior['pin_tip_left_tool_m'])
    assert result['fixed_parameters']['grasp'] and result['fixed_parameters']['camera']
    assert len(result['parameter_names'])==6
    assert all(n.startswith('base_') for n in result['parameter_names'])


def test_planar_base_only_cannot_tilt_or_float():
    rows,prior=synthetic_rows();result=fit(rows,prior,fit_camera=False,fit_grasp=False,fit_base=True,planar_base=True,base_translation_bound_m=.1,base_rotation_bound_deg=15,starts=1)
    t=np.asarray(result['T_right_base_left_base'])
    np.testing.assert_allclose(t[2],[0,0,1,0],atol=1e-12)
    np.testing.assert_allclose(t[:2,2],[0,0],atol=1e-12)
    assert len(result['parameter_names'])==3
