import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_wrist_from_table import texture_mask,plane_transfer,fit_plane,rigid,wood_mask,match_table_features
from scipy.spatial.transform import Rotation


def camera_pixels(world,tool,camera,K):
    pose=tool@camera
    points=(world-pose[:3,3])@pose[:3,:3]
    return cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,np.zeros(5))[0].reshape(-1,2)


def synthetic_pairs():
    rng=np.random.default_rng(9);K=np.array([[400.,0,320],[0,400,240],[0,0,1]])
    camera=rigid(np.diag([1.,-1,-1]),[0,0,.05])
    world=np.c_[rng.uniform(-.035,.035,(30,2)),np.zeros(30)]
    poses=[rigid(Rotation.from_rotvec(rng.normal(0,.08,3)).as_matrix(),rng.normal(0,.01,3)+[0,0,.2]) for _ in range(10)]
    pairs=[]
    for i in range(9):
        pairs.append(dict(a=dict(pose=poses[i]),b=dict(pose=poses[i+1]),
            pa=camera_pixels(world,poses[i],camera,K),pb=camera_pixels(world,poses[i+1],camera,K),split='train' if i<7 else 'validation'))
    prior=dict(T_right_tool_camera_opencv=camera.tolist(),K=K.tolist(),distortion=[0]*5)
    return pairs,prior,world


def test_ray_plane_transfer_uses_measured_metric_motion():
    pairs,prior,world=synthetic_pairs();pair=pairs[0];n=len(world)
    pred,distance,depth,xy=plane_transfer(pair['pa'],np.repeat(pair['a']['pose'][None],n,axis=0),np.repeat(pair['b']['pose'][None],n,axis=0),np.array(prior['T_right_tool_camera_opencv']),np.array(prior['K']),np.zeros(5))
    np.testing.assert_allclose(pred,pair['pb'],atol=1e-7)
    np.testing.assert_allclose(xy,world,atol=1e-8)
    assert np.all(distance>0) and np.all(depth>0)


def test_mask_excludes_gripper_and_far_image_edges():
    mask=texture_mask((480,640))
    assert mask[100,400]==255
    assert mask[450,320]==0
    assert mask[240,20]==0
    assert mask[20,320]==0


def test_exact_fit_has_no_validation_pixel_leakage():
    pairs,prior,_=synthetic_pairs();result=fit_plane(pairs,prior,starts=1)
    assert result['training']['mean_px']<1e-6
    assert result['validation']['mean_px']<1e-6
    changed=[dict(p,pb=p['pb']+[100,50]) if p['split']=='validation' else p for p in pairs]
    shifted=fit_plane(changed,prior,starts=1)
    np.testing.assert_allclose(result['T_right_tool_camera_opencv'],shifted['T_right_tool_camera_opencv'])
    assert shifted['validation']['mean_px']>100
    assert not result['real_calibrated']


def test_wood_color_filter_rejects_plain_room_wall():
    wall=np.full((480,640,3),200,np.uint8)
    assert np.count_nonzero(wood_mask(wall))==0
    matches,audit=match_table_features(wall,wall)
    assert matches is None
    assert audit['reason']=='not_wood_dominated_polygon'
    wood=np.full((480,640,3),(110,155,190),np.uint8)
    assert np.count_nonzero(wood_mask(wood))>0
