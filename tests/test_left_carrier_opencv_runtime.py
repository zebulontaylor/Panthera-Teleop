"""Stateless RGB-only registration and actual-rim visibility contracts."""
import json
import numpy as np
import pytest
cv2=pytest.importorskip('cv2')
from app.left_carrier_opencv import LeftCarrierOpenCVPredictor,cyclic_layout

@pytest.fixture
def bank(tmp_path):
    rng=np.random.default_rng(4)
    gray=rng.integers(110,195,(240,320),dtype=np.uint8)
    points=np.array([[110,90],[160,75],[180,125],[130,145]],float)
    for q in points:
        p=tuple(q.astype(int));cv2.circle(gray,p,12,220,-1);cv2.circle(gray,p,8,25,-1)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);path=tmp_path/'train.png';cv2.imwrite(str(path),gray)
    row={'id':'train0','split':'train','image_path':str(path),'unordered_visible_bores_px':points.tolist(),'bore_radii_px':[8]*4}
    manifest=tmp_path/'bank.json';manifest.write_text(json.dumps({'frames':[row]}))
    return manifest,image,points,row

def test_cyclic_layout_missing_fourth_uses_opposite_pair():
    p=np.array([[0,0],[50,-20],[70,30]],float)
    full=cyclic_layout(p)
    assert any(np.linalg.norm(q-[20,50])<1e-6 for q in full)
    with pytest.raises(ValueError):cyclic_layout([[0,0],[1,0],[.5,.86]])

def test_train_only_reference_bank(bank,tmp_path):
    path,image,points,row=bank;row['split']='validation'
    path.write_text(json.dumps({'frames':[row]}))
    with pytest.raises(ValueError,match='TRAIN'):LeftCarrierOpenCVPredictor(path)

def test_rgb_translation_relocalizes_without_stationary_hold(bank):
    path,image,points,row=bank;predictor=LeftCarrierOpenCVPredictor(path)
    first=predictor.predict(image)
    assert sum(first['visible'])>=3
    shifted=cv2.warpAffine(image,np.array([[1,0,9],[0,1,6]],float),(320,240),borderMode=cv2.BORDER_REFLECT)
    second=predictor.predict(shifted)
    assert sum(second['visible'])>=3
    a=np.array([q for q,v in zip(first['holes_px'],first['visible']) if v])
    b=np.array([q for q,v in zip(second['holes_px'],second['visible']) if v])
    assert np.mean(np.min(np.linalg.norm((a+[9,6])[:,None]-b[None,:],axis=-1),axis=1))<2
    lost=predictor.predict(np.full_like(image,180))
    assert lost['holes_px']==[None]*4 and not any(lost['visible']) and not lost['supported']
    assert not lost['pixel_hold']

def test_occluded_bore_is_not_declared_observed(bank):
    path,image,points,row=bank;predictor=LeftCarrierOpenCVPredictor(path)
    masked=image.copy();cv2.rectangle(masked,(96,76),(124,104),(190,190,190),-1)
    result=predictor.predict(masked)
    observed=[q for q,v in zip(result['holes_px'],result['visible']) if v]
    assert all(np.linalg.norm(np.asarray(q)-points[0])>10 for q in observed)
    assert not result['motor_permission'] and not result['canonical_physical_slot_identity_verified']

def synthetic_three_landmarks(opposite=False):
    gray=np.full((480,640),180,np.uint8);center=(320,240)
    for r,color in ((20,25),(17,165),(9,25)):cv2.circle(gray,center,r,color,-1)
    bores=[(280,240),(360,240) if opposite else (320,200)]
    for p in bores:cv2.circle(gray,p,12,220,-1);cv2.circle(gray,p,8,25,-1)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

def test_three_total_holes_map_entire_face_without_reference(bank):
    path,_,_,_=bank;predictor=LeftCarrierOpenCVPredictor(path);predictor.references=[]
    result=predictor.predict(synthetic_three_landmarks())
    assert result['supported'] and result['registration_kind']=='affine3landmark'
    assert result['registration_landmark_count']==3 and result['bearing_observed']
    assert sum(result['visible'])==2
    assert sum(h['inferred'] for h in result['holes'])==2
    assert len(result['layout_holes'])==13 and len(result['face_outline_px'])>=32
    assert all(np.isfinite(h['px']).all() for h in result['layout_holes'])
    bosses=[h for h in result['layout_holes'] if h['kind']=='boss_mounting']
    assert len(bosses)==4 and all(h['plane']=='raised_boss' and h['height_above_face_mm_hypothesis']==4 for h in bosses)
    assert all(not h['insertion_target'] for h in result['layout_holes'] if h['kind']!='insertion_bore')
    assert all(not h['observed'] for h in result['holes'] if h['inferred'])

def test_opposite_two_bores_plus_bearing_are_degenerate(bank):
    path,_,_,_=bank;predictor=LeftCarrierOpenCVPredictor(path);predictor.references=[]
    result=predictor.predict(synthetic_three_landmarks(opposite=True))
    assert not result['supported'] and not any(result['visible'])
    assert result['layout_holes']==[]

def test_isolated_foreground_circle_cannot_bootstrap_face(bank):
    path,_,_,_=bank;predictor=LeftCarrierOpenCVPredictor(path);predictor.references=[]
    image=np.full((480,640,3),180,np.uint8);cv2.circle(image,(300,250),9,(20,20,20),-1)
    result=predictor.predict(image)
    assert result['holes_px']==[None]*4 and not result['supported']

def test_raised_boss_observation_is_never_a_planar_metric_anchor(bank):
    path,_,_,_=bank;predictor=LeftCarrierOpenCVPredictor(path);predictor.references=[]
    image=synthetic_three_landmarks();cv2.circle(image,(283,203),3,(25,25,25),-1)
    result=predictor.predict(image)
    bosses=[p for p in result['face_landmarks'] if p['kind']=='boss_mounting']
    assert bosses and all(p['plane']=='raised_boss' and not p['metric_homography_eligible'] for p in bosses)
    # The extra raised feature can be drawn but does not increase mapping support.
    assert result['registration_landmark_count']==3

def test_secondary_small_holes_refine_map_with_less_influence_than_large_anchors():
    from app.left_carrier_opencv import weighted_affine_refine,transform
    H=np.array([[2000.,0.,320.],[0.,2000.,240.],[0.,0.,1.]])
    large=np.array([[.0179625,0],[0,.0179625]]);target=transform(large,H)
    small=np.array([[.008,.006],[-.006,.009]])
    observed=transform(small,H)+[[2.,-1.],[1.,-1.]]
    low,info=weighted_affine_refine(H,large,target,small,observed,bearing_px=[320,240],aux_weight=.25)
    high,_=weighted_affine_refine(H,large,target,small,observed,bearing_px=[320,240],aux_weight=1.)
    assert info['accepted'] and info['auxiliary_mean_after_px']<info['auxiliary_mean_before_px']
    low_shift=np.linalg.norm(transform(large,low)-target,axis=1)
    high_shift=np.linalg.norm(transform(large,high)-target,axis=1)
    assert low_shift.max()<2 and low_shift.mean()<high_shift.mean()
    assert info['weights']['raised_boss']==0 and not info['perspective_parameters_freed']

def test_gross_auxiliary_outlier_cannot_move_the_large_bore_fit():
    from app.left_carrier_opencv import weighted_affine_refine,transform
    H=np.array([[2000.,0.,320.],[0.,2000.,240.],[0.,0.,1.]])
    large=np.array([[.0179625,0],[0,.0179625]]);target=transform(large,H)
    small=np.array([[.008,.006],[-.006,.009]]);observed=transform(small,H)+[[2.,-1.],[1.,-1.]]
    clean,_=weighted_affine_refine(H,large,target,small,observed,bearing_px=[320,240])
    extra=np.vstack((small,[.01,-.008]));dirty=np.vstack((observed,transform([[.01,-.008]],H)[0]+[40,-50]))
    robust,info=weighted_affine_refine(H,large,target,extra,dirty,bearing_px=[320,240])
    np.testing.assert_allclose(robust,clean,atol=1e-10)
    assert info['auxiliary_outlier_count']==1
