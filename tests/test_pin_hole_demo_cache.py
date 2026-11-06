import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from app.pin_hole_demo_cache import DemoHoleCache


def fixture_bank(tmp_path):
    rng=np.random.default_rng(44)
    image=np.full((240,320,3),25,np.uint8)
    texture=rng.integers(20,230,(160,220,3),dtype=np.uint8)
    texture=cv2.GaussianBlur(texture,(3,3),.7)
    image[40:200,50:270]=texture
    for x,y in ((70,60),(240,65),(90,175),(220,175),(165,120)):
        cv2.circle(image,(x,y),9,(10,10,10),2)
    path=tmp_path/'reference.png';cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    ref={'id':'train_ref','split':'train','image_path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'hole_px':[165,120],'carrier_polygon_px':[[50,40],[269,40],[269,199],[50,199]],
        'hole_exclusion_radius_px':14,'exclude_polygons_px':[],'uncertainty_px':2,'annotation_provenance':'synthetic known fixture'}
    bank=tmp_path/'references.json';bank.write_text(json.dumps({'references':[ref]}));return bank,image


def update(cache,image,timestamp=1_000_000_000,**changes):
    args={'timestamp_ns':timestamp,'now_ns':timestamp,'sequence_id':'sequence','camera_id':'right_wrist','grasp_id':'grasp'}
    args.update(changes);return cache.update(image,**args)


@pytest.mark.parametrize('mode',['affine_partial','homography'])
def test_transformed_partially_occluded_carrier_tracks_true_hole(tmp_path,mode):
    bank,image=fixture_bank(tmp_path);cache=DemoHoleCache(bank,mode=mode)
    matrix=cv2.getRotationMatrix2D((160,120),8,1.02);matrix[:,2]+=[8,-5]
    matrix=np.vstack((matrix,[0,0,1]))
    if mode=='homography':matrix[2,:2]=[.00012,-.00008]
    current=cv2.warpPerspective(image,matrix,(320,240))
    true=cache._transform([[165,120]],matrix)[0]
    # Moving material covers the target aperture and a substantial left region;
    # the bank cannot match either the target hole or the moving occluder.
    cv2.rectangle(current,(65,65),(120,165),(0,0,0),-1)
    cv2.circle(current,tuple(np.rint(true).astype(int)),17,(0,0,0),-1)
    result=update(cache,current)
    assert result['mode']=='tracked',result
    assert np.linalg.norm(np.asarray(result['hole_px'])-true)<2
    assert result['inliers']>=8
    assert result['coverage']>=.06
    assert result['reference_id']=='train_ref'
    assert not result['gate_positive'] and not result['motor_action']


def test_brief_feature_drop_holds_then_loses_and_stale_resets(tmp_path):
    bank,image=fixture_bank(tmp_path);cache=DemoHoleCache(bank,max_hold_ns=500_000_000)
    first=update(cache,image);assert first['mode']=='tracked'
    blank=np.zeros_like(image)
    held=update(cache,blank,1_100_000_000)
    assert held['mode']=='held'
    assert held['hole_px']==first['hole_px']
    assert held['age_ns']==100_000_000
    lost=update(cache,blank,1_700_000_000)
    assert lost['mode']=='lost' and lost['hole_px'] is None
    update(cache,image,2_000_000_000)
    stale=update(cache,image,2_100_000_000,now_ns=3_000_000_000)
    assert stale['mode']=='lost' and 'stale' in stale['reason']


@pytest.mark.parametrize('change',[{'sequence_id':'other'},{'camera_id':'other'},{'grasp_id':'other'}])
def test_real_context_change_does_not_reuse_held_cache(tmp_path,change):
    bank,image=fixture_bank(tmp_path);cache=DemoHoleCache(bank)
    assert update(cache,image)['mode']=='tracked'
    result=update(cache,np.zeros_like(image),1_100_000_000,**change)
    assert result['mode']=='lost' and result['hole_px'] is None


def test_reference_hash_and_train_provenance_required(tmp_path):
    bank,_=fixture_bank(tmp_path);rows=json.loads(bank.read_text())
    rows['references'][0]['split']='test';bank.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='TRAIN'):DemoHoleCache(bank)
    rows['references'][0]['split']='train';rows['references'][0]['image_sha256']='0'*64;bank.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='SHA'):DemoHoleCache(bank)


def test_degenerate_clustered_and_mirrored_geometry_rejected(tmp_path):
    bank,_=fixture_bank(tmp_path);cache=DemoHoleCache(bank)
    polygon=np.array([[50,40],[269,40],[269,199],[50,199]],float);hole=np.array([165,120.])
    source=np.c_[np.arange(8)*10+70,np.full(8,100.)]
    value,reason=cache._quality(source,source,np.eye(3),np.ones(8,bool),polygon,hole,320,240)
    assert value is None and 'collinear' in reason
    source=np.array([[150+x,100+y] for x in (0,2,4,6) for y in (0,2)],float)
    value,reason=cache._quality(source,source,np.eye(3),np.ones(8,bool),polygon,hole,320,240)
    assert value is None and 'coverage' in reason
    source=np.array([[x,y] for x in (65,120,190,250) for y in (60,180)],float)
    reflection=np.array([[-1,0,320],[0,1,0],[0,0,1]],float)
    value,reason=cache._quality(source,cache._transform(source,reflection),reflection,np.ones(8,bool),polygon,hole,320,240)
    assert value is None and 'orientation' in reason


def test_conflicting_reference_hypotheses_rejected(tmp_path,monkeypatch):
    bank,image=fixture_bank(tmp_path);cache=DemoHoleCache(bank)
    candidates=[{'score':100,'hole':np.array([100.,100.])},{'score':99,'hole':np.array([200.,100.])}]
    monkeypatch.setattr(cache,'_reference_candidates',lambda *args:(candidates,[]))
    result=update(cache,image)
    assert result['mode']=='lost'
    assert 'conflicting' in result['reason']


def test_repeated_timestamp_loses_without_silent_reference_reseed(tmp_path):
    bank,image=fixture_bank(tmp_path);cache=DemoHoleCache(bank)
    assert update(cache,image)['mode']=='tracked'
    result=update(cache,image)
    assert result['mode']=='lost' and result['hole_px'] is None
    assert 'timestamp' in result['reason']


def test_prediction_merge_preserves_pin_gate_delta_and_inputs():
    import copy
    from app.pin_hole_demo_cache import merge_with_prediction
    prediction={'keypoints':[.123,.456,.7,.8],'active_logit':[4.2],'delta_xyz_m':[.001,.002,.003]}
    cache={'mode':'tracked','hole_px':[150,80],'reference_id':'train_ref','reason':None}
    original=copy.deepcopy(prediction);old_cache=copy.deepcopy(cache)
    merged=merge_with_prediction(prediction,cache,(300,200))
    assert merged['keypoints'][:2]==original['keypoints'][:2]
    assert merged['keypoints'][2:]==[.5,.4]
    assert merged['raw_cnn_hole_normalized']==[.7,.8]
    assert merged['active_logit']==original['active_logit']
    assert merged['delta_xyz_m']==original['delta_xyz_m']
    assert merged['cache_valid'] and merged['hole_source']=='demo_reference_tracked'
    assert not merged['gate_permission_from_cache']
    assert prediction==original and cache==old_cache
    merged['active_logit'][0]=99
    assert prediction==original


def test_prediction_merge_lost_and_invalid_cache_fallback_is_explicit():
    from app.pin_hole_demo_cache import merge_with_prediction
    prediction={'keypoints':[.1,.2,.3,.4],'active_logit':[0.],'delta_xyz_m':[0.,0.,0.]}
    for cache in ({'mode':'lost','hole_px':None},{'mode':'held','hole_px':[float('nan'),1]},
                  {'mode':'tracked','hole_px':[500,500]}):
        merged=merge_with_prediction(prediction,cache,(300,200))
        assert merged['keypoints']==prediction['keypoints']
        assert merged['hole_source']=='cnn_unverified_fallback'
        assert not merged['cache_valid']
        assert not merged['gate_permission_from_cache']
