import copy
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

torch=pytest.importorskip('torch')
from app.pin_insertion_cnn import build_model,SCHEMA,PinInsertionPredictor
from scripts.adapt_pin_insertion_cnn import read_manifest,affine_sample,train


def frame(identifier,split,episode=None):
    return {'id':identifier,'episode_id':episode or identifier,'session_id':episode or identifier,
        'split':split,'image_path':identifier+'.png','state14':[0.]*14,'image_width':40,'image_height':32,
        'keypoints_px':[20.,16.,24.,20.],'valid':[True,True],'gate_label':None,'uncertainty_px':[1.,2.]}


def manifest(tmp_path,frames):
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'frames':frames}));return path


def test_split_leakage_and_source_overlap(tmp_path):
    a=frame('a','train','same');b=frame('b','validation','same')
    with pytest.raises(ValueError,match='leakage'):read_manifest(manifest(tmp_path,[a,b]),{})
    b=frame('b','validation')
    with pytest.raises(ValueError,match='pretrained'):read_manifest(manifest(tmp_path,[a,b]),{'train_groups':['b']})
    b['session_id']='same'
    with pytest.raises(ValueError,match='leakage'):read_manifest(manifest(tmp_path,[a,b]),{})


def test_test_supervision_never_inspected(tmp_path):
    test={'id':'test','episode_id':'test','split':'test','image_path':'missing.png'}
    rows=read_manifest(manifest(tmp_path,[frame('a','train'),frame('b','validation'),test]),{})['frames']
    assert len(rows)==3


def test_invalid_annotation_and_duplicate_image(tmp_path):
    a=frame('a','train');b=frame('b','validation');b['keypoints_px'][0]=float('nan')
    with pytest.raises(ValueError,match='landmarks'):read_manifest(manifest(tmp_path,[a,b]),{})
    b=frame('b','validation');b['image_path']=a['image_path']
    with pytest.raises(ValueError,match='image path leakage'):read_manifest(manifest(tmp_path,[a,b]),{})


def test_affine_transforms_pixel_and_label_together():
    image=np.zeros((80,100,3),np.uint8);image[37:44,47:54]=255
    transformed,kp,valid=affine_sample(image,np.array([.5,.5,.5,.5],np.float32),np.ones(2),np.random.default_rng(4))
    y,x=np.unravel_index(transformed[:,:,0].argmax(),transformed.shape[:2])
    assert np.linalg.norm(kp[:2]*[100,80]-[x,y])<5
    assert valid.all()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_adaptation_preserves_checkpoint_and_freezes_delta(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    metadata={'schema':SCHEMA,'model_variant':'heatmap','controlled_arm':'left','coordinate_frame':'right_base',
        'image_size':32,'train_groups':['synthetic_train'],'val_groups':['synthetic_val'],
        'dataset_metadata':{'image_width':40,'image_height':32},'dataset_sha256':'unused'}
    model=build_model('heatmap');delta=copy.deepcopy(model.delta.state_dict())
    checkpoint=tmp_path/'pretrained.pt';torch.save({'model':model.state_dict(),'metadata':metadata},checkpoint)
    original=checkpoint.read_bytes()
    rows=[frame('a','train'),frame('b','validation'),{'id':'heldout','episode_id':'heldout','split':'test','image_path':'missing.png'}]
    for identifier in ('a','b'):Image.fromarray(np.zeros((32,40,3),np.uint8)).save(tmp_path/(identifier+'.png'))
    path=manifest(tmp_path,rows)
    args=Namespace(checkpoint=str(checkpoint),manifest=str(path),output=str(tmp_path/'adapted.pt'),
        sim_dataset=None,sim_weight=.25,epochs=1,batch_size=2,image_size=32,lr=.0001,seed=3,device=device,threads=1,
        keypoint_weight=1000.,heatmap_weight=10.,gate_weight=1.,affine=True,image_only=True,hole_occlusion_probability=0.)
    result=train(args)
    assert checkpoint.read_bytes()==original
    adapted=PinInsertionPredictor(args.output)
    for key,value in delta.items():assert torch.equal(value,adapted.model.delta.state_dict()[key])
    rgb=np.zeros((32,40,3),np.uint8)
    p0=adapted.predict(rgb,np.zeros(14))
    p1=adapted.predict(rgb,np.ones(14))
    assert p0==p1
    assert result['metadata']['direct_xyz_disabled']
    assert not result['metadata']['adaptation']['test_used_for_training_or_selection']
    assert result['metadata']['adaptation']['reserved_test_episode_ids']==['heldout']


def test_partial_hole_occlusion_preserves_known_amodal_labels_and_masks_gate():
    from scripts.adapt_pin_insertion_cnn import partial_hole_occlusion,make_batch
    import cv2
    image=np.full((128,160,3),120,np.uint8)
    cv2.circle(image,(90,60),12,(20,20,20),-1)
    image[58:63,28:33]=[180,30,30]
    kp=np.array([30/160,60/128,90/160,60/128],np.float32)
    original=kp.copy()
    transformed,audit=partial_hole_occlusion(image,kp,np.ones(2),np.random.default_rng(8),12)
    assert audit is not None
    assert .12<=audit['nominal_hole_covered_fraction']<=.55
    assert audit['gate_supervision_masked']
    assert np.array_equal(kp,original)
    assert np.array_equal(transformed[59:62,29:32],image[59:62,29:32])
    row={'image':image,'keypoints':kp,'valid':np.ones(2),'weight':np.ones(2),'joints':np.zeros(12),
        'gate':1.,'hole_radius_px':12,'wh':np.array([160,128]),'id':'source'}
    _,_,target,weights,gate=make_batch([row],64,'cpu',np.random.default_rng(8),hole_occlusion_probability=1.)
    assert np.allclose(target.numpy()[0],kp)
    assert gate.item()==-1
    assert weights.sum()==4
    _,_,_,_,gate=make_batch([row],64,'cpu',hole_occlusion_probability=1.)
    assert gate.item()==1 # No augmentation without explicit training RNG.


def test_bore_scale_uses_observed_contrast_and_skips_unknown_holes():
    from scripts.adapt_pin_insertion_cnn import estimate_bore_radius,partial_hole_occlusion
    import cv2
    image=np.full((480,640,3),180,np.uint8)
    cv2.circle(image,(320,240),12,(25,25,25),-1)
    assert abs(estimate_bore_radius(image,[320,240])-12)<3
    assert estimate_bore_radius(np.zeros_like(image),[320,240]) is None
    _,audit=partial_hole_occlusion(image,np.array([.4,.5,.5,.5]),np.array([1.,0.]),np.random.default_rng(1),12)
    assert audit is None


def test_contact_replay_rejects_diagnostic_geometry(tmp_path):
    from scripts.adapt_pin_insertion_cnn import load_contact_replay
    path=tmp_path/'diagnostic.npz'
    path.with_suffix('.json').write_text(json.dumps({'diagnostic_only':True,'training_use_approved':False}))
    with pytest.raises(ValueError,match='diagnostic only'):
        load_contact_replay(path,{'frames':[]})


def test_contact_replay_requires_exact_real_train_episode_and_session(tmp_path):
    from scripts.adapt_pin_insertion_cnn import load_contact_replay
    path=tmp_path/'contact.npz'
    rows=[frame('train','train'),frame('validation','validation'),frame('test','test')]
    metadata={'schema_version':1,'generator':'contact_geometry_test.py','camera':'right_wrist','controlled_object':'pin',
        'controlled_arm':'left','delta_frame':'right_native_base','delta_units':'meters','physics_simulated':False,'real_calibrated':False,
        'state_order':'left joints1..6,gripper,right joints1..6,gripper (radians)',
        'keypoints_order':['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v'],'sources':[{'id':'train','session_id':'train'}]}
    def save(meta,group='train'):
        path.with_suffix('.json').write_text(json.dumps(meta))
        np.savez(path,metadata_json=np.asarray(json.dumps(meta)),images=np.zeros((2,32,40,3),np.uint8),
            state=np.zeros((2,14),np.float32),keypoints=np.full((2,4),.5,np.float32),active=np.zeros((2,1),np.float32),
            keypoints_valid=np.ones((2,4),np.float32),group_ids=np.repeat(group,2))
    save(metadata)
    arrays,audit=load_contact_replay(path,{'frames':rows})
    assert len(arrays['images'])==2
    assert audit['groups']==['train']
    bad=copy.deepcopy(metadata);bad['sources'][0]={'id':'validation','session_id':'validation'}
    save(bad,'validation')
    with pytest.raises(ValueError,match='not exclusively real train'):load_contact_replay(path,{'frames':rows})
    bad=copy.deepcopy(metadata);bad['sources'][0]['session_id']='test'
    save(bad)
    with pytest.raises(ValueError,match='not exclusively real train'):load_contact_replay(path,{'frames':rows})
    save(metadata,'test')
    with pytest.raises(ValueError,match='groups contain non-train'):load_contact_replay(path,{'frames':rows})


def test_contact_only_selection_reports_both_landmarks_and_rejects_empty_contact():
    from scripts.adapt_pin_insertion_cnn import validation
    class Center(torch.nn.Module):
        def forward(self,x,q):return {'keypoints':torch.full((len(x),4),.5,device=x.device)}
    common={'image':np.zeros((32,40,3),np.uint8),'valid':np.ones(2),'weight':np.ones(2),'joints':np.zeros(12),'gate':-1.,'wh':np.array([40,32]),'id':'x'}
    approach=dict(common,keypoints=np.array([.1,.1,.1,.1],np.float32),contact_audit=False)
    contact=dict(common,keypoints=np.array([.6,.5,.5,.7],np.float32),contact_audit=True)
    args=Namespace(batch_size=2,image_size=32,image_only=False,select_contact_only=True)
    report=validation(Center(),[approach,contact],args,'cpu')
    assert report['selection_stratum']=='contact'
    assert report['selection_mean_error_px']==pytest.approx(5.2,abs=1e-5)
    assert report['strata']['contact']['landmarks']['pin_tip']['valid_count']==1
    assert report['strata']['contact']['landmarks']['target_hole']['mean_error_px']==pytest.approx(6.4,abs=1e-5)
    with pytest.raises(ValueError,match='no valid landmark'):validation(Center(),[approach],args,'cpu')


def test_contact_replay_honors_internal_heldout_session(tmp_path):
    from scripts.adapt_pin_insertion_cnn import load_contact_replay
    path=tmp_path/'contact.npz'
    a=frame('train_a','train');b=frame('train_b','train')
    metadata={'schema_version':1,'generator':'pin_contact_sim.py','camera':'right_wrist','controlled_object':'pin',
        'controlled_arm':'left','delta_frame':'right_native_base','delta_units':'meters','physics_simulated':False,'real_calibrated':False,
        'state_order':'left joints1..6,gripper,right joints1..6,gripper (radians)',
        'keypoints_order':['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v'],
        'sources':[{'id':'train_a','session_id':'train_a'},{'id':'train_b','session_id':'train_b'}],'sim_validation_session_id':'train_b'}
    path.with_suffix('.json').write_text(json.dumps(metadata))
    np.savez(path,metadata_json=np.asarray(json.dumps(metadata)),images=np.zeros((2,32,40,3),np.uint8),
        state=np.zeros((2,14),np.float32),keypoints=np.full((2,4),.5,np.float32),active=np.zeros((2,1),np.float32),
        keypoints_valid=np.ones((2,4),np.float32),group_ids=np.array(['train_a','train_b']),split=np.array(['train','validation']))
    arrays,audit=load_contact_replay(path,{'frames':[a,b]})
    assert arrays['group_ids'].tolist()==['train_a']
    assert audit['internal_split']['validation_samples']==1
    assert not audit['internal_split']['validation_used_for_optimization']


def make_block_manifest(tmp_path):
    from scripts.adapt_pin_insertion_cnn import sha256
    definition={'source_recording_id':'recording','split_unit':'stationary_carrier_pose_block_within_one_recording',
        'split_frozen_before_coordinate_annotation_or_fitting':True,'independent_recordings':1,
        'blocks':[{'block_id':'B1','split':'train','first_frame':0,'last_frame':9,'start_s':0,'end_s':1},
                  {'block_id':'B2','split':'validation','first_frame':20,'last_frame':29,'start_s':2,'end_s':3},
                  {'block_id':'B3','split':'test','first_frame':40,'last_frame':49,'start_s':4,'end_s':5}]}
    definition_path=tmp_path/'split.json';definition_path.write_text(json.dumps(definition))
    rows=[]
    for j,(block,split,index,time) in enumerate((('B1','train',1,.1),('B2','validation',21,2.1),('B3','test',41,4.1))):
        row=frame(block,split,'recording::'+block)
        row.update(source_recording_id='recording',block_id=block,source_frame_index=index,actual_episode_time_s=time,session_id='same_actual_grasp')
        image=tmp_path/(block+'.png');Image.fromarray(np.full((32,40,3),j*50,np.uint8)).save(image)
        row['image_path']=str(image);row['image_sha256']=sha256(image)
        rows.append(row)
    contract={'split_unit':'carrier_pose_block','split_definition_path':str(definition_path),
        'split_definition_sha256':sha256(definition_path),'frames':rows}
    path=tmp_path/'block-manifest.json';path.write_text(json.dumps(contract));return path,contract


def test_truthful_block_holdout_allows_shared_actual_session(tmp_path):
    path,contract=make_block_manifest(tmp_path)
    # Test pixels are reserved: declared SHA is checked for duplicate provenance,
    # but the adaptation reader must never open the held-out test image.
    Path(contract['frames'][2]['image_path']).unlink()
    result=read_manifest(path,{})
    audit=result['_split_audit']
    assert audit['independent_recordings']==1
    assert not audit['session_disjoint']
    assert not audit['independent_episode_holdout']
    assert audit['shared_sessions_across_splits']==['same_actual_grasp']
    assert audit['test_image_pixels_read_by_trainer'] is False


@pytest.mark.parametrize('mutation,match',[
    (lambda c:c['frames'][1].update(source_frame_index=1),'frozen block'),
    (lambda c:c['frames'][1].update(episode_id='pretend_independent_episode'),'truthful'),
    (lambda c:c['frames'][1].update(image_sha256=c['frames'][0]['image_sha256']),'hash leakage'),
    (lambda c:c['frames'][1].update(actual_episode_time_s=1.5),'time bounds'),
    (lambda c:c.update(split_definition_sha256='0'*64),'definition SHA'),
])
def test_block_contract_rejects_leakage_and_false_identity(tmp_path,mutation,match):
    path,contract=make_block_manifest(tmp_path);mutation(contract);path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match=match):read_manifest(path,{})


def test_affine_preserves_protocol_positive_when_pin_was_already_hidden():
    from scripts.adapt_pin_insertion_cnn import make_batch
    row={'image':np.zeros((32,40,3),np.uint8),'keypoints':np.array([.5,.5,.6,.6],np.float32),
        'valid':np.array([0.,1.]),'weight':np.array([0.,1.]),'joints':np.zeros(12),'gate':1.,'wh':np.array([40,32]),'id':'contact'}
    audit={'attempted':0,'radii':[],'coverage':[]}
    _,_,_,_,gate=make_batch([row],32,'cpu',np.random.default_rng(4),affine=True,augmentation_audit=audit)
    assert gate.item()==1
    assert audit['gate_input_positive']==1
    assert audit['gate_used_positive']==1
