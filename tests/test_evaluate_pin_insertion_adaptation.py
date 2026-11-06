import copy
import json
from argparse import Namespace
import numpy as np
import pytest
from PIL import Image

torch=pytest.importorskip('torch')
from app.pin_insertion_cnn import build_model,SCHEMA
from scripts.adapt_pin_insertion_cnn import sha256
from scripts.evaluate_pin_insertion_adaptation import (gate_confusion,landmark_metrics,verify_split,evaluate)


def test_known_negative_gates_do_not_invent_positive_accuracy():
    result=gate_confusion([.9,.1,.8],[0,0,None],.5)
    assert result['known_positive_labels']==0
    assert result['recall'] is None
    assert result['precision']==0
    assert result['unknown_labels']==1


def test_native_and_normalized_pixel_metrics_and_invalid_mask():
    prediction=np.array([[[.6,.5],[.2,.2]]])
    truth=np.array([[[.5,.5],[.9,.9]]])
    result=landmark_metrics(prediction,truth,np.array([[True,False]]),np.array([[200,100]]),np.array([[3,4]]))
    assert result['pin_tip']['error_native_px']['mean']==pytest.approx(20)
    assert result['pin_tip']['error_normalized_image_coordinates']['mean']==pytest.approx(.1)
    assert result['target_hole']['error_native_px']['mean'] is None
    assert result['pin_tip']['annotation_uncertainty_px']['mean']==3


def frame(identifier,split):
    return {'id':identifier,'episode_id':identifier,'session_id':identifier,'split':split,'image_path':identifier+'.png',
        'image_width':40,'image_height':32,'state14':[0]*14,'keypoints_px':[20,16,None,None],
        'valid':[True,False],'uncertainty_px':2,'gate_label':0}


def metadata():
    return {'schema':SCHEMA,'model_variant':'heatmap','controlled_arm':'left','coordinate_frame':'right_base',
        'image_size':32,'train_groups':['simtrain'],'val_groups':['simval'],'dataset_metadata':{'image_width':40,'image_height':32}}


def test_episode_and_session_leakage_rejected():
    rows=[frame('train','train'),frame('val','validation'),frame('test','test')]
    adapted={'adaptation':{'train_episode_ids':['train'],'validation_episode_ids':['val'],
        'reserved_test_episode_ids':['test'],'test_used_for_training_or_selection':False}}
    verify_split({'frames':rows},adapted,metadata())
    rows[-1]['session_id']='train'
    with pytest.raises(ValueError,match='leakage'):verify_split({'frames':rows},adapted,metadata())
    rows[-1]['session_id']='test'
    baseline=metadata();baseline['train_groups']=['test']
    with pytest.raises(ValueError,match='pretrained'):verify_split({'frames':rows},adapted,baseline)


def test_full_before_after_evaluation_and_manifest_identity(tmp_path):
    rows=[frame('train','train'),frame('val','validation'),frame('test','test')]
    for row in rows:
        row['image_path']=str(tmp_path/row['image_path'])
        Image.fromarray(np.zeros((32,40,3),np.uint8)).save(row['image_path'])
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'frames':rows,'provenance':'synthetic test fixture'}))
    baseline=tmp_path/'baseline.pt';adapted=tmp_path/'adapted.pt';model=build_model('heatmap')
    base_meta=metadata();torch.save({'model':model.state_dict(),'metadata':base_meta},baseline)
    meta=copy.deepcopy(base_meta);meta['adaptation']={'manifest_sha256':sha256(manifest),'pretrained_sha256':sha256(baseline),
        'train_episode_ids':['train'],'validation_episode_ids':['val'],'reserved_test_episode_ids':['test'],
        'test_used_for_training_or_selection':False}
    torch.save({'model':model.state_dict(),'metadata':meta},adapted)
    args=Namespace(checkpoint=str(adapted),baseline_checkpoint=str(baseline),manifest=str(manifest),output=str(tmp_path/'report.json'),device='cpu')
    report=evaluate(args)
    assert report['split_audit']['frames']['test']==1
    assert set(report['results']['adapted'])=={'normal','black_images','shuffled_images'}
    assert report['train_mean_baseline']['landmarks']['pin_tip']['error_native_px']['mean']==0
    assert report['results']['adapted']['normal']['gate']['0.8']['recall'] is None
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='manifest SHA'):evaluate(args)
