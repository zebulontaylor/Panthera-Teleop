import json
from argparse import Namespace
import numpy as np
import pytest
from PIL import Image

torch=pytest.importorskip('torch')
from app.pin_insertion_cnn import build_model,SCHEMA
from scripts.adapt_pin_insertion_cnn import sha256
from scripts.evaluate_pin_insertion_block_model import evaluate,validate_selection,balanced_score


def test_selection_requires_frozen_validation_provenance(tmp_path):
    model=tmp_path/'model.pt';model.write_bytes(b'checkpoint')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    with pytest.raises(ValueError,match='requires frozen'):validate_selection(None,model,manifest)
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps({'manifest_sha256':sha256(manifest),'selected_checkpoint_sha256':sha256(model),
        'selection_split':'validation','test_used_for_selection':False}))
    assert validate_selection(selection,model,manifest)['selection_split']=='validation'
    model.write_bytes(b'new checkpoint')
    with pytest.raises(ValueError,match='mismatch'):validate_selection(selection,model,manifest)


def test_balanced_score_does_not_count_unlabeled_landmark():
    assert balanced_score({'pin':{'error_native_px':{'mean':4}},'hole':{'error_native_px':{'mean':None}}})==4
    assert balanced_score({'pin':{'error_native_px':{'mean':4}},'hole':{'error_native_px':{'mean':8}}})==6


def test_validation_does_not_read_reserved_test_images(tmp_path):
    blocks=[{'block_id':f'B{i+1}','split':split,'first_frame':20*i,'last_frame':20*i+9,'start_s':2*i,'end_s':2*i+1}
        for i,split in enumerate(('train','validation','test'))]
    split={'source_recording_id':'recording','split_unit':'carrier_pose_block','split_frozen_before_coordinate_annotation_or_fitting':True,
        'independent_recordings':1,'blocks':blocks}
    split_path=tmp_path/'split.json';split_path.write_text(json.dumps(split))
    rows=[]
    for i,block in enumerate(blocks):
        path=tmp_path/(block['block_id']+'.png');Image.fromarray(np.full((32,40,3),50*i,np.uint8)).save(path)
        rows.append({'id':str(i),'split':block['split'],'episode_id':'recording::'+block['block_id'],'session_id':'shared_grasp',
            'source_recording_id':'recording','block_id':block['block_id'],'source_frame_index':block['first_frame'],
            'image_path':str(path),'image_sha256':sha256(path),'image_width':40,'image_height':32,'state14':[0]*14,
            'keypoints_px':[20,16,None,None],'valid':[True,False],'uncertainty_px':[2,0],'gate_label':0})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'split_unit':'carrier_pose_block','split_definition_path':str(split_path),
        'split_definition_sha256':sha256(split_path),'frames':rows}))
    # A strict validation-only evaluator must succeed even if test pixels are absent.
    (tmp_path/'B3.png').unlink()
    checkpoint=tmp_path/'model.pt';torch.save({'model':build_model('heatmap').state_dict(),'metadata':{
        'schema':SCHEMA,'model_variant':'heatmap','controlled_arm':'left','coordinate_frame':'right_base','image_size':32,
        'dataset_metadata':{'image_width':40,'image_height':32},'train_groups':['sim_train'],'val_groups':['sim_val']}},checkpoint)
    args=Namespace(checkpoint=str(checkpoint),baseline_checkpoint=str(checkpoint),manifest=str(manifest),split='validation',
        selection=None,output=str(tmp_path/'validation.json'),device='cpu')
    report=evaluate(args)
    assert report['evaluation_split']=='validation'
    assert not report['test_images_or_supervision_used']
    assert not report['split_audit']['session_disjoint']
    assert report['train_mean_baseline']['landmarks']['pin_tip']['error_native_px']['mean']==0
