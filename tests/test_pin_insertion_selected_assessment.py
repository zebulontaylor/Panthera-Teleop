import json
from argparse import Namespace
import numpy as np
import pytest
from scripts.adapt_pin_insertion_cnn import sha256
from scripts.evaluate_pin_insertion_frozen_challenge import image_plane_directions
from scripts.evaluate_pin_insertion_block_memory import evaluate


def test_direction_diagnostic_respects_annotation_uncertainty_and_sign():
    true=np.array([[[0.,0.],[.5,0.]],[[0,0],[.01,0]]])
    pred=np.array([[[.5,0.],[0.,0.]],[[0,0],[.01,0]]])
    metrics=image_plane_directions(pred,true,np.ones((2,2),bool),np.full((2,2),100.),np.ones((2,2)))
    assert metrics['both_landmarks_valid_count']==2
    assert metrics['mean_cosine_all_nondegenerate']==0
    assert metrics['separation_exceeds_uncertainty_count']==1
    assert metrics['mean_cosine_separation_exceeds_uncertainty']==-1


def test_causal_cache_scoring_excludes_seed_frames_and_rejects_drift(tmp_path):
    manifest=tmp_path/'manifest.json';protocol=tmp_path/'protocol.json';predictions=tmp_path/'video.json'
    rows=[{'block_id':'B4','source_frame_index':i,'split':'test','valid':[False,True],
        'image_width':100,'image_height':100,'keypoints_px':[None,None,50,50],'uncertainty_px':[0,2]} for i in (4,10)]
    manifest.write_text(json.dumps({'frames':rows}))
    protocol.write_text(json.dumps({'manifest_sha256':sha256(manifest),'selected_checkpoint_sha256':'model',
        'seed_frame_indices':{'B4':list(range(5))},'clear_visibility_verified':False,'autonomous_memory_eligible':False,
        'future_insertion_reference_or_GT_seed':False,'test_predictions_or_scores_seen_before_declaration':False}))
    frames=[{'block_id':'B4','source_frame_index':i,'source_image_monotonic_ns':1000+i,
        'current_keypoints_normalized':[.2,.3,.7,.5],'cached_hole_normalized':None} for i in range(5)]
    frames.append({'block_id':'B4','source_frame_index':10,'source_image_monotonic_ns':1010,
        'current_keypoints_normalized':[.2,.3,.6,.5],'cached_hole_normalized':[.7,.5]})
    video={'manifest_sha256':sha256(manifest),'checkpoint_sha256':'model','memory_protocol_sha256':sha256(protocol),
        'test_labels_used':False,'future_references_used':False,'frames':frames}
    predictions.write_text(json.dumps(video))
    args=Namespace(manifest=str(manifest),protocol=str(protocol),predictions=str(predictions),output=str(tmp_path/'report.json'))
    report=evaluate(args)
    assert report['summary']['label_count']==1
    assert report['summary']['current_hole_error_px']['mean']==pytest.approx(10)
    assert report['summary']['cached_hole_error_px']['mean']==pytest.approx(20)
    video['frames'][-1]['cached_hole_normalized']=[.5,.5];predictions.write_text(json.dumps(video))
    with pytest.raises(ValueError,match='fixed causal'):evaluate(args)
