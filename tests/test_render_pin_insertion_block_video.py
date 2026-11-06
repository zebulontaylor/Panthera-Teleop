import json
from pathlib import Path
import numpy as np
import pytest
from scripts.render_pin_insertion_block_video import (chronological_stream, CausalHoleCache,
    memory_ranges, load_contract, compose)
from scripts.adapt_pin_insertion_cnn import sha256


def state(stamp, value):
    return {'monotonic_ns':stamp, 'arms':{side:{'position_rad':[value]*6,
        'gripper':{'position':value}} for side in ('left','right')}}


def test_original_receipt_clock_and_actual_state_not_data_clock():
    frames=[{'camera':'wrist_port2','frame_index':i,'monotonic_ns':100_000_000+i*60_000_000,
             'data_monotonic_ns':9_000_000_000,'episode_time_s':i*.06} for i in range(3)]
    blocks=[{'block_id':'B4','first_frame':0,'last_frame':2,'start_s':0,'end_s':1}]
    states=[state(101_000_000,1),state(158_000_000,2),state(222_000_000,3)]
    rows=chronological_stream(frames,states,blocks)
    assert [r['state14'][0] for r in rows]==[1,2,3]
    assert [r['state_skew_ms'] for r in rows]==[1,-2,2]
    with pytest.raises(ValueError,match='25 ms'):
        chronological_stream(frames,[state(1,0)],blocks)
    with pytest.raises(ValueError,match='strictly increase'):
        chronological_stream([frames[1],frames[0],frames[2]],states,blocks)
    with pytest.raises(ValueError,match='missing'):
        chronological_stream(frames[:2],states,blocks)


def test_cache_is_current_prediction_only_causal_and_resets_blocks():
    cache=CausalHoleCache({'B4':(10,12)})
    assert cache.observe('B4',10,[.1,.2]) is None
    assert cache.observe('B4',11,[.3,.4]) is None
    np.testing.assert_allclose(cache.observe('B4',12,[.2,.3]),[.2,.3])
    np.testing.assert_allclose(cache.observe('B4',13,[.9,.9]),[.2,.3])
    assert cache.observe('B5',20,[.9,.9]) is None
    assert CausalHoleCache({}).observe('B4',10,[.1,.2]) is None


def test_memory_requires_bound_external_protocol_and_early_range(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    block={'block_id':'B4','first_frame':10,'last_frame':300}
    with pytest.raises(ValueError,match='predeclared'):
        memory_ranges(['B4:10:12'],None,manifest,[block])
    protocol=tmp_path/'protocol.json'
    doc={'manifest_sha256':sha256(manifest),'declared_before_test':True,
         'blocks':{'B4':{'frame_range':[10,12],'clear_visible':True,
                         'carrier_pose_locked':True,'grasp_locked':True}}}
    protocol.write_text(json.dumps(doc))
    assert memory_ranges(['B4:10:12'],protocol,manifest,[block])[0]=={'B4':(10,12)}
    doc['blocks']['B4']['frame_range']=[11,12];protocol.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='first block'):
        memory_ranges(['B4:11:12'],protocol,manifest,[block])


def test_selection_and_split_sha_checked_without_label_coordinates(tmp_path):
    checkpoint=tmp_path/'model';checkpoint.write_bytes(b'model')
    source=tmp_path/'source';source.mkdir();(source/'frames.jsonl').write_text('')
    split=tmp_path/'split.json';split.write_text(json.dumps({
        'split_frozen_before_coordinate_annotation_or_fitting':True,'source_path':str(source),
        'source_hashes':{'frames.jsonl':sha256(source/'frames.jsonl')},
        'blocks':[{'block_id':b,'split':'test','first_frame':i} for i,b in enumerate(('B4','B5'))]}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'split_unit':'carrier_pose_block',
        'split_definition_path':str(split),'split_definition_sha256':sha256(split),
        'frames':'no test coordinate rows are required or inspected'}))
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({
        'manifest_sha256':sha256(manifest),'selected_checkpoint_sha256':sha256(checkpoint),
        'selection_split':'validation','test_used_for_selection':False}))
    assert len(load_contract(manifest,checkpoint,selection)[1])==2
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'):
        load_contract(manifest,checkpoint,selection)


def test_panels_keep_pixel_aspect_and_separate_cache_marker():
    pytest.importorskip('cv2')
    rgb=np.zeros((480,640,3),np.uint8)
    prediction={'keypoints':[.4,.5,.6,.5],'keypoint_logits':np.zeros((2,10,10)),
                'active_logit':[0]}
    image=compose(rgb,prediction,np.array([.7,.5]),'current frame',[160,160,320,240])
    assert image.shape==(1146,1280,3)
    assert tuple(image[110+240,448])==(220,50,220) or np.any(image[340:360,438:458,0])
    with pytest.raises(ValueError,match='ROI'):
        compose(rgb,prediction,None,'bad',[0,0,300,240])


def test_mock_stream_renders_all_frames_and_preserves_receipt_durations(tmp_path, monkeypatch):
    import shutil
    from scripts.render_pin_insertion_block_video import ffmpeg_executable
    try:
        ffmpeg_executable()
    except ImportError:
        pytest.skip('ffmpeg unavailable')
    torch=pytest.importorskip('torch')
    from PIL import Image
    from argparse import Namespace
    from app.pin_insertion_cnn import build_model,SCHEMA
    from scripts.render_pin_insertion_block_video import render
    source=tmp_path/'recording';source.mkdir()
    frames=[];states=[]
    for i,t in enumerate([0,.053,1,1.071]):
        Image.fromarray(np.zeros((48,64,3),np.uint8)).save(source/f'{i}.png')
        stamp=1_000_000_000+round(t*1e9)
        frames.append({'camera':'wrist_port2','frame_index':i,'episode_time_s':t,
                       'monotonic_ns':stamp,'path':f'{i}.png','width':64,'height':48})
        states.append(state(stamp,0))
    for name,rows in [('frames.jsonl',frames),('states.jsonl',states)]:
        (source/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    blocks=[{'block_id':'B4','split':'test','first_frame':0,'last_frame':1,'start_s':0,'end_s':.2},
            {'block_id':'B5','split':'test','first_frame':2,'last_frame':3,'start_s':1,'end_s':1.2}]
    split=tmp_path/'split.json';split.write_text(json.dumps({'source_recording_id':'mock',
        'source_path':str(source),'split_frozen_before_coordinate_annotation_or_fitting':True,
        'source_hashes':{n:sha256(source/n) for n in ('frames.jsonl','states.jsonl')},'blocks':blocks}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'split_unit':'carrier_pose_block',
        'split_definition_path':str(split),'split_definition_sha256':sha256(split)}))
    checkpoint=tmp_path/'model.pt';torch.save({'model':build_model('heatmap').state_dict(),'metadata':{
        'schema':SCHEMA,'model_variant':'heatmap','controlled_arm':'left','coordinate_frame':'right_base',
        'image_size':32,'dataset_metadata':{'image_width':64,'image_height':48}}},checkpoint)
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'manifest_sha256':sha256(manifest),
        'selected_checkpoint_sha256':sha256(checkpoint),'selection_split':'validation','test_used_for_selection':False}))
    out=tmp_path/'render.mp4'
    report=render(Namespace(manifest=manifest,checkpoint=checkpoint,selection=selection,output=out,
        device='cpu',memory_init_frame_range=[],memory_protocol=None,roi=[16,16,32,24]))
    assert out.stat().st_size>1000
    assert [r['source_frame_index'] for r in report['frames']]==[0,1,2,3]
    assert [r['duration_s'] for r in report['frames']]==pytest.approx([.053,.053,.071,.071])
    assert all(r['cached_hole_normalized'] is None for r in report['frames'])
    assert not report['test_labels_used'] and not report['future_references_used']


def test_diagnostic_memory_is_sha_bound_unverified_first_five_only(tmp_path):
    manifest=tmp_path/'manifest';manifest.write_text('{}')
    checkpoint=tmp_path/'model';checkpoint.write_bytes(b'model')
    blocks=[{'block_id':b,'first_frame':i,'last_frame':i+100} for b,i in [('B4',10),('B5',200)]]
    doc={'protocol_kind':'diagnostic causal unverified initialization baseline',
         'manifest_sha256':sha256(manifest),'selected_checkpoint_sha256':sha256(checkpoint),
         'clear_visibility_verified':False,'autonomous_memory_eligible':False,
         'future_insertion_reference_or_GT_seed':False,'test_predictions_or_scores_seen_before_declaration':False,
         'seed_frame_indices':{'B4':list(range(10,15)),'B5':list(range(200,205))}}
    path=tmp_path/'protocol';path.write_text(json.dumps(doc))
    ranges,protocol=memory_ranges([],path,manifest,blocks,checkpoint)
    assert ranges=={'B4':(10,14),'B5':(200,204)}
    assert protocol['autonomous_memory_eligible'] is False
    doc['clear_visibility_verified']=True;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='remain unverified'):
        memory_ranges([],path,manifest,blocks,checkpoint)
