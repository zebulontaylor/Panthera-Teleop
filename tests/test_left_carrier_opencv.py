"""Behavioral checks: occlusion and tiny distractors must not become visible labels."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
s=importlib.util.spec_from_file_location('opencv_prepare',Path(__file__).parents[1]/'scripts/prepare_left_carrier_opencv.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def bore(center=(80,80),radius=12):
    im=np.full((160,160),180,np.uint8);cv2.circle(im,center,radius,25,-1);return im

def test_local_rim_requires_large_dark_hole():
    assert m.refine_bore(bore(),(80,80),12) is not None
    assert m.refine_bore(bore(radius=3),(80,80),12) is None
    assert m.refine_bore(np.full((160,160),80,np.uint8),(80,80),12) is None

def test_occlusion_terminates_tracking():
    first=bore(); second=bore((82,81));det=m.track_step(first,second,(80,80),12)
    assert det is not None and np.linalg.norm(np.array(det['center'])-[82,81])<1.5
    occluded=second.copy();occluded[65:98,65:100]=180
    assert m.track_step(second,occluded,(82,81),12) is None

def test_scale_distractor_and_partial_rim_rejected():
    assert m.refine_bore(bore(radius=23),(80,80),12) is None
    partial=bore();partial[:,80:]=180
    assert m.refine_bore(partial,(80,80),12) is None

def test_rim_reports_measured_ellipse_in_native_pixels():
    image=np.full((160,160),180,np.uint8)
    cv2.ellipse(image,(80,80),(14,10),25,0,360,25,-1)
    result=m.refine_bore(image,(80,80),12)
    assert result is not None
    assert np.linalg.norm(np.array(result['center'])-[80,80])<1.
    assert np.allclose(sorted(result['ellipse_axes_px']),[20,28],atol=2.)
    assert result['radius']==np.sqrt(np.prod(result['ellipse_axes_px']))/2

def test_heldout_seeds_never_create_train_labels(tmp_path):
    import json
    seeds=tmp_path/'seeds.json';split=tmp_path/'split.json'
    seeds.write_text(json.dumps({'frames':[{'split':'test','image_path':'does-not-exist'}]}))
    split.write_text(json.dumps({'blocks':[]}))
    report=m.generate(seeds,split,tmp_path/'output')
    assert report['counts']['train_frames']==0
