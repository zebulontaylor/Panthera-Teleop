"""Auxiliary evidence requires resolved dark apertures and excludes own-jaw mask."""
import cv2
import numpy as np
from app.carrier_aux_holes import refine_aperture,detect_aux_holes

def test_small_aperture_and_glint_noise_occlusion():
 im=np.full((100,100),170,np.uint8);cv2.circle(im,(50,50),3,20,-1)
 assert refine_aperture(im,(50,50),3) is not None
 bright=np.full_like(im,30);cv2.circle(bright,(50,50),3,230,-1)
 assert refine_aperture(bright,(50,50),3) is None
 noise=np.full_like(im,170);noise[50,50]=0
 assert refine_aperture(noise,(50,50),3) is None
 im[:,50:]=170
 assert refine_aperture(im,(50,50),3) is None

def test_aux_classes_plane_and_occlusion_mask():
 im=np.full((200,200),180,np.uint8)
 # H metres->pixels, scale2px/mm; inner13.37mm and boss23.3mm.
 H=np.array([[2000,0,100],[0,2000,100],[0,0,1]],float)
 cv2.circle(im,(127,100),3,20,-1);cv2.circle(im,(100,147),3,20,-1)
 poly=[[30,30],[170,30],[170,170],[30,170]]
 obs=detect_aux_holes(im,poly,face_homography=H)
 assert {o['type'] for o in obs}=={'inner_threaded','boss_mounting'}
 boss=next(o for o in obs if o['type']=='boss_mounting')
 assert boss['plane']=='raised_boss' and boss['height_above_face_mm_hypothesis']==4
 assert all(not o['insertion_target'] and not o['metric_homography_eligible'] for o in obs)
 clipped=[[30,30],[170,30],[170,125],[30,125]]
 assert all(o['type']!='boss_mounting' for o in detect_aux_holes(im,clipped,face_homography=H))

def test_global_concentric_bearing_requires_two_resolved_rims():
 from app.carrier_aux_holes import global_central_bearing_candidates
 im=np.full((180,220),170,np.uint8)
 cv2.circle(im,(127,83),20,30,-1);cv2.circle(im,(127,83),14,160,-1);cv2.circle(im,(127,83),8,20,-1)
 candidates=global_central_bearing_candidates(im)
 assert any(np.linalg.norm(np.array(o['center'])-[127,83])<2 and o['concentric_support']>=2 for o in candidates)
 single=np.full_like(im,170);cv2.circle(single,(127,83),20,20,-1)
 assert not global_central_bearing_candidates(single)
