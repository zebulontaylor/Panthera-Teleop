import cv2
import numpy as np
from app.carrier_face_edge import refine_face_outline

def test_visible_edge_corrects_approximate_outline_and_marks_hidden():
 im=np.full((160,180),40,np.uint8);cv2.circle(im,(90,80),40,180,-1)
 # Predicted profile radius37, actual visible metal40.
 H=np.array([[37/.02649,0,90],[0,37/.02649,80],[0,0,1]],float)
 template={'radial_distance_m':[.02649]*144,'source':'test'}
 out=refine_face_outline(im,H,template=template)
 visible=[o for o in out['edge_observations'] if o['observed']]
 assert len(visible)>60
 before=np.mean([abs(np.linalg.norm(np.array(o['projected_px'])-[90,80])-40) for o in visible])
 after=np.mean([abs(np.linalg.norm(np.array(o['point_px'])-[90,80])-40) for o in visible])
 assert after<before/2
 assert all(not o['metric_homography_eligible'] and o['secondary_fit_weight']<=.05 for o in visible)
 # Visible-only mask excludes own jaw half. Excluded edge stays inferred.
 hidden=refine_face_outline(im,H,face_polygon=[[45,35],[90,35],[90,125],[45,125]],template=template)
 assert all(not o['observed'] for o in hidden['edge_observations'] if o['projected_px'][0]>110)

def test_textureless_image_cannot_confirm_face_outline():
 H=np.array([[1400,0,80],[0,1400,80],[0,0,1]],float)
 out=refine_face_outline(np.full((160,160),120,np.uint8),H,template={'radial_distance_m':[.02649]*144})
 assert out['observed_arc_fraction']==0
 assert all(o['inferred'] for o in out['edge_observations'])
