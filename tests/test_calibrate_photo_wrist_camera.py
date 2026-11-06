import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_photo_wrist_camera import fill_envelope,silhouette_metrics


def test_filled_envelope_ignores_internal_rib_slots():
    mask=np.zeros((30,40),np.uint8);mask[5:25,5:35]=1;mask[10:20,12:16]=0
    filled=fill_envelope(mask)
    assert filled[15,14]
    assert not filled[0,0]
    assert np.count_nonzero(filled)==600


def test_silhouette_metrics_distinguish_translation():
    a=np.zeros((50,50),bool);a[10:40,10:30]=True
    exact=silhouette_metrics(a,a)
    assert exact['iou']==1 and exact['score']==0
    b=np.roll(a,8,axis=1);shifted=silhouette_metrics(b,a)
    assert shifted['iou']<.6 and shifted['chamfer_px']>0
