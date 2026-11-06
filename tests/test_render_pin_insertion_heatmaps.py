import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.render_pin_insertion_heatmaps import spatial_probability,load_manifest,compose_frame,assert_render_disjoint,manifest_truth

class HeatmapVideoTests(unittest.TestCase):
    def test_spatial_probability_is_per_landmark(self):
        logits=np.zeros((2,3,4));logits[0,1,2]=1000
        p=spatial_probability(logits)
        np.testing.assert_allclose(p.sum((1,2)),[1,1])
        self.assertEqual(np.unravel_index(p[0].argmax(),(3,4)),(1,2))
        with self.assertRaises(ValueError):spatial_probability(np.zeros((1,3,4)))

    def test_manifest_excludes_training(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json'
            p.write_text(json.dumps({'frames':[{'split':'train'},{'split':'test','image_path':'image.jpg','state14':[0]*14}]}))
            self.assertEqual(len(load_manifest(p)),1)
            p.write_text(json.dumps([{'split':'test','image_path':'i','state14':[0]*6}]))
            with self.assertRaises(ValueError):load_manifest(p)

    def test_adaptation_test_provenance_is_allowed(self):
        metadata={'train_groups':['simtrain'],'adaptation':{'train_episode_ids':['realtrain'],
            'validation_episode_ids':['realval'],'reserved_test_episode_ids':['realtest'],
            'manifest':{'episodes':[{'episode_id':'realtest','split':'test'}]}}}
        assert_render_disjoint(['realtest'],metadata)
        with self.assertRaises(ValueError):assert_render_disjoint(['realval'],metadata)

    def test_exact_manual_labels_mask_invisible(self):
        truth=manifest_truth({'keypoints_px':[100,50,None,None],'valid':[True,False],
                              'width':200,'height':100})
        np.testing.assert_allclose(truth[0],[.5,.5]);self.assertTrue(np.isnan(truth[1]).all())

    def test_video_frame_even_resolution(self):
        out={'keypoints':[.3,.4,.6,.7],'delta_xyz_m':[.001,0,0],'active_logit':[0],
             'keypoint_logits':np.zeros((2,8,8))}
        image=compose_frame(np.zeros((48,64,3),dtype=np.uint8),out,'REAL HELDOUT','unit test')
        self.assertEqual(image.shape,(506,1440,3))
        self.assertGreater(image.max(),0)

if __name__=='__main__':unittest.main()
