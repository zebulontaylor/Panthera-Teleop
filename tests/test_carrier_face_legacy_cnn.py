import tempfile
import unittest
from pathlib import Path
import torch
from app.carrier_face_legacy_cnn import LegacyLeftCarrierCNN

class LegacyReuseTests(unittest.TestCase):
 def test_exact_visual_weight_reuse_and_left_context_slice(self):
  model=LegacyLeftCarrierCNN();source={k:torch.full_like(v,.125) for k,v in model.state_dict().items() if k.startswith(('stem.','encoder_','decoder_'))};source['cross_view.0.weight']=torch.arange(96*288,dtype=torch.float32).reshape(96,288);source['cross_view.0.bias']=torch.full((96,),.25);source['heatmap.weight']=torch.arange(6*32,dtype=torch.float32).reshape(6,32,1,1);source['heatmap.bias']=torch.arange(6,dtype=torch.float32)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'original.pt';torch.save({'model_state':source},path);report=model.load_original_adapted(path)
  self.assertTrue(torch.equal(model.stem[0].weight,source['stem.0.weight']));self.assertTrue(torch.equal(model.encoder_sixteenth[3].weight,source['encoder_sixteenth.3.weight']));self.assertTrue(torch.equal(model.decoder_quarter[3].weight,source['decoder_quarter.3.weight']));self.assertTrue(torch.equal(model.cross_view[0].weight,source['cross_view.0.weight'][:,96:192]));self.assertTrue(torch.equal(model.heatmap.weight[1:5],source['heatmap.weight'][2:6]));self.assertEqual(report['fixed_legacy_camera_identity'],'left_wrist index1')
 def test_only_rgb_and_constant_left_identity_enter_encoder(self):
  model=LegacyLeftCarrierCNN().eval();captured=[];hook=model.stem.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach()))
  with torch.inference_mode():output=model(torch.ones(1,3,32,32)*.5)
  hook.remove();self.assertEqual(output['keypoints'].shape,(1,10,2));self.assertTrue(torch.equal(captured[0][:,:3],torch.zeros(1,3,32,32)));self.assertTrue(torch.equal(captured[0][:,5:],torch.tensor([0.,1.,0.])[None,:,None,None].expand(1,3,32,32)))
  with self.assertRaises(TypeError):model(torch.zeros(1,3,32,32),end_effector_positions=torch.zeros(1,6))

class PredictorThresholdTests(unittest.TestCase):
 def test_per_channel_threshold_floor_controls_fit_observation_and_pin(self):
  import numpy as np
  from unittest.mock import patch
  from app.carrier_face_cnn import CarrierFacePredictor,FACE_POINTS_MM
  points=np.concatenate((.5+FACE_POINTS_MM/120,np.array([[.5,.8]],np.float32)))
  probability=torch.tensor([[.7]*10]);probability[0,3]=.4
  class Dummy(torch.nn.Module):
   def forward(self,image):return {'keypoints':torch.tensor(points)[None],'visibility_logits':torch.logit(probability)}
  predictor=CarrierFacePredictor.__new__(CarrierFacePredictor);predictor.device=torch.device('cpu');predictor.model=Dummy();predictor.metadata={'image_size':32,'visibility_threshold':.35,'visibility_threshold_floor':.5,'visibility_thresholds':[.6,.8,.6,.1,.6,.6,.6,.6,.6,.9]}
  geometry={'projected':points[:9].tolist(),'homography':np.eye(3).tolist(),'support':[0,1,2,4]}
  with patch('app.carrier_face_cnn.fit_face_geometry',return_value=geometry) as fit:
   result=predictor.predict(np.zeros((64,64,3),np.uint8))
  self.assertAlmostEqual(float(fit.call_args.kwargs['threshold'][1]),.8);self.assertAlmostEqual(float(fit.call_args.kwargs['threshold'][3]),.5);self.assertFalse(result['visible'][0]);self.assertTrue(result['visible'][1]);self.assertFalse(result['visible'][2]);self.assertIsNone(result['pin_px'])

if __name__=='__main__':unittest.main()
