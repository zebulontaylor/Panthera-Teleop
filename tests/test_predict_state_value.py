"""Single-observation CLI parity against the training export on native camera frames."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from PIL import Image

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import state_value_train as trainer
import predict_state_value as predictor

class SingleObservationInference(unittest.TestCase):
    def test_native_rgb_parity_and_metadata_free_cli(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);refs=[];rng=np.random.default_rng(11)
            for cam in range(3):
                ref=f'view{cam}.png';Image.fromarray(rng.integers(0,256,(480,640,3),dtype=np.uint8)).save(root/ref);refs.append(ref)
            model=trainer.VisualValue(False,'resnet18').eval();ck={'config':{'backbone':'resnet18','image_size':224},'model':model.state_dict(),'goal_bank':torch.randn(2,256),'goal_bank_ids':['train_a','train_b']};checkpoint=root/'best.pt';torch.save(ck,checkpoint)
            loaded,checkpoint_data=predictor.load_model(checkpoint,torch.device('cpu'))
            args=SimpleNamespace(image_size=224,goal_images=None,dataset_root=root,batch_size=1,output=root,value_horizon=30.)
            model.goal_bank=ck['goal_bank'];model.goal_bank_ids=ck['goal_bank_ids']
            row={'episode_id':'reference','split':'predict','t':0.,'images':refs,'_phase':0.,'_remaining':0.,'_weak':False}
            reference=trainer.predict(model,[row],args,torch.device('cpu'))[0]
            score=predictor.score_images(loaded,checkpoint_data,[[root/r for r in refs]],torch.device('cpu'),goal_bank=True)[0]
            for expected,actual in [('weak_demo_progress','weak_demo_progress'),('weak_demo_discounted_value','weak_discounted_value'),('uncalibrated_train_goal_bank_proximity','uncalibrated_train_goal_bank_proximity')]:self.assertAlmostEqual(reference[expected],score[actual],places=7)
            cmd=[sys.executable,str(SCRIPTS/'predict_state_value.py'),'--checkpoint',str(checkpoint),'--device','cpu','--cpu-threads','2','--goal-bank','--image-root',str(root)]
            single=subprocess.run(cmd+['--images',*refs],check=True,capture_output=True,text=True);single_score=json.loads(single.stdout);self.assertAlmostEqual(single_score['weak_demo_progress'],score['weak_demo_progress'],places=7)
            manifest=root/'arbitrary.jsonl';manifest.write_text(json.dumps({'id':'observation_without_episode_metadata','images':refs})+'\n')
            batch=subprocess.run(cmd+['--manifest',str(manifest)],check=True,capture_output=True,text=True);batch_score=json.loads(batch.stdout);self.assertEqual(batch_score['id'],'observation_without_episode_metadata');self.assertTrue(batch_score['uncalibrated']);self.assertAlmostEqual(batch_score['weak_discounted_value'],score['weak_discounted_value'],places=7)

if __name__=='__main__':unittest.main()
