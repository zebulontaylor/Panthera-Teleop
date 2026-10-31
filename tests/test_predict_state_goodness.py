import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import state_goodness_train as g
import predict_state_goodness as prediction

class GoodnessInference(unittest.TestCase):
    def test_native_frames_prediction_parity_and_metadata_free_cli(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(7);refs=[]
            for camera in range(3):
                ref=f'camera{camera}.png';Image.fromarray(rng.integers(0,256,(480,640,3),dtype=np.uint8)).save(root/ref);refs.append(ref)
            model=g.Goodness('resnet18',False,camera_mode='wrists-only').eval();checkpoint={'kind':'visual_state_goodness','config':{'backbone':'resnet18','image_size':224,'freeze_encoder':False,'camera_mode':'wrists-only'},'model':model.state_dict()};path=root/'goodness.pt';torch.save(checkpoint,path)
            reader=g.Images(root,224,False)
            with torch.inference_mode():reference=float(torch.sigmoid(model(reader(refs)[None]))[0])
            loaded,ck=prediction.load_model(path,torch.device('cpu'));score=prediction.score_images(loaded,ck,[refs],torch.device('cpu'),root)[0];self.assertAlmostEqual(reference,score['state_goodness_uncalibrated'],places=7)
            base=[sys.executable,str(SCRIPTS/'predict_state_goodness.py'),'--checkpoint',str(path),'--image-root',str(root),'--cpu-threads','2']
            single=subprocess.run(base+['--images',*refs,'--device','cpu'],check=True,capture_output=True,text=True);self.assertAlmostEqual(json.loads(single.stdout)['state_goodness_uncalibrated'],reference,places=7)
            manifest=root/'batch.jsonl';manifest.write_text(json.dumps({'id':'metadata_free','images':refs})+'\n');batch=subprocess.run(base+['--manifest',str(manifest),'--device','cpu'],check=True,capture_output=True,text=True);result=json.loads(batch.stdout);self.assertEqual(result['id'],'metadata_free');self.assertAlmostEqual(result['state_goodness_uncalibrated'],reference,places=7);self.assertEqual(set(result),{'id','state_goodness_uncalibrated','interpretation'})
            if torch.cuda.is_available():
                gpu=subprocess.run(base+['--images',*refs,'--device','cuda','--gpu-memory-gb','4'],check=True,capture_output=True,text=True);self.assertAlmostEqual(json.loads(gpu.stdout)['state_goodness_uncalibrated'],reference,delta=2e-5)

if __name__=='__main__':unittest.main()
