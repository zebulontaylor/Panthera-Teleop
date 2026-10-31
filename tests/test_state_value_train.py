"""Meaningful smoke checks for end-to-end gradients, split isolation and CLI checkpoint use."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/state_value_train.py'
spec=importlib.util.spec_from_file_location('state_value_train',SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class StateValueSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.manifest=cls.root/'frames.jsonl';cls.rows=[]
        rng=np.random.default_rng(4)
        for ep,sp in [('a','train'),('b','train'),('c','val'),('d','test')]:
            for t in range(3):
                refs=[]
                for cam in range(3):
                    ref=f'{ep}-{t}-{cam}.jpg';Image.fromarray(rng.integers(0,256,(40,40,3),dtype=np.uint8)).save(cls.root/ref);refs.append(ref)
                cls.rows.append(dict(episode_id=ep,session_id=ep,split=sp,t=float(t),episode_end_s=2.,images=refs,weak_demo=ep!='b',source_category='discarded' if ep=='b' else 'kept',state=[t,cam]))
        for row in cls.rows:
            if row['episode_id'] in ('b','c') and row['t']==2.:row.update(terminal_outcome=0,terminal_outcome_confidence=.9)
        cls.manifest.write_text(''.join(json.dumps(r)+'\n' for r in cls.rows))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_gradients_and_unknown_mask(self):
        model=m.VisualValue(False,'resnet18');batch={'a':torch.randn(2,3,3,32,32),'aug':torch.randn(2,3,3,32,32),'b':torch.randn(2,3,3,32,32),'delta':torch.tensor([.5,-.5]),'phase':torch.tensor([[.1,.6],[.7,.2]]),'remaining':torch.tensor([[20.,10.],[10.,20.]]),'weak':torch.ones(2),'anchor_t':torch.tensor([0.,2.]),'episode_key':torch.tensor([1,2])}
        args=SimpleNamespace(temperature=.1,value_horizon=30.,ssl_weight=.25,temporal_weight=1.,progress_weight=1.,value_weight=1.)
        loss,stats=m.losses(model,batch,args);self.assertTrue(torch.isfinite(loss));loss.backward()
        for module in [model.backbone.conv1,model.distance[0],model.progress[0],model.value[0]]:
            self.assertGreater(float(module.weight.grad.abs().sum()),0)
        batch['weak'].zero_();_,stats=m.losses(model,batch,args)
        for name in ('weak_progress_mse','weak_value_mse','weak_ranking'):self.assertEqual(float(stats[name].detach()),0.)
    def test_near_frame_ssl_mask(self):
        model=m.VisualValue(False,'resnet18').eval();batch={'a':torch.randn(2,3,3,32,32),'aug':torch.randn(2,3,3,32,32),'b':torch.randn(2,3,3,32,32),'delta':torch.tensor([.5,-.5]),'phase':torch.zeros(2,2),'remaining':torch.ones(2,2),'weak':torch.ones(2),'episode_key':torch.tensor([1,1]),'anchor_t':torch.tensor([0.,.5])}
        args=SimpleNamespace(temperature=.1,value_horizon=30.,ssl_weight=.25,temporal_weight=1.,progress_weight=1.,value_weight=1.)
        with torch.no_grad():
            _,stats=m.losses(model,batch,args);self.assertEqual(float(stats['ssl']),0.)
            batch['anchor_t'][1]=2.;_,stats=m.losses(model,batch,args);self.assertGreater(float(stats['ssl']),0.)
    def test_reviewed_terminal_failure_only_trains_value(self):
        model=m.VisualValue(False,'resnet18').eval();batch={'a':torch.randn(2,3,3,32,32),'aug':torch.randn(2,3,3,32,32),'b':torch.randn(2,3,3,32,32),'delta':torch.tensor([.5,-.5]),'phase':torch.zeros(2,2),'remaining':torch.ones(2,2),'weak':torch.zeros(2),'episode_key':torch.tensor([1,2]),'anchor_t':torch.tensor([0.,2.]),'terminal_outcome':torch.zeros(2,2),'terminal_confidence':torch.tensor([[1.,0.],[0.,0.]])}
        args=SimpleNamespace(temperature=.1,value_horizon=30.,ssl_weight=0.,temporal_weight=0.,progress_weight=0.,value_weight=0.,terminal_weight=1.)
        loss,stats=m.losses(model,batch,args);loss.backward()
        self.assertGreater(float(model.value[-1].bias.grad),0.) # gradient descent lowers failed endpoint value
        self.assertEqual(float(model.progress[-1].bias.grad),0.)
        baseline=float(loss.detach());batch['terminal_outcome'][0,1]=1.
        changed,_=m.losses(model,batch,args);self.assertAlmostEqual(float(changed.detach()),baseline,places=6) # unlabeled companion unaffected
    def test_preference_gradient_and_split_guards(self):
        p=torch.tensor([[0.,0.]],requires_grad=True);v=torch.tensor([[0.,0.]],requires_grad=True)
        loss=m.preference_objective(p,v,torch.ones(1)).mean();loss.backward()
        for logits in (p,v):self.assertLess(float(logits.grad[0,0]),0.);self.assertGreater(float(logits.grad[0,1]),0.)
        _,episodes=m.read_manifest(self.manifest)
        pair={'episode_id_a':'a','episode_id_b':'a','split':'train','preference':'a_better','observation_a':{'current':self.rows[0]['images']},'observation_b':{'current':self.rows[1]['images']}}
        manifest=self.root/'prefs.jsonl';manifest.write_text(json.dumps(pair)+'\n');self.assertEqual(len(m.read_preferences(manifest,episodes)),1)
        args=SimpleNamespace(image_size=32,preference_image_root=self.root,preference_batch_size=1)
        model=m.VisualValue(False,'resnet18');pref=m.preference_batch(model,[pair],args,torch.device('cpu'));self.assertTrue(torch.isfinite(pref).all());pref.mean().backward()
        self.assertIsNotNone(model.backbone.conv1.weight.grad)
        pair['split']='test';manifest.write_text(json.dumps(pair)+'\n')
        with self.assertRaisesRegex(ValueError,'test preference'):m.read_preferences(manifest,episodes)
        pair['split']='val';manifest.write_text(json.dumps(pair)+'\n')
        with self.assertRaisesRegex(ValueError,'split leakage'):m.read_preferences(manifest,episodes)
    def test_split_leakage(self):
        rows=[dict(r) for r in self.rows];rows[-1]['session_id']='a';bad=self.root/'bad.jsonl';bad.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'session leakage'):m.read_manifest(bad)
    def test_train_resume_predict(self):
        out=self.root/'run'
        prefs=[]
        for ep,sp,offset in [('a','train',0),('c','val',6)]:prefs.append({'episode_id_a':ep,'episode_id_b':ep,'split':sp,'preference':'a_better','observation_a':{'current':self.rows[offset+2]['images']},'observation_b':{'current':self.rows[offset]['images']}})
        pref_manifest=self.root/'cli-preferences.jsonl';pref_manifest.write_text(''.join(json.dumps(p)+'\n' for p in prefs))
        base=[sys.executable,str(SCRIPT),'--manifest',str(self.manifest),'--dataset-root',str(self.root),'--output',str(out),'--backbone','resnet18','--no-pretrained','--image-size','32','--batch-size','2','--workers','0','--cpu-threads','2','--steps-per-epoch','4','--preference-manifest',str(pref_manifest),'--preference-image-root',str(self.root),'--preference-batch-size','1','--validation-steps','1','--gpu-memory-gb','4','--no-amp']
        subprocess.run(base+['--epochs','1','--export-predictions'],check=True,capture_output=True,text=True)
        first=torch.load(out/'last.pt',map_location='cpu',weights_only=False);self.assertEqual(first['epoch'],1);self.assertEqual(set(first['goal_bank_ids']),{'a'})
        hist=json.loads((out/'history.jsonl').read_text().splitlines()[0]);self.assertEqual(hist['val']['reviewed_terminal_full_coverage']['states'],1);self.assertGreater(hist['val']['reviewed_terminal_bce'],0.);self.assertEqual(hist['val']['preference_full_coverage']['pairs'],1);self.assertIn('preference_hinge',hist['train'])
        subprocess.run(base+['--epochs','2','--resume',str(out/'last.pt')],check=True,capture_output=True,text=True)
        self.assertEqual(torch.load(out/'last.pt',map_location='cpu',weights_only=False)['epoch'],2)
        stage2=self.root/'stage2';stage_args=list(base);stage_args[stage_args.index('--output')+1]=str(stage2)
        changed=subprocess.run(stage_args+['--epochs','3','--preference-weight','.3','--resume',str(out/'last.pt')],capture_output=True,text=True);self.assertNotEqual(changed.returncode,0);self.assertIn('selection objective changed',changed.stderr)
        prior=torch.load(out/'last.pt',map_location='cpu',weights_only=False);prior['best']=-999.;low=self.root/'prior-low.pt';torch.save(prior,low)
        subprocess.run(stage_args+['--epochs','3','--preference-weight','.3','--resume',str(low),'--reset-selection','--export-predictions'],check=True,capture_output=True,text=True)
        self.assertTrue((stage2/'best.pt').exists());reset_report=json.loads((stage2/'run.json').read_text());self.assertTrue(reset_report['selection_reset']['requested']);self.assertEqual(reset_report['selection_reset']['previous_best'],-999.)
        subprocess.run(base+['--predict','--checkpoint',str(out/'last.pt')],check=True,capture_output=True,text=True)
        pred=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()];self.assertEqual(len(pred),len(self.rows));self.assertTrue(all(np.isfinite(r['uncalibrated_train_goal_bank_proximity']) for r in pred));self.assertEqual({r['goal_bank_size'] for r in pred},{1})

if __name__=='__main__':unittest.main()
