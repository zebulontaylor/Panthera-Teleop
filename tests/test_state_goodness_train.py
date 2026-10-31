import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import state_goodness_train as g

class GoodnessSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.rows=[]
        rng=np.random.default_rng(9)
        for ep,sp,cat in [('a','train','kept'),('b','train','discarded'),('c','val','kept'),('d','test','kept')]:
            for i in range(3):
                refs=[]
                for camera in range(3):
                    ref=f'{ep}-{i}-{camera}.png';Image.fromarray(rng.integers(0,256,(40,40,3),dtype=np.uint8)).save(cls.root/ref);refs.append(ref)
                row=dict(episode_id=ep,session_id=ep,split=sp,source_category=cat,images=refs,t=float(i),state=[i,camera])
                if i==2 and ep in ('b','c','d'):row.update(terminal_outcome=0,terminal_outcome_confidence=.9)
                cls.rows.append(row)
        cls.manifest=cls.root/'frames.jsonl';cls.manifest.write_text(''.join(json.dumps(r)+'\n' for r in cls.rows));cls.pairs=[]
        for ep,sp,offset in [('a','train',0),('c','val',6)]:cls.pairs.append({'episode_id_a':ep,'episode_id_b':ep,'session_id':ep,'split':sp,'preference':'a_better','observation_a':{'current':cls.rows[offset]['images']},'observation_b':{'current':cls.rows[offset+2]['images']}})
        cls.pairs += [dict(cls.pairs[1],preference='equal',pair_id='equal-review'),dict(cls.pairs[1],preference='unknown',pair_id='unknown-review')]
        cls.pref=cls.root/'prefs.jsonl';cls.pref.write_text(''.join(json.dumps(r)+'\n' for r in cls.pairs))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_no_clock_targets_and_balancing(self):
        rows,episodes=g.load_frames(self.manifest,True);self.assertEqual([r['_label'] for r in rows[:3]],[1.,1.,1.]);self.assertEqual(rows[5]['_label'],0.);self.assertIsNone(rows[3]['_label'])
        changed=[dict(r,t=100000.,episode_end_s=-99.) for r in self.rows];path=self.root/'clocks-changed.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in changed));new,_=g.load_frames(path,True);self.assertEqual([r['_label'] for r in rows],[r['_label'] for r in new])
        ds=g.BalancedStates(episodes,self.root,32,8);self.assertEqual([int(ds[i][1]) for i in range(8)],[0,1,0,1,0,1,0,1])
    def test_gradients_and_fresh_head(self):
        model=g.Goodness('resnet18',False);self.assertFalse(hasattr(model,'value'));self.assertFalse(hasattr(model,'progress'));x=torch.randn(2,3,3,32,32);logits=model(x);loss=g.support_loss(logits,torch.tensor([0.,1.]),torch.ones(2),.1);loss.backward();self.assertGreater(float(model.backbone.conv1.weight.grad.abs().sum()),0.)
        p=torch.zeros(1,2,requires_grad=True);g.preference_loss(p,torch.ones(1)).sum().backward();self.assertLess(float(p.grad[0,0]),0.);self.assertGreater(float(p.grad[0,1]),0.)
        frozen=g.Goodness('resnet18',False,True);self.assertFalse(frozen.backbone.conv1.weight.requires_grad);self.assertTrue(frozen.fuse[0].weight.requires_grad)
    def test_preference_macro_ignores_episode_duplication(self):
        first={'session_id':'session_a','episode_id_a':'episode_a','loss':.1,'confidence':1.}
        second={'session_id':'session_b','episode_id_a':'episode_b','loss':.9,'confidence':1.}
        base=g.macro_preference_hinge([first,second]);duplicated=g.macro_preference_hinge([dict(first) for _ in range(100)]+[second]);self.assertAlmostEqual(base,.5);self.assertAlmostEqual(duplicated,.5)
        # A session containing two episodes still receives the same total macro weight.
        third={'session_id':'session_a','episode_id_a':'another_episode','loss':.3,'confidence':1.}
        self.assertAlmostEqual(g.macro_preference_hinge([first,third,second]),.55)
        # Weak confidence reduces absolute influence, even for a whole weak session.
        weak=[dict(first,confidence=.5),dict(second,confidence=.5)]
        self.assertAlmostEqual(g.macro_preference_hinge(weak),base*.5)
        logits=torch.zeros(2,2);labels=torch.ones(2)
        strong_loss=g.confidence_preference_loss(logits,labels,torch.ones(2));weak_loss=g.confidence_preference_loss(logits,labels,torch.full((2,),.5));self.assertAlmostEqual(float(weak_loss),float(strong_loss)*.5)
        fourth=dict(first,loss=.5,confidence=.5)
        self.assertAlmostEqual(g.macro_preference_hinge([first,fourth]),(.1+.5*.5)/2)
    def test_wrists_only_structurally_ignores_overhead(self):
        torch.manual_seed(17);model=g.Goodness('resnet18',False,camera_mode='wrists-only').eval();images=torch.randn(2,3,3,32,32,requires_grad=True)
        original=model(images);overhead_changed=images.detach().clone();overhead_changed[:,0]=torch.randn_like(overhead_changed[:,0])*100
        self.assertTrue(torch.equal(original.detach(),model(overhead_changed).detach()))
        wrist_changed=images.detach().clone();wrist_changed[:,1]=torch.randn_like(wrist_changed[:,1])*3
        self.assertGreater(float((original.detach()-model(wrist_changed).detach()).abs().max()),1e-7)
        original.sum().backward();self.assertEqual(float(images.grad[:,0].abs().sum()),0.);self.assertGreater(float(images.grad[:,1:].abs().sum()),0.)
    def test_selection_weight_is_independent_of_training_weight(self):
        val={'balanced_support_bce':.6,'preference_hinge':.2}
        first=SimpleNamespace(preference_weight=1.,selection_preference_weight=1.)
        third=SimpleNamespace(preference_weight=5.,selection_preference_weight=1.)
        self.assertAlmostEqual(g.validation_selection_score(val,first),.8);self.assertAlmostEqual(g.validation_selection_score(val,third),.8)
        self.assertAlmostEqual(g.validation_selection_score(val,SimpleNamespace(preference_weight=5.,selection_preference_weight=None)),1.6)
    def test_split_and_transfer_guards(self):
        _,eps=g.load_frames(self.manifest,True);bad=[dict(r) for r in self.rows];bad[-1]['session_id']='a';path=self.root/'bad.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
        with self.assertRaisesRegex(ValueError,'session split'):g.load_frames(path,True)
        digest=hashlib.sha256(self.manifest.read_bytes()).hexdigest();self.assertEqual(g.validate_encoder_provenance({'manifest_sha256':digest},eps,digest,None),'identical_frozen_manifest')
        with self.assertRaisesRegex(ValueError,'encoder checkpoint evidence differs'):g.validate_encoder_provenance({'manifest_sha256':'other'},eps,digest,None)
        prefs=[dict(self.pairs[0],split='test')];path.write_text(json.dumps(prefs[0])+'\n')
        with self.assertRaisesRegex(ValueError,'test preference'):g.read_preferences(path,eps)
    def test_cuda_train_resume_and_validation_only_evaluator(self):
        out=self.root/'train';base=[sys.executable,str(SCRIPTS/'state_goodness_train.py'),'--manifest',str(self.manifest),'--dataset-root',str(self.root),'--output',str(out),'--clean-kept-confirmed','--camera-mode','wrists-only','--preference-manifest',str(self.pref),'--preference-image-root',str(self.root),'--backbone','resnet18','--no-pretrained','--image-size','32','--batch-size','2','--preference-batch-size','1','--steps-per-epoch','2','--workers','0','--cpu-threads','2','--gpu-memory-gb','4','--no-amp','--wandb-mode','disabled']
        subprocess.run(base+['--epochs','1'],check=True,capture_output=True,text=True);subprocess.run(base+['--epochs','2','--resume',str(out/'last.pt')],check=True,capture_output=True,text=True)
        history=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()];self.assertEqual([r['step'] for r in history],[1,2,3,4]);self.assertEqual(history[-1]['val']['preference_pairs'],2);self.assertEqual(history[-1]['val']['directional_preference_pairs'],1);self.assertEqual(history[-1]['val']['equal_preference_pairs'],1);self.assertEqual(history[-1]['val']['unknown_preference_pairs'],1);self.assertEqual(history[-1]['val']['directional_sessions'],1);self.assertEqual(history[-1]['val']['evidence_source_metrics']['reviewed_local_pair']['known_pairs'],2);self.assertEqual(history[-1]['val']['evidence_source_metrics']['reviewed_local_pair']['unknown_pairs'],1);self.assertIn('loss',history[-1]['train']);self.assertTrue((out/'shortcut_baselines.json').exists())
        changed=subprocess.run(base+['--epochs','3','--resume',str(out/'last.pt'),'--label-smoothing','.2'],capture_output=True,text=True);self.assertNotEqual(changed.returncode,0);self.assertIn('resume label/selection semantics changed',changed.stderr)
        output=self.root/'val-evaluation.json';rejected=subprocess.run([sys.executable,str(SCRIPTS/'state_goodness_evaluate.py'),'--checkpoint',str(out/'best.pt'),'--manifest',str(self.manifest),'--dataset-root',str(self.root),'--output',str(output),'--split','val','--preference-manifest',str(self.pref),'--preference-image-root',str(self.root),'--device','cpu'],capture_output=True,text=True,check=False);self.assertNotEqual(rejected.returncode,0)
        # Evaluator deliberately rejects a mixed train/val pair file. Supply only validation.
        valfile=self.root/'valpairs.jsonl';valfile.write_text(json.dumps(self.pairs[1])+'\n');subprocess.run([sys.executable,str(SCRIPTS/'state_goodness_evaluate.py'),'--checkpoint',str(out/'best.pt'),'--manifest',str(self.manifest),'--dataset-root',str(self.root),'--output',str(output),'--split','val','--preference-manifest',str(valfile),'--preference-image-root',str(self.root),'--device','cpu'],capture_output=True,text=True,check=True);self.assertEqual(json.loads(output.read_text())['preference_pairs'],1)

if __name__=='__main__':unittest.main()
