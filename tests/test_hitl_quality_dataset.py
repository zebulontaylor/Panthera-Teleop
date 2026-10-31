import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('quality',Path(__file__).resolve().parents[1]/'scripts/hitl_quality_dataset.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

class QualityTest(unittest.TestCase):
    def test_full_horizon(self):
        self.assertEqual(q.valid_starts(49),[])
        self.assertEqual(q.valid_starts(50),[0])
        self.assertEqual(q.valid_starts(52,50,10),[10,11,12])
    def test_preserve_grippers_and_endpoints(self):
        raw=np.zeros((60,14));raw[15:30,6]=1;raw[20,0]=.01
        smooth,delta=q.smooth_actions(raw)
        np.testing.assert_array_equal(raw[:,q.GRIPPERS],smooth[:,q.GRIPPERS])
        np.testing.assert_array_equal(raw[[0,-1]],smooth[[0,-1]])
        self.assertLessEqual(delta,.008)
        self.assertGreater(smooth[19,0],0)
    def test_reject_excessive_deviation(self):
        raw=np.zeros((10,14));raw[5,0]=1
        with self.assertRaises(ValueError):q.smooth_actions(raw)
    def test_trim_split_never_join(self):
        raw=np.zeros((200,14));raw[20:60,0]=np.arange(40)*.01
        raw[60:120,0]=.39;raw[120:160,0]=.39+np.arange(40)*.01;raw[160:,0]=.78
        cuts=q.active_slices(raw)
        self.assertEqual(len(cuts),2)
        self.assertLess(cuts[0][1],cuts[1][0])
        self.assertGreater(cuts[0][0],0)
        self.assertLess(cuts[-1][1],len(raw))
    def test_hold_handoff_pause_gap(self):
        rows=[dict(control_source='human',recording_segment=0,data_monotonic_ns=1),dict(control_source='hold',recording_segment=0,data_monotonic_ns=2),dict(control_source='human',recording_segment=0,data_monotonic_ns=3),dict(control_source='human',recording_segment=1,data_monotonic_ns=4),dict(control_source='human',recording_segment=1,data_monotonic_ns=200000005)]
        for row in rows: row['monotonic_ns']=row['data_monotonic_ns']
        self.assertEqual(list(q.human_blocks(rows)),[(0,0),(2,2),(3,3),(4,4)])
    def test_camera_gap_splits_without_bridging(self):
        ticks=np.arange(130,dtype=np.int64)*50_000_000
        cameras={'camera':np.r_[ticks[:60],ticks[65:]]}
        self.assertEqual(q.covered_slices(ticks,ticks,cameras),[(0,60),(65,130)])
    def test_short_nudge_is_retained_without_duration_cutoff(self):
        ticks=np.arange(3,dtype=np.int64)*50_000_000
        self.assertEqual(q.covered_slices(ticks,ticks,{'camera':ticks}),[(0,3)])
    def test_masked_loss_excludes_padded_gradients(self):
        import torch
        prediction=torch.tensor([[[1.],[2.],[100.]]],requires_grad=True)
        mask=torch.tensor([[True,True,False]])
        loss=q.masked_action_mse(prediction,torch.zeros_like(prediction),mask)
        self.assertEqual(float(loss.detach()),2.5)
        loss.backward()
        self.assertEqual(float(prediction.grad[0,2,0]),0.)
        self.assertEqual(float(prediction.grad[0,1,0]),2.)
        with self.assertRaises(ValueError):q.masked_action_mean(prediction,torch.zeros_like(mask))
    def test_wrapper_zeros_endpoint_padding_for_short_episode(self):
        import torch,tempfile,json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'processing').mkdir()
            (root/'processing/valid_horizons.json').write_text(json.dumps({'horizon':50,'global_valid_lengths':[2,1]}))
            native=[{'action':torch.ones(50,14)},{'action':torch.ones(50,14)}]
            wrapped=q.MaskedActionDataset(native,root)
            self.assertEqual(int(wrapped[0]['action_valid_mask'].sum()),2)
            self.assertEqual(float(wrapped[0]['action'][2:].sum()),0.)
            self.assertEqual(float(native[0]['action'][2:].sum()),48*14.)
            self.assertEqual(int(wrapped[1]['action_is_pad'].sum()),49)
    def test_pi05_adapter_masks_unreduced_loss(self):
        import torch
        from types import SimpleNamespace
        losses=torch.tensor([[[1.],[4.],[100.]]],requires_grad=True)
        policy=SimpleNamespace(config=SimpleNamespace(rtc_config=None,output_features={'action':SimpleNamespace(shape=(1,))}),_preprocess_images=lambda batch:([],[]),prepare_action=lambda batch:batch['action'],model=SimpleNamespace(forward=lambda *args:losses))
        batch={'action':torch.zeros(1,3,1),'action_valid_mask':torch.tensor([[True,True,False]]),'observation.language.tokens':None,'observation.language.attention_mask':None}
        loss,_=q.masked_pi05_forward(policy,batch)
        self.assertEqual(float(loss.detach()),2.5)
        loss.backward();self.assertEqual(float(losses.grad[0,2,0]),0.)
        self.assertEqual(float(losses.grad[0,0,0]),.5)
        policy.config.rtc_config=object()
        with self.assertRaises(ValueError):q.masked_pi05_forward(policy,batch)
    def test_integrity_complete_does_not_override_stale_or_fault(self):
        import copy
        rows=[]
        for i in range(3):
            rows.append({'sequence':i,'monotonic_ns':i+1,'unix_ns':int((10+i*.01)*1e9),'arms':{arm:{'position_rad':[0.]*6,'gripper':{'position':0.},'motor_fault':[0]*7,'motor_feedback_time':[10+i*.01-.01]*7} for arm in ('left','right')}})
        self.assertIsNone(q.validate_interval_integrity(rows,rows,{'complete':True},{}))
        bad=copy.deepcopy(rows);bad[1]['arms']['left']['motor_fault'][0]=1
        with self.assertRaises(ValueError):q.validate_interval_integrity(bad,bad,{'complete':True},{})
        bad=copy.deepcopy(rows);bad[1]['arms']['left']['motor_feedback_time']=[9.]*7
        with self.assertRaises(ValueError):q.validate_interval_integrity(bad,bad,{'complete':True},{})
        with self.assertRaises(ValueError):q.validate_interval_integrity(rows,rows,{'complete':False},{'source_id':'other','recording_segment':0})
    def test_crop_mismatch_requires_explicit_normalization(self):
        import copy
        source={'fps':20,'image_size':256,'task':'task','image_crops_xyxy':{'overhead':[80,0,560,480]}}
        output=copy.deepcopy(source);output['image_crops_xyxy']={'overhead':[120,80,520,480]}
        with self.assertRaises(ValueError):q.validate_export_contract(output,source)
        output['image_crop_normalization']={'approved':True,'method':'recrop_original_640x480','target_crops_xyxy':output['image_crops_xyxy'],'allowed_source_crops_xyxy':[source['image_crops_xyxy']]}
        q.validate_export_contract(output,source)
        source['fps']=10
        with self.assertRaises(ValueError):q.validate_export_contract(output,source)
    def test_slow_ramp_motion_survives_multiscale_gate(self):
        slow=np.zeros((25,14));slow[:,0]=np.arange(25)*.001
        self.assertEqual(q.active_slices(slow),[(0,25)])
        self.assertEqual(q.active_slices(np.zeros_like(slow)),[])
    def test_static_is_not_candidate(self):
        self.assertEqual(q.active_slices(np.zeros((100,14))),[])

if __name__=='__main__':unittest.main()
