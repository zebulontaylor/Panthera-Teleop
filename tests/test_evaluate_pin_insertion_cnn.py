import json
import tempfile
import unittest
from pathlib import Path
from scripts.evaluate_pin_insertion_cnn import eligible_samples, annotated_error, prediction_stats, assert_disjoint, validate_camera

class EvaluationTests(unittest.TestCase):
    def test_independent_visible_labels_only(self):
        errors=annotated_error({'keypoints':[.5,.5,.25,.25]},
            {'pin_tip':{'position_supervision':True,'xy_px':[50,50]},
             'target_bore':{'position_supervision':False,'xy_px':[25,25]}},[100,100])
        self.assertEqual(errors,{'pin_tip':0.0})

    def test_sync_and_camera_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'metadata.json').write_text(json.dumps({'complete':True,'camera_mapping':{'right_wrist':'wrist_port2'}}))
            row={'state':[0]*14,'monotonic_ns':100000000,'cameras':{'right_wrist':{'monotonic_ns':50000000,'source_camera':'wrist_port2'}}}
            (p/'samples.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(len(eligible_samples(p,max_skew_ms=40)[1]),0)
            self.assertEqual(len(eligible_samples(p,max_skew_ms=60)[1]),1)
            row['cameras']['right_wrist']['source_camera']='wrist_port3'
            (p/'samples.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError): eligible_samples(p,max_skew_ms=60)

    def test_train_validation_and_sources_are_excluded(self):
        metadata={'train_groups':['train'],'val_groups':['val'],
                  'dataset_metadata':{'sources':[{'id':'source'}]}}
        for episode in ('train','val','source'):
            with self.assertRaises(ValueError): assert_disjoint([episode],metadata)
        self.assertEqual(assert_disjoint(['heldout'],metadata)['overlap'],[])

    def test_camera_contract(self):
        validate_camera('right_wrist',{})
        with self.assertRaises(ValueError): validate_camera('left_wrist',{})
        with self.assertRaises(ValueError): validate_camera('right_wrist',{'input_camera':'overhead'})

    def test_unlabeled_is_not_accuracy(self):
        s=prediction_stats([{'prediction':{'delta_xyz_m':[0,0,0]},'active_probability':.8,'camera_sample_skew_ms':2}])
        self.assertIsNone(s['accuracy']); self.assertIsNone(s['insertion_success'])

if __name__=='__main__': unittest.main()
