"""Checks for offline grasp evaluation bookkeeping and leakage rejection."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import unittest
import tempfile

SPEC=importlib.util.spec_from_file_location('visual_grasp',Path(__file__).parents[1]/'scripts/validate_visual_grasp.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class VisualGraspTests(unittest.TestCase):
 def test_gate_abstention_and_false_empty(self):
    result=m.metrics(np.asarray([0,0,1,2]),np.asarray([[.9,.05,.05],[.04,.9,.06],[.3,.4,.3],[.05,.05,.9]]))
    assert result['empty_false_grasp']['false_positives']==1
    assert result['empty_false_grasp']['empty_frames']==2
    assert result['accepted']==3
    assert result['gate_confusion_columns_empty_gear_pin_abstain'][1]==[0,0,0,1]
    assert result['classes']['gear']['gated_sensitivity']==0

 def test_wilson_uncertainty_zero_errors(self):
    assert m.wilson(0,0) is None
    lower,upper=m.wilson(0,4)
    assert lower==0
    assert .48<upper<.50

 def test_episode_split_leakage_rejected(self):
    tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);tmp_path=Path(tmp.name)
    data={'label_provenance':[], 'observations':[
        {'episode_id':'same','observation_id':'one','split':'train','images':{},
         'labels':{'grasp':{'left':{'object':'unknown','confidence':'low'}}}},
        {'episode_id':'same','observation_id':'two','split':'validation','images':{},
         'labels':{'grasp':{'left':{'object':'unknown','confidence':'low'}}}}]}
    path=tmp_path/'labels.json';path.write_text(json.dumps(data))
    with self.assertRaisesRegex(ValueError,'split leakage'):m.load_rows(path,verify=False)

 def test_identical_source_across_splits_rejected(self):
    tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);tmp_path=Path(tmp.name)
    data={'label_provenance':[], 'observations':[
        {'episode_id':split,'observation_id':split,'split':split,
         'images':{'wrist_port3':{'sha256':'same'}},
         'labels':{'grasp':{'left':{'object':'unknown','confidence':'low'}}}}
        for split in ('train','validation')]}
    path=tmp_path/'labels.json';path.write_text(json.dumps(data))
    with self.assertRaisesRegex(ValueError,'Duplicate image'):m.load_rows(path,verify=False)


if __name__=="__main__": unittest.main()
