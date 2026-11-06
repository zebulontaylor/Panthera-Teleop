"""Frozen-selection, reference-leakage and source-clock evaluation guards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.evaluate_pin_hole_demo_cache import (
    validate_frozen_selection,validate_bank,guard_reference_context,score_predictions,sha,main)
from scripts.render_pin_insertion_block_video import chronological_stream
from scripts.render_pin_hole_demo_cache import display_repetitions

class HarnessGuards(unittest.TestCase):
    def test_test_refuses_before_any_manifest_or_image_access(self):
        argv=['eval','--manifest','missing.json','--references','missing-refs.json','--config','missing-config.json',
              '--split','test','--output','never-created.json']
        with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'requires a frozen'):
            main()

    def test_freeze_binds_config_bank_and_validation_only(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);bank=d/'bank.json';config=d/'config.json';selection=d/'selection.json'
            bank.write_text('{}');config.write_text('{}')
            result={'selection_split':'validation','test_used_for_selection':False,'frozen_before_this_method_test':True,
                    'references_sha256':sha(bank),'config_sha256':sha(config)}
            selection.write_text(json.dumps(result));validate_frozen_selection(selection,bank,config)
            config.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):validate_frozen_selection(selection,bank,config)
            config.write_text('{}');result['test_used_for_selection']=True;selection.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'validation only'):validate_frozen_selection(selection,bank,config)

    def test_train_reference_block_boundary_cannot_be_retagged(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);split=d/'split.json';image=d/'train.jpg';image.write_bytes(b'image')
            split.write_text(json.dumps({'source_recording_id':'recording','blocks':[{'block_id':'B4','split':'test','first_frame':10,'last_frame':20}]}))
            manifest={'split_definition_path':str(split),'split_definition_sha256':sha(split)}
            ref={'id':'fake_train','split':'train','hole_px':[20,30],'source_recording_id':'recording','block_id':'B4',
                 'source_frame_index':12,'image_path':str(image),'image_sha256':sha(image)}
            with self.assertRaisesRegex(ValueError,'TRAIN block boundary'):validate_bank({'references':[ref]},manifest)
            ref['split']='test'
            with self.assertRaisesRegex(ValueError,'Only TRAIN'):validate_bank({'references':[ref]},manifest)

    def test_context_reset_scope_distinguishes_blocks_and_recordings(self):
        guard_reference_context(('recording','B4'),{('recording','B1')},'blocks')
        with self.assertRaises(ValueError):guard_reference_context(('recording','B1'),{('recording','B1')},'blocks')
        with self.assertRaises(ValueError):guard_reference_context(('recording',None),{('recording','B1')},'episodes')

    def test_unavailable_predictions_receive_predeclared_penalty(self):
        records=[{'context_id':'B2','frame_index':n,'demo_cache':{'hole_px':point,'mode':'tracked' if point else 'lost'},
                  'current_cnn_hole_px':[0,0],'first5_cnn_cache_hole_px':None} for n,point in [(1,[3,4]),(2,None)]]
        manifest={'frames':[{'split':'validation','block_id':'B2','source_frame_index':n,'valid':[False,True],
                    'target_hole_px':[0,0],'uncertainty_px':[None,6]} for n in (1,2)]}
        result=score_predictions(records,manifest,'validation','blocks')['summary']['demo_cache_error_px']
        self.assertEqual(result['coverage'],.5);self.assertEqual(result['mean_px'],5.)
        self.assertEqual(result['mean_with_unavailable_40px_penalty'],22.5)

    def test_renderer_preserves_causal_clock_and_resets_at_context(self):
        rows=[{'context_id':'B4','timestamp_ns':i*50_000_000+100} for i in range(20)]
        counts=display_repetitions(rows,30)
        self.assertEqual(sum(counts),round((.95+1/30)*30))
        self.assertTrue(all(n>=1 for n in counts))
        rows.append({'context_id':'B5','timestamp_ns':1})
        self.assertEqual(display_repetitions(rows,30)[-1],1)
        rows=[{'context_id':'B4','timestamp_ns':100},{'context_id':'B4','timestamp_ns':99}]
        with self.assertRaises(ValueError):display_repetitions(rows,30)

    def test_source_clock_regression_is_rejected(self):
        state={'monotonic_ns':100,'arms':{s:{'position_rad':[0]*6,'gripper':{'position':0}} for s in ('left','right')}}
        blocks=[{'block_id':'B2','first_frame':0,'last_frame':1,'start_s':0,'end_s':2}]
        frames=[{'camera':'wrist_port2','frame_index':i,'monotonic_ns':stamp,'episode_time_s':.1+i*.1} for i,stamp in [(0,100),(1,99)]]
        with self.assertRaisesRegex(ValueError,'strictly increase'):chronological_stream(frames,[state],blocks)

if __name__=='__main__':unittest.main()
