"""Hardware-free checks for conservative motor phase selection."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('precision',Path(__file__).resolve().parents[1]/'scripts/annotate_precision_phases.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def candidate(t,width):
    return dict(start_s=t-1,end_s=t,release_s=t,grasp_width_rad=width,stable_hold_s=1,tool_terminal_median_speed_m_s=.01,max_tool_distance_to_release_m=.02)
class PrecisionPhases(unittest.TestCase):
    def test_opening_requires_object_hold_and_clear_transition(self):
        self.assertTrue(p.is_object_opening(.5,.02,.3,.05))
        for args in [(-.01,.02,.3,.05),(.02,.02,.3,.05),(.5,.2,.3,.05),(.5,.02,.2,.05),(.5,.02,.3,.01)]:
            self.assertFalse(p.is_object_opening(*args))
    def test_unique_gear_survives_missing_pin(self):
        phases,abstentions=p.select_phases([candidate(6,.5)])
        self.assertEqual([x['phase'] for x in phases],['gear_insertion']);self.assertIn('pin_insertion',abstentions)
    def test_ambiguous_gear_abstains_even_with_pin(self):
        phases,_=p.select_phases([candidate(6,.5),candidate(9,.4),candidate(15,.1)])
        self.assertEqual(phases,[])
    def test_pin_must_be_unique_ordered_and_nonempty(self):
        for extras in [[candidate(3,.1)],[candidate(15,.01)],[candidate(15,.1),candidate(17,.12)]]:
            phases,_=p.select_phases([candidate(6,.5),*extras]);self.assertEqual(len(phases),1)
        phases,_=p.select_phases([candidate(6,.5),candidate(15,.1)])
        self.assertEqual(len(phases),2)
        for phase in phases:
            self.assertLess(phase['start_s'],phase['end_s']);self.assertEqual(phase['end_s'],phase['release_s']);self.assertEqual(phase['scope'],'pre_release_final_approach');self.assertFalse(phase['contact_verified'])
    def test_pin_only_visual_override_keeps_motor_gear(self):
        import tempfile
        gear=p.select_phases([candidate(6,.5)])[0][0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.json';path.write_text('{}')
            result={'phases':[gear],'phase_abstentions':{'pin_insertion':'No reliable grasp'}}
            visual={'episode_id':'example','pin':{'start_s':10,'end_s':14,'confidence':'medium'}}
            p.apply_visual(result,visual,path,{'start_s':0,'end_s':16})
            self.assertEqual(result['phases'][0],gear)
            self.assertEqual(result['phases'][1]['phase'],'pin_insertion')
            self.assertEqual(result['status'],'proposed')
            self.assertEqual(result['phase_abstentions'],{})
    def test_visual_override_clips_bounds_and_preserves_review(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.json';path.write_text('{}')
            result={'phases':[]};visual={'episode_id':'example','review_assets':['review.jpg'],'gear':{'start_s':1,'end_s':5,'release_s':3,'confidence':'medium'},'pin':{'start_s':7,'end_s':12,'release_s':9,'confidence':'medium','end_right_censored':True}}
            p.apply_visual(result,visual,path,{'start_s':2,'end_s':10})
            self.assertEqual([(x['start_s'],x['end_s']) for x in result['phases']],[(2,5),(7,10)])
            self.assertEqual(result['phases'][1]['provenance']['exact_review_bounds_s'],[7,12])
            self.assertTrue(result['phases'][1]['end_right_censored'])
            self.assertEqual(result['phases'][1]['scope'],'visual_phase_window')
if __name__=='__main__':unittest.main()
