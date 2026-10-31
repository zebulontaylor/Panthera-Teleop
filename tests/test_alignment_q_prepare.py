"""Weak milestone evidence, causal boundaries, and episode holdout contracts."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('alignment_q_prepare',Path(__file__).resolve().parents[1]/'scripts/alignment_q_prepare.py')
prepare=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(prepare)


def rows():
    result=[]
    for i in range(101):
        t=i/10
        grip=1.2 if t<1 or 4<=t<5 else .5
        arms={s:dict(position_rad=[t]*6,gripper={'position':grip if s=='left' else 1.},
                    velocity_rad_s=[1.]*6,tool_pose_base={'position_m':[0.]*3}) for s in ('left','right')}
        result.append(dict(episode_id='20261022Tfixture',time_s=t,arms=arms))
    return result


def manifest():
    return {'episodes':[{'id':'20261022Tfixture','gear':{'start_s':2.,'end_s':3.},'pin':{'start_s':6.,'end_s':7.}}]}


def encoder(row,arms,previous_prediction=None,previous_dt=None):
    return [row['time_s'],previous_prediction['time_s'] if previous_prediction else -1.,previous_dt or 0.]


class PreparationTests(unittest.TestCase):
    def build(self,rs):
        with patch.object(prepare,'encode_observation',side_effect=encoder):
            return prepare.build_transitions(rs,manifest())

    def test_four_phases_rewards_and_causal_history(self):
        transitions,audit=self.build(rows())
        self.assertEqual(set(audit['counts_by_phase']),{'gear_pickup','pin_pickup','gear_insertion','pin_insertion'})
        for phase in audit['counts_by_phase']:
            seq=[r for r in transitions if r['phase']==phase]
            self.assertEqual(seq[0]['observation'][1],-1.)
            self.assertEqual(sum(r['reward'] for r in seq),1.)
            self.assertTrue(seq[-1]['terminal'])
            self.assertFalse(any(r['terminal'] for r in seq[:-1]))
            self.assertEqual(seq[0]['session_id'],'20261022')
            self.assertAlmostEqual(seq[0]['action'][0],.1)
            for a,b in zip(seq,seq[1:]):self.assertEqual(a['next_observation'],b['observation'])
        self.assertEqual(len(set(r['split'] for r in transitions)),1)

    def test_missing_endpoint_cannot_receive_reward(self):
        transitions,audit=self.build([r for r in rows() if not 2.8<=r['time_s']<=3.])
        self.assertFalse(any(r['phase']=='gear_insertion' for r in transitions))
        self.assertEqual(audit['rejected']['gear_insertion:missing_endpoint'],1)

    def test_gap_cannot_bridge_or_receive_reward(self):
        transitions,audit=self.build([r for r in rows() if r['time_s']!=2.5])
        self.assertFalse(any(r['phase']=='gear_insertion' for r in transitions))
        self.assertEqual(audit['rejected']['gear_insertion:gap_or_duplicate'],1)

    def test_retry_uses_last_stable_closure(self):
        rs=rows()
        for r in rs:
            t=r['time_s']
            r['arms']['left']['gripper']['position']=.5 if 1<=t<1.3 or t>=2 else 1.2
        self.assertEqual(prepare.pickup_bounds(rs,0.,3.),(0.,2.))
        for r in rs:r['arms']['left']['gripper']['position']=.5
        self.assertIsNone(prepare.pickup_bounds(rs,0.,3.))

    def test_fixed_action_horizon_rejects_irregular_cadence(self):
        rs=rows()
        for r in rs:
            if r['time_s']==2.5:r['time_s']=2.51
        transitions,audit=self.build(rs)
        self.assertFalse(any(r['phase']=='gear_insertion' for r in transitions))
        self.assertEqual(audit['rejected']['gear_insertion:irregular_action_cadence'],1)
        self.assertEqual(audit['action_horizon_s'],.1)
        self.assertTrue(all(abs(r['dt']-.1)<=1e-6 for r in transitions))

    def test_stable_closure_accepts_jittered_observation_clock(self):
        rs=rows()
        for i,r in enumerate(rs):r['time_s']=i*.099
        bounds=prepare.pickup_bounds(rs,0.,2.)
        self.assertIsNotNone(bounds)
        self.assertAlmostEqual(bounds[1],.99)

    def test_source_manifest_rejects_partial_mutated_or_wrong_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'predictions.jsonl';p.write_text('{}\n')
            sidecar=p.with_suffix('.manifest.json')
            spec=prepare.feature_spec()
            valid=dict(complete=True,output_sha256=prepare.sha256(p),
                       camera_mapping=spec['camera_mapping'],
                       checkpoints={k:{'sha256':v} for k,v in spec['checkpoint_ids'].items()})
            sidecar.write_text(json.dumps(valid))
            self.assertEqual(prepare.validate_source_manifest(p),sidecar)
            for key,value in [('complete',False),('output_sha256','bad'),('camera_mapping',{}),('checkpoints',{})]:
                altered={**valid,key:value};sidecar.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):prepare.validate_source_manifest(p)

    def test_episode_split_is_reproducible_and_day_is_not_relabelled(self):
        ids=['20261022T%02d'%i for i in range(17)]
        a=prepare.episode_splits(ids)
        self.assertEqual(a,prepare.episode_splits(list(reversed(ids))))
        self.assertEqual(list(a.values()).count('train'),11)
        self.assertEqual(list(a.values()).count('val'),3)
        self.assertEqual(list(a.values()).count('test'),3)

if __name__=='__main__':unittest.main()
