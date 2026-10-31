import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from gear_grasp import GearGrasp


class GearGraspTests(unittest.TestCase):
    def test_small_bounded_step_is_held_without_accumulating(self):
        grip=GearGrasp()
        grip.start(10.,'trial',dict(position=.6,velocity=0.),dict(lower=0.,upper=2.))
        command=grip.tick(10.1,'trial',dict(position=.58,velocity=-.05),True)
        self.assertAlmostEqual(command['position_rad'],.55)
        self.assertEqual(command['torque_limit_nm'],.15)
        self.assertEqual(command['speed_limit_rad_s'],.1)
        self.assertFalse(grip.settled)
        grip.start(10.2,'trial',dict(position=.57,velocity=0.),dict(lower=0.,upper=2.))
        self.assertAlmostEqual(grip.target,.55)
        grip.tick(10.7,'trial',dict(position=.56,velocity=0.),True)
        self.assertTrue(grip.settled)

    def test_collection_return_or_episode_change_cancels_hold(self):
        for allowed,episode in ((False,'trial'),(True,'new')):
            grip=GearGrasp();grip.start(10.,'trial',dict(position=.6,velocity=0.),dict(lower=0.,upper=2.))
            self.assertIsNone(grip.tick(11.,episode,dict(position=.56,velocity=0.),allowed))
            self.assertFalse(grip.status()['active'])

    def test_unexpected_opening_and_invalid_feedback_cancel_or_reject(self):
        grip=GearGrasp();grip.start(10.,'trial',dict(position=.6,velocity=0.),dict(lower=0.,upper=2.))
        self.assertIsNone(grip.tick(11.,'trial',dict(position=.7,velocity=0.),True))
        for p,v in ((float('nan'),0.),(.6,.2),(3.,0.)):
            with self.assertRaises(ValueError):
                grip.start(10.,'new',dict(position=p,velocity=v),dict(lower=0.,upper=2.))


if __name__ == '__main__':
    unittest.main()
