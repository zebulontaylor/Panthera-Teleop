import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from alignment_servo import AlignmentServo
from test_supervised_alignment import supervised
from test_alignment_assist import advance


def fixture():
    c,s,a,limits=supervised()
    a.update(gravity_model_torque_nm=[0.]*6,tool_pose_base={'position_m':[.2,0.,.1]})
    other=dict(a,position_rad=[0.]*6)
    arms={'left':a,'right':other}
    servo=AlignmentServo(c.config,10.,arms,limits,[5,8,8,3,2,2])
    return servo,s,arms


class ServoTests(unittest.TestCase):
    def test_closed_loop_uses_fresh_frames_converges_and_holds(self):
        servo,s,arms=fixture()
        for i in range(401):
            now=10+i*.05
            q=np.asarray(arms['left']['position_rad'])
            advance(s,now)
            p=s['predictions']['gear']['keypoints_2d']
            p[1][0][1]=108+100*q[1]
            p[2][0][1]=108+100*q[3]
            target=servo.tick(now,s,arms,allowed=True)
            self.assertLessEqual(np.linalg.norm(target-q),.003000001)
            arms['left']['position_rad']=target.tolist()
            arms['left']['tool_pose_base']['position_m']=[.2,0.,.1-target[1]*.05]
            if not servo.running:break
        self.assertEqual(servo.reason,'aligned_holding')
        self.assertGreater(servo.steps,1)
        np.testing.assert_allclose(servo.tick(now+.1,s,arms,allowed=True),target)

    def test_repeated_prediction_does_not_generate_repeated_steps(self):
        servo,s,arms=fixture()
        for i in range(10):servo.tick(10+i*.05,s,arms,allowed=True)
        self.assertEqual(servo.steps,0)

    def test_reversed_image_response_stops_corrections(self):
        servo,s,arms=fixture()
        for i in range(401):
            now=10+i*.05;advance(s,now)
            q=np.asarray(arms['left']['position_rad'])
            p=s['predictions']['gear']['keypoints_2d']
            p[1][0][1]=108-1000*q[1];p[2][0][1]=108-1000*q[3]
            target=servo.tick(now,s,arms,allowed=True)
            arms['left']['position_rad']=target.tolist()
            if not servo.running:break
        self.assertEqual(servo.reason,'image_error_not_improving')
        self.assertLess(servo.steps,10)

    def test_workspace_and_tracking_bounds(self):
        servo,s,arms=fixture()
        arms['left']['tool_pose_base']['position_m']=[.231,0.,.1]
        servo.tick(10.1,s,arms,allowed=True)
        self.assertEqual(servo.reason,'workspace_travel_limit')
        servo,s,arms=fixture();arms['left']['position_rad'][0]=.05
        with self.assertRaisesRegex(ValueError,'tracking_deviation'):
            servo.tick(10.1,s,arms,allowed=True)

    def test_ended_collection_and_carrier_movement_stop(self):
        servo,s,arms=fixture()
        with self.assertRaisesRegex(ValueError,'collection_or_grasp_ended'):
            servo.tick(10.1,s,arms,allowed=False)
        arms['right']['position_rad'][0]=.04
        servo.tick(10.1,s,arms,allowed=True)
        self.assertEqual(servo.reason,'carrier_moved')


if __name__ == '__main__':unittest.main()
