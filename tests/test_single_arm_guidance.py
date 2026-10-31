"""Single-arm selection boundaries, without opening hardware or an SDK robot."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import robot_setup
import single_arm_hand_guide
from motor_backend import MotorEngine, SETTINGS


class SingleArmGuidanceTests(unittest.TestCase):
    def test_prepare_right_does_not_resolve_or_rewrite_left(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config'
            config.mkdir()
            sdk = root / 'sdk'
            params = sdk / 'robot_param'
            (params / 'motor_param').mkdir(parents=True)
            mappings = {'left': {'usb_path': 'missing-left'},
                        'right': {'usb_path': 'right-board'}}
            (config / 'arm-mapping.json').write_text(json.dumps(mappings))
            for filename in ('left.yaml', 'left-motors.yaml'):
                (config / filename).write_text('left must remain unchanged\n')
            motor = {'robot': {'CANboard': {'No_1_CANboard': {
                'CANport': {'CANport_1': {'serial_id': 7}}}}}}
            (params / 'motor_param/6dof_Panthera_params_follower.yaml').write_text(yaml.safe_dump(motor))
            (params / 'Follower.yaml').write_text(yaml.safe_dump({
                'robot': {}, 'urdf': {'file_path': 'robot.urdf', 'end_effector_link': 'tool'}}))
            model = Mock(nq=6)
            model.existFrame.return_value = True
            with patch.object(robot_setup, 'CONFIG', config), patch.object(robot_setup, 'SDK', sdk), \
                    patch.object(robot_setup, 'resolve_board', return_value=Path('/dev/test-right')) as resolve, \
                    patch.object(robot_setup.pin, 'buildModelFromUrdf', return_value=model):
                result = robot_setup.prepare(('right',))
            resolve.assert_called_once_with(mappings['right'])
            self.assertEqual(result, {'right': config / 'right.yaml'})
            saved = yaml.safe_load((config / 'right-motors.yaml').read_text())['robot']
            self.assertEqual(saved['Serial_Type'], '/dev/test-right')
            self.assertEqual(saved['CANboard']['No_1_CANboard']['CANport']['CANport_1']['serial_id'], 1)
            for filename in ('left.yaml', 'left-motors.yaml'):
                self.assertEqual((config / filename).read_text(), 'left must remain unchanged\n')

    def test_right_backend_accepts_only_complete_selected_arm_frame(self):
        robot = Mock()
        robot.joint_limits = {'lower': np.full(6, -2.), 'upper': np.full(6, 2.)}
        robot.gripper_limits = {'lower': 0., 'upper': 2.}
        engine = MotorEngine({'right': robot}, json.loads(SETTINGS.read_text()))
        command = {'joint': {'mode': 'gravity', 'assist_torque_nm': [0.]*6},
                   'gripper': {'mode': 'passive'}}
        self.assertEqual(engine.validate_commands({'right': command}), {'right': command})
        for invalid in ({}, {'left': command}, {'left': command, 'right': command},
                        {'right': {'joint': command['joint']}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                engine.validate_commands(invalid)
        robot.pos_vel_tqe_kp_kd.assert_not_called()

    def test_frontend_rejects_wrong_or_dual_arm_backend_before_commands(self):
        for sides in (('left',), ('left', 'right')):
            client = Mock()
            client.robots = {side: Mock() for side in sides}
            with self.subTest(sides=sides), \
                    patch.object(sys, 'argv', ['single_arm_hand_guide.py', '--arm', 'right', '--run']), \
                    patch.object(single_arm_hand_guide.signal, 'signal'), \
                    patch.object(single_arm_hand_guide, 'MotorClient', return_value=client), \
                    self.assertRaisesRegex(RuntimeError, 'only the right arm'):
                single_arm_hand_guide.main()
            client.open.assert_called_once_with(autostart=False)
            client.read.assert_not_called()
            client.flush.assert_not_called()
            client.close.assert_called_once_with()
            for robot in client.robots.values():
                robot.pos_vel_tqe_kp_kd.assert_not_called()
                robot.gripper_control_MIT.assert_not_called()


if __name__ == '__main__':
    unittest.main()
