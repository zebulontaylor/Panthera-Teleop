"""Hardware mapping and configuration preflight, shared by motor backends."""
import json
import math
from pathlib import Path
import sys
import time

import pinocchio as pin
import yaml

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / 'work/Panthera-HT_SDK/panthera_python'
CONFIG = ROOT / 'config/robot-configs'


class RobotConnectionError(RuntimeError):
    """An expected hardware preflight failure, before opening motor devices."""


def wait_for_joint_feedback(robot, label, timeout=5.0):
    """Reject SDK placeholder readings before any startup trajectory is built."""
    deadline = time.monotonic() + timeout
    consecutive = 0
    while time.monotonic() < deadline:
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
        time.sleep(0.1)
        positions = robot.get_current_pos()
        valid = len(positions) == 6 and all(
            math.isfinite(q) and low <= q <= high
            for q, low, high in zip(positions, robot.joint_limits['lower'], robot.joint_limits['upper'])
        ) and all(m.get_current_motor_state().fault == 0 for m in robot.Motors)
        consecutive = consecutive + 1 if valid else 0
        if consecutive >= 5:
            print(f'{label}: valid joint feedback received; motor faults clear.', flush=True)
            return
    raise RuntimeError(f'{label}: joint feedback invalid or motor fault present; startup refused')


def resolve_board(mapping, sys_tty=Path('/sys/class/tty'),
                  sys_usb=Path('/sys/bus/usb/devices'), dev_root=Path('/dev')):
    found = []
    for tty in sys_tty.glob('ttyACM*'):
        resolved = tty.resolve()
        interface = f"{mapping['usb_path']}:1.0"
        if interface not in resolved.parts or mapping['pci_controller'] not in resolved.parts:
            continue
        usb_device = next((p for p in resolved.parents if p.name == mapping['usb_path']), None)
        if usb_device is None:
            continue
        if (usb_device/'idVendor').read_text().strip() != 'caf1' or (usb_device/'idProduct').read_text().strip() != 'ffff':
            raise RobotConnectionError('Unexpected device at the saved robot USB connection')
        found.append(dev_root/tty.name)
    if len(found) != 1:
        board = (sys_usb / mapping['usb_path']).resolve()
        if board.exists() and mapping['pci_controller'] in board.parts:
            try:
                vendor = (board / 'idVendor').read_text().strip()
                product = (board / 'idProduct').read_text().strip()
                configured = (board / 'bConfigurationValue').read_text().strip()
            except OSError:
                pass  # The device may have disappeared during enumeration.
            else:
                if (vendor, product) != ('caf1', 'ffff'):
                    raise RobotConnectionError('Unexpected device at the saved robot USB connection')
                if configured in ('', '0'):
                    raise RobotConnectionError(
                        f"Robot board at USB {mapping['usb_path']} is detected but USB is unconfigured; "
                        'no serial port is available. Check the robot hub external power supply '
                        'and USB cable, then reconnect the hub. Linux may have rejected the '
                        'configuration because of insufficient available bus power '
                        '(check journalctl -k).')
                raise RobotConnectionError(
                    f"Robot board at USB {mapping['usb_path']} is detected but its serial interface "
                    'is unavailable. Wait for enumeration or check the cdc_acm driver '
                    'and reconnect the robot USB cable.')
        raise RobotConnectionError(f"Expected one robot at USB {mapping['usb_path']}; found {len(found)}. Reconnect its USB cable to the original hub port.")
    device = found[0]
    # The vendor SDK uses prefix matching, so refuse ambiguous tty names.
    if list(device.parent.glob(device.name+'*')) != [device]:
        raise RobotConnectionError(f'SDK prefix matching is ambiguous for {device}; do not launch.')
    return device


def prepare(sides=('left', 'right')):
    sides = tuple(sides)
    if not sides or len(set(sides)) != len(sides) or any(s not in ('left', 'right') for s in sides):
        raise ValueError('Select left, right, or both arms')
    mappings = json.loads((CONFIG/'arm-mapping.json').read_text())
    configs = {}
    devices = {}
    for side in sides:
        device = resolve_board(mappings[side])
        devices[side] = str(device)
        motor = yaml.safe_load((SDK/'robot_param/motor_param/6dof_Panthera_params_follower.yaml').read_text())
        motor['robot']['Serial_Type'] = str(device)
        motor['robot']['CANboard']['No_1_CANboard']['CANport']['CANport_1']['serial_id'] = 1
        motor_path = CONFIG/f'{side}-motors.yaml'
        motor_path.write_text(yaml.safe_dump(motor, sort_keys=False))
        config = yaml.safe_load((SDK/'robot_param/Follower.yaml').read_text())
        config['robot'].update(name=side, robot_name=side, param_file=str(motor_path))
        model_path = (SDK/'robot_param'/config['urdf']['file_path']).resolve()
        config['urdf']['file_path'] = str(model_path)
        model = pin.buildModelFromUrdf(str(model_path))
        assert model.nq == 6 and model.existFrame(config['urdf']['end_effector_link'])
        configs[side] = CONFIG/f'{side}.yaml'
        configs[side].write_text(yaml.safe_dump(config, sort_keys=False))
        print(f'{side.title()} controller -> USB {mappings[side]["usb_path"]} -> {device}', flush=True)
    if len(set(devices.values())) != len(sides):
        raise RuntimeError('Left and right must address different robot devices')
    return configs
