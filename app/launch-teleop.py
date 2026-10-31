"""Resolve the identified robot boards and launch the official dual-arm demo.

Without --run, only validate/generate configurations; no motor devices are opened.
The official demo moves both arms on startup and returns them to zero on exit.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import socket
import struct
import sys
import time

import yaml

from robot_setup import (CONFIG, ROOT, SDK, RobotConnectionError, prepare,
                         resolve_board, wait_for_joint_feedback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Enable the official demo; both arms move on startup and return to zero on exit')
    args = parser.parse_args()
    configs = prepare()
    if not args.run:
        print('Configurations checked. No motor devices opened; no motion commands sent.')
        return
    print('Checking live tracking from both controllers with both grip buttons released...', flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', 5005))
        receiver.settimeout(1)
        deadline = time.monotonic() + 20
        valid_since = None
        while time.monotonic() < deadline:
            try:
                packet, _ = receiver.recvfrom(4096)
            except socket.timeout:
                valid_since = None
                continue
            if len(packet) != 160:
                valid_since = None
                continue
            values = struct.unpack('40f', packet)
            valid = all(math.isfinite(v) for v in values) and values[2] == 1 and values[22] == 1 and values[13] < 0.1 and values[33] < 0.1
            if not valid:
                valid_since = None
                continue
            now = time.monotonic()
            valid_since = now if valid_since is None else valid_since
            if now - valid_since >= 0.5:
                break
        else:
            raise RuntimeError('No stable two-controller tracking with grips released. No robot objects were created.')
    sys.path.insert(0, str(SDK/'scripts'))
    spec = importlib.util.spec_from_file_location('panthera_dual_vr', SDK/'scripts/7_vr_dual_cartesian_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.Panthera

    def configured_robot(config_path):
        filename = Path(config_path).name
        side = {'Leader.yaml': 'left', 'Follower.yaml': 'right'}.get(filename)
        if side is None:
            raise RuntimeError(f'Unexpected robot configuration: {filename}')
        # Recheck endpoint identity immediately before opening hardware.
        mapping = json.loads((CONFIG/'arm-mapping.json').read_text())[side]
        expected = yaml.safe_load((CONFIG/f'{side}-motors.yaml').read_text())['robot']['Serial_Type']
        if str(resolve_board(mapping)) != expected:
            raise RuntimeError('USB endpoints changed after preflight; restart the launcher')
        robot = original(str(configs[side]))
        wait_for_joint_feedback(robot, side)
        return robot

    module.Panthera = configured_robot
    print('Starting official dual-arm demo: automatic startup pose and exit-to-zero are enabled.', flush=True)
    module.main()


if __name__ == '__main__':
    main()
