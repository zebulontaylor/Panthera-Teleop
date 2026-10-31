"""Single-arm gravity assistance through an already running motor backend.

No recording, automatic trajectories, or active gripper control. On exit the
backend retains powered hold; stop the backend explicitly to release control.
"""
import argparse
import signal
import time

import numpy as np

from motor_client import MotorClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('left', 'right'), required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required to enable gravity assistance')
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    client = MotorClient('single-arm-hand-guidance')
    try:
        client.open(autostart=False)
        if set(client.robots) != {args.arm}:
            raise RuntimeError(f'Backend must own only the {args.arm} arm')
        print(f'HAND GUIDANCE ACTIVE: {args.arm} arm gravity assisted; gripper passive. '
              'Support the handle. Exit leaves backend in powered hold.', flush=True)
        while not stopping:
            started = time.monotonic()
            samples = client.read()
            for side, (gravity, _) in samples.items():
                robot = client.robots[side]
                robot.pos_vel_tqe_kp_kd(np.zeros(6), np.zeros(6), gravity,
                                       np.zeros(6), np.zeros(6))
                robot.gripper_control_MIT(0, 0, 0, 0, 0)
            client.flush()
            time.sleep(max(0, 1 / 150 - (time.monotonic() - started)))
    finally:
        client.close()
        print('Hand guidance ended; backend retains powered hold.', flush=True)


if __name__ == '__main__':
    main()
