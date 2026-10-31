#!/usr/bin/env python3
"""Inspect or stop the independent motor backend without a frontend session."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from motor_client import MotorClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'stop'))
    args = parser.parse_args()
    client = MotorClient('operator')
    try:
        client.connect()
        result = client.rpc('status' if args.action == 'status' else 'shutdown')
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == '__main__':
    sys.exit(main())
