#!/usr/bin/env python3
"""Fit bidirectional measured task-axis displacements; offline candidate only."""
import argparse
import json
from pathlib import Path
from nudge_calibration_common import fit_response, load_rows, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--axis', required=True, nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = fit_response(load_rows(args.pairs), axis=args.axis)
    write_report(args.output, result, [args.pairs, Path(__file__), Path(__file__).with_name('nudge_calibration_common.py')])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
