#!/usr/bin/env python3
"""Compare requested, encoder/FK and external tool motion before visual fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from nudge_calibration_common import clean_pairs, finite, load_rows, write_report


def audit(rows):
    pairs, rejected = clean_pairs(rows)
    outcomes = Counter()
    details = []
    for pair in pairs:
        row = pair['row']
        result = dict(session_id=row['session_id'], trial_id=row['trial_id'],
            encoder_fk_delta_m=pair['dx'].tolist())
        try:
            requested = finite(row.get('requested_delta_m'), (3,))
            if not .0002 <= np.linalg.norm(requested) <= .003:
                raise ValueError('invalid_requested_displacement')
            if np.linalg.norm(pair['dx']-requested) > max(.0002, .25*np.linalg.norm(requested)):
                raise ValueError('encoder_motion_disagrees_with_request')
            external = finite(row.get('external_delta_m'), (3,))
            if np.linalg.norm(external-pair['dx']) > .0002:
                raise ValueError('external_motion_disagrees_with_encoder_fk')
            if np.linalg.norm(external-requested) > max(.0002, .25*np.linalg.norm(requested)):
                raise ValueError('external_motion_disagrees_with_request')
            outcome = 'three_way_motion_agreement'
        except (ValueError, TypeError) as exc:
            outcome = str(exc)
            if 'requested_delta_m' not in row:
                outcome = 'requested_displacement_missing'
            elif 'external_delta_m' not in row and outcome == 'invalid_shape_or_nonfinite':
                outcome = 'external_measurement_missing'
        result['outcome'] = outcome
        details.append(result); outcomes[outcome] += 1
    return dict(method='motion_first_audit', eligible_pairs=len(pairs), rejected=rejected,
        outcomes=dict(outcomes), details=details,
        note='Agreement on individual trials does not certify a controller or collision-free motion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    report = audit(load_rows(args.pairs))
    write_report(args.output, report, [args.pairs, Path(__file__)])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
