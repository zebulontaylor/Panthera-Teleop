#!/usr/bin/env python3
"""Supplementary scoring of frozen predictions against blind manual rim audits.

This does not run, seed, select, or update the estimator. Original evaluation
reports remain unchanged; these annotations are a separate measurement audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def score(report_path, audit_path, output):
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if str(audit.get('status', '')).startswith('DRAFT'):
        raise ValueError('Do not score against superseded draft annotations')
    records = {r['frame_index']: r for r in report['records']}
    if len(records) != len(report['records']):
        raise ValueError('Audit requires a single recording with unique frame indices')
    rows = []
    for annotation in audit['annotations']:
        frame = annotation['frame']
        if frame not in records:
            raise ValueError('Missing annotated source frame')
        r = records[frame]; truth = np.asarray(annotation['upper_mouth_center_uv'], float)
        row = {'frame_index': frame, 'context_id': r['context_id'],
               'blind_upper_mouth_px': truth.tolist(),
               'manual_uncertainty_radius_px': annotation['uncertainty_radius_px'],
               'demo_mode': r['demo_cache']['mode']}
        for name, point in [('demo_cache', r['demo_cache']['hole_px']),
                            ('current_cnn', r['current_cnn_hole_px']),
                            ('first5_cnn_cache', r['first5_cnn_cache_hole_px'])]:
            row[name+'_error_px'] = None if point is None else float(np.linalg.norm(np.asarray(point)-truth))
        rows.append(row)
    summary = {}
    common = [r for r in rows if all(r[n+'_error_px'] is not None
              for n in ('demo_cache', 'current_cnn', 'first5_cnn_cache'))]
    for name in ('demo_cache', 'current_cnn', 'first5_cnn_cache'):
        values = [r[name+'_error_px'] for r in rows if r[name+'_error_px'] is not None]
        summary[name] = {'available_labels':len(values), 'total_labels':len(rows),
                         'mean_error_px':float(np.mean(values)) if values else None,
                         'median_error_px':float(np.median(values)) if values else None,
                         'common_available_labels':len(common),
                         'common_available_mean_error_px':float(np.mean([r[name+'_error_px'] for r in common])) if common else None}
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'kind':'supplementary_blind_outer_rim_annotation_audit',
              'prediction_report_path':str(report_path.resolve()), 'prediction_report_sha256':digest(report_path),
              'annotation_audit_path':str(audit_path.resolve()), 'annotation_audit_sha256':digest(audit_path),
              'used_for_model_or_parameter_selection':False, 'original_evaluation_replaced':False,
              'summary':summary, 'rows':rows,
              'limitations':['Independent AI manual rim annotations, not measured metrology.',
                             'Some occluded targets are inferred from a clear view in the same stationary block.',
                             'Errors below manual annotation uncertainty do not certify physical alignment accuracy.']}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','audit','output'):p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args()
    if a.output.resolve() in (a.report.resolve(),a.audit.resolve()):raise ValueError('Preserve inputs')
    score(a.report,a.audit,a.output)
