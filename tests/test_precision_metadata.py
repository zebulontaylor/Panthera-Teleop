"""Loss-mask behavior at phase boundaries, uncertainty, and recovery clipping."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('precision_metadata', Path(__file__).parents[1] / 'scripts/add_precision_metadata.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_action_target_enters_phase_before_observation_does():
    phases = [dict(phase='pin_insertion', start_s=10., end_s=11., confidence='medium')]
    assert m.frame_label(phases, 9.96, 3.)['loss_weight'] == 1.
    assert m.frame_label(phases, 10.01, 3.)['loss_weight'] == 3.
    assert m.frame_label(phases, 11., 3.)['loss_weight'] == 1.


def test_uncertain_visual_phase_is_recorded_without_promoting_loss():
    phases = [dict(phase='gear_insertion', start_s=1., end_s=2., confidence='visual', visual_reviewer_confidence='medium-low')]
    label = m.frame_label(phases, 1.5, 3.)
    assert label['gear_insertion'] and label['loss_weight'] == 1.


def test_overlapping_phases_do_not_multiply_weights():
    phases = [dict(phase=p, start_s=1., end_s=2., confidence='medium') for p in ['gear_insertion', 'pin_insertion']]
    label = m.frame_label(phases, 1.5, 3.)
    assert label['gear_insertion'] and label['pin_insertion']
    assert label['loss_weight'] == 3.
