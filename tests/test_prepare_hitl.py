"""Assisted ownership cannot become an autonomous-success training interval."""
import pytest
from scripts.prepare_hitl import select_intervals


def metadata(**outcome):
    return {'label':'kept','complete':True,'context':{'mode':'human_in_the_loop'},
            'outcome':{'success':True,**outcome}}


def states(sources):
    return [{'monotonic_ns':(i+1)*1_000_000_000,'control_source':source}
            for i,source in enumerate(sources)]


@pytest.mark.parametrize('source',['operator_nudge','operator_load'])
def test_operator_assistance_never_exported_as_human_or_unassisted(source):
    rows=states(['policy',source,source,'policy'])
    assert select_intervals(metadata(),rows)==[]
    assert select_intervals(metadata(policy_success=True),rows)==[]
    # Physical human corrections remain eligible, with no relabeling of operator controls.
    rows=states([source,'human','human',source,'policy'])
    assert select_intervals(metadata(policy_success=False,human_assisted=True),rows)==[
        ('human_correction',2_000_000_000,3_000_000_000)]


def test_explicit_policy_success_overrides_legacy_success_flag():
    rows=states(['hold','policy','policy','hold'])
    assert select_intervals(metadata(policy_success=False),rows)==[]
    assert select_intervals(metadata(policy_success=True),rows)==[
        ('successful_unassisted',2_000_000_000,3_000_000_000)]
    assert select_intervals(metadata(),rows)==[
        ('successful_unassisted',2_000_000_000,3_000_000_000)]


@pytest.mark.parametrize('field',['human_interventions','operator_interventions','assistance_interventions'])
def test_legacy_metadata_assistance_counts_exclude_autonomous_success(field):
    meta=metadata();meta[field]=1
    assert select_intervals(meta,states(['policy','policy']))==[]


@pytest.mark.parametrize('flag',['human_assisted','operator_assisted'])
def test_legacy_assisted_outcome_excludes_autonomous_success(flag):
    assert select_intervals(metadata(**{flag:True}),states(['policy','policy']))==[]


def test_frozen_right_rows_exclude_success_without_any_new_metadata():
    rows=states(['policy','policy'])
    rows[1]['control_source_by_arm']={'left':'policy','right':'frozen_hold'}
    assert select_intervals(metadata(),rows)==[]
    assert select_intervals(metadata(policy_success=True),rows)==[]
    meta=metadata();meta['operator_intervention_counts']={'frozen_hold':1}
    assert select_intervals(meta,states(['policy','policy']))==[]


def test_unknown_ownership_remains_invalid():
    with pytest.raises(ValueError,match='ownership'):
        select_intervals(metadata(),states(['policy','unknown_operator']))
