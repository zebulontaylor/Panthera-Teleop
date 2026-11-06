import base64
import time
import numpy as np
import pytest
from app.left_carrier_alignment import LeftCarrierAlignmentWorker
from app.pin_alignment_monitor import PinAlignmentMonitor, validated_config


def request():
    stamp = time.monotonic_ns()
    return dict(config=validated_config(), timestamp_ns=stamp, measured_timestamp_ns=stamp,
        image=dict(kind='rgb', data=base64.b64encode(bytes(640*480*3)).decode()),
        state=[0.]*14, cache_epoch=0, context_id='grasp', camera_id='left_wrist', grasp_id='grasp')


@pytest.fixture
def worker(monkeypatch):
    worker = LeftCarrierAlignmentWorker.__new__(LeftCarrierAlignmentWorker)
    worker.context = None
    worker.target_index = None
    worker.checkpoint = 'left.pt'
    worker.prediction = dict(holes_px=[[300., 200.], [340., 240.], None, None],
        visible=[True, False, False, False], supported=True, pin_px=[280., 200.],
        homography=[[1000., 0., 300.], [0., 1000., 200.], [0., 0., 1.]])
    worker.predictor = type('P', (), {'predict': lambda _, rgb: worker.prediction})()
    worker.calls = []
    def estimate(state, pin, hole, **kw):
        from app.pin_alignment_geometry import preview_from_geometry
        worker.calls.append((list(pin), list(hole)))
        return preview_from_geometry(pin, hole, np.diag([100.,100.,1.]), np.eye(4),
            [0.,0.,1.], [0.,0.,1.], np.eye(3), **kw)
    monkeypatch.setattr('app.pin_alignment_geometry.estimate_left_preview_nudge', estimate)
    return worker


def test_left_fresh_observation_preserves_nudge_bounds_and_no_motor_permission(worker):
    result = worker.score(request())
    assert result['camera_role'] == 'left_wrist'
    assert result['hole_source'] == 'left_carrier_cnn_observed'
    assert result['nudge_valid']
    assert result['nudge_left_mm'] == pytest.approx([5.,0.,0.])
    assert not result['motor_permission']
    assert not result['cnn_xyz_head_used']
    assert result['carrier_holes'][1]['inferred']
    assert worker.calls == [([280.,200.],[300.,200.])]
    req = request();req['camera_id'] = 'right_wrist'
    with pytest.raises(ValueError, match='left_wrist'): worker.score(req)


def test_absent_pin_and_unsupported_hidden_hole_cannot_nudge(worker):
    worker.prediction['pin_px'] = None
    result = worker.score(request())
    assert result['status'] == 'waiting_for_pin'
    assert not result['nudge_valid']
    worker.prediction['pin_px'] = [280.,200.]
    worker.prediction['supported'] = False
    worker.target_index = 1
    req = request()
    req['captured_hole'] = dict(context={k:req[k] for k in ('camera_id','grasp_id','context_id')},
        hole_index=1, hole_px=[340.,240.], source='captured_operator_reference')
    result = worker.score(req)
    assert result['hole_px'] is None
    assert not result['nudge_valid']


def test_selected_hole_relocalizes_after_camera_motion_and_does_not_hold_pixels(worker):
    req = request()
    req['captured_hole'] = dict(context={k:req[k] for k in ('camera_id','grasp_id','context_id')},
        hole_index=1, hole_px=[340.,240.], source='captured_operator_reference')
    first = worker.score(req)
    assert first['hole_px'] == [340.,240.]
    worker.prediction['holes_px'][1] = [360.,250.]
    req['state'][0] = .1
    second = worker.score(req)
    assert second['hole_px'] == [360.,250.]
    assert second['target_hole_index'] == 1
    # Explicit loss of geometry cannot be covered by an old fixed pixel.
    worker.prediction['supported'] = False
    assert worker.score(req)['hole_px'] is None


def test_manual_pin_expires_after_pose_changes_and_stale_frames_block(worker):
    req = request()
    req['captured_pin'] = dict(context={k:req[k] for k in ('camera_id','grasp_id','context_id')},
        pin_px=[250.,200.], measured_state14=[0.]*14)
    assert worker.score(req)['pin_px'] == [250.,200.]
    req['state'][0] = .03
    assert worker.score(req)['pin_px'] is None
    req.pop('captured_pin')
    req['timestamp_ns'] -= 3_000_000_000
    req['measured_timestamp_ns'] = req['timestamp_ns']
    result = worker.score(req)
    assert result['stale_result'] and not result['nudge_valid']
    assert result['arrow_kind'] == 'none'


def test_monitor_captures_face_identity_and_maps_arbitrary_face_point():
    monitor = PinAlignmentMonitor()
    try:
        def publish():
            monitor._publish(dict(status='ready', localization_ready=True, camera_role='left_wrist',
                analyzed_timestamp_ns=time.monotonic_ns(), measured_state14=[0.]*14,
                hole_source='left_carrier_layout_inferred', hole_px=[300.,200.],
                carrier_holes=[dict(index=2,hole_px=[300.,200.])],
                carrier_homography=[[1000.,0.,300.],[0.,1000.,200.],[0.,0.,1.]], carrier_geometry_supported=True),
                dict(request_id=1,generation=monitor._generation), {}, time.monotonic())
        publish()
        captured = monitor.capture_hole(1)
        assert captured['camera_role'] == 'left_wrist'
        assert captured['hole_index'] == 2
        assert captured['source'] == 'captured_operator_reference'
        monitor.clear_hole()
        publish()
        captured = monitor.capture_hole(1, [350.,250.])
        assert 'hole_index' not in captured
        assert captured['face_point_m'] == pytest.approx([.05,.05])
    finally:
        monitor.close()


def test_unregistered_cnn_points_cannot_authorize_nudge(worker):
    worker.prediction['supported'] = False
    result = worker.score(request())
    assert not result['nudge_valid'] and not result['localization_ready']
    assert result['hole_source'] == 'cnn_unverified_fallback'
    assert result['cache']['mode'] == 'lost'
    assert not result['carrier_holes'][0]['observed']


def test_axis_model_overrides_coarse_pin_and_labels_inferred_bottom(worker):
    worker.pin_predictor=type('P',(),{'predict':lambda _,rgb:dict(
        top_px=[335.,310.],bottom_px=[335.,290.],bottom_inferred=True)})()
    result=worker.score(request())
    assert result['pin_px']==[335.,290.]
    assert result['pin_top_px']==[335.,310.]
    assert result['pin_bottom_inferred']
    assert result['pin_source']=='center_axis_inferred'
    worker.pin_predictor.predict=lambda rgb:dict(top_px=None,bottom_px=None)
    result=worker.score(request())
    assert result['pin_px'] is None and not result['nudge_valid']


def test_stationary_pool_is_exposed_without_claiming_a_fresh_detection(worker):
    worker.hole_backend='opencv'
    class Pool:
        def reset(self,reason): self.reason=reason
        def update(self,prediction,state,stamp,config,*,eligible):
            assert eligible
            return dict(prediction, visible=[False]*4,
                holes=[dict(accumulated=True,sample_count=4,last_seen_age_s=.7)]+[{}]*3,
                accumulation=dict(stationary=True,frames=5,point_count=6,retained_count=1))
    worker.accumulator=Pool()
    result=worker.score(request())
    assert result['hole_source']=='left_carrier_stationary_accumulated'
    assert result['cache']['mode']=='held'
    assert result['carrier_holes'][0]['accumulated']
    assert not result['carrier_holes'][0]['observed']
    assert not result['carrier_holes'][0]['inferred']
    assert result['carrier_holes'][0]['sample_count']==4
    assert result['nudge_valid'] and not result['motor_permission']


def test_automatic_target_waits_for_supported_face_map(worker):
    worker.prediction['supported']=False
    first=worker.score(request())
    assert first['target_hole_index'] is None
    worker.prediction['supported']=True
    worker.prediction['holes_px'][3]=[285.,200.]
    second=worker.score(request())
    assert second['target_hole_index']==3
    assert second['hole_px']==[285.,200.]
    assert second['target_selection']['mode']=='automatic'


def test_automatic_target_rechecks_nearest_bore_when_map_improves(worker):
    first = worker.score(request())
    assert first['target_hole_index'] == 0
    # A later map adds the occluded bore beside the pin. The old automatic
    # selection must not remain latched to the farther bore.
    worker.prediction['holes_px'][3] = [285., 200.]
    result = worker.score(request())
    assert result['target_hole_index'] == 3
    assert result['hole_px'] == [285., 200.]
    assert result['pixel_error_px'] == pytest.approx(5.)


def test_automatic_target_follows_position_after_slot_renumbering(worker):
    first = worker.score(request())
    worker.prediction['holes_px'][0], worker.prediction['holes_px'][1] = (
        worker.prediction['holes_px'][1], worker.prediction['holes_px'][0])
    worker.prediction['visible'][:2] = [False, True]
    result = worker.score(request())
    assert result['target_hole_index'] == 1
    assert result['hole_px'] == first['hole_px']


def test_fresh_accumulated_target_reaches_nudge_geometry_without_median_lag(worker):
    from app.stationary_face_accumulation import StationaryFaceAccumulator
    req = request()
    req['captured_hole'] = dict(context={k: req[k] for k in ('camera_id', 'grasp_id', 'context_id')},
        hole_index=0, source='captured_operator_reference')
    worker.score(req)  # Initialize context before attaching the warmed pool.
    pool = StationaryFaceAccumulator()
    for offset in (600_000_000, 400_000_000, 200_000_000):
        pool.update(worker.prediction, req['state'], req['timestamp_ns']-offset, req['config'])
    worker.accumulator = pool
    worker.prediction['holes_px'][0] = [303., 200.]
    result = worker.score(req)
    assert result['accumulation']['stationary']
    assert result['carrier_holes'][0]['observed']
    assert result['hole_px'] == [303., 200.]
    assert result['error_px'] == [23., 0.]
    assert worker.calls[-1] == ([280., 200.], [303., 200.])


def test_stationary_pin_is_pooled_and_retained_without_claiming_visibility(worker):
    from app.stationary_face_accumulation import StationaryFaceAccumulator
    req = request()
    worker.score(req)
    worker.accumulator = StationaryFaceAccumulator()
    for offset in (600_000_000, 400_000_000, 200_000_000):
        stamp = req['timestamp_ns']-offset
        worker.accumulator.update(worker.prediction, req['state'], stamp, req['config'])
        worker.accumulator.accumulate_pin(dict(top_px=[335., 310.], bottom_px=[335., 290.],
            bottom_inferred=True, top_observed=False), stamp, req['config'])
    worker.pin_predictor = type('P', (), {'predict': lambda _, rgb: dict(
        top_px=[335., 314.], bottom_px=[335., 294.], bottom_inferred=True,
        top_observed=False, automatic_nudge_eligible=False)})()
    pooled = worker.score(req)
    assert pooled['pin_top_px'] == [335., 312.]
    assert pooled['pin_px'] == [335., 292.]
    assert pooled['pin_accumulation']['sample_count'] == 2
    assert pooled['pin_bottom_inferred']
    assert not pooled['pin_endpoints']['automatic_nudge_eligible']
    worker.pin_predictor.predict = lambda rgb: dict(top_px=None, bottom_px=None)
    req = request()
    retained = worker.score(req)
    assert retained['pin_px'] == pooled['pin_px']
    assert retained['pin_accumulation']['retained']
    assert retained['pin_endpoints']['visible'] is False
    req = request()
    req['state'][0] = .02
    assert worker.score(req)['pin_px'] is None


def test_slow_search_blocks_stale_proposal_without_erasing_valid_memory(worker, monkeypatch):
    from app.stationary_face_accumulation import StationaryFaceAccumulator
    req = request()
    worker.score(req)
    worker.accumulator = StationaryFaceAccumulator()
    for offset in (600_000_000, 400_000_000, 200_000_000):
        worker.accumulator.update(worker.prediction, req['state'], req['timestamp_ns']-offset, req['config'])
    clock = [req['timestamp_ns']+1_000_000]
    monkeypatch.setattr('app.left_carrier_alignment.time.monotonic_ns', lambda: clock[0])
    missing = dict(worker.prediction, supported=False, holes_px=[None]*4, visible=[False]*4,
        homography=None)
    def slow_search(rgb):
        clock[0] += 1_500_000_000
        return missing
    worker.predictor.predict = slow_search
    late = worker.score(req)
    assert late['stale_result'] and not late['nudge_valid']
    assert late['accumulation']['stationary']
    assert late['accumulation']['point_count'] == 1
    # The next fresh result can reuse memory; a slow search did not reset it.
    worker.predictor.predict = lambda rgb: missing
    current = worker.score(request())
    assert not current['stale_result']
    assert current['accumulation']['stationary']
    assert current['hole_px'] is not None
    # Inputs already stale on receipt still clear history.
    req = request()
    req['timestamp_ns'] -= 2_000_000_000
    req['measured_timestamp_ns'] = req['timestamp_ns']
    rejected = worker.score(req)
    assert not rejected['accumulation']['stationary']
    assert rejected['accumulation']['point_count'] == 0
