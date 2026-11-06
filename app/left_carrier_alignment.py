"""Fresh left-camera carrier localization adapter for the existing nudge interface.

No cached right-camera references or motor interfaces are used. Hidden bores are
supplied by the detector's support-checked face registration, never a hidden-label
coordinate prediction. Captures select face locations rather than frozen pixels.
"""
from __future__ import annotations
import base64
import time
from pathlib import Path
import numpy as np


class LeftCarrierAlignmentWorker:
    def __init__(self, checkpoint, device='cpu', *, backend='cnn', references=None):
        import cv2
        import torch
        cv2.setNumThreads(4)
        torch.set_num_threads(2)
        if backend == 'opencv':
            from app.left_carrier_opencv import LeftCarrierOpenCVPredictor
            self.predictor = LeftCarrierOpenCVPredictor(reference_bank=references)
        elif backend == 'cnn':
            from app.carrier_face_cnn import CarrierFacePredictor
            self.predictor = CarrierFacePredictor(checkpoint, device=device)
        else:
            raise ValueError('unknown left carrier backend')
        self.hole_backend = backend
        self.references = str(references) if references is not None else None
        self.accumulator = None
        if backend == 'opencv':
            from app.stationary_face_accumulation import StationaryFaceAccumulator
            self.accumulator = StationaryFaceAccumulator()
        pin_checkpoint = Path(__file__).resolve().parents[1]/'work/held-pin-cnn/best.pt'
        self.pin_predictor = None
        self.pin_checkpoint = None
        if pin_checkpoint.is_file():
            from app.held_pin_cnn import HeldPinPredictor
            self.pin_predictor = HeldPinPredictor(pin_checkpoint, device=device)
            self.pin_checkpoint = str(pin_checkpoint)
        self.checkpoint = str(checkpoint) if backend == 'cnn' else None
        self.context = None
        self.target_index = None

    @staticmethod
    def _point(value):
        if value is None:
            return None
        point = np.asarray(value, float)
        if (point.shape != (2,) or not np.isfinite(point).all()
                or np.any(point < 0) or np.any(point >= [640, 480])):
            return None
        return point

    @staticmethod
    def _same_pose(capture, state):
        source = np.asarray(capture.get('measured_state14', []), float)
        state = np.asarray(state, float)
        joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        return (source.shape == (14,) and state.shape == (14,)
                and np.isfinite(source).all() and np.isfinite(state).all()
                and np.max(np.abs(source[joints]-state[joints])) <= .01)

    def score(self, request):
        import cv2
        from app.pin_alignment_geometry import estimate_left_preview_nudge
        from app.pin_alignment_monitor import validated_config
        started = time.monotonic_ns()
        config = validated_config(request['config'])
        stamp = request['timestamp_ns']
        if type(stamp) is not int or not 0 < stamp <= started:
            raise ValueError('invalid/future monotonic image timestamp')
        if request['camera_id'] != 'left_wrist':
            raise ValueError('left carrier model requires logical left_wrist camera')
        state = np.asarray(request['state'], float)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('finite measured native state14 required')
        raw = base64.b64decode(request['image']['data'], validate=True)
        if request['image']['kind'] == 'rgb':
            if len(raw) != 640*480*3:
                raise ValueError('raw RGB must be 640x480x3')
            rgb = np.frombuffer(raw, np.uint8).reshape(480, 640, 3).copy()
        elif request['image']['kind'] == 'jpeg':
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('invalid JPEG')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('unknown image encoding')
        if rgb.shape != (480, 640, 3):
            raise ValueError('full uncropped 640x480 left-wrist image required')
        context = {k: request[k] for k in ('camera_id', 'grasp_id', 'context_id')}
        identity = (request['cache_epoch'], *context.values())
        if identity != self.context:
            self.context = identity
            self.target_index = None
            reset = getattr(self.predictor, 'reset', None)
            if callable(reset):
                reset()
            if getattr(self,'accumulator',None) is not None:
                self.accumulator.reset('camera, grasp or alignment context changed')
        prediction = self.predictor.predict(rgb)
        if getattr(self,'accumulator',None) is not None:
            measured_at = request.get('measured_timestamp_ns')
            # Admission concerns the captured input at worker receipt. A slow
            # unsuccessful search must not erase previously valid observations.
            # Completed-result age is checked separately before any proposal.
            eligible = (type(measured_at) is int and 0 < measured_at <= started
                        and abs(measured_at-stamp)/1e6 <= config['max_state_skew_ms']
                        and (started-stamp)/1e6 <= config['max_frame_age_ms'])
            prediction = self.accumulator.update(prediction,state,stamp,config,eligible=eligible)
        holes = [self._point(p) for p in prediction['holes_px']]
        visible = prediction['visible']
        retained = [bool(h.get('accumulated')) for h in prediction.get('holes',[{}]*4)]
        accumulation = prediction.get('accumulation',{})
        supported = bool(prediction['supported'])
        pin = self._point(prediction.get('pin_px'))
        pin_source = 'cnn' if pin is not None else 'unavailable'
        pin_endpoints = None
        pin_accumulation = {}
        if getattr(self, 'pin_predictor', None) is not None:
            pin_endpoints = self.pin_predictor.predict(rgb)
            accumulate_pin = getattr(getattr(self, 'accumulator', None), 'accumulate_pin', None)
            if callable(accumulate_pin) and request.get('captured_pin') is None:
                pin_endpoints = accumulate_pin(pin_endpoints, stamp, config)
                pin_accumulation = pin_endpoints.get('accumulation', {})
            pin = self._point(pin_endpoints.get('bottom_px'))
            pin_source = 'center_axis_inferred' if pin is not None else 'unavailable'
        captured_pin = request.get('captured_pin')
        if captured_pin is not None:
            if captured_pin.get('context') != context:
                raise ValueError('captured pin context mismatch')
            pin = self._point(captured_pin['pin_px']) if self._same_pose(captured_pin, state) else None
            pin_source = 'operator_selected' if pin is not None else 'operator_reference_expired'
        available = [i for i, p in enumerate(holes) if p is not None and (visible[i] or supported)]
        # Automatic selection follows the nearest bore in the current supported
        # map. Numeric slots can change as references and visibility change;
        # only an explicit operator selection should latch a particular slot.
        self.target_index = (min(available, key=lambda i: np.linalg.norm(holes[i]-pin))
                             if supported and pin is not None and available else None)
        target_index = self.target_index
        hole = holes[target_index] if target_index in available else None
        observed_source = 'left_carrier_opencv_observed' if getattr(self,'hole_backend','cnn')=='opencv' else 'left_carrier_cnn_observed'
        hole_source = (observed_source if visible[target_index] else
                       'left_carrier_stationary_accumulated' if retained[target_index] else
                       'left_carrier_layout_inferred') if hole is not None else 'unavailable'
        captured = request.get('captured_hole')
        manual_pixel_reference = False
        if captured is not None:
            if captured.get('context') != context:
                raise ValueError('captured hole context mismatch')
            index = captured.get('hole_index')
            if type(index) is int and 0 <= index < len(holes):
                target_index = index
                self.target_index = index
                hole = holes[index] if visible[index] or supported else None
            elif captured.get('face_point_m') is not None and supported:
                point = np.asarray([*captured['face_point_m'], 1.])
                pixel = np.asarray(prediction['homography']) @ point
                hole = self._point(pixel[:2]/pixel[2]) if abs(pixel[2]) > 1e-9 else None
            else:
                hole = self._point(captured['hole_px']) if self._same_pose(captured, state) else None
                manual_pixel_reference = hole is not None
            hole_source = captured.get('source', 'captured_operator_reference') if hole is not None else 'operator_reference_expired'
        reference_supported = supported or manual_pixel_reference
        if captured is None and not supported and getattr(self,'hole_backend','cnn')!='opencv':
            hole_source = 'cnn_unverified_fallback'
        geometry = dict(valid=False, reason='carrier face registration unavailable' if not reference_supported
                        else 'visible pin and supported target hole required')
        if pin is not None and hole is not None and reference_supported:
            geometry = estimate_left_preview_nudge(state, pin, hole, gain=config['gain'],
                max_step_m=config['max_step_mm']/1000, deadband_px=config['deadband_px'],
                frame_yaw_deg=config['frame_yaw_deg'])
        measured = request.get('measured_timestamp_ns')
        measured_valid = type(measured) is int and measured > 0
        skew = abs(measured-stamp)/1e6 if measured_valid else None
        now = time.monotonic_ns()
        age = (now-stamp)/1e6
        stale = age > config['max_frame_age_ms']
        if not measured_valid or skew > config['max_state_skew_ms']:
            geometry.update(valid=False, reason='measured-state timestamp unavailable or image/state skew exceeds limit')
        if measured_valid and measured > now:
            geometry.update(valid=False, reason='measured-state timestamp is in the future')
        if stale:
            geometry.update(valid=False, reason='analyzed frame is stale; no current proposal')
        valid = bool(geometry['valid'])
        delta = np.asarray(geometry.get('delta_left_m', [0., 0., 0.]))
        has_step = valid and np.linalg.norm(delta) > 1e-12
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        outline = np.asarray(prediction.get('face_outline_px', []), float)
        if outline.ndim == 2 and outline.shape[1:] == (2,) and len(outline)>=3 and np.isfinite(outline).all():
            cv2.polylines(overlay,[np.rint(outline).astype(np.int32)],True,(160,160,160),1,cv2.LINE_AA)
        layout_holes = prediction.get('layout_holes', [])
        # Keep approximate layout metadata available without drawing every
        # projected opening as if it were a detected rim.
        face_landmarks = prediction.get('face_landmarks', [])
        for landmark in face_landmarks:
            point = self._point(landmark.get('px'))
            if point is None or not landmark.get('metric_homography_eligible') or not (landmark.get('observed') or landmark.get('accumulated')):
                continue
            center = tuple(np.rint(point).astype(int))
            cv2.drawMarker(overlay, center, (180,220,60) if landmark.get('accumulated') else (255,160,40), cv2.MARKER_CROSS, 5, 1)
        carrier_holes = []
        for i, point in enumerate(holes):
            observed = bool(visible[i] and (supported or getattr(self,'hole_backend','cnn')=='opencv'))
            accumulated = bool(retained[i] and accumulation.get('stationary'))
            inferred = bool(point is not None and not observed and not accumulated and supported)
            carrier_holes.append(dict(index=i, hole_px=point.tolist() if point is not None else None,
                                      observed=observed, inferred=inferred, accumulated=accumulated,
                                      sample_count=prediction.get('holes',[{}]*4)[i].get('sample_count',1 if observed else 0),
                                      last_seen_age_s=prediction.get('holes',[{}]*4)[i].get('last_seen_age_s')))
            if point is None:
                continue
            color = ((100,230,100) if observed else (180,220,60) if accumulated else
                     (0,180,255) if inferred else (140,140,140))
            center = tuple(np.rint(point).astype(int))
            evidence = prediction.get('holes',[{}]*4)[i].get('rim_evidence') or {}
            axes = np.asarray(evidence.get('ellipse_axes_px', []), float)
            angle = evidence.get('ellipse_angle_deg')
            if observed and axes.shape == (2,) and np.isfinite(axes).all() and np.all(axes > 0) and angle is not None and np.isfinite(angle):
                cv2.ellipse(overlay, (tuple(map(float, point)), tuple(map(float, axes)), float(angle)), color, 1, cv2.LINE_AA)
            cv2.drawMarker(overlay, center, color,
                           cv2.MARKER_CROSS if observed else cv2.MARKER_DIAMOND if accumulated else cv2.MARKER_TILTED_CROSS, 5, 1)
            cv2.putText(overlay, str(i+1)+(' retained' if accumulated else ' inferred' if inferred else ''),
                        (center[0]+9, center[1]-9), cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
        if hole is not None:
            cv2.drawMarker(overlay, tuple(np.rint(hole).astype(int)), (255, 0, 200), cv2.MARKER_SQUARE, 21, 1)
        if pin is not None:
            cv2.circle(overlay, tuple(np.rint(pin).astype(int)), 7, (255, 255, 0), 2)
        top = self._point(pin_endpoints.get('top_px')) if pin_endpoints else None
        if top is not None:
            cv2.circle(overlay, tuple(np.rint(top).astype(int)), 5, (255, 180, 80), 2)
            cv2.putText(overlay, 'pin top', (int(top[0])+9,int(top[1])-9), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,180,80), 1)
            if pin is not None:
                cv2.line(overlay,tuple(np.rint(top).astype(int)),tuple(np.rint(pin).astype(int)),(255,255,0),1)
                cv2.putText(overlay, 'bottom inferred', (int(pin[0])+9,int(pin[1])+16), cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,0),1)
        if has_step:
            cv2.arrowedLine(overlay, tuple(np.rint(pin).astype(int)), tuple(np.rint(hole).astype(int)), (0, 220, 255), 2, tipLength=.2)
        backend_label = 'OPENCV' if getattr(self,'hole_backend','cnn')=='opencv' else 'CARRIER CNN'
        pin_label = {'center_axis_inferred':'bottom inferred on centerline',
                     'operator_selected':'manual point', 'operator_reference_expired':'manual point expired',
                     'cnn':'CNN estimate', 'unavailable':'unavailable'}.get(pin_source,pin_source)
        if pin_accumulation.get('sample_count'):
            pin_label += f" / {'retained' if pin_accumulation.get('retained') else 'pooled'} {pin_accumulation['sample_count']} estimates"
        for line, text in enumerate((f'LEFT WRIST / {backend_label} / PREVIEW ONLY',
                                     'Green: fitted rim / teal diamond: retained / orange x: inferred',
                                     'Blue cross: supporting landmark / purple square: target',
                                     (f"Stationary memory: {accumulation.get('frames',0)} frames / {accumulation.get('point_count',0)} points"
                                      if accumulation.get('stationary') else 'Current frame / purple: selected target'),
                                     f'Pin: {pin_label} / '+('nudge ready' if valid else geometry['reason']))):
            cv2.putText(overlay, text, (8, 22+22*line), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
        okay, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not okay:
            raise RuntimeError('overlay JPEG encoding failed')
        localized = pin is not None and hole is not None and reference_supported
        status = ('stale' if stale else 'ready' if localized else
                  'waiting_for_pin' if reference_supported and (hole is not None or available) and pin is None else 'waiting_for_hole')
        retained_target = hole_source=='left_carrier_stationary_accumulated'
        target_age = prediction.get('holes',[{}]*4)[target_index].get('last_seen_age_s',0) if retained_target else 0
        cache = dict(mode='held' if retained_target else 'tracked' if hole is not None and reference_supported else 'lost',
                     source_ns=stamp-int((target_age or 0)*1e9),
                     hole_px=hole.tolist() if hole is not None else None,
                     assumption='verified observations pooled only at stationary both-arm pose' if accumulation.get('stationary') else 'fresh left-camera observation/layout')
        error = hole-pin if localized else None
        return dict(status=status, ready=localized and not stale, localization_ready=localized and not stale,
            reason=geometry['reason'], nudge_blocked_reason=None if valid else geometry['reason'],
            stale_result=stale, analyzed_timestamp_ns=stamp, measured_timestamp_ns=measured,
            measured_state14=state.tolist(), captured_hole=captured, captured_pin=captured_pin,
            completed_timestamp_ns=now, frame_age_ms=age, analyzed_age_s=age/1000,
            state_image_skew_ms=skew, queue_age_ms=(started-stamp)/1e6,
            image_size_wh=[640, 480], camera_role='left_wrist', pin_source=pin_source,
            hole_backend=getattr(self,'hole_backend','cnn'),
            accumulation=accumulation,
            pin_accumulation=pin_accumulation,
            pin_top_px=top.tolist() if top is not None else None,
            pin_bottom_inferred=pin_source=='center_axis_inferred', pin_endpoints=pin_endpoints,
            pin_checkpoint=getattr(self,'pin_checkpoint',None),
            pin_px=pin.tolist() if pin is not None else None, hole_px=hole.tolist() if hole is not None else None,
            pin_uv=(pin/[640,480]).tolist() if pin is not None else None,
            hole_uv=(hole/[640,480]).tolist() if hole is not None else None,
            error_px=error.tolist() if error is not None else None,
            pixel_error_px=float(np.linalg.norm(error)) if error is not None else None,
            hole_source=hole_source, target_hole_index=target_index, carrier_holes=carrier_holes,
            target_selection=dict(mode='operator' if captured is not None else 'automatic',
                                  index=target_index, policy='nearest pin in current supported face map'),
            face_landmarks=face_landmarks,
            registration_landmark_count=prediction.get('registration_landmark_count',0),
            face_outline_px=prediction.get('face_outline_px'), layout_holes=layout_holes,
            registration_kind=prediction.get('registration_kind'),
            layout_orientation_verified=prediction.get('layout_orientation_verified',False),
            carrier_homography=prediction.get('homography'), carrier_geometry_supported=supported,
            cache=cache, visual_cache=cache, nudge_valid=valid,
            nudge_left_mm=(delta*1000).tolist() if valid else [0.,0.,0.],
            nudge_right_mm=(np.asarray(geometry.get('delta_right_m',[0.,0.,0.]))*1000).tolist() if valid else [0.,0.,0.],
            geometry=geometry, arrow_kind='alignment_direction' if has_step else 'none',
            preview_only=True, motor_permission=False, cnn_gate_permission=False,
            cnn_xyz_head_used=False, checkpoint=self.checkpoint, reference_bank=getattr(self,'references',None),
            analyzed_jpeg_b64=base64.b64encode(jpeg.tobytes()).decode('ascii'))
