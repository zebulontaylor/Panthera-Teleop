"""Current-pose XY search and measured loaded-height termination.

Pure calculations: no robot, force-control, camera, or network interfaces.
A downward movement is an insertion *candidate*, not proof of insertion.
The external controller must supply genuine measured pose/contact information.
"""
from dataclasses import dataclass
import numpy as np


def _xyz(value):
    point = np.asarray(value, float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('invalid_measured_xyz')
    return point.copy()


@dataclass(frozen=True)
class SearchConfig:
    radius_m: float = .001
    turns: float = 2.
    speed_m_s: float = .001
    max_duration_s: float = 120.
    oscillation_amplitude_m: float = .0002
    oscillation_frequency_hz: float = 15.
    insertion_drop_m: float = .003
    insertion_debounce_s: float = .2
    baseline_dwell_s: float = .5
    baseline_stability_m: float = .0002
    max_down_travel_m: float = .004
    max_sample_gap_s: float = .25

    def __post_init__(self):
        bounds = dict(radius_m=(0,.020), turns=(0,10), speed_m_s=(0,.010),
            max_duration_s=(0,300),
            insertion_drop_m=(.0003,.003), insertion_debounce_s=(.1,2),
            baseline_dwell_s=(.2,3), baseline_stability_m=(.00001,.0002),
            max_down_travel_m=(.0005,.005), max_sample_gap_s=(.05,.5))
        for key,(minimum,maximum) in bounds.items():
            value = getattr(self,key)
            if not np.isfinite(value) or not minimum <= value <= maximum or value <= 0:
                raise ValueError('invalid_search_' + key)
        for key, maximum in [('oscillation_amplitude_m', .002),
                             ('oscillation_frequency_hz', 40.)]:
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError('invalid_search_' + key)
        if self.insertion_drop_m <= 2*self.baseline_stability_m:
            raise ValueError('drop_threshold_must_exceed_baseline_noise')
        if self.insertion_drop_m >= self.max_down_travel_m:
            raise ValueError('drop_threshold_must_precede_hard_travel_stop')


class SpiralSearch:
    """Outward Archimedean spiral centered on the measured starting position.

    Speed describes the nominal spiral. A zero-mean sinusoid along the XY
    diagonal is added separately, with the configured peak radial amplitude.
    All targets hold start Z; base -Z load and orientation hold are external.
    """
    def __init__(self, start_xyz, config=None):
        self.config = config or SearchConfig()
        self.start = _xyz(start_xyz)
        self.center = self.start.copy()
        theta = np.linspace(0,2*np.pi*self.config.turns,4097)
        radii = self.config.radius_m * theta/theta[-1]
        spiral = self.center + np.column_stack((radii*np.cos(theta),
                            radii*np.sin(theta), np.zeros_like(theta)))
        self._points = spiral
        lengths = np.linalg.norm(np.diff(self._points,axis=0),axis=1)
        distances = np.r_[0.,np.cumsum(lengths)]
        keep = np.r_[True,np.diff(distances)>1e-14]
        self._points = self._points[keep]
        self._distances = distances[keep]
        self.path_length_m = float(self._distances[-1])
        self.completion_s = self.path_length_m/self.config.speed_m_s
        self.duration_s = min(self.completion_s,self.config.max_duration_s)

    def sample(self, elapsed_s):
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('invalid_search_elapsed_time')
        sample_time = min(elapsed_s,self.duration_s)
        distance = sample_time*self.config.speed_m_s
        distance = min(distance,self.path_length_m)
        point = np.array([np.interp(distance,self._distances,self._points[:,i])
                          for i in range(3)])
        finished = elapsed_s >= self.duration_s
        reason = ('pattern_complete' if self.completion_s <= self.config.max_duration_s
                  else 'time_limit') if finished else None
        offset = np.zeros(3)
        offset[:2] = (self.config.oscillation_amplitude_m / np.sqrt(2)
                      * np.sin(2*np.pi*self.config.oscillation_frequency_hz*sample_time))
        return dict(target_xyz=(point+offset).tolist(), nominal_target_xyz=point.tolist(),
                    oscillation_offset_xyz=offset.tolist(), phase='spiral', done=finished,
                    stop_reason=reason, center_xyz=self.center.tolist(), center_source='current_pose',
                    elapsed_s=float(elapsed_s), reference_frame='left_base',
                    force_direction_left_base=[0.,0.,-1.])


class LoadedDropMonitor:
    """Stable loaded baseline plus debounced measured downward movement.

    Call at each fresh measurement. ``contact_confirmed`` means the operator
    explicitly confirmed that the pin rests on the carrier; it is not a
    measurement of force/contact. An armed automatic controller may instead
    supply ``contact_candidate=True``; its baseline remains explicitly inferred.
    This class cannot infer Newtons or contact.
    Initial settling may move downward before a stable loaded baseline exists.
    Insertion candidates require a later drop relative to that loaded baseline.
    Pre-arm travel still counts toward the independent hard descent stop.
    """
    def __init__(self, start_time_s, start_xyz, config=None):
        if not np.isfinite(start_time_s):
            raise ValueError('invalid_start_time')
        self.config = config or SearchConfig()
        self.start_time = float(start_time_s)
        self.last_time = self.start_time
        self.start = _xyz(start_xyz)
        self.baseline_z = None
        self._stable = []
        self._drop_since = None
        self._terminal = None

    def update(self, now_s, measured_xyz, contact_confirmed, *, contact_candidate=False):
        point = _xyz(measured_xyz)
        if not np.isfinite(now_s) or now_s < self.last_time:
            raise ValueError('nonmonotonic_measurement_time')
        sample_gap = float(now_s-self.last_time)
        self.last_time = float(now_s)
        if not isinstance(contact_confirmed,(bool,np.bool_)):
            raise ValueError('contact_confirmation_must_be_boolean')
        if not isinstance(contact_candidate,(bool,np.bool_)):
            raise ValueError('contact_candidate_must_be_boolean')
        drop_from_start = float(self.start[2]-point[2])
        if self._terminal is None:
            if drop_from_start >= self.config.max_down_travel_m:
                self._terminal = 'hard_down_travel_stop'
            elif now_s-self.start_time >= self.config.max_duration_s:
                self._terminal = 'time_limit'
            elif sample_gap > self.config.max_sample_gap_s:
                self._terminal = 'measurement_gap_stop'
            elif not (contact_confirmed or contact_candidate):
                self._terminal = 'operator_contact_confirmation_missing'
            elif self.baseline_z is None:
                self._stable.append((float(now_s),float(point[2])))
                # Keep one sample before the window boundary so the full
                # stable dwell is covered, even with irregular timestamps.
                cutoff = now_s-self.config.baseline_dwell_s
                while len(self._stable)>1 and self._stable[1][0] <= cutoff:
                    self._stable.pop(0)
                heights = [item[1] for item in self._stable]
                if max(heights)-min(heights) > self.config.baseline_stability_m:
                    self._stable = [(float(now_s),float(point[2]))]
                elif (len(self._stable) >= 3
                      and now_s-self._stable[0][0] >= self.config.baseline_dwell_s):
                    self.baseline_z = float(np.median(heights))
            else:
                drop = self.baseline_z-point[2]
                if drop >= self.config.insertion_drop_m:
                    if self._drop_since is None:
                        self._drop_since = float(now_s)
                    elif now_s-self._drop_since >= self.config.insertion_debounce_s:
                        self._terminal = 'insertion_candidate'
                else:
                    self._drop_since = None
        return dict(done=self._terminal is not None, stop_reason=self._terminal,
                    phase='armed' if self.baseline_z is not None else 'acquiring_contact_baseline',
                    baseline_z_m=self.baseline_z,
                    baseline_authority='operator_confirmation' if contact_confirmed else
                        'inferred_contact_candidate' if contact_candidate else 'none',
                    downward_from_start_m=drop_from_start,
                    drop_from_loaded_baseline_m=(None if self.baseline_z is None
                                                else float(self.baseline_z-point[2])),
                    insertion_confirmed=False)
