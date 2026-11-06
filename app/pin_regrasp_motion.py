"""Preflighted local regrasp trajectories. No camera or motor access."""
from dataclasses import dataclass
import numpy as np

TRAINING_PHASES = ('return', 'close')


def rotation_exp(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    x, y, z = v/a
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(a)*k + (1-np.cos(a))*(k@k)


def rotation_log(r):
    a = np.arccos(np.clip((np.trace(r)-1)/2, -1, 1))
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if np.pi-a < 1e-5:
        raise ValueError('Orientation discontinuity')
    return v/2 if a < 1e-7 else v*a/(2*np.sin(a))


@dataclass(frozen=True)
class RegraspConfig:
    task: str = 'regrasp'
    carrier_jitter_m: float = 0.
    hz: float = 30.
    retreat_m: float = .02
    cone_deg: float = 8.
    angle_jitter_deg: float = 3.
    tool_axis: str = 'toward_base'
    base_elevation_deg: float = 40.
    alignment_yaw_deg: float = 40.
    speed_m_s: float = .015
    angular_speed_deg_s: float = 10.
    joint_speed_rad_s: float = .15
    gripper_speed_rad_s: float = 2.0
    opening_speed_rad_s: float = 4.0
    closing_distance_m: float = .02
    open_gripper_rad: float = .5
    lift_m: float = .015
    placement_jitter_m: float = 0.
    release_s: float = .7  # Legacy metadata; release now takes only the bounded opening ramp.
    close_s: float = .7  # Legacy setting; close duration now follows constant finger speed.
    settle_s: float = .5

    def __post_init__(self):
        if self.task not in ('regrasp', 'alignment'):
            raise ValueError('Invalid collection task')
        positive = ('hz', 'retreat_m', 'speed_m_s', 'angular_speed_deg_s',
                    'joint_speed_rad_s', 'gripper_speed_rad_s', 'opening_speed_rad_s', 'lift_m',
                    'release_s', 'close_s', 'settle_s', 'closing_distance_m')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name, maximum in [('cone_deg', 45.), ('angle_jitter_deg', 6.),
                              ('placement_jitter_m', .10), ('carrier_jitter_m', .10), ('base_elevation_deg', 90.)]:
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(name + ' outside conservative bounds')
        if not np.isfinite(self.alignment_yaw_deg) or not -180. <= self.alignment_yaw_deg <= 180.:
            raise ValueError('alignment_yaw_deg outside conservative bounds')
        if self.retreat_m > .09 or self.lift_m > .04 or self.speed_m_s > .15:
            raise ValueError('Travel/speed exceeds local collection bounds')
        if self.joint_speed_rad_s > .6 or max(self.gripper_speed_rad_s, self.opening_speed_rad_s) > 5.:
            raise ValueError('Joint/gripper speed exceeds hardware collection bounds')
        if self.closing_distance_m > .02:
            raise ValueError('Closing must start within the final 20 mm')
        if self.tool_axis not in ('toward_base', '+x', '-x', '+y', '-y', '+z', '-z'):
            raise ValueError('Invalid tool_axis')
        if not np.isfinite(self.open_gripper_rad):
            raise ValueError('Invalid gripper target')


def retreat_axis(position, rotation, config):
    """Cone center in arm-base coordinates, fixed by the captured reference.

    Base mode points horizontally toward the base origin, then tilts upward
    from the base XY plane. Alignment rotates the horizontal bearing clockwise
    by alignment_yaw_deg, viewed from above +Z. Wrist orientation never changes
    this bearing. Explicit tool axes bypass this base-relative yaw.
    """
    if config.tool_axis != 'toward_base':
        return rotation[:, 'xyz'.index(config.tool_axis[1])] * (1 if config.tool_axis[0] == '+' else -1)
    toward = -np.asarray(position, dtype=float)[:2]
    distance = np.linalg.norm(toward)
    if distance < 1e-6:
        raise ValueError('Reference is directly above the base; capture a position away from the base axis')
    toward = toward / distance
    if config.task == 'alignment':
        yaw = -np.deg2rad(config.alignment_yaw_deg)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        toward = np.array([[cosine, -sine], [sine, cosine]]) @ toward
    elevation = np.deg2rad(config.base_elevation_deg)
    return np.r_[toward * np.cos(elevation), np.sin(elevation)]


def solve_pose(kinematics, seed, position, rotation, lower, upper):
    """Damped local IK in base coordinates, with bounded iterations and steps."""
    q = np.asarray(seed, dtype=float).copy()
    for _ in range(100):
        p, r = kinematics.pose(q)
        error = np.r_[position-p, rotation_log(rotation @ r.T)]
        if np.linalg.norm(error[:3]) < 2e-5 and np.linalg.norm(error[3:]) < 2e-4:
            return q
        j = np.asarray(kinematics.jacobian(q))
        if j.shape != (6, 6) or not np.isfinite(j).all():
            raise ValueError('Invalid kinematic Jacobian')
        step = j.T @ np.linalg.solve(j@j.T + np.eye(6)*1e-8, error)
        step *= min(1., .03/max(np.linalg.norm(step), 1e-12))
        q += step
        if not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
            raise ValueError('IK joint limit')
    raise ValueError('Cartesian target is unreachable')


def plan_regrasp(anchor14, kinematics, lower14, upper14, *, seed=0, config=None, current14=None):
    """Return (phase, target14) ticks; only return/close are training actions.

    Each cycle starts at current14, or the initial anchor when omitted,
    while placement variation stays centered on the initial anchor. The right
    arm and its gripper hold that fixed captured anchor. Gripper positions
    are encoder radians, unlike the simulator's per-finger metre convention.
    The entire trajectory is solved before any caller can execute it. Object
    support and grasp success require independent evidence from the caller.
    """
    c = config or RegraspConfig()
    if c.task != 'regrasp':
        raise ValueError('Regrasp planner requires regrasp task')
    anchor = np.asarray(anchor14, dtype=float)
    lower, upper = np.asarray(lower14), np.asarray(upper14)
    if any(v.shape != (14,) for v in (anchor, lower, upper)):
        raise ValueError('Expected 14 joint/gripper positions and limits')
    if not np.isfinite(np.r_[anchor, lower, upper]).all() or np.any(lower >= upper):
        raise ValueError('Invalid joint limits or anchor')
    if np.any(anchor < lower) or np.any(anchor > upper):
        raise ValueError('Captured anchor outside joint limits')
    if not lower[6] <= c.open_gripper_rad <= upper[6]:
        raise ValueError('Open gripper target outside limits')
    current = anchor.copy() if current14 is None else np.asarray(current14, dtype=float).copy()
    if (current.shape != (14,) or not np.isfinite(current).all()
            or np.any(current < lower) or np.any(current > upper)):
        raise ValueError('Current pose outside joint limits')
    if not np.array_equal(current[7:], anchor[7:]):
        raise ValueError('Carrier differs from captured anchor')
    p0, r0 = kinematics.pose(anchor[:6])
    p0, r0 = np.asarray(p0).copy(), np.asarray(r0).copy()
    if (p0.shape != (3,) or r0.shape != (3, 3) or not np.isfinite(np.r_[p0, r0.ravel()]).all()
            or not np.allclose(r0.T@r0, np.eye(3), atol=1e-5) or np.linalg.det(r0) < .999):
        raise ValueError('Invalid captured tool pose')
    grasp_p, grasp_r = kinematics.pose(current[:6])
    grasp_p, grasp_r = np.asarray(grasp_p).copy(), np.asarray(grasp_r).copy()
    rng = np.random.default_rng(seed)
    axis = retreat_axis(p0, r0, c)
    tangent = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]); tangent /= np.linalg.norm(tangent)
    angle = np.arccos(rng.uniform(np.cos(np.deg2rad(c.cone_deg)), 1.))
    azimuth = rng.uniform(-np.pi, np.pi)
    direction = np.cos(angle)*axis + np.sin(angle)*(np.cos(azimuth)*tangent + np.sin(azimuth)*np.cross(axis, tangent))
    retreat = grasp_p + direction*c.retreat_m*rng.uniform(.7, 1.)
    retreat_r = r0 @ rotation_exp(np.deg2rad(rng.uniform(-c.angle_jitter_deg, c.angle_jitter_deg, 3)))
    placed = p0 + np.r_[rng.uniform(-c.placement_jitter_m, c.placement_jitter_m, 2), 0.]
    plan = []
    p, r = grasp_p.copy(), grasp_r.copy()

    def move(phase, target_p, target_r, opening, duration=0., approach=False):
        nonlocal current, p, r
        start, start_p, start_r = current.copy(), p.copy(), r.copy()
        delta = rotation_log(target_r @ start_r.T)
        seconds = max(duration, 1.875*np.linalg.norm(target_p-start_p)/c.speed_m_s,
                      1.875*np.linalg.norm(delta)/np.deg2rad(c.angular_speed_deg_s),
                      (0. if approach else 1.875*abs(opening-start[6])/c.gripper_speed_rad_s), 1/c.hz)
        # Retry with longer duration when the joint-space solution needs it.
        for _ in range(8):
            n = max(1, int(np.ceil(seconds*c.hz)))
            segment, q = [], start[:6].copy()
            valid = True
            previous = start.copy()
            closing_step = None
            for tick_index, t in enumerate(np.linspace(0., 1., n+1)[1:]):
                u = t**3*(10+t*(-15+6*t))
                q = solve_pose(kinematics, q, start_p+u*(target_p-start_p),
                               rotation_exp(u*delta)@start_r, lower[:6], upper[:6])
                target = start.copy(); target[:6] = q
                if approach:
                    actual_p, _ = kinematics.pose(q)
                    may_close = np.linalg.norm(np.asarray(actual_p)-target_p) <= c.closing_distance_m
                    if may_close and closing_step is None:
                        closing_step = min(c.gripper_speed_rad_s/c.hz,
                                           (previous[6]-lower[6])/(n-tick_index))
                    target[6] = max(lower[6], previous[6]-closing_step) if closing_step is not None else previous[6]
                    if tick_index == n-1 and closing_step is not None and target[6]-lower[6] < 1e-12:
                        target[6] = lower[6]
                else:
                    target[6] = start[6]+u*(opening-start[6])
                if np.max(np.abs(target[:6]-previous[:6]))*c.hz > c.joint_speed_rad_s*1.00001:
                    valid = False
                    break
                # Check the midpoint of the actual joint interpolation too.
                mid_p, mid_r = kinematics.pose((previous[:6]+target[:6])/2)
                if not np.isfinite(np.r_[mid_p, np.asarray(mid_r).ravel()]).all():
                    raise ValueError('Invalid interpolated tool pose')
                segment.append((phase, target)); previous = target
            if valid:
                plan.extend(segment); current = segment[-1][1].copy()
                p, r = np.asarray(target_p).copy(), np.asarray(target_r).copy()
                return
            seconds *= 2
        raise ValueError('Joint speed exceeds planning bound')

    # Open independently of closing speed, with no slow easing tail or dwell.
    # Keep the arm still until the full-open target; retreat follows next tick.
    start_opening = current[6]
    count = max(1, int(np.ceil(abs(c.open_gripper_rad-start_opening)*c.hz/c.opening_speed_rad_s)))
    for tick in range(1, count+1):
        current = current.copy()
        current[6] = start_opening + (c.open_gripper_rad-start_opening)*tick/count
        if tick == count:
            current[6] = c.open_gripper_rad
        plan.append(('release', current.copy()))
    move('retreat', retreat, retreat_r, c.open_gripper_rad)
    move('return', grasp_p, r0, lower[6], approach=True)
    # Continue the same constant-rate finger command across the phase boundary.
    # Always emit a full-close tick, even when arrival already reached the stop.
    count = max(1, int(np.ceil((current[6]-lower[6])*c.hz/c.gripper_speed_rad_s)))
    for tick in range(count):
        current = current.copy()
        current[6] = lower[6] if tick == count-1 else max(lower[6], current[6]-c.gripper_speed_rad_s/c.hz)
        plan.append(('close', current.copy()))
    move('lift', grasp_p+[0, 0, c.lift_m], r0, lower[6])
    if np.linalg.norm(placed-grasp_p) > 1e-9:
        move('transfer', placed+[0, 0, c.lift_m], r0, lower[6])
    move('lower', placed, r0, lower[6])
    move('settle', placed, r0, lower[6], c.settle_s)
    return plan


def plan_alignment(anchor14, kinematics, lower14, upper14, *, right_kinematics=None,
                   T_left_base_right_base=None, seed=0, config=None, current14=None):
    """Preflight a held-pin retreat and return to a demonstrated hover anchor.

    Retreat and optional carrier translation run simultaneously on one easing
    clock, with the pin following carrier translation plus its retreat offset.
    Both grippers stay at their captured encoder positions. This is geometric
    planning, not object/clearance verification: callers must establish that the
    anchor is above the mouth and independently verify support and held objects.
    Optional carrier translation uses a supplied right-base to left-base
    displacement rotation. Matching parallel base axes use identity; the
    translation component is ignored because targets are reference-relative.
    Jitter is horizontal in the left base: callers must establish that its
    XY plane is parallel to the support table. The base transform alone does
    not establish table geometry, object contact, or insertion clearance.
    centered on the original anchor, so repeated cycles cannot random-walk.
    """
    c = config or RegraspConfig(task='alignment')
    if c.task != 'alignment':
        raise ValueError('Alignment planner requires alignment task')
    anchor, lower, upper = (np.asarray(v, dtype=float) for v in (anchor14, lower14, upper14))
    current = anchor.copy() if current14 is None else np.asarray(current14, dtype=float).copy()
    if (any(v.shape != (14,) for v in (anchor, lower, upper, current))
            or not np.isfinite(np.r_[anchor, lower, upper, current]).all()
            or np.any(lower >= upper) or np.any(anchor < lower) or np.any(anchor > upper)
            or np.any(current < lower) or np.any(current > upper)):
        raise ValueError('Invalid alignment positions or joint limits')
    if not np.array_equal(current[[6, 13]], anchor[[6, 13]]):
        raise ValueError('Held grippers differ from captured anchor')

    def pose(kin, q):
        p, r = (np.asarray(v, dtype=float) for v in kin.pose(q))
        if (p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(np.r_[p, r.ravel()]).all()
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1., atol=1e-5)):
            raise ValueError('Invalid alignment tool pose')
        return p.copy(), r.copy()

    rotation = None
    if c.carrier_jitter_m or right_kinematics is not None or T_left_base_right_base is not None:
        if right_kinematics is None or T_left_base_right_base is None:
            raise ValueError('Carrier drag requires right kinematics and calibrated T_left_base_right_base')
        transform = np.asarray(T_left_base_right_base, dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)):
            raise ValueError('Invalid calibrated base transform')
        rotation = transform[:3, :3]
    elif not np.array_equal(current[7:13], anchor[7:13]):
        raise ValueError('Carrier differs from captured anchor')

    reference_p, reference_r = pose(kinematics, anchor[:6])
    p, r = pose(kinematics, current[:6])
    rng = np.random.default_rng(seed)
    axis = retreat_axis(reference_p, reference_r, c)
    tangent = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]); tangent /= np.linalg.norm(tangent)
    angle = np.arccos(rng.uniform(np.cos(np.deg2rad(c.cone_deg)), 1.))
    azimuth = rng.uniform(-np.pi, np.pi)
    direction = np.cos(angle)*axis + np.sin(angle)*(np.cos(azimuth)*tangent + np.sin(azimuth)*np.cross(axis, tangent))
    retreat = p + direction*c.retreat_m*rng.uniform(.7, 1.)
    retreat_r = reference_r.copy()  # Preserve the demonstrated held-pin orientation.
    plan = []

    def move(phase, targets):
        """Solve all participating arms on one easing clock; retime together."""
        nonlocal current
        start = current.copy()
        requests = []
        seconds = 1/c.hz
        for kin, index, target_p, target_r in targets:
            start_p, start_r = pose(kin, start[index:index+6])
            delta = rotation_log(target_r @ start_r.T)
            seconds = max(seconds, 1.875*np.linalg.norm(target_p-start_p)/c.speed_m_s,
                          1.875*np.linalg.norm(delta)/np.deg2rad(c.angular_speed_deg_s))
            requests.append((kin, index, start_p, start_r, target_p, delta))
        for _ in range(8):
            count = max(1, int(np.ceil(seconds*c.hz)))
            previous = start.copy()
            previous_poses = [(item[2], item[3]) for item in requests]
            segment = []
            for t in np.linspace(0., 1., count+1)[1:]:
                u = t**3*(10+t*(-15+6*t))
                target = start.copy()
                actual_poses = []
                valid = True
                for request, (previous_p, previous_r) in zip(requests, previous_poses):
                    kin, index, start_p, start_r, target_p, delta = request
                    q = solve_pose(kin, previous[index:index+6], start_p+u*(target_p-start_p),
                                   rotation_exp(u*delta)@start_r,
                                   lower[index:index+6], upper[index:index+6])
                    actual_p, actual_r = pose(kin, q)
                    pose(kin, (previous[index:index+6]+q)/2)
                    if (np.max(np.abs(q-previous[index:index+6]))*c.hz > c.joint_speed_rad_s*1.00001
                            or np.linalg.norm(actual_p-previous_p)*c.hz > c.speed_m_s*1.00001
                            or np.linalg.norm(rotation_log(actual_r@previous_r.T))*c.hz > np.deg2rad(c.angular_speed_deg_s)*1.00001):
                        valid = False
                    target[index:index+6] = q
                    actual_poses.append((actual_p, actual_r))
                if not valid:
                    break
                segment.append((phase, target))
                previous, previous_poses = target, actual_poses
            else:
                plan.extend(segment); current = segment[-1][1].copy()
                return
            seconds *= 2
        raise ValueError('Alignment speed exceeds planning bound')

    goal = reference_p.copy()
    retreat_targets = [(kinematics, 0, retreat, retreat_r)]
    if rotation is not None:
        carrier_anchor_p, carrier_anchor_r = pose(right_kinematics, anchor[7:13])
        carrier_p, carrier_r = pose(right_kinematics, current[7:13])
        if np.linalg.norm(rotation_log(carrier_r @ carrier_anchor_r.T)) > 2e-4:
            raise ValueError('Carrier orientation differs from captured anchor')
        displacement = np.r_[rng.uniform(-c.carrier_jitter_m, c.carrier_jitter_m, 2), 0.]
        # Bound the displacement radius as well as its individual components.
        displacement *= min(1., c.carrier_jitter_m/max(np.linalg.norm(displacement), 1e-12))
        right_goal = carrier_anchor_p + rotation.T @ displacement
        carrier_translation = rotation @ (right_goal-carrier_p)
        # At every shared u the carrier translation cancels from relative pin
        # separation, leaving only the oblique retreat. This is not a collision
        # proof: object and whole-arm geometry are not available to this planner.
        retreat_targets = [(kinematics, 0, retreat+carrier_translation, retreat_r),
                           (right_kinematics, 7, right_goal, carrier_anchor_r)]
    move('retreat', retreat_targets)
    if rotation is not None:
        actual_right, _ = pose(right_kinematics, current[7:13])
        goal = reference_p + rotation @ (actual_right-carrier_anchor_p)
    move('return', [(kinematics, 0, goal, reference_r)])
    for _ in range(max(1, int(np.ceil(c.settle_s*c.hz)))):
        plan.append(('verify', current.copy()))
    return plan
