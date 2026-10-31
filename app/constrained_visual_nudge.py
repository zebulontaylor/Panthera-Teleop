"""Image-error minimization with Cartesian and no-retreat constraints.

Pure proposal calculations; runtime supervision is in bounded_alignment_trial.
"""
import numpy as np
from scipy.optimize import minimize

from cartesian_nudge import rotation_error


def vector(value, length, reason='invalid_vector'):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(reason)
    return result


def forward_axis(settings):
    axis = vector(settings.get('forward_axis_base'), 3, 'forward_axis_required')
    if abs(np.linalg.norm(axis)-1) > 1e-6:
        raise ValueError('forward_axis_must_be_unit_vector')
    return axis


def translation_inverse(q, kinematics):
    j = np.asarray(kinematics.jacobian(q))
    if j.shape != (6, 6) or not np.isfinite(j).all():
        raise ValueError('invalid_robot_jacobian')
    singular = np.linalg.svd(j, compute_uv=False)
    if singular[-1] < 1e-4 or singular[0]/singular[-1] > 1000:
        raise ValueError('kinematic_singularity')
    return np.linalg.solve(j, np.r_[np.eye(3), np.zeros((3, 3))])


def checked_translation(q, translation_m, kinematics, limits, settings):
    """Check a translation-only joint path and its physical direction."""
    q = vector(q, 6)
    translation = vector(translation_m, 3)
    axis = forward_axis(settings)
    if np.linalg.norm(translation) > .000500001 or np.dot(translation, axis) < -1e-9:
        raise ValueError('cartesian_step_constraint')
    inverse = translation_inverse(q, kinematics)
    delta = inverse @ translation
    scale = min(1., .003/max(np.linalg.norm(delta), 1e-12))
    delta *= scale
    translation *= scale
    if np.linalg.norm(translation) < .00002:
        raise ValueError('step_too_small_or_infeasible')
    target = q+delta
    lower, upper = vector(limits['lower'], 6), vector(limits['upper'], 6)
    if np.any(target <= lower+.03) or np.any(target >= upper-.03):
        raise ValueError('joint_target_limit')
    xyz, rotation = kinematics.pose(q)
    for fraction in np.linspace(0, 1, 11):
        p, r = kinematics.pose(q+fraction*delta)
        displacement = p-xyz
        if (not np.isfinite(displacement).all()
                or np.linalg.norm(displacement) > .000501
                or np.linalg.norm(displacement-fraction*translation) > .00005
                or np.dot(displacement, axis) < -.000005
                or rotation_error(r, rotation) > .001):
            raise ValueError('predicted_cartesian_path_violation')
    return dict(target=target, delta_xyz_m=translation)


def constrained_visual_proposal(q, error_px, image_jacobian, kinematics, limits, settings):
    """Optimize in millimeters for numerical scaling; hold tool orientation."""
    q = vector(q, 6)
    error = vector(error_px, 4)[[1, 3]]
    image_jacobian = np.asarray(image_jacobian, dtype=float)
    if image_jacobian.shape != (4, 6) or not np.isfinite(image_jacobian).all():
        raise ValueError('invalid_image_jacobian')
    axis = forward_axis(settings)
    inverse = translation_inverse(q, kinematics)
    response = image_jacobian[[1, 3]] @ inverse*.001  # pixels/mm
    if np.linalg.matrix_rank(response, tol=.1) < 2:
        raise ValueError('insufficient_translation_response')
    constraints = [
        {'type': 'ineq', 'fun': lambda x: .5**2-np.dot(x, x), 'jac': lambda x: -2*x},
        {'type': 'ineq', 'fun': lambda x: np.dot(axis, x), 'jac': lambda x: axis},
        # Do not trade a large error reduction in one view for harm in the other.
        {'type': 'ineq', 'fun': lambda x: np.r_[
            np.abs(error)+.01-error-response @ x,
            np.abs(error)+.01+error+response @ x],
         'jac': lambda x: np.r_[-response, response]},
    ]
    normalization = max(1., np.dot(error, error))
    result = minimize(lambda x: (np.sum((error+response @ x)**2)+.01*np.dot(x, x))/normalization,
        np.zeros(3), method='SLSQP', bounds=[(-.5, .5)]*3,
        jac=lambda x: 2*(response.T @ (error+response @ x)+.01*x)/normalization,
        constraints=constraints, options={'maxiter': 150, 'ftol': 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('constrained_solver_failed')
    proposal = checked_translation(q, result.x*.001, kinematics, limits, settings)
    prediction = error+response @ (proposal['delta_xyz_m']*1000)
    if (np.any(np.abs(prediction) > np.abs(error)+.011)
            or np.linalg.norm(error)-np.linalg.norm(prediction) < .25):
        raise ValueError('no_supported_constrained_improvement')
    proposal['predicted_residual'] = prediction
    return proposal
