"""Offline local visual-Jacobian estimation; this module never controls hardware.

Input coordinates must use the same, physically identified target in each view
throughout each episode. Camera order is left_wrist x/y, right_wrist x/y.
Error is moving landmark minus target landmark in original image pixels.
Offline predictive validation does not verify grasp, geometry, or safe contact.
"""
from collections import defaultdict

import numpy as np


def _pairs(samples, lag_seconds, max_pair_seconds):
    episodes = defaultdict(list)
    rejected = 0
    for sample in samples:
        try:
            t = float(sample['time_s'])
            q = np.r_[sample['q_left'], sample['q_right']].astype(float)
            e = np.asarray(sample['error_px'], dtype=float).reshape(-1)
            if q.shape != (12,) or e.shape != (4,) or not np.isfinite(np.r_[t, q, e]).all():
                raise ValueError('invalid shape or nonfinite')
            episodes[str(sample['episode_id'])].append((t, q, e))
        except (KeyError, ValueError, TypeError):
            rejected += 1
    pairs = []
    # Non-overlapping pairs limit duplicated evidence and never cross episodes.
    for episode, rows in sorted(episodes.items()):
        rows.sort(key=lambda row: row[0])
        i = 0
        while i < len(rows)-1:
            j = i+1
            while j < len(rows) and rows[j][0]-rows[i][0] < lag_seconds:
                j += 1
            if j >= len(rows):
                break
            dt = rows[j][0]-rows[i][0]
            if dt <= max_pair_seconds:
                pairs.append((episode, rows[i][1], rows[j][1], rows[j][2]-rows[i][2]))
                i = j+1
            else:
                i += 1
    return pairs, rejected


def _regress(x, z, y, *, min_excitation_rad=.002, max_condition=50.):
    """Weighted partial regression removes both-arm co-motion confounding.

    Only controlled-arm directions independently excited after projecting out
    the other arm are retained. Zero coefficients in the null space mean
    unknown response, not measured zero physical sensitivity.
    """
    weights = np.ones(len(x))
    nuisance = np.column_stack((z, np.ones(len(z))))
    for _ in range(8):
        root = np.sqrt(weights)[:, None]
        zw, xw, yw = nuisance*root, x*root, y*root
        xres = xw-zw @ np.linalg.lstsq(zw, xw, rcond=1e-6)[0]
        yres = yw-zw @ np.linalg.lstsq(zw, yw, rcond=1e-6)[0]
        u, singular, vt = np.linalg.svd(xres, full_matrices=False)
        threshold = max(min_excitation_rad*np.sqrt(weights.sum()),
                        singular[0]/max_condition if len(singular) else 0.)
        keep = singular > threshold
        basis = vt[keep].T
        coefficient = (vt[keep].T/singular[keep]) @ (u[:, keep].T @ yres) if keep.any() else np.zeros((6, 4))
        nuisance_coefficient = np.linalg.lstsq(zw, (y-x @ coefficient)*root, rcond=1e-6)[0]
        residual = y-x @ coefficient-nuisance @ nuisance_coefficient
        norms = np.linalg.norm(residual, axis=1)
        scale = max(.25, float(np.median(norms))/1.54)
        weights = np.minimum(1., 2.5*scale/np.maximum(norms, 1e-12))
    retained = singular[keep]
    return dict(coefficient=coefficient, nuisance=nuisance_coefficient,
                basis=basis, rank=int(keep.sum()), singular=singular,
                excitation=singular/np.sqrt(weights.sum()),
                condition=float(retained[0]/retained[-1]) if len(retained) else None,
                weights=weights)


def fit_local(samples, side, task, *, lag_seconds=.2, max_pair_seconds=.5,
              local_radius_rad=.25, min_excitation_rad=.002, bootstrap_replicates=64, reference_joint_positions=None):
    """Fit a local candidate and leave-one-trial-out predictive diagnostics.

    ``samples`` have episode_id, time_s, q_left[6], q_right[6], error_px[4].
    Caller must filter for visible, confident, stable-grasp, fixed-target frames.
    Optional moving_px/target_px and gripper measurements are provenance only.
    This function neither infers a safe grasp nor changes verified to true.
    """
    if side not in ('left', 'right') or task not in ('gear', 'pin'):
        raise ValueError('side must be left/right and task gear/pin')
    if not 0 < lag_seconds <= max_pair_seconds or not 0 < local_radius_rad <= .3:
        raise ValueError('invalid temporal or local envelope')
    axes = [1, 3] if task == 'gear' else [0, 1, 2, 3]
    pairs, rejected = _pairs(samples, lag_seconds, max_pair_seconds)
    result = dict(verified=False, candidate_usable=False, side=side, task=task,
                  residual_axes=axes, camera_names=['left_wrist', 'right_wrist'],
                  max_joint_distance_rad=local_radius_rad,
                  validation_scope='offline image-motion prediction only',
                  reasons=[], rejected_samples=rejected, temporal_pair_count=len(pairs))
    if not pairs:
        result['reasons'] = ['no_valid_temporal_pairs']
        return result
    reference = (np.median(np.asarray([p[1] for p in pairs]), axis=0) if reference_joint_positions is None
                 else np.asarray(reference_joint_positions, dtype=float))
    if reference.shape != (12,) or not np.isfinite(reference).all():
        raise ValueError('reference_joint_positions must be 12 finite radians')
    pairs = [p for p in pairs if max(np.max(np.abs(p[1]-reference)),
                                    np.max(np.abs(p[2]-reference))) <= local_radius_rad]
    result['local_pair_count'] = len(pairs)
    result['local_episode_count'] = len({p[0] for p in pairs})
    controlled = slice(0, 6) if side == 'left' else slice(6, 12)
    other = slice(6, 12) if side == 'left' else slice(0, 6)
    result['reference_joint_position_rad'] = reference[controlled].tolist()
    result['other_arm_reference_joint_position_rad'] = reference[other].tolist()
    if len(pairs) < 15:
        result['reasons'] = ['insufficient_local_pairs']
        return result
    delta = np.asarray([p[2]-p[1] for p in pairs])
    x, z = delta[:, controlled], delta[:, other]
    y = np.asarray([p[3] for p in pairs])
    nuisance_design = np.column_stack((z, np.ones(len(z))))
    independently_excited = x-nuisance_design @ np.linalg.lstsq(nuisance_design, x, rcond=1e-6)[0]
    result['controlled_arm_motion_pair_count'] = int(np.sum(np.linalg.norm(x, axis=1) >= min_excitation_rad))
    result['other_arm_motion_pair_count'] = int(np.sum(np.linalg.norm(z, axis=1) >= min_excitation_rad))
    result['independent_controlled_arm_motion_pair_count'] = int(np.sum(np.linalg.norm(independently_excited, axis=1) >= min_excitation_rad))
    result['independent_motion_count_threshold_rad'] = min_excitation_rad
    episodes = np.asarray([p[0] for p in pairs])
    fitted = _regress(x, z, y, min_excitation_rad=min_excitation_rad)
    result.update(image_jacobian_px_per_rad=fitted['coefficient'].T.tolist(),
                  other_arm_image_jacobian_px_per_rad=fitted['nuisance'][:6].T.tolist(),
                  drift_px_per_pair=fitted['nuisance'][6].tolist(),
                  identifiable_joint_basis=fitted['basis'].tolist(),
                  identifiable_rank=fitted['rank'],
                  conditional_excitation_rms_rad=fitted['excitation'].tolist(),
                  condition_number=fitted['condition'],
                  downweighted_pair_count=int(np.sum(fitted['weights'] < .9)),
                  episode_count=len(set(episodes)))
    # Resample whole episodes: adjacent observations are not independent trials.
    # Coefficient intervals describe this offline dataset, not hardware safety.
    episode_ids = sorted(set(episodes))
    boot = []
    ranks = []
    rng = np.random.default_rng(1927)
    for _ in range(max(0, int(bootstrap_replicates))):
        draws = rng.choice(episode_ids, len(episode_ids), replace=True)
        indices = np.concatenate([np.flatnonzero(episodes == ep) for ep in draws])
        model = _regress(x[indices], z[indices], y[indices], min_excitation_rad=min_excitation_rad)
        boot.append(model['coefficient'].T)
        ranks.append(model['rank'])
    if boot:
        boot = np.asarray(boot)
        low, high = np.quantile(boot, [.025, .975], axis=0)
        response = fitted['coefficient'].T[axes] @ fitted['basis']
        boot_response = boot[:, axes, :] @ fitted['basis']
        signs = np.sum(boot_response*response[None, :, :], axis=1) > 0
        consistency = float(np.min(np.mean(signs, axis=0))) if signs.shape[1] else 0.
        result['trial_bootstrap'] = dict(replicates=len(boot),
            coefficient_lower_95_px_per_rad=low.tolist(),
            coefficient_upper_95_px_per_rad=high.tolist(),
            minimum_identifiable_rank=min(ranks),
            minimum_direction_consistency=consistency)
    task_singular = np.linalg.svd(fitted['coefficient'].T[axes], compute_uv=False)
    task_rank = int(np.sum(task_singular > max(1e-8, task_singular[0]/50.)))
    result['task_image_rank'] = task_rank
    result['task_image_singular_values_px_per_rad'] = task_singular.tolist()
    folds, actual, predicted, nuisance_only, independent_actual, independent_predicted = [], [], [], [], [], []
    for episode in sorted(set(episodes)):
        test = episodes == episode
        train = ~test
        if train.sum() < 15:
            continue
        model = _regress(x[train], z[train], y[train], min_excitation_rad=min_excitation_rad)
        prediction = x[test] @ model['coefficient'] + np.column_stack((z[test], np.ones(test.sum()))) @ model['nuisance']
        # Train a separate nuisance-only baseline; improvement must come from the
        # controlled arm, not merely from following the camera on the other arm.
        baseline_coef = np.linalg.lstsq(np.column_stack((z[train], np.ones(train.sum()))), y[train], rcond=1e-6)[0]
        baseline = np.column_stack((z[test], np.ones(test.sum()))) @ baseline_coef
        target = y[test][:, axes]
        pred = prediction[:, axes]
        base = baseline[:, axes]
        other_response = (np.column_stack((z[test], np.ones(test.sum()))) @ model['nuisance'])[:, axes]
        actual.extend(target)
        predicted.extend(pred)
        nuisance_only.extend(base)
        independent_actual.extend(target-other_response)
        independent_predicted.extend((x[test] @ model['coefficient'])[:, axes])
        folds.append(dict(episode_id=episode, pairs=int(test.sum()), rank=model['rank'],
                          rmse_px=float(np.sqrt(np.mean((target-pred)**2))),
                          zero_motion_rmse_px=float(np.sqrt(np.mean(target**2))),
                          nuisance_only_rmse_px=float(np.sqrt(np.mean((target-base)**2)))))
    result['heldout_trials'] = folds
    if folds:
        actual, predicted, nuisance_only = map(np.asarray, (actual, predicted, nuisance_only))
        ia, ip = np.asarray(independent_actual), np.asarray(independent_predicted)
        rmse = float(np.sqrt(np.mean((actual-predicted)**2)))
        zero = float(np.sqrt(np.mean(actual**2)))
        nuisance_rmse = float(np.sqrt(np.mean((actual-nuisance_only)**2)))
        eligible = (np.linalg.norm(ia, axis=1) >= 2.) & (np.linalg.norm(ip, axis=1) >= 1.)
        directional = float(np.mean(np.sum(ia[eligible]*ip[eligible], axis=1) > 0)) if eligible.any() else None
        result['validation'] = dict(rmse_px=rmse, zero_motion_rmse_px=zero,
                                    nuisance_only_rmse_px=nuisance_rmse,
                                    improvement_over_zero=1-rmse/max(zero, 1e-12),
                                    improvement_over_nuisance=1-rmse/max(nuisance_rmse, 1e-12),
                                    conditional_directional_accuracy=directional,
                                    directional_pair_count=int(eligible.sum()),
                                    heldout_pair_count=len(actual))
    reasons = result['reasons']
    if len(pairs) < 40 or len(set(episodes)) < 4:
        reasons.append('insufficient_independent_trial_support')
    if result['task_image_rank'] < 2:
        reasons.append('insufficient_task_image_response_rank')
    if fitted['rank'] < 2:
        reasons.append('insufficient_independent_controlled_arm_excitation')
    if len(folds) < 4 or any(f['rank'] < 2 for f in folds):
        reasons.append('heldout_excitation_not_stable')
    val = result.get('validation', {})
    if val.get('improvement_over_zero', -1) < .1 or val.get('improvement_over_nuisance', -1) < .1:
        reasons.append('heldout_prediction_does_not_beat_baselines')
    if val.get('directional_pair_count', 0) < 20 or (val.get('conditional_directional_accuracy') or 0.) < .65:
        reasons.append('heldout_direction_not_reliable')
    bootstrap = result.get('trial_bootstrap', {})
    if (bootstrap.get('replicates', 0) < 32 or bootstrap.get('minimum_identifiable_rank', 0) < 2
            or bootstrap.get('minimum_direction_consistency', 0) < .9):
        reasons.append('trial_bootstrap_response_not_stable')
    result['candidate_usable'] = not reasons
    return result
