#!/usr/bin/env python3
"""Historical legacy-scene prototype, retained only for experiment reproduction.

Measured arm seeds are replayed exactly. Object poses are synthetic hypotheses,
not reconstructed real contacts. Gate labels describe geometry/visibility only.
New data must use generate_pin_insertion_photo_data.py and the existing photo
environment. This prototype bypassed its layout, appearance and camera builder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from real2sim_bridge import DEFAULT_SCENE, Real2SimBridge
from real2sim_data import load_episode
from real2sim_object_pose import CARRIER_BORE_HEIGHT_M, carrier_bore_points


def project(points, camera_position, camera_rotation, fovy, width, height):
    """MuJoCo camera (-Z forward,+Y up) to normalized top-left image pixels."""
    p = (np.asarray(points)-camera_position) @ camera_rotation
    depth = -p[..., 2]
    focal = .5*height / np.tan(np.deg2rad(fovy)*.5)
    uv = np.stack([.5*width+focal*p[..., 0]/depth,
                   .5*height-focal*p[..., 1]/depth], axis=-1)
    return uv/np.array([width, height]), depth


def correction(pin_tip, hole, right_base_rotation, controlled_object='pin', max_step_m=.004):
    """Bounded translation of the controlled object, expressed in RIGHT base."""
    if controlled_object not in ('pin', 'carrier'):
        raise ValueError('controlled_object must be pin or carrier')
    error = np.asarray(pin_tip)-np.asarray(hole)
    if controlled_object == 'pin':
        error = -error
    local = np.asarray(right_base_rotation).T @ error
    return local * min(1., max_step_m / max(np.linalg.norm(local), 1e-12))


def depth_visibility(landmark_depth_m, rendered_depth_m, in_frame=True, tolerance_m=.001, same_object=False):
    """Reject external occlusion; optionally support an amodal object landmark.

    same_object is renderer segmentation evidence, never a distance heuristic.
    """
    return (np.asarray(in_frame,dtype=bool) &
            ((np.asarray(rendered_depth_m) >= np.asarray(landmark_depth_m)-tolerance_m)
             | np.asarray(same_object,dtype=bool)))


def hole_gate_support(hole_visible, pin_at_hole_pixel, carrier_visible):
    """Expected pin-over-bore occlusion can support gating, not visible-keypoint loss."""
    return bool(hole_visible or (pin_at_hole_pixel and carrier_visible))


def alignment_gate(pin_tip, hole, plane_normal, visible, *, height_tolerance_m=.001, min_nudge_m=.0002):
    """Geometric edge eligibility; proximity alone never activates hovering pins.

    This tests a synthetic support-plane hypothesis, not observed physical contact.
    """
    normal = np.asarray(plane_normal,dtype=float)
    normal = normal/np.linalg.norm(normal)
    error = np.asarray(pin_tip)-np.asarray(hole)
    height = float(error @ normal)
    lateral = float(np.linalg.norm(error-height*normal))
    active = bool(np.asarray(visible).all() and abs(height) <= height_tolerance_m
                  and min_nudge_m <= lateral <= .012)
    return active, height, lateral


def _place(bridge, name, position, rotation):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation).flatten())
    body = bridge.model.body(name).id
    if bridge.model.body_jntnum[body] == 0:
        bridge.model.body_pos[body] = position
        bridge.model.body_quat[body] = quat
    else:
        adr = int(bridge.model.jnt_qposadr[bridge.model.body_jntadr[body]])
        bridge.data.qpos[adr:adr+7] = np.r_[position, quat]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(episodes, output, *, scene=DEFAULT_SCENE, samples_per_episode=256,
             image_size=128, seed=42, camera='right_wrist', controlled_object='pin',
             start_s=None, end_s=None, max_step_m=.004, min_nudge_m=.0002, max_pin_tilt_deg=15.,
             legacy_prototype=False):
    if not legacy_prototype:
        raise ValueError('Legacy opposed/frustum scene is superseded. Use generate_pin_insertion_photo_data.py; legacy_prototype=True is only for historical reproduction.')
    if samples_per_episode < 1 or image_size < 32 or max_step_m <= 0:
        raise ValueError('Require positive samples, image_size >=32, positive max_step_m')
    if not 0 < min_nudge_m < .002 or not 0 <= max_pin_tilt_deg <= 90:
        raise ValueError('Require min_nudge_m in (0,.002), max_pin_tilt_deg in [0,90]')
    rng = np.random.default_rng(seed)
    bridge = Real2SimBridge(scene)
    model, data = bridge.model, bridge.data
    model.geom_pos[model.geom('table_top').id,2] -= .05
    manifest_path = Path(scene).parent/'manifest.json'
    parts = json.loads(manifest_path.read_text())['parts']
    _place(bridge,'small_carrier',[3,3,3],np.eye(3))
    mujoco.mj_setConst(model,data)
    cid = model.camera(camera).id
    width = round(image_size*4/3)
    renderer = mujoco.Renderer(model, height=image_size, width=width)
    original_fovy = float(model.cam_fovy[cid])
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # visual CAD, no collision decomposition
    results = {k: [] for k in ('images', 'state', 'keypoints', 'pin_base_uv', 'delta_xyz_m',
                              'active', 'groups', 'source_time_s', 'regime', 'visible','tip_height_m','lateral_distance_m','pin_tilt_deg','gate_support','direct_visible')}
    audits, sources = [], []
    try:
        for episode_path in episodes:
            ep = load_episode(Path(episode_path))
            valid = np.flatnonzero(((ep.time_s >= start_s) if start_s is not None else True) &
                                   ((ep.time_s <= end_s) if end_s is not None else True) & np.ones(len(ep.state), bool))
            if not len(valid):
                raise ValueError(f'No states in requested interval: {episode_path}')
            rows = {int(r['monotonic_ns']): r for r in
                    map(json.loads, (ep.path/'states.jsonl').read_text().splitlines()) if r}
            errors, rotations = [], []
            sources.append(dict(id=ep.metadata['id'], path=str(ep.path),
                                states_sha256=_sha(ep.path/'states.jsonl')))
            for n in range(samples_per_episode):
                i = int(rng.choice(valid))
                state = ep.state[i]
                poses = bridge.set_kinematic_pose(state)
                row = rows[int(ep.state_monotonic_ns[i])]
                for side in ('left', 'right'):
                    recorded = row['arms'][side].get('tool_pose_base')
                    if recorded:
                        errors.append(float(np.linalg.norm(poses[side]['position']-recorded['position_m'])))
                        relative = poses[side]['rotation'].T @ np.asarray(recorded['rotation'])
                        rotations.append(float(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
                if (errors and max(errors) > 1e-5) or (rotations and max(rotations) > 1e-4):
                    raise ValueError('MuJoCo FK disagrees with recorded app FK by >10 micrometers')
                model.cam_fovy[cid] = original_fovy + rng.uniform(-4,4)
                model.light_diffuse[:] = rng.uniform(.5,.95,size=model.light_diffuse.shape)
                model.light_ambient[:] = rng.uniform(.15,.4,size=model.light_ambient.shape)
                for geom_name in ('large_carrier_visual','pin_1_visual'):
                    model.geom_rgba[model.geom(geom_name).id,:3] = rng.uniform(.2,.85,3)
                # CAD objects are placed in the measured wrist camera frustum.
                # This is an explicit tool-object pose PRIOR, not fitted extrinsics.
                cam_rot = data.cam_xmat[cid].reshape(3, 3).copy()
                cam_pos = data.cam_xpos[cid].copy()
                distance = rng.uniform(.065,.11)
                half_height = distance*np.tan(np.deg2rad(model.cam_fovy[cid])/2)
                center = cam_pos + cam_rot @ np.array([rng.uniform(-.25,.25)*half_height, rng.uniform(-.4,.2)*half_height, -distance])
                # Bore plane faces the wrist: local +Z points towards camera.
                ax,az = rng.uniform(-.35,.35,2)
                cx,sx,cz,sz = np.cos(ax),np.sin(ax),np.cos(az),np.sin(az)
                assembly_rotation = cam_rot @ np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
                bore_local = carrier_bore_points()[n % 4] + [0, 0, CARRIER_BORE_HEIGHT_M]
                hole = center.copy()
                carrier_pos = hole-assembly_rotation @ bore_local
                regime = ('aligned', 'edge', 'away','hover')[n % 4]
                radius = {'aligned': (0,min(.00015,min_nudge_m*.75)), 'edge': (min_nudge_m,.012), 'away': (.015,.025), 'hover': (0,.012)}[regime]
                angle = rng.uniform(-np.pi,np.pi)
                distance_xy = np.exp(rng.uniform(np.log(radius[0]),np.log(radius[1]))) if regime == 'edge' else rng.uniform(*radius)
                offset = distance_xy*np.array([np.cos(angle),np.sin(angle),0])
                # Tip at the bore plane; pin stands above it, consistent with CAD z=0 tip.
                if regime == 'hover':
                    offset[2] = rng.uniform(.008,.035)
                tip = hole + assembly_rotation @ offset
                theta = np.deg2rad(rng.uniform(0,max_pin_tilt_deg))
                azimuth = rng.uniform(-np.pi,np.pi)
                ca,sa = np.cos(azimuth),np.sin(azimuth)
                rz = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
                c,s = np.cos(theta),np.sin(theta)
                all_holes = carrier_pos + (carrier_bore_points()+[0,0,CARRIER_BORE_HEIGHT_M]) @ assembly_rotation.T
                hole = all_holes[np.argmin(np.linalg.norm(all_holes-tip,axis=1))]
                _,height,lateral = alignment_gate(tip,hole,assembly_rotation[:,2],True,min_nudge_m=min_nudge_m)
                regime = 'hover' if abs(height) > .001 else 'aligned' if lateral < min_nudge_m else 'edge' if lateral <= .012 else 'away'
                pin_rotation = assembly_rotation @ rz @ np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ rz.T
                _place(bridge, 'large_carrier', carrier_pos, assembly_rotation)
                for gear_name in ('gear_1','gear_2','gear_3','gear_4'):
                    offset_gear = np.asarray(parts[gear_name]['assembled_origin'])-parts['large_carrier']['assembled_origin']
                    _place(bridge,gear_name,carrier_pos+assembly_rotation @ offset_gear,assembly_rotation)
                _place(bridge, 'pin_1', tip, pin_rotation)
                # Park other workpieces outside the local view; preserve robot state.
                for body_name in ('pin_2','pin_3','pin_4'):
                    try:
                        _place(bridge, body_name, [3,3,3], np.eye(3))
                    except KeyError:
                        pass
                mujoco.mj_forward(model, data)
                points = np.array([tip, hole, tip + pin_rotation @ [0,0,.01788]])
                uv, depth = project(points, data.cam_xpos[cid], data.cam_xmat[cid].reshape(3,3), model.cam_fovy[cid], width, image_size)
                in_frame = (depth > 0) & np.all((uv >= 0) & (uv <= 1),axis=1)
                uv = np.clip(uv,0,1)
                renderer.update_scene(data, camera=camera, scene_option=option)
                rgb = renderer.render().copy()
                # Depth visibility rejects real occlusion by arm/CAD, no painted labels.
                renderer.enable_depth_rendering()
                depth_image = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                pixels = np.minimum((uv[:2]*[width,image_size]).astype(int), [width-1,image_size-1])
                # Pin bottom-axis center is amodal behind its own shaft. Allow
                # SAME pin surface, but never gear, carrier, arm, or table occlusion.
                x_pin,y_pin = pixels[0]
                same_pin = bool(segmentation[y_pin,x_pin,0] == model.geom('pin_1_visual').id and
                                segmentation[y_pin,x_pin,1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                direct_visible = depth_visibility(depth[:2], np.array([depth_image[y,x] for x,y in pixels]), in_frame[:2])
                visible = depth_visibility(depth[:2], np.array([depth_image[y,x] for x,y in pixels]), in_frame[:2],same_object=[same_pin,False])
                delta = correction(tip,hole,data.xmat[bridge.base_ids[1]].reshape(3,3),controlled_object,max_step_m)
                x_hole,y_hole = pixels[1]
                pin_at_hole = bool(segmentation[y_hole,x_hole,0] == model.geom('pin_1_visual').id and
                                   segmentation[y_hole,x_hole,1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                carrier_visible = np.count_nonzero(segmentation[:,:,0] == model.geom('large_carrier_visual').id) >= 20
                gate_support = [bool(visible[0]), hole_gate_support(visible[1],pin_at_hole,carrier_visible) and bool(in_frame[1])]
                active,height,lateral = alignment_gate(tip,hole,assembly_rotation[:,2],gate_support,min_nudge_m=min_nudge_m)
                for key, value in dict(images=rgb,state=state,keypoints=uv[:2].flatten(),pin_base_uv=uv[2],
                        delta_xyz_m=delta if active else np.zeros(3),active=[float(active)],groups=ep.metadata['id'],
                        source_time_s=float(ep.time_s[i]),regime=regime,visible=visible,tip_height_m=[height],lateral_distance_m=[lateral],pin_tilt_deg=[np.rad2deg(theta)],gate_support=gate_support,direct_visible=direct_visible).items():
                    results[key].append(value)
            audits.append(dict(episode=ep.metadata['id'],samples=len(errors),
                max_position_error_m=max(errors,default=None),max_rotation_error_rad=max(rotations,default=None)))
    finally:
        renderer.close()
    arrays = {key: np.asarray(value) for key,value in results.items()}
    for key in ('state','keypoints','pin_base_uv','delta_xyz_m','active'):
        arrays[key] = arrays[key].astype(np.float32)
    arrays['group_ids'] = arrays['groups']
    arrays['keypoints_valid'] = np.repeat(arrays['visible'],2,axis=1).astype(np.float32)
    metadata = dict(schema_version=1, generator_version=6, generator='pin_insertion_sim.py', seed=seed, scene=str(Path(scene).resolve()),
        superseded=True, diagnostic_only=True, training_use_approved=False,
        replacement='generate_pin_insertion_photo_data.py: side_by_side/photo_20261027 physical environment',
        scene_sha256=_sha(scene), cad_manifest_sha256=_sha(manifest_path), table_offset_m=.05, table_registration='table-only lowered 0.05m to top z=0 per existing real2sim registration hypothesis; not a measured calibration', camera=camera, controlled_object=controlled_object, controlled_arm='left' if controlled_object == 'pin' else 'right',
        delta_frame='right_native_base', delta_units='meters', max_step_m=max_step_m,min_nudge_m=min_nudge_m,max_pin_tilt_deg=max_pin_tilt_deg,
        delta_definition=f'translation of {controlled_object} towards the other landmark; zero when gate inactive',
        keypoint_validity_semantics='keypoints_valid allows exact own-pin segmentation support for amodal pin center; hole remains direct-visible only. direct_visible is strict1mm for both; gate_support also allows own-pin-over-hole when carrier visible.', keypoints_order=['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v'],
        image_format='RGB uint8 NHWC, native 4:3 aspect ratio', image_height=image_size,image_width=width, domain_randomization=['fovy +/-4 degrees','assembly tilt and roll +/-20 degrees',f'pin tilt 0..{max_pin_tilt_deg} degrees with random azimuth','CAD RGB and scene illumination'],target_bore_definition='closest of all four CAD bores to pin in every state; gate based on nearest tangential distance and signed plane height', state_order='left joints1..6,gripper,right joints1..6,gripper (radians)',
        active_definition=f'pin supported and bore visible or expected own-pin overlap with carrier visible, absolute tip height <=0.001m from carrier plane, nearest-bore tangential distance {min_nudge_m}..0.012m; hover (8..35mm), aligned, away and occluded inactive', class_counts={str(k):int(v) for k,v in zip(*np.unique(arrays['regime'],return_counts=True))},visibility_depth_tolerance_m=.001, pin_tip_visibility='amodal geometric bottom-axis center; own pin surface supported by exact geom segmentation at projected tip pixel; any external foreground rejected at1mm', hole_visibility='keypoint loss valid only with no foreground within1mm; gate separately supports own-pin overlap iff carrier visual occupies >=20 pixels, never gear/arm/table overlap',visible_landmark_counts=arrays['visible'].sum(axis=0).tolist(),active_count=int(arrays['active'].sum()),inactive_count=int(len(arrays['active'])-arrays['active'].sum()),
        samples=len(arrays['state']),sources=sources,fk_vs_recorded_app_pose=audits,
        physics_simulated=False, real_calibrated=False,
        assumptions=['Arm angles are exact measured demo seeds; no fabricated joint perturbations.',
            'Synthetic pin and large carrier CAD placements in measured wrist camera frustum; tool-object transforms NOT image fitted.',
            'Four-bore CENTER pitch-circle radius 0.0179625m (not individual bore radius); approximate bore radius4.17mm and pin radius4mm, CAD radial clearance~0.17mm; bore-mouth plane height 0.01725m; pin CAD tip z=0 and base z=0.01788m. Landmarks outside frame are clipped with validity false.',
            'Camera uses sibling scene extrinsics/FOV; real wrist intrinsics, distortion, extrinsics and base-to-base transform UNVERIFIED.',
            'Default 0.2mm nudge deadband is a geometric prototype threshold, not measured clearance or insertion success; pin orientation 0..15deg is unvalidated for insertion and translation alone cannot correct tilt.',
            'Edge offsets are geometric hypotheses; gravity, support/contact, compliance, slip and insertion success are NOT validated.',
            'Controlled arm follows the object assignment (pin left, carrier right); hand-object coupling is an assumed rigid translation, not validated contact mechanics.',
            'Large carrier plus four gears assembled using sibling manifest CAD origins; real object registration and intended bore identity UNVERIFIED.',
            'Gate is synthetic geometry/visibility supervision with a 1mm carrier support-plane tolerance; physical contact is unobserved, and this is not a real safety or success judgement.'])
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-prototype',action='store_true',help='Explicitly reproduce the superseded opposed/frustum experiment; not the real setup')
    parser.add_argument('--episode',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=Path,default=DEFAULT_SCENE)
    parser.add_argument('--samples-per-episode',type=int,default=256)
    parser.add_argument('--image-size',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--camera',default='right_wrist')
    parser.add_argument('--controlled-object',choices=['pin','carrier'],default='pin')
    parser.add_argument('--start-s',type=float)
    parser.add_argument('--end-s',type=float)
    parser.add_argument('--min-nudge-m',type=float,default=.0002)
    parser.add_argument('--max-pin-tilt-deg',type=float,default=15.)
    parser.add_argument('--max-step-m',type=float,default=.004)
    args=vars(parser.parse_args()); args['episodes']=args.pop('episode'); print(json.dumps(generate(**args),indent=2))

if __name__=='__main__': main()
