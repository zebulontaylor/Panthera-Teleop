#!/usr/bin/env python3
"""Diagnostic paired MuJoCo lighting hypotheses in corrected sibling photo scene.

No real calibration or training approval is claimed. Reference removes held pin
and its holding arm from rendering only; physics state is unchanged. All pairs
are rendered at one exact camera/carrier state and lighting. No live interfaces.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np


def generate(args):
    import mujoco
    from PIL import Image,ImageDraw
    from pin_insertion_environment import create_photo_environment,PhotoSetupBridge
    from generate_pin_insertion_photo_data import landmarks,project,visibility
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    env=create_photo_environment(setup='nominal',carrier_visual_profile=False)
    renderer=None
    try:
        env.reset(seed=args.seed,options={'randomize':False})
        model,data=env.model,env.data
        if env.sim.layout!='side_by_side':raise RuntimeError('legacy opposed scene rejected')
        w=round(args.height*4/3);h=args.height
        renderer=mujoco.Renderer(model,height=h,width=w)
        opts=mujoco.MjvOption();opts.geomgroup[:]=[1,1,1,0,0,0];opts.sitegroup[:]=0
        refopts=mujoco.MjvOption();refopts.geomgroup[:]=opts.geomgroup;refopts.sitegroup[:]=0
        refopts.geomgroup[5]=0
        # Put held pin and holding robot into a render-only group. Current keeps
        # that group visible; reference hides it, including its cast shadow.
        left=model.body('left_base').id
        def is_left(b):
            while b:
                if b==left:return True
                b=int(model.body_parentid[b])
            return False
        hidden=[i for i,b in enumerate(model.geom_bodyid) if is_left(int(b)) or int(b)==env.pin_id]
        model.geom_group[hidden]=5;opts.geomgroup[5]=1
        cam=model.camera('right_wrist').id
        points,_=landmarks(env)
        uv,depth=project(points,data.cam_xpos[cam],data.cam_xmat[cam].reshape(3,3),float(model.cam_fovy[cam]),w,h)
        pin_geoms=set(np.flatnonzero(model.geom_bodyid==env.pin_id).tolist())
        # Geometry visibility is checked once; all variations change light only.
        renderer.update_scene(data,camera='right_wrist',scene_option=opts)
        renderer.enable_depth_rendering();di=renderer.render().copy();renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
        _,valid,_,_=visibility(uv,depth,di,seg,pin_geoms)
        rng=np.random.default_rng(args.seed);images=[];refs=[];light_params=[]
        # Model lights cast real renderer shadows. High specular materials produce
        # actual surface highlights, without synthetic image-space impostor pins.
        model.vis.quality.shadowsize=2048
        model.mat_specular[:]=.8;model.mat_shininess[:]=.9
        carrier=data.xpos[env.carrier_id].copy()
        for i in range(args.pairs):
            theta=rng.uniform(0,2*np.pi);elevation=rng.uniform(.08,1.2)
            direction=np.array([np.cos(theta),np.sin(theta),-elevation]);direction/=np.linalg.norm(direction)
            model.light_active[:]=False
            if model.nlight:
                model.light_active[0]=True;model.light_type[0]=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL;model.light_castshadow[0]=True
                model.light_dir[0]=direction;model.light_pos[0]=carrier-direction*.5
                model.light_ambient[0]=rng.uniform(.015,.14,3)
                model.light_diffuse[0]=rng.uniform(.65,1.3,3)
                model.light_specular[0]=rng.uniform(.6,1.5,3)
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera='right_wrist',scene_option=opts);current=renderer.render().copy()
            renderer.update_scene(data,camera='right_wrist',scene_option=refopts);reference=renderer.render().copy()
            images.append(current);refs.append(reference);light_params.append(direction.copy())
        state=PhotoSetupBridge(environment=env).native_state()
        meta={'schema_version':1,'generator':Path(__file__).name,'diagnostic_only':True,'training_use_approved':False,
              'real_calibrated':False,'synthetic_hypothesis':True,'suggested_max_training_weight':.05,
              'source':'corrected sibling RobotArmTraining photo_20261027 nominal environment reset',
              'source_split':'synthetic_diagnostic_not_real_split','seed':args.seed,'scene_layout':env.sim.layout,
              'setup':'nominal','reference':'same camera/carrier/light; held pin and left holding arm hidden in renderer only',
              'geometry':'existing reset CAD hypothesis, not image calibrated; lighting-only repeated state',
              'keypoint_order':['pin_tip_u','pin_tip_v','hole_u','hole_v'],'keypoint_visibility':'depth + own-pin segmentation; geometry visibility only',
              'mujoco_version':mujoco.__version__,'pairs':args.pairs,'train_policy':'excluded by default; require deliberate small-weight diagnostic opt-in'}
        np.savez_compressed(out,images=np.asarray(images),reference_images=np.asarray(refs),
            keypoints=np.tile(uv.reshape(1,4),(args.pairs,1)).astype(np.float32),
            keypoints_valid=np.tile(np.repeat(valid,2),(args.pairs,1)),
            state=np.tile(state,(args.pairs,1)).astype(np.float32),light_direction=np.asarray(light_params),
            group_ids=np.repeat(f'diagnostic_photo_reset_seed_{args.seed}',args.pairs),metadata_json=np.asarray(json.dumps(meta)))
        out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        n=min(args.pairs,12);sheet=Image.new('RGB',(w*2,h*n+24*n))
        draw=ImageDraw.Draw(sheet)
        for i in range(n):
            y=i*(h+24);draw.text((4,y+4),f'{i}: CURRENT / CLEAN REFERENCE; synthetic uncalibrated lighting hypothesis',fill='white')
            sheet.paste(Image.fromarray(images[i]),(0,y+24));sheet.paste(Image.fromarray(refs[i]),(w,y+24))
        sheet.save(out.with_name(out.stem+'-contact-sheet.jpg'))
        return meta
    finally:
        if renderer is not None:renderer.close()
        env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--pairs',type=int,default=32);p.add_argument('--height',type=int,default=240);p.add_argument('--seed',type=int,default=319)
    a=p.parse_args()
    if not 1<=a.pairs<=2048 or not 64<=a.height<=480:p.error('pairs 1..2048, height 64..480')
    print(json.dumps(generate(a),indent=2))
