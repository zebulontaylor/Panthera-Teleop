"""Reproduce documented carrier_real_face_v4 appearance without changing physics.

This is a NEW visual reconstruction from recovered checkpoint metadata, not the
unavailable original mesh or corrected STEP. Original lower CAD remains; its
upper plate is rebuilt with documented holes/bosses. No collision/inertial/target
or original source asset is changed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import os
from importlib.metadata import version
from io import BytesIO
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PROFILE=ROOT/'config/pin-insertion-photo/carrier-real-face-v4-profile.json'
DEFAULT_ORIGINAL=ROOT.parent/'RobotArmTraining/sim/actuator/meshes/large_carrier/visual.stl'
DEFAULT_OUTPUT=ROOT/'work/pin-insertion-real/reproduced-carrier-v4'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path,data):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:f.write(data);temporary=f.name
    os.replace(temporary,path)


def build_carrier_visual_profile(profile_path=DEFAULT_PROFILE,original_path=DEFAULT_ORIGINAL,output_dir=DEFAULT_OUTPUT):
    profile_path=Path(profile_path);original_path=Path(original_path);output_dir=Path(output_dir)
    profile_sha,original_sha=_sha(profile_path),_sha(original_path)
    builder_sha=_sha(__file__)
    dependencies={name:version(name) for name in ('trimesh','manifold3d','numpy')}
    output_dir.mkdir(parents=True,exist_ok=True);mesh_path=output_dir/'reproduced_carrier_real_face_v4.stl';provenance_path=output_dir/'provenance.json'
    if provenance_path.exists() and mesh_path.exists():
        cached=json.loads(provenance_path.read_text())
        if cached.get('builder_source_sha256')==builder_sha and cached.get('dependency_versions')==dependencies and cached.get('profile_sha256')==profile_sha and cached.get('original_mesh_sha256')==original_sha and cached.get('mesh_sha256')==_sha(mesh_path):return cached
    import trimesh
    profile=json.loads(profile_path.read_text())
    original=trimesh.load_mesh(original_path,process=True)
    if not original.is_watertight:raise ValueError('Original lower CAD must be watertight')
    original_mm=original.copy();original_mm.apply_scale(1000)
    mouth=float(profile['upper_plate_mouth_z_mm'])
    # These plate planes are recovered from ORIGINAL CAD vertices, not guessed.
    zs=np.unique(np.round(original_mm.vertices[:,2],5))
    top_candidates=zs[(zs<mouth-1e-3)&(zs>mouth-5)]
    if len(top_candidates)!=1:raise ValueError('Original upper-plate bottom not uniquely recoverable')
    plate_bottom=float(top_candidates[0])
    lower_clip=trimesh.creation.box(extents=[100,100,plate_bottom+1.01]);lower_clip.apply_translation([0,0,(plate_bottom-1)/2+.005])
    lower=trimesh.boolean.intersection([original_mm,lower_clip],engine='manifold')
    def cylinder(radius,z0,z1,xy=(0,0)):
        a=trimesh.creation.cylinder(radius=radius,height=z1-z0,sections=128);a.apply_translation([*xy,(z0+z1)/2]);return a
    upper=cylinder(profile['plate_outer_diameter_mm']/2,plate_bottom,mouth)
    bosses=profile['raised_mounting_bosses'];pieces=[upper];boss_centers=[]
    # Existing CAD mounting centers fix the otherwise undocumented boss phase.
    high=original_mm.vertices[original_mm.vertices[:,2]>mouth+.1]
    for sx,sy in ((1,1),(-1,1),(-1,-1),(1,-1)):
        quadrant=high[(high[:,0]*sx>0)&(high[:,1]*sy>0)]
        if not len(quadrant):raise ValueError('Cannot recover original mounting-boss phase')
        center=(quadrant.min(0)+quadrant.max(0))/2;theta=float(np.arctan2(center[1],center[0]))
        xy=np.array([np.cos(theta),np.sin(theta)])*bosses['pitch_radius_mm'];boss_centers.append(xy.tolist())
        block=trimesh.creation.box(extents=[bosses['radial_width_mm'],bosses['tangential_width_mm'],bosses['height_above_plate_mm']+.01])
        pose=trimesh.transformations.rotation_matrix(theta,[0,0,1]);pose[:3,3]=[*xy,mouth+bosses['height_above_plate_mm']/2-.005];block.apply_transform(pose);pieces.append(block)
    upper=trimesh.boolean.union(pieces,engine='manifold')
    holes=[cylinder(profile['central_upper_bore_diameter_mm']/2,plate_bottom-.1,mouth+.1)]
    bore_centers=[];inner_centers=[]
    for i in range(4):
        angle=i*np.pi/2;xy=np.array([np.cos(angle),np.sin(angle)])*profile['insertion_bore_pitch_radius_mm'];bore_centers.append(xy.tolist())
        holes.append(cylinder(profile['insertion_bore_diameter_mm']/2,plate_bottom-.1,mouth+.1,xy))
        inner=profile['inner_threaded_holes'];angle+=np.deg2rad(inner['phase_relative_to_insertion_bores_deg']);xy=np.array([np.cos(angle),np.sin(angle)])*inner['pitch_radius_mm'];inner_centers.append(xy.tolist())
        holes.append(cylinder(inner['diameter_mm']/2,plate_bottom-.1,mouth+.1,xy))
    for xy in boss_centers:holes.append(cylinder(bosses['mounting_hole_diameter_mm']/2,plate_bottom-.1,mouth+bosses['height_above_plate_mm']+.1,xy))
    upper=trimesh.boolean.difference([upper,*holes],engine='manifold')
    result=trimesh.boolean.union([lower,upper],engine='manifold')
    # Collapse CSG seam slivers before float32 STL quantization. The 1um bound
    # is far below the documented approximate visual dimensions.
    import manifold3d
    solid=manifold3d.Manifold(manifold3d.Mesh64(np.asarray(result.vertices,dtype=np.float64),np.asarray(result.faces,dtype=np.uint64)))
    simplified=solid.simplify(.001).to_mesh64()
    result=trimesh.Trimesh(vertices=np.asarray(simplified.vert_properties)[:,:3],faces=np.asarray(simplified.tri_verts),process=False)
    result.apply_scale(.001)
    if not result.is_watertight or not result.is_winding_consistent or result.volume<=0:raise ValueError('Generated visual mesh invalid')
    if len(result.faces)>=200000:raise ValueError('Exceeds MuJoCo STL facet limit')
    if not np.allclose(result.bounds[0,:2],[-.02649,-.02649],atol=1e-6) or not np.isclose(result.bounds[1,2],.02125,atol=1e-6):raise ValueError('Documented visual bounds changed')
    raw=trimesh.exchange.stl.export_stl(result)
    reloaded=trimesh.load_mesh(BytesIO(raw),file_type='stl',process=True)
    if not reloaded.is_watertight or not reloaded.is_winding_consistent or reloaded.volume<=0:raise ValueError('Exported/reloaded float32 STL is not watertight')
    _atomic(mesh_path,raw)
    saved=trimesh.load_mesh(mesh_path,process=True)
    if not saved.is_watertight or not saved.is_winding_consistent:raise ValueError('Saved STL post-publication verification failed')
    report={'profile':'new_reproduction_of_carrier_real_face_v4_version2','newly_reproduced':True,'original_recovered_mesh':False,
      'builder_source_sha256':builder_sha,'dependency_versions':dependencies,'source_profile_path':str(profile_path.resolve()),'profile_sha256':profile_sha,'original_mesh_path':str(original_path.resolve()),'original_mesh_sha256':original_sha,
      'mesh_path':str(mesh_path.resolve()),'mesh_sha256':_sha(mesh_path),'post_export_verified':True,'post_export_triangles':len(reloaded.faces),'triangles':len(result.faces),'watertight':reloaded.is_watertight,'winding_consistent':result.is_winding_consistent,
      'bounds_m':result.bounds.tolist(),'original_upper_plate_bottom_mm':plate_bottom,'upper_plate_mouth_z_mm':mouth,
      'insertion_bore_centers_mm':bore_centers,'insertion_bore_diameter_mm':profile['insertion_bore_diameter_mm'],'central_upper_bore_diameter_mm':profile['central_upper_bore_diameter_mm'],
      'inner_hole_centers_mm':inner_centers,'boss_centers_mm':boss_centers,'physics_modified':False,'targets_modified':False,
      'csg_overlap_mm':.01,'seam_simplification_bound_mm':.001,'method':'Retain original CAD below upper plate; rebuild upper plate/holes and raised bosses using recovered profile dimensions. Manifold boolean CSG in millimetres.',
      'limitations':['New approximate visual reproduction, not recovered corrected STEP or manufacturing CAD.',
         '4mm boss height and smooth unthreaded holes are documented appearance assumptions; no helical thread geometry.',
         'Only upper visual plate uses8.24mm insertion bore diameter; original lower CAD, all collision pieces/contact hulls/inertias and target coordinates remain unchanged.',
         'Visual radius4.12mm is separate from original collision geometry; this does not certify hardware insertion clearance.']}
    _atomic(provenance_path,(json.dumps(report,indent=2)+'\n').encode());return report


def apply_carrier_visual_profile(root,artifacts=None):
    """Replace ONLY large_carrier_visual geom's density-zero visual mesh asset."""
    artifacts=artifacts or {};report=build_carrier_visual_profile(**artifacts)
    assets=root.find('asset');original=assets.find("mesh[@name='large_carrier_visual']")
    geom=root.find("worldbody/.//geom[@name='large_carrier_visual']")
    if original is None or geom is None:raise ValueError('Expected existing carrier visual mesh/geom')
    if geom.get('contype','0')!='0' or geom.get('conaffinity','0')!='0':raise ValueError('Carrier target geom is not visual-only')
    name='photo_reproduced_carrier_real_face_v4';existing=assets.find(f"mesh[@name='{name}']")
    if existing is None:
        import copy
        mesh=copy.deepcopy(original);mesh.set('name',name);mesh.set('file',report['mesh_path']);assets.append(mesh)
    geom.set('mesh',name)
    report['visual_geom']='large_carrier_visual';return report

if __name__=='__main__':print(json.dumps(build_carrier_visual_profile(),indent=2))
