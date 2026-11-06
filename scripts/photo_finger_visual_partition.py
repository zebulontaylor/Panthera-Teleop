"""Exact CAD connected-component color partition; original physics/assets untouched.

Component membership is externally audited. Each exported STL facet record is
copied byte-for-byte from the original source, with no spatial clipping, scaling,
coordinate changes, or remeshing. Only density-zero visual geoms are replaced.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import os
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
VERSIONED_AUDIT=ROOT/'config/pin-insertion-photo/finger-components.json'
DEFAULT_AUDIT=VERSIONED_AUDIT if VERSIONED_AUDIT.exists() else ROOT/'work/pin-insertion-real/cad-finger-audit/audit.json'
DEFAULT_OUTPUT=ROOT/'work/pin-insertion-real/finger-visual-partition'


def _sha(data):return hashlib.sha256(data).hexdigest()


def _write_atomic(path,data):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(data);temporary=f.name
    os.replace(temporary,path)


def export_exact_partition(audit_path=DEFAULT_AUDIT,output_dir=DEFAULT_OUTPUT):
    audit_path=Path(audit_path);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    audit=json.loads(audit_path.read_text());report={'audit_path':str(audit_path.resolve()),'audit_sha256':_sha(audit_path.read_bytes()),'meshes':{}}
    for side in ('L','R'):
        row=audit['mesh_report'][side];source=Path(row['source'])
        portable_source=ROOT.parent/f'RobotArmTraining/sim/panthera/meshes/visual/{side}_finger.STL'
        if not source.exists():source=portable_source
        raw=source.read_bytes()
        if _sha(raw)!=row['source_sha256']:raise ValueError('Original CAD source hash changed')
        count=struct.unpack('<I',raw[80:84])[0]
        if len(raw)!=84+count*50 or count!=row['source_triangles']:raise ValueError('Expected audited binary STL')
        selected=sorted(map(int,row['white_source_triangle_indices']))
        if len(selected)!=len(set(selected)) or not selected or min(selected)<0 or max(selected)>=count:raise ValueError('Invalid exact component indices')
        white=set(selected);groups={'white':selected,'black':[i for i in range(count) if i not in white]}
        mesh_report={'source':str(source.resolve()),'source_sha256':_sha(raw),'source_triangles':count,'white_component_id':row['white_component_id']}
        for color,indices in groups.items():
            header=('Exact original CAD facet partition '+side+' '+color).encode().ljust(80,b' ')[:80]
            output=header+struct.pack('<I',len(indices))+b''.join(raw[84+i*50:84+(i+1)*50] for i in indices)
            path=output_dir/f'{side}_{color}.STL';_write_atomic(path,output)
            mesh_report[color]={'path':str(path.resolve()),'sha256':_sha(output),'triangles':len(indices)}
        if sum(len(v) for v in groups.values())!=count:raise ValueError('Facet partition incomplete')
        report['meshes'][side]=mesh_report
    _write_atomic(output_dir/'provenance.json',(json.dumps(report,indent=2)+'\n').encode())
    return report


def apply_finger_visual_partition(root,artifacts=None):
    """Mutate existing photo-scene ElementTree root before MuJoCo compilation.

    ``artifacts`` may be an audit.json path or {audit_path,output_dir}. The adapter
    invokes this after the existing photo setup. No joints, inertials, collision
    geoms, cameras, body transforms, keyframes, or actuators are modified.
    """
    if isinstance(artifacts,(str,Path)):
        artifacts={'audit_path':artifacts}
    artifacts=artifacts or {}
    report=export_exact_partition(artifacts.get('audit_path',DEFAULT_AUDIT),artifacts.get('output_dir',DEFAULT_OUTPUT))
    assets=root.find('asset')
    if assets is None:raise ValueError('Scene has no asset section')
    for side in ('L','R'):
        original=assets.find(f"mesh[@name='{side}_finger']")
        if original is None:raise ValueError('Expected original finger visual mesh')
        for color in ('white','black'):
            name=f'photo_{side}_'+('white_digit' if color=='white' else 'black_mount')
            if assets.find(f"mesh[@name='{name}']") is None:
                mesh=copy.deepcopy(original);mesh.set('name',name);mesh.set('file',report['meshes'][side][color]['path']);assets.append(mesh)
    changed=[]
    for body in root.findall('worldbody/.//body'):
        for original in list(body.findall('geom')):
            mesh_name=original.get('mesh','')
            if original.get('class')!='visual' or mesh_name not in ('L_finger','R_finger'):continue
            side=mesh_name[0];index=list(body).index(original)
            for offset,color in enumerate(('white','black')):
                geom=copy.deepcopy(original)
                name=body.get('name',mesh_name)+'_photo_'+color+'_visual'
                geom.set('name',name);geom.set('mesh',f'photo_{side}_'+('white_digit' if color=='white' else 'black_mount'))
                geom.set('rgba','.83 .85 .81 1' if color=='white' else '.035 .038 .043 1')
                geom.set('contype','0');geom.set('conaffinity','0');geom.set('density','0');geom.set('group','2')
                # A legacy material must not override explicit component colors.
                geom.attrib.pop('material',None)
                body.insert(index+offset,geom);changed.append(name)
            body.remove(original)
    if not changed:
        prior=[g for g in root.iter('geom') if g.get('mesh','') in ('photo_L_white_digit','photo_R_white_digit','photo_L_black_mount','photo_R_black_mount')]
        if len(prior)==8:
            report.update(replaced_visual_geoms=[g.get('name') for g in prior],physics_modified=False,geometry_coordinates_modified=False,already_applied=True)
            return report
    if len(changed)!=8:raise ValueError(f'Expected four original visual fingers, replaced {len(changed)//2}')
    report.update(replaced_visual_geoms=changed,physics_modified=False,geometry_coordinates_modified=False,
                  method='Audited connected component11, exact source facet records; white front digit and black mount/fasteners')
    return report
