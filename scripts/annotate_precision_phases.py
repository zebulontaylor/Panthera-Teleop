#!/usr/bin/env python3
"""Conservative motor-only precision-window proposals, never contact ground truth."""
import argparse,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from hitl_recording import data_timestamp,data_origin

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rotation(axis,angle):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis);x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
def fk(q,xml):
    robot=ET.fromstring(xml);joints=list(robot.findall('joint'));by_child={j.find('child').get('link'):j for j in joints};chain=[];link='tool_link'
    while link in by_child:
        j=by_child[link];chain.append(j);link=j.find('parent').get('link')
    T=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();axis_index=0
    for j in reversed(chain):
        o=j.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3);rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3)
        A=np.eye(4);A[:3,3]=xyz;A[:3,:3]=rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0]);T=T@A
        if j.get('type') in ('revolute','continuous'):
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');B=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();B[:,:3,:3]=np.stack([rotation(axis,a) for a in q[:,axis_index]]);T=T@B;axis_index+=1
    assert axis_index==6
    return T[:,:3,3]
def is_object_opening(baseline, variation, delta, recent_rise):
    return .025 <= baseline <= .9 and variation < .1 and delta >= .25 and recent_rise > .025

def select_phases(candidates):
    gear=[c for c in candidates if c['grasp_width_rad']>=.25 and c['stable_hold_s']>=.5]
    phases=[];abstentions={}
    if len(gear)!=1:
        return [], {'gear_insertion':f'Expected one stable gear candidate; found {len(gear)}',
                    'pin_insertion':'No unique preceding gear anchor'}
    selected=[('gear_insertion',gear[0])]
    pin=[c for c in candidates if .025<=c['grasp_width_rad']<.25 and c['stable_hold_s']>=.5 and c['release_s']-gear[0]['release_s']>=1.]
    if len(pin)==1:selected.append(('pin_insertion',pin[0]))
    else:abstentions['pin_insertion']=f'Expected one stable ordered pin candidate; found {len(pin)}'
    for name,c in selected:
        confidence='medium' if c['tool_terminal_median_speed_m_s']<=.08 and c['max_tool_distance_to_release_m']<=.05 else 'low'
        phases.append(dict(phase=name,**c,confidence=confidence,scope='pre_release_final_approach',evidence='stable object-sized grasp then opening; own-arm URDF tool proximity to release pose; pin requires unique preceding gear',contact_verified=False))
    return phases,abstentions

def apply_visual(result, visual, path, annotation):
    original=result['phases'];by_name={p['phase']:p for p in original};abstentions=dict(result.get('phase_abstentions',{}))
    for key,name in [('gear','gear_insertion'),('pin','pin_insertion')]:
        if not visual.get(key):continue
        v=visual[key];lo=max(annotation['start_s'],v['start_s']);hi=min(annotation['end_s'],v['end_s'])
        assert lo<hi
        by_name[name]=dict(v,phase=name,start_s=lo,end_s=hi,confidence='visual',visual_reviewer_confidence=v['confidence'],scope='visual_phase_window',contact_verified=False,end_right_censored=bool(v.get('end_right_censored',False) or hi<v['end_s']),provenance=dict(source_file=str(path),source_sha256=sha(path),episode_id=visual['episode_id'],exact_review_bounds_s=[v['start_s'],v['end_s']],review_assets=visual.get('review_assets',[]),clipped_to_selected_interval=lo!=v['start_s'] or hi!=v['end_s']))
        abstentions.pop(name,None)
    phases=[by_name[name] for name in ('gear_insertion','pin_insertion') if name in by_name]
    result.update(phases=phases,motor_proposals=original,status='proposed' if len(phases)==2 else 'partial' if phases else 'abstained',phase_abstentions=abstentions,abstention_reason='; '.join(abstentions.values()) or None)

def detect(root,annotation):
    identifier=annotation.get('source_id',annotation['id']);p=root/identifier;m=json.loads((p/'metadata.json').read_text());r=[json.loads(x) for x in (p/'states.jsonl').read_text().splitlines() if x.strip()];origin=data_origin(m)
    times=np.array([(data_timestamp(x)-origin)/1e9 for x in r]);t=np.arange(annotation['start_s'],annotation['end_s']+.0001,.05)
    q=np.array([x['arms']['left']['position_rad'] for x in r]);q=np.stack([np.interp(t,times,q[:,i]) for i in range(6)],axis=1)
    grip=np.interp(t,times,[x['arms']['left']['gripper']['position'] for x in r]);g=np.array([np.median(grip[max(0,i-1):min(len(t),i+2)]) for i in range(len(t))])
    tool=fk(q,m['context']['urdf_sources']['left']);speed=np.linalg.norm(np.gradient(tool,.05,axis=0),axis=1)
    candidates=[];last=-100
    for i in range(12,len(t)-8):
        pre=g[i-10:i-2];post=g[i:i+9];baseline=float(np.median(pre));delta=float(max(post)-baseline)
        if t[i]-last<.8 or not is_object_opening(baseline,float(np.ptp(pre)),delta,float(g[i]-g[i-2])):continue
        # First clear opening sample after a stable object-sized grasp.
        release=i;last=t[i];start=max(0,i-40)
        distance=np.linalg.norm(tool[start:i+1]-tool[i],axis=1)
        far=np.flatnonzero(distance>.035)
        if len(far):start+=int(far[-1])+1
        start=min(start,max(0,i-10))
        while start<i-10 and abs(g[start]-baseline)>.12:start+=1
        hold=0
        for j in range(i-1,-1,-1):
            if abs(g[j]-baseline)>.1:break
            hold+=1
        pre_speed=float(np.median(speed[max(start,i-8):i]));approach_speed=float(np.percentile(speed[max(0,start-12):max(start,1)],75))
        candidates.append(dict(release_s=round(float(t[i]),3),start_s=round(float(t[start]),3),end_s=round(float(t[i]),3),grasp_width_rad=round(baseline,4),opening_delta_rad=round(delta,4),stable_hold_s=round(hold*.05,3),tool_terminal_median_speed_m_s=round(pre_speed,4),prior_approach_p75_speed_m_s=round(approach_speed,4),terminal_to_approach_speed_ratio=round(pre_speed/max(approach_speed,.001),3),max_tool_distance_to_release_m=round(float(np.max(np.linalg.norm(tool[start:i+1]-tool[i],axis=1))),4)))
    phases,abstentions=select_phases(candidates)
    reason='; '.join(abstentions.values()) or None
    return dict(id=annotation['id'],source_id=identifier,source_category='kept',selected_interval_s=[annotation['start_s'],annotation['end_s']],status=('proposed' if len(phases)==2 else 'partial' if phases else 'abstained'),phase_abstentions=abstentions,abstention_reason=reason,phases=phases,candidates=candidates,source_sha256={n:sha(p/n) for n in ('metadata.json','states.jsonl')})
def main(a):
    spec=json.loads(a.annotations.read_text());eps=[e for e in spec['episodes'] if e.get('source_category','kept')=='kept'];results=[]
    visual_data=json.loads(a.visual_checks.read_text()) if a.visual_checks else {'episodes':[]}
    visual_map={e['episode_id']:e for e in visual_data['episodes']}
    for i,e in enumerate(eps):
        try:
            result=detect(a.source,e)
            if e['id'] in visual_map:apply_visual(result,visual_map[e['id']],a.visual_checks,e)
            for phase in result['phases']:assert e['start_s']<=phase['start_s']<phase['end_s']<=e['end_s']
            results.append(result)
        except Exception as exc:results.append(dict(id=e['id'],status='error',error=f'{type(exc).__name__}: {exc}',phases=[]))
        print(f'{i+1}/{len(eps)} {e["id"]}: {results[-1]["status"]}',flush=True)
    counts={s:sum(e['status']==s for e in results) for s in ('proposed','partial','abstained','error')}
    out=dict(version=2,method='motor_release_anchored_precision_proposals_with_explicit_visual_overrides',limits=['No calibrated camera/base transforms or force/contact ground truth','Precision windows are approach/release proxies, not verified insertion or success','Grasp-width thresholds and gear-before-pin order may fail on recoveries; ambiguous episodes abstain','Motor proposals use no camera frames; explicit visual overrides include seating and retain reviewer bounds/provenance; raw recordings unchanged'],parameters=dict(resample_hz=20,gripper_median_filter_samples=3,minimum_opening_delta_rad=.25,object_grasp_range_rad=[.025,.9],gear_minimum_grasp_rad=.25,minimum_stable_hold_s=.5,maximum_window_s=2,minimum_window_s=.5,tool_neighborhood_m=.035),provenance=dict(script_sha256=sha(Path(__file__)),annotations_sha256=sha(a.annotations),visual_checks_sha256=sha(a.visual_checks) if a.visual_checks else None),coverage=dict(episodes=len(results),**counts,proposed_phases=sum(len(e['phases']) for e in results),gear_phases=sum(any(p['phase']=='gear_insertion' for p in e['phases']) for e in results),pin_phases=sum(any(p['phase']=='pin_insertion' for p in e['phases']) for e in results),visual_override_episodes=len(visual_map)),episodes=results)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['coverage']))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data/kept'));p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--visual-checks',type=Path);main(p.parse_args())
