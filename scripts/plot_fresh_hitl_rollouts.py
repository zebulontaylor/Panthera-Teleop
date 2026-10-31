#!/usr/bin/env python3
"""Plot frozen fresh-HITL rollout critic scores without calibration claims."""
from __future__ import annotations
import argparse, base64, html, json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CAMERAS=("overhead","wrist_port2","wrist_port3")

def read_jsonl(path:Path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
def enc(path:Path)->str:return 'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
def key(eid,t):return (str(eid),round(float(t),6))
def centered3(values):
 out=[]
 for i in range(len(values)):
  window=values[max(0,i-1):min(len(values),i+2)]
  out.append(sum(window)/len(window))
 return out
def rmse(values):
 d=[values[i]-values[i-1] for i in range(1,len(values))]
 return math.sqrt(sum(x*x for x in d)/len(d)) if d else 0.0
def stats(values):
 d=[abs(values[i]-values[i-1]) for i in range(1,len(values))]
 sd=sorted(d); p95=sd[min(len(sd)-1,int(.95*(len(sd)-1)))] if sd else 0.0
 return {'min':min(values),'max':max(values),'mean':sum(values)/len(values),'range':max(values)-min(values),
         'frame_delta_rmse':rmse(values),'frame_delta_abs_max':max(d) if d else 0.0,'frame_delta_abs_p95':p95}
def ext_key(e):return str(e['episode_id'])

def find_val(d,*names):
 for n in names:
  if n in d and isinstance(d[n],(int,float)):return float(d[n])
 return None

def event_times(episode):
 start=episode.get('start_data_monotonic_ns') or episode.get('start_monotonic_ns')
 if not isinstance(start,(int,float)):return []
 out=[]; pending_human=False
 transitions=episode.get('control_transitions') or []
 for event in transitions:
  mono=event.get('monotonic_ns') if isinstance(event,dict) else None
  if not isinstance(mono,(int,float)):continue
  source=event.get('source','unknown')
  if source=='human':
   out.append({'t':(float(mono)-float(start))/1e9,'source':'human takeover','sequence':event.get('sequence')});pending_human=True
  elif pending_human and source in ('hold','policy'):
   out.append({'t':(float(mono)-float(start))/1e9,'source':'human release → '+source,'sequence':event.get('sequence')});pending_human=False
 return out

def transition_windows(times,events,duration):
 # Start/mid/end plus before/after state-control changes. Deduplicate nearby points.
 targets=[(0.0,'start'),(duration/2,'midpoint'),(duration,'end')]
 for ev in events:
  if not ev['source'].startswith('human '):continue
  t=max(0.0,min(duration,float(ev['t'])))
  targets.extend([(max(0.0,t-.2),f"before {ev['source']} boundary"),(min(duration,t+.2),f"after {ev['source']} boundary")])
 targets.sort()
 chosen=[]
 for t,label in targets:
  idx=min(range(len(times)),key=lambda i:abs(times[i]-t))
  actual=times[idx]
  if chosen and idx==chosen[-1]['index']:
   continue
  chosen.append({'index':idx,'t':actual,'label':label})
 return chosen

def ymap(v,lo,hi,y0,h):return y0+h-(float(v)-lo)/(hi-lo)*h
def xmap(t,t0,t1,x0,w):return x0+(float(t)-t0)/(t1-t0)*w if t1>t0 else x0+w/2
def path(values,times,lo,hi,box):
 x0,y0,w,h=box
 return ' '.join(('M' if i==0 else 'L')+f"{xmap(t,times[0],times[-1],x0,w):.2f},{ymap(v,lo,hi,y0,h):.2f}" for i,(t,v) in enumerate(zip(times,values)))

def svg_chart(episode,score_metric,metric_label,raw,smoothed,items,events,zoom,main_bounds=(0.0,1.0)):
 eid=episode['episode_id']; times=[r['t'] for r in items]; width=1500;height=850
 x0=110;w=1320;top_y=105;panel_h=250;bottom_y=480
 lo,hi=main_bounds if not zoom else (min(raw),max(raw))
 if zoom:
  pad=max((hi-lo)*.15,0.002);lo-=pad;hi+=pad
  if hi-lo<.004:lo-=.002;hi+=.002
 minr,maxr=min(raw),max(raw)
 meta_out=episode.get('raw_metadata_outcome') or {}
 title=f"{eid} — {metric_label}"
 subtitle=(f"external fresh_test · {len(items)} samples · {items[-1]['t']:.1f}s · "+
           f"operator label={meta_out.get('label','unknown')} · human assisted={meta_out.get('human_assisted','unknown')}")
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
 '<rect width="100%" height="100%" fill="#fff"/>',
 '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:26px;font-weight:bold}.sub{font-size:14px;fill:#444}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.raw{fill:none;stroke:#2878b5;stroke-width:1.3;opacity:.65}.smooth{fill:none;stroke:#cb4938;stroke-width:3}.human{fill:#a78bfa;opacity:.18}.event{stroke:#50359b;stroke-dasharray:4 4;stroke-width:1.5}.legend{font-size:14px}</style>',
 f'<text x="60" y="42" class="title">{html.escape(title)}</text>',f'<text x="60" y="68" class="sub">{html.escape(subtitle)}</text>']
 main_lo,main_hi=main_bounds
 for panel_y,ylo,yhi,panel_label in ((top_y,main_lo,main_hi,f'Shared model output scale [{main_lo:.3f},{main_hi:.3f}]'),(bottom_y,lo,hi,'Raw-range zoom; same uncalibrated values')):
  parts.append(f'<text x="62" y="{panel_y-16}" font-size="15" font-weight="bold">{panel_label}</text>')
  for k in range(6):
   v=ylo+(yhi-ylo)*k/5;y=ymap(v,ylo,yhi,panel_y,panel_h)
   parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+w}" y2="{y:.2f}" class="grid"/>')
   parts.append(f'<text x="{x0-12}" y="{y+5:.2f}" font-size="13" text-anchor="end">{v:.3f}</text>')
  for i,item in enumerate(items):
   if item.get('human_control_active') is True:
    t0=item['t'];t1=items[i+1]['t'] if i+1<len(items) else min(items[-1]['t']+.2,items[-1].get('episode_end_s',items[-1]['t']+.2))
    parts.append(f'<rect x="{xmap(t0,times[0],times[-1],x0,w):.2f}" y="{panel_y}" width="{max(1,xmap(t1,times[0],times[-1],x0,w)-xmap(t0,times[0],times[-1],x0,w)):.2f}" height="{panel_h}" class="human"/>')
  for ev in events:
   if 0<=ev['t']<=times[-1]:
    x=xmap(ev['t'],times[0],times[-1],x0,w)
    parts.append(f'<line x1="{x:.2f}" y1="{panel_y}" x2="{x:.2f}" y2="{panel_y+panel_h}" class="event"/>')
  parts.append(f'<path d="{path(raw,times,ylo,yhi,(x0,panel_y,w,panel_h))}" class="raw"/>')
  parts.append(f'<path d="{path(smoothed,times,ylo,yhi,(x0,panel_y,w,panel_h))}" class="smooth"/>')
  parts.append(f'<line x1="{x0}" y1="{panel_y+panel_h}" x2="{x0+w}" y2="{panel_y+panel_h}" class="axis"/>')
  parts.append(f'<line x1="{x0}" y1="{panel_y}" x2="{x0}" y2="{panel_y+panel_h}" class="axis"/>')
 for k in range(6):
  t=times[0]+(times[-1]-times[0])*k/5;x=xmap(t,times[0],times[-1],x0,w)
  parts.append(f'<text x="{x:.2f}" y="{bottom_y+panel_h+22}" font-size="13" text-anchor="middle">{t:.1f}</text>')
 parts += [f'<text x="{x0+w/2}" y="{bottom_y+panel_h+50}" font-size="14" text-anchor="middle">seconds from HITL data recording start</text>',
 f'<line x1="{x0}" y1="800" x2="{x0+45}" y2="800" class="raw"/><text x="{x0+54}" y="805" class="legend">raw framewise score</text>',
 f'<line x1="{x0+270}" y1="800" x2="{x0+315}" y2="800" class="smooth"/><text x="{x0+324}" y="805" class="legend">centered 3-sample mean (~0.5s diagnostic only)</text>',
 f'<rect x="{x0+730}" y="789" width="25" height="15" class="human"/><text x="{x0+763}" y="802" class="legend">per-frame human control</text>',
 f'<line x1="{x0+1055}" y1="800" x2="{x0+1085}" y2="800" class="event"/><text x="{x0+1095}" y="805" class="legend">recorded control-source boundary</text>',
 '</svg>']
 return ''.join(parts)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--holdout',type=Path,default=Path('work/state-value-20261029/fresh-hitl-holdout'))
 ap.add_argument('--scores',type=Path,default=Path('work/state-value-20261029/final-evaluation/fresh_test_rollout_scores.jsonl'))
 ap.add_argument('--output',type=Path,default=Path('work/state-value-20261029/plots'))
 ap.add_argument('--value-key',default='weak_demo_discounted_value_uncalibrated')
 ap.add_argument('--progress-key',default='weak_demo_progress_uncalibrated')
 a=ap.parse_args();hold=a.holdout.resolve();score_path=a.scores.resolve();out=a.output.resolve()
 if not score_path.exists():raise SystemExit(f'waiting for trainer scores: {score_path}')
 out.mkdir(parents=True,exist_ok=True); base=read_jsonl(hold/'samples.jsonl');scores=read_jsonl(score_path);episodes=read_jsonl(hold/'episodes.jsonl')
 if len(base)!=1049:raise ValueError(f'holdout rows changed: expected 1049, found {len(base)}')
 score_map={}
 for row in scores:
  k=key(row['episode_id'],row['t'])
  if k in score_map:raise ValueError(f'duplicate score row {k}')
  score_map[k]=row
 base_keys={key(r['episode_id'],r['t']) for r in base}
 missing=base_keys-set(score_map);extra=set(score_map)-base_keys
 if missing or extra:raise ValueError(f'score/base row mismatch missing={len(missing)}, extra={len(extra)}')
 byid=defaultdict(list)
 for row in base:
  score=score_map[key(row['episode_id'],row['t'])]
  row['weak_value']=find_val(score,a.value_key,'weak_discounted_value_uncalibrated','weak_demo_discounted_value')
  row['progress_value']=find_val(score,a.progress_key,'weak_demo_progress_uncalibrated','progress_uncalibrated','progress')
  row['score_provenance']=score
  if row['weak_value'] is None:raise ValueError(f'missing {a.value_key} for {row["episode_id"]}')
  byid[row['episode_id']].append(row)
 epmap={e['episode_id']:e for e in episodes}; allstats={};all_svg={};html_data={}
 progress_values=[r['progress_value'] for rows_ in byid.values() for r in rows_ if r['progress_value'] is not None]
 progress_bounds=(0.0,1.0) # both model heads are sigmoid outputs; raw-range zoom is separate
 for eid,rows in sorted(byid.items()):
  if eid not in epmap:raise ValueError(f'unknown holdout episode {eid}')
  rows.sort(key=lambda r:r['t']);times=[r['t'] for r in rows]
  vals=[r['weak_value'] for r in rows];smooth=centered3(vals);st=stats(vals)
  st.update({'episode_id':eid,'samples':len(rows),'duration_s':times[-1]-times[0],'human_control_fraction':sum(r.get('human_control_active') is True for r in rows)/len(rows),
             'operator_label':(epmap[eid].get('raw_metadata_outcome') or {}).get('label'),'human_assisted_episode':(epmap[eid].get('raw_metadata_outcome') or {}).get('human_assisted')})
  allstats[eid]={'weak_value':st}
  events=event_times(epmap[eid]); all_svg[(eid,'value')]=svg_chart(epmap[eid],a.value_key,'weak discounted value (uncalibrated)',vals,smooth,rows,events,False)
  all_svg[(eid,'value_zoom')]=svg_chart(epmap[eid],a.value_key,'weak discounted value — raw-range zoom',vals,smooth,rows,events,True)
  if all(r.get('progress_value') is not None for r in rows):
   pr=[r['progress_value'] for r in rows];ps=centered3(pr);allstats[eid]['native_progress']={'min':min(pr),'max':max(pr),'mean':sum(pr)/len(pr),'range':max(pr)-min(pr),'frame_delta_rmse':rmse(pr),'frame_delta_abs_max':stats(pr)['frame_delta_abs_max']}
   all_svg[(eid,'progress')]=svg_chart(epmap[eid],a.progress_key,'native progress head (uncalibrated)',pr,ps,rows,events,False,progress_bounds)
   all_svg[(eid,'progress_zoom')]=svg_chart(epmap[eid],a.progress_key,'native progress head — raw-range zoom',pr,ps,rows,events,True,progress_bounds)
  snapshot_times=transition_windows(times,events,times[-1])
  snaps=[]
  for snap in snapshot_times:
   r=rows[snap['index']]
   imgs=[]
   for cam,rel in zip(CAMERAS,r['images']):imgs.append({'camera':cam,'src':enc(hold/rel)})
   snaps.append({'t':r['t'],'label':snap['label'],'human_control_active':r.get('human_control_active'),'control_source_by_arm':r.get('control_source_by_arm'),'images':imgs})
  html_data[eid]={'episode_id':eid,'duration_s':times[-1],'times':times,'value':vals,'value_smooth':smooth,
    'progress':[r['progress_value'] for r in rows] if all(r.get('progress_value') is not None for r in rows) else None,'progress_bounds':progress_bounds,
    'progress_smooth':centered3([r['progress_value'] for r in rows]) if all(r.get('progress_value') is not None for r in rows) else None,
    'human':[bool(r.get('human_control_active')) for r in rows],
    'control_source_by_arm':[r.get('control_source_by_arm') for r in rows],
    'events':events,'snapshots':snaps,'raw_stats':st,
    'episode_outcome':(epmap[eid].get('raw_metadata_outcome') or {}).get('label'),
    'human_assisted_episode':(epmap[eid].get('raw_metadata_outcome') or {}).get('human_assisted')}
  basename=eid
  for kind in ('value','value_zoom','progress','progress_zoom'):
   if (eid,kind) in all_svg:(out/f'{basename}_{kind}.svg').write_text(all_svg[(eid,kind)],encoding='utf8')

 # Compact 3x2 overview SVG (all six individual raw traces plus centered mean).
 def overview_svg(metric='value'):
  W,H=1800,1180;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.raw{fill:none;stroke:#2878b5;stroke-width:1;opacity:.5}.smooth{fill:none;stroke:#cb4938;stroke-width:2.3}.human{fill:#a78bfa;opacity:.18}.axis{stroke:#444}</style>']
  parts.append(f'<text x="60" y="42" font-size="28" font-weight="bold">Fresh HITL external holdout — {html.escape(metric)}</text>')
  colors=['#2878b5','#d1495b','#00798c','#edae49','#6a4c93','#4f772d']
  if metric=='weak discounted value': ylo,yhi=0.0,1.0
  else: ylo,yhi=progress_bounds
  for j,(eid,data) in enumerate(sorted(html_data.items())):
   col=j%2;row=j//2;x0=80+col*850;y0=105+row*345;w=760;h=250
   raw=data['value'] if metric=='weak discounted value' else data['progress']
   sm=data['value_smooth'] if metric=='weak discounted value' else data['progress_smooth']
   if raw is None:continue
   ts=data['times'];
   parts.append(f'<text x="{x0}" y="{y0-14}" font-size="17" font-weight="bold">{eid} · operator {data["episode_outcome"]} · human-assisted={data["human_assisted_episode"]}</text>')
   for k in range(6):
    y=y0+h*(1-k/5);v=ylo+(yhi-ylo)*k/5;parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+w}" y2="{y:.2f}" class="grid"/><text x="{x0-12}" y="{y+5:.2f}" text-anchor="end" font-size="12">{v:.1f}</text>')
   for i,human in enumerate(data['human']):
    if human:
     xa=xmap(ts[i],ts[0],ts[-1],x0,w);xb=xmap(ts[i+1] if i+1<len(ts) else ts[i]+.2,ts[0],ts[-1],x0,w)
     parts.append(f'<rect x="{xa:.2f}" y="{y0}" width="{max(1,xb-xa):.2f}" height="{h}" class="human"/>')
   parts.append(f'<path d="{path(raw,ts,ylo,yhi,(x0,y0,w,h))}" class="raw" style="stroke:{colors[j%len(colors)]}"/>')
   parts.append(f'<path d="{path(sm,ts,ylo,yhi,(x0,y0,w,h))}" class="smooth" style="stroke:{colors[j%len(colors)]}"/>')
   parts.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>')
   for k in range(6):
    t=ts[0]+(ts[-1]-ts[0])*k/5;x=xmap(t,ts[0],ts[-1],x0,w)
    parts.append(f'<line x1="{x:.2f}" y1="{y0+h}" x2="{x:.2f}" y2="{y0+h+5}" class="axis"/><text x="{x:.2f}" y="{y0+h+20}" text-anchor="middle" font-size="12">{t:.1f}</text>')
   parts.append(f'<text x="{x0+w/2}" y="{y0+h+40}" text-anchor="middle" font-size="12">Elapsed time (s)</text>')
  parts.append(f'<text x="900" y="1155" text-anchor="middle" font-size="16">raw + centered 3-sample mean; shaded intervals mark per-frame human control. All values are uncalibrated.</text></svg>')
  return ''.join(parts)
 (out/'overview_weak_value.svg').write_text(overview_svg('weak discounted value'),encoding='utf8')
 if all(d['progress'] is not None for d in html_data.values()):(out/'overview_native_progress.svg').write_text(overview_svg('native progress head'),encoding='utf8')

 # Self-contained browser UI. Canvas traces include a main [0,1] view and a
 # separate raw-range zoom. Snapshot images are embedded as data URIs.
 html_obj={}
 for eid,data in html_data.items():
  html_obj[eid]=data
 embedded=json.dumps(html_obj,separators=(',',':')).replace('</','<\\/')
 page='''<!doctype html><html><head><meta charset="utf-8"><title>Fresh HITL critic rollouts</title><style>
body{font-family:system-ui,Arial;margin:20px;color:#202124;background:#fafafa}h1{margin-bottom:4px}small{color:#555}.controls{display:flex;gap:18px;align-items:center;margin:16px 0}.panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}canvas{width:100%;height:330px;display:block}#zoom{height:280px}.legend{display:flex;gap:18px;font-size:13px}.legend i{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:4px}.raw{background:#2878b5}.mean{background:#cb4938}.human{background:#a78bfa;height:12px!important}.snapgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}.snap{background:#fff;border:1px solid #ddd;padding:8px;border-radius:6px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}.views img{width:100%;height:auto}.hint{font-size:13px;color:#444}.tip{position:fixed;pointer-events:none;background:#111;color:#fff;padding:6px 9px;border-radius:4px;font-size:12px;display:none;z-index:99}</style></head><body>
<h1>Fresh HITL rollout scores</h1><div class="hint">Selected model scores on a frozen external holdout. Outputs are uncalibrated model values, not success probabilities. Raw 0–1 view and raw-range zoom are both shown; each centered 3-sample (~0.5 s) mean is diagnostic only. Shading uses per-frame human-control ownership, not whole-episode assistance.</div>
<div class="controls"><label>Episode <select id="ep"></select></label><label>Head <select id="head"><option value="value">Weak discounted value (uncalibrated)</option><option value="progress">Native progress head (uncalibrated)</option></select></label><span id="meta"></span></div>
<div class="legend"><span><i class="raw"></i>raw framewise output</span><span><i class="mean"></i>centered 3-sample mean</span><span><i class="human"></i>human control active</span><span>Vertical dashed line: recorded control-source change</span></div>
<div class="panel"><b>Main view (fixed [0,1] for both heads)</b><canvas id="main" width="1500" height="370"></canvas></div><div class="panel"><b>Raw-range zoom (same values; no per-episode normalization)</b><canvas id="zoom" width="1500" height="320"></canvas></div>
<div class="panel"><h2>Frame snapshots</h2><div class="hint">Start, midpoint, end, and frames immediately before/after recorded control boundaries; three cameras are embedded for offline inspection.</div><div id="snaps" class="snapgrid"></div></div><div id="tip" class="tip"></div>
<script>const D=__DATA__;const ep=document.getElementById('ep'),head=document.getElementById('head');for(const id of Object.keys(D).sort()){const o=document.createElement('option');o.value=id;o.textContent=id;ep.append(o)}
function draw(id){const d=D[id],metric=head.value==='value'?'value':'progress',raw=d[metric],mean=d[metric+'_smooth'];document.getElementById('meta').textContent=`${d.times.length} samples · ${d.duration_s.toFixed(1)} s · operator=${d.episode_outcome} · episode human-assisted=${d.human_assisted_episode}`;if(!raw){document.getElementById('meta').textContent+=' · progress unavailable';return}
 function plot(canvas,zoom){const c=document.getElementById(canvas),ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);const L=68,R=22,T=20,B=52,pw=W-L-R,ph=H-T-B,t0=d.times[0],t1=d.times[d.times.length-1],xmin=Math.min(...raw),xmax=Math.max(...raw),pad=Math.max((xmax-xmin)*.15,.002),bounds=metric==='value'?[0,1]:d.progress_bounds,ymin=zoom?xmin-pad:bounds[0],ymax=zoom?xmax+pad:bounds[1];if(ymax-ymin<.004){ymin-=.002;ymax+=.002}
 const X=t=>L+(t-t0)/(t1-t0||1)*pw,Y=v=>T+(ymax-v)/(ymax-ymin)*ph;
 for(let k=0;k<=5;k++){const v=ymin+(ymax-ymin)*k/5,y=Y(v);ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();ctx.fillStyle='#333';ctx.font='12px Arial';ctx.textAlign='right';ctx.fillText(v.toFixed(3),L-8,y+4)}
 for(let i=0;i<d.human.length;i++){if(d.human[i]){const x=X(d.times[i]),x2=X(d.times[Math.min(i+1,d.times.length-1)]+(i+1===d.times.length?.2:0));ctx.fillStyle='rgba(167,139,250,.18)';ctx.fillRect(x,T,Math.max(1,x2-x),ph)}}
 for(const e of d.events){if(e.t>=t0&&e.t<=t1){ctx.strokeStyle='#594299';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(X(e.t),T);ctx.lineTo(X(e.t),T+ph);ctx.stroke();ctx.setLineDash([])}}
 function line(vals,color,width,alpha){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.globalAlpha=alpha;vals.forEach((v,i)=>{const x=X(d.times[i]),y=Y(v);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();ctx.globalAlpha=1}
 line(raw,'#2878b5',1.4,.62);line(mean,'#cb4938',2.5,1);ctx.strokeStyle='#333';ctx.beginPath();ctx.moveTo(L,T+ph);ctx.lineTo(W-R,T+ph);ctx.stroke();ctx.textAlign='center';ctx.fillStyle='#333';ctx.font='12px Arial';for(let k=0;k<=5;k++){let t=t0+(t1-t0)*k/5;ctx.fillText(t.toFixed(1),X(t),H-26)}ctx.fillText('Elapsed time (s)',L+pw/2,H-5)
 c.onmousemove=ev=>{const rect=c.getBoundingClientRect(),x=(ev.clientX-rect.left)*W/rect.width,frac=(x-L)/pw,i=Math.max(0,Math.min(d.times.length-1,Math.round(frac*(d.times.length-1)))),tip=document.getElementById('tip');tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.textContent=`t=${d.times[i].toFixed(2)}s · raw=${raw[i].toFixed(5)} · 3-sample mean=${mean[i].toFixed(5)} · human_control=${d.human[i]} · owner=${JSON.stringify(d.control_source_by_arm[i])}`};c.onmouseleave=()=>document.getElementById('tip').style.display='none'}
 plot('main',false);plot('zoom',true);const box=document.getElementById('snaps');box.replaceChildren();for(const s of d.snapshots){const card=document.createElement('div');card.className='snap';let owner=JSON.stringify(s.control_source_by_arm);card.innerHTML=`<b>${s.label}</b> · t=${s.t.toFixed(2)}s · human=${s.human_control_active} · ${owner}`;const views=document.createElement('div');views.className='views';for(const im of s.images){const el=document.createElement('img');el.src=im.src;el.alt=im.camera;el.title=im.camera;views.append(el)}card.append(views);box.append(card)}}
ep.onchange=()=>draw(ep.value);head.onchange=()=>draw(ep.value);draw(ep.value);</script></body></html>'''.replace('__DATA__',embedded)
 (out/'index.html').write_text(page,encoding='utf8')
 summary={'schema_version':1,'source_scores':str(score_path),'source_holdout':str(hold),'episodes':len(html_data),'rows':len(base),'metric_semantics':'uncalibrated model output; no probability interpretation','smoothing':'centered three-sample mean, approximately 0.5 seconds, diagnostic only','noise_interpretation':'Frame-to-frame score deltas quantify temporal variation, not ground-truth measurement noise.','summary_by_episode':allstats}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps({'output':str(out),'episodes':len(html_data),'rows':len(base),'files':len(list(out.iterdir())),'metrics':allstats},indent=2))
if __name__=='__main__':main()
