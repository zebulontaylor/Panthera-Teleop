#!/usr/bin/env python3
"""Compare frozen fresh-HITL outputs from critic v1/v4b; descriptive only."""
from __future__ import annotations
import argparse, base64, hashlib, html, json, math
from collections import defaultdict
from pathlib import Path

CAMERAS=('overhead','wrist_port2','wrist_port3')
V1_KEY='weak_demo_discounted_value_uncalibrated'
V4B_KEY='state_goodness_uncalibrated'

def read_jsonl(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(r): return str(r['episode_id']),round(float(r['t']),6)
def avg_abs_delta(vals): return sum(abs(vals[i]-vals[i-1]) for i in range(1,len(vals)))/(len(vals)-1) if len(vals)>1 else 0.0
def stats(vals):
 d=[vals[i]-vals[i-1] for i in range(1,len(vals))]
 return {'min':min(vals),'max':max(vals),'mean':sum(vals)/len(vals),'range':max(vals)-min(vals),'frame_delta_rms':math.sqrt(sum(x*x for x in d)/len(d)) if d else 0,'frame_delta_abs_max':max(map(abs,d)) if d else 0}
def enc(path): return 'data:image/jpeg;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
def xmap(t,t0,t1,x,w): return x+(float(t)-t0)/(t1-t0)*w if t1>t0 else x+w/2
def ymap(v,lo,hi,y,h): return y+h-(v-lo)/(hi-lo)*h
def pth(vals,times,lo,hi,x,y,w,h):
 return ' '.join(('M' if i==0 else 'L')+f'{xmap(t,times[0],times[-1],x,w):.2f},{ymap(v,lo,hi,y,h):.2f}' for i,(t,v) in enumerate(zip(times,vals)))
def transition_events(ep):
 start=ep.get('start_data_monotonic_ns') or ep.get('start_monotonic_ns')
 if not isinstance(start,(int,float)): return []
 res=[]; waiting=False
 for e in ep.get('control_transitions') or []:
  n=e.get('monotonic_ns'); src=e.get('source')
  if not isinstance(n,(int,float)):continue
  if src=='human':res.append({'t':(n-start)/1e9,'label':'human takeover'});waiting=True
  elif waiting and src in ('hold','policy'):res.append({'t':(n-start)/1e9,'label':'human release → '+str(src)});waiting=False
 return res
def pick_snaps(rows,events):
 ts=[r['t'] for r in rows]; targets=[(ts[0],'start'),((ts[0]+ts[-1])/2,'midpoint'),(ts[-1],'end')]
 for e in events:
  for off,lab in [(-.2,'before '+e['label']),(.2,'after '+e['label'])]:targets.append((max(ts[0],min(ts[-1],e['t']+off)),lab))
 got=[]
 for t,lab in sorted(targets):
  i=min(range(len(ts)),key=lambda j:abs(ts[j]-t))
  if not any(x['i']==i for x in got):got.append({'i':i,'label':lab})
 return got

def svg_episode(eid,rows,events):
 W,H=1500,700; x,y,w,h=90,100,1350,480; ts=[r['t'] for r in rows]; v1=[r['v1'] for r in rows];v4b=[r['v4b'] for r in rows]
 q=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#333}.v1{fill:none;stroke:#2878b5;stroke-width:1.4;opacity:.78}.v4b{fill:none;stroke:#e07a28;stroke-width:1.6;opacity:.9}.human{fill:#a78bfa;opacity:.2}.event{stroke:#594299;stroke-dasharray:5 4}</style>',f'<text x="70" y="40" font-size="26" font-weight="bold">{html.escape(eid)} · fixed [0,1] raw outputs</text>',f'<text x="70" y="68" font-size="14">Fresh Oct29 diagnostic holdout · 5 Hz · wrists-only v4b (overhead masked) · uncalibrated scores</text>']
 for k in range(6):
  val=k/5; yy=ymap(val,0,1,y,h);q.append(f'<line x1="{x}" y1="{yy:.2f}" x2="{x+w}" y2="{yy:.2f}" class="grid"/><text x="{x-10}" y="{yy+4:.2f}" text-anchor="end" font-size="13">{val:.1f}</text>')
 for i,r in enumerate(rows):
  if r.get('human_control_active') is True:
   a=xmap(r['t'],ts[0],ts[-1],x,w); b=xmap(rows[i+1]['t'] if i+1<len(rows) else ts[-1]+.2,ts[0],ts[-1],x,w)
   q.append(f'<rect x="{a:.2f}" y="{y}" width="{max(1,b-a):.2f}" height="{h}" class="human"/>')
 for ev in events:
  if ts[0]<=ev['t']<=ts[-1]:
   xx=xmap(ev['t'],ts[0],ts[-1],x,w);q.append(f'<line x1="{xx:.2f}" y1="{y}" x2="{xx:.2f}" y2="{y+h}" class="event"/>')
 q.extend([f'<path d="{pth(v1,ts,0,1,x,y,w,h)}" class="v1"/>',f'<path d="{pth(v4b,ts,0,1,x,y,w,h)}" class="v4b"/>',f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" class="axis"/>'])
 for k in range(6):
  t=ts[0]+(ts[-1]-ts[0])*k/5;xx=xmap(t,ts[0],ts[-1],x,w);q.append(f'<text x="{xx:.2f}" y="{y+h+22}" text-anchor="middle" font-size="13">{t:.1f}</text>')
 q.extend([f'<text x="{x+w/2}" y="{y+h+49}" text-anchor="middle" font-size="14">Elapsed time (s)</text>',f'<line x1="{x}" y1="660" x2="{x+42}" y2="660" class="v1"/><text x="{x+52}" y="665" font-size="14">Previous selected model (v1)</text>',f'<line x1="{x+365}" y1="660" x2="{x+407}" y2="660" class="v4b"/><text x="{x+417}" y="665" font-size="14">New selected model (v4b)</text>',f'<rect x="{x+760}" y="650" width="24" height="15" class="human"/><text x="{x+793}" y="664" font-size="14">Per-frame human control</text>',f'<line x1="{x+1060}" y1="660" x2="{x+1090}" y2="660" class="event"/><text x="{x+1100}" y="665" font-size="14">ownership boundary</text>','</svg>'])
 return ''.join(q)

def scatter_svg(rows,title,episode_facets=False):
 W,H=900,760;x,y,w,h=105,95,720,535
 q=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#333}.diag{stroke:#555;stroke-dasharray:6 5}.policy{fill:#2878b5;opacity:.38}.human{fill:#8b5cf6;opacity:.5}</style>',f'<text x="50" y="34" font-size="22" font-weight="bold">{html.escape(title)}</text>','<circle cx="120" cy="61" r="5" class="policy"/><text x="132" y="66" font-size="13">policy-owned samples</text>','<circle cx="320" cy="61" r="5" class="human"/><text x="332" y="66" font-size="13">human-controlled samples</text>']
 for k in range(6):
  v=k/5;xx=x+w*v;yy=y+h-h*v
  q += [f'<line x1="{xx:.2f}" y1="{y}" x2="{xx:.2f}" y2="{y+h}" class="grid"/>',f'<line x1="{x}" y1="{yy:.2f}" x2="{x+w}" y2="{yy:.2f}" class="grid"/>',f'<text x="{xx:.2f}" y="{y+h+22}" text-anchor="middle" font-size="13">{v:.1f}</text>',f'<text x="{x-10}" y="{yy+4:.2f}" text-anchor="end" font-size="13">{v:.1f}</text>']
 q += [f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y}" class="diag"/>',f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" class="axis"/>',f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" class="axis"/>']
 for r in rows:
  xx=x+w*r['v1'];yy=y+h-h*r['v4b'];cl='human' if r.get('human_control_active') is True else 'policy'
  q.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="2.2" class="{cl}"/>')
 q += [f'<text x="{x+w/2}" y="{y+h+52}" text-anchor="middle" font-size="15">Previous model v1 raw score</text>',f'<text transform="translate(30 {y+h/2}) rotate(-90)" text-anchor="middle" font-size="15">New model v4b raw score</text>','</svg>']
 return ''.join(q)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--holdout',type=Path,default=Path('work/state-value-20261029/fresh-hitl-holdout'));ap.add_argument('--scores-v1',type=Path,default=Path('work/state-value-20261029/final-evaluation/fresh_test_rollout_scores.jsonl'));ap.add_argument('--scores-v4b',type=Path,default=Path('work/state-value-v2-20261029/final-evaluation/fresh_test_rollout_v4b_scores.jsonl'));ap.add_argument('--out',type=Path,default=Path('work/state-value-v2-20261029/plots-wrists-only-v4b'));ap.add_argument('--v4b-key',default=V4B_KEY);ap.add_argument('--v4b-checkpoint-sha256',default='pending')
 a=ap.parse_args(); hold=a.holdout.resolve();p1=a.scores_v1.resolve();p2=a.scores_v4b.resolve();out=a.out.resolve()
 for p in (p1,p2):
  if not p.exists():raise SystemExit(f'waiting for score file: {p}')
 base=read_jsonl(hold/'samples.jsonl'); episodes=read_jsonl(hold/'episodes.jsonl');s1=read_jsonl(p1);s2=read_jsonl(p2)
 if len(base)!=1049:raise ValueError(f'expected unchanged 1049-row holdout, got {len(base)}')
 def make_map(rows,label,keyname):
  d={}
  for r in rows:
   k=key(r)
   if k in d:raise ValueError(f'duplicate {label} score key {k}')
   if keyname not in r:raise ValueError(f'missing {keyname} in {label} score row')
   d[k]=float(r[keyname])
  return d
 m1=make_map(s1,'v1',V1_KEY);m2=make_map(s2,'v4b',a.v4b_key);keys={key(r) for r in base}
 if keys!=set(m1):raise ValueError(f'v1 join mismatch missing={len(keys-set(m1))}, extra={len(set(m1)-keys)}')
 if keys!=set(m2):raise ValueError(f'v4b join mismatch missing={len(keys-set(m2))}, extra={len(set(m2)-keys)}')
 out.mkdir(parents=True,exist_ok=True);by=defaultdict(list)
 for r in base:
  k=key(r);by[r['episode_id']].append({**r,'v1':m1[k],'v4b':m2[k]})
 epmap={e['episode_id']:e for e in episodes};sums={};webdata={}
 for eid,rows in sorted(by.items()):
  rows.sort(key=lambda r:r['t']);times=[r['t'] for r in rows];events=transition_events(epmap[eid]);snaps=[]
  for ss in pick_snaps(rows,events):
   r=rows[ss['i']];snaps.append({'t':r['t'],'label':ss['label'],'human':r.get('human_control_active'),'owners':r.get('control_source_by_arm'),'images':[{'camera':cam,'src':enc(hold/rel)} for cam,rel in zip(CAMERAS,r['images'])]})
  sums[eid]={'rows':len(rows),'duration_s':times[-1]-times[0],'v1':stats([r['v1'] for r in rows]),'v4b':stats([r['v4b'] for r in rows]),'v1_v4b_abs_delta_mean':sum(abs(r['v1']-r['v4b']) for r in rows)/len(rows),'v4b_minus_v1_mean':sum(r['v4b']-r['v1'] for r in rows)/len(rows),'human_control_fraction':sum(r.get('human_control_active') is True for r in rows)/len(rows),'metadata_outcome':(epmap[eid].get('raw_metadata_outcome') or {}).get('label')}
  (out/f'{eid}_compare.svg').write_text(svg_episode(eid,rows,events))
  (out/f'{eid}_scatter.svg').write_text(scatter_svg(rows,f'{eid} · paired raw scores (fixed [0,1])'))
  webdata[eid]={'times':times,'v1':[r['v1'] for r in rows],'v4b':[r['v4b'] for r in rows],'human':[r.get('human_control_active') is True for r in rows],'owner':[r.get('control_source_by_arm') for r in rows],'events':events,'snaps':snaps,'metadata_outcome':(epmap[eid].get('raw_metadata_outcome') or {}).get('label')}
 # SVG overview (all episodes), fixed shared [0,1] and seconds ticks per panel.
 W,H=1800,1180;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#444}.v1{fill:none;stroke:#2878b5;stroke-width:1.3;opacity:.8}.v4b{fill:none;stroke:#e07a28;stroke-width:1.4;opacity:.86}.human{fill:#a78bfa;opacity:.18}</style>','<text x="60" y="42" font-size="28" font-weight="bold">Fresh HITL diagnostic holdout · v1 vs wrists-only v4b raw outputs</text>']
 parts.extend(['<line x1="65" y1="72" x2="105" y2="72" class="v1"/><text x="115" y="77" font-size="14">v1 previous selected model</text>','<line x1="360" y1="72" x2="400" y2="72" class="v4b"/><text x="410" y="77" font-size="14">v4b new selected model</text>','<rect x="690" y="63" width="24" height="14" class="human"/><text x="722" y="76" font-size="14">per-frame human control</text>'])
 for j,(eid,d) in enumerate(sorted(webdata.items())):
  col=j%2;rr=j//2;x=85+col*850;y=125+rr*345;w=760;h=250;ts=d['times'];parts.append(f'<text x="{x}" y="{y-14}" font-size="16" font-weight="bold">{eid} · metadata={d["metadata_outcome"]}</text>')
  for k in range(6):
   val=k/5;yy=ymap(val,0,1,y,h);parts.append(f'<line x1="{x}" y1="{yy:.2f}" x2="{x+w}" y2="{yy:.2f}" class="grid"/><text x="{x-12}" y="{yy+4:.2f}" text-anchor="end" font-size="12">{val:.1f}</text>')
  for i,human in enumerate(d['human']):
   if human:
    a0=xmap(ts[i],ts[0],ts[-1],x,w);a1=xmap(ts[i+1] if i+1<len(ts) else ts[i]+.2,ts[0],ts[-1],x,w);parts.append(f'<rect x="{a0:.2f}" y="{y}" width="{max(1,a1-a0):.2f}" height="{h}" class="human"/>')
  parts += [f'<path d="{pth(d["v1"],ts,0,1,x,y,w,h)}" class="v1"/>',f'<path d="{pth(d["v4b"],ts,0,1,x,y,w,h)}" class="v4b"/>',f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" class="axis"/>']
  for k in range(6):
   t=ts[0]+(ts[-1]-ts[0])*k/5;xx=xmap(t,ts[0],ts[-1],x,w);parts.append(f'<text x="{xx:.2f}" y="{y+h+18}" text-anchor="middle" font-size="11">{t:.1f}</text>')
  parts.append(f'<text x="{x+w/2}" y="{y+h+36}" text-anchor="middle" font-size="11">Elapsed time (s)</text>')
 parts.append('<text x="900" y="1155" text-anchor="middle" font-size="15">Unsmoothed raw outputs, fixed [0,1]; lavender = per-frame human control. Outputs are uncalibrated and not task-success probabilities.</text></svg>')
 (out/'overview_v1_vs_v4b.svg').write_text(''.join(parts))
 all_rows=[r for rr in by.values() for r in rr]
 (out/'paired_raw_scatter.svg').write_text(scatter_svg(all_rows,'All held-out samples · v1 vs v4b paired raw scores (fixed [0,1])'))
 # Fully offline HTML: common timeline, exact cursor hover, model selector is unnecessary as both always shown.
 payload=json.dumps(webdata,separators=(',',':')).replace('</','<\\/')
 page='''<!doctype html><html><head><meta charset="utf-8"><title>Fresh HITL v1 vs v4b</title><style>body{font:15px system-ui;margin:20px;background:#fafafa;color:#222}.hint{max-width:1100px}.panel,.card{background:white;border:1px solid #ddd;border-radius:7px;padding:12px;margin:12px 0}canvas{width:100%;height:410px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}.views img{width:100%;height:auto}.tip{position:fixed;display:none;background:#111;color:white;padding:8px;pointer-events:none}.sw{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:5px}.views img{border:1px solid #ddd}</style></head><body><h1>Fresh HITL diagnostic rollout scores · v1 vs v4b</h1><p class="hint">Same frozen Oct29 diagnostic holdout (previously exposed in earlier audits) and exact timestamp join. The v4b critic receives wrist views only; overhead pixels are masked. Both traces are unsmoothed and fixed to [0,1]. Outputs are uncalibrated model scores, not success probabilities or verified task quality. Lavender shading is per-frame human ownership; dashed lines show recorded human takeover/release boundaries.</p><label>Episode <select id="ep"></select></label> <span id="meta"></span><p><span class="sw" style="background:#2878b5"></span>previous v1 <span class="sw" style="background:#e07a28"></span>new v4b <span class="sw" style="background:#a78bfa;height:12px"></span>human control</p><div class="panel"><canvas id="c" width="1500" height="440"></canvas></div><div class="panel"><h2>Selected frames</h2><div id="snap" class="grid"></div></div><div id="tip" class="tip"></div><script>const D=__DATA__,sel=document.getElementById('ep');for(const id of Object.keys(D).sort()){let o=document.createElement('option');o.value=id;o.textContent=id;sel.append(o)}function draw(id){const d=D[id],c=document.getElementById('c'),x=c.getContext('2d'),W=c.width,H=c.height,L=70,R=25,T=20,B=55,w=W-L-R,h=H-T-B,ts=d.times,t0=ts[0],t1=ts[ts.length-1],X=t=>L+(t-t0)/(t1-t0||1)*w,Y=v=>T+(1-v)*h;document.getElementById('meta').textContent=`${ts.length} rows · ${t1.toFixed(1)}s · operator metadata=${d.metadata_outcome}`;x.clearRect(0,0,W,H);for(let k=0;k<=5;k++){let v=k/5,y=Y(v);x.strokeStyle='#ddd';x.beginPath();x.moveTo(L,y);x.lineTo(W-R,y);x.stroke();x.fillStyle='#333';x.textAlign='right';x.fillText(v.toFixed(1),L-8,y+4)}for(let i=0;i<d.human.length;i++)if(d.human[i]){let a=X(ts[i]),b=X(ts[Math.min(i+1,ts.length-1)]+(i===ts.length-1?.2:0));x.fillStyle='rgba(167,139,250,.2)';x.fillRect(a,T,Math.max(1,b-a),h)}for(const e of d.events)if(e.t>=t0&&e.t<=t1){x.strokeStyle='#594299';x.setLineDash([5,4]);x.beginPath();x.moveTo(X(e.t),T);x.lineTo(X(e.t),T+h);x.stroke();x.setLineDash([])}function line(v,color){x.beginPath();x.strokeStyle=color;x.lineWidth=1.6;v.forEach((q,i)=>{if(i===0)x.moveTo(X(ts[i]),Y(q));else x.lineTo(X(ts[i]),Y(q))});x.stroke()}line(d.v1,'#2878b5');line(d.v4b,'#e07a28');x.strokeStyle='#333';x.beginPath();x.moveTo(L,T+h);x.lineTo(W-R,T+h);x.stroke();x.textAlign='center';for(let k=0;k<=5;k++){let t=t0+(t1-t0)*k/5;x.fillText(t.toFixed(1),X(t),H-28)}x.fillText('Elapsed time (s)',L+w/2,H-7);c.onmousemove=e=>{let rect=c.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,ix=Math.max(0,Math.min(ts.length-1,Math.round((px-L)/w*(ts.length-1)))),tip=document.getElementById('tip');tip.style.display='block';tip.style.left=(e.clientX+10)+'px';tip.style.top=(e.clientY+10)+'px';tip.textContent=`t=${ts[ix].toFixed(2)}s · v1=${d.v1[ix].toFixed(5)} · v4b=${d.v4b[ix].toFixed(5)} · human=${d.human[ix]} · owners=${JSON.stringify(d.owner[ix])}`};c.onmouseleave=()=>document.getElementById('tip').style.display='none';let box=document.getElementById('snap');box.replaceChildren();for(const s of d.snaps){let card=document.createElement('div');card.className='card';card.innerHTML=`<b>${s.label}</b> · ${s.t.toFixed(2)}s · human=${s.human} · owners=${JSON.stringify(s.owners)}`;let views=document.createElement('div');views.className='views';for(const im of s.images){let z=document.createElement('img');z.src=im.src;z.alt=im.camera;z.title=im.camera;views.append(z)}card.append(views);box.append(card)}}sel.onchange=()=>draw(sel.value);draw(sel.value)</script></body></html>'''.replace('__DATA__',payload)
 (out/'index.html').write_text(page)
 summary={'episodes':len(by),'rows':len(base),'v1_scores':str(p1),'v1_scores_sha256':file_sha(p1),'v4b_scores':str(p2),'v4b_scores_sha256':file_sha(p2),'v1_key':V1_KEY,'v4b_key':a.v4b_key,'v4b_checkpoint_sha256':a.v4b_checkpoint_sha256,'model_input_mode':'wrists-only (overhead pixels structurally ignored); displayed camera snapshots are original images','join':'exact (episode_id, round(t,6)); all three manifests must match','scale':'fixed [0,1] for both unsmoothed heads; no per-episode normalization','semantics':'Descriptive previously-exposed Oct29 diagnostic inference only; uncalibrated outputs are not task success probabilities or a blind final test. All current metadata outcomes are review. Wrists-only scoring: overhead image pixels are masked from the model input; display snapshots show the original camera views.','human_ownership':'Per-sample human_control_active shading and recorded transition boundaries; not causal-effect evidence.','summary_by_episode':sums,'paired_score_interpretation':'Raw model score pair agreement only; scores are not calibrated probabilities or independent observations. Samples are temporally correlated.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'README.md').write_text(f"V1 vs wrists-only v4b raw score comparison on the unchanged six-episode external HITL holdout. Selected checkpoint SHA256: {a.v4b_checkpoint_sha256}. V4b score file SHA256: {file_sha(p2)}. Model input was wrists-only; overhead pixels were masked from the critic, though displayed snapshots are original camera frames. Exact (episode_id,t) join; no score interpolation. Both unsmoothed traces share the fixed [0,1] scale. Use HTML for per-frame hover and camera snapshots at start/mid/end and ownership transitions. This previously exposed diagnostic holdout was excluded from v4b training/selection; prior model-history audits mean it is not a blind final test. Model scores are uncalibrated and do not establish causal quality or task success.\n")
 print(json.dumps({'output':str(out),'episodes':len(by),'rows':len(base),'files':len(list(out.iterdir())),'v4b_checkpoint_sha256':a.v4b_checkpoint_sha256},indent=2))
if __name__=='__main__':main()
