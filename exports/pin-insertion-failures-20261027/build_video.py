from pathlib import Path
import json, math, bisect, subprocess
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FPS=20
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(FONT,23); small=ImageFont.truetype(FONT,18)
W,H=1440,800
filename=OUT/'pin-insertion-failures-two-latest-attempts.mp4'
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(filename)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
manifest={'video':filename.name,'fps':FPS,'timing':'Host receive timestamps within active segments; holds omitted; original observed motion speed. Camera views are approximately synchronized.','attempts':[]}
count=0
for num,(identifier,first_seg,label) in enumerate([('20261027T155447-6d9ceba2',19,'10:54 AM CDT'),('20261027T155906-04975fdb',13,'10:59 AM CDT')],1):
 d=ROOT/'data/hitl/review'/identifier
 states=[json.loads(x) for x in (d/'states.jsonl').read_text().splitlines()]
 frames=[json.loads(x) for x in (d/'frames.jsonl').read_text().splitlines()]
 segments=sorted({s['recording_segment'] for s in states if s['recording_segment']>=first_seg})
 cams=['wrist_port2','wrist_port3','overhead']; cached={}; start_count=count; seginfo=[]
 def render(t,seg):
  canvas=Image.new('RGB',(W,H),(17,22,30)); draw=ImageDraw.Draw(canvas)
  draw.text((20,10),f'PIN INSERTION | Attempt {num}/2 | Oct 27, 2026 - {label}',font=font,fill='white')
  draw.text((20,43),'Observed motion speed | Pauses removed | Two latest full attempts',font=small,fill='#a9bccb')
  for cam,box in zip(cams,[(0,80,960,720),(960,80,480,360),(960,440,480,360)]):
   fs=[f for f in frames if f['camera']==cam and f['recording_segment']==seg]
   if not fs: fs=[f for f in frames if f['camera']==cam]
   times=[f['monotonic_ns']/1e9 for f in fs]; ix=max(0,bisect.bisect_right(times,t)-1); f=fs[ix]; path=d/f['path']
   if path not in cached:cached[path]=Image.open(path).convert('RGB')
   x,y,w,h=box;canvas.paste(cached[path].resize((w,h),Image.Resampling.LANCZOS),(x,y))
   draw=ImageDraw.Draw(canvas);name={'wrist_port2':'Insertion wrist','wrist_port3':'Opposite wrist','overhead':'Overhead'}[cam]
   draw.rectangle((x,y,x+w,y+30),fill=(15,20,28));draw.text((x+10,y+3),name,font=small,fill='white')
  return canvas
 for j,seg in enumerate(segments):
  ss=[s for s in states if s['recording_segment']==seg]; a=ss[0]['monotonic_ns']/1e9;b=ss[-1]['monotonic_ns']/1e9
  seginfo.append({'segment':seg,'control_source':ss[0]['control_source'],'data_time_start_s':ss[0]['data_time_s'],'data_time_end_s':ss[-1]['data_time_s'],'host_episode_start_s':ss[0]['episode_time_s'],'host_episode_end_s':ss[-1]['episode_time_s']})
  if j==0:
   im=render(a,seg);im.save(OUT/f'attempt-{num}-preview.jpg')
   for _ in range(FPS):p.stdin.write(im.tobytes());count+=1
  for k in range(max(1,math.ceil((b-a)*FPS))):
   im=render(min(b,a+k/FPS),seg);p.stdin.write(im.tobytes());count+=1
 for _ in range(FPS):p.stdin.write(im.tobytes());count+=1
 manifest['attempts'].append({'source_id':identifier,'video_start_s':start_count/FPS,'video_end_s':count/FPS,'segments':seginfo})
p.stdin.close();assert p.wait()==0
manifest['duration_s']=count/FPS
(OUT/'clip-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'path':str(filename),'duration_s':count/FPS,'bytes':filename.stat().st_size}))
