#!/usr/bin/env python3
"""Novel CAD contact-case heatmaps; explicitly not a heldout real demonstration."""
import argparse,json,hashlib,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import cv2,numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fps',type=float,default=2)
    a=p.parse_args()
    import torch,imageio_ffmpeg
    from app.pin_insertion_cnn import PinInsertionPredictor
    from render_pin_insertion_heatmaps import infer,compose_frame
    torch.set_num_threads(4);predictor=PinInsertionPredictor(a.checkpoint)
    x=np.load(a.dataset,allow_pickle=False);metadata=json.loads(str(x['metadata_json']))
    if not metadata.get('diagnostic_only'):raise ValueError('Diagnostic-only dataset required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();process=subprocess.Popen([ffmpeg,'-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x506','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(a.output)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    reports=[]
    try:
        for i in range(len(x['images'])):
            out=infer(predictor,x['images'][i],x['state'][i]);category=str(x['category'][i]);visibility=str(x['visibility_category'][i])
            subtitle=f"{category} | {visibility} | CAD contact fraction {x['contact_support_fraction'][i]:.2f} | independent case {i+1}/{len(x['images'])}"
            frame=compose_frame(x['images'][i],out,'SIM contact geometry diagnostic / CAD grasp prior / NO DYNAMICS',subtitle,x['keypoints'][i],True)
            cv2.rectangle(frame,(0,470),(1440,506),(0,0,0),-1)
            cv2.putText(frame,'Novel renders from realTRAIN seeds. Green: CAD geometry references, NOT independent real labels.',(12,494),cv2.FONT_HERSHEY_SIMPLEX,.62,(230,230,230),1,cv2.LINE_AA)
            process.stdin.write(frame.tobytes())
            if i==0:cv2.imwrite(str(a.output.with_suffix('.preview.jpg')),frame)
            reports.append(dict(index=i,category=category,hole_visibility=visibility,source_id=str(x['source_id'][i]),predicted_keypoints=np.asarray(out['keypoints']).tolist()))
        process.stdin.close();stderr=process.stderr.read();status=process.wait()
        if status:raise RuntimeError(stderr.decode()[-2000:])
    finally:
        if process.poll() is None:process.kill()
    report=dict(kind='novel_sim_geometry_diagnostic_not_heldout_real_test',training_use=False,checkpoint=str(a.checkpoint),dataset=str(a.dataset),frames=reports,dataset_metadata=metadata,
        caveats=['Independent geometric cases, not a dynamical trajectory. Pin is arm-supported with geometric cap/plate contact only.',
                 'Fitted real grasp rejected by jaw geometry; fixed CAD fingertip grasp used instead. Camera/base/carrier registration remain hypotheses.',
                 'No motion executed; CNN localization and gates are advisory/unvalidated. Sources are realTRAIN seeds, so this is not independent heldout evaluation.'])
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
