#!/usr/bin/env python3
"""Evaluate frozen grasp candidate first, then fixed train-only negative refit.

Supplemental labels are manual paired-camera temporal-context image judgments;
encoder closure only retrieved candidates and does not determine their labels.
"""
from pathlib import Path
import json
import numpy as np
import torch
import validate_visual_grasp as g

ROOT=g.ROOT


def run():
    torch.set_num_threads(2);torch.manual_seed(17)
    out=ROOT/'logs/alignment-validation'
    label_path=ROOT/'data/calibration/manual-labels/supplemental-closed-empty.json'
    data=json.loads(label_path.read_text())
    original,excluded,episodes=g.load_rows(ROOT/'data/calibration/manual-labels/assembled/labels.json')
    if data['source_labels_sha256']!=g.sha256(ROOT/'data/calibration/manual-labels/assembled/labels.json'):
        raise ValueError('Original label source changed')
    supp=[];seen=set();image_splits={r['images']['wrist_port3']['sha256']:r['split'] for r in original}
    for r in data['observations']:
        if r['episode_id'] not in episodes or episodes[r['episode_id']]!=r['split']:
            raise ValueError('Supplemental episode leakage')
        if r['observation_id'] in seen: raise ValueError('Duplicate supplement')
        seen.add(r['observation_id'])
        for camera,image in r['images'].items():
            if g.sha256(image['path'])!=image['sha256']: raise ValueError('Changed midpoint image')
            if image!=r['context'][1]['images'][camera]: raise ValueError('Midpoint/context mismatch')
        for ctx in r['context']:
            for camera,image in ctx['images'].items():
                if g.sha256(image['path'])!=image['sha256']:raise ValueError('Changed image')
        for name,sha in r['source_hashes'].items():
            if g.sha256(ROOT/'data/kept'/r['episode_id']/name)!=sha:raise ValueError('Changed recording')
        h=r['images']['wrist_port3']['sha256']
        if h in image_splits: raise ValueError('Duplicate original/supplemental image')
        image_splits[h]=r['split']
        label=r['labels']['grasp']['left']
        if label['object'] not in g.CLASSES or label['confidence'] not in ('high','medium'):continue
        supp.append(dict(r,target=g.CLASSES.index(label['object'])))
    model_root=ROOT.parent/'RobotArmTraining/outputs/panthera-assembly-cnns'
    checkpoint=model_root/'gear_alignment/real_v4/adapted.pt'
    x,_=g.extract_features(supp,checkpoint,model_root,'cuda')
    original_report=json.loads((out/'grasp-report.json').read_text())
    if original_report['provenance']['encoder_sha256']!=g.sha256(checkpoint):
        raise ValueError('Original frozen encoder changed')
    frozen=dict(np.load(out/'grasp-candidate.npz'))
    frozen_hash=g.sha256(out/'grasp-candidate.npz')
    y=np.asarray([r['target'] for r in supp]);train=np.asarray([r['split']=='train' for r in supp])
    frozen_prob=g.predict(frozen,x)
    def predrows(name,rows,prob):
        return [{'model':name,'observation_id':r['observation_id'],'episode_id':r['episode_id'],
                 'split':r['split'],'truth':g.CLASSES[r['target']],
                 'probabilities':dict(zip(g.CLASSES,p.tolist())),
                 'predicted':g.CLASSES[int(p.argmax())],'accepted':bool(p.max()>=g.THRESHOLD)}
                for r,p in zip(rows,prob)]
    frozen_result={'all':g.metrics(y,frozen_prob),'train':g.metrics(y[train],frozen_prob[train]),
                   'validation':g.metrics(y[~train],frozen_prob[~train]),
                   'predictions':predrows('original_frozen',supp,frozen_prob),
                   'frozen_candidate_sha256':frozen_hash,'supplemental_labels_sha256':g.sha256(label_path)}
    # This artifact is written before any supplemental refit.
    (out/'grasp-supplement-frozen.json').write_text(json.dumps(frozen_result,indent=2)+'\n')
    ox,_=g.extract_features(original,checkpoint,model_root,'cuda')
    oy=np.asarray([r['target'] for r in original]);otrain=np.asarray([r['split']=='train' for r in original])
    augmented=g.fit_classifier(np.concatenate((ox[otrain],x[train])),np.concatenate((oy[otrain],y[train])))
    op=g.predict(augmented,ox);sp=g.predict(augmented,x)
    def scalar(rows):return np.asarray([[r['measured_grippers']['left']['position']] for r in rows])
    os,ss=scalar(original),scalar(supp)
    oldbase=g.fit_classifier(os[otrain],oy[otrain]);newbase=g.fit_classifier(np.concatenate((os[otrain],ss[train])),np.concatenate((oy[otrain],y[train])))
    results={'original_frozen_candidate':frozen_result,
       'supplemented_visual_candidate':{'original_validation':g.metrics(oy[~otrain],op[~otrain]),
          'supplemental_validation':g.metrics(y[~train],sp[~train]),
          'combined_validation':g.metrics(np.concatenate((oy[~otrain],y[~train])),np.concatenate((op[~otrain],sp[~train]))),
          'by_validation_episode':{}},
       'encoder_baseline_original_frozen_on_supplement':g.metrics(y[~train],g.predict(oldbase,ss[~train])),
       'encoder_baseline_supplemented':{'original_validation':g.metrics(oy[~otrain],g.predict(newbase,os[~otrain])),
         'supplemental_validation':g.metrics(y[~train],g.predict(newbase,ss[~train]))}}
    for ep in sorted(ep for ep,s in episodes.items() if s=='validation'):
        mask=np.asarray([r['episode_id']==ep for r in supp]);results['supplemented_visual_candidate']['by_validation_episode'][ep]=g.metrics(y[mask],sp[mask])
    predictions=predrows('supplemented',original,op)+predrows('supplemented',supp,sp)
    report={'schema_version':1,'offline_only':True,'runtime_motion_authorized':False,
      'grasp_safety_validation_passed':False,
      'method':'First evaluate byte-identical original frozen candidate. Then refit same image features, crop, L2=.01 and threshold=.85 on original49train + supplemental13train only. Same 4 validation episodes kept separate; no model selection.',
      'source_labels_sha256':g.sha256(label_path),'source_original_labels_sha256':g.sha256(ROOT/'data/calibration/manual-labels/assembled/labels.json'),
      'candidate_original_sha256_before':frozen_hash,'candidate_original_sha256_after':g.sha256(out/'grasp-candidate.npz'),
      'checkpoint_sha256':g.sha256(checkpoint),'script_sha256':g.sha256(__file__),
      'source_recording_and_images_hashes_verified':True,'training_rows':int(otrain.sum()+train.sum()),
      'original_train_rows':int(otrain.sum()),'supplemental_train_rows':int(train.sum()),
      'original_validation_rows':int((~otrain).sum()),'supplemental_validation_rows':int((~train).sum()),
      'secondary_ai_review':data.get('secondary_ai_review'),
      'supplemental_object_counts':{'empty':len(supp)},'supplemental_confidences':{c:sum(r['labels']['grasp']['left']['confidence']==c for r in supp) for c in ('high','medium')},
      'results':results,'limitations':['AI manual labels are not human-certified; only one supplemental midpoint per trial with +/-0.6s context.',
        'Four closed-empty heldout examples are four correlated same-session trials, not validation for slipping or dropped object under motion.',
        'Validation supplement inspected after identifying old coverage gap; supplementary challenge set, not a fresh untouched prospective test.',
        'Classifier hyperparameters and image encoder remain fixed; no task-time or encoder inputs enter visual classifier.',
        'No continuous temporal grasp-gate validation or hardware retention test; runtime remains unverified.']}
    if frozen_hash!=report['candidate_original_sha256_after']: raise ValueError('Frozen candidate changed')
    (out/'grasp-supplement-report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'grasp-supplement-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    np.savez(out/'grasp-supplement-candidate.npz',**augmented,classes=np.asarray(g.CLASSES),crop=np.asarray(g.CROP),threshold=g.THRESHOLD)
    print(json.dumps({'frozen_on_supplement_validation':frozen_result['validation'],
                      'augmented_original_validation':results['supplemented_visual_candidate']['original_validation'],
                      'augmented_supplemental_validation':results['supplemented_visual_candidate']['supplemental_validation']},indent=2))
    return report


if __name__=='__main__':run()
