#!/usr/bin/env python3
"""Bounded-memory Arrow-only overhead recrop and cohort merge; sources immutable."""
import argparse,copy,hashlib,io,json,os,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from postprocess_kept import prepare_image,sha256,write_json
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import unflatten_dict,flatten_dict,write_stats
OH='observation.images.overhead'

def row_digest(digest,row):
    numeric={k:v for k,v in row.items() if not k.startswith('observation.images.')}
    digest.update(json.dumps(numeric,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for key in sorted(k for k in row if k.startswith('observation.images.')):
        digest.update(key.encode());digest.update(row[key]['bytes']);digest.update(str(row[key]['path']).encode())

def transform(path,roi):
    data=prepare_image(path,roi,256);buf=io.BytesIO();Image.fromarray(data).save(buf,format='JPEG',quality=95,subsampling=0)
    encoded=buf.getvalue()
    with Image.open(io.BytesIO(encoded)) as im:decoded=np.asarray(im).copy()
    hist=np.stack([np.bincount(decoded[:,:,c].ravel(),minlength=256) for c in range(3)])
    return encoded,hist

def image_stats(hist,frames):
    count=hist.sum(axis=1);levels=np.arange(256)/255.;mean=(hist*levels).sum(axis=1)/count
    std=np.sqrt(np.maximum(0,(hist*levels**2).sum(axis=1)/count-mean**2))
    values={'mean':mean,'std':std,'min':np.argmax(hist>0,axis=1)/255.,'max':(255-np.argmax(hist[:,::-1]>0,axis=1))/255.}
    for q in (.01,.1,.5,.9,.99):values[f'q{int(q*100):02d}']=np.array([np.searchsorted(np.cumsum(h),q*h.sum())/255. for h in hist])
    return {**{k:v.reshape(3,1,1) for k,v in values.items()},'count':np.array([frames])}

def offset_stats(ep,episode_offset,frame_offset):
    # Shift location statistics but preserve std/count for global-index features.
    for feature,offset in [('episode_index',episode_offset),('index',frame_offset)]:
        for key in list(ep):
            if key.startswith(f'stats/{feature}/') and key.rsplit('/',1)[1] not in ('std','count'):
                ep[key]=(np.asarray(ep[key])+offset).tolist()

def rebuild(a):
    out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    roots=[p.resolve() for p in a.roots];manifests=[json.loads((p/'processing/manifest.json').read_text()) for p in roots]
    source_entries=json.loads(a.source_map.read_text());source_entries={e['export_id']:e for e in source_entries}
    contract=('fps','image_size','task','image_encoding','action_semantics','state_semantics')
    for m in manifests:
        assert all(m[k]==manifests[0][k] for k in contract)
        assert m['image_size']==256
        for cam in ('wrist_port2','wrist_port3'):assert m['image_crops_xyxy'][cam]==manifests[0]['image_crops_xyxy'][cam]
    selected=[e['id'] for m in manifests for e in m['episodes']];assert len(set(selected))==len(selected) and set(selected)==set(source_entries)
    roi=a.overhead_roi;assert 0<=roi[0]<roi[2]<=640 and 0<=roi[1]<roi[3]<=480 and roi[2]-roi[0]==roi[3]-roi[1]
    required=sum((p.stat().st_size) for r in roots for p in (r/'data').rglob('*.parquet'))
    assert shutil.disk_usage(out.parent).free>required*1.1+500_000_000,'Insufficient conservative Parquet workspace budget'
    out.mkdir();(out/'processing/source_union').mkdir(parents=True)
    expected_digest=hashlib.sha256();metadata=[];allstats=[];combinedepisodes=[];allannotations=[];precisionepisodes=[];weighttables=[];sourcemap=[]
    ep_offset=frame_offset=0;allchecks=0;global_file=0
    for root,manifest in zip(roots,manifests):
        oldmeta={int(e['episode_index']):e for p in sorted((root/'meta/episodes').rglob('*.parquet')) for e in pq.read_table(p).to_pylist()}
        episode_meta=[];writer=None;current=None;hist=None;count=0;alignment=None;source=None;buffer=[];schema=None;current_path=None
        def flush():
            nonlocal buffer
            if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=schema));buffer=[]
        def finish():
            nonlocal writer,global_file
            if writer is None:return
            flush();writer.close();ep=copy.deepcopy(oldmeta[current]);ep['episode_index']+=ep_offset;ep['dataset_from_index']+=frame_offset;ep['dataset_to_index']+=frame_offset
            ep['data/chunk_index']=0;ep['data/file_index']=global_file;ep['meta/episodes/chunk_index']=0;ep['meta/episodes/file_index']=0
            offset_stats(ep,ep_offset,frame_offset)
            for key,value in image_stats(hist,count).items():ep[f'stats/{OH}/{key}']=value.tolist()
            assert count==ep['length'];metadata.append(ep);allstats.append(unflatten_dict({k[6:]:np.asarray(v) for k,v in ep.items() if k.startswith('stats/')}));global_file+=1;writer=None
        for path in sorted((root/'data').rglob('*.parquet')):
            parquet=pq.ParquetFile(path);schema=parquet.schema_arrow
            for batch in parquet.iter_batches(batch_size=a.batch_size):
                for original in batch.to_pylist():
                    oldep=int(original['episode_index'])
                    if current!=oldep:
                        finish();current=oldep;count=0;hist=np.zeros((3,256),dtype=np.int64)
                        annotation=manifest['episodes'][oldep];identifier=annotation['id'];entry=source_entries[identifier];source=Path(entry['source_path'])
                        # Three incomplete runs must use already-validated source overlays.
                        if sha256(source/'metadata.json')!=annotation['source_hashes']['metadata.json']:
                            candidate=root.parent/root.name.replace('curated-20hz','curation-sources')/annotation.get('source_id',identifier)
                            assert candidate.exists();source=candidate
                        for name in ('metadata.json','states.jsonl','frames.jsonl'):assert sha256(source/name)==annotation['source_hashes'][name],(identifier,name)
                        union=out/'processing/source_union'/annotation.get('source_id',identifier);union.symlink_to(source.resolve(),target_is_directory=True)
                        sourcemap.append(dict(entry,source_path=str(source.resolve()),new_episode_index=oldep+ep_offset))
                        alignment=pq.read_table(root/f'processing/alignment/episode_{oldep:06d}.parquet').to_pylist()
                        aligned=[]
                        for ar in alignment:ar['index']+=frame_offset;ar['episode_index']+=ep_offset;aligned.append(ar)
                        ap=out/f'processing/alignment/episode_{oldep+ep_offset:06d}.parquet';ap.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(aligned),ap,compression='zstd')
                        for relative in [f'processing/source_metadata/{identifier}.json',f'trimmed_full_rate/{identifier}.tar.gz']:
                            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);os.link(root/relative,dest)
                            assert os.stat(root/relative).st_ino==os.stat(dest).st_ino
                        combinedepisodes.append(dict(annotation,episode_index=oldep+ep_offset))
                        current_path=out/f'data/chunk-000/file-{global_file:03d}.parquet';current_path.parent.mkdir(parents=True,exist_ok=True);writer=pq.ParquetWriter(current_path,schema,compression='zstd')
                    row=original.copy();ar=alignment[count];assert ar['frame_index']==original['frame_index']
                    imagepath=source/ar[f'overhead.source_path'];assert sha256(imagepath)==ar['overhead.source_sha256']
                    encoded,h=transform(imagepath,roi);hist+=h;row[OH]={'bytes':encoded,'path':original[OH]['path']};row['index']+=frame_offset;row['episode_index']+=ep_offset
                    # Every non-overhead field remains exactly original except indices.
                    assert all(row[k]==v for k,v in original.items() if k not in (OH,'index','episode_index'))
                    row_digest(expected_digest,row);buffer.append(row);count+=1;allchecks+=1
                    if len(buffer)>=a.batch_size:flush()
        finish()
        annotations=json.loads((root/'processing/annotations.json').read_text());allannotations.extend(annotations['episodes'])
        precision=json.loads((root/'meta/precision.json').read_text())
        precisionepisodes.extend(dict(e,episode_index=e['episode_index']+ep_offset) for e in precision['episodes'])
        weights=pq.read_table(root/'meta/precision_weights.parquet').to_pylist()
        for row in weights:row['index']+=frame_offset;row['episode_index']+=ep_offset
        weighttables.append(pa.Table.from_pylist(weights))
        # Retain review and annotation evidence in a collision-free batch subfolder.
        if (root/'processing/curation').exists():shutil.copytree(root/'processing/curation',out/f'processing/curation/batch-{len(weighttables)-1}')
        ep_offset+=len(manifest['episodes']);frame_offset+=manifest['totals']['observations'];print(f'Completed {root.name}: total{ep_offset} episodes/{frame_offset} rows',flush=True)
    assert allchecks==frame_offset
    info=json.loads((roots[0]/'meta/info.json').read_text());info.update(total_episodes=ep_offset,total_frames=frame_offset,splits={'train':f'0:{ep_offset}'})
    write_json(out/'meta/info.json',info);shutil.copy2(roots[0]/'meta/tasks.parquet',out/'meta/tasks.parquet')
    for root in roots:assert pq.read_table(root/'meta/tasks.parquet').equals(pq.read_table(out/'meta/tasks.parquet'))
    ep_path=out/'meta/episodes/chunk-000/file-000.parquet';ep_path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(metadata),ep_path,compression='zstd');write_stats(aggregate_stats(allstats),out)
    combined=copy.deepcopy(manifests[0]);combined.update(repo_id=a.repo_id,episodes=combinedepisodes,reused_processed_batches=[str(r) for r in roots],recording_session_local_dates=sorted({e['source_id'][:8] for e in sourcemap}))
    combined['image_crops_xyxy']['overhead']=roi;combined['crop_rationale']='User-reviewed October26 overhead crop; wrist crops and encoded wrist bytes preserved.'
    combined['totals']={key:sum(m['totals'][key] for m in manifests) for key in manifests[0]['totals']}
    combined['rebuild_contract']=dict(overhead_only_reencoded=True,wrist_encoded_bytes_preserved=True,full_rate_archives_hardlinked=True,overhead_statistics='Exact histogram of all decoded transformed JPEG pixels; quantiles from256-level channel histograms.',source_map='processing/source_map.json',source_union='processing/source_union')
    weights=pa.concat_tables(weighttables);assert weights['index'].to_pylist()==list(range(frame_offset));pq.write_table(weights,out/'meta/precision_weights.parquet',compression='zstd');assert pq.read_table(out/'meta/precision_weights.parquet').equals(weights)
    precision=json.loads((roots[0]/'meta/precision.json').read_text());precision['episodes']=precisionepisodes;precision['input_batches']=[str(r) for r in roots]
    precision['coverage']={k:sum(json.loads((r/'meta/precision.json').read_text())['coverage'][k] for r in roots) for k in precision['coverage']}
    combined['precision_loss_metadata']['coverage']=precision['coverage'];write_json(out/'meta/precision.json',precision)
    write_json(out/'processing/manifest.json',combined);annotations['episodes']=allannotations;annotations['image_crops_xyxy']=combined['image_crops_xyxy'];write_json(out/'processing/annotations.json',annotations);write_json(out/'processing/source_map.json',sourcemap)
    observed_digest=hashlib.sha256()
    for p in sorted((out/'data').rglob('*.parquet')):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=a.batch_size):
            for row in batch.to_pylist():row_digest(observed_digest,row)
    assert expected_digest.digest()==observed_digest.digest()
    # Native LeRobot item/action-chunk logic with bounded in-memory Arrow backend.
    # Stock load_hf_dataset would duplicate the entire image table in HF Arrow cache.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import get_hf_features_from_features,hf_transform_to_torch
    import datasets
    class BoundedNativeDataset(LeRobotDataset):
        def load_hf_dataset(self):
            assert len(self.episodes)==1
            episode=self.episodes[0]
            table=pq.read_table(out/f'data/chunk-000/file-{episode:03d}.parquet')
            dataset=datasets.Dataset(table,info=datasets.DatasetInfo(features=get_hf_features_from_features(self.features)))
            dataset.set_transform(hf_transform_to_torch)
            return dataset
    exact_overhead_checks=0
    for ep in metadata:
        episode=ep['episode_index'];source=Path(sourcemap[episode]['source_path'])
        aligned=pq.read_table(out/f'processing/alignment/episode_{episode:06d}.parquet').to_pylist()
        diskrows=pq.read_table(out/f'data/chunk-000/file-{episode:03d}.parquet',columns=[OH]).to_pylist()
        for i in sorted({0,len(diskrows)//2,len(diskrows)-1}):
            expected,_=transform(source/aligned[i]['overhead.source_path'],roi)
            assert diskrows[i][OH]['bytes']==expected
            exact_overhead_checks+=1
        native=BoundedNativeDataset(a.repo_id,root=out,episodes=[ep['episode_index']],video_backend='pyav',delta_timestamps={'action':[i/combined['fps'] for i in range(10)]})
        assert len(native)==ep['length']
        for index in (0,len(native)-1):
            loaded=native[index];assert tuple(loaded['action'].shape)==(10,14);assert tuple(loaded[OH].shape)==(3,256,256)
        del native
    write_json(out/'processing/rebuild_verification.json',dict(passed=True,checked_observations=allchecks,all_stored_rows_match_expected_sha256=observed_digest.hexdigest(),wrist_and_numeric_preserved=True,all_overhead_source_hashes_checked=True,exact_overhead_source_transform_checks=exact_overhead_checks,full_rate_archives_hardlinked=True,native_episode_first_last_checks=ep_offset*2,native_proof_backend='Inherited LeRobot item/chunk logic with per-episode in-memory Arrow table; no full HF disk cache',precision_weight_rows_preserved_with_index_offsets=frame_offset))
    print(json.dumps(combined['totals'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',type=Path,required=True);p.add_argument('--source-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repo-id',required=True);p.add_argument('--overhead-roi',nargs=4,type=int,required=True);p.add_argument('--batch-size',type=int,default=16);rebuild(p.parse_args())
