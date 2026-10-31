#!/usr/bin/env python3
"""Assemble an independently verifiable, portable LeRobot v3 real-data release.

Media memory is bounded by one source parquet (normally <=100 MB) and one
selected episode. Numeric columns alone are retained for exact global quantiles.
No HF dataset cache, image re-encoding, source writes, uploads, or training.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric_stats(values):
    raw = np.asarray(values)
    if raw.ndim == 1:
        raw = raw[:, None]
    values = raw.astype(np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite nonempty numeric data')
    stats = dict(min=raw.min(0).tolist(), max=raw.max(0).tolist(),
                 mean=values.mean(0).tolist(), std=values.std(0).tolist(), count=[len(values)])
    stats.update({f'q{int(q * 100):02}': np.quantile(values, q, axis=0).tolist()
                  for q in (.01, .1, .5, .9, .99)})
    return stats


def remap(table, episode_offset, index_offset):
    for name, offset in (('episode_index', episode_offset), ('index', index_offset)):
        if name in table.column_names and offset:
            pos = table.schema.get_field_index(name)
            table = table.set_column(pos, table.schema.field(name), pc.add(table[name], offset))
    return table


def episode_metadata(root):
    return sorted((row for p in sorted((root / 'meta/episodes').rglob('*.parquet'))
                   for row in pq.read_table(p).to_pylist()), key=lambda row: row['episode_index'])


def locate_episodes(root):
    result = {}
    for path in sorted((root / 'data').rglob('*.parquet')):
        for episode in pc.unique(pq.read_table(path, columns=['episode_index'])['episode_index']).to_pylist():
            result.setdefault(int(episode), []).append(path)
    return result


def load_episode(paths, episode):
    tables = [pq.read_table(p, filters=[('episode_index', '=', episode)]) for p in paths]
    table = pa.concat_tables(tables)
    return table.take(pc.sort_indices(table, sort_keys=[('frame_index', 'ascending')]))


def row_digest(table, episode_offset=0, index_offset=0):
    """Canonical exact typed row digest, including every encoded JPEG byte."""
    digest = hashlib.sha256()
    for batch in table.to_batches(max_chunksize=32):
        for row in batch.to_pylist():
            row['episode_index'] += episode_offset
            row['index'] += index_offset
            # Bytes are encoded without decoding images. JSON preserves exact float
            # values of the Arrow numeric types and stable dictionary ordering.
            def encode(value):
                if isinstance(value, bytes):
                    return {'bytes_sha256': hashlib.sha256(value).hexdigest(), 'size': len(value)}
                raise TypeError(type(value).__name__)
            digest.update(json.dumps(row, sort_keys=True, default=encode, separators=(',', ':'), allow_nan=False).encode())
            digest.update(b'\n')
    return digest.hexdigest()


def copy_provenance(source, target):
    """Copy actual evidence; original absolute paths are historical strings only."""
    target.mkdir(parents=True)
    shutil.copytree(source / 'processing', target / 'processing', symlinks=False)
    shutil.copytree(source / 'meta', target / 'meta', symlinks=False)
    return {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}


def identity_mapping(table, global_episode, source_episode, source_id, fps, alignment=None):
    rows = []
    for index, frame, timestamp in zip(table['index'].to_pylist(), table['frame_index'].to_pylist(), table['timestamp'].to_pylist()):
        rows.append(dict(index=index, episode_index=global_episode, frame_index=frame,
            source_episode_index=source_episode, source_episode_id=source_id,
            source_state_frame=float(frame), source_action_frame=float(frame + 1),
            source_image_frame=frame, source_episode_time_s=float(timestamp),
            action_source_episode_time_s=(frame + 1) / fps, identity_timing=True))
    if alignment is not None:
        original = alignment.to_pylist()
        if len(original) != len(rows):
            raise ValueError('Alignment and data lengths differ')
        for row, evidence in zip(rows, original):
            source_time = evidence['source_episode_time_s']
            row.update({key: value for key, value in evidence.items()
                        if key not in ('index', 'episode_index', 'frame_index')})
            row['source_episode_time_s'] = source_time
            row['action_source_episode_time_s'] = source_time + 1 / fps
    return pa.Table.from_pylist(rows)


def default_precision(table, mapping, template):
    rows = []
    for row, timing in zip(table.select(['index','episode_index','frame_index','timestamp']).to_pylist(), mapping.to_pylist()):
        row.update(timestamp=row['frame_index'] / 20, action_source_time_s=timing['action_source_episode_time_s'],
            observation_source_time_s=timing['source_episode_time_s'], gear_insertion=False,
            pin_insertion=False, loss_weight=1., annotation_scope='', confidence='')
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=template)


def assemble(old, new, output, repo_id, native=True):
    from lerobot.datasets.compute_stats import aggregate_stats
    old, new, output = (Path(p).resolve() for p in (old, new, output))
    if output.exists():
        raise FileExistsError(output)
    if any(output == source or source in output.parents for source in (old, new)):
        raise ValueError('Output must be separate from sources')
    infos = [read_json(p / 'meta/info.json') for p in (old,new)]
    if any(info['features'] != infos[0]['features'] or info['fps'] != 20 for info in infos):
        raise ValueError('Cohort features and 20 Hz clocks must match')
    if not pq.read_table(old / 'meta/tasks.parquet').equals(pq.read_table(new / 'meta/tasks.parquet')):
        raise ValueError('Task tables differ')
    output.mkdir(parents=True)
    numeric = [key for key, feature in infos[0]['features'].items() if feature['dtype'] not in ('image','video','string')]
    cameras = [key for key, feature in infos[0]['features'].items() if feature['dtype'] == 'image']
    numeric_parts = {key: [] for key in numeric}
    episode_rows, image_episode_stats, entries, proofs, cohort_reports, weights = [], [], [], [], [], []
    precision_schema = pq.read_schema(old / 'meta/precision_weights.parquet')
    ep_offset = index_offset = 0
    for source, info, label in zip((old,new),infos,('old-retimed','new-reference')):
        manifest = read_json(source / 'processing/manifest.json')
        evidence = output / 'processing/cohorts' / label
        hashes = copy_provenance(source, evidence)
        write_json(evidence / 'source_files_sha256.json', hashes)
        metadata = episode_metadata(source)
        files = locate_episodes(source)
        if sorted(files) != list(range(info['total_episodes'])):
            raise ValueError('Noncontiguous cohort episode indices')
        source_weights = pq.read_table(source / 'meta/precision_weights.parquet') if (source / 'meta/precision_weights.parquet').exists() else None
        for source_ep, meta in enumerate(metadata):
            episode = ep_offset + source_ep
            table = load_episode(files[source_ep], source_ep)
            if table['frame_index'].to_pylist() != list(range(len(table))):
                raise ValueError('Noncontiguous frame indices')
            table = remap(table, ep_offset, index_offset)
            destination = output / f'data/chunk-{episode // 1000:03d}/file-{episode % 1000:03d}.parquet'
            destination.parent.mkdir(parents=True,exist_ok=True)
            # Only share a complete unchanged file, never a filtered mixed file.
            if not ep_offset and not index_offset and len(files[source_ep]) == 1 and pq.ParquetFile(files[source_ep][0]).metadata.num_rows == len(table):
                try:
                    os.link(files[source_ep][0], destination)
                except OSError:
                    shutil.copy2(files[source_ep][0], destination)
            else:
                pq.write_table(table, destination, compression='zstd')
            meta = copy.deepcopy(meta)
            start = int(table['index'][0].as_py())
            meta.update(episode_index=episode, dataset_from_index=start, dataset_to_index=start+len(table),
                        **{'data/chunk_index': episode // 1000, 'data/file_index': episode % 1000,
                           'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0})
            image_stats = {}
            for camera in cameras:
                image_stats[camera] = {key.removeprefix(f'stats/{camera}/'): np.asarray(value)
                    for key,value in meta.items() if key.startswith(f'stats/{camera}/')}
            image_episode_stats.append(image_stats)
            for key in numeric:
                values = np.asarray(table[key].to_pylist())
                numeric_parts[key].append(values)
                for stat,value in numeric_stats(values).items():
                    meta[f'stats/{key}/{stat}'] = value
            episode_rows.append(meta)
            entry = copy.deepcopy(manifest['episodes'][source_ep])
            entry.update(episode_index=episode, cohort=label, cohort_episode_index=source_ep,
                         source_manifest=f'processing/cohorts/{label}/processing/manifest.json')
            sidecar = output / f'processing/retiming/episode_{episode:06d}'
            original = source / f'processing/retiming/episode_{source_ep:06d}'
            if original.with_suffix('.parquet').exists():
                mapping = remap(pq.read_table(original.with_suffix('.parquet')),ep_offset,index_offset)
                detail = read_json(original.with_suffix('.json'))
            else:
                alignment_path = source / f'processing/alignment/episode_{source_ep:06d}.parquet'
                alignment = pq.read_table(alignment_path) if alignment_path.exists() else None
                mapping = identity_mapping(table,episode,source_ep,entry['id'],20,alignment)
                detail = dict(source_episode_index=source_ep,source_id=entry['id'],source_frames=len(table),
                    output_frames=len(table),source_duration_s=len(table)/20,output_duration_s=len(table)/20,
                    added_duration_s=0.,identity_timing=True,reason='Preferred new-method reference cohort retained at recorded 20 Hz timing.')
            detail.update(episode_index=episode,cohort=label,cohort_episode_index=source_ep)
            write_json(sidecar.with_suffix('.json'),detail)
            pq.write_table(mapping,sidecar.with_suffix('.parquet'),compression='zstd')
            entry['retiming_report']=str(sidecar.with_suffix('.json').relative_to(output))
            entries.append(entry)
            if source_weights is None:
                weight = default_precision(table,mapping,precision_schema)
            else:
                weight = remap(source_weights.filter(pc.equal(source_weights['episode_index'],source_ep)),ep_offset,index_offset)
            weights.append(weight)
            proofs.append(dict(episode_index=episode,cohort=label,source_episode_index=source_ep,
                episode_offset=ep_offset,index_offset=index_offset,rows=len(table),
                source_data_files=[str(p.relative_to(source)) for p in files[source_ep]],
                data_path=str(destination.relative_to(output)),expected_row_sha256=row_digest(table)))
            del table
            if (episode + 1) % 25 == 0:
                print(f'Assembled {episode+1} episodes',flush=True)
        cohort_reports.append(dict(label=label,episodes=info['total_episodes'],frames=info['total_frames'],
            episode_offset=ep_offset,index_offset=index_offset,source_manifest=f'processing/cohorts/{label}/processing/manifest.json',
            source_files_sha256=f'processing/cohorts/{label}/source_files_sha256.json'))
        ep_offset += info['total_episodes']
        index_offset += info['total_frames']
    info = copy.deepcopy(infos[0])
    info.update(total_episodes=ep_offset,total_frames=index_offset,splits={'train':f'0:{ep_offset}'})
    write_json(output / 'meta/info.json',info)
    shutil.copy2(old / 'meta/tasks.parquet',output / 'meta/tasks.parquet')
    metadata_path = output / 'meta/episodes/chunk-000/file-000.parquet'
    metadata_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows),metadata_path,compression='zstd')
    stats = {key:{stat:value.tolist() for stat,value in feature.items()} for key,feature in aggregate_stats(image_episode_stats).items()}
    stats.update({key:numeric_stats(np.concatenate(parts)) for key,parts in numeric_parts.items()})
    write_json(output / 'meta/stats.json',stats)
    combined_weights = pa.concat_tables(weights)
    pq.write_table(combined_weights,output / 'meta/precision_weights.parquet',compression='zstd')
    precision = read_json(old / 'meta/precision.json')
    precision.pop('retimed_from',None)
    precision['coverage']=dict(total_action_targets=index_offset,
        weighted_action_targets=pc.sum(pc.greater(combined_weights['loss_weight'],1)).as_py(),
        gear_action_targets=pc.sum(combined_weights['gear_insertion']).as_py(),
        pin_action_targets=pc.sum(combined_weights['pin_insertion']).as_py(),episodes=ep_offset)
    precision['new_reference_labels']='No precision annotations; all new-reference action targets default to weight 1.'
    write_json(output / 'meta/precision.json',precision)
    write_json(output / 'processing/manifest.json',dict(repo_id=repo_id,format='LeRobot v3',fps=20,
        source_kind='combined_real_retimed_and_reference',training_changed=False,
        cohorts=cohort_reports,episodes=entries,totals=dict(episodes=ep_offset,observations=index_offset),
        precision_loss_metadata=info['precision_loss_metadata'],
        provenance_note='Cohort evidence is included under processing/cohorts; original paths in historical reports are informational, never load dependencies. Data contain original embedded JPEG bytes.'))
    write_json(output / 'processing/assembly_proof.json',dict(version=1,episodes=proofs))
    (output / 'README.md').write_text(f'---\ntask_categories:\n- robotics\n---\n# {repo_id}\n\n{ep_offset} real episodes, {index_offset} frames at 20 Hz. Older demonstrations are time-stretched; preferred new-method demonstrations retain their timing. Embedded JPEGs and all numeric targets preserve the reviewed cohort exports. Training is unchanged.\n\nFull cohort manifests, source hashes, timing maps, precision weights, and independent verification are under `processing/`. Missing new-cohort precision labels default to weight 1. Recorded effort is historical context, not retimed dynamics.\n')
    verify(old,new,output,native=native)
    return dict(episodes=ep_offset,frames=index_offset)


def verify(old,new,output,native=True):
    """Re-read original inputs independently and prove the complete release."""
    roots = {'old-retimed':Path(old),'new-reference':Path(new)}
    output = Path(output)
    manifest = read_json(output / 'processing/manifest.json')
    proofs = read_json(output / 'processing/assembly_proof.json')['episodes']
    image_checks = rows_checked = native_checks = 0
    output_info = read_json(output / 'meta/info.json')
    numeric = [key for key, feature in output_info['features'].items()
               if feature['dtype'] not in ('image', 'video', 'string')]
    numeric_parts = {key: [] for key in numeric}
    metadata = episode_metadata(output)
    weights = pq.read_table(output / 'meta/precision_weights.parquet')
    image_stats = []
    for proof in proofs:
        root = roots[proof['cohort']]
        source = load_episode([root / p for p in proof['source_data_files']],proof['source_episode_index'])
        destination = pq.read_table(output / proof['data_path'])
        expected = row_digest(source,proof['episode_offset'],proof['index_offset'])
        assert expected == row_digest(destination) == proof['expected_row_sha256']
        assert len(destination) == proof['rows']
        episode = proof['episode_index']
        meta = metadata[episode]
        start = int(destination['index'][0].as_py())
        assert meta['episode_index'] == episode
        assert meta['dataset_from_index'] == start and meta['dataset_to_index'] == start + len(destination)
        assert output_info['data_path'].format(chunk_index=meta['data/chunk_index'],file_index=meta['data/file_index']) == proof['data_path']
        for key in numeric:
            values = np.asarray(destination[key].to_pylist())
            numeric_parts[key].append(values)
            for stat, expected_value in numeric_stats(values).items():
                np.testing.assert_allclose(meta[f'stats/{key}/{stat}'], expected_value, atol=1e-12, rtol=1e-12)
        image_stats.append({key: {name.removeprefix(f'stats/{key}/'): np.asarray(value)
            for name,value in meta.items() if name.startswith(f'stats/{key}/')}
            for key,feature in output_info['features'].items() if feature['dtype']=='image'})
        mapping = pq.read_table(output / f'processing/retiming/episode_{episode:06d}.parquet')
        for name in ('index','episode_index','frame_index'):
            assert mapping[name].to_pylist() == destination[name].to_pylist()
        original_sidecar = root / f'processing/retiming/episode_{proof["source_episode_index"]:06d}.parquet'
        if original_sidecar.exists():
            expected_mapping = remap(pq.read_table(original_sidecar),proof['episode_offset'],proof['index_offset'])
            assert expected_mapping.equals(mapping)
        else:
            alignment_path = root / f'processing/alignment/episode_{proof["source_episode_index"]:06d}.parquet'
            alignment = pq.read_table(alignment_path) if alignment_path.exists() else None
            source_id = manifest['episodes'][episode]['id']
            expected_mapping = identity_mapping(destination,episode,proof['source_episode_index'],source_id,20,alignment)
            assert expected_mapping.equals(mapping)
        episode_weights = weights.filter(pc.equal(weights['episode_index'],episode))
        for name in ('index','episode_index','frame_index'):
            assert episode_weights[name].to_pylist() == destination[name].to_pylist()
        if proof['cohort']=='old-retimed':
            original_weights = pq.read_table(root / 'meta/precision_weights.parquet',
                filters=[('episode_index','=',proof['source_episode_index'])])
            assert remap(original_weights,proof['episode_offset'],proof['index_offset']).equals(episode_weights)
        if proof['cohort']=='new-reference':
            assert episode_weights['loss_weight'].to_pylist() == [1.] * len(destination)
            assert not any(episode_weights['gear_insertion'].to_pylist())
            assert not any(episode_weights['pin_insertion'].to_pylist())
            np.testing.assert_allclose(episode_weights['observation_source_time_s'].to_pylist(),mapping['source_episode_time_s'].to_pylist(),atol=0,rtol=0)
            np.testing.assert_allclose(episode_weights['action_source_time_s'].to_pylist(),mapping['action_source_episode_time_s'].to_pylist(),atol=0,rtol=0)
        for key in destination.column_names:
            if key.startswith('observation.images.'):
                for value in destination[key].to_pylist():
                    assert value['bytes'] and value['bytes'][:2] == b'\xff\xd8'
                    image_checks += 1
        rows_checked += len(destination)
        if (proof['episode_index']+1)%25==0:
            print(f'Independently verified {proof["episode_index"]+1} episodes',flush=True)
    output_stats = read_json(output / 'meta/stats.json')
    for key, parts in numeric_parts.items():
        for stat, expected_value in numeric_stats(np.concatenate(parts)).items():
            np.testing.assert_allclose(output_stats[key][stat],expected_value,atol=1e-12,rtol=1e-12)
    from lerobot.datasets.compute_stats import aggregate_stats
    for key, feature in aggregate_stats(image_stats).items():
        for stat, expected_value in feature.items():
            np.testing.assert_allclose(output_stats[key][stat],expected_value,atol=1e-12,rtol=1e-12)
    if native:
        import torch
        import datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import get_hf_features_from_features,hf_transform_to_torch
        torch.set_num_threads(1)
        class BoundedNativeDataset(LeRobotDataset):
            def load_hf_dataset(self):
                assert len(self.episodes)==1
                table=pq.read_table(self.root / self.meta.get_data_file_path(self.episodes[0]))
                bounded=datasets.Dataset(table,info=datasets.DatasetInfo(features=get_hf_features_from_features(self.features)))
                bounded.set_transform(hf_transform_to_torch)
                return bounded
        for proof in proofs:
            actions=np.asarray(pq.read_table(output / proof['data_path'],columns=['action'])['action'].to_pylist(),dtype=np.float32)
            loader=BoundedNativeDataset(manifest['repo_id'],root=output,episodes=[proof['episode_index']],
                video_backend='pyav',delta_timestamps={'action':[i/20 for i in range(50)]})
            assert len(loader)==len(actions)
            for local in sorted({0,len(actions)//2,len(actions)-1}):
                item=loader[local]
                np.testing.assert_array_equal(item['action'].numpy(),actions[np.minimum(local+np.arange(50),len(actions)-1)])
                np.testing.assert_array_equal(item['action_is_pad'].numpy(),local+np.arange(50)>=len(actions))
                native_checks+=1
            del loader
            if (proof['episode_index']+1)%25==0:
                print(f'Native 50-action chunks verified {proof["episode_index"]+1} episodes',flush=True)
    assert rows_checked==read_json(output / 'meta/info.json')['total_frames']
    assert not any(p.is_symlink() for p in output.rglob('*'))
    result=dict(passed=True,all_rows_match_sources_with_only_global_offsets=True,
        checked_rows=rows_checked,identical_embedded_jpeg_checks=image_checks,
        native_50_action_chunk_checks=native_checks,native_verified=bool(native),
        episodes=len(proofs),training_changed=False,no_symlinks=True,
        numeric_statistics_verified=True,image_statistics_aggregation_verified=True,
        metadata_locations_verified=True,precision_and_retiming_indices_verified=True)
    write_json(output / 'processing/assembly_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--skip-native',action='store_true',help='For development fixtures only; release verification must include native loading.')
    args=parser.parse_args()
    print(assemble(args.old,args.new,args.output,args.repo_id,native=not args.skip_native))
