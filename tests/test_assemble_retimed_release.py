"""Release assembly contracts: offsets, exact encoded media, labels and metadata."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPEC=importlib.util.spec_from_file_location('assemble_release',Path(__file__).parents[1]/'scripts/assemble_retimed_release.py')
release=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def fixture(root, episodes, precision):
    root.mkdir()
    features={key:dict(dtype='int64',shape=[1]) for key in ('index','episode_index','frame_index','task_index')}
    features.update(timestamp=dict(dtype='float32',shape=[1]),action=dict(dtype='float32',shape=[2]),
                    **{'observation.images.overhead':dict(dtype='image',shape=[2,2,3])})
    release.write_json(root/'meta/info.json',dict(features=features,fps=20,total_episodes=episodes,total_frames=episodes*3,
        codebase_version='v3.0',robot_type='test',data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        precision_loss_metadata=dict(schema='meta/precision.json',table='meta/precision_weights.parquet',join_key='index')))
    pq.write_table(pa.Table.from_pylist([dict(task_index=0,__index_level_0__='task')]),root/'meta/tasks.parquet')
    rows=[]; metadata=[]; weights=[]
    for ep in range(episodes):
        for frame in range(3):
            rows.append(dict(index=ep*3+frame,episode_index=ep,frame_index=frame,task_index=0,timestamp=frame/20,
                action=[ep+frame/8,1e-7+frame/1e8],**{'observation.images.overhead':dict(bytes=b'\xff\xd8payload'+bytes([ep,frame]),path=None)}))
            weights.append(dict(index=ep*3+frame,episode_index=ep,frame_index=frame,timestamp=frame/20,
                action_source_time_s=(frame+1)/20,observation_source_time_s=frame/20,
                gear_insertion=frame==1,pin_insertion=False,loss_weight=3. if frame==1 else 1.,annotation_scope='',confidence=''))
        meta=dict(episode_index=ep,length=3,tasks=['task'],dataset_from_index=ep*3,dataset_to_index=ep*3+3,
                  **{'data/chunk_index':0,'data/file_index':0})
        for stat,value in dict(min=[[[0.]],[[0.]],[[0.]]],max=[[[1.]],[[1.]],[[1.]]],
            mean=[[[.1+ep/10]],[[.2]],[[.3]]],std=[[[.01]],[[.02]],[[.03]]],count=[3]).items():
            meta[f'stats/observation.images.overhead/{stat}']=value
        metadata.append(meta)
    path=root/'data/chunk-000/file-000.parquet';path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),path)
    path=root/'meta/episodes/chunk-000/file-000.parquet';path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(metadata),path)
    release.write_json(root/'meta/stats.json',{})
    release.write_json(root/'processing/manifest.json',dict(episodes=[dict(id=f'{root.name}-{ep}',episode_index=ep,exported_observations=3) for ep in range(episodes)]))
    if precision:
        pq.write_table(pa.Table.from_pylist(weights),root/'meta/precision_weights.parquet')
        release.write_json(root/'meta/precision.json',dict(default_loss_weight=1,coverage={}))
        for ep in range(episodes):
            path=root/f'processing/retiming/episode_{ep:06d}'
            mapping=release.identity_mapping(pa.Table.from_pylist(rows[ep*3:ep*3+3]),ep,ep,f'{root.name}-{ep}',20)
            release.write_json(path.with_suffix('.json'),dict(protected_mask=[True,False,True],interval_scales=[1.,1.5,1.]))
            pq.write_table(mapping,path.with_suffix('.parquet'))
    return rows


class AssemblyTests(unittest.TestCase):
    def test_release_exact_offsets_labels_stats_and_portable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); old=base/'old';new=base/'new';out=base/'release'
            fixture(old,2,True);new_rows=fixture(new,1,False)
            before=release.sha256(old/'data/chunk-000/file-000.parquet')
            result=release.assemble(old,new,out,'org/test',native=False)
            self.assertEqual(result,dict(episodes=3,frames=9))
            self.assertEqual(release.sha256(old/'data/chunk-000/file-000.parquet'),before)
            actual=pq.read_table(out/'data/chunk-000/file-002.parquet')
            self.assertEqual(actual['index'].to_pylist(),[6,7,8])
            self.assertEqual(actual['episode_index'].to_pylist(),[2,2,2])
            self.assertEqual(release.row_digest(actual),release.row_digest(pa.Table.from_pylist(new_rows),2,6))
            weights=pq.read_table(out/'meta/precision_weights.parquet').to_pylist()
            self.assertEqual([r['loss_weight'] for r in weights],[1.,3.,1.,1.,3.,1.,1.,1.,1.])
            mapping=pq.read_table(out/'processing/retiming/episode_000002.parquet')
            self.assertEqual(mapping['index'].to_pylist(),[6,7,8])
            self.assertEqual(mapping['source_episode_index'].to_pylist(),[0,0,0])
            self.assertEqual(release.read_json(out/'processing/retiming/episode_000000.json')['protected_mask'],[True,False,True])
            meta=release.episode_metadata(out)
            self.assertEqual([m['data/file_index'] for m in meta],[0,1,2])
            self.assertEqual([m['dataset_from_index'] for m in meta],[0,3,6])
            stats=release.read_json(out/'meta/stats.json')
            self.assertEqual(stats['index']['mean'],[4.])
            self.assertEqual(stats['index']['count'],[9])
            np.testing.assert_allclose(stats['observation.images.overhead']['mean'],[[[(.1+.2+.1)/3]],[[.2]],[[.3]]])
            self.assertTrue((out/'processing/cohorts/new-reference/source_files_sha256.json').exists())
            self.assertTrue(release.read_json(out/'processing/assembly_verification.json')['passed'])
            with self.assertRaises(FileExistsError):
                release.assemble(old,new,out,'org/test',native=False)
            # Independent verification must catch an altered JPEG byte.
            rows=actual.to_pylist();rows[0]['observation.images.overhead']['bytes']=b'\xff\xd8corrupt'
            pq.write_table(pa.Table.from_pylist(rows),out/'data/chunk-000/file-002.parquet')
            with self.assertRaises(AssertionError):
                release.verify(old,new,out,native=False)

    def test_identity_preserves_original_source_clock_and_camera_evidence(self):
        table=pa.Table.from_pylist([dict(index=76273,episode_index=227,frame_index=0,timestamp=0.)])
        alignment=pa.Table.from_pylist([dict(index=0,episode_index=0,frame_index=0,
            source_episode_time_s=.05,source_episode_id='new-0',**{'overhead.offset_s':-.01,'overhead.source_path':'camera.jpg'})])
        mapping=release.identity_mapping(table,227,0,'new-0',20,alignment).to_pylist()[0]
        self.assertEqual(mapping['index'],76273)
        self.assertEqual(mapping['episode_index'],227)
        self.assertEqual(mapping['source_episode_index'],0)
        self.assertEqual(mapping['source_episode_time_s'],.05)
        self.assertEqual(mapping['action_source_episode_time_s'],.1)
        self.assertEqual(mapping['overhead.source_path'],'camera.jpg')
        self.assertEqual(mapping['overhead.offset_s'],-.01)

    def test_numeric_population_variance_avoids_float32_cancellation(self):
        values=np.array([[1000.001],[1000.002],[1000.003]],dtype=np.float32)
        stats=release.numeric_stats(values)
        self.assertGreater(stats['std'][0],0)
        np.testing.assert_allclose(stats['std'],values.astype(np.float64).std(0),atol=0,rtol=0)

    def test_row_digest_checks_numeric_and_nested_image_fields(self):
        row=dict(index=0,episode_index=0,action=[1.,2.],image=dict(bytes=b'jpeg',path=None))
        original=pa.Table.from_pylist([row]);row['action']=[1.,3.]
        self.assertNotEqual(release.row_digest(original),release.row_digest(pa.Table.from_pylist([row])))

if __name__=='__main__':
    unittest.main()
