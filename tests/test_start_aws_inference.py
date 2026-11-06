import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('start_aws', Path(__file__).resolve().parents[1]/'scripts/start-aws-inference.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StartupTests(unittest.TestCase):
    def test_discovery_excludes_partial_smoke_and_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run, step in [('normal','000100'),('smoke-new','000200'),('normal','000300')]:
                model = root/run/'checkpoints'/step/'pretrained_model'
                model.mkdir(parents=True)
                for name in m.REQUIRED:
                    (model/name).write_text('{}')
            (root/'normal/checkpoints/000300/pretrained_model/config.json').unlink()
            (root/'normal/checkpoints/last').symlink_to('000100')
            self.assertEqual(m.select_checkpoint(root).parent.name, '000100')

    def test_mismatch_and_absolute_actions_rejected(self):
        contract = dict(fps=20, task='task',dataset_repo='repo',dataset_revision=None,image_crops_xyxy={'a':[0,0,1,1]})
        meta = dict(fps=20,tasks=['task'],dataset_repo='repo',dataset_revision=None,action_representation='absolute',image_crops={'observation.images.a':[0,0,1,1]})
        m.validate_contract(meta, {'use_relative_actions':False},contract)
        for field,value in [('fps',10),('action_representation','delta'),('image_crops',{})]:
            with self.assertRaises(ValueError):
                m.validate_contract(dict(meta,**{field:value}),{'use_relative_actions':False},contract)
        with self.assertRaises(ValueError):
            m.validate_contract(meta,{'use_relative_actions':True},contract)

    def test_ssh_validation_and_payload_roundtrip(self):
        m.validate_target('54.161.106.103','ubuntu')
        for bad in ['-oProxyCommand=evil','host;touch x','user@host','host\nwhoami']:
            with self.assertRaises(ValueError): m.validate_target(bad,'ubuntu')
        payload = {'run_dir':"/mnt/pi05/runs/name';$(touch /tmp/nope)", 'action':'discover'}
        args = shlex.split(m.remote_command(payload))
        self.assertEqual(args[:2],['python3','-c'])
        self.assertEqual(json.loads(args[3]),payload)
        compile(args[2],'<remote>','exec')

    def test_metadata_normalization_and_dashboard_guard(self):
        source = {'training_tasks':['a','b']}
        self.assertEqual(m.normalize_metadata(source)['tasks'], ['a','b'])
        self.assertNotIn('tasks', source)
        m.ensure_dashboard_idle({'connected':True,'mode':'holding'})
        for state in [{'playing':True},{'rollout_active':True},{'collection_pending':True},
                      {'connected':True,'mode':'moving'}]:
            with self.assertRaises(ValueError):
                m.ensure_dashboard_idle(state)
