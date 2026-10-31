"""Regression checks for recorded action targets, split isolation and inference."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('action_critic_train', SCRIPTS / 'action_critic_train.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(split='train', index=0):
    return dict(episode_id=f'{split}-{index}', session_id=f'{split}-{index}', split=split,
                images=['current.png'] * 3, future_images=['future.png'] * 3,
                state=[float(index), 2.], action_chunk=[[float(index), 0.]], dt=2.)


class ActionCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_manifest_rejects_leakage_nonfinite_and_cross_episode_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            def read(rows):
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                return m.read_transitions(path)
            rows = [row(), row('val')]
            self.assertEqual(read(rows)[1:], (2, (1, 2)))
            for key in ('episode_id', 'session_id'):
                bad = copy.deepcopy(rows); bad[1][key] = bad[0][key]
                with self.assertRaisesRegex(ValueError, 'leakage'): read(bad)
            for key, value in [('state', [float('nan'), 0.]), ('action_chunk', [[float('inf'), 0.]]),
                               ('dt', float('inf')), ('dt', 0.), ('future_episode_id', 'other'),
                               ('future_split', 'test')]:
                bad = copy.deepcopy(rows); bad[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): read(bad)

    def test_normalization_uses_only_training_transitions(self):
        rows = [row(index=0), row(index=2), row('val', 10000), row('test', -10000)]
        norm = m.fit_normalization(rows)
        torch.testing.assert_close(norm['state_mean'], torch.tensor([1., 2.]))
        torch.testing.assert_close(norm['action_mean'], torch.tensor([[1., 0.]]))
        self.assertEqual(float(norm['state_std'][0]), 1.)
        state, actions = m.normalized_inputs(rows, norm)
        self.assertTrue(torch.isfinite(state).all() and torch.isfinite(actions).all())
        with self.assertRaisesRegex(ValueError, 'training transitions'): m.fit_normalization(rows[2:])

    def test_target_uses_recorded_future_and_elapsed_discount_without_gradients(self):
        class Teacher(nn.Module):
            def __init__(self):
                super().__init__(); self.value = nn.Linear(256, 1, bias=False)
                nn.init.constant_(self.value.weight, 1. / 256.)
            def encode(self, x): return x[:, :1].expand(-1, 256) * self.value.weight.sum()
        teacher = Teacher()
        for gamma, dt in [(0., 1.), (1.1, 1.), (float('nan'), 1.), (.9, 0.), (.9, float('inf'))]:
            with self.subTest(gamma=gamma, dt=dt), self.assertRaises(ValueError):
                m.future_value_target(teacher, torch.zeros(1, 256), dt, gamma)
        def batch(refs, root, size, device): return torch.tensor([[3. if refs[0] == 'future.png' else -3.]])
        rows = [row(), row(index=1)]; rows[1]['dt'] = 4.
        with patch.object(m, 'image_batch', side_effect=batch) as reader:
            z, targets = m.encode_rows(rows, teacher, '.', 32, .5, torch.device('cpu'))
        self.assertEqual(reader.call_count, 2)  # current/future shared across both rows
        torch.testing.assert_close(targets, torch.sigmoid(torch.tensor([3., 3.])) * torch.tensor([.25, .0625]))
        self.assertTrue(torch.all(z == -3.))
        self.assertFalse(z.requires_grad or targets.requires_grad)
        self.assertIsNone(teacher.value.weight.grad)
        # Cached inference tensors must remain usable in a trainable critic backward pass.
        critic = m.ActionCritic(2, (1, 2), 16)
        critic(z, torch.zeros(2, 2), torch.zeros(2, 1, 2), torch.zeros(2)).sum().backward()
        self.assertIsNotNone(critic.head[0].weight.grad)

    def test_feature_cache_hit_and_identity_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'features.pt'
            expected = (torch.ones(2, 256), torch.tensor([.2, .3]))
            with patch.object(m, 'encode_rows', return_value=expected) as encode:
                args = ([row(), row('val')], None, '.', 32, 1., torch.device('cpu'))
                m.cached_features(path, {'teacher_sha256': 'first'}, *args)
                z, target = m.cached_features(path, {'teacher_sha256': 'first'}, *args)
                self.assertEqual(encode.call_count, 1)
                torch.testing.assert_close(z, expected[0])
                torch.testing.assert_close(target, expected[1])
                m.cached_features(path, {'teacher_sha256': 'changed'}, *args)
                self.assertEqual(encode.call_count, 2)

    def test_critic_learns_action_signal_at_identical_current_state(self):
        torch.manual_seed(4)
        critic = m.ActionCritic(2, (1, 1), 24)
        actions = torch.linspace(-1, 1, 32).reshape(-1, 1, 1)
        z, state, dt = torch.zeros(32, 256), torch.zeros(32, 2), torch.zeros(32)
        target = torch.sigmoid(actions[:, 0, 0] * 3)
        optimizer = torch.optim.Adam(critic.parameters(), lr=.01)
        for _ in range(120):
            loss = (critic(z, state, actions, dt) - target).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scores = critic(z, state, actions, dt).detach()
        self.assertLess(float((scores - target).square().mean()), .002)
        self.assertGreater(float(scores[-1] - scores[0]), .7)

    def test_cpu_cli_checkpoint_and_prediction_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, shade in [('current.png', 60), ('future.png', 180)]:
                Image.fromarray(np.full((32, 32, 3), shade, np.uint8)).save(root / name)
            teacher = m.VisualValue(False, 'resnet18')
            teacher_path = root / 'teacher.pt'
            torch.save({'model': teacher.state_dict(), 'config': {'backbone': 'resnet18', 'image_size': 32}}, teacher_path)
            frozen = m.load_teacher(torch.load(teacher_path, weights_only=False), torch.device('cpu'))
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            rows = [row(index=0), row(index=2), row('val', 1000), row('test', -1000)]
            # Heldout test images need not be readable during fitting.
            rows[-1]['images'] = rows[-1]['future_images'] = ['unavailable.png'] * 3
            manifest = root / 'rows.jsonl'; manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            out = root / 'out'; env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            subprocess.run([sys.executable, str(SCRIPTS / 'action_critic_train.py'), 'train',
                            '--manifest', str(manifest), '--teacher-checkpoint', str(teacher_path),
                            '--dataset-root', str(root), '--output', str(out), '--epochs', '1',
                            '--hidden-dim', '16', '--device', 'cpu'], check=True, capture_output=True, text=True, env=env)
            bundle = torch.load(out / 'best.pt', weights_only=False)
            self.assertEqual(bundle['epoch'], 1)
            report = json.loads((out / 'validation_report.json').read_text())
            self.assertEqual(report['validation_rows'], 1)
            self.assertTrue(all(np.isfinite(report[key]) for key in
                            ('val_current_teacher_mse', 'val_train_target_mean_mse',
                             'val_shuffled_action_mse', 'val_mean_action_mse')))
            torch.testing.assert_close(bundle['normalization']['state_mean'], torch.tensor([1., 2.]))
            self.assertTrue(all(np.isfinite(v) for v in bundle['metrics'].values()))
            candidates = root / 'candidates.json'; candidates.write_text('[[[0,0]],[[2,0]]]')
            output = root / 'scores.json'
            subprocess.run([sys.executable, str(SCRIPTS / 'action_critic_train.py'), 'predict',
                            '--checkpoint', str(out / 'best.pt'), '--images', *[str(root / 'current.png')] * 3,
                            '--state', '[0,2]', '--dt', '2', '--candidates', str(candidates),
                            '--output', str(output), '--device', 'cpu'], check=True, capture_output=True, text=True, env=env)
            result = json.loads(output.read_text())
            model = m.ActionCritic(2, (1, 2), 16); model.load_state_dict(bundle['model']); model.eval()
            states, actions = m.normalized_inputs([dict(state=[0, 2], action_chunk=a) for a in json.loads(candidates.read_text())], bundle['normalization'])
            with torch.inference_mode():
                z = frozen.encode(m.image_batch([str(root / 'current.png')] * 3, '.', 32, 'cpu'))
                expected = model(z.expand(2, -1), states, actions, torch.zeros(2)).tolist()
            np.testing.assert_allclose(result['scores'], expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(result['ranking'], sorted(range(2), key=lambda i: expected[i], reverse=True))


if __name__ == '__main__':
    unittest.main()
