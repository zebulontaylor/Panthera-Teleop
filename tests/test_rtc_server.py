"""RTC protocol and sampler math, without checkpoint weights or robot devices."""
import base64
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'infra/inference'))
import rtc_guidance as rtc
import server

try:
    import torch
except ImportError:
    torch = None


class PrefixValidationTests(unittest.TestCase):
    def test_deployment_checks_and_memory_cap_are_preserved(self):
        import argparse
        meta = {key: server.CONTRACT[key] for key in
                ('fps', 'task', 'dataset_repo', 'dataset_revision')}
        meta.update(action_representation='absolute', image_crops={
            f'observation.images.{name}': crop
            for name, crop in server.CONTRACT['image_crops_xyxy'].items()})
        for key in ('dataset_manifest_sha256', 'dataset_mix'):
            if key in server.CONTRACT:
                meta[key] = server.CONTRACT[key]
        server.validate_deployment(meta, server.CONTRACT)
        with self.assertRaisesRegex(ValueError, 'image_crops'):
            server.validate_deployment(dict(meta, image_crops={}), server.CONTRACT)
        self.assertEqual(server.cuda_memory_fraction('.16'), .16)
        for value in ('0', '-1', '2', 'nan'):
            with self.assertRaises(argparse.ArgumentTypeError):
                server.cuda_memory_fraction(value)

    def test_deployment_accepts_task_list_only_without_singular_task(self):
        meta = {key: server.CONTRACT[key] for key in
                ('fps', 'dataset_repo', 'dataset_revision',
                 'dataset_manifest_sha256', 'dataset_mix') if key in server.CONTRACT}
        meta.update(action_representation='absolute', image_crops={
            f'observation.images.{name}': crop
            for name, crop in server.CONTRACT['image_crops_xyxy'].items()})
        task = server.CONTRACT['task']
        server.validate_deployment(dict(meta, tasks=['another task', task]), server.CONTRACT)
        server.validate_deployment(dict(meta, task=task, tasks=[]), server.CONTRACT)
        for tasks in (None, [], ['another task'], task, {'task': task}):
            with self.subTest(tasks=tasks), self.assertRaisesRegex(ValueError, 'mismatch: task'):
                server.validate_deployment(dict(meta, tasks=tasks), server.CONTRACT)
        for singular in (None, '', 'another task'):
            with self.subTest(task=singular), self.assertRaisesRegex(ValueError, 'mismatch: task'):
                server.validate_deployment(dict(meta, task=singular, tasks=[task]), server.CONTRACT)

    def test_supported_tasks_preserve_legacy_and_validate_metadata(self):
        task = server.CONTRACT['task']
        self.assertEqual(server.deployment_tasks(dict(task=task, tasks=['extra'])), [task])
        self.assertEqual(server.deployment_tasks(dict(tasks=[task, 'extra', task])), [task, 'extra'])
        for tasks in ([task, None], [task, ''], [task, '   '], [task, 1]):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                server.deployment_tasks(dict(tasks=tasks))

    def test_runtime_rejects_unsupported_tasks_before_reading_observations(self):
        runtime = server.Runtime('mock', mock=True)
        self.assertEqual(runtime.supported_tasks, [server.CONTRACT['task']])
        for task in ('extra', None, [], 1, ''):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'Unsupported task'):
                runtime.infer(dict(task=task))

    def test_valid(self):
        actions, overlap = rtc.validate_prefix(dict(previous_actions=[[1.] * 14] * 3, overlap=3), 50)
        self.assertEqual((actions.shape, overlap), ((3, 14), 3))

    def test_rejects_invalid_prefixes(self):
        good = dict(previous_actions=[[0.] * 14], overlap=1)
        bad = [None, {}, dict(good, extra=True)]
        bad += [dict(good, overlap=x) for x in (True, 0, -1, 51, 1.0, '1')]
        bad += [dict(good, previous_actions=x) for x in
                ([], [[0.] * 13], [[0.] * 14] * 2, [[float('nan')] * 14], [[float('inf')] * 14])]
        for value in bad:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                rtc.validate_prefix(value, 50)

    def test_mock_explicitly_rejects_rtc(self):
        runtime = server.Runtime('mock', mock=True)
        self.assertFalse(runtime.rtc['supported'])
        with self.assertRaisesRegex(RuntimeError, 'RTC unsupported'):
            runtime.infer({'rtc': dict(previous_actions=[[0.] * 14], overlap=1)})


@unittest.skipIf(torch is None, 'torch is installed only in the model runtime')
class GuidanceTests(unittest.TestCase):
    def processor(self):
        return SimpleNamespace(rtc_config=SimpleNamespace(execution_horizon=1, max_guidance_weight=10.),
                               get_prefix_weights=lambda start, end, total: torch.tensor([1., 0.]))

    def test_vjp_includes_denoiser_derivative_and_masks_padding(self):
        latent = torch.ones(1, 2, 32)
        previous = torch.full((1, 1, 32), 2.)
        # The caller uses no_grad, as does PI05.sample_actions. Guidance must
        # locally enable the graph before evaluating the actual denoiser.
        with torch.no_grad():
            result = rtc.guided_velocity(self.processor(), latent, previous, 0, .5, lambda x: x * .25, 1)
        expected = .25 - 2 * ((2 - .875) * .875)
        torch.testing.assert_close(result[0, 0, :14], torch.full((14,), expected))
        torch.testing.assert_close(result[0, 0, 14:], torch.full((18,), .25))
        torch.testing.assert_close(result[0, 1], torch.full((32,), .25))
        self.assertFalse(result.requires_grad)

    def test_initial_step_is_finite_and_prefix_changes_sample(self):
        latent = torch.ones(1, 2, 32)
        previous = torch.zeros(1, 1, 32)
        result = rtc.guided_velocity(self.processor(), latent, previous, 0, 1., lambda x: x * .25, 1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(result[0, 0, 0], .25)

    def test_no_previous_chunk_is_exact_base_velocity(self):
        latent = torch.ones(1, 2, 32)
        result = rtc.guided_velocity(self.processor(), latent, None, None, .5, lambda x: x * .25)
        torch.testing.assert_close(result, latent * .25)

    def test_nonzero_delay_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'inference_delay=0'):
            rtc.guided_velocity(self.processor(), torch.ones(1, 2, 32), torch.ones(1, 1, 32),
                                1, .5, lambda x: x, 1)

    def test_configures_real_rtc_processor_and_freezes_only_weights(self):
        try:
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
        except ImportError as exc:
            self.skipTest(f'LeRobot RTC unavailable: {exc}')

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)
                self.config = SimpleNamespace(compile_model=False)
                self.model.config = self.config
                self.model._rtc_enabled = lambda: self.config.rtc_config.enabled

            def init_rtc_processor(self):
                pass

        policy = Policy()
        processor = rtc.configure_rtc(policy)
        self.assertIsInstance(processor, RTCProcessor)
        self.assertIs(policy.model.rtc_processor, processor)
        self.assertTrue(policy.model._rtc_enabled())
        self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
        self.assertEqual(processor.denoise_step.__func__, rtc.guided_velocity)
        weights = processor.get_prefix_weights(0, 3, 5)
        torch.testing.assert_close(weights, torch.tensor([.75, .5, .25, 0., 0.]))

    def test_saved_action_normalizer_round_trip(self):
        try:
            from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
            from lerobot.processor import (AddBatchDimensionProcessorStep, NormalizerProcessorStep,
                                          PolicyProcessorPipeline, UnnormalizerProcessorStep)
            from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        except ImportError as exc:
            self.skipTest(f'Full LeRobot processor dependencies unavailable: {exc}')
        features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
        mapping = {FeatureType.ACTION: NormalizationMode.QUANTILES}
        stats = {'action': {'q01': torch.arange(14).float(), 'q99': torch.arange(14).float() + 4}}
        pre = PolicyProcessorPipeline(steps=[AddBatchDimensionProcessorStep(), NormalizerProcessorStep(
            features=features, norm_map=mapping, stats=stats)])
        post = PolicyProcessorPipeline(steps=[UnnormalizerProcessorStep(
            features=features, norm_map=mapping, stats=stats)],
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        physical = np.stack([np.arange(14) + 1., np.arange(14) + 3.]).astype(np.float32)
        processed, prefix = rtc.prepare_prefix({'observation.state': torch.zeros(14)}, physical, pre, torch, 32)
        self.assertNotIn('action', processed)
        self.assertEqual(tuple(prefix.shape), (1, 2, 32))
        torch.testing.assert_close(prefix[0, 0, :14], torch.full((14,), -.5))
        torch.testing.assert_close(prefix[0, 1, :14], torch.full((14,), .5))
        self.assertEqual(prefix[..., 14:].count_nonzero().item(), 0)
        torch.testing.assert_close(post(prefix[..., :14])[0], torch.from_numpy(physical))

    def test_runtime_passes_prefix_then_resets_to_independent_request(self):
        runtime = server.Runtime('test', mock=True)
        runtime.mock = False
        runtime.torch = torch
        runtime.rtc = dict(supported=True, protocol='prefix-v1')
        runtime.supported_tasks = [server.CONTRACT['task'], 'another task']
        calls = []
        prompts = []

        class Policy:
            config = SimpleNamespace(chunk_size=50, max_action_dim=32)

            def reset(self):
                pass

            def predict_action_chunk(self, batch, **kwargs):
                calls.append((kwargs, torch.is_inference_mode_enabled()))
                prompts.append(batch['task'])
                return torch.zeros(1, 50, 14)

        runtime.policy = Policy()
        runtime.pre = lambda batch: dict(batch, action=batch['action'] * 2) if 'action' in batch else batch
        runtime.post = lambda actions: actions
        buf = io.BytesIO()
        Image.new('RGB', (256, 256)).save(buf, format='JPEG')
        payload = dict(state=[0.] * 14, sequence=1,
                       images={name: base64.b64encode(buf.getvalue()).decode()
                               for name in server.CONTRACT['image_crops_xyxy']})
        guided = runtime.infer(dict(payload, rtc=dict(previous_actions=[[.2] * 14] * 3, overlap=3)))
        self.assertEqual(guided['rtc'], dict(overlap=3))
        self.assertFalse(calls[0][1])
        self.assertEqual(calls[0][0]['execution_horizon'], 3)
        self.assertEqual(calls[0][0]['inference_delay'], 0)
        torch.testing.assert_close(calls[0][0]['prev_chunk_left_over'][0, :, :14], torch.full((3, 14), .4))
        independent = runtime.infer(payload)
        self.assertNotIn('rtc', independent)
        self.assertEqual(calls[1], ({}, True))
        self.assertEqual(independent['task'], server.CONTRACT['task'])
        selected = runtime.infer(dict(payload, task='another task'))
        self.assertEqual(selected['task'], 'another task')
        self.assertEqual(prompts, [server.CONTRACT['task'], server.CONTRACT['task'], 'another task'])


if __name__ == '__main__':
    unittest.main()
