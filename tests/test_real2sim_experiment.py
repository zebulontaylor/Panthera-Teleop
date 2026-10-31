"""Check offline target sequencing independently of MuJoCo integration."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
try:
    import numpy as np
    import mujoco
    from PIL import Image
except ImportError as exc:
    raise unittest.SkipTest(f'Optional real2sim runtime unavailable: {exc}')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_real2sim as runner


class FakeBridge:
    def __init__(self, fail=False):
        self.model = SimpleNamespace(opt=SimpleNamespace(disableflags=128))
        self.calls = []
        self.fail = fail

    def reset_dynamic(self, state):
        self.initial = state.copy()

    def violations(self, action):
        return []

    def step_action(self, action, dt):
        self.calls.append((action.copy(), dt, self.model.opt.disableflags))
        if self.fail:
            raise RuntimeError('injected failure')
        self.value = action.copy()
        self.value[0] += .2

    def native_state(self):
        return self.value


class ExperimentTests(unittest.TestCase):
    def episode(self):
        state = np.zeros((3, 14))
        actions = np.zeros((3, 14))
        actions[:, 0] = [1, 2, 3]
        return SimpleNamespace(state=state, action=actions, fps=20.)

    def test_exact_successor_targets_duration_and_rmse(self):
        bridge = FakeBridge()
        trajectory, metrics = runner.replay_dynamic(bridge, self.episode(), .11, False)
        self.assertEqual(len(trajectory), 2)
        np.testing.assert_array_equal(bridge.initial, np.zeros(14))
        self.assertEqual([a[0][0] for a in bridge.calls], [1, 2])
        self.assertEqual([a[1] for a in bridge.calls], [.05, .05])
        self.assertTrue(all(a[2] & int(runner.mujoco.mjtDisableBit.mjDSBL_CONTACT) for a in bridge.calls))
        self.assertEqual(bridge.model.opt.disableflags, 128)
        self.assertAlmostEqual(metrics['arm_rmse_rad'], .2 / np.sqrt(12))
        self.assertAlmostEqual(metrics['arm_per_joint_rmse_rad'][0][0], .2)
        self.assertEqual(metrics['arm_per_joint_rmse_rad'][1], [0] * 6)

    def test_flags_restored_on_failure(self):
        bridge = FakeBridge(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            runner.replay_dynamic(bridge, self.episode(), 1., False)
        self.assertEqual(bridge.model.opt.disableflags, 128)

    def test_contact_ablation_preserves_unrelated_flags(self):
        bridge = FakeBridge()
        bridge.model.opt.disableflags |= int(runner.mujoco.mjtDisableBit.mjDSBL_CONTACT)
        original = bridge.model.opt.disableflags
        runner.replay_dynamic(bridge, self.episode(), .05, True)
        self.assertEqual(bridge.calls[0][2], original & ~int(runner.mujoco.mjtDisableBit.mjDSBL_CONTACT))
        self.assertEqual(bridge.model.opt.disableflags, original)

    def test_one_pair_render_selection(self):
        np.testing.assert_array_equal(runner.comparison_indices(1), [0])
        np.testing.assert_array_equal(runner.comparison_indices(2), [0, 1])
        self.assertTrue(np.all(runner.comparison_indices(12) < 12))


if __name__ == '__main__':
    unittest.main()
