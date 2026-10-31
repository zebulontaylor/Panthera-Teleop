"""Conversion and actual MuJoCo integration checks for the offline bridge."""
import importlib.util
import os
from pathlib import Path
import unittest
try:
    import numpy as np
    import mujoco
except ImportError as exc:
    raise unittest.SkipTest(f'Optional real2sim dependencies unavailable: {exc}') from exc

spec = importlib.util.spec_from_file_location('bridge', Path(__file__).parents[1]/'scripts/real2sim_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = Path(os.environ.get('REAL2SIM_SCENE', str(bridge.DEFAULT_SCENE))).expanduser()
        if not scene.is_file():
            raise unittest.SkipTest(f'Optional real2sim scene unavailable: {scene}')
        cls.sim = bridge.Real2SimBridge(scene=scene)

    def pose(self):
        return np.array([0.,1.2,1.4,0.,0.,0.,1., 0.,1.2,1.4,0.,0.,0.,1.])

    def test_mapping_roundtrip_and_no_arm_clipping(self):
        q = self.pose()
        q[1] = -.05
        mapped = self.sim.map_state(q)
        np.testing.assert_array_equal(mapped[bridge.ARM_COLUMNS], q[bridge.ARM_COLUMNS])
        np.testing.assert_allclose(mapped[[6,13]], .02)
        np.testing.assert_allclose(self.sim.inverse_state(mapped), q)
        self.assertEqual(self.sim.violations(q)[0]['index'], 1)
        self.sim.set_pose(q)
        self.assertEqual(self.sim.data.qpos[self.sim.arm_qadr[0,1]], -.05)
        with self.assertRaises(ValueError): self.sim.map_state([float('nan')]*14)

    def test_fk_base_invariance_and_next_action(self):
        q = self.pose()
        self.sim.reset_dynamic(q)
        p, r = self.sim.local_tcp()
        np.testing.assert_allclose(p[0], p[1], atol=1e-12)
        np.testing.assert_allclose(r[0], r[1], atol=1e-12)
        np.testing.assert_allclose(self.sim.native_state(), q, atol=1e-12)
        q[0] += .01
        before = self.sim.data.time
        self.sim.step_action(q)
        self.assertAlmostEqual(self.sim.data.time-before, .05)
        self.assertGreater(self.sim.native_state()[0], 0)
        np.testing.assert_allclose(self.sim.data.ctrl[self.sim.ctrl_ids], self.sim.map_state(q))
        self.assertIsInstance(self.sim.contact_summary()['contacts'], list)

if __name__ == '__main__': unittest.main()
