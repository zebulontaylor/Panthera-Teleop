import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from action_critic_monitor import ActionCriticMonitor


class MonitorTests(unittest.TestCase):
    def test_latest_and_invalidation_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'worker.py'
            worker.write_text('import sys,json,time\nfor line in sys.stdin:\n r=json.loads(line); time.sleep(.15); print(json.dumps(dict(status="ok",request_id=r["request_id"],generation=r["generation"])),flush=True)\n')
            results=[]
            event=threading.Event()
            monitor=ActionCriticMonitor(lambda r:(results.append(r),event.set()),python=sys.executable)
            monitor.command=[sys.executable,'-u',str(worker)]
            monitor.submit({},[0]*14,[[0]*14]*20)
            time.sleep(.05)
            monitor.invalidate()
            monitor.submit({},[0]*14,[[0]*14]*20,chunk_id=2)
            monitor.submit({},[0]*14,[[0]*14]*20,chunk_id=3)
            self.assertTrue(event.wait(3))
            self.assertEqual([r['chunk_id'] for r in results],[3])
            started=time.monotonic()
            monitor.close()
            self.assertLess(time.monotonic()-started,.1)
            monitor._thread.join(2)
            self.assertFalse(monitor._thread.is_alive())

    def test_partial_response_timeout_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'worker.py'
            worker.write_text('import sys,time\nsys.stdout.write("{partial"); sys.stdout.flush(); time.sleep(60)\n')
            results=[]
            event=threading.Event()
            monitor=ActionCriticMonitor(lambda r:(results.append(r),event.set()),python=sys.executable,timeout_s=.15)
            monitor.command=[sys.executable,'-u',str(worker)]
            monitor.submit({},[0]*14,[[0]*14]*20)
            self.assertTrue(event.wait(2))
            self.assertEqual(results[-1]['status'],'error')
            self.assertIn('timed out',results[-1]['reason'])
            event.clear()
            monitor.submit({},[0]*14,[[0]*14]*20)
            self.assertTrue(event.wait(2))
            self.assertEqual(len(results),2)
            monitor.close()
            monitor._thread.join(2)

    def test_close_during_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'worker.py'
            worker.write_text('import time\ntime.sleep(60)\n')
            monitor=ActionCriticMonitor(python=sys.executable)
            monitor.command=[sys.executable,str(worker)]
            monitor.submit({},[0]*14,[[0]*14]*20)
            time.sleep(.1)
            process=monitor._process
            monitor.close()
            monitor._thread.join(2)
            self.assertIsNotNone(process.poll())

class ResampleTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'worker numpy environment required')
    def test_horizon_and_timestamp_interpolation(self):
        from action_critic_worker import resample_actions
        self.assertIsNone(resample_actions([0]*14, [[1]*14]*19))
        actions = [[(i+1)/20]*14 for i in range(20)]
        sampled = resample_actions([0]*14, actions)
        for i, row in enumerate(sampled):
            self.assertAlmostEqual(float(row[0]), (i+1)/5, places=6)
        with self.assertRaises(ValueError):
            resample_actions([float('nan')]*14, actions)
        with self.assertRaises(ValueError):
            resample_actions([0]*14, actions, 0)

if __name__=='__main__': unittest.main()
