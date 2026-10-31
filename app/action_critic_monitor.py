"""Stdlib-only, bounded asynchronous adapter to the local torch sidecar."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import selectors
import os
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


class ActionCriticMonitor:
    def __init__(self, callback=None, python=None, checkpoint=None, device='auto', timeout_s=15):
        self.timeout_s = float(timeout_s)
        if not 0 < self.timeout_s < float('inf'):
            raise ValueError('timeout_s must be finite and positive')
        self.callback = callback or (lambda result: None)
        self.command = [str(python or ROOT/'work/state-critic-20261028/venv/bin/python'), '-u', str(ROOT/'app/action_critic_worker.py'), '--checkpoint', str(checkpoint or ROOT/'work/action-critic-mixed-20261029/run/best.pt'), '--device',device]
        self._condition = threading.Condition()
        self._pending = None
        self._process = None
        self._closed = False
        self._generation = 0
        self._request_id = 0
        self._thread = threading.Thread(target=self._run,name='action-critic-monitor',daemon=True)
        self._thread.start()

    def submit(self, images, state, actions, fps=20, **identity):
        # Copy small numeric inputs; image strings are immutable and shared.
        with self._condition:
            if self._closed:
                return None
            self._request_id += 1
            request={'images':dict(images),'state':list(state),'actions':[list(row) for row in actions], 'fps':fps,'request_id':self._request_id,'generation':self._generation}
            self._pending=(request,dict(identity),time.monotonic())
            self._condition.notify()
            return self._request_id

    def invalidate(self):
        with self._condition:
            self._generation += 1
            self._pending = None

    def _publish(self, result, request, identity, started):
        with self._condition:
            # Suppress both obsolete generations and superseded proposals.
            if self._closed or request['generation'] != self._generation or request['request_id'] != self._request_id:
                return
        result.update(identity)
        result.update(request_id=request['request_id'],generation=request['generation'],total_latency_ms=(time.monotonic()-started)*1000)
        try:
            self.callback(result)
        except Exception:
            pass  # A display callback must never break the worker lifecycle.

    def _run(self):
        process=None
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    request,identity,started=self._pending
                    self._pending=None
                try:
                    if process is None:
                        # Inherited stderr avoids a pipe filling or mixing logs with JSONL.
                        process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
                        with self._condition:
                            self._process=process
                            closed=self._closed
                        if closed:
                            return
                    process.stdin.write(json.dumps(request,allow_nan=False)+'\n')
                    process.stdin.flush()
                    line=self._read_response(process)
                    if not line:
                        raise RuntimeError('critic worker exited')
                    result=json.loads(line)
                    if result.get('request_id') != request['request_id'] or result.get('generation') != request['generation']:
                        raise RuntimeError('critic response identity mismatch')
                except Exception as exc:
                    result={'status':'error','reason':str(exc)[:500]}
                    self._stop(process)
                    process=None
                    with self._condition:
                        self._process=None
                self._publish(result,request,identity,started)
        finally:
            self._stop(process)

    def _read_response(self, process):
        # Read bytes directly so partial JSON lines cannot bypass the deadline.
        deadline = time.monotonic() + self.timeout_s
        data = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('critic worker response timed out')
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError('critic worker exited')
                data.extend(chunk)
                if len(data) > 1024 * 1024:
                    raise RuntimeError('critic worker response exceeds limit')
                if b'\n' in data:
                    line, extra = bytes(data).split(b'\n', 1)
                    if extra:
                        raise RuntimeError('critic worker emitted unsolicited output')
                    return line.decode('utf-8')

    @staticmethod
    def _stop(process):
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except ProcessLookupError:
            process.wait()
        finally:
            for stream in (process.stdin,process.stdout):
                if stream:
                    stream.close()

    def close(self):
        # Never join inference from the robot/control thread. Reaping is bounded
        # in a daemon cleanup thread and also happens in _run's finally block.
        with self._condition:
            self._closed=True
            self._generation+=1
            self._pending=None
            process=self._process
            self._condition.notify_all()
        if process is not None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            threading.Thread(target=self._reap,args=(process,),daemon=True).start()

    @staticmethod
    def _reap(process):
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
