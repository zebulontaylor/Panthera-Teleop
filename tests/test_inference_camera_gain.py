"""Hardware gain plumbing and dashboard interaction without changing devices."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import dashboard as capture
import inference_dashboard as inference
from test_inference_playback_ui import HARNESS as BASE_HARNESS, HTML, NODE

HARNESS = BASE_HARNESS.replace("return {value:", "return {classList:{toggle(){},remove(){}},replaceChildren(){},value:").replace(
    "vm.runInContext(process.argv[2],context)",
    "vm.runInContext('(async()=>{'+process.argv[2]+'})()',context)")


class CameraGainIntegrationTests(unittest.TestCase):
    def test_capture_status_reports_gain_and_updates_readback(self):
        cameras = capture.Cameras.__new__(capture.Cameras)
        cameras.config = {'cameras': {'overhead': '/dev/fake'}}
        cameras.lock = threading.Lock()
        cameras.gain_lock = threading.Lock()
        cameras.latest, cameras.errors, cameras.gains = {}, {}, {}
        gain = dict(supported=True, writable=True, min=0, max=100, step=1, value=42)
        with patch.object(capture, 'set_gain', return_value=gain) as setter:
            self.assertEqual(cameras.set_gain('overhead', 42), gain)
            setter.assert_called_once_with('/dev/fake', 42)
        self.assertEqual(cameras.status()['overhead']['gain'], gain)
        with self.assertRaises(ValueError):
            cameras.set_gain('../other', 42)

    def test_controller_and_supervisor_delegate(self):
        controller = inference.Controller('unused')
        with self.assertRaises(RuntimeError):
            controller.set_camera_gain('overhead', 12)
        controller.hardware = Mock()
        controller.set_camera_gain('overhead', 12)
        controller.hardware.cameras.set_gain.assert_called_once_with('overhead', 12)
        with self.assertRaises(ValueError):
            controller.set_camera_gain('unknown', 12)
        supervisor = inference.SessionSupervisor.__new__(inference.SessionSupervisor)
        supervisor.lock, supervisor.recovery_worker = threading.RLock(), None
        supervisor.rpc = Mock()
        supervisor.set_camera_gain('overhead', 12)
        supervisor.rpc.assert_called_once_with('set_camera_gain', 'overhead', 12)

    def test_endpoint_requires_session_and_propagates_errors(self):
        controller = inference.Controller('unused')
        controller.set_camera_gain = Mock()
        server = inference.make_server(controller, 0)
        port = server.server_port
        server.server_close()
        server = inference.make_server(controller, port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{port}'
        with urlopen(base + '/api/status') as response:
            token = json.load(response)['token']
        body = b'{"camera":"overhead","gain":42}'
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base + '/api/camera/gain', data=body))
        self.assertEqual(error.exception.code, 403)
        controller.set_camera_gain.assert_not_called()
        headers = {'X-Session-Token': token}
        with urlopen(Request(base + '/api/camera/gain', data=body, headers=headers)) as response:
            self.assertEqual(response.status, 202)
        controller.set_camera_gain.assert_called_once_with('overhead', 42)
        controller.set_camera_gain.side_effect = ValueError('Invalid gain')
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base + '/api/camera/gain', data=body, headers=headers))
        self.assertEqual(error.exception.code, 409)


@unittest.skipUnless(NODE, 'Node.js required')
class CameraGainUITests(unittest.TestCase):
    def run_js(self, script):
        result = subprocess.run([NODE, '-e', HARNESS, str(HTML), script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gain_range_and_unsaved_edits(self):
        self.run_js("""
const s=status({cameras:{overhead:{ok:true,age_s:0,gain:{supported:true,value:12,min:0,max:100,step:1}}}});
render(s);
assert.equal($('gain-overhead').disabled,false);
assert.equal($('gain-overhead').value,'12');
assert.equal($('gain-overhead').max,100);
$('gain-overhead').value='42';$('gain-overhead').oninput();render(s);
assert.equal($('gain-overhead').value,'42');
assert.equal($('gain-wrist_port2').disabled,true);
render(status({connected:false}));assert.equal($('gain-overhead').disabled,true);
""")

    def test_apply_posts_camera_and_validates_input(self):
        self.run_js("""
const s=status({cameras:{overhead:{ok:true,age_s:0,gain:{supported:true,value:12,min:0,max:100,step:2}}}});
render(s);const requests=[];
fetch=async(url,options={})=>{if(options.method==='POST')requests.push({url,body:JSON.parse(options.body)});return {ok:true,json:async()=>s};};
for(const value of ['', '101', 'NaN', '13']){
 $('gain-overhead').value=value;await $('gain-apply-overhead').onclick();
}
assert.equal(requests.length,0);
$('gain-overhead').value='42';await $('gain-apply-overhead').onclick();
assert.equal(requests[0].url,'/api/camera/gain');
assert.deepEqual(requests[0].body,{camera:'overhead',gain:42});
assert.equal(gainPending.size,0);
""")
