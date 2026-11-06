"""Dashboard automatic right-arm freeze contract without hardware access."""
from pathlib import Path
import subprocess
import unittest

from test_inference_alignment_ui import HARNESS, HTML, NODE


@unittest.skipUnless(Path(NODE).is_file(), 'Node.js runtime unavailable')
class AutoFreezeUITests(unittest.TestCase):
    def run_js(self, script):
        result = subprocess.run([NODE, '-e', HARNESS, str(HTML), script],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_supported_disconnected_toggle_posts_and_default_is_off(self):
        self.run_js(r"""(async()=>{
const supported=status({connected:false,right_arm_auto_freeze_supported:true,right_arm_auto_freeze:{enabled:false,status:'disabled',reason:'',stable_s:0,settle_s:1}});
render(supported);
assert.equal($('auto-freeze-right').checked,false);
assert.equal($('auto-freeze-right').disabled,false);
assert.equal($('freeze-right').disabled,true);
const posts=[];
fetch=async(url,options={})=>{if(options.method==='POST')posts.push([url,JSON.parse(options.body)]);return {ok:true,json:async()=>supported};};
$('auto-freeze-right').checked=true;
await $('auto-freeze-right').onchange();
assert.equal(posts.length,1);
assert.equal(posts[0][0],'/api/auto-freeze-right');
assert.equal(posts[0][1].enabled,true);
render(status());
assert.equal($('auto-freeze-right').disabled,true);
assert.equal($('auto-freeze-right').checked,false);
})()""")

    def test_settle_progress_reason_and_manual_unfreeze(self):
        self.run_js(r"""(async()=>{
const current=status({right_arm_frozen:true,right_arm_auto_freeze_supported:true,right_arm_auto_freeze:{enabled:true,status:'settling',reason:'Waiting for measured joints',stable_s:.45,settle_s:1}});
render(current);
assert.equal($('auto-freeze-right').checked,true);
assert.match($('auto-freeze-right-status').textContent,/settling.*0.5 \/ 1.0 s.*Waiting for measured joints/);
const posts=[];
fetch=async(url,options={})=>{if(options.method==='POST')posts.push([url,JSON.parse(options.body)]);return {ok:true,json:async()=>current};};
await $('freeze-right').onclick();
assert.equal(posts[0][0],'/api/freeze-right');
assert.equal(posts[0][1].frozen,false);
})()""")

    def test_visual_nudge_detection_is_advisory_and_configurable_disconnected(self):
        self.run_js(r"""(async()=>{
render(status());
assert.equal($('pin-nudge-detection').disabled,true);
assert.equal($('pin-nudge-detection').checked,false);
const current=status({connected:false,pin_nudge_detection_supported:true,pin_nudge_detection:{enabled:false,status:'disabled',needs_nudge:null,reason:'',frame_count:0,span_s:0}});
render(current);
assert.equal($('pin-nudge-detection').disabled,false);
assert.equal($('pin-nudge-detection').checked,false);
const posts=[];
fetch=async(url,options={})=>{if(options.method==='POST')posts.push([url,JSON.parse(options.body)]);return {ok:true,json:async()=>current};};
$('pin-nudge-detection').checked=true;
await $('pin-nudge-detection').onchange();
assert.equal(posts.length,1);
assert.equal(posts[0][0],'/api/pin-nudge-detection');
assert.equal(posts[0][1].enabled,true);
for(const [value,expected] of [[true,'Nudge suggested'],[false,'Visually aligned'],[null,'Waiting for visual evidence']]){
 renderPinNudgeDetection({pin_nudge_detection_supported:true,pin_nudge_detection:{enabled:true,status:'collecting',needs_nudge:value,reason:'New frames required',frame_count:3,span_s:.4}});
 assert.ok($('pin-nudge-detection-status').textContent.startsWith(expected));
 assert.match($('pin-nudge-detection-status').textContent,/New frames required.*3 frames \/ 0.4 s/);
}
assert.ok(document.getElementById('pin-nudge-detection-note'));
})()""")
        self.assertIn('does not establish pin contact or start motion', HTML.read_text())
