"""Exercise playback settings in the actual dashboard JavaScript without a browser."""
from pathlib import Path
import shutil
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[1] / 'app/web/inference.html'
NODE = shutil.which('node') or shutil.which('nodejs')

HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const elements = new Map();
function element() {
  return {value:'', dataset:{}, style:{}, parentElement:{setAttribute(){}},
    previousElementSibling:{style:{}}, setAttribute(){}, append(){}};
}
function get(id) {
  if (!elements.has(id)) {
    const el = element();
    const tag = html.match(new RegExp('<[^>]+id="'+id+'"[^>]*>'))?.[0] || '';
    el.value = tag.match(/value="([^"]*)"/)?.[1] || '';
    elements.set(id, el);
  }
  return elements.get(id);
}
const presets = [...html.matchAll(/data-count="(\d+)"/g)].map(m=>{
  const el=element();el.dataset.count=m[1];return el;
});
const document = {getElementById:get, createElement:element, addEventListener(){},
  querySelector:()=>element(), querySelectorAll:selector=>selector==='[data-count]'?presets:[]};
const context = vm.createContext({assert,document,window:{addEventListener(){}},
  setInterval(){},setTimeout,console});
let source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
// Disable only the automatic network poll; functions and handlers remain real.
source=source.replace('counts();poll();setInterval(poll,400);','counts();');
vm.runInContext(source,context);
vm.runInContext(`
const status=(overrides={})=>({token:'test-token',mode:'holding',connected:true,actions_per_step:15,
 blend_actions:10,cameras:{},step_count:15,action_index:0,...overrides});
token='test-token';
`,context);
(async()=>{await vm.runInContext(process.argv[2],context);})().catch(e=>{console.error(e);process.exitCode=1;});
"""


@unittest.skipUnless(NODE, 'Node.js is required for dashboard JavaScript tests')
class PlaybackSettingsUITests(unittest.TestCase):
    def run_js(self, script):
        result = subprocess.run([NODE, '-e', HARNESS, str(HTML), script],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_first_connected_status_loads_saved_settings_once(self):
        self.run_js("""
render(status({connected:false,actions_per_step:1}));
assert.equal(count(),15);
render(status({actions_per_step:23,blend_actions:6}));
assert.equal(count(),23);assert.equal(blend(),6);
render(status({actions_per_step:1,blend_actions:0}));
assert.equal(count(),23);assert.equal(blend(),6);
""")

    def test_input_and_preset_edits_before_connection_survive_initialization(self):
        for edit in ("$('count').value='27';$('count').oninput();",
                     "document.querySelectorAll('[data-count]').find(b=>b.dataset.count==='5').onclick();"):
            with self.subTest(edit=edit):
                expected = 27 if "27" in edit else 5
                self.run_js(f"""
render(status({{connected:false}}));
{edit}
$('blend').value='3';$('blend').oninput();
render(status({{actions_per_step:1,blend_actions:10}}));
assert.equal(count(),{expected});assert.equal(blend(),3);
""")
        self.run_js("""
$('count').value='27';$('count').oninput();
render(status({actions_per_step:1,blend_actions:6}));
assert.equal(count(),27);assert.equal(blend(),6);
""")

    def test_stop_and_play_preserve_settings_across_async_recording_start(self):
        self.run_js("""
(async()=>{
render(status({rollout_active:true,playing:true}));
$('count').value='29';$('count').oninput();
$('blend').value='4';$('blend').oninput();
let latest=status({rollout_active:false}),requests=[];
fetch=async(url,options={})=>{
 if(options.method==='POST'){
  const body=JSON.parse(options.body);requests.push({url,body});
  if(url==='/api/collection/begin'){
   // Simulate a status render arriving while begin recording is in flight.
   latest=status({rollout_active:true,actions_per_step:1,blend_actions:10});
   render(latest);
   $('count').value='8';$('blend').value='2';
  }
 }
 return {ok:true,json:async()=>latest};
};
await $('stop').onclick();
assert.equal(count(),29);assert.equal(blend(),4);
await $('play').onclick();
const play=requests.find(r=>r.url==='/api/play');
assert.ok(play,'Play must be submitted after recording begins');
assert.equal(play.body.count,29);assert.equal(play.body.blend,4);
})()
""")


if __name__ == '__main__':
    unittest.main()
