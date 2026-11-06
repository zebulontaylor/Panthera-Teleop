"""Automatic contact-assist arming UI contract, without hardware access."""
from pathlib import Path
import subprocess
import unittest

from test_inference_alignment_ui import HARNESS, HTML, NODE


@unittest.skipUnless(Path(NODE).is_file(), 'Node.js runtime unavailable')
class AutoPinContactUITests(unittest.TestCase):
    def run_js(self, script):
        result = subprocess.run([NODE, '-e', HARNESS, str(HTML), script],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_arm_disconnected_without_manual_contact_ack(self):
        self.run_js(r"""(async()=>{
render(status());
assert.equal($('auto-pin-contact').disabled,true);
assert.equal($('auto-pin-contact').checked,false);
const current=status({connected:false,auto_pin_contact_supported:true,auto_pin_contact:{enabled:false,status:'disabled'}});
render(current);
assert.equal($('auto-pin-contact').disabled,false);
const posts=[];
fetch=async(url,options={})=>{if(options.method==='POST'&&url==='/api/auto-pin-contact')posts.push([url,JSON.parse(options.body)]);return {ok:true,json:async()=>current};};
$('auto-pin-contact').checked=true;
await $('auto-pin-contact').onchange();
assert.equal(posts.length,1);
assert.equal(posts[0][0],'/api/auto-pin-contact');
assert.equal(posts[0][1].enabled,true);
assert.equal(posts[0][1].load.force_n,.5);
assert.equal(posts[0][1].load.max_duration_s,120);
assert.equal('pin_on_carrier' in posts[0][1].load,false);
})()""")

    def test_validation_and_disable_with_invalid_settings(self):
        self.run_js(r"""(async()=>{
const current=status({auto_pin_contact_supported:true,auto_pin_contact:{enabled:false,status:'disabled'}});
render(current);
const posts=[];
fetch=async(url,options={})=>{if(options.method==='POST'&&url==='/api/auto-pin-contact')posts.push([url,JSON.parse(options.body)]);return {ok:true,json:async()=>current};};
$('force-newtons').value='99';
$('auto-pin-contact').checked=true;
await $('auto-pin-contact').onchange();
assert.equal(posts.length,0);
assert.equal($('auto-pin-contact').checked,false);
assert.match($('force-error').textContent,/between/);
$('auto-pin-contact').checked=false;
await $('auto-pin-contact').onchange();
assert.equal(posts.length,1);
assert.equal(posts[0][1].enabled,false);
assert.equal('load' in posts[0][1],false);
})()""")

    def test_status_and_manual_load_without_extra_confirmation(self):
        self.run_js(r"""(async()=>{
render(status({right_arm_frozen:true,alignment_force_supported:true,auto_pin_contact_supported:true,auto_pin_contact:{enabled:true,status:'active',phase:'nudging',triggered:true,reason:'Motion contact inferred'}}));
assert.equal($('auto-pin-contact').checked,true);
assert.match($('auto-pin-contact-status').textContent,/active.*triggered.*nudging.*Motion contact inferred/);
assert.equal(forceActionReason('start'),'');
assert.equal($('force-start').disabled,false);
})()""")
        self.assertIn('Turning this off stops assist without resuming policy', HTML.read_text())
