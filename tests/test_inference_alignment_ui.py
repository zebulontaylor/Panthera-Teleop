"""Execute the dashboard JavaScript in a mock DOM; no browser or hardware."""
from pathlib import Path
import shutil
import subprocess
import unittest

HTML=Path(__file__).resolve().parents[1]/'app/web/inference.html'
NODE=shutil.which('node') or shutil.which('nodejs') or '/usr/lib/chatgpt/resources/cua_node/bin/node'
HARNESS=r"""
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const html=fs.readFileSync(process.argv[1],'utf8'),elements=new Map(),imageControl={auto:true,pending:[]};
function element(tag=''){
 const classes=new Set(),attributes={};let value=tag.match(/value="([^"]*)"/)?.[1]||'';
 return {get value(){return value},set value(v){value=String(v)},get src(){return this._src},set src(v){this._src=v;if(imageControl.auto)queueMicrotask(()=>this.onload?.());else imageControl.pending.push(this)},naturalWidth:640,naturalHeight:480,decode:async()=>{},checked:/\bchecked\b/.test(tag),
  style:{},dataset:{},textContent:'',hidden:false,disabled:false,title:'',
  parentElement:{setAttribute(){}},previousElementSibling:{style:{},textContent:'Setting'},
  classList:{toggle(name,force){const yes=force===undefined?!classes.has(name):force;if(yes)classes.add(name);else classes.delete(name);return yes},contains(name){return classes.has(name)},remove(name){classes.delete(name)},add(name){classes.add(name)}},
  getBoundingClientRect(){return {left:0,top:0,width:640,height:480}},setAttribute(k,v){attributes[k]=String(v)},getAttribute(k){return attributes[k]},append(){},replaceChildren(){},addEventListener(){}};
}
function get(id){if(!elements.has(id))elements.set(id,element(html.match(new RegExp('<[^>]+id="'+id+'"[^>]*>'))?.[0]||''));elements.get(id).id=id;return elements.get(id)}
const presets=[...html.matchAll(/data-count="(\d+)"/g)].map(m=>{const e=element();e.dataset.count=m[1];return e});
const document={getElementById:get,createElement:()=>element(),addEventListener(){},
 querySelector:s=>get('query:'+s),querySelectorAll:s=>s==='[data-count]'?presets:[]};
let serial=0;const revoked=[];
class FakeImage{naturalWidth=640;naturalHeight=480;set src(v){this._src=v;queueMicrotask(()=>this.onload())}}
const context=vm.createContext({assert,document,imageControl,window:{addEventListener(){},confirm(){return false}},
 Image:FakeImage,URL:{createObjectURL:()=>`blob:${++serial}`,revokeObjectURL:u=>revoked.push(u)},
 fetch:async()=>({status:409,ok:false}),setTimeout,setInterval(){},setImmediate,queueMicrotask,console,revoked});
let source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.ok(source.includes('counts();poll();setInterval(poll,400);'));
source=source.replace('counts();poll();setInterval(poll,400);','counts();');
new vm.Script(source);vm.runInContext(source,context);
vm.runInContext(`
token='test-token';
const baseConfig={enabled:true,frequency_hz:2,gain:.5,max_step_mm:1,deadband_px:3,hold_s:30,frame_yaw_deg:0,fixed_carrier_hold:true};
const preview=(overrides={})=>({status:'ready',generation:1,frame_id:1,analyzed_image_available:true,image_size_wh:[640,480],pin_px:[100,120],hole_px:[103,124],error_px:[3,4],pixel_error_px:5,nudge_left_mm:[.1,-.2,0],nudge_valid:true,analyzed_age_s:.2,cache:{mode:'tracked',age_s:0},geometry:{label:'Image-plane geometry'},config:{...baseConfig},...overrides});
const status=(overrides={})=>({token:'test-token',mode:'holding',connected:true,playing:false,actions_per_step:15,blend_actions:10,cameras:{},step_count:15,action_index:0,alignment:preview(),right_arm_frozen:false,right_arm_freeze_supported:true,...overrides});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const imageResponse=(id)=>({status:200,ok:true,headers:{get:()=>String(id)},blob:async()=>({})});
`,context);
(async()=>{await vm.runInContext(process.argv[2],context);})().catch(error=>{console.error(error);process.exitCode=1});
"""


@unittest.skipUnless(Path(NODE).is_file(),'Node.js runtime unavailable')
class AlignmentUITests(unittest.TestCase):
    def test_removed_search_controls_and_route_are_unavailable(self):
        html = HTML.read_text()
        self.assertNotIn('spiral', html.lower())
        self.assertNotIn('force-pin-on-carrier', html)
        self.assertNotIn('I confirm the pin is on the carrier', html)
        for name in ('radius', 'turns', 'speed', 'pitch', 'center'):
            self.assertNotIn(f'id="force-{name}"', html)
        self.run_js("""(async()=>{
let posts=[];fetch=async(url,options={})=>{if(options.method==='POST')posts.push(url);return {ok:true,json:async()=>status()};};
await forceCommand('spiral');
assert.deepEqual(posts,[]);
})()""")

    def run_js(self,script):
        result=subprocess.run([NODE,'-e',HARNESS,str(HTML),script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_left_wrist_carrier_cnn_labels_and_nudge_remain_available(self):
        html = HTML.read_text()
        self.assertIn('LEFT wrist · port 3', html)
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
render(status({right_arm_frozen:true,alignment_nudge_supported:true,alignment:preview({camera_role:'left_wrist',hole_source:'left_carrier_cnn_observed',cache:null,captured_hole:{camera_role:'left_wrist',hole_px:[300,300]},pin_source:'unavailable'})}));await tick();
assert.match($('alignment-frame-detail').textContent,/Analyzed LEFT wrist/);
assert.equal($('alignment-cache').textContent,'LEFT CNN · observed');
assert.equal($('alignment-hole-label').textContent,'Hole · LEFT carrier CNN');
assert.equal($('alignment-nudge').disabled,false);
assert.equal($('alignment-hole-marker').hidden,true);
assert.equal($('alignment-pin-label').textContent,'Pin · unavailable');
})()""")

    def test_legacy_right_frame_keeps_right_caption(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);render(status());await tick();
assert.match($('alignment-frame-detail').textContent,/Analyzed RIGHT wrist/);
assert.match(alignmentDisplayImage().alt,/RIGHT-wrist/);
})()""")

    def test_left_waiting_for_pin_shows_image_and_hides_expired_manual_pin(self):
        self.run_js("""(async()=>{
let requests=[];fetch=async url=>{requests.push(url);return imageResponse(1)};
render(status({right_arm_frozen:true,alignment_manual_pin_supported:true,alignment:preview({camera_role:'left_wrist',status:'waiting_for_pin',pin_source:'unavailable',pin_px:null,captured_pin:{pin_px:[10,20]},hole_source:'left_carrier_cnn_observed',nudge_valid:false,nudge_blocked_reason:'Select the pin'})}));await tick();
assert.deepEqual(requests,['/api/alignment/image?frame_id=1']);
assert.equal(alignmentDisplayImage().style.display,'block');
assert.match($('alignment-frame-detail').textContent,/Analyzed LEFT wrist/);
assert.match(alignmentDisplayImage().alt,/LEFT-wrist/);
assert.equal($('alignment-nudge').disabled,true);
assert.equal($('alignment-manual-pin').disabled,false);
assert.equal($('alignment-pin-marker').hidden,true);
assert.equal($('alignment-pin-label').textContent,'Pin · unavailable');
})()""")

    def test_inferred_pin_bottom_labels_and_manual_override_keep_controls(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
const inferred=preview({camera_role:'left_wrist',pin_source:'center_axis_inferred',pin_top_px:[320,100],pin_px:[320,240],pin_bottom_inferred:true,hole_source:'left_carrier_cnn_observed'});
render(status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_manual_pin_supported:true,alignment:inferred}));await tick();
assert.equal($('alignment-pin-label').textContent,'Pin bottom · inferred');
assert.equal($('alignment-pin-detail').textContent,'TOP y 100.0 px · estimated; BOTTOM y 240.0 px · inferred. Both use the fixed gripper centerline.');
assert.equal($('alignment-nudge').disabled,false);
assert.equal($('alignment-manual-pin').disabled,false);
render(status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_manual_pin_supported:true,alignment:preview({...inferred,pin_source:'operator_selected',captured_pin:{pin_px:[320,235]}})}));await tick();
assert.equal($('alignment-pin-label').textContent,'Pin · fixed manually');
assert.equal($('alignment-pin-detail').textContent,'Pin fixed manually · update after moving');
assert.equal($('alignment-nudge').disabled,false);
})()""")

    def test_pooled_and_retained_pin_estimates_are_labeled(self):
        self.run_js("""(async()=>{
let id=1;fetch=async()=>imageResponse(id);
const sample=retained=>preview({frame_id:id,camera_role:'left_wrist',pin_source:'center_axis_inferred',pin_top_px:[335,310],pin_px:[335,290],pin_accumulation:{sample_count:7,retained,last_seen_age_s:1.2}});
render(status({alignment:sample(false)}));await tick();
assert.equal($('alignment-pin-label').textContent,'Pin bottom · inferred · pooled');
assert.match($('alignment-pin-detail').textContent,/7 stationary estimates/);
id=2;render(status({alignment:sample(true)}));await tick();
assert.equal($('alignment-pin-label').textContent,'Pin bottom · inferred · retained');
assert.match($('alignment-pin-detail').textContent,/last estimate 1.2 s ago/);
})()""")

    def test_opencv_observed_and_inferred_holes_keep_nudge_and_capture_controls(self):
        self.run_js("""(async()=>{
let id=1;fetch=async()=>imageResponse(id);
const armed=source=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment:preview({frame_id:id,camera_role:'left_wrist',hole_source:source,hole_backend:'opencv',cache:null})});
render(armed('left_carrier_opencv_observed'));await tick();
assert.equal($('alignment-cache').textContent,'LEFT OpenCV · observed');
assert.equal($('alignment-hole-label').textContent,'Hole · OpenCV');
assert.equal($('alignment-nudge').disabled,false);assert.equal($('alignment-capture').disabled,false);
id=2;render(armed('left_carrier_layout_inferred'));await tick();
assert.equal($('alignment-cache').textContent,'LEFT OpenCV · layout inferred');
assert.equal($('alignment-hole-label').textContent,'Hole · OpenCV');
assert.equal($('alignment-nudge').disabled,false);assert.equal($('alignment-capture').disabled,false);
assert.match($('alignment-capture-detail').textContent,/detector updates/);
})()""")

    def test_stationary_accumulation_is_labeled_and_moving_retained_target_blocked(self):
        self.run_js("""(async()=>{
let id=1;fetch=async()=>imageResponse(id);
const armed=stationary=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment:preview({frame_id:id,camera_role:'left_wrist',hole_backend:'opencv',hole_source:'left_carrier_stationary_accumulated',cache:null,accumulation:{stationary,frames:8,point_count:6,retained_count:3,age_s:1.2}})});
render(armed(true));await tick();
assert.equal($('alignment-cache').textContent,'LEFT OpenCV · retained observation');
assert.equal($('alignment-hole-label').textContent,'Hole · retained');
assert.match($('alignment-accumulation').textContent,/Stationary · 8 frames · 6 points · 3 retained · 1.2 s/);
assert.equal($('alignment-nudge').disabled,false);assert.equal($('alignment-capture').disabled,false);
id=2;render(armed(false));await tick();
assert.match($('alignment-accumulation').textContent,/Moving · current frame only/);
assert.equal($('alignment-nudge').disabled,true);assert.equal($('alignment-capture').disabled,true);
assert.match($('alignment-action-hint').textContent,/both arms to stay steady/);
})()""")

    def test_numbered_bore_selector_captures_exact_displayed_point_without_motion(self):
        self.run_js("""(async()=>{
let posts=[],captured=false;const holes=[{index:0,hole_px:[100,110],observed:true},{index:1,hole_px:[200,220],inferred:true},{index:2,hole_px:[300,330],accumulated:true},{index:3,hole_px:[900,400],observed:true}];
const armed=extra=>status({right_arm_frozen:true,alignment_manual_capture_supported:true,alignment_capture_supported:true,alignment:preview({camera_role:'left_wrist',carrier_holes:holes,target_hole_index:1,target_selection:{mode:'automatic'},accumulation:{stationary:true},...extra})});
fetch=async(url,options={})=>{if(options.method==='POST'){posts.push([url,JSON.parse(options.body)]);captured=true;return {ok:true,json:async()=>({})}}if(url==='/api/status')return {ok:true,json:async()=>armed(captured?{captured_hole:{hole_index:2,camera_role:'left_wrist'}}:{})};return imageResponse(1)};
render(armed());await tick();
assert.equal($('alignment-target-detail').textContent,'Target bore 2 · automatic');
assert.match($('alignment-bore-0').textContent,/observed/);assert.match($('alignment-bore-1').textContent,/inferred/);assert.match($('alignment-bore-2').textContent,/retained/);
assert.equal($('alignment-bore-2').disabled,false);assert.equal($('alignment-bore-3').disabled,true);
await $('alignment-bore-2').onclick();
assert.equal(posts.filter(p=>p[0]==='/api/alignment/capture').length,1);assert.equal(posts.some(p=>p[0].includes('nudge')),false);assert.equal(posts[0][0],'/api/alignment/capture');
assert.equal(posts[0][1].frame_id,1);assert.deepEqual(posts[0][1].hole_px,[300,330]);
assert.equal($('alignment-bore-0').disabled,true);
assert.equal($('alignment-target-detail').textContent,'Target bore 3 · selected by operator');
assert.match($('alignment-target-hint').textContent,/Clear hole/);
})()""")

    def test_five_mm_manual_nudge_default_is_a_cap_with_declared_controller_support(self):
        html=HTML.read_text()
        self.assertIn('id="alignment-max-step" type="number" min="0" max="5" step="0.1" value="5"',html)
        self.run_js("""
render(status({alignment_limits:{max_nudge_mm:5},alignment:preview({config:{...baseConfig,max_step_mm:5}})}));
assert.equal($('alignment-max-step').getAttribute('max'),'5');
assert.equal($('alignment-max-step').value,'5');
assert.equal(alignmentConfig().max_step_mm,5);
$('alignment-max-step').value='5.1';assert.throws(()=>alignmentConfig(),/between 0 and 5/);
$('alignment-max-step').value='3';assert.equal(alignmentConfig().max_step_mm,3);
""")

    def test_connection_controls_are_at_top_and_reset_connects_only_when_needed(self):
        html = HTML.read_text()
        self.assertNotIn('id="connect"', html)
        for control in ('reset', 'recover'):
            self.assertEqual(html.count(f'id="{control}"'), 1)
            self.assertLess(html.index(f'id="{control}"'), html.index('class="workspace"'))
        self.run_js("""(async()=>{
let posts=[];
fetch=async(url,options={})=>{
 if(options.method==='POST')posts.push(url);
 return {ok:true,json:async()=>status({connected:false,mode:'stopped'})};
};
render(status({connected:false,mode:'stopped'}));
assert.equal($('reset').disabled,false);assert.equal($('recover').disabled,false);
await $('reset').onclick();assert.ok(posts.includes('/api/connect'));
posts=[];render(status());await $('reset').onclick();assert.ok(posts.includes('/api/reset'));
for(const extra of [{mode:'connecting'},{mode:'resetting'},{rollout_active:true},{recovering:true}]){
 render(status(extra));assert.equal($('reset').disabled,true);
}
})()""")

    def test_offline_policy_disables_playback_and_allows_operator_load(self):
        self.run_js("""
const local=(extra={})=>status({policy_ready:false,policy_error:'Connection refused',right_arm_frozen:true,alignment_force_supported:true,...extra});
render(local());forceButtons();
for(const id of ['play','step','predict'])assert.equal($(id).disabled,true);
assert.equal($('reset').disabled,false);assert.equal($('recover').disabled,false);
assert.equal($('force-start').disabled,false);
assert.match($('checkpoint').textContent,/Local control.*insertion assist available/);
assert.equal($('checkpoint').title,'Connection refused');
render(local({mode:'force_holding',alignment_force:{active:true,phase:'holding'}}));
assert.equal($('force-stop').disabled,false);
render(status({policy_ready:true}));assert.equal($('play').disabled,false);
""")

    def test_ready_image_is_versioned_and_values_commit_after_matching_image(self):
        self.run_js("""(async()=>{
let urls=[];fetch=async url=>{urls.push(url);return imageResponse(1)};
render(status());assert.equal($('alignment-dx').textContent,'—');await tick();
assert.deepEqual(urls,['/api/alignment/image?frame_id=1']);
assert.equal($('alignment-dx').textContent,'+0.10');assert.equal($('alignment-error-px').textContent,'5.0 px');
assert.match($('alignment-frame-detail').textContent,/frame 1/);
assert.match($('alignment-frame-detail').textContent,/s old/);
assert.match($('alignment-geometry').textContent,/metric scale unvalidated/);
$('alignment-zoom').onclick();assert.equal($('alignment-zoom').getAttribute('aria-pressed'),'true');
assert.ok($('alignment-frame').classList.contains('zoomed'));
})()""")

    def test_real_backend_millisecond_frame_age_is_shown(self):
        self.run_js(r"""(async()=>{
fetch=async()=>imageResponse(1);
render(status({alignment:preview({analyzed_age_s:undefined,frame_age_ms:2000})}));await tick();
assert.match($('alignment-frame-detail').textContent,/2\.0 s old/);
render(status({alignment:preview({analyzed_age_s:undefined,frame_age_ms:6500,status:'stale',nudge_valid:false})}));
assert.match($('alignment-frame-detail').textContent,/6\.5 s old/);
assert.equal($('alignment-dx').textContent,'—');
})()""")

    def test_wrong_frame_header_never_commits_another_frames_values(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(2);render(status());await tick();
assert.equal(alignmentShownId,null);assert.equal($('alignment-dx').textContent,'—');
assert.match($('alignment-error').textContent,/did not match/);
})()""")

    def test_waiting_hole_image_shows_pin_and_explicit_unverified_fallback(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
render(status({alignment:preview({status:'waiting_for_hole',nudge_valid:false,hole_source:'cnn_unverified_fallback',cache:{mode:'lost'},nudge_left_mm:[0,0,0]})}));await tick();
assert.equal($('alignment-image').style.display,'block');assert.equal($('alignment-pin').textContent,'100.0, 120.0');
assert.equal($('alignment-dx').textContent,'—');assert.match($('alignment-hole-label').textContent,/unverified/);
assert.match($('alignment-status').textContent,/waiting for hole/);
})()""")

    def test_retained_valid_delta_survives_transient_states_until_age_expires(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);render(status());await tick();assert.equal($('alignment-dx').textContent,'+0.10');
for(const phase of ['stale','pending','error']){
 render(status({alignment:preview({status:phase,nudge_valid:false,nudge_left_mm:[0,0,0]})}));
 assert.equal($('alignment-dx').textContent,'+0.10');assert.equal($('alignment-panel').dataset.stale,'false');
}
render(status({alignment:preview({nudge_valid:false,nudge_left_mm:[0,0,0],nudge_blocked_reason:'State timestamp unavailable'})}));
assert.equal($('alignment-dx').textContent,'—');assert.match($('alignment-error').textContent,/State timestamp/);
})()""")

    def test_config_edits_survive_poll_and_apply_uses_only_preview_endpoint(self):
        self.run_js("""(async()=>{
render(status({alignment:preview({status:'pending',frame_id:null})}));
$('alignment-gain').value='.8';$('alignment-gain').oninput();
render(status({alignment:preview({status:'pending',frame_id:null,config:{...baseConfig,gain:.1}})}));
assert.equal($('alignment-gain').value,'.8');
let requests=[];fetch=async(url,options={})=>{
 if(options.method==='POST')requests.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>status({alignment:preview({status:'pending',frame_id:null,config:{...baseConfig,gain:.8}})})};
};
await $('alignment-apply').onclick();
const request=requests.find(r=>r.url==='/api/alignment/config');assert.equal(request.body.gain,.8);
assert.ok(requests.every(r=>['/api/alignment/config','/api/heartbeat'].includes(r.url)));
$('alignment-max-step').value='999';let count=requests.length;await $('alignment-apply').onclick();assert.equal(requests.length,count);
assert.match($('alignment-error').textContent,/between/);
})()""")

    def test_reset_invalidates_inflight_image_and_freeze_is_explicit_boolean(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);render(status());await tick();
let requests=[];fetch=async(url,options={})=>{
 if(options.method==='POST')requests.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>status({alignment:preview({status:'pending',generation:2,frame_id:null}),right_arm_freeze_pending:true,right_arm_freeze_target:[1,2,3]})};
};
await $('alignment-reset').onclick();assert.equal(alignmentShownId,'1');assert.equal($('alignment-dx').textContent,'—');
assert.ok(requests.some(r=>r.url==='/api/alignment/reset'));
await $('freeze-right').onclick();
assert.equal(requests.find(r=>r.url==='/api/freeze-right').body.frozen,true);
assert.equal($('freeze-right').getAttribute('aria-pressed'),'false');assert.equal($('freeze-right').disabled,true);
assert.match($('freeze-right-status').textContent,/Applying right-arm hold change/);
render(status({mode:'human',right_arm_frozen:true}));assert.equal($('freeze-right').disabled,true);
})()""")

    def test_stale_or_reset_arriving_during_image_fetch_cannot_restore_old_nudge(self):
        self.run_js("""(async()=>{
let resolveImage;fetch=()=>new Promise(resolve=>{resolveImage=resolve});
render(status());render(status({alignment:preview({status:'stale',nudge_valid:false,analyzed_age_s:6})}));
resolveImage(imageResponse(1));await tick();
assert.equal($('alignment-image').style.display,'block');assert.equal($('alignment-dx').textContent,'—');
assert.equal($('alignment-panel').dataset.stale,'true');
fetch=()=>new Promise(resolve=>{resolveImage=resolve});
render(status({alignment:preview({frame_id:2})}));
clearAlignmentImage('Waiting after reset');
resolveImage(imageResponse(2));await tick();
assert.equal(alignmentShownId,null);assert.equal($('alignment-image').style.display,'none');
assert.equal($('alignment-dx').textContent,'—');
})()""")

    def test_intermediate_controller_without_freeze_capability_never_posts(self):
        self.run_js("""(async()=>{
render(status({right_arm_freeze_supported:undefined,right_arm_frozen:false,right_arm_freeze_pending:false}));
assert.equal($('freeze-right').disabled,true);
assert.equal($('freeze-right-status').textContent,'Controller restart required for right-arm freeze');
let calls=0;fetch=async()=>{calls++;throw Error('must not submit')};
await $('freeze-right').onclick();assert.equal(calls,0);
})()""")

    def test_actual_nudge_is_only_sent_once_on_explicit_click_for_displayed_frame(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment_nudge:{status:'idle'},...extra});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({mode:'nudge_queued',alignment_nudge:{status:'queued',frame_id:1}})};
};
render(armed());await tick();assert.equal(posts.length,0);
assert.equal($('alignment-nudge').disabled,false);assert.match($('alignment-nudge').textContent,/0.22 mm/);
await $('alignment-nudge').onclick();
assert.equal(posts.filter(r=>r.url==='/api/alignment/nudge').length,1);
assert.equal(posts.find(r=>r.url==='/api/alignment/nudge').body.frame_id,1);
assert.equal($('alignment-nudge').disabled,true);assert.match($('alignment-action-hint').textContent,/current nudge/);
})()""")

    def test_nudge_capability_pause_freeze_freshness_and_step_limits_are_guarded(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,...extra});
render(armed());await tick();
for(const extra of [{alignment_nudge_supported:undefined},{playing:true},{right_arm_frozen:false},{mode:'human'},{right_arm_freeze_pending:true},{collection_pending:true}]){
 render(armed(extra));let calls=0;fetch=async()=>{calls++;throw Error('must not submit')};
 assert.equal($('alignment-nudge').disabled,true);await $('alignment-nudge').onclick();assert.equal(calls,0);
}
render(armed({alignment:preview({analyzed_age_s:4.5})}));assert.equal($('alignment-nudge').disabled,false);
render(armed({alignment:preview({analyzed_age_s:5.1})}));assert.equal($('alignment-nudge').disabled,true);
render(armed({alignment:preview({nudge_left_mm:[2.1,0,0],config:{...baseConfig,max_step_mm:2}})}));
assert.match($('alignment-action-hint').textContent,/exceeds/);
$('alignment-max-step').value='2.1';assert.throws(()=>alignmentConfig(),/between/);
})()""")

    def test_capture_uses_exact_frame_and_latches_age_until_clear(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,...extra});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({alignment:preview({status:'pending',generation:2,frame_id:null,captured_hole:{source_frame_id:1,age_s:2.5}})})};
};
render(armed());await tick();assert.equal($('alignment-capture').disabled,false);assert.equal(posts.length,0);
await $('alignment-capture').onclick();assert.equal(posts.find(r=>r.url==='/api/alignment/capture').body.frame_id,1);
assert.match($('alignment-capture-detail').textContent,/source frame 1/);assert.match($('alignment-capture-detail').textContent,/2.5 s old/);
fetch=async()=>imageResponse(2);
render(armed({alignment:preview({frame_id:2,hole_source:'captured_demo_reference',cache:{mode:'captured',age_s:3},captured_hole:{source_frame_id:1,age_s:3}})}));await tick();
assert.equal($('alignment-capture').disabled,true);assert.match($('alignment-capture').title,/already captured/);
assert.equal($('alignment-nudge').disabled,false);assert.equal($('alignment-hole-label').textContent,'Hole · captured');
})()""")

    def test_capture_rejects_unverified_cnn_and_missing_capability_even_if_clicked(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,...extra});
render(armed());await tick();
for(const extra of [{alignment_capture_supported:undefined},{alignment:preview({hole_source:'cnn_unverified_fallback',cache:{mode:'lost'}})}]){
 render(armed(extra));let calls=0;fetch=async()=>{calls++;throw Error('must not submit')};
 assert.equal($('alignment-capture').disabled,true);await $('alignment-capture').onclick();assert.equal(calls,0);
}
})()""")

    def test_auto_insert_sends_one_start_with_dither_and_keeps_stop_enabled(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_force_supported:true,alignment_insert_supported:true,alignment_force:{active:false},...extra});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST'&&url.startsWith('/api/alignment/force/'))posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({mode:'force_auto',alignment_force:{active:true,phase:'nudging',nudge_count:2}})};
};
render(armed());await tick();assert.equal($('force-insert').disabled,false);
await $('force-insert').onclick();
assert.deepEqual(posts[0],{url:'/api/alignment/force/insert',body:{pin_on_carrier:true,force_n:.5,frame_id:1,oscillation_amplitude_mm:.2,oscillation_frequency_hz:15,max_duration_s:120,drop_threshold_mm:3,drop_debounce_s:.2}});
assert.equal(posts.length,1);assert.equal($('force-insert').disabled,true);
assert.equal($('alignment-nudge').disabled,true);assert.equal($('force-stop').disabled,false);
assert.ok($('force-hint').textContent.includes('2 / 3 visual nudge passes'));
await $('force-stop').onclick();assert.equal(posts[1].url,'/api/alignment/force/stop');
})()""")

    def test_auto_insert_displays_axial_retry_progress_and_cap(self):
        self.run_js(r"""(async()=>{
fetch=async()=>imageResponse(1);
for(const phase of ['axial_retry','axial_limit']){
 render(status({right_arm_frozen:true,mode:'force_auto',alignment_force:{active:true,phase,axial_retry_mm:phase==='axial_limit'?15:3.6}}));await tick();
 assert.match($('force-hint').textContent,/toward LEFT base at 2 mm\/s/);
 assert.match($('force-hint').textContent,/15.0 mm commanded/);
 assert.match($('force-hint').textContent,/Downward load stays active/);
 assert.equal($('force-stop').disabled,false);
 if(phase==='axial_limit')assert.match($('force-hint').textContent,/Travel limit reached/);
}
})()""")

    def test_auto_insert_explains_three_pass_sequence_without_extra_waits(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
render(status({right_arm_frozen:true,alignment_force_supported:true,alignment_insert_supported:true,mode:'force_auto',alignment_force:{active:true,phase:'nudging',nudge_count:2}}));await tick();
assert.ok($('force-hint').textContent.includes('2 / 3 visual nudge passes'));
assert.match($('force-hint').textContent,/spaced 1 s apart/);
assert.doesNotMatch($('force-hint').textContent,/settlement|accumulat|detection/);
assert.equal($('force-stop').disabled,false);
})()""")

    def test_auto_insert_allows_missing_detection_but_requires_capability_and_hold(self):
        self.run_js("""(async()=>{
fetch=async()=>imageResponse(1);
const aligned=preview({nudge_left_mm:[0,0,0]});
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_force_supported:true,alignment_insert_supported:true,alignment:aligned,...extra});
render(armed());await tick();forceButtons();
assert.equal($('alignment-nudge').disabled,true);assert.equal($('force-insert').disabled,false);
for(const extra of [{alignment:preview({nudge_valid:false})},{alignment:preview({analyzed_age_s:3})},{alignment:preview({status:'waiting_for_hole',frame_id:null})}]){
 render(armed(extra));await tick();forceButtons();assert.equal($('force-insert').disabled,false);
}
render(armed({mode:'force_auto',alignment_force:{active:true,phase:'waiting_for_preview',preview_unavailable:true,waiting_reason:'Carrier detection unavailable'}}));
assert.match($('force-hint').textContent,/Carrier detection unavailable/);assert.equal($('force-stop').disabled,false);
render(armed({mode:'force_auto',alignment_force:{active:true,phase:'nudging',nudge_count:2,preview_unavailable:false}}));
assert.doesNotMatch($('force-hint').textContent,/Waiting for carrier detection/);
assert.ok($('force-hint').textContent.includes('2 / 3 visual nudge passes'));
for(const extra of [{alignment_insert_supported:undefined},{playing:true},{right_arm_frozen:false}]){
 render(armed(extra));await tick();let calls=0;fetch=async()=>{calls++;throw Error('must not start')};
 assert.equal($('force-insert').disabled,true);await $('force-insert').onclick();assert.equal(calls,0);
}
})()""")

    def test_force_start_requires_capability_pause_and_freeze(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment_force_supported:true,alignment_force:{active:false,phase:'off'},...extra});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({mode:'force_holding',alignment_force:{active:true,phase:'holding',requested_force_n:.5,ramped_force_n:.25,downward_travel_mm:.1,config:{max_duration_s:20,drop_threshold_mm:.8,drop_debounce_s:.2}}})};
};
render(armed());await tick();assert.equal(posts.length,0);assert.equal($('force-start').disabled,false);
assert.equal($('force-start').disabled,false);
await $('force-start').onclick();
const start=posts.find(p=>p.url==='/api/alignment/force/start');
assert.deepEqual(start.body,{pin_on_carrier:true,force_n:.5,max_duration_s:120,drop_threshold_mm:3,drop_debounce_s:.2});
assert.equal($('force-newtons').disabled,true);assert.equal($('force-duration').disabled,true);
assert.equal($('force-requested').textContent,'0.50 N');assert.equal($('force-ramped').textContent,'0.25 N');
assert.equal($('force-start').disabled,true);assert.equal($('force-stop').disabled,false);
})()""")

    def test_force_missing_capability_or_policy_sends_no_start(self):
        self.run_js(r"""(async()=>{
fetch=async()=>imageResponse(1);
const armed=(extra={})=>status({right_arm_frozen:true,alignment_force_supported:true,alignment_force:{active:false},...extra});
render(armed());await tick();
for(const extra of [{alignment_force_supported:undefined},{playing:true},{right_arm_frozen:false},{mode:'human'}]){
 render(armed(extra));let calls=0;fetch=async()=>{calls++;throw Error('must not submit')};
 assert.equal($('force-start').disabled,true);await $('force-start').onclick();assert.equal(calls,0);
}
render(armed({alignment_force_supported:false}));assert.match($('force-hint').textContent,/Update\/restart motor controller/);
})()""")

    def test_force_holding_nudge_is_xy_only_and_capture_clear_are_disabled(self):
        self.run_js("""(async()=>{
let posts=[];const held=()=>status({mode:'force_holding',right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment_force_supported:true,alignment_force:{active:true,phase:'holding',requested_force_n:.5,ramped_force_n:.5},alignment:preview({nudge_left_mm:[.3,.4,.5]})});
fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>held()};
};
render(held());await tick();assert.match($('alignment-nudge').textContent,/Apply XY nudge · 0.50 mm/);
assert.equal($('alignment-dz').textContent,'Loaded');assert.match($('alignment-action-hint').textContent,/Z remains loaded/);
assert.equal($('alignment-capture').disabled,true);assert.equal($('alignment-reset').disabled,true);
await $('alignment-capture').onclick();await $('alignment-reset').onclick();assert.equal(posts.length,0);
await $('alignment-nudge').onclick();assert.equal(posts.find(p=>p.url==='/api/alignment/nudge').body.frame_id,1);
})()""")

    def test_loaded_five_mm_xy_nudge_preserves_downward_force_and_transition_guards(self):
        self.run_js("""(async()=>{
let posts=[];
const held=(extra={})=>status({mode:'force_holding',control_source:'operator_load',right_arm_frozen:true,alignment_nudge_supported:true,alignment_limits:{max_nudge_mm:5},alignment_force:{active:true,phase:'holding',requested_force_n:.8,ramped_force_n:.8},alignment:preview({camera_role:'left_wrist',hole_source:'left_carrier_opencv_observed',nudge_left_mm:[3,4,7],config:{...baseConfig,max_step_mm:5}}),...extra});
fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>held()};
};
render(held());await tick();assert.equal($('alignment-nudge').disabled,false);
assert.match($('alignment-nudge').textContent,/Apply XY nudge · 5.00 mm/);
assert.equal($('alignment-dz').textContent,'Loaded');
await $('alignment-nudge').onclick();
assert.equal(posts.filter(p=>p.url==='/api/alignment/nudge').length,1);
assert.equal(posts.find(p=>p.url==='/api/alignment/nudge').body.frame_id,1);
assert.equal(posts.some(p=>p.url.includes('force/stop')),false);
assert.equal(state.alignment_force.requested_force_n,.8);
for(const extra of [{mode:'force_arming'},{right_arm_frozen:false},{playing:true},{alignment:preview({nudge_valid:false})},{alignment:preview({analyzed_age_s:5.1})}]){
 render(held(extra));assert.equal($('alignment-nudge').disabled,true);
}
render(held({alignment:preview({nudge_left_mm:[3.1,4,0],config:{...baseConfig,max_step_mm:5}})}));assert.equal($('alignment-nudge').disabled,true);
})()""")

    def test_auto_insert_dither_settings_render_validate_and_forward(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_force_supported:true,alignment_insert_supported:true,...extra});
render(armed({alignment_force:{config:{oscillation_amplitude_mm:.4,oscillation_frequency_hz:22}}}));
assert.equal($('force-oscillation-amplitude').value,'0.4');assert.equal($('force-oscillation-frequency').value,'22');
for(const [amplitude,frequency] of [[0,15],[.2,0],[2,40]]){
 $('force-oscillation-amplitude').value=amplitude;$('force-oscillation-frequency').value=frequency;
 assert.equal(forceConfig().oscillation_amplitude_mm,amplitude);assert.equal(forceConfig().oscillation_frequency_hz,frequency);
}
forceButtons();
let posts=[];fetch=async(url,options={})=>{if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});return {ok:true,json:async()=>armed({mode:'force_auto',alignment_force:{active:true,phase:'aligned_oscillation',config:{oscillation_amplitude_mm:.3,oscillation_frequency_hz:18}}})};};
assert.equal($('force-insert').disabled,false);await $('force-insert').onclick();
assert.equal(posts[0].body.oscillation_amplitude_mm,2);assert.equal(posts[0].body.oscillation_frequency_hz,40);
assert.equal($('force-oscillation-amplitude').value,'0.3');assert.equal($('force-oscillation-frequency').value,'18');
assert.deepEqual(Object.keys(forceConfig()).sort(),['force_n','oscillation_amplitude_mm','oscillation_frequency_hz','max_duration_s','drop_threshold_mm','drop_debounce_s'].sort());
})()""")

    def test_stop_load_needs_no_ack_and_force_bounds_reject_without_post(self):
        self.run_js("""(async()=>{
const held=()=>status({mode:'force_holding',right_arm_frozen:true,alignment_force_supported:true,alignment_force:{active:true,phase:'holding',requested_force_n:.5,ramped_force_n:.5},alignment:preview({status:'stale',nudge_valid:false})});
let posts=[];fetch=async(url,options={})=>{
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>status({right_arm_frozen:true,alignment_force_supported:true,alignment_force:{active:false,phase:'off',requested_force_n:0,ramped_force_n:0,reason:'Operator stop'}})};
};
render(held());assert.equal($('force-stop').disabled,false);
await $('force-stop').onclick();assert.deepEqual(posts.find(p=>p.url==='/api/alignment/force/stop').body,{});
assert.equal($('force-reason').textContent,'Operator stop');
for(const [id,value] of [['force-newtons','3.1'],['force-duration','301'],['force-drop','.2'],['force-debounce','.09'],['force-oscillation-amplitude','2.1'],['force-oscillation-amplitude','-.1'],['force-oscillation-frequency','40.1'],['force-oscillation-frequency','-.1']]){
 const previous=$(id).value;$(id).value=value;assert.throws(()=>forceConfig(),/between/);$(id).value=previous;
}
})()""")

    def test_manual_load_needs_no_cnn_and_zero_correction_is_allowed(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_force_supported:true,alignment_nudge_supported:true,...extra});
render(armed({alignment:preview({status:'waiting_for_hole',nudge_valid:false,frame_id:null,cache:{mode:'lost'},hole_source:'cnn_unverified_fallback'})}));
forceButtons();assert.equal($('force-start').disabled,false);
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({mode:'force_holding',alignment_force:{active:true,phase:'holding',requested_force_n:.5},alignment:preview({nudge_left_mm:[0,0,0]})})};
};
await $('force-start').onclick();await tick();
assert.ok(posts.some(p=>p.url==='/api/alignment/force/start'));
assert.equal($('alignment-nudge').disabled,true);
assert.match($('alignment-action-hint').textContent,/no nudge needed/);
})()""")

    def test_manual_click_maps_zoomed_image_to_native_pixels_and_requires_confirm(self):
        self.run_js("""(async()=>{
const lost=()=>status({right_arm_frozen:true,alignment_manual_capture_supported:true,alignment_capture_supported:true,alignment_nudge_supported:true,alignment:preview({generation:1,status:'waiting_for_hole',nudge_valid:false,hole_source:'cnn_unverified_fallback',cache:{mode:'lost'},reason:'No supported carrier reference',nudge_blocked_reason:'No supported carrier reference'})});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>status({right_arm_frozen:true,alignment_manual_capture_supported:true,alignment:preview({status:'pending',generation:2,frame_id:null,captured_hole:{operator_selected:true,source_frame_id:1,age_s:.1}})})};
};
render(lost());await tick();assert.equal($('alignment-manual').disabled,false);assert.equal($('alignment-capture').disabled,true);
$('alignment-manual').onclick();assert.equal(alignmentManualActive,true);assert.equal(posts.length,0);
$('alignment-zoom').onclick();
$('alignment-image').getBoundingClientRect=()=>({left:-200,top:-100,width:800,height:600});
$('alignment-frame').getBoundingClientRect=()=>({left:0,top:0,width:400,height:300});
$('alignment-frame').onclick({clientX:200,clientY:200});
assert.deepEqual(alignmentManualPoint,[320,240]);assert.equal(posts.length,0);
assert.equal($('alignment-manual-marker').style.left,'200px');assert.equal($('alignment-manual-marker').style.top,'200px');
assert.equal($('alignment-manual-confirm').disabled,false);
await $('alignment-manual-confirm').onclick();
assert.deepEqual(posts.find(p=>p.url==='/api/alignment/capture').body,{frame_id:1,hole_px:[320,240]});
assert.equal(alignmentManualActive,false);assert.match($('alignment-capture-detail').textContent,/Operator-selected hole/);
assert.ok(!posts.some(p=>p.url==='/api/alignment/nudge'));
})()""")

    def test_manual_selection_cancels_on_context_hold_or_policy_change(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_manual_capture_supported:true,alignment:preview({generation:1}),...extra});
fetch=async()=>imageResponse(1);
for(const extra of [{playing:true},{right_arm_frozen:false},{right_arm_freeze_pending:true},{mode:'human'},{connected:false},{alignment:preview({generation:2})},{mode:'force_holding',alignment_force:{active:true}}]){
 clearAlignmentImage('Fresh setup');render(armed());await tick();$('alignment-manual').onclick();
 $('alignment-frame').onclick({clientX:200,clientY:200});assert.equal(alignmentManualActive,true);
 render(armed(extra));assert.equal(alignmentManualActive,false);assert.equal(alignmentManualPoint,null);assert.equal($('alignment-manual-marker').hidden,true);
}
})()""")

    def test_retained_fresh_displayed_frame_does_not_chase_latest_id(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment:preview({generation:1}),...extra});
let posts=[];fetch=async url=>imageResponse(1);render(armed());await tick();
fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return new Promise(()=>{});
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({mode:'nudge_queued',alignment_nudge:{status:'queued'}})};
};
render(armed({alignment:preview({generation:1,frame_id:2})}));
assert.equal(alignmentShownId,'1');assert.equal($('alignment-nudge').disabled,false);
assert.match($('alignment-frame-detail').textContent,/retained fresh frame/);
await $('alignment-nudge').onclick();assert.equal(posts.find(p=>p.url==='/api/alignment/nudge').body.frame_id,1);
})()""")

    def test_lost_reference_reason_remains_stable_through_image_loads_and_new_ids(self):
        self.run_js("""(async()=>{
const lost=id=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment:preview({generation:1,frame_id:id,status:'waiting_for_hole',nudge_valid:false,hole_source:'cnn_unverified_fallback',cache:{mode:'lost'},reason:'No supported carrier reference',nudge_blocked_reason:'No supported carrier reference'})});
fetch=async()=>imageResponse(1);render(lost(1));const reason=$('alignment-error').textContent;await tick();
assert.equal($('alignment-error').textContent,reason);assert.equal(reason,'No supported carrier reference');
const hint=$('alignment-action-hint').textContent;
fetch=async()=>new Promise(()=>{});render(lost(2));
assert.equal($('alignment-error').textContent,reason);assert.equal($('alignment-action-hint').textContent,hint);
assert.match(hint,/Set hole manually/);assert.equal($('alignment-capture').disabled,true);
})()""")

    def test_manual_capability_capture_lock_and_action_error_persistence(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_manual_capture_supported:true,...extra});
fetch=async()=>imageResponse(1);render(armed({alignment_manual_capture_supported:undefined}));await tick();
assert.equal($('alignment-manual').disabled,true);$('alignment-manual').onclick();assert.equal(alignmentManualActive,false);
render(armed({alignment:preview({captured_hole:{operator_selected:true},cache:{mode:'captured'}})}));assert.equal($('alignment-manual').disabled,true);
assert.match($('alignment-manual').title,/already captured/);
alignmentUiError='Operator action rejected';fetch=async()=>imageResponse(2);render(armed({alignment:preview({frame_id:2})}));await tick();
assert.equal($('alignment-error').textContent,'Operator action rejected');
})()""")

    def test_actual_dom_double_buffer_keeps_old_image_and_draft_until_decode(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_manual_pin_supported:true,alignment_nudge_supported:true,alignment:preview({generation:1}),...extra});
fetch=async url=>imageResponse(url.includes('=2')?2:1);render(armed());await tick();
const oldUrl=$('alignment-image').src;assert.equal($('alignment-image').style.display,'block');
imageControl.auto=false;render(armed({alignment:preview({generation:1,frame_id:2})}));await tick();
assert.equal($('alignment-image').style.display,'block');assert.equal($('alignment-image-next').style.display,'none');
assert.equal(alignmentShownId,'1');assert.equal($('alignment-nudge').disabled,false);assert.ok(!revoked.includes(oldUrl));
$('alignment-manual-pin').onclick();$('alignment-frame').onclick({clientX:273,clientY:307});
assert.deepEqual(alignmentManualPoint,[273,307]);assert.equal(alignmentManualActive,true);
$('alignment-image-next').onload();await tick();
assert.equal($('alignment-image-next').style.display,'block');assert.equal($('alignment-image').style.display,'none');
assert.equal(alignmentShownId,'2');assert.ok(revoked.includes(oldUrl));assert.equal(alignmentManualActive,true);assert.deepEqual(alignmentManualPoint,[273,307]);
})()""")

    def test_manual_pin_persists_without_blanking_and_can_be_reselected(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_manual_pin_supported:true,alignment_manual_capture_supported:true,alignment:preview({generation:1}),...extra});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(url.includes('=3')?3:url.includes('=2')?2:1);
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({alignment:preview({status:'pending',generation:2,frame_id:null,captured_pin:{pin_px:[273,307],source_frame_id:1},captured_hole:{hole_px:[300,310]}})})};
};
render(armed());await tick();$('alignment-manual-pin').onclick();$('alignment-frame').onclick({clientX:273,clientY:307});
await $('alignment-manual-confirm').onclick();
assert.deepEqual(posts.find(p=>p.url==='/api/alignment/pin').body,{frame_id:1,pin_px:[273,307]});
assert.equal($('alignment-image').style.display,'block');assert.equal(alignmentShownId,'1');
assert.equal($('alignment-pin-detail').textContent,'Pin fixed manually · update after moving');assert.equal($('alignment-pin-marker').hidden,false);
render(armed({alignment:preview({generation:2,frame_id:2,pin_px:[273,307],pin_source:'operator_selected',captured_pin:{pin_px:[273,307],source_frame_id:1}})}));await tick();
render(armed({alignment:preview({generation:2,frame_id:3,pin_px:[200,220],captured_pin:{pin_px:[273,307],source_frame_id:1}})}));await tick();
assert.equal($('alignment-pin').textContent,'273.0, 307.0');assert.equal($('alignment-manual-pin').disabled,false);
$('alignment-manual-pin').onclick();assert.equal(alignmentManualActive,true);assert.equal(alignmentManualTarget,'pin');
})()""")

    def test_manual_pin_capability_and_separate_clear_routes(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_manual_pin_supported:true,alignment:preview({captured_pin:{pin_px:[273,307]},captured_hole:{hole_px:[300,310]}}),...extra});
fetch=async()=>imageResponse(1);render(armed({alignment_manual_pin_supported:undefined}));await tick();
assert.equal($('alignment-manual-pin').disabled,true);$('alignment-manual-pin').onclick();assert.equal(alignmentManualActive,false);
let posts=[];fetch=async(url,options={})=>{
 if(options.method==='POST')posts.push({url,body:JSON.parse(options.body)});
 return {ok:true,json:async()=>armed({alignment:preview({status:'pending',generation:2,frame_id:null,captured_hole:{hole_px:[300,310]}})})};
};
render(armed());assert.equal($('alignment-clear-pin').disabled,false);await $('alignment-clear-pin').onclick();
assert.ok(posts.some(p=>p.url==='/api/alignment/pin/clear'));assert.ok(!posts.some(p=>p.url==='/api/alignment/reset'));
assert.equal(state.alignment.captured_hole.hole_px[0],300);assert.equal($('alignment-image').style.display,'block');
await $('alignment-reset').onclick();assert.ok(posts.some(p=>p.url==='/api/alignment/reset'));
})()""")

    def test_last_valid_sample_stays_actionable_during_pending_error_and_image_updates(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment_capture_supported:true,alignment:preview({generation:1}),...extra});
fetch=async()=>imageResponse(1);render(armed());await tick();
for(const phase of ['pending','stale','error']){
 alignmentAgeBase=1;alignmentAgeAt=Date.now();
 render(armed({alignment:{status:phase,frame_id:null,generation:1,config:{...baseConfig,max_frame_age_ms:1250}}}));
 assert.equal($('alignment-nudge').disabled,false);assert.equal($('alignment-dx').textContent,'+0.10');assert.equal(alignmentShownId,'1');
}
alignmentAgeBase=1.4;alignmentAgeAt=Date.now();render(armed({alignment:{status:'pending',frame_id:null,generation:1,config:{...baseConfig,max_frame_age_ms:1250}}}));
assert.equal($('alignment-nudge').disabled,true);assert.equal($('alignment-dx').textContent,'—');
assert.equal(alignmentActionWindow({config:{frequency_hz:2,max_frame_age_ms:2000}}),2);
assert.equal(alignmentActionWindow({config:{frequency_hz:.5,max_frame_age_ms:2000}}),2);
assert.equal(alignmentActionWindow({config:{frequency_hz:10}}),5);
assert.equal(alignmentActionWindow({config:{frequency_hz:10,max_frame_age_ms:20000}}),10);
})()""")

    def test_clear_hole_preserves_fixed_pin_and_visible_buffer(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_manual_pin_supported:true,alignment:preview({captured_pin:{pin_px:[273,307]},captured_hole:{hole_px:[300,310]}}),...extra});
let posts=[];fetch=async(url,options={})=>{
 if(url.startsWith('/api/alignment/image'))return imageResponse(1);
 if(options.method==='POST')posts.push(url);
 return {ok:true,json:async()=>armed({alignment:preview({status:'pending',generation:2,frame_id:null,captured_pin:{pin_px:[273,307]},captured_hole:null})})};
};
render(armed());await tick();await $('alignment-reset').onclick();
assert.ok(posts.includes('/api/alignment/reset'));assert.ok(!posts.includes('/api/alignment/pin/clear'));
assert.equal($('alignment-image').style.display,'block');assert.equal($('alignment-pin-detail').textContent,'Pin fixed manually · update after moving');
assert.equal($('alignment-pin-marker').hidden,false);assert.equal($('alignment-hole-marker').hidden,true);
})()""")

    def test_force_guard_speed_limit_and_retained_stop_diagnostics_are_visible(self):
        self.run_js(r"""
const diagnostics={tool_speed_m_s:.012,max_tool_speed_m_s:.008,down_speed_m_s:.006,max_down_speed_m_s:.01,velocity_source:'temporal_fk_window',velocity_window_s:.15,lateral_travel_m:.0005,vertical_travel_m:.0013,sdk_tool_speed_m_s:.11};
render(status({mode:'force_holding',alignment_force:{active:true,phase:'holding',diagnostics}}));
assert.equal($('force-guard-speed').textContent,'12.00 mm/s');assert.equal($('force-guard-limit').textContent,'8.00 mm/s');
assert.equal($('force-down-speed').textContent,'6.00 mm/s');assert.equal($('force-down-limit').textContent,'10.00 mm/s');
assert.equal($('force-travel').textContent,'1.30 mm');assert.equal($('force-lateral-travel').textContent,'0.50 mm');
assert.equal($('force-sdk-speed').textContent,'110.00 mm/s');assert.match($('force-velocity-source').textContent,/150 ms window/);
render(status({alignment_force:{active:false,phase:'stopped',reason_code:'tool_speed_exceeded',reason:'tool_speed_exceeded',diagnostics}}));
assert.equal($('force-speed-label').textContent,'Tool speed at stop');assert.equal($('force-guard-speed').textContent,'12.00 mm/s');
assert.match($('force-reason').textContent,/12.00 mm\/s.*limit 8.00 mm\/s.*1.30 mm/);
""")

    def test_force_speed_guards_disabled_remains_visible(self):
        self.run_js("""
render(status({mode:'force_holding',alignment_force:{active:true,diagnostics:{speed_guards_enabled:false,tool_speed_m_s:.08,down_speed_m_s:.03}}}));
assert.equal($('force-guard-limit').textContent,'Disabled');
assert.equal($('force-down-limit').textContent,'Disabled');
assert.equal($('force-speed-label').textContent,'Measured tool speed');
assert.equal($('force-guard-speed').textContent,'80.00 mm/s');
""")

    def test_missing_force_diagnostics_remain_unknown_not_fabricated_zero(self):
        self.run_js("""
render(status({alignment_force:{active:false,phase:'off',reason:'Operator stop'}}));
for(const id of ['force-guard-speed','force-guard-limit','force-down-speed','force-down-limit','force-sdk-speed'])assert.equal($(id).textContent,'—');
assert.equal($('force-reason').textContent,'Operator stop');assert.match($('force-velocity-source').textContent,/No guard velocity/);
""")

    def test_larger_motion_settings_require_declared_controller_limits(self):
        self.run_js("""
render(status());
$('alignment-max-step').value='10';
assert.throws(()=>alignmentConfig(),/between 0 and 2/);
assert.equal($('alignment-max-step').getAttribute('max'),'2');
assert.match($('alignment-limits-note').textContent,/Updated controller required/);
render(status({alignment_limits:{max_nudge_mm:10,max_load_duration_s:200}}));
$('alignment-max-step').value='10';
assert.throws(()=>alignmentConfig(),/between 0 and 5/);$('alignment-max-step').value='5';assert.equal(alignmentConfig().max_step_mm,5);
assert.equal($('alignment-max-step').getAttribute('max'),'5');
assert.equal($('force-duration').getAttribute('max'),'200');
$('force-duration').value='201';assert.throws(()=>forceConfig(),/between/);
$('force-duration').value='200';assert.equal(forceConfig().max_duration_s,200);
""")

    def test_larger_displayed_nudge_is_blocked_without_new_limits(self):
        self.run_js("""(async()=>{
const armed=(extra={})=>status({right_arm_frozen:true,alignment_nudge_supported:true,alignment:preview({nudge_left_mm:[8,0,0],config:{...baseConfig,max_step_mm:10}}),...extra});
fetch=async()=>imageResponse(1);render(armed());await tick();assert.equal($('alignment-nudge').disabled,true);
assert.match($('alignment-action-hint').textContent,/exceeds/);
render(armed({alignment_limits:{max_nudge_mm:10}}));
assert.equal($('alignment-nudge').disabled,true);assert.match($('alignment-action-hint').textContent,/exceeds/);
})()""")

    def test_old_backend_disables_new_controls_without_breaking_playback(self):
        self.run_js("""
render(status({alignment:undefined,right_arm_frozen:undefined}));
assert.equal($('alignment-apply').disabled,true);assert.equal($('freeze-right').disabled,true);
assert.equal($('play').disabled,false);assert.match($('alignment-placeholder').textContent,/preview service/);
""")
