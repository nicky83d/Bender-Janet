const $ = (id) => document.getElementById(id);
let currentView = 'front';
const SKILL_VIEWS = ['boot','motors','routines','detection','voice','speech','elevenlabs','faces','objects'];
function isSkillView(view){ return SKILL_VIEWS.includes(view); }

function setText(id, value) { const el=$(id); if (el) el.textContent = value ?? '-'; }
function setHtml(id, value) { const el=$(id); if (el) el.innerHTML = value; }
function esc(s){return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
async function api(url, opts={}){ const res=await fetch(url,{cache:'no-store',...opts}); let data; try{data=await res.json()}catch{data={status:'error',message:await res.text()}} if(!res.ok && !data.message)data.message=res.statusText; return data; }
async function postJson(url, body={}){ return api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }

function switchView(view){
  if(view==='settings' || view==='skills') view='boot';
  currentView=view;
  const skillMode = isSkillView(view);
  document.querySelectorAll('.main-tabs .tab').forEach(b=>b.classList.toggle('active', b.dataset.view===view || (b.dataset.view==='skills' && skillMode)));
  document.querySelectorAll('.skills-list .tab').forEach(b=>b.classList.toggle('active', b.dataset.view===view));
  const list=$('skills-list'); if(list) list.classList.toggle('hidden', !skillMode);
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  const target=$('view-'+view); if(target) target.classList.add('active');
  $('main-container').classList.toggle('full', !['front','rear'].includes(view));
  $('main-container').classList.toggle('skills-mode', skillMode);
  if(view==='boot') loadBoot(); if(view==='routines') loadRoutines(); if(view==='detection') loadDetectionInfo(); if(view==='faces') loadFaceInfo(); if(view==='objects') loadObjectInfo(); if(view==='elevenlabs') loadElevenLabs();
}

async function move(dir){ const d=$('duration-input')?.value || ''; const a=$('accel-input')?.value || ''; fetch(`/move/${dir}?duration=${encodeURIComponent(d)}&acceleration=${encodeURIComponent(a)}&t=${Date.now()}`,{cache:'no-store'}).catch(console.log); }
function bind(){
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
  document.querySelectorAll('[data-dir]').forEach(b=>b.addEventListener('click',()=>move(b.dataset.dir)));
  $('screenshot-btn')?.addEventListener('click',()=>{ location.href='/screenshot_front?t='+Date.now(); });
  $('duration-input')?.addEventListener('input',syncMotorLabels); $('accel-input')?.addEventListener('input',syncMotorLabels);
  $('boot-restart')?.addEventListener('click',()=>postJson('/boot_start').then(d=>{setText('boot-message',d.message); loadBoot();}));
  $('save-motors')?.addEventListener('click',saveMotors); document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>applyPreset(b.dataset.preset)));
  $('routine-stop')?.addEventListener('click',()=>postJson('/routine_stop').then(loadRoutines)); $('routine-random')?.addEventListener('click',()=>postJson('/routine_start/random').then(loadRoutines));
  $('scan-models')?.addEventListener('click',scanModels);
  $('save-speech')?.addEventListener('click',()=>postJson('/speech_settings',{device:$('speech-device-input').value}).then(d=>setText('speech-status',d.message)));
  $('list-speech')?.addEventListener('click',async()=>{const d=await api('/speech_devices'); setText('speech-devices-list',d.devices); if(d.suggested_device)$('speech-device-input').value=d.suggested_device;});
  $('test-beep')?.addEventListener('click',()=>postJson('/speech_test_beep').then(d=>setText('speech-status',d.message)));
  $('set-volume')?.addEventListener('click',()=>postJson('/speech_set_volume').then(d=>setText('speech-status',d.message)));
  $('speak-phrase')?.addEventListener('click',()=>postJson('/speech_say',{text:$('speech-phrase-input').value}).then(d=>setText('speech-status',d.message)));
  $('save-elevenlabs')?.addEventListener('click',saveElevenLabs);
  $('refresh-elevenlabs')?.addEventListener('click',loadElevenLabs);
  $('test-elevenlabs')?.addEventListener('click',testElevenLabs);
  $('precache-elevenlabs')?.addEventListener('click',precacheElevenLabs);
  $('save-voice')?.addEventListener('click',()=>postJson('/voice_settings',{device:$('voice-device-input').value}).then(d=>setText('voice-status',d.status)));
  $('list-voice')?.addEventListener('click',async()=>{const d=await api('/voice_devices'); setText('voice-devices-list',d.devices); if(d.suggested_device)$('voice-device-input').value=d.suggested_device;});
  $('add-face-camera')?.addEventListener('click',addFaceCamera); $('add-face-photo')?.addEventListener('click',addFacePhoto);
  $('save-hermes')?.addEventListener('click',saveHermes); $('quick-hermes')?.addEventListener('click',quickHermes); $('probe-hermes')?.addEventListener('click',probeHermes); $('diag-hermes')?.addEventListener('click',diagHermes); $('repair-hermes')?.addEventListener('click',repairHermes); $('discover-hermes')?.addEventListener('click',discoverHermes); $('test-hermes')?.addEventListener('click',()=>askHermes('Reply with OK if Hermes can hear Janet.'));
  $('ask-hermes')?.addEventListener('click',()=>askHermes($('hermes-question').value,false)); $('hermes-see')?.addEventListener('click',()=>askHermes('What can Janet see?',true)); $('hermes-status-question')?.addEventListener('click',()=>askHermes('Summarise Janet robot status.',true));
}

function syncMotorLabels(){ setText('duration-label', Number($('duration-input').value).toFixed(2)+'s'); setText('accel-label', Number($('accel-input').value).toFixed(2)); }
async function loadMotors(){ const d=await api('/motor_settings'); if($('duration-input')){$('duration-input').value=d.duration;$('accel-input').value=d.acceleration; syncMotorLabels(); setText('motor-status',`Motors: ${d.motors_available?'available':'not available'} | preset ${d.preset}`);} }
async function saveMotors(){ const d=await postJson('/motor_settings',{duration:Number($('duration-input').value),acceleration:Number($('accel-input').value)}); setText('motor-status',d.message); }
async function applyPreset(p){ const d=await postJson('/motor_preset/'+p); setText('motor-status',d.message); loadMotors(); }

function colorClass(v){ if(v<0 || v<20) return 'bad'; if(v<50) return 'warn'; return 'ok'; }
function updateReadings(data){
  const s=data.sonar||{}; const map={f:s.front,b:s.back,l:s.left,r:s.right};
  Object.entries(map).forEach(([id,v])=>{const el=$(id); if(el){el.textContent=(v==null||v<0)?'-':v+'cm'; el.className='val '+colorClass(Number(v));}});
  setText('detections-title',`Detections (${Number(data.fps||0).toFixed(1)} FPS)`);
  const det=data.detections||[]; setHtml('detections-list', det.length?det.map(d=>`<div class="det-card"><b>${esc(d.label)}</b><br><span class="note">${esc(d.confidence)}% confidence</span></div>`).join(''):'<p class="note">No detections</p>');
  const v=data.voice||{}; setText('voice-status',v.status); setText('voice-heard',v.last_heard||'-'); setText('voice-action',v.last_action||v.last_error||'-'); setText('voice-level',`${v.mic_level||0}% / ${v.mic_level_db??'-'} dB`);
  const sp=data.speech||{}; setText('speech-status',sp.status); setText('speech-error',sp.last_error||'-'); if(sp.speaker_device && $('speech-device-input')) $('speech-device-input').value=sp.speaker_device;
  const el=data.elevenlabs||{}; setText('el-status',el.status||'-'); setText('el-error',el.last_error||'-'); setText('el-cache-count',el.cache_count??'-'); setText('el-key-status',el.api_key_found?'found':'not found'); setText('el-key-source',el.api_key_source||'-');
  const h=data.hermes||{}; setText('hermes-status-value',h.status||'-'); setText('hermes-endpoint-value',h.endpoint||'-'); setText('hermes-error-value',h.last_error||'-');
  updateBootFields(data.boot||{});
  setText('dashboard-status','Dashboard live: '+new Date().toLocaleTimeString());
}

function updateBootFields(b){
  setText('boot-active', b.active ? 'yes' : 'no');
  setText('boot-complete', b.complete ? 'yes' : 'no');
  setText('boot-step', b.step || '-');
  setText('boot-progress', `${b.step_index||0} / ${b.total_steps||6}`);
  setText('boot-message', b.last_message || '-');
  setText('boot-error', b.last_error || '-');
  const hist = (b.history||[]).slice(-12).map(x=>`${new Date((x.time||0)*1000).toLocaleTimeString()}  ${x.step_index||'-'}. ${x.step||''}: ${x.message||''}${x.error?' | '+x.error:''}`).join('\n');
  setText('boot-history', hist || 'No boot history yet.');
}
async function loadBoot(){ const d=await api('/boot_status'); updateBootFields(d.boot||{}); }

async function poll(){ try{ const data=await api('/readings_basic'); updateReadings(data); }catch(e){ setText('dashboard-status','Dashboard error: '+e); } }

async function loadRoutines(){ const d=await api('/routine_info'); const r=d.routines||{}; setText('routine-message',r.last_message||'Ready'); if($('routine-progress')) $('routine-progress').style.width=Math.min(100,(Number(r.elapsed_seconds||0)/Number(r.target_seconds||20))*100)+'%'; const box=$('routine-buttons'); if(box && r.routines) box.innerHTML=r.routines.map(x=>`<button onclick="postJson('/routine_start/${esc(x.id)}').then(loadRoutines)"><b>${esc(x.emoji)} ${esc(x.name)}</b><br><span class="note">${esc(x.description)}</span></button>`).join(''); }
async function loadDetectionInfo(){ const d=await api('/detection_info'); const m=d.model||{}; setHtml('detection-info',`<p>Name: <b>${esc(m.name)}</b></p><p>FPS: ${Number(d.fps||0).toFixed(1)}</p><p>Labels: ${m.labels_count||0}</p>`); setHtml('model-list',(d.candidates||m.scan_results||[]).map(x=>`<div class="det-card"><b>${esc(x.name)}</b><p class="note">${esc(x.note||'')}</p><button onclick="useModel('${esc(x.name)}')">Use this model</button></div>`).join('')); }
async function scanModels(){ const d=await postJson('/detection_scan'); setHtml('model-list',(d.results||[]).map(x=>`<div class="det-card"><b>${esc(x.name)}</b><p class="note">${esc(x.note||'')}</p><button onclick="useModel('${esc(x.name)}')">Use this model</button></div>`).join('')); }
async function useModel(name){ const d=await postJson('/detection_use_model',{model:name}); alert(d.message||JSON.stringify(d)); loadDetectionInfo(); }

async function loadFaceInfo(){ const d=await api('/face_info'); const f=d.face||{}; setText('face-message',f.last_message||''); renderFaces(f.known_faces||[]); renderSeenFaces(f.last_seen||[]); }
function renderFaces(items){ const box=$('face-list'); if(!box)return; box.innerHTML=items.length?items.map(item=>`<div class="face-item"><b>${esc(item.name)}</b> <span class="note">${item.samples} sample(s)</span><div class="thumb-grid">${(item.images||[]).map(img=>`<div class="thumb-wrap"><a href="${img.url}" target="_blank"><img src="${img.thumb_url}"></a><button class="small-x" onclick="removeFaceSample(${img.sample_index})">×</button></div>`).join('')}</div><button class="danger" onclick="removeFaceName('${esc(item.name)}')">Remove all</button></div>`).join(''):'<p class="note">No known faces yet.</p>'; }
function renderSeenFaces(items){ setHtml('face-seen-list', items.length?items.map(f=>`<div class="det-card"><b>${esc(f.name)}</b> <span class="note">score ${esc(f.score)}</span></div>`).join(''):'<p class="note">No faces seen.</p>'); }
async function addFaceCamera(){ const d=await postJson('/face_add',{name:$('face-name-input').value}); setText('face-message',d.message); loadFaceInfo(); }
async function addFacePhoto(){ const fd=new FormData(); fd.append('name',$('face-photo-name').value); fd.append('photo',$('face-photo-file').files[0]); const res=await fetch('/face_upload',{method:'POST',body:fd}); const d=await res.json(); setText('face-message',d.message); loadFaceInfo(); }
async function removeFaceSample(i){ if(confirm('Delete this face sample?')){const d=await postJson('/face_remove_sample',{sample_index:i}); setText('face-message',d.message); loadFaceInfo();}}
async function removeFaceName(name){ if(confirm('Remove all samples for '+name+'?')){const d=await postJson('/face_remove',{name}); setText('face-message',d.message); loadFaceInfo();}}

async function loadObjectInfo(){ const d=await api('/object_info'); const o=d.object||{}; renderObjects(o.known_objects||[]); renderSeenObjects(o.last_seen||[]); }
function renderObjects(items){ const box=$('object-list'); if(!box)return; box.innerHTML=items.length?items.map(item=>`<div class="object-tile"><b>${esc(item.label)}</b><br><span class="note">${item.samples} sample(s)</span><div class="thumb-grid">${(item.images||[]).slice(0,6).map(img=>`<div class="thumb-wrap"><a href="${img.url}" target="_blank"><img src="${img.thumb_url}"></a><button class="small-x" onclick="removeObjectSample(${img.sample_index},'${esc(img.id)}')">×</button></div>`).join('')}</div><button class="danger" onclick="removeObjectLabel('${esc(item.label)}')">Remove all</button></div>`).join(''):'<p class="note">No known objects yet.</p>'; }
function renderSeenObjects(items){ setHtml('object-seen-list', items.length?items.map(o=>`<div class="det-card"><b>${esc(o.label)}</b> <span class="note">${esc(o.confidence)}%</span></div>`).join(''):'<p class="note">No objects seen.</p>'); }
async function removeObjectSample(i,id){ if(confirm('Delete this object sample?')){await postJson('/object_remove_sample',{sample_index:i,sample_id:id}); loadObjectInfo();}}
async function removeObjectLabel(label){ if(confirm('Remove all samples for '+label+'?')){await postJson('/object_remove',{label}); loadObjectInfo();}}

async function loadHermesSettings(){ try{ const d=await api('/hermes_settings'); if(d.base_url&&$('hermes-base')) $('hermes-base').value=d.base_url; if(d.model&&$('hermes-model')) $('hermes-model').value=d.model; if(d.endpoint&&$('hermes-endpoint')) $('hermes-endpoint').value=d.endpoint; setText('hermes-status-value',d.status||'-'); setText('hermes-endpoint-value',d.endpoint||'-'); }catch(e){} }
async function saveHermes(){ const d=await postJson('/hermes_settings',{base_url:$('hermes-base').value,api_key:$('hermes-key').value,model:$('hermes-model').value,endpoint:$('hermes-endpoint').value}); setText('hermes-status','Hermes settings saved.'); return d; }
async function quickHermes(){ $('hermes-output').textContent='Checking Janet route...'; await saveHermes(); const a=await api('/hermes_button_test'); $('hermes-output').textContent='Janet route: '+JSON.stringify(a)+'\nChecking Hermes TCP/API...'; const d=await api('/hermes_quick_check'); $('hermes-output').textContent += '\nHermes quick check: '+JSON.stringify(d,null,2); }
async function probeHermes(){ $('hermes-output').textContent='Probing Hermes API with /v1/chat/completions...'; await saveHermes(); const d=await postJson('/hermes_probe'); $('hermes-output').textContent=JSON.stringify(d,null,2); }
async function diagHermes(){ $('hermes-output').textContent='Running Hermes network diagnostics from Janet backend...'; await saveHermes(); const d=await api('/hermes_diagnostics'); $('hermes-output').textContent=JSON.stringify(d,null,2); }

async function repairHermes(){
  $('hermes-output').textContent='Repairing Hermes link using saved endpoint first, then discovery if needed...';
  await saveHermes();
  const d=await postJson('/hermes_repair',{});
  $('hermes-output').textContent=JSON.stringify(d,null,2);
  if(d && d.selected){
    if(d.selected.base_url) $('hermes-base').value=d.selected.base_url;
    if(d.selected.endpoint) $('hermes-endpoint').value=d.selected.endpoint;
    await saveHermes();
    setText('hermes-status','Hermes repaired and saved: '+(d.selected.url||d.selected.base_url));
  } else {
    setText('hermes-status','Hermes repair did not find a working chat endpoint. Check output.');
  }
}

async function discoverHermes(){
  $('hermes-output').textContent='Discovering Hermes on common local ports. This can take up to 20 seconds...';
  await saveHermes();
  const d=await postJson('/hermes_discover',{});
  $('hermes-output').textContent=JSON.stringify(d,null,2);
  if(d && d.selected){
    if(d.selected.base_url) $('hermes-base').value=d.selected.base_url;
    if(d.selected.endpoint) $('hermes-endpoint').value=d.selected.endpoint;
    await saveHermes();
    setText('hermes-status','Hermes discovered and saved: '+d.selected.url);
  } else if(d && d.open_ports && d.open_ports.length){
    setText('hermes-status','Open port found, but no chat endpoint yet. Check output.');
  } else {
    setText('hermes-status','No Hermes API port reachable from Janet. Check Hermes bind/firewall/IP.');
  }
}

async function askHermes(question, includeContext=false){ await saveHermes(); const q=question || $('hermes-question').value || 'Hello Hermes'; $('hermes-output').textContent='Asking Hermes via /v1/chat/completions: '+q; const d=await postJson('/hermes_ask',{question:q,speak:$('hermes-speak').value==='true',include_context:includeContext}); $('hermes-output').textContent=d.answer || d.error || JSON.stringify(d,null,2); if(d && d.discover && d.discover.selected){ $('hermes-base').value=d.discover.selected.base_url; $('hermes-endpoint').value=d.discover.selected.endpoint; await saveHermes(); } else if(d && d.repair && d.repair.selected){ $('hermes-base').value=d.repair.selected.base_url; $('hermes-endpoint').value=d.repair.selected.endpoint; await saveHermes(); } }


async function loadElevenLabs(){
  try{
    const d=await api('/elevenlabs_settings');
    if($('el-engine')) $('el-engine').value=d.engine||'edge';
    if($('el-enabled')) $('el-enabled').value=String(!!d.enabled);
    if($('el-bilingual')) $('el-bilingual').value=String(!!d.bilingual);
    if($('el-translate')) $('el-translate').value=String(!!d.translate_with_hermes);
    if($('el-en-voice')) $('el-en-voice').value=d.english_voice_id||'RlSVB64yXMZJjq67jbB1';
    if($('el-zh-voice')) $('el-zh-voice').value=d.chinese_voice_id||'APSIkVZudNbPAwyPoeVO';
    if($('el-edge-enabled')) $('el-edge-enabled').value=String(d.edge_enabled!==false);
    if($('el-edge-en')) $('el-edge-en').value=d.edge_english_voice||'en-GB-RyanNeural';
    if($('el-edge-zh')) $('el-edge-zh').value=d.edge_chinese_voice||'zh-CN-XiaoxiaoNeural';
    if($('el-model')) $('el-model').value=d.model_id||'eleven_multilingual_v2';
    if($('el-format')) $('el-format').value=d.output_format||'pcm_16000';
    setText('el-key-status',d.api_key_found?(d.api_key_saved?'saved in web settings':'found via environment'):'not set');
    setText('el-key-source',d.api_key_source||'-');
    setText('el-cache-count',d.cache_count??'-');
    setText('el-edge-status',d.edge_available?'available':'missing');
    setText('el-edge-cache-count',d.edge_cache_count??'-');
    setText('el-status',d.status||'ready');
    setText('el-error',d.last_error||'-');
    setText('el-output',JSON.stringify(d,null,2));
    if($('el-api-key')) $('el-api-key').value='';
    if($('el-clear-key')) $('el-clear-key').checked=false;
  }catch(e){ setText('el-output','Natural TTS status error: '+e); }
}
async function saveElevenLabs(){
  const d=await postJson('/elevenlabs_settings',{
    engine:$('el-engine')?$('el-engine').value:'edge',
    api_key:$('el-api-key')?$('el-api-key').value:'',
    clear_api_key:$('el-clear-key')?$('el-clear-key').checked:false,
    enabled:$('el-enabled').value==='true',
    bilingual:$('el-bilingual').value==='true',
    translate_with_hermes:$('el-translate').value==='true',
    english_voice_id:$('el-en-voice').value,
    chinese_voice_id:$('el-zh-voice').value,
    model_id:$('el-model').value,
    output_format:$('el-format').value,
    cache_enabled:true,
    edge_enabled:$('el-edge-enabled')?$('el-edge-enabled').value==='true':true,
    edge_english_voice:$('el-edge-en')?$('el-edge-en').value:'en-GB-RyanNeural',
    edge_chinese_voice:$('el-edge-zh')?$('el-edge-zh').value:'zh-CN-XiaoxiaoNeural' 
  });
  setText('el-output',JSON.stringify(d,null,2));
  await loadElevenLabs();
}
async function testElevenLabs(){
  await saveElevenLabs();
  const phrase=$('el-test-phrase').value || 'Hello, I am Janet. My speaker is working.';
  const d=await postJson('/elevenlabs_test',{text:phrase});
  setText('el-output',JSON.stringify(d,null,2));
}
async function precacheElevenLabs(){
  await saveElevenLabs();
  setText('el-output','Pre-caching common Janet vocabulary, known names, and known objects with the selected Natural TTS engine...');
  const d=await postJson('/elevenlabs_precache',{});
  setText('el-output',JSON.stringify(d,null,2));
  await loadElevenLabs();
}

bind(); loadHermesSettings(); loadElevenLabs(); loadMotors(); loadRoutines(); poll(); setInterval(poll, 700);
