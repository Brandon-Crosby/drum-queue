(() => {
const $ = s => document.querySelector(s);
const el = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };
const mmss = s => { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
const AUD = /\.(mp3|wav|flac|m4a|aiff?|ogg|opus)$/i;
const SITE = window.SITE || {};
const previewSound = window.DrumPreviewSound;
let sound = previewSound ? previewSound.load() : { includeBass: false, tracks: {} };
if (!previewSound) {
  $('.sound-hint').hidden = true;
  $('#preview-sound-reset').hidden = true;
}
let LIM = { left: 1, max_mb: 30, max_minutes: 8 }, tsWidget = null, tsToken = '', pollT = 0, current = null, busy = false;

const store = {
  get() { try { return JSON.parse(localStorage.getItem('ds_recent') || '[]'); } catch (_) { return []; } },
  add(id, name) { try { const r = this.get().filter(x => x.id !== id); r.unshift({ id, name, t: Date.now() }); localStorage.setItem('ds_recent', JSON.stringify(r.slice(0, 6))); } catch (_) {} },
};

function showMsg(text) { const m = $('#msg'); m.replaceChildren(); if (text) m.append(el('div', 'msg err', text)); }
// Beta team link (?team=<code>): remembered on this device and sent with uploads; the server keeps
// those splits for testing. Removed from the address bar so it isn't copied along with a split link.
let TEAM = '';
try {
  const q = new URLSearchParams(location.search);
  if (q.has('team')) {
    const c = q.get('team') || '';
    if (c) localStorage.setItem('dsTeam', c); else localStorage.removeItem('dsTeam');
    q.delete('team'); history.replaceState(null, '', location.pathname + (q.toString() ? '?' + q : '') + location.hash);
  }
  TEAM = localStorage.getItem('dsTeam') || '';
} catch (_) {}
function teamBadge() {
  const b = document.createElement('div'); b.className = 'teambadge';
  b.textContent = 'Team mode: your splits are kept for testing';
  document.body.prepend(b);
}
if (TEAM) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', teamBadge); else teamBadge(); }


function renderRecent() {
  const r = store.get().filter(x => Date.now() - x.t < 24 * 3600e3 && (!current || x.id !== current));
  const box = $('#recent'); box.replaceChildren();
  if (!r.length) return;
  box.append('Your splits from today: ');
  r.forEach((x, i) => { const a = el('a', null, x.name.replace(/\.[a-z0-9]{2,4}$/i, '')); a.href = '#' + x.id; a.title = x.name; box.append(i ? ' · ' : '', a); });
}

async function loadLimits() {
  try {
    LIM = await (await fetch('/api/limits')).json();
    const parts = TEAM ? [`Up to ${LIM.max_minutes} minutes per song`, 'team mode: no daily limit, splits kept for testing']
      : [`Up to ${LIM.max_minutes} minutes per song`, `${LIM.left} of ${LIM.per_day} free splits left today`, 'files deleted after 24 hours'];
    $('#limits').textContent = parts.join(' · ');
    if (TEAM) {} else if (LIM.site_full) showMsg("We've hit today's limit for free splits. Please come back tomorrow.");
    else if (LIM.left <= 0) showMsg("You've used today's free splits. Come back tomorrow.");
  } catch (_) { /* limits line keeps its default text */ }
}

// ---------- bot check (Cloudflare Turnstile), only if a site key is configured
function initTurnstile() {
  if (!SITE.turnstile) return;
  const go = () => {
    if (!window.turnstile) return setTimeout(go, 200);
    tsWidget = turnstile.render('#ts', {
      sitekey: SITE.turnstile, theme: 'auto',
      // Stays invisible unless Cloudflare actually needs a click, so the box doesn't pop back
      // after every upload (each pass only works once, so it quietly re-runs after each one).
      appearance: 'interaction-only',
      callback: t => { tsToken = t; showMsg(''); },
      'expired-callback': () => { tsToken = ''; },
      // The check can't reach Cloudflare: almost always an ad blocker, VPN or iCloud Private Relay.
      'error-callback': () => { tsToken = ''; showTsHelp(); return true; },
    });
  };
  go();
}
function showTsHelp() {
  const m = $('#msg'); m.replaceChildren();
  const box = el('div', 'msg err');
  box.append("Uploads paused: the \"verify you're human\" check couldn't load (usually an ad blocker, VPN or iCloud Private Relay). Allow this site, then ");
  const b = el('button', 'btn', 'try again'); b.style.marginLeft = '4px';
  b.onclick = () => { showMsg(''); if (window.turnstile && tsWidget != null) turnstile.reset(tsWidget); else location.reload(); };
  box.append(b, '.');
  m.append(box);
}
// The Turnstile script itself was blocked from loading
setTimeout(() => { if (SITE.turnstile && !window.turnstile) showTsHelp(); }, 8000);
function resetTurnstile() { tsToken = ''; if (tsWidget != null && window.turnstile) turnstile.reset(tsWidget); }
// Each pass of the bot check works for one upload. Asking again right after an upload made the
// empty box pop back while the song was still splitting, which looked like the check had failed.
// So the box hides during a split and comes back, fresh, when the split ends (or a new song is dropped).
let tsParked = false;
function tsPark() { tsToken = ''; tsParked = true; $('#ts').hidden = true; }
function tsUnpark() { if (!tsParked) return; tsParked = false; $('#ts').hidden = false; resetTurnstile(); }

// ---------- upload
const input = $('#file');
// every drop zone on the page (the hero one and the one at the bottom) opens the same picker
document.querySelectorAll('#drop, [data-drop]').forEach(drop => {
  drop.addEventListener('click', () => !busy && input.click());
  drop.addEventListener('keydown', e => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); input.click(); } });
  ['dragenter', 'dragover'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) start(f); });
});
input.addEventListener('change', () => { const f = input.files[0]; input.value = ''; if (f) start(f); });

// The invisible check takes a second or two after page load / after each upload; wait for it.
async function waitToken(ms = 12000) {
  for (let t = 0; SITE.turnstile && !tsToken && t < ms; t += 250) await new Promise(r => setTimeout(r, 250));
  return !SITE.turnstile || !!tsToken;
}
async function start(f) {
  showMsg('');
  if (busy) return;
  if (!AUD.test(f.name)) return showMsg('That file type is not supported. Use mp3, wav, flac, m4a, aiff or ogg.');
  if (f.size > LIM.max_mb * 1024 * 1024) return showMsg(`That file is over ${LIM.max_mb} MB. Try an mp3 of the song.`);
  busy = true;
  tsUnpark();
  const ready = await waitToken(); busy = false;
  if (!ready) return showMsg('Please finish the "verify you are human" check above, then try again.');
  busy = true; stopMixer(); openJob(f.name); setStep('upload', 0);
  const fd = new FormData(); fd.append('file', f, f.name); fd.append('token', tsToken);
  if (TEAM) fd.append('team', TEAM);
  const x = new XMLHttpRequest();
  x.open('POST', '/api/jobs');
  x.upload.onprogress = e => { if (e.lengthComputable) setStep('upload', Math.round(e.loaded / e.total * 100)); };
  x.onload = () => {
    busy = false;
    let d = {}; try { d = JSON.parse(x.responseText); } catch (_) {}
    if (x.status === 200) tsPark(); else resetTurnstile();
    if (x.status !== 200) { $('#job').style.display = 'none'; showMsg(d.error || 'Upload failed. Please try again.'); loadLimits(); return; }
    if (TEAM && !d.team) showMsg("That team code wasn't recognized, so this split won't be kept.");
    store.add(d.id, f.name); location.hash = d.id; loadLimits();
  };
  x.onerror = () => { busy = false; resetTurnstile(); $('#job').style.display = 'none'; showMsg('Upload failed. Check your connection and try again.'); };
  x.send(fd);
}

// ---------- job status
function openJob(name) {
  $('#job').style.display = 'block'; $('#jname').textContent = name; $('#jmeta').textContent = '';
  $('#result').hidden = true; $('#steps').hidden = false; $('#tracks').replaceChildren(); $('#cancelrow').hidden = true;
  logReset(); $('#log').hidden = false;
  $('#job').scrollIntoView({ behavior: 'smooth', block: 'start' });
  showWaitAd();
}
const ORDER = ['upload', 'queue', 'mix', 'kit', 'finish'];
function setStep(k, pct) {
  const i = ORDER.indexOf(k);
  document.querySelectorAll('#steps li').forEach((li, j) => {
    li.className = j < i ? 'ok' : j === i ? 'on' : '';
    const base = li.dataset.base || (li.dataset.base = li.lastChild.textContent);
    li.lastChild.textContent = base + (j === i && k === 'upload' && pct != null ? ` (${pct}%)` : '');
  });
}
// ---------- process log: short technical lines under the steps. Lines for the GPU steps describe
// what that step really does; one-shot lines come from the server with the song's real numbers.
const LOGS = {
  queue: ['Requesting a GPU · L4 → A10 → T4 → L40S → A100', 'Waiting for a container to start',
          'Cold start: loading both separation models into GPU memory'],
  mix: ['Decoding audio · 44.1 kHz stereo', 'Demucs ensemble · one fine-tuned model per source',
        'Processing in overlapping 7.8 s windows', 'Separating drums from bass, vocals and melodic parts'],
  kit: ['Helyx drum model · kick · snare · percussion · hi-hat', 'Overlap ×4 to smooth window edges',
        'Writing one stem per kit piece'],
  finish: ['Encoding MP3 previews for the mixer'],
};
const LOG = { lines: [], step: '', i: 0, t: 0, last: '' };
function logPush(line) {
  if (!line || line === LOG.lines[LOG.lines.length - 1]) return;
  LOG.lines.push(line); if (LOG.lines.length > 40) LOG.lines.shift(); logDraw();
}
function logDraw() {
  const box = $('#log'), show = LOG.lines.slice(-6);
  box.replaceChildren(...show.map((l, i) => el('div', i === show.length - 1 ? 'now' : null, '› ' + l)));
}
function logReset() { Object.assign(LOG, { lines: [], step: '', i: 0, t: 0, last: '' }); logDraw(); }
function logTick(j, k) {
  const [stage, detail] = (j.stage || '').split(' — ');
  if (k !== LOG.step) { LOG.step = k; LOG.i = 0; LOG.t = 0; }
  if (detail) {                                                  // real numbers from the server
    for (const l of (j.log && j.log.length ? j.log : [detail])) if (!LOG.lines.includes(l)) logPush(l);
    return;
  }
  if (stage && stage !== LOG.last && j.status === 'running') { LOG.last = stage; logPush(stage); }
  const now = Date.now(), lines = LOGS[k] || [];
  if (LOG.i < lines.length && now - LOG.t > 3500) { logPush(lines[LOG.i++]); LOG.t = now; }
}
function stepOf(j) {
  if (j.status === 'queued') return 'queue';
  const s = (j.stage || '').toLowerCase();
  if (s.includes('preview') || s.includes('one-shot')) return 'finish';
  if (s.includes('kit')) return 'kit';
  return 'mix';
}
async function poll(id) {
  clearTimeout(pollT);
  if (current !== id) return;
  let j;
  try { const r = await fetch('/api/jobs/' + id); j = await r.json(); if (!r.ok) throw new Error(j.error || 'Not found'); }
  catch (e) { $('#steps').hidden = true; showMsg(e.message === 'Failed to fetch' ? 'Lost connection. Retrying…' : e.message); if (e.message === 'Failed to fetch') pollT = setTimeout(() => poll(id), 4000); return; }
  if (current !== id) return;
  $('#jname').textContent = j.name || 'Your song';
  const live = j.status === 'queued' || j.status === 'running';
  $('#cancelrow').hidden = !live;
  if (j.status === 'done' || j.status === 'error') tsUnpark();
  if (j.status === 'done') return showResult(id, j);
  if (j.status === 'error') { $('#steps').hidden = true; $('#log').hidden = true; showMsg(j.error || 'Something went wrong.'); loadLimits(); return; }
  showMsg(''); const k = stepOf(j); setStep(k); logTick(j, k);
  $('#jmeta').textContent = mmss(Date.now() / 1000 - j.created) + etaText(j);
  pollT = setTimeout(() => poll(id), 2000);
}
// Rough time left, from the song's length and how fast recent splits ran (the server sends both).
function etaText(j) {
  if (!j.eta_run) return '';
  const now = Date.now() / 1000;
  let left = j.status === 'running'
    ? j.eta_run - (now - (j.started || now))
    : Math.max(15, j.eta_wait - (now - j.created)) + j.eta_run;
  if (left < 15) return ' · almost done';
  if (left < 60) return ' · under a minute left';
  return ` · about ${Math.ceil(left / 60)} min left`;
}
$('#cancel').onclick = async () => {
  const id = current; if (!id) return;
  const b = $('#cancel'); b.disabled = true; b.textContent = 'Cancelling…';
  try { await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }); } catch (_) { /* poll shows the result */ }
  b.disabled = false; b.textContent = 'Cancel';
  poll(id);
};
function route() {
  const id = location.hash.slice(1);
  if (!/^[a-f0-9]{16}$/.test(id)) return;
  current = id; stopMixer(); stopShot();
  const known = store.get().find(x => x.id === id);
  openJob(known ? known.name : 'Your song'); setStep('queue');
  renderRecent(); poll(id);
}
window.addEventListener('hashchange', route);

// ---------- ads while waiting (manual slot if configured; auto ads handle the rest)
let adShown = false;
function showWaitAd() {
  if (adShown || !SITE.client || !SITE.waitSlot) return;
  adShown = true;
  const ins = el('ins', 'adsbygoogle'); ins.style.display = 'block';
  ins.dataset.adClient = SITE.client; ins.dataset.adSlot = SITE.waitSlot; ins.dataset.adFormat = 'auto'; ins.dataset.fullWidthResponsive = 'true';
  $('#waitad').append(ins);
  try { (window.adsbygoogle = window.adsbygoogle || []).push({}); } catch (_) {}
}

// ---------- result + mixer
let MX = null, AC = null, raf = 0;
function showResult(id, j) {
  $('#steps').hidden = true; $('#log').hidden = true; $('#result').hidden = false; showMsg('');
  $('#jmeta').textContent = [j.bpm ? Number(j.bpm.toFixed(2)) + ' BPM' : null, j.duration ? mmss(j.duration) : null].filter(Boolean).join(' · ');
  $('#zipkit').href = `/api/jobs/${id}/zip?kit=1`; $('#zipall').href = `/api/jobs/${id}/zip`;
  const left = Math.max(0, j.expires - Date.now() / 1000);
  $('#expire').textContent = `These files are deleted in about ${Math.max(1, Math.round(left / 3600))} hours. Bookmark this page to come back.` + (j.note ? ' ' + j.note : '');
  buildMixer(id, j);
  buildShots(id, j);
}
// ---------- one-shots: one pad per sound, tap to hear, download one or the whole kit
let shotAudio = null;
function buildShots(id, j) {
  stopShot(); buildLoops(id, j);                 // loops show even when a split has no one-shots
  const box = $('#shots'), pads = $('#pads'); pads.replaceChildren();
  const shots = j.shots || [];
  box.hidden = !shots.length;
  if (!shots.length) return;
  $('#zipshots').href = `/api/jobs/${id}/zip?shots=1`;
  const gm = $('#groovemid'); gm.hidden = !j.groove; gm.href = `/api/jobs/${id}/groove`; gm.download = 'Groove.mid'; // The API exposes availability, not a filename.
  for (const s of shots) {
    const url = `/api/jobs/${id}/shots/${encodeURIComponent(s.name)}`;
    const pad = el('div', 'pad');
    const play = el('button', 'padhit'); play.type = 'button';
    play.append(el('b', null, s.label), el('small', null, `${s.seconds.toFixed(2)}s · ${s.count} hits`));
    play.onclick = () => {
      stopShot(); if (MX && MX.playing) stopMixer(true);
      shotAudio = new Audio(url); shotAudio.play().catch(() => {});
      play.classList.add('hit'); setTimeout(() => play.classList.remove('hit'), 150);
    };
    const dl = el('a', 'paddl', '↓'); dl.href = url + '?dl=1'; dl.title = 'Download ' + s.label; dl.setAttribute('aria-label', 'Download ' + s.label);
    pad.append(play, dl); pads.append(pad);
  }
}
// one sound at a time: a pad tap or a loop stops whatever pad/loop was playing
function stopShot() {
  if (shotAudio) { shotAudio.pause(); shotAudio.loop = false; }
  shotAudio = null;
  document.querySelectorAll('#looppads .padhit.on').forEach(b => { b.classList.remove('on'); b.setAttribute('aria-pressed', 'false'); });
}
function buildLoops(id, j) {
  const box = $('#loops'), pads = $('#looppads'); pads.replaceChildren();
  const loops = j.loops || [];
  box.hidden = !loops.length;
  for (const s of loops) {
    const url = `/api/jobs/${id}/shots/${encodeURIComponent(s.name)}`;
    const pad = el('div', 'pad');
    const play = el('button', 'padhit'); play.type = 'button'; play.setAttribute('aria-pressed', 'false');
    play.append(el('b', null, s.label), el('small', null, `${s.bars} bars · loops`));
    play.onclick = () => {
      const was = play.classList.contains('on');
      stopShot(); if (MX && MX.playing) stopMixer(true);
      if (was) return;
      shotAudio = new Audio(url); shotAudio.loop = true; shotAudio.play().catch(() => stopShot());
      play.classList.add('on'); play.setAttribute('aria-pressed', 'true');
    };
    const dl = el('a', 'paddl', '↓'); dl.href = url + '?dl=1'; dl.title = 'Download ' + s.label; dl.setAttribute('aria-label', 'Download ' + s.label);
    pad.append(play, dl); pads.append(pad);
  }
}
function gainOn(t) { const solo = MX.tracks.some(x => x.solo); return (solo ? t.solo : !t.mute) ? 1 : 0; }
function db(value) { return (value > 0 ? '+' : '') + value + ' dB'; }
function syncSoundControls() {
  const bass = MX?.tracks.find(t => t.x.stem === 'bass');
  $('#preview-include-bass').checked = bass ? !bass.mute : sound.includeBass;
  $('#preview-include-bass').disabled = !bass;
  $('#preview-sound-note').textContent = (bass ? '' : 'No bass stem in this split. ') + 'Playback only. Your stem downloads stay unchanged.';
  for (const t of MX?.tracks || []) {
    if (!t.toneControls) continue;
    const s = previewSound.trackSettings(t.x, sound);
    t.toneEnabled.checked = s.enabled;
    for (const [name, control] of Object.entries(t.toneControls)) {
      control.input.value = s[name]; control.input.disabled = name !== 'levelDb' && !s.enabled;
      control.input.setAttribute('aria-valuetext', db(s[name])); control.output.textContent = db(s[name]);
      const [min, max] = previewSound.ranges[name];
      control.dial.style.setProperty('--dial-angle', (-135 + 270 * (s[name] - min) / (max - min)) + 'deg');
    }
  }
}
function apply() {
  if (!MX) return;
  for (const t of MX.tracks) {
    const on = gainOn(t);
    t.row.classList.toggle('off', !on); t.mb.classList.toggle('on', t.mute); t.sb.classList.toggle('on', t.solo);
    t.mb.setAttribute('aria-pressed', t.mute); t.sb.setAttribute('aria-pressed', t.solo);
    if (t.gain) t.gain.gain.setTargetAtTime(on, AC.currentTime, .01);
    if (t.sound) t.sound.update(sound);
    if (on && MX.ready && !t.buf && !t.loading) loadTrack(t, MX).catch(() => {});
  }
  syncSoundControls();
}
function setTrackSound(t, name, value) {
  sound.tracks[t.x.stem] = { ...previewSound.trackSettings(t.x, sound), [name]: value };
  previewSound?.save(sound); apply();
}
function tonePanel(t, button, index) {
  const panel = el('div', 'stem-tone'); panel.hidden = true; panel.id = 'stem-tone-' + index;
  panel.setAttribute('role', 'group'); panel.setAttribute('aria-label', t.x.label + ' EQ');
  button.setAttribute('aria-controls', panel.id); button.setAttribute('aria-expanded', 'false');
  button.onclick = () => { panel.hidden = !panel.hidden; button.setAttribute('aria-expanded', String(!panel.hidden)); };
  const heading = el('div', 'tone-heading'); heading.append(el('strong', null, t.x.label + ' EQ'));
  const toggle = el('label', 'include-bass'), enabled = document.createElement('input'); enabled.type = 'checkbox'; enabled.setAttribute('aria-label', t.x.label + ' EQ on');
  enabled.onchange = () => setTrackSound(t, 'enabled', enabled.checked); toggle.append(enabled, 'EQ on'); t.toneEnabled = enabled;
  const reset = el('button', 'btn', 'Reset'); reset.type = 'button'; reset.setAttribute('aria-label', 'Reset ' + t.x.label + ' EQ');
  reset.onclick = () => { delete sound.tracks[t.x.stem]; previewSound?.save(sound); apply(); };
  heading.append(toggle, reset); panel.append(heading);
  const knobs = el('div', 'tone-knobs'); t.toneControls = {};
  for (const [name, label] of [['levelDb', 'Level'], ['lowDb', 'Low'], ['midsDb', 'Mid'], ['highDb', 'High']]) {
    const dial = el('label', 'tone-dial'), face = el('span', 'dial-face'), pointer = el('span', 'dial-pointer'), input = document.createElement('input'), output = document.createElement('output');
    input.type = 'range'; [input.min, input.max] = previewSound.ranges[name]; input.step = '.5'; input.setAttribute('aria-label', t.x.label + ' ' + label.toLowerCase());
    input.oninput = () => setTrackSound(t, name, Number(input.value));
    let drag = null;
    input.onpointerdown = event => { if (event.button !== 0) return; event.preventDefault(); input.focus(); input.setPointerCapture(event.pointerId); drag = { id: event.pointerId, y: event.clientY, value: Number(input.value) }; };
    input.onpointermove = event => {
      if (!drag || event.pointerId !== drag.id) return;
      const [min, max] = previewSound.ranges[name];
      input.value = Math.max(min, Math.min(max, Math.round((drag.value + (drag.y - event.clientY) / 4) * 2) / 2));
      input.dispatchEvent(new Event('input', { bubbles: true }));
    };
    input.onpointerup = input.onpointercancel = input.onlostpointercapture = () => { drag = null; };
    face.append(pointer, input); dial.append(el('span', 'dial-label', label), face, output); knobs.append(dial); t.toneControls[name] = { dial, input, output };
  }
  panel.append(knobs); return panel;
}
$('#preview-include-bass').onchange = event => {
  sound.includeBass = event.target.checked;
  if (MX) for (const t of MX.tracks) if (t.x.stem === 'bass') t.mute = !sound.includeBass;
  previewSound?.save(sound); apply();
};
$('#preview-sound-reset').onclick = () => {
  if (!previewSound) return;
  sound = previewSound.settings();
  if (MX) for (const t of MX.tracks) if (t.x.stem === 'bass') t.mute = true;
  previewSound?.save(sound); apply();
};
function pos() { return MX.playing ? Math.min(MX.len, MX.off + AC.currentTime - MX.t0) : MX.off; }
function stopMixer(keep) {
  if (!MX) return;
  if (MX.playing) { const p = pos(); for (const t of MX.tracks) { try { t.src && t.src.stop(); } catch (_) {} t.src = null; } MX.playing = false; MX.off = keep ? p : 0; }
  else if (!keep) MX.off = 0;
  $('#play').textContent = '▶ Play'; draw();
}
function play(from) {
  if (!MX || !MX.ready) return; stopMixer(true);
  if (from != null) MX.off = from; if (MX.off >= MX.len - .05) MX.off = 0;
  stopShot(); AC.resume(); const t0 = AC.currentTime + .06; MX.t0 = t0;
  for (const t of MX.tracks) if (t.buf) startTrack(t, t0, MX.off);
  MX.playing = true; $('#play').textContent = '❚❚ Pause'; apply(); cancelAnimationFrame(raf);
  const tick = () => { if (!MX || !MX.playing) return; if (pos() >= MX.len) { stopMixer(false); return; } draw(); raf = requestAnimationFrame(tick); }; tick();
}
function draw() {
  if (!MX) return; const p = pos(); const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink');
  for (const t of MX.tracks) { const cv = t.cv; if (!t.img && t.buf) peaks(t); const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height); if (t.img) g.drawImage(t.img, 0, 0); if (MX.len) { g.fillStyle = ink; g.fillRect(p / MX.len * cv.width, 0, Math.max(1.5, devicePixelRatio), cv.height); } }
  $('#time').textContent = MX.ready ? mmss(p) + ' / ' + mmss(MX.len) : 'Loading…';
}
function peaks(t) {
  const cv = t.cv, dpr = devicePixelRatio || 1;
  if (!cv.clientWidth || !cv.clientHeight) { t.img = null; return; }   // hidden; redrawn when shown
  cv.width = Math.max(10, cv.clientWidth * dpr); cv.height = cv.clientHeight * dpr;
  const W = cv.width, H = cv.height, off = document.createElement('canvas'); off.width = W; off.height = H; const g = off.getContext('2d');
  g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--wave');
  const d = t.buf.getChannelData(0), step = Math.max(1, Math.floor(d.length / W)); let mx = 0; const col = new Float32Array(W);
  for (let x = 0; x < W; x++) { let m = 0; const a = x * step, b = Math.min(d.length, a + step); for (let i = a; i < b; i += 4) { const v = Math.abs(d[i]); if (v > m) m = v; } col[x] = m; if (m > mx) mx = m; }
  const sc = mx > 0 ? (H / 2 - 1) / Math.max(mx, .25) : 0; for (let x = 0; x < W; x++) { const h = Math.max(.5, col[x] * sc); g.fillRect(x, H / 2 - h, 1, h * 2); } t.img = off;
}
function startTrack(t, when, offset) {
  if (!t.buf || offset >= t.buf.duration) return;
  const source = AC.createBufferSource(); source.buffer = t.buf; source.connect(t.gain);
  source.start(when, Math.max(0, offset)); t.src = source;
}
async function loadTrack(t, mixer) {
  if (t.buf || t.loading) return t.loading;
  t.row.setAttribute('aria-busy', 'true'); t.cv.title = 'Loading audio…';
  t.loading = (async () => {
    try {
      const url = SITE.local
        ? `/api/jobs/${mixer.id}/files/${encodeURIComponent(t.x.name)}`
        : `/api/jobs/${mixer.id}/preview/${encodeURIComponent(t.x.preview)}`;
      const response = await fetch(url, { signal: mixer.abort.signal });
      if (!response.ok) throw new Error('Preview unavailable');
      const buffer = await AC.decodeAudioData(await response.arrayBuffer());
      if (MX !== mixer) return;
      t.buf = buffer;
      if (previewSound) {
        t.sound = previewSound.createTrack(AC, t.x, AC.destination, sound); t.gain = t.sound.input;
      } else {
        t.gain = AC.createGain(); t.gain.connect(AC.destination);
      }
      t.gain.gain.setValueAtTime(gainOn(t), AC.currentTime);
      peaks(t); t.cv.title = '';
      if (mixer.playing) startTrack(t, AC.currentTime, pos());
      draw();
    } catch (error) {
      if (MX === mixer && error.name !== 'AbortError') t.cv.title = 'Audio could not load. Mute and unmute to retry; downloads still work.';
      throw error;
    } finally {
      t.loading = null; t.row.removeAttribute('aria-busy');
    }
  })();
  return t.loading;
}
async function buildMixer(id, j) {
  AC = AC || new (window.AudioContext || window.webkitAudioContext)();
  if (MX) { stopMixer(false); MX.abort.abort(); for (const t of MX.tracks) { if (t.sound) t.sound.disconnect(); else if (t.gain) t.gain.disconnect(); } }
  const box = $('#tracks'); box.replaceChildren(); MX = { id, abort: new AbortController(), tracks: [], len: j.duration || 0, off: 0, playing: false, ready: false }; const me = MX;
  sound.includeBass = false;
  for (const grp of ['kit', 'mix']) {
    const list = j.stems.filter(x => x.group === grp); if (!list.length) continue;
    box.append(el('div', 'grp', grp === 'kit' ? 'Kit pieces' : 'Rest of the song'));
    for (const x of list) {
      const row = el('div', 'trk'), lb = el('span', 'lb', x.label); if (x.quiet) lb.append(el('small', null, 'silent'));
      const mb = el('button', 'm', 'M'), sb = el('button', 's', 'S'), tone = el('button', 'tone-button', 'EQ'), cv = document.createElement('canvas'), dl = el('a', null, 'WAV');
      mb.title = 'Mute ' + x.label; sb.title = 'Solo ' + x.label; mb.setAttribute('aria-label', 'Mute ' + x.label); sb.setAttribute('aria-label', 'Solo ' + x.label);
      dl.href = `/api/jobs/${id}/files/${encodeURIComponent(x.name)}?dl=1`; dl.title = 'Download ' + x.label + ' (WAV)';
      tone.hidden = !previewSound; tone.type = 'button'; tone.setAttribute('aria-label', 'EQ for ' + x.label);
      row.append(lb, mb, sb, cv, tone, dl); box.append(row);
      const t = { x, row, mb, sb, cv, mute: grp !== 'kit', solo: false, buf: null, gain: null, src: null, img: null };
      mb.onclick = () => { t.mute = !t.mute; if (x.stem === 'bass') { sound.includeBass = !t.mute; previewSound?.save(sound); } apply(); };
      sb.onclick = e => { if (!e.shiftKey && !e.metaKey && !t.solo) MX.tracks.forEach(o => o.solo = false); t.solo = !t.solo; apply(); };
      cv.onclick = e => { if (!MX.ready) return; const r = cv.getBoundingClientRect(); const p = (e.clientX - r.left) / r.width * MX.len; if (MX.playing) play(p); else { MX.off = p; draw(); } };
      if (previewSound) row.append(tonePanel(t, tone, MX.tracks.length));
      if (t.mute) cv.title = 'Audio loads when unmuted or soloed.';
      MX.tracks.push(t);
    }
  }
  apply(); draw();
  try {
    await Promise.all(me.tracks.filter(t => t.x.group === 'kit').map(t => loadTrack(t, me)));
  } catch (_) { if (MX === me) $('#time').textContent = 'Preview unavailable, downloads still work.'; return; }
  if (MX !== me) return;
  MX.len = Math.max(MX.len, ...MX.tracks.filter(t => t.buf).map(t => t.buf.duration));
  MX.ready = true; apply(); draw();
}
$('#play').onclick = () => { if (!MX) return; if (MX.playing) stopMixer(true); else play(); };
$('#stop').onclick = () => { stopMixer(false); stopShot(); };
$('#reset').onclick = () => { if (!MX) return; MX.tracks.forEach(t => { t.solo = false; t.mute = t.x.group !== 'kit'; }); sound.includeBass = false; previewSound?.save(sound); apply(); };
$('#again').onclick = () => { stopMixer(false); stopShot(); current = null; history.replaceState(null, '', location.pathname); $('#job').style.display = 'none'; renderRecent(); loadLimits(); window.scrollTo({ top: 0, behavior: 'smooth' }); };
document.addEventListener('keydown', e => { if (e.code === 'Space' && MX && MX.ready && !/INPUT|SELECT|TEXTAREA|BUTTON|A/.test(document.activeElement.tagName)) { e.preventDefault(); $('#play').click(); } });
window.addEventListener('resize', () => { if (MX && MX.ready) { MX.tracks.forEach(peaks); draw(); } });

// Cosmetic local copy must not prevent upload/history initialization if the layout changes.
if (SITE.local) {
  document.title = 'Drum Splitter · On this Mac';
  const text = (selector, value) => { const node = $(selector); if (node) node.textContent = value; };
  text('.hero .sub', 'Choose a song. Separate its drums, bass, vocals and other parts on this Mac, then save stems, one-shots and groove MIDI.');
  text('.foot .tag', 'Drum Splitter · Local processing in Zone Six Studio Hub.');
  text('.foot .fine > span', 'Audio stays on this Mac. Save downloads before quitting. Only choose audio you have the right to use.');
  const faq = $('#faq'); if (faq) faq.hidden = true;
  document.querySelectorAll('a[href="/#faq"]').forEach(node => { node.hidden = true; });
  for (const [key, value] of [['upload', 'Reading audio'], ['queue', 'Starting on this Mac']]) {
    const step = document.querySelector(`[data-k=${key}]`);
    if (step && step.lastChild) step.lastChild.textContent = value;
  }
}
// Keep the handler inside the mixer closure so MX/draw are available.
window.addEventListener('themechange', () => { if (!MX) return; MX.tracks.forEach(t => { t.img = null; }); draw(); });

loadLimits(); initTurnstile(); renderRecent(); route();
})();
