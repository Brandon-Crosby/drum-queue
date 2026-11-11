(() => {
const $ = s => document.querySelector(s);
const el = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };
const mmss = s => { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
const AUD = /\.(mp3|wav|flac|m4a|aiff?|ogg|opus)$/i;
const SITE = window.SITE || {};
let LIM = { left: 1, max_mb: 30, max_minutes: 8 }, tsWidget = null, tsToken = '', pollT = 0, current = null, busy = false;

const store = {
  get() { try { return JSON.parse(localStorage.getItem('ds_recent') || '[]'); } catch (_) { return []; } },
  add(id, name) { try { const r = this.get().filter(x => x.id !== id); r.unshift({ id, name, t: Date.now() }); localStorage.setItem('ds_recent', JSON.stringify(r.slice(0, 6))); } catch (_) {} },
};

function showMsg(text) { const m = $('#msg'); m.replaceChildren(); if (text) m.append(el('div', 'msg err', text)); }

function renderRecent() {
  const r = store.get().filter(x => Date.now() - x.t < 24 * 3600e3 && (!current || x.id !== current));
  const box = $('#recent'); box.replaceChildren();
  if (!r.length) return;
  box.append('Your splits from today: ');
  r.forEach((x, i) => { const a = el('a', null, x.name); a.href = '#' + x.id; box.append(i ? ' · ' : '', a); });
}

async function loadLimits() {
  try {
    LIM = await (await fetch('/api/limits')).json();
    const parts = [`Up to ${LIM.max_minutes} minutes per song`, `${LIM.left} of ${LIM.per_day} free splits left today`, 'files deleted after 24 hours'];
    $('#limits').textContent = parts.join(' · ');
    if (LIM.site_full) showMsg("We've hit today's limit for free splits. Please come back tomorrow.");
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
  box.append("The \"verify you're human\" check couldn't load, so uploads are paused. It's usually an ad blocker, a VPN or iCloud Private Relay blocking Cloudflare. Allow this site (or switch them off), then ");
  const b = el('button', 'btn', 'try again'); b.style.marginLeft = '4px';
  b.onclick = () => { showMsg(''); if (window.turnstile && tsWidget != null) turnstile.reset(tsWidget); else location.reload(); };
  box.append(b, '.');
  m.append(box);
}
// The Turnstile script itself was blocked from loading
setTimeout(() => { if (SITE.turnstile && !window.turnstile) showTsHelp(); }, 8000);
function resetTurnstile() { tsToken = ''; if (tsWidget != null && window.turnstile) turnstile.reset(tsWidget); }

// ---------- upload
const drop = $('#drop'), input = $('#file');
drop.addEventListener('click', () => !busy && input.click());
drop.addEventListener('keydown', e => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); input.click(); } });
['dragenter', 'dragover'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) start(f); });
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
  const ready = await waitToken(); busy = false;
  if (!ready) return showMsg('Please finish the "verify you are human" check above, then try again.');
  busy = true; stopMixer(); openJob(f.name); setStep('upload', 0);
  const fd = new FormData(); fd.append('file', f, f.name); fd.append('token', tsToken);
  const x = new XMLHttpRequest();
  x.open('POST', '/api/jobs');
  x.upload.onprogress = e => { if (e.lengthComputable) setStep('upload', Math.round(e.loaded / e.total * 100)); };
  x.onload = () => {
    busy = false; resetTurnstile();
    let d = {}; try { d = JSON.parse(x.responseText); } catch (_) {}
    if (x.status !== 200) { $('#job').style.display = 'none'; showMsg(d.error || 'Upload failed. Please try again.'); loadLimits(); return; }
    store.add(d.id, f.name); location.hash = d.id; loadLimits();
  };
  x.onerror = () => { busy = false; resetTurnstile(); $('#job').style.display = 'none'; showMsg('Upload failed. Check your connection and try again.'); };
  x.send(fd);
}

// ---------- job status
function openJob(name) {
  $('#job').style.display = 'block'; $('#jname').textContent = name; $('#jmeta').textContent = '';
  $('#result').hidden = true; $('#steps').hidden = false; $('#tracks').replaceChildren(); $('#cancelrow').hidden = true;
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
function stepOf(j) {
  if (j.status === 'queued') return 'queue';
  const s = (j.stage || '').toLowerCase();
  if (s.includes('kit')) return 'kit';
  if (s.includes('preview')) return 'finish';
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
  if (j.status === 'done') return showResult(id, j);
  if (j.status === 'error') { $('#steps').hidden = true; showMsg(j.error || 'Something went wrong.'); loadLimits(); return; }
  showMsg(''); setStep(stepOf(j));
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
  current = id; stopMixer();
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
  $('#steps').hidden = true; $('#result').hidden = false; showMsg('');
  $('#jmeta').textContent = [j.bpm ? Math.round(j.bpm) + ' BPM' : null, j.duration ? mmss(j.duration) : null].filter(Boolean).join(' · ');
  $('#zipkit').href = `/api/jobs/${id}/zip?kit=1`; $('#zipall').href = `/api/jobs/${id}/zip`;
  const left = Math.max(0, j.expires - Date.now() / 1000);
  $('#expire').textContent = `These files are deleted in about ${Math.max(1, Math.round(left / 3600))} hours. Bookmark this page to come back.` + (j.note ? ' ' + j.note : '');
  buildMixer(id, j);
}
function gainOn(t) { const solo = MX.tracks.some(x => x.solo); return (solo ? t.solo : !t.mute) ? 1 : 0; }
function apply() { for (const t of MX.tracks) { const on = gainOn(t); t.row.classList.toggle('off', !on); t.mb.classList.toggle('on', t.mute); t.sb.classList.toggle('on', t.solo); t.mb.setAttribute('aria-pressed', t.mute); t.sb.setAttribute('aria-pressed', t.solo); if (t.gain) t.gain.gain.setTargetAtTime(on, AC.currentTime, .01); } }
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
  AC.resume(); const t0 = AC.currentTime + .06; MX.t0 = t0;
  for (const t of MX.tracks) { const s = AC.createBufferSource(); s.buffer = t.buf; s.connect(t.gain); s.start(t0, MX.off); t.src = s; }
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
async function buildMixer(id, j) {
  AC = AC || new (window.AudioContext || window.webkitAudioContext)();
  const box = $('#tracks'); box.replaceChildren(); MX = { tracks: [], len: 0, off: 0, playing: false, ready: false }; const me = MX;
  for (const grp of ['kit', 'mix']) {
    const list = j.stems.filter(x => x.group === grp); if (!list.length) continue;
    box.append(el('div', 'grp', grp === 'kit' ? 'Kit pieces' : 'Rest of the song'));
    for (const x of list) {
      const row = el('div', 'trk'), lb = el('span', 'lb', x.label); if (x.quiet) lb.append(el('small', null, 'silent'));
      const mb = el('button', 'm', 'M'), sb = el('button', 's', 'S'), cv = document.createElement('canvas'), dl = el('a', null, 'WAV');
      mb.title = 'Mute ' + x.label; sb.title = 'Solo ' + x.label; mb.setAttribute('aria-label', 'Mute ' + x.label); sb.setAttribute('aria-label', 'Solo ' + x.label);
      dl.href = `/api/jobs/${id}/files/${encodeURIComponent(x.name)}?dl=1`; dl.title = 'Download ' + x.label + ' (WAV)';
      row.append(lb, mb, sb, cv, dl); box.append(row);
      const t = { x, row, mb, sb, cv, mute: grp !== 'kit', solo: false, buf: null, gain: null, src: null, img: null };
      mb.onclick = () => { t.mute = !t.mute; apply(); };
      sb.onclick = e => { if (!e.shiftKey && !e.metaKey && !t.solo) MX.tracks.forEach(o => o.solo = false); t.solo = !t.solo; apply(); };
      cv.onclick = e => { if (!MX.ready) return; const r = cv.getBoundingClientRect(); const p = (e.clientX - r.left) / r.width * MX.len; if (MX.playing) play(p); else { MX.off = p; draw(); } };
      MX.tracks.push(t);
    }
  }
  apply(); draw();
  try {
    await Promise.all(MX.tracks.map(async t => {
      const r = await fetch(`/api/jobs/${id}/preview/${encodeURIComponent(t.x.preview)}`);
      t.buf = await AC.decodeAudioData(await r.arrayBuffer()); if (MX !== me) return;
      t.gain = AC.createGain(); t.gain.connect(AC.destination); peaks(t);
    }));
  } catch (_) { if (MX === me) $('#time').textContent = 'Preview unavailable, downloads still work.'; return; }
  if (MX !== me) return; MX.len = Math.max(...MX.tracks.map(t => t.buf.duration)); MX.ready = true; apply(); draw();
}
$('#play').onclick = () => { if (!MX) return; if (MX.playing) stopMixer(true); else play(); };
$('#stop').onclick = () => stopMixer(false);
$('#reset').onclick = () => { if (!MX) return; MX.tracks.forEach(t => { t.solo = false; t.mute = t.x.group !== 'kit'; }); apply(); };
$('#again').onclick = () => { stopMixer(false); current = null; history.replaceState(null, '', location.pathname); $('#job').style.display = 'none'; renderRecent(); loadLimits(); window.scrollTo({ top: 0, behavior: 'smooth' }); };
document.addEventListener('keydown', e => { if (e.code === 'Space' && MX && MX.ready && !/INPUT|SELECT|TEXTAREA|BUTTON|A/.test(document.activeElement.tagName)) { e.preventDefault(); $('#play').click(); } });
window.addEventListener('resize', () => { if (MX && MX.ready) { MX.tracks.forEach(peaks); draw(); } });

loadLimits(); initTurnstile(); renderRecent(); route();
})();
