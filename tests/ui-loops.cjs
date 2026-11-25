// Loop and one-shot preview playback on the real built page: one sound at a time, and nothing keeps
// playing after Stop or after switching results. Only HTTP and audio hardware are substituted.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require('jsdom');
const dist = process.env.DS_UI_DIST;
assert.ok(dist, 'Set DS_UI_DIST to the extracted desktop package web/dist');
const html = fs.readFileSync(path.join(dist, 'index.html'), 'utf8');
const previewScript = fs.readFileSync(path.join(dist, 'preview-sound.js'), 'utf8');
const script = fs.readFileSync(path.join(dist, 'app.js'), 'utf8');
const A = '0123456789abcdef', B = 'fedcba9876543210';
const tick = () => new Promise(resolve => setImmediate(resolve));

async function page(jobs) {
  const errors = [], audios = [], sources = [];
  const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(e.message));
  const dom = new JSDOM(html, {url: `http://localhost/#${A}`, runScripts: 'outside-only', virtualConsole: vc});
  const w = dom.window;
  w.HTMLElement.prototype.scrollIntoView = function () {};
  w.SITE = {local: false};
  w.requestAnimationFrame = () => 1; w.cancelAnimationFrame = () => {};
  w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){}, drawImage(){}, fillRect(){}});
  w.AudioContext = class {
    constructor() {this.currentTime = 0; this.destination = {};}
    resume() {return Promise.resolve();}
    createBufferSource() {const s = {stopped: false, connect(){}, start(){}, stop(){s.stopped = true;}}; sources.push(s); return s;}
    createGain() {return {connect(){}, disconnect(){}, gain: {setValueAtTime(){}, setTargetAtTime(){}}};}
    createBiquadFilter() {return {...this.createGain(), frequency: {value: 0}, Q: {value: 0}};}
    async decodeAudioData() {return {duration: 12, getChannelData: () => new Float32Array(1000)};}
  };
  w.Audio = class {
    constructor(src) {this.src = src; this.paused = true; this.loop = false; audios.push(this);}
    play() {this.paused = false; return Promise.resolve();}
    pause() {this.paused = true;}
  };
  w.fetch = async url => {
    const id = (url.match(/^\/api\/jobs\/([a-f0-9]{16})$/) || [])[1];
    return {ok: true, json: async () => url === '/api/limits' ? {left: 3, per_day: 3, max_minutes: 8, max_mb: 30} : jobs[id],
      arrayBuffer: async () => new ArrayBuffer(8)};
  };
  w.eval(previewScript); w.eval(script);
  await tick(); await tick();
  const $ = s => w.document.querySelector(s);
  const playing = () => audios.filter(a => !a.paused);
  const mixerPlaying = () => $('#play').textContent.includes('Pause');
  const loopPad = label => [...w.document.querySelectorAll('#looppads .padhit')].find(b => b.textContent.startsWith(label));
  return {w, $, errors, audios, sources, playing, mixerPlaying, loopPad};
}
const job = (shots, loops) => ({status: 'done', name: 'song.wav', bpm: 120, duration: 12, expires: Date.now() / 1000 + 3600,
  stems: [{stem: 'kick', label: 'kick', group: 'kit', name: 'kick.wav', preview: 'kick.mp3'}], shots, loops, groove: false});
const SHOTS = [{label: 'Kick 1', name: 'Kick 1.wav', seconds: .2, count: 10}];
const LOOPS = [{label: 'Kick loop', name: 'song - Kick loop 120 BPM (4 bars).wav', bars: 4, bpm: 120},
               {label: 'Snare-Perc loop', name: 'song - Snare-Perc loop (4 bars).wav', bars: 4}];

test('a one-shot stops the mixer', async () => {
  const p = await page({[A]: job(SHOTS, LOOPS)});
  try {
    p.$('#play').click(); assert.ok(p.mixerPlaying());
    p.$('#pads .padhit').click();
    assert.equal(p.mixerPlaying(), false, 'mixer must stop when a one-shot plays');
    assert.ok(p.sources.every(s => s.stopped));
    assert.equal(p.playing().length, 1);
    assert.deepEqual(p.errors, []);
  } finally {p.w.close();}
});

test('loops: one at a time, mixer handoff, Stop stops them', async () => {
  const p = await page({[A]: job(SHOTS, LOOPS)});
  try {
    const kick = p.loopPad('Kick loop'), sp = p.loopPad('Snare-Perc loop');
    kick.click();
    assert.equal(p.playing().length, 1); assert.equal(p.playing()[0].loop, true);
    assert.equal(kick.getAttribute('aria-pressed'), 'true');
    sp.click();
    assert.equal(p.playing().length, 1, 'a second loop replaces the first');
    assert.equal(kick.getAttribute('aria-pressed'), 'false'); assert.equal(sp.getAttribute('aria-pressed'), 'true');
    sp.click();
    assert.equal(p.playing().length, 0, 'tapping the playing loop stops it');
    kick.click(); p.$('#play').click();
    assert.equal(p.playing().length, 0, 'the mixer stops the loop'); assert.ok(p.mixerPlaying());
    assert.equal(kick.getAttribute('aria-pressed'), 'false');
    kick.click();
    assert.equal(p.mixerPlaying(), false, 'a loop stops the mixer'); assert.equal(p.playing().length, 1);
    p.$('#stop').click();
    assert.equal(p.playing().length, 0, 'Stop stops a playing loop');
    assert.equal(kick.getAttribute('aria-pressed'), 'false', 'Stop clears the pressed loop');
    assert.deepEqual(p.errors, []);
  } finally {p.w.close();}
});

test('switching results stops a loop', async () => {
  const p = await page({[A]: job(SHOTS, LOOPS), [B]: job(SHOTS, [])});
  try {
    p.loopPad('Kick loop').click(); assert.equal(p.playing().length, 1);
    p.w.location.hash = B; p.w.dispatchEvent(new p.w.HashChangeEvent('hashchange'));
    await tick(); await tick();
    assert.equal(p.playing().length, 0, 'the previous result\'s loop must stop');
    assert.equal(p.$('#loops').hidden, true, 'a result without loops shows no loop row');
    assert.deepEqual(p.errors, []);
  } finally {p.w.close();}
});

test('loops show without one-shots', async () => {
  const p = await page({[A]: job([], LOOPS)});
  try {
    assert.equal(p.$('#shots').hidden, true);
    assert.equal(p.$('#loops').hidden, false);
    assert.equal(p.w.document.querySelectorAll('#looppads .padhit').length, 2);
    p.loopPad('Kick loop').click(); assert.equal(p.playing().length, 1);
    assert.deepEqual(p.errors, []);
  } finally {p.w.close();}
});
