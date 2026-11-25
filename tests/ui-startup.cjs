// Test the real built page and script, with only HTTP/audio hardware substituted.
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

for (const variant of ['local result reload', 'local without cosmetic footer', 'hosted result reload', 'local missing EQ module', 'hosted missing EQ module']) {
  test(variant, async () => {
    const errors = [], requests = [], sources = [];
    let finishBass;
    const tick = () => new Promise(resolve=>setImmediate(resolve));
    const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(e.message));
    const dom = new JSDOM(html, {url: 'http://localhost/#0123456789abcdef', runScripts: 'outside-only', virtualConsole: vc});
    const w = dom.window;
    try {
      w.HTMLElement.prototype.scrollIntoView = function () {};
      w.SITE = {local: !variant.startsWith('hosted')};
      if (variant.includes('without')) w.document.querySelector('.foot').remove();
      w.requestAnimationFrame = () => 1; w.cancelAnimationFrame = () => {};
      w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){}, drawImage(){}, fillRect(){}});
      w.AudioContext = class {
        constructor() {this.currentTime = 0; this.destination = {}; w.audioContext = this;}
        resume() {return Promise.resolve();}
        createBufferSource() {const source = {connect(){}, stop(){}, start(when, offset){source.when=when;source.offset=offset;}}; sources.push(source);return source;}
        createGain() {return {connect(){}, disconnect(){}, gain:{setValueAtTime(){}, setTargetAtTime(){}}};}
        createBiquadFilter() {return {...this.createGain(), frequency:{value:0}, Q:{value:0}};}
        async decodeAudioData() {return {duration:12, getChannelData:()=>new Float32Array(1000)};}
      };
      const job = {status:'done', name:'127.5 bpm - swing.wav', bpm:127.5, duration:12, expires:Date.now()/1000+3600,
        stems:['kick','bass','vocals','drums'].map(stem=>({stem,label:stem,group:stem==='kick'?'kit':'mix',name:stem+'.wav',preview:stem+'.mp3'})),
        shots:[{label:'Kick 1', name:'Kick 1.wav', seconds:.2, count:10}], groove:true};
      w.fetch = async url => {
        requests.push(url);
        if (/\/bass\.(mp3|wav)$/.test(url)) return new Promise(resolve=>{finishBass=()=>resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});});
        return {ok:true, json:async()=>url==='/api/limits' ? {left:3,per_day:3,max_minutes:8,max_mb:30} : job,
          arrayBuffer:async()=>new ArrayBuffer(8)};
      };
      // A saved opt-in must not change kit-only defaults for a new result.
      w.localStorage.setItem('ds_preview_sound_v2', JSON.stringify({includeBass:true}));
      if (!variant.includes('missing')) w.eval(previewScript);
      w.eval(script);
      await new Promise(resolve=>setImmediate(resolve));
      assert.ok(requests.includes('/api/limits'), 'startup must reach limits');
      assert.ok(requests.includes('/api/jobs/0123456789abcdef'), 'reopened result must poll its job');
      assert.equal(w.document.querySelector('#time').textContent, '0:00 / 0:12', 'audio graph must finish loading');
      const eqButton = w.document.querySelector('[aria-label="EQ for kick"]');
      if (variant.includes('missing')) {
        assert.equal(eqButton.hidden, true, 'missing optional EQ must not block startup');
      } else {
        assert.equal(eqButton.getAttribute('aria-expanded'), 'false');
        eqButton.click();
        assert.equal(eqButton.getAttribute('aria-expanded'), 'true');
        assert.equal(w.document.querySelector('[aria-label="kick mid"]').value, '1.5');
      }
      const audioRequests = () => requests.filter(url=>/\/(preview|files)\//.test(url));
      const expected = stem => `/api/jobs/0123456789abcdef/${w.SITE.local?'files':'preview'}/${stem}.${w.SITE.local?'wav':'mp3'}`;
      assert.deepEqual(audioRequests(), [expected('kick')], 'muted backing stems must not fetch or decode up front');
      assert.equal(w.document.querySelector('[aria-label="Mute bass"]').getAttribute('aria-pressed'), 'true');
      assert.equal(w.document.querySelector('#preview-include-bass').checked, false);
      w.document.querySelector('#play').click();
      assert.equal(sources.length, 1, 'play only creates sources for decoded tracks');
      w.audioContext.currentTime = 3;
      w.document.querySelector('[aria-label="Mute bass"]').click();
      w.document.querySelector('[aria-label="Mute bass"]').click();
      w.document.querySelector('[aria-label="Mute bass"]').click();
      assert.equal(audioRequests().filter(url=>url===expected('bass')).length, 1, 'rapid toggles share one in-flight load');
      assert.equal(w.document.querySelector('[aria-label="Mute bass"]').closest('.trk').getAttribute('aria-busy'), 'true');
      finishBass(); await tick();
      assert.equal(sources.length, 2, 'unmuting during playback starts the loaded stem');
      assert.ok(Math.abs(sources[1].offset - 2.94) < .001, 'late stem joins current playback position');
      assert.equal(w.document.querySelector('#preview-include-bass').checked, true);
      w.document.querySelector('[aria-label="Solo vocals"]').click(); await tick();
      assert.ok(audioRequests().includes(expected('vocals')), 'solo loads a muted backing stem');
      assert.ok(!audioRequests().includes(expected('drums')), 'unused backing stems stay unloaded');
      w.document.querySelector('#stop').click();
      assert.equal(w.document.querySelector('#jmeta').textContent, '127.5 BPM · 0:12');
      assert.equal(w.document.querySelector('#groovemid').getAttribute('download'), 'Groove.mid');
      w.dispatchEvent(new w.Event('themechange'));
      assert.deepEqual(errors, [], 'startup and theme changes must not throw');
      if (w.SITE.local) assert.equal(w.document.querySelector('#faq').hidden, true);
    } finally {w.close();}
  });
}
