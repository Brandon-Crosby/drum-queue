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

for (const variant of ['local result reload', 'local without cosmetic footer', 'hosted result reload']) {
  test(variant, async () => {
    const errors = [], requests = [];
    const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(e.message));
    const dom = new JSDOM(html, {url: 'http://localhost/#0123456789abcdef', runScripts: 'outside-only', virtualConsole: vc});
    const w = dom.window;
    try {
      w.HTMLElement.prototype.scrollIntoView = function () {};
      w.SITE = {local: !variant.startsWith('hosted')};
      if (variant.includes('without')) w.document.querySelector('.foot').remove();
      w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){}, drawImage(){}, fillRect(){}});
      w.AudioContext = class {
        constructor() {this.currentTime = 0; this.destination = {};}
        createGain() {return {connect(){}, disconnect(){}, gain:{setValueAtTime(){}, setTargetAtTime(){}}};}
        createBiquadFilter() {return {...this.createGain(), frequency:{value:0}, Q:{value:0}};}
        async decodeAudioData() {return {duration:12, getChannelData:()=>new Float32Array(1000)};}
      };
      const job = {status:'done', name:'127.5 bpm - swing.wav', bpm:127.5, duration:12, expires:Date.now()/1000+3600,
        stems:[{stem:'kick', label:'kick', group:'kit', name:'kick.wav', preview:'kick.mp3'}],
        shots:[{label:'Kick 1', name:'Kick 1.wav', seconds:.2, count:10}], groove:true};
      w.fetch = async url => {
        requests.push(url);
        return {ok:true, json:async()=>url==='/api/limits' ? {left:3,per_day:3,max_minutes:8,max_mb:30} : job,
          arrayBuffer:async()=>new ArrayBuffer(8)};
      };
      w.eval(previewScript);
      w.eval(script);
      await new Promise(resolve=>setImmediate(resolve));
      assert.ok(requests.includes('/api/limits'), 'startup must reach limits');
      assert.ok(requests.includes('/api/jobs/0123456789abcdef'), 'reopened result must poll its job');
      assert.equal(w.document.querySelector('#time').textContent, '0:00 / 0:12', 'audio graph must finish loading');
      const eqButton = w.document.querySelector('[aria-label="EQ for kick"]');
      assert.equal(eqButton.getAttribute('aria-expanded'), 'false');
      eqButton.click();
      assert.equal(eqButton.getAttribute('aria-expanded'), 'true');
      assert.equal(w.document.querySelector('[aria-label="kick mid"]').value, '1.5');
      assert.equal(w.document.querySelector('#jmeta').textContent, '127.5 BPM · 0:12');
      assert.equal(w.document.querySelector('#groovemid').getAttribute('download'), 'Groove.mid');
      w.dispatchEvent(new w.Event('themechange'));
      assert.deepEqual(errors, [], 'startup and theme changes must not throw');
      if (w.SITE.local) assert.equal(w.document.querySelector('#faq').hidden, true);
    } finally {w.close();}
  });
}
