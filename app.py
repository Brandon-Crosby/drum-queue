#!/usr/bin/env python3
"""Drum Queue - drop in songs, get the drum kit back as separate stems.

  python app.py                      # http://127.0.0.1:8766
  python app.py --inbox ~/Drum_Inbox --out ~/DrumSplits

Each song: full mix -> drums (Demucs) -> kick / snare / toms / hi-hat / ride / crash (DrumSep).
Two ways in: drag files or whole folders onto the page, or drop audio into the inbox folder.
Jobs are saved to disk, so closing the app mid-queue just resumes next time.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

import engine

HERE = Path(__file__).resolve().parent


def _jsonable(o):
    return o.item() if hasattr(o, 'item') else str(o)   # numpy scalars etc.


def _copy(x):
    return json.loads(json.dumps(x, default=_jsonable))
DEFAULT_OPTS = dict(model='htdemucs_ft', overlap=4, keep_full=True)


# ------------------------------------------------------------------ job store
class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.jobs = []
        if self.path.exists():
            try:
                self.jobs = json.loads(self.path.read_text())
            except Exception:
                self.jobs = []
        for j in self.jobs:                    # interrupted last time -> run again
            if j['status'] == 'running':
                j['status'], j['stage'] = 'queued', 'Resuming'
        self._save()

    def _save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.jobs, indent=1, default=_jsonable))
        tmp.replace(self.path)

    def add(self, job):
        with self.lock:
            self.jobs.append(job)
            self._save()

    def update(self, jid, **kw):
        with self.lock:
            for j in self.jobs:
                if j['id'] == jid:
                    j.update(kw)
                    break
            self._save()

    def get(self, jid):
        with self.lock:
            for j in self.jobs:
                if j['id'] == jid:
                    return _copy(j)
        return None

    def next_queued(self):
        with self.lock:
            for j in self.jobs:
                if j['status'] == 'queued':
                    return _copy(j)
        return None

    def snapshot(self):
        with self.lock:
            return _copy(self.jobs)

    def clear_finished(self):
        with self.lock:
            gone = [j for j in self.jobs if j['status'] not in ('queued', 'running')]
            self.jobs = [j for j in self.jobs if j['status'] in ('queued', 'running')]
            self._save()
        for j in gone:                         # failed jobs kept their upload for Retry
            if j['status'] == 'error':
                shutil.rmtree(Path(j['src']).parent, ignore_errors=True)


def clean_opts(get, base):
    m = get('model', base['model'])
    try:
        ov = int(get('overlap', base['overlap']))
    except (TypeError, ValueError):
        ov = base['overlap']
    kf = get('keep_full', base['keep_full'])
    kf = kf if isinstance(kf, bool) else str(kf).lower() in ('1', 'true', 'on', 'yes')
    return dict(model=m if m in engine.MIX_MODELS else base['model'],
                overlap=ov if ov in (2, 4, 8, 16) else base['overlap'], keep_full=kf)


def new_job(name, src, opts, source):
    return dict(id=uuid.uuid4().hex[:10], name=name, src=str(src), opts=opts, source=source,
                status='queued', stage='Waiting', note='', error='', stems=[], out_dir='',
                bpm=None, duration=0, added=time.time(), started=0, finished=0)


# ------------------------------------------------------------------- threads
def worker_loop(store, cfg):
    models = engine.Models(cfg['data'] / 'work')
    while True:
        job = store.next_queued()
        if not job:
            time.sleep(0.7)
            continue
        jid = job['id']
        store.update(jid, status='running', stage='Starting', started=time.time(), error='', note='')
        try:
            res = engine.process(job, cfg['out'], models, lambda s, jid=jid: store.update(jid, stage=s))
            store.update(jid, status='done', stage='Done', finished=time.time(), **res)
            if not cfg['keep_uploads']:
                shutil.rmtree(Path(job['src']).parent, ignore_errors=True)
        except Exception as e:                 # keep the source so Retry works
            msg = engine.friendly_error(e)
            if type(e).__name__ in ('NoBackendError', 'LibsndfileError', 'DecodeError'):
                msg = "Couldn't read this as audio (corrupt file, or a format that needs ffmpeg, like some .m4a files)."
            store.update(jid, status='error', stage='Failed', finished=time.time(), error=msg[:600])


def inbox_loop(store, cfg):
    inbox = cfg['inbox']
    seen = {}
    while True:
        try:
            for p in sorted(inbox.iterdir()):
                if not p.is_file() or p.name.startswith('.') or p.suffix.lower() not in engine.AUDIO_EXTS:
                    continue
                st = p.stat()
                sig = (st.st_size, st.st_mtime)
                if seen.get(p) == sig and st.st_size > 0:   # unchanged for one scan: done copying
                    d = cfg['uploads'] / uuid.uuid4().hex[:10]
                    d.mkdir(parents=True)
                    shutil.move(str(p), str(d / p.name))
                    store.add(new_job(p.name, d / p.name, dict(cfg['opts']), 'inbox'))
                    seen.pop(p, None)
                else:
                    seen[p] = sig
        except Exception:
            pass
        time.sleep(3)


# ----------------------------------------------------------------------- web
def create_app(store, cfg):
    app = Flask(__name__)

    @app.get('/')
    def index():
        return Response(PAGE, mimetype='text/html')

    @app.get('/api/state')
    def state():
        return jsonify(jobs=store.snapshot(), ready=engine.separator_available(), now=time.time(),
                       inbox=str(cfg['inbox']), out=str(cfg['out']), opts=cfg['opts'],
                       can_reveal=sys.platform == 'darwin')

    @app.post('/api/upload')
    def upload():
        opts = clean_opts(request.form.get, cfg['opts'])
        made = 0
        for f in request.files.getlist('files'):
            name = re.sub(r'[\x00-\x1f/\\]', '_', Path(f.filename or '').name)
            if Path(name).suffix.lower() not in engine.AUDIO_EXTS:
                continue
            d = cfg['uploads'] / uuid.uuid4().hex[:10]
            d.mkdir(parents=True)
            f.save(str(d / name))
            store.add(new_job(name, d / name, opts, 'upload'))
            made += 1
        if not made:
            return 'No supported audio files in that upload.', 400
        return jsonify(queued=made)

    @app.post('/api/retry/<jid>')
    def retry(jid):
        j = store.get(jid)
        if not j or j['status'] != 'error' or not Path(j['src']).exists():
            abort(400)
        store.update(jid, status='queued', stage='Waiting', error='')
        return jsonify(ok=True)

    @app.post('/api/clear')
    def clear():
        store.clear_finished()
        return jsonify(ok=True)

    @app.post('/api/reveal/<jid>')
    def reveal(jid):
        j = store.get(jid)
        if not j or not j['out_dir'] or sys.platform != 'darwin':
            abort(404)
        subprocess.Popen(['open', str(cfg['out'] / j['out_dir'])])
        return jsonify(ok=True)

    def _stem_path(jid, name, sub=''):
        j = store.get(jid)
        if not j or name not in [s['name'] for s in j.get('stems', [])]:
            abort(404)
        p = (cfg['out'] / j['out_dir'] / sub / name).resolve()
        if cfg['out'].resolve() not in p.parents or not p.exists():
            abort(404)
        return j, p

    @app.get('/files/<jid>/<name>')
    def one_file(jid, name):
        j, p = _stem_path(jid, name)
        return send_file(p, as_attachment=True, download_name=f"{Path(j['name']).stem} - {name}",
                         mimetype='audio/wav')

    @app.get('/preview/<jid>/<name>')
    def preview(jid, name):
        _, p = _stem_path(jid, name, 'preview')
        return send_file(p, mimetype='audio/wav', conditional=True)

    def _zip(jobs, fname, kit_only):
        # Written to a temp file, not memory: an album of WAV stems can be several GB.
        tmp = tempfile.NamedTemporaryFile(dir=cfg['data'], suffix='.zip', delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_STORED) as z:   # WAV barely compresses
            for j in jobs:
                for s in j['stems']:
                    if kit_only and s['group'] != 'kit':
                        continue
                    p = cfg['out'] / j['out_dir'] / s['name']
                    if p.exists():
                        z.write(p, f"{j['out_dir']}/{s['name']}")
        f = open(tmp.name, 'rb')
        Path(tmp.name).unlink()        # macOS/Linux keep it readable until the download closes
        return send_file(f, as_attachment=True, download_name=fname, mimetype='application/zip')

    @app.get('/download/all.zip')
    def all_zip():
        kit = request.args.get('kit') == '1'
        return _zip([j for j in store.snapshot() if j['status'] == 'done'],
                    'drum_kits.zip' if kit else 'drum_stems_all.zip', kit)

    @app.get('/download/<jid>.zip')
    def job_zip(jid):
        j = store.get(jid)
        if not j or j['status'] != 'done':
            abort(404)
        kit = request.args.get('kit') == '1'
        return _zip([j], f"{j['out_dir']}{' - kit' if kit else ''}.zip", kit)

    return app


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drum Queue</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1c1a;--mute:#76736c;--line:#e4e1da;--acc:#d2462b;--ok:#1f8a4c;--err:#c43b3b;--run:#b7791f;--wave:#d2462b}
@media(prefers-color-scheme:dark){:root{--bg:#161513;--card:#201f1c;--ink:#f0eee9;--mute:#9a968d;--line:#34322d;--acc:#ff7a5c;--ok:#4cc27f;--err:#ff7a7a;--run:#e3b05a;--wave:#ff7a5c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:880px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:22px;margin:0 0 4px}p.sub{margin:0 0 20px;color:var(--mute)}
#drop{border:2px dashed var(--line);border-radius:14px;background:var(--card);padding:34px 16px;text-align:center;transition:.15s}
#drop.over{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 8%,var(--card))}
#drop b{font-size:17px}#drop small{display:block;color:var(--mute);margin-top:6px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}
button,label.btn{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 13px;border-radius:9px;cursor:pointer}
button:hover,label.btn:hover{border-color:var(--acc)}button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
.opts{display:flex;gap:10px 18px;flex-wrap:wrap;margin:14px 0 0;color:var(--mute);font-size:14px;align-items:center}
.opts select{font:inherit;font-size:14px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:3px 6px}
.opts label{cursor:pointer}
#upstat{color:var(--mute);margin-top:10px;min-height:20px}
#warn{display:none;background:color-mix(in srgb,var(--err) 10%,var(--card));border:1px solid var(--err);border-radius:10px;padding:10px 14px;margin-top:14px;font-size:14px}
.bar{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin:26px 0 10px}.bar h2{font-size:16px;margin:0}
.job{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:9px}
.top{display:flex;justify-content:space-between;gap:10px;align-items:baseline}.nm{font-weight:600;word-break:break-word}
.pill{font-size:12px;padding:2px 9px;border-radius:99px;border:1px solid currentColor;white-space:nowrap}
.queued{color:var(--mute)}.running{color:var(--run)}.done{color:var(--ok)}.error{color:var(--err)}
.meta{color:var(--mute);font-size:13px;margin-top:4px}.err{color:var(--err);font-size:13px;margin-top:4px;word-break:break-word}
.links{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}.links a{font-size:13px;color:var(--acc);text-decoration:none;border:1px solid var(--line);padding:3px 10px;border-radius:8px}
.links a:hover{border-color:var(--acc)}.links a.q{color:var(--mute)}.empty{color:var(--mute);text-align:center;padding:26px}
#pv{display:none;background:var(--card);border:1px solid var(--acc);border-radius:12px;padding:12px 14px;margin:18px 0 0}
#pv h3{margin:0 0 8px;font-size:15px;display:flex;justify-content:space-between;gap:10px;align-items:center}
.pvrow{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:4px 0 10px;font-size:13px;color:var(--mute)}
.grp{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em;margin:10px 0 4px}
.trk{display:grid;grid-template-columns:72px 30px 30px 1fr 28px;gap:6px;align-items:center;margin-bottom:5px}
.trk .lb{font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.trk button{padding:2px 0;font-size:12px;font-weight:600;border-radius:6px}
.trk button.on.m{background:var(--mute);color:var(--card);border-color:var(--mute)}.trk button.on.s{background:var(--run);color:#fff;border-color:var(--run)}
.trk canvas{width:100%;height:38px;display:block;background:var(--bg);border-radius:6px;cursor:pointer}
.trk.off canvas{opacity:.35}.trk a{text-align:center;text-decoration:none;color:var(--acc);font-size:15px}
.info{color:var(--mute);font-size:13px;margin-top:22px}code{background:var(--line);padding:1px 6px;border-radius:5px}
@media(max-width:520px){.trk{grid-template-columns:56px 28px 28px 1fr 22px}}
</style></head><body><main>
<h1>Drum Queue</h1><p class="sub">Drop in songs or whole folders. Each one is split into kick, snare, toms, hi-hat, ride and crash, one at a time in the background.</p>
<div id="drop"><b>Drag songs or folders here</b><small>mp3 · wav · flac · m4a · aiff · ogg</small>
<div class="row" style="justify-content:center">
<label class="btn">Choose files<input id="pf" type="file" multiple accept="audio/*,.mp3,.wav,.flac,.m4a,.aiff,.aif,.ogg,.opus" hidden></label>
<label class="btn">Choose folder<input id="pd" type="file" webkitdirectory multiple hidden></label></div></div>
<div class="opts">
<label>Drums from mix: <select id="model">
 <option value="htdemucs_ft">Best (htdemucs_ft)</option>
 <option value="htdemucs">Faster (htdemucs)</option>
 <option value="htdemucs_6s">6-stem (adds guitar &amp; piano)</option></select></label>
<label>Kit split: <select id="overlap">
 <option value="4">Normal (like Xport)</option>
 <option value="8">Extra clean (2× slower)</option></select></label>
<label><input type="checkbox" id="keepfull" checked> Also keep bass / vocals / other</label></div>
<div id="warn"></div>
<div id="upstat"></div>
<div id="pv"><h3><span id="pvt"></span><button id="pvx">Close</button></h3>
<div class="pvrow"><button class="pri" id="pvplay">▶ Play</button><button id="pvstop">Stop</button>
<button id="pvall">Unmute kit</button><span id="pvtime"></span></div>
<div id="pvtracks"></div></div>
<div class="bar"><h2 id="count">Queue</h2><div class="row" style="margin:0">
<button id="clr">Clear finished</button><button id="zipkit">Download all kits (.zip)</button><button class="pri" id="zip">Download everything (.zip)</button></div></div>
<div id="jobs"></div>
<div class="info" id="info"></div>
</main><script>
const $=s=>document.querySelector(s);const AUD=/\.(mp3|wav|flac|m4a|aiff?|ogg|opus)$/i;
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
const mmss=s=>{s=Math.max(0,Math.floor(s));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0')};
// remember the option picks between visits
for(const id of ['model','overlap','keepfull']){const e=$('#'+id);try{const v=localStorage.getItem('dq_'+id);if(v!=null){if(e.type==='checkbox')e.checked=v==='1';else e.value=v}}catch(_){}
 e.onchange=()=>{try{localStorage.setItem('dq_'+id,e.type==='checkbox'?(e.checked?'1':'0'):e.value)}catch(_){}}}
async function walk(e,out){if(e.isFile){out.push(await new Promise((r,j)=>e.file(r,j)))}
 else if(e.isDirectory){const rd=e.createReader();let b;do{b=await new Promise((r,j)=>rd.readEntries(r,j));for(const c of b)await walk(c,out)}while(b.length)}}
async function send(files){files=files.filter(f=>AUD.test(f.name)&&!f.name.startsWith('._'));
 if(!files.length){$('#upstat').textContent='No audio files found in that drop.';return}let bad=0;
 for(let i=0;i<files.length;i++){$('#upstat').textContent='Uploading '+(i+1)+' of '+files.length+'…';
  const fd=new FormData();fd.append('model',$('#model').value);fd.append('overlap',$('#overlap').value);fd.append('keep_full',$('#keepfull').checked?'1':'0');
  fd.append('files',files[i],files[i].name);
  try{const r=await fetch('/api/upload',{method:'POST',body:fd});if(!r.ok)bad++}catch(e){bad++}refresh()}
 $('#upstat').textContent=bad?('Queued '+(files.length-bad)+', '+bad+' failed to upload.'):'Queued '+files.length+' file'+(files.length>1?'s':'')+'.';}
const dz=$('#drop');
['dragenter','dragover'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.add('over')}));
['dragleave','drop'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.remove('over')}));
dz.addEventListener('drop',async e=>{const es=[...e.dataTransfer.items].map(i=>i.webkitGetAsEntry&&i.webkitGetAsEntry()).filter(Boolean);
 const fs=[];for(const x of es)await walk(x,fs);send(fs)});
$('#pf').onchange=e=>{send([...e.target.files]);e.target.value=''};
$('#pd').onchange=e=>{send([...e.target.files]);e.target.value=''};
$('#clr').onclick=async()=>{await fetch('/api/clear',{method:'POST'});refresh()};
$('#zip').onclick=()=>location.href='/download/all.zip';
$('#zipkit').onclick=()=>location.href='/download/all.zip?kit=1';
async function retry(id){await fetch('/api/retry/'+id,{method:'POST'});refresh()}
const MODEL_NAME={htdemucs_ft:'best',htdemucs:'faster',htdemucs_6s:'6-stem'};
function render(s){
 const w=$('#warn');if(!s.ready){w.style.display='block';w.textContent='audio-separator is not installed in this environment. Run ./setup.sh first.'}else w.style.display='none';
 $('#info').textContent='';$('#info').append('Inbox: ',el('code',null,s.inbox),'  ·  Output: ',el('code',null,s.out));
 const js=s.jobs.slice().reverse();const wrap=$('#jobs');wrap.replaceChildren();
 const q=s.jobs.filter(j=>j.status==='queued'||j.status==='running').length,d=s.jobs.filter(j=>j.status==='done').length;
 $('#count').textContent='Queue · '+q+' waiting · '+d+' done';
 if(!js.length){wrap.append(el('div','empty','Nothing yet. Drop some songs above.'));return}
 for(const j of js){const c=el('div','job');const t=el('div','top');
  const pill=j.status==='running'?j.stage+' · '+mmss(s.now-j.started):j.status;
  t.append(el('span','nm',j.name),el('span','pill '+j.status,pill));c.append(t);
  const o=j.opts||{};const how=(MODEL_NAME[o.model]||o.model)+(o.overlap>4?' · extra clean':'');
  if(j.status==='queued')c.append(el('div','meta',how));
  if(j.status==='done'){
   const kit=j.stems.filter(x=>x.group==='kit'),mix=j.stems.filter(x=>x.group==='mix');
   c.append(el('div','meta',[j.bpm?Math.round(j.bpm)+' BPM (detected)':null,j.duration?mmss(j.duration):null,
     kit.length+' kit pieces'+(mix.length?' + '+mix.length+' mix stems':''),how,'took '+mmss(j.finished-j.started)].filter(Boolean).join(' · ')));
   const l=el('div','links');const pb=el('a',null,'▶ Mixer');pb.href='#';pb.onclick=ev=>{ev.preventDefault();openMixer(j)};l.append(pb);
   for(const x of kit){const a=el('a',x.quiet?'q':null,x.label+(x.quiet?' (silent)':''));a.href='/files/'+j.id+'/'+encodeURIComponent(x.name);a.title=x.quiet?'Nothing detected for this piece':'Download '+x.name;l.append(a)}
   const zk=el('a',null,'kit (.zip)');zk.href='/download/'+j.id+'.zip?kit=1';l.append(zk);
   if(mix.length){const za=el('a',null,'all stems (.zip)');za.href='/download/'+j.id+'.zip';l.append(za)}
   if(s.can_reveal){const op=el('a',null,'Open folder');op.href='#';op.onclick=ev=>{ev.preventDefault();fetch('/api/reveal/'+j.id,{method:'POST'})};l.append(op)}
   c.append(l)}
  if(j.note)c.append(el('div','meta',j.note));
  if(j.status==='error'){c.append(el('div','err',j.error));const b=el('button',null,'Retry');b.style.marginTop='8px';b.onclick=()=>retry(j.id);c.append(b)}
  wrap.append(c)}}

// ---------------- kit mixer (outside #jobs so the refresh never rebuilds it)
let MX=null,AC=null,raf=0;
function mxGainOf(t){const solo=MX.tracks.some(x=>x.solo);return (solo?t.solo:!t.mute)?1:0}
function mxApply(){for(const t of MX.tracks){const on=mxGainOf(t);t.row.classList.toggle('off',!on);
 t.mb.classList.toggle('on',t.mute);t.sb.classList.toggle('on',t.solo);if(t.gain)t.gain.gain.setTargetAtTime(on,AC.currentTime,.01)}}
function mxPos(){return MX.playing?Math.min(MX.len,MX.off+AC.currentTime-MX.t0):MX.off}
function mxStop(keepPos){if(!MX)return;if(MX.playing){const p=mxPos();for(const t of MX.tracks){try{t.src&&t.src.stop()}catch(e){}t.src=null}
 MX.playing=false;MX.off=keepPos?p:0}else if(!keepPos)MX.off=0;$('#pvplay').textContent='▶ Play';mxDraw()}
function mxPlay(from){if(!MX||!MX.ready)return;mxStop(true);if(from!=null)MX.off=from;if(MX.off>=MX.len-.05)MX.off=0;
 AC.resume();const t0=AC.currentTime+.06;MX.t0=t0;
 for(const t of MX.tracks){const src=AC.createBufferSource();src.buffer=t.buf;src.connect(t.gain);src.start(t0,MX.off);t.src=src}
 MX.playing=true;$('#pvplay').textContent='❚❚ Pause';mxApply();cancelAnimationFrame(raf);
 const tick=()=>{if(!MX||!MX.playing)return;if(mxPos()>=MX.len){mxStop(false);return}mxDraw();raf=requestAnimationFrame(tick)};tick()}
function mxDraw(){if(!MX)return;const p=mxPos();const ink=getComputedStyle(document.documentElement).getPropertyValue('--ink');
 for(const t of MX.tracks){const cv=t.cv,g=cv.getContext('2d');const W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  if(t.img)g.drawImage(t.img,0,0);g.fillStyle=ink;g.fillRect(p/MX.len*W,0,Math.max(1.5,devicePixelRatio),H)}
 $('#pvtime').textContent=mmss(p)+' / '+mmss(MX.len)}
function peaks(t){const cv=t.cv,dpr=devicePixelRatio||1;cv.width=Math.max(10,cv.clientWidth*dpr);cv.height=cv.clientHeight*dpr;
 const W=cv.width,H=cv.height,off=document.createElement('canvas');off.width=W;off.height=H;const g=off.getContext('2d');
 g.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--wave');
 const d=t.buf.getChannelData(0),step=Math.max(1,Math.floor(d.length/W));let mx=0;const col=new Float32Array(W);
 for(let x=0;x<W;x++){let m=0;const a=x*step,b=Math.min(d.length,a+step);for(let i=a;i<b;i+=4){const v=Math.abs(d[i]);if(v>m)m=v}col[x]=m;if(m>mx)mx=m}
 const sc=mx>0?(H/2-1)/Math.max(mx,.25):0;for(let x=0;x<W;x++){const h=Math.max(.5,col[x]*sc);g.fillRect(x,H/2-h,1,h*2)}t.img=off}
async function openMixer(j){if(MX)mxStop(false);AC=AC||new (window.AudioContext||window.webkitAudioContext)();
 $('#pv').style.display='block';$('#pvt').textContent='Mixer · '+j.name;$('#pvtime').textContent='Loading…';
 const box=$('#pvtracks');box.replaceChildren();MX={job:j,tracks:[],len:0,off:0,playing:false,ready:false};const me=MX;
 for(const grp of ['kit','mix']){const list=j.stems.filter(x=>x.group===grp);if(!list.length)continue;
  box.append(el('div','grp',grp==='kit'?'Kit pieces':'From the mix'));
  for(const x of list){const row=el('div','trk');const lb=el('span','lb',x.label);lb.title=x.name;
   const mb=el('button','m','M'),sb=el('button','s','S'),cv=document.createElement('canvas'),dl=el('a',null,'↓');
   mb.title='Mute';sb.title='Solo';dl.href='/files/'+j.id+'/'+encodeURIComponent(x.name);dl.title='Download full-quality WAV';
   row.append(lb,mb,sb,cv,dl);box.append(row);
   // default: hear the kit pieces; drums + other mix stems start muted
   const t={x,row,mb,sb,cv,mute:grp!=='kit',solo:false,buf:null,gain:null,src:null,img:null};
   mb.onclick=()=>{t.mute=!t.mute;mxApply()};sb.onclick=e=>{if(!e.shiftKey&&!e.metaKey&&!t.solo)MX.tracks.forEach(o=>o.solo=false);t.solo=!t.solo;mxApply()};
   cv.onclick=e=>{const r=cv.getBoundingClientRect();const p=(e.clientX-r.left)/r.width*MX.len;if(MX.playing)mxPlay(p);else{MX.off=p;mxDraw()}};
   MX.tracks.push(t)}}
 mxApply();$('#pv').scrollIntoView({behavior:'smooth',block:'nearest'});
 await Promise.all(MX.tracks.map(async t=>{const r=await fetch('/preview/'+j.id+'/'+encodeURIComponent(t.x.name));
  t.buf=await AC.decodeAudioData(await r.arrayBuffer());if(MX!==me)return;t.gain=AC.createGain();t.gain.connect(AC.destination);peaks(t)}));
 if(MX!==me)return;MX.len=Math.max(...MX.tracks.map(t=>t.buf.duration));MX.ready=true;mxApply();mxDraw()}
$('#pvplay').onclick=()=>{if(!MX)return;if(MX.playing)mxStop(true);else mxPlay()};
$('#pvstop').onclick=()=>mxStop(false);
$('#pvall').onclick=()=>{if(!MX)return;MX.tracks.forEach(t=>{t.solo=false;t.mute=t.x.group!=='kit'});mxApply()};
$('#pvx').onclick=()=>{mxStop(false);$('#pv').style.display='none';MX=null};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&MX&&!/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)){e.preventDefault();$('#pvplay').click()}});
window.addEventListener('resize',()=>{if(MX&&MX.ready){MX.tracks.forEach(peaks);mxDraw()}});
async function refresh(){try{render(await (await fetch('/api/state')).json())}catch(e){}}
refresh();setInterval(refresh,1500);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--out', default='~/DrumSplits', help='where stems are written (default ~/DrumSplits)')
    ap.add_argument('--inbox', default=str(HERE / 'inbox'), help='drop audio files here to queue them')
    ap.add_argument('--data', default=str(HERE / 'data'), help='job list and temporary uploads')
    ap.add_argument('--model', choices=list(engine.MIX_MODELS), default=DEFAULT_OPTS['model'], help='mix model for inbox files')
    ap.add_argument('--overlap', type=int, choices=[2, 4, 8, 16], default=DEFAULT_OPTS['overlap'], help='DrumSep overlap for inbox files')
    ap.add_argument('--drums-only', action='store_true', help='inbox files: drop bass/vocals/other')
    ap.add_argument('--keep-uploads', action='store_true', help='keep the audio copy after splitting')
    a = ap.parse_args()

    data = Path(a.data).expanduser()
    cfg = dict(out=Path(a.out).expanduser(), inbox=Path(a.inbox).expanduser(), data=data,
               uploads=data / 'uploads', keep_uploads=a.keep_uploads,
               opts=dict(model=a.model, overlap=a.overlap, keep_full=not a.drums_only))
    for p in (cfg['out'], cfg['inbox'], cfg['uploads']):
        p.mkdir(parents=True, exist_ok=True)
    engine.link_xport_drumsep()
    ff = engine.ensure_ffmpeg()
    store = Store(data / 'jobs.json')
    threading.Thread(target=worker_loop, args=(store, cfg), daemon=True).start()
    threading.Thread(target=inbox_loop, args=(store, cfg), daemon=True).start()
    print(f'\n  Drum Queue running at http://127.0.0.1:{a.port}\n  Inbox:  {cfg["inbox"]}\n  Output: {cfg["out"]}\n'
          f'  Models: {engine.MODEL_DIR}\n')
    create_app(store, cfg).run(host='127.0.0.1', port=a.port, threaded=True)


if __name__ == '__main__':
    main()
