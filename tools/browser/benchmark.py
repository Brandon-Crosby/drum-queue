#!/usr/bin/env python3
"""Local first-use benchmark; diagnostics only, no product UI or cloud inference.

Each invocation uses a fresh origin. Cold runs clear model caches; --reuse-cache
keeps them after the first run.
An optional shared bandwidth cap models a connection limit, not a real CDN test.
Browser/driver shader caches are not reset. Input is read before timing, as for a
file already selected by a user. Timing covers decode through returned WAVs.
"""
import argparse
import json
import math
import re
import io
import wave
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PAGE = r'''<!doctype html><meta charset="utf-8"><title>Local processing speed check</title>
<pre role="status" id="status">Starting local diagnostic…</pre>
<script src="/browser/processor.js"></script><script>
(async()=>{
 const status=document.getElementById('status');
 const config=await(await fetch('/config')).json();
 let pipeline;
 if(config.postMode){
  for(const name of ['postprocess.js','pipeline.js']){
   await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='/browser/'+name;script.onload=resolve;script.onerror=reject;document.head.append(script);});
  }
  pipeline=DrumBrowserPipeline.create({overlap:config.postMode==='overlap',reuse:config.postReuse,allowKitOnly:true});
 }
 const save=async report=>{const response=await fetch('/report',{method:'POST',body:JSON.stringify(report)});if(!response.ok)throw Error('Could not save benchmark report');};
 const bytes=await(await fetch('/input.wav')).arrayBuffer();
 const file=new File([bytes],'synthetic-drum-clip.wav',{type:'audio/wav'});
 let latestProfile;const NativeWorker=Worker;
 window.Worker=class extends NativeWorker{constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='profile')latestProfile=data.stats;});}};
 const report={schema:1,scope:'Local diagnostics; no product UI changes',
   bandwidthMbps:config.mbps,modelEncoding:config.modelEncoding,runs:[],userAgent:navigator.userAgent};
 try{
  for(let index=0;index<config.runs;index++){
   if(!config.reuseCache||index===0)await caches.delete('drumsplitter-browser-v1');
   const before=await(await fetch('/snapshot')).json();
   const entry={index:index+1,cache:config.reuseCache&&index>0?'verified cached models':'cold model cache',progress:[]};report.runs.push(entry);
   latestProfile=undefined;entry.visibility=[{ms:0,state:document.visibilityState}];entry.focus=[{ms:0,focused:document.hasFocus()}];
   const start=performance.now();const visibility=()=>entry.visibility.push({ms:performance.now()-start,state:document.visibilityState});document.addEventListener('visibilitychange',visibility);const focus=()=>entry.focus.push({ms:performance.now()-start,focused:document.hasFocus()});window.addEventListener('focus',focus);window.addEventListener('blur',focus);let previous=start,ticks=0,maxGap=0,lastProgress=-Infinity,lastStage='';
   const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),config.timeoutSeconds*1000);
   const heartbeat=setInterval(()=>{const now=performance.now();maxGap=Math.max(maxGap,now-previous);previous=now;ticks++;},100);
   let result;
   try{result=await (pipeline||DrumBrowserProcessor).process(file,{signal:controller.signal,onProgress:event=>{
    const ms=performance.now()-start;
    if(event.stage!==lastStage||ms-lastProgress>=250){entry.progress.push({ms,stage:event.stage,message:event.message});lastProgress=ms;lastStage=event.stage;}
    status.textContent='Run '+(index+1)+'/'+config.runs+'\n'+event.message;
   }});entry.elapsedMs=performance.now()-start;}
   finally{const finish=performance.now();entry.elapsedMs=finish-start;maxGap=Math.max(maxGap,finish-previous);document.removeEventListener('visibilitychange',visibility);window.removeEventListener('focus',focus);window.removeEventListener('blur',focus);entry.profile=latestProfile;clearTimeout(deadline);clearInterval(heartbeat);entry.heartbeat={ticks,maxGapMs:maxGap};}
   const after=await(await fetch('/snapshot')).json();
   entry.networkBytes=after.total-before.total;
   entry.modelBytes=after.models-before.models;
   entry.sampleAssetBytes=(after.samples||0)-(before.samples||0);
   if(entry.modelBytes!==(config.reuseCache&&index>0?0:config.expectedModelBytes))throw Error('Unexpected model transfer count for cache mode');
   entry.duration=result.duration;entry.files=[];
   for(const file of result.files){
    const view=new DataView(file.bytes),rate=view.getUint32(24,true),channels=view.getUint16(22,true),bits=view.getUint16(34,true);
    if(rate!==44100||channels!==2||bits!==24||file.bytes.byteLength!==44+Math.round(result.duration*44100)*6)throw Error('Unexpected WAV format');
    let peak=0;for(let at=44;at<file.bytes.byteLength;at+=3){let n=view.getUint8(at)|(view.getUint8(at+1)<<8)|(view.getUint8(at+2)<<16);if(n&0x800000)n-=0x1000000;peak=Math.max(peak,Math.abs(n)/8388608);}
    const sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',file.bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
    entry.files.push({name:file.name,bytes:file.bytes.byteLength,rate,channels,bits,peak,sha256});
    if(config.saveWavs){const saved=await fetch('/wave/'+(index+1)+'/'+file.name,{method:'POST',body:file.bytes});if(!saved.ok)throw Error('Could not save local WAV result');}
   }
   entry.status='passed';
   if(result.artifacts){
    entry.samples={result:result.result,artifacts:[]};
    for(const artifact of result.artifacts){
     const sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',artifact.bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
     entry.samples.artifacts.push({name:artifact.name,kind:artifact.kind,bytes:artifact.bytes.byteLength,sha256});
     if(config.saveWavs){const saved=await fetch('/artifact/'+(index+1)+'/'+artifact.kind+'/'+encodeURIComponent(artifact.name),{method:'POST',body:artifact.bytes});if(!saved.ok)throw Error('Could not save sample artifact');}
    }
   }
   await save(report);
  }
  report.status='complete';
 }catch(error){report.status='failed';report.error=String(error);}
 status.textContent=JSON.stringify(report,null,2);
 try{await save(report);}finally{pipeline?.dispose();await caches.delete('drumsplitter-browser-v1').catch(()=>{});}
})();</script>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing v1/')
    parser.add_argument('--input', type=Path, required=True, help='Authorized WAV fixture, up to 30 seconds')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--scripts', type=Path, help='Compare a locally extracted browser/ directory from another revision')
    parser.add_argument('--save-wavs', action='store_true', help='Save generated WAVs locally after timed processing for quality comparisons')
    parser.add_argument('--profile', action='store_true', help='Inject diagnostic-only stage timers into the local worker')
    parser.add_argument('--gzip-models', type=Path, help='Serve precompressed lossless .onnx.gz model copies with Content-Encoding: gzip')
    parser.add_argument('--reuse-cache', action='store_true', help='Clear model cache on run 1 only; require zero model downloads on later runs')
    parser.add_argument('--mbps', type=float, default=0, help='0 = uncapped localhost; otherwise shared simulated Mbit/s cap')
    parser.add_argument('--timeout-seconds', type=float, default=300, help='Per-run cancellation deadline')
    parser.add_argument('--post-assets', type=Path, help='Staged CPU runtime directory containing v1/')
    parser.add_argument('--post-mode', choices=['serial', 'overlap'], help='Include sample preparation in timed processing')
    parser.add_argument('--post-reuse', action='store_true', help='Keep the CPU runtime between timed jobs')
    args = parser.parse_args()
    if args.runs < 1 or args.runs > 5 or not math.isfinite(args.mbps) or args.mbps < 0 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error('Use 1–5 runs and a nonnegative bandwidth cap')
    root = Path(__file__).resolve().parents[2]
    scripts, assets = (args.scripts or root / 'web/site/browser').resolve(), args.assets.resolve()
    post_assets = args.post_assets.resolve() if args.post_assets else None
    if args.post_mode and (not post_assets or not (post_assets / 'v1/manifest.json').is_file()):
        parser.error('Sample preparation requires --post-assets')
    audio = args.input.resolve()
    if not audio.is_file() or not (assets / 'v1/manifest.json').is_file():
        parser.error('Missing fixture or staged assets')
    manifest = json.loads((assets / 'v1/manifest.json').read_text())
    expected_model_bytes = sum(model['bytes'] for model in manifest['models'].values())
    compressed = args.gzip_models.resolve() if args.gzip_models else None
    if compressed:
        gzip_paths = [compressed / f"{name}-{model['sha256']}.onnx.gz" for name, model in manifest['models'].items()]
        if not all(path.is_file() for path in gzip_paths):
            parser.error('Missing precompressed model copies')
        expected_model_bytes = sum(path.stat().st_size for path in gzip_paths)
    args.output.mkdir(parents=True, exist_ok=True)
    lock, pacing_lock = threading.Lock(), threading.Lock()
    totals = {'total': 0, 'models': 0, 'samples': 0}
    next_send = 0.0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, content_type='application/json', model=False, encoding=None, sample=False):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            if encoding:
                self.send_header('Content-Encoding', encoding)
            self.end_headers()
            self.deliver(data, model, sample)

        def deliver(self, data, model=False, sample=False):
            nonlocal next_send
            for offset in range(0, len(data), 256 * 1024):
                block = data[offset:offset + 256 * 1024]
                if args.mbps:
                    with pacing_lock:
                        now = time.monotonic()
                        due = max(now, next_send)
                        next_send = due + len(block) / (args.mbps * 1_000_000 / 8)
                    if due > now:
                        time.sleep(due - now)
                self.wfile.write(block)
                with lock:
                    totals['total'] += len(block)
                    if model:
                        totals['models'] += len(block)
                    if sample:
                        totals['samples'] += len(block)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                return self.send(PAGE.encode(), 'text/html; charset=utf-8')
            if path == '/config':
                return self.send(json.dumps({'runs': args.runs, 'mbps': args.mbps, 'expectedModelBytes': expected_model_bytes, 'timeoutSeconds': args.timeout_seconds, 'reuseCache': args.reuse_cache, 'modelEncoding': 'gzip' if compressed else 'identity', 'saveWavs': args.save_wavs, 'postMode': args.post_mode, 'postReuse': args.post_reuse}).encode())
            if path == '/snapshot':
                with lock:
                    data = json.dumps(totals).encode()
                return self.send(data)
            if path == '/input.wav':
                return self.send(audio.read_bytes(), 'audio/wav')
            if path.startswith('/browser-assets/'):
                base, relative = assets, path.removeprefix('/browser-assets/')
            elif path.startswith('/browser/'):
                base, relative = scripts, path.removeprefix('/browser/')
            elif post_assets and path.startswith('/post-assets/'):
                base, relative = post_assets, path.removeprefix('/post-assets/')
            else:
                return self.send_error(404)
            target = (base / relative).resolve()
            if base not in target.parents or not target.is_file():
                return self.send_error(404)
            types = {'.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'}
            if compressed and target.suffix == '.onnx':
                return self.send((compressed / (target.name + '.gz')).read_bytes(), 'application/octet-stream', model=True, encoding='gzip')
            data = target.read_bytes()
            if args.profile and target == scripts / 'worker.js':
                data = (root / 'tools/browser/profile-worker.js').read_bytes() + b'\n' + data
            return self.send(data, types.get(target.suffix, 'application/octet-stream'), target.suffix == '.onnx', sample=bool(post_assets and post_assets in target.parents))

        def do_POST(self):
            length = int(self.headers.get('Content-Length', '0'))
            artifact = re.fullmatch(r'/artifact/([1-5])/(stem|shot|zip)/([^/]+)', unquote(self.path))
            if args.save_wavs and artifact and int(artifact[1]) <= args.runs and 0 < length <= 64 * 1024 * 1024:
                name = artifact[3]
                if name in ['.', '..'] or '\\' in name or '\x00' in name:
                    return self.send_error(400)
                target = args.output / ('run-' + artifact[1]) / artifact[2]
                target.mkdir(parents=True, exist_ok=True)
                (target / name).write_bytes(self.rfile.read(length))
                return self.send(b'{}')
            matched = re.fullmatch(r'/wave/([1-5])/(kick|snare|percussion|hi-hat)\.wav', self.path)
            if args.save_wavs and matched and 0 < length <= 44 + 30 * 44100 * 6 and int(matched[1]) <= args.runs:
                data = self.rfile.read(length)
                try:
                    with wave.open(io.BytesIO(data), 'rb') as audio_file:
                        frames = audio_file.getnframes()
                        if audio_file.getparams()[:3] != (2, 3, 44100) or not 0 < frames <= 30 * 44100:
                            return self.send_error(400)
                        if len(audio_file.readframes(frames)) != frames * 6:
                            return self.send_error(400)
                except (wave.Error, EOFError):
                    return self.send_error(400)
                target = args.output / ('run-' + matched[1]); target.mkdir(exist_ok=True)
                (target / (matched[2] + '.wav')).write_bytes(data)
                return self.send(b'{}')
            if self.path != '/report' or not 0 < length <= 1024 * 1024:
                return self.send_error(405)
            report = json.loads(self.rfile.read(length))
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            return self.send(b'{}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    url = f'http://127.0.0.1:{server.server_port}/'
    (args.output / 'url.txt').write_text(url)
    print(url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
