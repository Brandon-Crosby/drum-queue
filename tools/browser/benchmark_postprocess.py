#!/usr/bin/env python3
"""Check existing sample-extraction artifacts in a real browser, without models.

Provide an official staged Pyodide 0.29.0 runtime and authorized WAV stems.
This diagnostic page does not alter the product UI or contact cloud processing.
"""
import argparse
import hashlib
import io
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
import zipfile

PAGE = b'''<!doctype html><meta charset="utf-8"><title>Local artifact compatibility check</title>
<h1>Local artifact compatibility check</h1><p>Diagnostic only. No cloud processing.</p>
<pre role="status">Starting browser worker...</pre><script>
const worker=new Worker('/worker.js');
worker.onmessage=async ({data})=>{
 document.querySelector('[role=status]').textContent=JSON.stringify(data,null,2);
 if(data.status==='complete'||data.status==='failed'){
  try{await fetch('/report',{method:'POST',body:JSON.stringify(data)});}
  finally{worker.terminate();}
 }
};
worker.onerror=async event=>{
 const data={status:'failed',error:event.message};
 document.querySelector('[role=status]').textContent=JSON.stringify(data);
 await fetch('/report',{method:'POST',body:JSON.stringify(data)});worker.terminate();
};
worker.postMessage({type:'run'});
</script>'''

WORKER = r'''
importScripts('/runtime/pyodide.js');
onmessage=async()=>{
 const started=performance.now();
 try{
  const config=await (await fetch('/config')).json();
  postMessage({status:'loading',stage:'Loading local Python runtime'});
  const py=await loadPyodide({indexURL:'/runtime/'});
  if(py.version!=='0.29.0')throw Error('Expected Pyodide 0.29.0');
  await py.loadPackage(['numpy','scipy','soundfile']);
  const get=async path=>{
   const response=await fetch(path);if(!response.ok)throw Error('Missing local artifact: '+path);
   return new Uint8Array(await response.arrayBuffer());
  };
  if(config.bytecode){
   const bytes=await get('/bytecode.zip');
   const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
   if(hash!==config.bytecode.sha256)throw Error('Bytecode archive verification failed');
   py.FS.writeFile('/bytecode.zip',bytes);
   await py.runPythonAsync("import zipfile\nwith zipfile.ZipFile('/bytecode.zip') as archive:\n    archive.extractall('/')");
  }
  const ready=performance.now();
  py.FS.mkdirTree('/python');py.FS.mkdirTree('/input');
  py.FS.writeFile('/python.zip',await get('/python.zip'));
  py.FS.writeFile('/reference.npz',await get('/reference.npz'));
  for(const stem of config.stems)py.FS.writeFile('/input/'+stem+'.wav',await get('/stem/'+stem+'.wav'));
  py.globals.set('job_name',config.name);py.globals.set('require_full_mix',config.fullMix);
  py.globals.set('profile_enabled',config.profile);
  postMessage({status:'processing',stage:'Running existing sample-extraction code'});
  const processing=performance.now();
  const encoded=await py.runPythonAsync(`
import sys, zipfile, json, hashlib
with zipfile.ZipFile('/python.zip') as source:
    source.extractall('/python')
sys.path.insert(0, '/python')
from postprocess_job import finish
profile_text = None
if profile_enabled:
    import cProfile, pstats, io
    profiler = cProfile.Profile()
    profiler.enable()
result = finish(job_name, '/input', '/output', '/reference.npz', require_full_mix=require_full_mix)
if profile_enabled:
    profiler.disable()
    profile_stream = io.StringIO()
    pstats.Stats(profiler, stream=profile_stream).strip_dirs().sort_stats('cumulative').print_stats(30)
    profile_text = profile_stream.getvalue()
checks = []
for artifact in result.pop('artifacts'):
    if artifact['kind'] == 'zip':
        with zipfile.ZipFile(artifact['path']) as archive:
            members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
        checks.append(dict(kind='zip', name=artifact['name'], members=members))
    else:
        with open(artifact['path'], 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        checks.append(dict(kind=artifact['kind'], name=artifact['name'], sha256=digest))
json.dumps(dict(result=result, artifacts=checks, profile=profile_text))
`);
  postMessage({status:'complete',runtime:'Pyodide 0.29.0',
   runtimeMs:ready-started,processingMs:performance.now()-processing,
   elapsedMs:performance.now()-started,...JSON.parse(encoded)});
 }catch(error){postMessage({status:'failed',error:String(error)});}
};
'''


def verified_bytecode(directory, runtime):
    manifest = json.loads((directory / 'manifest.json').read_text())
    data = (directory / 'bytecode.zip').read_bytes()
    if manifest.get('runtime') != 'Pyodide 0.29.0' or manifest.get('invalidation') != 'checked-hash':
        raise ValueError('Unsupported bytecode runtime')
    if hashlib.sha256((runtime / 'pyodide-lock.json').read_bytes()).hexdigest() != manifest.get('runtimeLockSha256'):
        raise ValueError('Bytecode was prepared for a different runtime lock')
    if len(data) != manifest.get('bytes') or hashlib.sha256(data).hexdigest() != manifest.get('sha256'):
        raise ValueError('Bytecode archive verification failed')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if not 0 < len(members) <= 10000 or sum(x.file_size for x in members) > 100 * 1024 * 1024:
            raise ValueError('Unexpected bytecode archive size')
        for entry in members:
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.filename
                    or not entry.filename.startswith('lib/python3.13/')
                    or '__pycache__' not in path.parts or path.suffix != '.pyc'
                    or entry.file_size > 4 * 1024 * 1024):
                raise ValueError('Unsafe bytecode archive path')
            with archive.open(entry) as source:
                if source.read(8)[4:8] != bytes([3, 0, 0, 0]):
                    raise ValueError('Expected checked-hash bytecode')
    return data, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--stems', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', default='test.wav')
    parser.add_argument('--profile', action='store_true', help='Collect Python hot spots; adds diagnostic timing overhead')
    parser.add_argument('--bytecode', type=Path, help='Verified, precompiled runtime cache from prepare_bytecode.cjs')
    parser.add_argument('--kit-only-prototype', action='store_true', help='Explicitly test four stems; not full-site feature parity')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runtime = args.runtime.resolve()
    lock = json.loads((runtime / 'pyodide-lock.json').read_text())
    if lock['info']['arch'] != 'wasm32':
        parser.error('Expected a wasm32 runtime; the worker checks its loaded version')
    bytecode, bytecode_manifest = verified_bytecode(args.bytecode, runtime) if args.bytecode else (None, None)
    # Verify official wheel hashes before exposing them to the test browser.
    for package in lock['packages'].values():
        path = runtime / package['file_name']
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != package['sha256']:
            parser.error('Runtime package hash mismatch: ' + path.name)
    stems = ['kick', 'snare', 'perc', 'hh']
    if not args.kit_only_prototype:
        stems += ['drums', 'bass', 'vocals', 'other']
    for stem in stems:
        path = args.stems / (stem + '.wav')
        if not path.is_file() or not 0 < path.stat().st_size <= 30 * 44100 * 8 + 4096:
            parser.error('Missing or oversized stem: ' + stem)
    if not args.reference.is_file():
        parser.error('The naming reference is required')
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / 'engine.py', 'engine.py')
        archive.write(root / 'tools/browser/postprocess_job.py', 'postprocess_job.py')
        for path in sorted((root / 'kitmatch').glob('*.py')):
            archive.write(path, 'kitmatch/' + path.name)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, kind):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            fixed = {'/': (PAGE, 'text/html'), '/worker.js': (WORKER.encode(), 'text/javascript'),
                     '/python.zip': (bundle.getvalue(), 'application/zip'),
                     '/config': (json.dumps(dict(name=args.name, stems=stems, fullMix=not args.kit_only_prototype, profile=args.profile, bytecode=bytecode_manifest)).encode(), 'application/json')}
            if path in fixed:
                return self.send(*fixed[path])
            if path == '/reference.npz':
                return self.send(args.reference.read_bytes(), 'application/octet-stream')
            if path == '/bytecode.zip' and bytecode is not None:
                return self.send(bytecode, 'application/zip')
            if path in ['/stem/' + name + '.wav' for name in stems]:
                return self.send((args.stems / Path(path).name).read_bytes(), 'audio/wav')
            if path.startswith('/runtime/'):
                target = (runtime / path.removeprefix('/runtime/')).resolve()
                if runtime in target.parents and target.is_file():
                    return self.send(target.read_bytes(), mimetypes.guess_type(target)[0] or 'application/octet-stream')
            self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get('Content-Length', '0'))
            if self.path != '/report' or not 0 < length <= 1024 * 1024:
                return self.send_error(400)
            result = json.loads(self.rfile.read(length))
            (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
            self.send(b'{}', 'application/json')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    print(f'http://127.0.0.1:{server.server_port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
