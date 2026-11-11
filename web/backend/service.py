"""service.py - the website's API. Pure FastAPI; storage, job store and the GPU hand-off are
injected, so the same code runs locally (local_dev.py) and on Modal (modal_app.py).

  POST /api/jobs                       upload a song -> {id}
  GET  /api/jobs/{id}                  status, stage, stems
  GET  /api/jobs/{id}/files/{name}     a stem (?dl=1 to download) or preview/<name>
  GET  /api/jobs/{id}/zip              all stems (?kit=1 for kit pieces only)
  GET  /api/limits                     splits left today for this visitor
"""
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aiff', '.aif', '.ogg', '.opus'}
JOB_ID = re.compile(r'^[a-f0-9]{16}$')
SAFE_FILE = re.compile(r'^[a-z_]+\.(wav|mp3)$')


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    def __init__(self):
        self.max_mb = env_int('MAX_UPLOAD_MB', 30)
        self.max_minutes = env_int('MAX_MINUTES', 8)
        self.per_ip_per_day = env_int('LIMIT_PER_IP', 3)
        self.daily_cap = env_int('DAILY_CAP', 300)          # all visitors combined; protects your bill
        self.keep_hours = env_int('KEEP_HOURS', 24)
        self.turnstile_secret = os.environ.get('TURNSTILE_SECRET', '')
        self.salt = os.environ.get('IP_SALT', 'change-me')
        self.origins = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()]
        # Set in production: only the Cloudflare proxy (site/functions/api) knows it, so nobody can
        # call Modal directly or fake their IP to dodge the daily limit.
        self.proxy_secret = os.environ.get('PROXY_SECRET', '')


# ------------------------------------------------------------------ stores
class MemoryStore:
    """Job records + counters for local runs. modal_app.py swaps in a modal.Dict-backed one."""

    def __init__(self):
        self.d = {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def put(self, k, v):
        self.d[k] = v

    def pop(self, k):
        self.d.pop(k, None)

    def keys(self):
        return list(self.d.keys())


def update_job(store, jid, **kw):
    j = store.get(f'job:{jid}') or {}
    j.update(kw)
    store.put(f'job:{jid}', j)
    return j


def today():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')


def client_ip(request, trusted):
    """Behind the Cloudflare proxy, x-real-ip is CF-Connecting-IP (set by Cloudflare, not the visitor)."""
    if trusted and request.headers.get('x-real-ip'):
        return request.headers['x-real-ip']
    return request.client.host if request.client else '?'


def visitor_key(request, salt, trusted=False):
    ip = client_ip(request, trusted)
    # Only a salted hash of the IP is kept, and only for counting today's splits.
    return hashlib.sha256(f'{salt}|{today()}|{ip}'.encode()).hexdigest()[:20]


def verify_turnstile(secret, token, ip=None):
    if not secret:
        return True
    if not token:
        return False
    data = {'secret': secret, 'response': token}
    if ip:
        data['remoteip'] = ip
    try:
        req = urllib.request.Request('https://challenges.cloudflare.com/turnstile/v0/siteverify',
                                     data=urllib.parse.urlencode(data).encode(), method='POST')
        with urllib.request.urlopen(req, timeout=8) as r:
            return bool(json.load(r).get('success'))
    except Exception:
        return False


def audio_seconds(path):
    try:
        import mutagen
        f = mutagen.File(str(path))
        return float(f.info.length) if f is not None and f.info else None   # untagged files are falsy
    except Exception:
        return None


# --------------------------------------------------------------------- app
def create_app(root, store, submit, settings=None, refresh=lambda: None):
    """root: folder holding jobs/<id>/...; store: job records + counters;
    submit(job_id): start the GPU work; refresh(): see files another container just wrote."""
    S = settings or Settings()
    root = Path(root)
    app = FastAPI(title='Drum splitter API', docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=S.origins, allow_methods=['GET', 'POST'],
                       allow_headers=['*'])
    trusted = bool(S.proxy_secret)

    @app.middleware('http')
    async def only_via_proxy(request, call_next):
        if S.proxy_secret and request.headers.get('x-proxy-secret') != S.proxy_secret:
            return JSONResponse({'error': 'Forbidden'}, status_code=403)
        return await call_next(request)

    def counts(vkey):
        d = today()
        return store.get(f'n:{d}:{vkey}', 0), store.get(f'n:{d}:all', 0)

    def public(j, jid):
        keep = {k: j.get(k) for k in ('status', 'stage', 'error', 'name', 'bpm', 'duration', 'note',
                                       'created', 'started', 'finished')}
        keep['id'] = jid
        keep['stems'] = [{k: s[k] for k in ('name', 'preview', 'stem', 'label', 'group', 'quiet')}
                         for s in j.get('stems') or []]
        keep['expires'] = (j.get('created') or time.time()) + S.keep_hours * 3600
        return keep

    @app.get('/api/health')
    def health():
        return {'ok': True}

    @app.get('/api/limits')
    def limits(request: Request):
        mine, total = counts(visitor_key(request, S.salt, trusted))
        return {'left': max(0, S.per_ip_per_day - mine), 'per_day': S.per_ip_per_day,
                'site_full': total >= S.daily_cap, 'max_minutes': S.max_minutes, 'max_mb': S.max_mb,
                'turnstile': bool(S.turnstile_secret)}

    @app.post('/api/jobs')
    async def create_job(request: Request, file: UploadFile = File(...), token: str = Form('')):
        name = re.sub(r'[\x00-\x1f/\\]', '_', Path(file.filename or 'song').name)[:120]
        if Path(name).suffix.lower() not in AUDIO_EXTS:
            raise HTTPException(400, 'That file type is not supported. Use mp3, wav, flac, m4a, aiff or ogg.')
        vkey = visitor_key(request, S.salt, trusted)
        mine, total = counts(vkey)
        if total >= S.daily_cap:
            raise HTTPException(429, "We've hit today's limit for free splits. Please come back tomorrow.")
        if mine >= S.per_ip_per_day:
            raise HTTPException(429, "You've used today's free splits. Come back tomorrow.")
        if not verify_turnstile(S.turnstile_secret, token, client_ip(request, trusted)):
            raise HTTPException(403, 'Bot check failed. Refresh the page and try again.')

        jid = uuid.uuid4().hex[:16]
        d = root / 'jobs' / jid / 'input'
        d.mkdir(parents=True)
        dst = d / name
        size, limit = 0, S.max_mb * 1024 * 1024
        with open(dst, 'wb') as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    out.close()
                    shutil.rmtree(d.parent, ignore_errors=True)
                    raise HTTPException(413, f'File is over {S.max_mb} MB.')
                out.write(chunk)
        secs = audio_seconds(dst)
        if secs is not None and secs > S.max_minutes * 60:
            shutil.rmtree(d.parent, ignore_errors=True)
            raise HTTPException(400, f'Song is {secs / 60:.1f} minutes; the limit is {S.max_minutes}.')

        store.put(f'n:{today()}:{vkey}', mine + 1)
        store.put(f'n:{today()}:all', total + 1)
        update_job(store, jid, status='queued', stage='Waiting for a GPU', name=name, created=time.time(),
                   stems=[], error='')
        submit(jid)
        return {'id': jid, 'left': max(0, S.per_ip_per_day - mine - 1)}

    def job_or_404(jid):
        if not JOB_ID.match(jid):
            raise HTTPException(404)
        j = store.get(f'job:{jid}')
        if not j:
            raise HTTPException(404, 'This split has expired or does not exist.')
        return j

    @app.get('/api/jobs/{jid}')
    def get_job(jid: str):
        return public(job_or_404(jid), jid)

    def stem_file(jid, j, name, preview=False):
        if not SAFE_FILE.match(name):
            raise HTTPException(404)
        field = 'preview' if preview else 'name'
        if name not in [s.get(field) for s in j.get('stems') or []]:
            raise HTTPException(404)
        p = root / 'jobs' / jid / 'out' / j['out_dir'] / ('preview' if preview else '') / name
        if not p.exists():
            refresh()
        if not p.exists():
            raise HTTPException(404, 'File is gone (splits are kept for 24 hours).')
        return p

    @app.get('/api/jobs/{jid}/files/{name}')
    def get_file(jid: str, name: str, dl: int = 0):
        j = job_or_404(jid)
        p = stem_file(jid, j, name)
        song = Path(j.get('name') or 'song').stem
        return FileResponse(p, media_type='audio/wav',
                            filename=f'{song} - {name}' if dl else None,
                            content_disposition_type='attachment' if dl else 'inline')

    @app.get('/api/jobs/{jid}/preview/{name}')
    def get_preview(jid: str, name: str):
        j = job_or_404(jid)
        p = stem_file(jid, j, name, preview=True)
        return FileResponse(p, media_type='audio/mpeg' if name.endswith('.mp3') else 'audio/wav',
                            headers={'Cache-Control': 'private, max-age=86400'})

    @app.get('/api/jobs/{jid}/zip')
    def get_zip(jid: str, kit: int = 0):
        j = job_or_404(jid)
        if j.get('status') != 'done':
            raise HTTPException(409, 'Not finished yet.')
        song = Path(j.get('name') or 'song').stem
        tmp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_STORED) as z:
            for s in j['stems']:
                if kit and s['group'] != 'kit':
                    continue
                z.write(stem_file(jid, j, s['name']), f"{song}/{s['name']}")
        f = open(tmp.name, 'rb')
        os.unlink(tmp.name)                         # stays readable until the response closes
        from fastapi.responses import StreamingResponse
        return StreamingResponse(f, media_type='application/zip', headers={
            'Content-Disposition': f'attachment; filename="{_ascii(song)}{" - kit" if kit else ""}.zip"'})

    @app.exception_handler(HTTPException)
    async def http_err(_, e):
        return JSONResponse({'error': e.detail or 'Not found'}, status_code=e.status_code)

    return app


def _ascii(s):
    return re.sub(r'[^\w\- ().&]', '_', s.encode('ascii', 'ignore').decode()) or 'song'


# ----------------------------------------------------------------- the GPU job
def run_job(jid, root, store, models, mix_model='htdemucs_ft', max_seconds=None):
    """Runs where the GPU is. Splits jobs/<id>/input/<song> into jobs/<id>/out/<song>/."""
    import engine
    root = Path(root)
    j = store.get(f'job:{jid}') or {}
    src = next((root / 'jobs' / jid / 'input').glob('*'), None)
    if not src:
        update_job(store, jid, status='error', stage='Failed', error='Upload is missing. Try again.')
        return
    update_job(store, jid, status='running', stage='Starting', started=time.time())
    try:
        res = engine.process(
            dict(name=j.get('name', src.name), src=str(src),
                 opts=dict(model=mix_model, overlap=4, keep_full=True)),
            root / 'jobs' / jid / 'out', models,
            lambda s: update_job(store, jid, stage=s), preview_fmt='mp3', max_seconds=max_seconds)
        update_job(store, jid, status='done', stage='Done', finished=time.time(), **res)
    except Exception as e:
        msg = str(e) if isinstance(e, ValueError) else 'Something went wrong while splitting this song.'
        if type(e).__name__ in ('NoBackendError', 'LibsndfileError', 'DecodeError'):
            msg = "Couldn't read this file as audio. Try an mp3 or wav."
        update_job(store, jid, status='error', stage='Failed', finished=time.time(), error=msg,
                   detail=f'{type(e).__name__}: {e}'[:500])
    finally:
        shutil.rmtree(root / 'jobs' / jid / 'input', ignore_errors=True)   # never keep the upload


def cleanup(root, store, keep_hours=24):
    """Delete splits older than keep_hours, plus old daily counters."""
    root = Path(root)
    cutoff = time.time() - keep_hours * 3600
    n = 0
    for d in (root / 'jobs').glob('*') if (root / 'jobs').exists() else []:
        j = store.get(f'job:{d.name}')
        made = (j or {}).get('created') or d.stat().st_mtime
        if made < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            store.pop(f'job:{d.name}')
            n += 1
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).strftime('%Y-%m-%d')
    for k in store.keys():
        if k.startswith('n:') and k.split(':')[1] < old:
            store.pop(k)
    return n
