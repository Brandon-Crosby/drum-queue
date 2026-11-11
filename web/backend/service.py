"""service.py - the website's API. Pure FastAPI; storage, job store and the GPU hand-off are
injected, so the same code runs locally (local_dev.py) and on Modal (modal_app.py).

  POST /api/jobs                       upload a song -> {id}
  GET  /api/jobs/{id}                  status, stage, stems
  POST /api/jobs/{id}/cancel           stop a split; it doesn't count toward the daily limit
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


def bump(store, key, **inc):
    """Add to a daily stats record: numbers are summed, dicts are summed per key.
    Only totals are kept (no IPs, nothing tied to a person)."""
    d = store.get(key) or {}
    for k, v in inc.items():
        if isinstance(v, dict):
            sub = dict(d.get(k) or {})
            for kk, vv in v.items():
                sub[kk] = sub.get(kk, 0) + vv
            d[k] = sub
        else:
            d[k] = round(d.get(k, 0) + v, 3)
    store.put(key, d)


# ETA defaults until there are real splits to learn from (seconds of GPU per second of song; seconds
# from upload to the GPU starting, which includes a cold start loading both models).
ETA_RATE, ETA_WAIT = 0.5, 60


def eta_model(store, days=7):
    """How fast recent splits ran: (GPU seconds per second of audio, average wait for a GPU)."""
    gpu = audio = wait = done = 0
    d0 = dt.datetime.now(dt.timezone.utc).date()
    for i in range(days):
        g = store.get(f'stats:{(d0 - dt.timedelta(days=i)).isoformat()}:gpu') or {}
        gpu += g.get('gpu_s', 0); audio += g.get('audio_min', 0) * 60
        wait += g.get('wait_s', 0); done += g.get('done', 0)
    rate = min(3.0, max(0.1, gpu / audio)) if audio > 120 else ETA_RATE
    avg_wait = min(300, max(10, wait / done)) if done >= 2 else ETA_WAIT
    return rate, avg_wait


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
    """Asks Cloudflare whether the visitor's Turnstile token is valid. Returns (ok, reason).

    remoteip is left out on purpose: with iCloud Private Relay / IPv6 the address that solved the
    challenge often differs from the one that uploads, and a mismatch fails an otherwise good token.
    An explicit User-Agent matters too: Cloudflare can refuse Python's default 'Python-urllib' agent.
    """
    if not secret:
        return True, 'off'
    if not token:
        return False, 'missing-token'
    data = urllib.parse.urlencode({'secret': secret.strip(), 'response': token}).encode()
    req = urllib.request.Request('https://challenges.cloudflare.com/turnstile/v0/siteverify', data=data,
                                 method='POST', headers={'User-Agent': 'DrumSplitter/1.0 (+https://drumsplitter.com)',
                                                         'Content-Type': 'application/x-www-form-urlencoded'})
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = json.load(r)
            if body.get('success'):
                return True, 'ok'
            reason = ','.join(body.get('error-codes') or []) or 'unknown'
            print(f'turnstile rejected: {reason} hostname={body.get("hostname")}', flush=True)
            return False, reason
        except urllib.error.HTTPError as e:
            reason = f'http-{e.code}'
            print(f'turnstile verify HTTP {e.code}: {e.read()[:300]!r}', flush=True)
        except Exception as e:
            reason = f'{type(e).__name__}'
            print(f'turnstile verify error (try {attempt}): {e!r}', flush=True)
    return False, reason


def audio_seconds(path):
    try:
        import mutagen
        f = mutagen.File(str(path))
        return float(f.info.length) if f is not None and f.info else None   # untagged files are falsy
    except Exception:
        return None


# --------------------------------------------------------------------- app
QUEUE_TIMEOUT = 8 * 60      # waiting this long for a GPU = give up and hand the split back
RUN_TIMEOUT = 20 * 60       # a split running this long means the GPU worker died


def create_app(root, store, submit, settings=None, refresh=lambda: None, cancel=lambda call_id: None):
    """root: folder holding jobs/<id>/...; store: job records + counters;
    submit(job_id): start the GPU work, may return a call id; cancel(call_id): stop it;
    refresh(): see files another container just wrote."""
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

    def stop_job(jid, j, msg, why):
        """Cancel a queued/running split and give the visitor their split back (once)."""
        if j.get('status') not in ('queued', 'running'):
            return j
        try:
            if j.get('call'):
                cancel(j['call'])
        except Exception as e:
            print(f'cancel {jid} failed: {e!r}', flush=True)
        if not j.get('refunded') and j.get('day') and j.get('vkey'):
            for k in (f"n:{j['day']}:{j['vkey']}", f"n:{j['day']}:all"):
                store.put(k, max(0, store.get(k, 0) - 1))
        bump(store, f'stats:{today()}:api', **{why: 1})
        return update_job(store, jid, status='error', stage='Stopped', error=msg, cancelled=True,
                          refunded=True, finished=time.time())

    def eta_fields(secs):
        rate, wait = eta_model(store)
        return {'secs': round(secs or 180), 'eta_wait': round(wait), 'eta_run': round(15 + (secs or 180) * rate)}

    def public(j, jid):
        keep = {k: j.get(k) for k in ('status', 'stage', 'error', 'name', 'bpm', 'duration', 'note',
                                       'created', 'started', 'finished', 'eta_wait', 'eta_run')}
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
    def create_job(request: Request, file: UploadFile = File(...), token: str = Form('')):
        # Plain def (not async): FastAPI runs it in a worker thread, so the blocking job-store and
        # Modal calls below never stall other visitors' requests.
        name = re.sub(r'[\x00-\x1f/\\]', '_', Path(file.filename or 'song').name)[:120]
        if Path(name).suffix.lower() not in AUDIO_EXTS:
            raise HTTPException(400, 'That file type is not supported. Use mp3, wav, flac, m4a, aiff or ogg.')
        vkey = visitor_key(request, S.salt, trusted)
        mine, total = counts(vkey)
        if total >= S.daily_cap:
            bump(store, f'stats:{today()}:api', site_full=1)
            raise HTTPException(429, "We've hit today's limit for free splits. Please come back tomorrow.")
        if mine >= S.per_ip_per_day:
            bump(store, f'stats:{today()}:api', limited=1)
            raise HTTPException(429, "You've used today's free splits. Come back tomorrow.")
        ok, why = verify_turnstile(S.turnstile_secret, token)
        if not ok:
            bump(store, f'stats:{today()}:api', bot_blocked=1)
            # The code (e.g. invalid-input-secret, timeout-or-duplicate) shows up small in the message,
            # so a screenshot is enough to tell what went wrong.
            raise HTTPException(403, f'Bot check failed. Refresh the page and try again. ({why})')

        jid = uuid.uuid4().hex[:16]
        d = root / 'jobs' / jid / 'input'
        d.mkdir(parents=True)
        dst = d / name
        size, limit = 0, S.max_mb * 1024 * 1024
        with open(dst, 'wb') as out:
            while chunk := file.file.read(1024 * 1024):
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
        # Location comes from Cloudflare's lookup (passed by the proxy), kept only as daily totals.
        country = (request.headers.get('x-country') or '').upper()[:2] if trusted else ''
        region = (request.headers.get('x-region') or '').upper()[:3] if trusted else ''
        loc = f'{country}-{region}' if country and region else (country or '??')
        bump(store, f'stats:{today()}:api', started=1, uploaders=1 if mine == 0 else 0,
             countries={country or '??': 1}, regions={loc: 1})
        update_job(store, jid, status='queued', stage='Waiting for a GPU', name=name, created=time.time(),
                   stems=[], error='', day=today(), vkey=vkey, **eta_fields(secs))
        call = submit(jid)
        if call:
            update_job(store, jid, call=call)
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
        j = job_or_404(jid)
        now = time.time()
        # Watchdog: nobody should sit on "Waiting for a free GPU" (or a dead worker) forever.
        if j.get('status') == 'queued' and now - (j.get('created') or now) > QUEUE_TIMEOUT:
            j = stop_job(jid, j, "Our GPUs are all busy right now, so we stopped this one. It didn't count "
                                 "toward your free splits. Please try again in a few minutes.", 'stalled')
        elif j.get('status') == 'running' and now - (j.get('started') or now) > RUN_TIMEOUT:
            j = stop_job(jid, j, "This split stopped responding, so we cancelled it. It didn't count toward "
                                 "your free splits. Please try again.", 'stalled')
        return public(j, jid)

    @app.post('/api/jobs/{jid}/cancel')
    def cancel_job(jid: str):
        j = stop_job(jid, job_or_404(jid), "Cancelled. It didn't count toward your free splits.", 'cancelled')
        return public(j, jid)

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
class Cancelled(Exception):
    pass


def _progress(store, jid, stage):
    if (store.get(f'job:{jid}') or {}).get('cancelled'):
        raise Cancelled()                        # stop between steps instead of finishing for nobody
    update_job(store, jid, stage=stage)


def run_job(jid, root, store, models, mix_model='htdemucs_ft', max_seconds=None):
    """Runs where the GPU is. Splits jobs/<id>/input/<song> into jobs/<id>/out/<song>/."""
    root = Path(root)
    j = store.get(f'job:{jid}') or {}
    if j.get('cancelled'):                       # visitor cancelled (or watchdog gave up) before we got a GPU
        return
    import engine
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
            lambda st: _progress(store, jid, st), preview_fmt='mp3', max_seconds=max_seconds)
        done = time.time()
        if (store.get(f'job:{jid}') or {}).get('cancelled'):
            return
        update_job(store, jid, status='done', stage='Done', finished=done, **res)
        started = (store.get(f'job:{jid}') or {}).get('started') or done
        bump(store, f'stats:{today()}:gpu', done=1, audio_min=(res.get('duration') or 0) / 60,
             gpu_s=done - started, wait_s=started - (j.get('created') or started))
    except Cancelled:
        pass
    except Exception as e:
        if (store.get(f'job:{jid}') or {}).get('cancelled'):
            return
        msg = str(e) if isinstance(e, ValueError) else 'Something went wrong while splitting this song.'
        if type(e).__name__ in ('NoBackendError', 'LibsndfileError', 'DecodeError'):
            msg = "Couldn't read this file as audio. Try an mp3 or wav."
        update_job(store, jid, status='error', stage='Failed', finished=time.time(), error=msg,
                   detail=f'{type(e).__name__}: {e}'[:500])
        bump(store, f'stats:{today()}:gpu', failed=1, gpu_s=time.time() - ((store.get(f'job:{jid}') or {}).get('started') or time.time()))
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


GPU_DOLLARS_PER_S = 0.00022            # Modal L4; A10/T4 fallbacks cost about the same per song


def stats_report(store, days=14):
    """Plain-text daily summary for `modal run web/backend/modal_app.py::stats`."""
    day0 = dt.datetime.now(dt.timezone.utc).date()
    rows, tot, countries, regions = [], {}, {}, {}
    for i in range(days):
        d = (day0 - dt.timedelta(days=i)).isoformat()
        a = store.get(f'stats:{d}:api') or {}
        g = store.get(f'stats:{d}:gpu') or {}
        rec = {
            'splits': a.get('started', 0), 'done': g.get('done', 0), 'failed': g.get('failed', 0),
            'people': a.get('uploaders', 0), 'audio_min': g.get('audio_min', 0), 'gpu_min': g.get('gpu_s', 0) / 60,
            'wait_s': g.get('wait_s', 0), 'limited': a.get('limited', 0), 'bots': a.get('bot_blocked', 0),
            'full': a.get('site_full', 0), 'stalled': a.get('stalled', 0), 'cancelled': a.get('cancelled', 0),
        }
        for k, v in rec.items():
            tot[k] = tot.get(k, 0) + v
        for k, v in (a.get('countries') or {}).items():
            countries[k] = countries.get(k, 0) + v
        for k, v in (a.get('regions') or {}).items():
            regions[k] = regions.get(k, 0) + v
        rows.append((d, rec))

    def line(label, r):
        avg_wait = r['wait_s'] / r['done'] if r['done'] else 0
        cost = r['gpu_min'] * 60 * GPU_DOLLARS_PER_S
        return (f"{label:12}{r['splits']:>7}{r['done']:>6}{r['failed']:>7}{r['people']:>8}"
                f"{r['audio_min']:>9.1f}{r['gpu_min']:>8.1f}{cost:>8.2f}{avg_wait:>8.0f}"
                f"{r['limited']:>9}{r['bots']:>6}{r['full']:>6}{r['stalled']:>9}{r['cancelled']:>7}")

    out = [f"DrumSplitter, last {days} days (UTC dates)", '',
           f"{'day':12}{'splits':>7}{'done':>6}{'failed':>7}{'people':>8}{'audio m':>9}{'GPU m':>8}"
           f"{'cost $':>8}{'wait s':>8}{'limited':>9}{'bots':>6}{'full':>6}{'stalled':>9}{'cancel':>7}"]
    out += [line(d, r) for d, r in rows]
    out += ['-' * 110, line('total', tot), '',
            'splits = uploads accepted   people = different visitors who split something that day',
            'wait s = average seconds waiting for a GPU   limited = turned away at the daily limit',
            'bots = failed the bot check   full = turned away because the site-wide daily cap was hit',
            'stalled = gave up waiting for a GPU (split refunded)   cancel = visitor pressed Cancel', '']
    top = lambda d: ', '.join(f'{k} {v}' for k, v in sorted(d.items(), key=lambda x: -x[1])[:12]) or 'none yet'
    out += [f'Countries: {top(countries)}', f'States/regions: {top(regions)}']
    return '\n'.join(out)
