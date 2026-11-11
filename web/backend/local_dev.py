#!/usr/bin/env python3
"""Run the website + API on your own machine, no Modal needed.

  .venv/bin/python web/backend/local_dev.py          # http://127.0.0.1:8780

Uses the same service.py as production, with an in-memory job store and a background
thread instead of a cloud GPU. The site in web/site/ is served at /.
"""
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]          # service.py, engine.py

import uvicorn                                               # noqa: E402
from fastapi.staticfiles import StaticFiles                  # noqa: E402

import engine                                                # noqa: E402
import service                                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8780)
    ap.add_argument('--data', default=str(HERE / '.local_data'))
    ap.add_argument('--model', default='htdemucs', choices=list(engine.MIX_MODELS))
    a = ap.parse_args()

    root = Path(a.data)
    (root / 'jobs').mkdir(parents=True, exist_ok=True)
    store = service.MemoryStore()
    models = engine.Models(root / 'work')
    pool = ThreadPoolExecutor(max_workers=1)                  # one "GPU"
    S = service.Settings()

    def submit(jid):
        pool.submit(service.run_job, jid, root, store, models, a.model, S.max_minutes * 60)

    app = service.create_app(root, store, submit, S)
    import subprocess
    subprocess.run([sys.executable, str(HERE.parent / 'build_site.py')], check=True)
    dist = HERE.parent / 'dist'

    @app.middleware('http')
    async def pretty_urls(request, call_next):          # /privacy -> privacy.html, like Cloudflare Pages
        path = request.url.path
        if not path.startswith('/api/') and '.' not in path.rsplit('/', 1)[-1] and path != '/' \
                and (dist / (path.strip('/') + '.html')).is_file():
            request.scope['path'] = path.rstrip('/') + '.html'
        return await call_next(request)

    app.mount('/', StaticFiles(directory=dist, html=True), name='site')

    def sweeper():
        import time
        while True:
            service.cleanup(root, store, S.keep_hours)
            time.sleep(600)
    threading.Thread(target=sweeper, daemon=True).start()

    engine.ensure_ffmpeg()
    print(f'\n  Site + API at http://127.0.0.1:{a.port}\n')
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
