"""modal_app.py - runs the website's backend on Modal (GPU billed per second, $0 when idle).

  modal deploy web/backend/modal_app.py

Pieces:
  api        CPU web server (FastAPI, from service.py). Takes uploads, reports status, serves files.
  Splitter   L4 GPU worker. Runs engine.py: Demucs -> MDX23C DrumSep. Models are baked into the image.
  cleanup    Every hour: deletes splits older than KEEP_HOURS.

State: a modal.Volume holds files, a modal.Dict holds job records and daily counters.
Secrets (modal secret create drumsplit-secrets ...): PROXY_SECRET, IP_SALT, TURNSTILE_SECRET.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# ---- knobs ---------------------------------------------------------------
MIX_MODEL = 'htdemucs_ft'      # 'htdemucs' is ~4x cheaper per song, slightly bleedier drums
# Tried in order; Modal uses whichever has capacity (a lone 'L4' can sit in a queue for minutes).
# L4 ~$0.00022/s, A10 ~$0.00031/s (faster), T4 ~$0.00016/s (slower) - similar cost per song.
# L40S / A100 cost ~3-5x per second but finish faster; they're only used when the cheap ones are
# all taken, so a busy hour costs a few cents more instead of leaving visitors stuck in the queue.
GPU = ['L4', 'A10', 'T4', 'L40S', 'A100-40GB']
MAX_GPUS = 2                   # most songs splitting at once; extra jobs wait in line
MAX_MINUTES = 8
KEEP_HOURS = 24
PUBLIC_ENV = dict(MAX_MINUTES=str(MAX_MINUTES), KEEP_HOURS=str(KEEP_HOURS),
                  LIMIT_PER_IP='3', DAILY_CAP='300', MAX_UPLOAD_MB='30')
# --------------------------------------------------------------------------

app = modal.App('drumsplit')
vol = modal.Volume.from_name('drumsplit-data', create_if_missing=True)
kv = modal.Dict.from_name('drumsplit-state', create_if_missing=True)
secrets = [modal.Secret.from_name('drumsplit-secrets')]


def _download_models():
    """Runs once at image build, so containers start with both models on disk."""
    import engine
    from audio_separator.separator import Separator
    engine.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s = Separator(model_file_dir=str(engine.MODEL_DIR), output_dir='/tmp')
    for m in (engine.MIX_MODELS[MIX_MODEL], engine.DRUMSEP):
        s.download_model_files(m)


gpu_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg', 'libsndfile1')
    .uv_pip_install('audio-separator[gpu]==0.44.2', 'librosa', 'soundfile', 'audioread', 'mutagen',
                    'fastapi', 'python-multipart')
    .env(PUBLIC_ENV)
    .add_local_file(REPO / 'engine.py', '/root/engine.py', copy=True)
    .add_local_file(REPO / 'oneshots.py', '/root/oneshots.py', copy=True)
    .add_local_file(HERE / 'service.py', '/root/service.py', copy=True)
    .run_function(_download_models)
)

web_image = (
    modal.Image.debian_slim(python_version='3.11')
    .uv_pip_install('fastapi[standard]', 'python-multipart', 'mutagen')
    .env(PUBLIC_ENV)
    .add_local_file(HERE / 'service.py', '/root/service.py')
)


class DictStore:
    """service.py's store interface on top of modal.Dict."""

    def __init__(self, d):
        self.d = d

    def get(self, k, default=None):
        return self.d.get(k, default)

    def put(self, k, v):
        self.d[k] = v

    def pop(self, k):
        try:
            self.d.pop(k)
        except KeyError:
            pass

    def keys(self):
        return list(self.d.keys())


@app.cls(image=gpu_image, gpu=GPU, volumes={'/data': vol}, secrets=secrets, timeout=900,
         max_containers=MAX_GPUS, scaledown_window=90)
class Splitter:
    @modal.enter()
    def load(self):
        import engine
        self.models = engine.Models('/tmp/dq/work')
        self.models.get_mix(MIX_MODEL)          # warm both models before the first song
        self.models.get_kit(4)

    @modal.method()
    def split(self, jid: str):
        import service
        vol.reload()
        try:
            service.run_job(jid, '/data', DictStore(kv), self.models, MIX_MODEL, MAX_MINUTES * 60)
        finally:
            vol.commit()


@app.function(image=web_image, volumes={'/data': vol}, secrets=secrets, scaledown_window=300)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def api():
    import service

    def submit(jid):
        vol.commit()                            # make the upload visible to the GPU container
        return Splitter().split.spawn(jid).object_id

    def cancel(call_id):
        modal.FunctionCall.from_id(call_id).cancel()

    def refresh():
        try:
            vol.reload()
        except Exception:                       # reload refuses while files are open; retry next time
            pass

    return service.create_app('/data', DictStore(kv), submit, refresh=refresh, cancel=cancel)


@app.function(image=web_image, volumes={'/data': vol}, secrets=secrets, schedule=modal.Period(hours=1))
def cleanup():
    import service
    vol.reload()
    n = service.cleanup('/data', DictStore(kv), KEEP_HOURS)
    vol.commit()
    print(f'removed {n} old splits')


@app.function(image=web_image, secrets=secrets)
def stats(days: int = 14):
    """Daily split counts:  modal run web/backend/modal_app.py::stats   (add --days 30 for more)"""
    import service
    print(service.stats_report(DictStore(kv), days))
