"""modal_app.py - runs the website's backend on Modal (GPU billed per second, $0 when idle).

  modal deploy web/backend/modal_app.py

Pieces:
  api        CPU web server (FastAPI, from service.py). Takes uploads, reports status, serves files.
  Splitter   L4 GPU worker. Runs engine.py: Demucs -> Helyx (our fine-tuned MDX23C). Models are baked
             into the image: Helyx is copied from the training volume at deploy time, so later training
             runs never change the live model - redeploy to pick up a new one (HELYX_RUN).
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
HELYX_RUN = 'runs/helyx-4'     # kit model: this run's last_mdx23c.ckpt on the drumsplit-train volume;
                               # '' = back to the public MDX23C DrumSep (kick/snare/toms/hh/ride/crash)
MSST_COMMIT = '1d4b3c431b37de696b3795ed2018493aacba946f'   # same code that trained and graded Helyx
HELYX_DIR = '/models/helyx'
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
train_vol = modal.Volume.from_name('drumsplit-train')


def _download_models():
    """Runs once at image build, so containers start with both models on disk."""
    import engine
    from audio_separator.separator import Separator
    engine.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s = Separator(model_file_dir=str(engine.MODEL_DIR), output_dir='/tmp')
    for m in (engine.MIX_MODELS[MIX_MODEL], engine.DRUMSEP):
        s.download_model_files(m)


def _copy_helyx():
    """Image build: copy the chosen Helyx checkpoint + its config out of the training volume."""
    import shutil
    if not HELYX_RUN:
        return
    src = Path('/train') / HELYX_RUN
    dst = Path(HELYX_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'last_mdx23c.ckpt', dst / 'model.ckpt')
    shutil.copy(src / 'config.yaml', dst / 'config.yaml')
    print(f'Helyx from {src}: {(dst / "model.ckpt").stat().st_size / 1e6:.0f} MB')


# Names one-shots (Snare, Clap, Cowbell...) by the one-shot library they sound like. Built on the Mac
# with kitmatch/soundid.py build; fingerprints and categories only. Missing = generic names.
SOUNDID_REF = Path.home() / 'DrumSplits' / 'soundmatch' / 'soundid_ref.npz'

gpu_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg', 'libsndfile1')
    .uv_pip_install('audio-separator[gpu]==0.44.2', 'librosa==0.10.2.post1',   # 1.0 breaks audio-separator
                    'soundfile', 'audioread', 'mutagen',
                    'fastapi', 'python-multipart')
    .env(PUBLIC_ENV)
    .add_local_file(REPO / 'engine.py', '/root/engine.py', copy=True)
    .add_local_file(REPO / 'oneshots.py', '/root/oneshots.py', copy=True)
    .add_local_file(REPO / 'kitmatch' / 'clean.py', '/root/kitmatch/clean.py', copy=True)
    .add_local_file(REPO / 'kitmatch' / 'hits.py', '/root/kitmatch/hits.py', copy=True)
    .add_local_file(REPO / 'kitmatch' / 'soundmatch.py', '/root/kitmatch/soundmatch.py', copy=True)
    .add_local_file(REPO / 'kitmatch' / 'soundid.py', '/root/kitmatch/soundid.py', copy=True)
    .add_local_file(REPO / 'kitmatch' / 'groove.py', '/root/kitmatch/groove.py', copy=True)
    .add_local_file(HERE / 'service.py', '/root/service.py', copy=True)
    .run_function(_download_models)
    .apt_install('git')
    .run_commands('git clone -q https://github.com/ZFTurbo/Music-Source-Separation-Training /msst && '
                  f'cd /msst && git checkout -q {MSST_COMMIT}')
    .uv_pip_install('ml_collections', 'omegaconf', 'pyyaml', 'tqdm')
    .env({'HELYX_RUN': HELYX_RUN})                 # changing the run forces a fresh copy below
    .run_function(_copy_helyx, volumes={'/train': train_vol})
)
if SOUNDID_REF.exists():
    gpu_image = gpu_image.add_local_file(SOUNDID_REF, '/root/kitmatch/soundid_ref.npz', copy=True)

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
        helyx = (f'{HELYX_DIR}/model.ckpt', f'{HELYX_DIR}/config.yaml', '/msst') if HELYX_RUN else None
        self.models = engine.Models('/tmp/dq/work', helyx=helyx)
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


@app.function(image=web_image, volumes={'/data': vol}, secrets=secrets, scaledown_window=300,
              timeout=1800)                     # a full split's zip is a few hundred MB on a slow connection
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
