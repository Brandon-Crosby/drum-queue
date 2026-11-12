"""modal_train.py - train the DrumSplit drum model on Modal.

One-time upload of the training copies (never the test set):
  modal volume create drumsplit-train
  modal volume put drumsplit-train /Volumes/T7/drumsplit-trainset /trainset
  modal volume put drumsplit-train ~/DrumSplits/flp-test/train /trainset-flp     # FLP rebuilds (datagen/flp_render.py)

Then:
  modal run train/modal_train.py::prep                      # CPU: lay out real songs + render synth clips
  modal run train/modal_train.py::train --run trial --hours 2
  modal run train/modal_train.py::train --run trial --hours 10 --resume   # keep going
  modal volume ls drumsplit-train /runs/trial               # checkpoints + scores

Fine-tunes the DrumSep MDX23C model (what the site uses today for stage 2) with new outputs:
kick, snare, perc, hh, other. Checkpoints land in /runs/<run>/ on the volume; the best one is
model_mdx23c_ep_<n>_sdr_<score>.ckpt and last_mdx23c.ckpt is the latest.
"""
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

MSST = 'https://github.com/ZFTurbo/Music-Source-Separation-Training'
MSST_COMMIT = '1d4b3c431b37de696b3795ed2018493aacba946f'
START_CKPT_URLS = [
    'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/'
    'MDX23C-DrumSep-aufr33-jarredou.ckpt',
    ('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/'
              'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'),
]
SKIP_REQS = 'pyaudio|wxpython|keyboard|sageattention|bitsandbytes|moises-light'

app = modal.App('drumsplit-train')
vol = modal.Volume.from_name('drumsplit-train', create_if_missing=True)
V = Path('/vol')
HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'ffmpeg', 'libsndfile1', 'wget')
    .pip_install('torch==2.5.1', 'torchaudio==2.5.1')
    .run_commands(
        f'git clone -q {MSST} /msst && cd /msst && git checkout -q {MSST_COMMIT}',
        f"grep -v -i -E '{SKIP_REQS}' /msst/requirements.txt > /tmp/req.txt",
        'pip install -q -r /tmp/req.txt lameenc',
    )
    .add_local_dir(HERE.parent / 'datagen', '/repo/datagen', ignore=['__pycache__'])
    .add_local_file(HERE / 'prep.py', '/repo/train/prep.py')
    .add_local_file(HERE / 'config_drumsplit.yaml', '/repo/train/config_drumsplit.yaml')
)


def _prep_mod():
    import sys
    sys.path.insert(0, '/repo/train')
    import prep
    return prep


@app.function(image=image, volumes={V: vol}, cpu=2, memory=4096, timeout=3600)
def synth_part(seed: int, n: int):
    r = _prep_mod().synth_clips(V / 'data' / 'synth', n, seed)
    vol.commit()
    return r


@app.function(image=image, volumes={V: vol}, cpu=4, memory=16384, timeout=3 * 3600)
def prep_real():
    import shutil
    prep = _prep_mod()
    for d in ('real', 'valid'):
        shutil.rmtree(V / 'data' / d, ignore_errors=True)
    prep.all_real([V / 'trainset', V / 'trainset-flp'], V / 'data')   # stems + rebuilt FL projects
    vol.commit()
    return sorted(p.name for p in (V / 'data' / 'valid').iterdir())


@app.local_entrypoint()
def prep(synth: int = 80, synth_only: bool = False):
    """Lay out the uploaded training songs and render `synth` synthesized clips (dembow, trap, afro...)."""
    jobs = None if synth_only else prep_real.spawn()
    per = 10
    for line in synth_part.starmap([(s, per) for s in range(max(1, synth // per))]):
        print(line)
    if jobs:
        print('held out for validation:', jobs.get())


def _committer(stop):
    while not stop.wait(600):
        try:
            vol.commit()
        except Exception as e:                 # a failed commit just waits for the next one
            print('commit failed:', e)


@app.function(image=image, volumes={V: vol}, gpu='A100-40GB', cpu=8, memory=32768, timeout=24 * 3600)
def train_remote(run: str, hours: float, resume: bool, steps: int):
    import yaml
    out = V / 'runs' / run
    out.mkdir(parents=True, exist_ok=True)
    start = V / 'ckpt' / 'drumsep_mdx23c.ckpt'
    if not start.exists() or start.stat().st_size < 100_000_000:   # missing or a failed download
        start.parent.mkdir(parents=True, exist_ok=True)
        for url in START_CKPT_URLS:
            if subprocess.run(['wget', '-q', '-O', str(start) + '.part', url]).returncode == 0:
                os.replace(str(start) + '.part', start)
                break
        else:
            raise SystemExit('Could not download the DrumSep model. Upload it once from your Mac:\n'
                             '  modal volume put drumsplit-train <path>/MDX23C-DrumSep-aufr33-jarredou.ckpt '
                             '/ckpt/drumsep_mdx23c.ckpt')
    cfg = yaml.load(open('/repo/train/config_drumsplit.yaml'), Loader=yaml.FullLoader)
    cfg['training']['num_steps'] = steps
    cfg['training']['num_epochs'] = 10_000              # we stop on time instead
    (out / 'config.yaml').write_text(yaml.dump(cfg))
    last = out / 'last_mdx23c.ckpt'
    cmd = ['python', '/msst/train.py', '--model_type', 'mdx23c', '--config_path', str(out / 'config.yaml'),
           '--results_path', str(out), '--dataset_type', '1',
           # real songs listed twice so they're picked as often as the synthesized clips
           '--data_path', str(V / 'data/real'), str(V / 'data/real'), str(V / 'data/synth'),
           '--valid_path', str(V / 'data/valid'), '--num_workers', '6', '--pin_memory',
           '--device_ids', '0', '--metrics', 'sdr', '--metric_for_scheduler', 'sdr']
    if resume and last.exists():
        cmd += ['--start_check_point', str(last), '--load_optimizer', '--load_scheduler', '--load_epoch',
                '--load_best_metric']
    else:
        cmd += ['--start_check_point', str(start), '--pre_valid']   # pre_valid = score before any training
    print(' '.join(cmd), flush=True)
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    p = subprocess.Popen(cmd, cwd='/msst')
    deadline = time.time() + hours * 3600
    while p.poll() is None and time.time() < deadline:
        time.sleep(30)
    if p.poll() is None:
        print(f'time limit ({hours} h) reached: stopping; resume with --resume', flush=True)
        p.terminate()
        try:
            p.wait(120)
        except subprocess.TimeoutExpired:
            p.kill()
    stop.set()
    vol.commit()
    return sorted(f.name for f in out.glob('*.ckpt'))


@app.local_entrypoint()
def train(run: str = 'trial', hours: float = 2.0, resume: bool = False, steps: int = 1000):
    """Train for `hours` (A100, about $2.50/h incl. CPU). Checkpoints go to /runs/<run> on the volume."""
    print(train_remote.remote(run, hours, resume, steps))
