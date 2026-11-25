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
    .add_local_file(HERE / 'msst_patch.py', '/opt/msst_patch.py', copy=True)
    .add_local_file(HERE / 'mixaug.py', '/opt/mixaug.py', copy=True)
    .run_commands('python /opt/msst_patch.py /msst')       # adds stem_l1_snr_loss (see msst_patch.py)
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


@app.function(image=image, volumes={V: vol}, cpu=2, memory=8192, timeout=1800, max_containers=40)
def prep_song(src: str, dest: str, valid: bool):
    from pathlib import Path as P
    try:
        r = _prep_mod().real_song(P(src), P(dest), valid)
    except Exception as e:                        # one bad song shouldn't stop the rest
        r = f'skip {P(src).name}: {type(e).__name__}: {e}'
    vol.commit()
    return r


@app.function(image=image, volumes={V: vol}, cpu=2, memory=8192, timeout=3 * 3600)
def prep_real(redo: bool = False):
    """Lays out every training song. Only new songs are prepped (redo=True does them all); songs
    whose source is gone, or that moved between training and validation, are cleared first."""
    import shutil
    prep = _prep_mod()
    if redo:
        for d in ('real', 'valid'):
            shutil.rmtree(V / 'data' / d, ignore_errors=True)
    sets = sorted(V.glob('trainset*'))              # /trainset (stems) + /trainset-flp*, -als (rebuilt projects)
    print('reading', [p.name for p in sets], flush=True)
    jobs = prep.plan(sets, V / 'data')
    want = {dest for _, dest, _ in jobs}
    for d in ('real', 'valid'):
        for old in (V / 'data' / d).glob('*') if (V / 'data' / d).exists() else []:
            if str(old) not in want:
                shutil.rmtree(old, ignore_errors=True)
    vol.commit()
    todo = [j for j in jobs if not prep.done(Path(j[1]), j[2])]
    print(f'{len(jobs)} songs, {len(jobs) - len(todo)} already prepped, {len(todo)} to do', flush=True)
    for line in prep_song.starmap(todo):
        print(line, flush=True)
    return sorted(dest.split('/')[-1] for _, dest, v in jobs if v)


@app.local_entrypoint()
def prep(synth: int = 80, synth_only: bool = False):
    """Lay out the uploaded training songs and render `synth` synthesized clips (dembow, trap, afro...)."""
    jobs = None if synth_only else prep_real.spawn()
    per = 10
    for line in synth_part.starmap([(s, per) for s in range(max(1, synth // per))]):
        print(line)
    if jobs:
        print('held out for validation:', jobs.get())


@app.function(image=image, volumes={V: vol}, cpu=1, memory=2048, timeout=24 * 3600)
def pipeline(run: str, hours: float, steps: int, resume: bool = False, synth: int = 0, init: str = '',
             loss: str = 'stem', snare_weight: float = 1.5, bleedless: float = 0.0):
    """(Re)render `synth` synthesized clips, prep the songs, then train - all in the cloud."""
    if synth:
        for line in synth_part.starmap([(s, 10) for s in range(max(1, synth // 10))]):
            print(line, flush=True)
    held = prep_real.remote()
    print('held out for validation:', held, flush=True)
    return train_remote.remote(run, hours, resume, steps, init, loss, snare_weight, bleedless)


@app.local_entrypoint()
def go(run: str = 'helyx-4', hours: float = 4.0, steps: int = 1000, resume: bool = False, synth: int = 0,
       init: str = 'runs/helyx-3/last_mdx23c.ckpt', loss: str = 'mse', snare_weight: float = 1.0,
       bleedless: float = 0.0):
    """Synth + prep + training as ONE job on the deployed app: safe to close the Mac right after.
    First: modal deploy train/modal_train.py   Then: modal run train/modal_train.py::go
    init: the checkpoint a NEW run starts from (path on the volume); ignored with --resume.
    loss: 'stem' = per-stem L1-SNR (each stem counts equally, snare x snare_weight), 'mse' = MSST default.
    bleedless: >0 adds MSST's bleedless penalty (anything in a stem the true stem doesn't have, e.g. hiss
    between hits) with this coefficient; MSST's suggested value is 2e-5 (helyx-8)."""
    call = modal.Function.from_name('drumsplit-train', 'pipeline').spawn(run, hours, steps, resume, synth, init,
                                                                          loss, snare_weight, bleedless)
    print(f'Started ({call.object_id}): synth clips, prep, then {hours} h of training on {run}.')
    print('Safe to close the terminal or turn the Mac off now.')
    print('Watch it: modal.com -> Apps -> drumsplit-train (deployed) -> pipeline / train_remote')


@app.local_entrypoint()
def everything(run: str = 'helyx-1', hours: float = 6.0, steps: int = 1000):
    """modal run --detach train/modal_train.py::everything --run helyx-1 --hours 6"""
    print(pipeline.remote(run, hours, steps))


@app.local_entrypoint()
def launch(run: str = 'helyx-1', hours: float = 6.0, steps: int = 1000, resume: bool = False):
    """Start training on the DEPLOYED app, so it keeps going if this Mac disconnects or reboots.
    First: modal deploy train/modal_train.py   Then: modal run train/modal_train.py::launch --run helyx-1 --hours 6"""
    f = modal.Function.from_name('drumsplit-train', 'train_remote')
    call = f.spawn(run, hours, resume, steps)
    print(f'Training started ({call.object_id}). Safe to close the terminal or reboot now.')
    print('Watch it: modal.com -> Apps -> drumsplit-train (deployed) -> train_remote')


def _committer(stop):
    while not stop.wait(600):
        try:
            vol.commit()
        except Exception as e:                 # a failed commit just waits for the next one
            print('commit failed:', e)


@app.function(image=image, volumes={V: vol}, gpu='A100-40GB', cpu=8, memory=32768, timeout=24 * 3600)
def train_remote(run: str, hours: float, resume: bool, steps: int, init: str = '', loss: str = 'mse',
                 snare_weight: float = 1.0, bleedless: float = 0.0):
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
           '--device_ids', '0', '--metrics', 'sdr', 'bleedless', 'fullness', '--metric_for_scheduler', 'sdr']
    losses = ['masked_loss']           # MSST's default ('mse')
    if loss == 'stem':                 # each stem scored on its own, in dB, so a quiet snare counts (msst_patch.py)
        w = {'snare': snare_weight}
        losses = ['stem_l1_snr_loss']
        cmd += ['--stem_loss_weights'] + [str(w.get(k, 1.0)) for k in cfg['training']['instruments']]
    if bleedless > 0:                  # helyx-8: penalize what a stem has that the truth doesn't (hiss, bleed)
        losses.append('bleedless_penalty_loss')
        cmd += ['--bleedless_penalty_loss_coef', str(bleedless)]
    if losses != ['masked_loss']:
        cmd += ['--loss'] + losses
    if resume and last.exists():
        cmd += ['--start_check_point', str(last), '--load_optimizer', '--load_scheduler', '--load_epoch',
                '--load_best_metric']
    elif init and (V / init.lstrip('/')).exists():
        # a new run from another run's weights: fresh optimizer (its state belongs to the old loss)
        cmd += ['--start_check_point', str(V / init.lstrip('/')), '--pre_valid']
    else:
        if init:
            print(f'! init checkpoint {init} not found on the volume: starting from DrumSep', flush=True)
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
