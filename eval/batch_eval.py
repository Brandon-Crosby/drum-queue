"""batch_eval.py - score the drum splitter on the whole test set, on Modal GPUs (~$0.10 a beat).

  modal run eval/batch_eval.py --testset ~/DrumSplits/testset
  modal run eval/batch_eval.py --testset ~/DrumSplits/testset --only dembow
  modal run eval/batch_eval.py --ckpt /runs/trial/last_mdx23c.ckpt --label trial   # a model we trained

The test set is what eval/make_testset.py writes: one folder per beat with mix.flac and the
answer-key stems (kick, snare, hh, other_drums, rest). This uploads it to a private Modal volume
(only beats not uploaded yet), splits every beat with the site's pipeline on GPUs in parallel,
scores each against its answer key with eval/score.py, and prints one scoreboard.
Results are saved to eval_results/<date>-<label>.json, so a new model can be compared to this run.
Beats that contain sample-pack sounds are for TESTING only - never use this set for training.
"""
import datetime as dt
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App('drumsplit-batch-eval')
vol = modal.Volume.from_name('drumsplit-eval', create_if_missing=True)
train_vol = modal.Volume.from_name('drumsplit-train', create_if_missing=True)
MSST_COMMIT = '1d4b3c431b37de696b3795ed2018493aacba946f'
MODELS = '/models'
PIECES = ['kick', 'snare', 'hh']


def _download():
    from audio_separator.separator import Separator
    s = Separator(model_file_dir=MODELS, output_dir='/tmp')
    for m in ('htdemucs_ft.yaml', 'MDX23C-DrumSep-aufr33-jarredou.ckpt'):
        s.download_model_files(m)


image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg', 'libsndfile1')
    .uv_pip_install('audio-separator[gpu]==0.44.2', 'librosa==0.10.2.post1', 'soundfile', 'scipy', 'numpy')
    .run_function(_download)
    .apt_install('git')
    .run_commands('git clone -q https://github.com/ZFTurbo/Music-Source-Separation-Training /msst && '
                  f'cd /msst && git checkout -q {MSST_COMMIT}')
    .uv_pip_install('ml_collections', 'omegaconf', 'pyyaml', 'tqdm')
    .add_local_file(HERE / 'score.py', '/root/score.py')
    .add_local_file(HERE / 'junk.py', '/root/junk.py')
)


@app.cls(image=image, gpu=['L4', 'A10', 'T4'], volumes={'/eval': vol, '/train': train_vol}, timeout=1800,
         max_containers=4)
class Bench:
    source: str = modal.parameter(default='mix')     # 'mix' = stems summed, 'master' = the session's Master bounce
    ckpt: str = modal.parameter(default='')          # a checkpoint on the drumsplit-train volume, e.g. /runs/trial/x.ckpt
    chunk_s: float = modal.parameter(default=0.0)   # >0: run our model on windows of ~this many seconds

    @modal.enter()
    def load(self):
        import logging
        from audio_separator.separator import Separator
        self.work = Path('/tmp/bench')
        self.work.mkdir(exist_ok=True)
        self.mix = Separator(output_dir=str(self.work), model_file_dir=MODELS, output_format='WAV',
                             log_level=logging.WARNING)
        self.mix.load_model(model_filename='htdemucs_ft.yaml')
        self.kit = Separator(output_dir=str(self.work), model_file_dir=MODELS, output_format='WAV',
                             log_level=logging.WARNING,
                             mdxc_params={'segment_size': 256, 'override_model_segment_size': False,
                                          'batch_size': 1, 'overlap': 4, 'pitch_shift': 0})
        self.kit.load_model(model_filename='MDX23C-DrumSep-aufr33-jarredou.ckpt')
        self.ours = None
        if self.ckpt:                                   # our fine-tuned model replaces stage 2
            import sys
            import torch
            sys.path.insert(0, '/msst')
            from utils.settings import get_model_from_config
            ck = Path('/train') / self.ckpt.lstrip('/')
            model, cfg = get_model_from_config('mdx23c', str(ck.parent / 'config.yaml'))
            sd = torch.load(str(ck), map_location='cpu', weights_only=False)
            for k in ('state_dict', 'model_state_dict', 'state'):
                if isinstance(sd, dict) and k in sd:
                    sd = sd[k]
            model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in sd.items()})
            if self.chunk_s and self.chunk_s > 0:      # longer context per window, same model (convolutional in time)
                hop = cfg.audio.hop_length
                dim_t = max(32, int(round(self.chunk_s * cfg.audio.sample_rate / hop / 32)) * 32)   # 5 halvings: multiple of 32
                cfg.audio.dim_t = dim_t
                cfg.audio.chunk_size = hop * (dim_t - 1)
                if 'inference' in cfg:
                    cfg.inference.dim_t = dim_t
                print(f'chunk: {cfg.audio.chunk_size} samples ({cfg.audio.chunk_size / cfg.audio.sample_rate:.2f} s), dim_t {dim_t}')
            self.ours = (model.to('cuda').eval(), cfg)

    @modal.method()
    def run(self, beat: str, listen=False):
        import shutil
        import subprocess
        import time
        import numpy as np
        import soundfile as sf
        vol.reload()
        src = Path('/eval/testset') / beat
        W = self.work / 'job'
        shutil.rmtree(W, ignore_errors=True)
        (W / 'truth').mkdir(parents=True)
        want = 'mix_master' if self.source == 'master' else 'mix'
        print(f'{beat}: splitting {want}.flac', flush=True)
        if not (src / f'{want}.flac').exists():
            return {'beat': beat, 'error': f'no {want}.flac for this beat (run eval/add_master.py)'}
        for f in src.glob('*.flac'):                     # score.py reads WAV
            if f.stem.startswith('mix') and f.stem != want:
                continue
            y, sr = sf.read(str(f), dtype='float32')
            sf.write(str(W / 'truth' / (f.stem + '.wav')) if f.stem != want else str(W / 'mix.wav'), y, sr,
                     subtype='FLOAT')
        t = time.time()
        for f in self.work.glob('*.wav'):
            f.unlink()
        outs = self.mix.separate(str(W / 'mix.wav'))
        drums = next(o for o in outs if '(Drums)' in o)
        shutil.move(str(self.work / Path(drums).name), W / 'drums.wav')
        for o in outs:
            (self.work / Path(o).name).unlink(missing_ok=True)
        est = W / 'kit'
        est.mkdir()
        if self.ours:
            import torch
            from utils.model_utils import demix
            model, cfg = self.ours
            x, sr = sf.read(str(W / 'drums.wav'), dtype='float32', always_2d=True)
            with torch.inference_mode():
                res = demix(cfg, model, torch.tensor(x.T), torch.device('cuda'), 'mdx23c')
            names = {'perc': 'toms'}                     # score.py's kit names; perc sits where toms were
            for k, v in res.items():
                sf.write(str(est / f"{names.get(k, k)}.wav"), v.T, sr, subtype='FLOAT')
        for o in ([] if self.ours else self.kit.separate(str(W / 'drums.wav'))):
            nm = Path(o).name
            shutil.move(str(self.work / nm), est / (nm[nm.rfind('(') + 1:nm.rfind(')')].strip().lower() + '.wav'))
        gpu_s = time.time() - t
        if listen:                                    # save what the model actually produced, to listen to
            dest = Path('/eval/listen') / (listen if isinstance(listen, str) else '') / beat
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            def flac(a, b):                           # FLAC: ~1/4 the size of the float WAVs
                y, r = sf.read(str(a), dtype='float32')
                sf.write(str(b.with_suffix('.flac')), np.clip(y, -1, 1), r, subtype='PCM_16')
            flac(W / 'mix.wav', dest / 'mix.wav')
            for o in est.glob('*.wav'):
                flac(o, dest / f'ours_{o.name}')
            for t_ in (W / 'truth').glob('*.wav'):
                if t_.stem != 'rest':
                    flac(t_, dest / f'truth_{t_.name}')
            vol.commit()
        p = subprocess.run(['python', '/root/score.py', str(W / 'truth'), str(est), '--json', str(W / 's.json'),
                            '--label', beat], capture_output=True, text=True)
        if p.returncode:
            return {'beat': beat, 'error': (p.stdout + p.stderr)[-800:]}
        s = json.loads((W / 's.json').read_text())
        import sys
        sys.path.insert(0, '/root')
        from junk import hit_junk                     # how clean each hit is, the way it sounds (see junk.py)
        junk = {}
        for piece, parts in (('kick', ['kick']), ('snare', ['snare', 'toms']), ('hh', ['hh'])):
            tf = W / 'truth' / f'{piece}.wav'
            ps = [est / f'{q}.wav' for q in parts if (est / f'{q}.wav').exists()]
            if not tf.exists() or not ps:
                continue
            t_, srt = sf.read(str(tf), dtype='float32')
            ys = [sf.read(str(q), dtype='float32') for q in ps]
            m = min(len(y) for y, _ in ys)
            junk[piece] = hit_junk(t_, sum(y[:m] for y, _ in ys), srt, sr_ours=ys[0][1])
        # where the real snare/kick hits land among our stems, alone vs stacked on the other (see junk.routing)
        from junk import routing
        route = {}
        outs_ = {o.stem: sf.read(str(o), dtype='float32') for o in est.glob('*.wav')}
        for piece, partner in (('snare', 'kick'), ('kick', 'snare')):
            tf, pf = W / 'truth' / f'{piece}.wav', W / 'truth' / f'{partner}.wav'
            if tf.exists() and outs_:
                t_, srt = sf.read(str(tf), dtype='float32')
                p_ = sf.read(str(pf), dtype='float32')[0] if pf.exists() else None
                route[piece] = routing(t_, srt, outs_, partner=p_)
        return {'beat': beat, 'split': want, 'gpu_s': round(gpu_s), 'sdr': s.get('sdr', {}), 'leakage': s.get('leakage', {}),
                'hits': s.get('hits', {}), 'junk': junk, 'routing': route}


def board(results):
    ok = [r for r in results if 'sdr' in r]
    lines = [f"{'beat':34}{'kick':>8}{'snare':>8}{'hh':>8}   {'kick pure':>9}{'snare pure':>11}{'hh pure':>8}"]
    for r in sorted(ok, key=lambda r: r['beat'].lower()):
        sd, lk = r['sdr'], r['leakage']
        pure = [lk.get(p, {}).get(p) for p in PIECES]
        lines.append(f"{r['beat'][:33]:34}" + ''.join(f"{sd.get(p, float('nan')):8.1f}" for p in PIECES) + '   '
                     + ''.join(f"{(v if v is not None else float('nan')):9.0f}%" if i == 0 else
                               f"{(v if v is not None else float('nan')):10.0f}%" if i == 1 else
                               f"{(v if v is not None else float('nan')):7.0f}%" for i, v in enumerate(pure)))
    if ok:
        import statistics as st
        med = lambda k, p: st.median([r[k][p] if k == 'sdr' else r[k].get(p, {}).get(p, 0) for r in ok
                                      if (r[k].get(p) is not None)] or [float('nan')])
        lines.append('-' * 90)
        lines.append(f"{'MEDIAN (' + str(len(ok)) + ' beats)':34}" + ''.join(f"{med('sdr', p):8.1f}" for p in PIECES) + '   '
                     + f"{med('leakage', 'kick'):9.0f}%{med('leakage', 'snare'):10.0f}%{med('leakage', 'hh'):7.0f}%")
    jk = [r for r in ok if r.get('junk')]
    if jk:
        import statistics as st
        lines.append('\nHit cleanliness: leftover junk under each real hit, dB (more negative = cleaner; body = 200 Hz-2 kHz)')
        lines.append(f"{'':34}{'kick':>8}{'body':>7}{'snare':>9}{'body':>7}{'hh':>8}{'top':>7}")
        def cell(r, p, k):
            v = (r['junk'].get(p) or {}).get(k)
            return v if v is not None else float('nan')
        for r in sorted(jk, key=lambda r: r['beat'].lower()):
            lines.append(f"{r['beat'][:33]:34}{cell(r, 'kick', 'all'):8.1f}{cell(r, 'kick', 'body'):7.1f}"
                         f"{cell(r, 'snare', 'all'):9.1f}{cell(r, 'snare', 'body'):7.1f}{cell(r, 'hh', 'all'):8.1f}{cell(r, 'hh', 'top'):7.1f}")
        def jm(p, k):
            v = [cell(r, p, k) for r in jk if cell(r, p, k) == cell(r, p, k)]
            return st.median(v) if v else float('nan')
        lines.append(f"{'MEDIAN':34}{jm('kick', 'all'):8.1f}{jm('kick', 'body'):7.1f}{jm('snare', 'all'):9.1f}"
                     f"{jm('snare', 'body'):7.1f}{jm('hh', 'all'):8.1f}{jm('hh', 'top'):7.1f}")
    rt = [r for r in ok if r.get('routing')]
    if rt:
        import statistics as st
        lines.append('\nWhere the real hits land: share of each real snare hit found in our snare+perc stems, and of each')
        lines.append('real kick in our kick stem; "stacked" = the snare and kick hit together (higher is better)')
        lines.append(f"{'':34}{'snare alone':>12}{'stacked':>9}{'kick alone':>12}{'stacked':>9}")
        def share(r, piece, kind):
            d = (r['routing'].get(piece) or {}).get(kind) or {}
            ks_ = ['snare', 'toms'] if piece == 'snare' else ['kick']
            v = [d.get(k) for k in ks_ if d.get(k) is not None]
            return sum(v) if v else float('nan')
        cols = (('snare', 'alone'), ('snare', 'stacked'), ('kick', 'alone'), ('kick', 'stacked'))
        for r in sorted(rt, key=lambda r: r['beat'].lower()):
            vals = [share(r, p, k) for p, k in cols]
            lines.append(f"{r['beat'][:33]:34}" + ''.join(f"{v:>{w}.0%}" if v == v else f"{'-':>{w}}" for v, w in zip(vals, (12, 9, 12, 9))))
        meds = []
        for p, k in cols:
            v = [share(r, p, k) for r in rt]
            v = [x for x in v if x == x]
            meds.append(st.median(v) if v else float('nan'))
        lines.append(f"{'MEDIAN':34}" + ''.join(f"{v:>{w}.0%}" for v, w in zip(meds, (12, 9, 12, 9))))
    for r in results:
        if 'error' in r:
            lines.append(f"! {r['beat']}: {r['error'][-300:]}")
    lines.append('\nSDR in dB vs the real tracks (higher is better). pure = share of the stem that really is that piece.')
    return '\n'.join(lines)


@app.local_entrypoint()
def main(testset: str = '~/DrumSplits/testset', only: str = '', label: str = 'current', ckpt: str = '', mix: str = 'mix',
         listen: str = '', chunk: float = 0.0):
    """--listen "beat name" also saves the model's actual separated audio (not just scores) for that one
    beat to the eval volume, so you can download and listen to it: modal volume get drumsplit-eval
    /listen/<beat> ~/Desktop/listen. --listen all saves every beat, under /listen/<label>/."""
    root = Path(os.path.expanduser(testset))
    have = set()
    try:
        have = {Path(e.path).name for e in vol.listdir('/testset')}
    except Exception:
        pass
    pick = lambda name: not only or only.lower() in name.lower()  # noqa: E731
    if root.is_dir():
        beats = sorted(p.name for p in root.iterdir() if (p / 'mix.flac').exists() and pick(p.name))
    else:                                         # no local copy: score the beats already on the eval volume
        beats = sorted(b for b in have if pick(b))
        if beats:
            print(f'{root} not found; using the {len(beats)} beat(s) already uploaded to the eval volume')
    if not beats:
        raise SystemExit(f'No test beats in {root} or on the eval volume (run eval/make_testset.py first)')
    new = [b for b in beats if b not in have]
    if new:
        print(f'Uploading {len(new)} beat(s) to the private eval volume ...')
        with vol.batch_upload(force=True) as up:
            for b in new:
                up.put_directory(str(root / b), f'/testset/{b}')
    print(f'Splitting and scoring {len(beats)} beat(s) on GPUs ...')
    if ckpt:
        print(f'Stage 2 = our model {ckpt}')
    if listen.lower() == 'all':                   # save every beat's split under /listen/<label>/
        args = [(b, label) for b in beats]
        listen_beat = None
    else:
        listen_beat = next((b for b in beats if listen and listen.lower() in b.lower()), None)
        if listen and not listen_beat:
            print(f'! --listen "{listen}" matched no beat in the test set; scoring without saving audio')
        args = [(b, b == listen_beat) for b in beats]
    if mix == 'master':                           # the Master cuts are added to beats already uploaded
        print('Splitting each beat\'s Master bounce (scores vs the stems are only a guide here)')
        with vol.batch_upload(force=True) as up:
            for b in beats:
                if (root / b / 'mix_master.flac').exists():
                    up.put_file(str(root / b / 'mix_master.flac'), f'/testset/{b}/mix_master.flac')
    results = list(Bench(ckpt=ckpt, source=mix, chunk_s=chunk).run.starmap(args, return_exceptions=True))
    results = [r if isinstance(r, dict) else {'beat': b, 'error': repr(r)} for b, r in zip(beats, results)]
    print('\n' + board(results))
    out = Path('eval_results')
    out.mkdir(exist_ok=True)
    f = out / f"{dt.date.today().isoformat()}-{label}.json"
    f.write_text(json.dumps(results, indent=1))
    print(f'\nSaved {f}')
    if listen.lower() == 'all':
        print(f'\nTo download every split: modal volume get drumsplit-eval "/listen/{label}" ~/DrumSplits/splits')
    if listen_beat:
        print(f'\nTo listen: modal volume get drumsplit-eval "/listen/{listen_beat}" ~/Desktop/listen')
