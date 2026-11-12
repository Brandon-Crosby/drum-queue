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
)


@app.cls(image=image, gpu=['L4', 'A10', 'T4'], volumes={'/eval': vol, '/train': train_vol}, timeout=1800,
         max_containers=4)
class Bench:
    ckpt: str = modal.parameter(default='')          # a checkpoint on the drumsplit-train volume, e.g. /runs/trial/x.ckpt

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
            self.ours = (model.to('cuda').eval(), cfg)

    @modal.method()
    def run(self, beat: str):
        import shutil
        import subprocess
        import time
        import soundfile as sf
        vol.reload()
        src = Path('/eval/testset') / beat
        W = self.work / 'job'
        shutil.rmtree(W, ignore_errors=True)
        (W / 'truth').mkdir(parents=True)
        for f in src.glob('*.flac'):                     # score.py reads WAV
            y, sr = sf.read(str(f), dtype='float32')
            sf.write(str(W / 'truth' / (f.stem + '.wav')) if f.stem != 'mix' else str(W / 'mix.wav'), y, sr,
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
        p = subprocess.run(['python', '/root/score.py', str(W / 'truth'), str(est), '--json', str(W / 's.json'),
                            '--label', beat], capture_output=True, text=True)
        if p.returncode:
            return {'beat': beat, 'error': (p.stdout + p.stderr)[-800:]}
        s = json.loads((W / 's.json').read_text())
        return {'beat': beat, 'gpu_s': round(gpu_s), 'sdr': s.get('sdr', {}), 'leakage': s.get('leakage', {}),
                'hits': s.get('hits', {})}


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
    for r in results:
        if 'error' in r:
            lines.append(f"! {r['beat']}: {r['error'][-300:]}")
    lines.append('\nSDR in dB vs the real tracks (higher is better). pure = share of the stem that really is that piece.')
    return '\n'.join(lines)


@app.local_entrypoint()
def main(testset: str = '~/DrumSplits/testset', only: str = '', label: str = 'current', ckpt: str = ''):
    root = Path(os.path.expanduser(testset))
    beats = sorted(p.name for p in root.iterdir() if (p / 'mix.flac').exists() and (not only or only.lower() in p.name.lower()))
    if not beats:
        raise SystemExit(f'No test beats in {root} (run eval/make_testset.py first)')
    have = set()
    try:
        have = {Path(e.path).name for e in vol.listdir('/testset')}
    except Exception:
        pass
    new = [b for b in beats if b not in have]
    if new:
        print(f'Uploading {len(new)} beat(s) to the private eval volume ...')
        with vol.batch_upload(force=True) as up:
            for b in new:
                up.put_directory(str(root / b), f'/testset/{b}')
    print(f'Splitting and scoring {len(beats)} beat(s) on GPUs ...')
    if ckpt:
        print(f'Stage 2 = our model {ckpt}')
    results = list(Bench(ckpt=ckpt).run.map(beats, return_exceptions=True))
    results = [r if isinstance(r, dict) else {'beat': b, 'error': repr(r)} for b, r in zip(beats, results)]
    print('\n' + board(results))
    out = Path('eval_results')
    out.mkdir(exist_ok=True)
    f = out / f"{dt.date.today().isoformat()}-{label}.json"
    f.write_text(json.dumps(results, indent=1))
    print(f'\nSaved {f}')
