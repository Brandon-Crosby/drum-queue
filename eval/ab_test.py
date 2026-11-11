"""ab_test.py - does a heavier split setting actually sound better? Runs on a Modal GPU (~5 min, ~$0.10).

  modal run eval/ab_test.py                                   # Te Veo Sola answer key, 0:50-1:50
  modal run eval/ab_test.py --truth ~/path/to/truth --start 50 --seconds 60

--truth is a folder of answer-key stems (kick.wav snare.wav hh.wav other_drums.wav rest.wav, as made
for eval/score.py). The test mix is rebuilt from them, so the answer key matches it exactly.
The stems are uploaded to a private Modal volume for the run and deleted from it afterwards.

Compares, on the same clip:
  stage 1 (drums out of the mix): Demucs htdemucs_ft with 2 shifts / 25% overlap (now) vs 4 / 50%
  stage 2 (kit pieces):           DrumSep overlap 4 (now) vs 8 ("Extra clean")
"""
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App('drumsplit-ab-test')
vol = modal.Volume.from_name('drumsplit-eval', create_if_missing=True)
MODELS = '/models'


def _download():
    from audio_separator.separator import Separator
    s = Separator(model_file_dir=MODELS, output_dir='/tmp')
    for m in ('htdemucs_ft.yaml', 'MDX23C-DrumSep-aufr33-jarredou.ckpt'):
        s.download_model_files(m)


image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg', 'libsndfile1')
    .uv_pip_install('audio-separator[gpu]==0.44.2', 'soundfile', 'scipy', 'numpy')
    .run_function(_download)
    .add_local_file(HERE / 'score.py', '/root/score.py')
)

STAGE1 = {'now': dict(shifts=2, overlap=0.25), 'heavy': dict(shifts=4, overlap=0.5)}
STAGE2 = {'now': 4, 'extra': 8}


@app.function(image=image, gpu=['L4', 'A10', 'T4'], volumes={'/eval': vol}, timeout=1800)
def run(start: int, seconds: int):
    import logging
    import shutil
    import subprocess
    import time

    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator

    W = Path('/tmp/ab')
    shutil.rmtree(W, ignore_errors=True)
    (W / 'truth').mkdir(parents=True)
    parts = {}
    sr = None
    for f in sorted(Path('/eval/truth').glob('*.wav')):
        y, sr = sf.read(f, always_2d=True, dtype='float32')
        parts[f.stem] = y[start * sr:(start + seconds) * sr]
    n = min(len(v) for v in parts.values())
    mix = sum(v[:n] for v in parts.values())
    g = min(1.0, 0.89 / (np.abs(mix).max() + 1e-9))
    sf.write(W / 'mix.wav', mix * g, sr, subtype='FLOAT')
    for k, v in parts.items():
        sf.write(W / 'truth' / f'{k}.wav', v[:n] * g, sr, subtype='FLOAT')

    def sep(**kw):
        return Separator(output_dir=str(W / 'work'), model_file_dir=MODELS, output_format='WAV',
                         log_level=logging.WARNING, **kw)

    secs, out = {}, {}
    for a, dp in STAGE1.items():
        shutil.rmtree(W / 'work', ignore_errors=True)
        s = sep(demucs_params={'segment_size': 'Default', 'segments_enabled': True, **dp})
        s.load_model(model_filename='htdemucs_ft.yaml')
        t = time.time()
        res = s.separate(str(W / 'mix.wav'))
        secs[f'stage1 {a}'] = time.time() - t
        drum = next(o for o in res if '(Drums)' in o)
        shutil.move(str(W / 'work' / Path(drum).name), W / f'drums_{a}.wav')
        for b, ov in STAGE2.items():
            shutil.rmtree(W / 'work', ignore_errors=True)
            s2 = sep(mdxc_params={'segment_size': 256, 'override_model_segment_size': False,
                                  'batch_size': 1, 'overlap': ov, 'pitch_shift': 0})
            s2.load_model(model_filename='MDX23C-DrumSep-aufr33-jarredou.ckpt')
            t = time.time()
            res = s2.separate(str(W / f'drums_{a}.wav'))
            secs[f'stage2 {b} (after {a})'] = time.time() - t
            est = W / f'kit_{a}_{b}'
            est.mkdir()
            for o in res:
                lab = Path(o).stem.split('_(')[-1].rstrip(')').lower()
                shutil.move(str(W / 'work' / Path(o).name), est / f'{lab}.wav')
            j = W / f'{a}_{b}.json'
            subprocess.run(['python', '/root/score.py', str(W / 'truth'), str(est), '--json', str(j),
                            '--label', f'{a}/{b}'], check=True, capture_output=True)
            out[f'{a}/{b}'] = json.loads(j.read_text())
    return {'scores': out, 'seconds': secs, 'clip': [start, seconds]}


def table(r):
    rows = []
    rows.append(f"Clip {r['clip'][0]}s-{r['clip'][0] + r['clip'][1]}s.  stage 1 / stage 2 settings:\n")
    rows.append(f"{'settings':22}{'kick':>7}{'snare':>7}{'hh':>7}   {'kick pure':>9}{'snare pure':>11}{'hh pure':>8}")
    base = r['scores'].get('now/now')
    for name, s in r['scores'].items():
        sd, lk = s['sdr'], s['leakage']
        pure = [lk.get(p, {}).get(p, 0) for p in ('kick', 'snare', 'hh')]
        line = f"{name:22}" + ''.join(f"{sd.get(p, float('nan')):7.2f}" for p in ('kick', 'snare', 'hh'))
        line += f"   {pure[0]:8.1f}%{pure[1]:10.1f}%{pure[2]:7.1f}%"
        if base and name != 'now/now':
            line += '   (' + ', '.join(f"{p} {sd.get(p, 0) - base['sdr'].get(p, 0):+.2f}" for p in ('kick', 'snare', 'hh')) + ' dB)'
        rows.append(line)
    rows.append('\nSDR = how close each stem is to the real one (dB, higher is better).')
    rows.append('pure = share of the output stem that really is that instrument.\n')
    rows.append('GPU seconds:  ' + ', '.join(f'{k} {v:.0f}' for k, v in r['seconds'].items()))
    return '\n'.join(rows)


@app.local_entrypoint()
def main(truth: str = '~/XportStudio/DrumSplit/TVS_truth', start: int = 50, seconds: int = 60):
    src = Path(os.path.expanduser(truth))
    if not src.is_dir():
        raise SystemExit(f'No folder at {src}')
    print(f'Uploading answer-key stems from {src} ...')
    with vol.batch_upload(force=True) as b:
        b.put_directory(str(src), '/truth')
    try:
        r = run.remote(start, seconds)
    finally:
        vol.remove_file('/truth', recursive=True)        # don't keep private stems around
    print('\n' + table(r))
    Path('ab_result.json').write_text(json.dumps(r, indent=1))
    print('\nFull scores saved to ab_result.json')
