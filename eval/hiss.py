#!/usr/bin/env python3
"""Hiss between hits: how loud each of our drum stems is where the real stem is silent.

For each beat in a batch_eval listen folder (ours_*.flac next to truth_*.flac) and each drum stem:
  gaps  = 10 ms frames where the true stem is more than 50 dB under its own loud hits (silence between hits)
  hiss  = level of OUR stem in those gaps, in dB relative to the true stem's hits (lower = cleaner)
  truth = the same for the true stem (its own noise floor; our floor can't usefully go below it)
Our stems are resampled to the truth's rate first (ours 44.1 kHz, truth 48 kHz).

  python3 eval/hiss.py ~/DrumSplits/splits_h7/helyx-7 --label helyx-7      (resumable: --out file.jsonl)
"""
import argparse, json, time
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PAIRS = {'kick': ('ours_kick', 'truth_kick'), 'snare': ('ours_snare', 'truth_snare'),
         'perc': ('ours_toms', 'truth_other_drums'), 'hh': ('ours_hh', 'truth_hh')}
FR = 0.010


def load(p, to=None):
    x, sr = sf.read(str(p), dtype='float32', always_2d=True)
    x = x.mean(1)
    if to and sr != to:
        g = gcd(to, sr)
        x = resample_poly(x, to // g, sr // g).astype('float32')
        sr = to
    return x, sr


def frames_db(x, sr):
    h = int(FR * sr)
    n = len(x) // h
    return 10 * np.log10((x[:n * h].reshape(n, h) ** 2).mean(1) + 1e-14)


def beat(d):
    out = {}
    for stem, (o, t) in PAIRS.items():
        if not (d / f'{o}.flac').exists() or not (d / f'{t}.flac').exists():
            continue
        tr, sr = load(d / f'{t}.flac')
        ou, _ = load(d / f'{o}.flac', sr)
        n = min(len(tr), len(ou))
        T, O = frames_db(tr[:n], sr), frames_db(ou[:n], sr)
        hits = float(np.percentile(T, 99))
        if hits < -70:                               # this beat has no such stem
            continue
        # silent in the truth, and not just a frame between two close hits: 3 silent frames in a row
        quiet = T < hits - 50
        gap = quiet & np.roll(quiet, 1) & np.roll(quiet, -1)
        if gap.sum() < 50:
            continue
        p = lambda v: float(10 * np.log10(np.mean(10 ** (v / 10)) + 1e-14))
        out[stem] = dict(hiss=round(p(O[gap]) - hits, 1), truth=round(p(T[gap]) - hits, 1),
                         gap_s=round(gap.sum() * FR, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--budget', type=float, default=0, help='stop after this many seconds (resume later)')
    a = ap.parse_args()
    out = Path(a.out or f'hiss-{a.label or Path(a.root).name}.jsonl')
    done = {json.loads(l)['beat'] for l in out.read_text().splitlines()} if out.exists() else set()
    t0 = time.time()
    for d in sorted(p for p in Path(a.root).expanduser().iterdir() if p.is_dir()):
        if d.name in done:
            continue
        if a.budget and time.time() - t0 > a.budget:
            print('more to do: run again'); return
        r = beat(d)
        with open(out, 'a') as f:
            f.write(json.dumps(dict(beat=d.name, **r)) + '\n')
        print(d.name, {k: v['hiss'] for k, v in r.items()}, flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    for stem in PAIRS:
        v = [r[stem]['hiss'] for r in rows if stem in r]
        if v:
            print(f'{stem:5s} median hiss between hits {np.median(v):6.1f} dB under the hits  ({len(v)} beats)')


if __name__ == '__main__':
    main()
