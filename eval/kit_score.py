#!/usr/bin/env python3
"""kit_score.py - how much of the one-shot kit is real drums, judged against the beat's real tracks.

    python3 eval/kit_score.py SPLIT_DIR TRUTH_DIR [SPLIT_DIR TRUTH_DIR ...] [--json out.json]

SPLIT_DIR: the site's stems for the song (kick/snare/perc/hh.wav, bass.wav if there).
TRUTH_DIR: the real tracks: truth_kick/snare/hh/other_drums.flac + mix.flac (any sample rate, any offset:
           both are lined up here).

It makes the kit exactly like the site does (kitmatch/clean.write_kit), then for every one-shot looks at
the real tracks at the very moments that sound plays (its onsets) and asks, on magnitude spectra of the
first 200 ms, how the one-shot splits into: real kick / real snare / real other drums (perc) / real hats
/ the music (mix minus drums: 808, synths, vocals). A one-shot is REAL when most of it (>= 50%) is the
piece it claims to be (snare/perc one-shots: snare + other drums). Kit score = real one-shots / all.

Checked on stunyun137 against a by-ear and by-hand review (Nov 2026). Evaluation only.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import nnls
from scipy.signal import fftconvolve, resample_poly, stft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'kitmatch'))
import clean  # noqa: E402
import hits as H  # noqa: E402

SR = clean.SR
PARTS = ('kick', 'snare', 'other_drums', 'hh', 'music')
CLAIM = {'kick': ('kick',), 'sp': ('snare', 'other_drums'), 'hh': ('hh',)}


def _mono(x):
    return x.mean(1) if x.ndim == 2 else x


def _load(f, sr=SR):
    x, r = sf.read(str(f), dtype='float32', always_2d=True)
    x = _mono(x)
    return resample_poly(x, sr, r).astype(np.float32) if r != sr else x


def load_truth(tdir):
    t = {p: _load(tdir / f'truth_{p}.flac') for p in ('kick', 'snare', 'other_drums', 'hh')
         if (tdir / f'truth_{p}.flac').exists()}
    n = max(len(v) for v in t.values())
    t = {p: np.pad(v, (0, n - len(v))) for p, v in t.items()}
    for p in ('kick', 'snare', 'other_drums', 'hh'):
        t.setdefault(p, np.zeros(n, np.float32))
    mix = _load(tdir / 'mix.flac')
    m = min(n, len(mix))
    t = {p: v[:m] for p, v in t.items()}
    t['music'] = mix[:m] - sum(t[p] for p in ('kick', 'snare', 'other_drums', 'hh'))
    return t


def lag_of(stems_sum, truth_sum, max_s=3.0):
    """Samples the split runs late vs the real tracks (an mp3 upload adds ~29 ms of padding)."""
    n = min(len(stems_sum), len(truth_sum), 60 * SR)
    a, b = stems_sum[:n], truth_sum[:n]
    c = fftconvolve(a, b[::-1], 'full')
    lags = np.arange(-n + 1, n)
    c[np.abs(lags) > max_s * SR] = -np.inf
    return int(lags[int(np.argmax(c))])


def _spec(y):
    return np.abs(stft(y, SR, nperseg=1024, noverlap=768)[2]).ravel()


def score(split_dir, truth_dir, ref=None, log=lambda line: None):
    split_dir, truth_dir = Path(split_dir), Path(truth_dir)
    t = load_truth(truth_dir)
    have = lambda r: split_dir / f'{r}.wav' if (split_dir / f'{r}.wav').exists() else None  # noqa: E731
    kit_st = [H.load(have(r)) for r in ('kick', 'snare', 'perc', 'hh') if have(r)]
    m = min(len(v) for v in kit_st)
    lag = lag_of(_mono(sum(v[:m] for v in kit_st)), sum(t[p] for p in ('kick', 'snare', 'other_drums', 'hh')))
    ks = clean.kit_sounds(split_dir, lambda r: have(r) if r != 'other' else None)
    xs = [H.load(have(r)) for r in ('snare', 'perc') if have(r)]
    pool = sum(v[:min(len(v) for v in xs)] for v in xs) if xs else None
    with tempfile.TemporaryDirectory() as td:
        recs = clean.write_kit(split_dir, td, ref=ref, log=log)
        shots = [(r['label'], r['piece'], _mono(sf.read(r['file'], dtype='float32')[0])) for r in recs]
    out = []
    L = int(0.2 * SR)
    for label, piece, y in shots:
        role = {'snare': 'sp'}.get(piece, piece)
        y = y[:L]
        v = clean.fp(y)
        cands = [s for s in ks if clean.GROUP.get(s['role']) == role and s.get('fp') is not None]
        if not cands or v is None:
            continue
        s = max(cands, key=lambda s: float(v @ s['fp']))
        src = H.load(have('kick')) if role == 'kick' else (H.load(have('hh')) if role == 'hh' else pool)
        on = np.sort(np.asarray(s['onsets']))
        on = clean.align(src, on, src[on[0]:][:int(0.04 * SR)])
        n = len(y)
        tn = len(t['music'])
        on = [i - lag for i in on if 0 <= i - lag and i - lag + n <= tn]
        if len(on) < 2:
            continue
        X = np.stack([_spec(np.mean([t[p][i:i + n] for i in on], 0)) for p in PARTS], 1)
        w, _ = nnls(X, _spec(y))
        c = w * np.linalg.norm(X, axis=0)
        share = dict(zip(PARTS, (c / (c.sum() + 1e-12)).round(3)))
        claim = float(sum(share[p] for p in CLAIM[role]))
        fit = float(1 - np.linalg.norm(_spec(y) - X @ w) / (np.linalg.norm(_spec(y)) + 1e-12))
        out.append(dict(label=label, role=role, real=claim >= 0.5, claim=round(claim, 3),
                        top=max(share, key=share.get), share=share, fit=round(fit, 2), hits=len(on)))
    return dict(lag_ms=round(1000 * lag / SR, 1), shots=out,
                kit_score=round(sum(o['real'] for o in out) / max(1, len(out)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+')
    ap.add_argument('--json', default='')
    ap.add_argument('--ref', default=str(HERE.parent / 'kitmatch' / 'soundid_ref.npz'))
    a = ap.parse_args()
    if len(a.pairs) % 2:
        sys.exit(__doc__)
    ref = a.ref if Path(a.ref).exists() else None
    res = {}
    for sd, td in zip(a.pairs[::2], a.pairs[1::2]):
        name = Path(td).name
        r = score(sd, td, ref)
        res[name] = r
        real = sum(o['real'] for o in r['shots'])
        print(f'\n== {name}: {real}/{len(r["shots"])} real (kit score {r["kit_score"]:.0%}, split {r["lag_ms"]} ms late)', flush=True)
        for o in r['shots']:
            tag = 'REAL ' if o['real'] else 'junk '
            mix = ', '.join(f'{k} {v:.0%}' for k, v in sorted(o['share'].items(), key=lambda kv: -kv[1]) if v >= 0.08)
            print(f'  {tag} {o["label"]:13} {o["claim"]:4.0%} its own piece  | {mix}')
    if res:
        tot = [o['real'] for r in res.values() for o in r['shots']]
        print(f'\nALL: {sum(tot)}/{len(tot)} one-shots real ({sum(tot) / max(1, len(tot)):.0%})')
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
