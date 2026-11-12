#!/usr/bin/env python3
"""prep.py - turn training copies + synthesized clips into the MSST folder layout.

  real   TRAINSET/<song>/{kick,snare,hh,other_drums,rest}.flac  (from make_testset.py --trainset)
  ->     OUT/real/<song>/{kick,snare,perc,hh,other}.flac
         OUT/valid/<song>/{mixture,kick,snare,perc,hh,other}.flac   (a few held-out songs, 30 s each)
  synth  OUT/synth/clip_*/{...same...}.flac                          (datagen/render.py)

perc  = other_drums (real) or toms + perc + ride + crash (synth)
other = the song's non-drum music turned down to -26..-16 dB, like the music Demucs leaves in its
        drum stem. The model learns to push that into "other" instead of into the snare or hats.

Runs inside Modal (modal_train.py), but works locally too:
  python3 train/prep.py real /Volumes/T7/drumsplit-trainset OUT
  python3 train/prep.py synth OUT --n 20 --seed 0
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
INSTRUMENTS = ['kick', 'snare', 'perc', 'hh', 'other']
VALID_N = 3
VALID_SECONDS = 30


def _seed(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def _read(p, n=None):
    if not p.exists():
        return None
    x, sr = sf.read(str(p), dtype='float32', always_2d=True)
    x = x.T
    if sr != SR:                                  # e.g. 48 kHz sessions
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, 0)
    return x[:2]


def _fit(x, n):
    if x is None:
        return np.zeros((2, n), np.float32)
    return np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))))[:, :n]


def _write(d, stems, mixture=False):
    d.mkdir(parents=True, exist_ok=True)
    for k in INSTRUMENTS:
        sf.write(str(d / f'{k}.flac'), stems[k].T, SR, subtype='PCM_24')
    if mixture:
        sf.write(str(d / 'mixture.flac'), sum(stems.values()).T, SR, subtype='PCM_24')


def _headroom(stems):
    peak = np.abs(sum(stems.values())).max()
    g = min(1.0, 0.95 / (peak + 1e-9))
    return {k: v * g for k, v in stems.items()}


DRUMS = ('kick', 'snare', 'hh', 'other_drums')


def has_all_drums(p):
    """A validation song needs every drum part, or the silent part's score (-100 dB) swamps the rest."""
    for r in DRUMS:
        x = _read(p / f'{r}.flac')
        if x is None or np.abs(x).max() < 1e-3:
            return False
    return True


def valid_songs(songs):
    """The same few songs every time (chosen by name hash, among songs with all four drum parts),
    so validation scores stay comparable: 3 multitrack songs + 2 rebuilt FL projects (Latin)."""
    pick = []
    for flp, n in ((False, 3), (True, 2)):
        group = sorted((p for p in songs if p.name.endswith('(FLP)') == flp), key=lambda p: _seed(p.name))
        pick += [p.name for p in group if has_all_drums(p)][:n]
    return set(pick)


def real_song(src: Path, out: Path, valid: bool):
    raw = {r: _read(src / f'{r}.flac') for r in ('kick', 'snare', 'hh', 'other_drums', 'rest')}
    n = max((x.shape[1] for x in raw.values() if x is not None), default=0)
    if n < SR * 5:
        return f'skip {src.name}: no audio'
    rng = np.random.default_rng(_seed(src.name))
    stems = {'kick': _fit(raw['kick'], n), 'snare': _fit(raw['snare'], n), 'hh': _fit(raw['hh'], n),
             'perc': _fit(raw['other_drums'], n),
             'other': _fit(raw['rest'], n) * np.float32(10 ** (rng.uniform(-26, -16) / 20))}
    if valid:                                     # 30 s where every drum part plays (then the busiest)
        win = VALID_SECONDS * SR
        if n > win:
            hop = SR
            act = {k: np.abs(stems[k]).mean(0) for k in ('kick', 'snare', 'hh', 'perc')}
            best, i = None, 0
            for j in range(0, n - win, hop):
                lv = [a[j:j + win].mean() / (a.max() + 1e-9) for a in act.values()]
                key = (sum(v > 1e-3 for v in lv), min(lv), sum(lv))
                if best is None or key > best:
                    best, i = key, j
            stems = {k: v[:, i:i + win] for k, v in stems.items()}
    stems = _headroom(stems)
    _write(out, stems, mixture=valid)
    return f"{'valid' if valid else 'train'} {src.name}"


def all_real(trainsets, out: Path):
    if isinstance(trainsets, Path):
        trainsets = [trainsets]
    songs = sorted((p for t in trainsets if t.exists() for p in t.iterdir()
                    if p.is_dir() and (p / 'kick.flac').exists()), key=lambda p: p.name)
    vs = valid_songs(songs)
    for p in songs:
        v = p.name in vs
        try:
            print(real_song(p, out / ('valid' if v else 'real') / p.name, v), flush=True)
        except Exception as e:                    # one bad song shouldn't stop the rest
            print(f'skip {p.name}: {type(e).__name__}: {e}', flush=True)


def synth_clips(out: Path, n: int, seed: int):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'datagen'))
    import render
    for k in range(n):
        rng = np.random.default_rng(seed * 1_000_003 + k + 7)
        tracks, rest, _mix, meta = render.render_clip(rng)
        m = tracks['kick'].shape[1]
        stems = {'kick': tracks['kick'], 'snare': tracks['snare'], 'hh': tracks['hh'],
                 'perc': tracks['toms'] + tracks['perc'] + tracks['ride'] + tracks['crash'],
                 'other': rest}
        stems = _headroom({k2: v.astype(np.float32)[:, :m] for k2, v in stems.items()})
        _write(out / f'clip_{seed:03d}_{k:05d}', stems)
    return f'synth seed {seed}: {n} clips'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['real', 'synth'])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.what == 'real':
        all_real(Path(a.paths[0]), Path(a.paths[1]))
    else:
        print(synth_clips(Path(a.paths[0]) / 'synth', a.n, a.seed))
