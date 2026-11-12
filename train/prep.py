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
    assert sr == SR, (p, sr)
    x = x.T
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


def valid_songs(names):
    """The same few songs every time (chosen by name hash), so validation scores stay comparable."""
    return set(sorted(names, key=_seed)[:VALID_N])


def real_song(src: Path, out: Path, valid: bool):
    raw = {r: _read(src / f'{r}.flac') for r in ('kick', 'snare', 'hh', 'other_drums', 'rest')}
    n = max((x.shape[1] for x in raw.values() if x is not None), default=0)
    if n < SR * 5:
        return f'skip {src.name}: no audio'
    rng = np.random.default_rng(_seed(src.name))
    stems = {'kick': _fit(raw['kick'], n), 'snare': _fit(raw['snare'], n), 'hh': _fit(raw['hh'], n),
             'perc': _fit(raw['other_drums'], n),
             'other': _fit(raw['rest'], n) * np.float32(10 ** (rng.uniform(-26, -16) / 20))}
    if valid:                                     # the 30 s with the most drum activity
        drums = np.abs(stems['kick'] + stems['snare'] + stems['hh'] + stems['perc']).mean(0)
        win = VALID_SECONDS * SR
        if n > win:
            hop = SR
            energy = [drums[i:i + win].mean() for i in range(0, n - win, hop)]
            i = int(np.argmax(energy)) * hop
            stems = {k: v[:, i:i + win] for k, v in stems.items()}
    stems = _headroom(stems)
    _write(out, stems, mixture=valid)
    return f"{'valid' if valid else 'train'} {src.name}"


def all_real(trainset: Path, out: Path):
    songs = sorted(p for p in trainset.iterdir() if p.is_dir() and (p / 'kick.flac').exists())
    vs = valid_songs([p.name for p in songs])
    for p in songs:
        v = p.name in vs
        print(real_song(p, out / ('valid' if v else 'real') / p.name, v), flush=True)


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
