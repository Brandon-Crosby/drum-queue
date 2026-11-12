#!/usr/bin/env python3
"""render.py - render training clips: realistic drum loops where every class is its own track.

  python3 datagen/render.py OUT_DIR --n 100 [--seed 1] [--genre dembow]

Each clip is a folder: kick, snare, toms, hh, perc, ride, crash (+ rest = low leftover music, the
kind Demucs leaves in its drum stem) as FLAC, and mix.flac = their exact sum. A model learns to go
from mix -> the class tracks. Every clip differs: genre, tempo, swing, which sounds, how they're
played (velocity, timing), and how they're processed (EQ, saturation, compression, room).

Sounds come from synth.py (no samples, no licenses). Licensed one-shots can be added later through
the same CLASSES table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from . import synth, patterns
except ImportError:                              # run as a script
    import synth
    import patterns

SR = synth.SR
STEMS = ['kick', 'snare', 'toms', 'hh', 'perc', 'ride', 'crash']
LEVEL_DB = {'kick': 0, 'snare': -3, 'toms': -6, 'hh': -11, 'perc': -8, 'ride': -14, 'crash': -13}


# ---------------------------------------------------------------- processing
def _shelf(x, rng):
    """Random tilt EQ: a gentle low or high boost/cut, like a producer's quick EQ move."""
    from scipy.signal import butter, sosfilt
    fc = rng.uniform(200, 6000)
    g = 10 ** (rng.uniform(-6, 6) / 20)
    band = sosfilt(butter(1, fc, 'highpass' if rng.random() < 0.5 else 'lowpass', fs=SR, output='sos'), x, axis=-1)
    return x + (g - 1) * band


def _saturate(x, rng):
    d = rng.uniform(1, 5)
    return np.tanh(x * d) / np.tanh(d)


def _compress(x, rng):
    """Simple feed-forward compressor (peak follower), for punch/glue variety."""
    thr = 10 ** (rng.uniform(-24, -6) / 20)
    ratio = rng.uniform(2, 8)
    rel = np.exp(-1 / (SR * rng.uniform(0.03, 0.2)))
    env = np.abs(x).max(axis=0)
    e = np.empty_like(env)
    lvl = 0.0
    for i, v in enumerate(env):                 # loop is fine: clips are short
        lvl = v if v > lvl else lvl * rel
        e[i] = lvl
    gain = np.where(e > thr, (thr * (e / thr) ** (1 / ratio)) / np.maximum(e, 1e-9), 1.0)
    return x * gain


def _room(x, rng):
    """Short room/plate: convolve with decaying noise, mixed in low."""
    from scipy.signal import fftconvolve
    sec = rng.uniform(0.2, 1.2)
    t = np.arange(int(sec * SR)) / SR
    ir = rng.standard_normal((2, len(t))) * np.exp(-t / (sec / 5))
    wet = np.stack([fftconvolve(x[c], ir[c])[:x.shape[1]] for c in range(2)])
    wet *= np.abs(x).max() / (np.abs(wet).max() + 1e-9)
    return x + rng.uniform(0.05, 0.3) * wet


def process(x, rng):
    if rng.random() < 0.7:
        x = _shelf(x, rng)
    if rng.random() < 0.4:
        x = _saturate(x / (np.abs(x).max() + 1e-9), rng) * np.abs(x).max()
    if rng.random() < 0.5:
        x = _compress(x, rng)
    if rng.random() < 0.5:
        x = _room(x, rng)
    return x


# ---------------------------------------------------------------- clip
def pick_sounds(cls, genre, rng, k):
    """k sound variants for one class, leaning on the genre's favorites."""
    opts = synth.CLASSES[cls]
    fav = patterns.FAVOR.get(genre, {}).get(cls)
    out = []
    for _ in range(k):
        if fav and rng.random() < 0.75:
            name = fav[rng.integers(len(fav))]
            gen = dict(opts)[name]
        else:
            name, gen = opts[rng.integers(len(opts))]
        sound = gen(rng)
        pan = rng.uniform(-0.6, 0.6) if cls not in ('kick', 'snare') else rng.uniform(-0.1, 0.1)
        out.append((name, sound, pan))
    return out


def leftover_music(n, bpm, rng):
    """Quiet bass + chord tones, like what Demucs leaves in its drum stem."""
    t = np.arange(n) / SR
    beat = 60 / bpm
    root = 55 * 2 ** (rng.integers(0, 12) / 12)
    bass = np.zeros(n)
    for b in range(int(n / SR / beat)):
        i = int(b * beat * SR)
        f = root * 2 ** (rng.choice([0, 3, 5, 7, 10]) / 12)
        seg = np.sin(2 * np.pi * f * t[:int(beat * SR)]) * np.exp(-t[:int(beat * SR)] / 0.3)
        bass[i:i + len(seg)] += seg[:n - i]
    chord = sum(np.sin(2 * np.pi * root * 4 * 2 ** (s / 12) * t) for s in (0, 3, 7)) / 3
    x = 0.6 * bass + 0.2 * chord
    return np.stack([x, x])


def render_clip(rng, genre=None, bars=8):
    genre = genre or list(patterns.GENRES)[rng.integers(len(patterns.GENRES))]
    g = patterns.GENRES[genre]
    bpm = rng.uniform(*g['bpm'])
    swing = rng.uniform(*g['swing'])
    step = 60 / bpm / 4
    n = int((bars * 16 * step + 3.0) * SR)       # + tail for the last hits
    tracks = {c: np.zeros((2, n)) for c in STEMS}
    used = {}
    for cls, grids in g['parts'].items():
        if cls not in STEMS:
            continue
        if cls in ('toms', 'ride', 'crash') and rng.random() < 0.3:
            continue                             # not every beat has every piece
        sounds = pick_sounds(cls, genre, rng, 1 if cls in ('kick', 'crash') else int(rng.integers(1, 3)))
        used[cls] = [s[0] for s in sounds]
        grid = grids[rng.integers(len(grids))]
        for b in range(bars):
            if b % 4 == 0 and rng.random() < 0.3:
                grid = grids[rng.integers(len(grids))]       # switch variations every 4 bars sometimes
            if cls == 'crash' and b % 4:
                continue
            for s, p in enumerate(grid):
                if rng.random() >= p:
                    continue
                t = (b * 16 + s) * step + (swing * step if s % 2 else 0) + rng.normal(0, 0.004)
                i = max(0, int(t * SR))
                name, snd, pan = sounds[rng.integers(len(sounds))]
                vel = rng.uniform(0.75, 1.0) if s % 4 == 0 else rng.uniform(0.45, 0.95)
                seg = snd[:n - i] * vel
                tracks[cls][0, i:i + len(seg)] += seg * (1 - pan) / 2 * 2 ** 0.5
                tracks[cls][1, i:i + len(seg)] += seg * (1 + pan) / 2 * 2 ** 0.5
    for cls in STEMS:
        if np.abs(tracks[cls]).max() > 0:
            x = process(tracks[cls], rng)
            x *= 10 ** ((LEVEL_DB[cls] + rng.uniform(-4, 4)) / 20) / (np.abs(x).max() + 1e-9)
            tracks[cls] = x
    rest = leftover_music(n, bpm, rng) * 10 ** (rng.uniform(-34, -20) / 20) if rng.random() < 0.6 else np.zeros((2, n))
    mix = sum(tracks.values()) + rest
    gain = 0.89 / (np.abs(mix).max() + 1e-9)
    tracks = {k: v * gain for k, v in tracks.items()}
    meta = dict(genre=genre, bpm=round(bpm, 1), swing=round(swing, 3), bars=bars, sounds=used)
    return tracks, rest * gain, mix * gain, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--genre', default=None)
    a = ap.parse_args()
    out = Path(a.out)
    for k in range(a.n):
        rng = np.random.default_rng(a.seed * 1_000_003 + k)
        tracks, rest, mix, meta = render_clip(rng, a.genre)
        d = out / f'clip_{a.seed:03d}_{k:05d}'
        d.mkdir(parents=True, exist_ok=True)
        for c, x in tracks.items():
            sf.write(str(d / f'{c}.flac'), x.T, SR, subtype='PCM_24')
        sf.write(str(d / 'rest.flac'), rest.T, SR, subtype='PCM_24')
        sf.write(str(d / 'mix.flac'), mix.T, SR, subtype='PCM_24')
        (d / 'meta.json').write_text(json.dumps(meta))
        print(d.name, meta['genre'], meta['bpm'], 'bpm', {k: v for k, v in meta['sounds'].items()})


if __name__ == '__main__':
    main()
