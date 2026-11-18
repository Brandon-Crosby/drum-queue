#!/usr/bin/env python3
"""add_master.py - give each test beat a mix_master.flac: the same stretch, cut from the session's
real Master bounce (limiter, bus compression, clipping and all), lined up to the sample with mix.flac.

  python eval/add_master.py /Volumes/T7/drumsplit-testset [--only NAME]

The bounce is found in the beat's session folder (roles.json: stems_dir + prefix), among the files
roles.json lists under left_out as 'finished mix' - preferring one named Master. batch_eval.py
--mix master then splits this instead of the summed stems. Its scores against the stems are only
a guide (the master chain changed the drums); listening and the kit test are the real check.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


REMAP = []          # (from, to) path prefixes, for running where the session folders sit elsewhere


def local(path):
    for a, b in REMAP:
        if path.startswith(a):
            return Path(b + path[len(a):])
    return Path(path)


def find_master(spec):
    d, prefix = local(spec['stems_dir']), spec.get('prefix', '')
    names = [k for k, v in spec.get('left_out', {}).items() if 'finished mix' in str(v)]
    names.sort(key=lambda k: (0 if 'master' in k.lower() else 1, k))
    for nm in names:
        for ext in ('.wav', '.flac', '.aif', '.aiff'):
            f = d / f'{prefix}{nm}{ext}'
            if f.exists():
                return f
    for f in sorted(d.glob('*')):                      # nothing listed: any file called ...Master
        if f.suffix.lower() in ('.wav', '.flac', '.aif', '.aiff') and f.stem.lower().endswith('master'):
            return f
    return None


def env(x, sr, hop=64):
    m = np.abs(x.mean(1))
    n = len(m) // hop
    return m[:n * hop].reshape(n, hop).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('testset')
    ap.add_argument('--only', default='')
    ap.add_argument('--search', type=float, default=3.0, help='seconds either way to look for the alignment')
    ap.add_argument('--remap', action='append', default=[], help='FROM=TO path prefix (repeatable)')
    a = ap.parse_args()
    REMAP.extend(tuple(r.split('=', 1)) for r in a.remap)
    for beat in sorted(p for p in Path(a.testset).expanduser().iterdir() if (p / 'testcase.json').exists()):
        if a.only and a.only.lower() not in beat.name.lower():
            continue
        spec = json.loads((beat / 'roles.json').read_text())
        case = json.loads((beat / 'testcase.json').read_text())
        f = find_master(spec)
        if f is None:
            print(f'- {beat.name}: no Master bounce found in {spec.get("stems_dir")}')
            continue
        mix, sr = sf.read(str(beat / 'mix.flac'), dtype='float32', always_2d=True)
        info = sf.info(str(f))
        if info.samplerate != sr:
            print(f'- {beat.name}: {f.name} is {info.samplerate} Hz, stems {sr} Hz - skipped')
            continue
        s0 = int(case['start_s'] * sr)
        pad = int(a.search * sr)
        a0 = max(0, s0 - pad)
        y, _ = sf.read(str(f), start=a0, frames=len(mix) + 2 * pad, dtype='float32', always_2d=True)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, 1)
        y = y[:, :2]
        if len(y) < len(mix):
            print(f'- {beat.name}: {f.name} is too short')
            continue
        # line up by the loudness envelope first (robust to the master's EQ), then to the sample
        hop = 64
        em, ey = env(mix, sr, hop), env(y, sr, hop)
        em, ey = em - em.mean(), ey - ey.mean()
        c = np.correlate(ey, em, mode='valid')
        k = int(np.argmax(c)) * hop
        # a master chain often delays the audio (limiter lookahead, linear-phase EQ: a few ms) and some
        # flip polarity, so search +-15 ms on the waveform and allow either sign
        w = int(0.015 * sr)
        lo, hi = max(0, k - w), min(len(y) - len(mix), k + w)
        ref = mix[sr * 5:sr * 15].mean(1) if len(mix) > sr * 15 else mix.mean(1)
        r0 = sr * 5 if len(mix) > sr * 15 else 0
        best = max(range(lo, hi + 1), key=lambda j: abs(float(y[j + r0:j + r0 + len(ref)].mean(1) @ ref)))
        out = y[best:best + len(mix)]
        if float(out[r0:r0 + len(ref)].mean(1) @ ref) < 0:
            out = -out                                  # polarity flipped in the master chain
        r = float(np.corrcoef(out.mean(1), mix.mean(1))[0, 1])
        sf.write(str(beat / 'mix_master.flac'), np.clip(out, -1, 1), sr, subtype='PCM_24')
        loud = 20 * np.log10(np.sqrt((out ** 2).mean()) / (np.sqrt((mix ** 2).mean()) + 1e-9) + 1e-9)
        flag = '' if r > 0.6 else '   <- low match: is this the same arrangement/version?'
        print(f'+ {beat.name}: {f.name}  offset {(a0 + best - s0) / sr:+.3f} s  match {r:.2f}  '
              f'{loud:+.1f} dB louder than the stem sum{flag}')


if __name__ == '__main__':
    main()
