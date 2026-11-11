#!/usr/bin/env python3
"""make_truth.py - turn a beat's stems export into an answer key for eval/score.py and eval/ab_test.py.

  python3 eval/make_truth.py STEMS_DIR OUT_DIR --roles roles.json [--prefix "BEAT NAME_"]

roles.json maps each role to the track names (file names without .wav, after the prefix):
  {"kick": [...], "snare": [...], "hh": [...], "other_drums": [...], "rest": [...]}
A manifest.json written by an earlier run works too (its "roles" key is used).
Buses (a DRUMS or HIHATS group), the master and silent tracks simply aren't listed.

Writes OUT_DIR/{kick,snare,hh,other_drums,rest}.wav (missing roles as silence) and mix.wav,
all scaled by the same gain so the mix peaks at -1 dBFS. Needs numpy and soundfile.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROLES = ['kick', 'snare', 'hh', 'other_drums', 'rest']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stems')
    ap.add_argument('out')
    ap.add_argument('--roles', required=True)
    ap.add_argument('--prefix', default='')
    a = ap.parse_args()
    spec = json.loads(Path(a.roles).read_text())
    roles = spec.get('roles', spec)
    prefix = a.prefix or spec.get('prefix', '')
    stems, out = Path(a.stems), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summed, sr = {}, None
    for role in ROLES:
        for name in roles.get(role, []):
            f = stems / f'{prefix}{name}.wav'
            if not f.exists():
                raise SystemExit(f'Missing track for {role}: {f}')
            y, r = sf.read(str(f), always_2d=True, dtype='float32')
            if sr and r != sr:
                raise SystemExit(f'{f.name} is {r} Hz, the others are {sr} Hz')
            sr = r
            prev = summed.get(role)
            if prev is None:
                summed[role] = y
            else:
                n = min(len(prev), len(y))
                summed[role] = prev[:n] + y[:n]
    if not summed:
        raise SystemExit('No tracks listed in the roles file')
    n = min(len(v) for v in summed.values())
    ch = max(v.shape[1] for v in summed.values())
    fix = lambda v: np.repeat(v[:n], ch, axis=1) if v.shape[1] < ch else v[:n]
    parts = {r: fix(summed[r]) if r in summed else np.zeros((n, ch), 'float32') for r in ROLES}
    mix = sum(parts.values())
    g = min(1.0, 0.89 / (float(np.abs(mix).max()) + 1e-9))
    for r, v in parts.items():
        sf.write(str(out / f'{r}.wav'), v * g, sr, subtype='FLOAT')
    sf.write(str(out / 'mix.wav'), mix * g, sr, subtype='FLOAT')
    print(f'Wrote {len(ROLES)} role stems + mix.wav to {out}  ({n / sr:.1f} s, {sr} Hz, gain {g:.3f})')


if __name__ == '__main__':
    main()
