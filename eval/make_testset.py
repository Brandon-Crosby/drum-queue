#!/usr/bin/env python3
"""make_testset.py - turn sorted beat folders into a compact test set for eval/batch_eval.py.

  ~/XportStudio/DrumSplit/drum_queue/.venv/bin/python eval/make_testset.py STEMS_ROOT ~/DrumSplits/testset

For every beat folder under STEMS_ROOT that has a roles.json (made by eval/autosort.py, then
reviewed), it sums the tracks by role, picks the SECONDS-long stretch with the most drums going on,
and writes OUT/<beat>/{mix,kick,snare,hh,other_drums,rest}.flac plus a copy of roles.json.
FLAC at 24-bit keeps it small (~10 MB a file) so the whole set uploads to Modal quickly.
Already-built beats are skipped unless their roles.json changed (or --force).
Needs numpy + soundfile (Drum Queue's Python has both).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

ROLES = ['kick', 'snare', 'hh', 'other_drums', 'rest']


def build(beat, out, seconds):
    spec = json.loads((beat / 'roles.json').read_text())
    prefix, roles = spec.get('prefix', ''), spec['roles']
    parts, sr = {}, None
    for role in ROLES:
        for name in roles.get(role, []):
            f = beat / f'{prefix}{name}.wav'
            if not f.exists():
                raise FileNotFoundError(f.name)
            y, r = sf.read(str(f), always_2d=True, dtype='float32')
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            y = y[:, :2]
            if sr and r != sr:
                raise ValueError(f'{f.name} is {r} Hz, others {sr} Hz')
            sr = r
            p = parts.get(role)
            if p is None:
                parts[role] = y
            else:
                n = min(len(p), len(y))
                parts[role] = p[:n] + y[:n]
    if 'kick' not in parts and 'snare' not in parts:
        raise ValueError('no kick or snare tracks listed')
    n = min(len(v) for v in parts.values())
    full = {r: parts[r][:n] if r in parts else np.zeros((n, 2), 'float32') for r in ROLES}
    # the stretch with the most kit going on (kick + snare + hats), in 1 s steps
    kit = np.abs(full['kick']).mean(1) + np.abs(full['snare']).mean(1) + np.abs(full['hh']).mean(1)
    win = int(seconds * sr)
    if n > win:
        c = np.concatenate([[0], np.cumsum(kit)])
        starts = np.arange(0, n - win, sr)
        s0 = int(starts[np.argmax(c[starts + win] - c[starts])])
    else:
        s0, win = 0, n
    clip = {r: v[s0:s0 + win] for r, v in full.items()}
    mix = sum(clip.values())
    g = min(1.0, 0.89 / (float(np.abs(mix).max()) + 1e-9))
    out.mkdir(parents=True, exist_ok=True)
    for r, v in clip.items():
        sf.write(str(out / f'{r}.flac'), v * g, sr, subtype='PCM_24')
    sf.write(str(out / 'mix.flac'), mix * g, sr, subtype='PCM_24')
    meta = dict(beat=beat.name, start_s=round(s0 / sr, 1), seconds=round(win / sr, 1), sample_rate=sr,
                gain=round(g, 4), roles=roles)
    (out / 'testcase.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    shutil.copy(beat / 'roles.json', out / 'roles.json')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('out')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--only', default='')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    root, outroot = Path(a.root).expanduser(), Path(a.out).expanduser()
    beats = [root] if (root / 'roles.json').exists() else sorted(p for p in root.iterdir() if (p / 'roles.json').exists())
    ok = 0
    for b in beats:
        if a.only and a.only.lower() not in b.name.lower():
            continue
        out = outroot / b.name
        if not a.force and (out / 'roles.json').exists() and (out / 'roles.json').read_text() == (b / 'roles.json').read_text():
            print(f'= {b.name}: up to date')
            ok += 1
            continue
        try:
            m = build(b, out, a.seconds)
            ok += 1
            print(f'+ {b.name}: {m["seconds"]} s from {m["start_s"]} s')
        except Exception as e:
            print(f'! {b.name}: {type(e).__name__}: {e}')
    print(f'\n{ok} beat(s) ready in {outroot}')


if __name__ == '__main__':
    main()
