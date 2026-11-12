#!/usr/bin/env python3
"""make_testset.py - turn sorted beat folders into a compact test set for eval/batch_eval.py.

  ~/XportStudio/DrumSplit/drum_queue/.venv/bin/python eval/make_testset.py ~/DrumSplits/roles ~/DrumSplits/testset

ROLES is either a folder of <beat>.json roles files from `autosort.py --out` (each says where its stems
are), or a stems root whose beat folders each hold a roles.json.

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
# Whole-kit tracks (a 'DRUMS' print, a drum loop or break) hold kick+snare+hats in one file. Fine in a
# test (the mix is what matters), but in TRAINING they'd teach that kicks and snares are 'perc'.
import re
FULL_KIT = re.compile(r'(^|\W)(drums?|drum ?loops?|drumloop|drum edit|breaks?|main break|dnb loop|jungle loop'
                      r'|garage loop|outro drums|intro drums|beat chop|playero)(\W|$)', re.I)
NOT_FULL_KIT = re.compile(r'log ?drum|steel ?drum|talking ?drum|hand ?drum|frame ?drum', re.I)   # single instruments


def build(beat, out, seconds, spec_file, full=False):
    """60 s test clip (default), or with full=True the whole beat by role, for training
    (no mix.flac: training remixes the roles itself)."""
    spec = json.loads(spec_file.read_text())
    prefix, roles = spec.get('prefix', ''), spec['roles']
    parts, sr = {}, None
    for role in ROLES:
        for name in roles.get(role, []):
            if full and role == 'other_drums' and FULL_KIT.search(name) and not NOT_FULL_KIT.search(name):
                continue                                    # whole-kit track: keep it out of training
            f = next((beat / f'{prefix}{name}{ext}' for ext in ('.wav', '.flac', '.aif', '.aiff')
                      if (beat / f'{prefix}{name}{ext}').exists()), beat / f'{prefix}{name}.wav')
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
            else:                                           # pad the shorter one: never cut a song short
                n = max(len(p), len(y))
                parts[role] = np.pad(p, ((0, n - len(p)), (0, 0))) + np.pad(y, ((0, n - len(y)), (0, 0)))
    if 'kick' not in parts and 'snare' not in parts:
        raise ValueError('no kick or snare tracks listed')
    n = max(len(v) for v in parts.values())
    full = {r: np.pad(parts[r], ((0, n - len(parts[r])), (0, 0))) if r in parts else np.zeros((n, 2), 'float32')
            for r in ROLES}
    # the stretch with the most kit going on (kick + snare + hats), in 1 s steps
    kit = np.abs(full['kick']).mean(1) + np.abs(full['snare']).mean(1) + np.abs(full['hh']).mean(1)
    win = n if full else int(seconds * sr)
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
    if not full:
        sf.write(str(out / 'mix.flac'), mix * g, sr, subtype='PCM_24')
    meta = dict(beat=beat.name, start_s=round(s0 / sr, 1), seconds=round(win / sr, 1), sample_rate=sr,
                gain=round(g, 4), roles=roles)
    (out / ('trainbeat.json' if full else 'testcase.json')).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    shutil.copy(spec_file, out / 'roles.json')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('out')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--only', default='')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--exclude', default='/Volumes/T7/drumsplit-trainset',
                    help='training set folder: beats in it never get a test clip (default: the T7 trainset)')
    a = ap.parse_args()
    root, outroot = Path(a.root).expanduser(), Path(a.out).expanduser()
    if any(root.glob('*.json')) and not (root / 'roles.json').exists():      # a folder of roles files
        jobs = [(Path(json.loads(f.read_text())['stems_dir']), f) for f in sorted(root.glob('*.json'))
                if not f.name.startswith('_')]                     # _overrides.json is not a beat
    elif (root / 'roles.json').exists():
        jobs = [(root, root / 'roles.json')]
    else:
        jobs = [(p, p / 'roles.json') for p in sorted(root.iterdir()) if (p / 'roles.json').exists()]
    ok = 0
    for b, spec_file in jobs:
        if a.only and a.only.lower() not in b.name.lower():
            continue
        out = outroot / b.name
        if a.exclude and (Path(a.exclude).expanduser() / spec_file.stem).exists():
            continue                                         # a training song: never a test clip
        why = json.loads(spec_file.read_text()).get('skip')
        if why:
            print(f'- {b.name}: skipped ({why})')
            continue
        if not a.force and (out / 'roles.json').exists() and (out / 'roles.json').read_text() == spec_file.read_text():
            print(f'= {b.name}: up to date')
            ok += 1
            continue
        try:
            m = build(b, out, a.seconds, spec_file)
            ok += 1
            print(f'+ {b.name}: {m["seconds"]} s from {m["start_s"]} s')
        except Exception as e:
            print(f'! {b.name}: {type(e).__name__}: {e}')
    print(f'\n{ok} beat(s) ready in {outroot}')


if __name__ == '__main__':
    main()
