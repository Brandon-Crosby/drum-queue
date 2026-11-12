#!/usr/bin/env python3
"""autosort.py - guess each track's role in a folder of beat stem exports, for the test set.

  python3 eval/autosort.py STEMS_ROOT --out ~/DrumSplits/roles      # every beat folder under it
  python3 eval/autosort.py STEMS_ROOT --out ~/DrumSplits/roles --only "la intencion"

For each beat folder it writes OUT/<beat folder name>.json (eval/make_testset.py reads it) and prints
a table to review. Nothing is written into the stems folders (they may be shared, e.g. on Dropbox).
Without --out it writes roles.json inside each beat folder instead.

How it decides, in order:
  1. Master / Current / mixdown         -> left out (that's the finished mix)
  2. Silent tracks                      -> left out
  3. A track that is (nearly) the sum of other drum tracks (a DRUMS or HIHATS bus) -> left out
  4. Track name words                   -> kick / snare / hh / other_drums / rest
  5. No drum word                       -> rest (bass, synths, vocals, FX...)
Names win over sound; anything doubtful is marked "check" in the table so a person looks at it.
A roles.json that has "reviewed": true is left alone (so corrections aren't overwritten).
"""
import argparse
import sys
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

WORDS = [  # (role, regex on the lowercase track name); first match wins
    ('mix', r'\b(master|current|mixdown|full ?mix|bounce)\b'),
    ('kick', r'kick|kik|\bkck\b|\bbd\b|bombo'),
    ('snare', r'snare|sn\w?are|snr|clap|\bclp\b|\brims?\b|rimz|rimshot|\bsd\b'),
    ('hh', r'hi ?-?hats?|(open|closed) ?hats?|\bhats?\b|\bhh\b|\boh\b|shaker|maraca|tamb(ourine)?\b|tambo\b|top loop|\btops?\b'),
    ('other_drums', r'perc|prc\b|palito|cong[ao]|bongo|campana|cowbell|guiro|g[uü]iro|timbal|clave|tambora|\btoms?\b'
                    r'|crash|\brides?\b|cymbal|log drum|dnb loop|jungle loop|\bamen\b|\bbreaks?\b|garage loop'
                    r'|drum ?loop|drums?\b|\bloop.*(perc|drum)|dembow loop|break'),
]
DRUM_ROLES = ('kick', 'snare', 'hh', 'other_drums')


def track_name(f, prefix):
    s = f.stem
    return s[len(prefix):] if prefix and s.startswith(prefix) else s


def common_prefix(names):
    if not names:
        return ''
    p = names[0]
    for n in names[1:]:
        while not n.startswith(p):
            p = p[:-1]
    # cut back to the last separator so we don't eat part of a track name
    m = max(p.rfind('_'), p.rfind(' - '))
    return p[:m + 1] if m >= 0 else ''


def by_name(name):
    # '2_RIM', 'drum_rubberslover_hats', '909crash': make separators spaces so word matching works
    low = re.sub(r'[_\-=+()\[\]{}|.,#@!]+', ' ', name.lower())
    low = re.sub(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)', ' ', low)
    for role, rx in WORDS:
        if re.search(rx, low):
            return role
    return 'rest'


def load_mono(f, max_sec=None):
    info = sf.info(str(f))
    frames = int(max_sec * info.samplerate) if max_sec else -1
    y, sr = sf.read(str(f), frames=frames, always_2d=True, dtype='float32')
    return y.mean(axis=1), sr


def sort_beat(folder, dest=None):
    wavs = sorted(p for p in folder.glob('*.wav') if not p.name.startswith('._'))
    if len(wavs) < 3:
        return None
    prefix = common_prefix([p.stem for p in wavs])
    rows, audio = [], {}
    stems = [p.stem.strip() for p in wavs]
    for f in wavs:
        name = track_name(f, prefix)
        # a file whose whole name is what every other file starts with = the beat's own mixdown
        if any(o != f.stem.strip() and o.startswith(f.stem.strip()) for o in stems) and \
                sum(o.startswith(f.stem.strip()) for o in stems) > len(stems) * 0.8:
            name = name or f.stem
            rows.append(dict(name=name, file=f.name, db=0.0, guess='mix'))
            audio[name] = load_mono(f)[0]
            rows[-1]['db'] = round(20 * np.log10(float(np.sqrt(np.mean(audio[name] ** 2))) + 1e-12), 1)
            continue
        y, sr = load_mono(f)
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        db = 20 * np.log10(rms + 1e-12)
        rows.append(dict(name=name, file=f.name, db=round(db, 1), guess=by_name(name)))
        audio[name] = y
    n = min(len(a) for a in audio.values())
    masters = [r['name'] for r in rows if r['guess'] == 'mix']
    for r in rows:
        if r['guess'] != 'mix' and masters:
            m, t = audio[masters[0]][:n:16], audio[r['name']][:n:16]
            c = float(np.dot(m, t) / (np.sqrt(np.dot(m, m) * np.dot(t, t)) + 1e-12))
            mdb = next(x['db'] for x in rows if x['name'] == masters[0])
            if c > 0.98 and abs(r['db'] - mdb) < 1.0:     # same shape AND same level as the master
                r['guess'] = 'mix'
    for r in rows:
        r['role'], r['why'] = r['guess'], 'name'
        if r['guess'] == 'mix':
            r['role'], r['why'] = 'left_out', 'finished mix'
        elif r['db'] < -70:
            r['role'], r['why'] = 'left_out', 'silent'
    # Buses: a drum track that is a SUM of other drum tracks (a DRUMS or HIHATS group). Non-negative
    # least squares only allows adding tracks, so "kick = DRUMS - snares" can't fit, and several buses
    # side by side don't confuse it. Louder (wider) tracks are checked first and removed once found.
    from scipy.optimize import nnls
    hop = 64                                        # decimate for speed; a bus matches at any rate
    drum = sorted((r for r in rows if r['role'] in DRUM_ROLES), key=lambda r: -r['db'])
    for r in drum:
        others = [o for o in drum if o is not r and o['role'] in DRUM_ROLES]
        if len(others) < 2:
            continue
        X = np.stack([audio[o['name']][:n:hop] for o in others], 1).astype(np.float64)
        t = audio[r['name']][:n:hop].astype(np.float64)
        w, _ = nnls(X, t, maxiter=2000)
        resid = 10 * np.log10(np.sum((t - X @ w) ** 2) / (np.sum(t ** 2) + 1e-12) + 1e-12)
        if resid < -7 and int(np.sum(w > 0.3)) >= 2:   # -7 dB: a bus with compression isn't an exact sum
            r['role'], r['why'] = 'left_out', f'bus of other drum tracks ({resid:.0f} dB)'
    # flag doubtful ones for a human: drum word but sounds like bass/music, or loud unnamed tracks
    for r in rows:
        low = r['name'].lower()
        r['check'] = ((r['role'] == 'rest' and re.search(r'insert|audio|track|untitled|sample|^hit\b|^fx\b', low) is not None)
                      or (r['role'] == 'other_drums' and re.search(r'^(intro )?drums?\b|drums$', low) is not None))
    roles = {k: [r['name'] for r in rows if r['role'] == k] for k in ('kick', 'snare', 'hh', 'other_drums', 'rest')}
    spec = dict(prefix=prefix, roles=roles, reviewed=False,
                left_out={r['name']: r['why'] for r in rows if r['role'] == 'left_out'},
                check=[r['name'] for r in rows if r['check']],
                seconds=round(n / sr, 1), sample_rate=sr)
    spec['stems_dir'] = str(folder)
    if dest:
        dest.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    return spec, rows


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--only', default='')
    ap.add_argument('--out', default='', help='folder for the roles files (default: inside each beat folder)')
    ap.add_argument('--testset', default='', help='also build each beat\'s 60 s test clip here right away, so its '
                                                  'stems can go back to online-only afterwards')
    ap.add_argument('--trainset', default='', help='also save each beat full-length by role here, for training. '
                                                   'Beats already in --holdout (the test set) are never saved here.')
    ap.add_argument('--holdout', default='', help='the test set folder: beats in it stay out of the training set')
    ap.add_argument('--force', action='store_true', help='redo folders already marked reviewed')
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    folders = [root] if any(root.glob('*.wav')) else sorted(p for p in root.iterdir() if p.is_dir())
    done = 0
    for d in folders:
        if a.only and a.only.lower() not in d.name.lower():
            continue
        outdir = Path(a.out).expanduser() if a.out else None
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
        rj = (outdir / f'{d.name}.json') if outdir else (d / 'roles.json')
        if rj.exists() and not a.force and json.loads(rj.read_text()).get('reviewed'):
            print(f'= {d.name}: reviewed already, skipped')
            continue
        try:
            out = sort_beat(d, rj)
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            if 'Format not recognised' in msg or 'Error opening' in msg:
                msg = 'stems not readable - in Dropbox, right-click the folder > Make available offline, then rerun'
            print(f'! {d.name}: {msg}')
            continue
        if not out:
            print(f'- {d.name}: fewer than 3 WAVs, skipped')
            continue
        spec, rows = out
        done += 1
        if a.trainset:
            import make_testset
            held = Path(a.holdout).expanduser() if a.holdout else None
            if held and (held / d.name).exists():
                print(f'  -> {d.name} is in the test set: NOT saved for training')
            elif json.loads(rj.read_text()).get('skip'):
                print('  -> skipped for training (marked skip)')
            else:
                try:
                    m = make_testset.build(d, Path(a.trainset).expanduser() / d.name, 0, rj, full=True)
                    print(f'  -> training copy saved ({m["seconds"]} s by role)')
                except Exception as e:
                    print(f'  -> training copy failed: {type(e).__name__}: {e}')
        if a.testset:
            import make_testset
            try:
                m = make_testset.build(d, Path(a.testset).expanduser() / d.name, 60, rj)
                print(f'  -> test clip saved ({m["seconds"]} s from {m["start_s"]} s)')
            except Exception as e:
                print(f'  -> test clip failed: {type(e).__name__}: {e}')
        print(f'\n## {d.name}  ({spec["seconds"]} s)')
        for r in sorted(rows, key=lambda r: (r['role'], r['name'])):
            flag = '  <- check' if r['check'] else ''
            print(f"  {r['role']:12} {r['db']:7.1f} dB  {r['name']}" + (f"   ({r['why']})" if r['why'] != 'name' else '') + flag)
    print(f'\nWrote roles for {done} beat(s) to {outdir or "the beat folders"}.')


if __name__ == '__main__':
    main()
