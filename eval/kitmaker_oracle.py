"""kitmaker_oracle.py - test the kit maker alone, on perfect stems with a known answer.

    python3 eval/kitmaker_oracle.py RENDERS_DIR GROOVES_DIR [--json out]

RENDERS_DIR: flp_render.py output (<song> (FLP)/kick,snare,hh,other_drums.flac: each drum on its own
track, exactly). GROOVES_DIR: its <song>.json answer keys (every hit: time, role, sample). The kit maker
runs on those stems as if the model were perfect, so every miss here is the kit maker's own:
  - per one-shot: which real sample it is (purity = share of its hits that are that sample) and how
    many of its hits are real hits at all;
  - per real sample with 8+ hits: did the kit get a one-shot of it (found) - or two (duplicate);
  - per role: the kit's hits vs the real hits within 25 ms (recall / precision of the groove).
"""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ab_kits as A  # noqa: E402

SR = A.SR
TOL = int(0.025 * SR)
GROUPS = {'kick': ('kick',), 'sp': ('snare', 'other_drums'), 'hh': ('hh',)}
STEM = {'kick': 'kick', 'snare': 'snare', 'perc': 'other_drums', 'hh': 'hh'}


def group_of(label):
    l = label.lower()
    return 'kick' if l.startswith('kick') else 'hh' if 'hat' in l else 'sp'


def near(a, b):
    """For each time in a: index of the nearest time in b within TOL, else -1."""
    b = np.asarray(b)
    if not len(b):
        return np.full(len(a), -1)
    o = np.argsort(b)
    bs = b[o]
    i = np.clip(np.searchsorted(bs, a), 1, len(bs) - 1) if len(bs) > 1 else np.zeros(len(a), int)
    out = []
    for t, k in zip(a, i):
        cand = [j for j in (k - 1, k) if 0 <= j < len(bs)]
        j = min(cand, key=lambda j: abs(bs[j] - t))
        out.append(o[j] if abs(bs[j] - t) <= TOL else -1)
    return np.array(out)


def one(render, key):
    hits = [h for h in json.load(open(key))['hits'] if h['role'] in ('kick', 'snare', 'other_drums', 'hh')]
    with tempfile.TemporaryDirectory() as td:
        sd = Path(td) / 'split'
        sd.mkdir()
        for mine, theirs in STEM.items():
            f = render / f'{theirs}.flac'
            if f.exists():
                x, sr = sf.read(f, dtype='float32', always_2d=True)
                if np.abs(x).max() > 1e-4:
                    sf.write(sd / f'{mine}.wav', x, sr)
        recs, parts = A.cut(sd, Path(td) / 'kit')
    res = dict(shots=[], sounds=[], groove={})
    for g, roles in GROUPS.items():
        T = [h for h in hits if h['role'] in roles]
        tt = np.array([round(h['t'] * SR) for h in T])
        ps = [p for p in parts if group_of(p['label']) == g and p.get('onsets') is not None]
        allon = np.sort(np.concatenate([np.asarray(p['onsets']) for p in ps])) if ps else np.array([])
        if len(tt):
            m = near(allon, tt)
            got = set(m[m >= 0])
            res['groove'][g] = dict(real=len(tt), kit=len(allon), recall=round(len(got) / len(tt), 3),
                                    precision=round(float((m >= 0).mean()), 3) if len(allon) else None)
        dom = {}
        for p in ps:
            on = np.asarray(p['onsets'])
            m = near(on, tt)
            names = Counter(T[j]['sound'] for j in m if j >= 0)
            top, n = names.most_common(1)[0] if names else (None, 0)
            res['shots'].append(dict(label=p['label'], hits=len(on), real=round(float((m >= 0).mean()), 3),
                                     sample=top, purity=round(n / len(on), 3)))
            if top:
                dom.setdefault(top, []).append(p['label'])
        for s, n in Counter(h['sound'] for h in T).items():
            if n >= 8:
                res['sounds'].append(dict(sample=s, group=g, hits=n, kit=dom.get(s, [])))
    return res


def main():
    rd, gd = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    for r in sorted(rd.glob('* (FLP)')):
        key = gd / (r.name[:-len(' (FLP)')] + '.json')
        if not key.exists():
            continue
        try:
            out[r.name[:-6]] = one(r, key)
        except Exception as e:
            print(r.name, 'failed:', e)
            continue
        o = out[r.name[:-6]]
        f = sum(1 for s in o['sounds'] if s['kit'])
        print(f"{r.name[:-6][:36]:36} found {f}/{len(o['sounds'])} samples | one-shots {len(o['shots'])}, "
              f"pure {sum(s['purity'] >= 0.8 for s in o['shots'])} | groove recall "
              + ' '.join(f"{g} {v['recall']:.2f}" for g, v in o['groove'].items()), flush=True)
    if '--json' in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)


if __name__ == '__main__':
    main()
