#!/usr/bin/env python3
"""soundid.py - say what a cut sound is (kick, snare, clap, rim, cowbell, 808...) by the company it
keeps in a big one-shot library.

Every library sound gets a category from its file and folder names ("[calle] winter [snare 26].wav",
"808 Cowbell 3.wav"). A new sound's fingerprint is compared with the library; the categories of its
closest matches vote, weighted by similarity. Names are noisy and sounds are reused under many names,
so a vote over many neighbours is far steadier than the single nearest match.

  python soundid.py build INDEX_DIR OUT.npz [--per 2500]   labelled, compact reference set
  python soundid.py stats OUT.npz
  python soundid.py test  OUT.npz SPLITS_DIR              true test-beat sounds: does the vote agree?

The reference set holds fingerprints (numbers) and categories only - no audio, no file names.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# first match wins, so the specific drum words come before the generic ones ('808 clap' is a clap,
# '808 cowbell' a cowbell, a plain '808' the bass); filename is tried before the folder names
CATS = [
    ('fx', r'\b(fx|sfx|riser|uplifter|downlifter|impact|sweep|whoosh|transition|reverse|rev|noise|vinyl|scratch|stab)\b'),
    ('vocal', r'\b(vox|vocal|voice|chant|adlib|ad lib|shout|yell|hey)\b'),
    ('cowbell', r'cow ?bell|\bcow\b|\bagogo\b'),
    ('clap', r'\b(clap|claps|clp|handclap|snap|snaps|finger ?snap)\b'),
    ('rim', r'\b(rim|rims|rimshot|rim shot|sidestick|side stick)\b'),
    ('snare', r'\b(snare|snares|snr|sd|snre)\b'),
    ('open hat', r'\b(open ?hat|open ?hh|ohh|oh|o ?hat|opn ?hat|hat open|open)\b'),
    ('closed hat', r'\b(hat|hats|hh|hihat|hi hat|hi-hat|chh|ch|closed ?hat|hat closed)\b'),
    ('cymbal', r'\b(crash|ride|cymbal|cymbals|splash|china|cym)\b'),
    ('shaker', r'\b(shaker|shakers|shkr|maraca|maracas|tamb|tambourine|cabasa)\b'),
    ('tom', r'\b(tom|toms|floor tom|tom tom)\b'),
    ('hand perc', r'\b(conga|congas|bongo|bongos|timbal|timbale|timbales|djembe|tabla|darbuka|guiro|cajon|'
                  r'clave|claves|block|woodblock|wood block|triangle|log ?drums?|logs)\b'),
    ('kick', r'\b(kick|kicks|kik|kck|bd|bassdrum|bass drum)\b'),
    ('808', r'\b(808|808s|sub|subs|reese|bass)\b'),
    ('perc', r'\b(perc|percs|percussion)\b'),
]
NAMES = [c for c, _ in CATS]
RX = [(c, re.compile(p, re.I)) for c, p in CATS]
# what each category is in the kit
KIT = {'kick': 'kick', 'snare': 'snare', 'clap': 'snare', 'rim': 'snare', 'open hat': 'hh', 'closed hat': 'hh',
       'shaker': 'hh', 'cymbal': 'perc', 'tom': 'perc', 'hand perc': 'perc', 'cowbell': 'perc', 'perc': 'perc',
       '808': 'not a drum', 'fx': 'not a drum', 'vocal': 'not a drum'}
LABEL = {'kick': 'Kick', 'snare': 'Snare', 'clap': 'Clap', 'rim': 'Rim', 'open hat': 'Open Hat',
         'closed hat': 'Hat', 'shaker': 'Shaker', 'cymbal': 'Cymbal', 'tom': 'Tom', 'hand perc': 'Perc',
         'cowbell': 'Cowbell', 'perc': 'Perc', '808': '808', 'fx': 'FX', 'vocal': 'Vocal'}


def _words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)                  # SnareRoll -> Snare Roll
    s = re.sub(r'(\d)([A-Za-z])|([A-Za-z])(\d)', lambda m: ' '.join(g for g in m.groups() if g), s)
    return re.sub(r'[_\-\.\[\]\(\)@#,+&]+', ' ', s)


def category(path):
    """Category from the file name, else from the nearest folder that names one; '' if none do."""
    parts = Path(path).with_suffix('').parts[::-1]
    for part in parts[:4]:
        w = _words(part)
        for c, rx in RX:
            if rx.search(w):
                return c
    return ''


def build(index_dir, out, per=2500, seed=0):
    import soundmatch as M
    V, R, P, inv, first = M.load_index(Path(index_dir))
    V, P = V[first], P[first]                                   # one copy of each distinct sound
    cats = np.array([category(p) for p in P])
    rng = np.random.default_rng(seed)
    keep = []
    for c in NAMES:
        idx = np.flatnonzero(cats == c)
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        keep.extend(idx.tolist())
    keep = np.array(sorted(keep))
    np.savez_compressed(out, vecs=V[keep].astype(np.float16), cats=cats[keep])
    print(f'{len(V)} distinct sounds, {np.mean(cats != ""):.0%} named a category; kept {len(keep)} -> {out}')
    stats(out)


def load(ref):
    z = np.load(ref, allow_pickle=True)
    return z['vecs'].astype(np.float32), z['cats']


def stats(ref):
    _, C = load(ref)
    u, n = np.unique(C, return_counts=True)
    print('  ' + '  '.join(f'{a} {b}' for a, b in sorted(zip(u, n), key=lambda t: -t[1])))


def identify(vec, V, C, k=25, power=8):
    """-> (category, share of the vote, best similarity) for one fingerprint."""
    if vec is None:
        return '', 0.0, 0.0
    s = V @ vec.astype(np.float32)
    top = np.argpartition(-s, k)[:k]
    w = np.clip(s[top], 0, None) ** power                     # close matches count far more
    votes = {}
    for c, wi in zip(C[top], w):
        votes[c] = votes.get(c, 0.0) + wi
    c = max(votes, key=votes.get)
    return c, votes[c] / (sum(votes.values()) + 1e-12), float(s[top].max())


def name_sound(vec, V, C, k=50, power=16, sure=0.8, drop_sure=0.9):
    """What to call a cut sound, only as far as the library vote is sure.
    -> (kit piece or '' if unsure, label or '' if unsure, drop: bool)
    Tested on the 17 test beats' real sounds: at 80% agreement the kit piece is right 86% of the time."""
    if vec is None:
        return '', '', False
    s = V @ vec.astype(np.float32)
    top = np.argpartition(-s, k)[:k]
    w = np.clip(s[top], 0, None) ** power
    tot = w.sum() + 1e-12
    pieces, cats = {}, {}
    for c, wi in zip(C[top], w):
        pieces[KIT[c]] = pieces.get(KIT[c], 0.0) + wi
        cats[c] = cats.get(c, 0.0) + wi
    piece = max(pieces, key=pieces.get)
    cat = max(cats, key=cats.get)
    if piece == 'not a drum':
        return '', '', pieces[piece] / tot >= drop_sure
    if pieces[piece] / tot < sure:
        return '', '', False
    return piece, (LABEL[cat] if cats[cat] / tot >= sure else ''), False


def test(ref, splits):
    """The cut sounds of each test beat's real stems: does the vote land in the right kit piece?"""
    import hits as H
    import clean as C_
    V, C = load(ref)
    truth = {'kick': 'kick', 'snare': 'snare', 'hh': 'hh', 'other_drums': 'perc'}
    tot, ok, conf = 0, 0, {}
    for b in sorted(Path(splits).iterdir()):
        for t, piece in truth.items():
            f = b / f'truth_{t}.flac'
            if not f.exists():
                continue
            x, on, lab, snd = H.sounds(f)
            for s in snd[:4]:
                v = C_.fp(x[s['start']:s['end']])
                c, share, best = identify(v, V, C)
                got = KIT.get(c, '?')
                tot += s['hits']
                ok += s['hits'] * (got == piece)
                conf.setdefault(piece, {}).setdefault(got, 0)
                conf[piece][got] += s['hits']
    print(f'right kit piece for {ok / max(tot, 1):.0%} of hits')
    for p, d in conf.items():
        n = sum(d.values())
        print(f'  real {p:5}: ' + '  '.join(f'{g} {v / n:.0%}' for g, v in sorted(d.items(), key=lambda t: -t[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'stats', 'test'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--per', type=int, default=2500)
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.args[0], a.args[1], a.per)
    elif a.cmd == 'stats':
        stats(a.args[0])
    else:
        test(a.args[0], a.args[1])


if __name__ == '__main__':
    main()
