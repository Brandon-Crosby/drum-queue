"""ab_kits.py - A/B two models' kits for one song, ready to judge by ear.

    python3 eval/ab_kits.py OUT_DIR TRUTH_DIR NAME_A SPLIT_A NAME_B SPLIT_B [NAME_C SPLIT_C ...]

SPLIT_*: a model's stems for the song (kick/snare/perc/hh.wav). TRUTH_DIR: the real tracks
(truth_*.flac + mix.flac). Splits may cover a different stretch of the song than each other or the
truth (a 60 s test clip vs the full song): everything is lined up to the shortest split's timeline.

For every model it writes, under OUT_DIR/<NAME>/:
  - the kit, cut by the live kit maker, with the same sound numbered the same in every model:
    "03 - Snare-Perc 5.wav" (helyx-4) and "03 - Snare-Perc 3.wav" (helyx-6) are the sound that plays
    the same hits in the song. A sound only one model found is "only here - Hat 3.wav".
  - the groove MIDI (as the site makes it).
and in OUT_DIR:
  - "0 real drums + beat.flac": the song itself (the reference),
  - "<n> <NAME> kit + beat.flac": the song with its drums swapped for that kit, each one-shot played
    at its own hits, as hard as the stem hit there; each model's drums matched in loudness to the
    real drums so louder never wins by default,
  - "drums only/<n> <NAME> kit.flac" and "drums only/0 real drums.flac" (FLAC: half the size, same sound).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve, resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'kitmatch'))
import clean  # noqa: E402
import hits as H  # noqa: E402

SR = clean.SR
DRUMS = ('kick', 'snare', 'other_drums', 'hh')
TOL = int(0.03 * SR)          # two hits within 30 ms are the same hit


def _load(f):
    x, r = sf.read(str(f), dtype='float32', always_2d=True)
    return resample_poly(x, SR, r, axis=0).astype(np.float32) if r != SR else x


def _env(x, hop=441):
    m = np.abs(x.mean(1) if x.ndim == 2 else x)
    n = len(m) // hop
    return m[:n * hop].reshape(n, hop).max(1)


def offset(a, b, hop=441):
    """Samples to add to a time on timeline a to land on the same moment in b (envelope xcorr, then fine)."""
    ea, eb = _env(a, hop), _env(b, hop)
    c = fftconvolve(eb, ea[::-1], 'full')
    coarse = (int(np.argmax(c)) - (len(ea) - 1)) * hop
    am = a.mean(1) if a.ndim == 2 else a
    bm = b.mean(1) if b.ndim == 2 else b
    s0 = max(0, -coarse) + SR
    n = min(20 * SR, len(am) - s0 - SR, len(bm) - s0 - coarse - SR)
    if n < SR:
        return coarse
    w = 2 * hop
    seg_b = bm[s0 + coarse - w: s0 + coarse + n + w]
    c = fftconvolve(seg_b, am[s0:s0 + n][::-1], 'valid')
    return coarse - w + int(np.argmax(c))


def cut(split_dir, out_dir):
    """The live kit maker on one split; returns (records, parts with onsets)."""
    got = {}
    orig = clean.write_groove

    def grab(split_dir_, out_dir_, parts, *a, **k):
        got['parts'] = parts
        return orig(split_dir_, out_dir_, parts, *a, **k)
    clean.write_groove = grab
    try:
        ref = HERE.parent / 'kitmatch' / 'soundid_ref.npz'
        recs = clean.write_kit(split_dir, out_dir, ref=ref if ref.exists() else None)
    finally:
        clean.write_groove = orig
    return recs, got.get('parts', [])


def same_hits(a, b):
    """F-score of two onset lists (same timeline): 1 = the very same hits."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    if not len(a) or not len(b):
        return 0.0
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    near = np.minimum(np.abs(b[i] - a), np.abs(b[i - 1] - a)) if len(b) > 1 else np.abs(b[0] - a)
    hit = int((near <= TOL).sum())
    return 2 * hit / (len(a) + len(b))


def _own(label):
    """The stem a one-shot came from, by its name."""
    l = label.lower()
    return ('kick',) if l.startswith('kick') else ('hh',) if 'hat' in l else ('snare', 'perc')


def render(parts, kit_dir, stems, n, shift):
    """Each one-shot at its own hits (same sound chokes itself at its next hit), as hard as ITS OWN stem hit
    there (not every stem: a perc landing on a kick used to play at the kick's level, a blast)."""
    out = np.zeros(n, np.float32)
    for p in parts:
        stems_sum = sum(stems[r] for r in _own(p['label']) if r in stems)
        if not isinstance(stems_sum, np.ndarray):
            continue
        f = kit_dir / f'{p["label"]}.wav'
        if p.get('onsets') is None or not f.exists():
            continue
        shot = _load(f).mean(1)
        on = np.sort(np.asarray(p['onsets']))
        nxt = np.append(on[1:], 10 ** 12)
        for t, t2 in zip(on, nxt):
            pk = float(np.abs(stems_sum[t:t + int(0.03 * SR)]).max()) if t < len(stems_sum) else 0.0
            s = shot[:max(0, t2 - t)].copy()
            fade = min(len(s), int(0.015 * SR))
            if len(s) < len(shot) and fade:
                s[-fade:] *= np.linspace(1, 0, fade)
            u = t + shift
            if u < 0 or u >= n:
                continue
            s = s[:n - u]
            out[u:u + len(s)] += pk * s
    return out


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def main():
    a = sys.argv[1:]
    if len(a) < 6 or len(a) % 2:
        sys.exit(__doc__)
    out, tdir = Path(a[0]), Path(a[1])
    models = list(zip(a[2::2], [Path(p) for p in a[3::2]]))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'drums only').mkdir(exist_ok=True)

    truth = {p: _load(tdir / f'truth_{p}.flac') for p in DRUMS if (tdir / f'truth_{p}.flac').exists()}
    mix = _load(tdir / 'mix.flac')
    m = min([len(mix)] + [len(v) for v in truth.values()])
    real = sum(v[:m] for v in truth.values())
    music = mix[:m] - real

    def stems(sd):
        xs = [H.load(sd / f'{r}.wav') for r in ('kick', 'snare', 'perc', 'hh') if (sd / f'{r}.wav').exists()]
        k = min(len(x) for x in xs)
        return sum(x[:k] for x in xs).mean(1)

    def each(sd):
        return {r: H.load(sd / f'{r}.wav').mean(1) for r in ('kick', 'snare', 'perc', 'hh') if (sd / f'{r}.wav').exists()}

    allst = {sd: stems(sd) for _, sd in models}
    base_sd = min(allst, key=lambda sd: len(allst[sd]))    # the shortest split sets the timeline
    base = allst[base_sd]
    n = len(base)
    t0 = offset(base, real)                       # the base timeline -> truth's
    real_c = np.zeros((n, real.shape[1]), np.float32)
    music_c = np.zeros_like(real_c)
    lo, hi = max(0, t0), min(m, t0 + n)
    real_c[lo - t0:hi - t0], music_c[lo - t0:hi - t0] = real[lo:hi], music[lo:hi]
    files = {}

    tmp = out / '.cut'
    kits = []
    for name, sd in models:
        kd = tmp / name
        if kd.exists():
            shutil.rmtree(kd)
        recs, parts = cut(sd, kd)
        ss = allst[sd]
        shift = 0 if sd == base_sd else -offset(base, ss)   # this split's time -> the base timeline
        parts = [dict(p, onsets=None if p.get('onsets') is None else np.asarray(p['onsets']) + shift) for p in parts]
        kits.append(dict(name=name, dir=kd, parts=parts, stems=each(sd), shift=shift))
        print(f'{name}: {len(parts)} sounds, timeline shift {shift / SR:+.3f} s', flush=True)

    # same sound across models = plays the same hits; numbered in the first model's order
    first = kits[0]['parts']
    num = {(0, p['label']): i + 1 for i, p in enumerate(first)}
    nxt = len(first) + 1
    for k, kit in enumerate(kits[1:], 1):
        P = kit['parts']
        S = np.array([[same_hits(q['onsets'], p['onsets']) if p.get('onsets') is not None and q.get('onsets') is not None
                       else 0 for p in P] for q in first]) if first and P else np.zeros((0, 0))
        if S.size:
            r, c = linear_sum_assignment(-S)
            for i, j in zip(r, c):
                if S[i, j] >= 0.3:
                    num[(k, P[j]['label'])] = i + 1
        for p in P:
            if (k, p['label']) not in num:
                num[(k, p['label'])] = nxt
                nxt += 1
    used = {}
    for k, kit in enumerate(kits):
        for p in kit['parts']:
            used.setdefault(num[(k, p['label'])], set()).add(k)

    shared = sorted(i for i, ks in used.items() if len(ks) > 1)
    label_no = {i: j + 1 for j, i in enumerate(shared)}        # shared sounds numbered 01, 02, ... with no gaps
    for k, kit in enumerate(kits):
        dest = out / kit['name']
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for p in kit['parts']:
            i = num[(k, p['label'])]
            tag = f'{label_no[i]:02d} - ' if i in label_no else 'only here - '
            shutil.copy(kit['dir'] / f'{p["label"]}.wav', dest / f'{tag}{p["label"]}.wav')
        for f in kit['dir'].glob('*.mid'):
            shutil.copy(f, dest / f.name)
        drums = render([dict(p, onsets=p['onsets'] - kit['shift']) for p in kit['parts']],
                       kit['dir'], kit['stems'], n, kit['shift'])
        drums *= rms(real_c) / (rms(drums) + 1e-9)
        d2 = np.repeat(drums[:, None], real_c.shape[1], 1)
        files[f'{k + 1} {kit["name"]} kit + beat.flac'] = d2 + music_c
        files[f'drums only/{k + 1} {kit["name"]} kit.flac'] = d2
    files['0 real drums + beat.flac'] = real_c + music_c
    files['drums only/0 real drums.flac'] = real_c
    g = 10 ** (-1 / 20) / max(float(np.abs(x).max()) for x in files.values())   # one gain for all: levels comparable
    for f, x in files.items():
        sf.write(out / f, x * g, SR, subtype='PCM_24')
    shutil.rmtree(tmp)
    print('pairs:', {i: sorted(kits[k]['name'] for k in ks) for i, ks in sorted(used.items())})


if __name__ == '__main__':
    main()
