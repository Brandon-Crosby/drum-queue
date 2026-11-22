"""junk.py - how clean each hit of a separated drum stem is, the way a producer hears it.

At every hit of the real (answer-key) track, our stem is lined up with it (to the sample, best gain),
the real hit is subtracted, and what's left over is measured in dB under the real hit: overall, in the
body/knock range (200 Hz - 2 kHz) and the top (2 - 8 kHz). More negative = cleaner. The median over hits
is the score.

Why: SDR and the fingerprint used for naming hide exactly what makes a one-shot sound lo-fi (quiet bleed,
grit and top-end junk). On breakdown136 / church's chicken / hyperbole this measure agreed with the
producer's ears where the fingerprint didn't (Nov 2026).

    python eval/junk.py truth_kick.wav ours_kick.wav
"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

BANDS = {'all': None, 'body': (200, 2000), 'top': (2000, 8000)}


def _mono(x):
    return x.mean(1) if x.ndim == 2 else x


def onsets(x, sr, rise_db=12.0, floor_db=-30.0, gap_s=0.05):
    """Hits of an isolated track: 5 ms RMS steps that jump rise_db within 10 ms, above floor_db of the max."""
    b = int(0.005 * sr)
    k = len(x) // b
    e = 20 * np.log10(np.sqrt((x[:k * b] ** 2).reshape(k, b).mean(1)) + 1e-9)
    top = e.max()
    out, last = [], -10 ** 9
    for i in range(2, k):
        if e[i] > top + floor_db and e[i] - min(e[i - 2], e[i - 1]) >= rise_db and (i - last) * b >= gap_s * sr:
            out.append(i * b - b)
            last = i
    return np.array(out, int)


def hit_junk(truth, ours, sr, win_s=0.15, search_s=0.005, max_hits=200, sr_ours=None):
    """Median leftover (dB under the real hit) per band over the real track's hits. Ours is resampled to the
    real track's rate first if they differ (sessions are 44.1 or 48 kHz; the site works at 44.1)."""
    t, o = _mono(truth).astype(np.float64), _mono(ours).astype(np.float64)
    if sr_ours and sr_ours != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g_ = gcd(int(sr), int(sr_ours))
        o = resample_poly(o, int(sr) // g_, int(sr_ours) // g_)
    n = min(len(t), len(o))
    t, o = t[:n], o[:n]
    on = onsets(t, sr)
    if len(on) == 0:
        return None
    if len(on) > max_hits:                                    # evenly spaced subset keeps it quick
        on = on[np.linspace(0, len(on) - 1, max_hits).astype(int)]
    filt = {k: (sosfiltfilt(butter(4, v, 'band', fs=sr, output='sos'), np.stack([t, o]), axis=1) if v else np.stack([t, o]))
            for k, v in BANDS.items()}
    w, s = int(win_s * sr), int(search_s * sr)
    res = {k: [] for k in BANDS}
    for i, a in enumerate(on):
        end = min(a + w, on[i + 1] if i + 1 < len(on) else n, n)
        if end - a < int(0.03 * sr) or a - s < 0 or end + s > n:
            continue
        r = t[a:end]
        if np.sum(r ** 2) < 1e-10:
            continue
        c = np.correlate(o[a - s:end + s], r, 'valid')            # line ours up with the real hit
        sh = int(np.argmax(c)) - s
        oo = o[a + sh:end + sh]
        g = np.dot(oo, r) / (np.dot(oo, oo) + 1e-12)
        for k in BANDS:
            rt, ro = filt[k][0][a:end], filt[k][1][a + sh:end + sh]
            den = np.sum(rt ** 2)
            if den < 1e-12:
                continue
            res[k].append(10 * np.log10(np.sum((rt - g * ro) ** 2) / den + 1e-12))
    return {k: (round(float(np.median(v)), 1) if v else None) for k, v in res.items()} | {'hits': len(res['all'])}


if __name__ == '__main__':
    tr, sr = sf.read(sys.argv[1], dtype='float32')
    ou, sr2 = sf.read(sys.argv[2], dtype='float32')
    print(hit_junk(tr, ou, sr, sr_ours=sr2))


def routing(truth, sr, stems, partner=None, win_s=0.06, stack_s=0.015, max_hits=300):
    """Where each real hit of one piece ends up among our output stems.

    truth: the real track of the piece (e.g. snare). stems: {name: (audio, sr)} of our outputs. partner: the
    real track of the piece it gets confused with (the kick for a snare): a hit within stack_s of a partner hit
    counts as 'stacked', otherwise 'alone'. For each hit, each stem's share is the energy of its part that
    matches the real hit (projection onto it), so a stem that only has other sounds there gets ~0.
    -> {'alone': {stem: share}, 'stacked': {...}, 'hits': {'alone': n, 'stacked': n}}"""
    from math import gcd
    from scipy.signal import resample_poly
    t = _mono(truth).astype(np.float64)
    ys = {}
    for k, (y, s) in stems.items():
        m = _mono(y).astype(np.float64)
        if s != sr:
            g_ = gcd(int(sr), int(s))
            m = resample_poly(m, int(sr) // g_, int(s) // g_)
        ys[k] = m
    on = onsets(t, sr)
    if len(on) > max_hits:
        on = on[np.linspace(0, len(on) - 1, max_hits).astype(int)]
    pon = onsets(_mono(partner).astype(np.float64), sr) if partner is not None else np.array([], int)
    w, st = int(win_s * sr), int(stack_s * sr)
    acc = {'alone': {k: 0.0 for k in ys}, 'stacked': {k: 0.0 for k in ys}}
    n = {'alone': 0, 'stacked': 0}
    for a in on:
        r = t[a:a + w]
        rr = float(np.dot(r, r))
        if len(r) < w or rr < 1e-10:
            continue
        kind = 'stacked' if len(pon) and np.abs(pon - a).min() <= st else 'alone'
        e = {k: max(0.0, float(np.dot(r, y[a:a + w]))) ** 2 / rr if len(y) >= a + w else 0.0 for k, y in ys.items()}
        tot = sum(e.values())
        if tot <= 0:
            continue
        n[kind] += 1
        for k in ys:
            acc[kind][k] += e[k] / tot
    return {kind: {k: (round(v / n[kind], 3) if n[kind] else None) for k, v in acc[kind].items()} for kind in acc} | {'hits': n}
