#!/usr/bin/env python3
"""score.py - how good is a drum split? Compares separated stems against the real multitrack.

  python3 eval/score.py TRUTH_DIR EST_DIR [--json out.json] [--label baseline]

TRUTH_DIR holds the answer key, one WAV per role:
    kick.wav snare.wav hh.wav [toms.wav ride.wav crash.wav] other_drums.wav rest.wav
EST_DIR holds what the splitter produced (Drum Queue / site output):
    kick.wav snare.wav toms.wav hh.wav ride.wav crash.wav [drums.wav bass.wav ...]

Three views, all printed:
  1. SDR per kit piece (dB, higher is better) - the number separation leaderboards use.
  2. Leakage: for each output stem, what share of its energy came from each real source.
  3. Hits: for every real kick/snare hit, did it land in the right stem, the wrong one, or both.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

SR = 44100
KIT = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash']
TRUTH_ORDER = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash', 'other_drums', 'rest']


def load(path, sr=SR):
    y, r = sf.read(str(path), dtype='float32', always_2d=True)
    y = y.mean(1)                                   # scoring in mono keeps it simple and fair
    if r != sr:
        from math import gcd
        g = gcd(r, sr)
        y = resample_poly(y, sr // g, r // g).astype('float32')
    return y


def align(est, ref, max_lag=4096):
    """Model outputs can be shifted by a few samples; find and remove the lag."""
    n = min(len(est), len(ref), SR * 60)
    a, b = est[:n], ref[:n]
    F = np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n))
    c = np.fft.irfft(F)
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]])
    lag = int(np.argmax(c)) - max_lag
    return lag


def shift(x, lag, n):
    out = np.zeros(n, dtype='float32')
    if lag >= 0:
        seg = x[lag:lag + n]
    else:
        seg = np.concatenate([np.zeros(-lag, dtype='float32'), x[:n + lag]])
    out[:len(seg)] = seg[:n]
    return out


def sdr(ref, est):
    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2)
    return float(10 * np.log10((num + 1e-9) / (den + 1e-9)))


def db(x):
    return float(10 * np.log10(x + 1e-12))


def leakage(est, truths, band=8, block_s=1.0, ridge=1e-3):
    """Share of an output stem's energy that comes from each real source.
    The output is modelled, in each frequency band and ~1 s block, as a weighted mix of the real
    sources (complex least squares). Each source's share is the energy of its weighted part;
    whatever no source explains is reported as 'artifacts' (model noise, smearing)."""
    keys = list(truths)
    _, _, E = stft(est, SR, nperseg=2048, noverlap=1536)
    S = np.stack([stft(truths[k], SR, nperseg=2048, noverlap=1536)[2] for k in keys])   # (K, F, T)
    K, F, T = S.shape
    bt = max(1, int(block_s * SR / 512))
    part = np.zeros(K)
    resid = 0.0
    for f0 in range(0, F, band):
        for t0 in range(0, T, bt):
            A = S[:, f0:f0 + band, t0:t0 + bt].reshape(K, -1).T         # (N, K)
            b = E[f0:f0 + band, t0:t0 + bt].reshape(-1)
            if not np.any(b):
                continue
            G = A.conj().T @ A
            G += ridge * np.trace(G).real / K * np.eye(K) + 1e-12 * np.eye(K)
            w = np.linalg.solve(G, A.conj().T @ b)
            for j in range(K):
                part[j] += np.sum(np.abs(A[:, j] * w[j]) ** 2)
            resid += np.sum(np.abs(b - A @ w) ** 2)
    total = float(np.sum(np.abs(E) ** 2))
    shares = {k: part[j] / (part.sum() + resid + 1e-12) for j, k in enumerate(keys)}
    shares['artifacts'] = resid / (part.sum() + resid + 1e-12)
    return shares, total


def onsets(x, min_gap=0.09, rel=0.12):
    """Hit times (s) from a clean single-instrument track: energy-envelope peaks."""
    hop = 256
    n = len(x) // hop
    env = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    flux = np.maximum(0, np.diff(env, prepend=0))
    thr = rel * flux.max()
    hits, last = [], -1e9
    for i in range(1, n - 1):
        if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            t = i * hop / SR
            if t - last >= min_gap:
                hits.append(t)
                last = t
    return np.array(hits)


def hit_report(truths, ests, piece, other, win=0.06, coincide=0.03):
    """For each real `piece` hit that is not stacked on an `other` hit, measure where its
    energy went: the right stem, the `other` stem, both, or nowhere."""
    t_hits = onsets(truths[piece])
    o_hits = onsets(truths[other])
    res = {'right': 0, 'wrong': 0, 'both': 0, 'missing': 0, 'stacked_skipped': 0}
    w = int(win * SR)
    for t in t_hits:
        if len(o_hits) and np.min(np.abs(o_hits - t)) < coincide:
            res['stacked_skipped'] += 1
            continue
        i = int(t * SR)
        ref = np.sum(truths[piece][i:i + w] ** 2) + 1e-12
        in_right = np.sum(ests[piece][i:i + w] ** 2) / ref
        # what's in the other stem beyond that instrument's own sound (e.g. a long 808 tail)
        own = np.sum(truths[other][i:i + w] ** 2)
        in_wrong = max(0.0, np.sum(ests[other][i:i + w] ** 2) - own) / ref
        r_ok, w_ok = in_right > 0.1, in_wrong > 0.1                   # >~ -10 dB of the hit
        res['both' if r_ok and w_ok else 'right' if r_ok else 'wrong' if w_ok else 'missing'] += 1
    res['total'] = len(t_hits)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('truth')
    ap.add_argument('est')
    ap.add_argument('--json')
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    tdir, edir = Path(a.truth), Path(a.est)
    truths = {k: load(tdir / f'{k}.wav') for k in TRUTH_ORDER if (tdir / f'{k}.wav').exists()}
    n = min(len(v) for v in truths.values())
    truths = {k: v[:n] for k, v in truths.items()}
    raw = {k: load(edir / f'{k}.wav') for k in KIT if (edir / f'{k}.wav').exists()}
    if not raw:
        sys.exit(f'No kit stems found in {edir}')

    # one lag for the whole split, measured on the summed drums
    drum_truth = sum(truths[k] for k in truths if k not in ('rest',))
    lag = align(sum(raw.values())[:n], drum_truth)
    ests = {k: shift(v, lag, n) for k, v in raw.items()}
    for k in KIT:                                   # missing outputs count as silence
        ests.setdefault(k, np.zeros(n, dtype='float32'))
        truths.setdefault(k, np.zeros(n, dtype='float32'))

    mix_drums = db(np.sum(drum_truth ** 2))
    report = {'label': a.label, 'lag_samples': lag, 'sdr': {}, 'silent_piece_leak_db': {}, 'leakage': {}, 'hits': {}}

    print(f"\n=== {a.label or edir.name} ===   (alignment lag {lag} samples)\n")
    print('1. Score per kit piece (SDR, dB; higher is better)')
    for k in KIT:
        if np.sum(truths[k] ** 2) > 1e-6:
            v = sdr(truths[k], ests[k])
            report['sdr'][k] = round(v, 2)
            print(f'   {k:6} {v:6.2f}')
        else:
            leak = db(np.sum(ests[k] ** 2)) - mix_drums
            report['silent_piece_leak_db'][k] = round(leak, 1)
            print(f'   {k:6}   --   (not in this beat; output is {leak:.1f} dB vs. the drums)')

    srcs = [k for k in TRUTH_ORDER if k in truths and np.sum(truths[k] ** 2) > 1e-6]
    print('\n2. Where each output stem\'s sound came from (% of its energy)')
    cols = srcs + ['artifacts']
    print('   ' + ' ' * 8 + ''.join(f'{s[:9]:>10}' for s in cols) + '   level')
    for k in KIT:
        share, _ = leakage(ests[k], {s: truths[s] for s in srcs})
        level = db(np.sum(ests[k] ** 2)) - mix_drums           # same scale as mix_drums
        report['leakage'][k] = {s: round(share[s] * 100, 1) for s in cols}
        report['leakage'][k]['_level_db'] = round(level, 1)
        if level < -60:
            print(f'   {k:8}' + '  (silent)')
            continue
        print(f'   {k:8}' + ''.join(f'{share[s] * 100:9.1f}%' for s in cols) + f'  {level:5.1f} dB')

    print('\n3. Hit by hit (hits that are stacked on top of each other are left out)')
    for piece, other in (('kick', 'snare'), ('snare', 'kick')):
        r = hit_report(truths, ests, piece, other)
        report['hits'][piece] = r
        judged = r['total'] - r['stacked_skipped']
        print(f"   real {piece} hits: {judged} judged -> {r['right']} only in {piece}, {r['both']} in both, "
              f"{r['wrong']} only in {other}, {r['missing']} missing  ({r['stacked_skipped']} stacked, skipped)")

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    main()
