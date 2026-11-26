#!/usr/bin/env python3
"""hits.py - turn a drum stem into the kit's sounds: find every hit, group the repeats, pick the
cleanest copy of each.

  sounds STEM.flac OUT_DIR             write one .wav per distinct sound (+ sounds.json)
  eval   TRAIN_DIR GROOVE_DIR [--n 20] check against rebuilt projects, whose grooves say which
                                       channel played every hit

How:
- onsets: spectral flux (frame-to-frame rise in log-spectrum energy), peaks over a local median,
  at least 25 ms apart, quieter than 45 dB under the stem's loudest frame ignored
- each hit is described by its first ~46 ms (64-band log-mel, level-normalized, 30 dB floor):
  short enough that the next hit rarely overlaps it, long enough to tell a clap from a snare
- grouping: loudest hits first; a hit joins the closest group if similar enough, else starts one;
  then every hit is reassigned to its closest group; tiny groups (stray ghost notes) are dropped
- the sound kept for a group: its copy with the longest clean tail before the next hit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
N_FFT, HOP = 2048, 256
FEAT_SR, FEAT_FFT, FEAT_HOP, N_MELS, FEAT_FRAMES = 22050, 1024, 256, 64, 4


def _mel_fb(sr, n_fft, n_mels, fmin=30):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


FB_ON = _mel_fb(SR, N_FFT, 96)
FB_FEAT = _mel_fb(FEAT_SR, FEAT_FFT, N_MELS)


def load(path):
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if sr != SR:
        t = np.arange(0, x.shape[0] * SR / sr) * sr / SR
        x = np.stack([np.interp(t, np.arange(x.shape[0]), c) for c in x.T], 1).astype(np.float32)
    return x                                                    # [n, ch]


def onsets(x, min_gap=0.025, floor_db=45, delta=0.6, min_rise_db=0.0):
    mono = x.mean(1)
    pad = np.pad(mono, (N_FFT // 2, N_FFT // 2))
    # Construct only the selected frames. The full intermediate view can exceed
    # 32-bit NumPy's index limit even though the selected frames fit comfortably.
    fr = np.lib.stride_tricks.as_strided(
        pad, shape=(1 + (len(pad) - N_FFT) // HOP, N_FFT),
        strides=(pad.strides[0] * HOP, pad.strides[0]), writeable=False)
    mag = np.abs(np.fft.rfft(fr * np.hanning(N_FFT), axis=1))
    mel = mag ** 2 @ FB_ON.T
    energy = 10 * np.log10(mel.sum(1) + 1e-12)
    lm = np.log1p(1000 * mel)
    flux = np.maximum(0, np.diff(lm, axis=0, prepend=np.zeros((1, lm.shape[1])))).sum(1)   # silence before the file
    k = 16                                                      # ~90 ms on each side
    padf = np.pad(flux, (k, k), mode='edge')
    med = np.median(np.lib.stride_tricks.sliding_window_view(padf, 2 * k + 1), axis=1)
    thr = med + delta * flux.std() + 1e-9
    loud = energy > energy.max() - floor_db
    peaks, last = [], -10 ** 9
    gap = int(min_gap * SR / HOP)
    for i in range(1, len(flux) - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and (loud[i] or loud[min(i + 2, len(loud) - 1)]):
            if i - last < gap:
                if flux[i] > flux[last]:
                    peaks[-1] = i
                    last = i
                continue
            peaks.append(i)
            last = i
    # a frame's window is centered on frame*HOP, so its flux peak can sit up to half a window before or
    # after the hit: look there for the sharpest rise of a ~1.5 ms envelope, and call that the start
    out = []
    w = 64
    gmax = np.abs(mono).max() + 1e-12
    for p in peaks:
        a = max(0, p * HOP - N_FFT // 2 - HOP)
        b = min(len(mono), p * HOP + N_FFT // 2 + HOP)
        seg = np.abs(mono[a:b])
        if len(seg) <= 2 * w:
            out.append(p * HOP)
            continue
        env = np.sqrt(np.convolve(seg ** 2, np.ones(w) / w, mode='valid'))
        rise = env[w:] - env[:-w]
        i = int(np.argmax(rise))
        lo = env[i]                                              # level just before the rise
        hi = env[i + w]
        j = i
        while j < i + w and env[j] < lo + 0.1 * (hi - lo):      # first point the rise is under way
            j += 1
        peak = env[i:i + 6 * w].max()
        pre = env[max(0, i - 4 * w):i + 1].max()                 # what was already sounding just before
        if rise[i] <= 0 or (peak < pre * 10 ** (min_rise_db / 20) and pre > gmax * 1e-3):
            continue                                             # a fading hit changing tone, not a new hit
        out.append(a + j)
    return np.array(sorted(set(out)), dtype=np.int64)


def feature(x, i):
    """The hit's first ~46 ms as a level-normalized log-mel picture."""
    mono = x[i:i + int(0.08 * SR)].mean(1)
    n = int(len(mono) * FEAT_SR / SR)
    if n < FEAT_FFT:
        mono = np.pad(mono, (0, int(FEAT_FFT * SR / FEAT_SR) + 8 - len(mono)))
        n = int(len(mono) * FEAT_SR / SR)
    y = np.interp(np.arange(n) * SR / FEAT_SR, np.arange(len(mono)), mono)
    fr = np.lib.stride_tricks.sliding_window_view(y, FEAT_FFT)[::FEAT_HOP][:FEAT_FRAMES]
    if len(fr) < FEAT_FRAMES:
        fr = np.concatenate([fr, np.zeros((FEAT_FRAMES - len(fr), FEAT_FFT))])
    mel = np.log10((np.abs(np.fft.rfft(fr * np.hanning(FEAT_FFT), axis=1)) ** 2) @ FB_FEAT.T + 1e-10)
    mel = np.maximum(mel, mel.max() - 3)
    v = (mel - mel.mean()).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


DECAY_W = 0.6     # how much ringing length counts next to the attack's tone


def decay(x, on, nxt):
    """How long each hit rings: level at 40-60, 60-90 and 90-140 ms after the start, against the
    attack, in dB (open hat vs closed hat). Where the next hit comes first, the part it covers counts
    as 'unknown' (0, i.e. no pull either way)."""
    mono = np.abs(x.mean(1))
    out = np.zeros((len(on), 3))
    for k, (i, j) in enumerate(zip(on, nxt)):
        att = mono[i:i + int(0.02 * SR)].max() + 1e-9
        for c, (a, b) in enumerate(((0.04, 0.06), (0.06, 0.09), (0.09, 0.14))):
            if i + int(b * SR) <= j:
                lvl = mono[i + int(a * SR):i + int(b * SR)].max()
                out[k, c] = (np.clip(20 * np.log10(lvl / att + 1e-9), -40, 0) + 20) / 20   # -1 short .. +1 long
    return out * DECAY_W / np.sqrt(3)


def merge_same(x, on, nxt, lab, level, thr=0.97):
    """Groups that are one sound split by overlaps or velocity: compare each group's cleanest
    copy over its first 200 ms and merge near-identical ones."""
    g_n = int(lab.max()) + 1 if (lab >= 0).any() else 0
    if g_n < 2:
        return lab
    reps = []
    for g in range(g_n):
        idx = np.flatnonzero(lab == g)
        best = idx[int(np.argmax((nxt[idx] - on[idx]) * level[idx] ** 0.5))]
        seg = x[on[best]:min(nxt[best], on[best] + int(0.2 * SR))].mean(1)
        seg = np.pad(seg, (0, max(0, int(0.2 * SR) - len(seg))))
        mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        v = np.log10(mag[:4000] + 1e-6)
        v = v - v.mean()
        reps.append((v / (np.linalg.norm(v) + 1e-9), (nxt[best] - on[best]) / SR))
    parent = list(range(g_n))
    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for a in range(g_n):
        for b in range(a + 1, g_n):
            sim = float(reps[a][0] @ reps[b][0])
            if sim >= thr and min(reps[a][1], reps[b][1]) >= 0.06:   # only judge copies long enough to compare
                parent[root(b)] = root(a)
    new = np.array([root(g) if g >= 0 else -1 for g in lab])
    ids = {g: k for k, g in enumerate(sorted(set(new[new >= 0]), key=lambda g: -np.sum(new == g)))}
    return np.array([ids.get(g, -1) for g in new])


def group(F, level, thr=0.9, min_share=0.01):
    """Greedy grouping, loudest first, then one reassignment pass. Returns a group id per hit (-1 = dropped)."""
    order = np.argsort(-level)
    cents, members = [], []
    for i in order:
        if cents:
            sims = np.array(cents) @ F[i]
            j = int(np.argmax(sims))
            if sims[j] >= thr:
                members[j].append(i)
                c = F[members[j]].mean(0)
                cents[j] = c / (np.linalg.norm(c) + 1e-9)
                continue
        cents.append(F[i].copy())
        members.append([i])
    C = np.array(cents)
    lab = np.argmax(F @ C.T, axis=1)
    sim = (F @ C.T)[np.arange(len(F)), lab]
    lab[sim < thr - 0.1] = -1                                   # fits no group: a stray
    counts = np.bincount(lab[lab >= 0], minlength=len(C))
    keep = counts >= max(2, int(min_share * len(F)))
    lab = np.where((lab >= 0) & keep[np.maximum(lab, 0)], lab, -1)
    ids = {g: k for k, g in enumerate(sorted(set(lab[lab >= 0]), key=lambda g: -counts[g]))}
    return np.array([ids.get(g, -1) for g in lab])


def peak_at(x, on, a=0.0, b=0.03):
    """Loudest sample from a seconds before to b seconds after each onset."""
    mono = np.abs(x).max(1)
    return np.array([mono[max(0, i - int(a * SR)):i + int(b * SR)].max(initial=0) for i in on])


def drop_bleed(x, on, others, bleed_db=12.0, alike=0.85):
    """A separated stem carries faint copies of the other drums. A 'hit' here that lines up with a hit
    at least bleed_db louder in another stem AND sounds like it (feature cosine >= alike) is that other
    drum leaking through: drop it. A different sound layered on a louder one (a clap on a kick) stays."""
    if not others or len(on) == 0:
        return on
    here = peak_at(x, on)
    lv = np.stack([peak_at(o, on, a=0.01, b=0.03) for o in others])        # [stems, hits]
    keep = []
    for j, i in enumerate(on):
        k = int(np.argmax(lv[:, j]))
        if here[j] >= lv[k, j] * 10 ** (-bleed_db / 20):
            keep.append(i)
            continue
        a, b = feature(x, i), feature(others[k], i)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if cos < alike:
            keep.append(i)
    return np.array(keep, dtype=np.int64)


def sounds(stem, max_len=1.5, others=None, bleed_db=12.0):
    """others: the song's other drum stems (paths or arrays) - pass them for a separated song so
    bleed from them isn't cut into fake sounds."""
    x = load(stem) if not isinstance(stem, np.ndarray) else stem
    on = onsets(x)
    if others:
        ox = [load(o) if not isinstance(o, np.ndarray) else o for o in others]
        n = len(x)
        ox = [np.pad(o, ((0, max(0, n - len(o))), (0, 0)))[:n] for o in ox]
        on = drop_bleed(x, on, ox, bleed_db)
    if len(on) == 0:
        return x, on, np.array([], int), []
    nxt = np.append(on[1:], len(x))
    F = np.stack([feature(x, i) for i in on])
    level = np.array([np.abs(x[i:i + int(0.03 * SR)]).max() for i in on])
    F = np.concatenate([F, decay(x, on, nxt)], 1)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    lab = group(F, level)
    lab = merge_same(x, on, nxt, lab, level)
    out = []
    for g in range(lab.max() + 1 if (lab >= 0).any() else 0):
        idx = np.flatnonzero(lab == g)
        tail = (nxt[idx] - on[idx]) / SR
        score = np.minimum(tail, max_len) * (level[idx] / level[idx].max()) ** 0.5
        best = idx[int(np.argmax(score))]
        end = min(nxt[best], on[best] + int(max_len * SR))
        out.append(dict(group=g, hits=len(idx), start=int(on[best]), end=int(end), tail_s=round(float((end - on[best]) / SR), 3)))
    return x, on, lab, out


def write_sounds(stem, out_dir, role='sound'):
    x, on, lab, out = sounds(stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(out):
        seg = x[s['start']:s['end']].copy()
        fd = min(len(seg), int(0.01 * SR))
        seg[len(seg) - fd:] *= np.linspace(1, 0, fd)[:, None]
        name = f'{role} {k + 1}.wav'
        sf.write(str(out_dir / name), seg, SR, subtype='PCM_24')
        s['file'] = name
    (out_dir / 'sounds.json').write_text(json.dumps(dict(onsets=len(on), sounds=out), indent=1))
    return out


# ---------- checking against rebuilt projects
def evaluate(train_dir, groove_dir, n=20, roles=('kick', 'snare', 'hh', 'other_drums'), tol=0.02):
    projs = sorted(p for p in train_dir.iterdir() if p.is_dir() and (groove_dir / (p.name.replace(' (FLP)', '') + '.json')).exists())
    projs = projs[:n]
    tot = {r: dict(tp=0, det=0, true=0, pure=0, lab=0, found=0, sounds=0, groups=0) for r in roles}
    for p in projs:
        hits = json.loads((groove_dir / (p.name.replace(' (FLP)', '') + '.json')).read_text())['hits']
        for r in roles:
            if not (p / f'{r}.flac').exists():
                continue
            times = {}
            for h in hits:
                if h['role'] == r:
                    times.setdefault(round(h['t'], 3), set()).add(h['sound'])
            if not times:
                continue
            tt = np.array(sorted(times))
            tl = ['+'.join(sorted(times[t])) for t in tt]
            x, on, lab, out = sounds(p / f'{r}.flac')
            ts = on / SR
            a = tot[r]
            a['true'] += len(tt); a['det'] += len(ts)
            # match detected hits to true ones
            j = np.searchsorted(tt, ts)
            truth = []
            for k, t in enumerate(ts):
                c = [q for q in (j[k] - 1, j[k]) if 0 <= q < len(tt)]
                q = min(c, key=lambda q: abs(tt[q] - t)) if c else None
                truth.append(tl[q] if q is not None and abs(tt[q] - t) <= tol else None)
            matched = {truth[k] for k in range(len(ts)) if truth[k]}
            a['tp'] += sum(1 for t in truth if t)
            # groups vs true sounds (a true sound = a channel, or channels always stacked together)
            cnt = {}
            for l in tl:
                if '+' not in l:                                 # moments where two channels stack aren't a sound of their own
                    cnt[l] = cnt.get(l, 0) + 1
            real = {l for l, c in cnt.items() if c >= 2}
            a['sounds'] += len(real)
            a['groups'] += int(lab.max() + 1) if (lab >= 0).any() else 0
            maj = set()
            for g in range(int(lab.max()) + 1 if (lab >= 0).any() else 0):
                ls = [truth[k] for k in np.flatnonzero(lab == g) if truth[k]]
                if ls:
                    top = max(set(ls), key=ls.count)
                    a['pure'] += ls.count(top); a['lab'] += len(ls)
                    maj.add(top)
            a['found'] += len(real & maj)
        print(f'done {p.name}', flush=True)
    print(f'\n{len(projs)} rebuilt projects')
    print(f"{'role':12} {'hits found':>11} {'false hits':>11} {'group purity':>13} {'sounds found':>13} {'groups/sounds':>14}")
    for r, a in tot.items():
        if a['true'] == 0:
            continue
        rec = a['tp'] / a['true']; fp = 1 - a['tp'] / max(a['det'], 1)
        pur = a['pure'] / max(a['lab'], 1); found = a['found'] / max(a['sounds'], 1)
        print(f"{r:12} {100 * rec:10.1f}% {100 * fp:10.1f}% {100 * pur:12.1f}% {100 * found:12.1f}% {a['groups']:6d}/{a['sounds']:<6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sounds', 'eval'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    if a.cmd == 'sounds':
        for s in write_sounds(Path(a.args[0]), Path(a.args[1]), Path(a.args[0]).stem):
            print(s)
    else:
        evaluate(Path(a.args[0]), Path(a.args[1]), a.n)


if __name__ == '__main__':
    main()
