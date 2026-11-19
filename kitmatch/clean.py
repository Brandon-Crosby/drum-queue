#!/usr/bin/env python3
"""clean.py - get one clean copy of each sound out of a separated (bleedy) stem.

A separated stem's single hits carry bleed from the other drums, and it differs from hit to hit.
The sound itself is the same sample every time. So:

  average   line up every hit of a sound (to the sample) and average them. The sound adds up, the
            bleed mostly doesn't: N hits clean it up by roughly 10*log10(N) dB. Each point of the
            average only uses hits that haven't been cut off by their next hit yet.
  merge     bleed splits one sound into several groups; once averaged, groups that are really the
            same sound look alike again, so they are merged and averaged together.
  unstack   a kick stem often holds a snare or clap stacked on some kicks. The kick plays alone
            elsewhere, so its averaged lone copy is the kick; subtract it (best-fit level) from
            the averaged stacked copy and what's left is the stacked sound.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hits as H            # noqa: E402
import soundmatch as M      # noqa: E402

SR = H.SR
MAX_LEN = 1.5
SHIFT = int(0.004 * SR)          # how far a detected onset may be off from the true start


def _mono(x):
    return x.mean(1) if x.ndim == 2 else x


def align(x, on, ref, n=int(0.04 * SR)):
    """Move each onset by up to SHIFT samples so its first 40 ms lines up best with ref."""
    m = _mono(x)
    r = _mono(ref)[:n]
    out = []
    for i in on:
        a = max(0, i - SHIFT)
        seg = m[a:i + SHIFT + n]
        if len(seg) < n + 1:
            out.append(i)
            continue
        c = np.correlate(seg, r, mode='valid')
        out.append(a + int(np.argmax(c)))
    return np.array(out)


def average(x, on, nxt, max_len=MAX_LEN, ref=None):
    """Sample-aligned average of the hits at `on`; each hit only counts up to its next hit."""
    L = int(min(max_len * SR, max(nxt - on)))
    if ref is None:
        lv = np.array([np.abs(x[i:i + int(0.03 * SR)]).max() for i in on])
        b = int(np.argmax(lv))
        ref = x[on[b]:on[b] + int(0.04 * SR)]
    on = align(x, on, ref)
    acc = np.zeros((L, x.shape[1]))
    cnt = np.zeros(L)
    for i, j in zip(on, nxt):
        k = int(min(L, j - i, len(x) - i))
        if k <= 0:
            continue
        acc[:k] += x[i:i + k]
        cnt[:k] += 1
    # only where a fair share of the hits still count (a tail made of 2 hits is mostly whatever came next)
    keep = cnt >= max(min(3, len(on)), 0.25 * len(on))
    k = int(np.flatnonzero(keep)[-1]) + 1 if keep.any() else 1
    avg = acc[:k] / cnt[:k, None]
    k = tail_end(avg)
    avg = avg[:k].copy()
    fd = min(k, int(0.02 * SR))
    avg[k - fd:] *= np.linspace(1, 0, fd)[:, None] ** 2
    return avg.astype(np.float32), int(cnt.max())


def tail_end(seg, blk=int(0.005 * SR), rise_db=8.0):
    """Where a one-shot really ends: follow its decay from the peak and stop at the quietest point
    before the level climbs again by rise_db (the next hit or bleed creeping in)."""
    m = np.abs(seg).max(1) if seg.ndim == 2 else np.abs(seg)
    n = len(m) // blk
    if n < 3:
        return len(seg)
    db = 20 * np.log10(m[:n * blk].reshape(n, blk).max(1) + 1e-9)
    i = int(np.argmax(db))
    lo, lo_i = db[i], i
    for j in range(i + 1, n):
        if db[j] < lo:
            lo, lo_i = db[j], j
        elif db[j] > lo + rise_db:
            return (lo_i + 1) * blk
    return len(seg)


def fp(seg):
    mono = _mono(seg)
    n = int(len(mono) * M.SR / SR)
    y = np.interp(np.arange(n) * SR / M.SR, np.arange(len(mono)), mono).astype(np.float32)
    return M.fingerprint(y)


def sounds(stem, others=None, merge_thr=0.95, min_hits=2):
    """Like hits.sounds, but each sound is the average of all its hits, and look-alike groups merged.
    Returns x, list of dict(seg, hits, onsets)."""
    x, on, lab, snd = H.sounds(stem, others=others)
    if len(on) == 0:
        return x, []
    nxt = np.append(on[1:], len(x))
    groups = [np.flatnonzero(lab == g) for g in range(lab.max() + 1)] if (lab >= 0).any() else []
    groups = [g for g in groups if len(g) >= min_hits]
    # average, then merge groups whose averages are the same sound, until nothing merges
    while True:
        avgs = [average(x, on[g], nxt[g])[0] for g in groups]
        vs = [fp(a) for a in avgs]
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if vs[a] is not None and vs[b] is not None and float(vs[a] @ vs[b]) >= merge_thr:
                    groups[a] = np.sort(np.concatenate([groups[a], groups[b]]))
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    out = [dict(seg=a, hits=len(g), onsets=on[g]) for a, g in zip(avgs, groups)]
    out.sort(key=lambda s: -s['hits'])
    return x, out


def brightness(seg):
    m = _mono(seg)[:int(0.1 * SR)]
    X = np.abs(np.fft.rfft(m))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return float((X * f).sum() / (X.sum() + 1e-12))


def unstack(kick_sounds, min_left_db=-20.0):
    """kick_sounds: output of sounds() on a kick stem. The darkest sound with plenty of hits is the
    lone kick. For every other sound, subtract the lone kick (aligned, best-fit level); if what's
    left is loud enough, it's a stacked sound. Returns [(residual_seg, hits, from_index)]."""
    if len(kick_sounds) < 2:
        return []
    big = [s for s in kick_sounds if s['hits'] >= max(3, 0.1 * kick_sounds[0]['hits'])] or kick_sounds
    lone = min(big, key=lambda s: brightness(s['seg']))
    K = lone['seg']
    out = []
    for n, s in enumerate(kick_sounds):
        if s is lone:
            continue
        S = s['seg']
        L = min(len(S), len(K))
        k = _mono(K[:L])
        best = None
        for d in range(-SHIFT, SHIFT + 1, 4):                     # fine alignment
            kk = np.roll(k, d)
            if d > 0:
                kk[:d] = 0
            elif d < 0:
                kk[d:] = 0
            a = float(_mono(S[:L]) @ kk / (kk @ kk + 1e-12))
            r = _mono(S[:L]) - a * kk
            e = float(r @ r)
            if best is None or e < best[0]:
                best = (e, d, a)
        e, d, a = best
        kk = np.roll(K[:L], d, axis=0)
        res = S[:L] - a * kk
        left_db = 10 * np.log10(e / (float(_mono(S[:L]) @ _mono(S[:L])) + 1e-12) + 1e-12)
        if left_db >= min_left_db and a > 0.2:
            out.append((res.astype(np.float32), s['hits'], n, round(left_db, 1)))
    return out


# ---------- a whole kit from a split song
def kit_sounds(stems, find, dedupe=0.95, min_left_db=-12.0):
    """stems: folder; find(role) -> stem path or None, roles kick/snare/perc/hh/other.
    Kicks from the kick stem. Snares, claps and percussion from the snare and perc stems added
    together (the model often files a snare as percussion and vice versa, so neither stem alone is
    trustworthy), plus whatever was stacked on kicks (subtracted out of the kick stem). Hats from
    the hat stem. Every sound is the average of all its hits. Returns [dict(role, seg, hits)]."""
    import soundfile as sf  # noqa: F401
    get = {r: find(r) for r in ('kick', 'snare', 'perc', 'hh', 'other')}
    loaded = {r: H.load(p) for r, p in get.items() if p}
    n = min(len(v) for v in loaded.values())
    loaded = {r: v[:n] for r, v in loaded.items()}
    out = []
    if 'kick' in loaded:
        oth = [loaded[r] for r in loaded if r != 'kick']
        _, ks = sounds(loaded['kick'], others=oth)
        if ks:
            stacked = unstack(ks, min_left_db=min_left_db)
            stacked_idx = {i for _, _, i, _ in stacked}
            for i, s in enumerate(ks):
                if i not in stacked_idx:
                    out.append(dict(role='kick', seg=s['seg'], hits=s['hits']))
            for res, h, i, _ in stacked:
                out.append(dict(role='snare', seg=res, hits=h, source='stacked on kick'))
    sp = [loaded[r] for r in ('snare', 'perc') if r in loaded]
    if sp:
        mix = sum(sp)
        oth = [loaded[r] for r in ('kick', 'hh') if r in loaded]
        _, ss = sounds(mix, others=oth)
        for s in ss:
            # file it by where most of its first 50 ms of energy sits
            e = {r: float((loaded[r][s['onsets'][0]:s['onsets'][0] + int(0.05 * SR)] ** 2).sum())
                 for r in ('snare', 'perc') if r in loaded}
            out.append(dict(role=max(e, key=e.get) if e else 'snare', seg=s['seg'], hits=s['hits']))
    if 'hh' in loaded:
        oth = [loaded[r] for r in loaded if r != 'hh']
        _, hs = sounds(loaded['hh'], others=oth)
        out += [dict(role='hh', seg=s['seg'], hits=s['hits']) for s in hs]
    # drop near-duplicates across the pools (same sound found twice): keep the one with more hits
    vs = [fp(s['seg']) for s in out]
    keep = []
    for i in sorted(range(len(out)), key=lambda i: -out[i]['hits']):
        if vs[i] is None:
            continue
        if any(float(vs[i] @ vs[j]) >= dedupe for j in keep):
            continue
        keep.append(i)
    return [dict(out[i], fp=vs[i]) for i in keep]
