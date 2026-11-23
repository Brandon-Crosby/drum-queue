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
    # line up on the attack's edges, not its body: the first difference weights the click and
    # crack, which pin timing to the sample (a kick's low thump is too smooth to time exactly,
    # and averaging hits that are a few samples apart smears the attack away)
    m = np.diff(_mono(x), prepend=0.0)
    r = np.diff(_mono(ref), prepend=0.0)[:n]
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
    marked = [s for s in kick_sounds if s.get('is_kick')]
    big = [s for s in kick_sounds if s['hits'] >= max(3, 0.1 * kick_sounds[0]['hits'])] or kick_sounds
    lone = marked[0] if marked else min(big, key=lambda s: brightness(s['seg']))
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
        if left_db >= min_left_db:
            # a > 0.2: a kick with something on top -> keep what's left after taking the kick out.
            # otherwise there's hardly any kick in it at all: a dark snare or perc the model filed as
            # kick (it plays where no kick is) -> keep it whole.
            out.append(((res if a > 0.2 else S[:L]).astype(np.float32), s['hits'], n, round(left_db, 1)))
    return out


# ---------- a whole kit from a split song
def kit_sounds(stems, find, dedupe=0.95, min_left_db=-12.0, log=lambda line: None):
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
        ks, _ = clean_kicks(loaded['kick'], ks, log=log)
        if ks:
            stacked = unstack(ks, min_left_db=min_left_db)
            stacked_idx = {i for _, _, i, _ in stacked}
            for i, s in enumerate(ks):
                if i not in stacked_idx:
                    out.append(dict(role='kick', seg=s['seg'], hits=s['hits'], onsets=s['onsets'],
                                    shot=s.get('shot')))
            for res, h, i, _ in stacked:
                out.append(dict(role='snare', seg=res, hits=h, source='stacked on kick', onsets=ks[i]['onsets']))
    sp = [loaded[r] for r in ('snare', 'perc') if r in loaded]
    if sp:
        mix = sum(sp)
        oth = [loaded[r] for r in ('kick', 'hh') if r in loaded]
        _, ss = sounds(mix, others=oth)
        for s in ss:
            # file it by where most of its first 50 ms of energy sits
            e = {r: float((loaded[r][s['onsets'][0]:s['onsets'][0] + int(0.05 * SR)] ** 2).sum())
                 for r in ('snare', 'perc') if r in loaded}
            out.append(dict(role=max(e, key=e.get) if e else 'snare', seg=s['seg'], hits=s['hits'],
                            onsets=s['onsets']))
    if 'hh' in loaded:
        oth = [loaded[r] for r in loaded if r != 'hh']
        _, hs = sounds(loaded['hh'], others=oth)
        out += [dict(role='hh', seg=s['seg'], hits=s['hits'], onsets=s['onsets']) for s in hs]
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


# ---------- the website's one-shot kit
# Snare and perc share one label: the model often files a dembow snare as perc (and the other way
# round), so which stem a sound came out of doesn't say which it is. Ordered by how often it plays.
LABELS = {'kick': 'Kick', 'sp': 'Snare-Perc', 'hh': 'Hat'}
MAX_SOUNDS = {'kick': 3, 'sp': 6, 'hh': 3}
GROUP = {'kick': 'kick', 'snare': 'sp', 'perc': 'sp', 'hh': 'hh'}


def _spec(seg, lo=60, hi=3000):
    m = (seg.mean(1) if seg.ndim == 2 else seg)[:int(0.15 * SR)]
    X = np.abs(np.fft.rfft(m, 8192))
    f = np.fft.rfftfreq(8192, 1 / SR)
    X = X[(f > lo) & (f < hi)]
    return X / (np.linalg.norm(X) + 1e-9)


def _sub_share(seg, cut=150):
    m = seg.mean(1) if seg.ndim == 2 else seg
    X = np.abs(np.fft.rfft(m[:int(0.15 * SR)], 16384)) ** 2
    f = np.fft.rfftfreq(16384, 1 / SR)
    return float(X[f < cut].sum() / (X.sum() + 1e-12))


def _drop_impostors(ks, split_dir, log, bass_like=0.5, kick_like=0.9, smear_sub=0.1, smear_out=3.0):
    """Snare/perc sounds that are really the bass (distorted bass the first stage left in the drums)
    or really the kick (kick leaking into the perc stem). Bass: the sound's spectrum matches what the
    bass stem plays at the same moments. Kick: its fingerprint matches the main kick sound."""
    bass_f = Path(split_dir) / 'bass.wav'
    bass = H.load(bass_f) if bass_f.exists() else None
    kicks = [k for k in ks if k['role'] == 'kick']
    main_kick = max(kicks, key=lambda k: k['hits'])['fp'] if kicks else None
    keep = []
    for s in ks:
        if s['role'] in ('snare', 'perc'):
            if _sub_share(s['seg']) > 0.85:          # nearly all sub: a piece of 808/bass, not a drum
                log(f'dropped a {s["role"]} sound heard {s["hits"]}x: it is almost all sub-bass (808)')
                continue
            if _sub_share(s['seg']) > smear_sub and _outside_core_db(s['seg']) > smear_out:
                # sub under it AND more energy outside the drum range than in it: 808 + hats landing
                # together, not a drum (stunyun137, Nov 2026: two "Snare-Perc" one-shots were this)
                log(f'dropped a {s["role"]} sound heard {s["hits"]}x: it is 808 and hats smeared together')
                continue
            if main_kick is not None and float(s['fp'] @ main_kick) >= kick_like:
                log(f'dropped a {s["role"]} sound heard {s["hits"]}x: it is the kick leaking in')
                continue
            on = s.get('onsets')
            if bass is not None and on is not None and len(on):
                L = int(0.15 * SR)
                b = [bass[i:i + L] for i in on if i + L <= len(bass)]
                if b and float(_spec(s['seg']) @ _spec(np.mean(b, 0))) >= bass_like:
                    log(f'dropped a {s["role"]} sound heard {s["hits"]}x: it is the bass leaking in')
                    continue
        keep.append(s)
    return keep


def _outside_core_db(seg, lo=100, hi=5000):
    """Energy outside the 100 Hz - 5 kHz core, relative to the core: a hat on top, or bass under it."""
    m = seg.mean(1) if seg.ndim == 2 else seg
    X = np.abs(np.fft.rfft(m[:int(0.15 * SR)], 8192)) ** 2
    f = np.fft.rfftfreq(8192, 1 / SR)
    core = X[(f >= lo) & (f <= hi)].sum()
    return float(10 * np.log10(X[(f < lo) | (f > hi)].sum() / (core + 1e-12) + 1e-12))


# ---------- releases cut short by the next hit

def extend_tonal(seg, sr=SR, fit_ms=60, max_peaks=8, floor_db=-60.0, max_len=1.5):
    """Continue a tonal one-shot past where it was cut: fit the strongest partials of the last stretch
    (frequency, amplitude, phase, decay) and let them ring out. Returns (new seg, ms added, info)."""
    from scipy.signal import find_peaks as _find_peaks
    x = seg.mean(1) if seg.ndim == 2 else seg
    n = int(fit_ms / 1000 * sr)
    if len(x) < 3 * n:
        return seg, 0, 'too short'
    pk = np.abs(x).max()
    # decay over the clean tail: RMS of two halves of the last 2n samples
    a1 = np.sqrt(np.mean(x[-2 * n:-n] ** 2)) + 1e-12
    a2 = np.sqrt(np.mean(x[-n:] ** 2)) + 1e-12
    rate = np.log(a2 / a1) / (n / sr)                    # per second, negative when dying
    if rate >= -1.0:
        return seg, 0, f'not decaying ({rate:.1f}/s)'
    lvl = 20 * np.log10(a2 / pk)
    if lvl < floor_db + 6:
        return seg, 0, f'already quiet ({lvl:.0f} dB)'
    w = x[-n:]
    N = 1 << 16
    X = np.abs(np.fft.rfft(w * np.hanning(n), N))
    f = np.fft.rfftfreq(N, 1 / sr)
    p, _ = _find_peaks(X, distance=int(30 / (sr / N)))
    p = p[np.argsort(-X[p])][:max_peaks]
    p = p[X[p] > X[p].max() * 10 ** (-30 / 20)]
    fr = []
    for i in p:                                           # parabolic refinement
        if 0 < i < len(X) - 1:
            a, b, c = np.log(X[i - 1:i + 2] + 1e-12)
            fr.append(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (sr / N))
    fr = np.array(fr)
    t = (np.arange(n) - (n - 1)) / sr                     # t = 0 at the last sample
    env = np.exp(rate * t)
    M = np.concatenate([np.cos(2 * np.pi * fr[None] * t[:, None]), np.sin(2 * np.pi * fr[None] * t[:, None])], 1) * env[:, None]
    coef, *_ = np.linalg.lstsq(M, w, rcond=None)
    fit = M @ coef
    explained = 1 - np.sum((w - fit) ** 2) / np.sum(w ** 2)
    if explained < 0.7:
        return seg, 0, f'not tonal enough ({explained:.2f} explained)'
    # how long until floor_db below the peak
    need = (floor_db - lvl) / 20 * np.log(10) / rate      # seconds
    add = int(min(need, max_len - len(x) / sr) * sr)
    if add <= 0:
        return seg, 0, 'no room'
    tt = np.arange(1, add + 1) / sr
    M2 = np.concatenate([np.cos(2 * np.pi * fr[None] * tt[:, None]), np.sin(2 * np.pi * fr[None] * tt[:, None])], 1) * np.exp(rate * tt)[:, None]
    ext = M2 @ coef
    fd = min(add, int(0.01 * sr))
    ext[-fd:] *= np.linspace(1, 0, fd) ** 2
    # undo any fade-out already applied to the end of seg: replace the last 20 ms with the model
    k = min(int(0.02 * sr), n)
    out = np.concatenate([x[:-k], fit[-k:], ext]) if seg.ndim == 1 else None
    if seg.ndim == 2:
        g = np.concatenate([fit[-k:], ext])
        out = np.concatenate([seg[:-k], np.repeat(g[:, None], seg.shape[1], 1) * 1.0], 0)
    return out.astype(np.float32), add * 1000 // sr, f'{len(fr)} partials, {explained:.2f} explained, decay {rate * 20 / np.log(10):.0f} dB/s'


def _noisiness(seg):
    m = seg.mean(1) if seg.ndim == 2 else seg
    X = np.abs(np.fft.rfft(m[:int(0.1 * SR)], 8192)) ** 2
    f = np.fft.rfftfreq(8192, 1 / SR)
    X = X[(f > 40) & (f < 4000)]
    return float(np.exp(np.mean(np.log(X + 1e-12))) / (np.mean(X) + 1e-12))


def _merge_same_core(sounds, log, same=0.85, tonal=0.1):
    """A tonal sound (a cowbell, a clave) with a hi-hat landing on it louder in some places than
    others gets split into 'different' sounds. Below the hat range they're the same, so keep one
    copy - the one with the least outside its own range - counting all their hits. Noisy sounds
    (snares, claps) aren't merged: their top end is part of the sound."""
    core = [_spec(s['seg'], 100, 5000) for s in sounds]
    ton = [_noisiness(s['seg']) < tonal for s in sounds]
    used, out = set(), []
    for i in range(len(sounds)):
        if i in used:
            continue
        fam = [i]
        if ton[i]:
            fam += [j for j in range(i + 1, len(sounds))
                    if j not in used and ton[j] and float(core[i] @ core[j]) >= same]
        used.update(fam)
        best = min(fam, key=lambda j: _outside_core_db(sounds[j]['seg']))
        if len(fam) > 1:
            log(f'one sound heard {len(fam)} ways (with something on top in some places): kept the cleanest')
        ons = [sounds[j].get('onsets') for j in fam if sounds[j].get('onsets') is not None]
        out.append(dict(sounds[best], hits=sum(sounds[j]['hits'] for j in fam),
                        onsets=np.unique(np.concatenate(ons)) if ons else None))
    return sorted(out, key=lambda s: -s['hits'])


def _namer(ref, log):
    """Name each sound by what the one-shot library says it is - only when the vote is sure (80%);
    otherwise keep the safe generic name. Drops sounds the library is 90% sure aren't drums."""
    generic = {'kick': 'Kick', 'sp': 'Snare-Perc', 'hh': 'Hat'}
    if not ref or not Path(ref).exists():
        return lambda s, role: (generic[role], False)
    import soundid
    V, C = soundid.load(ref)

    def name(s, role):
        piece, label, drop = soundid.name_sound(s['fp'], V, C)
        if drop and role == 'sp':
            log(f'dropped a sound heard {s["hits"]}x: the library says it is not a drum')
            return '', True
        if role == 'sp' and piece in ('snare', 'perc'):
            return label or {'snare': 'Snare', 'perc': 'Perc'}[piece], False
        if role == 'hh' and label in ('Open Hat', 'Shaker'):
            return label, False
        return generic[role], False
    return name


def _drop_echo_hats(hats, ks, log, share=0.98, win=0.012):
    """A 'hat' whose every hit lands on a kick, snare or perc hit is that drum's sizzle bleeding into the
    hat stem, not a hat (17 test beats vs the real tracks: 9 of 25 fake kit hats did this 98-100% of the time;
    no real hat over 97%). The main hat always stays; the slot isn't refilled."""
    other = [np.asarray(s['onsets']) for s in ks if GROUP.get(s['role']) != 'hh' and s.get('onsets') is not None]
    if not hats or not other:
        return hats
    other = np.sort(np.concatenate(other))
    keep = [hats[0]]
    for h in hats[1:]:
        on = np.asarray(h.get('onsets') if h.get('onsets') is not None else [])
        if len(on) >= 4:
            j = np.clip(np.searchsorted(other, on), 1, len(other) - 1)
            near = np.minimum(np.abs(other[j] - on), np.abs(other[j - 1] - on)) < int(win * SR)
            if near.mean() >= share:
                log(f'hats: left out a {h["hits"]}-hit sound that only ever plays with another drum (its bleed)')
                continue
        keep.append(h)
    return keep


def _closed_hat_out(hats, log, min_corr=0.6, win=0.03, search=60):
    """An open/long hat triggered together with the main closed hat carries that closed hat in its first
    milliseconds (the producer heard it as 'a closed hat inside'). When the closed hat's top end (>2 kHz)
    matches this hat's first 30 ms closely (corr >= min_corr), subtract it there (best offset, fitted gain).
    17 test beats: only breakdown136's two long hats match (0.82, 0.87); everything else <= 0.36.
    Producer A/B on breakdown136: the subtracted open hat sounded closer to his real one."""
    if len(hats) < 2:
        return hats
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 2000, 'high', fs=SR, output='sos')
    main = hats[0]['seg']
    A = sosfiltfilt(sos, _mono(main))
    w = int(win * SR)
    out = [hats[0]]
    for h in hats[1:]:
        seg = h['seg']
        if len(seg) <= len(main):
            out.append(h)
            continue
        O = sosfiltfilt(sos, _mono(seg))
        best = (0.0, 0, 0.0)
        for sh in range(-search, search + 1):
            a = A[max(0, -sh):][:w]
            o = O[max(0, sh):][:len(a)]
            if len(a) < w or len(o) < w:
                continue
            c = float(np.dot(a, o) / (np.sqrt(np.dot(a, a) * np.dot(o, o)) + 1e-12))
            if c > best[0]:
                best = (c, sh, float(np.dot(a, o) / (np.dot(a, a) + 1e-12)))
        c, sh, g = best
        if c >= min_corr and g > 0:
            new = seg.astype(np.float32).copy()
            a = main[max(0, -sh):]
            n = min(len(a), len(new) - max(0, sh))
            new[max(0, sh):max(0, sh) + n] -= g * a[:n]
            log(f'hats: took the closed hat out of the start of a {h["hits"]}-hit longer hat (match {c:.2f})')
            h = dict(h, seg=new)
        out.append(h)
    return out


def write_kit(split_dir, out_dir, prefix='', min_hits=4, log=lambda line: None, ref=None):
    """split_dir holds kick/snare/perc/hh.wav from Helyx. Writes the strongest few sounds per piece
    (each averaged over all its hits, snare and perc read together, kick-stem stowaways rescued) as
    24-bit WAVs peaking at -1 dBFS. Returns oneshots.make_kit-style records."""
    import soundfile as sf
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {r: split_dir / f'{r}.wav' for r in ('kick', 'snare', 'perc', 'hh') if (split_dir / f'{r}.wav').exists()}
    log('Finding every hit and averaging the repeats')
    ks = kit_sounds(split_dir, lambda r: have.get(r), log=log)
    ks = _drop_impostors(ks, split_dir, log)
    namer = _namer(ref, log)
    pool = None                                    # snare + perc stems, loaded when a snare needs a real hit
    out, parts = [], []
    for role in ('kick', 'sp', 'hh'):
        mine = sorted([s for s in ks if GROUP.get(s['role']) == role and s['hits'] >= min_hits],
                      key=lambda s: -s['hits'])
        if role == 'sp':
            mine = _merge_same_core(mine, log)
        mine = mine[:MAX_SOUNDS[role]]
        if role == 'hh':
            mine = _drop_echo_hats(mine, ks, log)
            mine = _closed_hat_out(mine, log)
        log(f'{LABELS[role].lower()}: {len(mine)} sound{"" if len(mine) == 1 else "s"}'
            + (f' (heard {", ".join(str(s["hits"]) for s in mine)} times)' if mine else ''))
        count = {}
        for s in mine:
            name, drop = namer(s, role)
            if drop:
                continue
            count[name] = count.get(name, 0) + 1
            seg = s['shot'] if s.get('shot') is not None else s['seg']
            if role == 'sp' and name == 'Snare' and s.get('onsets') is not None and len(s['onsets']) >= 4:
                # a sound the namer is sure is a snare: one real, typical hit, like the kick (the producer
                # picked the single hit on all three snares A/B'd, Nov 2026; Snare-Perc and hats stay averaged)
                if pool is None:
                    xs = [H.load(have[r]) for r in ('snare', 'perc') if r in have]
                    m = min(len(v) for v in xs)
                    pool = sum(v[:m] for v in xs)
                on = np.sort(np.asarray(s['onsets']))
                on = on[on + len(seg) <= len(pool)]
                if len(on) >= 4:
                    nxt = np.append(on[1:], len(pool))
                    ref = pool[on[int(np.argmax([np.abs(pool[i:i + int(0.03 * SR)]).max() for i in on]))]:][:int(0.04 * SR)]
                    hit, _ = typical_hit(pool, align(pool, on, ref), nxt, len(seg))
                    if hit is not None:
                        seg = hit
            if role == 'sp' and _noisiness(seg) < 0.1:
                fd = int(0.02 * SR)
                longer, added, why = extend_tonal(seg[:-fd] if len(seg) > 3 * fd else seg)
                if added:
                    seg = longer
                    log(f'{name}: let its release ring out {added} ms longer ({why})')
            seg = seg * (10 ** (-1 / 20) / (np.abs(seg).max() + 1e-9))
            lab = f'{name} {count[name]}'
            fname = f'{prefix}{lab}.wav'
            sf.write(str(out_dir / fname), seg, SR, subtype='PCM_24')
            parts.append(dict(label=lab, role=role, onsets=s.get('onsets')))
            out.append(dict(name=fname, file=str(out_dir / fname), piece={'sp': 'snare'}.get(role, role),
                            label=lab, count=int(s['hits']),
                            seconds=round(len(seg) / SR, 2)))
    try:
        write_groove(split_dir, out_dir, parts, prefix, have, log)
    except Exception as e:                         # a bonus; never lose the kit over it
        log(f'no groove MIDI: {e}')
    return out


def write_groove(split_dir, out_dir, parts, prefix, have, log, title=''):
    """The kit's drum pattern as '<song> - Groove.mid' next to the one-shots (see groove.py)."""
    import groove
    parts = [p for p in parts if p['onsets'] is not None and len(p['onsets'])]
    if not parts:
        return None
    stems = {r: H.load(f) for r, f in have.items()}
    src = {'kick': [stems.get('kick')], 'sp': [stems.get('snare'), stems.get('perc')], 'hh': [stems.get('hh')]}
    song = prefix.rstrip(' -') or title
    times = np.sort(np.concatenate([np.asarray(p['onsets']) / SR for p in parts]))
    bpm = groove.bpm_from_name(song) or groove.estimate_bpm(times)
    mid = []
    for p in parts:
        xs = [x for x in src[p['role']] if x is not None]
        n = min(len(x) for x in xs)
        on = np.sort(np.asarray(p['onsets']))
        mid.append(dict(label=p['label'], times=on / SR, velocities=groove.velocities(sum(x[:n] for x in xs), on, SR)))
    mid, start = groove.align(mid, bpm)
    # Preserve swing, pickups and closely spaced hits in the full-song export.
    # The compact loop remains quantized for the existing drag-and-drop workflow.
    groove.write(Path(out_dir) / f'{prefix}Groove (full song).mid', mid, bpm, title=song)
    mid = groove.quantize(mid, bpm)
    lp, first, bars = groove.loop(mid, bpm)
    f = Path(out_dir) / f'{prefix}Groove.mid'
    groove.write(f, lp, bpm, title=f'{song} - {bars} bars')
    log(f'groove MIDI: a {bars}-bar loop from bar {first} ({sum(len(m["times"]) for m in lp)} notes) '
        f'and the full song, at {bpm:g} BPM')
    return f


# ---------- the kick: 808 thumps out, the kick's own groups back together
def _kick_feats(x, on):
    """Per hit: sub share (<60 Hz vs <300 Hz, first 60 ms) and decay (80-120 ms vs first 30 ms), dB."""
    out = []
    for i in on:
        k = (x[i:i + int(0.2 * SR)].mean(1) if x.ndim == 2 else x[i:i + int(0.2 * SR)])
        if len(k) < int(0.12 * SR):
            out.append((0.0, 0.0))
            continue
        X = np.abs(np.fft.rfft(k[:int(0.06 * SR)], 8192)) ** 2
        f = np.fft.rfftfreq(8192, 1 / SR)
        sub = 10 * np.log10(X[f < 60].sum() / (X[f < 300].sum() + 1e-12) + 1e-12)
        dec = 20 * np.log10((np.abs(k[int(0.08 * SR):int(0.12 * SR)]).max() + 1e-9)
                            / (np.abs(k[:int(0.03 * SR)]).max() + 1e-9))
        out.append((sub, dec))
    return np.array(out)


def split_808(x, on, min_gap_sub=3.0, min_gap_dec=5.0):
    """Hits of one kick 'sound' that are really two things: the kick, and 808 notes thumping through the
    kick stem. 2-means on sub share and decay; if the groups are clearly apart (the sub-heavy one more sub
    by min_gap_sub dB AND dying faster by min_gap_dec dB) that group is the 808. Else all are kick.
    -> mask of the hits that are the kick. (breakdown136 vs its FL MIDI: 93% of 808 thumps out; on test
    beats with no 808 problem, no real kick removed.)"""
    if len(on) < 8:
        return np.ones(len(on), bool)
    F = _kick_feats(x, on)
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    c = np.array([Z[np.argmin(Z[:, 0])], Z[np.argmax(Z[:, 0])]])
    lab = np.zeros(len(on), int)
    for _ in range(20):
        lab = np.argmin(((Z[:, None, :] - c[None]) ** 2).sum(2), 1)
        if lab.min() == lab.max():
            return np.ones(len(on), bool)
        c = np.array([Z[lab == j].mean(0) for j in (0, 1)])
    m0, m1 = F[lab == 0].mean(0), F[lab == 1].mean(0)
    heavy = 1 if m1[0] > m0[0] else 0
    mh, ml = (m1, m0) if heavy == 1 else (m0, m1)
    nh, nl = (lab == heavy).sum(), (lab != heavy).sum()
    if (mh[0] - ml[0] >= min_gap_sub and ml[1] - mh[1] >= min_gap_dec and nh >= 4
            and nl >= max(4, 0.15 * len(on))):
        return lab != heavy
    return np.ones(len(on), bool)


def _full_low(x, on, within=4.0):
    """Mask of the hits whose low end (<120 Hz, first 60 ms) is within `within` dB of the kick's strongest
    (90th percentile). The kick stem also carries snares the model filed there and bleed; those hits
    have 6-8 dB less low end (hyperbole: real kicks 57-60 dB, the rest 50-52), and the one-shot must not
    be one of them (the producer: 'kick has no low end'). Never empties the mask."""
    m = _mono(x)
    f = np.fft.rfftfreq(8192, 1 / SR)
    low = np.array([10 * np.log10((np.abs(np.fft.rfft(m[i:i + int(0.06 * SR)], 8192)) ** 2)[f < 120].sum() + 1e-12)
                    for i in on])
    mask = low >= np.percentile(low, 90) - within
    return mask if mask.sum() >= 2 else np.ones(len(on), bool)


def clean_kicks(kx, ks, log=lambda line: None):
    """The kick stem's biggest group is THE kick (so unstack() doesn't pick the darkest sound, an 808).
    808 notes thumping through it are taken out (split_808); then the one-shot is one real, typical hit
    (typical_hit) rather than an average, which dulls the top end. Its length is what the average allowed.
    -> (groups, biggest first; number of 808 thumps removed)"""
    if not ks:
        return ks, 0
    ks = sorted(ks, key=lambda k: -k['hits'])
    k0 = ks[0]
    keep = split_808(kx, k0['onsets'])
    dropped = int((~keep).sum())
    on = k0['onsets'][keep]
    allon = np.sort(np.concatenate([k['onsets'] for k in ks]))
    nxt = np.array([allon[allon > i][0] if (allon > i).any() else len(kx) for i in on])
    avg = average(kx, on, nxt)[0] if dropped else k0['seg']
    full = _full_low(kx, on)                       # the hit is picked among hits with the kick's full low end
    onf = np.sort(on[full])
    nf = np.append(onf[1:], len(kx))
    # its length is how long a real kick has before the next real kick, not before the next anything in the
    # stem: snares filed in the kick stem cut the average short (hyperbole helyx-6: a 25 ms click, 'no low end')
    length = int(max(len(avg), min(MAX_LEN * SR, np.median(nf - onf))))
    ref = kx[onf[int(np.argmax([np.abs(kx[i:i + int(0.03 * SR)]).max() for i in onf]))]:][:int(0.04 * SR)]
    seg, _ = typical_hit(kx, align(kx, onf, ref), nf, length)
    if seg is not None and len(seg) > len(avg):
        k = max(len(avg), tail_end(seg))
        seg = seg[:k].copy()
        fd = min(k, int(0.02 * SR))
        seg[k - fd:] *= (np.linspace(1, 0, fd) ** 2)[:, None] if seg.ndim == 2 else np.linspace(1, 0, fd) ** 2
    # decisions downstream (unstacking, look-alikes) keep using the average; only the file is the real hit
    kick = dict(k0, seg=avg, shot=seg, onsets=on, hits=len(on), is_kick=bool(dropped))
    if dropped:
        log(f'kick: {dropped} 808 notes taken out of the kick ({len(on)} real kick hits left)')
    return [kick] + ks[1:], dropped


# ---------- the kick: one real hit, not an average
def _sustain_db(x, on, nxt):
    """Low end (<100 Hz) left 150-250 ms after each hit vs its first 40 ms, dB (nan if the next hit comes first).
    A clean kick has died away by then; an 808 under it keeps ringing (breakdown136 vs its 808 stem:
    median -38 dB without, -3 dB with)."""
    from scipy.signal import butter, sosfiltfilt
    L = sosfiltfilt(butter(4, 100, 'low', fs=SR, output='sos'), _mono(x))
    out = []
    for i, j in zip(on, nxt):
        e = min(j, i + int(0.25 * SR))
        if e - i < int(0.17 * SR):
            out.append(np.nan)
            continue
        a0 = np.sqrt(np.mean(L[i:i + int(0.04 * SR)] ** 2)) + 1e-9
        out.append(20 * np.log10(np.sqrt(np.mean(L[i + int(0.15 * SR):e] ** 2)) / a0 + 1e-9))
    return np.array(out)


def typical_hit(x, on, nxt, length, gap=15.0):
    """The single hit that best represents the sound: among hits with no 808 still ringing under them, the
    one whose first 80 ms spectrum is closest to the median of all of them (so not one with a stray snare on
    top). Averaging dulls a kick's top end; the producer's ear preferred a real hit (breakdown136, Nov 2026).
    -> (segment of `length` samples with a short fade, index into on) or (None, None)."""
    if len(on) < 3:
        return None, None
    s = _sustain_db(x, on, nxt)
    ok = ~np.isnan(s)
    lower = s[ok][s[ok] <= np.median(s[ok])] if ok.sum() else s[ok]
    cand = np.arange(len(on))
    if ok.sum() >= 8 and len(lower):
        clean_ = ~(ok & (s > np.median(lower) + gap))
        if clean_.sum() >= max(3, 0.3 * len(on)):
            cand = np.flatnonzero(clean_)
    # A representative hit must fit before the next onset, just like the average.
    # Otherwise its tail can contain a second drum hit. Keep the average if too
    # few isolated candidates remain.
    cand = [c for c in cand if on[c] + length <= min(nxt[c], len(x))]
    if len(cand) < 3:
        return None, None
    w = min(length, int(0.08 * SR))
    win = np.hanning(w)
    sp = np.array([np.log10(np.abs(np.fft.rfft(_mono(x[on[c]:on[c] + w]) * win)) + 1e-9) for c in cand])
    pick = cand[int(np.argmin(((sp - np.median(sp, 0)) ** 2).sum(1)))]
    seg = x[on[pick]:on[pick] + length].astype(np.float32).copy()
    fd = min(length, int(0.02 * SR))
    seg[length - fd:] *= (np.linspace(1, 0, fd) ** 2)[:, None] if seg.ndim == 2 else np.linspace(1, 0, fd) ** 2
    return seg, pick
