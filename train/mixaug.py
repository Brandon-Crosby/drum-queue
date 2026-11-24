"""mixaug.py - training-mix augmentations for helyx-4 (copied into /msst/utils by msst_patch.py).

Every function takes res: [stems, 2, T] (torch or numpy, float32) in the config's instrument order
and keeps the rule MSST's trainer relies on: the targets are exactly what's in the mix (apart from
clipping, which is distortion no stem owns).

restack(res, instruments, aug)   cut the snare stem's main hit and lay copies on some of the kicks,
                                 sometimes darkened - the stacked dark snare Helyx keeps filing as
                                 kick. The copies go into the snare target, so the answer stays known.
darken(res, instruments, aug)    low-pass the whole snare stem (and sometimes warm it up), so the
                                 model hears dark snares on their own, not only stacked on kicks.
                                 Only the snare stem changes, so the target stays exactly right.
onbeat(res, instruments, aug)    cut short chops out of the music (synth stabs, vox chops, 808 attacks)
                                 and lay them right on the snare/perc hits, louder than the usual
                                 background bleed. They stay in the 'other' target: on-beat music is
                                 still music (stunyun137 put synths and a vox chop in the snare stem).
nokick(res, instruments, aug)    sometimes leave the kick out: 808-led beats with no kick exist, and
                                 the 808's attack must not become a kick or a snare there.
master(res, aug)                 a master chain on the whole mix: tilt EQ + low shelf, bus
                                 compressor, push into a lookahead limiter, sometimes a clipper.
                                 EQ/compressor/limiter are gains applied to every stem alike (so a
                                 ducked snare in the mix is a ducked snare in the target); clipping
                                 only touches the mix. Returns (res, mix).

Config (augmentations:): aligned_prob, restack_prob, darken_prob, onbeat_prob, nokick_prob, quiethats_prob, layerkick_prob,
master_prob. All off when missing.
"""
import numpy as np

SR = 44100
BLK = 32                                   # envelope block, ~0.7 ms


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy(), True
    except ImportError:
        pass
    return np.asarray(x), False


def _back(x, was_torch):
    if was_torch:
        import torch
        return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
    return x.astype(np.float32)


def _get(aug, key, default=0.0):
    try:
        v = aug[key] if key in aug else default
    except TypeError:
        v = getattr(aug, key, default)
    return float(v) if v is not None else default


def _blocks(env, blk=BLK):
    n = len(env) // blk
    return env[:n * blk].reshape(n, blk).max(1), n


def onsets(x, min_gap=0.09, rel=0.3):
    """Hit starts in a (2, T) stem: block-peak envelope jumps well above the last ~10 ms."""
    env = np.abs(x).max(0)
    b, n = _blocks(env)
    if n < 20 or b.max() < 1e-4:
        return np.array([], int)
    back = max(1, int(0.010 * SR / BLK))
    prev = np.concatenate([np.full(back, b[0]), b[:-back]])
    cand = np.flatnonzero((b > rel * b.max()) & (b > 3.0 * prev + 1e-5))
    out, last = [], -10 ** 9
    gap = int(min_gap * SR / BLK)
    for i in cand:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.array(out, int) * BLK


def _lowpass(seg, fc):
    X = np.fft.rfft(seg, axis=-1)
    f = np.fft.rfftfreq(seg.shape[-1], 1 / SR)
    X *= 1 / np.sqrt(1 + (f / fc) ** 4)                     # 2nd-order-ish, zero phase
    return np.fft.irfft(X, n=seg.shape[-1], axis=-1)


def restack(res, instruments, aug, rng=np.random):
    p = _get(aug, 'restack_prob')
    if p <= 0 or rng.uniform() >= p or 'kick' not in instruments or 'snare' not in instruments:
        return res
    r, was = _np(res)
    r = r.copy()
    k, s = instruments.index('kick'), instruments.index('snare')
    kon = onsets(r[k])
    if len(kon) == 0 or np.abs(r[s]).max() < 1e-3:
        return res
    # the snare stem's loudest hit, from where its rise starts
    env = np.abs(r[s]).max(0)
    pk = int(np.argmax(env))
    st = pk
    lim = max(0, pk - int(0.02 * SR))
    while st > lim and env[st] > 0.1 * env[pk]:
        st -= 1
    L = int(rng.uniform(0.18, 0.35) * SR)
    shot = r[s][:, st:st + L].copy()
    if shot.shape[1] < int(0.05 * SR):
        return res
    fd = min(shot.shape[1], int(0.01 * SR))
    shot[:, -fd:] *= np.linspace(1, 0, fd)
    if rng.uniform() < 0.5:                                  # the dark snares that fail today
        shot = _lowpass(shot, rng.uniform(700, 2500))
    shot *= np.abs(r[k]).max() * rng.uniform(0.2, 0.9) / (np.abs(shot).max() + 1e-9)
    keep = rng.uniform(0.3, 1.0)
    T = r.shape[-1]
    for o in kon:
        if rng.uniform() > keep:
            continue
        o = int(o + rng.uniform(-0.002, 0.002) * SR)
        if o < 0 or o >= T:
            continue
        n = min(shot.shape[1], T - o)
        r[s][:, o:o + n] += shot[:, :n]
    return _back(r, was)


def _smooth_db(target_db, att_blocks, rel_blocks):
    """One-pole attack/release over a block sequence (gain-reduction in dB, <= 0)."""
    out = np.empty_like(target_db)
    g = 0.0
    a = np.exp(-1.0 / max(att_blocks, 1e-3))
    rl = np.exp(-1.0 / max(rel_blocks, 1e-3))
    for i, t in enumerate(target_db):
        c = a if t < g else rl
        g = c * g + (1 - c) * t
        out[i] = g
    return out


def _apply_gain(r, gain_blocks, n):
    T = r.shape[-1]
    t_blk = (np.arange(n) + 0.5) * BLK
    g = np.interp(np.arange(T), t_blk, gain_blocks)
    return r * g[None, None, :]


def darken(res, instruments, aug, rng=np.random):
    p = _get(aug, 'darken_prob')
    if p <= 0 or rng.uniform() >= p or 'snare' not in instruments:
        return res
    r, was = _np(res)
    s = instruments.index('snare')
    x = r[s].astype(np.float64)
    if np.abs(x).max() < 1e-4:
        return res
    T = x.shape[-1]
    f = np.fft.rfftfreq(T, 1 / SR)
    g = 1 / np.sqrt(1 + (f / rng.uniform(800, 3000)) ** 4)         # high end mostly gone
    g *= 10 ** ((rng.uniform(0, 6) / (1 + (f / 250) ** 2)) / 20)    # sometimes a warm low-mid bump
    y = np.fft.irfft(np.fft.rfft(x, axis=-1) * g, n=T, axis=-1)
    y *= np.abs(x).max() / (np.abs(y).max() + 1e-9)                # same peak level as before
    r = r.copy()
    r[s] = y.astype(r.dtype)
    return _back(r, was)


def onbeat(res, instruments, aug, rng=np.random):
    p = _get(aug, 'onbeat_prob')
    if p <= 0 or rng.uniform() >= p or 'other' not in instruments:
        return res
    r, was = _np(res)
    o = instruments.index('other')
    hits_at = [instruments.index(k) for k in ('snare', 'perc') if k in instruments]
    if not hits_at or np.abs(r[o]).max() < 1e-4:
        return res
    tgt = np.concatenate([onsets(r[i]) for i in hits_at])
    src = onsets(r[o], rel=0.2)
    if len(tgt) == 0 or len(src) == 0:
        return res
    r = r.copy()
    T = r.shape[-1]
    drums_pk = max(np.abs(r[i]).max() for i in hits_at) + 1e-9
    for _ in range(rng.randint(1, 4)):                         # a few different chops per clip
        a = int(src[rng.randint(len(src))])
        L = int(rng.uniform(0.08, 0.3) * SR)
        chop = r[o][:, a:a + L].copy()
        if chop.shape[1] < int(0.04 * SR) or np.abs(chop).max() < 1e-5:
            continue
        fd = min(chop.shape[1], int(0.01 * SR))
        chop[:, -fd:] *= np.linspace(1, 0, fd)
        chop *= drums_pk * 10 ** (rng.uniform(-14, -3) / 20) / np.abs(chop).max()
        for t in tgt:
            if rng.uniform() > rng.uniform(0.3, 0.8):
                continue
            t = int(t + rng.uniform(-0.003, 0.003) * SR)
            if 0 <= t < T:
                n = min(chop.shape[1], T - t)
                r[o][:, t:t + n] += chop[:, :n]
    return _back(r, was)


def nokick(res, instruments, aug, rng=np.random):
    p = _get(aug, 'nokick_prob')
    if p <= 0 or rng.uniform() >= p or 'kick' not in instruments:
        return res
    r, was = _np(res)
    r = r.copy()
    r[instruments.index('kick')] = 0
    return _back(r, was)


def _kick_layer(rng):
    """One clip's kick top layer: a knock (a pitched body at 250-700 Hz, dropping in pitch, 15-40 ms) and
    a click (band-passed noise, 1.5-8 kHz, 2-8 ms), each at its own level (dB re the kick's peak)."""
    from scipy.signal import butter, sosfilt
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(250, 700)
    f = f0 * (1 + rng.uniform(0.2, 1.0) * np.exp(-t / 0.004))          # quick pitch drop
    knock = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / rng.uniform(0.015, 0.04))
    lo, hi = rng.uniform(1500, 3000), rng.uniform(5000, 8000)
    click = sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), rng.randn(n))
    click *= np.exp(-t / rng.uniform(0.002, 0.008))
    knock *= 10 ** (rng.uniform(-20, -8) / 20) / (np.abs(knock).max() + 1e-9)
    click *= 10 ** (rng.uniform(-26, -12) / 20) / (np.abs(click).max() + 1e-9)
    use = rng.uniform()
    return (knock if use < 0.8 else 0) + (click if use > 0.2 else 0)


def layerkick(res, instruments, aug, rng=np.random):
    """Kicks with a body/knock and click on top, all in the kick target. helyx-4/6 keep a kick's sub but
    file its 300 Hz+ knock and click in the snare/perc stems (providence, TABLA, ONEAAZZ: 5-10 dB
    missing at 300-700 Hz, 'the kick has no body'); this teaches that the layers belong to the kick.
    The same layer on every kick of the clip (as in a real kit), level jittered +-2 dB per hit."""
    p = _get(aug, 'layerkick_prob')
    if p <= 0 or rng.uniform() >= p or 'kick' not in instruments:
        return res
    r, was = _np(res)
    k = instruments.index('kick')
    if np.abs(r[k]).max() < 1e-4:
        return res
    on = onsets(r[k])
    if len(on) == 0:
        return res
    r = r.copy()
    T = r.shape[-1]
    layer = _kick_layer(rng)
    for t in on:
        t = int(t)
        pk = np.abs(r[k][:, t:t + int(0.03 * SR)]).max()
        n = min(len(layer), T - t)
        if n > 0:
            r[k][:, t:t + n] += pk * 10 ** (rng.uniform(-2, 2) / 20) * layer[:n]
    return _back(r, was)


def quiethats(res, instruments, aug, rng=np.random):
    """Turn the hats (shakers count as hats) 12-30 dB down under the other drums. church's chicken: its
    shaker and open hat sit ~30 dB under loud percs, and helyx-6 put 60% of them in the perc stem and 4% in
    hats, leaving kick bleed as the 'hats'. The target hh stem is turned down with the mix, so the model
    learns that a quiet hat is still a hat."""
    p = _get(aug, 'quiethats_prob')
    if p <= 0 or rng.uniform() >= p or 'hh' not in instruments:
        return res
    r, was = _np(res)
    r = r.copy()
    r[instruments.index('hh')] *= 10 ** (-rng.uniform(12, 30) / 20)
    return _back(r, was)


def master(res, aug, rng=np.random):
    r, was = _np(res)
    p = _get(aug, 'master_prob')
    if p <= 0 or rng.uniform() >= p:
        return res, _back(r.sum(0), was)
    r = r.astype(np.float64).copy()
    T = r.shape[-1]
    # 1. master EQ: a gentle tilt and a low shelf, the same on every stem
    if rng.uniform() < 0.7:
        f = np.fft.rfftfreq(T, 1 / SR)
        tilt = rng.uniform(-1.5, 1.5) * np.log2(np.maximum(f, 20) / 1000)
        shelf = rng.uniform(0, 4) / (1 + (f / 100) ** 2)
        g = 10 ** (np.clip(tilt + shelf, -6, 6) / 20)
        r = np.fft.irfft(np.fft.rfft(r, axis=-1) * g, n=T, axis=-1)
    # 2. bus compressor, keyed off the whole mix
    if rng.uniform() < 0.7:
        b, n = _blocks(np.abs(r.sum(0)).max(0))
        if n > 4:
            db = 20 * np.log10(b + 1e-9)
            thr = np.percentile(db, rng.uniform(55, 85))
            ratio = rng.uniform(2, 6)
            gr = -(1 - 1 / ratio) * np.maximum(0, db - thr)
            gr = _smooth_db(gr, rng.uniform(3, 30) * 1e-3 * SR / BLK, rng.uniform(50, 250) * 1e-3 * SR / BLK)
            r = _apply_gain(r, 10 ** (gr / 20), n)
    # 3. push into a lookahead limiter
    m = np.abs(r.sum(0)).max() + 1e-9
    ceil = rng.uniform(0.8, 0.98)
    r *= 10 ** (rng.uniform(4, 14) / 20) * (ceil / m)           # 4-14 dB over the ceiling: modern-master loud
    b, n = _blocks(np.abs(r.sum(0)).max(0))
    need = np.minimum(1.0, ceil / (b + 1e-9))
    look = max(1, int(rng.uniform(1, 6) * 1e-3 * SR / BLK))
    pad = np.concatenate([need, np.ones(look)])
    need = np.min(np.lib.stride_tricks.sliding_window_view(pad, look + 1), axis=1)[:n]
    gr = _smooth_db(20 * np.log10(need), 0.3, rng.uniform(20, 200) * 1e-3 * SR / BLK)
    gr = np.minimum(gr, 20 * np.log10(need))                    # the attack never lets a peak through
    r = _apply_gain(r, 10 ** (gr / 20), n)
    mix = r.sum(0)
    # 4. sometimes a clipper on the mix only (distortion no stem owns)
    if rng.uniform() < 0.4:
        drive = 10 ** (rng.uniform(1, 6) / 20)
        mix = np.tanh(mix * drive) / drive
    pk = np.abs(mix).max()
    if pk > 0.99:
        r *= 0.99 / pk
        mix *= 0.99 / pk
    return _back(r, was), _back(mix, was)
