"""groove.py - the beat's drum pattern as a MIDI file, one track per one-shot.

Each sound gets its own track, named like its one-shot WAV, with every note on C5 (MIDI 60) - the key
that plays a sample at its own pitch in FL Studio. The full-song export preserves detected timing.
The compact loop is snapped to the grid that fits each sound:
16ths for most, 16th triplets or 32nds only for sounds that really play that fast (hat rolls).
Velocity comes from how hard each hit was.

The tempo comes from the song's name when it says one ("breakdown136", "ONEAZZ 130 bpm"), else it
is estimated from the hits. In the full-song export, tempo only decides how detected times
line up with the DAW's grid.
"""
import re
import struct

import numpy as np

PPQ = 960


ROOT = 60          # C5 in FL Studio: a sample on a channel plays at its own pitch


def bpm_from_name(name):
    """A tempo written in the song's name, if there's a plausible one (60-200)."""
    for m in re.finditer(r'(?<!\d)(\d{2,3}(?:\.\d+)?)\s*(?:bpm)?(?!\d)', name, re.I):
        v = float(m.group(1))
        if 60 <= v <= 200:
            return v
    return None


def estimate_bpm(times, lo=70, hi=180):
    """Tempo from hit times (seconds): autocorrelation of the hit train, then the strongest
    beat-length lag. Reggaeton/house-friendly range by default."""
    if len(times) < 8:
        return 120.0
    res = 0.01
    n = int(max(times) / res) + 2
    x = np.zeros(n)
    x[(np.asarray(times) / res).astype(int)] = 1
    x = np.convolve(x, np.hanning(5), 'same')
    ac = np.correlate(x, x, 'full')[n - 1:]
    best, score = 120.0, -1
    for bpm in np.arange(lo, hi + 0.5, 0.5):
        lag = 60 / bpm / res
        s = sum(np.interp(lag * m, np.arange(len(ac)), ac) for m in (1, 2, 4))   # beat, 2 beats, bar
        if s > score:
            best, score = bpm, s
    return float(best)


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _track(events):
    """events: [(tick, bytes)] -> MTrk chunk."""
    data, last = b'', 0
    for t, ev in sorted(events, key=lambda e: (e[0], e[1][0] & 0xF0 == 0x90)):   # note-offs first
        data += _vlq(t - last) + ev
        last = t
    data += _vlq(0) + b'\xff\x2f\x00'
    return b'MTrk' + struct.pack('>I', len(data)) + data


def align(parts, bpm):
    """Line the song up with the DAW's grid. The hits' common offset from the 16th grid is the grid's
    phase (songs rarely start exactly on a beat); then the bar starts on the first kick (or first
    hit), one bar earlier if anything comes before it. The pattern starts at bar 1, every hit keeps
    its feel relative to the others. -> (parts with shifted times, seconds removed)."""
    step = 60 / bpm / 4
    t = np.concatenate([p['times'] for p in parts])
    ang = 2 * np.pi * (t % step) / step
    phase = (np.angle(np.exp(1j * ang).mean()) % (2 * np.pi)) / (2 * np.pi) * step
    kicks = [p for p in parts if p['label'].startswith('Kick')]
    first = (kicks[0]['times'] if kicks else t).min()
    start = phase + np.round((first - phase) / (4 * step)) * 4 * step       # nearest beat to the first kick
    bar = 16 * step
    while (t < start).any():
        start -= bar
    return [dict(p, times=p['times'] - start) for p in parts], start


def quantize(parts, bpm):
    """Snap each sound to the coarsest grid its hits fit: 16ths, else 16th triplets, else 32nds.
    A hit detected a little late (a kick's slow rise) lands back on its 16th; a real hat roll keeps
    its fast notes. Two hits in one slot become one (the louder)."""
    step = 60 / bpm / 4
    out = []
    for p in parts:
        t = np.asarray(p['times'], float)
        v = np.asarray(p['velocities'])
        g = step / 2
        for cand in (step, step * 2 / 3):
            err = np.abs(t - np.round(t / cand) * cand) / step
            if len(t) and np.median(err) < 0.12 and np.percentile(err, 90) < 0.3:
                g = cand
                break
        slot = np.round(t / g).astype(int)
        keep = {}
        for sl, vel in zip(slot, v):
            keep[sl] = max(keep.get(sl, 0), int(vel))
        sl = np.array(sorted(keep))
        out.append(dict(p, times=sl * g, velocities=np.array([keep[k] for k in sl]),
                        grid={step: '16th', step / 2: '32nd'}.get(g, '16th triplet')))
    return out


def loop(parts, bpm, max_bars=8):
    """The beat's main groove, max_bars long (4 if the pattern truly repeats every 4 bars). Bars are
    scored by how much of the kit plays in them plus how typical they are of the song (so the main
    groove wins over an intro, a breakdown or a one-off fill); the best-scoring window starting on a
    4-bar boundary is kept. parts must already be aligned (bar 1 at t=0).
    -> (parts cut to the loop and shifted to start at 0, first bar number, bars)"""
    step = 60 / bpm / 4
    bar = 16 * step
    nb = int(max(p['times'].max() for p in parts if len(p['times'])) / bar) + 1
    if nb <= max_bars:
        return parts, 1, nb
    G = np.zeros((nb, len(parts), 16))
    for k, p in enumerate(parts):
        st = np.round(np.asarray(p['times']) / step).astype(int)
        st = st[(st >= 0) & (st < nb * 16)]
        G[st // 16, k, st % 16] = 1
    V = G.reshape(nb, -1)
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    typical = (U @ U.T).mean(1)                                  # how much each bar looks like the rest
    coverage = (G.sum(2) > 0).mean(1)                            # share of the kit's sounds playing
    score = typical + coverage
    best, best_s = 0, -1
    for b in range(0, nb - max_bars + 1, 4):
        sc = score[b:b + max_bars].mean()
        if sc > best_s:
            best, best_s = b, sc
    bars = max_bars
    if max_bars == 8 and best + 8 <= nb:                         # 4 bars that repeat exactly: just 4
        a, b2 = V[best:best + 4], V[best + 4:best + 8]
        if a.sum() > 0 and np.array_equal(a > 0, b2 > 0):
            bars = 4
    t0, t1 = best * bar, (best + bars) * bar
    out = []
    for p in parts:
        t = np.asarray(p['times'])
        m = (t >= t0 - step / 2) & (t < t1 - step / 2)
        out.append(dict(p, times=np.maximum(t[m] - t0, 0), velocities=np.asarray(p['velocities'])[m]))
    return [p for p in out if len(p['times'])], best + 1, bars


def write(path, parts, bpm, title=''):
    """parts: [dict(label, times (s), velocities 1-127)]. Writes a type-1 MIDI file."""
    tick = lambda s: int(round(s * bpm / 60 * PPQ))            # noqa: E731
    head = [(0, b'\xff\x03' + _vlq(len(title.encode())) + title.encode()),
            (0, b'\xff\x51\x03' + int(round(60e6 / bpm)).to_bytes(3, 'big')),
            (0, b'\xff\x58\x04\x04\x02\x18\x08')]                 # 4/4
    tracks = [_track(head)]
    for p in parts:
        nm = p['label'].encode()
        n = ROOT
        ev = [(0, b'\xff\x03' + _vlq(len(nm)) + nm)]
        dur = PPQ // 8                                           # a 32nd: one-shots ignore note length
        for t, v in zip(p['times'], p['velocities']):
            a = max(0, tick(t))
            ev.append((a, bytes([0x99, n, int(np.clip(v, 1, 127))])))      # channel 10: drums
            ev.append((a + dur, bytes([0x89, n, 0])))
        tracks.append(_track(ev))
    with open(path, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), PPQ))
        for t in tracks:
            f.write(t)


def velocities(x, onsets, sr, win=0.02):
    """How hard each hit was: its peak, mapped so the loudest hit of the sound is 127 and every
    -6 dB is about -20 velocity."""
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    pk = np.array([m[i:i + int(win * sr)].max(initial=0) for i in onsets]) + 1e-9
    db = 20 * np.log10(pk / pk.max())
    return np.clip(127 + db * 20 / 6, 20, 127).astype(int)


def read(path):
    """Minimal reader for checking: -> (bpm, {track name: [(seconds, note, velocity)]})."""
    b = open(path, 'rb').read()
    ntr, ppq = struct.unpack('>HH', b[10:14])
    pos, bpm, out = 14, 120.0, {}
    for _ in range(ntr):
        ln = struct.unpack('>I', b[pos + 4:pos + 8])[0]
        d, i, t, name, notes = b[pos + 8:pos + 8 + ln], 0, 0, '', []
        while i < len(d):
            v = 0
            while True:
                c = d[i]
                i += 1
                v = (v << 7) | (c & 0x7F)
                if not c & 0x80:
                    break
            t += v
            if d[i] == 0xFF:
                typ, i = d[i + 1], i + 2
                L = 0
                while True:
                    c = d[i]
                    i += 1
                    L = (L << 7) | (c & 0x7F)
                    if not c & 0x80:
                        break
                if typ == 0x03:
                    name = d[i:i + L].decode()
                if typ == 0x51:
                    bpm = 60e6 / int.from_bytes(d[i:i + L], 'big')
                i += L
            else:
                st = d[i]
                if st & 0xF0 == 0x90 and d[i + 2] > 0:
                    notes.append((t / ppq * 60 / bpm, d[i + 1], d[i + 2]))
                i += 3
        out[name] = notes
        pos += 8 + ln
    return bpm, out
