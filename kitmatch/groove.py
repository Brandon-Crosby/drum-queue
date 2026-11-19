"""groove.py - the beat's drum pattern as a MIDI file, one track per one-shot.

Each note sits exactly where that sound plays in the song (not snapped to a grid, so the swing and
the push and pull survive), with a velocity from how hard that hit was. Tracks are named like the
one-shot WAVs, so in FL Studio each track lands on a channel you load the matching sample into;
notes follow the General MIDI drum map so a drum rack plays something sensible too.

The tempo comes from the song's name when it says one ("breakdown136", "ONEAZZ 130 bpm"), else it
is estimated from the hits. Either way the notes keep their real times: the tempo only decides how
they line up with the DAW's grid.
"""
import re
import struct

import numpy as np

GM = {'Kick': 36, 'Snare': 38, 'Clap': 39, 'Rim': 37, 'Hat': 42, 'Open Hat': 46, 'Shaker': 70, 'Cymbal': 49,
      'Tom': 45, 'Cowbell': 56, 'Perc': 63, 'Snare-Perc': 40}
PPQ = 960


def note_for(label):
    """'Snare 2' -> GM snare + 1: same-kind sounds step up a key each so they don't share one."""
    m = re.match(r'(.*?)\s*(\d+)$', label)
    base, n = (m.group(1), int(m.group(2))) if m else (label, 1)
    return min(127, GM.get(base, 60) + n - 1)


def bpm_from_name(name):
    """A tempo written in the song's name, if there's a plausible one (60-200)."""
    for m in re.finditer(r'(?<!\d)(\d{2,3})(?:\.\d+)?\s*(?:bpm)?(?!\d)', name, re.I):
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
    while (t < start - step / 2).any():
        start -= bar
    return [dict(p, times=p['times'] - start) for p in parts], start


def write(path, parts, bpm, title=''):
    """parts: [dict(label, times (s), velocities 1-127)]. Writes a type-1 MIDI file."""
    tick = lambda s: int(round(s * bpm / 60 * PPQ))            # noqa: E731
    head = [(0, b'\xff\x03' + _vlq(len(title.encode())) + title.encode()),
            (0, b'\xff\x51\x03' + int(round(60e6 / bpm)).to_bytes(3, 'big')),
            (0, b'\xff\x58\x04\x04\x02\x18\x08')]                 # 4/4
    tracks = [_track(head)]
    # every sound its own key, for drum racks; each kind's first sound gets its standard key
    keys, used = {}, set()
    for p in sorted(parts, key=lambda p: not re.search(r'\s1$', p['label'])):
        n = note_for(p['label'])
        while n in used and n < 127:
            n += 1
        keys[p['label']] = n
        used.add(n)
    for p in parts:
        nm = p['label'].encode()
        n = keys[p['label']]
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
