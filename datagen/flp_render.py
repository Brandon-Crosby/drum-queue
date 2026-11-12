#!/usr/bin/env python3
"""flp_render.py - rebuild an FL Studio project's drums track by track, from its notes and its samples.

  python flp_render.py FLP_ROOT --out TRAINSET --midi GROOVES [--only NAME] [--force]

For every .flp under FLP_ROOT (FL backups skipped): read the playlist arrangement (pattern clips and
their notes), find each sampler channel's one-shot next to the project, decide its role from the
channel name, sample file name and sample folder (Kicks\\, Snares\\...), and play every note into that
role's track (pitch, velocity, pan, channel volume). Writes

  TRAINSET/<song> (FLP)/{kick,snare,hh,other_drums,rest}.flac    training copy, same layout as autosort's
  GROOVES/<song>.mid + <song>.json                               the drum notes per role (answer key)

The tracks are rendered dry and only from channels we could classify and load, so every track is
exactly what's in the mix we make from them: missing plugins or unknown channels never put wrong
labels in the training data, they just leave that sound out. Drum loops (mixed kits in one sample) are
skipped for the same reason. Needs Python 3.10 for pyflp.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'eval'))
from autosort import WORDS  # noqa: E402

SR = 44100
ROLES = ['kick', 'snare', 'hh', 'other_drums', 'rest']
DRUMS = ROLES[:4]
NOTE = {n: i for i, n in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
MUTED = 0x2000                   # playlist item flag
MAX_SEC = 600
GM = {'kick': 36, 'snare': 38, 'hh': 42, 'other_drums': 47, 'rest': 35}   # GM drum notes for the MIDI


def _words(name):
    low = re.sub(r'[_\-=+()\[\]{}|.,#@!]+', ' ', name.lower())
    return re.sub(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)', ' ', low)


def role_of(*names):
    """First name that clearly says what it is wins. 808s/bass count as rest (music under the drums);
    anything we can't name is None and gets left out."""
    for name in names:
        if not name:
            continue
        low = _words(name)
        if re.search(r'\b808\b', low) and re.search(r'kick|snare|clap|hi ?hat|\bhh\b|\btom|rim|cowbell|conga', low):
            low = re.sub(r'\b808\b', ' ', low)     # "808 Kick": the TR-808 drum machine, not an 808 bass
        for role, rx in WORDS:
            if role in ('mix', 'fx_return'):
                continue
            if re.search(rx, low):
                if role in DRUMS and re.search(r'\bloops?\b|\bbreaks?\b|\bkit\b|\bdrums?\b|\bfill', low) \
                        and not re.search(r'\btop loop', low):
                    return 'loop'          # several drums in one sample: can't label it
                return role
    return None


def midi_key(k):
    if isinstance(k, (int, np.integer)):
        return int(k)
    m = re.match(r'([A-G]#?)(-?\d+)$', str(k))
    return NOTE[m.group(1)] + 12 * int(m.group(2))


def find_sample(path, roots):
    """FL stores the path on the producer's machine; look for the file name next to the project."""
    if not path:
        return None
    name = re.split(r'[\\/]', str(path))[-1]
    for r in roots:
        hits = [p for p in r.rglob('*') if p.name.lower() == name.lower()]
        if hits:
            return hits[0]
    return None


_cache = {}


def load(p):
    if p not in _cache:
        try:
            x, sr = sf.read(str(p), dtype='float32', always_2d=True)
        except Exception:                             # .wv (WavPack) and friends
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav') as t:
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(p), t.name], check=True)
                x, sr = sf.read(t.name, dtype='float32', always_2d=True)
        x = x.T
        if x.shape[0] == 1:
            x = np.repeat(x, 2, 0)
        x = x[:2]
        if sr != SR:
            n = int(x.shape[1] * SR / sr)
            t = np.linspace(0, x.shape[1] - 1, n)
            x = np.stack([np.interp(t, np.arange(x.shape[1]), c) for c in x]).astype(np.float32)
        _cache[p] = x[:, :SR * 8]                     # one-shots; cap long tails
    return _cache[p]


def repitch(x, semis):
    if abs(semis) < 0.01:
        return x
    r = 2 ** (semis / 12)
    n = int(x.shape[1] / r)
    t = np.arange(n) * r
    return np.stack([np.interp(t, np.arange(x.shape[1]), c) for c in x]).astype(np.float32)


def render(flp: Path, root: Path):
    import pyflp
    p = pyflp.parse(str(flp))
    if not p.tempo:
        return None, ['  no tempo stored in this project (newer FL format pyflp can\'t read)']
    tick = 60.0 / (float(p.tempo) * p.ppq)
    roots = [flp.parent, root]
    chans, log = {}, []
    for ch in p.channels:
        sp = getattr(ch, 'sample_path', None)
        if type(ch).__name__ != 'Sampler' or not sp:
            continue
        parts = re.split(r'[\\/]', str(sp))
        role = role_of(ch.name, parts[-1], parts[-2] if len(parts) > 1 else '')
        if role is None or role == 'loop':
            log.append(f'  left out  {ch.name!r} ({parts[-1]}): {"drum loop" if role else "unknown sound"}')
            continue
        f = find_sample(sp, roots)
        if f is None:
            log.append(f'  missing   {ch.name!r}: {parts[-1]} not found next to the project')
            continue
        try:
            snd = load(f)
        except Exception as e:
            log.append(f'  unreadable {parts[-1]}: {e}')
            continue
        kb = getattr(ch, 'keyboard', None)
        root_note = getattr(kb, 'root_note', 60) or 60
        vol = (getattr(ch, 'volume', 10000) or 10000) / 10000
        pan = ((getattr(ch, 'pan', 6400) or 6400) - 6400) / 6400
        cents = getattr(ch, 'pitch_shift', 0) or 0
        chans[ch.iid] = dict(role=role, name=ch.name, snd=snd, root=root_note, vol=vol, pan=pan, cents=cents)
        log.append(f'  {role:<11} {ch.name!r} <- {f.name}')
    pats = {pat.iid: pat for pat in p.patterns}
    hits = []                                       # (sec, chan, key, vel, note pan, fine pitch)
    clips = []                                      # one-shots dragged straight into the playlist
    for track_on, item in playlist(p):
        if not track_on or item['item_flags'] & MUTED:
            continue
        if item['item_index'] <= item['pattern_base']:
            c = item['item_index']
            if c in chans:
                off = item['start_offset']
                off = 0 if off != off or off < 0 else off          # ms into the sample
                clips.append((item['position'] * tick, c, off / 1000, item['length'] * tick))
            continue
        pat = pats.get(item['item_index'] - item['pattern_base'])
        if pat is None:
            continue
        start = item['start_offset']
        start = 0 if start != start or start < 0 else start                # nan: whole pattern
        for n in pat.notes:
            if n.rack_channel not in chans:
                continue
            rel = n.position - start
            if 0 <= rel < item['length']:
                hits.append(((item['position'] + rel) * tick, n.rack_channel, midi_key(n.key),
                             n.velocity, n.pan, n.fine_pitch, 0.0, n.length * tick, False))
    for sec, c, off, dur in clips:
        hits.append((sec, c, chans[c]['root'], 100, 64, 120, off, dur, True))
    if not hits:
        return None, log + ['  no notes on usable channels']
    end = min(MAX_SEC, max(h[0] for h in hits) + 3)
    n_out = int(end * SR)
    tracks = {r: np.zeros((2, n_out), np.float32) for r in ROLES}
    groove = []
    hits.sort(key=lambda h: h[0])
    nxt, last = {}, {}                              # next note start on the same channel
    for k in range(len(hits) - 1, -1, -1):
        nxt[k] = last.get(hits[k][1], 1e9)
        last[hits[k][1]] = hits[k][0]
    for k, h in enumerate(hits):
        sec, c, key, vel, npan, fine, off, dur, is_clip = h
        if sec >= end:
            continue
        ch = chans[c]
        x = repitch(ch['snd'], key - ch['root'] + (ch['cents'] + (fine - 120)) / 100)
        x = x[:, int(off * SR):]
        stop = None
        if ch['role'] == 'rest':                    # 808s/bass: a note stops at its length or the next note
            stop = min(dur if dur > 0 else 1e9, nxt[k] - sec)
        elif is_clip:
            stop = dur                              # audio clip: trimmed to the clip
        if stop is not None and stop * SR < x.shape[1]:
            m = max(1, int(stop * SR))
            x = x[:, :m].copy()
            f = min(m, int(0.01 * SR))
            x[:, m - f:] *= np.linspace(1, 0, f, dtype=np.float32)
        g = (vel / 100) * ch['vol']
        pn = np.clip(ch['pan'] + (npan - 64) / 64, -1, 1)
        i = int(sec * SR)
        seg = x[:, :n_out - i]
        tracks[ch['role']][0, i:i + seg.shape[1]] += seg[0] * g * min(1, 1 - pn)
        tracks[ch['role']][1, i:i + seg.shape[1]] += seg[1] * g * min(1, 1 + pn)
        groove.append(dict(t=round(sec, 4), role=ch['role'], sound=ch['name'], vel=vel))
    meta = dict(tempo=float(p.tempo), seconds=round(end, 2), fl_version=str(p.version),
                hits={r: sum(1 for h in groove if h['role'] == r) for r in ROLES})
    return (tracks, sorted(groove, key=lambda h: h['t']), meta), log


def playlist(p):
    """(track enabled?, raw playlist item) for the first arrangement. Reads the playlist event directly:
    pyflp's own Arrangement.tracks crashes on clips whose channel was deleted."""
    from pyflp.arrangement import ArrangementID, TrackID, Track
    from pyflp.types import FLVersion
    arr = next(iter(p.arrangements))
    if ArrangementID.Playlist not in arr.events.ids:
        return
    pl = arr.events.first(ArrangementID.Playlist)
    max_idx = 499 if arr._kw['version'] >= FLVersion(12, 9, 1) else 198
    enabled = {}
    for i, ed in enumerate(arr.events.divide(TrackID.Data, *TrackID)):
        try:
            enabled[i] = Track(ed, items=[]).enabled is not False
        except Exception:
            enabled[i] = True
    for item in pl:
        yield enabled.get(max_idx - item['track_rvidx'], True), item


def write_midi(groove, tempo, path):
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    evs = []
    for h in groove:
        t = int(round(h['t'] / (60 / tempo) * 480))
        note = GM[h['role']]
        evs += [(t, 1, mido.Message('note_on', channel=9, note=note, velocity=min(127, h['vel']))),
                (t + 60, 0, mido.Message('note_off', channel=9, note=note, velocity=0))]
    now = 0
    for t, _, m in sorted(evs, key=lambda e: (e[0], e[1])):
        tr.append(m.copy(time=t - now))
        now = t
    mid.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', required=True, help='training copies go here (one folder per project)')
    ap.add_argument('--midi', required=True, help='grooves (MIDI + JSON answer keys) go here')
    ap.add_argument('--only', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    root, out, gdir = Path(a.root), Path(a.out), Path(a.midi)
    gdir.mkdir(parents=True, exist_ok=True)
    flps = sorted(f for f in root.rglob('*.flp') if 'backup' not in str(f).lower() and not f.name.startswith('._'))
    seen = set()
    for flp in flps:
        name = flp.parent.name if flp.parent != root else flp.stem
        if a.only and a.only.lower() not in str(flp).lower():
            continue
        if name in seen:
            name = f'{name} - {flp.stem}'
        seen.add(name)
        dest = out / f'{name} (FLP)'
        print(f'\n## {name}  [{flp.name}]')
        if dest.exists() and not a.force:
            print('  already rendered (use --force to redo)')
            continue
        try:
            res, log = render(flp, flp.parent)
        except Exception as e:
            print(f'  could not read: {type(e).__name__}: {e}')
            continue
        print('\n'.join(log))
        if res is None:
            continue
        tracks, groove, meta = res
        print(f"  {meta['tempo']:g} bpm, {meta['seconds']} s, hits {meta['hits']}")
        if not (meta['hits']['kick'] or meta['hits']['snare']):
            print('  -> no kick or snare notes: not saved')
            continue
        peak = max(np.abs(sum(tracks.values())).max(), 1e-9)
        g = min(1.0, 0.9 / peak)
        dest.mkdir(parents=True, exist_ok=True)
        for r, x in tracks.items():
            sf.write(str(dest / f'{r}.flac'), (x * g).T, SR, subtype='PCM_24')
        (dest / 'trainbeat.json').write_text(json.dumps(dict(source=str(flp.relative_to(root)), **meta)))
        write_midi(groove, meta['tempo'], gdir / f'{name}.mid')
        (gdir / f'{name}.json').write_text(json.dumps(dict(meta=meta, hits=groove)))
        print(f'  -> training copy saved')


if __name__ == '__main__':
    main()
