#!/usr/bin/env python3
"""als_render.py - training copies from Ableton Live projects, the way flp_render.py does FL ones.

  python3 datagen/als_render.py ROOT --out TRAIN_DIR --midi GROOVE_DIR [--only NAME] [--force]

An .als file is gzipped XML. This replays every drum hit with the project's own samples, so each
piece comes out on its own track:
  - MIDI clips on Simpler tracks: note pitch vs the root key, transpose, one-shot or classic (gated)
    playback, sample start/end, Simpler volume, track volume and pan
  - audio clips (loops, one-shots): clip looping, start markers, warping (stretched by resampling)
Muted tracks, and tracks inside muted groups, are skipped. A track's role comes from its name, then
its sample's name, then the names of the groups it sits in ('Drums > Cymbals > Hat Loop').
Plugin instruments (Serum, Kontakt...) can't be played here, so those tracks are left out of the
stems AND the mix: the model never sees a sound with the wrong label.
Projects shipped in several versions (e.g. 'Ableton Piano' / 'PML Piano') have the same drums, so
one .als per project folder is used, preferring a '[Merged Drums]' one.
Output matches flp_render.py: OUT/<project> (ALS)/{kick,snare,hh,other_drums,rest}.flac + JSON + MIDI.
"""
import argparse
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flp_render import ROLES, SR, MAX_SEC, role_of, find_sample, repitch, write_midi  # noqa: E402


def _v(e, path, default=None, cast=float):
    x = e.find(path) if path else e
    if x is None or x.get('Value') is None:
        return default
    s = x.get('Value')
    if cast is bool:
        return s == 'true'
    try:
        return cast(s)
    except ValueError:
        return default


_cache = {}


def load(p):
    """Whole file at SR, stereo (loops can be long, so no cap like flp_render's one-shots)."""
    if p not in _cache:
        x, sr = sf.read(str(p), dtype='float32', always_2d=True)
        x = x.T
        if x.shape[0] == 1:
            x = np.repeat(x, 2, 0)
        x = x[:2]
        if sr != SR:
            n = int(x.shape[1] * SR / sr)
            t = np.linspace(0, x.shape[1] - 1, n)
            x = np.stack([np.interp(t, np.arange(x.shape[1]), c) for c in x]).astype(np.float32)
        _cache[p] = (x, sr)
    return _cache[p]


def sample_file(fileref, als, roots):
    rel = _v(fileref, 'RelativePath', '', str)
    if rel and (als.parent / rel).exists():
        return als.parent / rel
    return find_sample(_v(fileref, 'Path', '', str) or rel, roots)


def spans(c):
    """[(content start, content end, arrangement start)] in beats for one arrangement clip,
    following the clip's start marker and its loop."""
    start, end = _v(c, 'CurrentStart', 0), _v(c, 'CurrentEnd', 0)
    ls, le = _v(c, 'Loop/LoopStart', 0), _v(c, 'Loop/LoopEnd', 0)
    c0 = ls + _v(c, 'Loop/StartRelative', 0)
    if not _v(c, 'Loop/LoopOn', False, bool) or le - ls <= 1e-6:
        stop = le if le > c0 else c0 + (end - start)
        return [(c0, min(stop, c0 + end - start), start)]
    out, a, cf = [], start, c0
    while a < end - 1e-9 and len(out) < 10000:
        seg = min(le - cf, end - a)
        if seg <= 1e-9:
            cf = ls
            continue
        out.append((cf, cf + seg, a))
        a, cf = a + seg, ls
    return out


def tracks_of(ls):
    tr = list(ls.find('Tracks'))
    by_id = {t.get('Id'): t for t in tr}

    def chain(t):                                    # the track, then each group above it
        out = [t]
        while _v(out[-1], 'TrackGroupId', '-1', str) not in ('-1', None):
            g = by_id.get(_v(out[-1], 'TrackGroupId', '-1', str))
            if g is None:
                break
            out.append(g)
        return out
    for t in tr:
        if t.tag in ('MidiTrack', 'AudioTrack'):
            yield t, chain(t)


def voice(sim, als, roots, names, gain, pan, play):
    """One Simpler: its sample (trimmed to start/end), pitch, volume and label. `names` are tried in
    order for the label; None in it marks where the sample's own file name goes. `play` is the note a
    Drum Rack pad sends (None: the played note)."""
    fr = sim.find('.//SampleRef/FileRef')
    sname = Path(_v(fr, 'RelativePath', '', str) or _v(fr, 'Path', '', str)).name if fr is not None else ''
    names = [sname if n is None else n for n in names]
    role = role_of(*names)
    label = names[0]
    if role is None or role == 'loop':
        return dict(log=f'  left out  {label!r} ({sname}): {"drum loop" if role else "unknown sound"}')
    f = sample_file(fr, als, roots) if fr is not None else None
    if f is None:
        return dict(log=f'  missing   {label!r}: {sname} not found')
    snd, osr = load(f)
    part = sim.find('Player/MultiSampleMap/SampleParts/MultiSamplePart')
    r = SR / osr
    s0 = int(_v(part, 'SampleStart', 0) * r)
    s1 = int(_v(part, 'SampleEnd', snd.shape[1] / r) * r) or snd.shape[1]
    return dict(log='', role=role, label=label, file=f.name, snd=snd[:, s0:max(s0 + 1, s1)],
                root=_v(part, 'RootKey', 60), play=play,
                tr=_v(sim, 'Pitch/TransposeKey/Manual', 0) + _v(sim, 'Pitch/TransposeFine/Manual', 0) / 100,
                gain=gain * _v(part, 'Volume', 1.0) * 10 ** (_v(sim, 'VolumeAndPan/Volume/Manual', 0) / 20),
                one_shot=_v(sim, 'Globals/PlaybackMode', 0) == 1,
                release=min(2.0, _v(sim, 'VolumeAndPan/Envelope/ReleaseTime/Manual', 50) / 1000),
                pan=float(np.clip(pan + _v(sim, 'VolumeAndPan/Panorama/Manual', 0), -1, 1)))


def render(als: Path, root: Path):
    ls = ET.fromstring(gzip.open(als).read()).find('LiveSet')
    tempo = _v(ls, './/MasterTrack/DeviceChain/Mixer/Tempo/Manual') or _v(ls, './/Tempo/Manual', 120)
    spb = 60.0 / tempo                               # seconds per beat (tempo changes ignored)
    roots = [als.parent, root]
    events, log = [], []                             # (sec, role, name, audio, gain, pan, vel)
    for t, chain in tracks_of(ls):
        name = _v(t, 'Name/EffectiveName', '', str)
        if not all(_v(x, 'DeviceChain/Mixer/Speaker/Manual', True, bool) for x in chain):
            continue                                 # muted, or in a muted group
        gain = float(np.prod([_v(x, 'DeviceChain/Mixer/Volume/Manual', 1.0) for x in chain]))
        pan = _v(t, 'DeviceChain/Mixer/Pan/Manual', 0.0)
        groups = [_v(x, 'Name/EffectiveName', '', str) for x in chain[1:]]
        if t.tag == 'MidiTrack':
            clips = [c for c in t.findall('.//ArrangerAutomation/Events/MidiClip')
                     if not _v(c, 'Disabled', False, bool)]
            if not clips:
                continue
            # voices: MIDI note -> (sound, label). A Drum Rack has one per pad; a Simpler track one for all notes.
            dg = t.find('.//DrumGroupDevice')
            voices, label_log = {}, []
            if dg is not None:
                for b in dg.findall('Branches/DrumBranch'):
                    rn = _v(b, 'BranchInfo/ReceivingNote', None)
                    if rn is None or not _v(b, 'MixerDevice/Speaker/Manual', True, bool):
                        continue                     # no note, or the pad is muted
                    pad = _v(b, 'Name/EffectiveName', '', str)
                    sim = b.find('.//OriginalSimpler')
                    if sim is None:
                        if role_of(pad, name) not in (None, 'rest', 'loop'):
                            label_log.append(f'  left out  {name!r} pad {pad!r}: not a sample (plugin or synth)')
                        continue
                    v = voice(sim, als, roots, [pad, None, name, *groups], gain * _v(b, 'MixerDevice/Volume/Manual', 1.0),
                              pan + _v(b, 'MixerDevice/Pan/Manual', 0.0), _v(b, 'BranchInfo/SendingNote', 60))
                    label_log.append(v.pop('log'))
                    if v.get('role'):
                        voices[128 - int(rn)] = v    # Live stores the pad's note as 128 - note
            else:
                sim = t.find('.//OriginalSimpler')
                if sim is None:
                    if t.find('.//PluginDevice') is not None or t.find('.//InstrumentVector') is not None:
                        if role_of(name, *groups) not in (None, 'rest', 'loop'):
                            log.append(f'  left out  {name!r}: drums from a plugin (can\'t play it here)')
                    continue
                v = voice(sim, als, roots, [name, None, *groups], gain, pan, None)
                label_log.append(v.pop('log'))
                if v.get('role'):
                    voices = {k: v for k in range(128)}
            log += [m for m in label_log if m]
            n_notes = {}
            for c in clips:
                notes = []
                for kt in c.findall('Notes/KeyTracks/KeyTrack'):
                    key = _v(kt, 'MidiKey', 60, int)
                    for n in kt.findall('Notes/MidiNoteEvent'):
                        if n.get('IsEnabled', 'true') == 'true':
                            notes.append((float(n.get('Time')), float(n.get('Duration')), key,
                                          float(n.get('Velocity', 100))))
                for cf, ct, a in spans(c):
                    for nt, dur, key, vel in notes:
                        v = voices.get(key)
                        if v is None or not cf <= nt < ct:
                            continue
                        sec = (a + nt - cf) * spb
                        x = repitch(v['snd'], (v['play'] if v['play'] is not None else key) - v['root'] + v['tr'])
                        if not v['one_shot']:            # classic mode: the note gates the sample
                            m = int((min(dur, ct - nt) * spb + v['release']) * SR)
                            if m < x.shape[1]:
                                x = x[:, :m].copy()
                                fd = min(m, int(max(v['release'], 0.005) * SR))
                                x[:, m - fd:] *= np.linspace(1, 0, fd, dtype=np.float32)
                        events.append((sec, v['role'], v['label'], x, v['gain'] * vel / 127, v['pan'], int(vel)))
                        n_notes[(v['role'], v['label'], v['file'])] = n_notes.get((v['role'], v['label'], v['file']), 0) + 1
            for (role, label, fname), k in n_notes.items():
                log.append(f'  {role:<11} {label!r} <- {fname} ({k} hits)')
        else:                                        # audio track
            clips = [c for c in t.findall('.//ArrangerAutomation/Events/AudioClip')
                     if not _v(c, 'Disabled', False, bool)]
            if not clips:
                continue
            fr0 = clips[0].find('SampleRef/FileRef')
            sname = Path(_v(fr0, 'RelativePath', '', str) or _v(fr0, 'Path', '', str)).name
            role = role_of(name, sname, *groups)
            if role is None or role == 'loop':
                log.append(f'  left out  {name!r} ({sname}): {"drum loop" if role else "unknown sound"}')
                continue
            n_clips = 0
            for c in clips:
                fr = c.find('SampleRef/FileRef')
                f = sample_file(fr, als, roots)
                if f is None:
                    msg = f'  missing   {name!r}: {Path(_v(fr, "RelativePath", "", str)).name} not found'
                    if msg not in log:
                        log.append(msg)
                    continue
                src, _ = load(f)
                warped = _v(c, 'IsWarped', False, bool)
                wm = sorted((float(w.get('BeatTime')), float(w.get('SecTime'))) for w in c.findall('WarpMarkers/WarpMarker'))
                g = gain * _v(c, 'SampleVolume', 1.0)
                for cf, ct, a in spans(c):
                    n = int((ct - cf) * spb * SR)
                    if n <= 0:
                        continue
                    if warped and len(wm) >= 2:      # content beat -> second in the file
                        cb = cf + np.arange(n) / SR / spb
                        bt, st = np.array([w[0] for w in wm]), np.array([w[1] for w in wm])
                        sec = np.interp(cb, bt, st)
                        lo, hi = cb < bt[0], cb > bt[-1]
                        sec[lo] = st[0] + (cb[lo] - bt[0]) * (st[1] - st[0]) / max(bt[1] - bt[0], 1e-9)
                        sec[hi] = st[-1] + (cb[hi] - bt[-1]) * (st[-1] - st[-2]) / max(bt[-1] - bt[-2], 1e-9)
                    else:                            # unwarped: markers are seconds, plays at its own speed
                        sec = cf + np.arange(n) / SR
                    pos = sec * SR
                    ok = (pos >= 0) & (pos < src.shape[1] - 1)
                    if not ok.any():
                        continue
                    x = np.zeros((2, n), np.float32)
                    for ch in range(2):
                        x[ch, ok] = np.interp(pos[ok], np.arange(src.shape[1]), src[ch])
                    fd = min(n, int(0.003 * SR))       # no clicks at clip edges
                    x[:, :fd] *= np.linspace(0, 1, fd, dtype=np.float32)
                    x[:, n - fd:] *= np.linspace(1, 0, fd, dtype=np.float32)
                    events.append((a * spb, role, name, x, g, pan, 100))
                    n_clips += 1
            if n_clips:
                log.append(f'  {role:<11} {name!r} <- audio clips ({n_clips})')
    if not events:
        return None, log + ['  no playable drum tracks']
    end = min(MAX_SEC, max(e[0] for e in events) + 3)
    n_out = int(end * SR)
    out = {r: np.zeros((2, n_out), np.float32) for r in ROLES}
    groove = []
    for sec, role, name, x, g, pn, vel in sorted(events, key=lambda e: e[0]):
        if sec >= end:
            continue
        i = int(sec * SR)
        seg = x[:, :n_out - i]
        out[role][0, i:i + seg.shape[1]] += seg[0] * g * min(1, 1 - pn)
        out[role][1, i:i + seg.shape[1]] += seg[1] * g * min(1, 1 + pn)
        if seg.shape[1] < 4 * SR:                    # hits, not long loops, go in the groove answer key
            groove.append(dict(t=round(sec, 4), role=role, sound=name, vel=vel))
    meta = dict(tempo=float(tempo), seconds=round(end, 2),
                live_version=ET.fromstring(gzip.open(als).read()).get('Creator', ''),
                hits={r: sum(1 for h in groove if h['role'] == r) for r in ROLES},
                loud={r: round(float(20 * np.log10(np.sqrt(np.mean(out[r] ** 2)) + 1e-12)), 1) for r in ROLES})
    return (out, groove, meta), log


GENERIC_DIR = re.compile(r'^\W*(ableton( live)?( \d+)?|project( files?)?|projects|bonus project|als|live set)\W*$', re.I)


def project_name(als, root):
    """The folder's name, unless it's a generic one ('Ableton Live 11', 'Project File'): then the file's."""
    d = als.parent
    return als.stem if d == root or GENERIC_DIR.match(d.name) else d.name


def pick_projects(root):
    """One .als per project folder: '[Merged Drums]' first, else the shortest name; no backups. The same
    file in two copies of a pack (same name and size) is used once."""
    by_dir, seen = {}, set()
    for d, _, fs in os.walk(root, followlinks=True):
        if 'backup' in d.lower():
            continue
        for f in fs:
            if f.lower().endswith('.als') and not f.startswith('._'):
                by_dir.setdefault(Path(d), []).append(Path(d) / f)
    for d in sorted(by_dir):
        best = sorted(by_dir[d], key=lambda p: ('merged drums' not in p.name.lower(), len(p.name), p.name))[0]
        key = (best.name.lower(), best.stat().st_size)
        if key not in seen:
            seen.add(key)
            yield best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', required=True, help='training copies go here (one folder per project)')
    ap.add_argument('--midi', required=True, help='grooves (MIDI + JSON answer keys) go here')
    ap.add_argument('--only', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    root, out, gdir = Path(a.root), Path(a.out), Path(a.midi)
    out.mkdir(parents=True, exist_ok=True)
    gdir.mkdir(parents=True, exist_ok=True)
    tried_f = out / '_tried_als.txt'
    tried = set(tried_f.read_text().splitlines()) if tried_f.exists() and not a.force else set()
    seen = set()
    for als in pick_projects(root):
        if a.only and a.only.lower() not in str(als).lower():
            continue
        name = project_name(als, root)
        if name in seen:
            name = f'{name} - {als.stem}'
        seen.add(name)
        dest = out / f'{name} (ALS)'
        if str(als) in tried:
            continue
        print(f'\n## {name}  [{als.name}]')
        if dest.exists() and not a.force:
            print('  already rendered (use --force to redo)')
            continue
        try:
            res, log = render(als, root)
        except Exception as e:
            print(f'  could not read: {type(e).__name__}: {e}')
            with open(tried_f, 'a') as fh:
                fh.write(f'{als}\n')
            continue
        print('\n'.join(log))
        if res is None:
            with open(tried_f, 'a') as fh:
                fh.write(f'{als}\n')
            continue
        tracks, groove, meta = res
        print(f"  {meta['tempo']:g} bpm, {meta['seconds']} s, hits {meta['hits']}")
        if max(meta['loud'][r] for r in ('kick', 'snare', 'hh')) < -70:
            print('  -> no kick, snare or hats: not saved')
            with open(tried_f, 'a') as fh:
                fh.write(f'{als}\n')
            continue
        peak = max(np.abs(sum(tracks.values())).max(), 1e-9)
        g = min(1.0, 0.9 / peak)
        dest.mkdir(parents=True, exist_ok=True)
        for r, x in tracks.items():
            sf.write(str(dest / f'{r}.flac'), (x * g).T, SR, subtype='PCM_24')
        (dest / 'trainbeat.json').write_text(json.dumps(dict(source=str(als.relative_to(root)), **meta)))
        write_midi(groove, meta['tempo'], gdir / f'{name}.mid')
        (gdir / f'{name}.json').write_text(json.dumps(dict(meta=meta, hits=groove)))
        print('  -> training copy saved')


if __name__ == '__main__':
    main()
