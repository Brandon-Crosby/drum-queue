#!/usr/bin/env python3
"""kit.py - drum stems in, finished two-version kit out.

  make  STEMS_DIR OUT_DIR INDEX_DIR LIB_ROOT [--name NAME] [--midi GROOVE.mid] [--include-matches]
  eval  TRAIN_DIR GROOVE_DIR LOG.txt INDEX_DIR [--n 20]
  realtest SPLITS_DIR INDEX_DIR [--out KITS_DIR]   real songs: Helyx's split vs the true stems

STEMS_DIR may also be a batch_eval listen folder (ours_kick.wav ... / truth_kick.wav ...).

STEMS_DIR holds kick / snare / hh / other_drums .flac (what the splitter writes). For each stem the
hit cutter (hits.py) finds the distinct sounds; each is looked up in the fingerprint index
(soundmatch.py), trying the sound at a few pitches since producers retune samples. The kit:

  Kit - NAME/
    Kicks/Original/Kick 1.wav            the sound as cut from the song
    Kicks/Matched/Kick 1 - <file>.wav    the library's file (only with --include-matches)
    ... Snares & Claps, Hats, Percussion
    MIDI/NAME groove.mid                 (with --midi)
    kit.json                             every sound: hits, match, how sure, pitch

How sure: 'exact' = almost certainly the same sample, 'close' = a similar sound, 'none' = nothing
near it. Matched files are included only with --include-matches, for personal use; a product build
ships only files marked yours/licensed in the catalog (not built yet).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hits as H            # noqa: E402
import soundmatch as M      # noqa: E402

ROLES = [('kick', 'Kicks', 'Kick'), ('snare', 'Snares & Claps', 'Snare'), ('hh', 'Hats', 'Hat'),
         ('other_drums', 'Percussion', 'Perc')]
SHIFTS = [0, -1, 1, -2, 2, -3, 3]       # semitones tried (retuned samples); wider shifts mostly found look-alikes
SHIFT_COST = 0.004                       # similarity given up per semitone, so an unshifted match wins a tie
EXACT, CLOSE = 0.995, 0.95       # on rebuilt projects: >= 0.995 was the same sample 90% of the time


def shift(x, semis):
    """Repitch the way a sampler does (speed change)."""
    if semis == 0:
        return x
    r = 2 ** (semis / 12)
    n = int(len(x) / r)
    return np.interp(np.arange(n) * r, np.arange(len(x)), x).astype(np.float32)


def lookup(V, R, x_mono_22k, role, k=5):
    """Best library match over the pitches in SHIFTS. Returns [(index, similarity, semis)] best first."""
    best = {}
    for s in SHIFTS:
        v = M.fingerprint(shift(x_mono_22k, -s))   # the song plays the sample s semitones up: undo it
        if v is None:
            continue
        top, sims = M.search(V, R, v, role, k=k)
        for t, sim in zip(top, sims):
            sim = float(sim) - SHIFT_COST * abs(s)
            if t not in best or sim > best[t][0]:
                best[t] = (sim, s)
    return sorted(((t, sim, s) for t, (sim, s) in best.items()), key=lambda r: -r[1])[:k]


def to22k(seg):
    mono = seg.mean(1) if seg.ndim == 2 else seg
    n = int(len(mono) * M.SR / H.SR)
    return np.interp(np.arange(n) * H.SR / M.SR, np.arange(len(mono)), mono).astype(np.float32)


def confidence(sim):
    return 'exact' if sim >= EXACT else 'close' if sim >= CLOSE else 'none'


ALIASES = {'kick': ['kick'], 'snare': ['snare'], 'hh': ['hh'], 'other_drums': ['other_drums', 'toms', 'perc']}


def find_stem(stems, role, prefix=''):
    """kick.flac (splitter output), or ours_kick.wav / truth_kick.wav (batch_eval listen folders)."""
    for a in ALIASES[role]:
        for ext in ('.flac', '.wav'):
            f = stems / f'{prefix}{a}{ext}'
            if f.exists():
                return f
    return None


def make(stems, out, index_dir, lib_root, name=None, midi=None, include=False, V=None, R=None, P=None):
    name = name or stems.name.replace(' (FLP)', '').replace(' (ALS)', '')
    kit = out / f'Kit - {name}'
    if V is None:
        V, R, P, _, _ = M.load_index(index_dir)
    report = dict(name=name, sounds=[])
    for role, folder, label in ROLES:
        f = find_stem(stems, role) or find_stem(stems, role, 'ours_')
        if f is None:
            continue
        x, on, lab, sounds = H.sounds(f)
        for k, s in enumerate(sounds):
            seg = x[s['start']:s['end']].copy()
            fd = min(len(seg), int(0.01 * H.SR))
            seg[len(seg) - fd:] *= np.linspace(1, 0, fd)[:, None]
            o = kit / folder / 'Original' / f'{label} {k + 1}.wav'
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(o), seg, H.SR, subtype='PCM_24')
            top = lookup(V, R, to22k(seg), role)
            entry = dict(role=role, file=str(o.relative_to(kit)), hits=s['hits'], start_s=round(s['start'] / H.SR, 3))
            if top:
                t, sim, semis = top[0]
                entry.update(match=str(P[t]), similarity=round(sim, 3), pitch=semis, confidence=confidence(sim),
                             alternatives=[dict(match=str(P[a]), similarity=round(b, 3), pitch=c) for a, b, c in top[1:4]])
                if include and sim >= CLOSE:
                    m = kit / folder / 'Matched' / f'{label} {k + 1} - {Path(P[t]).name}'
                    m.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(lib_root / P[t], m)
                    entry['matched_file'] = str(m.relative_to(kit))
            report['sounds'].append(entry)
    if midi and Path(midi).exists():
        (kit / 'MIDI').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(midi, kit / 'MIDI' / f'{name} groove.mid')
    kit.mkdir(parents=True, exist_ok=True)
    (kit / 'kit.json').write_text(json.dumps(report, indent=1))
    return kit, report


# ---------- checking the matcher on rebuilt projects
def channel_samples(log):
    """{project: {channel: sample file name}} from flp_render's log."""
    out, cur = {}, None
    for line in Path(log).read_text(errors='replace').splitlines():
        m = re.match(r'## (.+?)\s+\[.*\]$', line)
        if m:
            cur = out.setdefault(m.group(1).strip(), {})
            continue
        m = re.match(r"\s+(kick|snare|hh|other_drums|rest)\s+'(.+)' <- (.+)$", line)
        if m and cur is not None:
            cur[m.group(2)] = m.group(3).strip()
    return out


def norm(name):
    return re.sub(r'\W+', '', Path(name).stem.lower())


def evaluate(train, grooves, log, index_dir, n=20):
    V, R, P, grp, _ = M.load_index(index_dir)
    names = np.array([norm(p) for p in P])
    samples = channel_samples(log)
    rows = []
    projs = [p for p in sorted(train.iterdir()) if p.is_dir() and p.name.replace(' (FLP)', '') in samples
             and (grooves / (p.name.replace(' (FLP)', '') + '.json')).exists()][:n]
    for p in projs:
        pname = p.name.replace(' (FLP)', '')
        truth = json.loads((grooves / f'{pname}.json').read_text())['hits']
        for role, _, _ in ROLES:
            f = p / f'{role}.flac'
            if not f.exists():
                continue
            times = {}
            for h in truth:
                if h['role'] == role:
                    times.setdefault(round(h['t'], 3), set()).add(h['sound'])
            if not times:
                continue
            tt = np.array(sorted(times))
            x, on, lab, sounds = H.sounds(f)
            for s in sounds:
                ks = np.flatnonzero(lab == s['group'])
                chans = []
                for k in ks:
                    q = int(np.argmin(np.abs(tt - on[k] / H.SR)))
                    if abs(tt[q] - on[k] / H.SR) <= 0.02 and len(times[tt[q]]) == 1:
                        chans.append(next(iter(times[tt[q]])))
                if not chans:
                    continue
                ch = max(set(chans), key=chans.count)
                smp = samples[pname].get(ch)
                if not smp:
                    continue
                in_lib = np.flatnonzero(names == norm(smp))
                seg = x[s['start']:s['end']]
                top = lookup(V, R, to22k(seg), role)
                if not top:
                    continue
                tv = V[in_lib]                                       # the true sample's own fingerprint(s)
                same = lambda t: names[t] == norm(smp) or bool(len(in_lib)) and (grp[t] in set(grp[in_lib]) or float((tv @ V[t]).max()) >= 0.995)
                rows.append(dict(project=pname, role=role, sample=smp, in_lib=bool(len(in_lib)),
                                 top1=bool(same(top[0][0])), top5=any(same(t) for t, _, _ in top),
                                 sim=top[0][1], pitch=top[0][2], got=Path(P[top[0][0]]).name))
        print(f'done {pname}', flush=True)
    inl = [r for r in rows if r['in_lib']]
    print(f'\n{len(rows)} sounds cut from {len(projs)} projects; the true sample is in the library for {len(inl)}')
    if inl:
        print(f"exact sample found first: {100 * np.mean([r['top1'] for r in inl]):.1f}%, in top 5: {100 * np.mean([r['top5'] for r in inl]):.1f}%")
        for role, _, _ in ROLES:
            rr = [r for r in inl if r['role'] == role]
            if rr:
                print(f"  {role:12} first {100 * np.mean([r['top1'] for r in rr]):5.1f}%  top-5 {100 * np.mean([r['top5'] for r in rr]):5.1f}%  (n={len(rr)})")
        ok = [r['sim'] for r in inl if r['top1']]
        bad = [r['sim'] for r in inl if not r['top1']]
        print('similarity when right:', np.round(np.percentile(ok, [10, 50, 90]), 3) if ok else '-',
              '| when wrong:', np.round(np.percentile(bad, [10, 50, 90]), 3) if bad else '-')
        for cut in (0.95, 0.9, 0.85, 0.8, 0.75):
            sel = [r for r in inl if r['sim'] >= cut]
            if sel:
                print(f'  sim >= {cut}: {len(sel)} sounds, {100 * np.mean([r["top1"] for r in sel]):.0f}% right')
        print('pitch shifts on right matches:', sorted({r['pitch'] for r in inl if r['top1']}))
    Path('kit_eval.json').write_text(json.dumps(rows, indent=1))


# ---------- real songs: what survives Helyx's split
def _cut(f):
    """Distinct sounds in a stem: [(fingerprint, hits, segment)] plus all onset times in seconds."""
    if f is None:
        return [], np.array([])
    x, on, lab, snd = H.sounds(f)
    out = []
    for s in snd:
        seg = x[s['start']:s['end']]
        v = M.fingerprint(to22k(seg))
        if v is not None:
            out.append((v, s['hits'], seg))
    return out, on / H.SR


def realtest(splits, index_dir, kits=None, tol=0.03):
    """For each beat (a batch_eval --listen folder): cut sounds from the true stem and from Helyx's stem.
    found     = share of true hits (by count) that Helyx's stem also has within 30 ms
    sounds    = share of the true distinct sounds, weighted by how often they play, that have a twin
                (>= CLOSE) among Helyx's sounds
    fake      = Helyx sounds with no true twin in that stem (bleed, smears, split hits)
    same_match= of true sounds whose library match is 'exact', share whose Helyx twin finds the same file
    """
    V, R, P, _, _ = M.load_index(index_dir)
    beats = sorted(d for d in splits.iterdir() if d.is_dir() and any(d.glob('ours_*.wav')))
    rows = []
    for b in beats:
        for role, _, _ in ROLES:
            tr, t_on = _cut(find_stem(b, role, 'truth_'))
            ou, o_on = _cut(find_stem(b, role, 'ours_'))
            if not tr:
                continue
            found = np.mean([np.abs(o_on - t).min() <= tol if len(o_on) else False for t in t_on])
            Ov = np.stack([v for v, _, _ in ou]) if ou else np.zeros((0, len(tr[0][0])))
            w = np.array([h for _, h, _ in tr], float)
            best = [(Ov @ v).max() if len(Ov) else 0 for v, _, _ in tr]
            twin = [int(np.argmax(Ov @ v)) if len(Ov) else -1 for v, _, _ in tr]
            got = np.array(best) >= CLOSE
            Tv = np.stack([v for v, _, _ in tr])
            fake = sum((Tv @ v).max() < CLOSE for v, _, _ in ou)
            same, tot = 0, 0
            for (v, h, seg), ok, j in zip(tr, got, twin):
                t1 = lookup(V, R, to22k(seg), role, k=1)
                if not t1 or t1[0][1] < EXACT:
                    continue
                tot += 1
                if ok:
                    o1 = lookup(V, R, to22k(ou[j][2]), role, k=1)
                    same += bool(o1) and o1[0][0] == t1[0][0]
            rows.append(dict(beat=b.name, role=role, true_hits=len(t_on), found=round(float(found), 3),
                             true_sounds=len(tr), ours_sounds=len(ou), sounds=round(float((w * got).sum() / w.sum()), 3),
                             fake=int(fake), exact_true=tot, same_match=same))
            r = rows[-1]
            print(f"{b.name[:24]:24} {role:11} hits {r['found']:5.0%} of {r['true_hits']:4d}   sounds {r['sounds']:5.0%} "
                  f"({r['true_sounds']} true, {r['ours_sounds']} ours, {r['fake']} fake)   same file {same}/{tot}", flush=True)
        if kits:
            make(b, kits, index_dir, Path('/'), name=b.name, V=V, R=R, P=P)
    print('\nOVERALL')
    for role, _, _ in ROLES:
        rs = [r for r in rows if r['role'] == role]
        if not rs:
            continue
        th = sum(r['true_hits'] for r in rs)
        print(f"  {role:11} hits found {sum(r['found'] * r['true_hits'] for r in rs) / th:5.0%}   "
              f"sounds kept {np.mean([r['sounds'] for r in rs]):5.0%}   fake sounds {sum(r['fake'] for r in rs):3d} "
              f"(of {sum(r['ours_sounds'] for r in rs)})   same library file {sum(r['same_match'] for r in rs)}/"
              f"{sum(r['exact_true'] for r in rs)}")
    Path('realtest.json').write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['make', 'eval', 'realtest'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--name')
    ap.add_argument('--midi')
    ap.add_argument('--include-matches', action='store_true')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'make':
        stems, out, idx, lib = map(Path, a.args[:4])
        kit, rep = make(stems, out, idx, lib, a.name, a.midi, a.include_matches)
        for s in rep['sounds']:
            print(f"{s['file']:34} {s['hits']:4d} hits  {s.get('confidence', '-'):6} {s.get('similarity', 0):.3f}  "
                  f"pitch {s.get('pitch', 0):+d}  {Path(s.get('match', '')).name}")
        print('->', kit)
    elif a.cmd == 'realtest':
        realtest(Path(a.args[0]), Path(a.args[1]), Path(a.out) if a.out else None)
    else:
        train, grooves, log, idx = map(Path, a.args[:4])
        evaluate(train, grooves, log, idx, a.n)


if __name__ == '__main__':
    main()
