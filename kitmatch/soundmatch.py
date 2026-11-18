#!/usr/bin/env python3
"""soundmatch.py - find the library one-shot a separated drum hit most likely came from.

  list   ROOT LIST.tsv              find drum one-shots under ROOT (role from the file name, then its
                                    folders: 'Kicks/x.wav'); skips loops, MIDI, vocals, stems
  index  LIST.tsv ROOT OUT_DIR      fingerprint every one-shot in LIST (role<TAB>path relative to ROOT);
                                    resumable: rerun until it says done
  test   OUT_DIR ROOT [--n 500]     self-test: dirty copies of random library sounds -> how often the
                                    right file comes back first / in the top 5
  query  OUT_DIR HIT.wav [--role kick]

Fingerprint: the first ~0.5 s after the hit starts, as a 64-band log-mel picture (pitch- and
shape-sensitive, so an exact re-use of a sample lands on that sample), normalized so loudness
doesn't matter. Matching is cosine similarity, searched only within the hit's role.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
SEC = 0.5
N_FFT, HOP, N_MELS = 1024, 512, 64
N_FR = int(SEC * SR) // HOP + 1
SHARD = 4000


def _mel_fb():
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(30), hz2mel(SR / 2), N_MELS + 2))
    bins = np.floor((N_FFT + 1) * pts / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            fb[m - 1, k] = (k - a) / max(b - a, 1)
        for k in range(b, c):
            fb[m - 1, k] = (c - k) / max(c - b, 1)
    return fb


FB = _mel_fb()
WIN = np.hanning(N_FFT)


def load(path, sec=SEC + 0.3):
    info = sf.info(str(path))
    n = int(min(info.frames, sec * 1.5 * info.samplerate + 2048))
    x, sr = sf.read(str(path), frames=n, dtype='float32', always_2d=True)
    x = x.mean(1)
    if sr != SR:
        t = np.arange(0, len(x) * SR / sr) * sr / SR
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


def onset(x):
    """Start of the hit: first sample within 30 dB of the peak (skips silence before it)."""
    a = np.abs(x)
    if a.max() < 1e-5:
        return None
    i = int(np.argmax(a > a.max() * 0.0316))
    return max(0, i - int(0.002 * SR))


def fingerprint(x):
    i = onset(x)
    if i is None:
        return None
    x = x[i:i + int(SEC * SR) + N_FFT]
    x = np.pad(x, (0, max(0, int(SEC * SR) + N_FFT - len(x))))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:N_FR]
    spec = np.abs(np.fft.rfft(frames * WIN, axis=1)) ** 2
    mel = np.log10(spec @ FB.T + 1e-10)                      # [frames, mels]
    mel = np.maximum(mel, mel.max() - 3)                     # 30 dB floor: bleed and music sit below it
    v = (mel - mel.mean()).ravel()
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float16)


# ---------- list
AUDIO = ('.wav', '.aif', '.aiff', '.flac', '.mp3')
LOOPY = r'\bloops?\b|\bbpm\b|\d{2,3} ?bpm|\bbreaks?\b|\bfill|\bmidi\b|construction kit|\bstems?\b|\bvox\b|vocal|melod|\bchord|\bbass ?line|preset'
SKIP_DIRS = ('.Spotlight-V100', '.Trashes', '.fseventsd', 'com.apple.MobileSMS', '90ssamplecds')


def list_shots(root, out_tsv, max_mb=2):
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'eval'))
    from autosort import by_name
    loopy, keep = re.compile(LOOPY, re.I), []
    for d, dirs, files in os.walk(root):
        rel = os.path.relpath(d, root)
        if rel.split(os.sep)[0] in SKIP_DIRS or rel.split(os.sep)[0].startswith(('drumsplit-', 'DrumSplits')):
            dirs[:] = []
            continue
        for f in files:
            if f.startswith('._') or not f.lower().endswith(AUDIO):
                continue
            p = os.path.join(rel, f) if rel != '.' else f
            if loopy.search(p) or os.path.getsize(os.path.join(d, f)) > max_mb * 2 ** 20:
                continue
            role = by_name(f)
            if role not in ('kick', 'snare', 'hh', 'other_drums'):
                for part in reversed(p.split(os.sep)[:-1][-3:]):
                    r = by_name(part)
                    if r in ('kick', 'snare', 'hh', 'other_drums'):
                        role = r
                        break
            if role in ('kick', 'snare', 'hh', 'other_drums'):
                keep.append(f'{role}\t{p}')
    Path(out_tsv).write_text('\n'.join(keep))
    print(f'{len(keep)} drum one-shots listed')


# ---------- index
def index(list_tsv, root, out):
    out.mkdir(parents=True, exist_ok=True)
    items = [l.split('\t', 1) for l in open(list_tsv).read().splitlines() if '\t' in l]
    shards = sorted(out.glob('shard_*.npz'))
    done = sum(len(np.load(s, allow_pickle=True)['paths']) for s in shards)  # files handled (incl. failed)
    start, t0 = len(shards) * SHARD, time.time()
    for s0 in range(start, len(items), SHARD):
        part = items[s0:s0 + SHARD]
        vecs, roles, paths = [], [], []
        for role, p in part:
            try:
                v = fingerprint(load(root / p))
            except Exception:
                v = None
            vecs.append(v if v is not None else np.zeros(N_FR * N_MELS, np.float16))
            roles.append(role if v is not None else '')
            paths.append(p)
        np.savez(out / f'shard_{s0 // SHARD:04d}.npz', vecs=np.stack(vecs), roles=np.array(roles), paths=np.array(paths, dtype=object))
        print(f'{min(s0 + SHARD, len(items))}/{len(items)} ({time.time() - t0:.0f} s)', flush=True)
    print('done')


def load_index(out):
    V, R, P = [], [], []
    for s in sorted(out.glob('shard_*.npz')):
        z = np.load(s, allow_pickle=True)
        V.append(z['vecs']); R.append(z['roles']); P.append(z['paths'])
    V, R, P = np.concatenate(V).astype(np.float32), np.concatenate(R), np.concatenate(P)
    ok = R != ''
    V, R, P = V[ok], R[ok], P[ok]
    # the same sample sits in many packs: collapse exact duplicates so a match isn't split between copies
    key = np.round(V[:, ::7] * 512).astype(np.int16)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V, R, P, inv.ravel(), first


def search(V, R, vec, role=None, k=5):
    m = np.ones(len(V), bool) if role is None else (R == role)
    idx = np.flatnonzero(m)
    sims = V[idx] @ vec.astype(np.float32)
    top = idx[np.argsort(-sims)[:k]]
    return top, V[top] @ vec.astype(np.float32)


# ---------- self-test
def dirty(x, others, rng):
    """A hit as it comes out of separation: bleed from other drums + music, EQ, level, a bit of clip."""
    i = onset(x)
    x = x[i:i + int(1.0 * SR)].copy()
    x /= np.abs(x).max() + 1e-9
    for o in others:                                        # other drums hitting at the same time
        j = onset(o)
        o = o[j:j + len(x)] if j is not None else o[:len(x)]
        o = np.pad(o, (0, len(x) - len(o)))
        x += o / (np.abs(o).max() + 1e-9) * 10 ** (rng.uniform(-24, -10) / 20) * rng.uniform(0.5, 1)
    noise = np.cumsum(rng.standard_normal(len(x))) * 0.002   # music bed (red-ish noise)
    x += noise / (np.abs(noise).max() + 1e-9) * 10 ** (rng.uniform(-30, -18) / 20)
    if rng.random() < 0.5:                                  # tone shaping
        from scipy.signal import butter, lfilter
        b, a = butter(1, rng.uniform(3000, 9000) / (SR / 2), 'low') if rng.random() < 0.5 else \
            butter(1, rng.uniform(40, 150) / (SR / 2), 'high')
        x = lfilter(b, a, x)
    if rng.random() < 0.4:
        d = rng.uniform(1.5, 4)
        x = np.tanh(x * d) / np.tanh(d)
    return (x * rng.uniform(0.2, 1)).astype(np.float32)


def selftest(out, root, n, seed=0):
    V, R, P, grp, _ = load_index(out)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(V), size=min(n, len(V)), replace=False)
    hit1 = hit5 = 0
    per = {}
    for c, q in enumerate(picks):
        x = load(root / P[q], sec=1.2)
        others = [load(root / P[j], sec=1.2) for j in rng.choice(len(V), size=rng.integers(0, 3), replace=False)
                  if R[j] != R[q]]
        v = fingerprint(dirty(x, others, rng))
        if v is None:
            continue
        top, _ = search(V, R, v, R[q], k=5)
        ok1 = grp[top[0]] == grp[q]
        ok5 = any(grp[t] == grp[q] for t in top)
        hit1 += ok1; hit5 += ok5
        a = per.setdefault(R[q], [0, 0, 0]); a[0] += ok1; a[1] += ok5; a[2] += 1
    tot = len(picks)
    print(f'library: {len(V)} one-shots ({grp.max() + 1} unique after removing duplicates across packs)')
    print(f'self-test on {tot} dirty hits: exact sample first {100 * hit1 / tot:.1f}%, in top 5 {100 * hit5 / tot:.1f}%')
    for r, (a, b, t) in sorted(per.items()):
        print(f'  {r:12} first {100 * a / t:5.1f}%  top-5 {100 * b / t:5.1f}%  (n={t})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'index', 'test', 'query'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--role', default=None)
    a = ap.parse_args()
    if a.cmd == 'list':
        list_shots(Path(a.args[0]), Path(a.args[1]))
    elif a.cmd == 'index':
        index(Path(a.args[0]), Path(a.args[1]), Path(a.args[2]))
    elif a.cmd == 'test':
        selftest(Path(a.args[0]), Path(a.args[1]), a.n)
    else:
        V, R, P, grp, _ = load_index(Path(a.args[0]))
        v = fingerprint(load(Path(a.args[1])))
        top, sims = search(V, R, v, a.role, k=10)
        for t, s in zip(top, sims):
            print(f'{s:.3f}  {R[t]:12} {P[t]}')


if __name__ == '__main__':
    main()
