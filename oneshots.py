"""oneshots.py - turns split kit stems into a small drum kit of one-shot samples.

For each kit piece (kick, snare, toms, hh, ride, crash):
  1. find every hit in its stem (onset detection)
  2. cut each hit from its start until it dies away or the next hit starts
  3. score the hits: loud, rings out before the next hit, and little else playing at that moment
     (the other stems are quiet), so the picked sample has as little bleed as possible
  4. group hits that sound alike, so two different snares become "Snare 1" and "Snare 2"
     instead of forty copies of the same one
  5. write the best hit of each group as a 24-bit WAV, faded and normalized

  python3 oneshots.py SPLIT_DIR [OUT_DIR]     (SPLIT_DIR holds kick.wav, snare.wav, ...)
"""
import sys
from pathlib import Path

import numpy as np

KIT = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash']
LABEL = {'kick': 'Kick', 'snare': 'Snare', 'toms': 'Tom', 'hh': 'Hat', 'ride': 'Ride', 'crash': 'Crash'}
MAX_LEN = {'kick': 1.2, 'snare': 1.0, 'toms': 1.4, 'hh': 0.6, 'ride': 2.0, 'crash': 3.0}   # seconds
MAX_SOUNDS = {'kick': 6, 'snare': 6, 'toms': 4, 'hh': 4, 'ride': 2, 'crash': 3}
SAME = 5.0          # dB: average attack difference under which two hits count as the same sample (tuned on TVS)
MIN_HITS = 3        # a sound has to come back at least this often (one-offs are usually bleed)
SR = 44100
MAX_BLEED = -6.0    # dB: skip hits where the other drums + bass are within 6 dB of the hit


def _load(path):
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=False)
    y = np.atleast_2d(y)
    return y                                     # (channels, samples)


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


ATTACK = 0.06       # seconds: the click + punch that identifies a sample, before tails/808s/bleed smear it


def _features(seg):
    """Fingerprint of a hit's attack (how its spectrum moves over the first 60 ms), for grouping
    hits of the same sample. Loudness is removed, so a softer hit of the same kick still matches."""
    import librosa
    n = int(ATTACK * SR)
    a = np.pad(seg[:n], (0, max(0, n - len(seg))))
    mel = librosa.feature.melspectrogram(y=a, sr=SR, n_fft=1024, hop_length=256, n_mels=32)
    return np.maximum(librosa.power_to_db(mel, ref=np.max), -50)


def hits(piece, y, others):
    """All usable hits in one stem. y: (ch, n). others: mono sum of every other stem (bleed meter)."""
    import librosa
    mono = y.mean(axis=0)
    peak_all = np.abs(mono).max()
    if peak_all < 1e-4:
        return []
    hop = 256
    env = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=hop)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, backtrack=True,
                                    units='samples', delta=0.08, wait=int(0.05 * SR / hop))
    on = np.asarray(on, dtype=int)
    out = []
    maxlen = int(MAX_LEN[piece] * SR)
    for i, s in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else len(mono)
        seg = mono[s:min(nxt, s + maxlen)]
        if len(seg) < int(0.04 * SR):
            continue
        pk = np.abs(seg[:int(0.05 * SR)]).max()
        if pk < peak_all * 0.18:                 # ~-15 dB under the stem's loudest hit: ghost note / bleed
            continue
        # where it dies away: envelope falls 45 dB under its peak
        frame = 512
        n_fr = len(seg) // frame
        if n_fr < 2:
            continue
        fr = _db(np.sqrt(np.mean(seg[:n_fr * frame].reshape(n_fr, frame) ** 2, axis=1)))
        top = fr.max()
        dead = np.where(fr < top - 45)[0]
        dead = dead[dead > np.argmax(fr)]
        end = int((dead[0] + 1) * frame) if len(dead) else len(seg)
        end = max(end, int(0.06 * SR))
        cut_short = end >= len(seg) - frame and nxt - s < maxlen    # next hit cut its tail off
        # bleed meter: how loud everything else is during the attack, relative to this hit
        w = slice(s, s + int(0.12 * SR))
        bleed = _db(_rms(others[w]) + 1e-9) - _db(_rms(mono[w]) + 1e-9)
        out.append(dict(start=int(s), end=int(s + end), peak=float(pk / peak_all),
                        ring=(nxt - s) / SR, bleed=float(bleed), cut_short=bool(cut_short),
                        fp=_features(mono[s:s + end])))
    return out


def score(h, piece):
    want = MAX_LEN[piece] * 0.6                  # enough room to ring out
    ring = min(h['ring'], want) / want
    return 1.0 * h['peak'] + 0.8 * ring - 0.04 * max(h['bleed'], -30) - (0.3 if h['cut_short'] else 0)


def same_sound(a, b, max_lag=256):
    """Peak normalized cross-correlation of two hits' attacks: ~1.0 = the same sample."""
    n = min(len(a), len(b), int(0.2 * SR))
    a, b = a[:n], b[:n]
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    c = np.correlate(np.pad(a, max_lag), b, 'valid')
    return float(np.abs(c).max() / den)


def pick(piece, hs, mono=None):
    """Group alike hits and return the best hit of each group (largest groups first)."""
    if not hs:
        return []
    hs = sorted(hs, key=lambda h: -score(h, piece))
    fps = np.array([h['fp'] for h in hs])
    # simple leader clustering on timbre: a hit joins the closest group within the threshold
    leaders, members = [], []
    for i, f in enumerate(fps):
        d = [np.mean(np.abs(f - fps[l])) for l in leaders]
        if d and min(d) < SAME:
            members[int(np.argmin(d))].append(i)
        else:
            leaders.append(i)
            members.append([i])
    groups = [(leaders[k], m) for k, m in enumerate(members) if len(m) >= MIN_HITS]
    groups.sort(key=lambda g: -len(g[1]))
    # Bleed can make one sample look like two groups; if two picks are the same waveform, keep one.
    kept = []
    for l, m in groups:
        h = dict(hs[l], count=len(m))
        if mono is not None:
            cut = mono[h['start']:h['end']]
            dup = next((k for k in kept if same_sound(cut, mono[k['start']:k['end']]) > 0.9), None)
            if dup:
                dup['count'] += h['count']
                continue
        kept.append(h)
        if len(kept) == MAX_SOUNDS[piece]:
            break
    return kept


def write(y, h, path):
    import soundfile as sf
    seg = y[:, h['start']:h['end']].copy()
    n = seg.shape[1]
    fin, fout = min(int(0.002 * SR), n // 4), min(int(0.03 * SR), n // 3)
    seg[:, :fin] *= np.linspace(0, 1, fin)
    seg[:, n - fout:] *= np.linspace(1, 0, fout) ** 2
    pk = np.abs(seg).max()
    if pk > 0:
        seg *= 10 ** (-1 / 20) / pk              # peak at -1 dBFS
    sf.write(str(path), seg.T, SR, subtype='PCM_24')


def make_kit(split_dir, out_dir, prefix='', stems=None, log=lambda line: None):
    """split_dir: folder with kick.wav, snare.wav, ... (and optionally bass/vocals/other).
    log(line): progress lines with the real numbers, shown to the visitor while it works.
    Returns [{name, file, piece, label, count, bleed_db, seconds}] for each one-shot written."""
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {p.stem: p for p in split_dir.glob('*.wav')}
    if stems:
        found.update({k: Path(v) for k, v in stems.items()})
    # Bleed only comes from what plays at the same time and leaks into a drum stem: the other kit
    # pieces and the bass/808. Vocals and chords don't count against a hit.
    log('Loading kit stems · 44.1 kHz')
    audio = {k: _load(p) for k, p in found.items() if k in KIT or k == 'bass'}
    if not audio:
        return []
    n = max(a.shape[1] for a in audio.values())
    mono = {k: np.pad(a.mean(axis=0), (0, n - a.shape[1])) for k, a in audio.items()}
    total = sum(mono.values())
    result = []
    for piece in KIT:
        if piece not in audio:
            continue
        y = np.pad(audio[piece], ((0, 0), (0, n - audio[piece].shape[1])))
        name = LABEL[piece].lower().replace('hat', 'hi-hat')
        log(f'{name}: scanning for transients')
        found_hits = hits(piece, y, total - mono[piece])
        hs = [h for h in found_hits if h['bleed'] < MAX_BLEED]
        if not found_hits:
            log(f'{name}: stem is silent · skipped')
            continue
        log(f'{name}: {len(found_hits)} transients → {len(hs)} pass bleed gate (−{abs(MAX_BLEED):.0f} dB)')
        chosen = pick(piece, hs, y.mean(axis=0))
        log(f'{name}: {len(chosen)} distinct sound{"s" if len(chosen) != 1 else ""} by {int(ATTACK * 1000)} ms attack fingerprint'
            if chosen else f'{name}: no sound repeats cleanly · skipped')
        for i, h in enumerate(chosen, 1):
            log(f'Rendering {LABEL[piece]} {i} · {(h["end"] - h["start"]) / SR:.2f} s · 24-bit · peak −1 dBFS')
            label = f'{LABEL[piece]} {i}'
            fname = f'{prefix}{label}.wav'
            write(y, h, out_dir / fname)
            result.append(dict(name=fname, file=str(out_dir / fname), piece=piece, label=label,
                               count=h['count'], bleed_db=round(h['bleed'], 1),
                               seconds=round((h['end'] - h['start']) / SR, 2), at=round(h['start'] / SR, 2)))
    log(f'Kit ready · {len(result)} one-shots')
    return result


if __name__ == '__main__':
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src / 'one-shots'
    for r in make_kit(src, dst):
        print(f"{r['label']:8} {r['seconds']:5.2f}s  heard {r['count']:3}x  bleed {r['bleed_db']:6.1f} dB  "
              f"at {r['at']:7.2f}s  -> {r['name']}")
