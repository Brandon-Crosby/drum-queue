"""compare_stems.py - score any drum splitter's stems against the real tracks, next to Helyx.

    python eval/compare_stems.py "<their stems folder>" "<test beat folder>" [more pairs...]

<test beat folder> holds truth_kick/snare/hh.flac and (optionally) helyx's ours_*.flac, like
~/DrumSplits/splits_h4/<beat>. Their stems are found by name: kick, snare, tom(s), hat/hh, cymbal(s)
(any .wav/.flac/.aif). Prints, per model:
  - where the real snare hits land (alone / stacked on a kick) and how much reaches snare + toms/perc
  - where the real kicks land
  - hit cleanliness (dB of leftover under each real hit; more negative = cleaner) for kick, snare, hats
Same measures as the model test (eval/junk.py). Evaluation only: nothing here feeds training.
"""
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from junk import hit_junk, routing  # noqa: E402

NAMES = [('kick', r'kick|\bbd\b'), ('snare', r'snare|snr'), ('toms', r'tom|perc'), ('hh', r'hi.?hat|\bhh\b|hats?\b'),
         ('cymbals', r'cymbal|ride|crash')]


def find_stems(folder):
    out = {}
    files = [p for p in Path(folder).rglob('*') if p.suffix.lower() in ('.wav', '.flac', '.aif', '.aiff')]
    for key, pat in NAMES:
        hits = [p for p in files if re.search(pat, p.stem, re.I) and p not in out.values()]
        if hits:
            out[key] = hits[0]
    return out


def score(stems, beat):
    T = {p: sf.read(str(beat / f'truth_{p}.flac')) for p in ('kick', 'snare', 'hh') if (beat / f'truth_{p}.flac').exists()}
    S = {k: sf.read(str(v)) for k, v in stems.items()}
    if not {'kick', 'snare'} <= set(S) or not {'kick', 'snare'} <= set(T):
        return None
    sr = S['kick'][1]
    rs = routing(T['snare'][0], T['snare'][1], S, partner=T['kick'][0])
    rk = routing(T['kick'][0], T['kick'][1], S, partner=T['snare'][0])
    sp = lambda r: None if r.get('snare') is None else r['snare'] + (r.get('toms') or 0)  # noqa: E731
    y = S['snare'][0]
    if 'toms' in S:
        m = min(len(y), len(S['toms'][0]))
        y = y[:m] + S['toms'][0][:m]
    j = {'kick': hit_junk(T['kick'][0], S['kick'][0], T['kick'][1], sr_ours=sr),
         'snare': hit_junk(T['snare'][0], y, T['snare'][1], sr_ours=sr)}
    if 'hh' in S and 'hh' in T:
        j['hh'] = hit_junk(T['hh'][0], S['hh'][0], T['hh'][1], sr_ours=sr)
    return dict(alone=sp(rs['alone']), stacked=sp(rs['stacked']), kick=rk['alone'].get('kick'),
                where_alone=rs['alone'], where_stacked=rs['stacked'], junk=j, hits=rs['hits'])


def show(name, r):
    if r is None:
        print(f'  {name:14} (no kick/snare stems found)')
        return
    f = lambda v: '   - ' if v is None else f'{v:5.0%}'  # noqa: E731
    g = lambda v: '    -' if not v or v.get('all') is None else f"{v['all']:5.1f}"  # noqa: E731
    print(f"  {name:14} snare right: alone {f(r['alone'])} stacked {f(r['stacked'])} | kick right {f(r['kick'])} | "
          f"clean dB: kick {g(r['junk'].get('kick'))} snare {g(r['junk'].get('snare'))} hats {g(r['junk'].get('hh'))}")
    w = lambda d: ', '.join(f'{k} {v:.0%}' for k, v in d.items() if v is not None and v >= 0.03)  # noqa: E731
    print(f"  {'':14} snare went to: alone [{w(r['where_alone'])}]  stacked [{w(r['where_stacked'])}]")


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2 or len(a) % 2:
        sys.exit(__doc__)
    for theirs, beat in zip(a[::2], a[1::2]):
        beat = Path(beat).expanduser()
        print(f'\n== {beat.name}')
        ours = {k: beat / f'ours_{k}.flac' for k in ('kick', 'snare', 'toms', 'hh') if (beat / f'ours_{k}.flac').exists()}
        if ours:
            show('helyx', score(ours, beat))
        st = find_stems(Path(theirs).expanduser())
        print(f"  ({Path(theirs).name}: {', '.join(f'{k}={v.name}' for k, v in st.items())})")
        show(Path(theirs).name[:14], score(st, beat))
