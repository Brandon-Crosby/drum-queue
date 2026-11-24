"""junk.py - does this one-shot sound like a clean drum, or like bleed?

A logistic model over features the production pipeline can see (no answer tracks): the one-shot's
spectrum shape, decay and duration, how loud its hits are in their own stem, and whether it only ever
plays when the kick or the snares play (bleed does). Trained on 338 one-shots from the 18 test beats,
scored against the real session tracks and corrected by the producer's ear verdicts (Nov 2026: he
auditioned every sound the first draft cut and named the good ones; his labels weighed 3x).

The producer auditioned every sound a first draft would cut (41): 33 junk, 8 good; his verdicts
override the scorer and weigh 3x. Held out by beat, cutting below 0.35 removes ~45 of 338 one-shots,
~70% of them junk; the good sounds it cuts (often quiet hats) aren't separable by these features yet.
He chose "fewer sounds that are all clean". Retrain with eval/junk_filter_train.py when more verdicts come in.
"""
import numpy as np

SR = 44100

# logistic weights over the raw features (scaler folded in); p = sigmoid(W @ x + B)
W = np.array([
    -1.210438e-02, -4.439160e-02, 3.768197e-04, -8.390814e-04, -1.215631e-02, 1.780659e-02,
    -2.222688e-02, -5.264996e-02, -3.973122e-02, 3.554765e-01, -7.503614e-02, 3.223517e-02,
    1.753312e+00, 8.929696e-02, 5.951943e-02, -2.454949e-02, -2.602365e-03, -3.549779e-03,
    7.843583e-01])
B = -1.815373

BANDS = ((30, 100), (100, 250), (250, 600), (600, 1500), (1500, 3500), (3500, 7000),
         (7000, 12000), (12000, 18000))


def _mono(x):
    return x.mean(1) if x.ndim == 2 else x


def features(y, stems, own, on, kick_on, sp_on):
    """y: the one-shot as written (normalized; mono'd here). stems: {role: mono stem}. own: the stem
    roles this sound was cut from. on: its hits in the stems' timeline. kick_on/sp_on: every kick/
    snare-perc hit in the song ([] for the sound's own group)."""
    y = _mono(np.asarray(y))[:int(0.3 * SR)]
    m = len(y)
    X = np.abs(np.fft.rfft(y * np.hanning(m), 16384)) ** 2
    f = np.fft.rfftfreq(16384, 1 / SR)
    tot = X[(f > 30) & (f < 18000)].sum() + 1e-12
    be = [10 * np.log10(X[(f >= a) & (f < b)].sum() / tot + 1e-12) for a, b in BANDS]
    cen = float((f * X).sum() / X.sum())
    mid = X[(f > 30) & (f < 16000)]
    flat = float(np.exp(np.mean(np.log(mid + 1e-12))) / (mid.mean() + 1e-12))
    env = np.abs(y)
    crest = 20 * np.log10(env.max() / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    pk = int(np.argmax(env))
    if m > pk + int(.06 * SR):
        dec = 20 * np.log10((np.sqrt(np.mean(y[pk + int(.03 * SR):pk + int(.06 * SR)] ** 2)) + 1e-9)
                            / (np.sqrt(np.mean(y[pk:pk + int(.01 * SR)] ** 2)) + 1e-9))
    else:
        dec = -60.0
    W_ = int(0.05 * SR)
    n0 = min(len(v) for v in stems.values())
    on = [int(i) for i in on if i + W_ < n0]
    if len(on) < 2:
        return None
    e = {r: np.mean([np.sum(v[i:i + W_] ** 2) for i in on]) for r, v in stems.items()}
    own_share = sum(e[r] for r in own if r in e) / (sum(e.values()) + 1e-12)
    own_sum = sum(stems[r] for r in own if r in stems)
    lvl_med = np.median([np.abs(own_sum[i:i + W_]).max() for i in on])
    stem_pk = np.percentile(np.abs(own_sum), 99.9) + 1e-9
    lvl = 20 * np.log10(lvl_med / stem_pk + 1e-9)

    def co(b):
        b = np.asarray(b)
        return float(np.mean([np.min(np.abs(b - x)) <= 0.015 * SR for x in on])) if len(b) else 0.0
    harsh = 10 * np.log10(X[(f >= 2000) & (f < 6000)].sum() / tot + 1e-12)
    return np.array(be + [cen / 1000, flat * 10, float(crest), float(dec), m / SR, own_share * 10,
                          float(lvl), co(kick_on) * 10, co(sp_on) * 10, float(harsh), np.log10(len(on))])


def clean_p(y, stems, own, on, kick_on, sp_on):
    """Probability this one-shot is a clean drum (None when it can't be judged: too few hits in range)."""
    x = features(y, stems, own, on, kick_on, sp_on)
    if x is None:
        return None
    return float(1 / (1 + np.exp(-(W @ x + B))))
