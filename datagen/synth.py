"""synth.py - license-free drum and percussion one-shots, synthesized from scratch.

Every call makes a slightly different sound (pitch, decay, tone, noise color are randomized), so a
training set sees thousands of kicks, snares and congas rather than one of each. Nothing here comes
from a sample pack, so there are no license questions about training on it.

Classes (the stems the new kit model will learn):
  kick   snare (snare, clap, rim)   toms   hh (closed/open hats, shakers, maracas)
  perc (conga, bongo, campana/cowbell, guiro, timbale, clave, tambora)   ride   crash
"""
import numpy as np

SR = 44100


def _t(sec):
    return np.arange(int(sec * SR)) / SR


def _env(sec, decay, attack=0.001):
    t = _t(sec)
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / max(decay, 1e-4))


def _noise(n, rng):
    return rng.standard_normal(n)


def _bp(x, lo, hi, order=2):
    from scipy.signal import butter, sosfilt
    hi = min(hi, SR / 2 - 100)
    lo = max(lo, 20)
    if lo >= hi:
        return x
    return sosfilt(butter(order, [lo, hi], 'bandpass', fs=SR, output='sos'), x)


def _hp(x, fc, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, min(fc, SR / 2 - 100), 'highpass', fs=SR, output='sos'), x)


def _lp(x, fc, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, min(fc, SR / 2 - 100), 'lowpass', fs=SR, output='sos'), x)


def _sweep(f0, f1, sweep_s, sec):
    """Sine whose pitch falls exponentially from f0 to f1 (kicks, toms, congas)."""
    t = _t(sec)
    f = f1 + (f0 - f1) * np.exp(-t / max(sweep_s, 1e-4))
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def _norm(x):
    p = np.abs(x).max()
    return x / p if p > 0 else x


# ---------------------------------------------------------------- kit
def kick(rng):
    sec = rng.uniform(0.25, 0.9)
    body = _sweep(rng.uniform(120, 260), rng.uniform(40, 62), rng.uniform(0.02, 0.06), sec)
    body *= _env(sec, rng.uniform(0.08, 0.35))
    click = _hp(_noise(int(0.01 * SR), rng), rng.uniform(2000, 6000)) * _env(0.01, 0.002)
    x = body.copy()
    x[:len(click)] += click * rng.uniform(0.05, 0.4)
    drive = rng.uniform(1, 4)
    return _norm(np.tanh(x * drive))


def snare(rng):
    sec = rng.uniform(0.15, 0.45)
    tone = _sweep(rng.uniform(220, 330), rng.uniform(160, 220), 0.02, sec) * _env(sec, rng.uniform(0.03, 0.08))
    nz = _bp(_noise(len(tone), rng), rng.uniform(1500, 3000), rng.uniform(7000, 12000)) * _env(sec, rng.uniform(0.05, 0.18))
    return _norm(tone * rng.uniform(0.3, 1) + nz * rng.uniform(0.6, 1.2))


def clap(rng):
    sec = rng.uniform(0.2, 0.5)
    x = np.zeros(int(sec * SR))
    t0 = 0
    for _ in range(rng.integers(3, 5)):               # a few quick bursts, then the tail
        b = _env(0.012, 0.004) * _noise(int(0.012 * SR), rng)
        x[t0:t0 + len(b)] += b
        t0 += int(rng.uniform(0.006, 0.014) * SR)
    tail = _noise(len(x), rng) * _env(sec, rng.uniform(0.04, 0.12))
    x[t0:] += tail[:len(x) - t0]
    return _norm(_bp(x, rng.uniform(800, 1200), rng.uniform(4000, 8000)))


def rim(rng):
    sec = 0.12
    tone = np.sin(2 * np.pi * rng.uniform(400, 900) * _t(sec)) * _env(sec, rng.uniform(0.01, 0.03))
    clk = _hp(_noise(len(tone), rng), 3000) * _env(sec, 0.004)
    return _norm(tone + 0.6 * clk)


def tom(rng):
    sec = rng.uniform(0.3, 0.8)
    f = rng.uniform(80, 220)
    x = _sweep(f * 1.6, f, rng.uniform(0.03, 0.1), sec) * _env(sec, rng.uniform(0.12, 0.35))
    x += _bp(_noise(len(x), rng), 300, 3000) * _env(sec, 0.02) * 0.2
    return _norm(x)


def _metal(sec, base, rng, n=6):
    """Inharmonic square-ish partials: the classic analog cymbal/hat recipe."""
    t = _t(sec)
    ratios = np.array([1, 1.342, 1.2312, 1.6532, 1.9523, 2.1523])[:n] * rng.uniform(0.9, 1.1, n)
    return sum(np.sign(np.sin(2 * np.pi * base * r * t)) for r in ratios)


def hat_closed(rng):
    sec = rng.uniform(0.04, 0.12)
    x = _metal(sec, rng.uniform(300, 420), rng) * 0.5 + _noise(int(sec * SR), rng)
    return _norm(_hp(x, rng.uniform(6000, 9000)) * _env(sec, rng.uniform(0.01, 0.04)))


def hat_open(rng):
    sec = rng.uniform(0.25, 0.6)
    x = _metal(sec, rng.uniform(300, 420), rng) * 0.5 + _noise(int(sec * SR), rng)
    return _norm(_hp(x, rng.uniform(5000, 8000)) * _env(sec, rng.uniform(0.1, 0.3)))


def shaker(rng):
    """Shakers / maracas: grainy band-passed noise with a soft swell."""
    sec = rng.uniform(0.06, 0.18)
    x = _noise(int(sec * SR), rng) * (rng.random(int(sec * SR)) > rng.uniform(0.3, 0.7))
    return _norm(_bp(x, rng.uniform(3000, 5000), rng.uniform(9000, 14000)) * _env(sec, sec / 3, attack=sec / 4))


def ride(rng):
    sec = rng.uniform(0.8, 2.0)
    x = _metal(sec, rng.uniform(400, 600), rng) * 0.4 + _noise(int(sec * SR), rng) * 0.3
    bell = np.sin(2 * np.pi * rng.uniform(2500, 4000) * _t(sec)) * rng.uniform(0, 0.4)
    return _norm(_hp(x + bell, 2500) * _env(sec, rng.uniform(0.3, 0.8)))


def crash(rng):
    sec = rng.uniform(1.2, 3.0)
    x = _metal(sec, rng.uniform(300, 500), rng) * 0.3 + _noise(int(sec * SR), rng)
    return _norm(_hp(x, rng.uniform(1500, 3000)) * _env(sec, rng.uniform(0.4, 1.2), attack=0.003))


# ---------------------------------------------------------------- latin percussion
def conga(rng, slap=None):
    """Conga / tumbadora: a pitched skin, with an optional slap (bright noise snap)."""
    sec = rng.uniform(0.2, 0.5)
    f = rng.uniform(170, 330)
    x = _sweep(f * 1.15, f, 0.01, sec) * _env(sec, rng.uniform(0.06, 0.18))
    x += 0.3 * np.sin(2 * np.pi * f * 2.3 * _t(sec)) * _env(sec, 0.03)
    if slap if slap is not None else rng.random() < 0.4:
        x += _bp(_noise(len(x), rng), 1500, 6000) * _env(sec, 0.015) * rng.uniform(0.5, 1.2)
    return _norm(x)


def bongo(rng):
    sec = rng.uniform(0.1, 0.3)
    f = rng.uniform(350, 650)
    x = _sweep(f * 1.1, f, 0.006, sec) * _env(sec, rng.uniform(0.03, 0.09))
    x += _bp(_noise(len(x), rng), 2000, 8000) * _env(sec, 0.006) * 0.4
    return _norm(x)


def campana(rng):
    """Campana / cowbell: two metallic partials, short and bright."""
    sec = rng.uniform(0.1, 0.4)
    f = rng.uniform(500, 900)
    t = _t(sec)
    x = np.sign(np.sin(2 * np.pi * f * t)) + 0.7 * np.sign(np.sin(2 * np.pi * f * rng.uniform(1.45, 1.55) * t))
    return _norm(_bp(x, f * 0.8, f * 6) * _env(sec, rng.uniform(0.03, 0.12)))


def guiro(rng):
    """Güiro: a scrape, i.e. a fast train of tiny clicks."""
    sec = rng.uniform(0.12, 0.35)
    x = np.zeros(int(sec * SR))
    rate = rng.uniform(80, 180)
    for k in range(int(sec * rate)):
        i = int(k / rate * SR)
        x[i:i + 60] += _noise(min(60, len(x) - i), rng)
    return _norm(_bp(x, 2000, 9000) * _env(sec, sec, attack=sec * 0.3))


def timbale(rng):
    sec = rng.uniform(0.15, 0.5)
    f = rng.uniform(350, 700)
    x = np.sin(2 * np.pi * f * _t(sec)) * _env(sec, rng.uniform(0.05, 0.2))
    x += _metal(sec, f * 0.9, rng, 3) * 0.15 * _env(sec, 0.04)
    x += _hp(_noise(len(x), rng), 2500) * _env(sec, 0.01) * 0.5
    return _norm(x)


def clave(rng):
    sec = 0.15
    return _norm(np.sin(2 * np.pi * rng.uniform(1800, 2600) * _t(sec)) * _env(sec, rng.uniform(0.02, 0.05)))


def tambora(rng):
    """Dominican tambora: low skin + a wooden rim hit."""
    sec = rng.uniform(0.2, 0.45)
    f = rng.uniform(110, 190)
    x = _sweep(f * 1.3, f, 0.01, sec) * _env(sec, rng.uniform(0.07, 0.15))
    x += _bp(_noise(len(x), rng), 800, 4000) * _env(sec, 0.01) * rng.uniform(0.2, 0.8)
    return _norm(x)


# class -> list of (sound name, generator)
CLASSES = {
    'kick': [('kick', kick)],
    'snare': [('snare', snare), ('clap', clap), ('rim', rim)],
    'toms': [('tom', tom)],
    'hh': [('hat closed', hat_closed), ('hat open', hat_open), ('shaker', shaker)],
    'perc': [('conga', conga), ('bongo', bongo), ('campana', campana), ('guiro', guiro),
             ('timbale', timbale), ('clave', clave), ('tambora', tambora)],
    'ride': [('ride', ride)],
    'crash': [('crash', crash)],
}
