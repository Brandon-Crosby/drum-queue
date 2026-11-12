"""patterns.py - drum patterns by genre, as 16th-note grids with hit probabilities.

Each pattern is one bar of 16 steps per sound. A value is the chance (0-1) that the step plays, so
every bar comes out a little different, the way real producers vary hats and perc. Velocities and
timing are humanized in render.py.

Sounds map to the training classes in synth.CLASSES:
  kick, snare (snare/clap/rim), hh (closed hat/open hat/shaker), perc (conga/bongo/campana/guiro/
  timbale/clave/tambora), toms, ride, crash
"""

def bar(*steps, p=1.0):
    """bar(0, 4, 8, 12) -> 16 probabilities with p at those steps."""
    g = [0.0] * 16
    for s in steps:
        g[s % 16] = p
    return g


def mix(*grids):
    return [max(v) for v in zip(*grids)]


EVERY_8TH = bar(0, 2, 4, 6, 8, 10, 12, 14)
EVERY_16TH = [1.0] * 16

GENRES = {
    # Dembow / reggaeton: four-on-the-floor kick, the "boom-ch-boom-chick" snare at 3/6/11/14 (0-based)
    'dembow': dict(
        bpm=(88, 100), swing=(0.0, 0.08),
        parts={
            'kick': [bar(0, 4, 8, 12)],
            'snare': [bar(3, 6, 11, 14), mix(bar(3, 6, 11, 14), bar(7, 15, p=0.25))],
            'hh': [EVERY_8TH, mix(EVERY_8TH, bar(1, 5, 9, 13, p=0.4)), mix(bar(2, 6, 10, 14), bar(0, 4, 8, 12, p=0.3))],
            'perc': [bar(2, 5, 7, 10, 13, 15, p=0.7), bar(1, 3, 6, 9, 11, 14, p=0.6), bar(3, 7, 11, 15, p=0.8)],
            'toms': [bar(12, 13, 14, 15, p=0.15)],
            'crash': [bar(0, p=0.25)],
        }),
    # Dembow at club tempo (Dominican dembow / "dembow dominicano"), busier perc
    'dembow_rd': dict(
        bpm=(112, 128), swing=(0.0, 0.05),
        parts={
            'kick': [bar(0, 4, 8, 12), mix(bar(0, 4, 8, 12), bar(7, 15, p=0.3))],
            'snare': [bar(3, 6, 11, 14)],
            'hh': [EVERY_16TH, mix(EVERY_8TH, bar(3, 7, 11, 15, p=0.6))],
            'perc': [bar(1, 2, 5, 6, 9, 10, 13, 14, p=0.7), bar(0, 3, 6, 8, 11, 14, p=0.7)],
            'crash': [bar(0, p=0.2)],
        }),
    # Perreo / old-school reggaeton: dembow with a heavier, sparser feel
    'perreo': dict(
        bpm=(90, 98), swing=(0.0, 0.06),
        parts={
            'kick': [bar(0, 4, 8, 12), mix(bar(0, 4, 8, 12), bar(10, p=0.3))],
            'snare': [bar(3, 6, 11, 14), bar(3, 7, 11, 15)],
            'hh': [bar(2, 6, 10, 14), EVERY_8TH],
            'perc': [bar(3, 7, 11, 15, p=0.5), bar(6, 14, p=0.6)],
            'ride': [EVERY_8TH],
        }),
    # Trap: half-time snare on 3, sparse syncopated kicks, rolling hats
    'trap': dict(
        bpm=(130, 160), swing=(0.0, 0.03),
        parts={
            'kick': [mix(bar(0), bar(3, 7, 10, 11, 14, p=0.45)), mix(bar(0, 10), bar(6, 13, p=0.4))],
            'snare': [bar(8), mix(bar(8), bar(15, p=0.2))],
            'hh': [EVERY_8TH, EVERY_16TH, mix(EVERY_8TH, bar(5, 13, 15, p=0.5))],
            'perc': [bar(6, 14, p=0.3)],
            'crash': [bar(0, p=0.2)],
        }),
    # R&B: laid-back half-time, ghost snares, light hats
    'rnb': dict(
        bpm=(62, 80), swing=(0.05, 0.15),
        parts={
            'kick': [mix(bar(0), bar(7, 10, p=0.5)), mix(bar(0, 9), bar(3, p=0.3))],
            'snare': [mix(bar(8), bar(5, 14, p=0.2))],
            'hh': [EVERY_8TH, mix(EVERY_8TH, bar(3, 11, p=0.4))],
            'perc': [bar(4, 12, p=0.3)],
            'ride': [bar(0, 4, 8, 12, p=0.3)],
        }),
    # Afrobeats: syncopated kick, rim/snare on 2&4 area, shakers and congas everywhere
    'afro': dict(
        bpm=(98, 112), swing=(0.03, 0.1),
        parts={
            'kick': [bar(0, 3, 6, 10), bar(0, 6, 8, 11)],
            'snare': [bar(4, 12), mix(bar(4, 12), bar(7, p=0.4))],
            'hh': [EVERY_16TH, mix(EVERY_8TH, bar(1, 3, 9, 11, p=0.6))],
            'perc': [bar(2, 5, 7, 10, 13, p=0.8), bar(1, 4, 6, 9, 12, 14, p=0.7)],
            'toms': [bar(13, 14, 15, p=0.1)],
        }),
    # Merengue / tambora: fast, tambora and güiro driving
    'merengue': dict(
        bpm=(140, 160), swing=(0.0, 0.03),
        parts={
            'kick': [bar(0, 4, 8, 12)],
            'snare': [bar(4, 12, p=0.6)],
            'hh': [EVERY_8TH],
            'perc': [EVERY_16TH, mix(bar(0, 2, 3, 4, 6, 8, 10, 11, 12, 14), bar(1, 9, p=0.5))],
            'crash': [bar(0, p=0.15)],
        }),
}

# Which synth sounds a genre leans on inside each class (others still appear sometimes).
FAVOR = {
    'dembow': {'perc': ['conga', 'bongo', 'campana', 'timbale', 'guiro'], 'hh': ['hat closed', 'shaker'], 'snare': ['snare', 'clap', 'rim']},
    'dembow_rd': {'perc': ['bongo', 'campana', 'conga', 'guiro'], 'hh': ['hat closed', 'shaker']},
    'perreo': {'perc': ['timbale', 'conga', 'clave'], 'snare': ['snare', 'clap']},
    'trap': {'hh': ['hat closed', 'hat open'], 'snare': ['clap', 'snare']},
    'rnb': {'snare': ['snare', 'rim', 'clap'], 'hh': ['hat closed']},
    'afro': {'perc': ['conga', 'bongo', 'campana', 'clave'], 'hh': ['shaker', 'hat closed'], 'snare': ['rim', 'clap']},
    'merengue': {'perc': ['tambora', 'guiro', 'campana'], 'hh': ['shaker', 'hat closed']},
}
