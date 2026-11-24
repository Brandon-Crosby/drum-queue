"""junk_filter_train.py - retrain kitmatch/junk.py's weights.

    python3 eval/junk_filter_train.py [--write]

Data: eval/baselines/junk-rows.json (one row per one-shot of the 18 test beats, both models: the
features junk.features() computes, plus the truth-based kit score's verdict) and
eval/baselines/ear-labels-junk-filter.json (the producer's verdicts, which override the scorer and
weigh 3x). Prints a leave-beats-out check; --write puts the new weights into kitmatch/junk.py.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
EAR_W = 3.0


def feat(r):
    return r['b'] + [r['cen'] / 1000, r['flat'] * 10, r['crest'], r['dec'], r['dur'], r['own_share'] * 10,
                     r['lvl'], r['co_kick'] * 10, r['co_sp'] * 10, r['harsh'], np.log10(r['hits'])]


def data():
    R = json.load(open(HERE / 'baselines' / 'junk-rows.json'))
    ear = {(e['beat'], e['model'], e['label']): e['ear'] == 'good'
           for e in json.load(open(HERE / 'baselines' / 'ear-labels-junk-filter.json'))}
    X = np.array([feat(r) for r in R])
    y = np.array([ear.get((r['beat'], r['model'], r['label']), r['real']) for r in R])
    w = np.array([EAR_W if (r['beat'], r['model'], r['label']) in ear else 1.0 for r in R])
    g = np.array([r['beat'] for r in R])
    isear = np.array([(r['beat'], r['model'], r['label']) in ear for r in R])
    return X, y, w, g, isear


def fit(X, y, w, C=0.5):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y, sample_weight=w)
    W = m.coef_[0] / sc.scale_
    B = float(m.intercept_[0] - (m.coef_[0] * sc.mean_ / sc.scale_).sum())
    return W, B


def main():
    X, y, w, g, isear = data()
    P = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=6).split(X, y, g):
        W, B = fit(X[tr], y[tr], w[tr])
        P[te] = 1 / (1 + np.exp(-(X[te] @ W + B)))
    for th in (0.25, 0.3, 0.35, 0.4):
        cut = P < th
        print(f'cut below {th}: cuts {cut.sum()}/{len(y)}; of cuts {(~y[cut]).mean():.0%} junk | '
              f'ear-checked: junk cut {(cut & isear & ~y).sum()}/{(isear & ~y).sum()}, '
              f'good cut {(cut & isear & y).sum()}/{(isear & y).sum()}')
    if '--write' in sys.argv:
        W, B = fit(X, y, w)
        p = HERE.parent / 'kitmatch' / 'junk.py'
        s = p.read_text()
        ws = ',\n    '.join(', '.join(f'{v:.6e}' for v in W[i:i + 6]) for i in range(0, len(W), 6))
        s = re.sub(r'W = np\.array\(\[.*?\]\)', f'W = np.array([\n    {ws}])', s, flags=re.S)
        s = re.sub(r'\nB = [-0-9.e]+', f'\nB = {B:.6f}', s)
        p.write_text(s)
        print('wrote', p)


if __name__ == '__main__':
    main()
