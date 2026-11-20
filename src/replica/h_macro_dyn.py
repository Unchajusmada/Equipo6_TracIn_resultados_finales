"""F1 macro con umbral dinámico (percentil elegido en la 1ª parte, aplicado a la 2ª), mejor (L,P) por dataset."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '..')
from pathlib import Path
from common import valid_mask
from paper_numbers import SYN_DATASETS, TABLE2
RES = Path(__file__).resolve().parents[2] / 'results' / 'synthetic' / 'scores'
CFGS = [f'L{L}P{P}' for L in (4, 8, 16) for P in (1, 2, 4)]
PCTS = np.arange(50, 100, 0.5)

def f1s(y, p):
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum(); tn = (~y & ~p).sum()
    fa = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    fnm = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return fa, (fa + fnm) / 2

rows = []
for ds in SYN_DATASETS:
    for seed in range(20):
        z = np.load(RES / f'{ds}_s{seed}.npz'); y = z['y'].astype(bool); m = valid_mask(len(y)); yv = y[m]
        for c in CFGS:
            s = z[f'lstm|{c}|tracin'].astype(float)[m]; s = s / s.max()
            for prop in (0.5, 0.8):
                k = int(len(s) * prop)
                for crit in ('bin', 'mac'):
                    j = 0 if crit == 'bin' else 1
                    best = max(PCTS, key=lambda q: f1s(yv[:k], s[:k] >= np.percentile(s, q))[j])
                    thr = np.percentile(s, best)
                    fa, fm = f1s(yv[k:], s[k:] >= thr)
                    rows.append(dict(ds=ds, seed=seed, cfg=c, prop=prop, crit=crit, f1_bin=fa, f1_mac=fm))
df = pd.DataFrame(rows)
g = df.groupby(['prop', 'crit', 'ds', 'cfg'])[['f1_bin', 'f1_mac']].mean()
best = g.groupby(level=[0, 1, 2]).max()
res = best.groupby(level=[0, 1]).mean()
print(res.round(3), '\nartículo LSTM+D medio:', np.mean(TABLE2['LSTM+D']).round(3))
print(best.loc[(0.5, 'mac')].assign(paper=[TABLE2['LSTM+D'][SYN_DATASETS.index(d)] for d in best.loc[(0.5,'mac')].index]).round(3))
