"""Hipótesis de evaluación sobre scores ya guardados (sintéticos, 20 semillas)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '..')
from pathlib import Path
from common import best_threshold, prf, valid_mask
from paper_numbers import SYN_DATASETS, TABLE2
RES = Path(__file__).resolve().parents[2] / 'results' / 'synthetic' / 'scores'
CFGS = [f'L{L}P{P}' for L in (4, 8, 16) for P in (1, 2, 4)]
rows = []
for ds in SYN_DATASETS:
    for seed in range(20):
        z = np.load(RES / f'{ds}_s{seed}.npz')
        y = z['y']; n = len(y); m = valid_mask(n)
        for c in CFGS:
            s = z[f'lstm|{c}|tracin'].astype(float)
            s = s / s.max()
            r = dict(ds=ds, seed=seed, cfg=c)
            yv, sv = y[m], s[m]
            r['oracle_all'] = best_threshold(yv, sv)[0]
            for prop in (0.5, 0.8):
                k = int(len(yv) * prop)
                f_tr, thr = best_threshold(yv[:k], sv[:k])
                r[f'test{prop}'] = prf(yv[k:], sv[k:] >= thr)[2] if np.isfinite(thr) else 0.0
                r[f'oracle_test{prop}'] = best_threshold(yv[k:], sv[k:])[0]
            rows.append(r)
df = pd.DataFrame(rows)
g = df.groupby(['ds', 'cfg']).mean(numeric_only=True)
best = g.groupby(level=0).max()   # mejor configuración por dataset (como el artículo)
best['paper_D'] = [TABLE2['LSTM+D'][SYN_DATASETS.index(d)] for d in best.index]
print(best.round(3)); print(best.mean().round(3))
