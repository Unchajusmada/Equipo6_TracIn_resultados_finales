"""Hipótesis: el artículo reporta F1 'macro' (promedio de F1 de la clase normal y anómala) en sintéticos."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '..')
from pathlib import Path
from sklearn.metrics import f1_score
from common import valid_mask
from paper_numbers import SYN_DATASETS, TABLE2
RES = Path(__file__).resolve().parents[2] / 'results' / 'synthetic' / 'scores'
rows = []
for ds in SYN_DATASETS:
    for seed in range(20):
        z = np.load(RES / f'{ds}_s{seed}.npz'); y = z['y'].astype(int); m = valid_mask(len(y))
        s = z['lstm|L4P1|tracin'].astype(float)[m]; yy = y[m]; s = s / s.max()
        r = dict(ds=ds, seed=seed)
        for nm, pred in [('frac085', s >= 0.85), ('pct85', s >= np.percentile(s, 85))]:
            r[nm + '_bin'] = f1_score(yy, pred, zero_division=0)
            r[nm + '_macro'] = f1_score(yy, pred, average='macro', zero_division=0)
            r[nm + '_weighted'] = f1_score(yy, pred, average='weighted', zero_division=0)
        rows.append(r)
df = pd.DataFrame(rows)
g = df.groupby('ds').agg(['mean', 'std']).drop(columns='seed')
g[('paper_S', '')] = [TABLE2['LSTM+S'][SYN_DATASETS.index(d)] for d in g.index]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(g.round(3)); print(df.drop(columns=['ds','seed']).mean().round(3))
