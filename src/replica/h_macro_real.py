"""Hipótesis F1 macro en reales: P, R, F1 con percentil 87 (estático) vs Tabla 4."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '..')
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support as prfs
from common import valid_mask
from data_real import load
from paper_numbers import TABLE4
RES = Path(__file__).resolve().parents[2] / 'results' / 'real' / 'scores'
for ds in ['creditcard', 'cicids', 'gecco', 'swan_sf']:
    y = load(ds)[1].astype(int); m = valid_mask(len(y)); yy = y[m]
    out = []
    for seed in range(3):
        s = np.load(RES / ds / f'lstm_4_1_{seed}.npz')['tracin'].astype(float)[m]; s = s / s.max()
        for nm, pred in [('pct87', s >= np.percentile(s, 87)), ('frac087', s >= 0.87)]:
            for avg in ['binary', 'macro', 'weighted']:
                p, r, f, _ = prfs(yy, pred.astype(int), average=avg, zero_division=0)
                out.append((nm, avg, p, r, f))
    d = pd.DataFrame(out, columns=['thr', 'avg', 'P', 'R', 'F1']).groupby(['thr', 'avg']).mean()
    print(ds, 'artículo LSTM+S (P,R,F1) =', TABLE4[ds]['LSTM+S']); print(d.round(3))
