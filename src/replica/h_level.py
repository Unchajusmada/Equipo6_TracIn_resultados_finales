"""¿A qué nivel evalúa el artículo? Scores por ventana desde las trazas (L4P1)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '..')
from pathlib import Path
from common import best_threshold, prf, project_sum, valid_mask
from paper_numbers import SYN_DATASETS, TABLE2
RES = Path(__file__).resolve().parents[2] / 'results' / 'synthetic' / 'scores'
L, P = 4, 1; span = L + P
rows = []
for ds in SYN_DATASETS:
    for seed in range(20):
        z = np.load(RES / f'{ds}_s{seed}.npz'); y = z['y'].astype(bool); n = len(y)
        loss = z['lstm|L4P1|trace_loss'].astype(float); hsq = z['lstm|L4P1|trace_hsq'].astype(float); lrs = z['lstm|L4P1|trace_lrs']
        si = (lrs[:, None] * loss * (1 + hsq)).sum(0)          # self-influence por ventana
        nw = len(si); win_any = np.array([y[i:i + span].any() for i in range(nw)]); win_tgt = y[L:L + nw]
        r = dict(ds=ds, seed=seed)
        for lab, yy in [('win_any', win_any), ('win_tgt', win_tgt)]:
            s = si / si.max()
            r[f'{lab}_oracle'] = best_threshold(yy, s)[0]
            r[f'{lab}_S085'] = prf(yy, s >= 0.85)[2]
            r[f'{lab}_Spct85'] = prf(yy, s >= np.percentile(s, 85))[2]
        ts = project_sum(si, span, n); m = valid_mask(n); ts = ts / ts[m].max()
        r['ts_oracle'] = best_threshold(y[m], ts[m])[0]
        # tolerancia: un instante marcado cuenta como TP si hay anomalía a <= span-1 a su derecha (la meseta izquierda)
        rows.append(r)
df = pd.DataFrame(rows); g = df.groupby('ds').mean(numeric_only=True).drop(columns='seed')
g['paper_S'] = [TABLE2['LSTM+S'][SYN_DATASETS.index(d)] for d in g.index]
g['paper_D'] = [TABLE2['LSTM+D'][SYN_DATASETS.index(d)] for d in g.index]
pd.set_option('display.width', 220); print(g.round(3)); print(g.mean().round(3))
