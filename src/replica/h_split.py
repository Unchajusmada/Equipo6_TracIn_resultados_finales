"""Lecturas alternativas de la partición del umbral dinámico en reales."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '..')
from common import valid_mask
from data_real import load
from paper_numbers import TABLE4
from real_screen import prf
from protocol import PCTS
RES = Path(__file__).resolve().parents[2] / 'results' / 'replica' / 'real_full'

def pick(y, s, s_all):
    thr = np.percentile(s_all, PCTS); best, bt = -1, thr[0]
    for t in thr:
        f = prf(y, s >= t)['F1_bin']
        if f > best: best, bt = f, t
    return bt

rows = []
for f in sorted(RES.glob('*.npz')):
    if '.tmp' in f.name: continue
    ds = f.stem.split('_L')[0]; cfg = 'L' + f.stem.split('_L')[1].split('_s')[0]
    y = load(ds)[1].astype(bool); m = valid_mask(len(y)); y = y[m]
    s = np.load(f)['tracin'][m].astype(float); s /= s.max(); n = len(s); k = n // 2
    rng = np.random.default_rng(0); tr = np.zeros(n, bool); tr[rng.choice(n, k, replace=False)] = True
    t_rand = pick(y[tr], s[tr], s); t_first = pick(y[:k], s[:k], s)
    r = dict(ds=ds, cfg=cfg)
    for nm, t, mask in [('aleatoria', t_rand, ~tr), ('1a mitad→serie completa', t_first, np.ones(n, bool)), ('contigua', t_first, np.arange(n) >= k)]:
        d = prf(y[mask], s[mask] >= t); r[nm] = f"{d['P_bin']:.3f}/{d['R_bin']:.3f}/{d['F1_bin']:.3f}"; r[nm + '_F1'] = d['F1_bin']
    rows.append(r)
df = pd.DataFrame(rows)
for ds, g in df.groupby('ds'):
    b = g.loc[g['aleatoria_F1'].idxmax()]
    print(ds, 'artículo LSTM+D', TABLE4[ds]['LSTM+D'])
    print(g[['cfg', 'aleatoria', '1a mitad→serie completa', 'contigua']].to_string(index=False))
