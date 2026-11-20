"""Evalúa results/final/<bloque>/*.npz con el protocolo de PROTOCOLO_FINAL.md.

Por (archivo, configuración, entrenamiento, proyección) calcula, sobre los
instantes válidos (se excluyen 19 en cada extremo):
  auprc, auroc                     calidad del score (serie completa)
  <thr>_bin, <thr>_mac             F1 binario / macro medido en la 2.ª mitad
     thr = static (percentil τ del artículo), dyn (umbral elegido con
     etiquetas en la 1.ª mitad: criterio binario para _bin, macro para _mac),
     iqr (Q3 + 1.5·IQR sin etiquetas, sobre toda la serie)
  static_full_*, iqr_full_*        lo mismo medido en la serie completa
  iqr_rate                         fracción de instantes marcados por IQR

Salida: results/final/<bloque>_eval.csv
Uso:    python evaluate.py synth|contam|real [--workers 20]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from core import RES  # noqa: E402  (fija sys.path)

import numpy as np  # noqa: E402

TRAINS = ["base", "pd", "soft_r1", "soft_r2", "trim_r1", "trim_r2", "adapt_r1", "adapt_r2", "adapt_r3"]
TAU = {"synth": 0.85, "contam": 0.85, "real": 0.87, "smd": 0.87, "nasa": 0.87, "semi": 0.87}
PCTS = np.arange(50.0, 99.95, 0.1)


def f1s(y, p):
    tp = np.sum(y & p); fp = np.sum(~y & p); fn = np.sum(y & ~p); tn = np.sum(~y & ~p)
    fa = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    fnm = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return float(fa), float((fa + fnm) / 2)


def metrics(y, s, tau):
    from common import thr_iqr
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = np.asarray(s, dtype=np.float64)
    y = y.astype(bool)
    k = len(s) // 2
    o = {"auprc": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s))}
    ts = np.percentile(s, 100 * tau)
    ti = thr_iqr(s)
    o["static_bin"], o["static_mac"] = f1s(y[k:], s[k:] >= ts)
    o["static_full_bin"], o["static_full_mac"] = f1s(y, s >= ts)
    o["iqr_bin"], o["iqr_mac"] = f1s(y[k:], s[k:] >= ti)
    o["iqr_full_bin"], o["iqr_full_mac"] = f1s(y, s >= ti)
    o["iqr_rate"] = float((s >= ti).mean())
    # dinámico: búsqueda vectorizada por bloques sobre la grilla de percentiles
    ya, sa, yb, sb = y[:k], s[:k], y[k:], s[k:]
    thr = np.percentile(s, PCTS)
    order = np.argsort(-sa)
    sa_sorted, ya_sorted = sa[order], ya[order]
    cnt = np.searchsorted(-sa_sorted, -thr, side="right")  # nº de puntos >= thr
    tp = np.r_[0, np.cumsum(ya_sorted)][cnt]
    fp = cnt - tp
    fn = ya.sum() - tp
    tn = len(ya) - tp - fp - fn
    fa = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    fnm = np.where(tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    tb = thr[int(np.argmax(fa))]
    tm = thr[int(np.argmax((fa + fnm) / 2))]
    o["dyn_bin"] = f1s(yb, sb >= tb)[0]
    o["dyn_mac"] = f1s(yb, sb >= tm)[1]
    o["dyn_bin_mac"] = f1s(yb, sb >= tb)[1]
    return o


def file_job(args):
    block, f = args
    from common import project_sum, project_targets_mean, valid_mask
    z = np.load(f)
    y = z["y"].astype(bool)
    n = len(y)
    m = valid_mask(n)
    stem = f.stem
    if block == "contam":
        mix, rr, sd = stem.split("_")
        meta = dict(ds=mix, ratio=float(rr[1:]), seed=int(sd[1:]))
    elif block == "semi":
        mach, mix, rr, sd = stem.split("_")
        meta = dict(ds=mach, mix=mix, ratio=float(rr[1:]), seed=int(sd[1:]))
    else:
        ds, sd = stem.rsplit("_s", 1)
        meta = dict(ds=ds, seed=int(sd))
    meta["anom_rate"] = float(y[m].mean())
    tau = TAU[block]
    rows = []
    cfgs = sorted({k.split("|")[0] for k in z.files if "|" in k}) or [None]
    for cfg in cfgs:
        if cfg is None:
            L, P, pre, cname = 4, 1, "", "L4P1"
        else:
            L, P = int(cfg[1:cfg.index("P")]), int(cfg[cfg.index("P") + 1:])
            pre, cname = cfg + "|", cfg
        for tr in TRAINS:
            if pre + tr not in z.files:
                continue
            s = z[pre + tr].astype(np.float64)
            for proj, sc in (("win", project_sum(s, L + P, n)), ("tgt", project_targets_mean(s, L, P, n))):
                rows.append(dict(**meta, cfg=cname, train=tr, proj=proj, **metrics(y[m], sc[m], tau)))
    for k in z.files:  # métodos de referencia ya por instante
        if k.startswith("ts_"):
            rows.append(dict(**meta, cfg="-", train=k[3:], proj="-", **metrics(y[m], z[k][m].astype(np.float64), tau)))
    diag_rows = []
    for k in z.files:
        if "diag_" in k:
            cfg = k.split("|")[0] if "|" in k else "L4P1"
            diag_rows.append(dict(**meta, cfg=cfg, key=k.split("diag_")[1], value=float(z[k])))
    return rows, diag_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["synth", "contam", "real", "smd", "nasa", "semi"])
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    import pandas as pd
    files = sorted(f for f in (RES / a.block).glob("*.npz") if ".tmp" not in f.name)
    rows, drows = [], []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r, d in pool.imap_unordered(file_job, [(a.block, f) for f in files]):
            rows += r
            drows += d
    pd.DataFrame(rows).to_csv(RES / f"{a.block}_eval.csv", index=False)
    pd.DataFrame(drows).to_csv(RES / f"{a.block}_diag.csv", index=False)
    print(a.block, len(files), "archivos,", len(rows), "filas")


if __name__ == "__main__":
    main()
