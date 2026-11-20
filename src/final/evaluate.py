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
from nasa_cohort import HISTORICAL, MAIN, run_files

TRAINS = ["base", "pd", "soft_r1", "soft_r2", "trim_r1", "trim_r2", "adapt_r1", "adapt_r2", "adapt_r3"]
TAU = {"synth": 0.85, "contam": 0.85, "real": 0.87, "smd": 0.87, "nasa": 0.87, "semi": 0.87}
PCTS = np.arange(50.0, 99.95, 0.1)


def f1s(y, p):
    tp = np.sum(y & p); fp = np.sum(~y & p); fn = np.sum(y & ~p); tn = np.sum(~y & ~p)
    fa = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    fnm = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return float(fa), float((fa + fnm) / 2)


def checked_vector(values, name, length=None):
    array = np.asarray(values)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ValueError(f"{name}: vector de longitud {length if length is not None else 'válida'} requerido; forma {array.shape}")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name}: se requieren valores numéricos finitos")
    return array


def checked_labels(values):
    y = checked_vector(values, "etiquetas")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Se requieren etiquetas binarias (0, 1)")
    return y.astype(bool)


def evaluation_mask(n):
    from common import valid_mask
    mask = valid_mask(n)
    if mask.sum() < 2:
        raise ValueError("El recorte deja menos de dos instantes para calibración y evaluación")
    return mask


def metrics(y, s, tau):
    from common import thr_iqr
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = checked_labels(y)
    s = checked_vector(s, "scores", len(y)).astype(np.float64)
    if not y.any() or y.all():
        raise ValueError("AUROC indefinida: se necesitan las dos clases en el tramo válido")
    k = len(s) // 2
    o = {"auprc": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s))}
    ts = np.percentile(s, 100 * tau)
    ti = thr_iqr(s)
    o["static_bin"], o["static_mac"] = f1s(y[k:], s[k:] >= ts)
    o["static_full_bin"], o["static_full_mac"] = f1s(y, s >= ts)
    o["iqr_bin"], o["iqr_mac"] = f1s(y[k:], s[k:] >= ti)
    o["iqr_full_bin"], o["iqr_full_mac"] = f1s(y, s >= ti)
    o["iqr_rate"] = float((s >= ti).mean())
    # Referencia histórica: grilla de ambas mitades; etiquetas solo de la primera.
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
    try:
        return evaluate_file(args)
    except ValueError as exc:
        raise ValueError(f"{args[1].name}: {exc}") from exc


def evaluate_file(args):
    block, f = args
    with np.load(f) as archive:
        return evaluate_arrays(block, f, archive)


def evaluate_arrays(block, f, z):
    from common import project_sum, project_targets_mean
    if "y" not in z.files:
        raise ValueError("Falta el vector de etiquetas y")
    y = checked_labels(z["y"])
    n = len(y)
    m = evaluation_mask(n)
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
        if min(L, P) < 1 or L + P > n:
            raise ValueError(f"Configuración {cname} incompatible con {n} instantes")
        for tr in TRAINS:
            if pre + tr not in z.files:
                continue
            s = checked_vector(z[pre + tr], pre + tr, n - L - P + 1)
            for proj, sc in (("win", project_sum(s, L + P, n)), ("tgt", project_targets_mean(s, L, P, n))):
                rows.append(dict(**meta, cfg=cname, train=tr, proj=proj, **metrics(y[m], sc[m], tau)))
    for k in z.files:  # métodos de referencia ya por instante
        if k.startswith("ts_"):
            s = checked_vector(z[k], k, n)
            rows.append(dict(**meta, cfg="-", train=k[3:], proj="-", **metrics(y[m], s[m], tau)))
    if not rows:
        raise ValueError("No hay scores evaluables")
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
    ap.add_argument("--nasa-first-row", action="store_true", help="Evalúa la cohorte histórica NASA de 81 en una copia aislada")
    a = ap.parse_args()
    import pandas as pd
    if a.nasa_first_row and a.block != "nasa":
        ap.error("--nasa-first-row solo se aplica a NASA")
    cohort = HISTORICAL if a.nasa_first_row else MAIN
    root = RES / a.block
    files = (run_files(root, cohort) if a.block == "nasa"
             else sorted(f for f in root.glob("*.npz") if ".tmp" not in f.name))
    if not files:
        raise ValueError(f"No hay corridas para {a.block}")
    rows, drows = [], []
    expected = None
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r, d in pool.imap_unordered(file_job, [(a.block, f) for f in files]):
            keys = {(row["cfg"], row["train"], row["proj"]) for row in r}
            if len(keys) != len(r) or (expected is not None and keys != expected):
                raise ValueError(f"Comparaciones no pareadas: variantes o configuraciones distintas en {r[0]['ds']} s{r[0]['seed']}")
            expected = keys
            rows += r
            drows += d
    if a.block == "nasa":
        for row in rows + drows:
            row["nasa_cohort"] = cohort
    pd.DataFrame(rows).to_csv(RES / f"{a.block}_eval.csv", index=False)
    pd.DataFrame(drows).to_csv(RES / f"{a.block}_diag.csv", index=False)
    print(a.block, len(files), "archivos,", len(rows), "filas")


if __name__ == "__main__":
    main()
