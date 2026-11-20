"""Evalúa la corrida de réplica (y los métodos de referencia ya guardados)
con el protocolo del artículo y con nuestros umbrales sin etiquetas.

Para cada (dataset, semilla, método, configuración) calcula, sobre los
mismos instantes (se excluyen 19 en cada extremo):
* auprc, oracle (F1 binario máximo)
* S_bin / S_mac : estático, percentil 85 (lectura literal), F1 binario / macro
* D_bin / D_mac : dinámico (percentil elegido en la 1ª mitad, medido en la 2ª)
* U_<regla>_bin / _mac : umbral sin etiquetas (IQR, MAD, log-IQR, log-MAD,
  POT) calculado sobre la serie completa SIN etiquetas y medido en la 2ª
  mitad (la misma parte que el dinámico, para que sean comparables).

Salida: results/replica/<subdir>_eval.csv
Uso:  python rep_eval.py --subdir synth_full
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import unsupervised_thresholds, valid_mask
from protocol import evaluate, f1_pair

RES = Path(__file__).resolve().parents[2] / "results"
BASE_KEYS = ["iforest|w1", "iforest|w4", "iforest|w8", "iforest|w16", "ocsvm|w1", "ocsvm|w4", "ocsvm|w8", "ocsvm|w16",
             "lof|w1", "lof|w4", "lof|w8", "lof|w16", "ae|w1", "ae|w4", "ae|w8", "ae|w16", "mp|m4", "mp|m8", "mp|m16"]


def eval_one(y, s, tau=0.85):
    e = evaluate(y, s, tau)
    k = len(s) // 2
    yb = y.astype(bool)
    for name, thr in unsupervised_thresholds(np.asarray(s, dtype=np.float64)).items():
        b, m = f1_pair(yb[k:], np.asarray(s[k:]) >= thr)
        e[f"U_{name}_bin"], e[f"U_{name}_mac"] = b, m
    return e


def file_job(args):
    f, base_dir = args
    z = np.load(f)
    ds, seed = f.stem.rsplit("_s", 1)
    y = z["y"].astype(bool)
    m = valid_mask(len(y))
    rows = []
    for k in z.files:
        if k == "y" or k.endswith("|epochs"):
            continue
        cfg, score = k.split("|", 1)
        rows.append(dict(ds=ds, seed=int(seed), method="tracin_lstm", cfg=cfg, score=score, **eval_one(y[m], z[k][m])))
    if base_dir is not None and (base_dir / f"{ds}_s{seed}.npz").exists():
        b = np.load(base_dir / f"{ds}_s{seed}.npz")
        for k in BASE_KEYS:
            meth, cfg = k.split("|")
            rows.append(dict(ds=ds, seed=int(seed), method=meth, cfg=cfg, score="score", **eval_one(y[m], b[k][m].astype(np.float64))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="synth_full")
    ap.add_argument("--baselines", default="", help="subcarpeta de results/replica con los métodos de referencia")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    files = sorted((RES / "replica" / a.subdir).glob("*_s*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    base_dir = RES / "replica" / a.baselines if a.baselines else None
    import multiprocessing as mp
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(file_job, [(f, base_dir) for f in files]):
            rows += r
    df = pd.DataFrame(rows)
    out = RES / "replica" / f"{a.subdir}_eval.csv"
    df.to_csv(out, index=False)
    print(out, len(df))


if __name__ == "__main__":
    main()
