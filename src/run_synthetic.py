"""Experimento 1: los 15 datasets sintéticos del artículo (Tabla 2).

Para cada dataset y semilla (la semilla fija tanto la realización de la
serie como la inicialización de los modelos) se calculan y guardan los
scores por instante de:

* LSTM + TracIn para L in {4, 8, 16}, P in {1, 2, 4}, con sus variantes de
  ablación (forecaster.Trace.timestep_scores) y las rondas de reponderación
  ``soft`` y ``trim`` (propuesta 2).
* IForest, OCSVM, LOF, AE con w in {1, 4, 8, 16} y Matrix Profile con
  m in {4, 8, 16}.

Para la configuración por defecto (L=4, P=1) se guarda además la traza
completa (pérdida y ||h||^2 por época y ventana) para el análisis de por qué
funciona la self-influence.

Uso:  python run_synthetic.py [--seeds 20] [--workers 22]
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "synthetic" / "scores"

LS, PS = [4, 8, 16], [1, 2, 4]
WS, MS = [1, 4, 8, 16], [4, 8, 16]
DEFAULT = (4, 1)


def task(args):
    dataset, seed = args
    path = OUT / f"{dataset}_s{seed}.npz"
    if path.exists():
        return dataset, seed, 0.0
    import torch
    torch.set_num_threads(1)
    from baselines import autoencoder, iforest, lof, matrix_profile, ocsvm
    from forecaster import reweighted_runs, train_trace
    from synthetic import make_dataset

    t0 = time.time()
    X, y = make_dataset(dataset, seed)
    res: dict[str, np.ndarray] = {"X": X, "y": y}
    for L, P in product(LS, PS):
        tag = f"lstm|L{L}P{P}"
        base = train_trace(X, L, P, seed)
        for k, s in base.timestep_scores().items():
            res[f"{tag}|{k}"] = s
        res[f"{tag}|epochs"] = np.array(len(base.lrs))
        if (L, P) == DEFAULT:
            res[f"{tag}|trace_loss"] = base.loss
            res[f"{tag}|trace_hsq"] = base.hsq
            res[f"{tag}|trace_lrs"] = base.lrs
        for mode in ("soft", "trim"):
            for r, (tr, w) in enumerate(reweighted_runs(X, L, P, seed, base, mode), start=1):
                ts = tr.timestep_scores()
                res[f"{tag}|rw_{mode}_r{r}"] = ts["tracin"]
                res[f"{tag}|rw_{mode}_r{r}_tgt"] = ts["tracin_tgt"]
                if (L, P) == DEFAULT:
                    res[f"{tag}|rw_{mode}_r{r}_w"] = w.astype(np.float32)
    for w in WS:
        res[f"iforest|w{w}"] = iforest(X, w, seed)
        res[f"ocsvm|w{w}"] = ocsvm(X, w, seed)
        res[f"lof|w{w}"] = lof(X, w)
        res[f"ae|w{w}"] = autoencoder(X, w, seed)
    for m in MS:
        res[f"mp|m{m}"] = matrix_profile(X, m)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)
    return dataset, seed, time.time() - t0


def main():
    from synthetic import DATASETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s) for s in range(a.seeds) for d in DATASETS]
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, (d, s, dt) in enumerate(pool.imap_unordered(task, jobs), 1):
            print(f"[{i}/{len(jobs)}] {d} s{s} {dt:.0f}s  (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
