"""Experimento 3 (propuesta): contaminación controlada.

Pregunta: ¿cómo se degrada cada detector cuando la proporción de anomalías
en la serie (que es también el conjunto de entrenamiento) crece, y la
reponderación (propuesta 2) mitiga la degradación de TracIn?

Se generan series de 4000 pasos (el artículo muestra que 400 pasos dan
varianza alta, Tabla 3) con los 5 tipos de anomalía, cada uno con
proporción r/5, para r in {1, 5, 10, 15, 20, 25, 30} %. Se reporta la
proporción real (los tramos pueden traslaparse). Configuración fija a priori:
L = 4, P = 1 (la del umbral estático del artículo), 10 semillas.

Uso:  python run_contamination.py [--seeds 10] [--workers 22]
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "contamination" / "scores"
RATIOS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LENGTH = 4000


def task(args):
    ratio, seed = args
    path = OUT / f"r{ratio:.2f}_s{seed}.npz"
    if path.exists():
        return ratio, seed, 0.0
    import torch
    torch.set_num_threads(1)
    from baselines import autoencoder, iforest, lof, matrix_profile, ocsvm
    from forecaster import reweighted_runs, train_trace
    from synthetic import make_dataset

    t0 = time.time()
    X, y = make_dataset("01234", 10_000 + seed, length=LENGTH, ratio=ratio / 5)
    res = {"X": X, "y": y}
    base = train_trace(X, 4, 1, seed)
    for k, s in base.timestep_scores().items():
        res[f"lstm|{k}"] = s
    res["lstm|trace_loss"] = base.loss
    res["lstm|trace_lrs"] = base.lrs
    for mode in ("soft", "trim"):
        for r, (tr, w) in enumerate(reweighted_runs(X, 4, 1, seed, base, mode), start=1):
            ts = tr.timestep_scores()
            res[f"lstm|rw_{mode}_r{r}"] = ts["tracin"]
            res[f"lstm|rw_{mode}_r{r}_tgt"] = ts["tracin_tgt"]
            res[f"lstm|rw_{mode}_r{r}_w"] = w.astype(np.float32)
            if r == 3:
                res[f"lstm|rw_{mode}_trace_loss"] = tr.loss
    res["iforest|w1"] = iforest(X, 1, seed)
    res["ocsvm|w1"] = ocsvm(X, 1, seed)
    res["lof|w1"] = lof(X, 1)
    res["ae|w1"] = autoencoder(X, 1, seed)
    res["mp|m8"] = matrix_profile(X, 8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)
    return ratio, seed, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, s) for s in range(a.seeds) for r in RATIOS]
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=2) as pool:
        for i, (r, s, dt) in enumerate(pool.imap_unordered(task, jobs), 1):
            print(f"[{i}/{len(jobs)}] r={r:.2f} s{s} {dt:.0f}s (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
