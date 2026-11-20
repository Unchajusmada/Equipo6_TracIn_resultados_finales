"""Métodos de referencia sobre los sintéticos estandarizados por canal
(la misma entrada que recibe TracIn en la réplica), con los mismos
hiperparámetros que el experimento principal.

Salida: results/replica/synth_baselines/<ds>_s<seed>.npz
Uso:  python rep_baselines.py --seeds 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from baselines import autoencoder, iforest, lof, matrix_profile, ocsvm
from lab import normalize
from synthetic import DATASETS, make_dataset

OUT = Path(__file__).resolve().parents[2] / "results" / "replica" / "synth_baselines"


def job(args):
    ds, seed = args
    path = OUT / f"{ds}_s{seed}.npz"
    if path.exists():
        return
    X, y = make_dataset(ds, seed)
    X = normalize(X.astype(np.float64), "z").astype(np.float32)
    res = {"y": y}
    for w in (1, 4, 8, 16):
        res[f"iforest|w{w}"] = iforest(X, w, seed)
        res[f"ocsvm|w{w}"] = ocsvm(X, w, seed)
        res[f"lof|w{w}"] = lof(X, w)
        res[f"ae|w{w}"] = autoencoder(X, w, seed)
    for m in (4, 8, 16):
        res[f"mp|m{m}"] = matrix_profile(X, m)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in res.items()})
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        list(pool.imap_unordered(job, [(ds, s) for s in range(a.seeds) for ds in DATASETS]))
    print("listo")


if __name__ == "__main__":
    main()
