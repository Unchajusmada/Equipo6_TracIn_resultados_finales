"""Contaminación controlada: series de 4000 pasos con proporción nominal r de
anomalías, tres mezclas de tipos, L4P1, todas las variantes + LOF e IForest.

La proporción por tipo es r / (número de tipos), como en rep_contam.py.
Salida: results/final/contam/<mezcla>_r<r>_s<seed>.npz
Uso:    python run_contam.py --seeds 20 --workers 20
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from core import RES, all_variants, save  # noqa: E402

RATIOS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MIXES = ["01234", "01", "234"]
LENGTH, L, P = 4000, 4, 1


def job(args):
    mix, ratio, seed = args
    path = RES / "contam" / f"{mix}_r{ratio:.2f}_s{seed}.npz"
    if path.exists():
        return path.name
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from baselines import iforest, lof
    from lab import normalize
    from screen import parse_variant
    from synthetic import make_dataset
    X, y = make_dataset(mix, 10_000 + seed, length=LENGTH, ratio=ratio / len(mix))
    res = {"y": y, **all_variants(X, y, L, P, seed, parse_variant("base"))}
    Xz = normalize(X.astype(np.float64), "z").astype(np.float32)
    res["ts_lof"] = lof(Xz, 1)
    res["ts_iforest"] = iforest(Xz, 1, seed)
    save(path, res)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    jobs = [(m, r, s) for s in range(a.seeds) for m in MIXES for r in RATIOS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, name in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"{i}/{len(jobs)} {name}", flush=True)


if __name__ == "__main__":
    main()
