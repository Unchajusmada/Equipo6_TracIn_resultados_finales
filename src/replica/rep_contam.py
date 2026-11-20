"""Contaminación controlada (experimento 3) repetida con la configuración
que reproduce al artículo: datos estandarizados por canal y el TracIn de
lab.py. Mismo diseño que run_contamination.py: series de 4000 pasos con los
5 tipos de anomalía, proporción total r, L = 4, P = 1 fijos a priori.

Por (r, semilla) guarda results/replica/contam/r<r>_s<seed>.npz con:
  tracin, tracin_tgt, pd, rw_<modo>_r<k>[_tgt]  (k = 1..2)
  iforest, ocsvm, lof, ae (ventana 1) y mp (m = 8), sobre los mismos datos.

Uso:  python rep_contam.py --seeds 10 --workers 22
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
from common import project_sum, project_targets_mean
from forecaster import weights_from_influence
from lab import normalize, run
from screen import parse_variant
from synthetic import make_dataset

OUT = Path(__file__).resolve().parents[2] / "results" / "replica" / "contam"
RATIOS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LENGTH = 4000
L, P = 4, 1


def job(args):
    variant, ratio, seed, rw_rounds = args
    path = OUT / f"r{ratio:.2f}_s{seed}.npz"
    if path.exists():
        return str(path)
    import torch
    torch.set_num_threads(1)
    cfg = parse_variant(variant)
    X, y = make_dataset("01234", 10_000 + seed, length=LENGTH, ratio=ratio / 5)
    n = len(y)
    res = {"y": y}
    r = run(X, L, P, seed, cfg)
    res["tracin"] = project_sum(r["si"], L + P, n)
    res["tracin_tgt"] = project_targets_mean(r["si"], L, P, n)
    res["pd"] = project_targets_mean(r["loss_final"], L, P, n)
    for mode in ("soft", "trim"):
        s = r["si"]
        for k in range(1, rw_rounds + 1):
            s = run(X, L, P, seed, cfg, weights=weights_from_influence(s, mode))["si"]
            res[f"rw_{mode}_r{k}"] = project_sum(s, L + P, n)
            res[f"rw_{mode}_r{k}_tgt"] = project_targets_mean(s, L, P, n)
    Xz = normalize(X.astype(np.float64), "z").astype(np.float32)
    res["iforest"] = iforest(Xz, 1, seed)
    res["ocsvm"] = ocsvm(Xz, 1, seed)
    res["lof"] = lof(Xz, 1)
    res["ae"] = autoencoder(Xz, 1, seed)
    res["mp"] = matrix_profile(Xz, 8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in res.items()})
    tmp.rename(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rw_rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a.variant, r, s, a.rw_rounds) for s in range(a.seeds) for r in RATIOS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, p in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"{i}/{len(jobs)} {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
