"""Corrida final de réplica en sintéticos con la configuración que reproduce
al artículo, más nuestras mejoras sobre esa misma configuración.

Por (dataset, semilla) guarda en results/replica/synth/<ds>_s<seed>.npz los
scores por instante de las 9 configuraciones (L, P):
  <cfg>|tracin      método del artículo
  <cfg>|tracin_tgt  mejora 3: proyección a los pasos predichos
  <cfg>|pd          error final del mismo modelo (prediction deviation)
  <cfg>|rw_<modo>_r<k>[ _tgt]  mejora 2: reponderación (k = 1..R)

Uso:  python rep_synth.py --variant full --seeds 20 --rw_rounds 2
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import project_sum, project_targets_mean
from forecaster import weights_from_influence
from lab import run
from screen import parse_variant
from synthetic import DATASETS, make_dataset

OUT = Path(__file__).resolve().parents[2] / "results" / "replica"
CFGS = list(product((4, 8, 16), (1, 2, 4)))


def job(args):
    variant, ds, seed, rw_rounds, rw_modes, subdir = args
    path = OUT / subdir / f"{ds}_s{seed}.npz"
    if path.exists():
        return str(path)
    cfg = parse_variant(variant)
    X, y = make_dataset(ds, seed)
    n = len(y)
    res = {"y": y}
    for L, P in CFGS:
        tag = f"L{L}P{P}"
        r = run(X, L, P, seed, cfg)
        res[f"{tag}|tracin"] = project_sum(r["si"], L + P, n).astype(np.float32)
        res[f"{tag}|tracin_tgt"] = project_targets_mean(r["si"], L, P, n).astype(np.float32)
        res[f"{tag}|pd"] = project_targets_mean(r["loss_final"], L, P, n).astype(np.float32)
        res[f"{tag}|epochs"] = np.array(r["epochs"])
        for mode in rw_modes:
            s = r["si"]
            for k in range(1, rw_rounds + 1):
                w = weights_from_influence(s, mode)
                rr = run(X, L, P, seed, cfg, weights=w)
                s = rr["si"]
                res[f"{tag}|rw_{mode}_r{k}"] = project_sum(s, L + P, n).astype(np.float32)
                res[f"{tag}|rw_{mode}_r{k}_tgt"] = project_targets_mean(s, L, P, n).astype(np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    tmp.rename(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rw_rounds", type=int, default=2)
    ap.add_argument("--rw_modes", default="soft,trim")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    subdir = f"synth_{a.variant}"
    (OUT / subdir).mkdir(parents=True, exist_ok=True)
    modes = [m for m in a.rw_modes.split(",") if m]
    jobs = [(a.variant, ds, s, a.rw_rounds, modes, subdir) for s in range(a.seeds) for ds in DATASETS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, p in enumerate(pool.imap_unordered(job, jobs), 1):
            if i % 15 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
