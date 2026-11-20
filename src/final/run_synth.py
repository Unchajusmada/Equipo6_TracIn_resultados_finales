"""Sintéticos: 15 datasets × semillas × 9 (L, P), todas las variantes.

Salida: results/final/synth/<ds>_s<seed>.npz con claves "L<L>P<P>|<variante>"
(scores por ventana) e "y".
Uso:    python run_synth.py --seeds 20 --workers 20
"""
from __future__ import annotations

import argparse
import os
from itertools import product

os.environ.setdefault("OMP_NUM_THREADS", "1")

from core import RES, all_variants, save  # noqa: E402

CFGS = list(product((4, 8, 16), (1, 2, 4)))


def job(args):
    ds, seed = args
    path = RES / "synth" / f"{ds}_s{seed}.npz"
    if path.exists():
        return path.name
    import torch
    torch.set_num_threads(1)
    from screen import parse_variant
    from synthetic import make_dataset
    X, y = make_dataset(ds, seed)
    res = {"y": y}
    for L, P in CFGS:
        for k, v in all_variants(X, y, L, P, seed, parse_variant("base")).items():
            res[f"L{L}P{P}|{k}"] = v
    save(path, res)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    from synthetic import DATASETS
    jobs = [(ds, s) for s in range(a.seeds) for ds in DATASETS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, name in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"{i}/{len(jobs)} {name}", flush=True)


if __name__ == "__main__":
    main()
