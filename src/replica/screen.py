"""Barrido de réplica en sintéticos: cada variante de lo no especificado,
evaluada con todas las lecturas del protocolo del artículo.

Uso:  python screen.py --variants base,lr1e-2 --seeds 5 --out screen.csv
Reanudable: omite (variante, dataset, semilla, L, P) ya presentes.
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
import pandas as pd

from common import project_sum, project_targets_mean, valid_mask
from synthetic import DATASETS, make_dataset
from lab import Cfg, run
from protocol import evaluate

OUT = Path(__file__).resolve().parents[2] / "results" / "replica"

VARIANTS = {
    "base": Cfg(),
    "lr1e-2": Cfg(lr=1e-2),
    "b8": Cfg(batch=8),
    "b1": Cfg(batch=1),
    "ep300": Cfg(max_epochs=300, patience=30),
    "val20": Cfg(val_frac=0.2),
    "minmax": Cfg(norm="minmax"),
    "nonorm": Cfg(norm="none"),
    "full": Cfg(grad="full"),
    "nolrw": Cfg(lr_weight=False),
    "h128": Cfg(hidden=128),
    "nosched": Cfg(sched=False),
    "h16": Cfg(hidden=16),
    "ep30": Cfg(max_epochs=30, patience=30),
}


def parse_variant(name: str) -> Cfg:
    """'a+b' combina variantes (los campos de b pisan a los de a)."""
    d = {}
    for part in name.split("+"):
        base = Cfg().__dict__
        d.update({k: v for k, v in VARIANTS[part].__dict__.items() if v != base[k]})
    return Cfg(**d)


def job(args):
    vname, ds, seed, L, P = args
    cfg = parse_variant(vname)
    X, y = make_dataset(ds, seed)
    r = run(X, L, P, seed, cfg)
    n = len(y)
    m = valid_mask(n)
    scores = {"tracin": project_sum(r["si"], L + P, n),
              "tracin_tgt": project_targets_mean(r["si"], L, P, n),
              "pd": project_targets_mean(r["loss_final"], L, P, n)}
    rows = []
    for sc, s in scores.items():
        e = evaluate(y[m], s[m], tau=0.85)
        rows.append(dict(variant=vname, ds=ds, seed=seed, L=L, P=P, score=sc, epochs=r["epochs"], **e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="screen.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / a.out
    done = set()
    if path.exists():
        d = pd.read_csv(path, dtype={"ds": str})
        done = set(zip(d.variant, d.ds, d.seed, d.L, d.P))
    jobs = [(v, ds, s, L, P) for v in a.variants.split(",") for ds in DATASETS
            for s in range(a.seed0, a.seed0 + a.seeds) for L, P in product((4, 8, 16), (1, 2, 4))]
    jobs = [j for j in jobs if j not in done]
    # los más lentos primero (L grande, variantes caras)
    jobs.sort(key=lambda j: (-("b1" in j[0]) - ("full" in j[0]), -j[3]))
    print(f"{len(jobs)} trabajos", flush=True)
    import multiprocessing as mp
    header = not path.exists()
    with mp.get_context("spawn").Pool(a.workers) as pool, open(path, "a") as fh:
        for i, rows in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
            pd.DataFrame(rows).to_csv(fh, header=header, index=False)
            header = False
            fh.flush()
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
