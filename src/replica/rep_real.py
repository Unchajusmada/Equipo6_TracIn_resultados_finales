"""Réplica en datos reales con la configuración reproducida (TracIn con
gradientes de todos los parámetros) y nuestras mejoras sobre ella.

Por (dataset, L, P, semilla) guarda results/replica/real_<variante>/<ds>_L<L>P<P>_s<seed>.npz
con los scores por instante: tracin, tracin_tgt, pd y rw_<modo>_r1[_tgt].

Uso:  python rep_real.py --datasets gecco,swan_sf --variant full --cfgs 4x1,8x2,16x4 --seeds 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import project_sum, project_targets_mean
from data_real import load
from forecaster import weights_from_influence
from lab import Cfg, run
from screen import parse_variant

OUT = Path(__file__).resolve().parents[2] / "results" / "replica"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--cfgs", default="4x1,8x2,16x4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rw_modes", default="soft,trim")
    a = ap.parse_args()
    base = parse_variant(a.variant)
    cfg = Cfg(**{**base.__dict__, "batch": 256, "max_epochs": 50})   # mismo entrenamiento que el experimento principal
    d = OUT / f"real_{a.variant}"
    d.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(","):
        X, y = load(ds)[:2]
        n = len(y)
        for seed in range(a.seeds):
            for cf in a.cfgs.split(","):
                L, P = map(int, cf.split("x"))
                path = d / f"{ds}_L{L}P{P}_s{seed}.npz"
                if path.exists():
                    continue
                t0 = time.time()
                r = run(X, L, P, seed, cfg, device="cuda")
                res = {"tracin": project_sum(r["si"], L + P, n), "tracin_tgt": project_targets_mean(r["si"], L, P, n),
                       "pd": project_targets_mean(r["loss_final"], L, P, n), "epochs": np.array(r["epochs"])}
                for mode in [m for m in a.rw_modes.split(",") if m]:
                    rr = run(X, L, P, seed, cfg, device="cuda", weights=weights_from_influence(r["si"], mode))
                    res[f"rw_{mode}_r1"] = project_sum(rr["si"], L + P, n)
                    res[f"rw_{mode}_r1_tgt"] = project_targets_mean(rr["si"], L, P, n)
                res = {k: (v.astype(np.float32) if v.ndim else v) for k, v in res.items()}
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **res)
                tmp.rename(path)
                print(f"{ds} L{L}P{P} s{seed} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
