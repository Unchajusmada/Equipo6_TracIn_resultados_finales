"""¿Se puede frenar la degradación restante con la contaminación? Variantes
sobre la misma serie de rep_contam.py (mismos datos, semillas, L4P1), todas
con hiperparámetros fijados antes de ver resultados y medidas con AUROC:

  early     TracIn sólo con el primer cuarto de las épocas. Las redes aprenden
            primero lo común y memorizan lo atípico al final (Arpit et al.,
            2017; Feldman y Zhang, 2020): las épocas tempranas deberían
            separar mejor las anomalías aunque sean muchas.
  adapt     Recorte adaptativo (mejoras 1 + 2): en cada ronda se descartan las
            ventanas cuya self-influence supera el umbral IQR (Q3 + 1.5·IQR)
            de la mejora 1, así la fracción recortada crece con la
            contaminación; el recorte z > 2 actual recorta una fracción casi
            fija. 3 rondas.
  h16       LSTM de 16 unidades en vez de 64: menos capacidad para memorizar.

Todas se proyectan al objetivo (mejora 3). Guarda también, como diagnóstico,
qué fracción de ventanas recorta cada regla y qué fracción de las ventanas
anómalas (las etiquetas sólo se usan para ese diagnóstico).

Salida: results/replica/contam_fix/r<r>_s<seed>.npz
Uso:    python rep_contam_fix.py --seeds 10 --workers 18
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import project_sum, project_targets_mean, thr_iqr
from forecaster import weights_from_influence
from lab import run
from rep_contam import L, LENGTH, P, RATIOS
from screen import parse_variant
from synthetic import make_dataset

OUT = Path(__file__).resolve().parents[2] / "results" / "replica" / "contam_fix"
EARLY_FRAC = 0.25
ADAPT_ROUNDS = 3


def adapt_weights(si: np.ndarray) -> np.ndarray:
    return np.where(si > thr_iqr(si), 0.0, 1.0)


def window_labels(y: np.ndarray, M: int) -> np.ndarray:
    """Ventana anómala si alguno de sus instantes (entrada u objetivo) lo es."""
    c = np.r_[0, np.cumsum(y.astype(int))]
    i = np.arange(M)
    return (c[i + L + P] - c[i]) > 0


def job(args):
    ratio, seed, redo = args
    path = OUT / f"r{ratio:.2f}_s{seed}.npz"
    if path.exists() and not redo:
        return str(path)
    import torch
    torch.set_num_threads(1)
    cfg = parse_variant("base")
    X, y = make_dataset("01234", 10_000 + seed, length=LENGTH, ratio=ratio / 5)
    n = len(y)
    tgt = lambda s: project_targets_mean(s, L, P, n)
    res = {"y": y}
    r = run(X, L, P, seed, cfg, keep_g2=True)
    M = len(r["si"])
    wy = window_labels(y, M)
    res["tracin_tgt"] = tgt(r["si"])
    k = max(1, math.ceil(EARLY_FRAC * r["epochs"]))
    res["early_tgt"] = tgt(r["g2"][:k].sum(0))
    res["epochs"] = r["epochs"]
    # diagnóstico del recorte actual (z > 2): cuánto recorta y cuántas anómalas
    w = weights_from_influence(r["si"], "trim")
    res["diag_trim_frac"], res["diag_trim_anom"] = (w == 0).mean(), (w[wy] == 0).mean()
    s = r["si"]
    for k_ in range(1, ADAPT_ROUNDS + 1):
        w = adapt_weights(s)
        res[f"diag_adapt_r{k_}_frac"], res[f"diag_adapt_r{k_}_anom"] = (w == 0).mean(), (w[wy] == 0).mean()
        s = run(X, L, P, seed, cfg, weights=w)["si"]
        res[f"adapt_r{k_}_tgt"] = tgt(s)
        res[f"adapt_r{k_}"] = project_sum(s, L + P, n)  # mejoras 1 + 2 sin la 3
    res["h16_tgt"] = tgt(run(X, L, P, seed, parse_variant("h16"))["si"])
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k_: np.asarray(v, dtype=np.float32) for k_, v in res.items()})
    tmp.rename(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--redo", action="store_true", help="recalcular aunque el archivo exista")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, s, a.redo) for s in range(a.seeds) for r in RATIOS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, p in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"{i}/{len(jobs)} {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
