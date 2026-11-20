"""Experimento 2: los 4 datasets reales del artículo (Tabla 4).

Cada trabajo es independiente, guarda su propio archivo y se omite si ya
existe (el experimento se puede reanudar). Fase A: forecasters (LSTM con 5
semillas y Transformer con 3) y métodos de referencia. Fase B: reponderación
(propuesta 2) a partir de la self-influence guardada en la fase A.

Uso:  python run_real.py [--workers 6] [--datasets gecco swan_sf ...]
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real" / "scores"

LS, PS = [4, 8, 16], [1, 2, 4]
WS, MS = [1, 4, 8, 16], [4, 8, 16]
DEFAULT = (4, 1)
LSTM_SEEDS, TF_SEEDS, RW_SEEDS, BASE_SEEDS = range(5), range(3), range(5), range(5)
# El barrido completo (L, P) se corre sólo para el LSTM base (comparación
# "estilo artículo"). Transformer y reponderación usan la configuración fijada
# a priori (L=4, P=1), que es la que se usa en todas las conclusiones; así el
# costo de la fase B baja de ~6 h a ~1 h en la GPU disponible.
TRAIN = dict(batch=256, max_epochs=50, patience=10)


def path_of(job) -> Path:
    ds, kind, *rest = job
    return OUT / ds / ("_".join([kind] + [str(r) for r in rest]) + ".npz")


def _save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run_job(job):
    """Envoltura tolerante a fallos: un error (p. ej. memoria de GPU) no
    detiene a los demás trabajos; se reporta y se reintenta al final."""
    import traceback
    try:
        return _run_job(job)
    except Exception as e:  # noqa: BLE001
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return job, -1.0, f"{type(e).__name__}: {e}".splitlines()[0] + "\n" + traceback.format_exc(limit=2)


def _run_job(job):
    path = path_of(job)
    if path.exists():
        return job, 0.0, ""
    import torch
    torch.set_num_threads(2)
    from baselines import autoencoder, iforest, lof, matrix_profile, ocsvm
    from data_real import load
    from forecaster import reweighted_runs, train_trace

    t0 = time.time()
    ds, kind, *rest = job
    X, y = load(ds)
    dev = "cuda"
    if kind in ("lstm", "tf"):
        L, P, seed = rest
        tr = train_trace(X, L, P, seed, arch=kind, device=dev, **TRAIN)
        out = {k: v.astype(np.float32) for k, v in tr.timestep_scores().items()}
        out["sample_tracin"] = tr.self_influence().astype(np.float32)
        out["epochs"] = np.array(len(tr.lrs))
        out["train_loss"] = tr.train_loss
        if kind == "lstm" and (L, P) == DEFAULT and seed == 0:
            out["trace_loss"] = tr.loss.astype(np.float16)
            out["trace_hsq"] = tr.hsq.astype(np.float16)
            out["trace_lrs"] = tr.lrs
        _save(path, **out)
    elif kind == "rw":
        L, P, seed, mode = rest
        base_file = path_of((ds, "lstm", L, P, seed))
        z = np.load(base_file)

        class _Base:  # sólo se necesita la self-influence por ventana
            def self_influence(self):
                return z["sample_tracin"].astype(np.float64)

        out = {}
        for r, (tr, w) in enumerate(reweighted_runs(X, L, P, seed, _Base(), mode, device=dev, **TRAIN), start=1):
            ts = tr.timestep_scores()
            out[f"r{r}"] = ts["tracin"].astype(np.float32)
            out[f"r{r}_tgt"] = ts["tracin_tgt"].astype(np.float32)
            out[f"r{r}_sample"] = tr.self_influence().astype(np.float32)
            out[f"r{r}_w"] = w.astype(np.float16)
        _save(path, **out)
    elif kind == "iforest":
        w, seed = rest
        _save(path, score=iforest(X, w, seed).astype(np.float32))
    elif kind == "ocsvm":
        w, seed = rest
        _save(path, score=ocsvm(X, w, seed).astype(np.float32))
    elif kind == "lof":
        (w,) = rest
        _save(path, score=lof(X, w, device=dev).astype(np.float32))
    elif kind == "ae":
        w, seed = rest
        _save(path, score=autoencoder(X, w, seed, epochs=20, batch=512, device=dev).astype(np.float32))
    elif kind == "mp":
        (m,) = rest
        _save(path, score=matrix_profile(X, m, device=dev).astype(np.float32))
    else:
        raise ValueError(kind)
    return job, time.time() - t0, ""


def jobs_phase_a(datasets):
    jobs = []
    for ds in datasets:
        jobs += [(ds, "lstm", L, P, s) for s in LSTM_SEEDS for L, P in product(LS, PS)]
        jobs += [(ds, "lof", w) for w in WS]
        jobs += [(ds, "mp", m) for m in MS]
        jobs += [(ds, "ae", w, s) for s in BASE_SEEDS for w in WS]
        jobs += [(ds, "iforest", w, s) for s in BASE_SEEDS for w in WS]
        jobs += [(ds, "ocsvm", w, s) for s in BASE_SEEDS for w in WS]
        jobs += [(ds, "tf", *DEFAULT, s) for s in TF_SEEDS]
    return jobs


def jobs_phase_b(datasets):
    return [(ds, "rw", *DEFAULT, s, mode) for ds in datasets for s in RW_SEEDS for mode in ("soft", "trim")]


def main():
    import multiprocessing as mp
    from data_real import NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", nargs="+", default=NAMES)
    ap.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    t0 = time.time()
    for ph, jobs in (("a", jobs_phase_a(a.datasets)), ("b", jobs_phase_b(a.datasets))):
        if ph not in a.phase:
            continue
        failed = []
        with ctx.Pool(a.workers, maxtasksperchild=8) as pool:
            for i, (job, dt, err) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                if err:
                    failed.append(job)
                    print(f"[{ph} {i}/{len(jobs)}] FALLÓ {job}: {err.splitlines()[0]} (se reintentará)", flush=True)
                else:
                    print(f"[{ph} {i}/{len(jobs)}] {job} {dt:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
        if failed:  # reintento en serie (sin competir por memoria de GPU)
            with ctx.Pool(1) as pool:
                for job, dt, err in pool.imap_unordered(run_job, failed):
                    print(f"[{ph} reintento] {job} " + (f"FALLÓ DE NUEVO: {err}" if err else f"{dt:.0f}s"), flush=True)


if __name__ == "__main__":
    main()
