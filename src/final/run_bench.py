"""Extensión preregistrada: SMD, SMAP/MSL y contaminación semisintética sobre
fondo real («semi»). Misma configuración que run_real.py (gradiente de todos
los parámetros, batch 256, 50 épocas máx., L4P1), todas las variantes de
entrenamiento + LOF e IForest.

Reparto dinámico con candados (como run_real.py --claim): se pueden lanzar
varios procesos (GPU o CPU) en cualquier momento sin repetir trabajo.

Salida: results/final/{smd,nasa,semi}/<nombre>.npz
Uso:    python run_bench.py --blocks smd,nasa,semi [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

from core import RES, ROOT, all_variants, save  # noqa: E402

sys.path.insert(0, str(ROOT))

L, P = 4, 1
SEEDS = {"smd": 5, "nasa": 5, "semi": 3}
MIXES = ["01234", "01", "234"]
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.30]


def _claim(lockdir, name: str) -> bool:
    """Candado atómico por trabajo; si su proceso murió, se reemplaza (igual que run_real.py)."""
    lockdir.mkdir(parents=True, exist_ok=True)
    lk = lockdir / f"{name}.lock"
    for _ in range(2):
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                os.kill(int(lk.read_text() or 0), 0)
                return False
            except (ValueError, ProcessLookupError):
                lk.unlink(missing_ok=True)
            except PermissionError:
                return False
    return False


def jobs(block):
    import data_bench as db
    if block == "smd":
        return [(f"{m}_s{s}", ("smd_" + m, s)) for s in range(SEEDS[block]) for m in db.smd_names()]
    if block == "nasa":
        return [(f"{c}_s{s}", ("nasa_" + c, s)) for s in range(SEEDS[block]) for c in db.nasa_names()]
    out = []
    for s in range(SEEDS[block]):
        for r in RATIOS:
            for mi, mix in enumerate(MIXES):
                for bi, m in enumerate(db.semi_machines()):
                    out.append((f"{m}_{mix}_r{r:.2f}_s{s}", (m, mix, r, s, bi, mi)))
    return out


def make(block, spec):
    import data_bench as db
    if block in ("smd", "nasa"):
        name, s = spec
        X, y = db.load_bench(name)
        return X, y, s
    m, mix, r, s, bi, mi = spec
    bg = db.standardize(__import__("numpy").load(db.OUT / f"bg_{m}.npz")["X"])
    X, y = db.inject(bg, mix, r, seed=30_000 + 1000 * s + 100 * bi + 10 * mi + RATIOS.index(r))
    return X, y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="smd,nasa,semi")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1 if a.device == "cuda" else 2)
    from baselines import iforest, lof
    from lab import Cfg, normalize
    from screen import parse_variant
    cfg = Cfg(**{**parse_variant("full").__dict__, "batch": 256, "max_epochs": 50})
    for block in a.blocks.split(","):
        for stem, spec in jobs(block):
            path = RES / block / f"{stem}.npz"
            if path.exists() or not _claim(RES / block / ".locks", stem):
                continue
            if path.exists():
                continue
            t0 = time.time()
            X, y, s = make(block, spec)
            res = {"y": y, **all_variants(X, y, L, P, s, cfg, device=a.device)}
            Xz = normalize(X.astype(np.float64), "z").astype(np.float32)
            res["ts_lof"] = lof(Xz, 1, device=a.device)
            res["ts_iforest"] = iforest(Xz, 1, s)
            save(path, res)
            print(f"{block} {stem} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
