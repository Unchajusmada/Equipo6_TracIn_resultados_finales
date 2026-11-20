"""Reales (GPU): Credit Card, CICIDS, GECCO, SWAN-SF, L4P1, todas las
variantes, con la configuración reproducida (gradiente de todos los
parámetros, batch 256, 50 épocas máx., igual que rep_real.py).

Los trabajos se ordenan por semilla (todos los datasets de la semilla 0, luego
la 1, ...). Con --claim cada proceso reclama un trabajo creando un candado
(results/final/real/.locks/), así se pueden lanzar o agregar procesos en
cualquier momento sin repetir trabajo; los candados de procesos muertos se
ignoran. Sin --claim se reparten con --nproc/--rank. Reanudable.

Salida: results/final/real/<ds>_s<seed>.npz
Uso:    python run_real.py --seeds 10 --nproc 3 --rank 0   (y rank 1, 2)
"""
from __future__ import annotations

import argparse
import os
import time

from core import RES, all_variants, save

DATASETS = ["gecco", "swan_sf", "cicids", "creditcard"]
L, P = 4, 1
LOCKS = RES / "real" / ".locks"


def claim(name: str) -> bool:
    """Crea el candado de forma atómica; si existe pero su proceso murió, lo reemplaza."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    lk = LOCKS / f"{name}.lock"
    for _ in range(2):
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(lk.read_text() or 0)
                os.kill(pid, 0)
                return False          # su dueño sigue vivo
            except (ValueError, ProcessLookupError):
                lk.unlink(missing_ok=True)
            except PermissionError:
                return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--claim", action="store_true", help="reparto dinámico con candados")
    a = ap.parse_args()
    from data_real import load
    from lab import Cfg
    from screen import parse_variant
    base = parse_variant("full")
    cfg = Cfg(**{**base.__dict__, "batch": 256, "max_epochs": 50})
    jobs = [(ds, s) for s in range(a.seed0, a.seed0 + a.seeds) for ds in a.datasets.split(",")]
    cache = {}
    for i, (ds, seed) in enumerate(jobs):
        if not a.claim and i % a.nproc != a.rank:
            continue
        path = RES / "real" / f"{ds}_s{seed}.npz"
        if path.exists() or (a.claim and not claim(path.stem)):
            continue
        if path.exists():               # terminó otro proceso entre ambas revisiones
            continue
        if ds not in cache:
            cache.clear()
            cache[ds] = load(ds)[:2]
        X, y = cache[ds]
        t0 = time.time()
        res = {"y": y, **all_variants(X, y, L, P, seed, cfg, device="cuda")}
        save(path, res)
        print(f"{ds} s{seed} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
