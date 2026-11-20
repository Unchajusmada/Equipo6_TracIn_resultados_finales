"""Evaluación: aplica a TODOS los scores guardados los mismos protocolos.

Para cada vector de scores por instante (restringido a los instantes
válidos, ver ``common.EDGE``) se calculan:

* Independientes del umbral: AUPRC, AUROC, F1 máximo (oráculo).
* Umbrales del artículo: estático (tau = 0.85 sintéticos / 0.87 reales, en
  sus dos lecturas) y dinámico (validación cruzada en 2 bloques).
* Barrido de "contaminación" tipo PyOD/TODS (marcar el c % más alto), que es
  como Lai et al. (2021) reportan los métodos de referencia del artículo.
* Umbrales sin etiquetas (propuesta 1): IQR, MAD, log-IQR, log-MAD, POT.

Salida: un CSV largo por experimento en results/<exp>/eval_long.csv.

Uso:  python evaluate.py synthetic|real|contamination
"""
from __future__ import annotations

import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from common import evaluate_scores, prf, valid_mask

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

PCTS = {"synthetic": [0.05, 0.10, 0.15, 0.20, 0.25], "real": [0.01, 0.10],
        "contamination": [0.05, 0.10, 0.15, 0.20, 0.25]}
TAU = {"synthetic": 0.85, "real": 0.87, "contamination": 0.85}


def eval_vector(y, s, exp):
    e = evaluate_scores(y, s, TAU[exp])
    for c in PCTS[exp]:
        thr = np.quantile(s, 1 - c)
        e[f"cont{c:.2f}_f1"] = prf(y, s >= thr)[2]
    return e


def parse_key(key: str) -> dict:
    """'lstm|L4P1|tracin' -> family/method/config/variant."""
    parts = key.split("|")
    if parts[0] in ("lstm", "tf"):
        if len(parts) == 2:  # experimento de contaminación: 'lstm|tracin'
            return {"method": parts[0], "config": "L4P1", "variant": parts[1]}
        return {"method": parts[0], "config": parts[1], "variant": parts[2]}
    return {"method": parts[0], "config": parts[1], "variant": "score"}


SKIP = re.compile(r"(trace|_w$|epochs|^X$|^y$)")  # rw_*_r{k}_w son pesos, no scores


def _eval_file_synth(path: Path, exp: str):
    z = np.load(path)
    y = z["y"].astype(int)
    v = valid_mask(len(y))
    m = re.match(r"(r?[\d.]+)_s(\d+)", path.stem)
    ident, seed = m.group(1), int(m.group(2))
    rows = []
    for k in z.files:
        if SKIP.search(k):
            continue
        row = {"dataset": ident, "seed": seed, **parse_key(k), **eval_vector(y[v], z[k][v].astype(np.float64), exp)}
        if exp == "contamination":
            row["true_rate"] = float(y[v].mean())
        rows.append(row)
    return rows


def evaluate_synthetic(exp: str):
    files = sorted((RES / exp / "scores").glob("*_s*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    with Pool(16) as pool:
        out = pool.starmap(_eval_file_synth, [(f, exp) for f in files])
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(RES / exp / "eval_long.csv", index=False)
    print(f"{exp}: {len(files)} archivos, {len(df)} filas")


def _eval_file_real(path: Path):
    from data_real import PROC
    ds = path.parent.name
    y = np.load(PROC / f"{ds}.npz")["y"].astype(int)
    v = valid_mask(len(y))
    z = np.load(path)
    parts = path.stem.split("_")
    kind = parts[0]
    rows = []
    if kind in ("lstm", "tf"):
        L, P, seed = parts[1:4]
        for var in ("tracin", "tracin_last", "losstraj", "lossfinal", "pd", "tracin_tgt", "losstraj_tgt"):
            rows.append({"dataset": ds, "seed": int(seed), "method": kind, "config": f"L{L}P{P}",
                         "variant": var, "epochs": int(z["epochs"]),
                         **eval_vector(y[v], z[var][v].astype(np.float64), "real")})
    elif kind == "rw":
        L, P, seed, mode = parts[1:5]
        for r in (1, 2, 3):
            for suf in ("", "_tgt"):
                rows.append({"dataset": ds, "seed": int(seed), "method": "lstm", "config": f"L{L}P{P}",
                             "variant": f"rw_{mode}_r{r}{suf}",
                             **eval_vector(y[v], z[f"r{r}{suf}"][v].astype(np.float64), "real")})
    else:
        cfg = parts[1]
        seed = int(parts[2]) if len(parts) > 2 else 0
        prefix = "m" if kind == "mp" else "w"
        rows.append({"dataset": ds, "seed": seed, "method": kind, "config": f"{prefix}{cfg}",
                     "variant": "score", **eval_vector(y[v], z["score"][v].astype(np.float64), "real")})
    return rows


def evaluate_real():
    files = sorted(f for f in (RES / "real" / "scores").glob("*/*.npz") if ".tmp" not in f.name)
    with Pool(8) as pool:
        out = pool.map(_eval_file_real, files)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(RES / "real" / "eval_long.csv", index=False)
    print(f"real: {len(files)} archivos, {len(df)} filas")


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "real":
        evaluate_real()
    else:
        evaluate_synthetic(which)
