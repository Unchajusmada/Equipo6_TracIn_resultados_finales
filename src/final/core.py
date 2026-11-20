"""Evaluación final (ver ../../PROTOCOLO_FINAL.md): todas las variantes de
entrenamiento de un mismo (serie, L, P, semilla) en una sola llamada, para que
las comparaciones sean pareadas exactamente.

Variantes de entrenamiento (score por VENTANA = self-influence de TracIn):
  base        método del artículo (pesos uniformes)
  pd          error final del mismo modelo base (contexto)
  soft_r1/2   mejora 2: reponderación suave, z robusto del log > 2, 2 rondas
  trim_r1/2   mejora 2: recorte duro z > 2, 2 rondas
  adapt_r1/3  mejoras 1+2: recorte de las ventanas sobre el umbral IQR, 3 rondas

La proyección a instantes (mejora 3) y los umbrales (mejora 1) se aplican al
evaluar, así un mismo entrenamiento sirve para todas las combinaciones.
Diagnóstico (no se usa en ningún método): fracción de ventanas recortadas y
fracción de ventanas anómalas recortadas en cada ronda.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "replica"))

import numpy as np

from common import thr_iqr
from forecaster import weights_from_influence
from lab import Cfg, run

RES = ROOT.parent / "results" / "final"
SOFT_ROUNDS = TRIM_ROUNDS = 2
ADAPT_ROUNDS = 3


def adapt_weights(si: np.ndarray) -> np.ndarray:
    """Mejoras 1+2: se descartan las ventanas cuya self-influence supera el
    umbral IQR de la mejora 1 (idéntico a rep_contam_fix.py)."""
    return np.where(si > thr_iqr(si), 0.0, 1.0)


def window_labels(y: np.ndarray, M: int, L: int, P: int) -> np.ndarray:
    c = np.r_[0, np.cumsum(y.astype(int))]
    i = np.arange(M)
    return (c[i + L + P] - c[i]) > 0


def all_variants(X, y, L, P, seed, cfg: Cfg, device="cpu", modes=("soft", "trim", "adapt")) -> dict:
    out = {}
    r = run(X, L, P, seed, cfg, device=device)
    M = len(r["si"])
    wy = window_labels(y, M, L, P) if y is not None else None
    out["base"], out["pd"] = r["si"], r["loss_final"]
    out["diag_epochs_base"] = r["epochs"]
    plan = {"soft": (SOFT_ROUNDS, lambda s: weights_from_influence(s, "soft")),
            "trim": (TRIM_ROUNDS, lambda s: weights_from_influence(s, "trim")),
            "adapt": (ADAPT_ROUNDS, adapt_weights)}
    for mode in modes:
        rounds, wfn = plan[mode]
        s = r["si"]
        for k in range(1, rounds + 1):
            w = wfn(s)
            out[f"diag_{mode}_r{k}_frac"] = float((w < 1).mean())
            out[f"diag_{mode}_r{k}_wmean"] = float(w.mean())
            if wy is not None:
                out[f"diag_{mode}_r{k}_anom"] = float((w[wy] < 1).mean()) if wy.any() else np.nan
                out[f"diag_{mode}_r{k}_norm"] = float((w[~wy] < 1).mean())
            s = run(X, L, P, seed, cfg, device=device, weights=w)["si"]
            out[f"{mode}_r{k}"] = s
    return out


def save(path: Path, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    arr = {k: (np.asarray(v, dtype=np.float32) if np.ndim(v) else np.asarray(v)) for k, v in res.items()}
    np.savez_compressed(tmp, **arr)
    tmp.rename(path)
