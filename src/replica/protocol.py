"""Protocolo de evaluación del artículo, en todas sus lecturas posibles.

Para un score por instante devuelve:
* auprc, oracle (F1 binario máximo): calidad del score sin umbral.
* Estático (Ec. 6, τ = 0.85 sintéticos / 0.87 reales) en dos lecturas del
  umbral (percentil ``pct`` o fracción del máximo ``frac``) y dos lecturas
  del F1 (binario ``bin`` o macro ``mac`` = promedio del F1 de ambas clases,
  ``f1_score(average='macro')`` de scikit-learn).
* Dinámico (Secc. 4.3 y Tabla 3): la serie se divide en una parte inicial
  (proporción ``prop`` = 0.5) donde se elige el percentil que maximiza el F1,
  y el resto, donde se aplica y se mide. Con criterio y medida binarios o
  macro.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

PCTS = np.arange(50.0, 99.95, 0.1)


def f1_pair(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(F1 binario de la clase anómala, F1 macro)."""
    tp = np.sum(y & p); fp = np.sum(~y & p); fn = np.sum(y & ~p); tn = np.sum(~y & ~p)
    fa = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    fn_ = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return float(fa), float((fa + fn_) / 2)


def oracle_f1(y, s):
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    return float(np.max(2 * tp / (k + ys.sum()))) if ys.sum() else float("nan")


def _best_pct(y, s, s_all, crit):
    """Percentil (de la distribución completa del score) que maximiza el F1
    en (y, s). Vectorizado sobre la grilla de percentiles."""
    thr = np.percentile(s_all, PCTS)
    P = s[None, :] >= thr[:, None]
    Y = y[None, :]
    tp = (P & Y).sum(1); fp = (P & ~Y).sum(1); fn = (~P & Y).sum(1); tn = (~P & ~Y).sum(1)
    fa = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    fnm = np.where(tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    f = fa if crit == "bin" else (fa + fnm) / 2
    return thr[int(np.argmax(f))]


def evaluate(y: np.ndarray, s: np.ndarray, tau: float, prop: float = 0.5) -> dict:
    y = y.astype(bool)
    s = np.asarray(s, dtype=np.float64)
    s = s / s.max()
    out = {"auprc": float(average_precision_score(y, s)), "oracle": oracle_f1(y, s)}
    for nm, thr in (("pct", np.percentile(s, tau * 100)), ("frac", tau)):
        b, m = f1_pair(y, s >= thr)
        out[f"S_{nm}_bin"], out[f"S_{nm}_mac"] = b, m
    k = int(len(s) * prop)
    for crit in ("bin", "mac"):
        thr = _best_pct(y[:k], s[:k], s, crit)
        b, m = f1_pair(y[k:], s[k:] >= thr)
        out[f"D_{crit}"] = b if crit == "bin" else m
    return out
