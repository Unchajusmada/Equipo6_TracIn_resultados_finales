"""Utilidades compartidas: proyección de scores, umbrales y métricas.

Convenciones
------------
* Una serie es un arreglo ``X`` de forma (N, C) y etiquetas ``y`` de forma (N,).
* Una "muestra" (ventana) i abarca los instantes [i, i + span), donde
  span = L + P para el forecaster y span = w para métodos basados en ventanas.
* Todos los métodos producen un score por instante; mayor = más anómalo.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

# Instantes excluidos en cada extremo de la serie para la evaluación. Es el
# mayor span posible del barrido (L=16, P=4 -> 19 pasos). Se usa el mismo
# valor para todos los métodos y configuraciones, así todos se evalúan sobre
# exactamente los mismos instantes (el artículo descarta los extremos).
EDGE = 19


# --------------------------------------------------------------------------
# Proyección muestra -> instante
# --------------------------------------------------------------------------
def project_sum(sample_scores: np.ndarray, span: int, n: int) -> np.ndarray:
    """Ec. (4) del artículo: cada instante suma el score de todas las
    ventanas que lo contienen. La ventana i cubre [i, i + span)."""
    s = np.asarray(sample_scores, dtype=np.float64)
    acc = np.zeros(n + 1)
    m = len(s)
    acc[:m] += s
    acc[span:span + m] -= s
    return np.cumsum(acc)[:n]


def project_targets_mean(sample_scores: np.ndarray, L: int, P: int, n: int) -> np.ndarray:
    """Proyección clásica de *prediction deviation*: el error de la ventana i
    se asigna sólo a los instantes que predice, [i + L, i + L + P), y se
    promedia sobre las predicciones que cubren cada instante."""
    s = np.asarray(sample_scores, dtype=np.float64)
    m = len(s)
    acc = np.zeros(n + 1)
    cnt = np.zeros(n + 1)
    acc[L:L + m] += s
    acc[L + P:L + P + m] -= s
    cnt[L:L + m] += 1
    cnt[L + P:L + P + m] -= 1
    acc, cnt = np.cumsum(acc)[:n], np.cumsum(cnt)[:n]
    out = np.zeros(n)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz]
    return out


def valid_mask(n: int, edge: int = EDGE) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[edge:n - edge] = True
    return m


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = np.sum(y & pred)
    fp = np.sum(~y & pred)
    fn = np.sum(y & ~pred)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return float(p), float(r), float(f)


def best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Umbral exacto que maximiza F1 (regla ``s >= thr``). Devuelve (F1, thr).
    Si no hay anomalías, devuelve (nan, inf)."""
    y = y.astype(bool)
    if y.sum() == 0:
        return float("nan"), float("inf")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    # sólo cortes donde cambia el valor del score
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, k, ss = tp[last], k[last], ss[last]
    f1 = 2 * tp / (k + y.sum())
    i = int(np.argmax(f1))
    return float(f1[i]), float(ss[i])


def best_f1(y: np.ndarray, s: np.ndarray) -> float:
    """F1 máximo sobre todos los umbrales posibles (cota superior/oráculo)."""
    return best_threshold(y, s)[0]


def threshold_free(y: np.ndarray, s: np.ndarray) -> dict:
    y = y.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auprc": float("nan"), "auroc": float("nan"), "best_f1": float("nan")}
    return {
        "auprc": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
        "best_f1": best_f1(y, s),
    }


# --------------------------------------------------------------------------
# Umbrales del artículo
# --------------------------------------------------------------------------
# El texto del artículo habla de "percentil", pero sus números sólo son
# alcanzables si tau es una fracción del score máximo (Ec. 5-6: s/max >= tau):
# con el percentil 87 se marca el 13 % de los instantes y, con 1.25 % de
# anomalías (GECCO), la precisión no podría pasar de ~0.096 (reportan 0.539).
# Se implementan ambas lecturas; la principal es la de fracción del máximo.
def thr_static_frac(s: np.ndarray, tau: float) -> float:
    """Umbral estático, lectura A (principal): s / max(s) >= tau."""
    return float(tau * np.max(s))


def thr_static_pct(s: np.ndarray, pct: float) -> float:
    """Umbral estático, lectura B (literal del texto): percentil ``pct``."""
    return float(np.percentile(s, pct))


def dynamic_crossfit(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Umbral dinámico del artículo (usa etiquetas) con validación cruzada de
    2 bloques contiguos: el umbral que maximiza F1 en una mitad de la serie
    se aplica a la otra, y viceversa. Así ningún instante se evalúa con un
    umbral elegido mirando su propia etiqueta.

    El umbral se busca de forma exacta sobre todos los valores del score
    (cubre cualquier percentil o fracción del máximo). Si la mitad de
    calibración no tiene anomalías, se usa su score máximo (sólo se marcan
    puntos más extremos que todo lo visto). Devuelve también los umbrales
    elegidos expresados como fracción del máximo global."""
    n = len(s)
    smax = float(np.max(s)) or 1.0
    halves = [np.arange(0, n // 2), np.arange(n // 2, n)]
    pred = np.zeros(n, dtype=bool)
    chosen = []
    for a, b in [(0, 1), (1, 0)]:
        ia, ib = halves[a], halves[b]
        f, t = best_threshold(y[ia], s[ia])
        if not np.isfinite(t):
            t = float(np.nextafter(s[ia].max(), np.inf))
        pred[ib] = s[ib] >= t
        chosen.append(t / smax)
    return pred, chosen


# --------------------------------------------------------------------------
# Umbrales no supervisados (propuesta 1)
# --------------------------------------------------------------------------
def _log(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    pos = s[s > 0]
    floor = pos.min() * 0.5 if pos.size else 1e-12
    return np.log(np.maximum(s, floor))


def thr_iqr(s: np.ndarray, k: float = 1.5) -> float:
    q1, q3 = np.percentile(s, [25, 75])
    return float(q3 + k * (q3 - q1))


def thr_mad(s: np.ndarray, k: float = 3.0) -> float:
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return float(med + k * mad)


def thr_pot(s: np.ndarray, q: float = 1e-3, level: float = 0.98) -> float:
    """Peaks-Over-Threshold (Siffer et al., 2017, SPOT en modo batch).
    Se ajusta una Pareto generalizada a los excesos sobre el cuantil
    ``level`` y se devuelve el cuantil z_q tal que P(S > z_q) = q."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    t = np.quantile(s, level)
    exc = s[s > t] - t
    nt = len(exc)
    if nt < 5:
        return float(s.max())
    try:
        gamma, _, sigma = stats.genpareto.fit(exc, floc=0)
    except Exception:  # pragma: no cover - ajuste numérico fallido
        return float(s.max())
    r = q * n / nt
    if abs(gamma) < 1e-8:
        return float(t - sigma * np.log(r))
    return float(t + (sigma / gamma) * (r ** (-gamma) - 1.0))


def unsupervised_thresholds(s: np.ndarray) -> dict[str, float]:
    """Todos los umbrales sin etiquetas, con hiperparámetros fijados a priori
    (k = 1.5 para IQR, k = 3 para MAD, q = 1e-3 para POT). Los de log-escala
    se devuelven ya transformados de vuelta a la escala original."""
    ls = _log(s)
    return {
        "iqr": thr_iqr(s),
        "mad": thr_mad(s),
        "log_iqr": float(np.exp(thr_iqr(ls))),
        "log_mad": float(np.exp(thr_mad(ls))),
        "pot": thr_pot(s),
    }


def evaluate_scores(y: np.ndarray, s: np.ndarray, static_tau: float) -> dict:
    """Evalúa un vector de scores por instante con todos los protocolos.
    ``y`` y ``s`` deben venir ya restringidos a los instantes válidos.
    ``static_tau`` es el umbral estático del artículo (0.85 sintéticos,
    0.87 reales), que se aplica en sus dos lecturas."""
    out = threshold_free(y, s)
    out["anomaly_rate"] = float(y.mean())
    pred = s >= thr_static_frac(s, static_tau)
    p, r, f = prf(y, pred)
    out.update({"static_p": p, "static_r": r, "static_f1": f, "static_rate": float(pred.mean())})
    p, r, f = prf(y, s >= thr_static_pct(s, 100 * static_tau))
    out.update({"staticpct_p": p, "staticpct_r": r, "staticpct_f1": f})
    pred, chosen = dynamic_crossfit(y, s)
    p, r, f = prf(y, pred)
    out.update({"dyn_p": p, "dyn_r": r, "dyn_f1": f, "dyn_tau_a": chosen[0], "dyn_tau_b": chosen[1]})
    for name, t in unsupervised_thresholds(s).items():
        pred = s >= t if np.isfinite(t) else np.zeros_like(y, dtype=bool)
        p, r, f = prf(y, pred)
        out.update({f"{name}_p": p, f"{name}_r": r, f"{name}_f1": f, f"{name}_rate": float(pred.mean())})
    return out
