"""Métodos de detección de referencia, de las tres familias del artículo.

* Majority modeling: Isolation Forest, One-Class SVM, LOF, Autoencoder.
* Prediction deviation: error del mismo forecaster LSTM (ver forecaster.py,
  variante ``pd``), así la única diferencia con TracIn es cómo se puntúa.
* Discord analysis: Matrix Profile (stumpy).

Todos se ajustan sobre la serie completa estandarizada (transductivo, igual
que el método del artículo) y devuelven un score por instante (mayor = más
anómalo). Con ventana w = 1 el método es puntual; con w > 1 opera sobre
ventanas deslizantes aplanadas (las variantes "-DA" de Lai et al., 2021), y
el score se proyecta a cada instante con la misma Ec. (4) del artículo.

Hiperparámetros por defecto de PyOD, que es lo que usa el benchmark de
Lai et al.: IForest(100 árboles, max_samples=256), OCSVM(rbf, nu=0.5,
gamma=1/d), LOF(k=20), AE([64, 32, 32, 64] adaptado a la dimensión,
dropout 0.2).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from torch import nn

from common import project_sum


def window_matrix(X: np.ndarray, w: int) -> np.ndarray:
    if w == 1:
        return X
    v = np.lib.stride_tricks.sliding_window_view(X, w, axis=0)  # (n-w+1, C, w)
    return np.ascontiguousarray(v.transpose(0, 2, 1).reshape(len(v), -1))


def _to_timesteps(win_scores: np.ndarray, w: int, n: int) -> np.ndarray:
    return project_sum(win_scores, w, n)


def iforest(X, w, seed):
    Z = window_matrix(X, w)
    m = IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=1).fit(Z)
    return _to_timesteps(-m.score_samples(Z), w, len(X))


def ocsvm(X, w, seed, max_fit=10000):
    Z = window_matrix(X, w)
    rng = np.random.default_rng(seed)
    fit = Z if len(Z) <= max_fit else Z[rng.choice(len(Z), max_fit, replace=False)]
    m = OneClassSVM(kernel="rbf", nu=0.5, gamma="auto").fit(fit)
    out = np.empty(len(Z))
    for s in range(0, len(Z), 50000):
        out[s:s + 50000] = -m.decision_function(Z[s:s + 50000])
    return _to_timesteps(out, w, len(X))


def lof(X, w, k=20, device="cpu"):
    Z = window_matrix(X, w)
    if device == "cpu" or len(Z) < 20000:
        m = LocalOutlierFactor(n_neighbors=k).fit(Z)
        return _to_timesteps(-m.negative_outlier_factor_, w, len(X))
    return _to_timesteps(lof_gpu(Z, k, device), w, len(X))


@torch.no_grad()
def lof_gpu(Z: np.ndarray, k: int = 20, device: str = "cuda", chunk: int = 1024) -> np.ndarray:
    """LOF exacto con k-NN por fuerza bruta en GPU (misma definición que
    sklearn: se excluye el propio punto, reach-dist = max(k-dist(o), d(p,o)),
    lrd = 1 / (media reach-dist + 1e-10)). Se valida contra sklearn en
    ``_selftest``."""
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=device)
    n = len(Zt)
    sq = (Zt ** 2).sum(1)
    dists = torch.empty(n, k, device=device)
    idxs = torch.empty(n, k, dtype=torch.long, device=device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (Zt[s:e] @ Zt.T)
        d2.clamp_(min=0)
        d2[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = float("inf")
        v, i = torch.topk(d2, k, dim=1, largest=False)
        dists[s:e], idxs[s:e] = v.sqrt(), i
    kdist = dists[:, -1]
    reach = torch.maximum(dists, kdist[idxs])
    lrd = 1.0 / (reach.mean(1) + 1e-10)
    lof_ = lrd[idxs].mean(1) / lrd
    return lof_.double().cpu().numpy()


class _AE(nn.Module):
    def __init__(self, D):
        super().__init__()
        h1, b = min(256, max(16, D)), min(64, max(2, D // 2))
        self.net = nn.Sequential(
            nn.Linear(D, h1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h1, b), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(b, h1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h1, D))

    def forward(self, x):
        return self.net(x)


def autoencoder(X, w, seed, epochs=100, batch=32, device="cpu"):
    torch.manual_seed(seed)
    Z = torch.as_tensor(window_matrix(X, w), dtype=torch.float32, device=device)
    m = _AE(Z.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for b in torch.randperm(len(Z), generator=gen).to(device).split(batch):
            loss = ((m(Z[b]) - Z[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        err = torch.cat([((m(Z[s:s + 65536]) - Z[s:s + 65536]) ** 2).mean(1)
                         for s in range(0, len(Z), 65536)])
    return _to_timesteps(err.cpu().numpy(), w, len(X))


def matrix_profile(X, m, device="cpu"):
    """Discord por canal con Matrix Profile (distancia z-normalizada al
    vecino más cercano no trivial). Score de la subsecuencia = máximo sobre
    canales (una anomalía en cualquier canal hace anómalo al instante).
    En CPU usa stumpy; en GPU, ``mp_torch`` (validado contra stumpy)."""
    n, C = X.shape
    P = np.full((C, n - m + 1), np.nan)
    for c in range(C):
        x = X[:, c].astype(np.float64)
        if np.ptp(x) == 0:
            continue
        if device == "cpu":
            import stumpy
            mp = stumpy.stump(x, m)[:, 0].astype(np.float64)
        else:
            mp = mp_torch(x, m, device)
        mp[~np.isfinite(mp)] = np.nan
        P[c] = mp
    win = np.nanmax(P, axis=0)
    win = np.nan_to_num(win, nan=0.0)
    return _to_timesteps(win, m, n)


@torch.no_grad()
def mp_torch(x: np.ndarray, m: int, device: str = "cuda", chunk: int = 1024) -> np.ndarray:
    """Matrix Profile exacto (self-join) por fuerza bruta en GPU.
    d^2(i, j) = 2m (1 - corr(i, j)); zona de exclusión |i - j| <= ceil(m/4)
    (la de stumpy). Subsecuencias constantes: d = 0 entre dos constantes y
    d = sqrt(m) entre una constante y una no constante (convención de stumpy)."""
    xt = torch.as_tensor(x, dtype=torch.float64, device=device)
    W = xt.unfold(0, m, 1)                       # (k, m)
    mu = W.mean(1, keepdim=True)
    sd = W.std(1, unbiased=False, keepdim=True)
    const = (sd.squeeze(1) < 1e-8)
    Zn = torch.where(const[:, None], torch.zeros_like(W), (W - mu) / sd.clamp_min(1e-12)).float()
    k = len(Zn)
    excl = int(np.ceil(m / 4))
    out = torch.empty(k, dtype=torch.float32, device=device)
    for s in range(0, k, chunk):
        e = min(s + chunk, k)
        corr = (Zn[s:e] @ Zn.T) / m
        d2 = (2 * m * (1 - corr)).clamp_(min=0)
        ci = const[s:e, None]
        cj = const[None, :]
        d2 = torch.where(ci & cj, torch.zeros_like(d2), d2)
        d2 = torch.where(ci ^ cj, torch.full_like(d2, float(m)), d2)
        rows = torch.arange(s, e, device=device)[:, None]
        cols = torch.arange(k, device=device)[None, :]
        d2.masked_fill_((rows - cols).abs() <= excl, float("inf"))
        out[s:e] = d2.min(1).values.sqrt()
    return out.double().cpu().numpy()


def _selftest():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3000, 6)).astype(np.float32)
    Z[:20] *= 5
    ref = -LocalOutlierFactor(n_neighbors=20).fit(Z).negative_outlier_factor_
    if torch.cuda.is_available():
        got = lof_gpu(Z, 20, "cuda")
        print("LOF GPU vs sklearn, max |dif| =", float(np.abs(ref - got).max()))
        import stumpy
        x = np.sin(np.arange(3000) * 0.1) + 0.1 * rng.normal(size=3000)
        x[1000:1050] = 0.0  # tramo constante
        for m in (4, 8, 16):
            a = stumpy.stump(x, m)[:, 0].astype(np.float64)
            b = mp_torch(x, m)
            print(f"MP GPU vs stumpy (m={m}), max |dif| =", float(np.nanmax(np.abs(a - b))))


if __name__ == "__main__":
    _selftest()
