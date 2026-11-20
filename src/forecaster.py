"""Tarea auxiliar (forecasting) + atribución de datos con TracIn.

Reproducción del método de Verdone et al. (2026), Secc. 3-4:

* Se entrena un forecaster (LSTM, o Transformer) que recibe L pasos y
  predice los P siguientes de todos los canales, con MSE, Adam (lr 1e-3) y
  ReduceLROnPlateau (factor 0.9, paciencia 3) + early stopping.
* Al final de cada época (= checkpoint k) se calcula, para cada ventana de
  entrenamiento x, el término de TracIn restringido a la última capa lineal
  y = W h + b:

      ||grad_W l||^2 + ||grad_b l||^2 = ||g||^2 (||h||^2 + 1),
      g = dl/dy = (2/D)(y - t),  D = P*C  =>  ||g||^2 = (4/D) l(x).

  Por lo tanto la self-influence (Ec. 3) es exactamente

      s(x) = sum_k eta_k * (4/D) * l_k(x) * (1 + ||h_k(x)||^2),

  donde l_k es la pérdida de x en el checkpoint k. Es una pérdida integrada a
  lo largo del entrenamiento, ponderada por el learning rate y por la norma
  de la representación. Se verifica contra autograd en ``_selftest``.

Se guardan l_k(x) y ||h_k(x)||^2 por época, de modo que la self-influence y
todas las variantes de ablación se obtienen sin reentrenar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from common import project_sum, project_targets_mean


class LSTMForecaster(nn.Module):
    def __init__(self, C: int, L: int, P: int, hidden: int = 64):
        super().__init__()
        self.P, self.C = P, C
        self.lstm = nn.LSTM(C, hidden, batch_first=True)
        self.head = nn.Linear(hidden, P * C)  # capa en la que se toma TracIn

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.head(h).view(-1, self.P, self.C), h


class TransformerForecaster(nn.Module):
    def __init__(self, C: int, L: int, P: int, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.P, self.C = P, C
        self.inp = nn.Linear(C, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, P * C)

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos)
        h = z[:, -1]
        return self.head(h).view(-1, self.P, self.C), h


ARCHS = {"lstm": LSTMForecaster, "tf": TransformerForecaster}


@dataclass
class Trace:
    """Estadísticas por checkpoint (época) y por ventana."""
    L: int
    P: int
    C: int
    n: int                  # longitud de la serie
    lrs: np.ndarray         # (E,)  learning rate usado en cada época
    loss: np.ndarray        # (E, M) pérdida de cada ventana en cada checkpoint
    hsq: np.ndarray         # (E, M) ||h||^2 de cada ventana en cada checkpoint
    train_loss: np.ndarray  # (E,)  pérdida (ponderada) monitoreada

    @property
    def D(self) -> int:
        return self.P * self.C

    # ---- scores por ventana -------------------------------------------
    def self_influence(self, last_only: bool = False) -> np.ndarray:
        g = (4.0 / self.D) * self.loss * (1.0 + self.hsq)
        if last_only:
            return self.lrs[-1] * g[-1]
        return np.einsum("e,em->m", self.lrs, g)

    def loss_traj(self) -> np.ndarray:
        """Pérdida integrada (sin el factor de la representación)."""
        return np.einsum("e,em->m", self.lrs, self.loss)

    def loss_final(self) -> np.ndarray:
        return self.loss[-1]

    # ---- scores por instante -------------------------------------------
    def timestep_scores(self) -> dict[str, np.ndarray]:
        """Ablación 3 x 2: {pérdida final, pérdida integrada, self-influence}
        x {proyección del artículo (suma sobre las L+P posiciones de la
        ventana), proyección a los P pasos predichos (promedio)}.
        ``tracin`` es el método del artículo; ``pd`` es el detector clásico
        de prediction deviation con el mismo modelo."""
        span = self.L + self.P
        si = self.self_influence()
        lt = self.loss_traj()
        lf = self.loss_final()
        return {
            "tracin": project_sum(si, span, self.n),
            "tracin_last": project_sum(self.self_influence(last_only=True), span, self.n),
            "losstraj": project_sum(lt, span, self.n),
            "lossfinal": project_sum(lf, span, self.n),
            "pd": project_targets_mean(lf, self.L, self.P, self.n),
            "tracin_tgt": project_targets_mean(si, self.L, self.P, self.n),
            "losstraj_tgt": project_targets_mean(lt, self.L, self.P, self.n),
        }


def _windows(Xt: torch.Tensor, idx: torch.Tensor, L: int, P: int):
    ar = torch.arange(L + P, device=Xt.device)
    w = Xt[idx[:, None] + ar]  # (B, L+P, C)
    return w[:, :L], w[:, L:]


@torch.no_grad()
def _checkpoint_stats(model, Xt, M, L, P, bs):
    model.eval()
    losses, hsqs = [], []
    for s in range(0, M, bs):
        idx = torch.arange(s, min(s + bs, M), device=Xt.device)
        xb, yb = _windows(Xt, idx, L, P)
        pred, h = model(xb)
        losses.append(((pred - yb) ** 2).mean(dim=(1, 2)))
        hsqs.append((h ** 2).sum(1))
    model.train()
    return torch.cat(losses), torch.cat(hsqs)


def train_trace(X: np.ndarray, L: int, P: int, seed: int, arch: str = "lstm",
                weights: np.ndarray | None = None, lr: float = 1e-3, batch: int = 32,
                max_epochs: int = 100, patience: int = 10, device: str = "cpu",
                eval_batch: int = 8192, model_kw: dict | None = None) -> Trace:
    """Entrena el forecaster y devuelve la traza de checkpoints.

    ``weights`` (opcional) pondera la pérdida de cada ventana (propuesta 2).
    El scheduler y el early stopping monitorean la pérdida ponderada de todas
    las ventanas evaluada en cada checkpoint (no hay conjunto aparte: el
    escenario es transductivo y no supervisado, como en el artículo)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n, C = X.shape
    M = n - L - P + 1
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    w = torch.ones(M, device=device) if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    if w.sum() == 0:
        raise ValueError("El conjunto monitoreado tiene peso total cero")
    model = ARCHS[arch](C, L, P, **(model_kw or {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=3)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    lrs, L_hist, H_hist, mon = [], [], [], []
    best, bad = math.inf, 0
    for _ in range(max_epochs):
        lr_now = opt.param_groups[0]["lr"]
        perm = torch.randperm(M, generator=gen).to(device)
        for b in perm.split(batch):
            xb, yb = _windows(Xt, b, L, P)
            pred, _ = model(xb)
            l = ((pred - yb) ** 2).mean(dim=(1, 2))
            wb = w[b]
            loss = (l * wb).sum() / wb.sum().clamp_min(1e-12)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        l_all, h_all = _checkpoint_stats(model, Xt, M, L, P, eval_batch)
        monitored = float((l_all * w).sum() / w.sum())
        lrs.append(lr_now)
        L_hist.append(l_all.cpu().numpy())
        H_hist.append(h_all.cpu().numpy())
        mon.append(monitored)
        sched.step(monitored)
        if monitored < best * (1 - 1e-4):
            best, bad = monitored, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return Trace(L=L, P=P, C=C, n=n, lrs=np.array(lrs), loss=np.stack(L_hist).astype(np.float32),
                 hsq=np.stack(H_hist).astype(np.float32), train_loss=np.array(mon))


# --------------------------------------------------------------------------
# Propuesta 2: reponderación iterativa guiada por self-influence
# --------------------------------------------------------------------------
def robust_z(v: np.ndarray) -> np.ndarray:
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return (v - med) / max(mad, 1e-12)


def weights_from_influence(s: np.ndarray, mode: str, z0: float = 2.0, w_min: float = 0.05) -> np.ndarray:
    """Pesos por ventana a partir de su self-influence (escala log, z robusto).

    * ``soft``: w = 1 si z <= z0;  w = max(w_min, exp(-(z - z0))) si z > z0.
      Baja el peso gradualmente sin descartar ninguna ventana.
    * ``trim``: w = 0 si z > z0 (recorte duro al estilo ITLM, Shen &
      Sanghavi, 2019). Se incluye como contraste de la versión suave.
    z0 = 2 y w_min = 0.05 se fijan a priori (no se ajustan con etiquetas)."""
    z = robust_z(np.log(np.maximum(s, 1e-30)))
    if mode == "soft":
        return np.where(z <= z0, 1.0, np.maximum(w_min, np.exp(-(z - z0))))
    if mode == "trim":
        return np.where(z <= z0, 1.0, 0.0)
    raise ValueError(mode)


def reweighted_runs(X: np.ndarray, L: int, P: int, seed: int, base: Trace, mode: str,
                    rounds: int = 3, **kw) -> list[tuple[Trace, np.ndarray]]:
    """Ronda 0 = ``base`` (pesos uniformes, método original). En cada ronda
    r >= 1 se reentrena desde cero (misma semilla) con pesos calculados de la
    self-influence de la ronda anterior. El score de la ronda r es la
    self-influence *sin ponderar* de cada ventana, sum_k eta_k ||grad l(x)||^2:
    mide cuánto le cuesta al modelo ajustar x aunque x pese poco en el
    entrenamiento (si se multiplicara por el peso, las ventanas sospechosas
    quedarían artificialmente ocultas)."""
    out = []
    s = base.self_influence()
    for _ in range(rounds):
        w = weights_from_influence(s, mode)
        tr = train_trace(X, L, P, seed, weights=w, **kw)
        out.append((tr, w))
        s = tr.self_influence()
    return out


def _selftest():
    """Comprueba la forma cerrada de ||grad||^2 contra autograd."""
    torch.manual_seed(0)
    C, L, P = 3, 5, 2
    m = LSTMForecaster(C, L, P)
    x, t = torch.randn(1, L, C), torch.randn(1, P, C)
    pred, h = m(x)
    loss = ((pred - t) ** 2).mean()
    gW, gb = torch.autograd.grad(loss, [m.head.weight, m.head.bias])
    auto = float((gW ** 2).sum() + (gb ** 2).sum())
    closed = float(4.0 / (P * C) * loss * (1 + (h ** 2).sum()))
    assert abs(auto - closed) / auto < 1e-5, (auto, closed)
    print(f"selftest OK: autograd={auto:.6e} forma cerrada={closed:.6e}")


if __name__ == "__main__":
    _selftest()
