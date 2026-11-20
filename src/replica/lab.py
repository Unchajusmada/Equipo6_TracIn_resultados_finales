"""Laboratorio de réplica: TracIn con todas las opciones que el artículo no fija.

El artículo (Verdone et al., 2026) no especifica: batch, épocas máximas,
paciencia, tamaño del LSTM, si hay conjunto de validación, normalización de la
serie, de qué parámetros se toman los gradientes en la práctica, ni si el F1
es binario o macro. Este módulo permite variar cada una de esas decisiones
sin tocar el código principal (``src/forecaster.py``), que queda como la
implementación de referencia ya validada.

Opciones de ``Cfg``
-------------------
lr, batch, max_epochs, patience, hidden : entrenamiento.
sched      : usar ReduceLROnPlateau(0.9, 3) (lo que describe el artículo).
val_frac   : fracción final de ventanas que se aparta para validación
             (scheduler y early stopping la monitorean); 0 = sin validación.
             Las ventanas de validación también reciben score.
norm       : 'z' (z-score por canal), 'minmax' (a [0, 1]) o 'none'.
grad       : 'last' = última capa lineal (forma cerrada, Ec. 3 del artículo);
             'full' = todos los parámetros (TracIn completo, Ec. 2).
lr_weight  : multiplicar cada checkpoint por su learning rate (TracIn
             original) o no.

La red es un LSTM escrito con operaciones elementales (misma ecuación que
``nn.LSTM``) para poder calcular gradientes por muestra con ``torch.func``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap


@dataclass(frozen=True)
class Cfg:
    lr: float = 1e-3
    batch: int = 32
    max_epochs: int = 100
    patience: int = 10
    hidden: int = 64
    sched: bool = True
    val_frac: float = 0.0
    norm: str = "z"
    grad: str = "last"
    lr_weight: bool = True

    def tag(self) -> str:
        d = asdict(self)
        base = asdict(Cfg())
        diff = {k: v for k, v in d.items() if v != base[k]}
        return "base" if not diff else ",".join(f"{k}={v}" for k, v in diff.items())


class LSTM(nn.Module):
    """LSTM de una capa + capa lineal final (como en ``forecaster.py``)."""

    def __init__(self, C: int, P: int, hidden: int):
        super().__init__()
        self.P, self.C, self.H = P, C, hidden
        self.ih = nn.Linear(C, 4 * hidden)
        self.hh = nn.Linear(hidden, 4 * hidden)
        self.head = nn.Linear(hidden, P * C)
        k = 1 / math.sqrt(hidden)
        for p in list(self.ih.parameters()) + list(self.hh.parameters()):
            nn.init.uniform_(p, -k, k)

    def forward(self, x):  # x: (B, L, C)
        B = x.shape[0]
        h = x.new_zeros(B, self.H)
        c = x.new_zeros(B, self.H)
        gx = self.ih(x)
        for t in range(x.shape[1]):
            i, f, g, o = (gx[:, t] + self.hh(h)).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        return self.head(h).view(B, self.P, self.C), h


def normalize(X: np.ndarray, how: str) -> np.ndarray:
    if how == "z":
        sd = X.std(0)
        return (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    if how == "minmax":
        lo, hi = X.min(0), X.max(0)
        return (X - lo) / np.where(hi > lo, hi - lo, 1)
    return X


def windows(Xt, idx, L, P):
    ar = torch.arange(L + P, device=Xt.device)
    w = Xt[idx[:, None] + ar]
    return w[:, :L], w[:, L:]


def _per_sample_sqgrad(model, Xt, M, L, P, bs=4096):
    """||grad_theta l(x)||^2 de cada ventana sobre TODOS los parámetros."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_fn(p, xb, yb):
        pred, _ = functional_call(model, p, (xb[None],))
        return ((pred[0] - yb) ** 2).mean()

    g_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    out = []
    for s in range(0, M, bs):
        idx = torch.arange(s, min(s + bs, M), device=Xt.device)
        xb, yb = windows(Xt, idx, L, P)
        g = g_fn(params, xb, yb)
        out.append(sum((v ** 2).flatten(1).sum(1) for v in g.values()))
    return torch.cat(out)


@torch.no_grad()
def _stats(model, Xt, M, L, P, bs=8192):
    loss, hsq = [], []
    for s in range(0, M, bs):
        idx = torch.arange(s, min(s + bs, M), device=Xt.device)
        xb, yb = windows(Xt, idx, L, P)
        pred, h = model(xb)
        loss.append(((pred - yb) ** 2).mean(dim=(1, 2)))
        hsq.append((h ** 2).sum(1))
    return torch.cat(loss), torch.cat(hsq)


def run(X: np.ndarray, L: int, P: int, seed: int, cfg: Cfg, device: str = "cpu",
        weights: np.ndarray | None = None, keep_g2: bool = False, callback=None) -> dict:
    """Entrena y devuelve scores por ventana: self-influence (según cfg),
    pérdida final y pérdida por época (para diagnósticos). Con keep_g2 devuelve
    también la contribución de cada época, η_k‖∇ℓ‖² (épocas × ventanas).
    ``callback(época, si_acumulada, pérdida_por_ventana)`` (opcional) se llama
    al final de cada época; sólo sirve para mostrar el avance (notebook)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xn = normalize(X.astype(np.float64), cfg.norm)
    n, C = Xn.shape
    M = n - L - P + 1
    Xt = torch.as_tensor(Xn, dtype=torch.float32, device=device)
    model = LSTM(C, P, cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=3) if cfg.sched else None
    n_val = int(M * cfg.val_frac)
    tr_idx = torch.arange(M - n_val, device=device)
    va_idx = torch.arange(M - n_val, M, device=device)
    w_all = torch.ones(M, device=device) if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    D = P * C
    si = torch.zeros(M, device=device)
    lrs, losses, g2s = [], [], []
    best, bad = math.inf, 0
    for _ in range(cfg.max_epochs):
        lr_now = opt.param_groups[0]["lr"]
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).to(device)]
        model.train()
        for b in perm.split(cfg.batch):
            xb, yb = windows(Xt, b, L, P)
            pred, _ = model(xb)
            l = ((pred - yb) ** 2).mean(dim=(1, 2))
            wb = w_all[b]
            loss = (l * wb).sum() / wb.sum().clamp_min(1e-12)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        l_all, h_all = _stats(model, Xt, M, L, P)
        if cfg.grad == "last":
            g2 = (4.0 / D) * l_all * (1 + h_all)
        else:
            g2 = _per_sample_sqgrad(model, Xt, M, L, P)
        si += (lr_now if cfg.lr_weight else 1.0) * g2
        if keep_g2:
            g2s.append(((lr_now if cfg.lr_weight else 1.0) * g2).cpu().numpy().astype(np.float32))
        lrs.append(lr_now)
        losses.append(l_all.cpu().numpy().astype(np.float32))
        if callback is not None:
            callback(len(lrs), si.cpu().numpy().astype(np.float64), losses[-1])
        ref = va_idx if n_val > 0 else tr_idx
        mon = float((l_all[ref] * w_all[ref]).sum() / w_all[ref].sum())
        if sched is not None:
            sched.step(mon)
        if mon < best * (1 - 1e-4):
            best, bad = mon, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    out = {"si": si.cpu().numpy().astype(np.float64), "loss_final": losses[-1].astype(np.float64),
           "loss": np.stack(losses), "epochs": len(lrs)}
    if keep_g2:
        out["g2"] = np.stack(g2s)
    return out


def _selftest():
    """El TracIn de la última capa (forma cerrada) coincide con el 'full'
    restringido a la cabeza; y el LSTM manual reproduce nn.LSTM."""
    torch.manual_seed(0)
    m = LSTM(3, 2, 8)
    ref = nn.LSTM(3, 8, batch_first=True)
    with torch.no_grad():
        ref.weight_ih_l0.copy_(m.ih.weight); ref.bias_ih_l0.copy_(m.ih.bias)
        ref.weight_hh_l0.copy_(m.hh.weight); ref.bias_hh_l0.copy_(m.hh.bias)
    x = torch.randn(4, 5, 3)
    _, h = m(x)
    out, _ = ref(x)
    print("LSTM manual vs nn.LSTM, max |dif| =", float((out[:, -1] - h).abs().max()))
    X = np.random.default_rng(0).normal(size=(60, 3))
    Xt = torch.as_tensor(X, dtype=torch.float32)
    full = _per_sample_sqgrad(m, Xt, 54, 5, 2)
    l, hs = _stats(m, Xt, 54, 5, 2)
    last = (4.0 / 6) * l * (1 + hs)
    print("||grad||^2 total >= cabeza:", bool((full >= last - 1e-6).all()))


if __name__ == "__main__":
    _selftest()
