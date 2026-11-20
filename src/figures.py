"""Figuras del reporte (PNG en figures/), a partir de los scores guardados,
results/*/eval_long.csv y results/deck_data.json.

Estilo: el de la presentación (blanco, negro, acento amarillo, Raleway /
Antonio). TracIn del artículo en negro, nuestra variante en amarillo oscuro,
el resto en gris o en los 3 primeros tonos categóricos validados
(azul #2a78d6, naranja #eb6834, aqua #1baf7a) con etiqueta directa.

Uso:  python figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from common import valid_mask

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
for f in (ROOT / "assets" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

INK, MUTED, GRID, SURF = "#111111", "#6b6a66", "#e6e5e1", "#ffffff"
YEL, YEL_D, GRAY = "#ffd84d", "#c98500", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "Raleway", "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def title(ax, text, sub=None):
    ax.set_title(text.upper(), loc="left", fontfamily="Antonio", fontweight="bold", fontsize=17, pad=16 if sub else 8, color=INK)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("figura:", name)


def norm01(s):
    s = np.asarray(s, float)
    return (s - s.min()) / (np.ptp(s) or 1.0)


def shade(ax, y, color=YEL, alpha=0.55):
    idx = np.flatnonzero(np.diff(np.r_[0, y, 0]))
    for a, b in zip(idx[::2], idx[1::2]):
        ax.axvspan(a - 0.5, b - 0.5, color=color, alpha=alpha, lw=0, zorder=0)


# --------------------------------------------------------------------------
def fig_mechanism(dataset="01234", seed=0):
    z = np.load(RES / "synthetic" / "scores" / f"{dataset}_s{seed}.npz")
    X, y = z["X"], z["y"]
    rows = [("TracIn (artículo)", z["lstm|L4P1|tracin"], INK),
            ("TracIn proy. objetivo (nuestro)", z["lstm|L4P1|tracin_tgt"], YEL_D),
            ("Error de predicción LSTM", z["lstm|L4P1|pd"], BLUE),
            ("Isolation Forest", z["iforest|w1"], GRAY),
            ("LOF", z["lof|w1"], GRAY),
            ("Matrix Profile", z["mp|m8"], GRAY)]
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.4] + [1] * len(rows)})
    v = valid_mask(len(y))
    t = np.arange(len(y))
    ax = axes[0]
    shade(ax, y)
    for c in range(X.shape[1]):
        ax.plot(t, X[:, c] - 4 * c, color=INK, lw=0.9)
    ax.set_yticks([-4 * c for c in range(5)], [f"canal {c} · tipo {c}" for c in range(5)], fontsize=8)
    ax.grid(False)
    title(ax, f"Dataset sintético «{dataset}»: la misma serie vista por cada detector",
          "Franjas amarillas = instantes anómalos (verdad de referencia). Scores normalizados a [0, 1].")
    for ax, (name, s, col) in zip(axes[1:], rows):
        shade(ax, y, alpha=0.35)
        s = np.where(v, s, np.nan)
        ax.fill_between(t, norm01(np.nan_to_num(s, nan=np.nanmin(s))), color=col, alpha=0.9, lw=0, step="mid")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.text(0.005, 0.92, name, transform=ax.transAxes, fontsize=9.5, va="top", fontweight="bold")
        ax.grid(False)
    axes[-1].set_xlabel("instante t")
    fig.tight_layout(h_pad=0.3)
    save(fig, "fig_mecanismo_serie")
    keys = {"TracIn (artículo)": "lstm|L4P1|tracin", "TracIn proy. objetivo": "lstm|L4P1|tracin_tgt",
            "Error LSTM (PD)": "lstm|L4P1|pd", "Isolation Forest": "iforest|w1", "LOF": "lof|w1",
            "Matrix Profile": "mp|m8"}
    return {"dataset": dataset, "seed": seed, "edge": int((~v).sum() // 2),
            "X": [np.round(X[:, c], 3).tolist() for c in range(X.shape[1])], "y": y.astype(int).tolist(),
            "scores": {k: np.round(norm01(np.where(v, z[key], np.min(z[key][v]))), 3).tolist() for k, key in keys.items()}}


def fig_trajectories():
    """Pérdida por época de ventanas cuyo paso predicho es anómalo vs normal,
    para cada tipo de anomalía (datasets 0-4, L=4, P=1)."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
    names = ["0 · puntual global", "1 · puntual contextual", "2 · shapelet", "3 · estacional", "4 · tendencia"]
    out = {}
    for ax, ds, nm in zip(axes, "01234", names):
        ratios = []
        for f in sorted((RES / "synthetic" / "scores").glob(f"{ds}_s*.npz")):
            z = np.load(f)
            loss = z["lstm|L4P1|trace_loss"]          # (E, M)
            y = z["y"]
            tgt = y[4:4 + loss.shape[1]].astype(bool)  # paso predicho de cada ventana
            if tgt.sum() == 0:
                continue
            r = loss[:, tgt].mean(1) / loss[:, ~tgt].mean(1)
            ratios.append(np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(r)), r))
        R = np.array(ratios)
        x = np.linspace(0, 100, 50)
        lo, med, hi = np.percentile(R, [25, 50, 75], axis=0)
        ax.fill_between(x, lo, hi, color=GRAY, alpha=0.45, lw=0)
        ax.plot(x, med, color=INK, lw=2)
        ax.set_yscale("log")
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_title(nm, fontsize=10.5, loc="left")
        ax.set_xlabel("% del entrenamiento")
        out[ds] = {"x": x.round(1).tolist(), "median": med.round(3).tolist(), "q25": lo.round(3).tolist(), "q75": hi.round(3).tolist()}
    axes[0].set_ylabel("pérdida anómala / pérdida normal")
    fig.suptitle("¿EL LSTM MEMORIZA LAS ANOMALÍAS? RAZÓN DE PÉRDIDAS DURANTE EL ENTRENAMIENTO",
                 x=0.01, y=0.995, ha="left", fontfamily="Antonio", fontweight="bold", fontsize=17)
    fig.text(0.01, 0.875, "Mediana (línea) y rango intercuartil (banda) entre semillas. Si la razón cayera hacia 1, "
             "el modelo estaría aprendiendo las anomalías como normales.", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "fig_mecanismo_trayectorias")
    return out


def _bars(ax, names, vals, errs=None, highlight=None, fmt="{:.2f}", xmax=None):
    highlight = highlight or {}
    y = np.arange(len(names))[::-1]
    cols = [highlight.get(n, GRAY) for n in names]
    ax.barh(y, vals, xerr=errs, color=cols, height=0.62, error_kw={"ecolor": MUTED, "lw": 1, "capsize": 0})
    ends = vals if errs is None else [v_ + (e if np.isfinite(e) else 0) for v_, e in zip(vals, errs)]
    span = xmax or (ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 1.0)
    for yi, v_, e_ in zip(y, vals, ends):
        ax.text(e_ + 0.012 * span, yi, fmt.format(v_), va="center", fontsize=9.5)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)


def fig_synthetic_comparison():
    df = pd.read_csv(RES / "synthetic" / "eval_long.csv", dtype={"dataset": str})
    from report import METHODS, default_rows
    D = default_rows(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    hl = {"TracIn (artículo)": INK, "TracIn proy. objetivo": YEL_D}
    for ax, (metric, lab) in zip(axes, [("auprc", "AUPRC"), ("dyn_f1", "F1 · umbral dinámico"),
                                       ("iqr_f1", "F1 · IQR (sin etiquetas, nuestro)")]):
        names, vals, errs = [], [], []
        for m, v, name in METHODS:
            d = D[(D.method == m) & (D.variant == v)]
            ps = d.groupby("seed")[metric].mean()
            names.append(name), vals.append(ps.mean()), errs.append(ps.std())
        _bars(ax, names, vals, errs, hl)
        ax.set_xlim(0, 1)
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("SINTÉTICOS · 15 DATASETS · CONFIGURACIÓN FIJA A PRIORI", x=0.01, y=0.99, ha="left", fontfamily="Antonio", fontweight="bold", fontsize=17)
    fig.text(0.01, 0.905, "Media de los 15 datasets; barra de error = desv. estándar entre semillas. "
             "IQR es el mejor de los 5 umbrales sin etiquetas evaluados (tabla syn_thresholds).", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig_sinteticos_comparacion")


def fig_heatmap():
    deck = json.loads((RES / "deck_data.json").read_text())
    A = deck["syn"]["auprc_by_dataset"]
    names = list(A["methods"])
    M = np.array([A["methods"][n] for n in names])
    fig, ax = plt.subplots(figsize=(13, 4.4))
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f5f4f0", "#1c5cab"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(A["datasets"])), A["datasets"])
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            best = M[i, j] == M[:, j].max()
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.55 else INK, fontweight="bold" if best else "normal")
    title(ax, "AUPRC por dataset sintético (negritas = mejor por columna)", "Tipos: 0 global · 1 contextual · 2 shapelet · 3 estacional · 4 tendencia")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    save(fig, "fig_sinteticos_heatmap")


def fig_ablation():
    df = pd.read_csv(RES / "synthetic" / "eval_long.csv", dtype={"dataset": str})
    L = df[(df.method == "lstm") & (df.config == "L4P1")]
    from report import ABLATION
    fig, ax = plt.subplots(figsize=(10, 4))
    names, vals, errs, hl = [], [], [], {}
    for var, score, proj in ABLATION:
        ps = L[L.variant == var].groupby("seed").auprc.mean()
        n = f"{score} · {proj}"
        names.append(n), vals.append(ps.mean()), errs.append(ps.std())
        if var == "tracin":
            hl[n] = INK
        if var == "tracin_tgt":
            hl[n] = YEL_D
    _bars(ax, names, vals, errs, hl)
    ax.set_xlim(0, 1)
    title(ax, "Ablación: ¿qué hace funcionar a la self-influence?", "AUPRC medio en los 15 sintéticos (L=4, P=1). Barras de error = desv. entre semillas.")
    save(fig, "fig_ablacion")


def fig_contamination():
    deck = json.loads((RES / "deck_data.json").read_text())
    C = deck["cont"]
    x = np.array(C["true_rate"]) * 100
    style = {"TracIn (artículo)": (INK, 2.4), "TracIn + reponderación soft": (BLUE, 2.2),
             "TracIn proy. objetivo": (YEL_D, 2.2), "LSTM error (PD)": (ORANGE, 1.8), "LOF": (AQUA, 1.8)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (metric, lab) in zip(axes, [("auprc", "AUPRC"), ("dyn_f1", "F1 · umbral dinámico")]):
        for name, ys in C[metric].items():
            col, lw = style.get(name, (GRAY, 1.0))
            ax.plot(x, ys, color=col, lw=lw, marker="o", ms=4, zorder=3 if name in style else 2)
            if name in style:
                ax.annotate(name, (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5, va="center", color=INK)
        if metric == "auprc":
            ax.plot(x, x / 100, color=INK, lw=1, ls=(0, (4, 4)))
            ax.annotate("azar", (x[-1], x[-1] / 100), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=MUTED, va="center")
        ax.set_xlabel("% real de instantes anómalos en la serie")
        ax.set_ylim(0, 1)
        ax.set_xlim(x[0] - 0.5, x[-1] + 9)
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("CONTAMINACIÓN CONTROLADA · 4000 PASOS · 5 TIPOS DE ANOMALÍA", x=0.01, ha="left", fontfamily="Antonio", fontweight="bold", fontsize=17)
    fig.text(0.01, 0.9, "Gris = otros métodos (OCSVM, IForest, AE, MP, recorte duro, proy. objetivo + soft). Media de 10 semillas.", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig_contaminacion")


def fig_real():
    deck = json.loads((RES / "deck_data.json").read_text())
    F = deck["real"]["fair_auprc"]
    names = list(F["methods"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    hl = {"TracIn (artículo)": INK, "TracIn proy. objetivo": YEL_D, "TracIn con Transformer": INK}
    for j, (ax, ds) in enumerate(zip(axes, F["datasets"])):
        vals = [F["methods"][n][j] for n in names]
        xmax = max(1e-3, max(vals)) * 1.3
        _bars(ax, names, vals, None, hl, xmax=xmax)
        ax.set_xlim(0, xmax)
        rate = deck["real"]["num"][deck["real"]["keys"][j]]["lstm|tracin"]["anomaly_rate"][0]
        ax.axvline(rate, color=INK, lw=1, ls=(0, (4, 4)))
        ax.text(rate, len(names) - 0.35, " azar", fontsize=8.5, color=MUTED, va="bottom")
        ax.set_title(ds, loc="left", fontsize=11, fontweight="bold", pad=14)
    fig.suptitle("DATOS REALES · AUPRC · CONFIGURACIÓN FIJA A PRIORI", x=0.01, ha="left", fontfamily="Antonio", fontweight="bold", fontsize=17)
    fig.text(0.01, 0.885, "Cada panel con su propia escala. Línea punteada = proporción de anomalías (AUPRC de un detector al azar). "
             "Media entre semillas (LSTM 5, Transformer 3, IForest/OCSVM/AE 5).", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save(fig, "fig_reales_auprc")


if __name__ == "__main__":
    import sys
    which = set(sys.argv[1:]) or {"syn", "real", "cont"}
    extra = {}
    if "syn" in which:
        extra["example"] = fig_mechanism()
        extra["trajectories"] = fig_trajectories()
        fig_synthetic_comparison()
        fig_heatmap()
        fig_ablation()
    if "cont" in which and (RES / "contamination" / "eval_long.csv").exists():
        fig_contamination()
    if "real" in which and (RES / "real" / "eval_long.csv").exists():
        fig_real()
    if extra:
        p = RES / "deck_data.json"
        deck = json.loads(p.read_text())
        deck.update(extra)
        p.write_text(json.dumps(deck, ensure_ascii=False, indent=1), encoding="utf-8")
