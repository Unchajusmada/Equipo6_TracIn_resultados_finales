"""Figuras de la réplica (figures/fig_replica_*.png), con el mismo estilo que
figures.py, a partir de results/replica/replica_numbers.json y de las tablas
de results/replica/tables/.

Uso:  python rep_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from figures import BLUE, GRAY, INK, MUTED, ORANGE, AQUA, YEL, YEL_D, plt, save, title
from paper_numbers import SYN_DATASETS, TABLE2

RES = Path(__file__).resolve().parents[2] / "results" / "replica"


def fig_table2():
    """F1 dinámico por dataset: artículo vs réplica (macro y binario)."""
    t = pd.read_csv(RES / "tables" / "rep_syn_table2.csv", dtype={"Dataset": str})
    t = t[t.Dataset != "Promedio"].set_index("Dataset").reindex(SYN_DATASETS)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(SYN_DATASETS))
    paper = np.array(TABLE2["LSTM+D"])
    mac, binr = t["Nuestro D macro"].astype(float).values, t["Nuestro D binario"].astype(float).values
    for i in x:
        ax.plot([i, i], [binr[i], max(paper[i], mac[i])], color=GRAY, lw=1, zorder=1)
    ax.scatter(x, binr, s=46, color=GRAY, zorder=2, label="Nuestro · F1 binario")
    ax.scatter(x, mac, s=60, color=YEL_D, zorder=3, label="Nuestro · F1 macro")
    ax.scatter(x, paper, s=60, facecolor="white", edgecolor=INK, linewidth=1.8, zorder=4, label="Artículo (Tabla 2, LSTM+D)")
    for v, c, lab, dy in ((paper.mean(), INK, "artículo", 0.018), (mac.mean(), YEL_D, "macro", -0.018), (binr.mean(), MUTED, "binario", 0)):
        ax.plot([-0.5, len(x) - 0.5], [v, v], color=c, lw=1, ls="--", zorder=0)
        ax.text(len(x) - 0.3, v + dy, f"promedio {lab} {v:.3f}", color=c, va="center", fontsize=9)
    ax.set_xticks(x, [f"«{d}»" for d in SYN_DATASETS], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(x) + 2.2)
    ax.set_ylabel("F1 dinámico (mejor (L, P) por dataset)")
    ax.legend(loc="lower left", ncol=3, fontsize=9.5)
    title(ax, "Réplica de la Tabla 2", "TracIn-LSTM, datos estandarizados, 20 semillas. El promedio coincide con F1 macro; el patrón por dataset, no.")
    save(fig, "fig_replica_tabla2")


def fig_contamination():
    n = json.loads((RES / "replica_numbers.json").read_text(encoding="utf-8"))["cont"]
    xs = np.array(n["rate"]) * 100
    style = {"TracIn (artículo)": (INK, 2.4, "TracIn (artículo)"), "Recorte": (BLUE, 2, "TracIn + recorte"),
             "Recorte + proy.": (YEL_D, 2.4, "Recorte + proy. objetivo"), "Error LSTM (PD)": (ORANGE, 2, "Error LSTM"),
             "LOF": (AQUA, 2, "LOF")}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.5, 7.4), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    for name, ys in n["auroc"].items():
        if name not in style:
            ax.plot(xs, ys, color=GRAY, lw=1, marker="o", ms=2.5, zorder=1)
    ends = []
    for name, (c, lw, lab) in style.items():
        ys = n["auroc"][name]
        ax.plot(xs, ys, color=c, lw=lw, marker="o", ms=4.5, zorder=3)
        ends.append([ys[-1], lab, c])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.016)
    for y, lab, c in ends:
        ax.text(xs[-1] + 0.6, y, lab, color=c, va="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.6, 1)
    ax.set_ylabel("AUROC (azar = 0.5)")
    title(ax, "Contaminación controlada · réplica", "Serie «01234», 4000 pasos, L4P1, 10 semillas. AUROC: no depende del % de anomalías. Gris: IForest, OCSVM, AE, MP, suave, proy. obj., suave + proy.")
    t = n["tests_auroc"]["Recorte − TracIn"]
    d = np.array([v[0] for v in t]); p = np.array([v[1] for v in t])
    bx.bar(xs, d, width=1.6, color=[YEL if (pi < 0.05 and di > 0) else GRAY for di, pi in zip(d, p)], edgecolor=INK, linewidth=0.6)
    for xi, di, v in zip(xs, d, t):
        bx.text(xi, max(di, 0) + 0.001, f"{di:+.3f}\n{v[2]}/{v[3]}", ha="center", va="bottom", fontsize=8.5)
    bx.axhline(0, color=MUTED, lw=0.8)
    bx.set_ylim(min(0, d.min()) - 0.003, d.max() + 0.012)
    bx.set_ylabel("Δ AUROC\nrecorte − TracIn")
    bx.set_xlabel("% real de instantes anómalos")
    bx.set_xticks(xs, [f"{v:.0f} %" for v in xs])
    ax.set_xlim(0, xs[-1] + 7)
    save(fig, "fig_replica_contaminacion")


def fig_contam_steps():
    """Aporte de cada mejora contra la contaminación (AUROC)."""
    n = json.loads((RES / "replica_numbers.json").read_text(encoding="utf-8")).get("cont_fix")
    if not n:
        return
    xs = np.array(n["rate"]) * 100
    A = n["auroc"]
    style = [("TracIn (artículo)", INK, 2.4, "TracIn (artículo)"), ("Proy. objetivo", BLUE, 2.2, "+ proyección (mejora 3)"),
             ("Recorte z > 2 + proy. (2 rondas)", YEL, 2.2, "+ recorte z > 2 (mejora 2)"),
             ("Recorte adaptativo IQR + proy. (3 rondas)", YEL_D, 2.8, "+ recorte adaptativo IQR (mejoras 1 + 2)"),
             ("LOF", AQUA, 1.6, "LOF"), ("Error LSTM (PD)", ORANGE, 1.6, "Error LSTM")]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ends = []
    for k, c, lw, lab in style:
        ys = A[k]
        ax.plot(xs, ys, color=c, lw=lw, marker="o", ms=4.5, ls="--" if k in ("LOF", "Error LSTM (PD)") else "-")
        ends.append([ys[-1], f"{lab}  {ys[-1]:.3f} ({ys[-1] - ys[0]:+.3f})", c])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.013)
    for y, lab, c in ends:
        ax.text(xs[-1] + 0.7, y, lab, color=c if c != YEL else MUTED, va="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0.76, 0.97)
    ax.set_xlim(0, xs[-1] + 21)
    ax.set_xticks(xs, [f"{v:.0f} %" for v in xs])
    ax.set_xlabel("% real de instantes anómalos")
    ax.set_ylabel("AUROC (azar = 0.5)")
    title(ax, "Cada mejora frena la degradación", "Serie «01234», 4000 pasos, L4P1, 10 semillas. Entre paréntesis: caída de 1 % a 26 %. Punteadas: referencias.")
    save(fig, "fig_replica_contam_mejoras")


if __name__ == "__main__":
    fig_table2()
    fig_contamination()
    fig_contam_steps()
