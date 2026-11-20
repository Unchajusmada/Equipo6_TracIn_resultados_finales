"""Figuras de la evaluación final (figures/fig_final_*.png) a partir de
results/final/*_eval.csv.

Uso: python figures.py
"""
from __future__ import annotations

from core import RES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats import FINAL, METHODS, method_frame

FIG = RES.parents[1] / "figures"
SHOW = [("Artículo · umbral dinámico (usa etiquetas)", "Artículo\n(dinámico,\ncon etiquetas)"),
        ("Artículo · umbral estático", "Artículo\n(estático)"),
        ("M1 · umbral IQR", "M1"), ("M2 · recorte z>2", "M2\nrecorte"), ("M2 · reponderación suave", "M2\nsuave"),
        ("M3 · proyección al objetivo", "M3"), ("M1+M3", "M1+M3"), ("M1+M2+M3 con recorte z>2", "M1+M2+M3\n(z>2)"),
        (FINAL, "FINAL"), ("Error del LSTM (PD) + IQR", "Error\nLSTM")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def bars():
    blocks = [b for b in ("synth", "contam", "real") if (RES / f"{b}_eval.csv").exists()]
    title = {"synth": "Sintéticos", "contam": "Contaminación", "real": "Reales"}
    fig, axes = plt.subplots(len(blocks), 2, figsize=(13, 3.2 * len(blocks)), squeeze=False)
    for i, b in enumerate(blocks):
        mf = method_frame(pd.read_csv(RES / f"{b}_eval.csv", dtype={"ds": str}), b)
        for j, (met, nm) in enumerate((("auprc", "AUPRC"), ("f1_bin", "F1 binario"))):
            ax = axes[i, j]
            names = [(m, s) for m, s in SHOW if m in set(mf.method)]
            g = [mf[mf.method == m][met] for m, _ in names]
            mu = [x.mean() for x in g]
            se = [1.96 * x.std() / np.sqrt(len(x)) for x in g]
            col = ["#c0392b" if m == FINAL else ("#7f7f7f" if m.startswith("Artículo") else "#8fb8d8") for m, _ in names]
            ax.bar(range(len(mu)), mu, yerr=se, color=col, capsize=2)
            ax.set_xticks(range(len(mu))); ax.set_xticklabels([s for _, s in names], fontsize=7)
            for k, v in enumerate(mu):
                ax.text(k, v + se[k], f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{title[b]} · {nm} (media ± IC 95 %)")
    plt.tight_layout()
    fig.savefig(FIG / "fig_final_resumen.png", dpi=150)


def contam():
    p = RES / "contam_eval.csv"
    if not p.exists():
        return
    ev = pd.read_csv(p, dtype={"ds": str})
    mf = method_frame(ev, "contam")
    rates = ev.groupby(["ds", "ratio"]).anom_rate.mean()
    show = [("Artículo · umbral estático", "Artículo", "#7f7f7f"), ("M3 · proyección al objetivo", "M3", "#1f77b4"),
            ("M2 · recorte z>2", "M2 recorte z>2", "#9467bd"), (FINAL, "FINAL", "#c0392b"),
            ("Error del LSTM (PD) + IQR", "Error LSTM", "#2ca02c"), ("LOF + IQR", "LOF", "#ff7f0e")]
    mixes = sorted(mf.ds.unique())
    nm = {"01234": "todas las anomalías (01234)", "01": "sólo puntuales (01)", "234": "sólo de patrón (234)"}
    fig, axes = plt.subplots(1, len(mixes), figsize=(14, 3.6), sharey=True)
    for ax, mix in zip(axes, mixes):
        for m, lab, c in show:
            g = mf[(mf.ds == mix) & (mf.method == m)].groupby("ratio").auroc
            x = [100 * rates[(mix, r)] for r in g.mean().index]
            ax.errorbar(x, g.mean().values, yerr=1.96 * g.sem().values, marker="o", ms=3, color=c,
                        lw=2.5 if m == FINAL else 1.2, label=lab, capsize=2)
        ax.set_title(nm.get(mix, mix)); ax.set_xlabel("% real de instantes anómalos")
    axes[0].set_ylabel("AUROC"); axes[-1].legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG / "fig_final_contaminacion.png", dpi=150)


if __name__ == "__main__":
    bars()
    contam()
    print("listo")
