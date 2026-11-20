"""Sensibilidad de la reproducción en GECCO a hiperparámetros que el
artículo no especifica (o especifica de forma ambigua).

El artículo reporta F1 = 0.604 (estático) y 0.761 (dinámico) para LSTM en
GECCO. Este script entrena el LSTM+TracIn con distintas variantes de
learning rate, batch, tamaño oculto y número de épocas, y reporta métricas
independientes del umbral, para saber si la brecha se explica por alguno
de esos detalles. Es un diagnóstico: los experimentos principales usan los
valores declarados por el artículo (lr = 1e-3) y fijados a priori.

Uso:  python repro_gecco.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from common import evaluate_scores, valid_mask
from data_real import load
from forecaster import train_trace

OUT = Path(__file__).resolve().parents[1] / "results" / "real" / "gecco_sensitivity.csv"

VARIANTS = {
    "base (lr 1e-3, b256, h64, 50 ép.)": dict(),
    "lr 1e-2": dict(lr=1e-2),
    "batch 64": dict(batch=64),
    "oculto 128": dict(model_kw={"hidden": 128}),
    "150 épocas": dict(max_epochs=150, patience=20),
    "lr 1e-2 + batch 64": dict(lr=1e-2, batch=64),
}


def main():
    X, y = load("gecco")
    v = valid_mask(len(y))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kw in VARIANTS.items():
        for L, P in [(4, 1), (8, 2), (16, 4)]:
            for seed in range(2):
                args = dict(batch=256, max_epochs=50, device="cuda") | kw
                t = time.time()
                tr = train_trace(X, L, P, seed, **args)
                for var, s in tr.timestep_scores().items():
                    if var not in ("tracin", "pd"):
                        continue
                    e = evaluate_scores(y[v], s[v], 0.87)
                    rows.append({"variante": name, "L": L, "P": P, "seed": seed, "score": var,
                                 "epocas": len(tr.lrs), **{k: e[k] for k in
                                 ("auprc", "auroc", "best_f1", "static_f1", "staticpct_f1", "dyn_f1")}})
                r = rows[-2]
                print(f"{name:32s} L{L}P{P} s{seed} ep={len(tr.lrs)} {time.time() - t:.0f}s "
                      f"AUPRC={r['auprc']:.3f} bestF1={r['best_f1']:.3f} est={r['static_f1']:.3f} "
                      f"din={r['dyn_f1']:.3f}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
