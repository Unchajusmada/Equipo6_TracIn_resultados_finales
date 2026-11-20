"""Diagnóstico: sensibilidad de LOF al número de vecinos k en los datasets reales.

Con el valor por defecto de PyOD (k = 20) LOF queda cerca del azar en
Credit Card y GECCO. La hipótesis es que las anomalías reales son colectivas
(eventos contiguos o fraudes parecidos entre sí) y forman grupos densos de más
de k puntos, así que localmente parecen normales. Si es así, la AUPRC debe
crecer con k.

Esto NO es un resultado comparativo justo: k se elige mirando las etiquetas.
Sólo se usa para explicar el fallo. Salida: results/real/lof_k.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines import lof_gpu
from data_real import NAMES, load

KS = (20, 50, 100, 200, 500)
OUT = Path(__file__).resolve().parent.parent / "results" / "real" / "lof_k.csv"


def main():
    rows = []
    for ds in NAMES:
        X, y = load(ds)[:2]
        for k in KS:
            s = lof_gpu(X, k, "cuda", chunk=512)
            rows.append(dict(dataset=ds, k=k, auprc=average_precision_score(y, s),
                             auroc=roc_auc_score(y, s), anomaly_rate=y.mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
