"""¿Qué lectura de la métrica explica la Tabla 2? Contraste de hipótesis
sobre los scores de la réplica (results/replica/synth_base, TracIn del
artículo, 15 datasets × 20 semillas × 9 configuraciones).

Lecturas del F1 comparadas, todas con el mismo protocolo (estático: L4P1,
percentil 85; dinámico: percentil elegido en la 1ª mitad, F1 en la 2ª, mejor
(L, P) por dataset):
  bin  F1 de la clase anómala (lo habitual)
  mac  F1 macro: promedio del F1 de la clase normal y de la anómala
  pa   F1 binario con point-adjust (un segmento anómalo cuenta como
       detectado entero si se marca cualquiera de sus instantes)
  wtd  F1 ponderado por soporte (f1_score(average='weighted'))
Para cada una: promedio, error absoluto medio y correlación por dataset
contra la Tabla 2.

Salida: results/replica/tables/rep_hipotesis_metrica.md (+ números en
results/replica/hypotheses.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import valid_mask
from paper_numbers import SYN_DATASETS, TABLE2
from protocol import PCTS

RES = Path(__file__).resolve().parents[2] / "results" / "replica"
READINGS = ["bin", "mac", "pa", "wtd"]


def segments(y):
    d = np.diff(np.r_[0, y.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def f1s(y, P, segs):
    """F1 de cada lectura para una matriz de predicciones P (umbrales × instantes)."""
    Y = y[None, :]
    tp = (P & Y).sum(1); fp = (P & ~Y).sum(1); fn = (~P & Y).sum(1); tn = (~P & ~Y).sum(1)
    fa = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    fnm = np.where(tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    pos = y.mean()
    Pa = P.copy()
    for a, b in segs:
        hit = P[:, a:b].any(1)
        Pa[hit, a:b] = True
    tpa = (Pa & Y).sum(1); fpa = (Pa & ~Y).sum(1); fna = (~Pa & Y).sum(1)
    fpa_ = np.where(tpa + fpa + fna > 0, 2 * tpa / np.maximum(2 * tpa + fpa + fna, 1), 0.0)
    return {"bin": fa, "mac": (fa + fnm) / 2, "pa": fpa_, "wtd": pos * fa + (1 - pos) * fnm}


def one(y, s):
    y = y.astype(bool)
    s = s / s.max()
    out = {}
    st = f1s(y, (s >= np.percentile(s, 85))[None, :], segments(y))
    for r in READINGS:
        out[f"S_{r}"] = float(st[r][0])
    k = len(s) // 2
    thr = np.percentile(s, PCTS)
    tr = f1s(y[:k], s[None, :k] >= thr[:, None], segments(y[:k]))
    for r in READINGS:
        t = thr[int(np.argmax(tr[r]))]
        out[f"D_{r}"] = float(f1s(y[k:], (s[k:] >= t)[None, :], segments(y[k:]))[r][0])
    return out


def job(f):
    z = np.load(f)
    ds, seed = f.stem.rsplit("_s", 1)
    y = z["y"].astype(bool)
    m = valid_mask(len(y))
    rows = []
    for k in z.files:
        if k.endswith("|tracin"):
            rows.append(dict(ds=ds, seed=int(seed), cfg=k.split("|")[0], **one(y[m], z[k][m].astype(np.float64))))
    return rows


def main():
    import multiprocessing as mp
    files = sorted(f for f in (RES / "synth_base").glob("*_s*.npz") if ".tmp" not in f.name)
    rows = []
    with mp.get_context("spawn").Pool(8) as pool:
        for r in pool.imap_unordered(job, files):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(RES / "hypotheses_synth.csv", index=False)
    g = df.groupby(["ds", "cfg"]).mean(numeric_only=True)
    paper_S = np.array(TABLE2["LSTM+S"]); paper_D = np.array(TABLE2["LSTM+D"])
    out, num = [], {}
    names = {"bin": "F1 binario (clase anómala)", "mac": "F1 macro (promedio de ambas clases)",
             "pa": "F1 binario con point-adjust", "wtd": "F1 ponderado por soporte"}
    for r in READINGS:
        S = g[f"S_{r}"].xs("L4P1", level="cfg").reindex(SYN_DATASETS).values
        D = g[f"D_{r}"].groupby(level=0).max().reindex(SYN_DATASETS).values
        num[r] = {"S": float(S.mean()), "D": float(D.mean()), "mae_S": float(np.abs(S - paper_S).mean()),
                  "mae_D": float(np.abs(D - paper_D).mean()), "corr_S": float(np.corrcoef(S, paper_S)[0, 1]),
                  "corr_D": float(np.corrcoef(D, paper_D)[0, 1])}
        out.append({"Lectura del F1": names[r], "Estático (promedio)": f"{S.mean():.3f}", "Error abs. medio (est.)": f"{num[r]['mae_S']:.3f}",
                    "Dinámico (promedio)": f"{D.mean():.3f}", "Error abs. medio (din.)": f"{num[r]['mae_D']:.3f}",
                    "Correlación por dataset (din.)": f"{num[r]['corr_D']:+.2f}"})
    out.insert(0, {"Lectura del F1": "Art. (Tabla 2)", "Estático (promedio)": f"{paper_S.mean():.3f}", "Error abs. medio (est.)": "—",
                   "Dinámico (promedio)": f"{paper_D.mean():.3f}", "Error abs. medio (din.)": "—", "Correlación por dataset (din.)": "—"})
    t = pd.DataFrame(out)
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(x[c]) for c in cols) + " |" for _, x in t.iterrows()]
    note = ("TracIn del artículo reproducido (datos estandarizados, 20 semillas). Estático: L4P1, percentil 85; dinámico: "
            "percentil elegido en la primera mitad con la misma lectura del F1 y medido en la segunda; mejor (L, P) por dataset. "
            "Error abs. medio: error absoluto medio contra la Tabla 2 sobre los 15 datasets.")
    (RES / "tables").mkdir(exist_ok=True)
    (RES / "tables" / "rep_hipotesis_metrica.md").write_text(
        "## ¿Qué lectura del F1 reproduce la Tabla 2?\n\n" + "\n".join(md) + f"\n\n{note}\n", encoding="utf-8")
    t.to_csv(RES / "tables" / "rep_hipotesis_metrica.csv", index=False)
    (RES / "hypotheses.json").write_text(json.dumps(num, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
