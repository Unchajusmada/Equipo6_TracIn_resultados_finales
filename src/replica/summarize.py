"""Resume el barrido de réplica al estilo de la Tabla 2 del artículo.

* Estático: L = 4, P = 1, media entre semillas y luego entre datasets.
* Dinámico: por dataset, la mejor (L, P) según la media entre semillas
  (como el artículo), luego media entre datasets.
* AUPRC (L4P1) y F1 oráculo (mejor configuración) miden el score sin umbral.
Artículo: LSTM+S = 0.662 (texto: ≈ 0.654), LSTM+D = 0.725.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_numbers import SYN_DATASETS, TABLE2

RES = Path(__file__).resolve().parents[2] / "results" / "replica"
PAPER_S, PAPER_D = np.mean(TABLE2["LSTM+S"]), np.mean(TABLE2["LSTM+D"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, sc), d in df.groupby(["variant", "score"]):
        st = d[(d.L == 4) & (d.P == 1)].groupby("ds").mean(numeric_only=True).mean()
        per = d.groupby(["ds", "L", "P"]).mean(numeric_only=True)
        dyn = per.groupby(level=0).max().mean()
        rows.append(dict(variant=v, score=sc, n_seeds=d.seed.nunique(), epochs=st.epochs,
                         auprc=st.auprc, oracle_best=dyn.oracle,
                         S_pct_bin=st.S_pct_bin, S_pct_mac=st.S_pct_mac,
                         S_frac_bin=st.S_frac_bin, S_frac_mac=st.S_frac_mac,
                         D_bin=dyn.D_bin, D_mac=dyn.D_mac))
    return pd.DataFrame(rows)


def per_dataset(df: pd.DataFrame, variant: str, score: str = "tracin") -> pd.DataFrame:
    d = df[(df.variant == variant) & (df.score == score)]
    st = d[(d.L == 4) & (d.P == 1)].groupby("ds").mean(numeric_only=True)
    dyn = d.groupby(["ds", "L", "P"]).mean(numeric_only=True).groupby(level=0).max()
    out = pd.DataFrame({"S_pct_mac": st.S_pct_mac, "paper_S": [TABLE2["LSTM+S"][SYN_DATASETS.index(i)] for i in st.index],
                        "D_mac": dyn.D_mac, "D_bin": dyn.D_bin,
                        "paper_D": [TABLE2["LSTM+D"][SYN_DATASETS.index(i)] for i in dyn.index]})
    return out.reindex(SYN_DATASETS)


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "screen.csv"
    df = pd.read_csv(RES / f, dtype={"ds": str})
    pd.set_option("display.width", 220)
    s = summarize(df)
    print(f"Artículo: LSTM+S {PAPER_S:.3f} · LSTM+D {PAPER_D:.3f}")
    print(s[s.score == "tracin"].sort_values("D_mac", ascending=False).round(3).to_string(index=False))
    print(s[s.score != "tracin"].sort_values(["score", "D_mac"], ascending=False).round(3).to_string(index=False))
