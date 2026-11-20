"""Tablas de resultados a partir de results/<exp>/eval_long.csv.

Genera results/tables/*.md y *.csv, y results/deck_data.json (los números
que usan las gráficas de la presentación).

Convenciones
------------
* Configuración por defecto (fijada a priori, sin etiquetas): LSTM/TF con
  L = 4, P = 1 (la del umbral estático del artículo); IForest, OCSVM, LOF y
  AE puntuales (w = 1, como en las filas principales de la Tabla 2); Matrix
  Profile con m = 8.
* "Estilo artículo": mejor configuración por dataset elegida con la métrica
  de evaluación (oráculo), como hace el artículo. Sólo sirve para comparar
  contra sus tablas, no como estimación honesta del desempeño.
* Media ± desviación estándar entre semillas. Para promedios sobre datasets,
  la desviación es la de los promedios por semilla.

Uso:  python report.py synthetic|real|contamination|all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from paper_numbers import SYN_DATASETS, TABLE2, TABLE4

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "tables"

DEFAULT_CFG = {"lstm": "L4P1", "tf": "L4P1", "iforest": "w1", "ocsvm": "w1", "lof": "w1", "ae": "w1", "mp": "m8"}
METHODS = [  # (method, variant, nombre)
    ("lstm", "tracin", "TracIn (artículo)"),
    ("lstm", "tracin_tgt", "TracIn proy. objetivo"),
    ("lstm", "pd", "LSTM error (PD)"),
    ("iforest", "score", "Isolation Forest"),
    ("ocsvm", "score", "One-Class SVM"),
    ("lof", "score", "LOF"),
    ("ae", "score", "Autoencoder"),
    ("mp", "score", "Matrix Profile"),
]
FAMILY = {"lstm": "Atribución / predicción", "iforest": "Mayoría", "ocsvm": "Mayoría", "lof": "Mayoría",
          "ae": "Mayoría", "mp": "Discordia", "tf": "Atribución / predicción"}
ABLATION = [
    ("lossfinal", "Pérdida final", "ventana completa"),
    ("losstraj", "Pérdida integrada", "ventana completa"),
    ("tracin", "Self-influence (TracIn)", "ventana completa"),
    ("pd", "Pérdida final", "pasos predichos"),
    ("losstraj_tgt", "Pérdida integrada", "pasos predichos"),
    ("tracin_tgt", "Self-influence (TracIn)", "pasos predichos"),
]
THRESHOLDS = [("static", "Estático τ=0.85·máx (art.)"), ("staticpct", "Estático percentil 85 (art., literal)"),
              ("dyn", "Dinámico (art., usa etiquetas)"), ("iqr", "IQR k=1.5"), ("mad", "MAD k=3"),
              ("log_iqr", "log-IQR k=1.5"), ("log_mad", "log-MAD k=3"), ("pot", "EVT-POT q=1e-3")]
POINT, PATTERN = ["0", "1"], ["2", "3", "4"]


def fmt(m, s=None, d=3):
    if pd.isna(m):
        return "—"
    return f"{m:.{d}f}" if s is None or pd.isna(s) else f"{m:.{d}f} ± {s:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write(name: str, title: str, df: pd.DataFrame, note: str = ""):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(f"## {title}\n\n{md_table(df)}\n\n{note}\n", encoding="utf-8")
    print(f"\n## {title}\n{md_table(df)}\n{note}")


def default_rows(df):
    mask = np.zeros(len(df), dtype=bool)
    for m, cfg in DEFAULT_CFG.items():
        mask |= (df.method == m) & (df.config == cfg)
    return df[mask]


def avg_over(df, metric, by, datasets=None):
    """Media entre datasets y desviación entre semillas de esa media."""
    if datasets is not None:
        df = df[df.dataset.isin(datasets)]
    per_seed = df.groupby(by + ["seed"])[metric].mean()
    g = per_seed.groupby(level=list(range(len(by))))
    return g.mean(), g.std()


def paired_test(a: pd.Series, b: pd.Series):
    """Wilcoxon pareado (dos colas) sobre pares (dataset, semilla)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = j.a - j.b
    if (d != 0).sum() < 5:
        return float(d.mean()), float("nan"), len(d)
    return float(d.mean()), float(wilcoxon(j.a, j.b).pvalue), len(d)


NUM_METRICS = ["auprc", "auroc", "best_f1", "dyn_f1", "static_f1", "staticpct_f1", "iqr_f1", "mad_f1",
               "log_iqr_f1", "log_mad_f1", "pot_f1", "iqr_rate", "mad_rate", "static_rate", "anomaly_rate"]


def numeric_summary(df: pd.DataFrame, datasets=None) -> dict:
    """{'method|variant': {metrica: [media, desv]}} con la desv. entre semillas
    del promedio sobre datasets (configuración por defecto)."""
    out = {}
    for (m, v), d in df.groupby(["method", "variant"]):
        if datasets is not None:
            d = d[d.dataset.isin(datasets)]
        per_seed = d.groupby("seed")[[c for c in NUM_METRICS if c in d.columns]].mean()
        out[f"{m}|{v}"] = {c: [round(float(per_seed[c].mean()), 4),
                               round(float(per_seed[c].std()), 4) if len(per_seed) > 1 else None]
                           for c in per_seed.columns}
    return out


def series_of(df, method, variant, cfg, metric):
    d = df[(df.method == method) & (df.variant == variant) & (df.config == cfg)]
    return d.set_index(["dataset", "seed"])[metric]


# ==========================================================================
# Sintéticos
# ==========================================================================
def synthetic(deck: dict):
    df = pd.read_csv(RES / "synthetic" / "eval_long.csv", dtype={"dataset": str})
    df["dataset"] = df.dataset.str.replace(r"\.0$", "", regex=True)
    n_seeds = df.seed.nunique()
    D = default_rows(df)
    deck["syn"] = {"n_seeds": int(n_seeds), "num": numeric_summary(D),
                   "num_point": numeric_summary(D, POINT), "num_pattern": numeric_summary(D, PATTERN)}

    # ---- 1. Comparación justa con la configuración por defecto -------------
    rows, per_ds = [], {}
    for m, v, name in METHODS:
        d = D[(D.method == m) & (D.variant == v)]
        r = {"Método": name, "Familia": FAMILY[m]}
        for metric, lab in [("auprc", "AUPRC"), ("auroc", "AUROC"), ("best_f1", "F1 oráculo"),
                            ("dyn_f1", "F1 dinámico (art., usa etiquetas)"), ("iqr_f1", "F1 IQR (sin etiquetas)")]:
            mu, sd = avg_over(d, metric, ["method"])
            r[lab] = fmt(mu.iloc[0], sd.iloc[0])
        rows.append(r)
        per_ds[name] = d.groupby("dataset").auprc.mean().reindex(SYN_DATASETS).round(4).tolist()
    # victorias por dataset (AUPRC media)
    best = pd.DataFrame(per_ds, index=SYN_DATASETS).idxmax(axis=1).value_counts()
    for r in rows:
        r["Gana (AUPRC)"] = int(best.get(r["Método"], 0))
    write("syn_fair", f"Sintéticos — comparación con configuración fija a priori (promedio de 15 datasets, {n_seeds} semillas)",
          pd.DataFrame(rows),
          "Todos los métodos se evalúan sobre los mismos instantes y con los mismos protocolos de umbral. "
          "‘F1 dinámico’ = umbral F1-óptimo elegido en la otra mitad de la serie (validación cruzada de 2 bloques). "
          "‘Gana’ = número de datasets en que el método tiene la mayor AUPRC media.")
    deck["syn"]["auprc_by_dataset"] = {"datasets": SYN_DATASETS, "methods": per_ds}
    deck["syn"]["fair"] = rows

    # ---- 2. Reproducción estilo artículo -----------------------------------
    rep = []
    tr = df[(df.method == "lstm") & (df.variant == "tracin")]
    static = tr[tr.config == "L4P1"].groupby("dataset").static_f1.agg(["mean", "std"])
    staticpct = tr[tr.config == "L4P1"].groupby("dataset").staticpct_f1.mean()
    dyn_cfg = tr.groupby(["dataset", "config"]).dyn_f1.agg(["mean", "std"]).reset_index()
    dyn_best = dyn_cfg.loc[dyn_cfg.groupby("dataset")["mean"].idxmax()].set_index("dataset")
    orc_best = tr.groupby(["dataset", "config"]).best_f1.mean().groupby("dataset").max()
    # Baselines como en Lai et al.: mejor (config, contaminación) por dataset.
    cont_cols = [c for c in df.columns if c.startswith("cont") and c.endswith("_f1")]
    base = df[df.method.isin(["iforest", "ocsvm", "lof", "ae", "mp"])]
    bmax = base.groupby(["dataset", "method", "config"])[cont_cols].mean().max(axis=1).groupby(["dataset", "method"]).max()
    for i, ds in enumerate(SYN_DATASETS):
        paper_base = {k: v[i] for k, v in TABLE2.items() if not k.startswith("LSTM+")}
        pb = max(paper_base, key=paper_base.get)
        ob = bmax.loc[ds].idxmax()
        rep.append({"Dataset": ds,
                    "Art. LSTM+S": f"{TABLE2['LSTM+S'][i]:.3f}",
                    "Nuestro TracIn+S (τ·máx)": fmt(static.loc[ds, "mean"], static.loc[ds, "std"]),
                    "Nuestro TracIn+S (pct)": fmt(staticpct.loc[ds]),
                    "Art. LSTM+D": f"{TABLE2['LSTM+D'][i]:.3f}",
                    "Nuestro TracIn+D": fmt(dyn_best.loc[ds, "mean"], dyn_best.loc[ds, "std"]),
                    "Nuestro TracIn F1 oráculo": fmt(orc_best.loc[ds]),
                    "Art. mejor baseline": f"{paper_base[pb]:.3f} ({pb})",
                    "Nuestro mejor baseline": f"{bmax.loc[ds].max():.3f} ({ob})"})
    mean_row = {"Dataset": "Promedio",
                "Art. LSTM+S": f"{np.mean(TABLE2['LSTM+S']):.3f}",
                "Nuestro TracIn+S (τ·máx)": fmt(static["mean"].mean()),
                "Nuestro TracIn+S (pct)": fmt(staticpct.mean()),
                "Art. LSTM+D": f"{np.mean(TABLE2['LSTM+D']):.3f}",
                "Nuestro TracIn+D": fmt(dyn_best["mean"].mean()),
                "Nuestro TracIn F1 oráculo": fmt(orc_best.mean()),
                "Art. mejor baseline": f"{np.mean([max(v[i] for k, v in TABLE2.items() if not k.startswith('LSTM+')) for i in range(15)]):.3f}",
                "Nuestro mejor baseline": f"{bmax.groupby('dataset').max().mean():.3f}"}
    rep.append(mean_row)
    write("syn_reproduction", "Sintéticos — reproducción de la Tabla 2 (F1, estilo artículo)", pd.DataFrame(rep),
          "TracIn+S: L=4, P=1, τ=0.85 (lectura principal: fracción del máximo; ‘pct’ = lectura literal como percentil). "
          "TracIn+D: mejor configuración (L, P) por dataset según F1 dinámico medio (oráculo de configuración, como el artículo). "
          "Baselines: mejor (ventana, contaminación) por dataset, como en Lai et al. (2021).")
    deck["syn"]["reproduction"] = {
        "paper_S": float(np.mean(TABLE2["LSTM+S"])), "paper_D": float(np.mean(TABLE2["LSTM+D"])),
        "ours_S": float(static["mean"].mean()), "ours_S_pct": float(staticpct.mean()),
        "ours_D": float(dyn_best["mean"].mean()), "ours_oracle": float(orc_best.mean()),
        "paper_best_base": float(mean_row["Art. mejor baseline"]), "ours_best_base": float(mean_row["Nuestro mejor baseline"]),
        "per_dataset": {"paper_D": TABLE2["LSTM+D"], "ours_D": dyn_best["mean"].reindex(SYN_DATASETS).round(4).tolist(),
                        "ours_oracle": orc_best.reindex(SYN_DATASETS).round(4).tolist()},
    }

    # ---- 3. Ablación: qué parte de la self-influence aporta -----------------
    ab = []
    L = df[(df.method == "lstm") & (df.config == "L4P1")]
    Lall = df[df.method == "lstm"]
    for var, score, proj in ABLATION:
        d = L[L.variant == var]
        r = {"Score": score, "Proyección": proj}
        for lab, dsets in [("Puntuales (0,1)", POINT), ("De patrón (2,3,4)", PATTERN), ("Los 15", None)]:
            mu, sd = avg_over(d, "auprc", ["variant"], dsets)
            r[f"AUPRC {lab}"] = fmt(mu.iloc[0], sd.iloc[0])
        mu, _ = avg_over(Lall[Lall.variant == var], "auprc", ["variant"])
        r["AUPRC 15 (9 configs)"] = fmt(mu.iloc[0])
        ab.append(r)
    t1 = paired_test(series_of(df, "lstm", "tracin", "L4P1", "auprc"), series_of(df, "lstm", "lossfinal", "L4P1", "auprc"))
    t2 = paired_test(series_of(df, "lstm", "tracin_tgt", "L4P1", "auprc"), series_of(df, "lstm", "tracin", "L4P1", "auprc"))
    t3 = paired_test(series_of(df, "lstm", "tracin_tgt", "L4P1", "auprc"), series_of(df, "lstm", "pd", "L4P1", "auprc"))
    note = (f"L=4, P=1. Wilcoxon pareado sobre (dataset, semilla): TracIn − pérdida final (misma proyección) = {t1[0]:+.3f} (p = {t1[1]:.2g}); "
            f"TracIn proy. objetivo − TracIn = {t2[0]:+.3f} (p = {t2[1]:.2g}); TracIn proy. objetivo − PD = {t3[0]:+.3f} (p = {t3[1]:.2g}).")
    write("syn_ablation", "Sintéticos — ablación del score de TracIn (AUPRC)", pd.DataFrame(ab), note)
    deck["syn"]["ablation"] = {"rows": ab, "tests": {"tracin_vs_lossfinal": t1, "tgt_vs_tracin": t2, "tgt_vs_pd": t3}}

    # ---- 4. Umbrales (propuesta 1) ------------------------------------------
    th = []
    for m, v, name in [("lstm", "tracin", "TracIn (artículo)"), ("lstm", "tracin_tgt", "TracIn proy. objetivo"),
                       ("lstm", "pd", "LSTM error (PD)"), ("lof", "score", "LOF"), ("iforest", "score", "Isolation Forest")]:
        d = D[(D.method == m) & (D.variant == v)]
        r = {"Score": name}
        for key, lab in THRESHOLDS:
            mu, sd = avg_over(d, f"{key}_f1", ["method"])
            r[lab] = fmt(mu.iloc[0], sd.iloc[0])
        r["F1 oráculo"] = fmt(d.best_f1.mean())
        th.append(r)
    rates = D[(D.method == "lstm") & (D.variant == "tracin_tgt")]
    rate_note = "Proporción marcada / proporción real (TracIn proy. objetivo): " + ", ".join(
        f"{lab.split(' ')[0]} {(rates[f'{k}_rate'] / rates.anomaly_rate).median():.2f}" for k, lab in THRESHOLDS
        if f"{k}_rate" in rates.columns)
    write("syn_thresholds", "Sintéticos — F1 según la regla de umbral (configuración por defecto)", pd.DataFrame(th),
          "Los umbrales sin etiquetas usan hiperparámetros fijados a priori (k = 1.5, k = 3, q = 1e-3). " + rate_note)
    deck["syn"]["thresholds"] = th

    # ---- 5. Reponderación (propuesta 2) -------------------------------------
    rw = []
    for base_v, suf, lab in [("tracin", "", "TracIn (artículo)"), ("tracin_tgt", "_tgt", "TracIn proy. objetivo")]:
        for v, name in [(base_v, "Original (ronda 0)")] + [(f"rw_{mo}_r{r}{suf}", f"{mo} ronda {r}") for mo in ("soft", "trim") for r in (1, 2, 3)]:
            d = L[L.variant == v]
            r_ = {"Score": lab, "Variante": name}
            for metric, ml in [("auprc", "AUPRC"), ("dyn_f1", "F1 dinámico"), ("log_mad_f1", "F1 log-MAD")]:
                mu, sd = avg_over(d, metric, ["variant"])
                r_[ml] = fmt(mu.iloc[0], sd.iloc[0])
            if v != base_v:
                dlt, p, _ = paired_test(series_of(df, "lstm", v, "L4P1", "auprc"), series_of(df, "lstm", base_v, "L4P1", "auprc"))
                r_["Δ AUPRC vs original (p)"] = f"{dlt:+.3f} ({p:.2g})"
            else:
                r_["Δ AUPRC vs original (p)"] = "—"
            rw.append(r_)
    write("syn_reweighting", "Sintéticos — reponderación iterativa (L=4, P=1)", pd.DataFrame(rw),
          "soft: w = max(0.05, exp(−(z−2))) para z > 2; trim: w = 0 para z > 2 (z robusto del log de la self-influence). "
          "p = Wilcoxon pareado sobre (dataset, semilla).")
    deck["syn"]["reweighting"] = rw


# ==========================================================================
# Reales
# ==========================================================================
REAL = ["creditcard", "cicids", "gecco", "swan_sf"]
PRETTY = {"creditcard": "Credit Card", "cicids": "CICIDS", "gecco": "GECCO", "swan_sf": "SWAN-SF"}


def real(deck: dict):
    df = pd.read_csv(RES / "real" / "eval_long.csv")
    D = default_rows(df)
    deck["real"] = {"num": {ds: numeric_summary(D[D.dataset == ds]) for ds in REAL},
                    "pretty": [PRETTY[d] for d in REAL], "keys": REAL}
    rwseeds_ = sorted(df[df.variant.str.startswith("rw_")].seed.unique())
    deck["real"]["num_rwseeds"] = {ds: numeric_summary(D[(D.dataset == ds) & D.seed.isin(rwseeds_)]) for ds in REAL}
    methods = METHODS + [("tf", "tracin", "TracIn con Transformer")]

    # ---- 1. Comparación justa por dataset ----------------------------------
    for metric, lab in [("auprc", "AUPRC"), ("dyn_f1", "F1 dinámico"), ("log_mad_f1", "F1 log-MAD"), ("best_f1", "F1 oráculo")]:
        rows = []
        for m, v, name in methods:
            d = D[(D.method == m) & (D.variant == v)]
            if d.empty:
                continue
            r = {"Método": name}
            for ds in REAL:
                x = d[d.dataset == ds][metric]
                r[PRETTY[ds]] = fmt(x.mean(), x.std() if len(x) > 1 else None)
            rows.append(r)
        write(f"real_fair_{metric}", f"Reales — {lab}, configuración fija a priori", pd.DataFrame(rows),
              "Media ± desv. estándar entre semillas (LSTM 5, Transformer 3, IForest/OCSVM/AE 5; LOF y MP son deterministas).")
        deck["real"][f"fair_{metric}"] = {
            "datasets": [PRETTY[d] for d in REAL],
            "methods": {name: [float(D[(D.method == m) & (D.variant == v) & (D.dataset == ds)][metric].mean())
                               for ds in REAL] for m, v, name in methods
                        if not D[(D.method == m) & (D.variant == v)].empty}}

    # ---- 2. Reproducción estilo artículo (Tabla 4) -------------------------
    rep = []
    tr = df[(df.method == "lstm") & (df.variant == "tracin")]
    cont_cols = [c for c in df.columns if c.startswith("cont") and c.endswith("_f1")]
    base = df[df.method.isin(["iforest", "ocsvm", "lof", "ae", "mp"])]
    for ds in REAL:
        t = tr[tr.dataset == ds]
        s = t[t.config == "L4P1"]
        dyn = t.groupby("config").dyn_f1.mean()
        orc = t.groupby("config").best_f1.mean()
        b = base[base.dataset == ds].groupby(["method", "config"])[cont_cols].mean().max(axis=1).groupby("method").max()
        pb = {k: v[2] for k, v in TABLE4[ds].items() if "+" not in k}
        deck.setdefault("real_rep_num", {})[ds] = {
            "paper_S": TABLE4[ds]["LSTM+S"][2], "paper_D": TABLE4[ds]["LSTM+D"][2],
            "paper_best_base": max(pb.values()), "paper_best_base_name": max(pb, key=pb.get),
            "ours_S": float(s.static_f1.mean()), "ours_S_pct": float(s.staticpct_f1.mean()),
            "ours_D": float(dyn.max()), "ours_D_cfg": dyn.idxmax(), "ours_oracle": float(orc.max()),
            "ours_best_base": float(b.max()) if len(b) else None, "ours_best_base_name": b.idxmax() if len(b) else None}
        rep.append({"Dataset": PRETTY[ds],
                    "Art. LSTM+S": f"{TABLE4[ds]['LSTM+S'][2]:.3f}",
                    "Nuestro TracIn+S (τ·máx)": fmt(s.static_f1.mean(), s.static_f1.std()),
                    "Nuestro TracIn+S (pct)": fmt(s.staticpct_f1.mean()),
                    "Art. LSTM+D": f"{TABLE4[ds]['LSTM+D'][2]:.3f}",
                    "Nuestro TracIn+D": f"{dyn.max():.3f} ({dyn.idxmax()})",
                    "Nuestro TracIn F1 oráculo": f"{orc.max():.3f} ({orc.idxmax()})",
                    "Art. mejor baseline": f"{max(pb.values()):.3f} ({max(pb, key=pb.get)})",
                    "Nuestro mejor baseline": f"{b.max():.3f} ({b.idxmax()})" if len(b) else "—"})
    g = base[(base.dataset == "gecco") & (base.method == "iforest") & (base.config == "w1")]
    if len(g):
        deck["real_if_gecco"] = float(g[cont_cols].mean().max())
    write("real_reproduction", "Reales — reproducción de la Tabla 4 (F1, estilo artículo)", pd.DataFrame(rep),
          "Mismas convenciones que en sintéticos; τ = 0.87.")
    deck["real"]["reproduction"] = rep

    # ---- 3. Ablación, umbrales y reponderación por dataset -----------------
    ab = []
    L = df[(df.method == "lstm") & (df.config == "L4P1")]
    for var, score, proj in ABLATION:
        r = {"Score": score, "Proyección": proj}
        for ds in REAL:
            x = L[(L.variant == var) & (L.dataset == ds)].auprc
            r[PRETTY[ds]] = fmt(x.mean(), x.std())
        ab.append(r)
    write("real_ablation", "Reales — ablación del score de TracIn (AUPRC, L=4, P=1)", pd.DataFrame(ab))
    deck["real"]["ablation"] = ab

    th = []
    for m, v, name in [("lstm", "tracin", "TracIn (artículo)"), ("lstm", "tracin_tgt", "TracIn proy. objetivo"),
                       ("iforest", "score", "Isolation Forest")]:
        d = D[(D.method == m) & (D.variant == v)]
        for ds in REAL:
            x = d[d.dataset == ds]
            r = {"Score": name, "Dataset": PRETTY[ds]}
            for key, lab in THRESHOLDS:
                lab = lab.replace("0.85", "0.87").replace("percentil 85", "percentil 87")
                r[lab] = fmt(x[f"{key}_f1"].mean())
            r["F1 oráculo"] = fmt(x.best_f1.mean())
            th.append(r)
    write("real_thresholds", "Reales — F1 según la regla de umbral (configuración por defecto)", pd.DataFrame(th))
    deck["real"]["thresholds"] = th

    rw = []
    rwseeds = sorted(df[df.variant.str.startswith("rw_")].seed.unique())
    for base_v, suf, lab in [("tracin", "", "TracIn (artículo)"), ("tracin_tgt", "_tgt", "TracIn proy. objetivo")]:
        for v, name in [(base_v, "Original")] + [(f"rw_{mo}_r{r}{suf}", f"{mo} r{r}") for mo in ("soft", "trim") for r in (1, 2, 3)]:
            r_ = {"Score": lab, "Variante": name}
            for ds in REAL:
                x = L[(L.variant == v) & (L.dataset == ds) & (L.seed.isin(rwseeds))].auprc
                r_[PRETTY[ds]] = fmt(x.mean(), x.std() if len(x) > 1 else None)
            rw.append(r_)
    write("real_reweighting", "Reales — reponderación iterativa (AUPRC, L=4, P=1, semillas de la fase B)", pd.DataFrame(rw))
    deck["real"]["reweighting"] = rw
    rwd = {}
    for ds in REAL:
        for v in ("rw_soft_r3", "rw_trim_r3", "rw_soft_r3_tgt", "rw_trim_r3_tgt"):
            base_v = "tracin_tgt" if v.endswith("_tgt") else "tracin"
            a = L[(L.variant == v) & (L.dataset == ds)].set_index("seed").auprc
            b = L[(L.variant == base_v) & (L.dataset == ds)].set_index("seed").auprc
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if len(j):
                rwd.setdefault(ds, {})[v] = [float((j.a - j.b).mean()), int((j.a > j.b).sum()), len(j)]
    deck["real"]["rw_delta"] = rwd

    # ---- 4. ¿Dónde aporta la trayectoria? memorización por dataset ---------
    # Razón = pérdida media de ventanas cuyo paso predicho es anómalo / normales,
    # por época (traza guardada del LSTM L4P1, semilla 0). Δ por semilla (5).
    from data_real import load
    mem, rows = {}, []
    for ds in REAL:
        f = RES / "real" / "scores" / ds / "lstm_4_1_0.npz"
        if not f.exists():
            continue
        loss = np.load(f)["trace_loss"].astype(np.float64)
        y = load(ds)[1]
        tgt = y[4:4 + loss.shape[1]].astype(bool)
        ratio = loss[:, tgt].mean(1) / loss[:, ~tgt].mean(1)
        d = {}
        for a_v, b_v in [("tracin", "lossfinal"), ("tracin_tgt", "pd")]:
            a = L[(L.variant == a_v) & (L.dataset == ds)].set_index("seed").auprc
            b = L[(L.variant == b_v) & (L.dataset == ds)].set_index("seed").auprc
            d[f"{a_v}-{b_v}"] = [float((a - b).mean()), int((a > b).sum()), int(len(a))]
        mem[ds] = {"rate": float(y.mean()), "max": float(ratio.max()), "final": float(ratio[-1]),
                   "final_over_max": float(ratio[-1] / ratio.max()), **d}
        rows.append({"Dataset": PRETTY[ds], "% anómalo": f"{y.mean() * 100:.2f} %",
                     "Razón máx.": f"{ratio.max():.1f}×", "Razón final": f"{ratio[-1]:.1f}×",
                     "Final / máx.": f"{ratio[-1] / ratio.max():.2f}",
                     "Δ AUPRC TracIn − pérdida final (ventana)": f"{d['tracin-lossfinal'][0]:+.3f} ({d['tracin-lossfinal'][1]}/{d['tracin-lossfinal'][2]})",
                     "Δ AUPRC proy. obj. − PD": f"{d['tracin_tgt-pd'][0]:+.3f} ({d['tracin_tgt-pd'][1]}/{d['tracin_tgt-pd'][2]})"})
    if rows:
        write("real_memorizacion", "Reales — memorización del LSTM y dónde aporta la trayectoria (L=4, P=1)", pd.DataFrame(rows),
              "Razón = pérdida media de ventanas cuyo paso predicho es anómalo / de las normales (semilla 0). "
              "Final/máx < 1: el modelo aprendió (memorizó) parte de las anomalías. Razón < 1: las anomalías se predicen "
              "mejor que lo normal. Entre paréntesis, semillas en que gana el primer score.")
        deck["real"]["memo"] = mem

    # ---- 5. Diagnóstico LOF vs k (k elegido con etiquetas: no es comparación justa)
    f = RES / "real" / "lof_k.csv"
    if f.exists():
        lk = pd.read_csv(f)
        deck["real"]["lof_k"] = {ds: dict(zip(g.k.astype(str), g.auprc.round(4))) for ds, g in lk.groupby("dataset")}
        t = lk.pivot(index="dataset", columns="k", values="auprc").reindex(REAL)
        t.index = [PRETTY[d] for d in t.index]
        t = t.map(lambda v: f"{v:.3f}").reset_index().rename(columns={"index": "Dataset"})
        t.columns = ["Dataset"] + [f"k={c}" for c in t.columns[1:]]
        write("real_lof_k", "Reales — diagnóstico: AUPRC de LOF según el número de vecinos k", t,
              "k = 20 es el valor por defecto (PyOD) usado en la comparación. Los demás k se eligen mirando etiquetas: "
              "sólo explican el fallo de LOF (anomalías colectivas más numerosas que k), no son un resultado comparable.")


# ==========================================================================
# Contaminación
# ==========================================================================
def contamination(deck: dict):
    df = pd.read_csv(RES / "contamination" / "eval_long.csv")
    df["ratio"] = df.dataset.str.replace("r", "").astype(float)
    sel = [("lstm", "tracin", "TracIn (artículo)"), ("lstm", "tracin_tgt", "TracIn proy. objetivo"),
           ("lstm", "rw_soft_r3", "TracIn + reponderación soft"), ("lstm", "rw_soft_r3_tgt", "TracIn proy. objetivo + soft"),
           ("lstm", "rw_trim_r3", "TracIn + recorte"), ("lstm", "pd", "LSTM error (PD)"),
           ("iforest", "score", "Isolation Forest"), ("ocsvm", "score", "One-Class SVM"), ("lof", "score", "LOF"),
           ("ae", "score", "Autoencoder"), ("mp", "score", "Matrix Profile")]
    ratios = sorted(df.ratio.unique())
    true_rate = df.groupby("ratio").true_rate.mean()
    deck["cont"] = {"ratios": ratios, "true_rate": true_rate.round(4).tolist()}
    for metric, lab in [("auprc", "AUPRC"), ("dyn_f1", "F1 dinámico"), ("log_mad_f1", "F1 log-MAD")]:
        rows, series = [], {}
        for m, v, name in sel:
            d = df[(df.method == m) & (df.variant == v)]
            if d.empty:
                continue
            g = d.groupby("ratio")[metric].agg(["mean", "std"])
            rows.append({"Método": name, **{f"{true_rate[r] * 100:.1f} %": fmt(g.loc[r, "mean"], g.loc[r, "std"]) for r in ratios}})
            series[name] = g["mean"].round(4).tolist()
        write(f"cont_{metric}", f"Contaminación controlada — {lab} vs. proporción real de anomalías (4000 pasos, L=4, P=1)",
              pd.DataFrame(rows))
        deck["cont"][metric] = series
    # prueba: reponderación vs original por nivel
    tests = {}
    for r in ratios:
        a = df[(df.ratio == r) & (df.variant == "rw_soft_r3")].set_index("seed").auprc
        b = df[(df.ratio == r) & (df.variant == "tracin")].set_index("seed").auprc
        tests[f"{r:.2f}"] = paired_test(a, b)
    deck["cont"]["rw_tests"] = tests
    # TracIn proyectado al objetivo vs error final, por nivel (¿dónde aporta la trayectoria?)
    tt = {}
    for r in ratios:
        a = df[(df.ratio == r) & (df.variant == "tracin_tgt")].set_index("seed").auprc
        b = df[(df.ratio == r) & (df.variant == "pd")].set_index("seed").auprc
        tt[f"{r:.2f}"] = paired_test(a, b)
    deck["cont"]["tgt_vs_pd"] = tt
    print("TracIn proy. objetivo − PD (AUPRC, Wilcoxon):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in tt.items()})
    # memorización: razón pérdida anómala/normal (paso predicho) al final vs su máximo
    memo = {}
    for f in sorted((RES / "contamination" / "scores").glob("r*_s*.npz")):
        z = np.load(f)
        r = float(f.stem.split("_")[0][1:])
        loss, y = z["lstm|trace_loss"], z["y"]
        tgt = y[4:4 + loss.shape[1]].astype(bool)
        ratio = loss[:, tgt].mean(1) / loss[:, ~tgt].mean(1)
        memo.setdefault(r, []).append((ratio.max(), ratio[-1]))
    deck["cont"]["memo"] = {f"{r:.2f}": {"max": float(np.median([m for m, _ in v])), "final": float(np.median([e for _, e in v])),
                                          "final_over_max": float(np.median([e / m for m, e in v]))} for r, v in sorted(memo.items())}
    rows = [{"Contaminación": f"{true_rate[r] * 100:.1f} %", "Razón máx.": f"{deck['cont']['memo'][f'{r:.2f}']['max']:.1f}×",
             "Razón final": f"{deck['cont']['memo'][f'{r:.2f}']['final']:.1f}×",
             "Final / máx.": f"{deck['cont']['memo'][f'{r:.2f}']['final_over_max']:.2f}",
             "Δ AUPRC proy. obj. − PD (p)": f"{tt[f'{r:.2f}'][0]:+.3f} ({tt[f'{r:.2f}'][1]:.2g})",
             "Δ AUPRC soft − original (p)": f"{tests[f'{r:.2f}'][0]:+.3f} ({tests[f'{r:.2f}'][1]:.2g})"} for r in ratios]
    write("cont_memorizacion", "Contaminación — memorización del LSTM y dónde aporta la trayectoria",
          pd.DataFrame(rows), "Razón = pérdida media de ventanas cuyo paso predicho es anómalo / de las normales (mediana entre semillas). "
          "Final/máx < 1 indica que el modelo empezó a aprender (memorizar) las anomalías.")
    print("\nreponderación soft r3 − original (AUPRC, Wilcoxon):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in tests.items()})


def gecco_sensitivity(deck: dict):
    f = RES / "real" / "gecco_sensitivity.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    t = g[g.score == "tracin"]
    by = t.groupby(["variante", "L", "P"])[["best_f1", "dyn_f1", "static_f1", "auprc"]].mean().reset_index()
    best = by.loc[by.best_f1.idxmax()]
    deck["gecco_sens"] = {"n_variants": int(g.variante.nunique()), "n_runs": int(len(t)),
                          "max_best_f1": float(by.best_f1.max()), "max_dyn_f1": float(by.dyn_f1.max()),
                          "max_static_f1": float(by.static_f1.max()),
                          "best_variant": f"{best.variante}, L={best.L}, P={best.P}"}
    rows = [{"Variante": r.variante, "L": r.L, "P": r.P, "AUPRC": fmt(r.auprc), "F1 oráculo": fmt(r.best_f1),
             "F1 dinámico": fmt(r.dyn_f1), "F1 estático (τ·máx)": fmt(r.static_f1)} for r in by.itertuples()]
    write("gecco_sensitivity", "GECCO — sensibilidad de TracIn a hiperparámetros no especificados (media de 2 semillas)",
          pd.DataFrame(rows), "Artículo: LSTM+S 0.604, LSTM+D 0.761.")


def counts(deck: dict):
    """Cuenta lo efectivamente entrenado a partir de los archivos guardados."""
    syn = [f for f in (RES / "synthetic" / "scores").glob("*_s*.npz") if ".tmp" not in f.name]
    con = [f for f in (RES / "contamination" / "scores").glob("*_s*.npz") if ".tmp" not in f.name]
    real = [f for f in (RES / "real" / "scores").glob("*/*.npz") if ".tmp" not in f.name]
    kind = lambda f: f.stem.split("_")[0]
    fc = (len(syn) * 9 * 7 + len(con) * 7 + sum(kind(f) in ("lstm", "tf") for f in real)
          + 3 * sum(kind(f) == "rw" for f in real))
    bl = len(syn) * 19 + len(con) * 5 + sum(kind(f) not in ("lstm", "tf", "rw") for f in real)
    deck["counts"] = {"forecasters": int(fc), "baselines": int(bl), "syn_files": len(syn),
                      "cont_files": len(con), "real_files": len(real)}


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    deck_path = RES / "deck_data.json"
    deck = json.loads(deck_path.read_text()) if deck_path.exists() else {}
    for name, fn in [("synthetic", synthetic), ("real", real), ("contamination", contamination)]:
        if name in which or "all" in which:
            if (RES / {"synthetic": "synthetic", "real": "real", "contamination": "contamination"}[name] / "eval_long.csv").exists():
                fn(deck)
    gecco_sensitivity(deck)
    counts(deck)
    deck_path.write_text(json.dumps(deck, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
