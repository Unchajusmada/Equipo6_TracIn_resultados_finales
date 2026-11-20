"""Tablas de la réplica (results/replica/tables/*.md) y números para el informe.

Sintéticos (results/replica/synth_base_eval.csv):
  1. Reproducción de la Tabla 2: TracIn estático (L4P1, percentil 85) y
     dinámico (mejor (L, P) por dataset), F1 binario y macro, vs artículo.
  2. Mismo protocolo para todos: métodos de referencia evaluados igual.
  3. Nuestras mejoras sobre la configuración reproducida:
     proyección al objetivo, umbrales sin etiquetas, reponderación.
Reales (results/replica/real_full/*.npz): igual, vs Tabla 4.
Contaminación (results/replica/contam/*.npz): AUPRC vs proporción de
anomalías y pruebas pareadas de las mejoras con la configuración reproducida.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_numbers import SYN_DATASETS, TABLE2, TABLE4

RES = Path(__file__).resolve().parents[2] / "results" / "replica"
TAB = RES / "tables"
NAMES = {("tracin_lstm", "tracin"): "TracIn (artículo)", ("tracin_lstm", "tracin_tgt"): "TracIn proy. objetivo",
         ("tracin_lstm", "pd"): "Error LSTM (PD)", ("iforest", "score"): "Isolation Forest",
         ("ocsvm", "score"): "One-Class SVM", ("lof", "score"): "LOF", ("ae", "score"): "Autoencoder",
         ("mp", "score"): "Matrix Profile"}
UNSUP = ["iqr", "mad", "log_iqr", "log_mad", "pot"]


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def write(name, title, df, note=""):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(f"## {title}\n\n{md(df)}\n\n{note}\n", encoding="utf-8")
    print(f"\n## {title}\n{md(df)}\n{note}")


def paper_style(d: pd.DataFrame, metric: str, fixed_cfg: str | None = None) -> pd.Series:
    """Media entre semillas por (ds, cfg); con fixed_cfg usa esa config, si
    no la mejor por dataset (como el artículo). Devuelve una serie por ds."""
    g = d.groupby(["ds", "cfg"])[metric].mean()
    if fixed_cfg is not None:
        return g.xs(fixed_cfg, level="cfg")
    return g.groupby(level=0).max()


def per_seed_best(d: pd.DataFrame, metric: str) -> pd.Series:
    """Para pruebas pareadas: por dataset fija la mejor config (media) y
    devuelve el valor por (ds, semilla) en esa config."""
    g = d.groupby(["ds", "cfg"])[metric].mean()
    best = g.groupby(level=0).idxmax()
    rows = [d[(d.ds == ds) & (d.cfg == cfg)].set_index(["ds", "seed"])[metric] for ds, cfg in best.values]
    return pd.concat(rows)


def pair(a: pd.Series, b: pd.Series):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    dlt = j.a - j.b
    p = wilcoxon(j.a, j.b).pvalue if (dlt != 0).sum() >= 5 else float("nan")
    return float(dlt.mean()), float(p), int((dlt > 0).sum()), len(dlt)


def synthetic(num: dict):
    df = pd.read_csv(RES / "synth_base_eval.csv", dtype={"ds": str})
    T = df[df.method == "tracin_lstm"]
    tr = T[T.score == "tracin"]
    # ---- 1. reproducción ----
    rows = []
    for ds in SYN_DATASETS:
        if not (tr.ds == ds).any():
            continue
        r = {"Dataset": ds}
        r["Art. LSTM+S"] = f"{TABLE2['LSTM+S'][SYN_DATASETS.index(ds)]:.3f}"
        for m, lab in (("S_pct_mac", "Nuestro S macro"), ("S_pct_bin", "Nuestro S binario")):
            r[lab] = f"{paper_style(tr[tr.ds == ds], m, 'L4P1').iloc[0]:.3f}"
        r["Art. LSTM+D"] = f"{TABLE2['LSTM+D'][SYN_DATASETS.index(ds)]:.3f}"
        for m, lab in (("D_mac", "Nuestro D macro"), ("D_bin", "Nuestro D binario")):
            r[lab] = f"{paper_style(tr[tr.ds == ds], m).iloc[0]:.3f}"
        rows.append(r)
    avg = {"Dataset": "Promedio", "Art. LSTM+S": f"{np.mean(TABLE2['LSTM+S']):.3f}",
           "Nuestro S macro": f"{paper_style(tr, 'S_pct_mac', 'L4P1').mean():.3f}",
           "Nuestro S binario": f"{paper_style(tr, 'S_pct_bin', 'L4P1').mean():.3f}",
           "Art. LSTM+D": f"{np.mean(TABLE2['LSTM+D']):.3f}",
           "Nuestro D macro": f"{paper_style(tr, 'D_mac').mean():.3f}",
           "Nuestro D binario": f"{paper_style(tr, 'D_bin').mean():.3f}"}
    rows.append(avg)
    write("rep_syn_table2", "Réplica de la Tabla 2 (sintéticos, 20 semillas, datos estandarizados)", pd.DataFrame(rows),
          "S: L=4, P=1, umbral = percentil 85 de los scores (lectura literal del texto). D: umbral (percentil) elegido en la "
          "primera mitad de la serie, F1 medido en la segunda; mejor (L, P) por dataset según la media de 20 semillas, como el "
          "artículo. 'macro' = promedio del F1 de la clase normal y de la anómala (f1_score(average='macro')).")
    num["syn"] = {"paper_S": float(np.mean(TABLE2["LSTM+S"])), "paper_D": float(np.mean(TABLE2["LSTM+D"])),
                  "S_mac": float(paper_style(tr, "S_pct_mac", "L4P1").mean()), "S_bin": float(paper_style(tr, "S_pct_bin", "L4P1").mean()),
                  "D_mac": float(paper_style(tr, "D_mac").mean()), "D_bin": float(paper_style(tr, "D_bin").mean()),
                  "S_frac_mac": float(paper_style(tr, "S_frac_mac", "L4P1").mean()), "S_frac_bin": float(paper_style(tr, "S_frac_bin", "L4P1").mean())}
    dm = paper_style(tr, "D_mac").reindex(SYN_DATASETS)
    ok = dm.notna().values
    corr = np.corrcoef(dm.values[ok], np.array(TABLE2["LSTM+D"])[ok])[0, 1]
    num["syn"]["corr_D"] = float(corr)

    # ---- 2. mismo protocolo para todos (mejor config por dataset, como el artículo) ----
    rows = []
    for (meth, sc), name in NAMES.items():
        d = df[(df.method == meth) & (df.score == sc)]
        if d.empty:
            continue
        rows.append({"Método": name, "AUPRC (mejor config)": f"{paper_style(d, 'auprc').mean():.3f}",
                     "F1 dinámico macro": f"{paper_style(d, 'D_mac').mean():.3f}",
                     "F1 dinámico binario": f"{paper_style(d, 'D_bin').mean():.3f}",
                     "F1 IQR macro (sin etiquetas)": f"{paper_style(d, 'U_iqr_mac').mean():.3f}",
                     "F1 IQR binario (sin etiquetas)": f"{paper_style(d, 'U_iqr_bin').mean():.3f}"})
        num.setdefault("syn_cmp", {})[name] = {m: float(paper_style(d, m).mean()) for m in ("auprc", "D_mac", "D_bin", "U_iqr_mac", "U_iqr_bin")}
        for m in ("auprc", "D_mac", "D_bin"):
            # dispersión entre semillas del promedio sobre datasets (cada dataset en su mejor configuración)
            num["syn_cmp"][name][m + "_sd"] = float(per_seed_best(d, m).groupby(level="seed").mean().std())
        for typ, dss in (("point", ["0", "1"]), ("pattern", ["2", "3", "4"])):
            num["syn_cmp"][name]["auprc_" + typ] = float(paper_style(d[d.ds.isin(dss)], "auprc").mean())
    write("rep_syn_cmp", "Sintéticos: TracIn reproducido vs. otros detectores, mismo protocolo del artículo", pd.DataFrame(rows),
          "Mejor configuración por dataset (L, P para el LSTM; ventana para los demás), media de 20 semillas, promedio de 15 "
          "datasets. Todos sobre los mismos datos estandarizados y los mismos instantes.")

    # ---- 3. mejoras sobre la configuración reproducida ----
    base = per_seed_best(tr, "D_mac")
    rows = []
    def add(name, d, metric_d="D_mac", metric_b="D_bin", src=None):
        a_m = per_seed_best(d, metric_d)
        a_b = per_seed_best(d, metric_b)
        dm = pair(a_m, per_seed_best(tr, "D_mac"))
        db = pair(a_b, per_seed_best(tr, "D_bin"))
        auprc = per_seed_best(d, "auprc")
        da = pair(auprc, per_seed_best(tr, "auprc"))
        dfmt = lambda t: "—" if d is tr else f"{t[0]:+.3f} ({t[1]:.1e})"
        rows.append({"Variante": name, "AUPRC": f"{auprc.mean():.3f}", "Δ AUPRC (p)": dfmt(da),
                     "F1 macro": f"{a_m.mean():.3f}", "Δ F1 macro (p)": dfmt(dm),
                     "F1 binario": f"{a_b.mean():.3f}", "Δ F1 binario (p)": dfmt(db)})
        num.setdefault("syn_impr", {})[name] = {"auprc": float(auprc.mean()), "d_auprc": da[0], "p_auprc": da[1],
                                                 "mac": float(a_m.mean()), "d_mac": dm[0], "p_mac": dm[1],
                                                 "bin": float(a_b.mean()), "d_bin": db[0], "p_bin": db[1]}
    add("TracIn del artículo (reproducido)", tr)
    for sc, name in [("tracin_tgt", "Mejora 3 · proyección al objetivo"),
                     ("rw_soft_r2", "Mejora 2 · reponderación suave (2 rondas)"), ("rw_trim_r2", "Mejora 2 · recorte (2 rondas)"),
                     ("rw_soft_r2_tgt", "Mejoras 2 + 3 · suave + proyección"), ("rw_trim_r2_tgt", "Mejoras 2 + 3 · recorte + proyección")]:
        d = T[T.score == sc]
        if not d.empty:
            add(name, d)
    write("rep_syn_mejoras", "Sintéticos: nuestras mejoras sobre el TracIn reproducido (protocolo del artículo)", pd.DataFrame(rows),
          "F1 con el umbral dinámico del artículo. Δ y p (Wilcoxon pareado sobre 15 datasets × 20 semillas) respecto del TracIn "
          "reproducido, cada uno en su mejor (L, P) por dataset.")

    # ---- trayectoria vs. error final, misma proyección (tracin_tgt vs pd) ----
    for m in ("auprc", "D_mac"):
        a, b = T[T.score == "tracin_tgt"], T[T.score == "pd"]
        num.setdefault("traj", {})[f"syn_best_{m}"] = pair(per_seed_best(a, m), per_seed_best(b, m))
        fx = lambda d: d[d.cfg == "L4P1"].set_index(["ds", "seed"])[m]
        num["traj"][f"syn_L4P1_{m}"] = pair(fx(a), fx(b))

    # ---- 4. mejora 1: umbral sin etiquetas vs dinámico (que usa etiquetas) ----
    rows = []
    for sc, name in [("tracin", "TracIn (artículo)"), ("tracin_tgt", "TracIn proy. objetivo")]:
        d = T[T.score == sc]
        r = {"Score": name, "Dinámico (usa etiquetas) macro": f"{paper_style(d, 'D_mac').mean():.3f}",
             "Dinámico binario": f"{paper_style(d, 'D_bin').mean():.3f}",
             "Estático pct 85 macro": f"{paper_style(d, 'S_pct_mac').mean():.3f}"}
        for u in UNSUP:
            r[f"{u} macro"] = f"{paper_style(d, f'U_{u}_mac').mean():.3f}"
            r[f"{u} binario"] = f"{paper_style(d, f'U_{u}_bin').mean():.3f}"
            num.setdefault("syn_thr", {}).setdefault(name, {})[u] = (float(paper_style(d, f"U_{u}_mac").mean()),
                                                                      float(paper_style(d, f"U_{u}_bin").mean()))
        num["syn_thr"][name]["dyn"] = (float(paper_style(d, "D_mac").mean()), float(paper_style(d, "D_bin").mean()))
        rows.append(r)
    write("rep_syn_umbrales", "Sintéticos: mejora 1 (umbral sin etiquetas) sobre el TracIn reproducido", pd.DataFrame(rows),
          "Todas las reglas se miden en la segunda mitad de la serie (la misma que el dinámico). Las reglas sin etiquetas usan "
          "k = 1.5 (IQR), k = 3 (MAD) y q = 1e-3 (POT), fijados a priori; mejor (L, P) por dataset.")
    # completamente sin etiquetas: además del umbral, (L, P) fija a priori (L4P1), sin elegir la mejor con etiquetas
    rows = []
    for sc, name in [("tracin", "TracIn (artículo)"), ("tracin_tgt", "TracIn proy. objetivo"), ("pd", "Error LSTM (PD)")]:
        d = T[T.score == sc]
        f = lambda m: float(paper_style(d, m, "L4P1").mean())
        r = {"Score": name, "Dinámico (usa etiquetas)": f"{f('D_mac'):.3f}", "Estático pct 85": f"{f('S_pct_mac'):.3f}"}
        num.setdefault("syn_thr_L4P1", {})[name] = {"dyn": f("D_mac"), "pct": f("S_pct_mac"), "oracle": f("oracle_mac") if "oracle_mac" in d else None}
        for u in UNSUP:
            r[u] = f"{f(f'U_{u}_mac'):.3f}"
            num["syn_thr_L4P1"][name][u] = f(f"U_{u}_mac")
        rows.append(r)
    write("rep_syn_umbrales_L4P1", "Sintéticos, método completamente sin etiquetas: L4P1 fijo + umbral sin etiquetas (F1 macro)",
          pd.DataFrame(rows), "L = 4, P = 1 fijados a priori para todos los datasets (no se elige la mejor (L, P) con etiquetas). "
          "F1 macro en la segunda mitad de la serie, 20 semillas × 15 datasets.")


def real(num: dict):
    from common import valid_mask
    from data_real import load
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from real_screen import evaluate_real
    d = RES / "real_full"
    files = sorted(f for f in d.glob("*.npz") if ".tmp" not in f.name)
    if not files:
        return
    rows = []
    cache = {}
    for f in files:
        ds, cfg, seed = f.stem.split("_L")[0], "L" + f.stem.split("_L")[1].split("_s")[0], int(f.stem.split("_s")[-1])
        if ds not in cache:
            y = load(ds)[1]
            cache[ds] = (y, valid_mask(len(y)))
        y, m = cache[ds]
        z = np.load(f)
        for sc in z.files:
            if sc == "epochs":
                continue
            rows.append(dict(ds=ds, cfg=cfg, seed=seed, score=sc, **evaluate_real(y[m], z[sc][m].astype(np.float64))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "real_full_eval.csv", index=False)
    # elegir la mejor (L, P) sólo entre configuraciones con todas sus semillas: una
    # configuración a medio correr tendría un promedio más ruidoso (y un máximo sesgado)
    ns = df.groupby(["ds", "cfg"]).seed.nunique()
    full = ns[ns == ns.groupby(level=0).transform("max")].index
    pending = sorted(f"{a} {b}" for a, b in ns.index.difference(full))
    if pending:
        print("configuraciones incompletas (excluidas):", ", ".join(pending))
    num["real_pending"] = pending
    df = df.set_index(["ds", "cfg"]).loc[full].reset_index()
    out = []
    for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
        t = df[(df.ds == ds) & (df.score == "tracin")]
        if t.empty:
            continue
        g = t.groupby("cfg").mean(numeric_only=True)
        bc = g.D_bin_F1_bin.idxmax()
        P_, R_, F_ = TABLE4[ds]["LSTM+D"]
        Ps, Rs, Fs = TABLE4[ds]["LSTM+S"]
        st = g.loc["L4P1"] if "L4P1" in g.index else g.iloc[0]
        out.append({"Dataset": ds, "Art. LSTM+S (P/R/F1)": f"{Ps:.3f}/{Rs:.3f}/{Fs:.3f}",
                    "Nuestro S pct87 binario": f"{st.S_pct_P_bin:.3f}/{st.S_pct_R_bin:.3f}/{st.S_pct_F1_bin:.3f}",
                    "Nuestro S τ·máx binario": f"{st.S_frac_P_bin:.3f}/{st.S_frac_R_bin:.3f}/{st.S_frac_F1_bin:.3f}",
                    "Art. LSTM+D (P/R/F1)": f"{P_:.3f}/{R_:.3f}/{F_:.3f}",
                    "Nuestro D binario": f"{g.loc[bc, 'D_bin_P_bin']:.3f}/{g.loc[bc, 'D_bin_R_bin']:.3f}/{g.loc[bc, 'D_bin_F1_bin']:.3f} ({bc})",
                    "Nuestro D macro": f"{g.D_mac_F1_mac.max():.3f}",
                    "Nuestro D binario, partición aleatoria": f"{g.Drand_F1_bin.max():.3f}",
                    "AUPRC": f"{g.auprc.max():.3f}"})
        num.setdefault("real", {})[ds] = {"paper_D": F_, "paper_S": Fs, "D_bin": float(g.D_bin_F1_bin.max()),
                                          "D_mac": float(g.D_mac_F1_mac.max()), "D_rand": float(g.Drand_F1_bin.max()), "S_bin": float(st.S_pct_F1_bin),
                                          "S_mac": float(st.S_pct_F1_mac), "S_frac": float(st.S_frac_F1_bin), "auprc": float(g.auprc.max()),
                                          "best_cfg": bc, "n_seeds": int(t.groupby("cfg").seed.nunique().min()), "n_cfgs": int(t.cfg.nunique())}
    write("rep_real_table4", "Réplica de la Tabla 4 (reales, TracIn con gradientes de todos los parámetros)", pd.DataFrame(out),
          "S: L=4, P=1; percentil 87 (lectura literal) o τ = 0.87 como fracción del máximo. D: umbral elegido en la primera mitad "
          "(contigua) y F1 medido en la segunda, o en una partición aleatoria 50/50 de los instantes. Mejor (L, P) por dataset; "
          "3 semillas por configuración, como el artículo. F1 binario (clase anómala), como en la Tabla 4. (L, P) evaluadas: "
          + "; ".join(f"{ds}: " + ", ".join(sorted(df[df.ds == ds].cfg.unique(), key=lambda c: (int(c[1:].split('P')[0]), int(c.split('P')[1]))))
                      for ds in ["creditcard", "cicids", "gecco", "swan_sf"] if ds in set(df.ds)) + ".")
    # mejoras en reales
    out = []
    for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
        t = df[df.ds == ds]
        if t.empty:
            continue
        r = {"Dataset": ds}
        for sc, lab in [("tracin", "TracIn"), ("tracin_tgt", "proy. objetivo"), ("rw_soft_r1", "suave"),
                        ("rw_trim_r1", "recorte"), ("rw_trim_r1_tgt", "recorte + proy.")]:
            x = t[t.score == sc].groupby("cfg").mean(numeric_only=True)
            if x.empty:
                continue
            r[f"{lab} AUPRC"] = f"{x.auprc.max():.3f}"
            r[f"{lab} F1 dinámico"] = f"{x.D_bin_F1_bin.max():.3f}"
            r[f"{lab} F1 dinámico (aleatoria)"] = f"{x.Drand_F1_bin.max():.3f}"
            num.setdefault("real_impr", {}).setdefault(ds, {})[sc] = {"auprc": float(x.auprc.max()), "D_bin": float(x.D_bin_F1_bin.max()), "D_rand": float(x.Drand_F1_bin.max()),
                                                                     "D_mac": float(x.D_mac_F1_mac.max())}
        out.append(r)
    # métodos de referencia del experimento principal, mismo protocolo
    bpath = RES / "real_baselines_eval.csv"
    if not bpath.exists():
        brows = []
        sc_dir = RES.parent / "real" / "scores"
        for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
            y, m = cache.get(ds, (None, None))
            if y is None:
                y = load(ds)[1]; m = valid_mask(len(y))
            for f in sorted((sc_dir / ds).glob("*.npz")):
                parts = f.stem.split("_")
                meth = parts[0]
                if meth not in ("iforest", "ocsvm", "ae", "lof", "mp"):
                    continue
                if meth in ("iforest", "ocsvm", "ae") and int(parts[2]) > 2:
                    continue
                s = np.load(f)["score"][m].astype(np.float64)
                brows.append(dict(ds=ds, method=meth, cfg=parts[1], seed=int(parts[2]) if len(parts) > 2 else 0,
                                  **evaluate_real(y[m], s)))
        pd.DataFrame(brows).to_csv(bpath, index=False)
    bdf = pd.read_csv(bpath)
    cmp_rows = []
    for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
        r = {"Dataset": ds}
        t = df[(df.ds == ds) & (df.score == "tracin")].groupby("cfg").mean(numeric_only=True)
        if t.empty:
            continue
        r["TracIn (réplica)"] = f"{t.auprc.max():.3f} / {t.D_bin_F1_bin.max():.3f}"
        tg = df[(df.ds == ds) & (df.score == "tracin_tgt")].groupby("cfg").mean(numeric_only=True)
        r["TracIn proy. objetivo"] = f"{tg.auprc.max():.3f} / {tg.D_bin_F1_bin.max():.3f}"
        for meth, lab in [("iforest", "IForest"), ("ocsvm", "OCSVM"), ("lof", "LOF"), ("ae", "Autoencoder"), ("mp", "Matrix Profile")]:
            x = bdf[(bdf.ds == ds) & (bdf.method == meth)].groupby("cfg").mean(numeric_only=True)
            if len(x):
                r[lab] = f"{x.auprc.max():.3f} / {x.D_bin_F1_bin.max():.3f}"
                num.setdefault("real_cmp", {}).setdefault(ds, {})[lab] = (float(x.auprc.max()), float(x.D_bin_F1_bin.max()))
        num.setdefault("real_cmp", {}).setdefault(ds, {})["TracIn"] = (float(t.auprc.max()), float(t.D_bin_F1_bin.max()))
        num["real_cmp"][ds]["TracIn proy. objetivo"] = (float(tg.auprc.max()), float(tg.D_bin_F1_bin.max()))
        cmp_rows.append(r)
    for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
        a = df[(df.ds == ds) & (df.score == "tracin_tgt")]
        b = df[(df.ds == ds) & (df.score == "pd")]
        if a.empty:
            continue
        bc = a.groupby("cfg").auprc.mean().idxmax()
        x = a[a.cfg == bc].set_index("seed").auprc
        y = b[b.cfg == bc].set_index("seed").auprc
        dl = (x - y).dropna()
        num.setdefault("traj", {})[f"real_{ds}"] = (float(dl.mean()), float("nan"), int((dl > 0).sum()), len(dl),
                                                     float(x.mean()), float(y.mean()), bc)
    # mejora 1 en reales: L4P1 fijo (sin elegir configuración con etiquetas) y reglas sin etiquetas
    thr_rows = []
    rules = [("D_bin", "Dinámico (usa etiquetas)"), ("S_pct", "Estático pct 87"), ("U_iqr", "IQR"), ("U_mad", "MAD"),
             ("U_log_iqr", "log-IQR"), ("U_log_mad", "log-MAD"), ("U_pot", "EVT-POT")]
    for sc, name in [("tracin", "TracIn (artículo)"), ("tracin_tgt", "TracIn proy. objetivo")]:
        for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
            x = df[(df.ds == ds) & (df.score == sc) & (df.cfg == "L4P1")]
            if x.empty or "U_iqr_F1_bin" not in x:
                continue
            x = x.mean(numeric_only=True)
            r = {"Score": name, "Dataset": ds}
            for key, lab in rules:
                r[lab] = f"{x[key + '_F1_bin']:.3f}"
                num.setdefault("real_thr", {}).setdefault(name, {}).setdefault(ds, {})[key] = float(x[key + "_F1_bin"])
            thr_rows.append(r)
    if thr_rows:
        write("rep_real_umbrales", "Reales, mejora 1: umbral sin etiquetas (L4P1 fijo, F1 binario)", pd.DataFrame(thr_rows),
              "Configuración L4P1 fijada a priori; todas las reglas medidas en la segunda mitad de la serie, como el dinámico; "
              "3 semillas.")
    write("rep_real_cmp", "Reales: TracIn de la réplica vs. otros detectores (AUPRC / F1 dinámico binario)", pd.DataFrame(cmp_rows),
          "Mejor configuración por dataset; mismo protocolo (umbral elegido en la primera mitad, F1 en la segunda).")
    write("rep_real_mejoras", "Reales: nuestras mejoras sobre el TracIn de la réplica", pd.DataFrame(out),
          "Mejor (L, P) por dataset; F1 binario con el umbral dinámico del artículo.")


SCREEN_DESC = {"base": "Base: z-score por canal, lr 1e-3, batch 32, ≤ 100 épocas (paciencia 10), LSTM 64, gradiente de la última capa",
               "full": "Gradiente de todos los parámetros (Ec. 2 completa)", "lr1e-2": "lr 1e-2", "b8": "batch 8",
               "ep300": "≤ 300 épocas (paciencia 30)", "ep30": "30 épocas fijas", "h128": "LSTM de 128", "h16": "LSTM de 16",
               "nosched": "Sin reducción del lr", "nolrw": "Sin ponderar por el lr (η_k = 1)",
               "val20": "20 % final como validación para early stopping", "minmax": "Normalización min-max",
               "nonorm": "Sin normalizar (nuestro primer experimento)"}


def screen(num: dict):
    from summarize import summarize
    path = RES / "screen.csv"
    if not path.exists():
        return
    s = summarize(pd.read_csv(path, dtype={"ds": str}))
    s = s[s.score == "tracin"].set_index("variant")
    rows = []
    for v in [v for v in SCREEN_DESC if v in s.index]:
        r = s.loc[v]
        rows.append({"Variante": SCREEN_DESC[v], "AUPRC (L4P1)": f"{r.auprc:.3f}", "S macro": f"{r.S_pct_mac:.3f}",
                     "D macro": f"{r.D_mac:.3f}", "D binario": f"{r.D_bin:.3f}"})
        num.setdefault("screen", {})[v] = {k: float(r[k]) for k in ("auprc", "S_pct_mac", "D_mac", "D_bin")}
    rows.insert(0, {"Variante": "Art. (Tabla 2)", "AUPRC (L4P1)": "—", "S macro": f"{np.mean(TABLE2['LSTM+S']):.3f}",
                    "D macro": f"{np.mean(TABLE2['LSTM+D']):.3f}", "D binario": "—"})
    write("rep_syn_screen", "Sintéticos: barrido de lo que el artículo no especifica (TracIn, 5 semillas)", pd.DataFrame(rows),
          f"Cada fila cambia sólo eso respecto de la base. {int(s.n_seeds.min())} semillas × 15 datasets × 9 (L, P); mismo protocolo "
          "que la Tabla 2 (estático L4P1 percentil 85, dinámico con la mejor (L, P) por dataset).")


def real_hyp(num: dict):
    """Hipótesis descartadas en reales: hiperparámetros (GECCO) y fuga de la etiqueta."""
    rows = []
    sc = RES / "real_screen.csv"
    if sc.exists():
        d = pd.read_csv(sc)
        d = d[d.score == "tracin"]
        for v in [v for v in SCREEN_DESC if v in set(d.variant)]:
            g = d[d.variant == v].groupby(["L", "P"]).mean(numeric_only=True)
            lab = "Base: z-score por canal, lr 1e-3, LSTM 64, gradiente de la última capa" if v == "base" else SCREEN_DESC[v]
            rows.append({"Hipótesis": f"GECCO · {lab}", "AUPRC": f"{g.auprc.max():.3f}",
                         "F1 dinámico binario": f"{g.D_bin_F1_bin.max():.3f}", "Art.": f"{TABLE4['gecco']['LSTM+D'][2]:.3f}"})
            num.setdefault("real_hyp", {}).setdefault("gecco_screen", {})[v] = (float(g.auprc.max()), float(g.D_bin_F1_bin.max()))
    lk = RES / "real_leak.csv"
    if lk.exists():
        d = pd.read_csv(lk)
        d = d[d.score == "tracin"]
        for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
            g = d[d.ds == ds].groupby(["L", "P"]).mean(numeric_only=True)
            if len(g):
                rows.append({"Hipótesis": f"{ds} · la etiqueta entra como canal (fuga)", "AUPRC": f"{g.auprc.max():.3f}",
                             "F1 dinámico binario": f"{g.D_bin_F1_bin.max():.3f}", "Art.": f"{TABLE4[ds]['LSTM+D'][2]:.3f}"})
                num.setdefault("real_hyp", {}).setdefault("leak", {})[ds] = (float(g.auprc.max()), float(g.D_bin_F1_bin.max()))
    if rows:
        write("rep_real_hipotesis", "Reales: hipótesis probadas para explicar la diferencia con la Tabla 4", pd.DataFrame(rows),
              "Barrido en GECCO: 2 semillas, L4P1 y L16P4, 50 épocas, batch 256. Fuga: se añade la etiqueta estandarizada como un "
              "canal más (los conteos de canales de la Tabla 1 coinciden con atributos + etiqueta); 2 semillas, L4P1 y L16P4. "
              "Mejor configuración por dataset. Diagnóstico, no un método válido.")


def counts(num: dict):
    """Cuántos forecasters (cada uno con su traza de checkpoints) y ajustes de referencia se corrieron en la réplica."""
    ok = lambda d: [f for f in (RES / d).glob("*.npz") if ".tmp" not in f.name] if (RES / d).exists() else []
    n_syn = len(ok("synth_base")) * 9 * 5
    n_scr = len(pd.read_csv(RES / "screen.csv")) // 3 if (RES / "screen.csv").exists() else 0
    n_con = len(ok("contam")) * 5
    n_real = len(ok("real_full")) * 3
    n_rs = sum(len(pd.read_csv(RES / f)) // 3 for f in ("real_screen.csv", "real_leak.csv") if (RES / f).exists())
    n_cfx = len(ok("contam_fix")) * 5  # base + 3 rondas de recorte adaptativo + LSTM de 16
    num["counts"] = {"forecasters": n_syn + n_scr + n_con + n_real + n_rs + n_cfx, "syn": n_syn, "screen": n_scr, "contam": n_con,
                     "contam_fix": n_cfx,
                     "real": n_real, "real_screen": n_rs, "baselines": len(ok("synth_baselines")) * 19 + len(ok("contam")) * 5}


CONT_SCORES = [("tracin", "TracIn (artículo)"), ("tracin_tgt", "TracIn proy. objetivo"), ("pd", "Error LSTM (PD)"),
               ("rw_soft_r2", "Reponderación suave"), ("rw_trim_r2", "Recorte"), ("rw_soft_r2_tgt", "Suave + proy."),
               ("rw_trim_r2_tgt", "Recorte + proy."), ("iforest", "IForest"), ("ocsvm", "OCSVM"), ("lof", "LOF"),
               ("ae", "Autoencoder"), ("mp", "Matrix Profile")]


def contamination(num: dict):
    from sklearn.metrics import roc_auc_score
    from common import valid_mask
    from rep_eval import eval_one
    files = sorted(f for f in (RES / "contam").glob("r*_s*.npz") if ".tmp" not in f.name)
    if not files:
        return
    rows = []
    for f in files:
        r, seed = f.stem[1:].split("_s")
        z = np.load(f)
        y = z["y"].astype(bool)
        m = valid_mask(len(y))
        for sc, _ in CONT_SCORES:
            if sc in z.files:
                rows.append(dict(ratio=float(r), seed=int(seed), score=sc, rate=float(y[m].mean()),
                                 auroc=float(roc_auc_score(y[m], z[sc][m].astype(np.float64))),
                                 **eval_one(y[m], z[sc][m].astype(np.float64))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "contam_eval.csv", index=False)
    ratios = sorted(df.ratio.unique())
    rate = df.groupby("ratio").rate.mean()
    num["cont"] = {"ratios": ratios, "rate": [float(rate[r]) for r in ratios], "n_seeds": int(df.seed.nunique())}
    # AUROC no depende de la proporción de anomalías (el AUPRC del azar es esa proporción y sube con ella),
    # así que es la métrica para ver si un método se degrada con la contaminación
    for metric, lab in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("D_bin", "F1 dinámico binario")):
        t = df.pivot_table(index="score", columns="ratio", values=metric, aggfunc="mean")
        out = []
        for sc, name in CONT_SCORES:
            if sc in t.index:
                out.append({"Método": name, **{f"{rate[r] * 100:.1f} %": f"{t.loc[sc, r]:.3f}" for r in ratios}})
                num["cont"].setdefault(metric, {})[name] = [float(t.loc[sc, r]) for r in ratios]
        write(f"rep_cont_{metric}", f"Contaminación (configuración reproducida) — {lab} vs. proporción real de anomalías",
              pd.DataFrame(out), f"Serie «01234» de 4000 pasos, L = 4, P = 1 fijos a priori, datos estandarizados, "
              f"media de {df.seed.nunique()} semillas. Columnas: proporción real de instantes anómalos.")
    # pruebas pareadas por proporción (semillas)
    for metric, lab in (("auroc", "AUROC"), ("auprc", "AUPRC")):
        out = []
        piv = df.pivot_table(index=["ratio", "seed"], columns="score", values=metric)
        for a, b, name in [("rw_soft_r2", "tracin", "Suave − TracIn"), ("rw_trim_r2", "tracin", "Recorte − TracIn"),
                           ("rw_trim_r2_tgt", "tracin_tgt", "Recorte + proy. − Proy. objetivo"),
                           ("tracin_tgt", "tracin", "Proy. objetivo − TracIn"), ("tracin_tgt", "pd", "Proy. objetivo − PD"),
                           ("tracin", "pd", "TracIn − PD")]:
            r_ = {f"Comparación ({lab})": name}
            for r in ratios:
                x = piv.xs(r, level="ratio")
                dlt, p, k, n = pair(x[a], x[b])
                r_[f"{rate[r] * 100:.1f} %"] = f"{dlt:+.3f} ({k}/{n})"
                num["cont"].setdefault("tests" if metric == "auprc" else "tests_auroc", {}).setdefault(name, []).append((dlt, p, k, n))
            out.append(r_)
        write("rep_cont_tests" + ("" if metric == "auprc" else "_auroc"),
              f"Contaminación (configuración reproducida) — diferencias pareadas de {lab}", pd.DataFrame(out),
              "Media de la diferencia por semilla; entre paréntesis, semillas en que el primer método gana.")


def contam_fix(num: dict):
    """Variantes para frenar la degradación restante (rep_contam_fix.py), AUROC."""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from common import valid_mask
    files = sorted(f for f in (RES / "contam_fix").glob("r*_s*.npz") if ".tmp" not in f.name)
    if not files:
        return
    rk = lambda v: rankdata(v) / len(v)
    rows = []
    for f in files:
        r, seed = f.stem[1:].split("_s")
        z, o = np.load(f), np.load(RES / "contam" / f.name)
        y = z["y"].astype(bool); m = valid_mask(len(y)); y = y[m]
        sc = {"TracIn (artículo)": o["tracin"], "Proy. objetivo": z["tracin_tgt"], "Recorte z > 2 (2 rondas)": o["rw_trim_r2"],
              "Recorte adaptativo IQR (3 rondas)": z["adapt_r3"], "Recorte z > 2 + proy. (2 rondas)": o["rw_trim_r2_tgt"],
              "Recorte adaptativo IQR + proy. (3 rondas)": z["adapt_r3_tgt"], "Épocas tempranas + proy.": z["early_tgt"],
              "LSTM de 16 + proy.": z["h16_tgt"], "LOF": o["lof"], "Error LSTM (PD)": o["pd"]}
        sc = {k: v[m].astype(np.float64) for k, v in sc.items()}
        sc["Ensamble rangos (recorte z > 2 + proy., LOF)"] = rk(sc["Recorte z > 2 + proy. (2 rondas)"]) + rk(sc["LOF"])
        d = dict(ratio=float(r), seed=int(seed), rate=float(y.mean()))
        for k, v in sc.items():
            rows.append(dict(d, score=k, auroc=float(roc_auc_score(y, v))))
        for k in ("trim", "adapt_r3"):
            rows.append(dict(d, score=f"diag_{k}", frac=float(z[f"diag_{k}_frac"]), anom=float(z[f"diag_{k}_anom"])))
    df = pd.DataFrame(rows)
    ratios = sorted(df.ratio.unique()); rate = df.groupby("ratio").rate.mean()
    t = df.dropna(subset=["auroc"]).pivot_table(index="score", columns="ratio", values="auroc", aggfunc="mean")
    order = ["TracIn (artículo)", "Proy. objetivo", "Recorte z > 2 (2 rondas)", "Recorte adaptativo IQR (3 rondas)",
             "Recorte z > 2 + proy. (2 rondas)", "Recorte adaptativo IQR + proy. (3 rondas)",
             "Épocas tempranas + proy.", "LSTM de 16 + proy.", "Ensamble rangos (recorte z > 2 + proy., LOF)", "LOF", "Error LSTM (PD)"]
    out = []
    num["cont_fix"] = {"rate": [float(rate[r]) for r in ratios], "auroc": {}}
    for k in order:
        v = [float(t.loc[k, r]) for r in ratios]
        num["cont_fix"]["auroc"][k] = v
        out.append({"Método": k, **{f"{rate[r] * 100:.1f} %": f"{x:.3f}" for r, x in zip(ratios, v)}, "Caída 1 % → 26 %": f"{v[-1] - v[0]:+.3f}"})
    write("rep_cont_fix", "Contaminación: ¿se puede frenar la degradación? (AUROC)", pd.DataFrame(out),
          "Misma serie, semillas y L4P1 que la tabla anterior; 10 semillas. Variantes fijadas a priori: recorte adaptativo = descartar "
          "las ventanas con self-influence sobre el umbral IQR de la mejora 1 (3 rondas); épocas tempranas = TracIn con el primer "
          "cuarto de las épocas; ensamble = suma de rangos, pesos iguales.")
    piv = df.dropna(subset=["auroc"]).pivot_table(index=["ratio", "seed"], columns="score", values="auroc")
    num["cont_fix"]["tests"] = {}
    for a_, b_ in [("Recorte adaptativo IQR + proy. (3 rondas)", "Recorte z > 2 + proy. (2 rondas)"),
                   ("Épocas tempranas + proy.", "Proy. objetivo"), ("LSTM de 16 + proy.", "Proy. objetivo"),
                   ("Ensamble rangos (recorte z > 2 + proy., LOF)", "Recorte z > 2 + proy. (2 rondas)")]:
        num["cont_fix"]["tests"][f"{a_} − {b_}"] = [pair(piv.xs(r, level="ratio")[a_], piv.xs(r, level="ratio")[b_]) for r in ratios]
    dg = df[df.score.str.startswith("diag")].groupby(["score", "ratio"])[["frac", "anom"]].mean()
    out = []
    for k, lab in (("diag_trim", "Recorte z > 2 (1.ª ronda)"), ("diag_adapt_r3", "Recorte adaptativo IQR (3.ª ronda)")):
        out.append({"Regla": lab, **{f"{rate[r] * 100:.1f} %": f"{dg.loc[(k, r), 'frac'] * 100:.0f} % / {dg.loc[(k, r), 'anom'] * 100:.0f} %" for r in ratios}})
        num["cont_fix"][k] = [(float(dg.loc[(k, r), "frac"]), float(dg.loc[(k, r), "anom"])) for r in ratios]
    write("rep_cont_fix_diag", "Contaminación: cuánto recorta cada regla (ventanas recortadas / ventanas anómalas recortadas)",
          pd.DataFrame(out), "Ventana anómala: algún instante de su entrada o de su objetivo es anómalo. Las etiquetas sólo se usan "
          "para este diagnóstico.")


def trajectory(num: dict):
    T = num.get("traj", {})
    rows = []
    f = lambda t: f"{t[0]:+.3f} ({t[2]}/{t[3]})"
    if "syn_best_auprc" in T:
        rows.append({"Datos": "Sintéticos, mejor (L, P) por dataset", "Δ AUPRC (gana TracIn)": f(T["syn_best_auprc"]),
                     "p (Wilcoxon)": f"{T['syn_best_auprc'][1]:.1e}", "Δ F1 dinámico macro": f(T["syn_best_D_mac"])})
        rows.append({"Datos": "Sintéticos, L4P1 fijo", "Δ AUPRC (gana TracIn)": f(T["syn_L4P1_auprc"]),
                     "p (Wilcoxon)": f"{T['syn_L4P1_auprc'][1]:.1e}", "Δ F1 dinámico macro": f(T["syn_L4P1_D_mac"])})
    ct = num.get("cont", {}).get("tests", {}).get("Proy. objetivo − PD")
    if ct:
        for r, t in zip(num["cont"]["rate"], ct):
            rows.append({"Datos": f"Contaminación {r * 100:.1f} %", "Δ AUPRC (gana TracIn)": f(t), "p (Wilcoxon)": f"{t[1]:.1e}",
                         "Δ F1 dinámico macro": "—"})
    for ds in ["creditcard", "cicids", "gecco", "swan_sf"]:
        t = T.get(f"real_{ds}")
        if t:
            rows.append({"Datos": f"{ds} ({t[6]})", "Δ AUPRC (gana TracIn)": f(t), "p (Wilcoxon)": "—", "Δ F1 dinámico macro": "—"})
    if rows:
        write("rep_trayectoria", "¿Aporta la trayectoria? TracIn proyectado al objetivo − error final del mismo LSTM", pd.DataFrame(rows),
              "Misma proyección (pasos predichos) y mismo modelo: la única diferencia es medir la pérdida a lo largo del "
              "entrenamiento (self-influence) o sólo al final. Entre paréntesis, (dataset, semilla) en que gana TracIn. Configuración "
              "reproducida (datos estandarizados; reales con gradiente completo).")


if __name__ == "__main__":
    num = {}
    counts(num)
    screen(num)
    real_hyp(num)
    contamination(num)
    contam_fix(num)
    if (RES / "synth_base_eval.csv").exists():
        synthetic(num)
    real(num)
    trajectory(num)
    (RES / "replica_numbers.json").write_text(json.dumps(num, indent=1, default=float), encoding="utf-8")
