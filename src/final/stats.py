"""Tablas y pruebas de PROTOCOLO_FINAL.md a partir de results/final/*_eval.csv.

Salida: results/final/tables/*.md (una tabla por archivo) y
        results/final/final_numbers.json (todas las cifras, para el informe
        y para el notebook de Colab).
Uso:    python stats.py
"""
from __future__ import annotations

import json

from core import RES  # noqa: F401  (fija sys.path)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

TAB = RES / "tables"
RNG = np.random.default_rng(0)

# (nombre, entrenamiento, proyección, umbral)  — umbral None = sólo métricas sin umbral
METHODS = [
    ("Artículo · umbral estático", "base", "win", "static"),
    ("Artículo · umbral dinámico (usa etiquetas)", "base", "win", "dyn"),
    ("M1 · umbral IQR", "base", "win", "iqr"),
    ("M2 · reponderación suave", "soft_r2", "win", "static"),
    ("M2 · recorte z>2", "trim_r2", "win", "static"),
    ("M3 · proyección al objetivo", "base", "tgt", "static"),
    ("M1+M3", "base", "tgt", "iqr"),
    ("M2+M3 (recorte z>2)", "trim_r2", "tgt", "static"),
    ("M1+M2 (recorte adaptativo, sin M3)", "adapt_r3", "win", "iqr"),
    ("M1+M2+M3 con recorte z>2", "trim_r2", "tgt", "iqr"),
    ("**Método final** (recorte adaptativo + proyección + IQR)", "adapt_r3", "tgt", "iqr"),
    ("Error del LSTM (PD) + IQR", "pd", "tgt", "iqr"),
]
FINAL = METHODS[10][0]
PAPER_S, PAPER_D = METHODS[0][0], METHODS[1][0]
M3 = METHODS[5][0]
M13 = METHODS[6][0]
BLOCK_NAME = {"synth": "Sintéticos (15 datasets × 20 semillas, L4P1)",
              "contam": "Contaminación controlada (3 mezclas × 7 niveles × 20 semillas, L4P1)",
              "real": "Reales (4 datasets × semillas, L4P1)",
              "smd": "SMD (28 máquinas × 5 semillas, L4P1; unidad de prueba = máquina)",
              "nasa": "SMAP + MSL (82 canales × 5 semillas, L4P1; unidad de prueba = canal)",
              "semi": "Contaminación semisintética sobre fondo real SMD (10 máquinas × 3 mezclas × 5 niveles × 3 semillas)"}
BLOCKS = ("synth", "contam", "real", "smd", "nasa", "semi")
PER_SERIES = ("smd", "nasa")      # pruebas entre series (promedio sobre semillas)


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def fp(p):
    return "—" if p is None or not np.isfinite(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def keys(block):
    return {"contam": ["ds", "seed", "ratio"], "semi": ["ds", "mix", "ratio", "seed"]}.get(block, ["ds", "seed"])


def method_frame(ev: pd.DataFrame, block: str) -> pd.DataFrame:
    """Una fila por (unidad, método) con auprc, auroc, f1_bin, f1_mac."""
    e = ev[ev.cfg == "L4P1"]
    out = []
    for name, tr, pj, th in METHODS:
        x = e[(e.train == tr) & (e.proj == pj)]
        d = x[keys(block)].copy()
        d["method"] = name
        d["auprc"], d["auroc"] = x.auprc.values, x.auroc.values
        d["f1_bin"], d["f1_mac"] = x[f"{th}_bin"].values, x[f"{th}_mac"].values
        out.append(d)
    if (ev.cfg == "-").any():
        for nm, tr in (("LOF + IQR", "lof"), ("IForest + IQR", "iforest")):
            x = ev[ev.train == tr]
            d = x[keys(block)].copy()
            d["method"] = nm
            d["auprc"], d["auroc"], d["f1_bin"], d["f1_mac"] = x.auprc.values, x.auroc.values, x.iqr_bin.values, x.iqr_mac.values
            out.append(d)
    return pd.concat(out, ignore_index=True)


def per_series(mf):
    """Promedia sobre semillas: una fila por (serie, método); seed = 0."""
    g = mf.groupby(["ds", "method"], as_index=False)[["auprc", "auroc", "f1_bin", "f1_mac"]].mean()
    g["seed"] = 0
    return g


def paired(mf, block, a, b, metric):
    k = keys(block)
    A = mf[mf.method == a].set_index(k)[metric]
    B = mf[mf.method == b].set_index(k)[metric]
    j = A.index.intersection(B.index)
    d = (A.loc[j] - B.loc[j]).values
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(n=0)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else float("nan")
    ranks = pd.Series(np.abs(nz)).rank().values
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rb = float((wp - wm) / (wp + wm)) if wp + wm else 0.0
    boots = RNG.choice(d, size=(10000, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), p=p, rb=rb,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m = len(ps)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[idx]))
        adj[idx] = run
    return adj


def summary_table(mf, block):
    rows = []
    for name in mf.method.unique():
        x = mf[mf.method == name]
        dA = paired(mf, block, name, PAPER_S, "auprc") if name != PAPER_S else {}
        dF = paired(mf, block, name, PAPER_D, "f1_bin") if name != PAPER_D else {}
        rows.append({"Método": name,
                     "AUPRC": f"{x.auprc.mean():.3f} ± {x.auprc.std():.3f}",
                     "AUROC": fmt(x.auroc.mean()),
                     "F1 binario": f"{x.f1_bin.mean():.3f} ± {x.f1_bin.std():.3f}",
                     "F1 macro": fmt(x.f1_mac.mean()),
                     "Δ AUPRC vs artículo (p)": f"{dA['mean']:+.3f} ({fp(dA['p'])})" if dA else "—",
                     "Δ F1 bin vs artículo dinámico (p)": f"{dF['mean']:+.3f} ({fp(dF['p'])})" if dF else "—"})
    return pd.DataFrame(rows)


def hypotheses(mf, block, ev):
    tests = [("H1", "Final > Artículo", FINAL, PAPER_S, "auprc"),
             ("H2", "Final (sin etiquetas) ≥ Artículo dinámico (con etiquetas)", FINAL, PAPER_D, "f1_bin"),
             ("H3a", "Final > M3 (mismo score sin recorte)", FINAL, M13, "auprc"),
             ("H3b", "Final > M1+M3 (todo menos el recorte)", FINAL, M13, "f1_bin")]
    rows = []
    for h, desc, a, b, met in tests:
        r = paired(mf, block, a, b, met)
        rows.append(dict(h=h, desc=desc, metric=met, **r))
    if block == "contam":
        r = drop_test(ev)
        rows.append(dict(h="H4", desc="Caída de AUROC 1 %→30 % menor en Final que en Artículo", metric="auroc_drop", **r))
    if block in ("smd", "nasa", "semi"):        # extensión preregistrada: H5 en la misma familia
        rows += h5_rows(mf, block)
    if block == "semi":
        r = drop_test(ev, idx=["ds", "mix", "seed"])
        rows.append(dict(h="H4r", desc="Caída de AUROC 1 %→30 % menor en Final que en Artículo", metric="auroc_drop", **r))
        r = drop_test(ev, idx=["ds", "mix", "seed"], other=("base", "tgt"))
        rows.append(dict(h="H5c", desc="Caída de AUROC 1 %→30 % menor en M3 que en Artículo", metric="auroc_drop", **r))
    ps = [r["p"] for r in rows]
    for r, pa in zip(rows, holm(ps)):
        r["p_holm"] = float(pa)
    return rows


def h5_rows(mf, block):
    out = []
    for h, met in (("H5a", "auroc"), ("H5b", "auprc")):
        r = paired(mf, block, M3, PAPER_S, met)
        out.append(dict(h=h, desc="M3 (sólo proyección al objetivo) > Artículo", metric=met, **r))
    return out


def drop_test(ev, idx=("ds", "seed"), other=("adapt_r3", "tgt")):
    e = ev[ev.cfg == "L4P1"]
    idx = list(idx)
    def drop(tr, pj):
        x = e[(e.train == tr) & (e.proj == pj)]
        lo = x[x.ratio == x.ratio.min()].set_index(idx).auroc
        hi = x[x.ratio == x.ratio.max()].set_index(idx).auroc
        return lo - hi
    dp, df_ = drop("base", "win"), drop(*other)
    j = dp.index.intersection(df_.index)
    d = (dp.loc[j] - df_.loc[j]).values  # > 0: el final cae menos
    nz = d[d != 0]
    ranks = pd.Series(np.abs(nz)).rank().values
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    boots = RNG.choice(d, size=(10000, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), p=float(wilcoxon(nz).pvalue),
                rb=float((wp - wm) / (wp + wm)), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                drop_paper=float(dp.mean()), drop_final=float(df_.mean()))


def hyp_table(rows):
    t = []
    for r in rows:
        t.append({"Hipótesis": r["h"], "Comparación": r["desc"], "Métrica": r["metric"],
                  "Δ media [IC 95 %]": f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
                  "Gana / pierde": f"{r['wins']} / {r['losses']} de {r['n']}",
                  "r rank-biserial": f"{r['rb']:+.2f}", "p (Wilcoxon)": fp(r["p"]), "p Holm": fp(r["p_holm"]),
                  "¿Se sostiene?": ("sí" if r["p_holm"] < 0.05 and r["mean"] > 0 else
                                    ("no (peor)" if r["p_holm"] < 0.05 else "no significativa"))})
    return pd.DataFrame(t)


def factorial(ev, block):
    e = ev[ev.cfg == "L4P1"]
    trains = [("base", "Artículo"), ("soft_r2", "Suave z>2"), ("trim_r2", "Recorte z>2"), ("adapt_r3", "Recorte adaptativo IQR")]
    rows = []
    for tr, tn in trains:
        for pj, pn in (("win", "ventana (artículo)"), ("tgt", "objetivo (M3)")):
            x = e[(e.train == tr) & (e.proj == pj)]
            rows.append({"Entrenamiento": tn, "Proyección": pn, "AUPRC": fmt(x.auprc.mean()), "AUROC": fmt(x.auroc.mean()),
                         "F1 bin · estático": fmt(x.static_bin.mean()), "F1 bin · dinámico (etiquetas)": fmt(x.dyn_bin.mean()),
                         "F1 bin · IQR (M1)": fmt(x.iqr_bin.mean()),
                         "F1 macro · estático": fmt(x.static_mac.mean()), "F1 macro · dinámico": fmt(x.dyn_mac.mean()),
                         "F1 macro · IQR": fmt(x.iqr_mac.mean())})
    return pd.DataFrame(rows)


def per_dataset(mf, block, names):
    rows = []
    for ds in sorted(mf.ds.unique(), key=str):
        x = mf[mf.ds == ds]
        r = {"Dataset": ds}
        for nm, short in names:
            y_ = x[x.method == nm]
            r[f"{short} AUPRC"] = fmt(y_.auprc.mean())
            r[f"{short} F1"] = fmt(y_.f1_bin.mean())
        dA = paired(x, block, FINAL, PAPER_S, "auprc")
        dF = paired(x, block, FINAL, PAPER_D, "f1_bin")
        r["Final − Artículo AUPRC (gana/n)"] = f"{dA['mean']:+.3f} ({dA['wins']}/{dA['n']})"
        r["Final − Art. dinámico F1 (gana/n)"] = f"{dF['mean']:+.3f} ({dF['wins']}/{dF['n']})"
        rows.append(r)
    return pd.DataFrame(rows)


def contam_curves(mf, col="ds"):
    names = [PAPER_S, M3, "M2 · recorte z>2", FINAL, "Error del LSTM (PD) + IQR", "LOF + IQR", "IForest + IQR"]
    out = {}
    for mix in sorted(mf[col].unique()):
        x = mf[mf[col] == mix]
        rows = []
        for nm in names:
            r = {"Método": nm}
            for ra in sorted(x.ratio.unique()):
                r[f"{ra:.0%}"] = fmt(x[(x.method == nm) & (x.ratio == ra)].auroc.mean())
            g = x[x.method == nm]
            r["Caída"] = fmt(g[g.ratio == g.ratio.min()].auroc.mean() - g[g.ratio == g.ratio.max()].auroc.mean())
            rows.append(r)
        out[mix] = pd.DataFrame(rows)
    return out


def h6(evs):
    """H6: Spearman entre tasa de anomalías de la serie y Δ AUROC (M3 − Artículo), SMD + SMAP/MSL."""
    rows = []
    for b in PER_SERIES:
        if b not in evs:
            continue
        e = evs[b][evs[b].cfg == "L4P1"]
        a = e[(e.train == "base") & (e.proj == "tgt")].groupby("ds")[["auroc", "auprc", "anom_rate"]].mean()
        p = e[(e.train == "base") & (e.proj == "win")].groupby("ds")[["auroc", "auprc"]].mean()
        f = e[(e.train == "adapt_r3") & (e.proj == "tgt")].groupby("ds")[["auroc", "auprc"]].mean()
        d = pd.DataFrame({"block": b, "anom_rate": a.anom_rate, "d_auroc_m3": a.auroc - p.auroc,
                          "d_auprc_m3": a.auprc - p.auprc, "d_auroc_final": f.auroc - p.auroc})
        rows.append(d.reset_index())
    if not rows:
        return None, None
    d = pd.concat(rows, ignore_index=True)
    res = {}
    for k in ("d_auroc_m3", "d_auprc_m3", "d_auroc_final"):
        rho, p = spearmanr(d.anom_rate, d[k], alternative="greater")
        res[k] = dict(rho=float(rho), p=float(p), n=len(d))
    q = pd.qcut(d.anom_rate, 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (mayor)"])
    res["cuartiles"] = d.groupby(q, observed=True).agg(tasa_min=("anom_rate", "min"), tasa_max=("anom_rate", "max"),
                                                       d_auroc_m3=("d_auroc_m3", "mean"), d_auprc_m3=("d_auprc_m3", "mean"),
                                                       gana=("d_auroc_m3", lambda v: int((v > 0).sum())),
                                                       n=("d_auroc_m3", "size")).reset_index().to_dict("records")
    return d, res


def paper_protocol_synth(ev):
    """Artículo con SU protocolo: mejor (L, P) por dataset elegida con la
    métrica de prueba (media sobre semillas) + umbral dinámico. Frente al
    método final con L4P1 fijo y sin etiquetas."""
    b = ev[(ev.train == "base") & (ev.proj == "win")]
    best = b.groupby(["ds", "cfg"]).dyn_bin.mean().reset_index().sort_values("dyn_bin").groupby("ds").tail(1)
    bestA = b.groupby(["ds", "cfg"]).auprc.mean().reset_index().sort_values("auprc").groupby("ds").tail(1)
    pf = b.merge(best[["ds", "cfg"]], on=["ds", "cfg"]).set_index(["ds", "seed"])["dyn_bin"]
    pa = b.merge(bestA[["ds", "cfg"]], on=["ds", "cfg"]).set_index(["ds", "seed"])["auprc"]
    f = ev[(ev.train == "adapt_r3") & (ev.proj == "tgt") & (ev.cfg == "L4P1")].set_index(["ds", "seed"])
    res = {}
    for nm, A, B in (("f1_bin", f.iqr_bin, pf), ("auprc", f.auprc, pa)):
        d = (A - B.reindex(A.index)).dropna().values
        nz = d[d != 0]
        boots = RNG.choice(d, size=(10000, len(d)), replace=True).mean(1)
        res[nm] = dict(final=float(A.mean()), paper=float(B.mean()), mean=float(d.mean()),
                       lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                       p=float(wilcoxon(nz).pvalue), wins=int((d > 0).sum()), n=len(d))
    return res


def diag_table(block):
    p = RES / f"{block}_diag.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"ds": str, "mix": str})
    d = d[d.cfg == "L4P1"]
    rows = []
    for mode, nm in (("trim_r1", "Recorte z>2, ronda 1"), ("trim_r2", "Recorte z>2, ronda 2"),
                     ("adapt_r1", "Adaptativo IQR, ronda 1"), ("adapt_r3", "Adaptativo IQR, ronda 3")):
        g = {k: d[d.key == f"{mode}_{k}"].value.mean() for k in ("frac", "anom", "norm")}
        rows.append({"Regla": nm, "Ventanas recortadas": f"{g['frac']:.1%}", "De las anómalas": f"{g['anom']:.1%}",
                     "De las normales": f"{g['norm']:.1%}"})
    return pd.DataFrame(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    numbers = {}
    evs = {}
    for block in BLOCKS:
        p = RES / f"{block}_eval.csv"
        if not p.exists():
            continue
        ev = pd.read_csv(p, dtype={"ds": str, "mix": str})
        evs[block] = ev
        mf = method_frame(ev, block)
        mfh = per_series(mf) if block in PER_SERIES else mf      # unidad de las pruebas
        nb = numbers.setdefault(block, {})
        nb["n_units"] = int(mfh[mfh.method == FINAL].shape[0])
        nb["n_runs"] = int(mf[mf.method == FINAL].shape[0])
        nb["methods"] = {m: {k: float(mf[mf.method == m][k].mean()) for k in ("auprc", "auroc", "f1_bin", "f1_mac")}
                         for m in mf.method.unique()}
        st = summary_table(mfh, block)
        (TAB / f"{block}_resumen.md").write_text(f"### {BLOCK_NAME[block]} — todos los métodos\n\n" + md(st) + "\n")
        hy = hypotheses(mfh, block, ev)
        nb["hyp"] = hy
        (TAB / f"{block}_hipotesis.md").write_text(f"### {BLOCK_NAME[block]} — hipótesis primarias\n\n" + md(hyp_table(hy)) + "\n")
        if block in ("synth", "contam", "real"):     # H5 fuera de la familia original (reporte)
            h5 = h5_rows(mfh, block)
            for r, pa in zip(h5, holm([r["p"] for r in h5])):
                r["p_holm"] = float(pa)
            nb["hyp_h5"] = h5
            (TAB / f"{block}_h5.md").write_text(f"### {BLOCK_NAME[block]} — H5 (M3 sola vs. artículo; familia aparte)\n\n" + md(hyp_table(h5)) + "\n")
        (TAB / f"{block}_factorial.md").write_text(f"### {BLOCK_NAME[block]} — diseño factorial (medias)\n\n" + md(factorial(ev, block)) + "\n")
        names = [(PAPER_S, "Art."), (M3, "M3"), (FINAL, "Final")]
        if block not in ("contam", "semi"):
            (TAB / f"{block}_por_dataset.md").write_text(f"### {BLOCK_NAME[block]} — por dataset\n\n" + md(per_dataset(mf, block, names)) + "\n")
        dt = diag_table(block)
        if dt is not None:
            (TAB / f"{block}_recorte.md").write_text(f"### {BLOCK_NAME[block]} — qué recorta cada regla (diagnóstico con etiquetas)\n\n" + md(dt) + "\n")
        if block in ("contam", "semi"):
            col = "ds" if block == "contam" else "mix"
            cc = contam_curves(mf, col)
            txt = ""
            for mix, t in cc.items():
                txt += f"### {BLOCK_NAME[block]}, mezcla «{mix}» — AUROC vs. proporción nominal\n\n" + md(t) + "\n\n"
            (TAB / f"{block}_curvas.md").write_text(txt)
            nb["curves"] = {mix: t.to_dict("records") for mix, t in cc.items()}
            nb["anom_rate"] = ev.groupby([col, "ratio"]).anom_rate.mean().reset_index().to_dict("records")
        if block == "synth":
            nb["paper_protocol"] = paper_protocol_synth(ev)
        if block in ("real", "smd", "nasa"):
            nb["per_ds"] = {ds: {m: {k: float(mf[(mf.ds == ds) & (mf.method == m)][k].mean()) for k in ("auprc", "auroc", "f1_bin")}
                                 for m in mf.method.unique()} for ds in mf.ds.unique()}
            nb["seeds"] = mf[mf.method == FINAL].groupby("ds").size().to_dict()
        print(block, "ok")
    d6, r6 = h6(evs)
    if r6 is not None:
        numbers["h6"] = r6
        d6.to_csv(RES / "h6_series.csv", index=False)
        t = pd.DataFrame(r6["cuartiles"])
        (TAB / "h6.md").write_text("### H6 — ¿la ventaja de M3 crece con la tasa natural de anomalías? (SMD + SMAP/MSL)\n\n"
                                   + "\n".join(f"* {k}: ρ = {v['rho']:+.3f}, p (unilateral) = {fp(v['p'])}, n = {v['n']}"
                                                for k, v in r6.items() if k != "cuartiles") + "\n\n" + md(t.round(4)) + "\n")
    (RES / "final_numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
