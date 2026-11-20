"""Análisis y salidas para el artículo (secundarios; ver «Desviaciones» en
PROTOCOLO_FINAL.md). No cambia ninguna hipótesis primaria: sólo agrega
(a) rankings de Friedman / Nemenyi, (b) pruebas centradas en la mejora 3
(proyección al objetivo) bajo contaminación y en reales, (c) el mecanismo por
el que la proyección al objetivo resiste la contaminación («halo» de falsos
positivos alrededor de cada anomalía) y (d) tablas LaTeX + figuras PDF.

Entradas: results/final/{synth,contam,real}_eval.csv y los .npz.
Salidas:  results/final/paper/{tablas/*.tex, figs/*.pdf, numeros.json}
Uso:      python paper.py [--workers 20]
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from core import RES  # noqa: E402  (fija sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

from stats import FINAL, M3, PAPER_D, PAPER_S, PER_SERIES, holm, method_frame, paired, per_series  # noqa: E402

OUT = RES / "paper"
TAB, FIG = OUT / "tablas", OUT / "figs"
RNG = np.random.default_rng(1)
L, P = 4, 1
SPAN = L + P

# nombres cortos para tablas/figuras
SHORT = {
    PAPER_S: "Artículo (estático)",
    PAPER_D: "Artículo (dinámico, etiquetas)",
    "M1 · umbral IQR": "M1",
    "M2 · reponderación suave": "M2 suave",
    "M2 · recorte z>2": "M2 recorte",
    M3: "M3",
    "M1+M3": "M1+M3",
    "M2+M3 (recorte z>2)": "M2+M3",
    "M1+M2 (recorte adaptativo, sin M3)": "M1+M2",
    "M1+M2+M3 con recorte z>2": "M1+M2+M3 (z>2)",
    FINAL: "M1+M2+M3 (final)",
    "Error del LSTM (PD) + IQR": "Error LSTM + IQR",
    "LOF + IQR": "LOF + IQR",
    "IForest + IQR": "IForest + IQR",
}
ORDER = list(SHORT)
RANK_SET = [PAPER_S, PAPER_D, "M1 · umbral IQR", "M2 · recorte z>2", M3, "M1+M3", "M2+M3 (recorte z>2)",
            "M1+M2 (recorte adaptativo, sin M3)", FINAL, "Error del LSTM (PD) + IQR"]
# valores críticos q_0.05 de Nemenyi (Demšar, 2006), k = 2..12
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
DSNAME = {"creditcard": "Credit Card", "cicids": "CICIDS", "gecco": "GECCO", "swan_sf": "SWAN-SF"}
MIXNAME = {"01234": "Todas (01234)", "01": "Puntuales (01)", "234": "De patrón (234)"}
BNAME = {"synth": "Sintéticos", "contam": "Contaminación", "real": "Reales (4)", "smd": "SMD", "nasa": "SMAP+MSL",
         "semi": "Semisintético"}
MIXCOL = {"contam": "ds", "semi": "mix"}
UNIT = {"contam": ["ds", "seed"], "semi": ["ds", "mix", "seed"]}


def load(block):
    p = RES / f"{block}_eval.csv"
    return pd.read_csv(p, dtype={"ds": str, "mix": str}) if p.exists() else None


def tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace(">", r"$>$")


def f3(x, d=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pstr(p):
    if p is None or not np.isfinite(p):
        return "--"
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return rf"$<10^{{{e + 1}}}$"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def write_tab(name, body):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(body)


# ---------------------------------------------------------------------------
# (a) Friedman + rango medio + diferencia crítica de Nemenyi
# ---------------------------------------------------------------------------
def friedman(mf, unit_cols, metric, methods):
    g = mf[mf.method.isin(methods)].groupby(unit_cols + ["method"])[metric].mean().unstack("method")[methods].dropna()
    R = g.rank(axis=1, ascending=False)  # 1 = mejor
    stat, p = friedmanchisquare(*[g[m].values for m in methods])
    k, n = len(methods), len(g)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    return dict(n=n, k=k, chi2=float(stat), p=float(p), cd=float(cd), ranks=R.mean().to_dict())


def cd_plot(res, title, path):
    import matplotlib.pyplot as plt
    ranks = sorted(res["ranks"].items(), key=lambda t: t[1])
    k = len(ranks)
    rs = [r for _, r in ranks]
    cliques = []
    for i in range(k):
        j = max(j for j in range(i, k) if rs[j] - rs[i] <= res["cd"])
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    half = int(np.ceil(k / 2))
    y0 = -0.35 - 0.16 * len(cliques)            # primera fila de etiquetas
    nrows = max(half, k - half)
    fig, ax = plt.subplots(figsize=(7.0, 1.0 + 0.24 * nrows + 0.1 * len(cliques)))
    lo, hi = 1, k
    ax.set_xlim(lo - 3.2, hi + 3.2)
    ax.set_ylim(y0 - 0.32 * nrows, 1.0)
    ax.hlines(0, lo, hi, color="k", lw=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.1, color="k", lw=0.8)
        ax.text(t, 0.15, str(t), ha="center", va="bottom", fontsize=7)
    ax.hlines(0.7, lo, lo + res["cd"], color="k", lw=1.5)
    ax.vlines([lo, lo + res["cd"]], 0.65, 0.75, color="k", lw=1)
    ax.text(lo + res["cd"] / 2, 0.78, f"DC = {res['cd']:.2f}", ha="center", va="bottom", fontsize=7)
    for i, (m, r) in enumerate(ranks):
        left = i < half
        y = y0 - 0.32 * (i if left else k - 1 - i)
        x_end = lo - 0.4 if left else hi + 0.4
        c = "#c0392b" if m == FINAL else "k"
        ax.plot([r, r, x_end], [0, y, y], color=c, lw=0.8, zorder=1)
        ax.text(x_end + (-0.08 if left else 0.08), y, f"{SHORT[m]} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7, fontweight="bold" if m == FINAL else None, color=c)
    for q, (a, b) in enumerate(cliques):
        yb = -0.15 - 0.16 * q
        ax.hlines(yb, rs[a] - 0.06, rs[b] + 0.06, color="#444", lw=3, zorder=3)
    ax.axis("off")
    ax.set_title(title, fontsize=8)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) mecanismo: halo de la proyección y perfil alrededor de anomalías
# ---------------------------------------------------------------------------
def _dist_to_anom(y):
    n = len(y)
    idx = np.flatnonzero(y)
    if len(idx) == 0:
        return np.full(n, np.inf)
    pos = np.searchsorted(idx, np.arange(n))
    left = np.abs(np.arange(n) - idx[np.clip(pos - 1, 0, len(idx) - 1)])
    right = np.abs(idx[np.clip(pos, 0, len(idx) - 1)] - np.arange(n))
    return np.minimum(left, right).astype(float)


def halo_job(args):
    block, f = args
    from common import project_sum, project_targets_mean, valid_mask
    from sklearn.metrics import roc_auc_score
    z = np.load(f)
    pre = "L4P1|" if block == "synth" else ""
    y = z["y"].astype(bool)
    n = len(y)
    m = valid_mask(n)
    d = _dist_to_anom(y)
    halo = (~y) & (d >= 1) & (d <= SPAN - 1) & m
    far = (~y) & (d >= SPAN) & m
    anom = y & m
    stem = f.stem
    if block == "contam":
        mix, rr, sd = stem.split("_")
        meta = dict(ds=mix, ratio=float(rr[1:]), seed=int(sd[1:]))
    elif block == "semi":
        mach, mix, rr, sd = stem.split("_")
        meta = dict(ds=mach, mix=mix, ratio=float(rr[1:]), seed=int(sd[1:]))
    else:
        ds, sd = stem.rsplit("_s", 1)
        meta = dict(ds=ds, seed=int(sd))
    meta.update(anom_rate=float(anom.sum() / m.sum()), halo_frac=float(halo.sum() / max((~y & m).sum(), 1)))
    rows, prof = [], {}
    pos_curve = {}
    start = y & ~np.r_[False, y[:-1]]
    sid = np.cumsum(start)                                   # id de evento
    st_idx = np.r_[0, np.flatnonzero(start)]
    offset = np.where(y, np.arange(n) - st_idx[sid], -1)     # posición dentro del evento
    elen = np.bincount(sid[y], minlength=sid.max() + 1) if y.any() else np.zeros(1, int)
    long_ev = y & (elen[sid] >= 8)                           # eventos colectivos (≥ 8 pasos)
    iso = np.flatnonzero(anom & (np.r_[np.zeros(1, bool), y[:-1]] == 0))  # inicios de evento
    iso = iso[(iso > 30) & (iso < n - 30)]
    for tr in ("base", "adapt_r3", "pd"):
        if pre + tr not in z.files:
            continue
        s = z[pre + tr].astype(np.float64)
        for pj, sc in (("win", project_sum(s, SPAN, n)), ("tgt", project_targets_mean(s, L, P, n))):
            r = dict(**meta, train=tr, proj=pj)
            if anom.any() and halo.any() and far.any():
                yy = np.r_[np.ones(anom.sum()), np.zeros(halo.sum())]
                r["auroc_anom_vs_halo"] = float(roc_auc_score(yy, np.r_[sc[anom], sc[halo]]))
                yy = np.r_[np.ones(anom.sum()), np.zeros(far.sum())]
                r["auroc_anom_vs_far"] = float(roc_auc_score(yy, np.r_[sc[anom], sc[far]]))
                k = int(anom.sum())                       # top-k con k = nº de anomalías
                top = np.argsort(-np.where(m, sc, -np.inf))[:k]
                r["topk_halo_share"] = float(halo[top].mean())
                r["topk_anom_share"] = float(anom[top].mean())
            if long_ev.any():
                pr = np.empty(n)
                pr[m] = (np.argsort(np.argsort(sc[m])) + 0.5) / m.sum()   # percentil del score
                pos_curve[f"{tr}|{pj}"] = [float(pr[long_ev & m & (offset == k)].mean()) if (long_ev & m & (offset == k)).any()
                                           else None for k in range(12)]
            rows.append(r)
            if len(iso):
                med = np.median(sc[m]); iqr = np.subtract(*np.percentile(sc[m], [75, 25])) or 1.0
                zz = (sc - med) / iqr
                lags = np.arange(-12, 13)
                prof[f"{tr}|{pj}"] = zz[iso[:, None] + lags[None, :]].mean(0).tolist()
    return rows, dict(**meta, prof=prof, pos=pos_curve)


def run_halo(block, workers):
    files = sorted(f for f in (RES / block).glob("*.npz") if ".tmp" not in f.name)
    import multiprocessing as mp
    rows, profs = [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for r, p in pool.imap_unordered(halo_job, [(block, f) for f in files]):
            rows += r
            profs.append(p)
    pd.DataFrame(rows).to_csv(OUT / f"halo_{block}.csv", index=False)
    (OUT / f"perfil_{block}.json").write_text(json.dumps(profs))
    return pd.DataFrame(rows), profs


# ---------------------------------------------------------------------------
# (c) pruebas centradas en M3
# ---------------------------------------------------------------------------
def test_vec(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else float("nan")
    boots = RNG.choice(d, size=(10000, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ranks = pd.Series(np.abs(nz)).rank().values
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), p=p,
                rb=float((wp - wm) / (wp + wm)) if wp + wm else 0.0, wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def contam_m3(mf, col="ds", unit=("seed",)):
    """Por mezcla y nivel: M3 − Artículo y Final − Artículo (AUROC, AUPRC, F1).
    Holm sobre las 21 celdas de cada comparación/métrica."""
    out = []
    for a, lab in ((M3, "M3"), (FINAL, "Final"), ("M2 · recorte z>2", "M2"), ("M1+M2 (recorte adaptativo, sin M3)", "M1+M2")):
        for met in ("auroc", "auprc", "f1_bin"):
            cells = []
            for mix in sorted(mf[col].unique()):
                for r in sorted(mf.ratio.unique()):
                    x = mf[(mf[col] == mix) & (mf.ratio == r)]
                    A = x[x.method == a].set_index(list(unit))[met]
                    B = x[x.method == (PAPER_D if met == "f1_bin" else PAPER_S)].set_index(list(unit))[met]
                    j = A.index.intersection(B.index)
                    cells.append(dict(cmp=lab, metric=met, mix=mix, ratio=r, **test_vec(A[j] - B[j])))
            for c, pa in zip(cells, holm([c["p"] for c in cells])):
                c["p_holm"] = float(pa)
            out += cells
    return pd.DataFrame(out)


def drop_tests(ev, idx=("ds", "seed"), col="ds"):
    """Caída de AUROC 1 %→30 % por método, y diferencias de caída frente al artículo."""
    e = ev[ev.cfg == "L4P1"]
    combos = {"Artículo": ("base", "win"), "M2 (recorte z>2)": ("trim_r2", "win"), "M1+M2": ("adapt_r3", "win"),
              "M3": ("base", "tgt"), "M2+M3": ("trim_r2", "tgt"), "Final": ("adapt_r3", "tgt"), "Error LSTM": ("pd", "tgt")}
    drops = {}
    for nm, (tr, pj) in combos.items():
        x = e[(e.train == tr) & (e.proj == pj)]
        lo = x[x.ratio == x.ratio.min()].set_index(list(idx)).auroc
        hi = x[x.ratio == x.ratio.max()].set_index(list(idx)).auroc
        drops[nm] = lo - hi
    for nm, tr in (("LOF", "lof"), ("IForest", "iforest")):
        x = ev[ev.train == tr]
        drops[nm] = (x[x.ratio == x.ratio.min()].set_index(list(idx)).auroc
                     - x[x.ratio == x.ratio.max()].set_index(list(idx)).auroc)
    rows = []
    for nm, d in drops.items():
        r = dict(method=nm, drop=float(d.mean()), drop_lo=float(np.percentile(RNG.choice(d.values, (10000, len(d))).mean(1), 2.5)),
                 drop_hi=float(np.percentile(RNG.choice(d.values, (10000, len(d))).mean(1), 97.5)))
        lv = list(idx).index(col)
        for mix in sorted(d.index.get_level_values(lv).unique()):
            r[f"drop_{mix}"] = float(d[d.index.get_level_values(lv) == mix].mean())
        if nm != "Artículo":
            j = d.index.intersection(drops["Artículo"].index)
            t = test_vec(drops["Artículo"].loc[j] - d.loc[j])   # > 0: cae menos que el artículo
            r.update({f"vs_{k}": v for k, v in t.items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    m = df.vs_p.notna()
    df.loc[m, "vs_p_holm"] = holm(df.loc[m, "vs_p"].values)
    return df


def real_tests(mf):
    """Por dataset real (n = semillas): método − artículo, Wilcoxon exacto + signo."""
    out = []
    for a in (M3, FINAL, "M1+M3", "M2 · recorte z>2", "M1+M2 (recorte adaptativo, sin M3)", "M1 · umbral IQR"):
        for met, ref in (("auprc", PAPER_S), ("auroc", PAPER_S), ("f1_bin", PAPER_D)):
            for ds in sorted(mf.ds.unique()):
                x = mf[mf.ds == ds]
                A = x[x.method == a].set_index("seed")[met]
                B = x[x.method == ref].set_index("seed")[met]
                j = A.index.intersection(B.index)
                out.append(dict(cmp=SHORT[a], metric=met, ds=ds, a=float(A[j].mean()), b=float(B[j].mean()),
                                **test_vec(A[j] - B[j])))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# (d) tablas LaTeX
# ---------------------------------------------------------------------------
def tab_main(mf, block, caption, label, ref_auprc=PAPER_S, ref_f1=PAPER_D):
    names = [m for m in ORDER if m in set(mf.method)]
    best = {k: max(mf[mf.method == m][k].mean() for m in names) for k in ("auprc", "auroc", "f1_bin", "f1_mac")}
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Método & Etiq. & AUPRC & AUROC & F1 bin. & F1 macro & $\Delta$AUPRC / $\Delta$F1 \\", r"\midrule"]
    for m in names:
        x = mf[mf.method == m]
        cells = []
        for k in ("auprc", "auroc", "f1_bin", "f1_mac"):
            v = x[k].mean()
            s = f"{v:.3f}" + (rf"{{\scriptsize$\pm${x[k].std():.2f}}}" if k in ("auprc", "f1_bin") else "")
            cells.append(rf"\textbf{{{s}}}" if abs(v - best[k]) < 5e-4 else s)
        dA = paired(mf, block, m, ref_auprc, "auprc") if m != ref_auprc else None
        dF = paired(mf, block, m, ref_f1, "f1_bin") if m != ref_f1 else None
        def star(d):
            if not d or d.get("n", 0) == 0 or not np.isfinite(d.get("p", np.nan)):
                return "--"
            st = "***" if d["p"] < 1e-3 else "**" if d["p"] < 1e-2 else "*" if d["p"] < 0.05 else ""
            return f"{d['mean']:+.3f}{st}" if abs(d["mean"]) >= 5e-4 or st else "0"
        lab = "sí" if m == PAPER_D else "no"
        name = tex_escape(SHORT[m])
        if m == FINAL:
            name = rf"\textbf{{{name}}}"
        lines.append(f"{name} & {lab} & " + " & ".join(cells) + f" & {star(dA)} / {star(dF)}" + r" \\")
        if m in (PAPER_D, "M1+M2 (recorte adaptativo, sin M3)", FINAL):
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_hyp(hyps, caption, label):
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Datos & Hip. & Métrica & $\Delta$ media [IC 95\%] & G/P & $r_{rb}$ & $p_{\text{Holm}}$ & ¿Se sostiene? \\",
             r"\midrule"]
    met = {"auprc": "AUPRC", "auroc": "AUROC", "f1_bin": "F1 bin.", "auroc_drop": "Caída AUROC"}
    for blk, rows in hyps:
        for i, r in enumerate(rows):
            ok = "sí" if r["p_holm"] < 0.05 and r["mean"] > 0 else ("no (peor)" if r["p_holm"] < 0.05 else "no sig.")
            lines.append((blk if i == 0 else "") + f" & {r['h']} & {met[r['metric']]} & "
                         rf"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] & {r['wins']}/{r['losses']} & {r['rb']:+.2f} & "
                         f"{pstr(r['p_holm'])} & {ok}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_factorial(ev, caption, label):
    e = ev[ev.cfg == "L4P1"]
    trains = [("base", "Ninguno (artículo)"), ("soft_r2", "Suave $z>2$"), ("trim_r2", "Recorte $z>2$"), ("adapt_r3", "Recorte IQR (M1+M2)")]
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"& & & & \multicolumn{3}{c}{F1 binario según umbral} \\ \cmidrule(lr){5-7}",
             r"Entrenamiento & Proyección & AUPRC & AUROC & Estático & Dinámico$^\dagger$ & IQR (M1) \\", r"\midrule"]
    for tr, tn in trains:
        for pj, pn in (("win", "ventana"), ("tgt", "objetivo (M3)")):
            x = e[(e.train == tr) & (e.proj == pj)]
            lines.append(f"{tn if pj == 'win' else ''} & {pn} & {f3(x.auprc.mean())} & {f3(x.auroc.mean())} & "
                         f"{f3(x.static_bin.mean())} & {f3(x.dyn_bin.mean())} & {f3(x.iqr_bin.mean())}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\\[2pt]{\footnotesize $^\dagger$ umbral elegido con etiquetas en la primera mitad (protocolo del artículo).}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_contam(mf, drops, caption, label):
    show = [(PAPER_S, "Artículo"), ("M2 · recorte z>2", "M2 recorte"), ("M1+M2 (recorte adaptativo, sin M3)", "M1+M2"),
            (M3, "M3"), ("M2+M3 (recorte z>2)", "M2+M3"), (FINAL, "Final (M1+M2+M3)"),
            ("Error del LSTM (PD) + IQR", "Error LSTM"), ("LOF + IQR", "LOF"), ("IForest + IQR", "IForest")]
    dmap = {"Artículo": PAPER_S, "M2 (recorte z>2)": "M2 · recorte z>2", "M1+M2": "M1+M2 (recorte adaptativo, sin M3)",
            "M3": M3, "M2+M3": "M2+M3 (recorte z>2)", "Final": FINAL, "Error LSTM": "Error del LSTM (PD) + IQR",
            "LOF": "LOF + IQR", "IForest": "IForest + IQR"}
    inv = {v: k for k, v in dmap.items()}
    ratios = sorted(mf.ratio.unique())
    lines = [r"\begin{table*}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * len(ratios) + "cc}", r"\toprule",
             r"& \multicolumn{" + str(len(ratios)) + r"}{c}{AUROC por proporción nominal de anomalías} & & \\ \cmidrule(lr){2-" + str(len(ratios) + 1) + "}",
             "Método & " + " & ".join(f"{100 * r:.0f}\\%" for r in ratios) + r" & Caída & vs.\ artículo \\", r"\midrule"]
    for m, nm in show:
        x = mf[mf.method == m]
        vals = [x[x.ratio == r].auroc.mean() for r in ratios]
        best = [max(mf[(mf.method == mm) & (mf.ratio == r)].auroc.mean() for mm, _ in show) for r in ratios]
        cells = [rf"\textbf{{{v:.3f}}}" if abs(v - b) < 5e-4 else f"{v:.3f}" for v, b in zip(vals, best)]
        dr = drops[drops.method == inv[m]].iloc[0]
        cmp = "--" if inv[m] == "Artículo" else (f"{dr.vs_mean:+.3f}" + ("*" if dr.vs_p_holm < 0.05 else ""))
        lines.append((rf"\textbf{{{nm}}}" if m == FINAL else nm) + " & " + " & ".join(cells) +
                     f" & {dr['drop']:.3f} & {cmp}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def tab_real(mf, rt, caption, label):
    show = [PAPER_S, PAPER_D, "M1 · umbral IQR", "M2 · recorte z>2", M3, "M1+M3", "M1+M2 (recorte adaptativo, sin M3)", FINAL,
            "Error del LSTM (PD) + IQR"]
    dss = [d for d in ("creditcard", "cicids", "gecco", "swan_sf") if d in set(mf.ds)]
    lines = [r"\begin{table*}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "cc" * len(dss) + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSNAME[d]}}}" for d in dss) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(dss))),
             "Método & " + " & ".join(["AUPRC & F1 bin."] * len(dss)) + r" \\", r"\midrule"]
    for m in show:
        cells = []
        for d in dss:
            x = mf[(mf.ds == d) & (mf.method == m)]
            for met, ref in (("auprc", PAPER_S), ("f1_bin", PAPER_D)):
                v = x[met].mean()
                bst = max(mf[(mf.ds == d) & (mf.method == mm)][met].mean() for mm in show)
                s = f"{v:.3f}"
                if m != ref:
                    t = rt[(rt.cmp == SHORT.get(m)) & (rt.metric == met) & (rt.ds == d)]
                    if len(t) and t.iloc[0].p < 0.05:
                        s += r"$^{\uparrow}$" if t.iloc[0]["mean"] > 0 else r"$^{\downarrow}$"
                cells.append(rf"\textbf{{{s}}}" if abs(v - bst) < 5e-4 else s)
        nm = tex_escape(SHORT[m])
        lines.append((rf"\textbf{{{nm}}}" if m == FINAL else nm) + " & " + " & ".join(cells) + r" \\")
        if m in (PAPER_D, "M1+M2 (recorte adaptativo, sin M3)", FINAL):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def tab_halo(h, caption, label):
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"Datos & Proyección & \makecell{AUROC\\anom.\ vs.\ halo} & \makecell{AUROC\\anom.\ vs.\ lejanos} & "
             r"\makecell{Top-$k$\\en halo} & \makecell{Top-$k$\\anómalos} \\", r"\midrule"]
    for blk, d in h:
        x = d[d.train == "base"]
        for i, (pj, pn) in enumerate((("win", "ventana (artículo)"), ("tgt", "objetivo (M3)"))):
            y = x[x.proj == pj]
            lines.append((blk if i == 0 else "") + f" & {pn} & {f3(y.auroc_anom_vs_halo.mean())} & {f3(y.auroc_anom_vs_far.mean())} & "
                         f"{100 * y.topk_halo_share.mean():.1f}\\% & {100 * y.topk_anom_share.mean():.1f}\\%" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_bench(mfhs, caption, label):
    """SMD y SMAP/MSL: medias por serie; marcas = Wilcoxon entre series frente al artículo."""
    show = [PAPER_S, PAPER_D, "M1 · umbral IQR", "M2 · recorte z>2", M3, "M1+M3", "M1+M2 (recorte adaptativo, sin M3)", FINAL,
            "Error del LSTM (PD) + IQR", "LOF + IQR", "IForest + IQR"]
    bl = [b for b in ("smd", "nasa") if b in mfhs]
    lines = [r"\begin{table*}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "ccc" * len(bl) + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{BNAME[b]} ($n={mfhs[b].ds.nunique()}$ series)}}" for b in bl) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(bl))),
             "Método & " + " & ".join(["AUPRC & AUROC & F1 bin."] * len(bl)) + r" \\", r"\midrule"]
    for m in show:
        cells = []
        for b in bl:
            mf = mfhs[b]
            for met, ref in (("auprc", PAPER_S), ("auroc", PAPER_S), ("f1_bin", PAPER_D)):
                x = mf[mf.method == m]
                if not len(x):
                    cells.append("--")
                    continue
                v = x[met].mean()
                bst = max(mf[mf.method == mm][met].mean() for mm in show if (mf.method == mm).any())
                txt = f"{v:.3f}"
                if m != ref:
                    d = paired(mf, b, m, ref, met)
                    if d.get("n", 0) and np.isfinite(d["p"]) and d["p"] < 0.05:
                        txt += r"$^{\uparrow}$" if d["mean"] > 0 else r"$^{\downarrow}$"
                cells.append(rf"\textbf{{{txt}}}" if abs(v - bst) < 5e-4 else txt)
        nm = tex_escape(SHORT[m])
        lines.append((rf"\textbf{{{nm}}}" if m == FINAL else nm) + " & " + " & ".join(cells) + r" \\")
        if m in (PAPER_D, "M1+M2 (recorte adaptativo, sin M3)", FINAL):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------
def figs(mfs, evs, halos, profs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "font.family": "serif"})
    FIG.mkdir(parents=True, exist_ok=True)
    C = {PAPER_S: ("Artículo", "#7f7f7f", "-"), "M2 · recorte z>2": ("M2 (recorte)", "#9467bd", "--"),
         "M1+M2 (recorte adaptativo, sin M3)": ("M1+M2", "#8c564b", "--"), M3: ("M3", "#1f77b4", "-"),
         FINAL: ("M1+M2+M3", "#c0392b", "-"), "Error del LSTM (PD) + IQR": ("Error LSTM", "#2ca02c", ":"),
         "LOF + IQR": ("LOF", "#ff7f0e", ":"), "IForest + IQR": ("IForest", "#bcbd22", ":")}
    # 1) contaminación (sintética y sobre fondo real): AUROC y AUPRC vs. tasa real, por mezcla
    for blk, fname in (("contam", "contaminacion.pdf"), ("semi", "contaminacion_semi.pdf")):
        if blk not in mfs:
            continue
        mf, ev, col = mfs[blk], evs[blk], MIXCOL[blk]
        rates = ev.groupby([col, "ratio"]).anom_rate.mean()
        mixes = [m for m in ("01234", "01", "234") if m in set(mf[col])]
        fig, axes = plt.subplots(2, len(mixes), figsize=(7.2, 4.4), sharex="col", squeeze=False)
        for j, mix in enumerate(mixes):
            for i, met in enumerate(("auroc", "auprc")):
                ax = axes[i, j]
                for m, (lab, c, ls) in C.items():
                    g = mf[(mf[col] == mix) & (mf.method == m)].groupby("ratio")[met]
                    if not len(g):
                        continue
                    x = [100 * rates[(mix, r)] for r in g.mean().index]
                    mu, se = g.mean().values, 1.96 * g.sem().fillna(0).values
                    ax.plot(x, mu, ls, color=c, lw=2 if m == FINAL else 1.1, marker="o", ms=2.2, label=lab)
                    ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.12, lw=0)
                if i == 0:
                    ax.set_title(MIXNAME[mix])
                if i == 1:
                    ax.set_xlabel("% real de instantes anómalos")
                if j == 0:
                    ax.set_ylabel(met.upper())
        axes[0, 0].legend(fontsize=6, ncol=2, frameon=False)
        plt.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)
    # 1b) H6: ventaja de M3 frente a la tasa natural de anomalías (SMD + SMAP/MSL)
    p6 = RES / "h6_series.csv"
    if p6.exists():
        d = pd.read_csv(p6)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
        for ax, (k, tl) in zip(axes, (("d_auroc_m3", "$\\Delta$AUROC (M3 $-$ artículo)"),
                                      ("d_auprc_m3", "$\\Delta$AUPRC (M3 $-$ artículo)"))):
            for b, c, mk in (("smd", "#1f77b4", "o"), ("nasa", "#c0392b", "s")):
                x = d[d.block == b]
                ax.scatter(100 * x.anom_rate, x[k], s=10, color=c, marker=mk, alpha=0.7, lw=0,
                           label="SMD" if b == "smd" else "SMAP/MSL")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_xlabel("% de instantes anómalos de la serie (escala log)")
            ax.set_title(tl)
        axes[0].legend(fontsize=6, frameon=False)
        plt.tight_layout()
        fig.savefig(FIG / "h6_tasa.pdf")
        plt.close(fig)
    # 2) mecanismo (2 × 2)
    if "contam" in profs:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
        ax = axes[0, 0]
        lags = np.arange(-12, 13)
        sel = [p for p in profs["contam"] if p["ds"] == "01" and p["ratio"] <= 0.05 and p["prof"]]
        for key, lab, c in (("base|win", "ventana (artículo)", "#7f7f7f"), ("base|tgt", "objetivo (M3)", "#1f77b4"),
                            ("adapt_r3|tgt", "M1+M2+M3", "#c0392b")):
            arr = np.array([p["prof"][key] for p in sel if key in p["prof"]])
            if len(arr):
                ax.plot(lags, arr.mean(0), color=c, marker="o", ms=2, lw=1.2, label=lab)
        ax.axvspan(-0.5, 0.5, color="#f4c7c3", alpha=0.6, lw=0)
        ax.set_xlabel("desfase respecto a la anomalía puntual (pasos)")
        ax.set_ylabel("score (z robusto)")
        ax.set_title("(a) Respuesta a una anomalía puntual aislada", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        h = halos["contam"]
        hb = h[h.train == "base"]
        ax = axes[0, 1]
        for mix, ls in (("01", "-"), ("234", "--")):
            for pj, lab, c in (("win", "ventana", "#7f7f7f"), ("tgt", "objetivo (M3)", "#1f77b4")):
                g = hb[(hb.proj == pj) & (hb.ds == mix)].groupby("ratio")
                ax.plot(100 * g.anom_rate.mean().values, 100 * g.topk_halo_share.mean().values, ls, color=c, marker="o",
                        ms=2.2, label=f"{lab}, {MIXNAME[mix].lower()}")
        ax.set_xlabel("% real de instantes anómalos"); ax.set_ylabel("% del top-$k$")
        ax.set_title("(b) Top-$k$ ocupado por normales vecinas (halo)", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False, loc="center right")
        ax = axes[1, 0]
        for col, ls, nm in (("auroc_anom_vs_halo", "-", "vs. vecinos"), ("auroc_anom_vs_far", "--", "vs. lejanos")):
            for pj, lab, c in (("win", "ventana", "#7f7f7f"), ("tgt", "objetivo (M3)", "#1f77b4")):
                g = hb[(hb.proj == pj) & (hb.ds == "234")].groupby("ratio")
                ax.plot(100 * g.anom_rate.mean().values, g[col].mean().values, ls, color=c, marker="o", ms=2.2,
                        label=f"{lab}, {nm}")
        ax.set_xlabel("% real de instantes anómalos"); ax.set_ylabel("AUROC")
        ax.set_title("(c) Anomalías de patrón: separación de vecinos y de lejanos", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False)
        ax = axes[1, 1]
        for key, lab, c in (("base|win", "ventana (artículo)", "#7f7f7f"), ("base|tgt", "objetivo (M3)", "#1f77b4"),
                            ("adapt_r3|tgt", "M1+M2+M3", "#c0392b")):
            arr = np.array([[np.nan if v is None else v for v in p["pos"][key]] for p in profs["contam"]
                            if p["ds"] == "234" and p.get("pos") and key in p["pos"]], dtype=float)
            if len(arr):
                ax.plot(np.arange(arr.shape[1]), 100 * np.nanmean(arr, 0), color=c, marker="o", ms=2.2, label=lab)
        ax.axvspan(-0.5, 9.5, color="#f4c7c3", alpha=0.35, lw=0)
        ax.set_xlabel("posición dentro del evento (pasos desde su inicio)")
        ax.set_ylabel("percentil medio del score")
        ax.set_title("(d) Anomalías de patrón: score a lo largo del evento", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        plt.tight_layout()
        fig.savefig(FIG / "mecanismo.pdf")
        plt.close(fig)
    # 3) reales: diferencia pareada por semilla, M3 y final vs artículo
    if "real" in mfs:
        mf = mfs["real"]
        dss = [d for d in ("creditcard", "cicids", "gecco", "swan_sf") if d in set(mf.ds)]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
        for ax, (met, ref, tl) in zip(axes, (("auprc", PAPER_S, "$\\Delta$AUPRC vs. artículo"),
                                             ("f1_bin", PAPER_D, "$\\Delta$F1 bin. vs. artículo dinámico (con etiquetas)"))):
            for k, (m, lab, c) in enumerate(((M3, "M3", "#1f77b4"), ("M1+M3", "M1+M3", "#17becf"), (FINAL, "M1+M2+M3", "#c0392b"))):
                for i, d in enumerate(dss):
                    x = mf[mf.ds == d]
                    A = x[x.method == m].set_index("seed")[met]
                    B = x[x.method == ref].set_index("seed")[met]
                    dd = (A - B.reindex(A.index)).dropna().values
                    xs = i + (k - 1) * 0.25 + RNG.uniform(-0.05, 0.05, len(dd))
                    ax.scatter(xs, dd, s=6, color=c, alpha=0.6, lw=0, label=lab if i == 0 else None)
                    ax.hlines(dd.mean(), i + (k - 1) * 0.25 - 0.1, i + (k - 1) * 0.25 + 0.1, color=c, lw=2)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(dss))); ax.set_xticklabels([DSNAME[d] for d in dss])
            ax.set_title(tl)
        axes[0].legend(fontsize=6, frameon=False)
        plt.tight_layout()
        fig.savefig(FIG / "reales.pdf")
        plt.close(fig)
    # 4) sintéticos: ablación (barras F1/AUPRC)
    if "synth" in mfs:
        mf = mfs["synth"]
        show = [PAPER_D, PAPER_S, "M1 · umbral IQR", "M2 · recorte z>2", M3, "M1+M3", "M2+M3 (recorte z>2)",
                "M1+M2 (recorte adaptativo, sin M3)", FINAL]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
        for ax, (met, nm) in zip(axes, (("auprc", "AUPRC"), ("f1_bin", "F1 binario"))):
            mu = [mf[mf.method == m][met].mean() for m in show]
            se = [1.96 * mf[mf.method == m][met].sem() for m in show]
            col = ["#c0392b" if m == FINAL else "#7f7f7f" if m.startswith("Artículo") else
                   "#1f77b4" if "M3" in m else "#9ecae1" for m in show]
            ax.bar(range(len(show)), mu, yerr=se, color=col, capsize=1.5, width=0.75)
            ax.set_xticks(range(len(show)))
            ax.set_xticklabels([SHORT[m].replace(" (", "\n(") for m in show], rotation=60, ha="right", fontsize=6)
            ax.set_ylabel(nm)
            ax.set_ylim(0, max(mu) * 1.15)
        plt.tight_layout()
        fig.savefig(FIG / "sinteticos.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--no-halo", action="store_true", help="reutiliza halo_*.csv / perfil_*.json ya calculados")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    num = json.loads((RES / "final_numbers.json").read_text())
    evs = {b: load(b) for b in ("synth", "contam", "real", "smd", "nasa", "semi")}
    evs = {b: e for b, e in evs.items() if e is not None}
    mfs = {b: method_frame(e, b) for b, e in evs.items()}
    mfh = {b: (per_series(mf) if b in PER_SERIES else mf) for b, mf in mfs.items()}
    out = {}
    write_tab("hipotesis", tab_hyp([(BNAME[b], num[b]["hyp"]) for b in mfs if b in num],
              "Hipótesis preregistradas. Wilcoxon pareado; unidad: (dataset, semilla) en sintéticos y reales, "
              "(mezcla, nivel, semilla) en contaminación, serie (promedio de semillas) en SMD y SMAP/MSL. "
              "$p$ corregido por Holm dentro de cada bloque. G/P: unidades en que gana/pierde el primer método.", "tab:hip"))
    h5 = [(BNAME[b], num[b]["hyp_h5"]) for b in ("synth", "contam", "real") if b in num and "hyp_h5" in num[b]]
    if h5:
        write_tab("h5_originales", tab_hyp(h5, "H5 en los bloques originales (M3 sola $>$ artículo; familia de Holm aparte "
                                               "para no alterar las familias preregistradas H1--H4).", "tab:h5"))
    # Friedman / Nemenyi
    fr = {}
    units = {"synth": ["ds"], "contam": ["ds", "ratio"], "real": ["ds", "seed"], "smd": ["ds"], "nasa": ["ds"],
             "semi": ["ds", "mix", "ratio"]}
    for b, mf in mfs.items():
        for met in ("auprc", "auroc", "f1_bin"):
            meths = RANK_SET if met == "f1_bin" else [PAPER_S, "M2 · recorte z>2", "M1+M2 (recorte adaptativo, sin M3)", M3,
                                                      "M2+M3 (recorte z>2)", FINAL, "Error del LSTM (PD) + IQR"]
            if met != "f1_bin" and "LOF + IQR" in set(mf.method):
                meths = meths + ["LOF + IQR", "IForest + IQR"]
            try:
                fr[f"{b}|{met}"] = friedman(mf, units[b], met, meths)
            except ValueError:
                pass
    out["friedman"] = fr
    for key, r in fr.items():
        b, met = key.split("|")
        cd_plot(r, f"{BNAME[b]} · {dict(auprc='AUPRC', auroc='AUROC', f1_bin='F1 binario')[met]}  "
                   f"(Friedman χ²={r['chi2']:.1f}, p={r['p']:.1e}, N={r['n']})", FIG / f"cd_{b}_{met}.pdf")
    # tablas principales
    if "synth" in mfs:
        write_tab("sinteticos", tab_main(mfs["synth"], "synth",
                  "Sintéticos (15 datasets $\\times$ 20 semillas, $L{=}4$, $P{=}1$). Media $\\pm$ d.e. "
                  "$\\Delta$: diferencia pareada frente al artículo (AUPRC) y frente al artículo con umbral dinámico "
                  "que usa etiquetas (F1); * $p<0.05$, ** $p<0.01$, *** $p<0.001$ (Wilcoxon).", "tab:synth"))
        write_tab("factorial_synth", tab_factorial(evs["synth"], "Diseño factorial en sintéticos: efecto de cada pieza.", "tab:fact"))
    for b, title in (("contam", "Contaminación controlada sintética (promedio de las tres mezclas, 20 semillas)"),
                     ("semi", "Contaminación semisintética sobre fondo real SMD (10 máquinas, tres mezclas, 3 semillas)")):
        if b not in mfs:
            continue
        dr = drop_tests(evs[b], idx=UNIT[b], col=MIXCOL[b])
        dr.to_csv(OUT / f"{b}_caidas.csv", index=False)
        out[f"{b}_drops"] = dr.to_dict("records")
        cm = contam_m3(mfs[b], col=MIXCOL[b], unit=[u for u in UNIT[b] if u != MIXCOL[b]])
        cm.to_csv(OUT / f"{b}_celdas.csv", index=False)
        out[f"{b}_cells"] = cm.to_dict("records")
        lo, hi = 100 * mfs[b].ratio.min(), 100 * mfs[b].ratio.max()
        write_tab(f"{b}_auroc", tab_contam(mfs[b], dr,
                  f"{title}. Caída: AUROC({lo:.0f}\\%) $-$ AUROC({hi:.0f}\\%). "
                  "«vs.\\ artículo»: cuánto menos cae que el artículo (Wilcoxon pareado, * $p_{\\text{Holm}}<0.05$).",
                  f"tab:{b}"))
        write_tab(f"{b}_main", tab_main(mfs[b], b, f"{title}: todos los niveles juntos. Mismo formato que la "
                                                    "Tabla~\\ref{tab:synth}.", f"tab:{b}_all"))
    if "real" in mfs:
        rt = real_tests(mfs["real"])
        rt.to_csv(OUT / "reales_pruebas.csv", index=False)
        out["real_tests"] = rt.to_dict("records")
        write_tab("reales", tab_real(mfs["real"], rt,
                  "Datos reales del artículo ($L{=}4$, $P{=}1$, 10 semillas). AUPRC (sin umbral) y F1 binario. "
                  "$^{\\uparrow}$/$^{\\downarrow}$: diferente del artículo con $p<0.05$ (Wilcoxon pareado sobre semillas; "
                  "AUPRC frente al artículo, F1 frente al artículo con umbral dinámico con etiquetas).", "tab:real"))
    if any(b in mfh for b in PER_SERIES):
        write_tab("benchmarks", tab_bench(mfh,
                  "Benchmarks reales adicionales (5 semillas por serie; media de las medias por serie). "
                  "$^{\\uparrow}$/$^{\\downarrow}$: diferente del artículo con $p<0.05$ (Wilcoxon pareado entre series; "
                  "F1 frente al artículo con umbral dinámico con etiquetas).", "tab:bench"))
        for b in PER_SERIES:
            if b in mfs:
                rows = []
                for m in (M3, FINAL, "M1+M3", "M1+M2 (recorte adaptativo, sin M3)", "M2 · recorte z>2"):
                    for met, ref in (("auprc", PAPER_S), ("auroc", PAPER_S), ("f1_bin", PAPER_D)):
                        rows.append(dict(block=b, cmp=SHORT[m], metric=met, **paired(mfh[b], b, m, ref, met)))
                out[f"{b}_tests"] = rows
    # mecanismo
    halos, profs = {}, {}
    for b in mfs:
        if a.no_halo and (OUT / f"halo_{b}.csv").exists():
            halos[b] = pd.read_csv(OUT / f"halo_{b}.csv", dtype={"ds": str, "mix": str})
            profs[b] = json.loads((OUT / f"perfil_{b}.json").read_text())
        else:
            halos[b], profs[b] = run_halo(b, a.workers)
    write_tab("halo", tab_halo([(BNAME[b], halos[b]) for b in halos],
              "Mecanismo de la mejora 3. Halo: instantes normales a distancia $1..L{+}P{-}1$ de una anomalía. "
              "Top-$k$: los $k$ instantes con mayor score, $k$ = número de anomalías.", "tab:halo"))
    out["halo"] = {b: h.groupby(["train", "proj"])[["auroc_anom_vs_halo", "auroc_anom_vs_far", "topk_halo_share",
                                                     "topk_anom_share", "halo_frac"]].mean().reset_index().to_dict("records")
                   for b, h in halos.items()}
    figs(mfs, evs, halos, profs)
    (OUT / "numeros.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print("paper ok:", sorted(p.name for p in TAB.glob("*.tex")), sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
