"""Exporta las cifras de final_numbers.json y paper/numeros.json como macros
LaTeX (results/final/paper/tablas/macros.tex), para que el texto de
resultados.tex cite exactamente los números calculados, sin transcribir a mano.

Uso en LaTeX:
  \\Hy{bloque}{H1}{mean|lo|hi|ci|wins|losses|n|ph|rb}   hipótesis
  \\Me{bloque}{clave}{auprc|auroc|f1|f1m}                 media de un método
  \\Dr{bloque}{clave}{drop|vs|vsp|d01|d234|d01234}       caída de AUROC
  \\Rt{dataset}{m3|fin}{auprc}                           reales, Δ vs artículo
  \\Bt{bloque}{m3|fin}{auprc|auroc|f1}{mean|ci|wins|losses|n|p}   SMD/NASA
  \\Ha{bloque}{win|tgt|finwin|fintgt}{halo|far|topk|anom}   halo
  \\Hs{rho|p|n}{auroc|auprc|fin}                          H6
  \\Fr{bloque}{auprc|auroc|f1_bin}{p|chi|n}               Friedman
Clave de método: art, artdyn, m1, m2s, m2, m3, m1m3, m2m3, m1m2, finz, fin, pd, lof, ifo.
"""
from __future__ import annotations

import json
import math

from core import RES

KEY = {
    "Artículo · umbral estático": "art", "Artículo · umbral dinámico (usa etiquetas)": "artdyn",
    "M1 · umbral IQR": "m1", "M2 · reponderación suave": "m2s", "M2 · recorte z>2": "m2", "M3 · proyección al objetivo": "m3",
    "M1+M3": "m1m3", "M2+M3 (recorte z>2)": "m2m3", "M1+M2 (recorte adaptativo, sin M3)": "m1m2",
    "M1+M2+M3 con recorte z>2": "finz", "**Método final** (recorte adaptativo + proyección + IQR)": "fin",
    "Error del LSTM (PD) + IQR": "pd", "LOF + IQR": "lof", "IForest + IQR": "ifo",
}
CMPKEY = {"M3": "m3", "M1+M2+M3 (final)": "fin", "M1+M3": "m1m3", "M2 recorte": "m2", "M1+M2": "m1m2", "M1": "m1"}
DROPKEY = {"Artículo": "art", "M2 (recorte z>2)": "m2", "M1+M2": "m1m2", "M3": "m3", "M2+M3": "m2m3",
           "Final": "fin", "Final (M1+M2+M3)": "fin", "Error LSTM": "pd", "LOF": "lof", "IForest": "ifo"}


def f3(x, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def fp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-3:
        e = math.floor(math.log10(p))
        return f"$<10^{{{e + 1}}}$"
    return f"{p:.3f}"


def main():
    out = []

    def d(name, val):
        out.append(f"\\expandafter\\def\\csname {name}\\endcsname{{{val}}}")

    N = json.loads((RES / "final_numbers.json").read_text(encoding="utf-8"))
    P = json.loads((RES / "paper" / "numeros.json").read_text(encoding="utf-8"))
    for b, nb in N.items():
        if b == "h6":
            continue
        d(f"nu@{b}", nb.get("n_units", "--"))
        d(f"nr@{b}", nb.get("n_runs", "--"))
        for h in nb.get("hyp", []) + nb.get("hyp_h5", []):
            k = f"hy@{b}@{h['h']}"
            d(k + "@mean", f3(h["mean"], True))
            d(k + "@lo", f3(h["lo"], True))
            d(k + "@hi", f3(h["hi"], True))
            d(k + "@ci", f"[{f3(h['lo'], True)}, {f3(h['hi'], True)}]")
            d(k + "@wins", h["wins"])
            d(k + "@losses", h["losses"])
            d(k + "@n", h["n"])
            d(k + "@ph", fp(h["p_holm"]))
            d(k + "@rb", f3(h["rb"], True))
        for m, v in nb["methods"].items():
            if m in KEY:
                for met, mk in (("auprc", "auprc"), ("auroc", "auroc"), ("f1_bin", "f1"), ("f1_mac", "f1m")):
                    d(f"me@{b}@{KEY[m]}@{mk}", f3(v.get(met)))
        for mix, rows in (nb.get("curves") or {}).items():       # AUROC por nivel (curvas de contaminación)
            for r in rows:
                mk = KEY.get(r["Método"])
                if mk is None:
                    continue
                for lv, v in r.items():
                    if lv.endswith("%"):
                        d(f"cu@{b}@{mix}@{mk}@{lv[:-1]}", v)
                    elif lv == "Caída":
                        d(f"cu@{b}@{mix}@{mk}@caida", v)
        pp = nb.get("paper_protocol")
        if pp:
            for met, mk in (("f1_bin", "f1"), ("auprc", "auprc")):
                x = pp[met]
                d(f"pp@{b}@{mk}@mean", f3(x["mean"], True))
                d(f"pp@{b}@{mk}@ci", f"[{f3(x['lo'], True)}, {f3(x['hi'], True)}]")
                d(f"pp@{b}@{mk}@wins", x["wins"])
                d(f"pp@{b}@{mk}@n", x["n"])
                d(f"pp@{b}@{mk}@p", fp(x["p"]))
                d(f"pp@{b}@{mk}@final", f3(x["final"]))
                d(f"pp@{b}@{mk}@paper", f3(x["paper"]))
    d("nr@total", f"{sum(v.get('n_runs', 0) for k, v in N.items() if k != 'h6'):,}".replace(",", "\\,"))
    for b in ("contam", "semi"):
        for r in P.get(f"{b}_drops", []):
            k = f"dr@{b}@{DROPKEY.get(r['method'], r['method'])}"
            d(k + "@drop", f3(r["drop"]))
            d(k + "@ci", f"[{f3(r['drop_lo'])}, {f3(r['drop_hi'])}]")
            for mix in ("01", "234", "01234"):
                d(k + f"@d{mix}", f3(r.get(f"drop_{mix}")))
            d(k + "@vs", f3(r["vs_mean"], True))
            d(k + "@vsp", fp(r["vs_p_holm"]))
            d(k + "@vsw", "--" if r["vs_wins"] != r["vs_wins"] else int(r["vs_wins"]))
            d(k + "@vsn", "--" if r["vs_n"] != r["vs_n"] else int(r["vs_n"]))
    for b in ("contam", "semi"):           # celdas (mezcla, nivel): media sobre niveles y nº de celdas significativas
        acc = {}
        for r in P.get(f"{b}_cells", []):
            c = {"M3": "m3", "Final": "fin", "M2": "m2", "M1+M2": "m1m2"}.get(r["cmp"], r["cmp"])
            met = {"f1_bin": "f1"}.get(r["metric"], r["metric"])
            a_ = acc.setdefault((c, met, r["mix"]), [0.0, 0, 0, 0])
            a_[0] += r["mean"]; a_[3] += 1
            ph = r.get("p_holm", float("nan"))
            if ph == ph and ph < 0.05:
                a_[1 if r["mean"] > 0 else 2] += 1
        for (c, met, mix), (sm, pos, neg, n) in acc.items():
            k = f"ce@{b}@{c}@{met}@{mix}"
            d(k + "@mean", f3(sm / n, True))
            d(k + "@pos", pos)
            d(k + "@neg", neg)
            d(k + "@n", n)
    for r in P.get("real_tests", []):
        c = CMPKEY[r["cmp"]]
        k = f"rt@{r['ds']}@{c}@{r['metric']}"
        d(k + "@mean", f3(r["mean"], True))
        d(k + "@wins", r["wins"])
        d(k + "@a", f3(r["a"]))
        d(k + "@b", f3(r["b"]))
    for b in ("smd", "nasa"):
        for r in P.get(f"{b}_tests", []):
            c = CMPKEY[r["cmp"]]
            met = {"f1_bin": "f1"}.get(r["metric"], r["metric"])
            k = f"bt@{b}@{c}@{met}"
            d(k + "@mean", f3(r["mean"], True))
            d(k + "@ci", f"[{f3(r['lo'], True)}, {f3(r['hi'], True)}]")
            d(k + "@wins", r["wins"])
            d(k + "@losses", r["losses"])
            d(k + "@n", r["n"])
            d(k + "@p", fp(r["p"]))
    for b, rows in P.get("halo", {}).items():
        for r in rows:
            if r["train"] not in ("base", "adapt_r3"):
                continue
            k = f"ha@{b}@{'' if r['train'] == 'base' else 'fin'}{r['proj']}"
            d(k + "@halo", f3(r["auroc_anom_vs_halo"]))
            d(k + "@far", f3(r["auroc_anom_vs_far"]))
            d(k + "@topk", f"{100 * r['topk_halo_share']:.0f}\\%")
            d(k + "@anom", f"{100 * r['topk_anom_share']:.0f}\\%")
    import pandas as pd                       # halo por mezcla (contaminación sintética y semisintética)
    for b, col in (("contam", "ds"), ("semi", "mix")):
        f = RES / "paper" / f"halo_{b}.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f, dtype={"ds": str, "mix": str})
        if col not in h:
            continue
        g = h.groupby([col, "train", "proj"])[["auroc_anom_vs_halo", "auroc_anom_vs_far", "topk_halo_share",
                                                "topk_anom_share"]].mean()
        gr = h[h.train == "base"].groupby([col, "ratio", "proj"])[["auroc_anom_vs_halo", "auroc_anom_vs_far",
                                                                   "halo_frac", "topk_halo_share"]].mean()
        for (mix, ra, pj), r in gr.iterrows():
            k = f"hr@{b}@{mix}@{round(100 * ra)}@{pj}"
            d(k + "@halo", f3(r.auroc_anom_vs_halo))
            d(k + "@far", f3(r.auroc_anom_vs_far))
            d(k + "@frac", f"{100 * r.halo_frac:.0f}\\%")
            d(k + "@topk", f"{100 * r.topk_halo_share:.0f}\\%")
        for (mix, tr, pj), r in g.iterrows():
            if tr not in ("base", "adapt_r3"):
                continue
            k = f"hm@{b}@{mix}@{'' if tr == 'base' else 'fin'}{pj}"
            d(k + "@halo", f3(r.auroc_anom_vs_halo))
            d(k + "@far", f3(r.auroc_anom_vs_far))
            d(k + "@topk", f"{100 * r.topk_halo_share:.0f}\\%")
            d(k + "@anom", f"{100 * r.topk_anom_share:.0f}\\%")
    h6 = N.get("h6")
    if h6:
        for key, nm in (("d_auroc_m3", "auroc"), ("d_auprc_m3", "auprc"), ("d_auroc_final", "fin")):
            d(f"hs@rho@{nm}", f3(h6[key]["rho"], True))
            d(f"hs@p@{nm}", fp(h6[key]["p"]))
            d(f"hs@n@{nm}", h6[key]["n"])
        for i, q in enumerate(h6.get("cuartiles", []), 1):
            d(f"hs@q{i}@min", f"{100 * q['tasa_min']:.1f}\\%")
            d(f"hs@q{i}@max", f"{100 * q['tasa_max']:.1f}\\%")
            d(f"hs@q{i}@auroc", f3(q["d_auroc_m3"], True))
            d(f"hs@q{i}@auprc", f3(q["d_auprc_m3"], True))
            d(f"hs@q{i}@gana", q["gana"])
            d(f"hs@q{i}@n", q["n"])
    for k, v in P.get("friedman", {}).items():
        b, met = k.split("|")
        d(f"fr@{b}@{met}@p", fp(v["p"]))
        d(f"fr@{b}@{met}@chi", f"{v['chi2']:.1f}")
        d(f"fr@{b}@{met}@n", v["n"])
    head = [
        "% Generado por src/final/macros.py: NO editar a mano.",
        "\\makeatletter",
        "\\def\\@numq#1{\\@ifundefined{#1}{\\textbf{??}}{\\csname #1\\endcsname}}",
        "\\newcommand\\Hy[3]{\\@numq{hy@#1@#2@#3}}",
        "\\newcommand\\Me[3]{\\@numq{me@#1@#2@#3}}",
        "\\newcommand\\Pp[3]{\\@numq{pp@#1@#2@#3}}",
        "\\newcommand\\Dr[3]{\\@numq{dr@#1@#2@#3}}",
        "\\newcommand\\Rt[4]{\\@numq{rt@#1@#2@#3@#4}}",
        "\\newcommand\\Bt[4]{\\@numq{bt@#1@#2@#3@#4}}",
        "\\newcommand\\Ha[3]{\\@numq{ha@#1@#2@#3}}",
        "\\newcommand\\Hs[2]{\\@numq{hs@#1@#2}}",
        "\\newcommand\\Fr[3]{\\@numq{fr@#1@#2@#3}}",
        "\\newcommand\\Ce[5]{\\@numq{ce@#1@#2@#3@#4@#5}}",
        "\\newcommand\\Hm[4]{\\@numq{hm@#1@#2@#3@#4}}",
        "\\newcommand\\Hr[5]{\\@numq{hr@#1@#2@#3@#4@#5}}",
        "\\newcommand\\Cu[4]{\\@numq{cu@#1@#2@#3@#4}}",
        "\\newcommand\\Nu[1]{\\@numq{nu@#1}}",
        "\\newcommand\\Nr[1]{\\@numq{nr@#1}}",
        "\\makeatother",
    ]
    dst = RES / "paper" / "tablas" / "macros.tex"
    dst.write_text("\n".join(head + out) + "\n", encoding="utf-8")
    print("macros:", len(out), "->", dst)


if __name__ == "__main__":
    main()
