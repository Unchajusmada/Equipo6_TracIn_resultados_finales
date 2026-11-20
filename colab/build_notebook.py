"""Genera colab/Equipo6_TracIn_mejoras.ipynb: un notebook autocontenido para
Google Colab con nuestro código (copiado tal cual de src/), una versión
reducida de los experimentos, una sección por mejora y nuestros resultados
completos como referencia.

    python colab/build_notebook.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RES = ROOT / "results" / "replica"
OUT = ROOT / "colab" / "Equipo6_TracIn_mejoras.ipynb"
sys.path.insert(0, str(SRC))
from common import valid_mask as _vm  # noqa: E402

# Parámetros de la versión reducida (los mismos que usa el notebook)
SEEDS_A = 5
LEVELS_B = [0.01, 0.15, 0.30]
SEEDS_B = 3
POINT, PATTERN = ["0", "1"], ["2", "3", "4"]


# ------------------------------------------------------------- código fuente
def pick(path: Path, names: list[str]) -> str:
    """Copia literal de las definiciones pedidas (funciones, clases o constantes)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out, found = [], set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            start = node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
            out.append("\n".join(src.splitlines()[start - 1:node.end_lineno]))
            found.add(name)
    assert found == set(names), set(names) - found
    return "\n\n\n".join(out)


# synthetic.py se copia entero (sin la línea de __future__): es el generador de Lai et al.
syn_txt = (SRC / "synthetic.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", "")
common_code = pick(SRC / "common.py", ["EDGE", "project_sum", "project_targets_mean", "valid_mask", "_log", "thr_iqr",
                                       "thr_mad", "thr_pot", "unsupervised_thresholds"])
protocol_code = pick(SRC / "replica" / "protocol.py", ["PCTS", "f1_pair", "oracle_f1", "_best_pct", "evaluate"])
rw_code = pick(SRC / "forecaster.py", ["robust_z", "weights_from_influence"])
lof_code = pick(SRC / "baselines.py", ["window_matrix", "_to_timesteps", "lof"])
adapt_code = pick(SRC / "replica" / "rep_contam_fix.py", ["adapt_weights"])
lab_code = pick(SRC / "replica" / "lab.py", ["Cfg", "LSTM", "normalize", "windows", "_per_sample_sqgrad", "_stats", "run"])

# ------------------------------------------------------------- referencias (nuestras corridas)
n = json.loads((RES / "replica_numbers.json").read_text(encoding="utf-8"))
r4 = lambda x: round(float(x), 4)

ev = pd.read_csv(RES / "synth_base_eval.csv", dtype={"ds": str})
A_KEYS = {"tracin": "tracin", "m2": "rw_trim_r2", "m3": "tracin_tgt", "m23": "rw_trim_r2_tgt", "pd": "pd"}
ev = ev[((ev.method == "tracin_lstm") & (ev.cfg == "L4P1") & ev.score.isin(A_KEYS.values()))
        | ((ev.method == "lof") & (ev.cfg == "w1"))].copy()
ev["m"] = ev.method.where(ev.method == "lof", ev.score.map({v: k for k, v in A_KEYS.items()}))
cols = ["auprc", "D_mac", "U_iqr_mac", "U_mad_mac"]


def summarize(d):
    out = d.groupby("m")[cols].mean().round(4).to_dict(orient="index")
    for m_, g in d.groupby("m"):
        out[m_]["auprc_puntual"] = r4(g[g.ds.isin(POINT)].auprc.mean())
        out[m_]["auprc_patron"] = r4(g[g.ds.isin(PATTERN)].auprc.mean())
    return out


refA_sub, refA_all = summarize(ev[ev.seed < SEEDS_A]), summarize(ev)

B_NAMES = ["TracIn original", "Sólo mejora 2", "Sólo mejora 3", "Mejoras 1 + 2", "Mejoras 2 + 3", "Mejoras 1 + 2 + 3",
           "LOF", "Error LSTM"]
refB = {}
for r in LEVELS_B:
    rows = {}
    for s_ in range(SEEDS_B):
        z = np.load(RES / "contam" / f"r{r:.2f}_s{s_}.npz")
        zf = np.load(RES / "contam_fix" / f"r{r:.2f}_s{s_}.npz")
        y = z["y"].astype(bool); m = _vm(len(y))
        sc = [z["tracin"], z["rw_trim_r2"], zf["tracin_tgt"], zf["adapt_r3"], z["rw_trim_r2_tgt"], zf["adapt_r3_tgt"],
              z["lof"], z["pd"]]
        for k, v in zip(B_NAMES, sc):
            rows.setdefault(k, []).append(roc_auc_score(y[m], v[m]))
    refB[f"{r:.2f}"] = {k: r4(np.mean(v)) for k, v in rows.items()}
ce = pd.read_csv(RES / "contam_eval.csv")
ce = ce[ce.ratio.isin(LEVELS_B) & (ce.seed < SEEDS_B) & ce.score.isin(["tracin", "tracin_tgt"])]
THR_NAMES = {"tracin": "TracIn original", "tracin_tgt": "TracIn con mejora 3"}
refB_thr = {THR_NAMES[sc]: {f"{r:.2f}": {"dyn": r4(g[g.ratio == r].D_mac.mean()), "iqr": r4(g[g.ratio == r].U_iqr_mac.mean()),
                                         "mad": r4(g[g.ratio == r].U_mad_mac.mean())} for r in LEVELS_B}
            for sc, g in ce.groupby("score")}
FA = n["cont_fix"]["auroc"]
refB_full = {"rate": [r4(x) for x in n["cont_fix"]["rate"]]}
for k, src in zip(B_NAMES, ["TracIn (artículo)", "Recorte z > 2 (2 rondas)", "Proy. objetivo", "Recorte adaptativo IQR (3 rondas)",
                            "Recorte z > 2 + proy. (2 rondas)", "Recorte adaptativo IQR + proy. (3 rondas)", "LOF", "Error LSTM (PD)"]):
    refB_full[k] = [r4(x) for x in FA[src]]
CA = n["cont"]["auroc"]
for k in ["IForest", "OCSVM", "Autoencoder", "Matrix Profile"]:
    refB_full[k] = [r4(x) for x in CA[k]]

full = {
    "sinteticos": {m: {"AUPRC": r4(n["syn_cmp"][m]["auprc"]), "F1 macro": r4(n["syn_cmp"][m]["D_mac"])}
                   for m in ["TracIn (artículo)", "TracIn proy. objetivo", "Error LSTM (PD)", "LOF", "One-Class SVM",
                             "Isolation Forest", "Autoencoder", "Matrix Profile"]},
    "reales_auprc": {d: {k: r4(v[0]) for k, v in n["real_cmp"][d].items()} for d in n["real_cmp"]},
    "reales_mejoras": {d: {k: r4(v["auprc"]) for k, v in n["real_impr"][d].items()} for d in n["real_impr"]},
    "reales_umbral": {sc: {d: {k: r4(v) for k, v in n["real_thr"][sc][d].items()} for d in n["real_thr"][sc]}
                      for sc in n["real_thr"]},
}

# ------------------------------------------------------------- celdas
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t, hidden=False, **subs):
    for k, v in subs.items():
        t = t.replace(f"__{k}__", v)
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": t.strip("\n")})


J = lambda x: json.dumps(x, ensure_ascii=False)

md("""
# TracIn para detección de anomalías: nuestras mejoras

Equipo 6 (Alan Rangel, Carlos Bruzual) · Aprendizaje no supervisado, INAOE

El método del artículo (Verdone et al., 2026) **no entrena una red para identificar anomalías: usa el proceso de
aprendizaje de una red para detectarlas**. Entrena un LSTM a predecir la serie y mide con TracIn cuánto le costó aprender
cada parte; lo que más le costó es lo anómalo. Propusimos tres mejoras y aquí se prueba cada una.

**Cómo usarlo:** *Entorno de ejecución → Ejecutar todo*. No hay que subir nada. Con la CPU gratuita de Colab tarda unos
40 minutos; casi todo es la sección 2. Después de correrla, las secciones 3 a 8 sólo leen resultados.

**Contenido**

0. Preparación y código
1. El método original
2. Correr los experimentos
3. **Mejora 1:** umbral sin etiquetas
4. **Mejora 2:** reponderación
5. **Mejora 3:** proyección al objetivo
6. Las mejoras combinadas
7. Comparación con otros métodos de detección
8. Resultados completos y resumen

Los experimentos usan las mismas semillas que nuestras corridas, así que cada tabla trae al lado nuestra cifra para
comparar (puede diferir en el tercer decimal por la versión de PyTorch o del procesador). Las corridas completas usaron
más semillas; esas cifras también se muestran.
""")

# ---------------------------------------------------------------- 0
md("""
## 0. Preparación y código

Las celdas de esta sección son el código que usamos, copiado de nuestro repositorio. Las tres primeras están ocultas (doble
clic para verlas): el generador de series sintéticas, las proyecciones y umbrales, y el protocolo de evaluación.
""")
code("""
import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
try:
    from IPython.display import display
except ImportError:
    display = print
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3)
print("PyTorch", torch.__version__, "· hilos:", torch.get_num_threads())
""")
code("# @title Generador de series sintéticas (Lai et al., 2021)\nfrom __future__ import annotations\n" + syn_txt, hidden=True)
code("# @title Proyecciones, umbrales sin etiquetas (mejora 1), reponderación (mejora 2) y LOF\n"
     "from __future__ import annotations\nfrom scipy import stats\nfrom sklearn.neighbors import LocalOutlierFactor\n\n"
     + common_code + "\n\n\n" + rw_code + "\n\n\n" + adapt_code + "\n\n\n" + lof_code, hidden=True)
code("# @title Protocolo de evaluación del artículo (umbral dinámico, F1 binario y macro)\nfrom __future__ import annotations\n"
     "from sklearn.metrics import average_precision_score\n\n" + protocol_code, hidden=True)
md("""
**Modelo y TracIn.** Un LSTM recibe 4 instantes de la serie y predice el siguiente. Al final de cada época se guarda, para
cada ventana, el cuadrado del gradiente de su pérdida por la tasa de aprendizaje; la suma sobre las épocas es la
*self-influence* de TracIn. Con la última capa y pérdida cuadrática tiene forma cerrada:

$$SI(x) = \\sum_k \\eta_k \\, \\frac{4}{D} \\, \\ell_k(x) \\, (1 + \\lVert h_k(x) \\rVert^2)$$

es decir, el error de cada ventana acumulado durante todo el entrenamiento. TracIn no cambia el entrenamiento, sólo lo
observa. El parámetro `weights` de `run` es lo que usa la mejora 2.
""")
code("from __future__ import annotations\nimport math\nfrom dataclasses import dataclass, asdict\nfrom torch import nn\n"
     "from torch.func import functional_call, grad, vmap\n\n" + lab_code)

# ---------------------------------------------------------------- 1
md("""
## 1. El método original

1. Se toma la serie, sin etiquetas.
2. Se entrena un LSTM para que prediga la serie.
3. Con TracIn se mide cuánto le costó aprender cada ventana (4 instantes de entrada + 1 que predice).
4. El puntaje de cada ventana se pasa a los instantes: el artículo suma, en cada instante, el puntaje de todas las ventanas
   que lo contienen.
5. Lo que supera un umbral se marca como anomalía. El artículo elige ese umbral con etiquetas.

Ejemplo en una serie sintética con anomalías puntuales (tipo 0). Las franjas rojas son las anomalías reales.
""")
code("""
X1, y1 = make_dataset("0", seed=0)
r1 = run(X1, 4, 1, seed=0, cfg=Cfg())
n1, m1 = len(y1), valid_mask(len(y1))
s_original = project_sum(r1["si"], 5, n1)
print(f"Épocas: {r1['epochs']} · AUPRC del método original: {average_precision_score(y1[m1], s_original[m1]):.3f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
ax[0].plot(X1[:, 0], lw=.8, color="k"); ax[0].set_ylabel("serie (canal 0)")
ax[1].plot(s_original / s_original.max(), lw=.9, color="k"); ax[1].set_ylabel("puntaje TracIn")
for a in ax:
    for i in np.flatnonzero(y1):
        a.axvspan(i - .5, i + .5, color="tab:red", alpha=.25, lw=0)
ax[1].set_xlabel("instante"); plt.tight_layout(); plt.show()
""")
md("""
Los puntajes altos coinciden con las anomalías: el modelo sí tuvo más dificultad con ellas. Pero cada pico se ve como una
meseta de varios instantes; eso es lo que corrige la mejora 3.
""")

# ---------------------------------------------------------------- 2
md(f"""
## 2. Correr los experimentos

Aquí se hace todo el cálculo; las secciones siguientes sólo analizan los resultados.

**Experimento A, 15 sintéticos.** Los 15 conjuntos de Lai et al. (400 instantes, 5 canales, anomalías puntuales y de
patrón, solas o combinadas), {SEEDS_A} semillas (nosotros: 20). Ventana fija de 4 + 1. Para cada serie se entrena el LSTM
una vez y se reentrena dos veces para la mejora 2. Se calculan: TracIn original, sólo mejora 2, sólo mejora 3, mejoras 2 + 3,
el error de predicción final del mismo LSTM y LOF.

**Experimento B, contaminación.** Una serie de 4000 instantes con los cinco tipos de anomalía, con proporción nominal de
anomalías {LEVELS_B} (real ≈ 1 %, 14 % y 26 %), {SEEDS_B} semillas (nosotros: 7 niveles × 10 semillas). Además de lo anterior
se calcula el recorte adaptativo (mejoras 1 + 2), que se reentrena tres veces. Aquí se mide con AUROC, que no depende de la
proporción de anomalías.
""")
code("""
SEMILLAS_A = __SEEDS_A__   # nosotros: 20

def evaluar(y, s):
    \"\"\"AUPRC, F1 con el umbral del artículo y F1 con los umbrales sin etiquetas (misma segunda mitad).\"\"\"
    s = np.asarray(s, dtype=np.float32).astype(np.float64)   # igual que en nuestros archivos
    e = evaluate(y, s, 0.85)
    k = len(s) // 2
    for nombre, thr in unsupervised_thresholds(s).items():
        e[f"U_{nombre}_bin"], e[f"U_{nombre}_mac"] = f1_pair(y[k:].astype(bool), s[k:] >= thr)
    return e

filas, t0 = [], time.time()
for i, ds in enumerate(DATASETS):
    for seed in range(SEMILLAS_A):
        X, y = make_dataset(ds, seed)
        n = len(y); m = valid_mask(n)
        r = run(X, 4, 1, seed, Cfg())
        s2 = r["si"]
        for _ in range(2):                                   # mejora 2: recorte z > 2, 2 rondas
            s2 = run(X, 4, 1, seed, Cfg(), weights=weights_from_influence(s2, "trim"))["si"]
        Xz = normalize(X.astype(np.float64), "z").astype(np.float32)
        scores = {"tracin": project_sum(r["si"], 5, n), "m2": project_sum(s2, 5, n),
                  "m3": project_targets_mean(r["si"], 4, 1, n), "m23": project_targets_mean(s2, 4, 1, n),
                  "pd": project_targets_mean(r["loss_final"], 4, 1, n), "lof": lof(Xz, 1)}
        for nombre, sc in scores.items():
            filas.append(dict(ds=ds, seed=seed, m=nombre, **evaluar(y[m], sc[m])))
    print(f"A: {i + 1}/15 conjuntos · {time.time() - t0:.0f} s")
A = pd.DataFrame(filas)
""", SEEDS_A=str(SEEDS_A))
code("""
NIVELES = __LEVELS_B__
SEMILLAS_B = __SEEDS_B__   # nosotros: 10

filas, filas_thr, t0 = [], [], time.time()
for r_ in NIVELES:
    for seed in range(SEMILLAS_B):
        X, y = make_dataset("01234", 10_000 + seed, length=4000, ratio=r_ / 5)
        n = len(y); m = valid_mask(n)
        base = run(X, 4, 1, seed, Cfg())
        s2 = base["si"]
        for _ in range(2):                                   # mejora 2: recorte z > 2, 2 rondas
            s2 = run(X, 4, 1, seed, Cfg(), weights=weights_from_influence(s2, "trim"))["si"]
        s12 = base["si"]
        for _ in range(3):                                   # mejoras 1 + 2: recorte con el umbral IQR, 3 rondas
            s12 = run(X, 4, 1, seed, Cfg(), weights=adapt_weights(s12))["si"]
        Xz = normalize(X.astype(np.float64), "z").astype(np.float32)
        scores = {"TracIn original": project_sum(base["si"], 5, n), "Sólo mejora 2": project_sum(s2, 5, n),
                  "Sólo mejora 3": project_targets_mean(base["si"], 4, 1, n), "Mejoras 1 + 2": project_sum(s12, 5, n),
                  "Mejoras 2 + 3": project_targets_mean(s2, 4, 1, n), "Mejoras 1 + 2 + 3": project_targets_mean(s12, 4, 1, n),
                  "LOF": lof(Xz, 1), "Error LSTM": project_targets_mean(base["loss_final"], 4, 1, n)}
        fila = dict(nivel=r_, seed=seed, anomalias=y[m].mean())
        for k, v in scores.items():
            fila[k] = roc_auc_score(y[m], np.asarray(v, dtype=np.float32)[m])
        filas.append(fila)
        for k, nombre in (("TracIn original", "TracIn original"), ("Sólo mejora 3", "TracIn con mejora 3")):
            e = evaluar(y[m], scores[k][m])
            filas_thr.append(dict(nivel=r_, seed=seed, score=nombre, dyn=e["D_mac"], iqr=e["U_iqr_mac"], mad=e["U_mad_mac"]))
        print(f"B: nivel {r_:.2f} · semilla {seed} · {time.time() - t0:.0f} s")
B = pd.DataFrame(filas); Bthr = pd.DataFrame(filas_thr)
""", LEVELS_B=str(LEVELS_B), SEEDS_B=str(SEEDS_B))
code("""
# Nuestras cifras de referencia (corridas completas) y utilidades para las tablas
REF_A_MISMAS = __REF_A_SUB__
REF_A_20 = __REF_A_ALL__
REF_B_MISMAS = __REF_B__
REF_B_UMBRAL = __REF_B_THR__
REF_B_COMPLETO = __REF_B_FULL__
NOMBRES_A = {"tracin": "TracIn original", "m2": "Sólo mejora 2", "m3": "Sólo mejora 3", "m23": "Mejoras 2 + 3",
             "pd": "Error de predicción del LSTM", "lof": "LOF"}
resA = A.groupby("m")[["auprc", "D_mac", "U_iqr_mac", "U_mad_mac"]].mean()
resA["auprc_puntual"] = A[A.ds.isin(["0", "1"])].groupby("m").auprc.mean()
resA["auprc_patron"] = A[A.ds.isin(["2", "3", "4"])].groupby("m").auprc.mean()
resB = B.groupby("nivel").mean(numeric_only=True)
ETIQ_B = [f"{v * 100:.0f} %" for v in resB.anomalias]

def tabla_A(claves, col="auprc"):
    \"\"\"Métrica del experimento A: este notebook vs nuestra corrida (mismas semillas y 20 semillas).\"\"\"
    return pd.DataFrame({"Este notebook": [resA.loc[k, col] for k in claves],
                         "Nuestra corrida (mismas semillas)": [REF_A_MISMAS[k][col] for k in claves],
                         "Nuestra corrida (20 semillas)": [REF_A_20[k][col] for k in claves]},
                        index=[NOMBRES_A[k] for k in claves]).round(3)

def tabla_B(metodos):
    \"\"\"AUROC del experimento B por nivel de anomalías, con nuestra referencia de las mismas semillas.\"\"\"
    t = resB[metodos].T; t.columns = [f"{e} (notebook)" for e in ETIQ_B]
    ref = pd.DataFrame(REF_B_MISMAS).loc[metodos]; ref.columns = [f"{e} (nuestra)" for e in ETIQ_B]
    return pd.concat([t, ref], axis=1)[[c for p in zip(t.columns, ref.columns) for c in p]].round(3)

def grafica_B(metodos, titulo):
    \"\"\"Curvas de nuestra corrida completa (10 semillas, 7 niveles) y puntos de este notebook.\"\"\"
    x = [v * 100 for v in REF_B_COMPLETO["rate"]]
    plt.figure(figsize=(7, 3.4))
    for k in metodos:
        l, = plt.plot(x, REF_B_COMPLETO[k], "--" if k in ("LOF", "Error LSTM") else "-", lw=1.5, label=k)
        if k in resB:
            plt.plot(resB.anomalias * 100, resB[k], "o", color=l.get_color(), ms=5)
    plt.title(titulo + "\\nlíneas: nuestra corrida completa · puntos: este notebook", fontsize=9, loc="left")
    plt.xlabel("% de instantes anómalos"); plt.ylabel("AUROC"); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.show()

print("Listo. Las secciones siguientes analizan estos resultados.")
""", REF_A_SUB=J(refA_sub), REF_A_ALL=J(refA_all), REF_B=J(refB), REF_B_THR=J(refB_thr), REF_B_FULL=J(refB_full))

# ---------------------------------------------------------------- 3 mejora 1
md("""
## 3. Mejora 1: umbral sin etiquetas

**Qué es.** El artículo elige el umbral del paso 5 con etiquetas: prueba varios umbrales en la primera mitad de la serie,
se queda con el que da mejor F1 y lo aplica en la segunda. La mejora lo sustituye por reglas estadísticas sobre los propios
puntajes, sin etiquetas:
* **IQR:** se marca lo que supera Q3 + 1.5 · (Q3 − Q1).
* **MAD:** se marca lo que supera la mediana + 3 desviaciones absolutas medianas.

**Por qué.** En un problema no supervisado no hay etiquetas; un método que las necesita para decidir qué es anomalía no se
puede usar en la práctica. El objetivo no es detectar mejor, sino no depender de etiquetas.

**Cómo se mide.** Es un umbral: no cambia el orden de los puntajes, así que no cambia el AUPRC. Se mide con F1 (macro,
como el artículo en sintéticos) en la misma segunda mitad de la serie.
""")
code("""
print("Sintéticos: F1 macro con el umbral del artículo (con etiquetas) y con IQR y MAD (sin etiquetas)")
t = resA.loc[["tracin", "m3"], ["D_mac", "U_iqr_mac", "U_mad_mac"]].copy()
t.index = ["TracIn original", "TracIn con mejora 3"]
t.columns = ["Con etiquetas", "IQR", "MAD"]
t["Con etiquetas (nuestra)"] = [REF_A_MISMAS[k]["D_mac"] for k in ("tracin", "m3")]
t["IQR (nuestra)"] = [REF_A_MISMAS[k]["U_iqr_mac"] for k in ("tracin", "m3")]
t["MAD (nuestra)"] = [REF_A_MISMAS[k]["U_mad_mac"] for k in ("tracin", "m3")]
display(t.round(3))

print("Serie contaminada: F1 macro con y sin etiquetas")
t = Bthr.groupby(["score", "nivel"])[["dyn", "iqr", "mad"]].mean().unstack("nivel").round(3)
t.columns = [f"{ {'dyn': 'Con etiquetas', 'iqr': 'IQR', 'mad': 'MAD'}[a] } · {e}" for a in ("dyn", "iqr", "mad") for e in ETIQ_B]
display(t)
print("Nuestra corrida, mismas semillas")
display(pd.DataFrame({k: [v[r]["dyn"] for r in v] + [v[r]["iqr"] for r in v] + [v[r]["mad"] for r in v]
                      for k, v in REF_B_UMBRAL.items()}, index=t.columns).T.loc[t.index].round(3))
""")
md("""
**Resultado.**
* En los sintéticos, IQR y MAD dan prácticamente el mismo F1 que el umbral con etiquetas: el método queda completamente no
  supervisado sin perder desempeño.
* En la serie contaminada se ve el límite: la regla adecuada depende de cuántas anomalías hay. Con la mejora 3, IQR falla
  cuando las anomalías son muy pocas (1 %), porque marca demasiados instantes, e iguala al umbral con etiquetas desde 5 %.
  Con el TracIn original, IQR falla con mucha contaminación y MAD aguanta.
* En datos reales (sección 8) pasa lo mismo: cada conjunto prefiere una regla distinta.

**¿Sirvió?** Sí, con un matiz: elimina las etiquetas sin costo en sintéticos, pero falta una forma de elegir la regla según
la proporción de anomalías.
""")

# ---------------------------------------------------------------- 4 mejora 2
md("""
## 4. Mejora 2: reponderación

**Qué es.** Se calcula la self-influence, se pone peso cero a las ventanas más sospechosas (z robusto mayor que 2), se
reentrena el LSTM desde cero con esos pesos y se repite. El puntaje final es la self-influence del último entrenamiento.

**Por qué.** TracIn supone que las anomalías son raras y por eso difíciles de aprender. Si la serie tiene muchas, el modelo
las ve con frecuencia, las aprende y su error baja: TracIn deja de distinguirlas. Al quitarles peso, el modelo aprende sobre
todo lo normal y las anomalías le siguen costando. Es la idea del aprendizaje robusto con datos ruidosos (Shen y Sanghavi,
2019). Es la única mejora que cambia el entrenamiento.

**Variante con la mejora 1 (mejoras 1 + 2).** En lugar de z > 2, se quitan las ventanas que superan el umbral IQR; así se
recorta más cuanto más contaminada está la serie.
""")
code("""
print("Sintéticos (AUPRC): la mejora 2 sola")
display(tabla_A(["tracin", "m2"]))

print("Serie contaminada (AUROC): la mejora 2 sola y con el criterio de la mejora 1")
display(tabla_B(["TracIn original", "Sólo mejora 2", "Mejoras 1 + 2"]))
grafica_B(["TracIn original", "Sólo mejora 2", "Mejoras 1 + 2", "LOF", "Error LSTM"], "Mejora 2")
""")
md("""
**Resultado.**
* En los sintéticos de 400 instantes casi no cambia: hay pocas anomalías y el modelo no alcanza a aprenderlas, así que no
  hay nada que corregir.
* En la serie contaminada, el TracIn original se degrada mucho al aumentar las anomalías. La mejora 2 sola recupera una parte
  pequeña de esa caída, de forma consistente; con el criterio adaptativo de la mejora 1, un poco más.
* No basta sola: con 26 % de anomalías sigue por debajo de LOF y del error de predicción. El recorte sólo logra quitar cerca
  de un tercio de las ventanas anómalas, porque el modelo ya aprendió las demás.
* En datos reales (sección 8) el efecto es mixto: ayuda en GECCO, empeora Credit Card y no cambia los otros dos.

**¿Sirvió?** En parte: ataca el problema correcto, pero sola aporta poco. Funciona bien combinada con la mejora 3
(sección 6).
""")

# ---------------------------------------------------------------- 5 mejora 3
md("""
## 5. Mejora 3: proyección al objetivo

**Qué es.** Cambia el paso 4: el puntaje de cada ventana se asigna sólo al instante que la ventana predice, no a todos sus
instantes. Cada instante recibe el promedio de las ventanas que lo predicen. El modelo y el entrenamiento no cambian.

**Por qué.** El error de una ventana mide qué tan mal predijo su salida, así que la anomalía está en lo que no pudo
predecir, no en lo que vio. Si en t5 hay un pico, la ventana que predice t5 falla; el artículo reparte ese puntaje también a
t1–t4, que son normales, y el pico vuelve a sumar en las ventanas siguientes que lo tienen como entrada. Un pico de un
instante se vuelve una meseta. Es la forma en que los detectores de error de predicción asignan el error (Malhotra et al.,
2015).

La misma serie de la sección 1, con el mismo modelo y la misma self-influence:
""")
code("""
s_mejora3 = project_targets_mean(r1["si"], 4, 1, n1)
fig, ax = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
for a, sc, t in [(ax[0], s_original, "Método original"), (ax[1], s_mejora3, "Mejora 3")]:
    a.plot(sc / sc.max(), lw=.9, color="k")
    a.set_title(f"{t} · AUPRC {average_precision_score(y1[m1], sc[m1]):.3f}", loc="left", fontsize=10)
    for i in np.flatnonzero(y1):
        a.axvspan(i - .5, i + .5, color="tab:red", alpha=.25, lw=0)
ax[1].set_xlabel("instante (rojo: anomalía)"); plt.tight_layout(); plt.show()

print("Sintéticos (AUPRC), total y por tipo de anomalía")
t = tabla_A(["tracin", "m3"])
t["Puntuales, tipos 0 y 1 (notebook)"] = [resA.loc[k, "auprc_puntual"] for k in ("tracin", "m3")]
t["Puntuales (nuestra, mismas semillas)"] = [REF_A_MISMAS[k]["auprc_puntual"] for k in ("tracin", "m3")]
t["De patrón, tipos 2-4 (notebook)"] = [resA.loc[k, "auprc_patron"] for k in ("tracin", "m3")]
t["De patrón (nuestra, mismas semillas)"] = [REF_A_MISMAS[k]["auprc_patron"] for k in ("tracin", "m3")]
display(t.round(3))

print("Serie contaminada (AUROC)")
display(tabla_B(["TracIn original", "Sólo mejora 3"]))
grafica_B(["TracIn original", "Sólo mejora 3", "LOF", "Error LSTM"], "Mejora 3")
""")
md("""
**Resultado.**
* Con la mejora 3 el puntaje se queda en el instante de la anomalía en lugar de formar mesetas.
* Es la mejora con más efecto: el AUPRC en sintéticos sube de forma muy marcada, sobre todo en anomalías puntuales, que eran
  la debilidad del método original.
* En la serie contaminada es la que más frena la degradación por sí sola.
* En datos reales (sección 8) es la mejor en Credit Card (fraudes puntuales). En GECCO, con anomalías largas, la proyección
  original funciona un poco mejor.

**¿Sirvió?** Sí, y es la de mayor efecto.
""")

# ---------------------------------------------------------------- 6 combinadas
md("""
## 6. Las mejoras combinadas

* **Mejoras 2 + 3:** reponderación (z > 2) con la proyección al objetivo.
* **Mejoras 1 + 2 + 3:** reponderación con el umbral IQR como criterio de recorte y proyección al objetivo.
""")
code("""
print("Sintéticos (AUPRC)")
display(tabla_A(["tracin", "m2", "m3", "m23"]))

print("Serie contaminada (AUROC)")
display(tabla_B(["TracIn original", "Sólo mejora 2", "Sólo mejora 3", "Mejoras 1 + 2", "Mejoras 2 + 3", "Mejoras 1 + 2 + 3"]))
print("Caída de 1 % a 26 % de anomalías (nuestra corrida completa, 10 semillas)")
display(pd.Series({k: REF_B_COMPLETO[k][-1] - REF_B_COMPLETO[k][0] for k in
                   ["TracIn original", "Sólo mejora 2", "Mejoras 1 + 2", "Sólo mejora 3", "Mejoras 2 + 3", "Mejoras 1 + 2 + 3",
                    "LOF", "Error LSTM"]}).round(3).to_frame("caída de AUROC"))
grafica_B(["TracIn original", "Sólo mejora 3", "Mejoras 2 + 3", "Mejoras 1 + 2 + 3", "LOF", "Error LSTM"], "Mejoras combinadas")
""")
md("""
**Resultado.** Las mejoras se complementan. En los sintéticos manda la mejora 3. Con contaminación, las tres juntas son las
que menos se degradan y el mejor detector cuando hay muchas anomalías; el costo es una ligera pérdida con 1 % y reentrenar
el modelo tres veces más.
""")

# ---------------------------------------------------------------- 7 comparación
md("""
## 7. Comparación con otros métodos de detección

Todos los métodos reciben la misma serie, dan un puntaje a cada instante y se califican igual contra las anomalías reales.
Hay dos grupos:
* **Usan el mismo LSTM:** TracIn original, TracIn con nuestras mejoras y el error de predicción final. Sólo cambia cómo se
  aprovecha el modelo; es la comparación más limpia.
* **No usan el LSTM:** LOF (densidad de vecinos) en este notebook; en nuestras corridas completas también Isolation Forest,
  One-Class SVM, un autoencoder y Matrix Profile.
""")
code("""
print("Sintéticos (AUPRC)")
display(tabla_A(["m3", "pd", "lof", "tracin"]))

print("Serie contaminada (AUROC), nuestra corrida completa con todos los métodos")
t = pd.DataFrame({k: REF_B_COMPLETO[k] for k in ["Mejoras 1 + 2 + 3", "Sólo mejora 3", "LOF", "Error LSTM", "IForest",
                                                  "Matrix Profile", "OCSVM", "Autoencoder", "TracIn original"]}).T
t.columns = [f"{v * 100:.0f} %" for v in REF_B_COMPLETO["rate"]]
display(t.round(3))
""")
md("""
**Resultado.** Con la misma métrica, el TracIn original queda por debajo del error de predicción y de LOF en sintéticos. Con
la mejora 3 queda primero, y con las tres mejoras es el mejor en todos los niveles de contaminación. La comparación con el
error de predicción del mismo modelo muestra que usar el proceso de aprendizaje sí aporta, una vez que el puntaje se asigna
bien.
""")

# ---------------------------------------------------------------- 8 completos
md("""
## 8. Resultados completos y resumen

Estas tablas no se calculan aquí: son de nuestras corridas completas (sintéticos con 20 semillas y la mejor de 9
configuraciones de ventana; reales con 3 semillas). Los datos reales (Credit Card, CICIDS, GECCO y SWAN-SF) pesan cerca de
600 MB y tardan horas, por eso no se corren en este notebook.
""")
code("""
FULL = __FULL__
print("Sintéticos, todos los métodos (mejor configuración de cada uno)")
display(pd.DataFrame(FULL["sinteticos"]).T.rename(index={"TracIn (artículo)": "TracIn original",
        "TracIn proy. objetivo": "TracIn con mejora 3", "Error LSTM (PD)": "Error de predicción"}).round(3))
print("Reales: AUPRC de cada método")
display(pd.DataFrame(FULL["reales_auprc"]).rename(index={"TracIn": "TracIn original",
        "TracIn proy. objetivo": "TracIn con mejora 3"}).round(3))
print("Reales: AUPRC con las mejoras 2 y 3 (reponderación de 1 ronda, criterio z > 2)")
display(pd.DataFrame(FULL["reales_mejoras"]).rename(index={"tracin": "TracIn original", "tracin_tgt": "Sólo mejora 3",
        "rw_soft_r1": "Sólo mejora 2 (suave)", "rw_trim_r1": "Sólo mejora 2 (recorte)", "rw_trim_r1_tgt": "Mejoras 2 + 3"}).round(3))
print("Reales: mejora 1, F1 binario con umbral con etiquetas (D_bin) y sin etiquetas (U_*), ventana 4 + 1")
for sc, d in FULL["reales_umbral"].items():
    print(sc); display(pd.DataFrame(d).round(3))
""", FULL=J(full))
md("""
| Mejora | Qué cambia | ¿Sirvió? |
|---|---|---|
| 1 · Umbral sin etiquetas | El umbral final | Sí en sintéticos: iguala al umbral con etiquetas. La regla adecuada depende de la proporción de anomalías. |
| 2 · Reponderación | El entrenamiento | En parte: sola aporta poco; combinada con 1 y 3 deja la menor degradación con contaminación. En reales es mixta. |
| 3 · Proyección al objetivo | Cómo se asigna el puntaje | Sí: la de mayor efecto; mejor detector en sintéticos y en Credit Card. |
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "toc_visible": True},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    src = c["source"]
    c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
