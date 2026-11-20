"""Genera colab/Equipo6_TracIn_paso_a_paso.ipynb: cuaderno "para dummies" que
explica TODO el proyecto (el artículo, cada mejora, cómo está implementada,
cómo se evaluó, todos los resultados —también los negativos— y cómo probarlo
en datos reales o en un CSV propio).

El código del método se copia literal de src/ (las mismas funciones que usaron
los experimentos); los resultados completos se leen de results/final/ y se
incrustan en el cuaderno.

    python colab/build_paso_a_paso.py
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "colab" / "Equipo6_TracIn_paso_a_paso.ipynb"
NUM = ROOT / "results" / "final" / "final_numbers.json"
PAPERNUM = ROOT / "results" / "final" / "paper" / "numeros.json"
H6CSV = ROOT / "results" / "final" / "h6_series.csv"


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


syn_code = (SRC / "synthetic.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", "")
common_code = pick(SRC / "common.py", ["EDGE", "project_sum", "project_targets_mean", "valid_mask", "thr_iqr"])
rw_code = pick(SRC / "forecaster.py", ["robust_z", "weights_from_influence"])
adapt_code = pick(SRC / "final" / "core.py", ["adapt_weights"])
lab_code = pick(SRC / "replica" / "lab.py", ["Cfg", "LSTM", "normalize", "windows", "_per_sample_sqgrad", "_stats", "run"])
inject_code = pick(SRC / "data_bench.py", ["standardize", "SEG", "inject"])

# Sólo se incrustan bloques COMPLETOS de la evaluación final (nunca cifras parciales).
# SMAP+MSL: 81 canales únicos × 5 semillas (P-2 aparece dos veces en el archivo de etiquetas de la NASA).
EXPECTED = {"synth": 300, "contam": 420, "real": 40, "smd": 140, "nasa": 405, "semi": 450}
NUMBERS = json.loads(NUM.read_text(encoding="utf-8")) if NUM.exists() else {}
NUMBERS = {b: v for b, v in NUMBERS.items()
           if b == "h6" or v.get("n_runs", v.get("n_units", 0)) >= EXPECTED.get(b, 10**9)}
if not {"smd", "nasa"} <= set(NUMBERS):
    NUMBERS.pop("h6", None)
EXTRA = {}
if PAPERNUM.exists():
    P = json.loads(PAPERNUM.read_text(encoding="utf-8"))
    EXTRA["halo"] = P.get("halo", {})
    EXTRA["caidas"] = {b: [{k: r[k] for k in ("method", "drop") if k in r} for r in P.get(f"{b}_drops", [])]
                       for b in ("contam", "semi")}
REALCSV = ROOT / "results" / "final" / "real_eval.csv"
if REALCSV.exists() and "real" in NUMBERS:
    acc = {}
    with REALCSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cfg"] == "L4P1" and r["train"] == "adapt_r3" and r["proj"] == "tgt":
                a = acc.setdefault(r["ds"], [0.0, 0.0, 0])
                a[0] += float(r["anom_rate"]); a[1] += float(r["iqr_rate"]); a[2] += 1
    EXTRA["real_iqr"] = {ds: {"anom": a / k, "iqr": b / k} for ds, (a, b, k) in acc.items()}
if H6CSV.exists() and "h6" in NUMBERS:
    with H6CSV.open(encoding="utf-8") as f:
        EXTRA["h6_series"] = [{k: (v if k in ("ds", "block") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
print("bloques completos incrustados:", list(NUMBERS) or "ninguno")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t, title=None, hidden=False):
    src = (f"# @title {title}\n" if title else "") + t.strip("\n")
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src})


# =====================================================================================
# PORTADA
# =====================================================================================
md(r"""
# Detección de anomalías con TracIn, explicada desde cero
## Qué hizo el Equipo 6, por qué, cómo está programado y qué tan bien funciona

**Equipo 6 · Aprendizaje no supervisado · INAOE**

Este cuaderno es para alguien que **no ha leído el artículo** ni nuestro código. No da nada por sabido: cada
concepto se explica con palabras simples, con un ejemplo de juguete y con el código real que usamos. Al
terminarlo deberías poder:

1. Explicar con tus palabras cómo detecta anomalías el artículo (Verdone et al., 2026, *TracIn* para series de tiempo).
2. Explicar cada una de nuestras **tres mejoras**: qué problema ataca, cómo lo resuelve y en qué líneas de código está.
3. Explicar el **método final** (las tres juntas) y **en qué casos funciona y en cuáles no**.
4. Entender cómo lo evaluamos (pruebas estadísticas incluidas) y leer todas las tablas de resultados.
5. Correrlo tú mismo en un dataset real o en tu propio archivo CSV.

---

### Cómo usar este cuaderno

* Ve al menú **Entorno de ejecución → Ejecutar todo** (o `Ctrl + F9`). No necesitas GPU ni subir archivos.
* Todo el cuaderno tarda **unos 20–30 minutos** en la CPU gratuita de Colab. Mientras corre puedes ir leyendo:
  cada sección imprime lo que va haciendo.
* Las celdas con un título y ▶ (por ejemplo *"Código del método"*) están **plegadas** porque son largas. Puedes
  abrirlas si quieres ver el código completo, pero **no hace falta**: en el texto te explicamos las partes
  importantes línea por línea.
* Si algo falla, vuelve a ejecutar desde la sección 0.

### Mapa del cuaderno

| Parte | Sección | Qué vas a aprender |
|---|---|---|
| **A. Lo básico** | 0 | Preparar el entorno (sólo ejecutar) |
| | 1 | Conceptos básicos: serie de tiempo, anomalía, red neuronal, gradiente, época… con ejemplos de juguete |
| | 2 | Los datos: cómo se fabrican series con anomalías y qué tipos de anomalía existen |
| **B. El artículo** | 3 | El método del artículo paso a paso: ventanas → LSTM → TracIn → proyección → umbral → métricas |
| **C. Nuestras mejoras** | 4 | **Mejora 1 (M1):** un umbral que no necesita etiquetas |
| | 5 | **Mejora 2 (M2):** quitar lo sospechoso del entrenamiento y volver a entrenar |
| | 6 | **Mejora 3 (M3):** proyección al objetivo, el "halo" y por qué importa tanto |
| | 7 | **Método final = M1 + M2 + M3**, y un laboratorio interactivo |
| **D. ¿Funciona?** | 8 | Cómo se evalúa bien: protocolo, hipótesis, pruebas estadísticas (con un mini-experimento en vivo) |
| | 9 | **Todos los resultados** de la evaluación grande (1 755 corridas), también los negativos |
| **E. Úsalo tú** | 10 | Pruébalo en datos reales (SMD, descargados en vivo) y en tu propio CSV |
| | 11 | Cómo está organizado el código del proyecto y cómo reproducir los experimentos |
| | 12 | Resumen, glosario y preguntas que te pueden hacer (con respuestas) |

### La historia del proyecto en un minuto

1. **El artículo** propone detectar anomalías en series de tiempo así: se entrena una red neuronal (LSTM) a
   predecir la serie y se mide **cuánto le costó aprender cada pedazo** (eso es *TracIn*). Lo que más cuesta
   aprender es lo raro.
2. **Reprodujimos** el artículo y encontramos tres puntos débiles:
   * su mejor umbral **usa las respuestas** (las etiquetas), cosa que en un problema real no se tiene;
   * cuando hay **muchas anomalías** (datos "contaminados"), la red las aprende y deja de verlas como raras;
   * su fórmula para pasar de "score por ventana" a "score por instante" **reparte la culpa** de una anomalía
     entre sus vecinos normales.
3. Propusimos **una mejora para cada punto** (M1, M2, M3) y las **combinamos** en un método final que no usa
   ninguna etiqueta.
4. Lo evaluamos con un **protocolo escrito antes de correr** (para no hacer trampa sin querer), en 6 bloques de
   datos, con pruebas estadísticas.
5. **Resultado:** la combinación funciona muy bien con series contaminadas y anomalías cortas, y explicamos
   **por qué** (el "halo"). Pero **no** funciona en benchmarks donde las anomalías son tramos muy largos (SMD,
   SMAP/MSL). Lo reportamos tal cual.
""")

# =====================================================================================
# 0. PREPARACIÓN
# =====================================================================================
md(r"""
# PARTE A · Lo básico

## 0. Preparación

Ejecuta las celdas de esta sección y sigue adelante. Qué hace cada una:

1. **Librerías.** Carga las herramientas que usamos:
   * `numpy` (cálculo con arreglos de números), `pandas` (tablas), `matplotlib` (gráficas);
   * `torch` (**PyTorch**, para la red neuronal);
   * `scipy` (pruebas estadísticas) y `sklearn` (métricas como AUPRC y AUROC).
2. **Generador de series sintéticas.** Es el mismo generador que usa el artículo (de Lai et al., 2021). Fabrica
   series con anomalías de tipos conocidos, así sabemos exactamente dónde están.
3. **Código del método.** La red LSTM y el cálculo de TracIn, **copiados literalmente** de nuestro archivo
   `src/replica/lab.py`. No es una versión simplificada: es exactamente lo que corrimos.
4. **Código de las mejoras.** Las funciones de proyección, umbral IQR, reponderación y recorte (de `src/common.py`,
   `src/forecaster.py` y `src/final/core.py`), y el inyector de anomalías sobre datos reales (`src/data_bench.py`).
5. **Funciones auxiliares** para dibujar y medir.
""")
code(r'''
import time, warnings, math, io, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from IPython.display import display, clear_output, Markdown
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3)
pd.set_option("display.max_colwidth", 120)
torch.set_num_threads(max(1, torch.get_num_threads()))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
print("Listo. PyTorch", torch.__version__)
''', title="1. Librerías")
code("from __future__ import annotations\n" + syn_code,
     title="2. Generador de series sintéticas (el mismo benchmark del artículo, Lai et al. 2021)", hidden=True)
code("from __future__ import annotations\nimport math\nfrom dataclasses import dataclass, asdict\nfrom torch import nn\n"
     "from torch.func import functional_call, grad, vmap\n\n" + lab_code,
     title="3. Código del método: LSTM + TracIn (copiado de src/replica/lab.py)", hidden=True)
code("from __future__ import annotations\n\n" + common_code + "\n\n\n" + rw_code + "\n\n\n" + adapt_code + "\n\n\n" + inject_code,
     title="4. Código de las mejoras (copiado de src/common.py, src/forecaster.py, src/final/core.py, src/data_bench.py)",
     hidden=True)
code(r'''
TIPOS = {"0": "puntual global", "1": "puntual contextual", "2": "de forma (shapelet)",
         "3": "estacional", "4": "de tendencia"}
CFG = Cfg()           # configuración que reproduce al artículo en sintéticos (datos estandarizados, última capa)
L, P = 4, 1           # el LSTM ve 4 instantes y predice el siguiente


def intervalos(y):
    """Tramos [inicio, fin) donde y == 1."""
    d = np.diff(np.r_[0, np.asarray(y).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def sombrear(ax, y, color="#f4a3a3", alpha=0.35, label="anomalía real"):
    for i, (a, b) in enumerate(intervalos(y)):
        ax.axvspan(a - 0.5, b - 0.5, color=color, alpha=alpha, lw=0, label=label if i == 0 else None)


def dibujar_serie(X, y, titulo="", ax=None, max_canales=6):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))
    Xn = normalize(X.astype(np.float64), "z")
    for c in range(min(Xn.shape[1], max_canales)):
        ax.plot(Xn[:, c] - 4 * c, lw=0.8, label=f"canal {c}")
    sombrear(ax, y)
    ax.set_yticks([]); ax.set_xlabel("tiempo"); ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=7, ncol=7)
    return ax


def proyectar(s_ventana, n, objetivo):
    """Score por ventana -> score por instante.
    objetivo=False: Ec. 4 del artículo (suma sobre los L+P instantes de la ventana).
    objetivo=True : mejora 3 (sólo el instante predicho)."""
    return project_targets_mean(s_ventana, L, P, n) if objetivo else project_sum(s_ventana, L + P, n)


def f1(y, pred):
    y = np.asarray(y).astype(bool); pred = np.asarray(pred).astype(bool)
    tp = np.sum(y & pred); fp = np.sum(~y & pred); fn = np.sum(y & ~pred)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def umbral_dinamico(y, s):
    """Umbral del artículo: elige, CON ETIQUETAS, el percentil que maximiza el F1 en la 1.ª mitad."""
    k = len(s) // 2
    mejor, thr = -1, None
    for p in np.arange(50, 99.95, 0.1):
        t = np.percentile(s, p)
        f = f1(y[:k], s[:k] >= t)
        if f > mejor:
            mejor, thr = f, t
    return thr


def evaluar(y, s, thr):
    """Métricas en los instantes válidos. El F1 se mide en la 2.ª mitad (la 1.ª la usa el umbral dinámico)."""
    m = valid_mask(len(y)); y, s = np.asarray(y)[m].astype(bool), s[m]
    k = len(s) // 2
    pred = s[k:] >= thr
    tp = np.sum(y[k:] & pred)
    return {"AUPRC": average_precision_score(y, s), "AUROC": roc_auc_score(y, s),
            "% marcado": pred.mean(), "precisión": tp / max(pred.sum(), 1),
            "exhaustividad": tp / max(y[k:].sum(), 1), "F1": f1(y[k:], pred)}


def dibujar_deteccion(y, s, thr, titulo, ax):
    ax.plot(s, color="#3182bd", lw=0.9, label="score")
    sombrear(ax, y)
    ax.axhline(thr, color="k", ls="--", lw=1, label="umbral")
    y = np.asarray(y).astype(bool); pred = s >= thr
    ax.plot(np.where(pred & y)[0], s[pred & y], "o", ms=3, color="#31a354", label="acierto")
    ax.plot(np.where(pred & ~y)[0], s[pred & ~y], "x", ms=4, color="#e6550d", label="falsa alarma")
    ax.axvline(len(y) // 2, color="gray", lw=0.8); ax.set_title(titulo); ax.legend(loc="upper right", fontsize=7, ncol=5)


print("Funciones auxiliares listas.")
''', title="5. Funciones auxiliares para dibujar y medir")

# =====================================================================================
# 1. CONCEPTOS BÁSICOS
# =====================================================================================
md(r"""
## 1. Conceptos básicos (si ya los sabes, sáltate esta sección)

### 1.1 Serie de tiempo
Una **serie de tiempo** es una lista de mediciones tomadas una tras otra en el tiempo: la temperatura cada hora,
el uso de CPU de un servidor cada minuto, las transacciones de una tarjeta… Si en cada instante se miden
**varias cosas a la vez** (varios sensores), es **multivariada**; cada sensor es un **canal**.

En el código, una serie es una matriz `X` de tamaño *(instantes × canales)*. Por ejemplo, `X[100, 2]` es lo que
midió el canal 2 en el instante 100.

### 1.2 Anomalía
Una **anomalía** es un instante o un tramo que **no se comporta como el resto**: un pico imposible, un sensor que
se queda trabado, un patrón que de pronto cambia de forma. Las **etiquetas** `y` dicen cuáles instantes son
anómalos (`y[t] = 1`) y cuáles normales (`y[t] = 0`).

### 1.3 No supervisado
En la vida real casi nunca tenemos etiquetas: nadie revisó la serie instante por instante. Un método **no
supervisado** tiene que encontrar las anomalías **sin ver nunca las etiquetas**. Las etiquetas sólo se usan **al
final**, para calificar qué tan bien lo hizo (como la hoja de respuestas de un examen que el alumno no ve).

> ⚠️ Esto es importante para todo el proyecto: si un método usa las etiquetas para tomar alguna decisión, ya no es
> totalmente no supervisado. El artículo lo hace en un paso (el umbral) y nosotros lo quitamos (Mejora 1).

### 1.4 Contaminación
Llamamos **contaminación** a la **proporción de anomalías** dentro de los datos. Con 1 % de contaminación hay 1
instante raro de cada 100; con 30 % hay casi uno de cada tres. Mucha contaminación es un problema porque el
modelo se entrena con esos mismos datos y puede acabar **aprendiendo las anomalías como si fueran normales**.

### 1.5 Red neuronal, entrenamiento, gradiente, época, tasa de aprendizaje
* Una **red neuronal** es una función con muchos números ajustables (**parámetros**). Aquí la usamos para
  **predecir** el siguiente valor de la serie a partir de los anteriores.
* **Entrenar** es ajustar esos parámetros para que el **error de predicción** (la **pérdida**) sea pequeño.
* El **gradiente** dice en qué dirección y cuánto mover cada parámetro para reducir el error **de un ejemplo**.
  Un gradiente **grande** significa "este ejemplo está muy mal predicho; me obliga a cambiar mucho".
* Una **época** es una pasada completa por todos los datos. Se entrenan muchas épocas.
* La **tasa de aprendizaje** (η, *learning rate*) es el tamaño del paso que se da en cada ajuste.

### 1.6 La idea de TracIn con un ejemplo de juguete
TracIn (Pruthi et al., 2020) mide **cuánto "empujó" cada ejemplo al modelo durante todo el entrenamiento**: suma,
época por época, el tamaño del gradiente de ese ejemplo (al cuadrado) multiplicado por la tasa de aprendizaje.
Los ejemplos que el modelo **no logra aprender** (los raros) empujan mucho, época tras época, y acumulan un
valor alto.

Vamos a verlo con el modelo más simple posible: **una recta** `ŷ = a·x + b` que se ajusta a 40 puntos. Tres
puntos están "fuera de lugar" (son las anomalías). Entrenamos con descenso de gradiente y vamos sumando, para
cada punto, `η · ‖gradiente‖²`. **No le decimos al modelo cuáles son los raros.**
""")
code(r'''
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 40)
yv = 2 * x + 1 + rng.normal(0, 0.05, 40)          # datos normales: una recta con un poco de ruido
raros = [8, 25, 33]
yv[raros] += [1.2, -1.0, 0.9]                      # tres puntos anómalos

a, b, eta = 0.0, 0.0, 0.1                          # parámetros de la recta y tasa de aprendizaje
esfuerzo = np.zeros(40)                            # aquí acumulamos la "self-influence" de cada punto
for epoca in range(200):
    err = (a * x + b) - yv                         # error de cada punto
    grad_a, grad_b = 2 * err * x, 2 * err          # gradiente del error al cuadrado, punto por punto
    esfuerzo += eta * (grad_a ** 2 + grad_b ** 2)  # TracIn: suma de η·‖gradiente‖² a lo largo de las épocas
    a -= eta * grad_a.mean(); b -= eta * grad_b.mean()   # paso de entrenamiento normal

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.4))
a1.scatter(x, yv, c=["#de2d26" if i in raros else "#3182bd" for i in range(40)])
a1.plot(x, a * x + b, "k--", label=f"recta aprendida: {a:.2f}·x + {b:.2f}")
a1.set_title("Datos (rojo = anómalos, el modelo NO lo sabe)"); a1.legend()
a2.bar(range(40), esfuerzo, color=["#de2d26" if i in raros else "#9ecae1" for i in range(40)])
a2.set_title("Esfuerzo acumulado (TracIn) de cada punto"); a2.set_xlabel("punto")
plt.tight_layout(); plt.show()
top3 = np.argsort(esfuerzo)[-3:][::-1]
print("Los 3 puntos con más esfuerzo son:", sorted(top3.tolist()), " | los anómalos eran:", raros)
''')
md(r"""
**Qué acabas de ver.** Sin saber cuáles eran los raros, los tres puntos con mayor esfuerzo acumulado son
exactamente los anómalos. **Ésa es toda la idea del artículo**, sólo que en lugar de una recta usa una red LSTM
y en lugar de puntos sueltos usa *ventanas* de una serie de tiempo.

Fíjate también en algo que será importante en la Mejora 2: los puntos raros **jalan la recta** hacia ellos (la
recta aprendida no es exactamente `2x + 1`). Si hubiera muchos puntos raros, la recta se movería tanto que ya no
parecerían raros. Eso es la **contaminación**.
""")

# =====================================================================================
# 2. DATOS
# =====================================================================================
md(r"""
## 2. Los datos: cómo se fabrica una serie con anomalías

Para saber si un detector funciona necesitamos series donde **sepamos exactamente** dónde están las anomalías.
El artículo usa el generador de **Lai et al. (2021)**, que fabrica series de 5 canales con ondas (senos, cosenos,
ondas cuadradas) más un poco de ruido, y **les inyecta** anomalías de cinco tipos. En nuestro código el canal
`t` recibe la anomalía de tipo `t`:

| Tipo | Nombre | Qué es | Familia |
|---|---|---|---|
| 0 | Puntual global | Un solo instante con un valor fuera del rango normal (un pico enorme) | **Puntual** |
| 1 | Puntual contextual | Un solo instante con un valor posible en general, pero raro **para ese momento** | **Puntual** |
| 2 | De forma (*shapelet*) | Un tramo donde la forma de la onda cambia | **De patrón** |
| 3 | Estacional | Un tramo donde el ritmo (la frecuencia) cambia | **De patrón** |
| 4 | De tendencia | Un tramo donde se suma una pendiente | **De patrón** |

Las **puntuales** duran **un instante**; las **de patrón** duran **un tramo** (unos 10 instantes). Esta
diferencia va a ser **clave** para entender dónde funciona nuestra Mejora 3 y dónde no.

La función que usamos es `make_dataset(tipos, semilla, length, ratio)`:
* `tipos`: qué tipos de anomalía meter, como texto (`"01234"` = todos, `"01"` = sólo puntuales, `"234"` = sólo de patrón);
* `semilla`: número que fija el azar (misma semilla → misma serie, así los experimentos son repetibles);
* `length`: largo de la serie (400 en el benchmark del artículo);
* `ratio`: proporción de anomalías **por tipo**.

Veamos cada tipo por separado y luego todos juntos.
""")
code(r'''
fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
for t, ax in zip("01234", axes):
    Xt, yt = make_dataset(t, seed=3)
    Xn = normalize(Xt.astype(np.float64), "z")
    ax.plot(Xn[:, int(t)], lw=0.9, color="k")
    sombrear(ax, yt)
    ax.set_title(f"Tipo {t}: {TIPOS[t]} (canal {t})", fontsize=9); ax.set_yticks([])
axes[-1].set_xlabel("tiempo")
plt.tight_layout(); plt.show()
''')
code(r'''
X, y = make_dataset("01234", seed=3)
print(f"Serie de {len(y)} instantes y {X.shape[1]} canales; {y.mean():.1%} de los instantes son anómalos.")
print("Forma de X:", X.shape, "  · primeras 3 filas:")
print(np.round(X[:3], 3))
print("Etiquetas y (primeros 40 instantes):", y[:40])
dibujar_serie(X, y, "Serie sintética con los 5 tipos de anomalía (franjas rojas = anomalías reales)")
plt.show()
''')
md(r"""
**Para llevarte:** las franjas rojas son las respuestas correctas. **El algoritmo nunca las ve.** Sólo las usamos
al final para calificar.
""")

# =====================================================================================
# 3. EL ARTÍCULO
# =====================================================================================
md(r"""
# PARTE B · El método del artículo

## 3. El método del artículo, paso a paso

El método tiene **cinco pasos**. Los vamos a recorrer uno por uno, con el código real:

```
 serie X ──► (3.1) cortar en ventanas ──► (3.2) entrenar LSTM a predecir ──► (3.3) TracIn: esfuerzo por ventana
                                                                                        │
                       anomalías ◄── (3.5) umbral ◄── (3.4) pasar de ventanas a instantes (Ec. 4)
```

### 3.1 Ventanas
La red recibe **L = 4** instantes seguidos y debe predecir el **siguiente** (P = 1). Cada pedazo de
L + P = 5 instantes es una **ventana**. Hay una ventana por cada posición: la ventana `i` usa los instantes
`i, i+1, i+2, i+3` como entrada y predice el instante `i+4` (su **objetivo**).

En el código (`windows` en `lab.py`):
```python
def windows(Xt, idx, L, P):
    ar = torch.arange(L + P)          # [0, 1, 2, 3, 4]
    w = Xt[idx[:, None] + ar]         # para cada inicio i, toma los instantes i..i+4
    return w[:, :L], w[:, L:]         # entrada (4 instantes) y objetivo (1 instante)
```
""")
code(r'''
fig, ax = plt.subplots(figsize=(12, 2.6))
Xn = normalize(X.astype(np.float64), "z")
t0 = 60
ax.plot(np.arange(40, 100), Xn[40:100, 0], "k.-", lw=0.8, ms=4)
ax.axvspan(t0 - 0.5, t0 + L - 0.5, color="#9ecae1", alpha=0.6, label="entrada (L = 4 instantes)")
ax.axvspan(t0 + L - 0.5, t0 + L + P - 0.5, color="#fdae6b", alpha=0.8, label="objetivo: lo que debe predecir (P = 1)")
ax.set_title(f"La ventana que empieza en t = {t0}: el LSTM ve los instantes azules y predice el naranja")
ax.legend(loc="upper right")
plt.show()
n = len(y)
print(f"Con n = {n} instantes hay M = n - L - P + 1 = {n - L - P + 1} ventanas.")
print("Cada instante (salvo los de las orillas) pertenece a L + P = 5 ventanas distintas. Esto importará en la Mejora 3.")
''')
md(r"""
### 3.2 La red: un LSTM que predice el siguiente instante
Un **LSTM** (*Long Short-Term Memory*) es una red neuronal hecha para secuencias: lee los instantes de la ventana
**uno por uno** y va guardando una "memoria" (`h` y `c`) de lo que ha visto. Al final, una capa lineal (`head`)
convierte esa memoria en la predicción del siguiente instante.

Nuestro `LSTM` (en `lab.py`) está escrito a mano para poder calcular gradientes por ventana de forma eficiente,
pero hace exactamente lo mismo que el `nn.LSTM` de PyTorch (lo verificamos con una prueba automática). Lo esencial:

```python
for t in range(x.shape[1]):                          # recorre los L = 4 instantes de la ventana
    i, f, g, o = (gx[:, t] + self.hh(h)).chunk(4, dim=1)   # cuatro "compuertas"
    c = sigmoid(f) * c + sigmoid(i) * tanh(g)        # actualiza la memoria (olvida y agrega)
    h = sigmoid(o) * tanh(c)                         # salida de este paso
return self.head(h)                                  # predicción del instante siguiente
```

La pérdida de una ventana es el **error cuadrático medio** entre la predicción y el valor real del objetivo.

### 3.3 TracIn: el "esfuerzo" de cada ventana
Al final de **cada época** `k` calculamos, para **cada ventana** `x`, el tamaño de su gradiente al cuadrado y lo
multiplicamos por la tasa de aprendizaje de esa época. Sumando sobre las épocas obtenemos la **self-influence**:

$$\text{TracIn}(x) = \sum_{\text{épocas } k} \eta_k \, \lVert \nabla_\theta \ell_k(x) \rVert^2$$

Leído término a término:
* $\ell_k(x)$: el error de la ventana $x$ con el modelo como estaba al final de la época $k$;
* $\nabla_\theta \ell_k(x)$: el gradiente de ese error respecto a los parámetros $\theta$ (cuánto "jala" esa ventana al modelo);
* $\lVert\cdot\rVert^2$: su tamaño al cuadrado;
* $\eta_k$: la tasa de aprendizaje en esa época.

En el código (función `run` de `lab.py`), dentro del ciclo de épocas:
```python
l_all, h_all = _stats(model, Xt, M, L, P)       # error y memoria final de TODAS las ventanas
if cfg.grad == "last":                          # gradiente sólo de la última capa (fórmula cerrada, rápido)
    g2 = (4.0 / D) * l_all * (1 + h_all)
else:                                           # gradiente de TODOS los parámetros (vmap, más lento)
    g2 = _per_sample_sqgrad(model, Xt, M, L, P)
si += lr_now * g2                               # acumula η_k · ‖∇ℓ‖²  → self-influence
```
* En los **sintéticos** el artículo usa el gradiente de la **última capa** (hay una fórmula exacta y rápida).
* En los **datos reales** usamos el gradiente de **todos los parámetros**, como en el artículo.

Además, `run` usa **Adam** como optimizador, baja la tasa de aprendizaje si el error se estanca
(`ReduceLROnPlateau`) y para el entrenamiento si no mejora en 10 épocas (*early stopping*).

**Ejecuta la celda.** A la izquierda verás bajar el error de predicción. A la derecha se va acumulando la
self-influence, ya convertida a instantes. Los picos deberían ir apareciendo sobre las franjas rojas.
""")
code(r'''
historia = []
def en_vivo(epoca, si, perdida):
    historia.append(float(perdida.mean()))
    if epoca % 10 and epoca > 3:
        return
    clear_output(wait=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1, 3]})
    a1.plot(historia, "k-"); a1.set_title("error de predicción promedio"); a1.set_xlabel("época"); a1.set_yscale("log")
    a2.plot(proyectar(si, len(y), objetivo=False), color="#3182bd", lw=0.9)
    sombrear(a2, y); a2.set_title(f"self-influence acumulada tras {epoca} épocas"); a2.set_xlabel("tiempo"); a2.legend(loc="upper right")
    plt.show()

t = time.time()
r = run(X, L, P, seed=0, cfg=CFG, callback=en_vivo)
print(f"Entrenamiento terminado: {r['epochs']} épocas en {time.time() - t:.0f} s.")
print("r['si'] tiene un número por VENTANA:", r["si"].shape, "· primeros valores:", np.round(r["si"][:5], 4))
''')
md(r"""
### 3.4 De "esfuerzo por ventana" a "score por instante": la Ecuación 4 del artículo
TracIn da **un número por ventana**, pero queremos saber **qué instantes** son anómalos. El artículo (Ec. 4) le da a
cada instante la **suma** de los scores de **todas las ventanas que lo contienen** (hasta 5 ventanas).

Veámoslo con un ejemplo de juguete: una serie de 12 instantes (8 ventanas). Todas las ventanas tienen score 1,
salvo la ventana 4, que tiene score 9 (porque su objetivo, el instante 8, es una anomalía).
""")
code(r'''
n_j = 12
s_j = np.array([1, 1, 1, 1, 9, 1, 1, 1], dtype=float)         # score de las 8 ventanas
tabla = pd.DataFrame({"instante": range(n_j),
                      "ventanas que lo contienen": [", ".join(str(i) for i in range(len(s_j)) if i <= t < i + L + P) for t in range(n_j)],
                      "score Ec. 4 (suma)": project_sum(s_j, L + P, n_j)})
display(tabla.set_index("instante"))
print("La ventana 4 cubre los instantes 4, 5, 6, 7 y 8 (entrada 4-7, objetivo 8).")
print("Con la Ec. 4, su score alto se reparte a LOS CINCO instantes: 4, 5, 6 y 7 son normales pero quedan altos.")
''')
md(r"""
En el código, `project_sum(s, span, n)` hace esa suma de forma eficiente (con sumas acumuladas en lugar de dos
ciclos anidados):
```python
acc = np.zeros(n + 1)
acc[:m] += s              # cada ventana i "empieza" a sumar en el instante i
acc[span:span + m] -= s   # ...y "deja" de sumar en el instante i + span
return np.cumsum(acc)[:n] # la suma acumulada da, en cada instante, la suma de las ventanas que lo cubren
```
Guarda este ejemplo: **es el corazón de la Mejora 3.**

### 3.5 El umbral: ¿a partir de qué score decimos "anómalo"?
Hay que elegir un corte. El artículo usa dos:
* **Estático:** un percentil fijo (el 85 en sintéticos, el 87 en reales). Siempre marca el 13–15 % de los instantes.
* **Dinámico:** prueba muchos cortes y se queda con el que da **el mejor F1 en la primera mitad de la serie,
  mirando las etiquetas**. Después lo aplica a la segunda mitad. Es el que mejor le funciona, pero **usa las
  respuestas**.

Por eso, para comparar de forma justa, **todos los F1 se miden sólo en la segunda mitad** (la línea gris en las
gráficas).

### 3.6 Cómo se califica: AUPRC, AUROC y F1 (con ejemplo de juguete)
* **AUROC** (0.5 = azar, 1 = perfecto): la probabilidad de que un instante anómalo elegido al azar tenga **más score**
  que uno normal elegido al azar. **No depende del umbral.**
* **AUPRC** (de la proporción de anomalías hasta 1): qué tan "arriba" quedan las anomalías al ordenar por score,
  dando más peso a los primeros lugares. Es más exigente que el AUROC cuando hay pocas anomalías. **No depende del
  umbral.** Mide la **calidad del score**.
* **F1** (0 a 1): con un umbral dado, combina **precisión** (de lo que marqué, cuánto era anómalo de verdad) y
  **exhaustividad** (de lo anómalo, cuánto encontré). **Sí depende del umbral.**
""")
code(r'''
y_j  = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
sc_j = np.array([.1, .3, .2, .9, .4, .8, .7, .1, .2, .3])
pred = sc_j >= 0.6
print("etiquetas :", y_j)
print("scores    :", sc_j)
print("marcado   :", pred.astype(int), "(umbral 0.6)")
tp = int((pred & (y_j == 1)).sum()); fp = int((pred & (y_j == 0)).sum()); fn = int((~pred & (y_j == 1)).sum())
print(f"aciertos = {tp}, falsas alarmas = {fp}, anomalías perdidas = {fn}")
print(f"precisión = {tp}/{tp + fp} = {tp / (tp + fp):.2f}  · exhaustividad = {tp}/{tp + fn} = {tp / (tp + fn):.2f}  · F1 = {f1(y_j, pred):.2f}")
print(f"AUROC = {roc_auc_score(y_j, sc_j):.3f} (hay un normal, el 0.8, por encima de una anomalía, el 0.7)")
print(f"AUPRC = {average_precision_score(y_j, sc_j):.3f}")
''')
md(r"""
### 3.7 Todo junto: el método del artículo sobre nuestra serie
""")
code(r'''
s_art = proyectar(r["si"], len(y), objetivo=False)
m = valid_mask(len(y))                       # se excluyen 19 instantes en cada orilla (ahí hay menos ventanas)
thr_art = umbral_dinamico(np.asarray(y)[m].astype(bool), s_art[m])   # usa las etiquetas de la 1.ª mitad
res_art = evaluar(y, s_art, thr_art)
fig, ax = plt.subplots(figsize=(12, 3.2))
dibujar_deteccion(y, s_art, thr_art, "Método del artículo: score y detecciones (el F1 se mide a la derecha de la línea gris)", ax)
plt.show()
display(pd.DataFrame([res_art], index=["Artículo (umbral dinámico, con etiquetas)"]).round(3))
''')

# =====================================================================================
# 4. MEJORA 1
# =====================================================================================
md(r"""
# PARTE C · Nuestras tres mejoras

## 4. Mejora 1 (M1) · Un umbral que **no** necesita etiquetas

### El problema
El umbral dinámico del artículo se calibra **viendo las respuestas**. En un problema realmente no supervisado esas
etiquetas no existen. Es como fijar la calificación mínima para aprobar **después** de ver quién sacó qué.

### La idea
Usamos la regla clásica del **diagrama de caja** (*boxplot*): es atípico lo que queda por encima de

$$\text{umbral} = Q_3 + 1.5 \times (Q_3 - Q_1)$$

* $Q_1$ es el valor que deja por debajo al 25 % de los scores; $Q_3$, el que deja por debajo al 75 %.
* $Q_3 - Q_1$ se llama **rango intercuartil (IQR)** y mide qué tan dispersos están los scores "típicos".
* El 1.5 es el valor estándar en estadística (Tukey, 1977). **Lo fijamos antes de ver resultados** y nunca lo ajustamos.

Sólo usa el score, **nunca** las etiquetas. Y se adapta solo: si hay más anomalías, la cola de scores altos crece y
el umbral cambia.

### El código (`thr_iqr` en `src/common.py`)
```python
def thr_iqr(s, k=1.5):
    q1, q3 = np.percentile(s, [25, 75])   # cuartiles del score
    return float(q3 + k * (q3 - q1))      # Q3 + 1.5 · IQR
```
Son dos líneas: la mejora es conceptual (quitar las etiquetas del proceso), no de programación.

### Ejemplo de juguete
""")
code(r'''
sc_j = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 15, 18], dtype=float)
q1, q3 = np.percentile(sc_j, [25, 75])
print(f"scores = {sc_j.tolist()}")
print(f"Q1 = {q1}, Q3 = {q3}, IQR = Q3 - Q1 = {q3 - q1}")
print(f"umbral = Q3 + 1.5·IQR = {q3} + 1.5·{q3 - q1} = {thr_iqr(sc_j)}")
print("Se marcan como anómalos:", sc_j[sc_j > thr_iqr(sc_j)].tolist())
''')
md(r"""
### En nuestra serie
""")
code(r'''
m = valid_mask(len(y))
thr_m1 = thr_iqr(s_art[m])
res_m1 = evaluar(y, s_art, thr_m1)
fig, ax = plt.subplots(figsize=(11, 3))
ax.hist(s_art[m][np.asarray(y)[m] == 0], bins=80, alpha=0.6, label="instantes normales", color="#9ecae1")
ax.hist(s_art[m][np.asarray(y)[m] == 1], bins=80, alpha=0.7, label="instantes anómalos", color="#fb6a4a")
ax.axvline(thr_art, color="k", ls="--", label="umbral dinámico (usa etiquetas)")
ax.axvline(thr_m1, color="#31a354", lw=2, label="umbral IQR (mejora 1, sin etiquetas)")
ax.set_yscale("log"); ax.set_xlabel("score"); ax.legend(fontsize=8); ax.set_title("¿Dónde cortar?")
plt.show()
display(pd.DataFrame([res_art, res_m1], index=["Artículo · dinámico (CON etiquetas)", "Mejora 1 · IQR (SIN etiquetas)"]).round(3))
print("Nota: el AUPRC y el AUROC no cambian (el score es el mismo); sólo cambia dónde se corta.")
print(f"En esta serie hay {np.mean(y):.0%} de instantes anómalos y el IQR marca {res_m1['% marcado']:.0%}.")
''')
md(r"""
**¿Por qué aquí M1 sola no le gana al umbral con etiquetas?** El umbral IQR se pone donde la distribución del score
dice que empiezan los valores atípicos. Si el score es malo (con el halo del artículo, muchos normales tienen score
alto y muchas anomalías no), ningún umbral sin etiquetas puede hacer milagros. Sobre un score mejor (con M3), el
IQR sí rinde. En la evaluación grande de sintéticos (300 corridas), el F1 fue:
* artículo con etiquetas: 0.469;
* M1 sobre el score del artículo: 0.481;
* M1 sobre el score de M3: **0.665**.

**M1 vale sobre todo en combinación.**

### ⚠️ Cuándo **no** funciona bien la Mejora 1
Cuando las anomalías son **muy raras** (menos del 2 % de los instantes) y el score tiene **colas pesadas** (algunos
normales con valores muy altos, como pasa en datos reales), la regla IQR marca **demasiado**. Muchas falsas
alarmas → precisión baja → F1 bajo. El umbral dinámico del artículo no tiene ese problema porque **mira las
respuestas**. Es el precio de no usar etiquetas, y lo reportamos (ver sección 9).

En las series sintéticas esto casi no se nota. Por eso aquí mostramos lo que pasó en los **4 datasets reales del
artículo**, en la evaluación grande (método final, promedio de 10 semillas):
""")
code("REAL_IQR = " + json.dumps(EXTRA.get("real_iqr", {})) + r'''
if REAL_IQR:
    t_ = pd.DataFrame(REAL_IQR).T.rename(columns={"anom": "% real de anomalías", "iqr": "% que marca el IQR"})
    display((100 * t_).round(1))
    print("En Credit Card, CICIDS y GECCO las anomalías son menos del 1.3 %, pero el IQR marca entre 12 y 17 %:")
    print("aunque encuentre las anomalías, la mayoría de lo marcado son falsas alarmas.")
''')

# =====================================================================================
# 5. MEJORA 2
# =====================================================================================
md(r"""
## 5. Mejora 2 (M2) · Quitar lo sospechoso del entrenamiento y volver a entrenar

### El problema: el modelo "memoriza" las anomalías
¿Recuerdas la recta del ejemplo de juguete, que se movía hacia los puntos raros? Con una red neuronal pasa lo
mismo, y peor: la red es tan flexible que, si hay **muchas** anomalías (digamos 25 % de los datos), termina
**aprendiéndoselas**. Ya no le cuestan y su self-influence baja, así que dejan de destacar. El artículo menciona
esto para explicar por qué le va mal en SWAN-SF (24 % de anomalías), pero **no lo mide ni lo corrige**.

> **Analogía.** Si una cuarta parte del cuadernillo son ejercicios con trampa, el estudiante acaba memorizando las
> trampas y ya no le parecen raras.

Primero comprobemos que el problema existe: la misma clase de serie con 2 % y con 25 % de anomalías.
""")
code(r'''
def serie_contaminada(tipos="01234", contaminacion=0.25, semilla=0, largo=2000):
    """Serie larga con una proporción TOTAL `contaminacion` de anomalías (repartida entre los tipos)."""
    return make_dataset(tipos, 10_000 + semilla, length=largo, ratio=contaminacion / len(tipos))

filas = {}
for c in (0.02, 0.25):
    Xc, yc = serie_contaminada(contaminacion=c)
    rc = run(Xc, L, P, seed=0, cfg=CFG)
    sc = proyectar(rc["si"], len(yc), objetivo=False)
    mc = valid_mask(len(yc))
    filas[f"{yc.mean():.0%} de anomalías"] = {"AUROC del artículo": roc_auc_score(yc[mc], sc[mc]),
                                              "AUPRC del artículo": average_precision_score(yc[mc], sc[mc])}
    print(f"contaminación {yc.mean():.0%}: listo")
display(pd.DataFrame(filas).T.round(3))
print("Con muchas anomalías el método del artículo empeora. (En la sección 6 veremos que buena parte de")
print("esa caída NO es por memorización, sino por la proyección; pero la memorización también existe.)")
''')
md(r"""
### La idea: entrenar sólo con lo que parece normal
1. Entrenamos normal y calculamos la self-influence de cada ventana.
2. Las ventanas con self-influence **muy alta** (sospechosas de ser anómalas) se **quitan del entrenamiento**
   (peso 0) o se les **baja el peso**.
3. **Volvemos a entrenar desde cero** sin ellas y medimos la self-influence de **todas** las ventanas, **también de
   las quitadas**.
4. Repetimos varias rondas.

**Punto clave:** quitar una ventana del entrenamiento **no** la quita del análisis. Al contrario: como el modelo ya
no la ve al entrenar, no puede memorizarla, y cuando le calculamos el score **le cuesta más que antes**. Lo
anómalo resalta más.

### Las tres variantes que probamos
| Variante | Qué ventanas | Qué se les hace | Rondas |
|---|---|---|---|
| **Suave** (`soft`) | $z > 2$ | se les baja el peso gradualmente: $w = \max(0.05,\; e^{-(z-2)})$ | 2 |
| **Recorte** (`trim`) | $z > 2$ | peso 0 (se quitan) | 2 |
| **Recorte adaptativo** (`adapt`, la del método final) | las que pasan el **umbral IQR de la Mejora 1** | peso 0 | 3 |

Aquí $z$ es un **z-score robusto** del logaritmo de la self-influence: cuántas "desviaciones típicas" está cada
ventana por encima de la mediana, usando mediana y MAD en vez de media y desviación estándar (así las propias
anomalías no deforman la escala). Los valores 2, 0.05, 2 rondas y 3 rondas **se fijaron antes de evaluar**.

### El código
```python
def robust_z(v):                                   # src/forecaster.py
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))      # MAD escalada: equivale a la desv. estándar si fuera normal
    return (v - med) / max(mad, 1e-12)

def weights_from_influence(s, mode, z0=2.0, w_min=0.05):
    z = robust_z(np.log(np.maximum(s, 1e-30)))     # z robusto del log de la self-influence
    if mode == "soft":
        return np.where(z <= z0, 1.0, np.maximum(w_min, np.exp(-(z - z0))))
    if mode == "trim":
        return np.where(z <= z0, 1.0, 0.0)

def adapt_weights(si):                             # src/final/core.py  ← la del método final
    return np.where(si > thr_iqr(si), 0.0, 1.0)    # quita lo que pasa el umbral IQR (¡la Mejora 1!)
```
¿Y dónde se usan los pesos? En `run`, la pérdida de cada lote se pondera:
```python
loss = (l * wb).sum() / wb.sum()    # una ventana con peso 0 no aporta nada al entrenamiento
```

**¿Por qué el recorte adaptativo es el elegido?** Porque reutiliza el umbral IQR de la Mejora 1, y así **la
cantidad que se quita se ajusta sola**: si hay más anomalías, el umbral deja pasar más ventanas sospechosas y se
quitan más. Con un corte fijo $z > 2$, la cantidad quitada es casi constante.

### En vivo: tres rondas de recorte adaptativo con 25 % de anomalías
Las marcas moradas son las ventanas quitadas. La tabla dice qué fracción se quitó y **qué fracción de las
ventanas anómalas** quedó fuera (esto último usa las etiquetas, **sólo para diagnosticar**; el método no las ve).
""")
code(r'''
Xc, yc = serie_contaminada(contaminacion=0.25)
n = len(yc); mc = valid_mask(n)
M = n - L - P + 1
cum = np.r_[0, np.cumsum(yc.astype(int))]
ventana_anomala = (cum[np.arange(M) + L + P] - cum[np.arange(M)]) > 0   # sólo para el diagnóstico

r0 = run(Xc, L, P, seed=0, cfg=CFG)
s = r0["si"]
tabla = [{"ronda": 0, "ventanas quitadas": "0 %", "de las anómalas": "—", "de las normales": "—",
          "AUROC": roc_auc_score(yc[mc], proyectar(s, n, False)[mc])}]
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axes[0].plot(proyectar(s, n, False), lw=0.7); sombrear(axes[0], yc); axes[0].set_title("ronda 0 (artículo)")
for k in range(1, 4):
    w = adapt_weights(s)                       # mejora 1 + 2: quitar lo que pasa el umbral IQR
    s = run(Xc, L, P, seed=0, cfg=CFG, weights=w)["si"]
    tabla.append({"ronda": k, "ventanas quitadas": f"{(w == 0).mean():.0%}",
                  "de las anómalas": f"{(w[ventana_anomala] == 0).mean():.0%}",
                  "de las normales": f"{(w[~ventana_anomala] == 0).mean():.0%}",
                  "AUROC": roc_auc_score(yc[mc], proyectar(s, n, False)[mc])})
    axes[k].plot(proyectar(s, n, False), lw=0.7); sombrear(axes[k], yc)
    quitadas = np.where(w == 0)[0] + L
    axes[k].plot(quitadas, np.full(len(quitadas), axes[k].get_ylim()[1] * 0.95), "|", color="purple", ms=4, label="ventanas quitadas")
    axes[k].set_title(f"ronda {k}"); axes[k].legend(loc="upper left", fontsize=7)
    print(f"ronda {k} lista")
plt.tight_layout(); plt.show()
display(pd.DataFrame(tabla).round(3))
s_recorte = s          # lo guardamos para la sección 6
print("Lo que se quita es, en su mayoría, anómalo; y el AUROC sube ronda a ronda.")
''')
md(r"""
**Para llevarte:** la Mejora 2 sola ayuda **poco** cuando hay pocas anomalías (no hay mucho que memorizar), y
ayuda más cuando hay muchas. Pero su papel más importante aparece **combinada con la Mejora 3**, como veremos
en la sección 6.4.
""")

# =====================================================================================
# 6. MEJORA 3
# =====================================================================================
md(r"""
## 6. Mejora 3 (M3) · Proyección al objetivo: culpar al instante correcto

Ésta es **la mejora más importante del proyecto**, así que le dedicamos más espacio.

### 6.1 El problema
Vuelve al ejemplo de juguete de la sección 3.4. El error de una ventana se mide **sólo en el instante que predice**
(su objetivo). Pero la Ec. 4 del artículo reparte ese score entre **los 5 instantes** de la ventana. Resultado: una
anomalía "contagia" su score alto a los **4 instantes normales anteriores**, que aparecen como falsas alarmas.

> **Analogía.** Un jugador mete un autogol y la mala calificación se la ponen a los cinco que estaban en la cancha.

A esos instantes normales contagiados los llamamos el **halo** de la anomalía: son los normales que están a
distancia 1 a L + P − 1 = 4 de una anomalía.

### 6.2 La idea y el código
Cada ventana pasa su score **sólo al instante que predijo** (su objetivo). Si varias ventanas predicen el mismo
instante (cuando P > 1), se promedian. Ésta es la forma estándar en los detectores basados en predicción: **no la
inventamos nosotros**. Lo nuestro es usarla con TracIn y medir cuánto cambia y por qué.

```python
def project_targets_mean(s, L, P, n):      # src/common.py
    acc[L:L + m] += s                     # la ventana i aporta a partir del instante i + L (su objetivo)...
    acc[L + P:L + P + m] -= s             # ...y sólo hasta i + L + P (con P = 1, sólo el instante i + 4)
    cnt[L:L + m] += 1; cnt[L + P:L + P + m] -= 1   # cuántas ventanas predicen cada instante
    out = cumsum(acc) / cumsum(cnt)       # promedio
```
Es el mismo truco de sumas acumuladas de `project_sum`, pero cubriendo **sólo el objetivo**.

Comparemos las dos proyecciones con el ejemplo de juguete:
""")
code(r'''
tabla = pd.DataFrame({"score Ec. 4 (artículo)": project_sum(s_j, L + P, n_j),
                      "score M3 (al objetivo)": project_targets_mean(s_j, L, P, n_j)}, index=pd.Index(range(n_j), name="instante"))
display(tabla)
print("Con M3, sólo el instante 8 (el objetivo de la ventana anómala) queda alto. Los instantes 4-7 ya no se contagian.")
print("(Los instantes 0-3 no son objetivo de ninguna ventana: por eso se excluyen las orillas al evaluar.)")
''')
md(r"""
### 6.3 En una serie de verdad: acercamiento a una anomalía puntual
""")
code(r'''
X0, y0 = make_dataset("0", seed=1)                 # sólo anomalías puntuales
r_0 = run(X0, L, P, seed=0, cfg=CFG)
s_win = proyectar(r_0["si"], len(y0), objetivo=False)
s_tgt = proyectar(r_0["si"], len(y0), objetivo=True)
pico = [a for a, b in intervalos(y0) if 30 < a < len(y0) - 30][0]
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax, s_, tit in ((axes[0], s_win, "Artículo (Ec. 4): el score se reparte en 5 instantes"),
                    (axes[1], s_tgt, "Mejora 3: el score va al instante predicho")):
    rng_ = np.arange(pico - 12, pico + 8)
    ax.bar(rng_, s_[rng_], color=["#fb6a4a" if y0[i] else "#9ecae1" for i in rng_])
    ax.set_title(tit, fontsize=9); ax.set_xlabel("tiempo (rojo = anomalía real)")
plt.tight_layout(); plt.show()
m0 = valid_mask(len(y0))
display(pd.DataFrame({"AUROC": [roc_auc_score(y0[m0], s_win[m0]), roc_auc_score(y0[m0], s_tgt[m0])],
                      "AUPRC": [average_precision_score(y0[m0], s_win[m0]), average_precision_score(y0[m0], s_tgt[m0])]},
                     index=["Artículo (Ec. 4)", "Mejora 3 (al objetivo)"]).round(3))
''')
md(r"""
### 6.4 El descubrimiento principal: el halo crece con la contaminación
Aquí está la explicación de **por qué el método del artículo empeora tanto cuando hay muchas anomalías**.

Piensa en el AUROC global como una mezcla de dos comparaciones:
* anomalías **contra normales lejanas** (fácil: las lejanas tienen score bajo);
* anomalías **contra normales del halo** (difícil con la Ec. 4: el halo tiene score contagiado).

El peso de cada parte depende de **cuántas normales caen en el halo**. Con 1 % de anomalías, casi ninguna normal
está cerca de una anomalía. Con 30 %, **casi todas** lo están. Entonces, con la Ec. 4, el AUROC global se va
pareciendo cada vez más al AUROC **contra el halo**, que es bajo.

Vamos a medirlo en vivo con anomalías puntuales a 1 %, 10 % y 30 %. Para cada serie calculamos:
* **% de normales en el halo:** qué fracción de los instantes normales está a distancia 1–4 de una anomalía;
* **AUROC anom. vs halo** y **AUROC anom. vs lejanas**, con cada proyección;
* el **AUROC global**.

(Esto usa las etiquetas **sólo para analizar**; ningún método las usa.)
""")
code(r'''
def distancia_a_anomalia(y):
    idx = np.flatnonzero(y); n = len(y)
    if len(idx) == 0:
        return np.full(n, np.inf)
    pos = np.searchsorted(idx, np.arange(n))
    izq = np.abs(np.arange(n) - idx[np.clip(pos - 1, 0, len(idx) - 1)])
    der = np.abs(idx[np.clip(pos, 0, len(idx) - 1)] - np.arange(n))
    return np.minimum(izq, der)

def medir_halo(y, sc):
    y = np.asarray(y).astype(bool); m = valid_mask(len(y)); d = distancia_a_anomalia(y)
    halo = ~y & (d >= 1) & (d <= L + P - 1) & m; lejos = ~y & (d >= L + P) & m; anom = y & m
    auc = lambda a, b: roc_auc_score(np.r_[np.ones(a.sum()), np.zeros(b.sum())], np.r_[sc[a], sc[b]])
    return {"% normales en el halo": halo.sum() / (~y & m).sum(), "AUROC anom. vs halo": auc(anom, halo),
            "AUROC anom. vs lejanas": auc(anom, lejos), "AUROC global": roc_auc_score(y[m], sc[m])}

filas = []
for c in (0.01, 0.10, 0.30):
    Xh, yh = serie_contaminada("01", c, semilla=0)
    rh = run(Xh, L, P, seed=0, cfg=CFG)
    for nombre, obj in (("Artículo (Ec. 4)", False), ("Mejora 3", True)):
        filas.append({"contaminación": f"{c:.0%}", "proyección": nombre, **medir_halo(yh, proyectar(rh["si"], len(yh), obj))})
    print(f"contaminación {c:.0%}: listo")
halo_df = pd.DataFrame(filas).set_index(["contaminación", "proyección"])
display(halo_df.round(3))

fig, ax = plt.subplots(figsize=(8, 3.4))
for nombre, col in (("Artículo (Ec. 4)", "#7f7f7f"), ("Mejora 3", "#1f77b4")):
    d_ = halo_df.xs(nombre, level="proyección")
    xs = 100 * d_["% normales en el halo"].values
    ax.plot(xs, d_["AUROC global"], "o-", color=col, label=f"{nombre}: AUROC global")
    ax.plot(xs, d_["AUROC anom. vs halo"], "s--", color=col, alpha=0.6, label=f"{nombre}: AUROC vs halo")
ax.set_xlabel("% de normales que caen en el halo"); ax.set_ylabel("AUROC"); ax.legend(fontsize=7)
ax.set_title("Con la Ec. 4, el AUROC global cae hacia el AUROC contra el halo")
plt.show()
''')
md(r"""
**Qué deberías ver:** con la Ec. 4, el % de normales en el halo pasa de pocos a casi todas. El AUROC **contra el
halo** es bajo, y el AUROC global va cayendo hasta acercarse a él. Con la Mejora 3, el AUROC contra el halo es
alto, así que el AUROC global casi no cae.

En la evaluación grande (20 semillas, ver sección 9), con anomalías puntuales al 30 %, el **91 %** de las normales
caen en el halo. El AUROC del artículo pasa de 0.916 a 0.711, que es prácticamente su AUROC contra el halo
(0.686). Con M3 se queda en 0.931 (cae 0.022 en vez de 0.205). Es decir, **con anomalías puntuales casi toda la
degradación del artículo se debe a la proyección**, no a que el modelo aprenda las anomalías.

### 6.5 ⚠️ Dónde falla la Mejora 3: anomalías largas
La suma de la Ec. 4 tiene una **virtud**: en un evento **largo** (muchos instantes anómalos seguidos), cada instante
interior recibe el score de **5 ventanas anómalas**, así que **acumula evidencia**. La Mejora 3 sólo conserva la
ventana que predice ese instante, y **a mitad de un patrón el LSTM ya lo predice relativamente bien**: el
interior del evento queda con score bajo.

Ahí entra la Mejora 2: si quitamos las ventanas anómalas del entrenamiento, el modelo **no aprende el patrón**, y su
interior vuelve a ser difícil de predecir. Veámoslo con anomalías sólo de patrón (tipos 2, 3 y 4) al 20 %:
""")
code(r'''
Xp, yp = serie_contaminada("234", 0.20, semilla=1)
rp = run(Xp, L, P, seed=0, cfg=CFG)
sp = rp["si"]
for _ in range(3):
    sp = run(Xp, L, P, seed=0, cfg=CFG, weights=adapt_weights(sp))["si"]
mp = valid_mask(len(yp))
variantes = {"Artículo (Ec. 4)": proyectar(rp["si"], len(yp), False), "Mejora 3 sola": proyectar(rp["si"], len(yp), True),
             "M2 (recorte) + Ec. 4": proyectar(sp, len(yp), False), "M2 + M3 (método final)": proyectar(sp, len(yp), True)}
display(pd.DataFrame({k: {"AUROC": roc_auc_score(yp[mp], v[mp]), "AUPRC": average_precision_score(yp[mp], v[mp])}
                      for k, v in variantes.items()}).T.round(3))

# percentil del score a lo largo de cada evento largo
def perfil_eventos(y, sc, largo_min=8):
    pct = pd.Series(sc).rank(pct=True).values
    perf = []
    for a, b in intervalos(y):
        if b - a >= largo_min and a > 20 and b < len(y) - 20:
            perf.append(np.interp(np.linspace(0, 1, 10), np.linspace(0, 1, b - a), pct[a:b]))
    return np.mean(perf, axis=0)

fig, ax = plt.subplots(figsize=(8, 3.2))
for (k, v), c in zip(variantes.items(), ("#7f7f7f", "#1f77b4", "#8c564b", "#c0392b")):
    ax.plot(np.linspace(0, 100, 10), perfil_eventos(yp, v), "o-", color=c, label=k)
ax.set_xlabel("posición dentro del evento (% del largo)"); ax.set_ylabel("percentil del score")
ax.set_title("Anomalías de patrón: ¿qué tan alto queda el score DENTRO del evento?"); ax.legend(fontsize=8)
plt.show()
print("Con M3 sola, el interior del evento baja (el LSTM ya predice el patrón). El recorte (M2) lo recupera.")
print("Fíjate también en 'M2 (recorte) + Ec. 4': con anomalías largas, el recorte SIN M3 suele dar el mejor AUROC.")
print("Lo mismo pasó en la evaluación grande (mezcla 234) y en SMD.")
''')
md(r"""
**Para llevarte (muy importante para entender los resultados):**
* La Mejora 3 **cambia integración de evidencia por localización**: ubica mejor el instante exacto, pero deja de
  sumar evidencia a lo largo de eventos largos.
* Gana claramente con anomalías **cortas** (puntuales) y cuando hay **mucha contaminación** (mucho halo).
* Con anomalías **largas**, sola puede **empeorar**; necesita la Mejora 2 para compensar, y aun así puede no
  alcanzar. Por eso en los benchmarks reales SMD y SMAP/MSL (anomalías de cientos de pasos) **no ayuda**.
""")

# =====================================================================================
# 7. MÉTODO FINAL
# =====================================================================================
md(r"""
## 7. El método final = Mejora 1 + Mejora 2 + Mejora 3

Las tres piezas encajan en un solo procedimiento, **sin ninguna etiqueta** y con L = 4, P = 1 fijos de antemano (el
artículo, además del umbral, elige L y P para cada dataset mirando las etiquetas):

```
serie ──► entrenar LSTM ──► self-influence ──► quitar ventanas sobre el umbral IQR (M1 + M2) ──┐
              ▲                                                                                │  ×3 rondas
              └────────────────────────── volver a entrenar sin ellas ◄────────────────────────┘
                                   │
                                   ▼
                 proyectar al instante predicho (M3) ──► umbral IQR (M1) ──► anomalías
```

**Lo que es propio de la combinación:**
* la self-influence hace **dos trabajos**: es el detector **y** decide qué limpiar del entrenamiento;
* **el mismo umbral IQR** sirve para limpiar (M2) y para decidir (M1);
* cada pieza **corrige un punto débil de otra**: M3 quita el halo, pero pierde evidencia en eventos largos → M2 la
  recupera; M2 y M3 mejoran el score, pero hace falta un umbral → M1 lo da sin etiquetas.

Cada pieza por separado tiene antecedentes en la literatura; **la combinación, el diagnóstico del halo y la
evaluación son nuestra contribución**.

La función siguiente implementa el método completo. Con los interruptores `M1`, `M2`, `M3` puedes apagar cada
mejora. Con las tres apagadas es el método del artículo (y entonces el umbral hay que calibrarlo con etiquetas).
""")
code(r'''
def metodo_final(X, semilla=0, M1=True, M2=True, M3=True, rondas=3, avisar=print, cfg=None):
    """Devuelve (score por instante, umbral, score del artículo)."""
    cfg = cfg or CFG
    n = len(X)
    r0 = run(X, L, P, seed=semilla, cfg=cfg)               # 1) entrenar y medir self-influence
    s = r0["si"]
    s_articulo = proyectar(s, n, objetivo=False)
    if M2:                                                  # 2) rondas de recorte adaptativo (M1 + M2)
        for k in range(1, rondas + 1):
            w = adapt_weights(s)                            #    peso 0 a lo que pasa el umbral IQR
            s = run(X, L, P, seed=semilla, cfg=cfg, weights=w)["si"]   # reentrenar sin eso, medir TODO
            avisar(f"  ronda {k}/{rondas}: se quitaron {(w == 0).mean():.0%} de las ventanas")
    score = proyectar(s, n, objetivo=M3)                    # 3) proyección (M3 o Ec. 4)
    thr = thr_iqr(score[valid_mask(n)]) if M1 else None     # 4) umbral IQR (M1) o ninguno
    return score, thr, s_articulo


def comparar(tipos="01234", contaminacion=0.15, semilla=0, M1=True, M2=True, M3=True, largo=2000):
    Xs, ys = serie_contaminada(tipos, contaminacion, semilla, largo)
    m = valid_mask(len(ys))
    print(f"Serie '{tipos}' con {ys.mean():.1%} de anomalías. Entrenando…")
    t = time.time()
    s, thr, s_art = metodo_final(Xs, semilla, M1, M2, M3)
    thr_art = umbral_dinamico(np.asarray(ys)[m].astype(bool), s_art[m])
    nombre = "Nuestro (" + " + ".join([f"M{i}" for i, on in ((1, M1), (2, M2), (3, M3)) if on]) + ")" if (M1 or M2 or M3) else "Artículo"
    if thr is None:
        thr = umbral_dinamico(np.asarray(ys)[m].astype(bool), s[m]); nombre += " · umbral con etiquetas"
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    dibujar_serie(Xs, ys, f"Serie ({TIPOS.get(tipos, 'mezcla ' + tipos)}), {ys.mean():.0%} anómalo", axes[0])
    dibujar_deteccion(ys, s_art, thr_art, "Artículo (umbral dinámico, USA etiquetas)", axes[1])
    dibujar_deteccion(ys, s, thr, nombre, axes[2])
    plt.tight_layout(); plt.show()
    tabla = pd.DataFrame([evaluar(ys, s_art, thr_art), evaluar(ys, s, thr)],
                         index=["Artículo (con etiquetas)", nombre]).round(3)
    display(tabla)
    print(f"({time.time() - t:.0f} s)")
    return tabla

_ = comparar("01234", 0.20)
''')
md(r"""
### Laboratorio interactivo

Mueve los controles y pulsa **Run Interact / Ejecutar**. Cosas para probar (y qué deberías ver):

| Prueba | Qué esperar |
|---|---|
| Contaminación 2 % vs 30 % con todas las mejoras | el artículo se cae mucho más que el método final |
| Sólo puntuales (`01`), prender/apagar **M3** | M3 marca la diferencia: sin ella aparecen falsas alarmas antes de cada pico |
| Sólo de patrón (`234`), **M3 sin M2** | M3 sola puede empeorar; al prender M2 se recupera |
| Apagar **M1** | el método vuelve a necesitar etiquetas para el umbral |

(Cada corrida tarda entre 20 s y 1 min en la CPU de Colab.)
""")
code(r'''
try:
    import ipywidgets as W
    W.interact_manual(comparar,
        tipos=W.Dropdown(options=[("todas (01234)", "01234"), ("puntuales (01)", "01"), ("de patrón (234)", "234"),
                                  ("sólo puntual global (0)", "0"), ("sólo tendencia (4)", "4")], value="01234", description="anomalías"),
        contaminacion=W.FloatSlider(value=0.15, min=0.01, max=0.30, step=0.01, description="contaminación", readout_format=".0%"),
        semilla=W.IntSlider(value=0, min=0, max=20, description="semilla"),
        M1=W.Checkbox(value=True, description="Mejora 1 (umbral IQR)"),
        M2=W.Checkbox(value=True, description="Mejora 2 (recorte)"),
        M3=W.Checkbox(value=True, description="Mejora 3 (proyección)"),
        largo=W.fixed(2000))
except ImportError:
    print("ipywidgets no está disponible; usa comparar(tipos, contaminacion, semilla, M1, M2, M3) directamente.")
''', title="Laboratorio (controles)")

# =====================================================================================
# 8. EVALUACIÓN
# =====================================================================================
md(r"""
# PARTE D · ¿Funciona de verdad?

## 8. Cómo se evalúa bien un método

Una sola serie no prueba nada: pudo salir bien por casualidad. Para afirmar que un método es mejor hay que
cuidar varias cosas. Así lo hicimos:

### 8.1 Protocolo escrito ANTES de correr (`PROTOCOLO_FINAL.md`)
Antes de lanzar los experimentos dejamos por escrito: qué métodos, qué datos, qué métricas, qué hipótesis y qué
prueba estadística. **Nada se ajustó mirando los resultados.** Esto evita el error más común: probar muchas cosas
y reportar sólo lo que salió bien. Cuando agregamos datos nuevos (SMD, SMAP/MSL, semisintético), también
escribimos su protocolo antes de descargar los datos. Todo cambio posterior quedó anotado como "desviación".

### 8.2 Las hipótesis (en simple)
| Hipótesis | Pregunta | Métrica |
|---|---|---|
| **H1** | ¿El método final da un mejor score que el artículo? | AUPRC |
| **H2** | ¿El final **sin** etiquetas detecta al menos tan bien como el artículo **con** etiquetas? | F1 |
| **H3a** | ¿El recorte (M2) aporta algo encima de M3? | AUPRC |
| **H3b** | ¿El recorte aporta encima de M1 + M3? | F1 |
| **H4** | Con más anomalías (1 % → 30 %), ¿el final se degrada menos que el artículo? | caída de AUROC |
| **H5a/H5b** | ¿M3 sola mejora al artículo? | AUROC / AUPRC |
| **H4r/H5c** | Lo mismo que H4, pero con anomalías inyectadas sobre datos reales (final / M3) | caída de AUROC |
| **H6** | En series reales, ¿la ventaja de M3 crece con la proporción de anomalías? | correlación |

### 8.3 Los 6 bloques de datos
| Bloque | Qué es | Tamaño |
|---|---|---|
| **Sintéticos** | las 15 series del benchmark del artículo (400 pasos) | 15 series × 20 semillas = 300 corridas |
| **Contaminación** | series de 4 000 pasos con 1, 5, 10, 15, 20, 25 y 30 % de anomalías; tres mezclas (todas, sólo puntuales, sólo de patrón) | 3 × 7 × 20 semillas = 420 |
| **Reales del artículo** | Credit Card (fraude con tarjeta), CICIDS (ataques de red), GECCO (calidad del agua), SWAN-SF (erupciones solares) | 4 × 10 semillas = 40 |
| **SMD** | *Server Machine Dataset*: 28 servidores de una empresa de internet, 38 métricas cada uno | 28 × 5 = 140 |
| **SMAP + MSL** | telemetría de un satélite y del rover Curiosity de la NASA; 81 canales | 81 × 5 = 405 |
| **Semisintético** | datos reales de 10 servidores SMD **sin** anomalías etiquetadas + anomalías inyectadas por nosotros (1–30 %) | 10 × 3 mezclas × 5 niveles × 3 semillas = 450 |

Total: **1 755 corridas** (cada una entrena varias veces el LSTM: el artículo, el recorte, etc.).

¿Por qué tantos bloques? Los **sintéticos** y la **contaminación** permiten controlar exactamente el tipo y la
cantidad de anomalías. Los **reales** prueban que sirve fuera del laboratorio. Pero el artículo sólo tiene 4
datasets reales, que son muy pocos para concluir algo general. Por eso agregamos **SMD y SMAP/MSL** (109 series
reales más). El **semisintético** es el punto intermedio: fondo real, pero anomalías controladas.

### 8.4 La prueba estadística, en simple
* **Wilcoxon pareada.** Para cada serie calculamos *(nuestro − artículo)*. Si nuestro método no fuera mejor, esas
  diferencias serían positivas y negativas más o menos por igual. El **p-valor** es la probabilidad de ver
  diferencias tan cargadas hacia un lado *si en realidad no hubiera mejora*. Si p < 0.05, decimos que la
  diferencia es **significativa**. "Pareada" quiere decir que se comparan los dos métodos **sobre la misma
  serie y la misma semilla**.
* **Corrección de Holm.** Si haces 20 pruebas, alguna saldrá "significativa" por puro azar. Holm ajusta los
  p-valores para que eso no pase.
* **Intervalo de confianza (IC 95 %) bootstrap.** Rango donde muy probablemente está la mejora real. Se obtiene
  remuestreando las series 10 000 veces.
* **Tamaño del efecto** (*rank-biserial*, de −1 a 1). No basta con que sea significativo: también importa qué
  tanto. 1 = el nuestro gana siempre, 0 = empate, −1 = pierde siempre.
* **Gana / total.** En cuántas series ganó. Es lo más fácil de entender.

### 8.5 Mini-experimento en vivo
Una versión pequeña de la evaluación (pocas semillas), para que veas el procedimiento completo. **Con tan pocas
series el p-valor no puede ser muy pequeño**: por eso corrimos la evaluación grande (sección 9).
""")
code(r'''
N_SEMILLAS = 3                      # súbelo si tienes tiempo (cada semilla ≈ 1–2 min)
NIVELES = [0.05, 0.15, 0.25]
filas = []
t = time.time()
for sem in range(N_SEMILLAS):
    for c in NIVELES:
        Xs, ys = serie_contaminada("01234", c, sem, largo=1500)
        m = valid_mask(len(ys)); yb = np.asarray(ys)[m].astype(bool)
        s, thr, s_art = metodo_final(Xs, sem, avisar=lambda *_: None)
        e_fin = evaluar(ys, s, thr); e_art = evaluar(ys, s_art, umbral_dinamico(yb, s_art[m]))
        filas.append({"semilla": sem, "contaminación": f"{c:.0%}", "AUPRC artículo": e_art["AUPRC"], "AUPRC final": e_fin["AUPRC"],
                      "F1 artículo (con etiquetas)": e_art["F1"], "F1 final (sin etiquetas)": e_fin["F1"]})
        print(f"semilla {sem}, contaminación {c:.0%}: listo ({time.time() - t:.0f} s)")
mini = pd.DataFrame(filas)
display(mini.round(3))
for met in ("AUPRC", "F1"):
    a = mini[[c_ for c_ in mini if c_.startswith(met) and "artículo" in c_][0]]
    b = mini[[c_ for c_ in mini if c_.startswith(met) and "final" in c_][0]]
    d = b - a
    p = wilcoxon(d).pvalue if (d != 0).sum() >= 2 else float("nan")
    print(f"{met}: el final gana en {(d > 0).sum()} de {len(d)} series; diferencia media {d.mean():+.3f}; p (Wilcoxon) = {p:.3f}")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.scatter(mini["AUPRC artículo"], mini["AUPRC final"], c="#3182bd")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("AUPRC artículo"); ax.set_ylabel("AUPRC método final"); ax.set_title("Cada punto es una serie: arriba de la diagonal = gana el final")
plt.show()
''')

# =====================================================================================
# 9. RESULTADOS
# =====================================================================================
md(r"""
## 9. Todos los resultados de la evaluación grande

Los experimentos grandes se corrieron en nuestra máquina (GPU y CPU, muchas horas de cómputo). Aquí sólo se muestran
sus cifras, que vienen de `results/final/final_numbers.json`. Ejecuta las celdas en orden; cada una trae su
explicación.

**Nombres de los métodos en las tablas:**
* **Artículo (estático / dinámico):** el método original con su umbral fijo o con su umbral calibrado con etiquetas.
* **M1, M2, M3:** cada mejora sola. **M1+M3, M2+M3, M1+M2:** combinaciones de dos.
* **Método final:** M1 + M2 (recorte adaptativo) + M3.
* **Referencias:** "Error del LSTM" (usar el error de predicción como score, la forma clásica), **LOF** e
  **Isolation Forest** (dos detectores clásicos que no usan redes neuronales).
""")
code("RESULTADOS = " + json.dumps(NUMBERS, ensure_ascii=False, indent=None, default=float) + "\n"
     "EXTRA = " + json.dumps(EXTRA, ensure_ascii=False, indent=None, default=float),
     title="Resultados completos (datos incrustados)", hidden=True)
code(r'''
FINAL = "**Método final** (recorte adaptativo + proyección + IQR)"
ART_S, ART_D = "Artículo · umbral estático", "Artículo · umbral dinámico (usa etiquetas)"
M3N, M1M2N = "M3 · proyección al objetivo", "M1+M2 (recorte adaptativo, sin M3)"
CORTO = {ART_D: "Artículo (dinámico, con etiquetas)", ART_S: "Artículo (estático)", "M1 · umbral IQR": "M1",
         "M2 · reponderación suave": "M2 suave", "M2 · recorte z>2": "M2 recorte", M3N: "M3", "M1+M3": "M1+M3",
         "M2+M3 (recorte z>2)": "M2+M3", M1M2N: "M1+M2", "M1+M2+M3 con recorte z>2": "M1+M2+M3 (z>2)",
         FINAL: "MÉTODO FINAL (M1+M2+M3)", "Error del LSTM (PD) + IQR": "Error LSTM", "LOF + IQR": "LOF", "IForest + IQR": "IForest"}
BLOQUES = {"synth": "Sintéticos", "contam": "Contaminación", "real": "Reales del artículo", "smd": "SMD",
           "nasa": "SMAP+MSL", "semi": "Semisintético (fondo real)"}
BLQ = [b for b in BLOQUES if b in RESULTADOS]
if not BLQ:
    display(Markdown("> ⏳ **Pendiente.** La evaluación final todavía no termina; este cuaderno se regenera con las cifras."))
else:
    filas = []
    for b in BLQ:
        mm = RESULTADOS[b]["methods"]
        fila = {"bloque": BLOQUES[b]}
        for m, nom in ((ART_S, "artículo"), (M3N, "M3 sola"), (M1M2N, "M1+M2"), (FINAL, "FINAL")):
            fila[f"AUPRC {nom}"] = mm[m]["auprc"]
        fila["AUROC artículo"], fila["AUROC FINAL"] = mm[ART_S]["auroc"], mm[FINAL]["auroc"]
        fila["F1 artículo (con etiquetas)"], fila["F1 FINAL (sin etiquetas)"] = mm[ART_D]["f1_bin"], mm[FINAL]["f1_bin"]
        filas.append(fila)
    resumen = pd.DataFrame(filas).set_index("bloque")
    display(Markdown("### 9.1 Vista general: el método final frente al artículo en los 6 bloques"))
    display(resumen.round(3))
    fig, ax = plt.subplots(figsize=(12, 3.6))
    xs = np.arange(len(BLQ)); wd = 0.2
    for i, (col, c) in enumerate((("AUPRC artículo", "#7f7f7f"), ("AUPRC M3 sola", "#1f77b4"),
                                  ("AUPRC M1+M2", "#8c564b"), ("AUPRC FINAL", "#c0392b"))):
        ax.bar(xs + (i - 1.5) * wd, resumen[col], wd, color=c, label=col.replace("AUPRC ", ""))
    ax.set_xticks(xs); ax.set_xticklabels([BLOQUES[b] for b in BLQ], fontsize=8); ax.set_ylabel("AUPRC")
    ax.legend(fontsize=8, ncol=4); ax.set_title("AUPRC medio por bloque (más alto = mejor score)")
    plt.show()
''')
md(r"""
**Cómo leer la vista general:**
* En **sintéticos y contaminación** el método final mejora muchísimo el AUPRC (de ~0.5 a ~0.8) y también el F1,
  **aun comparándolo con el artículo que usa etiquetas**.
* En el **semisintético** y en los **reales del artículo** mejora el AUPRC, con menor magnitud.
* En **SMD** queda **por debajo** del artículo, y en **SMAP+MSL** empata. Fíjate en la columna **M1+M2**: en SMD es
  la mejor variante. Lo que hace daño ahí es **M3** (anomalías largas, sección 6.5).
* El **F1 sin etiquetas** supera al del artículo con etiquetas en sintéticos y contaminación, empata en el
  semisintético y **pierde en los datos reales** (sección 4: el umbral IQR marca de más cuando las anomalías son raras).
""")
code(r'''
def veredicto(h):
    if h["p_holm"] < 0.05 and h["mean"] > 0:
        return "✅ sí (significativa)"
    if h["p_holm"] < 0.05:
        return "❌ no: es PEOR"
    return "➖ no significativa"

EXPLICA = {"H1": "¿El score del final es mejor que el del artículo? (AUPRC)",
           "H2": "¿El final SIN etiquetas detecta al menos tan bien como el artículo CON etiquetas? (F1)",
           "H3a": "¿El recorte (M2) aporta encima de M3? (AUPRC)",
           "H3b": "¿El recorte aporta encima de M1+M3? (F1)",
           "H4": "Con más anomalías, ¿el final se degrada menos que el artículo? (caída de AUROC)",
           "H4r": "Sobre fondo real, con más anomalías, ¿el final se degrada menos? (caída de AUROC)",
           "H5a": "¿M3 sola mejora al artículo? (AUROC)",
           "H5b": "¿M3 sola mejora al artículo? (AUPRC)",
           "H5c": "Sobre fondo real, con más anomalías, ¿M3 se degrada menos? (caída de AUROC)"}
if BLQ:
    filas = []
    for b in BLQ:
        nb = RESULTADOS[b]
        for h in nb["hyp"] + nb.get("hyp_h5", []):
            filas.append({"bloque": BLOQUES[b], "hip.": h["h"], "pregunta": EXPLICA.get(h["h"], h["h"]),
                          "diferencia media": f"{h['mean']:+.3f}", "IC 95 %": f"[{h['lo']:+.3f}, {h['hi']:+.3f}]",
                          "gana / total": f"{h['wins']} / {h['n']}",
                          "p (Holm)": f"{h['p_holm']:.1e}" if h["p_holm"] < 1e-3 else f"{h['p_holm']:.3f}",
                          "¿se sostiene?": veredicto(h)})
    display(Markdown("### 9.2 Todas las hipótesis, bloque por bloque"))
    display(pd.DataFrame(filas).set_index(["bloque", "hip."]))
''')
md(r"""
**Cómo leer la tabla de hipótesis:**
* **diferencia media**: cuánto mejora (positivo) o empeora (negativo) en promedio.
* **IC 95 %**: si no incluye el 0, la diferencia es consistente.
* **gana / total**: en cuántas unidades ganó. En SMD y SMAP+MSL la unidad es la serie (promedio de 5 semillas);
  en los demás, cada (serie, semilla).
* **p (Holm)**: menor que 0.05 → significativa. Valores como `1e-60` quieren decir que es prácticamente imposible
  que sea suerte.

**Resumen honesto:**
* ✅ **Contaminación:** todas las hipótesis se sostienen con p < 10⁻¹⁰. Es el resultado más fuerte del proyecto.
* ✅ **Sintéticos:** todas se sostienen.
* ✅ **Semisintético:** H1, H3b, H4r, H5 y H5c se sostienen; H2 empata (sin etiquetas ≈ con etiquetas); H3a sale al revés.
* ⚖️ **Reales del artículo:** H1 sí (gana en las 40 corridas), H2 **no** (el umbral sin etiquetas pierde en F1).
* ❌ **SMD:** M3 pierde en 27 de 28 máquinas; el final no supera al artículo.
* ➖ **SMAP+MSL:** casi todo empata; el recorte sí aporta un poco (H3a).
""")
code(r'''
MIX = {"01234": "todas las anomalías", "01": "sólo puntuales", "234": "sólo de patrón"}
for bq in ("contam", "semi"):
    if not RESULTADOS.get(bq):
        continue
    curvas = RESULTADOS[bq]["curves"]
    display(Markdown(f"### 9.3 {BLOQUES[bq]}: AUROC al aumentar la proporción de anomalías"))
    fig, axes = plt.subplots(1, len(curvas), figsize=(15, 3.6), sharey=True)
    for ax, mix in zip(np.atleast_1d(axes), [m_ for m_ in ("01234", "01", "234") if m_ in curvas]):
        for f_ in curvas[mix]:
            niveles = [k for k in f_ if k.endswith("%")]
            xs = [float(k[:-1]) for k in niveles]
            es_final = f_["Método"] == FINAL
            ax.plot(xs, [float(f_[k]) for k in niveles], "o-" if es_final else ".-", lw=2.5 if es_final else 1,
                    color="#c0392b" if es_final else None, label=CORTO.get(f_["Método"], f_["Método"]))
        ax.set_title(MIX.get(mix, mix)); ax.set_xlabel("% de anomalías (nominal)")
    np.atleast_1d(axes)[0].set_ylabel("AUROC"); np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower left")
    plt.suptitle("Más plano = más robusto a la contaminación"); plt.tight_layout(); plt.show()
    caidas = pd.DataFrame(EXTRA.get("caidas", {}).get(bq, []))
    if len(caidas):
        caidas = caidas.rename(columns={"method": "método", "drop": "caída de AUROC 1 % → 30 %"}).set_index("método")
        display(caidas.round(3))
''')
md(r"""
**Cómo leer las curvas de contaminación:**
* **Contaminación sintética, todas las anomalías:** el AUROC del artículo cae 0.170 de 1 % a 30 %; el del método
  final, sólo 0.043.
* **Sólo puntuales:** el método final queda **plano**. Es el caso ideal para M3: anomalías cortas y mucho halo.
* **Sólo de patrón:** M3 sola queda **por debajo** del artículo. El final (con el recorte) lo alcanza en AUROC y lo
  supera en AUPRC, pero no en F1.
* **Sobre fondo real (semisintético):** el mismo patrón, con diferencias más pequeñas, porque ahí el artículo casi
  no se degrada. Ojo: en ese bloque **LOF** y el **error del LSTM** son mejores que cualquier variante de TracIn.
""")
code(r'''
if RESULTADOS.get("real"):
    display(Markdown("### 9.4 Los 4 datasets reales del artículo (10 semillas cada uno)"))
    pdx = RESULTADOS["real"]["per_ds"]
    filas = []
    for ds, mm in pdx.items():
        filas.append({"dataset": ds, "AUPRC artículo": mm[ART_S]["auprc"], "AUPRC M3": mm[M3N]["auprc"],
                      "AUPRC M1+M2": mm[M1M2N]["auprc"], "AUPRC FINAL": mm[FINAL]["auprc"],
                      "F1 artículo (con etiquetas)": mm[ART_D]["f1_bin"], "F1 FINAL (sin etiquetas)": mm[FINAL]["f1_bin"],
                      "AUPRC error LSTM": mm["Error del LSTM (PD) + IQR"]["auprc"]})
    display(pd.DataFrame(filas).set_index("dataset").round(3))
''')
md(r"""
**Cómo leer los reales del artículo:**
* El método final mejora el AUPRC en **los 4 datasets** (10 de 10 semillas en cada uno): mucho en Credit Card y
  GECCO, algo en SWAN-SF y casi nada en CICIDS, donde **ningún método funciona** (AUPRC ≈ 0.02, igual que el azar).
* **No todas las mejoras ayudan en todos lados:** M3 ayuda en Credit Card y SWAN-SF, pero perjudica en GECCO. Ahí
  la mejora viene del recorte (M1+M2).
* En **F1** el artículo con etiquetas gana, porque estas anomalías son muy raras (0.2–1.3 %) y el IQR marca de más
  (sección 4).
""")
code(r'''
filas = []
for bq in ("smd", "nasa"):
    if not RESULTADOS.get(bq):
        continue
    mm = RESULTADOS[bq]["methods"]
    for m in (ART_S, ART_D, "M1 · umbral IQR", "M2 · recorte z>2", M3N, M1M2N, FINAL, "Error del LSTM (PD) + IQR", "LOF + IQR", "IForest + IQR"):
        if m in mm:
            filas.append({"benchmark": f"{BLOQUES[bq]} ({RESULTADOS[bq]['n_units']} series)", "método": CORTO.get(m, m),
                          "AUROC": mm[m]["auroc"], "AUPRC": mm[m]["auprc"], "F1": mm[m]["f1_bin"]})
if filas:
    display(Markdown("### 9.5 Benchmarks reales adicionales: SMD y SMAP+MSL (promedio por serie)"))
    display(pd.DataFrame(filas).set_index(["benchmark", "método"]).round(3))
''')
md(r"""
**Cómo leer SMD y SMAP+MSL (el resultado negativo más importante):**
* En **SMD**, M3 sola pierde frente al artículo en **27 de 28 máquinas** (AUROC −0.023, AUPRC −0.054). El método
  final hereda esa pérdida (AUPRC 0.392 frente a 0.454). En cambio **M1+M2** (recorte sin M3) es la mejor variante
  (AUPRC 0.472).
* En **SMAP+MSL** casi todo empata alrededor de AUROC 0.6 (cerca del azar). Aquí el mejor es Isolation Forest.
* **¿Por qué?** Las anomalías etiquetadas en estos benchmarks son **tramos larguísimos**: en SMD el 98 % de los
  instantes anómalos están en eventos de 10 pasos o más (la mitad en eventos de más de 400 pasos); en SMAP/MSL, el
  100 % (la mitad en eventos de más de 2 700 pasos). Es justo el caso de la sección 6.5, donde M3 pierde la
  evidencia acumulada.
* **Conclusión práctica:** si sabes que tus anomalías son tramos largos, usa **M1 + M2 sin M3**.
""")
code(r'''
if EXTRA.get("h6_series"):
    display(Markdown("### 9.6 H6: ¿M3 ayuda más en las series reales con más anomalías?"))
    h6 = pd.DataFrame(EXTRA["h6_series"])
    rho, p = spearmanr(h6["anom_rate"], h6["d_auprc_m3"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, col, nom in ((axes[0], "d_auroc_m3", "ΔAUROC (M3 − artículo)"), (axes[1], "d_auprc_m3", "ΔAUPRC (M3 − artículo)")):
        for blk, c in (("smd", "#1f77b4"), ("nasa", "#d62728")):
            d_ = h6[h6.block == blk]
            ax.scatter(100 * d_.anom_rate, d_[col], s=14, color=c, alpha=0.7, label="SMD" if blk == "smd" else "SMAP+MSL")
        ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("% de instantes anómalos de la serie (escala log)")
        ax.set_title(nom); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    print(f"Correlación de Spearman entre % de anomalías y ventaja de M3 (AUPRC): ρ = {rho:+.2f}, p = {p:.1e}, n = {len(h6)} series")
    display(pd.DataFrame(RESULTADOS["h6"]["cuartiles"]).rename(columns={
        "anom_rate": "cuartil de % anomalías", "tasa_min": "desde", "tasa_max": "hasta", "d_auroc_m3": "ΔAUROC M3",
        "d_auprc_m3": "ΔAUPRC M3", "gana": "series donde gana M3", "n": "series"}).round(3))
''')
md(r"""
**Cómo leer H6:** cada punto es una serie real. Con pocas anomalías (izquierda), M3 suele perder; con muchas
(derecha), se acerca a cero o gana. La correlación es positiva y muy significativa. Esto **confirma en datos
reales** el mecanismo del halo: cuantas más anomalías, más halo, y más útil es M3. Pero ojo: en estos benchmarks
la contaminación alta **sólo neutraliza** la desventaja de M3 con eventos largos; no la convierte en ventaja.
""")
code(r'''
if EXTRA.get("halo"):
    display(Markdown("### 9.7 El mecanismo medido en todos los bloques (método del artículo vs M3, mismo entrenamiento)"))
    filas = []
    for b, rows in EXTRA["halo"].items():
        for r_ in rows:
            if r_["train"] == "base":
                filas.append({"bloque": BLOQUES.get(b, b), "proyección": "Ec. 4 (artículo)" if r_["proj"] == "win" else "M3 (objetivo)",
                              "AUROC anom. vs halo": r_["auroc_anom_vs_halo"], "AUROC anom. vs lejanas": r_["auroc_anom_vs_far"],
                              "% del top-k que es halo": 100 * r_["topk_halo_share"]})
    display(pd.DataFrame(filas).set_index(["bloque", "proyección"]).round(3))
''')
md(r"""
**Cómo leer la tabla del mecanismo:** el "top-k" son los *k* instantes con mayor score (*k* = número de anomalías).
Si una buena parte de ellos son normales del halo, son falsas alarmas causadas por la proyección.
* En todos los bloques, **M3 sube el AUROC contra el halo** y **baja el % del top-k que es halo**: el halo existe y
  M3 lo quita.
* Pero en bloques con anomalías largas (SMD, SMAP+MSL) el AUROC **contra lejanas** baja un poco con M3, y el halo
  es una fracción muy pequeña de las normales, así que pesa más lo segundo.

### 9.8 En una frase: qué funciona y qué no
| ✅ Se sostiene | ❌ No se sostiene / límites |
|---|---|
| La combinación mejora mucho el score con contaminación (AUPRC +0.33, 402 de 420) | En SMD el final queda por debajo del artículo (M3 perjudica con anomalías largas) |
| Con contaminación se degrada 4 veces menos que el artículo (0.043 vs 0.170) | En SMAP+MSL no hay diferencia; nadie funciona bien (AUROC ≈ 0.6) |
| Sin etiquetas supera en F1 al artículo con etiquetas (sintéticos y contaminación) | Con anomalías muy raras (< 2 %), el umbral IQR pierde en F1 frente al umbral con etiquetas |
| Mejora el AUPRC en los 4 reales del artículo (40 de 40) | M3 sola empeora con anomalías de patrón; necesita al recorte |
| El mecanismo (halo) está identificado y medido, y H6 lo confirma en 109 series reales | En el semisintético, LOF y el error del LSTM superan a todas las variantes de TracIn |
""")

# =====================================================================================
# 10. ÚSALO TÚ
# =====================================================================================
md(r"""
# PARTE E · Úsalo tú

## 10. Pruébalo en datos reales

### 10.1 Una máquina del SMD, descargada en vivo
El **SMD** está publicado en GitHub (repositorio *OmniAnomaly*). La celda siguiente descarga los datos de prueba de
una máquina con sus etiquetas, corre **las cuatro variantes** y las compara:
* **Artículo**: Ec. 4, umbral dinámico **con** etiquetas;
* **M3 sola**: proyección al objetivo, umbral IQR;
* **M1+M2**: recorte adaptativo con la Ec. 4, umbral IQR (la recomendada para anomalías largas);
* **Final**: M1+M2+M3.

Truco: las cuatro salen de **sólo dos entrenamientos** (el normal y el de 3 rondas de recorte), porque M3 y la
Ec. 4 son sólo dos formas de proyectar el mismo score.

> Para que corra rápido en Colab usamos el gradiente de la **última capa**, batch 256 y máximo 30 épocas. En la
> evaluación grande usamos el gradiente de todos los parámetros y 50 épocas, así que los números pueden variar un
> poco, pero la tendencia es la misma.

Puedes cambiar `MAQUINA` (de `machine-1-1` a `machine-1-8`, `machine-2-1` a `machine-2-9`, `machine-3-1` a
`machine-3-11`).
""")
code(r'''
URL_SMD = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
CFG_RAPIDA = Cfg(batch=256, max_epochs=30)

def bajar_smd(maquina, parte="test"):
    leer = lambda ruta: np.loadtxt(io.StringIO(urllib.request.urlopen(f"{URL_SMD}/{ruta}/{maquina}.txt", timeout=60).read().decode()), delimiter=",")
    X = leer(parte).astype(np.float32)
    y = leer("test_label").astype(np.int8) if parte == "test" else None
    return X, y

def cuatro_variantes(X, y, semilla=0, cfg=None, titulo=""):
    cfg = cfg or CFG_RAPIDA
    n = len(y); m = valid_mask(n); yb = np.asarray(y)[m].astype(bool)
    t = time.time()
    s0 = run(X, L, P, seed=semilla, cfg=cfg)["si"]; s = s0
    for k in range(3):
        s = run(X, L, P, seed=semilla, cfg=cfg, weights=adapt_weights(s))["si"]
        print(f"  ronda de recorte {k + 1}/3 lista ({time.time() - t:.0f} s)")
    sc = {"Artículo (con etiquetas)": proyectar(s0, n, False), "M3 sola": proyectar(s0, n, True),
          "M1+M2 (sin M3)": proyectar(s, n, False), "FINAL (M1+M2+M3)": proyectar(s, n, True)}
    filas = {}
    for k, v in sc.items():
        thr = umbral_dinamico(yb, v[m]) if k.startswith("Artículo") else thr_iqr(v[m])
        filas[k] = evaluar(y, v, thr)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    dibujar_serie(X, y, titulo, axes[0])
    for ax, k in zip(axes[1:], ("Artículo (con etiquetas)", "FINAL (M1+M2+M3)")):
        v = sc[k]; thr = umbral_dinamico(yb, v[m]) if k.startswith("Artículo") else thr_iqr(v[m])
        dibujar_deteccion(y, v, thr, k, ax)
    plt.tight_layout(); plt.show()
    tabla = pd.DataFrame(filas).T
    display(tabla.round(3))
    largos = [b - a for a, b in intervalos(y)]
    print(f"{len(largos)} eventos anómalos; largo mediano {np.median(largos):.0f} pasos; {np.mean(y):.1%} de instantes anómalos.")
    return tabla

MAQUINA = "machine-1-1"
try:
    Xs_, ys_ = bajar_smd(MAQUINA)
    print(f"{MAQUINA}: {Xs_.shape[0]} instantes × {Xs_.shape[1]} métricas")
    _ = cuatro_variantes(Xs_, ys_, titulo=f"SMD {MAQUINA} (primeras 6 métricas)")
except Exception as e:
    print("No se pudo descargar (¿sin internet?):", e)
''')
md(r"""
**Qué esperar:** en la mayoría de las máquinas del SMD, el **artículo** o **M1+M2** quedan arriba en AUPRC y el
final un poco abajo, igual que en la evaluación grande. Fíjate en la última línea: los eventos anómalos duran
decenas o cientos de pasos. **Éste es el caso donde no conviene M3.**

### 10.2 Semisintético en vivo: fondo real + anomalías cortas inyectadas
Ahora tomamos 10 000 pasos de la parte de **entrenamiento** de la misma máquina (que no trae anomalías
etiquetadas) e **inyectamos** anomalías **puntuales** con nuestra función `inject` (de `src/data_bench.py`):
* **tipo 0:** un valor fuera del rango del canal;
* **tipo 1:** un valor lejos de su vecindad, pero dentro del rango.

Cada anomalía cae en un canal elegido al azar. Así tenemos un fondo real con anomalías cortas, el caso donde la
combinación sí debería ayudar.
""")
code(r'''
try:
    Xbg, _ = bajar_smd(MAQUINA, parte="train")
    bg = standardize(Xbg[:10_000])
    Xi, yi = inject(bg, "01", 0.20, seed=123)          # 20 % de anomalías puntuales inyectadas
    print(f"Fondo real {bg.shape}, {yi.mean():.1%} de instantes con anomalía inyectada")
    _ = cuatro_variantes(Xi, yi, titulo=f"Fondo real ({MAQUINA}, entrenamiento) + 20 % de anomalías puntuales")
except Exception as e:
    print("No se pudo descargar (¿sin internet?):", e)
''')
md(r"""
**Qué esperar:** aquí M3 y el método final deberían quedar **por encima** del artículo en AUROC y AUPRC, igual que
en el bloque semisintético de la evaluación grande (mezcla de anomalías puntuales).

### 10.3 Con tu propio archivo CSV
Prepara un CSV donde **cada fila es un instante** y **cada columna una medición** (todas numéricas). Si tienes
etiquetas, ponlas en una columna (por ejemplo `label`, con 1 = anómalo y 0 = normal); si no, el cuaderno sólo te
dará el score y los instantes marcados.

1. Ejecuta la celda siguiente y elige tu archivo.
2. Cambia `COLUMNA_ETIQUETA` si tu columna se llama distinto (o pon `None` si no tienes etiquetas).
3. Cambia `USAR_M3` a `False` si tus anomalías son tramos largos (decenas de pasos o más).
""")
code(r'''
COLUMNA_ETIQUETA = "label"     # None si no hay etiquetas
USAR_M3 = True                 # False si tus anomalías son tramos largos

def probar_csv(ruta, col_etiqueta=COLUMNA_ETIQUETA, usar_m3=USAR_M3, semilla=0):
    df = pd.read_csv(ruta)
    y = None
    if col_etiqueta and col_etiqueta in df:
        y = df.pop(col_etiqueta).astype(int).values
    X = df.select_dtypes("number").ffill().fillna(0).values.astype(np.float32)
    print(f"{len(X)} instantes × {X.shape[1]} columnas numéricas")
    s, thr, s_art = metodo_final(X, semilla, M1=True, M2=True, M3=usar_m3, cfg=CFG_RAPIDA)
    marcados = np.flatnonzero(s >= thr)
    print(f"El método marcó {len(marcados)} instantes ({len(marcados) / len(X):.1%}). Primeros: {marcados[:20].tolist()}")
    fig, ax = plt.subplots(figsize=(12, 3))
    if y is not None:
        dibujar_deteccion(y, s, thr, "Tu serie: score del método final", ax)
        plt.show()
        m = valid_mask(len(y))
        display(pd.DataFrame([evaluar(y, s_art, umbral_dinamico(np.asarray(y)[m].astype(bool), s_art[m])), evaluar(y, s, thr)],
                             index=["Artículo (con etiquetas)", "Final (sin etiquetas)"]).round(3))
    else:
        ax.plot(s, lw=0.8); ax.axhline(thr, color="k", ls="--"); ax.set_title("Tu serie: score y umbral IQR"); plt.show()
    return s, thr

try:
    from google.colab import files
    subido = files.upload()
    for nombre in subido:
        _ = probar_csv(nombre)
except ImportError:
    print("Fuera de Colab: llama a probar_csv('ruta/a/tu_archivo.csv').")
except Exception as e:
    print("No se subió ningún archivo o hubo un problema:", e)
''', title="Subir y probar tu CSV (opcional)")

# =====================================================================================
# 11. CÓDIGO DEL PROYECTO
# =====================================================================================
md(r"""
## 11. Cómo está organizado el código del proyecto

Todo vive en la carpeta `experimentos/`:

| Archivo / carpeta | Qué contiene |
|---|---|
| `PROTOCOLO_FINAL.md` | el protocolo escrito **antes** de correr: hipótesis, métodos, datos; y las desviaciones anotadas |
| `src/synthetic.py` | el generador de series sintéticas (Lai et al., 2021) |
| `src/replica/lab.py` | la réplica del artículo: el LSTM, el entrenamiento y TracIn (`run`) |
| `src/common.py` | proyecciones (`project_sum` = Ec. 4, `project_targets_mean` = M3), umbral IQR (`thr_iqr`) |
| `src/forecaster.py` | z robusto y pesos para M2 suave / recorte (`weights_from_influence`) |
| `src/final/core.py` | recorte adaptativo (`adapt_weights`) y `all_variants`, que corre todas las variantes de una vez |
| `src/data_bench.py` | descarga y preparación de SMD y SMAP/MSL; `inject` para el semisintético |
| `src/baselines.py` | LOF e Isolation Forest |
| `src/final/run_synth.py`, `run_contam.py`, `run_real.py`, `run_bench.py` | corren cada bloque y guardan los scores en `results/final/<bloque>/*.npz` |
| `src/final/evaluate.py` | calcula todas las métricas de cada corrida → `results/final/<bloque>_eval.csv` |
| `src/final/stats.py` | pruebas de hipótesis → `results/final/final_numbers.json` y `results/final/tables/*.md` |
| `src/final/paper.py` | tablas LaTeX, figuras, Friedman/Nemenyi y el análisis del halo → `results/final/paper/` |
| `src/final/macros.py` | pasa las cifras a macros LaTeX, para que el texto cite números calculados y no copiados a mano |
| `results/final/paper/resultados.pdf` | la sección de resultados lista para el artículo |
| `colab/build_paso_a_paso.py` | genera **este** cuaderno |

### Cómo se reproduce todo (en una máquina con GPU)
```bash
cd experimentos/src/final
python run_synth.py --seeds 20 --workers 18
python run_contam.py --seeds 20 --workers 18
python run_real.py --seeds 10 --claim          # se pueden lanzar varios en paralelo
python ../data_bench.py                         # prepara SMD, SMAP/MSL y fondos (datos en data/raw/)
python run_bench.py --blocks smd,nasa,semi      # GPU; con --device cpu también funciona
for b in synth contam real smd nasa semi; do python evaluate.py $b --workers 20; done
python stats.py && python paper.py --workers 20 && python macros.py
cd ../../results/final/paper && pdflatex resultados.tex
```
Cada corrida está fijada por su semilla: repetirla da exactamente el mismo resultado.
""")

# =====================================================================================
# 12. RESUMEN
# =====================================================================================
md(r"""
## 12. Resumen, glosario y preguntas que te pueden hacer

### El proyecto en cinco frases
1. El artículo detecta anomalías midiendo cuánto le costó a un LSTM aprender cada pedazo de la serie (TracIn).
2. **Mejora 1:** cambiamos el umbral que se calibra con etiquetas por la regla IQR, que no las necesita.
3. **Mejora 2:** con muchas anomalías el modelo las memoriza; quitamos del entrenamiento lo sospechoso (con el mismo
   umbral IQR) y reentrenamos tres veces.
4. **Mejora 3:** el score de cada ventana va sólo al instante que predijo. Así desaparece el "halo" de falsas
   alarmas que la fórmula del artículo crea alrededor de cada anomalía, y que crece con la contaminación.
5. **Método final:** las tres juntas, sin etiquetas. Funciona muy bien con contaminación y anomalías cortas; con
   anomalías largas (SMD, SMAP/MSL) conviene quitar la Mejora 3.

### Cómo explicarlo en dos minutos
> "El artículo usa TracIn: entrena un LSTM a predecir la serie y mide cuánto le cuesta aprender cada ventana. Le
> encontramos tres problemas: su mejor umbral usa etiquetas, con muchas anomalías el modelo se las aprende, y su
> forma de pasar de ventanas a instantes contagia el score de una anomalía a sus vecinos normales. Lo arreglamos con
> un umbral IQR, con rondas de recorte de las ventanas sospechosas y proyectando el score sólo al instante predicho.
> Con un protocolo escrito antes de correr, en 1 755 corridas, la combinación mejora el AUPRC hasta en 0.33 y se
> degrada cuatro veces menos con la contaminación. Mostramos que la causa principal de la degradación del artículo
> es ese contagio, al que llamamos halo. Pero en benchmarks con anomalías muy largas la proyección al objetivo no
> ayuda, y ahí conviene usar sólo el umbral y el recorte."

### Glosario
| Término | En simple |
|---|---|
| Serie de tiempo | Mediciones tomadas una tras otra en el tiempo |
| Canal | Cada una de las cosas que se miden a la vez (cada columna de `X`) |
| Anomalía puntual / de patrón | Un solo instante raro / un tramo cuya forma es rara |
| No supervisado | El algoritmo nunca ve las etiquetas de qué es anómalo |
| Contaminación | Proporción de anomalías en los datos |
| LSTM | Red neuronal para secuencias; aquí predice el siguiente instante |
| Gradiente | Hacia dónde y cuánto mover los parámetros para reducir el error de un ejemplo |
| Época | Una pasada completa por todos los datos de entrenamiento |
| Tasa de aprendizaje (η) | Tamaño de cada paso de ajuste |
| Self-influence (TracIn) | Suma, sobre las épocas, de η · ‖gradiente‖² de cada ventana: alto = le costó aprenderla |
| Ventana | Pedazo de L + P instantes consecutivos (aquí 4 de entrada + 1 a predecir) |
| Objetivo | El instante que una ventana predice |
| Proyección | Cómo pasar de un score por ventana a un score por instante |
| Ec. 4 | La proyección del artículo: cada instante suma el score de todas las ventanas que lo contienen |
| Halo | Normales a 1–4 pasos de una anomalía, que con la Ec. 4 heredan su score alto |
| Umbral | Valor a partir del cual se marca "anómalo" |
| IQR | Rango intercuartil, $Q_3 - Q_1$; la regla $Q_3 + 1.5\,IQR$ es la del diagrama de caja |
| z robusto | Cuántas "desviaciones" está algo de la mediana, usando mediana y MAD |
| Recorte / reponderación | Quitar / bajar el peso de ventanas sospechosas al reentrenar |
| AUROC | Probabilidad de que una anomalía tenga más score que un normal (0.5 = azar) |
| AUPRC | Qué tan arriba quedan las anomalías al ordenar por score (sensible a pocas anomalías) |
| F1 | Equilibrio entre precisión y exhaustividad con un umbral dado |
| Semilla | Número que fija el azar para poder repetir un experimento exactamente |
| Wilcoxon pareado | Prueba que compara dos métodos sobre las mismas series |
| Holm | Corrección del p-valor cuando se hacen varias pruebas a la vez |
| IC 95 % bootstrap | Rango probable de la mejora real, obtenido remuestreando |
| Preregistro | Escribir hipótesis y análisis antes de ver los resultados |

### Preguntas que te pueden hacer (y cómo contestarlas)
* **"¿La proyección al objetivo es invento suyo?"** No: es la práctica estándar en detectores basados en predicción.
  Lo nuestro es mostrar que la Ec. 4 del artículo crea un halo de falsos positivos, medir que ese halo explica la
  caída con la contaminación y combinarla con las otras dos mejoras.
* **"¿Por qué IQR con 1.5?"** Es el valor estándar del diagrama de caja. Lo fijamos antes de ver resultados y no lo
  ajustamos.
* **"¿El recorte no quita anomalías que luego ya no detectan?"** No: se quitan del *entrenamiento*, pero se les sigue
  calculando el score. Justamente por no haberlas aprendido, su score queda más alto.
* **"¿Funciona en datos reales?"** Depende del tipo de anomalía. En los 4 datasets del artículo mejora el AUPRC en las
  40 corridas. En SMD y SMAP/MSL, con anomalías de cientos de pasos, no: ahí conviene quitar M3. Y en 109 series
  reales, la ventaja de M3 crece con la proporción de anomalías (H6).
* **"¿Por qué el umbral sin etiquetas pierde en F1 en reales?"** Porque las anomalías son muy raras (0.2–1.3 %) y
  la regla IQR marca 10–17 % de los instantes. El umbral del artículo gana porque mira las respuestas.
* **"¿Cómo saben que no es suerte?"** Pruebas de Wilcoxon pareadas sobre cientos de series, corrección de Holm,
  intervalos de confianza y un protocolo escrito antes de correr.
* **"¿Le ganan a los métodos clásicos?"** En contaminación sintética sí (salvo Isolation Forest, que cae poco pero
  parte de un AUROC mucho más bajo). En el semisintético, LOF y el error del LSTM son mejores que todas las
  variantes de TracIn, incluida la del artículo. Lo decimos tal cual.
* **"¿Qué harían después?"** Una proyección híbrida que use M3 en anomalías cortas y la suma en eventos largos, y un
  umbral sin etiquetas que funcione mejor con anomalías muy raras.
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "toc_visible": True},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT, len(cells), "celdas")
