# Resultados — segunda entrega (Equipo 6)

Reproducción y extensión de Verdone, Scardapane & Panella (2026), *A data
attribution approach for unsupervised anomaly detection on multivariate time
series*, ESWA 297, 129285. Cómo correr todo: [`README.md`](README.md).

> **Convención.** Las cifras marcadas **"Art."** se copiaron de las Tablas 2 y 4
> del artículo; **todo lo demás son resultados nuestros**, calculados con el
> código de `src/`. Este archivo se genera con `python src/write_results.py`
> a partir de `results/tables/`, `results/deck_data.json` y `results/replica/`:
> ninguna cifra está escrita a mano. Las tablas que dicen **"réplica"** usan la
> configuración que reproduce al artículo (§2); las demás, la del primer
> experimento.

## 0. Resumen

1. **Mecanismo (observación 1).** Con MSE y gradientes de la última capa
   lineal, la self-influence de TracIn tiene forma cerrada:
   `SI(x) = Σ_k η_k · (4/D) · ℓ_k(x) · (1 + ‖h_k(x)‖²)`. Es la pérdida de la
   ventana **acumulada durante todo el entrenamiento** (ponderada por el
   learning rate y la norma de su representación). El error de predicción
   pregunta "¿el modelo final se equivoca aquí?"; TracIn pregunta "¿le costó
   ajustarlo a lo largo del entrenamiento?".
2. **Reproducción (§2).** Los datos y casi todos los métodos de referencia se
   reproducen (IForest en GECCO: F1 0.374 vs 0.391 Art.). TracIn
   se reproduce **en sintéticos** con dos supuestos que el artículo no declara:
   **canales estandarizados** y **F1 macro** (promedio del F1 de ambas clases).
   Con ellos, F1 estático 0.710 vs 0.662 Art. y dinámico
   0.703 vs 0.725 Art.; con el F1 binario habitual serían
   0.494 y 0.478. El nivel promedio coincide; el patrón
   por dataset no. En reales, **Credit Card se reproduce**
   (0.225 vs 0.236 Art.), **GECCO y CICIDS no**
   (0.355 vs 0.761; 0.003 vs 0.135) y en
   SWAN-SF obtenemos más (0.378 vs 0.200). Descartamos
   hiperparámetros, gradiente completo vs. última capa, point-adjust, F1
   ponderado y fuga de la etiqueta.
3. **Comparación (observación 2).** Con la configuración reproducida y la
   **misma métrica para todos**, TracIn del artículo queda debajo de los
   detectores clásicos: F1 dinámico macro 0.703
   vs 0.797 (LOF), 0.744 (OCSVM),
   0.736 (IForest) y
   0.832 (error del mismo LSTM). Las referencias
   de su Tabla 2 vienen de Lai et al. (2021), con F1 binario: si TracIn se
   reportó con F1 macro, esa tabla compara métricas distintas. En reales
   tampoco es el mejor en AUPRC en ningún dataset; con la proyección al
   objetivo sí lo es en Credit Card (0.214 vs
   0.160 de IForest; §3.2).
4. **¿Aporta la trayectoria?** Sí, comparando con la misma proyección: con la
   configuración reproducida, TracIn proyectado a los pasos predichos supera al
   error final del mismo LSTM en sintéticos (+0.084 AUPRC
   con L4P1, p = 1.2e-27), en **todos** los niveles de
   contaminación (+0.057 a
   +0.110) y en GECCO
   (+0.227) y Credit Card (+0.147). En el
   primer experimento (datos sin estandarizar) sólo aparecía con 1 % de
   contaminación y en reales: **la estandarización cambia esta conclusión**.
   La ganancia sigue ordenada por cuánto memoriza el modelo las anomalías (§5).
5. **Mejoras propuestas, sobre el TracIn reproducido.**
   *Proyección al objetivo* (mejora 3, salió de la ablación):
   +0.226 AUPRC y
   +0.129 F1 macro
   (p = 4.1e-40): lo lleva al
   nivel de los mejores detectores. *Umbral IQR sin etiquetas* (mejora 1):
   F1 macro 0.838 vs
   0.832 del umbral dinámico, que usa
   etiquetas. *Reponderación* (mejora 2): ≈ 0 en los sintéticos de 400 pasos.
   Con contaminación controlada confirmamos el problema que la motiva: TracIn
   pasa de ser el mejor detector con 1 % de anomalías (AUROC
   0.954) a quedar debajo del error del LSTM con
   26 % (0.789). El recorte frena la caída poco
   (+0.024 AUROC, 10/10 semillas); con la
   proyección al objetivo la caída casi desaparece y es el mejor detector desde
   ≈ 10 % (AUROC 0.896 vs 0.852 de
   LOF con 26 %). En GECCO la reponderación suave
   sube el AUPRC de 0.461 a
   0.522; en SWAN-SF y CICIDS no cambia y en
   Credit Card empeora.

## 1. Qué se corrió

| Experimento | Datos | Métodos | Semillas |
|---|---|---|---|
| 1. Sintéticos | 15 datasets del generador de Lai et al. (400 × 5) | TracIn-LSTM (9 configuraciones L × P), 6 referencias con 4 ventanas, reponderación (3 rondas × 2 variantes) | 20 |
| 2. Reales | Credit Card, CICIDS, GECCO, SWAN-SF | TracIn-LSTM (9 configs, 5 semillas), TracIn-Transformer (L4P1, 3 semillas), 6 referencias, reponderación (L4P1, 5 semillas) | 3–5 |
| 3. Contaminación | Serie «01234» de 4000 pasos, 1–26 % de anomalías | TracIn, proy. objetivo, reponderación, 6 referencias | 10 |
| Diagnósticos | GECCO (sensibilidad a hiperparámetros), reales (LOF vs k) | — | 2 / — |
| **R1. Barrido de réplica** | 15 sintéticos | 13 configuraciones de lo no especificado × 9 (L, P) | 5 |
| **R2. Réplica en sintéticos** | 15 sintéticos, estandarizados | TracIn, proy. objetivo, error final, reponderación (2 rondas × 2), 6 referencias | 20 |
| **R3. Réplica en reales** | Credit Card, CICIDS, GECCO, SWAN-SF | TracIn con gradiente completo, proy. objetivo, error final, reponderación (1 ronda × 2) | 3 |
| **R4. Contaminación (réplica)** | «01234», 4000 pasos, estandarizada | ídem R2 con L4P1 + 5 referencias | 10 |

Primer experimento: 19 702 forecasters entrenados (cada
uno con su traza de checkpoints) y 6 318 ajustes de métodos
de referencia. Réplica: 23 330 forecasters y
6 050 ajustes más.

**Configuración por defecto del primer experimento (fijada antes de ver
resultados).** LSTM/Transformer L = 4, P = 1; IForest, OCSVM, LOF y AE puntuales
(w = 1); Matrix Profile m = 8; datos sin estandarizar; F1 binario. Las
comparaciones "justas" usan esta configuración. Las tablas "estilo artículo"
(incluidas todas las de la réplica) eligen la mejor configuración por dataset
con la métrica de prueba, como hace el artículo.

**Protocolo común.** Mismos instantes evaluados para todos (se excluyen los
19 primeros y últimos), AUPRC y AUROC (sin umbral), F1 con el umbral estático
y dinámico del artículo y con nuestros umbrales sin etiquetas. El umbral
dinámico se elige en la primera mitad de la serie y se mide en la segunda (en
el primer experimento, validación cruzada en ambas direcciones).

## 2. ¿Se reproduce el artículo? Réplica

Nuestro primer experimento (§2.5) no reproducía las cifras de TracIn. Como el
artículo deja sin especificar varias decisiones, montamos una réplica
controlada (`src/replica/`, ver README): probamos cada decisión por separado y,
una vez encontrada la configuración que reproduce sus números, **volvimos a
evaluar sobre ella a los demás detectores y a nuestras mejoras**. No elegimos
nada mirando las etiquetas de prueba más allá de lo que hace el propio
protocolo del artículo (mejor (L, P) por dataset).

### 2.1 Sintéticos: se reproduce con dos supuestos no declarados

1. **Estandarizar cada canal (z-score).** Nuestro primer experimento
   estandarizaba los datos reales pero entrenaba con los sintéticos crudos del
   generador. Sólo con estandarizar, el AUPRC de TracIn
   (L4P1) pasa de 0.450 a 0.583 y el F1
   dinámico macro de 0.631 a 0.712.
2. **F1 macro.** Las cifras de la Tabla 2 coinciden con el promedio del F1 de
   la clase normal y de la anómala (`f1_score(average='macro')`), no con el F1
   de la clase anómala. Ninguna otra lectura que probamos se acerca:

### ¿Qué lectura del F1 reproduce la Tabla 2?

| Lectura del F1 | Estático (promedio) | Error abs. medio (est.) | Dinámico (promedio) | Error abs. medio (din.) | Correlación por dataset (din.) |
|---|---|---|---|---|---|
| Art. (Tabla 2) | 0.662 | — | 0.725 | — | — |
| F1 binario (clase anómala) | 0.494 | 0.184 | 0.478 | 0.249 | -0.41 |
| F1 macro (promedio de ambas clases) | 0.710 | 0.072 | 0.703 | 0.085 | -0.32 |
| F1 binario con point-adjust | 0.564 | 0.148 | 0.508 | 0.224 | -0.50 |
| F1 ponderado por soporte | 0.882 | 0.219 | 0.889 | 0.163 | +0.30 |

TracIn del artículo reproducido (datos estandarizados, 20 semillas). Estático: L4P1, percentil 85; dinámico: percentil elegido en la primera mitad con la misma lectura del F1 y medido en la segunda; mejor (L, P) por dataset. Error abs. medio: error absoluto medio contra la Tabla 2 sobre los 15 datasets.

### Réplica de la Tabla 2 (sintéticos, 20 semillas, datos estandarizados)

| Dataset | Art. LSTM+S | Nuestro S macro | Nuestro S binario | Art. LSTM+D | Nuestro D macro | Nuestro D binario |
|---|---|---|---|---|---|---|
| 0 | 0.660 | 0.632 | 0.337 | 0.807 | 0.597 | 0.265 |
| 1 | 0.644 | 0.612 | 0.303 | 0.807 | 0.602 | 0.278 |
| 2 | 0.619 | 0.694 | 0.448 | 0.700 | 0.804 | 0.461 |
| 3 | 0.648 | 0.705 | 0.469 | 0.695 | 0.774 | 0.446 |
| 4 | 0.615 | 0.670 | 0.405 | 0.746 | 0.691 | 0.419 |
| 01 | 0.693 | 0.623 | 0.339 | 0.778 | 0.590 | 0.333 |
| 12 | 0.655 | 0.703 | 0.481 | 0.642 | 0.676 | 0.444 |
| 23 | 0.653 | 0.841 | 0.720 | 0.770 | 0.825 | 0.668 |
| 34 | 0.667 | 0.814 | 0.674 | 0.781 | 0.795 | 0.646 |
| 012 | 0.690 | 0.635 | 0.377 | 0.701 | 0.624 | 0.432 |
| 123 | 0.671 | 0.765 | 0.599 | 0.673 | 0.716 | 0.514 |
| 234 | 0.655 | 0.825 | 0.703 | 0.636 | 0.769 | 0.647 |
| 0123 | 0.681 | 0.699 | 0.500 | 0.737 | 0.673 | 0.495 |
| 1234 | 0.706 | 0.739 | 0.565 | 0.720 | 0.716 | 0.557 |
| 01234 | 0.678 | 0.690 | 0.496 | 0.687 | 0.691 | 0.559 |
| Promedio | 0.662 | 0.710 | 0.494 | 0.725 | 0.703 | 0.478 |

S: L=4, P=1, umbral = percentil 85 de los scores (lectura literal del texto). D: umbral (percentil) elegido en la primera mitad de la serie, F1 medido en la segunda; mejor (L, P) por dataset según la media de 20 semillas, como el artículo. 'macro' = promedio del F1 de la clase normal y de la anómala (f1_score(average='macro')).

![Réplica de la Tabla 2](figures/fig_replica_tabla2.png)

Con estos dos supuestos el **nivel** de la Tabla 2 se reproduce: F1 estático
0.710 vs 0.662 Art. y dinámico 0.703 vs 0.725 Art. El
**patrón por dataset no**: la correlación entre nuestros F1 dinámicos y los
suyos es −0.32. Ellos obtienen sus mejores cifras en las
anomalías puntuales (0 y 1: 0.807), donde nosotros obtenemos las peores. Lo
atribuimos a cómo se etiquetan y proyectan las anomalías puntuales (un único
instante marcado contra un score repartido en L + P posiciones), pero no
pudimos verificarlo sin su código.

Lo demás que el artículo no especifica casi no mueve el resultado:

### Sintéticos: barrido de lo que el artículo no especifica (TracIn, 5 semillas)

| Variante | AUPRC (L4P1) | S macro | D macro | D binario |
|---|---|---|---|---|
| Art. (Tabla 2) | — | 0.662 | 0.725 | — |
| Base: z-score por canal, lr 1e-3, batch 32, ≤ 100 épocas (paciencia 10), LSTM 64, gradiente de la última capa | 0.583 | 0.710 | 0.712 | 0.502 |
| Gradiente de todos los parámetros (Ec. 2 completa) | 0.586 | 0.713 | 0.712 | 0.495 |
| lr 1e-2 | 0.539 | 0.701 | 0.704 | 0.482 |
| batch 8 | 0.561 | 0.705 | 0.715 | 0.498 |
| ≤ 300 épocas (paciencia 30) | 0.558 | 0.703 | 0.713 | 0.495 |
| 30 épocas fijas | 0.586 | 0.703 | 0.690 | 0.465 |
| LSTM de 128 | 0.578 | 0.711 | 0.708 | 0.495 |
| LSTM de 16 | 0.550 | 0.686 | 0.690 | 0.463 |
| Sin reducción del lr | 0.584 | 0.710 | 0.714 | 0.501 |
| Sin ponderar por el lr (η_k = 1) | 0.584 | 0.710 | 0.714 | 0.502 |
| 20 % final como validación para early stopping | 0.566 | 0.695 | 0.678 | 0.452 |
| Normalización min-max | 0.504 | 0.673 | 0.686 | 0.421 |
| Sin normalizar (nuestro primer experimento) | 0.450 | 0.656 | 0.631 | 0.392 |

Cada fila cambia sólo eso respecto de la base. 5 semillas × 15 datasets × 9 (L, P); mismo protocolo que la Tabla 2 (estático L4P1 percentil 85, dinámico con la mejor (L, P) por dataset).

### 2.2 Reales: Credit Card sí, GECCO y CICIDS no

En reales el F1 de la Tabla 4 tiene que ser **binario**: en Credit Card y
CICIDS la clase normal es > 98 % de los datos; con cualquier umbral que marque
una minoría de instantes su F1 es ≈ 1 y el F1 macro no baja de ≈ 0.5, pero
reportan 0.236 y 0.135. La réplica en reales usa el
gradiente de todos los parámetros (Ec. 2 completa), que en GECCO subió el
AUPRC de 0.398 a 0.469.

### Réplica de la Tabla 4 (reales, TracIn con gradientes de todos los parámetros)

| Dataset | Art. LSTM+S (P/R/F1) | Nuestro S pct87 binario | Nuestro S τ·máx binario | Art. LSTM+D (P/R/F1) | Nuestro D binario | Nuestro D macro | Nuestro D binario, partición aleatoria | AUPRC |
|---|---|---|---|---|---|---|---|---|
| creditcard | 0.119/0.609/0.200 | 0.010/0.772/0.020 | 0.583/0.003/0.005 | 0.280/0.204/0.236 | 0.208/0.253/0.225 (L4P1) | 0.612 | 0.203 | 0.116 |
| cicids | 0.059/0.130/0.081 | 0.016/0.158/0.028 | 0.000/0.000/0.000 | 0.073/0.961/0.135 | 0.001/0.692/0.003 (L4P2) | 0.498 | 0.035 | 0.017 |
| gecco | 0.539/0.686/0.604 | 0.072/0.754/0.132 | 1.000/0.003/0.005 | 0.750/0.784/0.761 | 0.352/0.368/0.355 (L4P4) | 0.672 | 0.510 | 0.461 |
| swan_sf | 0.148/0.190/0.167 | 0.405/0.221/0.286 | 0.594/0.000/0.000 | 0.143/0.335/0.200 | 0.379/0.380/0.378 (L4P1) | 0.548 | 0.375 | 0.339 |

S: L=4, P=1; percentil 87 (lectura literal) o τ = 0.87 como fracción del máximo. D: umbral elegido en la primera mitad (contigua) y F1 medido en la segunda, o en una partición aleatoria 50/50 de los instantes. Mejor (L, P) por dataset; 3 semillas por configuración, como el artículo. F1 binario (clase anómala), como en la Tabla 4. (L, P) evaluadas: creditcard: L4P1, L4P2, L4P4, L8P1, L8P2, L8P4, L16P1, L16P2, L16P4; cicids: L4P1, L4P2, L4P4, L8P1, L8P2, L8P4, L16P1, L16P2, L16P4; gecco: L4P1, L4P2, L4P4, L8P1, L8P2, L8P4, L16P1, L16P2, L16P4; swan_sf: L4P1, L4P2, L4P4, L8P1, L8P2, L8P4, L16P1, L16P2, L16P4.

* **Credit Card se reproduce**: F1 dinámico 0.225 vs
  0.236 Art.
* **GECCO no**: 0.355 vs 0.761 Art. (con partición
  aleatoria de los instantes, 0.510).
* **CICIDS no**: 0.003 vs 0.135 Art. Los ataques están
  concentrados en la primera mitad de la serie, así que un umbral elegido en
  la primera mitad y medido en la segunda casi no tiene anomalías que
  detectar; con partición aleatoria sube a 0.035.
* **SWAN-SF**: obtenemos más que el artículo (0.378 vs
  0.200 Art.).
* El umbral estático: con el percentil 87 y F1 binario la precisión no puede
  superar la tasa de anomalías / 0.13 (≈ 0.096 en GECCO), pero reportan
  P = 0.539; en reales sus cifras sólo son posibles si τ es una fracción del
  máximo. Con esa lectura, nuestro TracIn casi no marca instantes (F1 ≤ 0.01 en
  los cuatro datasets): el umbral estático en reales tampoco se reproduce. En sintéticos, en cambio, la lectura como percentil (con F1 macro) es
  la que reproduce: con τ·máx el F1 estático macro sería
  0.552, contra 0.710 con el percentil y 0.662 Art.

### 2.3 Hipótesis descartadas

| Hipótesis | Prueba | Resultado |
|---|---|---|
| Hiperparámetros de entrenamiento (lr, batch, épocas, tamaño, scheduler, validación) | 10 variantes de entrenamiento en sintéticos (tabla de §2.1), 6 en GECCO | F1 dinámico macro entre 0.678 y 0.715 en sintéticos; ninguna variante acerca GECCO a 0.761 |
| Gradiente de la última capa vs. de todos los parámetros | variante `full` | Igual en sintéticos (0.712 vs 0.712); mejora GECCO, por eso se usa en reales |
| Point-adjust | lectura `pa` (tabla de §2.1) | Sintéticos: 0.508; GECCO (primer experimento): IForest pasaría de 0.39 a 0.62, lejos del 0.391 publicado, que sí coincide sin point-adjust |
| F1 ponderado por soporte | lectura `wtd` | 0.889: muy por encima de la Tabla 2 |
| Evaluar a nivel de ventana, o sólo en la parte de prueba | `h_level.py`, `h_eval.py` (primer experimento) | No cierra la diferencia |
| La etiqueta entra como canal (los canales de la Tabla 1 coinciden con atributos + etiqueta) | `real_screen.py --leak` | Inconsistente: GECCO sube a 0.650, pero CICIDS queda en 0.004 y SWAN-SF en 0.473 (Art. 0.200) |

### Reales: hipótesis probadas para explicar la diferencia con la Tabla 4

| Hipótesis | AUPRC | F1 dinámico binario | Art. |
|---|---|---|---|
| GECCO · Base: z-score por canal, lr 1e-3, LSTM 64, gradiente de la última capa | 0.398 | 0.268 | 0.761 |
| GECCO · Gradiente de todos los parámetros (Ec. 2 completa) | 0.469 | 0.340 | 0.761 |
| GECCO · lr 1e-2 | 0.441 | 0.291 | 0.761 |
| GECCO · Sin ponderar por el lr (η_k = 1) | 0.405 | 0.282 | 0.761 |
| GECCO · 20 % final como validación para early stopping | 0.378 | 0.260 | 0.761 |
| GECCO · Normalización min-max | 0.358 | 0.294 | 0.761 |
| creditcard · la etiqueta entra como canal (fuga) | 0.123 | 0.153 | 0.236 |
| cicids · la etiqueta entra como canal (fuga) | 0.025 | 0.004 | 0.135 |
| gecco · la etiqueta entra como canal (fuga) | 0.728 | 0.650 | 0.761 |
| swan_sf · la etiqueta entra como canal (fuga) | 0.387 | 0.473 | 0.200 |

Barrido en GECCO: 2 semillas, L4P1 y L16P4, 50 épocas, batch 256. Fuga: se añade la etiqueta estandarizada como un canal más (los conteos de canales de la Tabla 1 coinciden con atributos + etiqueta); 2 semillas, L4P1 y L16P4. Mejor configuración por dataset. Diagnóstico, no un método válido.

### 2.4 Qué implica para la comparación del artículo

Las cifras de los métodos de referencia de su Tabla 2 provienen de Lai et al.
(2021), que usan F1 **binario** (las reprodujimos así en §2.5). Si TracIn se
reportó con F1 macro, como indica la réplica, la Tabla 2 compara métricas
distintas: cuando las anomalías son minoría, el F1 de la clase normal es
cercano a 1 y el F1 macro de un detector queda muy por encima de su F1 binario
(TracIn: 0.703 macro vs 0.478 binario). Con la **misma** métrica para todos (§3.1),
los detectores clásicos quedan por encima de TracIn.

### 2.5 Primer intento (sin estandarizar, F1 binario)

### Sintéticos — reproducción de la Tabla 2 (F1, estilo artículo)

| Dataset | Art. LSTM+S | Nuestro TracIn+S (τ·máx) | Nuestro TracIn+S (pct) | Art. LSTM+D | Nuestro TracIn+D | Nuestro TracIn F1 oráculo | Art. mejor baseline | Nuestro mejor baseline |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.660 | 0.148 ± 0.049 | 0.302 | 0.807 | 0.313 ± 0.059 | 0.367 | 0.900 (OCSVM) | 0.840 (ocsvm) |
| 1 | 0.644 | 0.130 ± 0.044 | 0.252 | 0.807 | 0.250 ± 0.066 | 0.331 | 0.750 (OCSVM) | 0.601 (lof) |
| 2 | 0.619 | 0.269 ± 0.151 | 0.215 | 0.700 | 0.464 ± 0.284 | 0.587 | 0.470 (LSTM) | 0.793 (mp) |
| 3 | 0.648 | 0.244 ± 0.124 | 0.449 | 0.695 | 0.694 ± 0.134 | 0.820 | 0.700 (IForest) | 0.867 (iforest) |
| 4 | 0.615 | 0.141 ± 0.119 | 0.227 | 0.746 | 0.356 ± 0.251 | 0.546 | 0.670 (LSTM) | 0.734 (lof) |
| 01 | 0.693 | 0.122 ± 0.074 | 0.352 | 0.778 | 0.338 ± 0.049 | 0.385 | 0.810 (OCSVM) | 0.759 (ocsvm) |
| 12 | 0.655 | 0.095 ± 0.035 | 0.395 | 0.642 | 0.379 ± 0.103 | 0.443 | 0.538 (IForest) | 0.556 (lof) |
| 23 | 0.653 | 0.188 ± 0.110 | 0.571 | 0.770 | 0.617 ± 0.122 | 0.721 | 0.525 (OCSVM) | 0.725 (iforest) |
| 34 | 0.667 | 0.154 ± 0.098 | 0.455 | 0.781 | 0.448 ± 0.245 | 0.624 | 0.753 (IForest) | 0.749 (lof) |
| 012 | 0.690 | 0.091 ± 0.054 | 0.356 | 0.701 | 0.370 ± 0.066 | 0.441 | 0.649 (OCSVM) | 0.723 (ocsvm) |
| 123 | 0.671 | 0.129 ± 0.070 | 0.564 | 0.673 | 0.543 ± 0.092 | 0.606 | 0.593 (IForest) | 0.668 (lof) |
| 234 | 0.655 | 0.107 ± 0.084 | 0.438 | 0.636 | 0.431 ± 0.146 | 0.566 | 0.433 (OCSVM) | 0.651 (mp) |
| 0123 | 0.681 | 0.089 ± 0.066 | 0.476 | 0.737 | 0.504 ± 0.048 | 0.554 | 0.696 (OCSVM) | 0.743 (ocsvm) |
| 1234 | 0.706 | 0.090 ± 0.057 | 0.434 | 0.720 | 0.445 ± 0.120 | 0.524 | 0.675 (IForest) | 0.612 (lof) |
| 01234 | 0.678 | 0.070 ± 0.054 | 0.421 | 0.687 | 0.471 ± 0.111 | 0.539 | 0.673 (OCSVM) | 0.661 (ocsvm) |
| Promedio | 0.662 | 0.138 | 0.394 | 0.725 | 0.441 | 0.537 | 0.656 | 0.712 |

TracIn+S: L=4, P=1, τ=0.85 (lectura principal: fracción del máximo; ‘pct’ = lectura literal como percentil). TracIn+D: mejor configuración (L, P) por dataset según F1 dinámico medio (oráculo de configuración, como el artículo). Baselines: mejor (ventana, contaminación) por dataset, como en Lai et al. (2021).

### Reales — reproducción de la Tabla 4 (F1, estilo artículo)

| Dataset | Art. LSTM+S | Nuestro TracIn+S (τ·máx) | Nuestro TracIn+S (pct) | Art. LSTM+D | Nuestro TracIn+D | Nuestro TracIn F1 oráculo | Art. mejor baseline | Nuestro mejor baseline |
|---|---|---|---|---|---|---|---|---|
| Credit Card | 0.200 | 0.006 ± 0.004 | 0.020 | 0.236 | 0.129 (L4P4) | 0.139 (L4P4) | 0.230 (USAD) | 0.183 (ocsvm) |
| CICIDS | 0.081 | 0.000 ± 0.000 | 0.029 | 0.135 | 0.038 (L4P1) | 0.038 (L4P1) | 0.046 (LSTM) | 0.047 (ae) |
| GECCO | 0.604 | 0.002 ± 0.000 | 0.123 | 0.761 | 0.323 (L16P4) | 0.481 (L16P4) | 0.558 (USAD) | 0.581 (ae) |
| SWAN-SF | 0.167 | 0.000 ± 0.000 | 0.274 | 0.200 | 0.328 (L16P4) | 0.398 (L4P4) | 0.583 (IForest) | 0.408 (iforest) |

Mismas convenciones que en sintéticos; τ = 0.87.

En reales, nuestras referencias quedan cerca de las publicadas salvo en SWAN-SF
(IForest 0.408 vs 0.583 Art.); USAD no lo
implementamos. TracIn queda muy por debajo excepto en SWAN-SF, donde nuestro
F1 (0.328) supera al publicado (0.200).

**Umbral estático (primer intento).** El artículo lo describe como el
percentil τ = 0.85/0.87. Con esa lectura se marca el 13–15 % de los instantes;
en GECCO (1.25 % de anomalías) la precisión no podría pasar de ≈ 0.096, pero
la Tabla 4 reporta P = 0.539. Sus cifras sólo son posibles si τ es una
**fracción del máximo** del score normalizado. Calculamos ambas lecturas; con
ninguna nos acercamos a lo publicado. Esto corrige lo que dijimos en la
primera presentación ("marca siempre el 13–15 %").

**Sensibilidad en GECCO (primer intento).** Probamos lo que el artículo no especifica (learning
rate, batch, tamaño oculto, épocas) para descartar que el hueco se deba a un
mal ajuste:

### GECCO — sensibilidad de TracIn a hiperparámetros no especificados (media de 2 semillas)

| Variante | L | P | AUPRC | F1 oráculo | F1 dinámico | F1 estático (τ·máx) |
|---|---|---|---|---|---|---|
| 150 épocas | 4 | 1 | 0.245 | 0.339 | 0.316 | 0.002 |
| 150 épocas | 8 | 2 | 0.360 | 0.415 | 0.297 | 0.006 |
| 150 épocas | 16 | 4 | 0.374 | 0.438 | 0.295 | 0.016 |
| base (lr 1e-3, b256, h64, 50 ép.) | 4 | 1 | 0.286 | 0.386 | 0.297 | 0.002 |
| base (lr 1e-3, b256, h64, 50 ép.) | 8 | 2 | 0.393 | 0.443 | 0.299 | 0.006 |
| base (lr 1e-3, b256, h64, 50 ép.) | 16 | 4 | 0.401 | 0.476 | 0.319 | 0.015 |
| batch 64 | 4 | 1 | 0.304 | 0.384 | 0.323 | 0.002 |
| batch 64 | 8 | 2 | 0.387 | 0.437 | 0.319 | 0.006 |
| batch 64 | 16 | 4 | 0.396 | 0.478 | 0.317 | 0.016 |
| lr 1e-2 | 4 | 1 | 0.375 | 0.401 | 0.318 | 0.003 |
| lr 1e-2 | 8 | 2 | 0.417 | 0.443 | 0.338 | 0.007 |
| lr 1e-2 | 16 | 4 | 0.454 | 0.512 | 0.342 | 0.016 |
| lr 1e-2 + batch 64 | 4 | 1 | 0.362 | 0.411 | 0.364 | 0.003 |
| lr 1e-2 + batch 64 | 8 | 2 | 0.433 | 0.448 | 0.316 | 0.007 |
| lr 1e-2 + batch 64 | 16 | 4 | 0.455 | 0.478 | 0.318 | 0.016 |
| oculto 128 | 4 | 1 | 0.353 | 0.428 | 0.328 | 0.002 |
| oculto 128 | 8 | 2 | 0.395 | 0.443 | 0.309 | 0.006 |
| oculto 128 | 16 | 4 | 0.404 | 0.480 | 0.321 | 0.015 |

Artículo: LSTM+S 0.604, LSTM+D 0.761.

## 3. Comparación con otros detectores (observación 2)

### 3.1 Sintéticos

**Réplica (protocolo del artículo, misma métrica para todos).**

### Sintéticos: TracIn reproducido vs. otros detectores, mismo protocolo del artículo

| Método | AUPRC (mejor config) | F1 dinámico macro | F1 dinámico binario | F1 IQR macro (sin etiquetas) | F1 IQR binario (sin etiquetas) |
|---|---|---|---|---|---|
| TracIn (artículo) | 0.606 | 0.703 | 0.478 | 0.712 | 0.488 |
| TracIn proy. objetivo | 0.832 | 0.832 | 0.679 | 0.838 | 0.709 |
| Error LSTM (PD) | 0.810 | 0.832 | 0.673 | 0.832 | 0.699 |
| Isolation Forest | 0.664 | 0.736 | 0.527 | 0.715 | 0.475 |
| One-Class SVM | 0.608 | 0.744 | 0.517 | 0.754 | 0.554 |
| LOF | 0.736 | 0.797 | 0.614 | 0.805 | 0.650 |
| Autoencoder | 0.660 | 0.711 | 0.477 | 0.699 | 0.451 |
| Matrix Profile | 0.553 | 0.686 | 0.436 | 0.665 | 0.391 |

Mejor configuración por dataset (L, P para el LSTM; ventana para los demás), media de 20 semillas, promedio de 15 datasets. Todos sobre los mismos datos estandarizados y los mismos instantes.

Con los datos estandarizados y la mejor configuración por dataset para cada
método, TracIn del artículo queda por debajo de LOF, OCSVM, IForest y del
error de su propio LSTM, en F1 macro, F1 binario y AUPRC. Con la proyección al
objetivo (mejora 3) pasa al primer lugar, empatado con el error del LSTM. El
autoencoder y Matrix Profile quedan al nivel de TracIn o por debajo.

**Primer experimento (configuración fija a priori, datos sin estandarizar,
F1 binario).**

### Sintéticos — comparación con configuración fija a priori (promedio de 15 datasets, 20 semillas)

| Método | Familia | AUPRC | AUROC | F1 oráculo | F1 dinámico (art., usa etiquetas) | F1 IQR (sin etiquetas) | Gana (AUPRC) |
|---|---|---|---|---|---|---|---|
| TracIn (artículo) | Atribución / predicción | 0.434 ± 0.055 | 0.789 ± 0.030 | 0.518 ± 0.044 | 0.426 ± 0.063 | 0.395 ± 0.061 | 0 |
| TracIn proy. objetivo | Atribución / predicción | 0.626 ± 0.039 | 0.818 ± 0.031 | 0.641 ± 0.035 | 0.566 ± 0.055 | 0.580 ± 0.045 | 2 |
| LSTM error (PD) | Atribución / predicción | 0.678 ± 0.037 | 0.858 ± 0.021 | 0.682 ± 0.034 | 0.610 ± 0.040 | 0.609 ± 0.030 | 6 |
| Isolation Forest | Mayoría | 0.579 ± 0.031 | 0.824 ± 0.023 | 0.591 ± 0.032 | 0.506 ± 0.039 | 0.464 ± 0.043 | 0 |
| One-Class SVM | Mayoría | 0.644 ± 0.027 | 0.811 ± 0.013 | 0.675 ± 0.024 | 0.612 ± 0.032 | 0.618 ± 0.024 | 2 |
| LOF | Mayoría | 0.678 ± 0.034 | 0.822 ± 0.020 | 0.702 ± 0.030 | 0.632 ± 0.045 | 0.656 ± 0.032 | 2 |
| Autoencoder | Mayoría | 0.393 ± 0.045 | 0.655 ± 0.037 | 0.436 ± 0.044 | 0.358 ± 0.048 | 0.325 ± 0.062 | 0 |
| Matrix Profile | Discordia | 0.497 ± 0.042 | 0.826 ± 0.026 | 0.556 ± 0.035 | 0.467 ± 0.046 | 0.397 ± 0.043 | 3 |

Todos los métodos se evalúan sobre los mismos instantes y con los mismos protocolos de umbral. ‘F1 dinámico’ = umbral F1-óptimo elegido en la otra mitad de la serie (validación cruzada de 2 bloques). ‘Gana’ = número de datasets en que el método tiene la mayor AUPRC media.

![Comparación en sintéticos](figures/fig_sinteticos_comparacion.png)

![Mapa de calor por dataset](figures/fig_sinteticos_heatmap.png)

TracIn del artículo es fuerte en anomalías de patrón y débil en puntuales
(AUPRC 0.217 en puntuales vs
0.488 en las de patrón): al repartir el score
sobre las L + P posiciones de cada ventana, un pico de un instante se vuelve
una meseta ancha.

### 3.2 Reales

**Réplica (TracIn con gradiente completo, mejor (L, P); referencias del primer
experimento con el mismo protocolo de evaluación).**

### Reales: TracIn de la réplica vs. otros detectores (AUPRC / F1 dinámico binario)

| Dataset | TracIn (réplica) | TracIn proy. objetivo | IForest | OCSVM | LOF | Autoencoder | Matrix Profile |
|---|---|---|---|---|---|---|---|
| creditcard | 0.116 / 0.225 | 0.214 / 0.282 | 0.160 / 0.192 | 0.131 / 0.206 | 0.014 / 0.067 | 0.121 / 0.163 | 0.001 / 0.002 |
| cicids | 0.017 / 0.003 | 0.015 / 0.003 | 0.022 / 0.004 | 0.017 / 0.004 | 0.018 / 0.003 | 0.021 / 0.004 | 0.010 / 0.001 |
| gecco | 0.461 / 0.355 | 0.438 / 0.357 | 0.414 / 0.313 | 0.463 / 0.323 | 0.086 / 0.061 | 0.621 / 0.467 | 0.091 / 0.114 |
| swan_sf | 0.339 / 0.378 | 0.376 / 0.383 | 0.616 / 0.542 | 0.464 / 0.466 | 0.419 / 0.483 | 0.413 / 0.478 | 0.327 / 0.497 |

Mejor configuración por dataset; mismo protocolo (umbral elegido en la primera mitad, F1 en la segunda).

* En la réplica, TracIn del artículo no es el mejor en AUPRC en ningún
  dataset. Con la proyección al objetivo sí lo es en **Credit Card**
  (AUPRC 0.214, F1
  0.282, contra
  0.160 / 0.192 de IForest);
  en GECCO gana el autoencoder (0.621) y en SWAN-SF
  IForest (0.616).

**Primer experimento (configuración fija a priori).**

### Reales — AUPRC, configuración fija a priori

| Método | Credit Card | CICIDS | GECCO | SWAN-SF |
|---|---|---|---|---|
| TracIn (artículo) | 0.065 ± 0.003 | 0.017 ± 0.000 | 0.282 ± 0.021 | 0.325 ± 0.006 |
| TracIn proy. objetivo | 0.172 ± 0.002 | 0.018 ± 0.000 | 0.284 ± 0.023 | 0.383 ± 0.006 |
| LSTM error (PD) | 0.069 ± 0.004 | 0.022 ± 0.000 | 0.103 ± 0.011 | 0.314 ± 0.003 |
| Isolation Forest | 0.144 ± 0.046 | 0.022 ± 0.002 | 0.399 ± 0.045 | 0.613 ± 0.018 |
| One-Class SVM | 0.108 ± 0.008 | 0.017 ± 0.000 | 0.430 ± 0.006 | 0.462 ± 0.003 |
| LOF | 0.002 | 0.009 | 0.023 | 0.222 |
| Autoencoder | 0.123 ± 0.023 | 0.016 ± 0.001 | 0.492 ± 0.051 | 0.417 ± 0.009 |
| Matrix Profile | 0.001 | 0.008 | 0.016 | 0.326 |
| TracIn con Transformer | 0.068 ± 0.004 | 0.018 ± 0.000 | 0.273 ± 0.016 | 0.287 ± 0.004 |

Media ± desv. estándar entre semillas (LSTM 5, Transformer 3, IForest/OCSVM/AE 5; LOF y MP son deterministas).

### Reales — F1 dinámico, configuración fija a priori

| Método | Credit Card | CICIDS | GECCO | SWAN-SF |
|---|---|---|---|---|
| TracIn (artículo) | 0.102 ± 0.005 | 0.038 ± 0.000 | 0.291 ± 0.019 | 0.323 ± 0.008 |
| TracIn proy. objetivo | 0.189 ± 0.003 | 0.033 ± 0.027 | 0.320 ± 0.014 | 0.300 ± 0.003 |
| LSTM error (PD) | 0.121 ± 0.009 | 0.062 ± 0.003 | 0.207 ± 0.030 | 0.278 ± 0.004 |
| Isolation Forest | 0.244 ± 0.040 | 0.051 ± 0.026 | 0.339 ± 0.039 | 0.498 ± 0.006 |
| One-Class SVM | 0.223 ± 0.019 | 0.062 ± 0.000 | 0.372 ± 0.006 | 0.428 ± 0.004 |
| LOF | 0.006 | 0.026 | 0.066 | 0.384 |
| Autoencoder | 0.209 ± 0.016 | 0.024 ± 0.025 | 0.434 ± 0.061 | 0.405 ± 0.006 |
| Matrix Profile | 0.003 | 0.025 | 0.027 | 0.372 |
| TracIn con Transformer | 0.100 ± 0.002 | 0.033 ± 0.008 | 0.371 ± 0.012 | 0.304 ± 0.001 |

Media ± desv. estándar entre semillas (LSTM 5, Transformer 3, IForest/OCSVM/AE 5; LOF y MP son deterministas).

![Reales](figures/fig_reales_auprc.png)

* TracIn del artículo no es el mejor en ningún dataset. IForest, OCSVM y el
  autoencoder lo superan en Credit Card, GECCO y SWAN-SF.
* En CICIDS ningún método llega a 2× el azar (véase §5: los ataques se
  predicen mejor que el tráfico normal).
* LOF (k = 20) y Matrix Profile se desploman en reales. Para LOF la causa es
  que las anomalías reales son colectivas (eventos contiguos, fraudes
  parecidos) y forman grupos más grandes que k:

### Reales — diagnóstico: AUPRC de LOF según el número de vecinos k

| Dataset | k=20 | k=50 | k=100 | k=200 | k=500 |
|---|---|---|---|---|---|
| Credit Card | 0.002 | 0.002 | 0.002 | 0.007 | 0.208 |
| CICIDS | 0.009 | 0.009 | 0.008 | 0.009 | 0.009 |
| GECCO | 0.023 | 0.104 | 0.186 | 0.276 | 0.712 |
| SWAN-SF | 0.222 | 0.218 | 0.224 | 0.262 | 0.268 |

k = 20 es el valor por defecto (PyOD) usado en la comparación. Los demás k se eligen mirando etiquetas: sólo explican el fallo de LOF (anomalías colectivas más numerosas que k), no son un resultado comparable.

## 4. Ablación: ¿qué hace funcionar al score?

Mismo LSTM y mismos checkpoints; sólo cambia qué se mide (pérdida final,
pérdida integrada sobre el entrenamiento, self-influence) y cómo se proyecta a
instantes (las L + P posiciones de la ventana, como el artículo, o sólo los P
pasos predichos).

### Sintéticos — ablación del score de TracIn (AUPRC)

| Score | Proyección | AUPRC Puntuales (0,1) | AUPRC De patrón (2,3,4) | AUPRC Los 15 | AUPRC 15 (9 configs) |
|---|---|---|---|---|---|
| Pérdida final | ventana completa | 0.273 ± 0.042 | 0.655 ± 0.114 | 0.493 ± 0.047 | 0.405 |
| Pérdida integrada | ventana completa | 0.240 ± 0.054 | 0.484 ± 0.138 | 0.442 ± 0.056 | 0.370 |
| Self-influence (TracIn) | ventana completa | 0.217 ± 0.038 | 0.488 ± 0.164 | 0.434 ± 0.055 | 0.365 |
| Pérdida final | pasos predichos | 0.844 ± 0.059 | 0.497 ± 0.097 | 0.678 ± 0.037 | 0.665 |
| Pérdida integrada | pasos predichos | 0.772 ± 0.062 | 0.472 ± 0.104 | 0.630 ± 0.041 | 0.571 |
| Self-influence (TracIn) | pasos predichos | 0.758 ± 0.056 | 0.467 ± 0.106 | 0.626 ± 0.039 | 0.563 |

L=4, P=1. Wilcoxon pareado sobre (dataset, semilla): TracIn − pérdida final (misma proyección) = -0.058 (p = 1.3e-07); TracIn proy. objetivo − TracIn = +0.191 (p = 4.2e-30); TracIn proy. objetivo − PD = -0.053 (p = 3.5e-11).

### Reales — ablación del score de TracIn (AUPRC, L=4, P=1)

| Score | Proyección | Credit Card | CICIDS | GECCO | SWAN-SF |
|---|---|---|---|---|---|
| Pérdida final | ventana completa | 0.019 ± 0.001 | 0.019 ± 0.000 | 0.136 ± 0.007 | 0.272 ± 0.002 |
| Pérdida integrada | ventana completa | 0.029 ± 0.000 | 0.019 ± 0.000 | 0.229 ± 0.001 | 0.280 ± 0.001 |
| Self-influence (TracIn) | ventana completa | 0.065 ± 0.003 | 0.017 ± 0.000 | 0.282 ± 0.021 | 0.325 ± 0.006 |
| Pérdida final | pasos predichos | 0.069 ± 0.004 | 0.022 ± 0.000 | 0.103 ± 0.011 | 0.314 ± 0.003 |
| Pérdida integrada | pasos predichos | 0.108 ± 0.001 | 0.022 ± 0.000 | 0.224 ± 0.000 | 0.338 ± 0.001 |
| Self-influence (TracIn) | pasos predichos | 0.172 ± 0.002 | 0.018 ± 0.000 | 0.284 ± 0.023 | 0.383 ± 0.006 |

![Ablación](figures/fig_ablacion.png)

* En sintéticos la trayectoria **no** aporta: TracIn − pérdida final =
  −0.058 (p = 1.3e-07).
* La **proyección** domina: asignar el score sólo a los pasos predichos sube
  el AUPRC en +0.191. Es nuestra mejora 3, no
  prevista en el plan de la primera presentación.
* En reales, en cambio, la trayectoria sí aporta en GECCO, Credit Card y
  SWAN-SF. La sección siguiente explica por qué.

**Réplica: con datos estandarizados la trayectoria sí aporta.** La comparación
limpia es TracIn proyectado al objetivo contra el error final del mismo LSTM
(misma proyección, mismo modelo):

### ¿Aporta la trayectoria? TracIn proyectado al objetivo − error final del mismo LSTM

| Datos | Δ AUPRC (gana TracIn) | p (Wilcoxon) | Δ F1 dinámico macro |
|---|---|---|---|
| Sintéticos, mejor (L, P) por dataset | +0.023 (212/300) | 4.3e-13 | +0.000 (139/300) |
| Sintéticos, L4P1 fijo | +0.084 (234/300) | 1.2e-27 | +0.029 (180/300) |
| Contaminación 1.1 % | +0.110 (9/10) | 9.8e-03 | — |
| Contaminación 4.9 % | +0.105 (10/10) | 2.0e-03 | — |
| Contaminación 9.5 % | +0.088 (10/10) | 2.0e-03 | — |
| Contaminación 13.8 % | +0.069 (10/10) | 2.0e-03 | — |
| Contaminación 18.2 % | +0.068 (10/10) | 2.0e-03 | — |
| Contaminación 22.3 % | +0.057 (10/10) | 2.0e-03 | — |
| Contaminación 25.9 % | +0.057 (10/10) | 2.0e-03 | — |
| creditcard (L4P1) | +0.147 (3/3) | — | — |
| cicids (L4P1) | -0.007 (0/3) | — | — |
| gecco (L4P4) | +0.227 (3/3) | — | — |
| swan_sf (L4P1) | +0.063 (3/3) | — | — |

Misma proyección (pasos predichos) y mismo modelo: la única diferencia es medir la pérdida a lo largo del entrenamiento (self-influence) o sólo al final. Entre paréntesis, (dataset, semilla) en que gana TracIn. Configuración reproducida (datos estandarizados; reales con gradiente completo).

* Con la configuración reproducida, la trayectoria gana en sintéticos
  (+0.084 AUPRC con L4P1; +0.023 con la
  mejor (L, P)), en todos los niveles de contaminación y en GECCO y Credit
  Card. En F1 dinámico con la mejor (L, P) la diferencia en sintéticos es nula.
* Esto **contradice** la conclusión del primer experimento para los
  sintéticos (allí la pérdida final era igual o mejor). La diferencia entre
  ambos es la estandarización de los canales; no verificamos el mecanismo.
* La **proyección** sigue siendo lo que más pesa: TracIn proyectado al
  objetivo − TracIn del artículo = +0.226
  AUPRC en la réplica.

## 5. ¿Cuándo aporta la trayectoria? Memorización

Medimos, época por época, la razón entre la pérdida de las ventanas cuyo
paso predicho es anómalo y la de las normales. Si cae desde su máximo, el
modelo está aprendiendo (memorizando) las anomalías y el error final las deja
de ver; ése es el caso que motiva al artículo.

![Trayectorias](figures/fig_mecanismo_trayectorias.png)

### Contaminación — memorización del LSTM y dónde aporta la trayectoria

| Contaminación | Razón máx. | Razón final | Final / máx. | Δ AUPRC proy. obj. − PD (p) | Δ AUPRC soft − original (p) |
|---|---|---|---|---|---|
| 1.1 % | 34.7× | 16.8× | 0.44 | +0.121 (0.002) | +0.034 (0.049) |
| 4.9 % | 25.7× | 14.7× | 0.60 | -0.047 (0.85) | +0.029 (0.027) |
| 9.5 % | 20.2× | 16.0× | 0.75 | -0.117 (0.064) | +0.021 (0.027) |
| 13.8 % | 13.9× | 12.6× | 0.88 | -0.303 (0.0039) | +0.008 (0.0098) |
| 18.2 % | 11.1× | 9.4× | 0.87 | -0.251 (0.002) | +0.001 (1) |
| 22.3 % | 9.2× | 8.6× | 0.95 | -0.255 (0.002) | -0.001 (0.23) |
| 25.9 % | 8.3× | 7.5× | 0.92 | -0.194 (0.0039) | -0.000 (0.92) |

Razón = pérdida media de ventanas cuyo paso predicho es anómalo / de las normales (mediana entre semillas). Final/máx < 1 indica que el modelo empezó a aprender (memorizar) las anomalías.

### Reales — memorización del LSTM y dónde aporta la trayectoria (L=4, P=1)

| Dataset | % anómalo | Razón máx. | Razón final | Final / máx. | Δ AUPRC TracIn − pérdida final (ventana) | Δ AUPRC proy. obj. − PD |
|---|---|---|---|---|---|---|
| Credit Card | 0.17 % | 24.8× | 13.8× | 0.56 | +0.046 (5/5) | +0.103 (5/5) |
| CICIDS | 1.28 % | 0.7× | 0.6× | 0.99 | -0.002 (0/5) | -0.004 (0/5) |
| GECCO | 1.25 % | 251.0× | 70.6× | 0.28 | +0.146 (5/5) | +0.181 (5/5) |
| SWAN-SF | 23.80 % | 2.3× | 2.1× | 0.93 | +0.053 (5/5) | +0.070 (5/5) |

Razón = pérdida media de ventanas cuyo paso predicho es anómalo / de las normales (semilla 0). Final/máx < 1: el modelo aprendió (memorizó) parte de las anomalías. Razón < 1: las anomalías se predicen mejor que lo normal. Entre paréntesis, semillas en que gana el primer score.

La ganancia de TracIn sobre el error final está ordenada por la memorización:
GECCO (final/máx 0.28) gana
+0.181; Credit Card
(0.56) +0.103;
SWAN-SF (0.93) +0.070;
CICIDS (0.99) −0.004.
En los sintéticos de 400 pasos la pérdida de las anomalías sigue varias veces
por encima de la normal al final del entrenamiento (figura de arriba): el error
final ya las ve y la trayectoria no añade nada (primer experimento). En la
contaminación controlada del primer experimento sólo ayudaba al 1 %; en la
réplica ayuda en todos los niveles (AUPRC +0.110
con 1.1 %, +0.057 con 26 %; en AUROC, que no
depende de la proporción de anomalías, +0.063
y +0.051: la ventaja casi no cambia). Son cuatro datasets reales:
es una relación ordenada, no una correlación estimada.

## 6. Propuesta 1: umbral sin etiquetas

**Réplica.**

### Sintéticos: mejora 1 (umbral sin etiquetas) sobre el TracIn reproducido

| Score | Dinámico (usa etiquetas) macro | Dinámico binario | Estático pct 85 macro | iqr macro | iqr binario | mad macro | mad binario | log_iqr macro | log_iqr binario | log_mad macro | log_mad binario | pot macro | pot binario |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.703 | 0.478 | 0.718 | 0.712 | 0.488 | 0.724 | 0.530 | 0.664 | 0.391 | 0.704 | 0.477 | 0.503 | 0.066 |
| TracIn proy. objetivo | 0.832 | 0.679 | 0.785 | 0.838 | 0.709 | 0.838 | 0.710 | 0.828 | 0.689 | 0.831 | 0.695 | 0.502 | 0.063 |

Todas las reglas se miden en la segunda mitad de la serie (la misma que el dinámico). Las reglas sin etiquetas usan k = 1.5 (IQR), k = 3 (MAD) y q = 1e-3 (POT), fijados a priori; mejor (L, P) por dataset.

Con la configuración reproducida se confirma: IQR y MAD, sin etiquetas, quedan
al nivel del umbral dinámico del artículo (que sí las usa) — con TracIn del
artículo 0.712 y 0.724
vs 0.703; con la proyección al objetivo
0.838 vs 0.832
(F1 macro). POT sigue sin funcionar en series de 400 pasos.

**Objetivo de la mejora: que el método sea completamente no supervisado.** El
artículo usa etiquetas en dos lugares: para calibrar el umbral dinámico y para
elegir la mejor (L, P) de cada dataset. Para medir el método sin ninguna
etiqueta fijamos L4P1 a priori y usamos sólo reglas sin etiquetas:

### Sintéticos, método completamente sin etiquetas: L4P1 fijo + umbral sin etiquetas (F1 macro)

| Score | Dinámico (usa etiquetas) | Estático pct 85 | iqr | mad | log_iqr | log_mad | pot |
|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.697 | 0.710 | 0.708 | 0.721 | 0.655 | 0.698 | 0.493 |
| TracIn proy. objetivo | 0.817 | 0.763 | 0.815 | 0.815 | 0.804 | 0.801 | 0.494 |
| Error LSTM (PD) | 0.787 | 0.730 | 0.788 | 0.788 | 0.743 | 0.722 | 0.490 |

L = 4, P = 1 fijados a priori para todos los datasets (no se elige la mejor (L, P) con etiquetas). F1 macro en la segunda mitad de la serie, 20 semillas × 15 datasets.

* **Sintéticos: objetivo cumplido.** Sin ninguna etiqueta, IQR y MAD igualan
  al dinámico que sí las usa: TracIn del artículo
  0.708 (IQR) y
  0.721 (MAD) vs
  0.697; con la proyección al objetivo
  0.815 vs
  0.817. Con la proyección además
  superan al umbral estático del artículo
  (0.763), que supone conocida la
  proporción de anomalías.

### Reales, mejora 1: umbral sin etiquetas (L4P1 fijo, F1 binario)

| Score | Dataset | Dinámico (usa etiquetas) | Estático pct 87 | IQR | MAD | log-IQR | log-MAD | EVT-POT |
|---|---|---|---|---|---|---|---|---|
| TracIn (artículo) | creditcard | 0.225 | 0.020 | 0.018 | 0.017 | 0.026 | 0.025 | 0.237 |
| TracIn (artículo) | cicids | 0.003 | 0.028 | 0.002 | 0.003 | 0.000 | 0.000 | 0.000 |
| TracIn (artículo) | gecco | 0.325 | 0.132 | 0.184 | 0.138 | 0.325 | 0.332 | 0.105 |
| TracIn (artículo) | swan_sf | 0.378 | 0.286 | 0.272 | 0.291 | 0.153 | 0.138 | 0.009 |
| TracIn proy. objetivo | creditcard | 0.282 | 0.024 | 0.027 | 0.025 | 0.059 | 0.069 | 0.314 |
| TracIn proy. objetivo | cicids | 0.003 | 0.013 | 0.002 | 0.002 | 0.003 | 0.002 | 0.000 |
| TracIn proy. objetivo | gecco | 0.318 | 0.118 | 0.075 | 0.067 | 0.295 | 0.308 | 0.103 |
| TracIn proy. objetivo | swan_sf | 0.383 | 0.326 | 0.293 | 0.332 | 0.103 | 0.084 | 0.008 |

Configuración L4P1 fijada a priori; todas las reglas medidas en la segunda mitad de la serie, como el dinámico; 3 semillas.

* **Reales: objetivo cumplido sólo en parte.** Hay una regla sin etiquetas que
  iguala o supera al dinámico en Credit Card (EVT-POT,
  0.314 vs
  0.282) y en GECCO (log-MAD,
  0.332 vs
  0.325); en SWAN-SF la mejor (MAD,
  0.332) queda por debajo del
  dinámico (0.383), y en CICIDS
  ninguna regla funciona, tampoco la del artículo.
* Pero **no hay una regla única**: la que sirve depende de la tasa de
  anomalías (POT con anomalías muy raras, la escala logarítmica en GECCO, la
  lineal con 24 % en SWAN-SF), y cualquier regla fija falla en algún dataset
  (MAD en Credit Card: 0.025).
  Elegir la regla con un criterio sin etiquetas queda como siguiente paso; no
  lo diseñamos aquí para no ajustarlo a estos cuatro datasets viendo sus
  resultados.

**Primer experimento.**

### Sintéticos — F1 según la regla de umbral (configuración por defecto)

| Score | Estático τ=0.85·máx (art.) | Estático percentil 85 (art., literal) | Dinámico (art., usa etiquetas) | IQR k=1.5 | MAD k=3 | log-IQR k=1.5 | log-MAD k=3 | EVT-POT q=1e-3 | F1 oráculo |
|---|---|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.138 ± 0.025 | 0.394 ± 0.036 | 0.426 ± 0.063 | 0.395 ± 0.061 | 0.398 ± 0.058 | 0.208 ± 0.047 | 0.175 ± 0.044 | 0.041 ± 0.010 | 0.518 |
| TracIn proy. objetivo | 0.100 ± 0.022 | 0.462 ± 0.035 | 0.566 ± 0.055 | 0.580 ± 0.045 | 0.585 ± 0.044 | 0.401 ± 0.046 | 0.348 ± 0.046 | 0.045 ± 0.009 | 0.641 |
| LSTM error (PD) | 0.087 ± 0.017 | 0.514 ± 0.025 | 0.610 ± 0.040 | 0.609 ± 0.030 | 0.606 ± 0.028 | 0.547 ± 0.033 | 0.501 ± 0.030 | 0.040 ± 0.008 | 0.682 |
| LOF | 0.182 ± 0.025 | 0.498 ± 0.024 | 0.632 ± 0.045 | 0.656 ± 0.032 | 0.660 ± 0.033 | 0.658 ± 0.033 | 0.661 ± 0.032 | 0.054 ± 0.007 | 0.702 |
| Isolation Forest | 0.417 ± 0.041 | 0.456 ± 0.026 | 0.506 ± 0.039 | 0.464 ± 0.043 | 0.468 ± 0.045 | 0.452 ± 0.047 | 0.454 ± 0.050 | 0.051 ± 0.015 | 0.591 |

Los umbrales sin etiquetas usan hiperparámetros fijados a priori (k = 1.5, k = 3, q = 1e-3). Proporción marcada / proporción real (TracIn proy. objetivo): Estático 0.05, IQR 0.60, MAD 0.64, log-IQR 0.33, log-MAD 0.27, EVT-POT 0.02

### Reales — F1 según la regla de umbral (configuración por defecto)

| Score | Dataset | Estático τ=0.87·máx (art.) | Estático percentil 87 (art., literal) | Dinámico (art., usa etiquetas) | IQR k=1.5 | MAD k=3 | log-IQR k=1.5 | log-MAD k=3 | EVT-POT q=1e-3 | F1 oráculo |
|---|---|---|---|---|---|---|---|---|---|---|
| TracIn (artículo) | Credit Card | 0.006 | 0.020 | 0.102 | 0.020 | 0.019 | 0.029 | 0.029 | 0.115 | 0.133 |
| TracIn (artículo) | CICIDS | 0.000 | 0.029 | 0.038 | 0.028 | 0.032 | 0.002 | 0.001 | 0.001 | 0.038 |
| TracIn (artículo) | GECCO | 0.002 | 0.123 | 0.291 | 0.252 | 0.174 | 0.344 | 0.350 | 0.128 | 0.381 |
| TracIn (artículo) | SWAN-SF | 0.000 | 0.274 | 0.323 | 0.222 | 0.227 | 0.147 | 0.133 | 0.003 | 0.384 |
| TracIn proy. objetivo | Credit Card | 0.000 | 0.024 | 0.189 | 0.033 | 0.031 | 0.077 | 0.092 | 0.259 | 0.275 |
| TracIn proy. objetivo | CICIDS | 0.000 | 0.013 | 0.033 | 0.013 | 0.021 | 0.000 | 0.000 | 0.000 | 0.053 |
| TracIn proy. objetivo | GECCO | 0.001 | 0.111 | 0.320 | 0.101 | 0.092 | 0.377 | 0.376 | 0.112 | 0.380 |
| TracIn proy. objetivo | SWAN-SF | 0.000 | 0.323 | 0.300 | 0.259 | 0.273 | 0.090 | 0.073 | 0.008 | 0.404 |
| Isolation Forest | Credit Card | 0.233 | 0.024 | 0.244 | 0.056 | 0.059 | 0.066 | 0.072 | 0.233 | 0.263 |
| Isolation Forest | CICIDS | 0.002 | 0.008 | 0.051 | 0.012 | 0.008 | 0.015 | 0.009 | 0.000 | 0.065 |
| Isolation Forest | GECCO | 0.051 | 0.149 | 0.339 | 0.301 | 0.275 | 0.349 | 0.333 | 0.207 | 0.387 |
| Isolation Forest | SWAN-SF | 0.116 | 0.458 | 0.498 | 0.268 | 0.294 | 0.225 | 0.240 | 0.003 | 0.589 |

* En sintéticos, **IQR y MAD** quedan prácticamente al nivel del umbral
  dinámico del artículo, que usa etiquetas. Es decir, se puede quitar la parte
  supervisada casi sin costo.
* La **escala logarítmica** y **EVT-POT** no funcionan en series de 400
  pasos (≈ 7 excesos para ajustar la cola).
* En reales la regla adecuada depende de la tasa de anomalías: POT funciona con
  anomalías muy raras (Credit Card, proy. objetivo: F1
  0.259 vs
  0.189 del dinámico), pero no
  cuando son el 24 % (SWAN-SF: 0.008);
  en GECCO las versiones logarítmicas son las mejores.
* El umbral estático del artículo (τ·máx) depende de un único valor extremo;
  con TracIn casi no marca nada en reales.

## 7. Propuesta 2: reponderación iterativa

### 7.1 Réplica

### Sintéticos: nuestras mejoras sobre el TracIn reproducido (protocolo del artículo)

| Variante | AUPRC | Δ AUPRC (p) | F1 macro | Δ F1 macro (p) | F1 binario | Δ F1 binario (p) |
|---|---|---|---|---|---|---|
| TracIn del artículo (reproducido) | 0.606 | — | 0.703 | — | 0.478 | — |
| Mejora 3 · proyección al objetivo | 0.832 | +0.226 (4.1e-40) | 0.832 | +0.129 (1.5e-33) | 0.679 | +0.201 (1.8e-32) |
| Mejora 2 · reponderación suave (2 rondas) | 0.614 | +0.008 (9.9e-13) | 0.700 | -0.002 (6.1e-01) | 0.473 | -0.004 (2.7e-01) |
| Mejora 2 · recorte (2 rondas) | 0.607 | +0.000 (1.7e-06) | 0.700 | -0.003 (7.1e-01) | 0.470 | -0.007 (9.2e-01) |
| Mejoras 2 + 3 · suave + proyección | 0.836 | +0.229 (1.4e-36) | 0.840 | +0.137 (5.9e-35) | 0.691 | +0.213 (1.7e-33) |
| Mejoras 2 + 3 · recorte + proyección | 0.822 | +0.216 (2.2e-31) | 0.839 | +0.136 (1.7e-34) | 0.691 | +0.214 (2.9e-34) |

F1 con el umbral dinámico del artículo. Δ y p (Wilcoxon pareado sobre 15 datasets × 20 semillas) respecto del TracIn reproducido, cada uno en su mejor (L, P) por dataset.

![Contaminación, réplica](figures/fig_replica_contaminacion.png)

Para ver si un método **se degrada** con la contaminación usamos AUROC: el
AUPRC de un detector al azar es la proporción de anomalías, así que el AUPRC
sube solo cuando hay más anomalías y una curva plana de AUPRC esconde una
degradación. Las diferencias entre métodos en un mismo nivel sí se pueden
leer en ambas métricas.

### Contaminación (configuración reproducida) — AUROC vs. proporción real de anomalías

| Método | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.954 | 0.931 | 0.892 | 0.858 | 0.832 | 0.811 | 0.789 |
| TracIn proy. objetivo | 0.945 | 0.941 | 0.919 | 0.902 | 0.897 | 0.879 | 0.873 |
| Error LSTM (PD) | 0.882 | 0.887 | 0.859 | 0.852 | 0.841 | 0.831 | 0.823 |
| Reponderación suave | 0.956 | 0.935 | 0.906 | 0.871 | 0.843 | 0.818 | 0.794 |
| Recorte | 0.954 | 0.935 | 0.911 | 0.881 | 0.856 | 0.835 | 0.813 |
| Suave + proy. | 0.947 | 0.946 | 0.934 | 0.913 | 0.906 | 0.887 | 0.882 |
| Recorte + proy. | 0.944 | 0.945 | 0.941 | 0.925 | 0.921 | 0.904 | 0.896 |
| IForest | 0.807 | 0.773 | 0.786 | 0.785 | 0.780 | 0.780 | 0.766 |
| OCSVM | 0.632 | 0.664 | 0.674 | 0.681 | 0.684 | 0.683 | 0.675 |
| LOF | 0.910 | 0.898 | 0.900 | 0.881 | 0.868 | 0.873 | 0.852 |
| Autoencoder | 0.671 | 0.644 | 0.660 | 0.643 | 0.660 | 0.654 | 0.639 |
| Matrix Profile | 0.854 | 0.845 | 0.805 | 0.773 | 0.741 | 0.721 | 0.711 |

Serie «01234» de 4000 pasos, L = 4, P = 1 fijos a priori, datos estandarizados, media de 10 semillas. Columnas: proporción real de instantes anómalos.

### Contaminación (configuración reproducida) — diferencias pareadas de AUROC

| Comparación (AUROC) | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| Suave − TracIn | +0.002 (9/10) | +0.004 (10/10) | +0.013 (10/10) | +0.013 (10/10) | +0.011 (10/10) | +0.006 (10/10) | +0.005 (10/10) |
| Recorte − TracIn | +0.000 (7/10) | +0.004 (9/10) | +0.019 (10/10) | +0.023 (10/10) | +0.024 (10/10) | +0.023 (10/10) | +0.024 (10/10) |
| Recorte + proy. − Proy. objetivo | -0.001 (8/10) | +0.004 (9/10) | +0.022 (10/10) | +0.024 (10/10) | +0.024 (10/10) | +0.024 (10/10) | +0.023 (10/10) |
| Proy. objetivo − TracIn | -0.008 (2/10) | +0.010 (8/10) | +0.027 (10/10) | +0.043 (10/10) | +0.065 (10/10) | +0.068 (10/10) | +0.084 (10/10) |
| Proy. objetivo − PD | +0.063 (10/10) | +0.054 (10/10) | +0.060 (10/10) | +0.050 (10/10) | +0.056 (10/10) | +0.049 (10/10) | +0.051 (10/10) |
| TracIn − PD | +0.072 (10/10) | +0.044 (9/10) | +0.033 (10/10) | +0.006 (7/10) | -0.009 (3/10) | -0.019 (2/10) | -0.034 (0/10) |

Media de la diferencia por semilla; entre paréntesis, semillas en que el primer método gana.

### Contaminación (configuración reproducida) — AUPRC vs. proporción real de anomalías

| Método | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.466 | 0.436 | 0.433 | 0.452 | 0.479 | 0.500 | 0.517 |
| TracIn proy. objetivo | 0.713 | 0.752 | 0.752 | 0.757 | 0.770 | 0.761 | 0.778 |
| Error LSTM (PD) | 0.602 | 0.647 | 0.664 | 0.687 | 0.702 | 0.704 | 0.721 |
| Reponderación suave | 0.503 | 0.504 | 0.511 | 0.519 | 0.514 | 0.517 | 0.526 |
| Recorte | 0.466 | 0.506 | 0.536 | 0.554 | 0.567 | 0.581 | 0.588 |
| Suave + proy. | 0.702 | 0.777 | 0.794 | 0.786 | 0.791 | 0.772 | 0.788 |
| Recorte + proy. | 0.648 | 0.751 | 0.810 | 0.806 | 0.819 | 0.808 | 0.823 |
| IForest | 0.128 | 0.250 | 0.427 | 0.499 | 0.547 | 0.588 | 0.615 |
| OCSVM | 0.296 | 0.271 | 0.389 | 0.429 | 0.463 | 0.496 | 0.531 |
| LOF | 0.738 | 0.745 | 0.787 | 0.752 | 0.757 | 0.788 | 0.770 |
| Autoencoder | 0.160 | 0.152 | 0.260 | 0.289 | 0.353 | 0.385 | 0.411 |
| Matrix Profile | 0.329 | 0.413 | 0.412 | 0.426 | 0.435 | 0.466 | 0.494 |

Serie «01234» de 4000 pasos, L = 4, P = 1 fijos a priori, datos estandarizados, media de 10 semillas. Columnas: proporción real de instantes anómalos.

### Contaminación (configuración reproducida) — diferencias pareadas de AUPRC

| Comparación (AUPRC) | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| Suave − TracIn | +0.036 (9/10) | +0.069 (10/10) | +0.078 (10/10) | +0.067 (10/10) | +0.036 (10/10) | +0.017 (10/10) | +0.009 (9/10) |
| Recorte − TracIn | -0.000 (8/10) | +0.070 (9/10) | +0.102 (10/10) | +0.102 (10/10) | +0.088 (10/10) | +0.081 (10/10) | +0.071 (10/10) |
| Recorte + proy. − Proy. objetivo | -0.065 (6/10) | -0.001 (8/10) | +0.058 (10/10) | +0.050 (10/10) | +0.050 (10/10) | +0.048 (10/10) | +0.045 (10/10) |
| Proy. objetivo − TracIn | +0.247 (10/10) | +0.316 (10/10) | +0.318 (10/10) | +0.305 (10/10) | +0.291 (10/10) | +0.261 (10/10) | +0.261 (10/10) |
| Proy. objetivo − PD | +0.110 (9/10) | +0.105 (10/10) | +0.088 (10/10) | +0.069 (10/10) | +0.068 (10/10) | +0.057 (10/10) | +0.057 (10/10) |
| TracIn − PD | -0.136 (0/10) | -0.211 (0/10) | -0.231 (0/10) | -0.235 (0/10) | -0.223 (0/10) | -0.204 (0/10) | -0.204 (0/10) |

Media de la diferencia por semilla; entre paréntesis, semillas en que el primer método gana.

### Contaminación (configuración reproducida) — F1 dinámico binario vs. proporción real de anomalías

| Método | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.372 | 0.488 | 0.494 | 0.523 | 0.531 | 0.575 | 0.587 |
| TracIn proy. objetivo | 0.704 | 0.734 | 0.725 | 0.711 | 0.714 | 0.711 | 0.720 |
| Error LSTM (PD) | 0.617 | 0.657 | 0.652 | 0.643 | 0.654 | 0.657 | 0.660 |
| Reponderación suave | 0.406 | 0.502 | 0.523 | 0.530 | 0.547 | 0.575 | 0.590 |
| Recorte | 0.407 | 0.488 | 0.531 | 0.539 | 0.553 | 0.583 | 0.593 |
| Suave + proy. | 0.716 | 0.755 | 0.751 | 0.741 | 0.735 | 0.724 | 0.730 |
| Recorte + proy. | 0.696 | 0.771 | 0.769 | 0.765 | 0.758 | 0.749 | 0.749 |
| IForest | 0.207 | 0.277 | 0.369 | 0.425 | 0.466 | 0.516 | 0.537 |
| OCSVM | 0.382 | 0.347 | 0.375 | 0.381 | 0.437 | 0.462 | 0.487 |
| LOF | 0.720 | 0.775 | 0.733 | 0.686 | 0.674 | 0.709 | 0.676 |
| Autoencoder | 0.295 | 0.188 | 0.256 | 0.276 | 0.359 | 0.415 | 0.438 |
| Matrix Profile | 0.425 | 0.446 | 0.430 | 0.419 | 0.444 | 0.478 | 0.501 |

Serie «01234» de 4000 pasos, L = 4, P = 1 fijos a priori, datos estandarizados, media de 10 semillas. Columnas: proporción real de instantes anómalos.

### Reales: nuestras mejoras sobre el TracIn de la réplica

| Dataset | TracIn AUPRC | TracIn F1 dinámico | TracIn F1 dinámico (aleatoria) | proy. objetivo AUPRC | proy. objetivo F1 dinámico | proy. objetivo F1 dinámico (aleatoria) | suave AUPRC | suave F1 dinámico | suave F1 dinámico (aleatoria) | recorte AUPRC | recorte F1 dinámico | recorte F1 dinámico (aleatoria) | recorte + proy. AUPRC | recorte + proy. F1 dinámico | recorte + proy. F1 dinámico (aleatoria) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| creditcard | 0.116 | 0.225 | 0.203 | 0.214 | 0.282 | 0.310 | 0.085 | 0.195 | 0.171 | 0.080 | 0.183 | 0.177 | 0.201 | 0.249 | 0.309 |
| cicids | 0.017 | 0.003 | 0.035 | 0.015 | 0.003 | 0.041 | 0.017 | 0.003 | 0.037 | 0.017 | 0.003 | 0.036 | 0.016 | 0.003 | 0.044 |
| gecco | 0.461 | 0.355 | 0.510 | 0.438 | 0.357 | 0.443 | 0.522 | 0.446 | 0.570 | 0.511 | 0.381 | 0.540 | 0.499 | 0.364 | 0.565 |
| swan_sf | 0.339 | 0.378 | 0.375 | 0.376 | 0.383 | 0.405 | 0.340 | 0.380 | 0.371 | 0.347 | 0.374 | 0.364 | 0.372 | 0.377 | 0.394 |

Mejor (L, P) por dataset; F1 binario con el umbral dinámico del artículo.

* **Sintéticos de 400 pasos:** la reponderación no cambia el F1
  (−0.002 suave,
  −0.003 recorte; p > 0.2) y apenas
  el AUPRC.
* **El problema que motiva la mejora es real.** TracIn del artículo es el
  mejor detector con 1 % de anomalías (AUROC
  0.954, contra 0.910 de LOF y
  0.882 del error del LSTM) y el que más se degrada:
  con 26 % cae a 0.789, por debajo del error del
  LSTM (0.823) y de LOF (0.852),
  que pierden mucho menos. Es lo que el artículo anticipa (el modelo aprende las
  anomalías y su pérdida deja de distinguirlas), aunque el artículo no lo mide:
  sólo lo propone para explicar SWAN-SF.
* **La reponderación frena la caída, pero poco.** El recorte sube el AUROC de
  TracIn +0.019 a
  +0.024 desde 10 % de anomalías, en las 10
  semillas (p = 2.0e-03); en AUPRC son
  +0.070 a +0.102. De
  una caída de ≈ 0.17 de AUROC recupera ≈ 0.02. Con 1 % no hace nada.
* **Con la proyección al objetivo (mejoras 2 + 3) la caída casi desaparece**:
  0.944 con 1 % y 0.896
  con 26 %; es el mejor detector de la serie desde ≈ 10 % (con 26 %, LOF
  0.852). Sobre la proyección sola, el recorte suma
  +0.023 de AUROC (10/10 semillas
  desde 10 %).
* **¿Se puede frenar la degradación restante?** Probamos cuatro variantes,
  todas fijadas antes de ver resultados:

### Contaminación: ¿se puede frenar la degradación? (AUROC)

| Método | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % | Caída 1 % → 26 % |
|---|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.954 | 0.931 | 0.892 | 0.858 | 0.832 | 0.811 | 0.789 | -0.165 |
| Proy. objetivo | 0.945 | 0.941 | 0.919 | 0.902 | 0.897 | 0.879 | 0.873 | -0.072 |
| Recorte z > 2 (2 rondas) | 0.954 | 0.935 | 0.911 | 0.881 | 0.856 | 0.835 | 0.813 | -0.141 |
| Recorte adaptativo IQR (3 rondas) | 0.946 | 0.935 | 0.913 | 0.886 | 0.862 | 0.841 | 0.821 | -0.125 |
| Recorte z > 2 + proy. (2 rondas) | 0.944 | 0.945 | 0.941 | 0.925 | 0.921 | 0.904 | 0.896 | -0.048 |
| Recorte adaptativo IQR + proy. (3 rondas) | 0.937 | 0.944 | 0.943 | 0.931 | 0.930 | 0.912 | 0.905 | -0.032 |
| Épocas tempranas + proy. | 0.946 | 0.943 | 0.933 | 0.914 | 0.910 | 0.892 | 0.886 | -0.060 |
| LSTM de 16 + proy. | 0.942 | 0.933 | 0.918 | 0.901 | 0.897 | 0.885 | 0.878 | -0.064 |
| Ensamble rangos (recorte z > 2 + proy., LOF) | 0.934 | 0.930 | 0.932 | 0.920 | 0.917 | 0.912 | 0.903 | -0.030 |
| LOF | 0.910 | 0.898 | 0.900 | 0.881 | 0.868 | 0.873 | 0.852 | -0.058 |
| Error LSTM (PD) | 0.882 | 0.887 | 0.859 | 0.852 | 0.841 | 0.831 | 0.823 | -0.059 |

Misma serie, semillas y L4P1 que la tabla anterior; 10 semillas. Variantes fijadas a priori: recorte adaptativo = descartar las ventanas con self-influence sobre el umbral IQR de la mejora 1 (3 rondas); épocas tempranas = TracIn con el primer cuarto de las épocas; ensamble = suma de rangos, pesos iguales.

### Contaminación: cuánto recorta cada regla (ventanas recortadas / ventanas anómalas recortadas)

| Regla | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| Recorte z > 2 (1.ª ronda) | 2 % / 35 % | 5 % / 31 % | 7 % / 27 % | 9 % / 25 % | 11 % / 25 % | 12 % / 24 % | 13 % / 24 % |
| Recorte adaptativo IQR (3.ª ronda) | 6 % / 44 % | 8 % / 41 % | 10 % / 37 % | 12 % / 34 % | 14 % / 32 % | 16 % / 31 % | 17 % / 30 % |

Ventana anómala: algún instante de su entrada o de su objetivo es anómalo. Las etiquetas sólo se usan para este diagnóstico.

  * **Recorte adaptativo (mejoras 1 + 2):** recortar las ventanas que superan
    el umbral IQR de la mejora 1, en vez de z > 2, sube el AUROC con 26 % de
    0.896 a
    0.905
    (+0.009,
    10/10 semillas desde 14 %); la caída de 1 % a 26 % pasa de −0.048 a
    −0.032. Es la mejor variante de TracIn y la que menos se degrada; con 1 %
    pierde un poco
    (−0.007).
  * **Épocas tempranas** ayuda sobre la proyección sola
    (+0.013, 10/10 desde
    10 %), pero menos que el recorte. **Un LSTM más chico** no ayuda.
  * **Ensamble con LOF**: es el que menos cae, pero porque empieza más abajo:
    con pocas anomalías es peor que recorte + proyección.
  * **Por qué no desaparece:** aun el recorte adaptativo sólo saca del
    entrenamiento ≈ 30 % de las ventanas anómalas con 26 % de contaminación.
    Cuando las anomalías son una cuarta parte de los datos, el modelo las
    aprende desde el principio y su self-influence se parece a la de las
    normales: no hay forma de reconocerlas para quitarlas. Todos los detectores
    se degradan (LOF 0.910 → 0.852, error
    del LSTM 0.882 → 0.823); el
    artículo mismo dice que con 24 % (SWAN-SF) las anomalías «casi quedan fuera
    de la definición de anomalía». Con el recorte adaptativo nuestra variante
    se degrada menos que LOF y que el error del LSTM.
  * Sólo se probó en esta serie sintética; falta comprobarlo en reales.
* **Reales (1 ronda):** en GECCO la suave sube TracIn de
  0.461 a 0.522 de AUPRC y
  de 0.355 a 0.446 de F1
  dinámico; en Credit Card empeora (0.085 vs
  0.116); en CICIDS y SWAN-SF no cambia
  (0.340 vs 0.339). Son 3 semillas.
* **Corrección.** En versiones anteriores de este informe dijimos que el error
  del LSTM *mejoraba* con la contaminación y que TracIn no se degradaba en la
  réplica: era un efecto del AUPRC, que sube con la proporción de anomalías.
  En AUROC ambos se degradan, y TracIn mucho más
  (0.882 → 0.823 el error
  del LSTM; 0.954 → 0.789
  TracIn).

### 7.2 Primer experimento

### Sintéticos — reponderación iterativa (L=4, P=1)

| Score | Variante | AUPRC | F1 dinámico | F1 log-MAD | Δ AUPRC vs original (p) |
|---|---|---|---|---|---|
| TracIn (artículo) | Original (ronda 0) | 0.434 ± 0.055 | 0.426 ± 0.063 | 0.175 ± 0.044 | — |
| TracIn (artículo) | soft ronda 1 | 0.443 ± 0.054 | 0.428 ± 0.059 | 0.204 ± 0.049 | +0.009 (2.7e-20) |
| TracIn (artículo) | soft ronda 2 | 0.445 ± 0.055 | 0.426 ± 0.060 | 0.210 ± 0.051 | +0.011 (1.4e-19) |
| TracIn (artículo) | soft ronda 3 | 0.445 ± 0.054 | 0.426 ± 0.060 | 0.212 ± 0.052 | +0.010 (1.3e-18) |
| TracIn (artículo) | trim ronda 1 | 0.445 ± 0.053 | 0.427 ± 0.059 | 0.215 ± 0.051 | +0.010 (5.5e-15) |
| TracIn (artículo) | trim ronda 2 | 0.444 ± 0.053 | 0.428 ± 0.059 | 0.224 ± 0.052 | +0.010 (6.3e-13) |
| TracIn (artículo) | trim ronda 3 | 0.443 ± 0.052 | 0.428 ± 0.059 | 0.225 ± 0.053 | +0.008 (8.4e-12) |
| TracIn proy. objetivo | Original (ronda 0) | 0.626 ± 0.039 | 0.566 ± 0.055 | 0.348 ± 0.046 | — |
| TracIn proy. objetivo | soft ronda 1 | 0.635 ± 0.041 | 0.577 ± 0.050 | 0.386 ± 0.046 | +0.010 (1.6e-24) |
| TracIn proy. objetivo | soft ronda 2 | 0.636 ± 0.041 | 0.578 ± 0.050 | 0.392 ± 0.046 | +0.011 (1.3e-24) |
| TracIn proy. objetivo | soft ronda 3 | 0.636 ± 0.041 | 0.577 ± 0.049 | 0.395 ± 0.046 | +0.011 (2.4e-23) |
| TracIn proy. objetivo | trim ronda 1 | 0.633 ± 0.039 | 0.578 ± 0.052 | 0.400 ± 0.051 | +0.007 (5.5e-14) |
| TracIn proy. objetivo | trim ronda 2 | 0.631 ± 0.039 | 0.581 ± 0.053 | 0.408 ± 0.050 | +0.005 (3.8e-11) |
| TracIn proy. objetivo | trim ronda 3 | 0.629 ± 0.039 | 0.580 ± 0.054 | 0.409 ± 0.052 | +0.003 (2.3e-09) |

soft: w = max(0.05, exp(−(z−2))) para z > 2; trim: w = 0 para z > 2 (z robusto del log de la self-influence). p = Wilcoxon pareado sobre (dataset, semilla).

![Contaminación](figures/fig_contaminacion.png)

### Contaminación controlada — AUPRC vs. proporción real de anomalías (4000 pasos, L=4, P=1)

| Método | 1.1 % | 4.9 % | 9.5 % | 13.8 % | 18.2 % | 22.3 % | 25.9 % |
|---|---|---|---|---|---|---|---|
| TracIn (artículo) | 0.488 ± 0.068 | 0.373 ± 0.103 | 0.335 ± 0.083 | 0.273 ± 0.099 | 0.338 ± 0.086 | 0.357 ± 0.088 | 0.411 ± 0.088 |
| TracIn proy. objetivo | 0.678 ± 0.062 | 0.578 ± 0.183 | 0.526 ± 0.156 | 0.357 ± 0.176 | 0.437 ± 0.161 | 0.434 ± 0.152 | 0.504 ± 0.149 |
| TracIn + reponderación soft | 0.521 ± 0.051 | 0.402 ± 0.127 | 0.356 ± 0.107 | 0.282 ± 0.113 | 0.338 ± 0.090 | 0.356 ± 0.088 | 0.410 ± 0.089 |
| TracIn proy. objetivo + soft | 0.687 ± 0.061 | 0.570 ± 0.200 | 0.534 ± 0.167 | 0.363 ± 0.185 | 0.434 ± 0.164 | 0.431 ± 0.149 | 0.510 ± 0.156 |
| TracIn + recorte | 0.504 ± 0.096 | 0.377 ± 0.175 | 0.329 ± 0.128 | 0.273 ± 0.122 | 0.333 ± 0.100 | 0.350 ± 0.082 | 0.411 ± 0.094 |
| LSTM error (PD) | 0.557 ± 0.061 | 0.625 ± 0.032 | 0.644 ± 0.037 | 0.660 ± 0.047 | 0.688 ± 0.028 | 0.689 ± 0.020 | 0.699 ± 0.025 |
| Isolation Forest | 0.128 ± 0.084 | 0.250 ± 0.102 | 0.427 ± 0.116 | 0.499 ± 0.074 | 0.547 ± 0.081 | 0.588 ± 0.056 | 0.615 ± 0.050 |
| One-Class SVM | 0.394 ± 0.240 | 0.408 ± 0.198 | 0.533 ± 0.167 | 0.519 ± 0.095 | 0.579 ± 0.138 | 0.625 ± 0.112 | 0.686 ± 0.054 |
| LOF | 0.730 ± 0.077 | 0.761 ± 0.054 | 0.805 ± 0.037 | 0.790 ± 0.060 | 0.811 ± 0.038 | 0.830 ± 0.012 | 0.845 ± 0.017 |
| Autoencoder | 0.111 ± 0.094 | 0.127 ± 0.043 | 0.203 ± 0.096 | 0.223 ± 0.052 | 0.280 ± 0.094 | 0.289 ± 0.069 | 0.322 ± 0.066 |
| Matrix Profile | 0.329 ± 0.056 | 0.413 ± 0.068 | 0.412 ± 0.024 | 0.426 ± 0.019 | 0.435 ± 0.034 | 0.466 ± 0.029 | 0.494 ± 0.013 |

### Reales — reponderación iterativa (AUPRC, L=4, P=1, semillas de la fase B)

| Score | Variante | Credit Card | CICIDS | GECCO | SWAN-SF |
|---|---|---|---|---|---|
| TracIn (artículo) | Original | 0.065 ± 0.003 | 0.017 ± 0.000 | 0.282 ± 0.021 | 0.325 ± 0.006 |
| TracIn (artículo) | soft r1 | 0.075 ± 0.004 | 0.017 ± 0.000 | 0.366 ± 0.022 | 0.319 ± 0.006 |
| TracIn (artículo) | soft r2 | 0.070 ± 0.003 | 0.017 ± 0.000 | 0.381 ± 0.017 | 0.317 ± 0.005 |
| TracIn (artículo) | soft r3 | 0.069 ± 0.003 | 0.017 ± 0.000 | 0.388 ± 0.015 | 0.317 ± 0.005 |
| TracIn (artículo) | trim r1 | 0.069 ± 0.004 | 0.017 ± 0.000 | 0.494 ± 0.005 | 0.319 ± 0.002 |
| TracIn (artículo) | trim r2 | 0.068 ± 0.003 | 0.017 ± 0.000 | 0.501 ± 0.004 | 0.320 ± 0.002 |
| TracIn (artículo) | trim r3 | 0.066 ± 0.002 | 0.017 ± 0.000 | 0.502 ± 0.004 | 0.321 ± 0.005 |
| TracIn proy. objetivo | Original | 0.172 ± 0.002 | 0.018 ± 0.000 | 0.284 ± 0.023 | 0.383 ± 0.006 |
| TracIn proy. objetivo | soft r1 | 0.216 ± 0.002 | 0.018 ± 0.000 | 0.356 ± 0.018 | 0.375 ± 0.007 |
| TracIn proy. objetivo | soft r2 | 0.211 ± 0.001 | 0.018 ± 0.000 | 0.370 ± 0.015 | 0.371 ± 0.008 |
| TracIn proy. objetivo | soft r3 | 0.211 ± 0.001 | 0.018 ± 0.000 | 0.377 ± 0.013 | 0.371 ± 0.009 |
| TracIn proy. objetivo | trim r1 | 0.210 ± 0.003 | 0.019 ± 0.000 | 0.481 ± 0.004 | 0.374 ± 0.002 |
| TracIn proy. objetivo | trim r2 | 0.207 ± 0.006 | 0.019 ± 0.000 | 0.492 ± 0.004 | 0.376 ± 0.005 |
| TracIn proy. objetivo | trim r3 | 0.201 ± 0.005 | 0.019 ± 0.000 | 0.494 ± 0.003 | 0.378 ± 0.003 |

Δ AUPRC tras 3 rondas, media de 5 semillas (entre paréntesis, semillas en que mejora):

| Dataset | Suave − original | Recorte − original | Suave − original (proy. obj.) | Recorte − original (proy. obj.) |
|---|---|---|---|---|
| Credit Card | +0.004 (5/5) | +0.001 (3/5) | +0.039 (5/5) | +0.030 (5/5) |
| CICIDS | +0.000 (3/5) | +0.000 (3/5) | +0.000 (5/5) | +0.000 (4/5) |
| GECCO | +0.106 (5/5) | +0.220 (5/5) | +0.093 (5/5) | +0.210 (5/5) |
| SWAN-SF | −0.009 (0/5) | −0.004 (1/5) | −0.013 (0/5) | −0.006 (0/5) |

* **GECCO es el caso donde la reponderación funciona**: TracIn sube de
  0.282 a 0.388 (suave) y
  0.502 (recorte), en las 5 semillas. Es el dataset con más
  memorización (§5). Nuestra interpretación, que todavía no verificamos
  directamente: al quitarle peso a las ventanas de mayor self-influence, el
  modelo deja de aprender las anomalías y su pérdida queda alta. Con el
  umbral log-MAD, sin etiquetas, el F1 de TracIn + recorte es
  0.517, contra
  0.291 del TracIn original con el umbral
  dinámico (que usa etiquetas). Sigue lejos del 0.761 publicado.
* En Credit Card ayuda sobre todo a la proyección al objetivo
  (+0.039, suave); en SWAN-SF (24 % anómalo) empeora
  ligeramente en todas las semillas, y en CICIDS no cambia nada.
* En los 15 sintéticos la mejora es pequeña pero muy consistente (tabla de
  arriba). En la contaminación controlada: +0.034 con 1 %,
  ≈ 0 desde el 18 %.
* La reponderación ayuda sólo donde hay anomalías raras que el modelo llega a
  aprender. Con contaminación alta, las ventanas anómalas no se distinguen lo
  suficiente en la primera ronda como para bajarles el peso.
* Lo que aquí parecía una mejora del error del LSTM con la contaminación
  (AUPRC de 0.56 a 0.70) es el efecto de la
  proporción de anomalías sobre el AUPRC; véase la corrección en §7.1.

## 8. Limitaciones y desviaciones del artículo

* **Supuestos de la réplica.** Que el artículo estandariza los canales y
  reporta F1 macro en sintéticos lo inferimos de que es la única combinación que
  reproduce sus cifras (§2.1); no está escrito en el artículo. El patrón por
  dataset no coincide (correlación −0.32), así que puede haber otra
  diferencia que no identificamos. GECCO y CICIDS no se reproducen.
* **Hiperparámetros no especificados** (lr, batch, épocas máximas, tamaño del
  LSTM, criterio de early stopping): usamos valores estándar y medimos la
  sensibilidad (§2.1, §2.3); casi no mueven el resultado.
* **Réplica en reales:** 3 semillas (como el artículo) y 1 ronda de
  reponderación; los métodos de referencia son los del primer experimento
  (que en reales ya usaba datos estandarizados), evaluados con el protocolo de
  la réplica.
* **Reales: alcance reducido por costo.** El Transformer se corrió sólo con
  L4P1 y 3 semillas; la reponderación, sólo con L4P1 y 5 semillas. OCSVM se
  ajusta sobre una submuestra de 10 000 ventanas. El autoencoder en reales usa
  20 épocas y batch 512.
* **Umbral dinámico.** El artículo no detalla cómo se elige; usamos el
  F1-óptimo validado cruzado en dos mitades, que no ve las etiquetas del
  instante que evalúa.
* **Fuentes de datos.** Credit Card y CICIDS se descargaron de réplicas
  (OpenML y el servidor de UNB no respondían); se verificó que los tamaños y
  el % de anomalías coinciden exactamente con la Tabla 1.
* **Wilcoxon con 5 semillas** no alcanza p < 0.05 en dos colas; en reales
  reportamos cuántas semillas favorecen a cada método en vez de un p-valor.
* El diagnóstico de LOF vs k elige k viendo etiquetas: explica el fallo,
  no es un resultado comparable.
