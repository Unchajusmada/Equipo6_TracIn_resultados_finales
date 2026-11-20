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
   reproducen (IForest en GECCO: F1 {{v:real_if_gecco|3}} vs 0.391 Art.). TracIn
   se reproduce **en sintéticos** con dos supuestos que el artículo no declara:
   **canales estandarizados** y **F1 macro** (promedio del F1 de ambas clases).
   Con ellos, F1 estático {{w:syn/S_mac|3}} vs 0.662 Art. y dinámico
   {{w:syn/D_mac|3}} vs 0.725 Art.; con el F1 binario habitual serían
   {{w:syn/S_bin|3}} y {{w:syn/D_bin|3}}. El nivel promedio coincide; el patrón
   por dataset no. En reales, **Credit Card se reproduce**
   ({{w:real/creditcard/D_bin|3}} vs 0.236 Art.), **GECCO y CICIDS no**
   ({{w:real/gecco/D_bin|3}} vs 0.761; {{w:real/cicids/D_bin|3}} vs 0.135) y en
   SWAN-SF obtenemos más ({{w:real/swan_sf/D_bin|3}} vs 0.200). Descartamos
   hiperparámetros, gradiente completo vs. última capa, point-adjust, F1
   ponderado y fuga de la etiqueta.
3. **Comparación (observación 2).** Con la configuración reproducida y la
   **misma métrica para todos**, TracIn del artículo queda debajo de los
   detectores clásicos: F1 dinámico macro {{w:syn_cmp/TracIn (artículo)/D_mac|3}}
   vs {{w:syn_cmp/LOF/D_mac|3}} (LOF), {{w:syn_cmp/One-Class SVM/D_mac|3}} (OCSVM),
   {{w:syn_cmp/Isolation Forest/D_mac|3}} (IForest) y
   {{w:syn_cmp/Error LSTM (PD)/D_mac|3}} (error del mismo LSTM). Las referencias
   de su Tabla 2 vienen de Lai et al. (2021), con F1 binario: si TracIn se
   reportó con F1 macro, esa tabla compara métricas distintas. En reales
   tampoco es el mejor en AUPRC en ningún dataset; con la proyección al
   objetivo sí lo es en Credit Card ({{w:real_cmp/creditcard/TracIn proy. objetivo/0|3}} vs
   {{w:real_cmp/creditcard/IForest/0|3}} de IForest; §3.2).
4. **¿Aporta la trayectoria?** Sí, comparando con la misma proyección: con la
   configuración reproducida, TracIn proyectado a los pasos predichos supera al
   error final del mismo LSTM en sintéticos ({{w:traj/syn_L4P1_auprc/0|+3}} AUPRC
   con L4P1, p = {{w:traj/syn_L4P1_auprc/1|e}}), en **todos** los niveles de
   contaminación ({{w:cont/tests/Proy. objetivo − PD/6/0|+3}} a
   {{w:cont/tests/Proy. objetivo − PD/0/0|+3}}) y en GECCO
   ({{w:traj/real_gecco/0|+3}}) y Credit Card ({{w:traj/real_creditcard/0|+3}}). En el
   primer experimento (datos sin estandarizar) sólo aparecía con 1 % de
   contaminación y en reales: **la estandarización cambia esta conclusión**.
   La ganancia sigue ordenada por cuánto memoriza el modelo las anomalías (§5).
5. **Mejoras propuestas, sobre el TracIn reproducido.**
   *Proyección al objetivo* (mejora 3, salió de la ablación):
   {{w:syn_impr/Mejora 3 · proyección al objetivo/d_auprc|+3}} AUPRC y
   {{w:syn_impr/Mejora 3 · proyección al objetivo/d_mac|+3}} F1 macro
   (p = {{w:syn_impr/Mejora 3 · proyección al objetivo/p_auprc|e}}): lo lleva al
   nivel de los mejores detectores. *Umbral IQR sin etiquetas* (mejora 1):
   F1 macro {{w:syn_thr/TracIn proy. objetivo/iqr/0|3}} vs
   {{w:syn_thr/TracIn proy. objetivo/dyn/0|3}} del umbral dinámico, que usa
   etiquetas. *Reponderación* (mejora 2): ≈ 0 en los sintéticos de 400 pasos.
   Con contaminación controlada confirmamos el problema que la motiva: TracIn
   pasa de ser el mejor detector con 1 % de anomalías (AUROC
   {{w:cont/auroc/TracIn (artículo)/0|3}}) a quedar debajo del error del LSTM con
   26 % ({{w:cont/auroc/TracIn (artículo)/6|3}}). El recorte frena la caída poco
   ({{w:cont/tests_auroc/Recorte − TracIn/6/0|+3}} AUROC, 10/10 semillas); con la
   proyección al objetivo la caída casi desaparece y es el mejor detector desde
   ≈ 10 % (AUROC {{w:cont/auroc/Recorte + proy./6|3}} vs {{w:cont/auroc/LOF/6|3}} de
   LOF con 26 %). En GECCO la reponderación suave
   sube el AUPRC de {{w:real_impr/gecco/tracin/auprc|3}} a
   {{w:real_impr/gecco/rw_soft_r1/auprc|3}}; en SWAN-SF y CICIDS no cambia y en
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

Primer experimento: {{v:counts/forecasters|int}} forecasters entrenados (cada
uno con su traza de checkpoints) y {{v:counts/baselines|int}} ajustes de métodos
de referencia. Réplica: {{w:counts/forecasters|int}} forecasters y
{{w:counts/baselines|int}} ajustes más.

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
   (L4P1) pasa de {{w:screen/nonorm/auprc|3}} a {{w:screen/base/auprc|3}} y el F1
   dinámico macro de {{w:screen/nonorm/D_mac|3}} a {{w:screen/base/D_mac|3}}.
2. **F1 macro.** Las cifras de la Tabla 2 coinciden con el promedio del F1 de
   la clase normal y de la anómala (`f1_score(average='macro')`), no con el F1
   de la clase anómala. Ninguna otra lectura que probamos se acerca:

{{r:rep_hipotesis_metrica}}

{{r:rep_syn_table2}}

![Réplica de la Tabla 2](figures/fig_replica_tabla2.png)

Con estos dos supuestos el **nivel** de la Tabla 2 se reproduce: F1 estático
{{w:syn/S_mac|3}} vs 0.662 Art. y dinámico {{w:syn/D_mac|3}} vs 0.725 Art. El
**patrón por dataset no**: la correlación entre nuestros F1 dinámicos y los
suyos es {{w:syn/corr_D|+2}}. Ellos obtienen sus mejores cifras en las
anomalías puntuales (0 y 1: 0.807), donde nosotros obtenemos las peores. Lo
atribuimos a cómo se etiquetan y proyectan las anomalías puntuales (un único
instante marcado contra un score repartido en L + P posiciones), pero no
pudimos verificarlo sin su código.

Lo demás que el artículo no especifica casi no mueve el resultado:

{{r:rep_syn_screen}}

### 2.2 Reales: Credit Card sí, GECCO y CICIDS no

En reales el F1 de la Tabla 4 tiene que ser **binario**: en Credit Card y
CICIDS la clase normal es > 98 % de los datos; con cualquier umbral que marque
una minoría de instantes su F1 es ≈ 1 y el F1 macro no baja de ≈ 0.5, pero
reportan 0.236 y 0.135. La réplica en reales usa el
gradiente de todos los parámetros (Ec. 2 completa), que en GECCO subió el
AUPRC de {{w:real_hyp/gecco_screen/base/0|3}} a {{w:real_hyp/gecco_screen/full/0|3}}.

{{r:rep_real_table4}}

* **Credit Card se reproduce**: F1 dinámico {{w:real/creditcard/D_bin|3}} vs
  0.236 Art.
* **GECCO no**: {{w:real/gecco/D_bin|3}} vs 0.761 Art. (con partición
  aleatoria de los instantes, {{w:real/gecco/D_rand|3}}).
* **CICIDS no**: {{w:real/cicids/D_bin|3}} vs 0.135 Art. Los ataques están
  concentrados en la primera mitad de la serie, así que un umbral elegido en
  la primera mitad y medido en la segunda casi no tiene anomalías que
  detectar; con partición aleatoria sube a {{w:real/cicids/D_rand|3}}.
* **SWAN-SF**: obtenemos más que el artículo ({{w:real/swan_sf/D_bin|3}} vs
  0.200 Art.).
* El umbral estático: con el percentil 87 y F1 binario la precisión no puede
  superar la tasa de anomalías / 0.13 (≈ 0.096 en GECCO), pero reportan
  P = 0.539; en reales sus cifras sólo son posibles si τ es una fracción del
  máximo. Con esa lectura, nuestro TracIn casi no marca instantes (F1 ≤ 0.01 en
  los cuatro datasets): el umbral estático en reales tampoco se reproduce. En sintéticos, en cambio, la lectura como percentil (con F1 macro) es
  la que reproduce: con τ·máx el F1 estático macro sería
  {{w:syn/S_frac_mac|3}}, contra {{w:syn/S_mac|3}} con el percentil y 0.662 Art.

### 2.3 Hipótesis descartadas

| Hipótesis | Prueba | Resultado |
|---|---|---|
| Hiperparámetros de entrenamiento (lr, batch, épocas, tamaño, scheduler, validación) | 10 variantes de entrenamiento en sintéticos (tabla de §2.1), 6 en GECCO | F1 dinámico macro entre {{w:screen/val20/D_mac|3}} y {{w:screen/b8/D_mac|3}} en sintéticos; ninguna variante acerca GECCO a 0.761 |
| Gradiente de la última capa vs. de todos los parámetros | variante `full` | Igual en sintéticos ({{w:screen/full/D_mac|3}} vs {{w:screen/base/D_mac|3}}); mejora GECCO, por eso se usa en reales |
| Point-adjust | lectura `pa` (tabla de §2.1) | Sintéticos: {{w:hyp/pa/D|3}}; GECCO (primer experimento): IForest pasaría de 0.39 a 0.62, lejos del 0.391 publicado, que sí coincide sin point-adjust |
| F1 ponderado por soporte | lectura `wtd` | {{w:hyp/wtd/D|3}}: muy por encima de la Tabla 2 |
| Evaluar a nivel de ventana, o sólo en la parte de prueba | `h_level.py`, `h_eval.py` (primer experimento) | No cierra la diferencia |
| La etiqueta entra como canal (los canales de la Tabla 1 coinciden con atributos + etiqueta) | `real_screen.py --leak` | Inconsistente: GECCO sube a {{w:real_hyp/leak/gecco/1|3}}, pero CICIDS queda en {{w:real_hyp/leak/cicids/1|3}} y SWAN-SF en {{w:real_hyp/leak/swan_sf/1|3}} (Art. 0.200) |

{{r:rep_real_hipotesis}}

### 2.4 Qué implica para la comparación del artículo

Las cifras de los métodos de referencia de su Tabla 2 provienen de Lai et al.
(2021), que usan F1 **binario** (las reprodujimos así en §2.5). Si TracIn se
reportó con F1 macro, como indica la réplica, la Tabla 2 compara métricas
distintas: cuando las anomalías son minoría, el F1 de la clase normal es
cercano a 1 y el F1 macro de un detector queda muy por encima de su F1 binario
(TracIn: {{w:syn/D_mac|3}} macro vs {{w:syn/D_bin|3}} binario). Con la **misma** métrica para todos (§3.1),
los detectores clásicos quedan por encima de TracIn.

### 2.5 Primer intento (sin estandarizar, F1 binario)

{{t:syn_reproduction}}

{{t:real_reproduction}}

En reales, nuestras referencias quedan cerca de las publicadas salvo en SWAN-SF
(IForest {{v:real_rep_num/swan_sf/ours_best_base|3}} vs 0.583 Art.); USAD no lo
implementamos. TracIn queda muy por debajo excepto en SWAN-SF, donde nuestro
F1 ({{v:real_rep_num/swan_sf/ours_D|3}}) supera al publicado (0.200).

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

{{t:gecco_sensitivity}}

## 3. Comparación con otros detectores (observación 2)

### 3.1 Sintéticos

**Réplica (protocolo del artículo, misma métrica para todos).**

{{r:rep_syn_cmp}}

Con los datos estandarizados y la mejor configuración por dataset para cada
método, TracIn del artículo queda por debajo de LOF, OCSVM, IForest y del
error de su propio LSTM, en F1 macro, F1 binario y AUPRC. Con la proyección al
objetivo (mejora 3) pasa al primer lugar, empatado con el error del LSTM. El
autoencoder y Matrix Profile quedan al nivel de TracIn o por debajo.

**Primer experimento (configuración fija a priori, datos sin estandarizar,
F1 binario).**

{{t:syn_fair}}

![Comparación en sintéticos](figures/fig_sinteticos_comparacion.png)

![Mapa de calor por dataset](figures/fig_sinteticos_heatmap.png)

TracIn del artículo es fuerte en anomalías de patrón y débil en puntuales
(AUPRC {{v:syn/num_point/lstm|tracin/auprc/0|3}} en puntuales vs
{{v:syn/num_pattern/lstm|tracin/auprc/0|3}} en las de patrón): al repartir el score
sobre las L + P posiciones de cada ventana, un pico de un instante se vuelve
una meseta ancha.

### 3.2 Reales

**Réplica (TracIn con gradiente completo, mejor (L, P); referencias del primer
experimento con el mismo protocolo de evaluación).**

{{r:rep_real_cmp}}

* En la réplica, TracIn del artículo no es el mejor en AUPRC en ningún
  dataset. Con la proyección al objetivo sí lo es en **Credit Card**
  (AUPRC {{w:real_cmp/creditcard/TracIn proy. objetivo/0|3}}, F1
  {{w:real_cmp/creditcard/TracIn proy. objetivo/1|3}}, contra
  {{w:real_cmp/creditcard/IForest/0|3}} / {{w:real_cmp/creditcard/IForest/1|3}} de IForest);
  en GECCO gana el autoencoder ({{w:real_cmp/gecco/Autoencoder/0|3}}) y en SWAN-SF
  IForest ({{w:real_cmp/swan_sf/IForest/0|3}}).

**Primer experimento (configuración fija a priori).**

{{t:real_fair_auprc}}

{{t:real_fair_dyn_f1}}

![Reales](figures/fig_reales_auprc.png)

* TracIn del artículo no es el mejor en ningún dataset. IForest, OCSVM y el
  autoencoder lo superan en Credit Card, GECCO y SWAN-SF.
* En CICIDS ningún método llega a 2× el azar (véase §5: los ataques se
  predicen mejor que el tráfico normal).
* LOF (k = 20) y Matrix Profile se desploman en reales. Para LOF la causa es
  que las anomalías reales son colectivas (eventos contiguos, fraudes
  parecidos) y forman grupos más grandes que k:

{{t:real_lof_k}}

## 4. Ablación: ¿qué hace funcionar al score?

Mismo LSTM y mismos checkpoints; sólo cambia qué se mide (pérdida final,
pérdida integrada sobre el entrenamiento, self-influence) y cómo se proyecta a
instantes (las L + P posiciones de la ventana, como el artículo, o sólo los P
pasos predichos).

{{t:syn_ablation}}

{{t:real_ablation}}

![Ablación](figures/fig_ablacion.png)

* En sintéticos la trayectoria **no** aporta: TracIn − pérdida final =
  {{v:syn/ablation/tests/tracin_vs_lossfinal/0|+3}} (p = {{v:syn/ablation/tests/tracin_vs_lossfinal/1|e}}).
* La **proyección** domina: asignar el score sólo a los pasos predichos sube
  el AUPRC en {{v:syn/ablation/tests/tgt_vs_tracin/0|+3}}. Es nuestra mejora 3, no
  prevista en el plan de la primera presentación.
* En reales, en cambio, la trayectoria sí aporta en GECCO, Credit Card y
  SWAN-SF. La sección siguiente explica por qué.

**Réplica: con datos estandarizados la trayectoria sí aporta.** La comparación
limpia es TracIn proyectado al objetivo contra el error final del mismo LSTM
(misma proyección, mismo modelo):

{{r:rep_trayectoria}}

* Con la configuración reproducida, la trayectoria gana en sintéticos
  ({{w:traj/syn_L4P1_auprc/0|+3}} AUPRC con L4P1; {{w:traj/syn_best_auprc/0|+3}} con la
  mejor (L, P)), en todos los niveles de contaminación y en GECCO y Credit
  Card. En F1 dinámico con la mejor (L, P) la diferencia en sintéticos es nula.
* Esto **contradice** la conclusión del primer experimento para los
  sintéticos (allí la pérdida final era igual o mejor). La diferencia entre
  ambos es la estandarización de los canales; no verificamos el mecanismo.
* La **proyección** sigue siendo lo que más pesa: TracIn proyectado al
  objetivo − TracIn del artículo = {{w:syn_impr/Mejora 3 · proyección al objetivo/d_auprc|+3}}
  AUPRC en la réplica.

## 5. ¿Cuándo aporta la trayectoria? Memorización

Medimos, época por época, la razón entre la pérdida de las ventanas cuyo
paso predicho es anómalo y la de las normales. Si cae desde su máximo, el
modelo está aprendiendo (memorizando) las anomalías y el error final las deja
de ver; ése es el caso que motiva al artículo.

![Trayectorias](figures/fig_mecanismo_trayectorias.png)

{{t:cont_memorizacion}}

{{t:real_memorizacion}}

La ganancia de TracIn sobre el error final está ordenada por la memorización:
GECCO (final/máx {{v:real/memo/gecco/final_over_max|2}}) gana
{{v:real/memo/gecco/tracin_tgt-pd/0|+3}}; Credit Card
({{v:real/memo/creditcard/final_over_max|2}}) {{v:real/memo/creditcard/tracin_tgt-pd/0|+3}};
SWAN-SF ({{v:real/memo/swan_sf/final_over_max|2}}) {{v:real/memo/swan_sf/tracin_tgt-pd/0|+3}};
CICIDS ({{v:real/memo/cicids/final_over_max|2}}) {{v:real/memo/cicids/tracin_tgt-pd/0|+3}}.
En los sintéticos de 400 pasos la pérdida de las anomalías sigue varias veces
por encima de la normal al final del entrenamiento (figura de arriba): el error
final ya las ve y la trayectoria no añade nada (primer experimento). En la
contaminación controlada del primer experimento sólo ayudaba al 1 %; en la
réplica ayuda en todos los niveles (AUPRC {{w:cont/tests/Proy. objetivo − PD/0/0|+3}}
con 1.1 %, {{w:cont/tests/Proy. objetivo − PD/6/0|+3}} con 26 %; en AUROC, que no
depende de la proporción de anomalías, {{w:cont/tests_auroc/Proy. objetivo − PD/0/0|+3}}
y {{w:cont/tests_auroc/Proy. objetivo − PD/6/0|+3}}: la ventaja casi no cambia). Son cuatro datasets reales:
es una relación ordenada, no una correlación estimada.

## 6. Propuesta 1: umbral sin etiquetas

**Réplica.**

{{r:rep_syn_umbrales}}

Con la configuración reproducida se confirma: IQR y MAD, sin etiquetas, quedan
al nivel del umbral dinámico del artículo (que sí las usa) — con TracIn del
artículo {{w:syn_thr/TracIn (artículo)/iqr/0|3}} y {{w:syn_thr/TracIn (artículo)/mad/0|3}}
vs {{w:syn_thr/TracIn (artículo)/dyn/0|3}}; con la proyección al objetivo
{{w:syn_thr/TracIn proy. objetivo/iqr/0|3}} vs {{w:syn_thr/TracIn proy. objetivo/dyn/0|3}}
(F1 macro). POT sigue sin funcionar en series de 400 pasos.

**Objetivo de la mejora: que el método sea completamente no supervisado.** El
artículo usa etiquetas en dos lugares: para calibrar el umbral dinámico y para
elegir la mejor (L, P) de cada dataset. Para medir el método sin ninguna
etiqueta fijamos L4P1 a priori y usamos sólo reglas sin etiquetas:

{{r:rep_syn_umbrales_L4P1}}

* **Sintéticos: objetivo cumplido.** Sin ninguna etiqueta, IQR y MAD igualan
  al dinámico que sí las usa: TracIn del artículo
  {{w:syn_thr_L4P1/TracIn (artículo)/iqr|3}} (IQR) y
  {{w:syn_thr_L4P1/TracIn (artículo)/mad|3}} (MAD) vs
  {{w:syn_thr_L4P1/TracIn (artículo)/dyn|3}}; con la proyección al objetivo
  {{w:syn_thr_L4P1/TracIn proy. objetivo/iqr|3}} vs
  {{w:syn_thr_L4P1/TracIn proy. objetivo/dyn|3}}. Con la proyección además
  superan al umbral estático del artículo
  ({{w:syn_thr_L4P1/TracIn proy. objetivo/pct|3}}), que supone conocida la
  proporción de anomalías.

{{r:rep_real_umbrales}}

* **Reales: objetivo cumplido sólo en parte.** Hay una regla sin etiquetas que
  iguala o supera al dinámico en Credit Card (EVT-POT,
  {{w:real_thr/TracIn proy. objetivo/creditcard/U_pot|3}} vs
  {{w:real_thr/TracIn proy. objetivo/creditcard/D_bin|3}}) y en GECCO (log-MAD,
  {{w:real_thr/TracIn (artículo)/gecco/U_log_mad|3}} vs
  {{w:real_thr/TracIn (artículo)/gecco/D_bin|3}}); en SWAN-SF la mejor (MAD,
  {{w:real_thr/TracIn proy. objetivo/swan_sf/U_mad|3}}) queda por debajo del
  dinámico ({{w:real_thr/TracIn proy. objetivo/swan_sf/D_bin|3}}), y en CICIDS
  ninguna regla funciona, tampoco la del artículo.
* Pero **no hay una regla única**: la que sirve depende de la tasa de
  anomalías (POT con anomalías muy raras, la escala logarítmica en GECCO, la
  lineal con 24 % en SWAN-SF), y cualquier regla fija falla en algún dataset
  (MAD en Credit Card: {{w:real_thr/TracIn proy. objetivo/creditcard/U_mad|3}}).
  Elegir la regla con un criterio sin etiquetas queda como siguiente paso; no
  lo diseñamos aquí para no ajustarlo a estos cuatro datasets viendo sus
  resultados.

**Primer experimento.**

{{t:syn_thresholds}}

{{t:real_thresholds}}

* En sintéticos, **IQR y MAD** quedan prácticamente al nivel del umbral
  dinámico del artículo, que usa etiquetas. Es decir, se puede quitar la parte
  supervisada casi sin costo.
* La **escala logarítmica** y **EVT-POT** no funcionan en series de 400
  pasos (≈ 7 excesos para ajustar la cola).
* En reales la regla adecuada depende de la tasa de anomalías: POT funciona con
  anomalías muy raras (Credit Card, proy. objetivo: F1
  {{v:real/num/creditcard/lstm|tracin_tgt/pot_f1/0|3}} vs
  {{v:real/num/creditcard/lstm|tracin_tgt/dyn_f1/0|3}} del dinámico), pero no
  cuando son el 24 % (SWAN-SF: {{v:real/num/swan_sf/lstm|tracin_tgt/pot_f1/0|3}});
  en GECCO las versiones logarítmicas son las mejores.
* El umbral estático del artículo (τ·máx) depende de un único valor extremo;
  con TracIn casi no marca nada en reales.

## 7. Propuesta 2: reponderación iterativa

### 7.1 Réplica

{{r:rep_syn_mejoras}}

![Contaminación, réplica](figures/fig_replica_contaminacion.png)

Para ver si un método **se degrada** con la contaminación usamos AUROC: el
AUPRC de un detector al azar es la proporción de anomalías, así que el AUPRC
sube solo cuando hay más anomalías y una curva plana de AUPRC esconde una
degradación. Las diferencias entre métodos en un mismo nivel sí se pueden
leer en ambas métricas.

{{r:rep_cont_auroc}}

{{r:rep_cont_tests_auroc}}

{{r:rep_cont_auprc}}

{{r:rep_cont_tests}}

{{r:rep_cont_D_bin}}

{{r:rep_real_mejoras}}

* **Sintéticos de 400 pasos:** la reponderación no cambia el F1
  ({{w:syn_impr/Mejora 2 · reponderación suave (2 rondas)/d_mac|+3}} suave,
  {{w:syn_impr/Mejora 2 · recorte (2 rondas)/d_mac|+3}} recorte; p > 0.2) y apenas
  el AUPRC.
* **El problema que motiva la mejora es real.** TracIn del artículo es el
  mejor detector con 1 % de anomalías (AUROC
  {{w:cont/auroc/TracIn (artículo)/0|3}}, contra {{w:cont/auroc/LOF/0|3}} de LOF y
  {{w:cont/auroc/Error LSTM (PD)/0|3}} del error del LSTM) y el que más se degrada:
  con 26 % cae a {{w:cont/auroc/TracIn (artículo)/6|3}}, por debajo del error del
  LSTM ({{w:cont/auroc/Error LSTM (PD)/6|3}}) y de LOF ({{w:cont/auroc/LOF/6|3}}),
  que pierden mucho menos. Es lo que el artículo anticipa (el modelo aprende las
  anomalías y su pérdida deja de distinguirlas), aunque el artículo no lo mide:
  sólo lo propone para explicar SWAN-SF.
* **La reponderación frena la caída, pero poco.** El recorte sube el AUROC de
  TracIn {{w:cont/tests_auroc/Recorte − TracIn/2/0|+3}} a
  {{w:cont/tests_auroc/Recorte − TracIn/6/0|+3}} desde 10 % de anomalías, en las 10
  semillas (p = {{w:cont/tests_auroc/Recorte − TracIn/6/1|e}}); en AUPRC son
  {{w:cont/tests/Recorte − TracIn/1/0|+3}} a {{w:cont/tests/Recorte − TracIn/3/0|+3}}. De
  una caída de ≈ 0.17 de AUROC recupera ≈ 0.02. Con 1 % no hace nada.
* **Con la proyección al objetivo (mejoras 2 + 3) la caída casi desaparece**:
  {{w:cont/auroc/Recorte + proy./0|3}} con 1 % y {{w:cont/auroc/Recorte + proy./6|3}}
  con 26 %; es el mejor detector de la serie desde ≈ 10 % (con 26 %, LOF
  {{w:cont/auroc/LOF/6|3}}). Sobre la proyección sola, el recorte suma
  {{w:cont/tests_auroc/Recorte + proy. − Proy. objetivo/6/0|+3}} de AUROC (10/10 semillas
  desde 10 %).
* **¿Se puede frenar la degradación restante?** Probamos cuatro variantes,
  todas fijadas antes de ver resultados:

{{r:rep_cont_fix}}

{{r:rep_cont_fix_diag}}

  * **Recorte adaptativo (mejoras 1 + 2):** recortar las ventanas que superan
    el umbral IQR de la mejora 1, en vez de z > 2, sube el AUROC con 26 % de
    {{w:cont_fix/auroc/Recorte z > 2 + proy. (2 rondas)/6|3}} a
    {{w:cont_fix/auroc/Recorte adaptativo IQR + proy. (3 rondas)/6|3}}
    ({{w:cont_fix/tests/Recorte adaptativo IQR + proy. (3 rondas) − Recorte z > 2 + proy. (2 rondas)/6/0|+3}},
    10/10 semillas desde 14 %); la caída de 1 % a 26 % pasa de −0.048 a
    −0.032. Es la mejor variante de TracIn y la que menos se degrada; con 1 %
    pierde un poco
    ({{w:cont_fix/tests/Recorte adaptativo IQR + proy. (3 rondas) − Recorte z > 2 + proy. (2 rondas)/0/0|+3}}).
  * **Épocas tempranas** ayuda sobre la proyección sola
    ({{w:cont_fix/tests/Épocas tempranas + proy. − Proy. objetivo/6/0|+3}}, 10/10 desde
    10 %), pero menos que el recorte. **Un LSTM más chico** no ayuda.
  * **Ensamble con LOF**: es el que menos cae, pero porque empieza más abajo:
    con pocas anomalías es peor que recorte + proyección.
  * **Por qué no desaparece:** aun el recorte adaptativo sólo saca del
    entrenamiento ≈ 30 % de las ventanas anómalas con 26 % de contaminación.
    Cuando las anomalías son una cuarta parte de los datos, el modelo las
    aprende desde el principio y su self-influence se parece a la de las
    normales: no hay forma de reconocerlas para quitarlas. Todos los detectores
    se degradan (LOF {{w:cont_fix/auroc/LOF/0|3}} → {{w:cont_fix/auroc/LOF/6|3}}, error
    del LSTM {{w:cont_fix/auroc/Error LSTM (PD)/0|3}} → {{w:cont_fix/auroc/Error LSTM (PD)/6|3}}); el
    artículo mismo dice que con 24 % (SWAN-SF) las anomalías «casi quedan fuera
    de la definición de anomalía». Con el recorte adaptativo nuestra variante
    se degrada menos que LOF y que el error del LSTM.
  * Sólo se probó en esta serie sintética; falta comprobarlo en reales.
* **Reales (1 ronda):** en GECCO la suave sube TracIn de
  {{w:real_impr/gecco/tracin/auprc|3}} a {{w:real_impr/gecco/rw_soft_r1/auprc|3}} de AUPRC y
  de {{w:real_impr/gecco/tracin/D_bin|3}} a {{w:real_impr/gecco/rw_soft_r1/D_bin|3}} de F1
  dinámico; en Credit Card empeora ({{w:real_impr/creditcard/rw_soft_r1/auprc|3}} vs
  {{w:real_impr/creditcard/tracin/auprc|3}}); en CICIDS y SWAN-SF no cambia
  ({{w:real_impr/swan_sf/rw_soft_r1/auprc|3}} vs {{w:real_impr/swan_sf/tracin/auprc|3}}). Son 3 semillas.
* **Corrección.** En versiones anteriores de este informe dijimos que el error
  del LSTM *mejoraba* con la contaminación y que TracIn no se degradaba en la
  réplica: era un efecto del AUPRC, que sube con la proporción de anomalías.
  En AUROC ambos se degradan, y TracIn mucho más
  ({{w:cont/auroc/Error LSTM (PD)/0|3}} → {{w:cont/auroc/Error LSTM (PD)/6|3}} el error
  del LSTM; {{w:cont/auroc/TracIn (artículo)/0|3}} → {{w:cont/auroc/TracIn (artículo)/6|3}}
  TracIn).

### 7.2 Primer experimento

{{t:syn_reweighting}}

![Contaminación](figures/fig_contaminacion.png)

{{t:cont_auprc}}

{{t:real_reweighting}}

{{RW_REAL}}

* **GECCO es el caso donde la reponderación funciona**: TracIn sube de
  {{v:real/num_rwseeds/gecco/lstm|tracin/auprc/0|3}} a {{v:real/num_rwseeds/gecco/lstm|rw_soft_r3/auprc/0|3}} (suave) y
  {{v:real/num_rwseeds/gecco/lstm|rw_trim_r3/auprc/0|3}} (recorte), en las 5 semillas. Es el dataset con más
  memorización (§5). Nuestra interpretación, que todavía no verificamos
  directamente: al quitarle peso a las ventanas de mayor self-influence, el
  modelo deja de aprender las anomalías y su pérdida queda alta. Con el
  umbral log-MAD, sin etiquetas, el F1 de TracIn + recorte es
  {{v:real/num_rwseeds/gecco/lstm|rw_trim_r3/log_mad_f1/0|3}}, contra
  {{v:real/num_rwseeds/gecco/lstm|tracin/dyn_f1/0|3}} del TracIn original con el umbral
  dinámico (que usa etiquetas). Sigue lejos del 0.761 publicado.
* En Credit Card ayuda sobre todo a la proyección al objetivo
  ({{v:real/rw_delta/creditcard/rw_soft_r3_tgt/0|+3}}, suave); en SWAN-SF (24 % anómalo) empeora
  ligeramente en todas las semillas, y en CICIDS no cambia nada.
* En los 15 sintéticos la mejora es pequeña pero muy consistente (tabla de
  arriba). En la contaminación controlada: {{v:cont/rw_tests/0.01/0|+3}} con 1 %,
  ≈ 0 desde el 18 %.
* La reponderación ayuda sólo donde hay anomalías raras que el modelo llega a
  aprender. Con contaminación alta, las ventanas anómalas no se distinguen lo
  suficiente en la primera ronda como para bajarles el peso.
* Lo que aquí parecía una mejora del error del LSTM con la contaminación
  (AUPRC de {{v:cont/auprc/LSTM error (PD)/0|2}} a {{v:cont/auprc/LSTM error (PD)/6|2}}) es el efecto de la
  proporción de anomalías sobre el AUPRC; véase la corrección en §7.1.

## 8. Limitaciones y desviaciones del artículo

* **Supuestos de la réplica.** Que el artículo estandariza los canales y
  reporta F1 macro en sintéticos lo inferimos de que es la única combinación que
  reproduce sus cifras (§2.1); no está escrito en el artículo. El patrón por
  dataset no coincide (correlación {{w:syn/corr_D|+2}}), así que puede haber otra
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
