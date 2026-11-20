# Auditoría estadística de TracIn · Fase 1

19 de noviembre de 2026 · Hallazgos C-01 y A-02 de [AUDITORIA_CODEX.md](AUDITORIA_CODEX.md)

## Resultado

**H2 queda sin evaluar para no inferioridad en los seis bloques.** Se recalcularon las diferencias de F1, pero el protocolo carece de un margen de pérdida admisible. El resultado histórico bilateral sigue disponible como contraste de diferencia. La cercanía a cero del efecto semisintético y su p histórico de 0.557 dejan abierta la cuestión de no inferioridad.

**La evidencia en los cuatro reales originales es descriptiva entre datasets.** El método final mejora la AP en los cuatro datasets y pierde F1 frente a la referencia dinámica en tres. El IC y p históricos basados en 40 pares de semillas se muestran solo como antecedente. En semi, la unidad pasa de 450 filas a 10 fondos para H1–H3/H5a–b y de 90 pares de caídas a 10 fondos para H4r/H5c.

## Archivos y alcance

| Archivo nuevo | Contenido |
|---|---|
| [src/final/audit_statistics.py](src/final/audit_statistics.py) | Lee métricas L4P1, verifica el diseño, empareja, agrega y emite JSON a stdout. |
| [tests/test_audit_statistics.py](tests/test_audit_statistics.py) | Pruebas unitarias e integración con los seis CSV archivados. |
| [results/audit_phase1/statistics.json](results/audit_phase1/statistics.json) | Resultado corregido, incluidos deltas por unidad, p crudos y ajustados, dispersión, tamaños de efecto y estabilidad por semilla en reales. |
| `AUDITORIA_ESTADISTICA_FASE1.md` | Metodología, verificación y comparación con el histórico. |

`git status --short --branch` informó que la carpeta no pertenece a un repositorio Git, por lo que la rama `audit/fix-statistics` no se creó. Se conservaron `src/final/stats.py`, `src/final/paper.py`, los CSV, JSON, tablas originales y fuentes LaTeX. No se descargaron datasets ni se ejecutaron entrenamientos. El histórico de la tabla procede de `results/final/final_numbers.json`, que almacena el Wilcoxon bilateral, Holm e IC bootstrap de la media. Los nuevos contrastes se recalcularon desde los seis CSV, con Python 3.14.7, NumPy 2.5.3 y SciPy 1.18.1 disponibles.

## Emparejamiento, estimandos y unidades

Se emparejan métodos por escenario/serie, semilla y, cuando corresponda, mezcla y dosis. Se comprueba que las tres variantes tengan el mismo conjunto de llaves, que no existan duplicados y que el diseño archivado esté completo y balanceado. Primero se calcula la diferencia por par y luego se agregan por unidad con peso uniforme para las celdas observadas. H4, H4r y H5c usan la diferencia de caídas entre 1 % y 30 % nominales, emparejando ambos niveles antes de agregar. Las medias históricas y nuevas coinciden al redondear por el balance del diseño. Cambian la unidad, la varianza y el contraste.

| Bloque | Unidad usada | Alcance y dependencia residual |
|---|---|---|
| Sintéticos | 20 semillas generadoras, cada una promedia los 15 escenarios fijos | La semilla cruza escenarios y cada escenario se observa bajo las mismas 20 semillas. El estimando es la media sobre nuevas realizaciones del menú fijo. Se guardan las 15 medias por escenario como descripción de heterogeneidad. Para generalizar a escenarios nuevos haría falta justificar cómo fueron muestreados y modelar el cruce escenario–semilla. |
| Contaminación controlada | 20 semillas generadoras, cada una promedia las mezclas y niveles incluidos | Las 21 celdas son condiciones fijas. Se conserva la dependencia inducida por reutilizar la semilla. H4 usa tres diferencias de caídas por semilla y luego promedia. El contraste describe los extremos del diseño, sin extenderse a otros patrones de inyección. |
| Reales originales | 4 datasets, media de 10 semillas por dataset | Solo descriptivo, sin p ni IC de generalización. El JSON conserva los deltas por semilla y dataset para estabilidad algorítmica. |
| SMD | 28 máquinas, media de 5 semillas | Unidad ya usada históricamente. Los IC y p son condicionales al supuesto de independencia entre máquinas. Falta justificar la independencia entre grupos de infraestructura para extrapolar. |
| SMAP/MSL | 81 canales observados, media de 5 semillas | Unidad ya usada históricamente. Los intervalos y contrastes se condicionan a independencia entre canales. Los CSV no identifican misión por fila para modelar esa dependencia. P-2 conserva las etiquetas históricas. |
| Semi | 10 fondos SMD, cada uno promedia mezclas, niveles y semillas | Los escenarios de inyección comparten fondo. La variación entre medias de fondos informa la incertidumbre. Se asume independencia entre estos fondos, con solo 10 unidades y posible dependencia entre máquinas. |

El procedimiento elegido aplica una t de una muestra sobre diferencias ya agregadas. El estimando es la **media de las diferencias de las unidades indicadas**. El JSON también informa desviación muestral, signos por unidad y `dz` (media dividida por la desviación muestral de esas diferencias). La t y el IC se apoyan en independencia entre unidades y normalidad aproximada de sus efectos o una aproximación muestral suficiente. Con varianza nula se informa el efecto sin inferencia t. La aparente mayor precisión en algunos sintéticos refleja un estimando condicionado a escenarios fijos, en el que se promedia heterogeneidad entre celdas dentro de cada semilla. No es evidencia de mayor precisión sobre escenarios nuevos. [NIST, análisis pareado](https://itl.nist.gov/div898/handbook/prc/section3/prc311.htm)

Un bootstrap agrupado sobre unidades independientes, conservando juntas las celdas del clúster, serviría como sensibilidad para la **media** con suficientes clústeres. Con cuatro datasets tiene resolución y representatividad limitadas. Un Wilcoxon desplazado por el margen y un IC compatible de localización requerirían simetría de diferencias y cambiarían el estimando general de media a localización. Una prueba de signos o su inversión respondería a un estimando de mediana y exige tratar empates. Para unidades cruzadas o anidadas, un modelo jerárquico exigiría una población objetivo y evidencia para las dependencias y componentes de varianza. [SciPy, Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)

## H2 y multiplicidad

PROTOCOLO_FINAL.md:61–66 compara **Final sin etiquetas ≥ Artículo dinámico con etiquetas** en F1 binario de la mitad de evaluación. En los CSV, el primer término es `adapt_r3+tgt/iqr_bin` y el segundo `base+win/dyn_bin`. Con una pérdida absoluta admisible δ > 0 justificada por el equipo, la prueba implementada evalúa H0: media(d) ≤ −δ frente a H1: media(d) > −δ. Se utiliza la cola superior de t con estadístico (media(d)+δ)/EE. El límite inferior unilateral nominal de 95 % supera −δ exactamente cuando el p crudo es menor que 0.05, salvo redondeo. La decisión ajustada usa p Holm y el límite nominal no representa cobertura simultánea. El IC bilateral de la tabla es un intervalo descriptivo de la misma media, disponible aunque falte δ en bloques con inferencia. Con cuatro reales se mantienen solo estimaciones descriptivas incluso si el equipo define un margen.

El script exige `--delta` y `--delta-justification` juntos si se desea realizar la prueba. El argumento configura un margen común para la ejecución. Si el equipo requiere márgenes específicos por bloque, deberá versionar esa decisión y adaptar explícitamente la interfaz. La corrida entregada dejó ambos argumentos vacíos. La ejecución automatizada con valores artificiales en la prueba unitaria valida código, sin asignar un margen científico a los resultados reales.

Se conservaron los contrastes primarios archivados por bloque, incluido el lugar de H2 en Holm. H2 pendiente recibe p nulo y cuenta para el tamaño de la familia, equivalente para los otros p a reservarle un p de 1. H5a–b en sintéticos, contaminación y reales conservan una familia exploratoria aparte. En SMD/NASA y semi H5 conserva la familia archivada del bloque. En particular, el código histórico incluye H1–H3 en semi, aunque la extensión del protocolo menciona H1–H3 explícitamente para SMD y SMAP/MSL. Mantener la familia archivada es una decisión de comparabilidad, con carácter confirmatorio de esas celdas pendiente de ratificación. Las pruebas nuevas son un reanálisis de datos ya observados. H1, H3, H4 y H5 usan la dirección favorable que expresaba el protocolo. Un p nuevo cercano a 1 con delta negativo indica falta de evidencia en esa dirección. El signo y el IC bilateral muestran el perjuicio estimado. Los IC de la tabla son nominales y los p están ajustados dentro de familia, por lo que sus decisiones pueden diferir.

H6 conserva su registro histórico como análisis de asociación. La comprobación de sensibilidad calculó Spearman para ΔAUROC de M3 con una media por serie. El histórico conjunto es ρ=0.299, n=109, p=0.000793 aproximadamente. El reanálisis descriptivo dio ρ=−0.113 en SMD, +0.293 en NASA y +0.299 agrupado. Las diferencias de signo y la estructura por benchmark limitan la interpretación del resultado agrupado. No se publican p nuevos para H6. Las comparaciones secundarias de `paper.py` (por celda, pruebas por dataset, Friedman/Nemenyi y halo) siguen siendo análisis históricos y quedan fuera del contraste primario corregido.

## Interpretación que cambia

- **H2 en todos los bloques.** Estado pendiente de margen. La diferencia media sigue cuantificable. En los reales originales, las cuatro diferencias por dataset son las siguientes.

  | Dataset | ΔAP H1 | ΔF1 H2 |
  |---|---:|---:|
  | CICIDS | +0.00272 | +0.00034 |
  | Credit Card | +0.07238 | −0.19067 |
  | GECCO | +0.07845 | −0.25652 |
  | SWAN-SF | +0.03181 | −0.07142 |

- **Reales originales, H1 y H5.** Los p históricos menores que 0.05 provenían de comparar semillas como si representaran series distintas. H1 conserva cuatro signos positivos para estas series. La inferencia hacia datasets nuevos queda sin estimar.
- **Semi, H3.** El IC de ΔAP incluye cero tras agrupar por fondo (−0.020 a +0.005). Para H3b, el nuevo p Holm es 0.0673 frente a 2.27e−20 histórico. H1, H4r, H5a/b y H5c mantienen efectos positivos y p ajustados por debajo de 0.05 en este modelo de unidades, condicionado a los supuestos anteriores.
- **SMD y NASA.** Se mantiene la unidad por serie. Los p cambian por el contraste de media en dirección de superioridad. En SMD los deltas negativos para H5 permanecen visibles en los intervalos. Los p cercanos a 1 no representan evidencia de empate. En NASA H3a sigue por debajo de 0.05 ajustado bajo independencia condicional.

## Verificaciones ejecutadas y límites

- `python -B -m unittest discover -s tests -v` ejecutó 7 pruebas con resultado OK. Cubren emparejamiento independiente del orden, rechazo de pares faltantes y duplicados, promedio de semillas, efecto estandarizado, signo de diferencia de caídas, H2 con y sin margen, coherencia entre límite unilateral y p crudo, reserva de Holm y conteos de unidades en los seis CSV.
- `python -B src/final/audit_statistics.py` recalculó el análisis sin margen. Se guardó stdout como `results/audit_phase1/statistics.json`. El evaluador solo lee CSV y no importa PyTorch ni ejecuta los scripts de entrenamiento o reportes originales.
- Una segunda ejecución reprodujo el JSON guardado tras normalizar finales de línea. Al generar la tabla se comprobó que las 39 medias nuevas coinciden con las archivadas dentro de una tolerancia absoluta de 10⁻¹².
- Las tablas históricas se contrastan con los agregados originales archivados. Los IC y p históricos se recuperan de ese archivo, sin volver a ejecutar el bootstrap original ni sobrescribir tablas. Las puntuaciones por instante, etiquetas completas y checkpoints no están disponibles en esta copia. El reanálisis no valida el cálculo original de F1/AP/AUROC, los scores de P-2 ni el modelo de dependencia dentro de un benchmark.

## Decisiones científicas pendientes

1. Justificar δ en puntos absolutos de F1, con un criterio operacional independiente de estos resultados, y especificar si corresponde un margen común.
2. Ratificar si el estimando sintético se limita al menú fijo o busca generalizar a nuevas familias. Acordar el universo de fondos y grupos de máquinas en semi.
3. Establecer si máquinas o canales pueden tratarse como independientes y recuperar identificadores de grupo/misión. La disponibilidad de dos misiones impone límites a una inferencia entre misiones.
4. Ratificar el alcance de H1–H3 en semi, la familia de Holm y el papel separado de H6 y demás análisis secundarios antes de revisar las conclusiones del artículo.
5. Acordar nuevos datasets independientes para una conclusión generalizable. Las diez semillas de cada real permiten valorar estabilidad dentro de la serie.

## Comparación completa de resultados

Las columnas de efecto muestran el Δ medio en unidades originales de la métrica. H1/H3a/H5b usan AP, H2/H3b F1 binario, H5a AUROC y H4/H4r/H5c la ventaja en caída de AUROC. Cada n indica el número de unidades usado por el análisis correspondiente. «—» indica inferencia omitida o H2 pendiente. Los valores de p histórico y reanálisis son Holm. Las cifras pequeñas se muestran sin sustituirlas por cero.


| Bloque | Contraste | n anterior / nuevo | Efecto anterior / nuevo | IC 95 % anterior | IC 95 % nuevo | p Holm anterior | p Holm nuevo |
|---|---|---:|---:|---:|---:|---:|---:|
| synth | H1 | 300 / 20 | +0.215 / +0.215 | [+0.185, +0.245] | [+0.197, +0.233] | 1.72e-27 | 1.16e-15 |
| synth | H2 | 300 / 20 | +0.218 / +0.218 | [+0.190, +0.244] | [+0.185, +0.250] | 1e-31 | — |
| synth | H3a | 300 / 20 | +0.009 / +0.009 | [+0.002, +0.015] | [+0.001, +0.016] | 1.35e-17 | 0.023 |
| synth | H3b | 300 / 20 | +0.022 / +0.022 | [+0.015, +0.028] | [+0.015, +0.029] | 8.61e-14 | 3.59e-06 |
| synth | H5a | 300 / 20 | +0.019 / +0.019 | [+0.011, +0.028] | [+0.013, +0.026] | 6.82e-06 | 9.9e-07 |
| synth | H5b | 300 / 20 | +0.206 / +0.206 | [+0.177, +0.236] | [+0.191, +0.222] | 6.43e-28 | 8.49e-17 |
| contam | H1 | 420 / 20 | +0.328 / +0.328 | [+0.306, +0.350] | [+0.321, +0.336] | 3.79e-61 | 3.37e-26 |
| contam | H2 | 420 / 20 | +0.167 / +0.167 | [+0.147, +0.188] | [+0.160, +0.175] | 1.84e-36 | — |
| contam | H3a | 420 / 20 | +0.045 / +0.045 | [+0.035, +0.055] | [+0.039, +0.052] | 4.28e-53 | 6.96e-12 |
| contam | H3b | 420 / 20 | +0.047 / +0.047 | [+0.042, +0.051] | [+0.044, +0.050] | 2.01e-51 | 9.69e-18 |
| contam | H4 | 60 / 20 | +0.127 / +0.127 | [+0.108, +0.146] | [+0.119, +0.136] | 3.46e-11 | 9.69e-18 |
| contam | H5a | 420 / 20 | +0.044 / +0.044 | [+0.035, +0.054] | [+0.042, +0.046] | 6.34e-14 | 1.5e-22 |
| contam | H5b | 420 / 20 | +0.283 / +0.283 | [+0.260, +0.306] | [+0.278, +0.288] | 2.58e-53 | 2.83e-29 |
| real | H1 | 40 / 4 | +0.046 / +0.046 | [+0.036, +0.056] | — | 7.28e-12 | — |
| real | H2 | 40 / 4 | -0.130 / -0.130 | [-0.160, -0.098] | — | 3.11e-07 | — |
| real | H3a | 40 / 4 | +0.020 / +0.020 | [+0.005, +0.035] | — | 0.856 | — |
| real | H3b | 40 / 4 | +0.002 / +0.002 | [-0.001, +0.006] | — | 0.856 | — |
| real | H5a | 40 / 4 | +0.018 / +0.018 | [+0.007, +0.029] | — | 0.0127 | — |
| real | H5b | 40 / 4 | +0.027 / +0.027 | [+0.013, +0.041] | — | 0.0127 | — |
| smd | H1 | 28 / 28 | -0.062 / -0.062 | [-0.116, -0.002] | [-0.122, -0.002] | 0.0612 | 1 |
| smd | H2 | 28 / 28 | -0.071 / -0.071 | [-0.166, +0.029] | [-0.175, +0.034] | 0.372 | — |
| smd | H3a | 28 / 28 | -0.008 / -0.008 | [-0.055, +0.043] | [-0.060, +0.044] | 0.399 | 1 |
| smd | H3b | 28 / 28 | -0.019 / -0.019 | [-0.033, -0.006] | [-0.034, -0.005] | 0.0138 | 1 |
| smd | H5a | 28 / 28 | -0.023 / -0.023 | [-0.028, -0.018] | [-0.028, -0.017] | 8.94e-08 | 1 |
| smd | H5b | 28 / 28 | -0.054 / -0.054 | [-0.072, -0.036] | [-0.073, -0.035] | 4.7e-05 | 1 |
| nasa | H1 | 81 / 81 | -0.007 / -0.007 | [-0.025, +0.010] | [-0.025, +0.012] | 0.946 | 1 |
| nasa | H2 | 81 / 81 | -0.071 / -0.071 | [-0.115, -0.029] | [-0.115, -0.027] | 0.0102 | — |
| nasa | H3a | 81 / 81 | +0.015 / +0.015 | [+0.005, +0.025] | [+0.004, +0.025] | 0.00388 | 0.0174 |
| nasa | H3b | 81 / 81 | +0.011 / +0.011 | [+0.000, +0.023] | [-0.000, +0.022] | 0.243 | 0.141 |
| nasa | H5a | 81 / 81 | -0.004 / -0.004 | [-0.012, +0.004] | [-0.012, +0.004] | 0.715 | 1 |
| nasa | H5b | 81 / 81 | -0.021 / -0.021 | [-0.035, -0.010] | [-0.034, -0.009] | 0.0139 | 1 |
| semi | H1 | 450 / 10 | +0.040 / +0.040 | [+0.033, +0.047] | [+0.023, +0.057] | 1.15e-17 | 0.00117 |
| semi | H2 | 450 / 10 | -0.005 / -0.005 | [-0.012, +0.001] | [-0.015, +0.004] | 0.557 | — |
| semi | H3a | 450 / 10 | -0.008 / -0.008 | [-0.010, -0.006] | [-0.020, +0.005] | 7.78e-11 | 1 |
| semi | H3b | 450 / 10 | +0.008 / +0.008 | [+0.006, +0.009] | [+0.000, +0.016] | 2.27e-20 | 0.0673 |
| semi | H4r | 90 / 10 | +0.017 / +0.017 | [+0.010, +0.023] | [+0.010, +0.024] | 2.21e-05 | 0.00117 |
| semi | H5a | 450 / 10 | +0.033 / +0.033 | [+0.030, +0.037] | [+0.027, +0.040] | 4.73e-43 | 2.44e-06 |
| semi | H5b | 450 / 10 | +0.048 / +0.048 | [+0.041, +0.055] | [+0.035, +0.061] | 4.42e-31 | 6.36e-05 |
| semi | H5c | 90 / 10 | +0.016 / +0.016 | [+0.010, +0.022] | [+0.012, +0.021] | 3.92e-06 | 6.36e-05 |
