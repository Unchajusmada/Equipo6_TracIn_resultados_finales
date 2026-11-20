# Auditoría científica de TracIn · Fase 2: anotaciones SMAP/MSL

19 de noviembre de 2026 · Hallazgo A-01 de [AUDITORIA_CODEX.md](AUDITORIA_CODEX.md)

## Cierre de código

Telemanom aclara que las dos filas de P-2 describen eventos y datos de periodos distintos del mismo canal. Los índices parten de cero y el solapamiento es entre coordenadas locales de secuencias distintas. Se retira la política de combinar filas sobre un solo NPY. [Telemanom, issue #8, preguntas 1 y 5](https://github.com/khundman/telemanom/issues/8)

- El modo predeterminado excluye P-2 de la preparación, el resumen de series y la planificación de trabajos. Cada otra serie conserva su matriz, los intervalos inclusivos de su propia fila y cinco semillas. Un ID repetido inesperado se rechaza hasta conocer la correspondencia entre sus filas y matrices.
- Para reproducir la máscara histórica de P-2, usar `--nasa-first-row` tanto en `src/data_bench.py --block nasa` como en `src/final/run_bench.py --blocks nasa`, en una copia aislada con los insumos históricos y sin reutilizar NPZ de corridas previas. La opción selecciona la primera fila física del CSV y programa una serie por semilla. Su cobertura se limita a la configuración histórica de un evento. El planificador omite P-2 en modo predeterminado aunque exista un NPZ anterior.
- La suite completa pasó **14 pruebas**. La regresión usa dos filas P-2 con el mismo ID, misión y longitud de 8209. En modo principal se omite P-2. En modo histórico se etiquetan únicamente los 1226 índices inclusivos de la primera fila, 5350–6575. También se comprueban las máscaras de filas únicas, los límites de índices, la validación de entradas y la planificación de ambos modos.

## Sensibilidad con CSV archivados

Se cargaron y validaron los seis `*_eval.csv` con `load_block` de `src/final/audit_statistics.py`. Para el modo principal se eliminaron en memoria las cinco parejas de semillas de P-2 de cada comparación L4P1. Se reutilizó `analyze_block` en ambas cohortes, con la media de las cinco semillas por canal y el mismo ajuste Holm. El escenario con P-2 usa las métricas archivadas con la máscara histórica. El intervalo bilateral de H2 describe el efecto bajo los supuestos del análisis. H2 sigue pendiente de margen en ambas cohortes.

| Contraste | Δ medio con P-2 (81) | Δ medio sin P-2 (80) | IC 95 % con P-2 | IC 95 % sin P-2 | p Holm con / sin P-2 |
|---|---:|---:|---:|---:|---:|
| H1 | −0.00657 | −0.00691 | [−0.02469, +0.01155] | [−0.02525, +0.01143] | 1 / 1 |
| H2 | −0.07079 | −0.07034 | [−0.11505, −0.02654] | [−0.11515, −0.02553] | — / — |
| H3a | +0.01482 | +0.01465 | [+0.00442, +0.02521] | [+0.00413, +0.02517] | 0.01735 / 0.02090 |
| H3b | +0.01105 | +0.01206 | [−0.00032, +0.02241] | [+0.00074, +0.02339] | 0.14143 / 0.09272 |
| H5a | −0.00400 | −0.00365 | [−0.01233, +0.00434] | [−0.01206, +0.00477] | 1 / 1 |
| H5b | −0.02138 | −0.02156 | [−0.03377, −0.00900] | [−0.03410, −0.00902] | 1 / 1 |

Los signos de las medias y las decisiones con p Holm al 5 % coinciden en ambas cohortes bajo el supuesto de independencia entre canales. Para H6, Spearman descriptivo cambia de +0.29310 a +0.30652 en NASA y de +0.29917 (109 series) a +0.31037 (108) al agrupar con SMD. SMD por separado conserva ρ = −0.11330 (28).

La cohorte completa reprodujo exactamente los resultados por bloque del JSON de fase 1. Los deltas por canal de las otras 80 series NASA y los cinco bloques ajenos a NASA coincidieron exactamente al excluir P-2. Se verificaron los SHA-256 de los 109 archivos de resultados antes y después. El análisis se efectuó sin descargas ni entrenamiento.

## Validación experimental pendiente

Para evaluar los dos eventos se necesitan las dos matrices originales de prueba y un manifiesto verificable que vincule cada fila del CSV con su archivo, misión y periodo o identificador de secuencia. El manifiesto debe identificar las versiones o hashes, documentar el orden de las muestras y confirmar la longitud y la alineación de los índices inclusivos de cada evento. Para calcular métricas hacen falta también puntuaciones por instante de cada secuencia con procedencia y configuración reproducibles. Los CSV archivados contienen métricas agregadas y permiten medir la sensibilidad a excluir P-2.
