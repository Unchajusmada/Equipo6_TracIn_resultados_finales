# Auditoría final de TracIn · M1, evaluación y reproducibilidad

19 de noviembre de 2026 · Rama `audit/check-m1-evaluation` desde `783f193`

## 1. Componentes verificados

- **M1 sin etiquetas.** El método final usa Q3 + 1.5·(Q3−Q1) sobre los scores de todos los instantes válidos. MAD usa mediana + 3·1.4826·mediana(|score−mediana|). MAD, las variantes logarítmicas y POT son alternativas exploratorias. POT usa cuantil base 0.98, cola 0.001 y el máximo si hay menos de cinco excesos o falla el ajuste. La referencia estática final usa percentil 85 en sintéticos y contaminación, y 87 en los otros bloques. La fracción del máximo es una lectura histórica separada. Las reglas sin supervisión conservan sus predicciones al cambiar las etiquetas. Con empates se aplica `score >= umbral`. Una serie constante queda enteramente marcada por IQR y MAD, incluidos los ceros. Los umbrales por sí solos aceptan entradas de un punto y los extremos finitos probados. [Umbrales](src/common.py) · [Protocolo](PROTOCOLO_FINAL.md)
- **Evaluación final.** AUPRC es *average precision* y AUROC es el área ROC de scikit-learn en todo el tramo válido. Se quitan 19 pasos por extremo antes de calcular métricas y umbrales. El F1 principal se mide en la segunda mitad de ese tramo, desde `floor(n_válido/2)`. El F1 binario usa la clase anómala y el macro promedia los F1 de ambas clases fijas, asignando cero a un denominador nulo. `dyn_bin_mac` mide macro con el corte que optimizó F1 binario. Con ambas clases y score constante, AUROC = 0.5 y AP equivale a la prevalencia. `win` y `tgt` toman el mismo vector de ventanas por variante. Las dos proyecciones corregidas de fase 4 se usan en evaluación y halo. La prueba de integración verificó la alineación de etiquetas y scores incluso con un valor extremo antes del recorte. [Evaluador](src/final/evaluate.py) · [Halo](src/final/paper.py)
- **Alcance científico.** El ajuste, la normalización y la calibración no supervisada sobre la misma serie puntuable constituyen una evaluación transductiva. La referencia dinámica es supervisada en la primera mitad y usa una grilla de scores de ambas mitades. H2 queda pendiente sin δ, con diferencias de F1 descriptivas. `audit_statistics.py` conserva el análisis corregido de unidades, separado de `stats.py` y de las tablas derivadas de su JSON. H3 estima el aporte incremental de `adapt_r3` con proyección `tgt` y, para F1, la misma regla IQR. El contraste no estima una interacción entre las tres mejoras. [Fase 1](AUDITORIA_ESTADISTICA_FASE1.md) · [Fase 4](AUDITORIA_FASE4_M2_M3.md)

## 2. Correcciones y límites del cierre

| Componente | Resultado final |
|---|---|
| Cohorte NASA | La preparación comprueba 80 canales sin P-2 y la planificación, evaluación y halo comprueban semillas 0–4 para cada canal. Evaluación y halo excluyen con aviso cualquier NPZ histórico de P-2 presente en la ruta principal. El lector estadístico valida tamaño, semillas, pertenencia de P-2 y emparejamiento. Los CSV y derivados nuevos identifican la cohorte. Se rechazan cachés de halo de otra cohorte y se cotejan sus corridas con la evaluación. La reproducción histórica de 81 requiere `--nasa-first-row` y una copia aislada. |
| Validación de métricas | El evaluador exige etiquetas binarias, vectores numéricos finitos y longitud exacta de scores por ventana o instante. Rechaza un recorte con menos de dos puntos y señala AUROC indefinida si falta una clase en el tramo válido. El error identifica el archivo y detiene el bloque antes de escribir CSV. Las agregaciones pareadas rechazan llaves incompletas, duplicadas o diferencias indefinidas. Registrar el archivo y motivo en la bitácora, corregir su entrada y repetir el bloque completo. AP, AUROC y F1 conservan sus fórmulas para casos válidos. |
| Referencia dinámica histórica | Se conservaron los cortes candidatos definidos por percentiles de los scores de ambas mitades. Solo las etiquetas de la primera mitad intervienen en la selección del corte. La prueba confirma que cambiar otros scores de la segunda mitad puede cambiar la predicción de un punto con score fijo. Una referencia calibrada estrictamente con la primera mitad queda como experimento futuro separado. |
| Evaluador legado | `replica/protocol.py` sigue fallando al dividir por el máximo cuando los scores son todos cero. Su oráculo puede informar F1=1 con un empate cuyo máximo alcanzable es 2/3. Se caracterizaron ambos casos con pruebas ordinarias. Los ejecutores finales usan `screen.parse_variant` para resolver la configuración. La evaluación final llama a `final/evaluate.py`, con una prueba que comprueba la independencia de la ruta legada. |
| Productos históricos y corregidos | El JSON de auditoría se identifica como `corrected-audit` y los derivados de `stats.py` y `paper.py` como `historical-statistical-method`. El manuscrito archivado conserva su texto histórico. Para la versión definitiva, H2 debe quedar pendiente, H3 debe interpretarse como aporte incremental y el rendimiento debe identificarse como transductivo. |

## 3. Pruebas ejecutadas

`python -B -m unittest discover -s tests -v` ejecutó **50 pruebas, todas aprobadas, sin fallos esperados**. Se usaron arreglos y archivos temporales artificiales en CPU con scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1 y PyTorch 2.14.0+cpu. La auditoría histórica de 81 canales, ejecutada en memoria, reprodujo exactamente los seis bloques y H6 del JSON de fase 1. H2 quedó pendiente en los seis bloques. Se comprobó en Git que los 109 archivos históricos, el protocolo y el evaluador legado quedaron intactos. No se descargaron datasets, no se ejecutaron los experimentos definitivos y no se generaron cifras del artículo.

Archivos del cierre: `src/nasa_cohort.py`, `src/data_bench.py`, `src/final/run_bench.py`, `src/final/evaluate.py`, `src/final/audit_statistics.py`, `src/final/stats.py`, `src/final/paper.py`, `tests/test_final_evaluation.py`, `tests/test_audit_statistics.py` y este informe.

## 4. Experimentos que deberá ejecutar mi compañero

**Entradas y diseño.** Preparar una copia de trabajo aislada del commit aprobado, con directorios de corridas nuevos. Recuperar archivos originales o NPZ procesados con hashes y procedencia. Las rutas siguientes son relativas a esa copia.

| Bloque | Insumos y corridas esperadas |
|---|---|
| Sintéticos | Generador `src/synthetic.py`. 15 familias × semillas 0–19, 300 NPZ con las nueve combinaciones de L∈{4,8,16}, P∈{1,2,4}. El análisis primario usa L4P1. |
| Contaminación | Generador local, 4 000 puntos. Tres mezclas `01234`, `01`, `234` × siete dosis nominales {1,5,10,15,20,25,30}% × semillas 0–19, 420 NPZ L4P1. La semilla generadora es 10 000 + semilla. |
| Reales | En `data/raw`: `gecco/gecco.csv`, `creditcard/creditcard.csv`, `cicids/GeneratedLabelledFlows.zip` con el miembro *Thursday-WorkingHours-Morning-WebAttacks* y los TSV con extensión CSV de `swan_sf/petdataset_01/`. Alternativa equivalente: los cuatro `data/processed/<id>.npz` verificados. Cuatro datasets × semillas 0–9, 40 NPZ L4P1. |
| SMD y semi | `data/raw/smd/{test,test_label,train}/machine-*.txt`. SMD usa 28 máquinas de prueba × semillas 0–4, 140 NPZ. Semi usa 10 fondos de entrenamiento, uno de cada tres por orden de máquina, primeros 10 000 puntos, tres mezclas, cinco dosis {1,5,10,20,30}% y semillas 0–2, 450 NPZ. Conservar el orden usado para la semilla de inyección de `run_bench.py`. |
| NASA | `data/raw/smap_msl/labeled_anomalies.csv` y `test/<chan_id>.npy` dentro del mismo directorio, del snapshot SMAP/MSL verificado. Principal de 80 canales × semillas 0–4, 400 NPZ L4P1 y ningún P-2. Evaluar ambos eventos de P-2 exige el manifiesto y las dos matrices de la fase 2. |

Conservar los hiperparámetros de `lab.Cfg` y `screen.parse_variant`. Sintéticos y contaminación usan última capa, CPU y un hilo por proceso, batch 32 y máximo 100 épocas. Reales, SMD, NASA y semi usan gradiente completo, batch 256 y máximo 50 épocas. El ejecutor de reales fija CUDA y el de benchmarks admite `--device`. El entorno original documentado es Python 3.13, PyTorch 2.10/CUDA 12.8, scikit-learn 1.8, NumPy 2.4, SciPy 1.17 y pandas 3.0, GPU RTX 5090. Recuperar las versiones exactas cuando sea posible, fijar Matplotlib para derivados y registrar sistema, driver, hardware e hilos. [Entorno e insumos originales](README.md)

**Reutilización y repeticiones.** Reutilizar etiquetas alineadas y scores de ventana de base, PD, `soft` completado y referencias por instante con procedencia comprobable. Repetir `trim` y `adapt` desde la primera ronda con un lote descartado de peso cero, incluidas las rondas posteriores. Si no se puede identificar esas corridas, repetir las ramas cubiertas. Repetir `soft` si se abortó por desbordamiento. Reproyectar y reevaluar los seis bloques con el código estable de M3. Un cambio de calibración dinámica exige reevaluar sus métricas. Los CLI de entrenamiento generan un NPZ por trabajo y omiten archivos existentes, sin selector por rama. Por ello, usar los ejecutores completos para trabajos que se deban regenerar y reunir en una carpeta de evaluación únicamente NPZ verificados. Los CSV archivados sirven para comparación histórica y sensibilidad, pero no reconstruyen scores por ventana.

**Comandos para uso posterior en la copia aislada.** Ejecutar tras verificar el manifiesto y las precondiciones de la sección 5. Preparar datos solo cuando haga falta generar los NPZ procesados.

```powershell
$env:PYTHONUTF8 = '1'
python -B src/data_real.py
python -B src/data_bench.py --block all
python -B src/final/run_synth.py --seeds 20 --workers 4
python -B src/final/run_contam.py --seeds 20 --workers 4
python -B src/final/run_real.py --seeds 10 --seed0 0
python -B src/final/run_bench.py --blocks smd,nasa,semi --device cuda
foreach ($block in 'synth','contam','real','smd','nasa','semi') {
    python -B src/final/evaluate.py $block --workers 4
    if ($LASTEXITCODE) { throw "Falló evaluate.py para $block" }
}
# Análisis corregido con la cohorte principal NASA de 80
python -B src/final/audit_statistics.py --input results/final > statistics_audit.json
```

En una copia adicional para los derivados del método estadístico histórico, la secuencia disponible es `python -B src/final/stats.py`, `python -B src/final/paper.py --workers 4` y `python -B src/final/macros.py`. Escribe tablas, perfiles, figuras y JSON en `results/final`. Evitar `--no-halo` con cachés de otra cohorte o revisión. Si se requiere reproducir el P-2 de primera fila, usar otra copia aislada y pasar `--nasa-first-row` a la preparación NASA, `run_bench.py`, `evaluate.py nasa`, `audit_statistics.py` y, si se generan esos derivados, a `stats.py` y `paper.py`.

## 5. Criterios para declarar el proyecto listo

1. Verificar la disponibilidad y los hashes de los insumos reales, los 80 canales y sus cinco semillas, y los directorios aislados por cohorte. Registrar el alcance transductivo de la referencia dinámica.
2. Antes de resumir cada bloque, comprobar conteos esperados, semillas y configuraciones completas, llaves únicas, finitud, etiquetas binarias alineadas, longitudes `n−L−P+1` o `n` según el tipo de score, cobertura tras el recorte y pares sobre los mismos instantes. Registrar cualquier métrica indefinida o corrida incompleta. Verificar que P-2 esté ausente de NPZ, CSV y derivados principales.
3. Entregar manifiesto con commit, comandos, versiones, dispositivo, semillas, hashes y procedencia por trabajo, además de NPZ con scores por ventana y etiquetas (y scores `ts_*` cuando proceda), registros de ejecución, seis pares `*_eval.csv` y `*_diag.csv` y el JSON estadístico corregido. Entregar tablas, figuras, perfiles y macros con su variante estadística y cohorte identificadas. Cotejar los hashes de los históricos antes y después.
4. Antes de redactar conclusiones, comprobar que H2 siga pendiente sin δ, que H3 se describa como ganancia incremental y que las cifras se identifiquen como transductivas.

**La fase 5 queda cerrada en código dentro del alcance autorizado.** La preparación para los experimentos definitivos requiere verificar los insumos reales y sus manifiestos y pasar los controles de cada bloque en la copia aislada. Esa verificación experimental sigue pendiente.
