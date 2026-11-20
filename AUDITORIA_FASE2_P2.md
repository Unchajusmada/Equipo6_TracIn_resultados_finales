# Auditoría científica de TracIn · Fase 2: anotaciones SMAP/MSL

Hallazgo A-01 de [AUDITORIA_CODEX.md](AUDITORIA_CODEX.md) · Rama `audit/fix-labels-p2`

## Cambio y alcance

El cargador agrupaba de hecho los archivos por ID, pero enumeraba un canal por cada fila del CSV y recuperaba siempre la primera fila para producir la máscara. Ahora hay un grupo por canal y una fila de resumen por serie. El modo predeterminado combina todas las anotaciones compatibles en una máscara binaria. Para cada intervalo inclusivo [a,b] asigna 1 a las posiciones de a a b. Los solapamientos y los duplicados cuentan una sola vez y los huecos entre intervalos separados se conservan.

| Archivo | Cambio |
|---|---|
| [src/data_bench.py](src/data_bench.py) | Valida el CSV, agrupa por canal y construye la máscara. Añade `--block nasa` para preparar solo este bloque y `--nasa-first-row` para comparar con la política histórica en datos válidos. |
| [src/final/run_bench.py](src/final/run_bench.py) | Obtiene la lista validada de canales NASA una sola vez y programa un trabajo por canal y semilla. |
| [tests/test_nasa_annotations.py](tests/test_nasa_annotations.py) | Crea CSV/NPY/NPZ artificiales temporales y comprueba deduplicación, unión, rechazo de datos incompatibles y planificación. |

El identificador debe ser apto para nombres de archivo, la misión debe ser SMAP o MSL y la longitud declarada debe ser un entero positivo. Todas las filas de un canal deben coincidir en misión y longitud porque el `.npy` y los resultados se identifican por canal. Los intervalos deben ser pares de enteros, sin valores booleanos, y satisfacer `0 <= inicio <= fin < num_values`. Una lista vacía representa una fila sin intervalos. Para cada canal anotado debe existir el `.npy` correspondiente. Durante la preparación se comprueba que sea una matriz con la longitud declarada y al menos una columna. Las matrices sin fila de anotación quedan fuera de este benchmark. El campo textual `class` no participa en la máscara binaria. La identidad de la misión se valida entre filas del CSV. Un NPY con la longitud esperada no permite confirmar por sí solo su misión, la alineación temporal ni la correspondencia con el snapshot de anotaciones.

En modo histórico se seleccionan los intervalos de la primera fila, en el orden físico del CSV, después de validar todas las filas. Este modo sirve para reproducir la máscara anterior con entradas válidas. Conserva la deduplicación de trabajos y rechaza entradas defectuosas, incluidos canales anotados sin matriz que antes se omitían silenciosamente.

## Verificación sin datasets ni entrenamiento

Se ejecutó `python -B -m unittest discover -s tests -v`. Pasaron **13 pruebas**, seis nuevas y siete de la fase 1. Las nuevas comprobaron:

- Unión de intervalos solapados, disjuntos y repetidos, incluidos el índice 0 y el último punto.
- En una matriz artificial de 8209 pasos, los intervalos P-2 documentados producen 1226 positivos en modo histórico y 1276 en modo unión. Las 50 posiciones añadidas son 5300–5349. Es una comprobación de la máscara, no de las métricas del benchmark.
- Rechazo de intervalos invertidos, fuera de límites, mal formados o con extremos fraccionarios o booleanos. El modo histórico también valida las filas que no selecciona.
- Rechazo de misiones, longitudes e IDs incompatibles o inválidos, archivos NPY ausentes y longitud real distinta de la declarada.
- Una salida por canal y cinco trabajos únicos por canal en el planificador, comprobado sin importar ni ejecutar el entrenamiento real. La preparación exclusiva de NASA no accede a SMD.

`git diff --check` no detectó defectos de espacios. Los datos y resultados históricos quedan intactos. El alcance empírico de esta fase son las pruebas artificiales. Quedan pendientes cotejar los snapshots completos y decidir si las filas públicas de P-2 son complementarias o si alguna constituye una revisión de la otra. La política implementada para filas compatibles es la unión.

## Reevaluación posterior en una copia aislada

### Entradas y dependencias

Trabajar en una copia del código ya revisado por el equipo. Tras incorporar los cambios a una revisión Git, crear dos directorios aislados, uno por modo. El siguiente ejemplo usa PowerShell y debe ejecutarse desde la raíz del proyecto original.

```powershell
git worktree add --detach ../tracin-nasa-union audit/fix-labels-p2
git worktree add --detach ../tracin-nasa-first-row audit/fix-labels-p2
```

En cada copia colocar los mismos archivos de entrada, manteniendo el orden del CSV y registrando revisión o hashes. Si se usan otros nombres de directorio, conservar la estructura relativa al proyecto:

- `data/raw/smap_msl/labeled_anomalies.csv`, etiquetas de Telemanom con columnas `chan_id`, `spacecraft`, `num_values` y `anomaly_sequences`. Para el comparador histórico importa el orden de sus filas.
- `data/raw/smap_msl/test/<chan_id>.npy`, matrices de prueba del snapshot de TranAD/Telemanom utilizado. La evaluación es transductiva y este bloque no necesita la partición pública de entrenamiento.
- Para consolidar todos los bloques, conservar en la copia los `*_eval.csv` y `*_diag.csv` de los otros bloques. Para reconstruir el artículo con los demás bloques sin recuperar sus puntuaciones, conservar también sus `halo_*.csv` y `perfil_*.json`. Si los artefactos quedan fuera de Git, copiarlos expresamente desde el archivo histórico a los directorios aislados.

Preparación y pruebas de etiquetas requieren Python, NumPy y pandas. El ejecutor requiere PyTorch con `torch.func`, scikit-learn, NumPy, pandas y SciPy. Usar PyTorch con CUDA y GPU compatible para la ejecución comparable a la original; `--device cpu` es una alternativa más lenta que conviene registrar como cambio de entorno. Las figuras del artículo requieren Matplotlib y la compilación eventual del PDF requiere el entorno LaTeX del proyecto. El README documenta el entorno original. Registrar las versiones, el hardware y el modo de etiquetas en ambos directorios.

### Preparar y evaluar

Desde la raíz de cada copia, activar el entorno y ejecutar **una sola** de estas dos líneas de preparación según el modo:

```powershell
# Copia de unión
python -B src/data_bench.py --block nasa

# Copia de primera fila
python -B src/data_bench.py --block nasa --nasa-first-row
```

La preparación escribe `data/processed/bench/nasa_<canal>.npz` con `X`, `y` y `spacecraft`, y un `data/processed/bench/resumen.csv` de NASA. Para comparar máscaras, comprobar número de canales, longitudes, identidad de las matrices `X` y posiciones donde cambia `y`. Con el snapshot histórico equivalente se esperan 81 IDs observados, sin duplicar P-2. Si los conteos cambian, revisar las fuentes antes de continuar.

En cada copia, para reproducir el protocolo completo desde matrices y etiquetas, ejecutar estas líneas. El directorio de corridas `results/final/nasa/` debe estar inicialmente vacío para una ejecución nueva y pertenecer al modo de esa copia. El ejecutor omite archivos existentes por nombre.

```powershell
$env:PYTHONUTF8 = '1'
python -B src/final/run_bench.py --blocks nasa --device cuda
python -B src/final/evaluate.py nasa --workers 4
```

El ejecutor escribe `results/final/nasa/<canal>_s<semilla>.npz`, normalmente cinco por canal, 405 con el snapshot de 81 canales. Evaluación produce `results/final/nasa_eval.csv` y `nasa_diag.csv`. Verificar corridas completas y llaves únicas antes de resumir. El número de trabajadores controla el paralelismo de la evaluación y se puede ajustar a la máquina.

**Si se recuperan las corridas NPZ históricas**, se puede evitar entrenar de nuevo. Primero confirmar que exactamente las mismas matrices y configuración generaron los scores. Copiar las corridas a la carpeta aislada y actualizar `y` para cada canal cuya máscara cambie. Actualizar también las claves diagnósticas de recorte que terminan en `_anom` y `_norm`. Esas claves se calculan con `window_labels` sobre etiquetas nuevas y con los pesos derivados del score anterior a la ronda (primero `base`, luego la ronda anterior de la rama). Las reglas están en `src/final/core.py`, `src/forecaster.py` y `src/replica/lab.py`. Los scores y pesos de ajuste no dependen de las etiquetas. La simple sustitución de `y` basta para evaluar AP, AUROC, F1 y halo, pero dejaría obsoletos los diagnósticos de recorte. Esta vía requiere preparar y verificar las copias de los NPZ antes de ejecutar `evaluate.py`. La fase actual no implementó ni ejecutó esa migración.

### Agregados y productos a regenerar

Con los seis CSV de evaluación presentes en cada copia, ejecutar el análisis de fase 1 sin margen para mantener H2 pendiente:

```powershell
New-Item -ItemType Directory -Force results/audit_phase2 | Out-Null
python -B src/final/audit_statistics.py > results/audit_phase2/statistics.json
```

Para reconstruir además los derivados con el protocolo estadístico **histórico**, ejecutar en las copias aisladas:

```powershell
python -B src/final/stats.py
if (Test-Path results/final/paper/halo_nasa.csv) {
    Rename-Item results/final/paper/halo_nasa.csv halo_nasa.pre_phase2.csv
}
if (Test-Path results/final/paper/perfil_nasa.json) {
    Rename-Item results/final/paper/perfil_nasa.json perfil_nasa.pre_phase2.json
}
python -B src/final/paper.py --no-halo --workers 4
python -B src/final/macros.py
```

En esta ruta, `stats.py` escribe tablas, H6 y `final_numbers.json`. Con `--no-halo`, `paper.py` reutiliza los cachés de otros bloques que existan y recalcula NASA cuando su caché se ha apartado. Si otro bloque carece de caché, recuperar sus pares de archivos de halo/perfil o sus NPZ antes de construir el artículo. Estas salidas mantienen la estadística histórica y deben distinguirse del JSON corregido de la fase 1.

| Artefactos que deben actualizarse | Motivo |
|---|---|
| NPZ procesados y resumen, corridas de canales con máscaras cambiadas, `nasa_eval.csv`, `nasa_diag.csv` | Fuente de etiquetas, métricas, prevalencia y diagnósticos de recorte. |
| `tables/nasa_*.md`, entrada NASA de `final_numbers.json`, estadísticas de fase 1 | Medias y pruebas recalculadas para las comparaciones que incluyen NASA. |
| `h6_series.csv`, `tables/h6.md`, resumen H6 de los JSON y `paper/figs/h6_tasa.pdf` | H6 combina NASA y SMD. Pueden cambiar prevalencia, deltas, rangos y cuartiles. |
| `paper/halo_nasa.csv`, `paper/perfil_nasa.json`, tablas y figuras de benchmarks, hipótesis, halo y diagramas CD de NASA, `paper/numeros.json` y `paper/tablas/macros.tex` | Derivados que usan etiquetas NASA directamente o agregados que las contienen. |
| Estadísticas de duración de eventos NASA, texto del artículo y PDF | Recalcular duración con las máscaras nuevas y revisar afirmaciones antes de recompilar. Las cifras de duración en el texto requieren una comprobación específica, aparte de la generación automática de tablas. |

Las métricas de los bloques ajenos a NASA pueden reutilizarse. En esta fase no se regeneró ninguno de los productos anteriores ni se estimaron métricas corregidas.
