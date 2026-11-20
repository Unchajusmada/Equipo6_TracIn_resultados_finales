# Protocolo de la evaluación final (fijado antes de correr)

Fecha: 2026-11-12. Este documento se escribió **antes** de lanzar los
experimentos de `src/final/`. Ningún hiperparámetro, umbral ni comparación se
eligió mirando los resultados de esta evaluación. Si algo cambia después, se
anota al final en "Desviaciones".

## Pregunta

¿Las tres mejoras, por separado y combinadas, mejoran de forma significativa al
método del artículo (TracIn, Verdone et al., 2026) en la configuración que lo
reproduce? En particular: ¿la combinación de las tres supera a cada mejora sola?

## Métodos

Todos usan el mismo LSTM y la misma configuración reproducida (`lab.py`):
sintéticos con gradiente de la última capa (`base`), reales con gradiente de
todos los parámetros (`full`, batch 256, 50 épocas máx.). Datos estandarizados.

Tres piezas independientes:

| Pieza | Opciones |
|---|---|
| Entrenamiento (mejora 2) | `base` (artículo) · `soft` z>2, 2 rondas · `trim` z>2, 2 rondas · `adapt` recorte con umbral IQR, 3 rondas (mejoras 1+2) |
| Proyección (mejora 3) | `win` Ec. 4 del artículo (suma sobre la ventana) · `tgt` sólo pasos predichos (promedio) |
| Umbral (mejora 1) | `static` τ del artículo (percentil 85 / 87) · `dyn` dinámico del artículo (USA etiquetas) · `iqr` Q3+1.5·IQR sin etiquetas |

Métodos con nombre:

* **Artículo**: `base + win`, con umbral `dyn` (su mejor protocolo, usa etiquetas) y con `static`.
* **M1**: `base + win + iqr`.
* **M2**: `soft`/`trim` + `win` + `static`.
* **M3**: `base + tgt` + `static`.
* **Método final (M1+M2+M3)**: `adapt (3 rondas) + tgt + iqr`. L = 4, P = 1 fijos. **No usa ninguna etiqueta.**
* Ablaciones del final: sin M3 (`adapt + win + iqr`), sin recorte (`base + tgt + iqr`), recorte z>2 en vez de adaptativo (`trim + tgt + iqr`).
* Contexto: error final del mismo LSTM (PD, `tgt`), LOF, IForest.

Hiperparámetros de las mejoras (ya fijados en entregas anteriores): z0 = 2,
peso mínimo 0.05, IQR k = 1.5, 2 rondas (z>2) y 3 rondas (adaptativo).

## Datos

1. **Sintéticos**: 15 datasets de Lai et al. (2021), 400 pasos, 20 semillas.
   Las 9 configuraciones (L, P) para poder aplicar también el protocolo del
   artículo (mejor (L, P) elegida con etiquetas).
2. **Contaminación controlada**: 4000 pasos, proporción nominal r ∈ {1, 5, 10,
   15, 20, 25, 30} %, tres mezclas de anomalías (`01234` todas, `01` puntuales,
   `234` de patrón), 20 semillas, L4P1.
3. **Reales**: Credit Card, CICIDS, GECCO, SWAN-SF, L4P1, 10 semillas.

## Evaluación

Mismos instantes para todos (se excluyen 19 en cada extremo). Umbrales `dyn` e
`iqr` se miden en la segunda mitad de la serie (el dinámico elige su umbral en
la primera mitad con etiquetas; el IQR se calcula sobre toda la serie sin
etiquetas). `static` se mide en la serie completa y en la segunda mitad.

* Calidad del score (sin umbral): **AUPRC** (primaria), AUROC.
* Detección (con umbral): **F1 binario** de la clase anómala (primaria), F1 macro.

## Hipótesis primarias (Wilcoxon pareado sobre (dataset, semilla), Holm)

* **H1** Método final > Artículo en AUPRC.
* **H2** Método final (sin etiquetas) ≥ Artículo con umbral dinámico (con etiquetas) en F1 binario.
* **H3** La combinación aporta: Método final > M3 sola (`base + tgt + iqr`) en AUPRC y en F1 binario.
* **H4** Contaminación: la caída de AUROC de 1 % a 30 % es menor en el Método final que en el Artículo.

Cada hipótesis se prueba por separado en sintéticos, contaminación y reales.
Corrección de Holm sobre la familia de pruebas primarias de cada bloque de
datos. Se reportan además: diferencia media, IC 95 % bootstrap (10 000
remuestreos por (dataset, semilla)), tamaño de efecto (rank-biserial) y
número de victorias. En reales hay sólo 4 datasets: la generalización entre
datasets se juzga por el signo en cada uno, no por el p-valor agregado.

Análisis secundario: diseño factorial completo
{base, soft, trim, adapt} × {win, tgt} × {static, dyn, iqr}.

## Desviaciones

1. **2026-11-12/17, interrupción.** La corrida se interrumpió (reinicio del
   equipo) con 121/300 sintéticos, 0/420 de contaminación y 4/40 reales; se
   reanudó el 2026-11-17 con el mismo código y las mismas semillas (los
   archivos ya terminados se conservan; cada archivo es determinista por
   (dataset, semilla)). Las tablas parciales de sintéticos (47 pares) se vieron
   antes de reanudar; no se cambió nada del protocolo por ello.
2. **Análisis secundarios agregados para el artículo (2026-11-17)**, escritos
   en `src/final/paper.py` después de ver las tablas parciales de sintéticos.
   No sustituyen a H1–H4 y se reportan como exploratorios:
   * Friedman + rango medio + diferencia crítica de Nemenyi (α = 0.05) sobre
     datasets (sintéticos: N = 15 medias por dataset; contaminación: N = 21
     celdas mezcla × nivel; reales: N = 40 pares (dataset, semilla), sólo
     descriptivo).
   * Pruebas centradas en la mejora 3: M3 − Artículo y Final − Artículo en
     cada celda (mezcla, nivel) de contaminación, Holm sobre las 21 celdas;
     caída de AUROC 1 %→30 % de cada método frente a la del artículo (Holm
     sobre los métodos); en reales, Wilcoxon exacto por dataset sobre las 10
     semillas.
   * Mecanismo («halo»): AUROC de anomalías contra normales a distancia
     1..L+P−1 de una anomalía, fracción del top-k ocupada por esas normales y
     perfil medio del score alrededor del inicio de cada evento. Usa
     etiquetas sólo para analizar, nunca en un método.

## Extensión preregistrada (2026-11-17, escrita antes de descargar los datos nuevos)

Motivo: en reales sólo hay 4 datasets, así que la generalización entre
datasets no se puede probar con un p-valor. Se agregan tres bloques. Métodos,
hiperparámetros y configuración del modelo **idénticos** al bloque de reales
(gradiente de todos los parámetros, batch 256, 50 épocas máx., L4P1, τ
estático = percentil 87). Se agregan LOF e IForest como referencia (igual
que en contaminación). Nada de esto se ajusta mirando resultados.

1. **SMD** (Server Machine Dataset, Su et al., KDD 2019): las 28 máquinas,
   partición de prueba (con etiquetas), 5 semillas. Escenario transductivo
   como en el artículo (se entrena sobre la misma serie que se puntúa).
2. **SMAP + MSL** (Hundman et al., KDD 2018): los 82 canales (55 + 27),
   partición de prueba, 5 semillas. Réplica de los datos en
   github.com/imperial-qore/TranAD (el enlace S3 original ya no responde).
3. **Contaminación semisintética sobre fondo real («semi»)**: fondo = primeros
   10 000 pasos de la partición de entrenamiento de 10 máquinas SMD
   (una de cada tres en orden alfabético, fijado a priori), estandarizado por
   canal. Se inyectan anomalías de los cinco tipos de Lai et al. adaptadas a
   fondos reales (canal elegido al azar por evento): 0 puntual global (fuera
   del rango del canal), 1 puntual contextual (±2.5 d.e. local, dentro del
   rango global), 2 shapelet (segmento de 10 pasos reemplazado por una onda
   cuadrada), 3 estacional (segmento de 10 pasos con el patrón local
   acelerado ×3), 4 tendencia (rampa de 0.5 d.e. por paso dentro de un
   segmento de 10). Mezclas `01234`, `01`, `234`; proporción nominal r ∈
   {1, 5, 10, 20, 30} %; 3 semillas. Limitación: el fondo de entrenamiento de
   SMD no tiene etiquetas y puede contener anomalías propias no marcadas
   (afecta igual a todos los métodos).

### Hipótesis adicionales

Unidad de análisis en SMD y SMAP/MSL: la **serie** (promedio sobre semillas),
así la prueba es entre datasets, no entre semillas. Holm dentro de cada bloque.

* **H1–H3** igual que arriba, en SMD y en SMAP/MSL.
* **H4r** (semi): la caída de AUROC de 1 % a 30 % es menor en el Método final
  que en el Artículo (unidad: máquina × mezcla × semilla).
* **H5** (mejora 3 sola): M3 > Artículo en AUROC y en AUPRC, en cada bloque.
* **H5c** (semi): la caída de AUROC de M3 es menor que la del Artículo.
* **H6** (la ventaja de M3 crece con la contaminación natural): correlación
  de Spearman entre la tasa de anomalías de cada serie y Δ AUROC (M3 −
  Artículo), unilateral ρ > 0, sobre las 110 series de SMD + SMAP/MSL.

## Desviaciones de la extensión (2026-11-18)

3. **SMAP+MSL: 81 canales, no 82.** El canal P-2 aparece dos veces en
   `labeled_anomalies.csv` (dos intervalos distintos). `nasa_names()` lo
   devolvía duplicado y las corridas se identifican por nombre, así que se
   evaluaron 81 canales únicos × 5 semillas = 405 corridas, usando las
   etiquetas de la primera fila de P-2. No se corrigió a la vista de
   resultados; sólo se documenta.
4. **Análisis descriptivo agregado después de ver los resultados de SMD y
   SMAP/MSL:** largo de los eventos anómalos (fracción de instantes anómalos en
   eventos ≥ 10 pasos y mediana ponderada por instante) para explicar el
   resultado negativo de M3. No cambia ninguna prueba ni hipótesis.
