# Experimentos — Segunda entrega (Equipo 6)

Reproducción y extensión de **Verdone, Scardapane & Panella (2026)**, *A data
attribution approach for unsupervised anomaly detection on multivariate time
series*, Expert Systems with Applications 297, 129285.

Todo el código está en `src/`; los resultados se escriben en `results/`, las
figuras en `figures/` y la presentación se arma con `deck/build_deck.py`.

## Entorno

```bash
python3 -m venv --system-site-packages .venv      # usa torch/sklearn del sistema
.venv/bin/pip install pandas matplotlib stumpy pyarrow fonttools brotli
```

Versiones usadas: Python 3.13, PyTorch 2.10 (CUDA 12.8), scikit-learn 1.8,
NumPy 2.4, SciPy 1.17, pandas 3.0, stumpy 1.14. GPU: RTX 5090 (sólo para los
datasets reales; los sintéticos corren en CPU, 1 hilo por proceso).

## Datos

| Dataset | Fuente | Notas |
|---|---|---|
| 15 sintéticos | `src/synthetic.py` | Copia fiel del generador oficial de Lai et al. (2021), repo `datamllab/tods`, rama `benchmark`. 400 pasos, 5 canales. La semilla fija la realización. |
| Credit Card | réplica de TensorFlow del CSV de Kaggle/ULB | OpenML (fuente del script original) no respondía. Mismo archivo: 284 807 filas. |
| CICIDS | `GeneratedLabelledFlows.zip` (réplica del ZIP de UNB en Hugging Face, `bencorn/CICIDS2017`) | Archivo *Thursday-WorkingHours-Morning-WebAttacks*. |
| GECCO | Zenodo 3884398 | |
| SWAN-SF | `petdataset_01.zip` (Bitbucket gsudmlab) | |

```bash
mkdir -p data/raw/{gecco,creditcard,cicids,swan_sf}
curl -L -o data/raw/gecco/gecco.csv "https://zenodo.org/records/3884398/files/1_gecco2018_water_quality.csv?download=1"
curl -L -o data/raw/creditcard/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
curl -L -o data/raw/swan_sf/petdataset_01.zip https://bitbucket.org/gsudmlab/mvtsdata_toolkit/downloads/petdataset_01.zip
(cd data/raw/swan_sf && unzip petdataset_01.zip)
curl -L -o data/raw/cicids/GeneratedLabelledFlows.zip https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/GeneratedLabelledFlows.zip
.venv/bin/python src/data_real.py       # -> data/processed/*.npz
```

El preprocesamiento replica los scripts oficiales del benchmark. Longitudes y
proporción de anomalías coinciden exactamente con la Tabla 1 del artículo
(284 807 / 0.173 %, 170 231 / 1.281 %, 138 521 / 1.246 %, 120 000 / 23.80 %).

## Ejecución

```bash
cd src
python forecaster.py                 # verifica la forma cerrada de TracIn contra autograd
python baselines.py                  # verifica LOF-GPU vs sklearn y MP-GPU vs stumpy
python run_synthetic.py --seeds 20   # Exp. 1: 15 sintéticos x 20 semillas
python run_real.py --phase a         # Exp. 2a: reales, forecasters + métodos de referencia
python run_real.py --phase b         # Exp. 2b: reales, reponderación
python run_contamination.py          # Exp. 3: contaminación controlada
python repro_gecco.py                # diagnóstico de sensibilidad en GECCO
python diag_lof_k.py                 # diagnóstico: LOF vs número de vecinos k (reales)
python evaluate.py synthetic && python evaluate.py real && python evaluate.py contamination
python report.py all                 # tablas en results/tables/ y results/deck_data.json
python figures.py                    # figuras en figures/
python write_results.py              # ../RESULTADOS.md (informe con tablas y figuras)
python ../deck/build_deck.py         # presentación (../../Presentacion_2_Equipo6.html)
```

Todos los ejecutores son reanudables (omiten trabajos cuyo archivo ya existe).

## Qué se implementó

**Método del artículo (`src/forecaster.py`).** LSTM (1 capa, 64 unidades) que
recibe L pasos y predice los P siguientes de todos los canales; MSE, Adam
lr = 1e-3, ReduceLROnPlateau (factor 0.9, paciencia 3), early stopping
(paciencia 10). Un checkpoint por época. Self-influence de TracIn sobre la
última capa lineal, en forma cerrada y exacta:

    s(x) = Σ_k η_k · (4/D) · ℓ_k(x) · (1 + ‖h_k(x)‖²),   D = P·C

Proyección a instantes (Ec. 4), normalización por el máximo (Ec. 5) y umbral
estático o dinámico (Ec. 6). Barrido L ∈ {4, 8, 16}, P ∈ {1, 2, 4}. También
con un Transformer (2 capas, d = 64) en los datos reales.

**Métodos de referencia (`src/baselines.py`).** Isolation Forest, One-Class
SVM, LOF, Autoencoder (hiperparámetros por defecto de PyOD, como el benchmark
de Lai et al.), Matrix Profile y el error de predicción del mismo LSTM. Con
w = 1 (puntual) y w ∈ {4, 8, 16} (ventanas, las variantes "-DA").

**Propuesta 1, umbrales sin etiquetas (`src/common.py`).** IQR (k = 1.5),
MAD (k = 3), ambos también en escala logarítmica, y EVT-POT (q = 1e-3,
cuantil inicial 0.98). Hiperparámetros fijados a priori.

**Propuesta 2, reponderación iterativa (`src/forecaster.py`).** 3 rondas; los
pesos se calculan del z robusto del log de la self-influence de la ronda
anterior (umbral z0 = 2). Variante suave (peso mínimo 0.05) y recorte duro.

## Protocolo de evaluación (idéntico para todos los métodos)

* Escenario transductivo: cada método se ajusta sobre la serie completa
  (con anomalías y sin etiquetas), como en el artículo.
* Se evalúan los mismos instantes para todos: se excluyen los 19 primeros y
  últimos (el mayor span del barrido).
* Métricas independientes del umbral: AUPRC, AUROC, F1 máximo (oráculo).
* Umbral estático del artículo (τ = 0.85 sintéticos / 0.87 reales). El texto
  dice "percentil", pero sus cifras sólo son alcanzables si τ es fracción del
  máximo (con el percentil 87 y 1.25 % de anomalías la precisión no puede
  pasar de ≈ 0.096, y reportan 0.539 en GECCO); se calculan ambas lecturas.
* Umbral dinámico del artículo (usa etiquetas): umbral F1-óptimo en una mitad
  de la serie aplicado a la otra (validación cruzada en 2 bloques contiguos).
* Umbrales sin etiquetas (propuesta 1).
* "Estilo artículo": mejor configuración por dataset elegida con la métrica
  de prueba, sólo para comparar con sus tablas.
* Pruebas de Wilcoxon pareadas sobre (dataset, semilla).

## Réplica del artículo (`src/replica/`)

El primer experimento (arriba) no reproducía las cifras de TracIn. La carpeta
`src/replica/` busca, de forma controlada, qué supuestos no declarados las
reproducen, y vuelve a probar nuestras mejoras **sobre la configuración que
sí reproduce**. No modifica el código ni los resultados del primer
experimento; escribe en `results/replica/`.

| Archivo | Qué hace |
|---|---|
| `lab.py` | LSTM escrito a mano (verificado contra `nn.LSTM`, error 4e-8) y entrenamiento configurable (`Cfg`): normalización, lr, batch, épocas, paciencia, validación, tamaño, gradiente de la última capa (forma cerrada) o de **todos** los parámetros (Ec. 2 completa, con `torch.func.vmap(grad)`). |
| `protocol.py` | Protocolo del artículo en todas sus lecturas: umbral estático como percentil o fracción del máximo; F1 binario o macro; umbral dinámico elegido en la primera mitad y medido en la segunda. |
| `screen.py`, `summarize.py` | Barrido de 13 variantes de lo no especificado (15 sintéticos × 5 semillas × 9 (L, P)). |
| `rep_hypotheses.py` | Contrasta lecturas del F1 (binario, macro, point-adjust, ponderado) contra la Tabla 2. |
| `real_screen.py` | Barrido en reales (GPU); `--leak` prueba la hipótesis de fuga de la etiqueta. |
| `rep_synth.py` | Réplica final en sintéticos: 15 × 20 semillas × 9 (L, P), TracIn + proyección al objetivo + error final + reponderación (2 rondas, suave y recorte). |
| `rep_baselines.py` | Métodos de referencia sobre los mismos sintéticos estandarizados. |
| `rep_real.py` | Réplica en reales con gradiente completo (3 semillas, hasta 9 (L, P)) + mejoras. |
| `rep_contam.py` | Contaminación controlada con la configuración reproducida. |
| `rep_contam_fix.py` | Variantes para frenar la degradación con la contaminación: recorte adaptativo (umbral IQR), épocas tempranas, LSTM de 16. |
| `rep_eval.py`, `rep_report.py`, `rep_figures.py` | Evaluación, tablas (`results/replica/tables/`, `results/replica/replica_numbers.json`) y figuras (`figures/fig_replica_*.png`). |
| `h_*.py` | Pruebas rápidas de hipótesis de evaluación sobre el primer experimento (sólo imprimen). |

```bash
cd src/replica
python screen.py --variants base,full,lr1e-2,b8,ep300,ep30,h128,h16,nosched,nolrw,val20,minmax,nonorm --seeds 5
python rep_synth.py --variant base --seeds 20 --workers 21
python rep_baselines.py --seeds 20
python rep_contam.py --seeds 10
python rep_contam_fix.py --seeds 10
python rep_real.py --datasets creditcard --variant full --cfgs 4x1,8x2,16x4,4x2,8x1,4x4,16x1,8x4,16x2 --seeds 3   # ídem cicids, gecco, swan_sf
python real_screen.py --variants base,full,lr1e-2,val20,minmax,nolrw --datasets gecco
python real_screen.py --variants base --leak --out real_leak.csv
python rep_eval.py --subdir synth_base --baselines synth_baselines
python rep_hypotheses.py
python rep_report.py
python rep_figures.py
cd .. && python write_results.py && python ../deck/build_deck.py
cd .. && python report/build_report.py   # reporte PDF para el equipo (report/)
python colab/build_notebook.py         # notebook autocontenido para Colab (colab/)
```

**Configuración que reproduce los sintéticos** (variante `base`): canales
estandarizados (z-score), lr 1e-3, batch 32, ≤ 100 épocas con paciencia 10
sobre la pérdida de entrenamiento, ReduceLROnPlateau, LSTM de 64, sin
partición de validación; F1 **macro**. En reales se usa además el gradiente
de todos los parámetros (variante `full`), batch 256 y ≤ 50 épocas.
