# Auditoría científica de TracIn · Fase 3

19 de noviembre de 2026 · Rama `audit/check-tracin-math` desde `58c529b`

La revisión contrastó el hallazgo M-01 y el protocolo final con TracInCP. La ecuación (1) de Pruthi et al. evalúa los gradientes en el checkpoint posterior y los pondera por la tasa del intervalo que acaba de concluir. Para la misma ventana a ambos lados del producto escalar, el término se reduce a ηₖ‖∇ℓₖ‖². [Hallazgo M-01](AUDITORIA_CODEX.md#medio) · [Protocolo final](PROTOCOLO_FINAL.md) · [Pruthi et al. (2020), §§3.2–3.3 y 4.1](https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf)

| Componente | Estado | Evidencia | ¿Necesita corrección? |
|---|---|---|---|
| Última capa y escalado MSE | **correcto** | Para D = P·C, ℓ = ‖r‖²/D. Autograd confirmó ∇W = (2/D)rhᵀ y ∇b = (2/D)r, y por tanto ‖∇cabeza‖² = (4/D)ℓ(1 + ‖h‖²). Se probaron 1 y 3 canales con P = 1, y 2 canales con P = 3, en ambos LSTM. Tolerancias float64 rtol 1e−11 y atol 1e−12. | No |
| Gradiente completo | **correcto** | Por ventana y parámetro, los tensores del `vmap(grad(...))` de producción coincidieron con Autograd explícito en forma y valor, con entradas finitas y las mismas tolerancias. La norma cuadrada coincide con la suma de las contribuciones de cabeza y parámetros recurrentes, ambas positivas en el ejemplo. | No |
| Acumulación, tasa y `keep_g2` | **correcto** | Una referencia calculó las normas con Autograd en cada estado de fin de época y observó la tasa efectiva en cada paso de Adam. Los scores coinciden con su suma independiente (float32, rtol 2e−6 y atol 1e−12). `keep_g2` guarda las contribuciones ya multiplicadas por la tasa, o por 1 con `lr_weight=False`, y deja el resto de la salida idéntica. Con scheduler apagado la tasa fue constante. En la prueba del scheduler activo se redujo su paciencia a 0 mediante un parche externo para observar una bajada en tres épocas. La época de la reducción conservó su tasa y la siguiente usó la nueva. | No |
| Early stopping y época final | **correcto** | Los dos entrenadores ejecutaron y acumularon dos épocas de un máximo de tres bajo el criterio de mejora relativa. La pérdida final coincide con la del último estado efectivo, distinto del primero. No se restaura el mejor checkpoint. La salida `last_only` de `Trace` usa solo la última contribución. | No |
| Reproducibilidad y uso en `core.py` | **correcto** | Ejecuciones CPU con semilla fija y un único hilo produjeron arrays idénticos en ambos entrenadores. Con `modes=()`, `core.all_variants` propagó score base, pérdida final y número de épocas sin ejecutar rondas. | No |
| Checkpoints por época y Adam | **aproximación metodológica** | El score aplica TracInCP por ventana en estados de fin de época. Ese estado aproxima los estados de los minibatches intermedios. Los momentos y el escalado adaptativo de Adam intervienen en el desplazamiento real de los parámetros. La tasa escalar usada en el score pondera el checkpoint. | No |
| Pérdida sin ponderar en el score | **aproximación metodológica** | Una ventana con peso cero conservó score positivo con otras ventanas de peso positivo. El score mide la sensibilidad de su propia MSE en los checkpoints obtenidos de la pérdida ponderada. Su término directo en el gradiente de entrenamiento vale cero. | No |
| Monitor con suma de pesos cero | **correcto** | La auditoría inicial reprodujo un `NaN` por división entre cero en ambos entrenadores. Ahora se comprueba la suma de los pesos del conjunto monitoreado antes de crear el optimizador. Ambos lanzan `ValueError` si la suma es cero. La prueba de `lab.py` cubre también validación de peso cero con entrenamiento de peso positivo. | No. Corrección aplicada y verificada. |
| Validez del score en datos reales | **pendiente de validación experimental** | Las pruebas usan ventanas artificiales. Queda por establecer cuánto aproxima el score el efecto de una ventana sobre la trayectoria y qué evidencia aporta en datasets reales. | Por determinar con evidencia experimental |

## Verificación

`python -B -m unittest discover -s tests -v` ejecutó **21 pruebas aprobadas en CPU** con PyTorch 2.14.0. Las dos pruebas del monitor ahora exigen un `ValueError` antes de crear Adam. La prueba de checkpoints incluye lotes individuales de peso cero con suma positiva en el conjunto monitoreado.

En 11 ejecuciones válidas sobre una serie artificial de seis instantes, los SHA-256 de las salidas antes y después coincidieron bit a bit. La comparación incluyó pérdidas, puntuaciones, contribuciones por época y las tasas y estadísticas de checkpoint que devuelve `Trace`. Las corridas duraron entre una y tres épocas. Las fórmulas de gradiente y de acumulación se conservaron, junto con M1–M3, H2, P-2, el protocolo y los 109 archivos históricos. No se descargaron datasets.

## Conclusión

**La fase 3 queda cerrada en código para el alcance matemático auditado.** Se verificaron los gradientes, la acumulación TracInCP y el rechazo explícito cuando el conjunto monitoreado suma peso cero. La validación experimental del score en datos reales sigue pendiente.
