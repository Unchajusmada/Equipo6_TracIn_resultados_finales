### SMAP + MSL (82 canales × 5 semillas, L4P1; unidad de prueba = canal) — hipótesis primarias

| Hipótesis | Comparación | Métrica | Δ media [IC 95 %] | Gana / pierde | r rank-biserial | p (Wilcoxon) | p Holm | ¿Se sostiene? |
|---|---|---|---|---|---|---|---|---|
| H1 | Final > Artículo | auprc | -0.007 [-0.025, +0.010] | 43 / 38 de 81 | -0.01 | 0.946 | 0.946 | no significativa |
| H2 | Final (sin etiquetas) ≥ Artículo dinámico (con etiquetas) | f1_bin | -0.071 [-0.115, -0.029] | 29 / 47 de 81 | -0.41 | 0.002 | 0.010 | no (peor) |
| H3a | Final > M3 (mismo score sin recorte) | auprc | +0.015 [+0.005, +0.025] | 55 / 26 de 81 | +0.44 | 6.5e-04 | 0.004 | sí |
| H3b | Final > M1+M3 (todo menos el recorte) | f1_bin | +0.011 [+0.000, +0.023] | 41 / 32 de 81 | +0.24 | 0.081 | 0.243 | no significativa |
| H5a | M3 (sólo proyección al objetivo) > Artículo | auroc | -0.004 [-0.012, +0.004] | 37 / 44 de 81 | -0.12 | 0.357 | 0.715 | no significativa |
| H5b | M3 (sólo proyección al objetivo) > Artículo | auprc | -0.021 [-0.035, -0.010] | 31 / 50 de 81 | -0.37 | 0.003 | 0.014 | no (peor) |
