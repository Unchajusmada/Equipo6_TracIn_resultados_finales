### Contaminación semisintética sobre fondo real SMD (10 máquinas × 3 mezclas × 5 niveles × 3 semillas) — hipótesis primarias

| Hipótesis | Comparación | Métrica | Δ media [IC 95 %] | Gana / pierde | r rank-biserial | p (Wilcoxon) | p Holm | ¿Se sostiene? |
|---|---|---|---|---|---|---|---|---|
| H1 | Final > Artículo | auprc | +0.040 [+0.033, +0.047] | 296 / 154 de 450 | +0.48 | 2.3e-18 | 1.2e-17 | sí |
| H2 | Final (sin etiquetas) ≥ Artículo dinámico (con etiquetas) | f1_bin | -0.005 [-0.012, +0.001] | 238 / 211 de 450 | -0.03 | 0.557 | 0.557 | no significativa |
| H3a | Final > M3 (mismo score sin recorte) | auprc | -0.008 [-0.010, -0.006] | 159 / 291 de 450 | -0.37 | 1.9e-11 | 7.8e-11 | no (peor) |
| H3b | Final > M1+M3 (todo menos el recorte) | f1_bin | +0.008 [+0.006, +0.009] | 311 / 137 de 450 | +0.51 | 3.8e-21 | 2.3e-20 | sí |
| H5a | M3 (sólo proyección al objetivo) > Artículo | auroc | +0.033 [+0.030, +0.037] | 338 / 112 de 450 | +0.76 | 5.9e-44 | 4.7e-43 | sí |
| H5b | M3 (sólo proyección al objetivo) > Artículo | auprc | +0.048 [+0.041, +0.055] | 308 / 142 de 450 | +0.64 | 6.3e-32 | 4.4e-31 | sí |
| H4r | Caída de AUROC 1 %→30 % menor en Final que en Artículo | auroc_drop | +0.017 [+0.010, +0.023] | 60 / 30 de 90 | +0.53 | 1.1e-05 | 2.2e-05 | sí |
| H5c | Caída de AUROC 1 %→30 % menor en M3 que en Artículo | auroc_drop | +0.016 [+0.010, +0.022] | 65 / 25 de 90 | +0.59 | 1.3e-06 | 3.9e-06 | sí |
