### Contaminación controlada (3 mezclas × 7 niveles × 20 semillas, L4P1) — hipótesis primarias

| Hipótesis | Comparación | Métrica | Δ media [IC 95 %] | Gana / pierde | r rank-biserial | p (Wilcoxon) | p Holm | ¿Se sostiene? |
|---|---|---|---|---|---|---|---|---|
| H1 | Final > Artículo | auprc | +0.328 [+0.306, +0.350] | 402 / 18 de 420 | +0.93 | 7.6e-62 | 3.8e-61 | sí |
| H2 | Final (sin etiquetas) ≥ Artículo dinámico (con etiquetas) | f1_bin | +0.167 [+0.147, +0.188] | 306 / 111 de 420 | +0.72 | 9.2e-37 | 1.8e-36 | sí |
| H3a | Final > M3 (mismo score sin recorte) | auprc | +0.045 [+0.035, +0.055] | 390 / 30 de 420 | +0.87 | 1.1e-53 | 4.3e-53 | sí |
| H3b | Final > M1+M3 (todo menos el recorte) | f1_bin | +0.047 [+0.042, +0.051] | 351 / 63 de 420 | +0.86 | 6.7e-52 | 2.0e-51 | sí |
| H4 | Caída de AUROC 1 %→30 % menor en Final que en Artículo | auroc_drop | +0.127 [+0.108, +0.146] | 57 / 3 de 60 | +0.98 | 3.5e-11 | 3.5e-11 | sí |
