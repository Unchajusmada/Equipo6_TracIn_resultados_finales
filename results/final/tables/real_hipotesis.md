### Reales (4 datasets × semillas, L4P1) — hipótesis primarias

| Hipótesis | Comparación | Métrica | Δ media [IC 95 %] | Gana / pierde | r rank-biserial | p (Wilcoxon) | p Holm | ¿Se sostiene? |
|---|---|---|---|---|---|---|---|---|
| H1 | Final > Artículo | auprc | +0.046 [+0.036, +0.056] | 40 / 0 de 40 | +1.00 | 1.8e-12 | 7.3e-12 | sí |
| H2 | Final (sin etiquetas) ≥ Artículo dinámico (con etiquetas) | f1_bin | -0.130 [-0.160, -0.098] | 10 / 30 de 40 | -0.87 | 1.0e-07 | 3.1e-07 | no (peor) |
| H3a | Final > M3 (mismo score sin recorte) | auprc | +0.020 [+0.005, +0.035] | 23 / 17 de 40 | +0.15 | 0.428 | 0.856 | no significativa |
| H3b | Final > M1+M3 (todo menos el recorte) | f1_bin | +0.002 [-0.001, +0.006] | 20 / 20 de 40 | -0.10 | 0.599 | 0.856 | no significativa |
