### Sintéticos (15 datasets × 20 semillas, L4P1) — hipótesis primarias

| Hipótesis | Comparación | Métrica | Δ media [IC 95 %] | Gana / pierde | r rank-biserial | p (Wilcoxon) | p Holm | ¿Se sostiene? |
|---|---|---|---|---|---|---|---|---|
| H1 | Final > Artículo | auprc | +0.215 [+0.185, +0.245] | 237 / 62 de 300 | +0.73 | 5.7e-28 | 1.7e-27 | sí |
| H2 | Final (sin etiquetas) ≥ Artículo dinámico (con etiquetas) | f1_bin | +0.218 [+0.190, +0.244] | 245 / 40 de 300 | +0.81 | 2.5e-32 | 1.0e-31 | sí |
| H3a | Final > M3 (mismo score sin recorte) | auprc | +0.009 [+0.002, +0.015] | 227 / 69 de 300 | +0.58 | 6.7e-18 | 1.3e-17 | sí |
| H3b | Final > M1+M3 (todo menos el recorte) | f1_bin | +0.022 [+0.015, +0.028] | 170 / 69 de 300 | +0.56 | 8.6e-14 | 8.6e-14 | sí |
