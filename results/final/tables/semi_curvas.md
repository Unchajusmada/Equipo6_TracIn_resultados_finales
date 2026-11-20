### Contaminación semisintética sobre fondo real SMD (10 máquinas × 3 mezclas × 5 niveles × 3 semillas), mezcla «01» — AUROC vs. proporción nominal

| Método | 1% | 5% | 10% | 20% | 30% | Caída |
|---|---|---|---|---|---|---|
| Artículo · umbral estático | 0.631 | 0.622 | 0.617 | 0.606 | 0.599 | 0.031 |
| M3 · proyección al objetivo | 0.695 | 0.690 | 0.693 | 0.694 | 0.700 | -0.005 |
| M2 · recorte z>2 | 0.637 | 0.628 | 0.622 | 0.610 | 0.603 | 0.034 |
| **Método final** (recorte adaptativo + proyección + IQR) | 0.708 | 0.704 | 0.707 | 0.708 | 0.714 | -0.006 |
| Error del LSTM (PD) + IQR | 0.754 | 0.752 | 0.754 | 0.759 | 0.766 | -0.012 |
| LOF + IQR | 0.797 | 0.799 | 0.800 | 0.798 | 0.761 | 0.036 |
| IForest + IQR | 0.573 | 0.577 | 0.585 | 0.598 | 0.608 | -0.036 |

### Contaminación semisintética sobre fondo real SMD (10 máquinas × 3 mezclas × 5 niveles × 3 semillas), mezcla «01234» — AUROC vs. proporción nominal

| Método | 1% | 5% | 10% | 20% | 30% | Caída |
|---|---|---|---|---|---|---|
| Artículo · umbral estático | 0.643 | 0.637 | 0.629 | 0.632 | 0.629 | 0.013 |
| M3 · proyección al objetivo | 0.660 | 0.661 | 0.656 | 0.664 | 0.667 | -0.007 |
| M2 · recorte z>2 | 0.648 | 0.641 | 0.633 | 0.633 | 0.631 | 0.017 |
| **Método final** (recorte adaptativo + proyección + IQR) | 0.672 | 0.674 | 0.671 | 0.677 | 0.681 | -0.009 |
| Error del LSTM (PD) + IQR | 0.722 | 0.722 | 0.724 | 0.728 | 0.731 | -0.009 |
| LOF + IQR | 0.781 | 0.787 | 0.785 | 0.787 | 0.779 | 0.002 |
| IForest + IQR | 0.561 | 0.565 | 0.573 | 0.579 | 0.592 | -0.031 |

### Contaminación semisintética sobre fondo real SMD (10 máquinas × 3 mezclas × 5 niveles × 3 semillas), mezcla «234» — AUROC vs. proporción nominal

| Método | 1% | 5% | 10% | 20% | 30% | Caída |
|---|---|---|---|---|---|---|
| Artículo · umbral estático | 0.650 | 0.646 | 0.653 | 0.645 | 0.653 | -0.003 |
| M3 · proyección al objetivo | 0.649 | 0.639 | 0.645 | 0.639 | 0.643 | 0.005 |
| M2 · recorte z>2 | 0.660 | 0.652 | 0.658 | 0.648 | 0.656 | 0.004 |
| **Método final** (recorte adaptativo + proyección + IQR) | 0.663 | 0.653 | 0.657 | 0.650 | 0.655 | 0.008 |
| Error del LSTM (PD) + IQR | 0.707 | 0.703 | 0.707 | 0.703 | 0.704 | 0.003 |
| LOF + IQR | 0.776 | 0.776 | 0.781 | 0.771 | 0.765 | 0.011 |
| IForest + IQR | 0.569 | 0.554 | 0.563 | 0.562 | 0.573 | -0.004 |

