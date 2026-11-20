### Contaminación controlada (3 mezclas × 7 niveles × 20 semillas, L4P1), mezcla «01» — AUROC vs. proporción nominal

| Método | 1% | 5% | 10% | 15% | 20% | 25% | 30% | Caída |
|---|---|---|---|---|---|---|---|---|
| Artículo · umbral estático | 0.916 | 0.870 | 0.817 | 0.780 | 0.753 | 0.728 | 0.711 | 0.205 |
| M3 · proyección al objetivo | 0.953 | 0.958 | 0.954 | 0.952 | 0.945 | 0.938 | 0.931 | 0.022 |
| M2 · recorte z>2 | 0.915 | 0.870 | 0.818 | 0.782 | 0.754 | 0.731 | 0.714 | 0.201 |
| **Método final** (recorte adaptativo + proyección + IQR) | 0.953 | 0.959 | 0.956 | 0.958 | 0.956 | 0.955 | 0.954 | -0.002 |
| Error del LSTM (PD) + IQR | 0.955 | 0.958 | 0.955 | 0.950 | 0.944 | 0.934 | 0.928 | 0.027 |
| LOF + IQR | 0.955 | 0.954 | 0.826 | 0.731 | 0.692 | 0.672 | 0.656 | 0.300 |
| IForest + IQR | 0.922 | 0.927 | 0.924 | 0.922 | 0.922 | 0.915 | 0.910 | 0.012 |

### Contaminación controlada (3 mezclas × 7 niveles × 20 semillas, L4P1), mezcla «01234» — AUROC vs. proporción nominal

| Método | 1% | 5% | 10% | 15% | 20% | 25% | 30% | Caída |
|---|---|---|---|---|---|---|---|---|
| Artículo · umbral estático | 0.957 | 0.931 | 0.891 | 0.857 | 0.830 | 0.809 | 0.789 | 0.167 |
| M3 · proyección al objetivo | 0.943 | 0.939 | 0.923 | 0.902 | 0.895 | 0.880 | 0.873 | 0.070 |
| M2 · recorte z>2 | 0.955 | 0.937 | 0.911 | 0.879 | 0.854 | 0.833 | 0.813 | 0.142 |
| **Método final** (recorte adaptativo + proyección + IQR) | 0.935 | 0.945 | 0.946 | 0.931 | 0.927 | 0.913 | 0.904 | 0.031 |
| Error del LSTM (PD) + IQR | 0.880 | 0.881 | 0.864 | 0.853 | 0.841 | 0.831 | 0.824 | 0.055 |
| LOF + IQR | 0.895 | 0.903 | 0.902 | 0.884 | 0.873 | 0.866 | 0.853 | 0.042 |
| IForest + IQR | 0.814 | 0.782 | 0.782 | 0.787 | 0.775 | 0.776 | 0.764 | 0.050 |

### Contaminación controlada (3 mezclas × 7 niveles × 20 semillas, L4P1), mezcla «234» — AUROC vs. proporción nominal

| Método | 1% | 5% | 10% | 15% | 20% | 25% | 30% | Caída |
|---|---|---|---|---|---|---|---|---|
| Artículo · umbral estático | 0.989 | 0.952 | 0.924 | 0.892 | 0.879 | 0.862 | 0.851 | 0.138 |
| M3 · proyección al objetivo | 0.944 | 0.897 | 0.862 | 0.826 | 0.814 | 0.798 | 0.788 | 0.156 |
| M2 · recorte z>2 | 0.983 | 0.976 | 0.959 | 0.928 | 0.911 | 0.893 | 0.877 | 0.106 |
| **Método final** (recorte adaptativo + proyección + IQR) | 0.939 | 0.937 | 0.921 | 0.887 | 0.872 | 0.853 | 0.841 | 0.098 |
| Error del LSTM (PD) + IQR | 0.864 | 0.784 | 0.754 | 0.733 | 0.717 | 0.710 | 0.700 | 0.164 |
| LOF + IQR | 0.888 | 0.853 | 0.849 | 0.808 | 0.768 | 0.758 | 0.732 | 0.156 |
| IForest + IQR | 0.756 | 0.732 | 0.709 | 0.697 | 0.711 | 0.702 | 0.709 | 0.047 |

