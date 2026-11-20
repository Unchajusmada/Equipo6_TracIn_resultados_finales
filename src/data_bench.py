"""Benchmarks reales adicionales (extensión preregistrada, PROTOCOLO_FINAL.md).

* SMD  (Su et al., KDD 2019): 28 máquinas, 38 canales. Prueba + etiquetas de
  github.com/NetManAIOps/OmniAnomaly (ServerMachineDataset).
* SMAP / MSL (Hundman et al., KDD 2018): 82 canales de telemetría (25 / 55
  dimensiones). Series de prueba de la réplica en github.com/imperial-qore/TranAD
  (copia de telemanom/data.zip); etiquetas de khundman/telemanom
  (labeled_anomalies.csv, intervalos inclusivos).
* Fondo para la contaminación semisintética: primeros 10 000 pasos de la
  partición de entrenamiento de 10 máquinas SMD (una de cada tres).

Uso: python data_bench.py   (genera data/processed/bench/*.npz)
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "bench"
SEMI_LEN = 10_000


def smd_names() -> list[str]:
    return sorted(p.stem for p in (RAW / "smd" / "test").glob("machine-*.txt"))


def semi_machines() -> list[str]:
    return smd_names()[::3]          # fijado a priori: una de cada tres (10 máquinas)


def nasa_names() -> list[str]:
    d = pd.read_csv(RAW / "smap_msl" / "labeled_anomalies.csv")
    have = {p.stem for p in (RAW / "smap_msl" / "test").glob("*.npy")}
    return sorted(c for c in d.chan_id if c in have)


def standardize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return ((X - X.mean(0)) / sd).astype(np.float32)


def load_bench(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"{name}.npz")
    return standardize(z["X"]), z["y"].astype(np.int8)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in smd_names():
        X = np.loadtxt(RAW / "smd" / "test" / f"{m}.txt", delimiter=",")
        y = np.loadtxt(RAW / "smd" / "test_label" / f"{m}.txt").astype(np.int8)
        np.savez_compressed(OUT / f"smd_{m}.npz", X=X.astype(np.float32), y=y)
        rows.append(("smd", m, len(y), X.shape[1], y.mean()))
    lab = pd.read_csv(RAW / "smap_msl" / "labeled_anomalies.csv")
    for c in nasa_names():
        r = lab[lab.chan_id == c].iloc[0]
        X = np.load(RAW / "smap_msl" / "test" / f"{c}.npy")
        assert len(X) == int(r.num_values), (c, len(X), r.num_values)
        y = np.zeros(len(X), np.int8)
        for a, b in ast.literal_eval(r.anomaly_sequences):
            y[a:b + 1] = 1
        np.savez_compressed(OUT / f"nasa_{c}.npz", X=X.astype(np.float32), y=y, spacecraft=r.spacecraft)
        rows.append((r.spacecraft.lower(), c, len(y), X.shape[1], y.mean()))
    for m in semi_machines():
        X = np.loadtxt(RAW / "smd" / "train" / f"{m}.txt", delimiter=",")[:SEMI_LEN]
        np.savez_compressed(OUT / f"bg_{m}.npz", X=X.astype(np.float32))
    df = pd.DataFrame(rows, columns=["block", "name", "n", "C", "anom_rate"])
    df.to_csv(OUT / "resumen.csv", index=False)
    print(df.groupby("block").agg(series=("name", "size"), n_med=("n", "median"), C=("C", "median"),
                                  anom_med=("anom_rate", "median"), anom_min=("anom_rate", "min"),
                                  anom_max=("anom_rate", "max")))


# ---------------------------------------------------------------------------
# Contaminación semisintética: tipos de Lai et al. (2021) adaptados a fondos reales
# ---------------------------------------------------------------------------
SEG = 10   # longitud de los eventos colectivos (2·radius, radius = 5 como en Lai et al.)


def inject(bg: np.ndarray, mix: str, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """bg: fondo (n × C) ya estandarizado. ``ratio`` = proporción nominal total
    de instantes anómalos; cada tipo de ``mix`` recibe ratio / len(mix).
    Devuelve (X, y). El canal de cada evento se elige al azar."""
    rng = np.random.default_rng(seed)
    X = bg.astype(np.float64).copy()
    n, C = X.shape
    y = np.zeros(n, np.int8)
    lo, hi = X.min(0), X.max(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    r = ratio / len(mix)
    for t in mix:
        if t in "01":
            for i in rng.integers(0, n, round(n * r)):
                c = rng.integers(C)
                if t == "0":   # puntual global: fuera del rango del canal
                    s = rng.choice([-1, 1])
                    X[i, c] = (hi[c] if s > 0 else lo[c]) + s * rng.uniform(0.5, 1.5) * sd[c]
                else:          # puntual contextual: lejos de su vecindad, dentro del rango global
                    w = bg[max(0, i - 5):i + 5, c]
                    ls = max(w.std(), 0.1 * sd[c])
                    X[i, c] = np.clip(w.mean() + rng.choice([-1, 1]) * 2.5 * ls * rng.uniform(1, 2), lo[c], hi[c])
                y[i] = 1
        else:
            for i in rng.integers(0, n - SEG, round(n * r / SEG)):
                c = rng.integers(C)
                seg = slice(i, i + SEG)
                base = bg[seg, c].astype(np.float64)
                if t == "2":   # shapelet: onda cuadrada de amplitud 1.5 d.e.
                    k = np.arange(SEG)
                    X[seg, c] = base.mean() + 1.5 * sd[c] * np.sign(np.sin(2 * np.pi * k / 5 + 0.1))
                elif t == "3":  # estacional: patrón local acelerado ×3
                    src = bg[i:min(n, i + 3 * SEG):3, c]
                    X[i:i + len(src), c] = src
                else:          # tendencia: rampa de 0.5 d.e. por paso
                    X[seg, c] = base + rng.choice([-1, 1]) * 0.5 * sd[c] * np.arange(SEG)
                y[seg] = 1
    return X.astype(np.float32), y


if __name__ == "__main__":
    build()
