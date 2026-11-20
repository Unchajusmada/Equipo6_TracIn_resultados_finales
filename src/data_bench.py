"""Benchmarks reales adicionales (extensión preregistrada, PROTOCOLO_FINAL.md).

* SMD  (Su et al., KDD 2019): 28 máquinas, 38 canales. Prueba + etiquetas de
  github.com/NetManAIOps/OmniAnomaly (ServerMachineDataset).
* SMAP / MSL (Hundman et al., KDD 2018): canales de telemetría (25 / 55
  dimensiones). Series de prueba de la réplica en github.com/imperial-qore/TranAD
  (copia de telemanom/data.zip); etiquetas de khundman/telemanom
  (labeled_anomalies.csv, intervalos inclusivos).
* Fondo para la contaminación semisintética: primeros 10 000 pasos de la
  partición de entrenamiento de 10 máquinas SMD (una de cada tres).

Uso: python data_bench.py [--block nasa] [--nasa-first-row]
"""
from __future__ import annotations

import argparse
import ast
import re
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


def nasa_annotations():
    """Agrupa filas validadas por el identificador usado para nombrar la serie."""
    root = RAW / "smap_msl"
    labels = pd.read_csv(root / "labeled_anomalies.csv", dtype=str, keep_default_na=False)
    columns = ["chan_id", "spacecraft", "num_values", "anomaly_sequences"]
    if labels.empty or not set(columns).issubset(labels.columns):
        raise ValueError("NASA: CSV vacío o faltan columnas de anotaciones")
    channels = {}
    for row, (channel, mission, size, encoded) in enumerate(
        labels[columns].itertuples(index=False, name=None), start=1
    ):
        context = f"NASA fila de datos {row} ({channel!r})"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", channel):
            raise ValueError(f"{context}: chan_id inválido")
        if mission not in {"SMAP", "MSL"} or not re.fullmatch(r"[0-9]+", size) or int(size) == 0:
            raise ValueError(f"{context}: spacecraft o num_values inválido")
        length = int(size)
        try:
            intervals = ast.literal_eval(encoded)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"{context}: anomaly_sequences inválido") from exc
        if not isinstance(intervals, (list, tuple)):
            raise ValueError(f"{context}: se esperaba una secuencia de intervalos")
        for interval in intervals:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(type(index) is not int for index in interval)
                    or not 0 <= interval[0] <= interval[1] < length):
                raise ValueError(f"{context}: intervalo inclusivo fuera de especificación: {interval!r}")
        previous_mission, previous_length, rows = channels.setdefault(channel, (mission, length, []))
        if (previous_mission, previous_length) != (mission, length):
            raise ValueError(f"{context}: misión o longitud incompatible con otra fila del canal")
        rows.append(intervals)
    missing = sorted(c for c in channels if not (root / "test" / f"{c}.npy").is_file())
    if missing:
        raise FileNotFoundError(f"NASA: faltan series .npy anotadas: {', '.join(missing)}")
    return channels


def nasa_names() -> list[str]:
    return sorted(nasa_annotations())


def build_nasa(*, first_row: bool = False):
    channels = nasa_annotations()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for channel, (mission, length, intervals_by_row) in sorted(channels.items()):
        X = np.load(RAW / "smap_msl" / "test" / f"{channel}.npy")
        if X.ndim != 2 or X.shape[0] != length or X.shape[1] == 0:
            raise ValueError(f"NASA {channel}: se esperaba una matriz con {length} pasos, forma {X.shape}")
        y = np.zeros(length, np.int8)
        selected = intervals_by_row[:1] if first_row else intervals_by_row
        for intervals in selected:
            for a, b in intervals:
                y[a:b + 1] = 1
        np.savez_compressed(OUT / f"nasa_{channel}.npz", X=X.astype(np.float32), y=y, spacecraft=mission)
        rows.append((mission.lower(), channel, length, X.shape[1], y.mean()))
    return rows


def standardize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return ((X - X.mean(0)) / sd).astype(np.float32)


def load_bench(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / f"{name}.npz")
    return standardize(z["X"]), z["y"].astype(np.int8)


def build(*, block: str = "all", nasa_first_row: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in smd_names() if block == "all" else ():
        X = np.loadtxt(RAW / "smd" / "test" / f"{m}.txt", delimiter=",")
        y = np.loadtxt(RAW / "smd" / "test_label" / f"{m}.txt").astype(np.int8)
        np.savez_compressed(OUT / f"smd_{m}.npz", X=X.astype(np.float32), y=y)
        rows.append(("smd", m, len(y), X.shape[1], y.mean()))
    rows.extend(build_nasa(first_row=nasa_first_row))
    for m in semi_machines() if block == "all" else ():
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("all", "nasa"), default="all")
    parser.add_argument("--nasa-first-row", action="store_true",
                        help="Usa la primera fila validada de cada canal para la comparación histórica")
    args = parser.parse_args()
    build(block=args.block, nasa_first_row=args.nasa_first_row)
