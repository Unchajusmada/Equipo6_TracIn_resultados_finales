"""Carga y preprocesamiento de los 4 datasets reales del artículo.

Se replican los scripts oficiales del benchmark de Lai et al. (2021)
(``tods/benchmark/realworld_data/data/script/*.py``, rama ``benchmark``),
que son la fuente de los datos del artículo. Diferencias documentadas:

* Credit Card: OpenML no respondía; se usa la réplica pública de TensorFlow
  (``storage.googleapis.com/download.tensorflow.org/data/creditcard.csv``),
  que es el mismo CSV de Kaggle/ULB (284 807 filas, 30 columnas + Class).
* CICIDS: el servidor de UNB ya no sirve el ZIP sin formulario; se usa la
  réplica byte a byte del ZIP original ``GeneratedLabelledFlows.zip`` en
  huggingface.co/datasets/bencorn/CICIDS2017.
* Los ordenamientos por tiempo usan ``kind="stable"`` para que el orden de
  filas con el mismo timestamp sea reproducible.

Uso:  python data_real.py   (genera data/processed/<nombre>.npz)
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
NAMES = ["creditcard", "cicids", "gecco", "swan_sf"]
PRETTY = {"creditcard": "Credit Card", "cicids": "CICIDS", "gecco": "GECCO", "swan_sf": "SWAN-SF"}


def _creditcard() -> pd.DataFrame:
    df = pd.read_csv(RAW / "creditcard" / "creditcard.csv").dropna()
    df = df.sort_values(by=["Time"], kind="stable").drop(columns=["Time"])
    df["Class"] = df["Class"].astype(str).str.replace("'", "").astype(int)
    return df.rename(columns={"Class": "label"})


def _gecco() -> pd.DataFrame:
    df = pd.read_csv(RAW / "gecco" / "gecco.csv").dropna()
    df = df.drop(columns=["Time", df.columns[0]])
    df["EVENT"] = df["EVENT"].astype(str).str.upper().map({"FALSE": 0, "TRUE": 1})
    return df.rename(columns={"EVENT": "label"})


def _cicids() -> pd.DataFrame:
    member = "TrafficLabelling /Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    with zipfile.ZipFile(RAW / "cicids" / "GeneratedLabelledFlows.zip") as z, z.open(member) as f:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    df = df.replace([np.inf, -np.inf, "Infinity", ""], np.nan).dropna()
    df[" Timestamp"] = pd.to_datetime(df[" Timestamp"], dayfirst=True, format="mixed")
    df = df.sort_values(by=[" Timestamp"], kind="stable")
    cols = list(df.columns)
    df = df.drop(columns=cols[0:5] + [cols[6]])
    # El script original mapea las 3 clases "Web Attack ..." a 1; aquí las
    # etiquetas traen un carácter mal codificado, así que se mapea BENIGN -> 0
    # y cualquier otra etiqueta -> 1 (son exactamente esas tres clases).
    df[" Label"] = (df[" Label"].str.strip() != "BENIGN").astype(int)
    df = df.rename(columns={" Label": "label"})
    return df.apply(pd.to_numeric)


def _swan_sf() -> pd.DataFrame:
    d = RAW / "swan_sf" / "petdataset_01"
    frames = []
    for f in sorted(os.listdir(d)):
        if "csv" not in f:
            continue
        lab = f.split("lab[")[1].split("]")[0]
        df = pd.read_csv(d / f, header=0, sep="\t")
        df["label"] = lab
        frames.append(df)
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.sort_values(by=["Timestamp"], kind="stable")
    drop = [c for c in df.columns if "label" in c or "loc" in c or "Timestamp" in c][:-1]
    df = df.drop(columns=drop).reset_index(drop=True)
    df = df.ffill().dropna(axis="columns")
    df["label"] = df["label"].map({"NF": 0, "C": 1, "B": 1, "M": 1, "X": 1}).astype(int)
    df["IS_TMFI"] = df["IS_TMFI"].astype(int)
    return df


LOADERS = {"creditcard": _creditcard, "gecco": _gecco, "cicids": _cicids, "swan_sf": _swan_sf}


def build(name: str) -> None:
    df = LOADERS[name]()
    y = df["label"].to_numpy().astype(np.int8)
    X = df.drop(columns=["label"]).to_numpy().astype(np.float64)
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROC / f"{name}.npz", X=X.astype(np.float32), y=y,
                        columns=np.array(df.drop(columns=["label"]).columns, dtype=str))
    print(f"{name:10s} N={len(y):7d} C={X.shape[1]:3d} anomalías={y.mean() * 100:.3f}%")


def standardize(X: np.ndarray) -> np.ndarray:
    """z-score por canal sobre toda la serie (escenario transductivo, como en
    el artículo). Canales constantes quedan en 0."""
    X = X.astype(np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(PROC / f"{name}.npz")
    return standardize(z["X"]), z["y"].astype(np.int8)


if __name__ == "__main__":
    for n in NAMES:
        build(n)
