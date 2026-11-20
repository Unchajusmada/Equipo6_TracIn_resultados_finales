"""Barras de progreso de la evaluación final (cuenta los archivos terminados).

Uso: python progress.py      (Ctrl+C para salir)
"""
import time
from pathlib import Path

from tqdm import tqdm

RES = Path(__file__).resolve().parents[2] / "results" / "final"
TOTAL = {"Sintéticos (15 × 20 semillas)": ("synth", 300),
         "Contaminación (3 mezclas × 7 niveles × 20)": ("contam", 420),
         "Reales (4 datasets × 10 semillas)": ("real", 40)}


def done(sub):
    d = RES / sub
    return sum(1 for f in d.glob("*.npz") if ".tmp" not in f.name) if d.exists() else 0


def main():
    grand = sum(n for _, n in TOTAL.values())
    bars = {k: tqdm(total=n, desc=k.ljust(44), position=i, unit="arch", dynamic_ncols=True,
                    bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for i, (k, (_, n)) in enumerate(TOTAL.items())}
    tot = tqdm(total=grand, desc="TOTAL".ljust(44), position=len(TOTAL), unit="arch", dynamic_ncols=True,
               bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}")
    try:
        while True:
            s = 0
            for k, (sub, n) in TOTAL.items():
                c = done(sub)
                bars[k].update(c - bars[k].n)
                s += c
            tot.update(s - tot.n)
            if s >= grand:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    for b in [*bars.values(), tot]:
        b.close()
    print("\nTodo terminado." if tot.n >= grand else "")


if __name__ == "__main__":
    main()
