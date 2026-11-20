"""Barrido de réplica en datos reales (GPU): variantes de lo no especificado,
evaluadas con P, R y F1 (binario y macro) para comparar con la Tabla 4.

Uso:  python real_screen.py --variants base,lr1e-2 --datasets gecco --cfgs 4x1,16x4 --seeds 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import project_sum, project_targets_mean, unsupervised_thresholds, valid_mask
from data_real import load
from lab import run
from protocol import PCTS, oracle_f1
from screen import parse_variant
from sklearn.metrics import average_precision_score

OUT = Path(__file__).resolve().parents[2] / "results" / "replica"
TAU = 0.87


def prf(y, p):
    tp = np.sum(y & p); fp = np.sum(~y & p); fn = np.sum(y & ~p); tn = np.sum(~y & ~p)
    pa = tp / (tp + fp) if tp + fp else 0.0
    ra = tp / (tp + fn) if tp + fn else 0.0
    fa = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    pn = tn / (tn + fn) if tn + fn else 0.0
    rn = tn / (tn + fp) if tn + fp else 0.0
    fn_ = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return dict(P_bin=pa, R_bin=ra, F1_bin=fa, P_mac=(pa + pn) / 2, R_mac=(ra + rn) / 2, F1_mac=(fa + fn_) / 2)


def best_thr(y, s, s_all, key):
    best, bt = -1, None
    for t in np.percentile(s_all, PCTS):
        f = prf(y, s >= t)[key]
        if f > best:
            best, bt = f, t
    return bt


def evaluate_real(y, s):
    y = y.astype(bool)
    s = s / s.max()
    out = {"auprc": float(average_precision_score(y, s)), "oracle": oracle_f1(y, s),
           "flag_frac087": float(np.mean(s >= TAU))}
    for nm, thr in (("pct", np.percentile(s, TAU * 100)), ("frac", TAU)):
        out.update({f"S_{nm}_{k}": v for k, v in prf(y, s >= thr).items()})
    k = len(s) // 2
    for crit in ("F1_bin", "F1_mac"):
        t = best_thr(y[:k], s[:k], s, crit)
        out.update({f"D{crit[2:]}_{kk}": v for kk, v in prf(y[k:], s[k:] >= t).items()})
    # lectura alternativa: partición aleatoria (no contigua) 50/50 de los instantes
    tr = np.zeros(len(s), bool)
    tr[np.random.default_rng(0).choice(len(s), k, replace=False)] = True
    t = best_thr(y[tr], s[tr], s, "F1_bin")
    out.update({f"Drand_{kk}": v for kk, v in prf(y[~tr], s[~tr] >= t).items()})
    # mejora 1: reglas sin etiquetas, medidas en la misma segunda mitad que el dinámico
    for nm, thr in unsupervised_thresholds(s).items():
        out.update({f"U_{nm}_{kk}": v for kk, v in prf(y[k:], s[k:] >= thr).items()})
        out[f"U_{nm}_flag"] = float(np.mean(s[k:] >= thr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--datasets", default="creditcard,cicids,gecco,swan_sf")
    ap.add_argument("--cfgs", default="4x1,16x4")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="real_screen.csv")
    ap.add_argument("--leak", action="store_true",
                    help="añade la etiqueta como canal de entrada (hipótesis de fuga; sólo diagnóstico)")
    a = ap.parse_args()
    path = OUT / a.out
    done = set()
    if path.exists():
        d = pd.read_csv(path)
        done = set(zip(d.variant, d.ds, d.seed, d.L, d.P))
    for ds in a.datasets.split(","):
        X, y = load(ds)[:2]
        if a.leak:
            lab = y.astype(np.float64)
            X = np.column_stack([X, (lab - lab.mean()) / lab.std()]).astype(np.float32)
        n = len(y)
        m = valid_mask(n)
        for v in a.variants.split(","):
            cfg = parse_variant(v)
            # en reales el batch y las épocas por defecto son los del experimento principal
            over = {}
            if "b8" not in v and "b1" not in v:
                over["batch"] = a.batch
            if "ep300" not in v and "ep30" not in v:
                over["max_epochs"] = a.max_epochs
            cfg = cfg.__class__(**{**cfg.__dict__, **over})
            for cf in a.cfgs.split(","):
                L, P = map(int, cf.split("x"))
                for seed in range(a.seeds):
                    if (v, ds, seed, L, P) in done:
                        continue
                    t0 = time.time()
                    # los datos reales ya vienen estandarizados (z) desde data_real.py
                    r = run(X, L, P, seed, cfg, device="cuda")
                    rows = []
                    for sc, s in {"tracin": project_sum(r["si"], L + P, n),
                                  "tracin_tgt": project_targets_mean(r["si"], L, P, n),
                                  "pd": project_targets_mean(r["loss_final"], L, P, n)}.items():
                        rows.append(dict(variant=v, ds=ds, seed=seed, L=L, P=P, score=sc, epochs=r["epochs"],
                                         **evaluate_real(y[m], s[m])))
                    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
                    t = rows[0]
                    print(f"{v:22s} {ds:10s} L{L}P{P} s{seed} ep={r['epochs']} {time.time() - t0:.0f}s "
                          f"AUPRC={t['auprc']:.3f} orác={t['oracle']:.3f} S_pct_mac={t['S_pct_F1_mac']:.3f} "
                          f"D_bin={t['D_bin_F1_bin']:.3f} D_mac={t['D_mac_F1_mac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
