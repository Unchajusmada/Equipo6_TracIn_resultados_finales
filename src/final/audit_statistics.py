"""Read-only reanalysis of final CSVs for C-01 and A-02."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nasa_cohort import HISTORICAL, MAIN, N_SEEDS, validate_pairs as validate_nasa_pairs
BASE = ("base", "win")
M3 = ("base", "tgt")
FINAL = ("adapt_r3", "tgt")
METHODS = {BASE, M3, FINAL}
# Unit counts describe this archive. A changed cohort requires an explicit update.
DESIGNS = {
    "synth": (("ds", "seed"), (15, 20), "seed"),
    "contam": (("ds", "ratio", "seed"), (3, 7, 20), "seed"),
    "real": (("ds", "seed"), (4, 10), "ds"),
    "smd": (("ds", "seed"), (28, 5), "ds"),
    "nasa": (("ds", "seed"), (80, N_SEEDS), "ds"),
    "semi": (("ds", "mix", "ratio", "seed"), (10, 3, 5, 3), "ds"),
}
RATIOS = {
    "contam": {"0.01", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"},
    "semi": {"0.01", "0.05", "0.1", "0.2", "0.3"},
}


def validate_pairs(methods):
    keys = methods[BASE].keys()
    for method in METHODS - {BASE}:
        if methods[method].keys() != keys:
            raise ValueError(f"Unpaired keys for {method}")
    if not keys:
        raise ValueError("Empty comparison")
    return sorted(keys)


def load_block(path, block, *, nasa_cohort=MAIN):
    columns, sizes, _ = DESIGNS[block]
    methods = {method: {} for method in METHODS}
    nasa_pairs = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if block == "nasa":
                if row.get("nasa_cohort", nasa_cohort) != nasa_cohort:
                    raise ValueError(f"NASA: cohorte incompatible con {nasa_cohort}")
                nasa_pairs.add((row["ds"], row["seed"]))
            method = (row["train"], row["proj"])
            if row["cfg"] != "L4P1" or method not in METHODS:
                continue
            key = tuple(row[col] for col in columns)
            if key in methods[method]:
                raise ValueError(f"{block}: duplicate key {key}, {method}")
            values = {col: float(row[col]) for col in
                      ("auprc", "auroc", "iqr_bin", "dyn_bin", "anom_rate")}
            if any(not 0 <= v <= 1 for v in values.values()):
                raise ValueError(f"{block}: invalid metric at {key}, {method}")
            methods[method][key] = values
    keys = validate_pairs(methods)
    if block == "nasa":
        validate_nasa_pairs(nasa_pairs, nasa_cohort)
        if set(keys) != nasa_pairs:
            raise ValueError("NASA: faltan comparaciones primarias para la cohorte completa")
        sizes = (len({ds for ds, _ in keys}), N_SEEDS)
    levels = [set(axis) for axis in zip(*keys)]
    if tuple(map(len, levels)) != sizes or set(itertools.product(*levels)) != set(keys):
        raise ValueError(f"{block}: incomplete or unexpected factorial design")
    if levels[columns.index("seed")] != {str(i) for i in range(sizes[-1])}:
        raise ValueError(f"{block}: unexpected seed set")
    if block in RATIOS and levels[columns.index("ratio")] != RATIOS[block]:
        raise ValueError(f"{block}: unexpected contamination levels")
    return methods


def differences(methods, first, second, metric, reference_metric=None):
    reference_metric = reference_metric or metric
    return {key: methods[first][key][metric] - methods[second][key][reference_metric]
            for key in validate_pairs(methods)}


def drop_differences(deltas, axis):
    # d(30%) - d(1%) = drop(reference) - drop(method).
    return {key[:axis] + key[axis + 1:]:
            deltas[key[:axis] + ("0.3",) + key[axis + 1:]] - value
            for key, value in deltas.items() if key[axis] == "0.01"}


def aggregate(deltas, axis):
    groups = defaultdict(list)
    for key, value in deltas.items():
        groups[key[axis]].append(value)
    return {unit: float(np.mean(values)) for unit, values in sorted(groups.items())}


def estimate(units, *, inference, boundary=0.0):
    """Mean effect and matched t inference at the independent-unit level."""
    x = np.array(list(units.values()), dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else None
    result = dict(n=n, mean=mean, sd=sd, dz=mean / sd if sd else None,
                  wins=int((x > 0).sum()), losses=int((x < 0).sum()),
                  ci95=None, lower95=None, p=None)
    if inference and sd:
        se = sd / np.sqrt(n)
        radius = float(t.ppf(0.975, n - 1) * se)
        result.update(ci95=[mean - radius, mean + radius],
                      lower95=mean - float(t.ppf(0.95, n - 1) * se),
                      p=float(t.sf((mean - boundary) / se, n - 1)))
    return result


def noninferiority(units, margin, *, inference):
    if margin is not None and not 0 < margin <= 1:
        raise ValueError("The F1 margin must be in (0, 1].")
    result = estimate(units, inference=inference, boundary=-margin if margin else 0.0)
    result["margin"] = margin
    if margin is None:
        result.update(status="pending_margin", p=None, lower95=None)
    elif result["p"] is None:
        result["status"] = "descriptive_only"
    else:
        result["status"] = "evaluated"
    return result


def holm_family(rows):
    # Preserve the size of the planned family if H2 is pending or a test undefined.
    ordered = sorted((row for row in rows if row["p"] is not None), key=lambda row: row["p"])
    running = 0.0
    for i, row in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - i) * row["p"]))
        row["p_holm"] = running


def analyze_block(methods, block, margin=None):
    columns, _, unit = DESIGNS[block]
    specs = [
        ("H1", differences(methods, FINAL, BASE, "auprc"), False),
        ("H2", differences(methods, FINAL, BASE, "iqr_bin", "dyn_bin"), False),
        ("H3a", differences(methods, FINAL, M3, "auprc"), False),
        ("H3b", differences(methods, FINAL, M3, "iqr_bin"), False),
        ("H5a", differences(methods, M3, BASE, "auroc"), False),
        ("H5b", differences(methods, M3, BASE, "auprc"), False),
    ]
    if block in RATIOS:
        specs.append(("H4" if block == "contam" else "H4r",
                      differences(methods, FINAL, BASE, "auroc"), True))
    if block == "semi":
        specs.append(("H5c", differences(methods, M3, BASE, "auroc"), True))

    out = {}
    original = block in {"synth", "contam", "real"}
    for hypothesis, deltas, is_drop in specs:
        index = columns
        if is_drop:
            axis = columns.index("ratio")
            deltas = drop_differences(deltas, axis)
            index = columns[:axis] + columns[axis + 1:]
        units = aggregate(deltas, index.index(unit))
        if hypothesis == "H2":
            result = noninferiority(units, margin, inference=block != "real")
        else:
            result = estimate(units, inference=block != "real")
        family = "exploratory_h5" if original and hypothesis in {"H5a", "H5b"} else "primary"
        result.update(unit=unit, units=units, n_pairs=len(deltas), family=family, p_holm=None)
        if block == "real":
            result["seed_deltas_by_dataset"] = {
                ds: {seed: deltas[ds, seed] for dataset, seed in sorted(deltas) if dataset == ds}
                for ds in units}
        # Sensitivity summaries describe heterogeneity; they carry no inferential p-value.
        if block == "synth":
            result["by_scenario"] = aggregate(deltas, index.index("ds"))
        out[hypothesis] = result
    for family in ("primary", "exploratory_h5"):
        holm_family([row for row in out.values() if row["family"] == family])
    h2 = out["H2"]
    if h2["status"] == "evaluated":
        h2["status"] = ("noninferiority_supported" if h2["p_holm"] < 0.05
                        else "inconclusive")
    return out


def h6_summary(blocks):
    rows = {}
    for block in ("smd", "nasa"):
        methods = blocks[block]
        x = aggregate({key: row["anom_rate"] for key, row in methods[BASE].items()}, 0)
        y = aggregate(differences(methods, M3, BASE, "auroc"), 0)
        rows[block] = [(x[ds], y[ds]) for ds in x]
    rows["pooled"] = rows["smd"] + rows["nasa"]
    return {block: {"n": len(pairs), "rho": float(spearmanr(*zip(*pairs)).statistic)}
            for block, pairs in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "final")
    parser.add_argument("--delta", type=float, help="Scientifically justified absolute F1 margin")
    parser.add_argument("--delta-justification", help="Scientific rationale for the supplied margin")
    parser.add_argument("--nasa-first-row", action="store_true", help="Select the historical NASA cohort of 81 channels")
    args = parser.parse_args()
    if (args.delta is not None) != bool(args.delta_justification and args.delta_justification.strip()):
        parser.error("--delta and --delta-justification must be supplied together")
    if args.delta is not None and not 0 < args.delta <= 1:
        parser.error("--delta must be in (0, 1]")
    cohort = HISTORICAL if args.nasa_first_row else MAIN
    blocks = {block: load_block(args.input / f"{block}_eval.csv", block, nasa_cohort=cohort) for block in DESIGNS}
    output = {"analysis": "corrected-audit", "nasa_cohort": cohort,
              "delta_justification": args.delta_justification,
              "blocks": {block: analyze_block(data, block, args.delta)
                         for block, data in blocks.items()},
              "h6_descriptive": h6_summary(blocks)}
    print(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
