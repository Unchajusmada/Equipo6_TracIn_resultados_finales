"""Shared cohort contract for final SMAP/MSL runs and derived artifacts."""

MAIN = "main"
HISTORICAL = "historical-first-row"
N_SEEDS = 5


def validate_channels(channels, cohort):
    channels = set(channels)
    expected = {MAIN: 80, HISTORICAL: 81}[cohort]
    if len(channels) != expected or ("P-2" in channels) != (cohort == HISTORICAL):
        raise ValueError(f"NASA {cohort}: se esperan {expected} canales con la política P-2 indicada; hay {len(channels)}")


def validate_pairs(pairs, cohort):
    pairs = {(channel, str(seed)) for channel, seed in pairs}
    channels = {channel for channel, _ in pairs}
    validate_channels(channels, cohort)
    if pairs != {(channel, str(seed)) for channel in channels for seed in range(N_SEEDS)}:
        raise ValueError(f"NASA {cohort}: se requieren las semillas 0–4 por canal")


def run_files(root, cohort):
    selected = {}
    excluded = 0
    for path in sorted(root.glob("*.npz")):
        if ".tmp" in path.name:
            continue
        channel, sep, seed = path.stem.rpartition("_s")
        if not sep or not channel:
            raise ValueError(f"NASA: nombre de corrida inválido: {path.name}")
        if cohort == MAIN and channel == "P-2":
            excluded += 1
            continue
        selected[channel, seed] = path
    validate_pairs(selected, cohort)
    if excluded:
        print(f"NASA {cohort}: se excluyeron {excluded} archivos históricos de P-2")
    return list(selected.values())
