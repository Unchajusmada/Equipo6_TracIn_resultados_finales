"""Arma ../RESULTADOS.md a partir de resultados_template.md.

Marcadores del template:
  {{t:nombre}}         inserta results/tables/nombre.md (título como ###)
  {{v:ruta|formato}}   valor de results/deck_data.json (ruta separada por '/';
                       formato: 2, 3, +2, +3, pct, int, e)
  {{RW_REAL}}          resumen de la reponderación en reales (fase B)
  {{r:nombre}}         inserta results/replica/tables/nombre.md (réplica)
  {{w:ruta|formato}}   valor de results/replica/replica_numbers.json
                       (+ hypotheses.json bajo la clave 'hyp')
Así ninguna cifra del informe se escribe a mano.

Uso:  python write_results.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
DECK = json.loads((ROOT / "results" / "deck_data.json").read_text(encoding="utf-8"))
REP = ROOT / "results" / "replica"
REPN = json.loads((REP / "replica_numbers.json").read_text(encoding="utf-8"))
REPN["hyp"] = json.loads((REP / "hypotheses.json").read_text(encoding="utf-8"))


def get(path: str, root=None):
    o = DECK if root is None else root
    for k in path.split("/"):
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


def fmt(v, f: str) -> str:
    if f == "pct":
        return f"{v * 100:.1f} %"
    if f == "e":
        return f"{v:.1e}"
    if f == "int":
        return f"{round(v):,}".replace(",", " ")
    if f.startswith("+"):
        d = int(f[1:])
        return f"{'+' if v >= 0 else '−'}{abs(v):.{d}f}"
    return f"{v:.{int(f)}f}"


def table(name: str, tab: Path = TAB) -> str:
    txt = (tab / f"{name}.md").read_text(encoding="utf-8").strip()
    return re.sub(r"^## ", "### ", txt, count=1)


def rw_real() -> str:
    rwd = DECK.get("real", {}).get("rw_delta", {})
    if not rwd:
        return "*Reponderación en reales: pendiente (fase B).*"
    pretty = dict(zip(DECK["real"]["keys"], DECK["real"]["pretty"]))
    rows = ["| Dataset | Suave − original | Recorte − original | Suave − original (proy. obj.) | Recorte − original (proy. obj.) |",
            "|---|---|---|---|---|"]
    for ds, d in rwd.items():
        cell = lambda v: f"{fmt(d[v][0], '+3')} ({d[v][1]}/{d[v][2]})" if v in d else "—"
        rows.append(f"| {pretty[ds]} | {cell('rw_soft_r3')} | {cell('rw_trim_r3')} | {cell('rw_soft_r3_tgt')} | {cell('rw_trim_r3_tgt')} |")
    return ("Δ AUPRC tras 3 rondas, media de 5 semillas (entre paréntesis, semillas en que mejora):\n\n"
            + "\n".join(rows))


def main():
    src = (Path(__file__).parent / "resultados_template.md").read_text(encoding="utf-8")
    out = re.sub(r"\{\{t:([\w]+)\}\}", lambda m: table(m.group(1)), src).replace("{{RW_REAL}}", rw_real())
    out = re.sub(r"\{\{r:([\w]+)\}\}", lambda m: table(m.group(1), REP / "tables"), out)

    def val(m, root=None):
        # las claves de método llevan '|' (p. ej. lstm|tracin): el formato es lo que va tras el último '|'
        path, _, f = m.group(1).rpartition("|")
        if not re.fullmatch(r"\+?\d+|pct|int|e", f):
            path, f = m.group(1), "3"
        return fmt(get(path, root), f)
    out = re.sub(r"\{\{v:([^}]+)\}\}", val, out)
    out = re.sub(r"\{\{w:([^}]+)\}\}", lambda m: val(m, REPN), out)
    (ROOT / "RESULTADOS.md").write_text(out, encoding="utf-8")
    print("RESULTADOS.md escrito")


if __name__ == "__main__":
    main()
