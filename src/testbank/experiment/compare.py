"""Tabla comparativa de ejecuciones. Recorre runs/, imprime y exporta CSV.

Dos reglas que no se negocian:

1. **Toda fila lleva n e intervalo.** Con ~500 imagenes, validacion queda en
   ~100 y las diferencias entre candidatos caen dentro del ruido. Una tabla de
   medias sin intervalos invita a leer como mejora lo que es dispersion. Si una
   ejecucion no trae intervalo, la celda dice `sin IC`, no un numero pelado.

2. **Lo no apto se marca visiblemente.** Una ejecucion con detector AGPL o con
   un dataset cuya cadena de fuentes lo es sirve como referencia de rendimiento
   y NO como candidato. Confundirlos es el error caro.
"""

from __future__ import annotations

import csv
from pathlib import Path

from testbank.experiment.run import discover_runs

#: Metricas que se muestran, en orden. Las dos ultimas son las que deciden.
COLUMNS = (
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("coverage_p5", "cobertura p5"),
)

#: Contaminacion, una columna por escena y cada una contra su propio umbral.
#: Un solo numero agregado esconde las dos: el suelo de un billete es cero
#: exacto y el de abanico 0.89.
SCENE_COLUMNS = (("single", "contam.1bill"), ("fan", "contam.abanico"))

NOT_READY = "NO APTO"


def _cell(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    if isinstance(value, dict):
        point = value.get("value")
        low, high = value.get("ci_low"), value.get("ci_high")
        if point is None:
            return "-"
        if low is None or high is None:
            return f"{point:.3f} sin IC"
        return f"{point:.3f} [{low:.3f},{high:.3f}]"
    return f"{value:.3f}"


def _scene_cell(metrics: dict, scene: str) -> str:
    """`p95 (umbral)` con marca si lo pasa. El umbral va al lado del numero
    porque los dos son distintos y sin verlo la cifra no se puede leer."""
    entry = ((metrics.get("contamination") or {}).get("by_scene") or {}).get(scene)
    if not entry or not entry.get("n"):
        return "-"
    mark = "" if entry.get("passes") else " !"
    return f"{entry['p95']:.3f}/{entry['threshold']:.2f}{mark}"


def build_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        metrics = run.get("metrics") or {}
        detector = run.get("detector") or {}
        row = {
            "run": run.get("name", "?"),
            "timestamp": run.get("timestamp", ""),
            "detector": detector.get("name", "-"),
            "license": detector.get("license", "-"),
            "n": str(metrics.get("n_images", "-")),
            "apto": "si" if run.get("production_ready") else NOT_READY,
            "reproducible": "si" if run.get("reproducible") else "NO",
        }
        for key, header in COLUMNS:
            row[header] = _cell(metrics, key)
        for scene, header in SCENE_COLUMNS:
            row[header] = _scene_cell(metrics, scene)
        row["_license_blockers"] = run.get("license_blockers", [])
        row["_provenance_blockers"] = run.get("provenance_blockers", [])
        rows.append(row)
    return rows


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "No hay ejecuciones en runs/."
    headers = ["run", "detector", "license", "n", "apto", "reproducible"]
    headers += [header for _, header in COLUMNS]
    headers += [header for _, header in SCENE_COLUMNS]
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        lines.append("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

    flagged = [r for r in rows if r["apto"] == NOT_READY]
    if flagged:
        lines.append("")
        lines.append(f"{len(flagged)} ejecucion(es) NO APTAS para produccion:")
        for row in flagged:
            lines.append(f"  {row['run']}:")
            for blocker in row["_license_blockers"]:
                lines.append(f"    - {blocker}")
        lines.append("")
        lines.append(
            "  Sirven como referencia de rendimiento, no como candidato."
        )
    if any("!" in str(row.get(header, "")) for row in rows for _, header in SCENE_COLUMNS):
        lines.append("")
        lines.append(
            "  '!' = supera su umbral de contaminacion. La columna es "
            "p95/umbral, y los dos umbrales son distintos a proposito: en "
            "imagenes de un billete el suelo es cero exacto, en abanicos 0.89."
        )

    unreproducible = [r for r in rows if r["reproducible"] == "NO"]
    if unreproducible:
        lines.append("")
        lines.append(
            f"{len(unreproducible)} ejecucion(es) NO REPRODUCIBLES "
            "(problema distinto de la aptitud: afecta a poder repetir el "
            "numero, no a poder desplegarlo):"
        )
        for reason in sorted({
            b for r in unreproducible for b in (r["_provenance_blockers"] or [])
        }):
            lines.append(f"  - {reason}")
    return "\n".join(lines)


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["run", "timestamp", "detector", "license", "n", "apto", "reproducible"]
    headers += [header for _, header in COLUMNS]
    headers += [header for _, header in SCENE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def compare(runs_dir: str | Path) -> tuple[list[dict], str]:
    rows = build_rows(discover_runs(runs_dir))
    return rows, render_table(rows)
