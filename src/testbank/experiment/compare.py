"""Comparison table of runs. Walks runs/, prints and exports CSV.

Two non-negotiable rules:

1. **Every row carries n and an interval.** With ~500 images, validation is
   ~100 and the differences between candidates fall within the noise. A table
   of means without intervals invites reading as improvement what is
   dispersion. If a run brings no interval, the cell says `no CI`, not a bare
   number.

2. **What is unfit is marked visibly.** A run with an AGPL detector or with a
   dataset whose source chain is AGPL serves as a performance reference and
   NOT as a candidate. Confusing them is the expensive mistake.

3. **Every row says which split version it ran on.** Two runs on different
   versions were evaluated on different images: their numbers sit in the same
   table but are not comparable, and the column is what says so.
"""

from __future__ import annotations

import csv
from pathlib import Path

from testbank.experiment.run import discover_runs

#: Metrics shown, in order. The last one is the one that decides.
COLUMNS = (
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("coverage_p5", "coverage p5"),
)

#: Contamination, one column per scene and each against its own threshold.
#: A single aggregate number hides both: the floor of a single banknote is
#: exactly zero and that of a fan is 0.89.
SCENE_COLUMNS = (("single", "contam.single"), ("fan", "contam.fan"))

NOT_READY = "NOT READY"


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
            return f"{point:.3f} no CI"
        return f"{point:.3f} [{low:.3f},{high:.3f}]"
    return f"{value:.3f}"


def _scene_cell(metrics: dict, scene: str) -> str:
    """`p95 (threshold)` with a mark if it exceeds it. The threshold goes next
    to the number because the two are different and without seeing it the
    figure cannot be read."""
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
            "split": (run.get("split") or {}).get("version", "-"),
            "aug": (run.get("augmented") or {}).get("version", "-"),
            "ready": "yes" if run.get("production_ready") else NOT_READY,
            "reproducible": "yes" if run.get("reproducible") else "NO",
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
        return "No runs in runs/."
    headers = ["run", "detector", "license", "split", "aug", "n", "ready", "reproducible"]
    headers += [header for _, header in COLUMNS]
    headers += [header for _, header in SCENE_COLUMNS]
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        lines.append("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

    flagged = [r for r in rows if r["ready"] == NOT_READY]
    if flagged:
        lines.append("")
        lines.append(f"{len(flagged)} run(s) NOT READY for production:")
        for row in flagged:
            lines.append(f"  {row['run']}:")
            for blocker in row["_license_blockers"]:
                lines.append(f"    - {blocker}")
        lines.append("")
        lines.append(
            "  They serve as a performance reference, not as a candidate."
        )
    if any("!" in str(row.get(header, "")) for row in rows for _, header in SCENE_COLUMNS):
        lines.append("")
        lines.append(
            "  '!' = exceeds its contamination threshold. The column is "
            "p95/threshold, and the two thresholds differ on purpose: in "
            "single-banknote images the floor is exactly zero, in fans 0.89."
        )

    versions = {r["split"] for r in rows if r["split"] != "-"}
    if len(versions) > 1:
        lines.append("")
        lines.append(
            f"  Runs on {len(versions)} different split versions "
            f"({', '.join(sorted(versions))}): rows on different versions were "
            "evaluated on different images and are NOT comparable."
        )

    unreproducible = [r for r in rows if r["reproducible"] == "NO"]
    if unreproducible:
        lines.append("")
        lines.append(
            f"{len(unreproducible)} NOT REPRODUCIBLE run(s) "
            "(a different problem from fitness: it affects being able to "
            "repeat the number, not being able to deploy it):"
        )
        for reason in sorted({
            b for r in unreproducible for b in (r["_provenance_blockers"] or [])
        }):
            lines.append(f"  - {reason}")
    return "\n".join(lines)


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["run", "timestamp", "detector", "license", "split", "n", "ready", "reproducible"]
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
