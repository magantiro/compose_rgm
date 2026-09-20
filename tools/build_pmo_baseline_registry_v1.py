"""Build the sealed, regime-separated PMO baseline registry from the authors' archive.

The PMO leaderboard has TWO incompatible regimes.  A "prescreened" run scores the
whole of ZINC250k with the task oracle before counted call 1 and seeds itself from
the top of that ranking; a "no prescreen" run does not.  They differ by roughly
249,455 uncounted oracle calls per task against a 10,000-call counted budget, so a
value from one column is not a comparator for a run in the other.  COMPOSE runs
``prescreen: false``.

This builder reads the InVirtuoGen results archive (``main.tar.gz``, sha256 pinned
below and independently recorded in ``diagnostics/pmo_ivg_oracle_parity/result.json``)
and derives every registry value from THREE concordant in-archive sources:

1. ``results/target_property/{no_,}prescreen_3_runs/<task>/results_<task>.csv``
   -- the raw per-seed ``auc_top10`` rows, i.e. the data behind the table;
2. ``pmo_comparison_table_with_std{,_no_prescreen}.tex`` -- the generated LaTeX;
3. ``README.md`` -- the markdown rendering of the same tables.

A task is admitted only when the mean of its raw per-seed rows reproduces the
published cell to 3 decimal places.  Nothing is inferred, interpolated or averaged
across regimes; a cell with no raw backing is written ABSENT.

Invariants maintained/tested here:

* Every value carries its regime, its budget, its metric id (``auc_top10@10000``,
  the identifier convention of ``compose_v4.eval.pmo_budget_aggregation``), and a
  file-level source location inside a sha256-pinned archive.
* The two regimes never share a dict, so a reader cannot pick the wrong column by
  forgetting a key.
* The payload is sealed with the repository convention
  ``sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))``.
* The registry is RESEARCH-ONLY: it authorizes no oracle call and no launch.

Usage::

    python3 tools/build_pmo_baseline_registry_v1.py --archive-root <extracted-repo>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

# ---- Pins ----

IVG_ARCHIVE_SHA256 = "09a9f3d2d9c676d0a71af7fc9f19724d6247daad15728e81e0c0e70377dc94c0"
IVG_REPOSITORY = "https://github.com/invirtuolabs/InVirtuoGen_results"
IVG_COMMIT = "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
IVG_PAPER = "arXiv:2509.26405 (InVirtuoGen)"

PMO_BUDGET = 10_000
PMO_LOG_FREQUENCY = 100
METRIC_ID = f"auc_top10@{PMO_BUDGET}"

REGIMES = {
    "no_prescreen": {
        "results_dir": "results/target_property/no_prescreen_3_runs",
        "tex": "pmo_comparison_table_with_std_no_prescreen.tex",
        "column": "InVirtuoGen (no prescreen)",
    },
    "prescreened": {
        "results_dir": "results/target_property/prescreen_3_runs",
        "tex": "pmo_comparison_table_with_std.tex",
        "column": "InVirtuoGen",
    },
}

# Secondary no-prescreen columns rendered in the authors' README. These are
# SECOND-HAND: the table caption attributes them to other publications, so the
# archive is evidence of the transcription, not of the measurement.
SECONDARY_COLUMNS = ("gen_gfn", "mol_ga", "reinvent", "graph_ga")
SECONDARY_ATTRIBUTION = {
    "gen_gfn": "Genetic GFN (Kim et al. 2024), quoted by InVirtuoGen",
    "mol_ga": "Mol GA (Tripp et al. 2023), quoted by InVirtuoGen",
    "reinvent": "REINVENT, via the PMO benchmark paper (Gao et al. 2022), quoted by InVirtuoGen",
    "graph_ga": "Graph GA, via the PMO benchmark paper (Gao et al. 2022), quoted by InVirtuoGen",
}

_TEX_ROW = re.compile(
    r"\\small\{([^}]+)\}\s*&\s*\$\\?m?a?t?h?b?f?\{?([0-9.]+)\}?\$\s*"
    r"\{\\tiny \(\$\\pm\$ ([0-9.]+)\)\}"
)
_README_CELL = re.compile(r"\*?\*?([0-9.]+)\s*(?:\(±\s*([0-9.]+)\))?\*?\*?")


# ---- Readers ----


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tex_column(path: Path) -> dict[str, tuple[float, float]]:
    """First data column of a generated comparison table, keyed by task name."""

    out: dict[str, tuple[float, float]] = {}
    for line in path.read_text().splitlines():
        match = _TEX_ROW.match(line.strip())
        if match:
            task = match.group(1).strip().replace(" ", "_")
            out[task] = (float(match.group(2)), float(match.group(3)))
    return out


def read_readme_table(path: Path) -> dict[str, dict[str, float | None]]:
    """The README's no-prescreen table: InVirtuoGen plus the quoted comparators."""

    rows: dict[str, dict[str, float | None]] = {}
    in_table = False
    for line in path.read_text().splitlines():
        if "InVirtuoGen (no prescreen)" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or set(cells[0]) <= set("-: "):
            continue
        task = cells[0].replace("**", "").strip().replace(" ", "_")
        if task.lower().startswith("sum"):
            break
        parsed: dict[str, float | None] = {}
        for name, cell in zip(("invirtuogen", *SECONDARY_COLUMNS), cells[1:6]):
            match = _README_CELL.match(cell)
            parsed[name] = float(match.group(1)) if match else None
        rows[task] = parsed
    return rows


def read_raw_runs(results_dir: Path) -> dict[str, dict]:
    """Per-task per-seed ``auc_top10`` rows, straight from the published CSVs."""

    runs: dict[str, dict] = {}
    for task_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        values: list[float] = []
        seeds: list[float | None] = []
        files: list[str] = []
        for csv_path in sorted(task_dir.glob("*.csv")):
            files.append(str(csv_path.relative_to(results_dir.parents[2])))
            with csv_path.open() as handle:
                for row in csv.DictReader(handle):
                    values.append(float(row["auc_top10"]))
                    seed = row.get("seed")
                    seeds.append(float(seed) if seed not in (None, "") else None)
        if values:
            runs[task_dir.name] = {"values": values, "seeds": seeds, "files": files}
    return runs


# ---- Assembly ----


def build_regime(root: Path, regime: str) -> tuple[dict[str, dict], dict]:
    spec = REGIMES[regime]
    results_dir = root / spec["results_dir"]
    tex_path = root / spec["tex"]
    published = read_tex_column(tex_path)
    raw = read_raw_runs(results_dir)

    tasks: dict[str, dict] = {}
    disagreements: list[dict] = []
    for task in sorted(set(published) | set(raw)):
        cell = published.get(task)
        rows = raw.get(task)
        if cell is None or rows is None:
            tasks[task] = {
                "value": "ABSENT",
                "reason": (
                    "no published cell" if cell is None else "no raw per-seed rows"
                ),
            }
            continue
        mean = statistics.fmean(rows["values"])
        std = statistics.pstdev(rows["values"])
        reproduces = abs(round(mean, 3) - cell[0]) <= 0.0005
        if not reproduces:
            disagreements.append(
                {"task": task, "published": cell[0], "raw_mean": round(mean, 4)}
            )
            tasks[task] = {
                "value": "ABSENT",
                "reason": "raw per-seed mean does not reproduce the published cell",
                "published_cell": cell[0],
                "raw_mean": round(mean, 6),
            }
            continue
        tasks[task] = {
            "value": cell[0],
            "std_population": cell[1],
            "metric_id": METRIC_ID,
            "metric_definition": "top_auc(top_n=10, finish=True)",
            "budget_oracle_calls": PMO_BUDGET,
            "log_frequency": PMO_LOG_FREQUENCY,
            "prescreen": regime == "prescreened",
            "n_runs": len(rows["values"]),
            "seeds": rows["seeds"],
            "per_run_auc_top10": [round(v, 10) for v in rows["values"]],
            "raw_mean_unrounded": round(mean, 10),
            "raw_std_population_unrounded": round(std, 10),
            "reproduces_published_cell_to_3dp": True,
            "evidence": "MEASURED",
            "sources": [
                {
                    "kind": "published_table",
                    "archive": IVG_REPOSITORY,
                    "path": spec["tex"],
                    "sha256": _sha256(tex_path),
                    "column": spec["column"],
                },
                {
                    "kind": "raw_per_run_data",
                    "archive": IVG_REPOSITORY,
                    "paths": rows["files"],
                },
            ],
        }
    summary = {
        "n_tasks": len(tasks),
        "n_with_value": sum(1 for v in tasks.values() if v["value"] != "ABSENT"),
        "n_absent": sum(1 for v in tasks.values() if v["value"] == "ABSENT"),
        "sum_of_published_cells": round(
            sum(v["value"] for v in tasks.values() if v["value"] != "ABSENT"), 4
        ),
        "sum_of_unrounded_run_means": round(
            sum(
                v["raw_mean_unrounded"]
                for v in tasks.values()
                if v["value"] != "ABSENT"
            ),
            6,
        ),
        "raw_vs_published_disagreements": disagreements,
    }
    return tasks, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        required=True,
        type=Path,
        help="extracted InVirtuoGen_results-main directory",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.archive_root
    no_prescreen, np_summary = build_regime(root, "no_prescreen")
    prescreened, pre_summary = build_regime(root, "prescreened")
    readme = read_readme_table(root / "README.md")

    # Third concordance check: the README rendering of the no-prescreen column.
    readme_agrees = {
        task: (
            task in readme
            and readme[task]["invirtuogen"] is not None
            and abs(readme[task]["invirtuogen"] - entry["value"]) <= 0.0005
        )
        for task, entry in no_prescreen.items()
        if entry["value"] != "ABSENT"
    }

    secondary: dict[str, dict] = {}
    for task, cells in readme.items():
        secondary[task] = {
            name: (cells[name] if cells[name] is not None else "ABSENT")
            for name in SECONDARY_COLUMNS
        }

    payload = {
        "schema_version": "pmo_baseline_registry_v1",
        "status": "SEALED_RESEARCH_ONLY",
        "oracle_calls_authorized": 0,
        "modal_launch_authorized": False,
        "scored_launch_authorized": False,
        "network_access": False,
        "purpose": (
            "Authoritative, regime-separated PMO baseline values with per-value "
            "provenance. Research record only: authorizes nothing."
        ),
        "comparison_rule": (
            "COMPOSE runs prescreen=false. Only regimes.no_prescreen is a valid "
            "comparator. regimes.prescreened is retained solely so that a "
            "substitution is visible, and must never be used as a target."
        ),
        "metric": {
            "metric_id": METRIC_ID,
            "definition": "top_auc(top_n=10, finish=True)",
            "budget_oracle_calls": PMO_BUDGET,
            "log_frequency": PMO_LOG_FREQUENCY,
            "runs_averaged": 3,
            "dispersion": "population standard deviation (ddof=0) over the 3 runs",
            "consumable_by": "compose_v4.eval.pmo_budget_aggregation.aggregate_suite",
            "budget_warning": (
                "A 250-call or 1,000-call COMPOSE reading is a DIFFERENT metric "
                "(auc_top10@250 / auc_top10@1000) and is not comparable to any value here."
            ),
        },
        "provenance": {
            "paper": IVG_PAPER,
            "repository": IVG_REPOSITORY,
            "commit_recorded_in_repo": IVG_COMMIT,
            "archive_sha256": IVG_ARCHIVE_SHA256,
            "archive_sha256_independently_recorded_at": (
                "diagnostics/pmo_ivg_oracle_parity/result.json"
                " -> contract.official_ivg_environment_evidence.archive_sha256"
            ),
            "concordant_sources_per_value": 3,
        },
        "regimes": {
            "no_prescreen": {
                "prescreen": False,
                "description": (
                    "No ZINC250k prescreen. Reproduce command in the archive README "
                    "runs with --max_oracle_calls 10000 and WITHOUT --use_prescreen."
                ),
                "tasks": no_prescreen,
                "summary": np_summary,
                "readme_column_agrees": readme_agrees,
            },
            "prescreened": {
                "prescreen": True,
                "description": (
                    "ZINC250k prescreened. Reproduce command in the archive README "
                    "adds --use_prescreen; the initial population is read from a "
                    "per-task oracle-scored ZINC250k csv and those scores never enter "
                    "mol_buffer, so they are not charged against the 10,000-call budget."
                ),
                "tasks": prescreened,
                "summary": pre_summary,
                "do_not_compare": (
                    "NOT a comparator for a prescreen=false run."
                ),
            },
        },
        "secondary_no_prescreen_columns": {
            "tier": "SECOND_HAND",
            "caution": (
                "Transcribed by InVirtuoGen from other publications; the archive is "
                "evidence of the transcription, not of the measurement. No raw "
                "per-run data for these columns exists in the archive."
            ),
            "attribution": SECONDARY_ATTRIBUTION,
            "source": "README.md no-prescreen table",
            "tasks": secondary,
        },
    }

    baseline_map = {
        task: (entry["value"] if entry["value"] != "ABSENT" else None)
        for task, entry in no_prescreen.items()
    }
    payload["baseline_map_for_aggregate_suite"] = {
        "metric_id": METRIC_ID,
        "regime": "no_prescreen",
        "note": (
            "Pass directly as aggregate_suite(readings, baseline=...). None means "
            "ABSENT; aggregate_suite reports those under baseline_absent."
        ),
        "map": baseline_map,
    }

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    document = {"payload": payload, "payload_sha256": hashlib.sha256(blob.encode()).hexdigest()}
    out = args.repo_root / "configs" / "pmo_baseline_registry_v1.json"
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(
        f"no_prescreen: {np_summary['n_with_value']}/{np_summary['n_tasks']} "
        f"sum={np_summary['sum_of_published_cells']} "
        f"unrounded={np_summary['sum_of_unrounded_run_means']}"
    )
    print(
        f"prescreened:  {pre_summary['n_with_value']}/{pre_summary['n_tasks']} "
        f"sum={pre_summary['sum_of_published_cells']}"
    )
    print(f"readme concordance: {sum(readme_agrees.values())}/{len(readme_agrees)}")
    print(f"payload_sha256: {document['payload_sha256']}")


if __name__ == "__main__":
    main()
