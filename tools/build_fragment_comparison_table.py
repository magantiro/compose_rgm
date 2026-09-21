#!/usr/bin/env python3
"""Put the COMPOSE suite table beside the published baselines it is compared to.

Baselines come from two PINNED upstream artifacts, never from memory:

``references/reference_metrics.csv``
    SAFE-GPT and GenMol, as transcribed inside the InVirtuoGen results
    repository.  This is the only place in the released code where a
    ``distance`` column exists at all -- ``evaluate_smiles`` does not compute
    one, so upstream reads the baselines' distance from here and drops the
    column from its own LaTeX table.

``ivg_results_with_std.json``
    InVirtuoGen's own numbers, carried in the same repository.  It has no
    distance entry, so distance is reported as absent rather than imputed.

Numbers that appear in neither file are recorded as UNSOURCED.  A promotion or
a headline claim decided by an unsourced baseline is not defensible, and the
cheapest way to prevent that is to make the gap visible in the table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASK_TO_CATEGORY = {
    "motif_extension": "motif",
    "linker_design": "linker",
    "scaffold_morphing": "morphing",
    "superstructure_generation": "superstructure",
    "scaffold_decoration": "decoration",
}
METRICS = ("validity", "uniqueness", "quality", "diversity", "distance")
def _upstream_root() -> Path:
    """The hash-verified upstream tree, FETCHED rather than vendored.

    The blobs are not committed (CC BY-NC-SA 4.0), so resolve them through the
    fetcher, which downloads and SHA-256 verifies them on a fresh clone.
    """
    import sys as _sys

    tools = Path(__file__).resolve().parent
    if str(tools) not in _sys.path:
        _sys.path.insert(0, str(tools))
    from fetch_official_fragment_evaluator import fetch, verify_only

    root = fetch()
    verify_only()
    return Path(root)


def load_reference_metrics() -> dict[str, dict[str, dict]]:
    path = _upstream_root() / "references" / "reference_metrics.csv"
    out: dict[str, dict[str, dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entry = {}
            for metric in METRICS:
                raw = (row.get(metric) or "").strip()
                std = (row.get(f"{metric}_std") or "").strip()
                entry[metric] = {
                    "mean": float(raw) if raw else None,
                    "std": float(std) if std else None,
                }
            out.setdefault(row["category"], {})[row["method"]] = entry
    return out


def load_ivg() -> dict[str, dict]:
    # InVirtuoGen's own results are not among the pinned blobs; they are optional.
    local = Path(".vendor_official/ivg_results_with_std.json")
    if not local.exists():
        return {}
    payload = json.loads(local.read_text())
    out = {}
    for category, block in payload.items():
        entry = block.get("InVirtuoGen", {})
        metrics, stds = entry.get("metrics", {}), entry.get("std", {})
        out[category] = {
            metric: {"mean": metrics.get(metric), "std": stds.get(metric)}
            for metric in METRICS
        }
    return out


def fmt(entry: dict | None, metric: str) -> str:
    if not entry or entry.get(metric, {}).get("mean") is None:
        return "     --"
    mean = entry[metric]["mean"]
    return f"{mean:7.3f}" if metric == "diversity" or metric == "distance" else f"{mean:7.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("diagnostics/fragment_official_suite_v2/suite_table.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/fragment_official_suite_v2/comparison_table.json"),
    )
    args = parser.parse_args()

    suite = json.loads(args.table.read_text())
    references = load_reference_metrics()
    ivg = load_ivg()

    rows = []
    for task, category in TASK_TO_CATEGORY.items():
        block = suite["tasks"].get(task)
        if block is None:
            continue
        compose = {m: block["summary"].get(m) for m in METRICS}
        rows.append(
            {
                "task": task,
                "category": category,
                "complete": block["complete"],
                "withheld_reason": block.get("withheld_reason"),
                "copied_from": block.get("copied_from"),
                "constraint_decomposition": block.get("constraint_decomposition"),
                "COMPOSE": compose,
                "InVirtuoGen": ivg.get(category),
                "GenMol": references.get(category, {}).get("GenMol"),
                "SAFE-GPT": references.get(category, {}).get("SAFE-GPT"),
            }
        )

    payload = {
        "schema": "compose_fragment_comparison_v1",
        "identity": suite["identity"],
        "baseline_provenance": {
            "GenMol": "references/reference_metrics.csv @ b50bb3ae (pinned)",
            "SAFE-GPT": "references/reference_metrics.csv @ b50bb3ae (pinned)",
            "InVirtuoGen": "ivg_results_with_std.json @ b50bb3ae (no distance column)",
            "note": (
                "distance is NOT produced by the released evaluator; the "
                "baselines' distance is transcribed in reference_metrics.csv, "
                "and COMPOSE's is computed with the official "
                "calculate_average_tanimoto against the dummy-stripped prompt"
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print("!! COMPOSE's row below prints TASK SUCCESS in the validity column.")
    print("!! That is NOT the comparator's definition of validity, which never")
    print("!! inspects the prompt fragment. For the like-for-like comparison read")
    print("!! diagnostics/fragment_official_suite_v2/corrected_table.json.")
    print(f"sampler {suite['identity']['sampler_config_sha256'][:16]}  "
          f"rdkit {suite['identity']['kernel']['rdkit']}")
    print(f"{'task':24s} {'method':13s} {'valid':>7s} {'uniq':>7s} {'qual':>7s} "
          f"{'div':>7s} {'dist':>7s}")
    for row in rows:
        flag = "" if row["complete"] else "  [PARTIAL]"
        if row.get("withheld_reason"):
            flag += f"  [WITHHELD: {row['withheld_reason']}]"
        for method in ("COMPOSE(task_success)", "GenMol", "SAFE-GPT", "InVirtuoGen"):
            entry = row["COMPOSE" if method.startswith("COMPOSE") else method]
            line = " ".join(fmt(entry, m) for m in METRICS)
            label = row["task"] if method.startswith("COMPOSE") else ""
            print(f"{label:24s} {method:21s} {line}"
                  f"{flag if method.startswith('COMPOSE') else ''}")
        print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
