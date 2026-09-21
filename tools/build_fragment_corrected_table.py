#!/usr/bin/env python3
"""Recompute the suite table with each measurement under its own name.

The first table reported one column called "validity" and set it beside
GenMol's validity column.  They measure different things.  GenMol's validity is
the fraction of generated SMILES that are chemically valid; its evaluator never
inspects the prompt fragment.  Ours was the fraction of ATTEMPTS that produced a
molecule satisfying the whole task, because the sampler withholds any endpoint
failing the task constraint and emits an unparseable placeholder instead.

This rebuilds the table from the per-shard counters, which already record each
condition separately, and reports:

    chemical_validity        committed and chemically valid, over ATTEMPTS.
                             The like-for-like comparison against a baseline
                             whose evaluator does not check the fragment.
    valid_share_of_produced  of the molecules actually committed, how many are
                             valid -- the ratio that tests the by-construction
                             claim. It is not a benchmark validity.
    fragment_containment     prompt fragment present, over ATTEMPTS.
    task_success             every declared attachment site extended, over
                             ATTEMPTS. This is what the old "validity" was.

A WARNING THIS TOOL WILL NOT LET YOU SKIP: uniqueness, quality, diversity and
distance were scored over the EMITTED set, which is the self-censored one.
Where censoring is material those numbers describe the surviving subset, not the
model, and they cannot be corrected from counters -- only by re-scoring the
committed endpoints, which needs the molecules. They are flagged, not silently
carried over.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TASK_ORDER = ("motif_extension", "superstructure_generation", "scaffold_decoration")
WITHHELD = {"linker_design", "scaffold_morphing"}
# Above this share of committed endpoints censored, the secondary metrics
# describe the survivors rather than the generator.
CENSORING_MATERIAL_PCT = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=Path,
        default=Path("diagnostics/fragment_official_suite_v2/shards"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("diagnostics/fragment_official_suite_v2/corrected_table.json"))
    args = parser.parse_args()

    rows: dict[str, list[dict]] = defaultdict(list)
    sampler_hashes = set()
    for path in sorted(args.shards.glob("*.json")):
        shard = json.loads(path.read_text())
        sampler_hashes.add(shard["sampler"]["config_sha256"])
        for task, result in shard["results"].items():
            for drug, entries in result["per_drug"].items():
                for row in entries:
                    rows[task].append({"drug": drug, **row})
    if len(sampler_hashes) != 1:
        raise SystemExit(f"shards disagree on the sampler: {sorted(sampler_hashes)}")

    tasks = {}
    for task, entries in rows.items():
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in entries:
            by_seed[int(row["seed"])].append(row)

        per_seed = {}
        for seed, seed_rows in sorted(by_seed.items()):
            def mean(fn, _rows=seed_rows):
                return statistics.fmean([fn(r) for r in _rows])

            per_seed[seed] = {
                "chemical_validity": mean(
                    lambda r: 100.0 * r["committed_chemically_valid"] / r["attempts"]),
                "fragment_containment": mean(
                    lambda r: 100.0 * r["committed_fragment_preserving"] / r["attempts"]),
                "task_success": mean(
                    lambda r: 100.0 * r["emitted_nonempty"] / r["attempts"]),
                "produced": mean(
                    lambda r: 100.0 * r["committed_endpoints"] / r["attempts"]),
            }

        summary = {
            key: {
                "mean": statistics.fmean([per_seed[s][key] for s in sorted(per_seed)]),
                "std": statistics.pstdev([per_seed[s][key] for s in sorted(per_seed)]),
            }
            for key in ("chemical_validity", "fragment_containment", "task_success", "produced")
        }

        committed = sum(r["committed_endpoints"] for r in entries)
        valid = sum(r["committed_chemically_valid"] for r in entries)
        emitted = sum(r["emitted_nonempty"] for r in entries)
        censored = committed - emitted
        censoring_pct = 100.0 * censored / committed if committed else 0.0

        tasks[task] = {
            "rows": len(entries),
            "withheld": task in WITHHELD,
            "summary": summary,
            "valid_share_of_produced_pct": 100.0 * valid / committed if committed else float("nan"),
            "committed_endpoints": committed,
            "emitted_after_task_filter": emitted,
            "censored_by_task_filter": censored,
            "censoring_pct_of_committed": censoring_pct,
            "secondary_metrics_valid": censoring_pct <= CENSORING_MATERIAL_PCT,
            "secondary_metrics_note": (
                "uniqueness/quality/diversity/distance were scored over the "
                f"emitted set, which excludes {censored} of {committed} committed "
                f"endpoints ({censoring_pct:.1f}%). "
                + (
                    "Censoring is immaterial, so they stand."
                    if censoring_pct <= CENSORING_MATERIAL_PCT
                    else "They describe the surviving subset, NOT the generator, "
                         "and cannot be corrected from counters -- re-scoring the "
                         "committed endpoints requires the molecules, which were "
                         "not persisted."
                ),
            ),
        }

    payload = {
        "schema": "compose_fragment_corrected_table_v1",
        "sampler_config_sha256": sampler_hashes.pop(),
        "correction": (
            "The column previously printed as 'validity' is task_success. "
            "chemical_validity is the GenMol-comparable quantity and is what "
            "belongs beside a baseline validity column."
        ),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    hdr = (f"{'task':26s} {'n':>4s} {'chem_valid':>11s} {'contain':>9s} "
           f"{'task_succ':>10s} {'valid/produced':>15s} {'2ndary':>8s}")
    print(hdr)
    for task in TASK_ORDER:
        if task not in tasks:
            continue
        b = tasks[task]; s = b["summary"]
        print(f"{task:26s} {b['rows']:>4d} "
              f"{s['chemical_validity']['mean']:>8.2f}+-{s['chemical_validity']['std']:<.2f} "
              f"{s['fragment_containment']['mean']:>8.2f} "
              f"{s['task_success']['mean']:>9.2f} "
              f"{b['valid_share_of_produced_pct']:>14.4f}% "
              f"{'OK' if b['secondary_metrics_valid'] else 'INVALID':>8s}")
    # Beside the pinned baselines, with the CORRECT column opposite each other.
    import csv as _csv

    ref = {}
    import sys as _sys

    _tools = Path(__file__).resolve().parent
    if str(_tools) not in _sys.path:
        _sys.path.insert(0, str(_tools))
    from fetch_official_fragment_evaluator import fetch, verify_only

    fetch()
    verify_only()
    ref_path = Path(fetch()) / "references" / "reference_metrics.csv"
    if ref_path.exists():
        with ref_path.open(newline="", encoding="utf-8") as handle:
            for r in _csv.DictReader(handle):
                ref.setdefault(r["category"], {})[r["method"]] = r
    cat = {
        "motif_extension": "motif",
        "superstructure_generation": "superstructure",
        "scaffold_decoration": "decoration",
    }
    print("\nChemical validity -- the comparator's own definition -- against the pinned baselines.")
    print("Their evaluators never inspect the prompt fragment, so this is the like-for-like column.")
    print(f"\n{'task':26s} {'COMPOSE':>9s} {'GenMol':>9s} {'SAFE-GPT':>9s} {'delta vs GenMol':>17s}")
    for task in TASK_ORDER:
        if task not in tasks or task not in cat:
            continue
        ours = tasks[task]["summary"]["chemical_validity"]["mean"]
        block = ref.get(cat[task], {})
        gen = float(block["GenMol"]["validity"]) if "GenMol" in block else None
        safe = float(block["SAFE-GPT"]["validity"]) if "SAFE-GPT" in block else None
        d = f"{ours - gen:+.2f}" if gen is not None else "--"
        print(f"{task:26s} {ours:9.2f} "
              f"{gen if gen is not None else float('nan'):9.2f} "
              f"{safe if safe is not None else float('nan'):9.2f} {d:>17s}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
