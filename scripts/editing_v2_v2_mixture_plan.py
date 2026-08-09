"""Two decision tables that fix the V2 compile recipe, then emit the recipe.

WHY THIS EXISTS
---------------
The census established WHICH lanes are worth compiling.  It does not fix two
mixture decisions, and both change what gets built:

1. `real_endpoint_multistep_path` is row-heavy and diversity-light -- 1.73M rows
   and 375,894 sources collapsing to 23,524 scaffolds, one of which carries
   31,569 sources.  A per-scaffold cap is clearly right, but the cap must come
   from a RETENTION CURVE.  Picking 4 because the census happened to instrument
   `@cap4` would be choosing a parameter by accident.

2. Synthetic walks supply 89% of the apparent scaffold-diversity gain, which is
   exactly why they must NOT be sampled for breadth.  Optimising synthetic
   sampling for scaffold count is what produced a 90%-synthetic held-out set
   last time.  They are a CAPABILITY REGULARIZER: sample them to guarantee
   minimum coverage per operator family and per capability cell -- especially
   the cycle/topology families, for which they are the only source -- and then
   cap their share of the mixture.

The intended inductive bias for the final reference model:

    plausibility is learned primarily from REAL molecular transitions, while
    synthetic walks keep the legal operator basis usable everywhere.

Writes a diagnostics JSON. Carries no authority and compiles nothing.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
from pathlib import Path

CAPS = (1, 2, 4, 8, 16)
CYCLE_FAMILIES = {"cycle_insert", "cycle_attach", "ring_system_restate"}


def _scaffold_fn():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    def scaffold(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception:  # noqa: BLE001
            return None

    return scaffold


def collect(active8_root: Path, compiled: set[str]):
    """Per-lane, uncompiled: rows and families per source, plus capability cells."""

    per_lane_source_rows = collections.defaultdict(collections.Counter)
    per_lane_source_fams = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    per_lane_cells = collections.defaultdict(collections.Counter)
    compiled_sources: set[str] = set()

    for task in sorted(os.listdir(active8_root / "tasks")):
        receipt = active8_root / "tasks" / task / "RECEIPT.json"
        stream = active8_root / "tasks" / task / "transitions.jsonl.gz"
        if not (receipt.exists() and stream.exists()):
            continue
        lane = json.loads(receipt.read_text()).get("data_lane", "?")
        is_compiled = task in compiled
        for line in gzip.open(stream, "rt"):
            record = json.loads(line)
            evidence = record["candidate_evidence"]
            if evidence.get("exclusion_reason") is not None:
                continue
            source = evidence["source_canonical_key"]
            if is_compiled:
                compiled_sources.add(source)
                continue
            family = record["model_family"]
            per_lane_source_rows[lane][source] += 1
            per_lane_source_fams[lane][source][family] += 1
            per_lane_cells[lane][record.get("capability_cell_id", "?")] += 1
    return per_lane_source_rows, per_lane_source_fams, per_lane_cells, compiled_sources


def retention_curve(source_rows, source_fams, scaffold_of, compiled_scaffolds):
    """Rows / scaffolds / new scaffolds / family mix retained at each per-scaffold cap."""

    by_scaffold = collections.defaultdict(list)
    for source, rows in source_rows.items():
        sc = scaffold_of(source)
        if sc is not None:
            by_scaffold[sc].append((source, rows))
    # Keep the most productive sources first so a small cap still yields rows.
    for sc in by_scaffold:
        by_scaffold[sc].sort(key=lambda pair: -pair[1])

    total_rows = sum(source_rows.values())
    all_new = {sc for sc in by_scaffold if sc not in compiled_scaffolds}
    out = []
    for cap in CAPS:
        rows = 0
        fams: collections.Counter = collections.Counter()
        kept_scaffolds = 0
        kept_new = 0
        for sc, entries in by_scaffold.items():
            chosen = entries[:cap]
            if not chosen:
                continue
            kept_scaffolds += 1
            if sc not in compiled_scaffolds:
                kept_new += 1
            for source, n in chosen:
                rows += n
                fams.update(source_fams[source])
        insert_delete = fams.get("atom_insert", 0) + fams.get("atom_delete", 0)
        out.append({
            "cap": cap,
            "retained_rows": rows,
            "retained_row_fraction": rows / max(total_rows, 1),
            "retained_scaffolds": kept_scaffolds,
            "retained_new_scaffolds": kept_new,
            "new_scaffold_fraction": kept_new / max(len(all_new), 1),
            "insert_delete_share": insert_delete / max(sum(fams.values()), 1),
        })
    return out, total_rows, len(all_new)


def synthetic_coverage(source_rows, source_fams, cells, per_family_target):
    """Smallest capability-driven synthetic sample: greedy on unmet family need."""

    need = collections.Counter()
    supply = collections.Counter()
    for source, fam_counter in source_fams.items():
        supply.update(fam_counter)
    for family, available in supply.items():
        need[family] = min(per_family_target, available)

    remaining = collections.Counter(need)
    chosen: list[str] = []
    got: collections.Counter = collections.Counter()
    # Sources that serve the scarcest families first.
    order = sorted(
        source_fams,
        key=lambda s: -sum(
            v for f, v in source_fams[s].items()
            if remaining[f] > 0 and f in CYCLE_FAMILIES
        ) - 0.001 * sum(source_fams[s].values()),
    )
    for source in order:
        if not any(remaining[f] > 0 for f in source_fams[source]):
            continue
        chosen.append(source)
        for f, v in source_fams[source].items():
            got[f] += v
            remaining[f] = max(0, remaining[f] - v)
        if all(v == 0 for v in remaining.values()):
            break
    return {
        "per_family_target": per_family_target,
        "sources_selected": len(chosen),
        "rows_selected": int(sum(source_rows[s] for s in chosen)),
        "family_coverage": dict(got.most_common()),
        "families_short": {f: int(v) for f, v in remaining.items() if v > 0},
        "capability_cells_available": len(cells),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--prep-subset", default="configs/process_v2_prep_subset.json")
    ap.add_argument("--synthetic-family-target", type=int, default=12000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.prep_subset).read_text())
    compiled = {c["task_identity_sha256"] for c in plan["chunks"]}
    rows_by_lane, fams_by_lane, cells_by_lane, compiled_sources = collect(
        Path(args.active8_root), compiled
    )
    scaffold = _scaffold_fn()
    cache: dict[str, str | None] = {}

    def scaffold_of(s: str):
        if s not in cache:
            cache[s] = scaffold(s)
        return cache[s]

    compiled_scaffolds = {s for s in (scaffold_of(k) for k in compiled_sources) if s}
    print(f"compiled scaffolds {len(compiled_scaffolds):,}", flush=True)

    lane = "real_endpoint_multistep_path"
    curve, total_rows, total_new = retention_curve(
        rows_by_lane[lane], fams_by_lane[lane], scaffold_of, compiled_scaffolds
    )
    print(f"\n{lane}: {total_rows:,} rows, {total_new:,} new scaffolds available")
    print(f"{'cap':>4} {'rows':>10} {'row%':>6} {'scaffolds':>10} {'new':>8} {'new%':>6} {'ins/del':>8}")
    for row in curve:
        print(f"{row['cap']:>4} {row['retained_rows']:>10,} "
              f"{100 * row['retained_row_fraction']:>5.1f}% {row['retained_scaffolds']:>10,} "
              f"{row['retained_new_scaffolds']:>8,} {100 * row['new_scaffold_fraction']:>5.1f}% "
              f"{100 * row['insert_delete_share']:>7.1f}%")

    synth = "reversible_synthetic_walk"
    cov = synthetic_coverage(rows_by_lane[synth], fams_by_lane[synth],
                             cells_by_lane[synth], args.synthetic_family_target)
    print(f"\n{synth}: capability-driven sample (target {cov['per_family_target']:,}/family)")
    print(f"  sources {cov['sources_selected']:,}  rows {cov['rows_selected']:,} "
          f"of {sum(rows_by_lane[synth].values()):,}")
    print(f"  coverage {cov['family_coverage']}")
    if cov["families_short"]:
        print(f"  SHORT: {cov['families_short']}")

    report = {
        "schema": "compose.editing_v2.v2_mixture_plan",
        "schema_version": 1,
        "status": "PLAN_EVIDENCE_ONLY_NO_AUTHORITY",
        "multistep_retention_curve": curve,
        "multistep_total_rows": total_rows,
        "multistep_total_new_scaffolds": total_new,
        "synthetic_capability_sample": cov,
        "take_whole": ["operator_aware_real_endpoint", "linker_positional_topology_analogue"],
        "skip": ["observed_local_analogue (uncompiled)"],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
