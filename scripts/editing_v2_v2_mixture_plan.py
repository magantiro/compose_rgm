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
SOURCE_CELLS: dict = collections.defaultdict(set)


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
    """Per-lane rows/families per source and capability cells, both sides.

    The compiled side matters: synthetic selection must be INCREMENTAL against
    coverage the corpus already has.  A fresh per-family quota would re-buy
    capability that is already present -- the compiled corpus already holds
    ~9.8k cycle_insert and ~6.8k cycle_attach examples.
    """

    per_lane_source_rows = collections.defaultdict(collections.Counter)
    per_lane_source_fams = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    per_lane_cells = collections.defaultdict(collections.Counter)
    compiled_fams: collections.Counter = collections.Counter()
    compiled_cells: collections.Counter = collections.Counter()
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
                compiled_fams[record["model_family"]] += 1
                compiled_cells[record.get("capability_cell_id", "?")] += 1
                continue
            family = record["model_family"]
            per_lane_source_rows[lane][source] += 1
            per_lane_source_fams[lane][source][family] += 1
            per_lane_cells[lane][record.get("capability_cell_id", "?")] += 1
            SOURCE_CELLS[(lane, source)].add(record.get("capability_cell_id", "?"))
    return (per_lane_source_rows, per_lane_source_fams, per_lane_cells,
            compiled_sources, compiled_fams, compiled_cells)


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


def incremental_synthetic_sweep(source_rows, source_fams, source_cells,
                                compiled_fams, compiled_cells, all_cells,
                                real_rows_existing, real_rows_new, synthetic_rows_existing,
                                targets):
    """How much ADDITIONAL synthetic is needed to close capability gaps.

    Not a per-family quota: the compiled corpus already holds ~9.8k cycle_insert
    and ~6.8k cycle_attach, so a fresh quota re-buys coverage that exists.

    CAPABILITY CELLS ARE NOT THE BINDING CRITERION.  Measured: the compiled
    corpus already covers 19 of 21 cells and the uncompiled synthetic adds only
    2 more, saturating below the smallest target swept.  What is actually scarce
    is the RAREST FAMILY -- `ring_system_restate` sits at 623 compiled -- so the
    target is a floor on the rarest family's combined old+new count, and sources
    are ranked by how much they advance whichever family is currently furthest
    below that floor.

    An earlier version ranked by total cycle-family density while stopping on
    `cycle_insert` alone, so cycle_attach-heavy sources sorted first without
    advancing the stop condition and every row count was inflated.
    """

    covered0 = set(compiled_cells)
    out = []
    for target in targets:
        covered = set(covered0)
        got: collections.Counter = collections.Counter()
        chosen: list[str] = []
        families = set(compiled_fams) | {f for s in source_fams for f in source_fams[s]}

        def deficit(counter):
            return {f: max(0, target - (compiled_fams.get(f, 0) + counter.get(f, 0)))
                    for f in families}

        # Rank by unseen cells first, then by how much a source advances the
        # families currently furthest below the floor -- aligned with the stop
        # condition rather than a proxy for it.
        short0 = {f for f, d in deficit(collections.Counter()).items() if d > 0}
        order = sorted(
            source_fams,
            key=lambda s: (-len(source_cells.get(s, ()) - covered0),
                           -sum(v for f, v in source_fams[s].items() if f in short0),
                           sum(source_fams[s].values())),
        )
        for source in order:
            if not any(d > 0 for d in deficit(got).values()):
                break
            chosen.append(source)
            got.update(source_fams[source])
            covered |= source_cells.get(source, set())
        rows = int(sum(source_rows[s] for s in chosen))
        total_real = real_rows_existing + real_rows_new
        total_synth = synthetic_rows_existing + rows
        combined = {f: int(compiled_fams.get(f, 0) + got.get(f, 0))
                    for f in set(compiled_fams) | set(got)}
        out.append({
            "rarest_family_floor": target,
            "sources_selected": len(chosen),
            "rows_selected": rows,
            "synthetic_share_including_existing": total_synth / max(total_real + total_synth, 1),
            "capability_cells_covered": len(covered),
            "capability_cells_total": len(all_cells),
            "new_cells_vs_compiled": len(covered - covered0),
            "cells_still_uncovered": len(all_cells - covered),
            "family_totals_existing_plus_new": dict(sorted(combined.items(), key=lambda kv: -kv[1])),
            "rarest_family_count": min(combined.values()) if combined else 0,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--prep-subset", default="configs/process_v2_prep_subset.json")
    ap.add_argument("--multistep-cap", type=int, default=1)
    ap.add_argument("--sweep", default="1000,2000,3000,5000,8000")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.prep_subset).read_text())
    compiled = {c["task_identity_sha256"] for c in plan["chunks"]}
    (rows_by_lane, fams_by_lane, cells_by_lane,
     compiled_sources, compiled_fams, compiled_cells) = collect(Path(args.active8_root), compiled)

    scaffold = _scaffold_fn()
    cache: dict[str, str | None] = {}

    def scaffold_of(s: str):
        if s not in cache:
            cache[s] = scaffold(s)
        return cache[s]

    compiled_scaffolds = {s for s in (scaffold_of(k) for k in compiled_sources) if s}
    lane = "real_endpoint_multistep_path"
    curve, total_rows, total_new = retention_curve(
        rows_by_lane[lane], fams_by_lane[lane], scaffold_of, compiled_scaffolds
    )
    print(f"{lane}: {total_rows:,} rows, {total_new:,} new scaffolds")
    print(f"{'cap':>4} {'rows':>10} {'row%':>6} {'new scaffolds':>14} {'new%':>6}")
    for row in curve:
        print(f"{row['cap']:>4} {row['retained_rows']:>10,} "
              f"{100 * row['retained_row_fraction']:>5.1f}% {row['retained_new_scaffolds']:>14,} "
              f"{100 * row['new_scaffold_fraction']:>5.1f}%")

    chosen_cap = next(r for r in curve if r["cap"] == args.multistep_cap)
    real_new = (chosen_cap["retained_rows"]
                + sum(rows_by_lane["operator_aware_real_endpoint"].values())
                + sum(rows_by_lane["linker_positional_topology_analogue"].values()))
    real_existing = sum(
        sum(rows_by_lane[k].values()) for k in rows_by_lane if k != "reversible_synthetic_walk"
    ) * 0  # compiled-side real rows come from the census, not the uncompiled tallies
    census_path = Path("diagnostics/editing_v2_corpus_census.json")
    real_existing = 0
    synth_existing = 0
    if census_path.exists():
        cen = json.loads(census_path.read_text())
        for e in cen["lanes"]:
            if e["lane"] == "reversible_synthetic_walk":
                synth_existing += e["compiled"]["rows"]
            else:
                real_existing += e["compiled"]["rows"]

    synth = "reversible_synthetic_walk"
    all_cells = set(compiled_cells) | set(cells_by_lane[synth])
    src_cells = {s: SOURCE_CELLS[(synth, s)] for s in fams_by_lane[synth]}
    targets = [int(x) for x in args.sweep.split(",")]
    sweep = incremental_synthetic_sweep(
        rows_by_lane[synth], fams_by_lane[synth], src_cells,
        compiled_fams, compiled_cells, all_cells,
        real_existing, real_new, synth_existing, targets,
    )
    print(f"\nINCREMENTAL synthetic sweep (real: {real_existing:,} existing + {real_new:,} new; "
          f"synthetic already compiled: {synth_existing:,})")
    print(f"{'floor':>9} {'sources':>8} {'rows':>9} {'synth%':>7} {'cells':>7} {'new cells':>10} "
          f"{'uncov':>6} {'rarest fam':>11}")
    for row in sweep:
        print(f"{row['rarest_family_floor']:>9,} {row['sources_selected']:>8,} "
              f"{row['rows_selected']:>9,} {100 * row['synthetic_share_including_existing']:>6.1f}% "
              f"{row['capability_cells_covered']:>7} {row['new_cells_vs_compiled']:>10} "
              f"{row['cells_still_uncovered']:>6} {row['rarest_family_count']:>11,}")

    report = {
        "schema": "compose.editing_v2.v2_mixture_plan",
        "schema_version": 2,
        "status": "PLAN_EVIDENCE_ONLY_NO_AUTHORITY",
        "multistep_retention_curve": curve,
        "multistep_cap_selected": args.multistep_cap,
        "real_rows_existing": real_existing,
        "real_rows_new": real_new,
        "synthetic_rows_existing": synth_existing,
        "incremental_synthetic_sweep": sweep,
        "take_whole": ["operator_aware_real_endpoint", "linker_positional_topology_analogue"],
        "skip": ["observed_local_analogue (uncompiled)"],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
