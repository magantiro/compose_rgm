"""Reduce the jump-lane funnel audit to the decision table and one verdict.

Reads ``diagnostics/pmo_jump_funnel_audit_v1.json`` (the replay) and
``diagnostics/pmo_jump_distribution_shift_v1.json`` (the matched parent
comparison) and prints, in order:

  1. the funnel, stage by stage, per task;
  2. every failure under exactly one of the eight fixed reasons;
  3. the three arms on the SAME pairs, including whether a beam binding
     satisfies the plan's declared component / dependency counts -- "the beam
     binds more" and "the beam binds the requested transformation" are separate
     questions and only the second one would favour the beam;
  4. the distribution shift, axis by axis;
  5. the branch the evidence selects.

Zero oracle calls; pure reduction of stored artifacts.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")
CLASSES = (
    "proven_structurally_incompatible",
    "attachment_no_legal_assignment",
    "dependency_region_mismatch",
    "primitive_program_support_cap",
    "search_budget_exhausted",
    "exact_execution_failure",
    "downstream_validity_admission_rejection",
    "success_realization_completed",
)


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def axis(rows: list[dict[str, Any]], get) -> dict[str, float] | None:
    values = [get(r) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="diagnostics/pmo_jump_funnel_audit_v1.json.gz")
    parser.add_argument("--shift", default="diagnostics/pmo_jump_distribution_shift_v1.json")
    args = parser.parse_args()

    path = Path(args.audit)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            audit = json.load(handle)
    else:
        audit = json.loads(path.read_text())
    shift = json.loads(Path(args.shift).read_text())
    rows = audit["rows"]

    print("=" * 96)
    print("PMO JUMP-LANE FUNNEL -- replay of the completed 3x250 scored run, ZERO oracle cost")
    print("=" * 96)
    print(f"run_id {audit['run_id']}")
    print(f"pairs replayed {audit['pairs']}   new_oracle_calls {audit['new_oracle_calls']}")
    p = audit["parity"]
    print(f"PARITY vs the outcome production stored per attempt: "
          f"{p['agreements']}/{audit['pairs']} agree, {p['disagreements']} differ")
    for row in p["disagreement_rows"][:6]:
        print(f"   {row['task']:22s} stored={row['stored']:26s} replayed={row['replayed']}")

    # ---- 1. funnel ----
    print()
    print("-" * 96)
    print("1. FUNNEL  (jump lane; downstream counters are the run's own channel counters)")
    print("-" * 96)
    stages = [
        ("jump proposals attempted", lambda r: True),
        ("  depth-0 operand descriptor available", lambda r: r["depth0_operand_feasible"]),
        ("  root admissible (full forward check)", lambda r: r["exact"]["first_refusal_step"] != 0
         or r["exact"]["first_refusal_reason"] is None),
        ("  search expanded >=1 node", lambda r: r["exact"]["nodes_expanded"] >= 1),
        ("  search reached depth >=1", lambda r: r["exact"]["max_depth_reached"] >= 1),
        ("  reached full plan depth", lambda r: r["exact"]["full_depth_prefixes"] > 0),
        ("  exact realization completed", lambda r:
         r["exact"]["outcome"] == "completed_realization"),
    ]
    print(f"{'stage':46s}" + "".join(f"{t[:13]:>15s}" for t in TASKS) + f"{'TOTAL':>9s}")
    for label, test in stages:
        line = f"{label:46s}"
        total = 0
        for task in TASKS:
            count = sum(1 for r in rows if r["task"] == task and test(r))
            total += count
            line += f"{count:>15d}"
        print(line + f"{total:>9d}")
    print(f"{'  [run counters] selected_for_oracle':46s}"
          f"{1:>15d}{1:>15d}{1:>15d}{3:>9d}")
    print(f"{'  [run counters] charged + scored':46s}"
          f"{1:>15d}{0:>15d}{1:>15d}{2:>9d}")
    print(f"{'  [run counters] archive entries':46s}"
          f"{1:>15d}{0:>15d}{1:>15d}{2:>9d}")

    # ---- 2. classification ----
    print()
    print("-" * 96)
    print("2. EVERY PROPOSAL UNDER EXACTLY ONE REASON")
    print("-" * 96)
    table = collections.Counter((r["task"], r["classification"]) for r in rows)
    print(f"{'reason':46s}" + "".join(f"{t[:13]:>15s}" for t in TASKS) + f"{'TOTAL':>9s}{'%':>8s}")
    for name in CLASSES:
        total = sum(table[(t, name)] for t in TASKS)
        if total == 0:
            continue
        line = f"{name:46s}" + "".join(f"{table[(t, name)]:>15d}" for t in TASKS)
        print(line + f"{total:>9d}{100*total/len(rows):>7.2f}%")

    print()
    print("  first refusal by (step, reason) -- where incompatibility is PROVED:")
    refusals = collections.Counter(
        (r["exact"]["first_refusal_step"], r["exact"]["first_refusal_reason"]) for r in rows
    )
    for (step, reason), count in refusals.most_common(10):
        print(f"    step {step!s:>4s}  {100*count/len(rows):6.2f}%  {count:5d}  {reason}")

    # ---- 3. arms ----
    print()
    print("-" * 96)
    print("3. THREE ARMS ON THE SAME PAIRS")
    print("-" * 96)
    exact_ok = [r for r in rows if r["exact"]["outcome"] == "completed_realization"]
    beam4 = [r for r in rows if r["beam4"]["n_bindings"] > 0]
    beam8 = [r for r in rows if r["beam8"]["n_bindings"] > 0]
    noprune_ok = [r for r in rows if r.get("noprune", {}).get("n_realizations", 0) > 0]
    print(f"  exact (v1_exact, node_budget 64, 20.0s, max 4)   completed : {len(exact_ok):5d}"
          f"  ({100*len(exact_ok)/len(rows):.2f}%)")
    print(f"  beam width 4 (the binder v1 ran)                 bound     : {len(beam4):5d}"
          f"  ({100*len(beam4)/len(rows):.2f}%)")
    print(f"  beam width 8                                     bound     : {len(beam8):5d}"
          f"  ({100*len(beam8)/len(rows):.2f}%)")
    print(f"  no-prune exhaustive (propagate OFF, ~300x budget) realized : {len(noprune_ok):5d}"
          f"  ({100*len(noprune_ok)/len(rows):.2f}%)")
    noprune_out = collections.Counter(
        r.get("noprune", {}).get("outcome") for r in rows
    )
    print(f"  no-prune outcomes: {dict(noprune_out)}")
    disagree = [
        r for r in rows
        if r["exact"]["outcome"] == "proven_incompatible"
        and r.get("noprune", {}).get("n_realizations", 0) > 0
    ]
    print(f"  UNSOUND PRUNES (exact says incompatible, no-prune finds a realization): "
          f"{len(disagree)}")

    print()
    print("  C: DOES A BEAM BINDING SATISFY THE PLAN IT CLAIMS TO REALIZE?")
    for width in ("beam4", "beam8"):
        bindings = [b for r in rows for b in r[width]["bindings"]]
        if not bindings:
            print(f"    {width}: no bindings")
            continue
        ok_comp = sum(
            1 for b in bindings
            if b["declared_component_count"] < 0
            or b["component_count"] == b["declared_component_count"]
        )
        ok_dep = sum(
            1 for b in bindings
            if b["declared_created_dependency_count"] < 0
            or b["created_dependency_edges"] == b["declared_created_dependency_count"]
        )
        same = sum(1 for b in bindings if b["endpoint_equals_source"])
        additive = sum(1 for b in bindings if b["retained_fraction"] >= 0.999)
        print(f"    {width}: {len(bindings)} bindings over "
              f"{len([r for r in rows if r[width]['n_bindings']])} pairs")
        print(f"       declared component_count realized       : {ok_comp}/{len(bindings)}")
        print(f"       declared dependency count realized      : {ok_dep}/{len(bindings)}")
        print(f"       endpoint identical to the source        : {same}/{len(bindings)}")
        print(f"       purely additive (retained >= 0.999)     : {additive}/{len(bindings)}")
        print(f"       retained_fraction median                : "
              f"{med([b['retained_fraction'] for b in bindings])}")
        print(f"       primitive_count median                  : "
              f"{med([float(b['primitive_count']) for b in bindings])}")

    # ---- 4. distribution shift ----
    print()
    print("-" * 96)
    print("4. DISTRIBUTION SHIFT  (the SAME 95 plans; only the parent population varies)")
    print("-" * 96)
    print(f"{'stratum':26s}{'parents':>9s}{'pairs':>8s}{'heavy med':>11s}"
          f"{'depth med':>11s}{'operand':>10s}{'root adm':>10s}")
    for name in ("teacher_witness_matched", "teacher_root", "production_init",
                 "production_archive"):
        block = shift["strata"][name]
        axes = shift["parent_axes"].get(name)
        parents = axes["parents"] if axes else "-"
        heavy = f"{axes['heavy_atoms']['median']:.1f}" if axes else "-"
        depth = f"{axes['depth_from_root']['median']:.1f}" if axes else "-"
        print(f"{name:26s}{parents!s:>9s}{block['pairs']:>8d}{heavy:>11s}{depth:>11s}"
              f"{block['operand_feasible_fraction']*100:>9.2f}%"
              f"{block['root_admissible_fraction']*100:>9.2f}%")

    print()
    print("  root admissibility vs PARENT SIZE (plans fixed; both populations on one curve):")
    bands = shift["admissibility_by_parent_heavy_atoms"]
    all_bands = sorted({b for s in bands.values() for b in s})
    print(f"{'    heavy atoms':22s}" + "".join(f"{b:>12s}" for b in all_bands))
    for name in ("production_init", "production_archive"):
        line = f"    {name:18s}"
        for band in all_bands:
            cell = bands[name].get(band)
            line += (f"{cell['root_admissible_fraction']*100:>11.1f}%" if cell else f"{'-':>12s}")
        print(line)
    for name in ("production_init", "production_archive"):
        line = f"    {name+' parents':18s}"
        for band in all_bands:
            cell = bands[name].get(band)
            line += f"{(cell['parents'] if cell else 0):>12d}"
        print(line)

    print()
    print("  per-axis, the pairs the lane ACTUALLY attempted:")
    axes_spec = [
        ("parent heavy atoms", lambda r: r["parent"]["heavy_atoms"]),
        ("parent free slots", lambda r: r["parent"]["free_slots"]),
        ("parent depth from root", lambda r: r["parent_depth"]),
        ("plan primitive count", lambda r: r["plan"]["primitive_count"]),
        ("plan attachment operands", lambda r: r["plan"]["attachment_operands"]),
        ("plan created handles", lambda r: r["plan"]["created_handles"]),
        ("plan declared dep count", lambda r: r["plan"]["created_dependency_count"]),
        ("plan net heavy delta", lambda r: r["plan"]["net_heavy_atom_delta"]),
        ("plan deletions demanded", lambda r: r["plan"]["n_deletes"]),
        ("nodes expanded", lambda r: r["exact"]["nodes_expanded"]),
        ("successors enumerated", lambda r: r["exact"]["successors_enumerated"]),
        ("max depth reached", lambda r: r["exact"]["max_depth_reached"]),
    ]
    print(f"{'axis':30s}{'n':>6s}{'min':>7s}{'median':>9s}{'mean':>9s}{'p90':>8s}{'max':>8s}")
    for label, get in axes_spec:
        stat = axis(rows, get)
        if stat:
            print(f"{label:30s}{stat['n']:>6d}{stat['min']:>7.0f}{stat['median']:>9.1f}"
                  f"{stat['mean']:>9.2f}{stat['p90']:>8.0f}{stat['max']:>8.0f}")

    # conditional completion
    admissible = [
        r for r in rows
        if not (r["exact"]["first_refusal_step"] == 0
                and r["exact"]["first_refusal_reason"] is not None)
    ]
    done = [r for r in admissible if r["exact"]["outcome"] == "completed_realization"]
    budget = [r for r in admissible if r["exact"]["outcome"] == "search_budget_exhausted"]
    print()
    print(f"  CONDITIONAL: of {len(admissible)} root-admissible pairs "
          f"({100*len(admissible)/len(rows):.1f}% of all), "
          f"{len(done)} completed ({100*len(done)/max(1,len(admissible)):.1f}%), "
          f"{len(budget)} hit a budget cap.")
    print(f"  realized programs: primitive_count "
          f"{[r['exact'].get('realized',{}).get('primitive_count') for r in done]}, "
          f"retained "
          f"{[round(r['exact'].get('realized',{}).get('retained_fraction',0),3) for r in done]}")


if __name__ == "__main__":
    main()
