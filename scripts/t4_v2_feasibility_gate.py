"""Zero-oracle gate for the T4-v2 feasibility-conditioned proposal law.

Runs, for every source, three things and writes one artifact:

  1. BOTTLENECK ATTRIBUTION -- which single decision loses the probability mass, measured
     against a program-free witness search and against what v1 actually proposed.
  2. THE v1-vs-v2 FUNNEL -- executed / sim / QED / SA / all-three / eligible, at a matched
     endpoint budget, with the margin distributions.
  3. ABLATIONS -- a 2x2 over (headroom-conditioned proposal) x (margin-guided selection)
     that separates the move vocabulary from the conditioning, plus a per-component freeze
     of h that says which component carries the signal.

No docking call, no oracle call, no Modal launch.  Nothing is wired into production.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import t4_v2_feasibility_proposal as law
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
V1_AUDIT = ROOT / "diagnostics/t4_support_stage_audit_v1.json"
#: Independent attributions produced elsewhere in this campaign.  They are CONSUMED, not
#: trusted: where they disagree with the measurement here, both readings are recorded.
SIBLING_ATTRIBUTIONS = (
    ROOT / "diagnostics/t4_failure_counterfactual_v1.json",
    ROOT / "diagnostics/t4_region_law_counterfactual_v1.json",
)

#: Grouping labels only.  Nothing in the proposal law reads them, and no target or cell
#: identifier is ever a feature: the law sees a SMILES, a delta and three thresholds.
FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")
SANITY = ("parp1_0", "parp1_1", "parp1_2")

ARMS = {
    "v2_full": {"conditioned_proposal": True, "guided_selection": True},
    "v2_blind_proposal": {"conditioned_proposal": False, "guided_selection": True},
    "v2_unguided_selection": {"conditioned_proposal": True, "guided_selection": False},
    "v2_moveset_only": {"conditioned_proposal": False, "guided_selection": False},
}

FREEZES = ("qed", "sa", "sim", "size")


def load_v1() -> dict:
    return json.loads(V1_AUDIT.read_text())


def v1_sources(audit: dict) -> dict[str, str]:
    return {
        cell: payload["slot_semantics_preflight"]["smiles"]
        for cell, payload in audit["cells"].items()
    }


def v1_funnel(audit: dict, cell: str) -> dict:
    funnel = audit["cells"][cell]["funnel"]
    return {
        "endpoints": funnel["distinct_endpoints"],
        "chemically_valid": funnel["chemically_valid"],
        "reached_gate": funnel["capacity_valid"],
        "similarity_pass": funnel["similarity_pass"],
        "qed_pass": funnel["qed_pass"],
        "sa_pass": funnel["sa_pass"],
        "similarity_and_qed_pass": funnel["similarity_and_qed_pass"],
        "all_three_pass": funnel["all_three_pass"],
        "eligible": funnel["fully_eligible"],
    }


def v1_endpoint_sample(audit: dict, cell: str) -> list[dict]:
    payload = audit["cells"][cell]
    rows: list[dict] = []
    for key in (
        "best_qed_among_similarity_passing",
        "best_similarity_among_qed_passing",
        "pareto_front_sim_vs_qed",
        "eligible_endpoints",
    ):
        rows.extend(payload.get(key) or [])
    unique: dict[str, dict] = {}
    for row in rows:
        if row.get("smiles"):
            unique.setdefault(row["smiles"], row)
    return list(unique.values())


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 5)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 5),
        "p10": at(0.10),
        "median": at(0.50),
        "p90": at(0.90),
        "max": round(ordered[-1], 5),
        "mean": round(statistics.fmean(ordered), 5),
    }


def summarise_arm(result: dict) -> dict:
    rows = result["rows"]
    reached = [row for row in rows if row.get("reached_gate")]
    sim_pass = [row for row in reached if row["sim_ok"]]
    qed_pass = [row for row in reached if row["qed_ok"]]
    sa_pass = [row for row in reached if row["sa_ok"]]
    sim_qed = [row for row in sim_pass if row["qed_ok"]]
    eligible = result["eligible"]
    unique_eligible = {row["smiles"]: row for row in eligible}
    # The UNFILTERED benchmark pool, recomputed from the rows so the filter's cost is
    # visible in the same artifact that reports what it kept.
    unique_benchmark = {
        row["smiles"]: row for row in reached if row.get("benchmark_eligible")
    }
    pathological_benchmark = {
        smiles: row
        for smiles, row in unique_benchmark.items()
        if not row.get("chemistry_sane", True)
    }
    scaffolds = set()
    for smiles in unique_eligible:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
        except (ValueError, RuntimeError):
            pass
    return {
        "arm": result["arm"],
        "funnel": {
            "endpoints": len(rows),
            "chemically_valid": sum(1 for row in rows if row.get("parses")),
            "reached_gate": len(reached),
            "similarity_pass": len(sim_pass),
            "qed_pass": len(qed_pass),
            "sa_pass": len(sa_pass),
            "similarity_and_qed_pass": len(sim_qed),
            "all_three_pass": sum(1 for row in sim_qed if row["sa_ok"]),
            "eligible": len(unique_eligible),
        },
        "chemistry_filter": {
            "enabled": result["arm"].get("chemistry_filter"),
            "prune_pathological_parents": result["arm"].get("prune_pathological_parents"),
            "benchmark_eligible_distinct": len(unique_benchmark),
            "clean_eligible_distinct": len(unique_benchmark) - len(pathological_benchmark),
            "pathological_eligible_distinct": len(pathological_benchmark),
            "motif_census": result.get("pathological_motif_census", {}),
            "removed_examples": [
                {"smiles": smiles, "motifs": row["pathological_motifs"]}
                for smiles, row in list(pathological_benchmark.items())[:6]
            ],
        },
        "distinct_eligible_scaffolds": len(scaffolds),
        "gate_reconstruction_disagreements": result["gate_reconstruction_disagreements"],
        "steps": result["steps"],
        "policy_contexts": result["policy_contexts"],
        "mode_census": result["mode_census"],
        "mode_census_eligible": result["mode_census_eligible"],
        "margins": {
            "delta_sim": quantiles([row["delta_sim"] for row in reached]),
            "delta_qed": quantiles([row["delta_qed"] for row in reached]),
            "delta_sa": quantiles([row["delta_sa"] for row in reached]),
        },
        "eligible_quality": {
            "similarity": quantiles([row["similarity"] for row in unique_eligible.values()]),
            "qed": quantiles([row["qed"] for row in unique_eligible.values()]),
            "sa": quantiles([row["sa"] for row in unique_eligible.values()]),
            "heavy_delta": quantiles(
                [float(row["heavy_delta"]) for row in unique_eligible.values()]
            ),
        },
        "eligible_witnesses": [
            {
                "smiles": row["smiles"],
                "similarity": round(row["similarity"], 5),
                "qed": round(row["qed"], 5),
                "sa": round(row["sa"], 5),
                "heavy_delta": row["heavy_delta"],
                "mode": row.get("mode"),
            }
            for row in sorted(unique_eligible.values(), key=lambda row: -row["qed"])[:12]
        ],
    }


def run_cell(cell: str, smiles: str, audit: dict, *, budget: int, seed: int,
             ablate: bool, delta: float = 0.6, arms: tuple[str, ...] = tuple(ARMS),
             chemistry_filter: bool = True,
             prune_pathological_parents: bool = True,
             attribution: bool = True) -> dict:
    started = time.time()
    payload: dict = {
        "cell": cell,
        "source_smiles": smiles,
        # The OPERATIVE similarity corridor.  ``v1`` below is the delta=0.6 audit, so a
        # run at another delta must not be read as a paired before/after against it.
        "delta": delta,
        "v1_reference_delta": 0.6,
        "observed_status": audit["cells"][cell]["observed_status"],
        "source": audit["cells"][cell]["source"],
        "v1": v1_funnel(audit, cell),
        "arms": {},
        "h_component_freeze": {},
    }
    if attribution:
        payload["attribution"] = law.attribute_failure(
            smiles, delta, v1_endpoint_sample(audit, cell)
        )
    filter_kwargs = {
        "chemistry_filter": chemistry_filter,
        "prune_pathological_parents": prune_pathological_parents,
    }
    for name, configuration in ARMS.items():
        if name not in arms:
            continue
        result = law.run_search(
            smiles, delta, budget=budget, seed=seed, **configuration, **filter_kwargs
        )
        payload["arms"][name] = summarise_arm(result)
        payload.setdefault("slot_preflight", result["slot_preflight"])
    if ablate:
        for component in FREEZES:
            result = law.run_search(
                smiles, delta, budget=budget, seed=seed, frozen=(component,), **filter_kwargs
            )
            payload["h_component_freeze"][component] = summarise_arm(result)["funnel"]
    payload["elapsed_seconds"] = round(time.time() - started, 2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "reduce", "sensitivity", "fallback"))
    parser.add_argument("--cells", default="")
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--no-chemistry-filter",
        action="store_true",
        help="run the proposal law WITHOUT the pathological-motif filter (the baseline arm)",
    )
    parser.add_argument(
        "--keep-pathological-parents",
        action="store_true",
        help="filter what is EMITTED but still allow the search to grow out of a "
             "pathological intermediate",
    )
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.6,
        help="the OPERATIVE similarity corridor the fiber is built with",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", default="")
    # Explicit, because the shard directory may be shared with other work: a bare
    # ``*.json`` glob would silently fold a foreign artifact into this one.
    parser.add_argument("--pattern", default="t4v2_*.json")
    parser.add_argument("--sensitivity", default="")
    arguments = parser.parse_args()

    audit = load_v1()
    if arguments.stage == "fallback":
        sources = v1_sources(audit)
        payload = {
            "schema_version": "t4_v2_bounded_support_expansion_v1",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "oracle_calls": 0,
            "docking_calls": 0,
            "delta": arguments.delta,
            "seed": arguments.seed,
            "ladder": list(EXPANSION_LADDER),
            "k_required": EXPANSION_K,
            "cells": {},
        }
        for cell in (c for c in arguments.cells.split(",") if c):
            payload["cells"][cell] = expansion_fallback(
                cell, sources[cell], delta=arguments.delta, seed=arguments.seed
            )
            print(f"[{cell}] {json.dumps(payload['cells'][cell]['proposals_to_k_clean'])}", flush=True)
        Path(arguments.out).write_text(json.dumps(payload, indent=1, default=str) + "\n")
        return
    if arguments.stage == "sensitivity":
        sources = v1_sources(audit)
        cells = [cell for cell in arguments.cells.split(",") if cell]
        payload = normaliser_sensitivity(
            cells, sources, budget=arguments.budget, seed=arguments.seed
        )
        Path(arguments.out).write_text(json.dumps(payload, indent=1))
        print(json.dumps(payload["arms"], indent=1))
        return
    if arguments.stage == "run":
        sources = v1_sources(audit)
        cells = [cell for cell in arguments.cells.split(",") if cell]
        payload = {
            "schema_version": law.SCHEMA_VERSION,
            "budget": arguments.budget,
            "seed": arguments.seed,
            "delta": arguments.delta,
            "chemistry_filter": not arguments.no_chemistry_filter,
            "prune_pathological_parents": not arguments.keep_pathological_parents,
            "cells": {},
        }
        for cell in cells:
            payload["cells"][cell] = run_cell(
                cell,
                sources[cell],
                audit,
                budget=arguments.budget,
                seed=arguments.seed,
                ablate=arguments.ablate,
                delta=arguments.delta,
                arms=tuple(a for a in arguments.arms.split(",") if a),
                chemistry_filter=not arguments.no_chemistry_filter,
                prune_pathological_parents=not arguments.keep_pathological_parents,
                attribution=not arguments.no_attribution,
            )
            print(f"[{cell}] done", flush=True)
        Path(arguments.out).write_text(json.dumps(payload, indent=1, default=str))
        return

    merged: dict = {
        "schema_version": law.SCHEMA_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "oracle_calls": 0,
        "docking_calls": 0,
        "v1_reference": str(V1_AUDIT.relative_to(ROOT)),
        "cells": {},
    }
    matched = sorted(Path(arguments.shards).glob(arguments.pattern))
    if not matched:
        raise SystemExit(f"no shard matched {arguments.pattern!r} under {arguments.shards}")
    for path in matched:
        shard = json.loads(path.read_text())
        merged["budget"] = shard["budget"]
        merged["seed"] = shard["seed"]
        merged["cells"].update(shard["cells"])
    merged["shards"] = [path.name for path in matched]
    merged["independent_attribution_cross_check"] = cross_check_attributions(merged)
    if arguments.sensitivity and Path(arguments.sensitivity).exists():
        merged["normaliser_sensitivity"] = json.loads(Path(arguments.sensitivity).read_text())
    merged["groups"] = {"failed": list(FAILED), "controls": list(CONTROLS), "sanity": list(SANITY)}
    merged["gate"] = gate_verdict(merged)
    Path(arguments.out).write_text(json.dumps(merged, indent=1, default=str))
    print(report(merged))


def cross_check_attributions(merged: dict) -> dict:
    """Compare this run's attribution against the campaign's independent ones.

    Agreement is recorded as agreement, and disagreement is recorded as disagreement with
    both readings and the evidence that separates them.  Nothing here overrides a local
    measurement.
    """

    out: dict = {"sources": {}, "per_cell": {}}
    payloads: dict[str, dict] = {}
    for path in SIBLING_ATTRIBUTIONS:
        if path.exists():
            payloads[path.name] = json.loads(path.read_text())
            out["sources"][path.name] = payloads[path.name].get("schema_version")
    for cell, payload in merged["cells"].items():
        mine = payload["attribution"]
        row: dict = {
            "this_run": mine.get("attribution"),
            "this_run_witness": (mine.get("witness") or {}).get("smiles"),
            "this_run_v2_eligible": payload["arms"]["v2_full"]["funnel"]["eligible"],
        }
        counterfactual = payloads.get("t4_failure_counterfactual_v1.json", {})
        sibling = (counterfactual.get("cells") or {}).get(cell)
        if sibling:
            row["counterfactual_axis"] = sibling.get("attributed_axis")
            row["counterfactual_binding_gate"] = sibling.get("binding_gate")
            row["counterfactual_witness"] = (sibling.get("known_feasible_witness") or {}).get(
                "smiles"
            )
        region = payloads.get("t4_region_law_counterfactual_v1.json", {})
        region_cell = (region.get("cells") or {}).get(cell)
        if region_cell:
            arms = region_cell.get("arms", {})
            row["region_law_v1_best_rank"] = (arms.get("v1") or {}).get("best_eligible_rank")
            row["region_law_conditioned_best_rank"] = (arms.get("both") or {}).get(
                "best_eligible_rank"
            )
            row["region_law_best_smiles"] = (arms.get("both") or {}).get("best_eligible_smiles")
        out["per_cell"][cell] = row
    return out


#: The bounded support-expansion ladder, in ADDITIONAL distinct endpoint evaluations per
#: stage.  Cumulative ceiling 3,840; the trailing 4096 block is a diagnostic extension for
#: a cell still at literally zero support after the ceiling, never the default.
EXPANSION_LADDER = (256, 512, 1024, 2048)
EXPANSION_OVERRUN_BLOCK = 4096

#: Distinct CLEAN eligible endpoints required before the expansion stops.  Deliberately
#: above one: a cell must not qualify on a single boundary molecule.
EXPANSION_K = 4


def expansion_fallback(cell: str, smiles: str, *, delta: float, seed: int,
                       allow_overrun: bool = True) -> dict:
    """Cost of recovering K clean eligible endpoints by bounded proposal expansion.

    One search is run to the ceiling and the ladder is read off it, rather than restarting
    per stage: an expansion that threw away the previous stage's work would overstate the
    cost of every stage after the first.  ``proposals`` are DISTINCT endpoint evaluations,
    which is the same denominator the funnel uses, and no oracle call is made.
    """

    ceiling = sum(EXPANSION_LADDER)
    budget = ceiling + (EXPANSION_OVERRUN_BLOCK if allow_overrun else 0)
    started = time.time()
    result = law.run_search(smiles, delta, budget=budget, seed=seed)
    elapsed = time.time() - started

    # Cache insertion order is evaluation order, and one new SMILES is exactly one gate
    # call, so a row's position in that order IS the proposal count at which it was found.
    order = {row["smiles"]: index + 1 for index, row in enumerate(result["rows"])}
    clean = sorted({row["smiles"] for row in result["eligible"]}, key=lambda s: order[s])
    at = [order[s] for s in clean]

    def stage_of(count: int | None) -> str | None:
        if count is None:
            return None
        used = 0
        for index, block in enumerate(EXPANSION_LADDER):
            used += block
            if count <= used:
                return f"stage_{index + 1}_cum_{used}"
        return f"overrun_cum_{ceiling + EXPANSION_OVERRUN_BLOCK}"

    first = at[0] if at else None
    kth = at[EXPANSION_K - 1] if len(at) >= EXPANSION_K else None
    scaffolds = set()
    for endpoint in clean[: max(EXPANSION_K, 1)]:
        molecule = Chem.MolFromSmiles(endpoint)
        if molecule is None:
            continue
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
        except (ValueError, RuntimeError):
            pass
    return {
        "cell": cell,
        "delta": delta,
        "seed": seed,
        "ladder": list(EXPANSION_LADDER),
        "cumulative_ceiling": ceiling,
        "overrun_block": EXPANSION_OVERRUN_BLOCK if allow_overrun else 0,
        "proposals_evaluated": result["gate_calls"],
        "k_required": EXPANSION_K,
        "proposals_to_first_clean": first,
        "proposals_to_k_clean": kth,
        "stage_for_first_clean": stage_of(first),
        "stage_for_k_clean": stage_of(kth),
        "within_ceiling": bool(kth is not None and kth <= ceiling),
        "clean_eligible_total": len(clean),
        "distinct_scaffolds_in_first_k": len(scaffolds),
        "wall_seconds": round(elapsed, 1),
        "seconds_per_1k_proposals": round(1000.0 * elapsed / max(result["gate_calls"], 1), 1),
        "clean_yield_per_1k": round(1000.0 * len(clean) / max(result["gate_calls"], 1), 2),
        "gate_reconstruction_disagreements": result["gate_reconstruction_disagreements"],
        "benchmark_eligible_filtered_out": result["benchmark_eligible_filtered_out"],
        "first_k_witnesses": [
            {
                "smiles": endpoint,
                "found_at_proposal": order[endpoint],
                "qed": round(result["rows"][order[endpoint] - 1]["qed"], 4),
                "similarity": round(result["rows"][order[endpoint] - 1]["similarity"], 4),
                "sa": round(result["rows"][order[endpoint] - 1]["sa"], 3),
            }
            for endpoint in clean[:EXPANSION_K]
        ],
    }


def gate_verdict(merged: dict) -> dict:
    cells = merged["cells"]
    failures = []
    for cell in FAILED:
        if cell not in cells:
            continue
        eligible = cells[cell]["arms"]["v2_full"]["funnel"]["eligible"]
        if eligible < 2:
            failures.append({"cell": cell, "eligible": eligible, "reason": "fewer than two witnesses"})
    regressions = []
    for cell in CONTROLS + SANITY:
        if cell not in cells:
            continue
        before = cells[cell]["v1"]["eligible"]
        after = cells[cell]["arms"]["v2_full"]["funnel"]["eligible"]
        if after < before:
            regressions.append({"cell": cell, "v1": before, "v2": after})
    return {
        "criterion": "every failed source yields >=2 distinct eligible witnesses; no control regresses",
        "failed_sources_without_pool": failures,
        "controls_regressed": regressions,
        "passed": not failures and not regressions,
    }


def normaliser_sensitivity(
    cells: list[str], sources: dict[str, str], *, budget: int, seed: int,
    axes: tuple[str, ...] = ("qed", "sim"), factors: tuple[float, ...] = (0.5, 2.0)
) -> dict:
    """Does the result rest on the choice of margin normalisers?

    ``Z_SCALE`` sets the units the margin vector is compared in, so it is the one free
    choice in the scalarisation.  Halving and doubling an axis moves the point at which
    that constraint starts to dominate the Tchebycheff min; if the eligible pools survive
    both, the finding is not a knife-edge artefact of the units.
    """

    baseline = dict(law.Z_SCALE)
    out: dict = {"baseline": baseline, "budget": budget, "arms": {}}
    try:
        for axis in axes:
            for factor in factors:
                law.Z_SCALE[axis] = baseline[axis] * factor
                key = f"{axis}x{factor}"
                out["arms"][key] = {}
                for cell in cells:
                    result = law.run_search(sources[cell], 0.6, budget=budget, seed=seed)
                    out["arms"][key][cell] = len({row["smiles"] for row in result["eligible"]})
                law.Z_SCALE[axis] = baseline[axis]
    finally:
        law.Z_SCALE.clear()
        law.Z_SCALE.update(baseline)
    return out


def report(merged: dict) -> str:
    """The v1-vs-v2 funnel for every source, plus the ablations and the attribution."""

    cells = merged["cells"]
    order = [cell for cell in FAILED + CONTROLS + SANITY if cell in cells]
    lines: list[str] = []
    lines.append(f"T4-v2 feasibility-conditioned proposal law -- {merged['schema_version']}")
    lines.append(
        f"budget {merged['budget']} distinct endpoint evaluations per source per arm; "
        f"oracle calls {merged['oracle_calls']}; docking calls {merged['docking_calls']}"
    )
    lines.append("")
    lines.append("FUNNEL  (endpoints | sim | QED | SA | sim&QED | all three | eligible)")
    header = (
        f"{'source':9s} {'grp':5s} {'arm':6s} {'endpts':>7s} {'sim':>6s} {'QED':>6s} "
        f"{'SA':>6s} {'s&q':>6s} {'all3':>6s} {'elig':>6s} {'scaf':>5s} {'elig/1k':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for cell in order:
        payload = cells[cell]
        group = (
            "FAIL" if cell in FAILED else "ctrl" if cell in CONTROLS else "sane"
        )
        v1 = payload["v1"]
        lines.append(
            f"{cell:9s} {group:5s} {'v1':6s} {v1['endpoints']:7d} {v1['similarity_pass']:6d} "
            f"{v1['qed_pass']:6d} {v1['sa_pass']:6d} {v1['similarity_and_qed_pass']:6d} "
            f"{v1['all_three_pass']:6d} {v1['eligible']:6d} {'-':>5s}"
            f"{1000.0 * v1['eligible'] / max(v1['reached_gate'], 1):9.2f}"
        )
        for name in ("v2_full", "v2_blind_proposal", "v2_unguided_selection", "v2_moveset_only"):
            if name not in payload["arms"]:
                continue
            arm = payload["arms"][name]
            funnel = arm["funnel"]
            tag = {
                "v2_full": "v2",
                "v2_blind_proposal": "v2-hb",
                "v2_unguided_selection": "v2-ug",
                "v2_moveset_only": "v2-mv",
            }[name]
            lines.append(
                f"{'':9s} {'':5s} {tag:6s} {funnel['endpoints']:7d} {funnel['similarity_pass']:6d} "
                f"{funnel['qed_pass']:6d} {funnel['sa_pass']:6d} "
                f"{funnel['similarity_and_qed_pass']:6d} {funnel['all_three_pass']:6d} "
                f"{funnel['eligible']:6d} {arm['distinct_eligible_scaffolds']:5d}"
                f"{1000.0 * funnel['eligible'] / max(funnel['reached_gate'], 1):9.2f}"
            )
        lines.append("")
    lines.append("v2 = headroom-conditioned proposal + margin-guided selection")
    lines.append("v2-hb = h-BLIND proposal (one context), same selection")
    lines.append("v2-ug = conditioned proposal, UNGUIDED selection")
    lines.append("v2-mv = move vocabulary only (blind + unguided)")
    lines.append("")
    lines.append("ATTRIBUTION  (which decision loses the mass)")
    lines.append(
        f"{'source':9s} {'verdict':20s} {'wit d(heavy)':>12s} {'wit sim':>8s} "
        f"{'v1 best sim|QED ok':>19s} {'overlap@scale':>14s}"
    )
    for cell in order:
        evidence = cells[cell]["attribution"]
        witness = evidence.get("witness") or {}
        lines.append(
            f"{cell:9s} {evidence.get('attribution', '?'):20s} "
            f"{evidence.get('witness_heavy_delta', 0):12d} "
            f"{witness.get('similarity', float('nan')):8.3f} "
            f"{evidence.get('v1_max_similarity_among_qed_passing', float('nan')):19.3f} "
            f"{(evidence.get('v1_max_region_overlap_at_witness_scale') or float('nan')):14.3f}"
        )
    lines.append("")
    lines.append("h-COMPONENT FREEZE  (eligible count with that axis collapsed to a constant)")
    freeze_header = f"{'source':9s} {'v2 full':>8s}" + "".join(
        f"{('-' + name):>10s}" for name in FREEZES
    )
    lines.append(freeze_header)
    for cell in order:
        payload = cells[cell]
        if not payload.get("h_component_freeze"):
            continue
        row = f"{cell:9s} {payload['arms']['v2_full']['funnel']['eligible']:8d}"
        for name in FREEZES:
            frozen = payload["h_component_freeze"].get(name)
            row += f"{(frozen['eligible'] if frozen else -1):10d}"
        lines.append(row)
    lines.append("")
    lines.append(json.dumps(merged["gate"], indent=1))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
