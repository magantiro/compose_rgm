#!/usr/bin/env python
"""ZERO-ORACLE 2x2: completion SCALE x completion CONTENT.

                    | simple linear CNO content | structured component content
    old scale (<=8) | v1_current                | content_only
    expanded scale  | scale_only                | scale_content

Every arm runs the SAME production synthesis entry points the PMO campaign runs
(``synthesize_dynamic_program`` for the shallow lane, ``synthesize_structured_
program`` for the structured lane) over the SAME parents with the SAME seeds.
The arms differ only in the ``completion_law`` contract value; the absent field
is v1 verbatim.

Parents are the frozen PMO initialization states, decoded with ``decode_state``
from the initialization lock -- never re-parsed from SMILES, because a SMILES
round trip returns a TIGHT graph and PMO proposal states are 48 slots.

NO ORACLE IS CONSTRUCTED OR CALLED.
"""

from __future__ import annotations

import ast
import collections
import json
import os
import statistics
import sys
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from compose_v4.rewrite.trace_shard import decode_state  # noqa: E402
from compose_v4.experiments.global_delta_census import (  # noqa: E402
    compute_global_delta,
)
from compose_v4.control.completion_law_contract import (  # noqa: E402
    completion_law_identity,
    resolve_completion_law,
)
from compose_v4.control.dynamic_program_synthesis import (  # noqa: E402
    synthesize_dynamic_program,
)
from compose_v4.control.dynamic_program_synthesis_v2 import (  # noqa: E402
    synthesize_structured_program,
)

INIT = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
OUT = "diagnostics/pmo_completion_repair_v1"
ARMS = {
    "v1_current": None,
    "content_only": "content_only_v1",
    "scale_only": "scale_only_v1",
    "scale_content": "scale_content_v1",
}


def parents() -> list[tuple[str, MolecularGraph]]:
    lock = json.load(open(INIT))
    out = []
    for row in lock["candidates"]:
        payload = (
            ast.literal_eval(row["state"])
            if isinstance(row["state"], str)
            else row["state"]
        )
        out.append((row["endpoint"], decode_state(payload)))
    return out


def global_delta(source_smiles: str, product_smiles: str) -> dict | None:
    """The CENSUS's own instrument, imported rather than transcribed.

    Using ``compute_global_delta`` is what makes these arms directly comparable
    to the productive-transition census (median largest coherent changed region
    14, median retention 0.40) instead of to a re-implementation of it.
    """

    delta = compute_global_delta(source_smiles, product_smiles)
    if delta.status != "ok":
        return None
    return {
        "largest_changed_region": int(delta.largest_changed_region),
        "excised_atoms": int(delta.excised_atoms),
        "installed_atoms": int(delta.installed_atoms),
        "retained_fraction_source": float(delta.retained_fraction_source),
        "installed_has_ring": bool(delta.installed_region_has_ring),
        "installed_branch_points": int(delta.installed_branch_points),
        "installed_expressible_as_grow_chain": bool(
            delta.installed_expressible_as_grow_chain
        ),
        "n_anchor_groups": int(delta.n_anchor_groups),
        "multi_region": bool(delta.multi_region),
        "total_changed_atoms": int(delta.total_changed_atoms),
        "heavy_delta": int(delta.delta_heavy),
        "mcs_timed_out": bool(delta.mcs_timed_out),
    }


def run_arm(name: str, declared, pool, draws: int) -> dict:
    law = resolve_completion_law(declared)
    rows, failures = [], collections.Counter()
    endpoints: set[str] = set()
    began = perf_counter()
    for parent_index, (parent_smiles, graph) in enumerate(pool):
        for draw in range(draws):
            for lane in ("shallow", "structured"):
                seed = 7_000_003 * parent_index + 10_007 * draw + (0 if lane == "shallow" else 1)
                rng = np.random.default_rng(seed)
                try:
                    if lane == "shallow":
                        _, _program, _binding, trace, detail = synthesize_dynamic_program(
                            graph, rng, max_modules=3, max_primitives=32,
                            max_blocks=8, completion_law=law,
                        )
                    else:
                        _, _program, _binding, trace, detail = synthesize_structured_program(
                            graph, rng, max_modules=3, max_primitives=32,
                            max_blocks=8, completion_law=law,
                        )
                except Exception as error:  # noqa: BLE001 - census of refusals
                    failures[f"{lane}:{type(error).__name__}"] += 1
                    continue
                endpoint = trace["endpoint"]
                endpoints.add(endpoint)
                delta = global_delta(parent_smiles, endpoint)
                if delta is None:
                    failures[f"{lane}:mcs"] += 1
                    continue
                families = [m["family"] for m in detail.get("modules", [])]
                rows.append({
                    "lane": lane, "parent": parent_index,
                    "endpoint": endpoint, "families": families,
                    "primitives": len(trace["actions"]),
                    **delta,
                })
    elapsed = perf_counter() - began

    def frac(predicate):
        return round(sum(1 for r in rows if predicate(r)) / max(1, len(rows)), 4)

    def med(key):
        values = [r[key] for r in rows]
        return round(statistics.median(values), 4) if values else None

    attempts = len(pool) * draws * 2
    completion_rows = [
        r for r in rows if set(r["families"]) & {"segment_replace", "segment_grow"}
    ]

    def cfrac(predicate):
        return round(
            sum(1 for r in completion_rows if predicate(r))
            / max(1, len(completion_rows)),
            4,
        )

    def cmed(key):
        values = [r[key] for r in completion_rows]
        return round(statistics.median(values), 4) if values else None

    conditional = {
        "n": len(completion_rows),
        "share_of_executed": round(len(completion_rows) / max(1, len(rows)), 4),
        "median_largest_changed_region": cmed("largest_changed_region"),
        "frac_largest_region_ge_6": cfrac(lambda r: r["largest_changed_region"] >= 6),
        "frac_largest_region_ge_13": cfrac(lambda r: r["largest_changed_region"] >= 13),
        "median_installed_atoms": cmed("installed_atoms"),
        "median_excised_atoms": cmed("excised_atoms"),
        "median_retained_fraction_source": cmed("retained_fraction_source"),
        "frac_installs_ring": cfrac(lambda r: r["installed_has_ring"]),
        "frac_installs_branch": cfrac(lambda r: r["installed_branch_points"] > 0),
        "frac_installed_expressible_as_grow_chain": cfrac(
            lambda r: r["installed_expressible_as_grow_chain"]
        ),
    }
    return {
        "conditional_on_a_completion_module": conditional,
        "family_counts": dict(
            collections.Counter(f for r in rows for f in r["families"]).most_common()
        ),
        "arm": name,
        "declared": declared,
        "law_identity": completion_law_identity(law),
        "attempts": attempts,
        "executed": len(rows),
        "executed_yield": round(len(rows) / attempts, 4),
        "distinct_endpoints": len(endpoints),
        "diversity": round(len(endpoints) / max(1, len(rows)), 4),
        "median_largest_changed_region": med("largest_changed_region"),
        "mean_largest_changed_region": round(
            statistics.mean(r["largest_changed_region"] for r in rows), 3
        ) if rows else None,
        "frac_largest_region_ge_6": frac(lambda r: r["largest_changed_region"] >= 6),
        "frac_largest_region_ge_13": frac(lambda r: r["largest_changed_region"] >= 13),
        "median_retained_fraction_source": med("retained_fraction_source"),
        "median_installed_atoms": med("installed_atoms"),
        "median_excised_atoms": med("excised_atoms"),
        "frac_installs_ring": frac(lambda r: r["installed_has_ring"]),
        "frac_installs_branch": frac(lambda r: r["installed_branch_points"] > 0),
        "frac_installed_expressible_as_grow_chain": frac(
            lambda r: r["installed_expressible_as_grow_chain"]
        ),
        "frac_multi_region": frac(lambda r: r["multi_region"]),
        "median_anchor_groups": med("n_anchor_groups"),
        "median_total_changed_atoms": med("total_changed_atoms"),
        "median_primitives": med("primitives"),
        "median_heavy_delta": med("heavy_delta"),
        "seconds": round(elapsed, 1),
        "failures": dict(failures.most_common(12)),
        "rows": rows,
    }


def main() -> None:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    selected = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ARMS)
    pool = parents()
    os.makedirs(OUT, exist_ok=True)
    report = {
        "schema_version": "pmo_completion_2x2_v1",
        "new_oracle_calls": 0,
        "kernel": "rdkit 2023.09.6 (PMO production pin)",
        "initialization": INIT,
        "parents": len(pool),
        "draws_per_parent_per_lane": draws,
        "arms": {},
    }
    for name in selected:
        declared = ARMS[name]
        result = run_arm(name, declared, pool, draws)
        rows = result.pop("rows")
        with open(os.path.join(OUT, f"rows_{name}.json"), "w") as handle:
            json.dump(rows, handle)
        report["arms"][name] = result
        print(
            f"{name:15s} exec={result['executed']:5d}/{result['attempts']:<5d} "
            f"region_med={result['median_largest_changed_region']} "
            f">=6={result['frac_largest_region_ge_6']:.3f} "
            f">=13={result['frac_largest_region_ge_13']:.3f} "
            f"ret={result['median_retained_fraction_source']} "
            f"ins={result['median_installed_atoms']} "
            f"ring={result['frac_installs_ring']:.3f} "
            f"branch={result['frac_installs_branch']:.3f} "
            f"div={result['diversity']:.3f} ({result['seconds']}s)",
            flush=True,
        )
    tag = "_".join(selected) if len(selected) < len(ARMS) else "all"
    path = os.path.join(OUT, f"two_by_two_v1_{tag}.json")
    with open(path, "w") as handle:
        json.dump(report, handle, sort_keys=True, indent=1)
    print("wrote", path)


main()
