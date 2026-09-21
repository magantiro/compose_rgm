"""What SHAPE do compiled teacher routes have? A prerequisite census, zero oracle calls.

The route-prototype design trains on SEMANTIC structure -- retained region, released
region, replacement topology, interface, ring add/remove, size change -- rather than on
literal atom-index scripts. Those features are only informative if the routes actually
RETAIN something. This census measures that before any prototype is fitted.

TWO CANDIDATE ROUTE SOURCES, and they differ categorically.

1. `compile_source_to_target` (`rewrite/compiler.py`), the only compiler that accepts an
   ARBITRARY pair, is `delete_to_null_then_construct_v1`: it deletes the source through
   the inverse of a legal construction, reaches the formal NULL STATE, and then builds
   the target. Its own docstring says it "is intentionally not an edit-minimal
   alignment". Every semantic feature above is therefore degenerate on its output --
   retained region empty, released region everything -- no matter how similar the pair.

2. `diagnostics/ivg_winner_paths/pairs/*.json.gz`, the corpus the sibling route-prior
   work admits, holds SEARCH-FOUND witness routes that never pass through null.

This matters because the repository has already recorded the failure mode once:
`real_endpoint_multistep_path` is a demolish-then-rebuild decomposition of A->B whose
family share moves monotonically from `atom_delete` early to `atom_insert` late, and the
standing note is "a compiler artifact, not chemistry -- do not train on it as-is."
Fitting route prototypes on source-to-target output would reproduce exactly that.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import statistics
from pathlib import Path

import numpy as np
from rdkit import RDLogger

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.compiler import compile_source_to_target
from compose_v4.rewrite.kernel import default_rewrite_system
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.trace_shard import decode_state

RDLogger.DisableLog("rdApp.*")

#: Ordinary drug-like pairs. The first three differ by a SINGLE substituent, which is the
#: strongest possible test: if a one-methyl change still demolishes the molecule, the
#: compiler's route shape is independent of how close the pair is.
GENERIC_PAIRS = (
    ("CC(=O)Nc1ccc(O)cc1", "CC(=O)Nc1ccc(OC)cc1"),
    ("O=C(NC1CCNCC1)c1ccccc1", "O=C(NC1CCN(C)CC1)c1ccccc1"),
    ("COc1ccc(CCN)cc1", "COc1ccc(CCNC)cc1"),
    ("CC(=O)Nc1ccc(O)cc1", "O=C(NC1CCNCC1)c1ccccc1"),
)
WITNESS_GLOB = "diagnostics/ivg_winner_paths/pairs/*.json.gz"


def census_generic_compiler(seed: int = 11) -> dict:
    system = default_rewrite_system()
    rng = np.random.default_rng(seed)
    rows = []
    for source_smiles, target_smiles in GENERIC_PAIRS:
        g0 = smiles_to_molecular_graph(source_smiles)
        g1 = smiles_to_molecular_graph(target_smiles)
        slots = max(g0.n_atoms, g1.n_atoms) + 8
        try:
            trace = compile_source_to_target(
                pad_molecular_graph(g0, slots),
                pad_molecular_graph(g1, slots),
                system=system,
                rng=rng,
            )
        except Exception as error:  # noqa: BLE001 - a refusal is a result here
            rows.append({"source": source_smiles, "target": target_smiles,
                         "status": f"{type(error).__name__}: {error}"})
            continue
        current = pad_molecular_graph(g0, slots)
        sizes = [current.n_real_atoms]
        for step in trace.steps:
            current = execute_trace(current, (step,), system=system)
            sizes.append(current.n_real_atoms)
        rows.append({
            "source": source_smiles,
            "target": target_smiles,
            "status": "compiled",
            "steps": len(trace.steps),
            "source_heavy_atoms": g0.n_real_atoms,
            "min_heavy_atoms_on_route": min(sizes),
            "retained_fraction": round(min(sizes) / max(1, g0.n_real_atoms), 4),
            "passes_through_null": min(sizes) == 0,
            "compiler": trace.metadata.get("compiler"),
        })
    compiled = [r for r in rows if r["status"] == "compiled"]
    return {
        "compiler": "compile_source_to_target",
        "pairs": len(rows),
        "compiled": len(compiled),
        "retained_fraction_median": (
            round(statistics.median(r["retained_fraction"] for r in compiled), 4)
            if compiled else None
        ),
        "routes_through_null": sum(r["passes_through_null"] for r in compiled),
        "rows": rows,
    }


def census_witness_corpus(pattern: str = WITNESS_GLOB) -> dict:
    files = sorted(glob.glob(pattern))
    retained, steps, families = [], [], collections.Counter()
    targets, ring_changed, statuses = set(), 0, collections.Counter()
    for path in files:
        record = json.loads(gzip.decompress(Path(path).read_bytes()).decode())
        route = record["payload"]["path"]
        statuses[route["status"]] += 1
        if route["status"] != "witness_found":
            continue
        source_heavy = route["source_stats"]["heavy_atoms"]
        sizes = [decode_state(state).n_real_atoms for state in route["states"]]
        retained.append(min(sizes) / max(1, source_heavy))
        steps.append(len(route["actions"]))
        for action in route["actions"]:
            families[action["executor_rule"]] += 1
        targets.add(route["target_2d"])
        if route["source_stats"]["cycle_rank"] != route["target_stats"]["cycle_rank"]:
            ring_changed += 1
    return {
        "corpus": pattern,
        "files": len(files),
        "statuses": dict(statuses),
        "witness_routes": len(retained),
        "distinct_targets": len(targets),
        "retained_fraction": {
            "min": round(min(retained), 4),
            "median": round(statistics.median(retained), 4),
            "mean": round(statistics.mean(retained), 4),
            "max": round(max(retained), 4),
        } if retained else None,
        "routes_through_null": sum(1 for value in retained if value == 0.0),
        "steps": {
            "min": min(steps), "median": statistics.median(steps), "max": max(steps)
        } if steps else None,
        "ring_rank_changing_routes": ring_changed,
        "families": dict(families.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    generic = census_generic_compiler()
    witness = census_witness_corpus()
    verdict = (
        "GENERIC_COMPILER_UNUSABLE_FOR_SEMANTIC_ROUTES"
        if generic["compiled"] and generic["routes_through_null"] == generic["compiled"]
        else "GENERIC_COMPILER_RETAINS_STRUCTURE"
    )
    report = {
        "schema_version": "pmo_route_corpus_shape_census_v1",
        "oracle_calls": 0,
        "generic_pair_compiler": generic,
        "witness_corpus": witness,
        "verdict": verdict,
    }
    print(f"generic compiler: {generic['compiled']}/{generic['pairs']} compiled, "
          f"retained_fraction median {generic['retained_fraction_median']}, "
          f"{generic['routes_through_null']} through null")
    print(f"witness corpus:   {witness['witness_routes']} routes, "
          f"retained_fraction median {witness['retained_fraction']['median']}, "
          f"{witness['routes_through_null']} through null, "
          f"{witness['ring_rank_changing_routes']} change ring rank")
    print(f"VERDICT: {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
