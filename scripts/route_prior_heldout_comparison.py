"""The held-out comparison: teacher-route prior vs a generic structural control.

Two parent populations, and they answer different questions:

``t4_heldout``   the fifteen T4 lead-optimization sources, each scored under the
                 prior fitted WITHOUT it (leave-source-out). This is the
                 in-domain transfer test -- same kind of molecule, unseen
                 instance.
``generic``      ZINC-derived drug-like molecules from the held-out h_phi
                 validation split, scored under the train-on-all prior. No
                 generic molecule appears in any teacher route, so there is
                 nothing to leave out; this is the OUT-OF-DOMAIN test, and it is
                 the one that speaks to PMO, whose parents are initialization-pool
                 molecules rather than docking leads.

Every number is zero-oracle. Nothing here calls a docking function, a PMO
oracle, or any task objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from route_prior_fit import ADMITTED_CORPUS_GLOB, assert_admissible, load_corpus

from compose_v4.control.route_program_prior import RouteProgramPrior
from compose_v4.experiments.route_program_prior_eval import (
    MAX_DECLARED_MODULES,
    PROPOSAL_SLOTS,
    _module_count,
    parent_state,
    prior_families,
    run_declared_arm,
    run_production_arm,
    teacher_region_targets,
    uniform_families,
)

SCHEMA_VERSION = "route_prior_heldout_comparison_v1"


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"


def _canonical(smiles: str) -> str:
    """Endpoint identity through the SAME kernel path the arms report with.

    An arm records `molecular_graph_to_smiles` of an executed state; a teacher
    endpoint arrives as a SMILES string. Comparing them requires putting both
    through one canonicalizer, and the production one is the executor's -- so
    the teacher string is parsed into a state and written back out rather than
    being canonicalized by a separate RDKit call.
    """

    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

    try:
        return molecular_graph_to_smiles(parent_state(smiles)) or smiles
    except (ValueError, KeyError, RuntimeError):
        return smiles


def _teacher_released(states, actions, source):
    """Source slots each teacher component released, for region recall."""

    from compose_v4.chem.molecular_graph import is_element
    from compose_v4.control.dependency_region_program import (
        DependencyRegionConfig,
        dependency_region_program,
        trace_structure,
    )

    program = dependency_region_program(states, actions, DependencyRegionConfig())
    graphs = trace_structure(states, actions)["graphs"]
    out = []
    for component in program["components"]:
        indices = component["primitive_indices"]
        start, stop = int(indices[0]), int(indices[-1])
        before = {int(i) for i in np.flatnonzero(is_element(graphs[start].atom_types))}
        after = {int(i) for i in np.flatnonzero(is_element(graphs[stop + 1].atom_types))}
        released = before - after
        if released:
            out.append(released)
    return out


def evaluate_parent(
    smiles: str,
    prior: RouteProgramPrior,
    *,
    draws: int,
    seed: int,
    teacher_regions,
    teacher_endpoints=None,
) -> dict:
    source = parent_state(smiles)
    counts = [
        _module_count(
            np.random.default_rng(seed + 991 * draw),
            source,
            maximum=MAX_DECLARED_MODULES,
        )
        for draw in range(draws)
    ]
    arms = {}
    arms["production_v1"] = run_production_arm(
        source, draws=draws, seed=seed, region_law=None,
        teacher_regions=teacher_regions, teacher_endpoints=teacher_endpoints,
    )
    arms["production_route_law"] = run_production_arm(
        source, draws=draws, seed=seed, region_law=prior,
        teacher_regions=teacher_regions, teacher_endpoints=teacher_endpoints,
    )
    declared = {
        "declared_uniform": (uniform_families, None),
        "declared_prior": (prior_families(prior), prior),
        "declared_prior_families": (prior_families(prior), None),
        "declared_prior_region": (uniform_families, prior),
    }
    for name, (families_for, law) in declared.items():
        arms[name] = run_declared_arm(
            source,
            draws=draws,
            seed=seed,
            families_for=families_for,
            region_law=law,
            module_counts=counts,
            teacher_regions=teacher_regions,
            teacher_endpoints=teacher_endpoints,
        )
    return {
        "heavy_atoms": int(source.n_real_atoms),
        "declared_module_counts": dict(sorted(Counter(counts).items())),
        "arms": {name: tally.summary() for name, tally in arms.items()},
        "_tallies": arms,
    }


def _pool(results: list[dict]) -> dict:
    """Source-balanced pooling: every parent weighs the same.

    Not row-pooled. A parent that happens to admit many programs would
    otherwise dominate, and the question is about parents, not about draws.
    """

    names = sorted(results[0]["arms"])
    pooled = {}
    for name in names:
        tallies = [r["_tallies"][name] for r in results]
        draws = sum(t.draws for t in tallies)
        pooled[name] = {
            "parents": len(tallies),
            "draws": draws,
            "complete_program_yield": float(
                np.mean([t.complete / t.draws if t.draws else 0.0 for t in tallies])
            ),
            "legal_execution_rate": float(
                np.mean([t.legal / t.draws if t.draws else 0.0 for t in tallies])
            ),
            "complete_programs": int(sum(t.complete for t in tallies)),
            "distinct_endpoints": int(sum(len(t.endpoints) for t in tallies)),
            "distinct_endpoints_per_parent": float(
                np.mean([len(t.endpoints) for t in tallies])
            ),
            "distinct_family_multisets_per_parent": float(
                np.mean([len(t.family_multisets) for t in tallies])
            ),
            "teacher_endpoint_hits": int(sum(t.teacher_endpoint_hits for t in tallies)),
            "teacher_region_recall": float(
                np.mean(
                    [
                        t.teacher_region_hits / t.complete if t.complete else 0.0
                        for t in tallies
                    ]
                )
            ),
            "released_region_size_median": float(
                np.median([v for t in tallies for v in t.released_sizes] or [0.0])
            ),
            "changed_fraction_median": float(
                np.median([v for t in tallies for v in t.changed_fractions] or [0.0])
            ),
            "primitive_count_median": float(
                np.median([v for t in tallies for v in t.primitive_counts] or [0.0])
            ),
            "wall_seconds": round(sum(t.wall_seconds for t in tallies), 2),
            "compile_attempts": int(sum(t.compile_attempts for t in tallies)),
            "parents_with_zero_complete": int(sum(1 for t in tallies if not t.complete)),
        }
    return pooled


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--draws", type=int, default=24)
    parser.add_argument("--generic-parents", type=int, default=12)
    parser.add_argument("--generic-pool", default="data/jin/hphi_valid_128.txt")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    fit_raw = Path(args.fit).read_bytes()
    fit = json.loads(fit_raw.decode())
    folds = fit["folds"]
    priors = {k: RouteProgramPrior.from_payload(v) for k, v in fit["priors"].items()}

    # Not for its return value: this re-runs the corpus admission check, so a
    # comparison cannot be produced against a corpus the fit would have refused.
    assert_admissible(ADMITTED_CORPUS_GLOB)
    load_corpus()
    by_source: dict[str, list] = {}
    smiles_of: dict[str, str] = {}
    endpoints_of: dict[str, set] = {}
    import glob as _glob
    import gzip as _gzip

    for path in sorted(_glob.glob("diagnostics/ivg_winner_paths/pairs/*.json.gz")):
        record = json.loads(_gzip.decompress(Path(path).read_bytes()).decode())
        route = record["payload"]["path"]
        if route["status"] != "witness_found":
            continue
        source = str(record["pair"]["source"])
        source_id = hashlib.sha256(source.encode()).hexdigest()
        smiles_of[source_id] = source
        by_source.setdefault(source_id, []).append(
            (tuple(route["states"]), tuple(route["actions"]))
        )
        endpoints_of.setdefault(source_id, set()).add(str(route["target_2d"]))

    started = time.monotonic()
    t4_results, t4_meta = [], []
    for fold_name, held in sorted(folds.items()):
        prior = priors[fold_name.replace("fold_", "fold_")]
        for source_id in held:
            smiles = smiles_of[source_id]
            source = parent_state(smiles)
            released = []
            for states, actions in by_source[source_id]:
                released.extend(_teacher_released(states, actions, source))
            targets = teacher_region_targets(source, released)
            seed = args.seed + int(source_id[:8], 16) % 1_000_003
            result = evaluate_parent(
                smiles,
                prior,
                draws=args.draws,
                seed=seed,
                teacher_regions=targets,
                teacher_endpoints=frozenset(
                    _canonical(s) for s in endpoints_of.get(source_id, ())
                ),
            )
            t4_results.append(result)
            t4_meta.append(
                {
                    "source_id": source_id,
                    "fold": fold_name,
                    "heavy_atoms": result["heavy_atoms"],
                    "teacher_routes": len(by_source[source_id]),
                    "teacher_region_targets": len(targets),
                    "teacher_endpoints": len(endpoints_of.get(source_id, ())),
                    "arms": result["arms"],
                }
            )

    generic_pool = [
        line.strip()
        for line in Path(args.generic_pool).read_text().splitlines()
        if line.strip()
    ]
    ordered = sorted(generic_pool, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    generic_results, generic_meta = [], []
    for smiles in ordered[: args.generic_parents]:
        seed = args.seed + int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16) % 1_000_003
        try:
            result = evaluate_parent(
                smiles,
                priors["train_on_all"],
                draws=args.draws,
                seed=seed,
                teacher_regions=None,
            )
        except (ValueError, KeyError) as error:
            generic_meta.append({"smiles": smiles, "skipped": str(error)[:80]})
            continue
        generic_results.append(result)
        generic_meta.append(
            {
                "smiles_sha256": hashlib.sha256(smiles.encode()).hexdigest(),
                "heavy_atoms": result["heavy_atoms"],
                "arms": result["arms"],
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "headline": {
            "t4_heldout_leave_source_out": _pool(t4_results),
            "generic_out_of_domain_train_on_all": (
                _pool(generic_results) if generic_results else {}
            ),
        },
        "per_parent": {"t4_heldout": t4_meta, "generic": generic_meta},
        "regime": {
            "t4_heldout": "leave-source-out; each parent scored under a prior "
            "fitted without any route from that source molecule",
            "generic": "train-on-all; no generic molecule appears in any teacher "
            "route, so there is nothing to leave out",
        },
        "settings": {
            "draws_per_parent_per_arm": args.draws,
            "proposal_slots": PROPOSAL_SLOTS,
            "max_declared_modules": MAX_DECLARED_MODULES,
            "seed": args.seed,
            "module_count_matched_across_declared_arms": True,
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "inputs": {
            "fit": {"path": args.fit, "sha256": hashlib.sha256(fit_raw).hexdigest()},
            "generic_pool": {
                "path": args.generic_pool,
                "sha256": hashlib.sha256(
                    Path(args.generic_pool).read_bytes()
                ).hexdigest(),
            },
        },
        "implementation": {
            "code_revision": _revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
            "wall_seconds": round(time.monotonic() - started, 2),
            "wall_clock_shares_this_machine": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True)
    out.write_text(body + "\n")
    print(json.dumps(payload["headline"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
