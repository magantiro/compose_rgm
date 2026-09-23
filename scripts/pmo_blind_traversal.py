"""Can FiberControl CHAIN the individually available improving macros, blind?

The boundary probe established controller-scale proposal SUPPORT: from 7 of 8 climbing states
blind COMPOSE proposes a next state better than the recorded teacher route's own next state, at
+0.079..+0.168 against a median requirement of +0.047.  That is a statement about ONE step from a
GIVEN state.  It does not establish that a controller starting at G0 will chain such steps.

This run answers the chaining question directly:

    G0 --blind--> H1 --blind--> H2 --blind--> ...

INFORMATION BOUNDARY.  No teacher trajectory, no route boundary, no witness fragment, no target
intermediate, no celecoxib-specific structural hint and no answer-known region size enters any
proposal or any selection.  Only the ordinary celecoxib PMO objective ranks candidates, exactly as
a blind run would.  The starting states are the ORIGINAL PMO init-bank molecules, not teacher
intermediates.

OBJECTIVE EVALUATIONS ARE COUNTED AND REPORTED.  Scoring a candidate to select it IS objective
evaluation, so this is a development DIAGNOSTIC with an explicit evaluation count -- it is not a
scored benchmark result and its numbers must never be spliced into a charged ledger or an AUC.

LINEAGE RETENTION IS EXPLICIT.  Greedy single-path selection would conflate "the controller cannot
chain" with "the harness threw the good child away", which is precisely the distinction this run
exists to draw, so several distinct lineages are carried forward every generation.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
import types

import numpy as np

ROUTES = "diagnostics/pmo_macro_horizon_v1/celecoxib_routes_v1.json"
OUT = "diagnostics/pmo_blind_traversal_v1/blind_traversal_v1.json"
EDGES = "diagnostics/pmo_blind_traversal_v1/control_dataset_edges_v1.jsonl"
RESUME_OUT = "diagnostics/pmo_blind_traversal_v1/blind_traversal_resumed_v1.json"


def _oracle():
    stub = types.ModuleType("rdkit.six")
    stub.string_types = (str,)
    stub.iteritems = lambda d: iter(d.items())
    import rdkit

    sys.modules["rdkit.six"] = stub
    rdkit.six = stub
    from tdc import Oracle

    return Oracle(name="celecoxib_rediscovery")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    generations = int(args[0]) if args else 8
    lineages = int(args[1]) if len(args) > 1 else 6
    per_lineage = int(args[2]) if len(args) > 2 else 32
    use_option = "--no-option" not in sys.argv

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.control.region_replacement_option import RegionReplacementOption
    from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw

    score = _oracle()
    option = RegionReplacementOption(region_law=ScaleBalancedRegionLaw()) if use_option else None

    # `--resume=<name>` continues a completed start from its SAVED FRONTIER rather than
    # redoing generations already paid for.  The frontier is the run's own committed output,
    # so this introduces no information the blind run did not already generate itself.
    resume = next(
        (a.removeprefix("--resume=") for a in sys.argv if a.startswith("--resume=")), None
    )
    if resume:
        with open(OUT) as handle:
            prior = json.load(handle)
        entry = next(s for s in prior["starts"] if s["name"] == resume)
        seeded = [
            {"smiles": r["smiles"], "score": r["score"], "depth": r["depth"]}
            for r in entry["final_frontier"]
        ]
        print(
            f"resuming {resume} from {len(seeded)} frontier entries, "
            f"best {max(r['score'] for r in seeded):.4f}, depth<= {max(r['depth'] for r in seeded)}",
            flush=True,
        )
    else:
        seeded = None

    with open(ROUTES) as handle:
        document = json.load(handle)
    # The ORIGINAL init-bank starting molecules, never a teacher intermediate.
    starts = [
        {"name": row["source_name"], "smiles": row["source_smiles"]}
        for row in document["routes"]
        if row["status"] == "witness_found" and row["source_name"].startswith("init_")
    ]
    if resume:
        starts = [s for s in starts if s["name"] == resume]

    report = {
        "schema_version": "pmo_blind_traversal_v1",
        "evidence_role": "blind_development_diagnostic_with_counted_objective_evaluations",
        "teacher_information_used": "none",
        "objective": "tdc celecoxib_rediscovery",
        "generations": generations,
        "lineages_retained": lineages,
        "proposals_per_lineage": per_lineage,
        "region_option_enabled": use_option,
        "starts": [],
    }

    for start in starts:
        seed = int.from_bytes(
            hashlib.blake2b(start["name"].encode(), digest_size=4).digest(), "big"
        )
        rng = np.random.default_rng(20260923 + seed)
        edge_log = []
        frontier = seeded or [
            {"smiles": start["smiles"], "score": float(score(start["smiles"])), "depth": 0}
        ]
        evaluations = 1
        best_overall = frontier[0]["score"]
        history, started = [], time.time()

        for generation in range(1, generations + 1):
            children, families, primitives = [], [], []
            for parent in frontier:
                graph = pad_molecular_graph(smiles_to_molecular_graph(parent["smiles"]), 48)
                for _ in range(per_lineage):
                    try:
                        _s, program, _a, trace, meta = synthesize_dynamic_program(
                            graph,
                            rng,
                            max_modules=3,
                            region_law=ScaleBalancedRegionLaw() if use_option else None,
                            replacement_option=option,
                            replacement_option_rate=0.5 if use_option else 0.0,
                        )
                    except (ValueError, RuntimeError):
                        continue
                    endpoint = trace.get("endpoint")
                    if not endpoint:
                        continue
                    value = float(score(endpoint))
                    evaluations += 1
                    record = {
                        "start": start["name"],
                        "generation": generation,
                        "endpoint": endpoint,
                        "score": value,
                        "depth": parent["depth"] + 1,
                        "parent": parent["smiles"],
                        "parent_score": parent["score"],
                        "delta": value - parent["score"],
                        # Stage-normalised: the FRACTION of remaining headroom closed. A +0.10
                        # from 0.10 and a +0.10 from 0.80 are not the same control event.
                        "headroom_fraction_closed": (value - parent["score"])
                        / max(1e-9, 1.0 - parent["score"]),
                        # Siblings share a parent AND a generation, so a contrast between them
                        # holds parent score and search stage fixed by construction.
                        "sibling_group": f"{start['name']}|g{generation}|{parent['smiles']}",
                        "families": [m["family"] for m in meta.get("modules", [])],
                        "module_count": len(meta.get("modules", [])),
                        "primitives": len(program.marks),
                        "rebuild_options": [
                            m["parameters"].get("rebuild_option")
                            for m in meta.get("modules", [])
                            if isinstance(m.get("parameters"), dict)
                            and m["parameters"].get("rebuild_option")
                        ],
                        "capacity_aware": meta.get("capacity_aware"),
                        "requested_modules": meta.get("requested_module_count"),
                    }
                    edge_log.append(record)
                    children.append(record)
                    families.extend(child["families"] for child in children[-1:])
                    primitives.append(len(program.marks))

            pool = {row["smiles"]: row for row in [*frontier, *children]}
            ranked = sorted(pool.values(), key=lambda r: -r["score"])
            # Retain several DISTINCT lineages so a dropped good child is distinguishable
            # from an absent one.
            frontier = ranked[:lineages]
            improving = [c for c in children if c["delta"] > 0]
            distinct_improving_parents = len({c["parent"] for c in improving})
            top10 = float(np.mean([r["score"] for r in ranked[:10]])) if ranked else 0.0
            best_overall = max(best_overall, frontier[0]["score"])
            history.append(
                {
                    "generation": generation,
                    "best_score": frontier[0]["score"],
                    "top10_mean": top10,
                    "children": len(children),
                    "improving_children": len(improving),
                    "distinct_improving_lineages": distinct_improving_parents,
                    "median_delta_improving": (
                        float(np.median([c["delta"] for c in improving])) if improving else None
                    ),
                    "median_primitives": float(np.median(primitives)) if primitives else None,
                    "best_smiles": frontier[0]["smiles"],
                    "best_depth": frontier[0]["depth"],
                    "evaluations_so_far": evaluations,
                }
            )
            print(
                f"{start['name']:9s} gen{generation:2d} best={frontier[0]['score']:.4f} "
                f"top10={top10:.4f} depth={frontier[0]['depth']:2d} "
                f"improving={len(improving):3d}/{len(children):3d} "
                f"lineages={distinct_improving_parents} evals={evaluations}",
                flush=True,
            )

        with open(EDGES, "a") as handle:
            for record in edge_log:
                handle.write(json.dumps(record) + "\n")
        report["starts"].append(
            {
                "name": start["name"],
                "start_smiles": start["smiles"],
                "start_score": float(score(start["smiles"])),
                "best_score": best_overall,
                "objective_evaluations": evaluations,
                "edges_logged": len(edge_log),
                "seconds": round(time.time() - started, 1),
                "history": history,
                "final_frontier": [
                    {"smiles": r["smiles"], "score": r["score"], "depth": r["depth"]}
                    for r in frontier
                ],
            }
        )
        target = RESUME_OUT if resume else OUT
        pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as handle:
            json.dump(report, handle, indent=1)
    print("WROTE", RESUME_OUT if resume else OUT)


if __name__ == "__main__":
    sys.exit(main())
