"""Can blind COMPOSE propose the next useful macro at each known Celecoxib boundary?

The macro-horizon work left 14 boundaries `G_i -> G_{i+1}` on routes that reach celecoxib, with
mostly monotone boundary scores.  This probe asks the only question that matters before another
scored run: standing at `G_i` with NO knowledge of `G_{i+1}`, does the controller PROPOSE a move of
the needed kind, and does that move improve the objective?

INFORMATION BOUNDARY, and it is the whole design.  `G_{i+1}` is used ONLY after generation, as a
diagnostic reference.  Nothing about it -- not the molecule, not its fragments, not its ring count,
not its size -- reaches the proposal draw.  The controller sees `G_i` and its own chemistry, exactly
as it would blind.  Scores here are an OFFLINE DIAGNOSTIC against a pure-RDKit evaluator, not a
charged campaign ledger, and must never be spliced into one.

SUCCESS IS NOT EXACT RECOVERY.  A different macro that reaches comparable objective value is
BETTER evidence of a general proposal mechanism than reproducing the recorded route, which a
memorising system could do.  So the table reports exact recovery, structural proximity, and score
improvement as three separate columns and does not collapse them.

ARM A  production shallow synthesis, no region option
ARM B  the same, plus the full-vocabulary region replacement offered before the family lottery
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
import types

import numpy as np

SEGMENTS = "diagnostics/pmo_macro_horizon_v1/macro_segments_v1.json"
CONTROLLER_SCALE = "diagnostics/pmo_macro_boundary_v1/controller_scale_boundaries_v1.json"
OUT = "diagnostics/pmo_macro_boundary_v1/macro_boundary_probe_v1.json"


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
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    draws = int(positional[0]) if positional else 300
    rate = float(positional[1]) if len(positional) > 1 else 0.5

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.control.region_replacement_option import RegionReplacementOption
    from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw

    score = _oracle()

    def canon(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None

    def fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprint(mol, 2) if mol else None

    def arm(source_graph, option, seed):
        rng = np.random.default_rng(seed)
        rows, failures = {}, 0
        for _ in range(draws):
            try:
                _s, _p, _a, trace, meta = synthesize_dynamic_program(
                    source_graph,
                    rng,
                    max_modules=3,
                    region_law=None if option is None else ScaleBalancedRegionLaw(),
                    replacement_option=option,
                    replacement_option_rate=0.0 if option is None else rate,
                )
            except (ValueError, RuntimeError):
                failures += 1
                continue
            endpoint = trace.get("endpoint")
            if not endpoint or endpoint in rows:
                continue
            families = [m["family"] for m in meta.get("modules", [])]
            rows[endpoint] = families
        return rows, failures

    # `--controller-scale` retargets the probe at boundaries rebuilt in the measured useful
    # macro regime (~12-18 primitives).  The default boundary file segments at cap 23, which
    # collapses each route's tail into ONE jump straight to celecoxib -- a compression artifact
    # of the diagnostic, not the action size a sequential controller is meant to take in one
    # proposal.  Failure on those says nothing about the proposal mechanism.
    controller_scale = "--controller-scale" in sys.argv
    if controller_scale:
        with open(CONTROLLER_SCALE) as handle:
            document = json.load(handle)
        segments = []
        for route in document["routes"]:
            for boundary in route.get("boundaries") or []:
                segments.append(
                    {
                        "route": route["source_name"],
                        "seg": boundary["from_index"],
                        "source_smiles": boundary["source_smiles"],
                        "target_smiles": boundary["next_smiles"],
                        "required_primitives": boundary["primitives"],
                        "target_score": boundary["next_score"],
                        "boundary_kind": (
                            "terminal_cliff" if boundary["next_score"] >= 0.99 else "climb"
                        ),
                    }
                )
    else:
        with open(SEGMENTS) as handle:
            segments = json.load(handle)

    option = RegionReplacementOption(region_law=ScaleBalancedRegionLaw())
    report = {
        "schema_version": "pmo_macro_boundary_v1",
        "evidence_role": "answer_known_offline_diagnostic",
        "new_charged_oracle_calls": 0,
        "oracle": "tdc celecoxib_rediscovery (pure rdkit, no asset)",
        "draws_per_arm": draws,
        "replacement_option_rate": rate,
        "boundary_source": CONTROLLER_SCALE if controller_scale else SEGMENTS,
        "information_boundary": (
            "G_{i+1} is read only AFTER generation. No property of it reaches any proposal draw."
        ),
        "boundaries": [],
    }

    # Sharding is a WALL-CLOCK convenience only: boundaries are independent and every draw is
    # seeded from the boundary's own index, so a shard produces byte-identical rows to the same
    # boundary in an unsharded run.  Shards write separate files and are merged at read time.
    shard, shards = 0, 1
    for argument in sys.argv[1:]:
        if argument.startswith("--shard="):
            shard, shards = (int(part) for part in argument.removeprefix("--shard=").split("/"))
    if shards > 1:
        segments = [row for i, row in enumerate(segments) if i % shards == shard]
        print(f"shard {shard}/{shards}: {len(segments)} boundaries", flush=True)

    for index, row in enumerate(segments):
        source_smiles, next_smiles = row["source_smiles"], row["target_smiles"]
        graph = pad_molecular_graph(smiles_to_molecular_graph(source_smiles), 48)
        parent_score = float(score(source_smiles))
        next_canon, next_fp = canon(next_smiles), fp(next_smiles)
        entry = {
            "route": row["route"],
            "seg": row["seg"],
            "boundary_kind": row.get("boundary_kind", "cap23_compressed"),
            "source_smiles": source_smiles,
            "next_smiles": next_smiles,
            "required_primitives": row["required_primitives"],
            "parent_score": parent_score,
            "next_score": float(row["target_score"]),
            "arms": {},
        }
        for label, opt in (("A_production", None), ("B_region_option", option)):
            started = time.time()
            # A STABLE digest, never `hash()`: str hashing is PYTHONHASHSEED-salted, so a
            # hash-derived seed differs between processes and no sharded row would be
            # reproducible by anyone, including us.
            identity = int.from_bytes(
                hashlib.blake2b(
                    f"{row['route']}:{row['seg']}".encode(), digest_size=4
                ).digest(),
                "big",
            )
            rows, failures = arm(graph, opt, 20260923 + identity)
            scored = []
            for endpoint, families in rows.items():
                value = float(score(endpoint))
                cand_fp = fp(endpoint)
                sim = (
                    float(DataStructs.TanimotoSimilarity(cand_fp, next_fp))
                    if cand_fp is not None and next_fp is not None
                    else 0.0
                )
                scored.append(
                    {
                        "smiles": endpoint,
                        "score": value,
                        "sim_to_next": sim,
                        "families": families,
                    }
                )
            scored.sort(key=lambda r: -r["score"])
            for rank, record in enumerate(scored, start=1):
                record["rank"] = rank
            exact = [r for r in scored if canon(r["smiles"]) == next_canon]
            improving = [r for r in scored if r["score"] > parent_score]
            best = scored[0] if scored else None
            nearest = max(scored, key=lambda r: r["sim_to_next"]) if scored else None
            entry["arms"][label] = {
                "distinct_endpoints": len(scored),
                "failures": failures,
                "exact_next_proposed": bool(exact),
                "exact_next_rank": exact[0]["rank"] if exact else None,
                "n_improving_parent": len(improving),
                "fraction_improving": len(improving) / max(len(scored), 1),
                "best_score": best["score"] if best else None,
                "best_score_delta_vs_parent": (best["score"] - parent_score) if best else None,
                "best_rank_families": best["families"] if best else None,
                "best_smiles": best["smiles"] if best else None,
                "max_sim_to_next": nearest["sim_to_next"] if nearest else None,
                "max_sim_candidate_improves": (
                    nearest["score"] > parent_score if nearest else None
                ),
                "seconds": round(time.time() - started, 1),
            }
        report["boundaries"].append(entry)
        a, b = entry["arms"]["A_production"], entry["arms"]["B_region_option"]
        print(
            f"{row['route'][:16]:16s} s{row['seg']} parent={parent_score:.4f} next={row['target_score']:.4f} "
            f"| A best {a['best_score']:.4f} d{a['best_score_delta_vs_parent']:+.4f} imp {a['fraction_improving']:.2f} "
            f"| B best {b['best_score']:.4f} d{b['best_score_delta_vs_parent']:+.4f} imp {b['fraction_improving']:.2f} "
            f"| exact A={a['exact_next_proposed']} B={b['exact_next_proposed']}",
            flush=True,
        )
        destination = (
            OUT.replace("_v1.json", "_controller_scale_v1.json") if controller_scale else OUT
        )
        if shards > 1:
            destination = destination.replace(".json", f".shard{shard}of{shards}.json")
        pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w") as handle:
            json.dump(report, handle, indent=1)
    print("WROTE", destination)


if __name__ == "__main__":
    sys.exit(main())
