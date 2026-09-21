"""Does the molecule a PMO proposal PRODUCES equal the molecule the campaign SCORES?

ZERO ORACLE CALLS. This is a launch gate, not an experiment.

WHY IT EXISTS
-------------
A parallel T4 investigation measured that `t4_fiber_campaign.expand` does NOT gate the
molecule its proposal law ranks: it abstracts the synthesized program through
`extract_structural_goal`, expands `_variants`, re-binds each subgoal via
`attachment_bindings(...).assignments[0]`, and gates whatever `instantiate_goal` builds.
Measured `recovered_fraction = 0.0000` over 150 draws -- the program's own endpoint never
appeared in the gated set. Any proposal law wired into that path conditions an object
nobody scores.

Before building a route-prototype channel into the PMO controller, the same question has
to be answered for the PMO path, BY EXECUTION rather than by reading the source. If PMO
shares the indirection, the route channel is unbuildable as designed and that is the
finding.

WHAT IS CHECKED, END TO END
---------------------------
1. PROPOSAL END. For every candidate the real `propose_batch` returns, the endpoint
   string it carries is recomputed from its own executed trace through the production
   decoder (`decode_state` -> `molecular_graph_to_smiles`) and must match. This catches a
   record whose endpoint was written from anything other than the executed program.
2. PROGRAM/TRACE AGREEMENT. The candidate's recorded program size profile must agree with
   the executed trace's heavy-atom count, which is the production invariant
   `_generate_channel_pool` itself asserts.
3. SCORING END. The batch is driven through the real allocation and the real
   `observe_batch`, and the endpoint recorded against each scored candidate id must be the
   same string again. This is the hop the T4 defect broke.

`recovered_fraction` is reported on the same definition as the T4 measurement: the share
of proposals whose own produced molecule is the one that reaches scoring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.rewrite.trace_shard import decode_state

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

#: PMO endpoint eligibility is RDKit-parseability only -- correct for no-prescreen, and
#: deliberately reproduced here rather than relaxed to "always eligible", so the gate
#: runs the rule production runs.
def _eligibility(row: dict) -> dict:
    return {"oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None}


def _canonical_from_state(payload) -> str | None:
    """The production decode path: packed state -> graph -> canonical SMILES."""
    return molecular_graph_to_smiles(decode_state(payload))


def run(*, smiles: str, seed: int, rounds: int) -> dict:
    # PMO states are 48 SLOTS with the heavy-atom ceiling enforced at construction. The
    # 40-slot editing-corpus preflight is the WRONG one for this path.
    source = production_state_from_smiles(smiles, max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=24,
        candidates_per_batch=8,
        wall_seconds=20,
        parent_allocation="niche_score",
    )
    batch = initial_dynamic_program_batch_v21(
        source,
        (),
        config,
        source_group="identity-gate",
        oracle_protocol="free-no-oracle",
        eligibility=_eligibility,
    )
    controller = PmoPopulationController(
        config,
        source_group="identity-gate",
        oracle_protocol="free-no-oracle",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
        enable_online_memory=True,
    )
    # Seed the archive with counted-looking observations. The SCORE is a free constant:
    # nothing here optimizes, so no oracle is consulted and none is needed.
    for index, candidate in enumerate(batch["candidates"][:4]):
        controller.add_measured_program(candidate, receipt_id=f"seed{index}", score=0.5)

    checked = 0
    endpoint_mismatch: list[dict] = []
    size_mismatch: list[dict] = []
    scored_mismatch: list[dict] = []
    recovered = 0

    for _round_index in range(rounds):
        proposed = controller.propose_batch(_eligibility)
        candidates = proposed["candidates"]
        if not candidates:
            break
        # The campaign LOCKS the subset it will charge before any oracle call, and the
        # archive is written from the LOCKED candidate record. Locking here keeps the
        # hop under test rather than short-circuiting it.
        locked = controller.lock_query_subset(
            proposed["batch_id"],
            [row["candidate_id"] for row in candidates],
            {"policy": "identity_gate_all", "selected_ids": [r["candidate_id"] for r in candidates]},
        )
        candidates = locked["candidates"]
        # ---- 1 + 2: proposal end ----
        declared: dict[str, str] = {}
        for candidate in candidates:
            checked += 1
            endpoint = candidate["endpoint"]
            declared[candidate["candidate_id"]] = endpoint
            produced = _canonical_from_state(candidate["trace"]["states"][-1])
            if produced != endpoint:
                endpoint_mismatch.append(
                    {"candidate_id": candidate["candidate_id"],
                     "recorded_endpoint": endpoint, "executed_endpoint": produced}
                )
            final_atoms = decode_state(candidate["trace"]["states"][-1]).n_real_atoms
            profile = candidate["provenance"]["program_size"]["final_heavy_atoms"]
            if final_atoms != profile:
                size_mismatch.append(
                    {"candidate_id": candidate["candidate_id"],
                     "trace_heavy_atoms": final_atoms, "program_profile": profile}
                )
        # ---- 3: scoring end ----
        receipts = {
            row["candidate_id"]: f"receipt-{row['candidate_id'][:16]}" for row in candidates
        }
        outcomes = [
            {
                "candidate_id": row["candidate_id"],
                "receipt_id": receipts[row["candidate_id"]],
                "score": 0.5,
                "oracle_protocol": controller.oracle_protocol,
            }
            for row in candidates
        ]
        controller.observe_batch(locked["batch_id"], outcomes)
        # The archive is keyed by RECEIPT, and its endpoint is written from the locked
        # candidate. This is the hop the T4 defect broke.
        for candidate in candidates:
            observed = controller.observations.get(receipts[candidate["candidate_id"]])
            if observed is None:
                continue
            executed = _canonical_from_state(candidate["trace"]["states"][-1])
            if observed["endpoint"] != executed:
                scored_mismatch.append(
                    {"candidate_id": candidate["candidate_id"],
                     "program_produced": executed,
                     "scored": observed["endpoint"]}
                )
            else:
                recovered += 1

    return {
        "source_smiles": smiles,
        "seed": seed,
        "candidates_checked": checked,
        "scored_candidates": recovered + len(scored_mismatch),
        "endpoint_mismatches": endpoint_mismatch[:5],
        "endpoint_mismatch_count": len(endpoint_mismatch),
        "program_size_mismatch_count": len(size_mismatch),
        "scored_mismatch_count": len(scored_mismatch),
        # Same definition as the T4 measurement that motivated this gate.
        "recovered_fraction": (
            round(recovered / (recovered + len(scored_mismatch)), 6)
            if (recovered + len(scored_mismatch))
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # Ordinary drug-like starting molecules; nothing task-specific, no target structure.
    sources = (
        "CC(=O)Nc1ccc(O)cc1",
        "O=C(NC1CCNCC1)c1ccccc1",
        "COc1ccc(CCN)cc1",
    )
    rows = []
    for smiles in sources:
        for seed in args.seeds:
            row = run(smiles=smiles, seed=seed, rounds=args.rounds)
            rows.append(row)
            print(
                f"{smiles:<26} seed={seed} checked={row['candidates_checked']:<4}"
                f" scored={row['scored_candidates']:<4}"
                f" endpoint_mismatch={row['endpoint_mismatch_count']}"
                f" size_mismatch={row['program_size_mismatch_count']}"
                f" scored_mismatch={row['scored_mismatch_count']}"
                f" recovered={row['recovered_fraction']}",
                flush=True,
            )
    total_checked = sum(r["candidates_checked"] for r in rows)
    total_scored = sum(r["scored_candidates"] for r in rows)
    mismatches = sum(
        r["endpoint_mismatch_count"] + r["program_size_mismatch_count"]
        + r["scored_mismatch_count"]
        for r in rows
    )
    verdict = (
        "PASS_PROPOSAL_IS_SCORED"
        if total_scored and mismatches == 0
        else "FAIL_PROPOSAL_IS_NOT_WHAT_IS_SCORED"
        if total_scored
        else "INCONCLUSIVE_NO_SCORED_CANDIDATES"
    )
    report = {
        "schema_version": "pmo_proposal_scoring_identity_gate_v1",
        "oracle_calls": 0,
        "candidates_checked": total_checked,
        "scored_candidates": total_scored,
        "total_mismatches": mismatches,
        "verdict": verdict,
        "rows": rows,
    }
    print(f"\nchecked={total_checked}  scored={total_scored}  mismatches={mismatches}")
    print(f"VERDICT: {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
