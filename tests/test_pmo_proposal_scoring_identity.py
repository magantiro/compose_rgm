"""The molecule a PMO proposal PRODUCES is the molecule the campaign SCORES.

This is a launch-gating invariant, not a performance property, and it is checked by
EXECUTION rather than by reading the source.

WHY. `t4_fiber_campaign.expand` does not gate the molecule its proposal law ranks: it
abstracts the synthesized program through `extract_structural_goal`, expands `_variants`,
re-binds via `attachment_bindings(...).assignments[0]`, and gates whatever
`instantiate_goal` builds. Measured `recovered_fraction = 0.0000` over 150 draws. Any
proposal law wired into that path conditions an object nobody scores.

The PMO path measures 1.0000 over 144 candidates
(`diagnostics/pmo_discovery_v1/proposal_scoring_identity_gate_v1.json`). These tests
keep it that way, because a route-prototype channel is only meaningful if the endpoint it
produces is the endpoint that gets charged.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rdkit import Chem

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

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

SOURCE = "CC(=O)Nc1ccc(O)cc1"


def _eligibility(row):
    # PMO endpoint eligibility is RDKit-parseability only. Reproduced, not relaxed.
    return {"oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None}


def _drive_one_batch(seed=11):
    """Bootstrap, propose, lock. Returns the controller and the LOCKED candidates."""
    # PMO states are 48 SLOTS; the 40-slot editing preflight is the wrong one here.
    source = production_state_from_smiles(SOURCE, max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=12,
        candidates_per_batch=4,
        wall_seconds=20,
        parent_allocation="niche_score",
    )
    batch = initial_dynamic_program_batch_v21(
        source, (), config,
        source_group="identity-test", oracle_protocol="free-no-oracle",
        eligibility=_eligibility,
    )
    controller = PmoPopulationController(
        config, source_group="identity-test", oracle_protocol="free-no-oracle",
        hierarchy=None, jump_checkpoint=CHECKPOINT, enable_online_memory=True,
    )
    for index, candidate in enumerate(batch["candidates"][:4]):
        controller.add_measured_program(candidate, receipt_id=f"seed{index}", score=0.5)
    proposed = controller.propose_batch(_eligibility)
    ids = [row["candidate_id"] for row in proposed["candidates"]]
    locked = controller.lock_query_subset(
        proposed["batch_id"], ids, {"policy": "identity_test_all", "selected_ids": ids}
    )
    return controller, locked


def test_every_proposal_endpoint_is_its_own_executed_program_endpoint():
    """The endpoint string must be recomputable from the candidate's executed trace.

    Recomputed through the production decoder, so a record whose endpoint came from
    anywhere other than the executed program fails here.
    """
    _, locked = _drive_one_batch()
    assert locked["candidates"], "no candidates were proposed; the test measured nothing"
    for candidate in locked["candidates"]:
        executed = molecular_graph_to_smiles(decode_state(candidate["trace"]["states"][-1]))
        assert candidate["endpoint"] == executed, (
            f"proposal endpoint {candidate['endpoint']!r} is not the molecule its own "
            f"program produced ({executed!r}) -- the T4 indirection has reached PMO"
        )


def test_the_scored_archive_records_the_produced_molecule():
    """The hop the T4 defect broke: proposal -> lock -> charged archive."""
    controller, locked = _drive_one_batch()
    candidates = locked["candidates"]
    assert candidates, "no candidates were locked; the test measured nothing"
    receipts = {row["candidate_id"]: f"r-{row['candidate_id'][:12]}" for row in candidates}
    controller.observe_batch(
        locked["batch_id"],
        [
            {
                "candidate_id": row["candidate_id"],
                "receipt_id": receipts[row["candidate_id"]],
                "score": 0.5,
                "oracle_protocol": controller.oracle_protocol,
            }
            for row in candidates
        ],
    )
    checked = 0
    for candidate in candidates:
        observed = controller.observations.get(receipts[candidate["candidate_id"]])
        if observed is None:
            continue
        checked += 1
        executed = molecular_graph_to_smiles(decode_state(candidate["trace"]["states"][-1]))
        assert observed["endpoint"] == executed, (
            "the archive charged a molecule the proposal did not produce"
        )
    assert checked, "nothing reached the archive; the test measured nothing"
