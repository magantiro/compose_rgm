from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.control.reference_programs import (
    FrozenProgramReference,
    ProgramPanelGuidance,
    t4_program_input,
)
from compose_v4.control.route_distilled_goal_expert import make_route_expert
from compose_v4.control.structural_subgoal_policy import (
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_integrated_route_fiber import attach_features, select_batch
from compose_v4.experiments.t4_route_proposals import (
    RouteProposalConfig,
    expand_route,
    load_route_expert,
)
from compose_v4.model.reference_checkpoint import load_frozen_reference

LEAD = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "experiments/reference/model.json").read_text())
BOUNDS = RouteProposalConfig(
    pool_size=4,
    realization_limit=4,
    beam_width=4,
    expansion_width=4,
    max_bindings_per_template=2,
    maximum_expansions=64,
)


@pytest.fixture
def expert():
    templates = tuple(
        sorted(
            (
                StructuralDeltaTemplate(
                    input_atoms=((2, 0, 3, 1),),
                    input_bonds=((0,),),
                    target_atoms=((element, 0, hydrogens, 1),),
                    output_atoms=(),
                    target_bonds=((0,),),
                )
                for element, hydrogens in ((3, 2), (4, 1))
            ),
            key=lambda row: row.template_id,
        )
    )
    marginal = MarginalSubgoalPolicy(
        tuple(row.template_id for row in templates),
        (0.5, 0.5),
        (0.97, 0.01, 0.01, 0.01),
        0.1,
        "unit-test-fixture",
    )
    return make_route_expert(templates, marginal, training_evidence_identity="unit-test-fixture")


def test_exact_route_receipts_leave_proposals_unchanged(expert):
    fiber = Fiber(LEAD, 0.4)
    baseline, plain_work = expand_route(LEAD, -7.0, fiber, expert, config=BOUNDS)
    captured, trace_work = expand_route(
        LEAD, -7.0, fiber, expert, config=BOUNDS, include_realized_actions=True
    )
    assert len(captured) >= 2
    trace_fields = {"realized_actions", "realized_endpoint_key", "source_state"}
    assert [{k: v for k, v in row.items() if k not in trace_fields} for row in captured] == baseline
    assert {k: v for k, v in trace_work.items() if k != "realized_actions_included"} == plain_work
    for row in captured:
        program = t4_program_input(row, candidate_id=row["smiles"])
        assert program.trace_json is not None
        assert row["parent"] == LEAD and row["parent_score"] == -7.0
        assert row["similarity"] >= fiber.delta and row["qed"] >= 0.6 and row["sa"] <= 4.0
    assert plain_work["complete_programs_committed"] == sum(
        plain_work[k] for k in ("eligible_endpoints", "ineligible_endpoints", "self_endpoints")
    )


@pytest.mark.external_artifact
def test_route_proposals_receive_real_neural_reference_guidance(expert):
    location = Path(
        os.environ.get(
            "COMPOSE_REFERENCE_CHECKPOINT", ROOT / "local_assets/fragments/r_theta_nll.pt"
        )
    )
    if not location.is_file():
        pytest.skip("fetch the NLL reference checkpoint through Git LFS")
    loaded = load_frozen_reference(
        location,
        expected_sha256=REFERENCE["checkpoint"]["sha256"],
        expected_catalog_fingerprint=REFERENCE["catalog"]["fingerprint"],
        catalog_path=ROOT / "local_assets/fragments/catalog.json",
        expected_catalog_sha256=REFERENCE["catalog"]["sha256"],
    )
    fiber = Fiber(LEAD, 0.4)
    candidates, _ = expand_route(
        LEAD, 0.0, fiber, expert, config=BOUNDS, include_realized_actions=True
    )
    programs = tuple(t4_program_input(row, candidate_id=row["smiles"]) for row in candidates)
    reference = FrozenProgramReference(loaded)
    scores = reference.score(programs)
    assert len(scores.scores) >= 2 and all(row.status == "scored" for row in scores.scores)
    assert len({row.value for row in scores.scores}) > 1
    state = SearchState()
    rows = attach_features(candidates, state, fiber)
    guide = ProgramPanelGuidance(
        programs, reference, GuidanceConfig("active", strength=0.25), identity_field="smiles"
    )
    receipts = []
    chosen = select_batch(
        rows,
        ProgramValue(),
        state,
        np.random.default_rng(4),
        round_index=1,
        batch=1,
        exploration=1,
        expert_floor_rounds=0,
        reference_guide=guide,
        reference_receipts=receipts,
    )
    assert len(chosen) == 1
    assert receipts and receipts[0]["probabilities_changed"]
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())


def test_route_asset_identity_and_inner_envelope_are_verified(tmp_path, expert):
    path = tmp_path / "expert.json"
    payload = {
        "schema_version": "t4_route_complete_region_expert_checkpoint_v1",
        "expert": expert.checkpoint(),
        "new_oracle_calls": 0,
        "split_audit": {},
    }
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    path.write_text(json.dumps(envelope))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_route_expert(path, expected_sha256=digest) == expert
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_route_expert(path, expected_sha256="0" * 64)
    payload["new_oracle_calls"] = 1
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_route_expert(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("digest", ["", "0" * 63, "G" * 64, None])
def test_route_asset_requires_explicit_sha_before_reading(tmp_path, digest):
    with pytest.raises(ValueError, match="explicit lowercase SHA-256"):
        load_route_expert(tmp_path / "absent.json", expected_sha256=digest)


@pytest.mark.parametrize(
    "payload,error,message",
    [
        ([], TypeError, "payload must be an object"),
        ({"schema_version": "other"}, ValueError, "unsupported route expert schema"),
        (
            {"schema_version": "t4_route_complete_region_expert_checkpoint_v1"},
            TypeError,
            "payload.expert must be an object",
        ),
    ],
)
def test_route_asset_rejects_bad_payload_schema(tmp_path, payload, error, message):
    path = tmp_path / "expert.json"
    path.write_text(json.dumps({"payload": payload, "payload_sha256": identity(payload)}))
    with pytest.raises(error, match=message):
        load_route_expert(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_route_limits_and_balance_flag_are_validated():
    with pytest.raises(ValueError, match="must not exceed pool_size"):
        replace(BOUNDS, realization_limit=BOUNDS.pool_size + 1)
    with pytest.raises(TypeError, match="scale_balanced"):
        replace(BOUNDS, scale_balanced=1)


@pytest.mark.parametrize("field", [k for k in asdict(BOUNDS) if k != "scale_balanced"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_route_work_limits_must_be_positive_integer_counts(field, value):
    with pytest.raises(ValueError, match=field):
        replace(BOUNDS, **{field: value})


@pytest.mark.parametrize("delta", [True, -0.01, 1.01, float("nan"), float("inf")])
def test_fiber_rejects_invalid_similarity_threshold(delta):
    with pytest.raises(ValueError, match="delta"):
        Fiber(LEAD, delta)


@pytest.mark.parametrize("lead", ["", "invalid", "CC.O"])
def test_fiber_rejects_invalid_reference_molecule(lead):
    with pytest.raises(ValueError, match="nonempty connected molecule"):
        Fiber(lead, 0.4)
