import ast
import inspect
import math
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import pytest

import compose_v4.control.route_complete_region_particles as particles
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_policy import StructuralDeltaTemplate


def _record(spec, *, smiles, score, rank, program):
    origin = {
        "job_id": spec.job_id,
        "binding_particle": spec.binding_particle,
        "depth": spec.depth,
        "shard": spec.shard,
        "shard_count": spec.shard_count,
        "local_rank": rank,
        "program_id": program,
    }
    return {
        "smiles": smiles,
        "proposal_lane": particles.ROUTE_EXPERT,
        "proposal_experts": [particles.ROUTE_EXPERT],
        "families": [particles.ROUTE_EXPERT],
        "program_families": [particles.ROUTE_EXPERT],
        "regions": spec.depth,
        "created": 0,
        "deleted": 0,
        "route_prior_score": score,
        "route_proposal_rank": rank,
        "route_program_id": program,
        "route_template_ids": ["template"],
        "rewrite_events": 1,
        "rewrite_scale": "local",
        "realized_primitives": 1,
        "realized_primitive_band": "small",
        "compiler_strategy": "fixture",
        "route_generation_modes": ["complete_combination_particle"],
        "route_particle_local_rank": rank,
        "route_particle_origins": [origin],
    }


def _receipt(spec, records=(), *, status="complete", source_hash="1" * 64):
    rows = [dict(row) for row in records]
    telemetry = {
        "candidate_count": len(rows),
        "candidate_set_sha256": identity(rows),
        "source_state_sha256": source_hash,
        "expert_training_identity_sha256": "2" * 64,
        "planning_expansions": spec.shard_stop - spec.shard_start,
        "combination_particle_assigned_combinations": (
            spec.shard_stop - spec.shard_start
        ),
    }
    payload = {
        "schema_version": particles.RECEIPT_SCHEMA_VERSION,
        "job": spec.payload(),
        "status": status,
        "records": rows,
        "telemetry": telemetry,
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def _receipts(records_by_job=None):
    records_by_job = {} if records_by_job is None else records_by_job
    return [
        _receipt(spec, records_by_job.get(spec.job_id, ()))
        for spec in particles.complete_combination_job_specs()
    ]


def test_fixed_schedule_has_exact_formulas_and_no_depth_four() -> None:
    specs = particles.complete_combination_job_specs()

    assert particles.COMBINATION_COUNTS == {
        1: 48,
        2: 1_128,
        3: 17_296,
        4: 194_580,
    }
    assert particles.REQUIRED_SHARD_COUNTS == {1: 1, 2: 1, 3: 5, 4: 48}
    assert particles.SCHEDULED_SHARD_COUNTS == {1: 1, 2: 1, 3: 5}
    assert len(specs) == 4 * (1 + 1 + 5) == 28
    assert len({spec.job_id for spec in specs}) == len(specs)
    assert {spec.binding_particle for spec in specs} == {0, 1, 2, 3}
    assert {spec.depth for spec in specs} == {1, 2, 3}
    assert all(spec.depth != 4 for spec in specs)
    assert particles.DEPTH_4_COMPLETE_COVERAGE is False

    for binding_particle in range(4):
        for depth, shard_count in particles.SCHEDULED_SHARD_COUNTS.items():
            subset = [
                spec
                for spec in specs
                if spec.binding_particle == binding_particle and spec.depth == depth
            ]
            assert len(subset) == shard_count
            assert subset[0].shard_start == 0
            assert subset[-1].shard_stop == math.comb(48, depth)
            assert all(
                left.shard_stop == right.shard_start for left, right in pairwise(subset)
            )
            assert all(spec.shard_stop - spec.shard_start <= 4_096 for spec in subset)


def test_schedule_has_no_target_cell_delta_or_source_input() -> None:
    signature = inspect.signature(particles.complete_combination_job_specs)
    run_signature = inspect.signature(particles.run_complete_combination_job)
    serialized = str(
        [row.payload() for row in particles.complete_combination_job_specs()]
    )

    assert not signature.parameters
    assert tuple(run_signature.parameters) == ("source", "expert", "spec")
    assert all(
        token not in serialized for token in ("target", "cell", "delta", "source")
    )
    assert (
        particles.complete_combination_job_specs()
        == particles.complete_combination_job_specs()
    )


def test_frozen_particle_budgets_match_production_allocation() -> None:
    budgets = particles.frozen_particle_budgets()

    assert budgets.expansion_width == 48
    assert budgets.max_bindings_per_template == 4
    assert budgets.max_planning_expansions == 4_096
    assert budgets.max_targets == 96
    assert budgets.max_realization_attempts == 96
    assert budgets.max_realization_expansions == 65_536
    assert budgets.maximum_expansions_per_realization == 4_000
    assert budgets.maximum_primitives == 32


def test_virtual_proposal_maps_to_legacy_route_schema_and_bands() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    template = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=((3, 0, 2, 1),),
        output_atoms=(),
        target_bonds=((0,),),
    )
    step = SimpleNamespace(
        patch=SimpleNamespace(output_atoms=(), target_atoms=template.target_atoms),
        template_id=template.template_id,
    )
    proposal = SimpleNamespace(
        endpoint=source,
        depth=1,
        templates=(template,),
        steps=(step,),
        log_probability=-3.5,
        program_id="program",
        primitive_count=12,
        compiler_strategy="fixture",
    )
    spec = particles.complete_combination_job_specs()[0]

    record = particles.route_record_from_virtual_proposal(proposal, spec, 7)

    assert record["proposal_lane"] == particles.ROUTE_EXPERT
    assert record["proposal_experts"] == [particles.ROUTE_EXPERT]
    assert record["families"] == [particles.ROUTE_EXPERT]
    assert record["program_families"] == [particles.ROUTE_EXPERT]
    assert record["route_proposal_rank"] == 7
    assert record["route_particle_local_rank"] == 7
    assert record["realized_primitive_band"] == "large"
    assert record["rewrite_scale"] == "local"
    assert record["route_generation_modes"] == ["complete_combination_particle"]
    assert "actions" not in record


def test_particle_job_uses_frozen_arguments_and_sanitizes_receipt(monkeypatch) -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    spec = particles.complete_combination_job_specs()[9]
    observed = {}

    def propose(candidate_source, expert, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            proposals=(),
            telemetry={
                "task_cell_route_or_endpoint_input_used": False,
                "primitive_teacher_actions_used": 0,
                "virtual_prefix_commits": 0,
                "partial_endpoint_evaluations": 0,
                "binding_particle_index": spec.binding_particle,
                "combination_particle_depth": spec.depth,
                "combination_particle_index": spec.shard,
                "combination_particle_count": spec.shard_count,
                "planning_expansions": 1,
                "compiler_abstentions": 1,
                "bound_constituent_ranks": [
                    {
                        "binding": [3],
                        "slot": 4,
                        "action": {"teacher": "secret"},
                    }
                ],
            },
        )

    monkeypatch.setattr(particles, "propose_virtual_joint_region_paths", propose)
    receipt = particles.run_complete_combination_job(
        source, SimpleNamespace(training_identity="2" * 64), spec
    )

    assert observed["constituent_allocation"] == "template_binding_particle"
    assert observed["binding_particle_index"] == spec.binding_particle
    assert observed["combination_planner"] == "complete_combination_particle"
    assert observed["combination_particle_depth"] == spec.depth
    assert observed["combination_particle_index"] == spec.shard
    assert observed["combination_particle_count"] == spec.shard_count
    assert receipt["payload"]["status"] == "complete_with_realization_abstentions"
    telemetry = receipt["payload"]["telemetry"]
    assert "bound_constituent_ranks" not in telemetry
    assert telemetry["compiler_abstentions"] == 1
    assert identity(receipt["payload"]) == receipt["payload_sha256"]


def test_telemetry_allowlist_recursively_excludes_sensitive_payloads() -> None:
    raw = {
        "planning_expansions": 12,
        "valid_stop_targets_depth_3": 4,
        "bound_constituent_ranks": [
            {
                "binding": [1, 2],
                "slots": [3],
                "actions": [{"teacher_route": "hidden"}],
                "target": {"cell": "hidden", "endpoint": "hidden"},
                "source_state": {"smiles": "hidden"},
            }
        ],
        "teacher": {"nested": {"action": "hidden"}},
        "target_name": "hidden",
    }

    sanitized = particles.sanitize_particle_telemetry(raw)

    assert sanitized == {
        "planning_expansions": 12,
        "valid_stop_targets_depth_3": 4,
    }
    assert all(
        not isinstance(value, (dict, list, tuple)) for value in sanitized.values()
    )


def test_shuffled_receipts_reduce_identically_and_legacy_wins_alias() -> None:
    specs = particles.complete_combination_job_specs()
    particle_legacy_alias = _record(
        specs[0], smiles="CC", score=100.0, rank=1, program="particle-alias"
    )
    particle_worse_alias = _record(
        specs[1], smiles="CCC", score=-3.0, rank=1, program="worse"
    )
    particle_better_alias = _record(
        specs[2], smiles="CCC", score=-1.0, rank=2, program="better"
    )
    receipts = _receipts(
        {
            specs[0].job_id: [particle_legacy_alias],
            specs[1].job_id: [particle_worse_alias],
            specs[2].job_id: [particle_better_alias],
        }
    )
    legacy = [
        {
            "smiles": "CC",
            "proposal_lane": particles.ROUTE_EXPERT,
            "proposal_experts": [particles.ROUTE_EXPERT],
            "families": [particles.ROUTE_EXPERT],
            "program_families": [particles.ROUTE_EXPERT],
            "route_prior_score": -9.0,
            "route_proposal_rank": 4,
            "route_program_id": "legacy",
            "legacy_sentinel": True,
        }
    ]

    forward = particles.reduce_route_complete_region_pool(legacy, receipts)
    reverse = particles.reduce_route_complete_region_pool(legacy, reversed(receipts))

    assert forward == reverse
    records, telemetry = forward
    by_smiles = {row["smiles"]: row for row in records}
    assert by_smiles["CC"]["legacy_sentinel"] is True
    assert by_smiles["CC"]["route_program_id"] == "legacy"
    assert by_smiles["CC"]["route_generation_modes"] == [
        "complete_combination_particle",
        "legacy_complete_region",
    ]
    assert by_smiles["CCC"]["route_program_id"] == "better"
    assert len(by_smiles["CCC"]["route_particle_origins"]) == 2
    assert [row["route_proposal_rank"] for row in records] == list(
        range(1, len(records) + 1)
    )
    assert telemetry["proposal_experts"] == [particles.ROUTE_EXPERT]
    assert telemetry["controller_rng_consumed"] is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "missing_required_receipt"),
        ("corrupt", "corrupt_required_receipt"),
        ("failed", "operationally_failed_required_receipt"),
    ],
)
def test_incomplete_or_corrupt_particle_set_abstains_to_unchanged_legacy(
    mutation, reason
) -> None:
    legacy = [{"smiles": "CC", "route_proposal_rank": 9, "sentinel": [1, 2]}]
    receipts = _receipts()
    if mutation == "missing":
        receipts.pop()
    elif mutation == "corrupt":
        receipts[3] = {**receipts[3], "payload_sha256": "0" * 64}
    else:
        receipt = receipts[4]
        receipt["payload"]["status"] = "failed"
        receipt["payload_sha256"] = identity(receipt["payload"])

    records, telemetry = particles.reduce_route_complete_region_pool(legacy, receipts)

    assert records == legacy
    assert telemetry["particle_augmentation_status"] == "abstained"
    assert telemetry["particle_augmentation_abstention_reason"] == reason
    assert telemetry["depth_4_complete_coverage"] is False


def test_job_spec_rejects_any_attempt_to_add_depth_four() -> None:
    spec = particles.complete_combination_job_specs()[0]
    with pytest.raises(ValueError, match="depths 1..3"):
        replace(
            spec,
            depth=4,
            shard_count=48,
            total_combinations=194_580,
            shard_start=0,
            shard_stop=4_053,
        )


def test_particle_integration_imports_and_accepts_no_rng() -> None:
    source = inspect.getsource(particles)
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert "random" not in imported_roots
    assert "numpy" not in imported_roots
    assert (
        "rng"
        not in inspect.signature(particles.run_complete_combination_job).parameters
    )
    assert (
        "rng"
        not in inspect.signature(particles.reduce_route_complete_region_pool).parameters
    )
