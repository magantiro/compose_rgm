import json
from dataclasses import replace
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_population_controller import PmoPopulationController, _niche_order

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]


def _controller(seed=23):
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=32,
        candidates_per_batch=8,
        wall_seconds=5,
        parent_allocation="niche_score",
    )
    return PmoPopulationController(
        config,
        source_group="fixture-source",
        oracle_protocol="fixture-oracle",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
    )


def test_plan_chunks_cover_every_shared_latent_without_omission():
    controller = _controller()
    order = [row["plan_id"] for row in controller._jump_plan_order()]
    chunks = []
    for batch in range(4):
        offset = batch * 32 % len(order)
        chunks.extend(
            order[offset : offset + 32]
            if offset + 32 <= len(order)
            else order[offset:] + order[: (offset + 32) % len(order)]
        )
    assert len(set(chunks)) == len(CHECKPOINT["plan_latents"])
    assert set(chunks) == {row["plan_id"] for row in CHECKPOINT["plan_latents"]}


def test_plan_order_is_deterministic_and_scale_diverse():
    first, second = _controller(31), _controller(31)
    first.batches = second.batches = 2
    assert [row["plan_id"] for row in first._jump_plan_order()] == [
        row["plan_id"] for row in second._jump_plan_order()
    ]
    selected = first._jump_plan_order()[:32]
    bands = {
        "small"
        if row["primitive_count"] <= 7
        else "medium"
        if row["primitive_count"] <= 15
        else "large"
        for row in selected
    }
    assert bands == {"small", "medium", "large"}


def test_niche_order_accepts_serialized_fingerprints():
    rows = [
        {"candidate_id": "a", "fiber_fingerprint": [1, 2]},
        {"candidate_id": "b", "fiber_fingerprint": [9, 10]},
        {"candidate_id": "c", "fiber_fingerprint": [1, 2, 3]},
    ]
    order = _niche_order(rows, seed=7)
    assert sorted(order) == [0, 1, 2]


def test_snapshot_restore_binds_same_joint_checkpoint():
    controller = _controller(37)
    controller.jump_rng.random()
    restored = PmoPopulationController.restore(
        controller.snapshot(), hierarchy=None, jump_checkpoint=CHECKPOINT
    )
    assert restored.jump_checkpoint_id == controller.jump_checkpoint_id
    assert restored.jump_rng.bit_generator.state == controller.jump_rng.bit_generator.state
    assert restored.population_state == controller.population_state
