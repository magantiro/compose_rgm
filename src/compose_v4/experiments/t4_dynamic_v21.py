"""Five-cell Dynamic COMPOSE v2.1 pooled-channel development experiment."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNEL_CANDIDATE_LIMIT,
    EXPLORATION_FLOOR,
    UCB_EXPLORATION,
    DynamicV21ProgramOptimizer,
    initial_dynamic_program_batch_v21,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import canonical_bytes
from compose_v4.experiments.t4_matched_pilot import unseal

KIND = "t4_dynamic_v21_development"
CONTRACT = "configs/t4_dynamic_v21_development_v1.json"
PREFLIGHT = "diagnostics/t4_dynamic_v21/preflight.json"
APP = "modal_apps/t4_dynamic_v21_app.py"
APP_NAME = "compose-t4-dynamic-v21"
SOURCE_CONTRACT = "configs/t4_dynamic_v1_development_v1.json"
EMPTY_LIBRARY = "diagnostics/t4_dynamic_v1/empty_library.json"
SELECTED_UNITS = (
    "parp1_0_r0",
    "fa7_0_r0",
    "5ht1b_0_r0",
    "braf_1_r0",
    "jak2_1_r0",
)


def v21_contract_payload() -> dict:
    return {
        "schema_version": "t4_dynamic_v21_development_v1",
        "development_units": list(SELECTED_UNITS),
        "new_call_ceiling": 5000,
        "max_concurrent_workers": 5,
        "automatic_retries": 0,
        "confirmation_calls": 0,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "planner_channels": [SHALLOW_CHANNEL, STRUCTURED_CHANNEL],
        "channel_candidate_limit": CHANNEL_CANDIDATE_LIMIT,
        "attempts_per_channel_per_batch": 128,
        "wall_seconds_per_channel_per_batch": 45,
        "arbitrated_candidate_limit": 16,
        "ucb_exploration": UCB_EXPLORATION,
        "eligible_exploration_floor": EXPLORATION_FLOOR,
        "independent_rng_streams": ["shallow", "structured", "arbitration"],
        "max_modules": 3,
        "intermediate_task_evaluations": 0,
        "offline_comparators": ["Dynamic-v0", "Dynamic-v1", "Full-146"],
        "comparison_outcomes_available_to_runtime": False,
        "remaining_t4_cells_authorized": False,
    }


def validate_dynamic_v21_contract(root: Path, contract: dict) -> None:
    if contract.get("dynamic_v21_development") != v21_contract_payload():
        raise ValueError("Dynamic-v2.1 development policy changed")
    if (
        contract.get("library_path") != EMPTY_LIBRARY
        or contract.get("library_programs") != 0
    ):
        raise ValueError("Dynamic-v2.1 must use the exact empty initial library")
    if (root / EMPTY_LIBRARY).read_text() != "[]\n":
        raise ValueError("Dynamic-v2.1 empty library changed")
    source = unseal(root / SOURCE_CONTRACT)
    protected = (
        "delta",
        "cells",
        "units",
        "search_replicates",
        "calls_per_unit",
        "cold_start_seed",
        "max_rounds",
        "max_consecutive_empty_rounds",
        "plateau_stop",
        "controller",
        "runtime_input_sha256",
        "ivg_reported",
        "upstream_protocol",
    )
    for field in protected:
        if canonical_bytes(contract[field]) != canonical_bytes(source[field]):
            raise ValueError(f"Dynamic-v2.1 changed protected source field {field}")
    forbidden = (
        "comparison_reference",
        "dynamic_v2_offline_comparison_reference",
        "dynamic_v1_offline_comparison_reference",
    )
    if any(name in contract.get("inputs", {}) for name in forbidden):
        raise ValueError("Dynamic-v2.1 comparison outcomes entered runtime inputs")


@contextmanager
def _v21_namespace():
    original = (
        benchmark.KIND,
        benchmark.CONTRACT,
        benchmark.PREFLIGHT,
        benchmark.load_library,
        benchmark.initial_program_batch,
        benchmark.ProgramOptimizer,
    )
    benchmark.KIND = KIND
    benchmark.CONTRACT = CONTRACT
    benchmark.PREFLIGHT = PREFLIGHT
    benchmark.load_library = lambda _root, _contract: ()
    benchmark.initial_program_batch = initial_dynamic_program_batch_v21
    benchmark.ProgramOptimizer = DynamicV21ProgramOptimizer
    try:
        yield
    finally:
        (
            benchmark.KIND,
            benchmark.CONTRACT,
            benchmark.PREFLIGHT,
            benchmark.load_library,
            benchmark.initial_program_batch,
            benchmark.ProgramOptimizer,
        ) = original


def load_contract(root: Path) -> dict:
    with _v21_namespace():
        contract = benchmark.load_contract(root)
    validate_dynamic_v21_contract(root, contract)
    return contract


def validate_launch(task, root, validate_revision):
    with _v21_namespace():
        contract = benchmark.validate_launch(task, root, validate_revision)
    validate_dynamic_v21_contract(root, contract)
    return contract


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    if task.get("unit_id") not in SELECTED_UNITS:
        raise ValueError("unit is outside the sealed Dynamic-v2.1 development cells")
    load_contract(root)
    with _v21_namespace():
        return benchmark.run_unit(
            task, root, artifacts, volume, validate_revision, dock
        )
