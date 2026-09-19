"""Isolated three-cell Dynamic COMPOSE v1 development experiment."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from compose_v4.control.dynamic_program_synthesis_v1 import (
    CONTEXT_EXPLORATION,
    GENERIC_MODULES,
    GENERIC_SUBSTITUENT_TEMPLATES,
    MAX_CONTEXT_CANDIDATES,
    MAX_CONTEXT_TRIALS,
    MAX_PANEL_CACHE_ENTRIES,
    MAX_RING_PATH_INTERNAL_ATOMS,
    MAX_RING_PATH_REPLACEMENT_ATOMS,
    MAX_RING_SUBSTITUENT_ATOMS,
    V1_CAPABILITY_PROBABILITY,
    V1_COMPOSITION_PATTERNS,
    V1_PATTERN_PROBABILITY,
    DynamicV1ProgramOptimizer,
    initial_dynamic_program_batch_v1,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

KIND = "t4_dynamic_v1_development"
CONTRACT = "configs/t4_dynamic_v1_development_v1.json"
PREFLIGHT = "diagnostics/t4_dynamic_v1/preflight.json"
APP = "modal_apps/t4_dynamic_v1_app.py"
APP_NAME = "compose-t4-dynamic-v1"
SOURCE_CONTRACT = "configs/t4_no_complete_routes_diagnostic_v1.json"
EMPTY_LIBRARY = "diagnostics/t4_dynamic_v1/empty_library.json"
REACHABILITY = "diagnostics/t4_dynamic_v1/teacher_reachability_v2.json"
SELECTED_UNITS = ("5ht1b_0_r0", "braf_1_r0", "jak2_1_r0")


def v1_contract_payload() -> dict:
    return {
        "schema_version": "t4_dynamic_v1_development_v1",
        "development_units": list(SELECTED_UNITS),
        "new_call_ceiling": 3000,
        "max_concurrent_workers": 3,
        "automatic_retries": 0,
        "confirmation_calls": 0,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "generic_modules": list(GENERIC_MODULES),
        "max_modules": 3,
        "max_ring_path_internal_atoms": MAX_RING_PATH_INTERNAL_ATOMS,
        "max_ring_path_replacement_atoms": MAX_RING_PATH_REPLACEMENT_ATOMS,
        "max_ring_substituent_atoms": MAX_RING_SUBSTITUENT_ATOMS,
        "max_context_candidates": MAX_CONTEXT_CANDIDATES,
        "max_context_trials": MAX_CONTEXT_TRIALS,
        "max_panel_cache_entries": MAX_PANEL_CACHE_ENTRIES,
        "context_exploration": CONTEXT_EXPLORATION,
        "v1_capability_probability": V1_CAPABILITY_PROBABILITY,
        "v1_pattern_probability": V1_PATTERN_PROBABILITY,
        "composition_patterns": [list(row) for row in V1_COMPOSITION_PATTERNS],
        "generic_substituent_templates": list(GENERIC_SUBSTITUENT_TEMPLATES),
        "intermediate_task_evaluations": 0,
        "full_146_and_dynamic_v0_reused_read_only": True,
        "remaining_t4_cells_authorized": False,
    }


def validate_dynamic_v1_contract(root: Path, contract: dict) -> None:
    if contract.get("dynamic_v1_development") != v1_contract_payload():
        raise ValueError("Dynamic-v1 development policy changed")
    if (
        contract.get("library_path") != EMPTY_LIBRARY
        or contract.get("library_programs") != 0
    ):
        raise ValueError("Dynamic-v1 must use the exact empty initial library")
    if (root / EMPTY_LIBRARY).read_text() != "[]\n":
        raise ValueError("Dynamic-v1 empty library changed")
    reachability = unseal(root / REACHABILITY)
    if (
        reachability.get("schema_version") != "t4_dynamic_v1_teacher_reachability_v1"
        or reachability.get("scope", {}).get("new_oracle_calls") != 0
        or reachability.get("summary", {}).get("coverage") != 1.0
        or reachability.get("summary", {}).get("execution_precision") != 1.0
    ):
        raise ValueError("Dynamic-v1 teacher-forced reachability gate did not pass")
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
            raise ValueError(f"Dynamic-v1 changed protected source field {field}")
    if contract["inputs"].get(REACHABILITY) != sha256_file(root / REACHABILITY):
        raise ValueError("Dynamic-v1 reachability input identity changed")


@contextmanager
def _v1_namespace():
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
    benchmark.initial_program_batch = initial_dynamic_program_batch_v1
    benchmark.ProgramOptimizer = DynamicV1ProgramOptimizer
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
    with _v1_namespace():
        contract = benchmark.load_contract(root)
    validate_dynamic_v1_contract(root, contract)
    return contract


def validate_launch(task, root, validate_revision):
    with _v1_namespace():
        contract = benchmark.validate_launch(task, root, validate_revision)
    validate_dynamic_v1_contract(root, contract)
    return contract


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    if task.get("unit_id") not in SELECTED_UNITS:
        raise ValueError("unit is outside the sealed Dynamic-v1 development cells")
    load_contract(root)
    with _v1_namespace():
        return benchmark.run_unit(
            task, root, artifacts, volume, validate_revision, dock
        )
