"""Frozen 15-cell Dynamic-v0 T4 comparator at strict similarity above 0.6."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from compose_v4.control.dynamic_program_synthesis import (
    ONLINE_COMPOSITION_PROBABILITY,
    DynamicProgramOptimizer,
    initial_dynamic_program_batch,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import unseal

KIND = "t4_dynamic_v0_full_suite_delta06"
CONTRACT = "configs/t4_dynamic_v0_full_suite_delta06.json"
PREFLIGHT = "diagnostics/t4_dynamic_v0_full_suite_delta06/preflight.json"
APP = "modal_apps/t4_dynamic_v0_full_suite_app.py"
APP_NAME = "compose-t4-dynamic-v0-full-suite-delta06"
EMPTY_LIBRARY = "diagnostics/t4_dynamic_v0_full_suite_delta06/empty_library.json"
SOURCE_CONTRACT = "configs/t4_no_complete_routes_diagnostic_v1.json"
_BASE_CONFIGURED = benchmark.configured


def configured(contract: dict, unit: dict):
    """Reproduce the exact historical Dynamic-v0 search configuration."""

    return replace(
        _BASE_CONFIGURED(contract, unit),
        current_state_edit_probability=0.0,
        composition_probability=ONLINE_COMPOSITION_PROBABILITY,
        max_composed_programs=3,
    )


def _no_plateau(curve, ivg_mean, policy):
    del ivg_mean, policy
    return {"stop": False, "reason": "disabled_by_frozen_contract", "calls": len(curve)}


def load_contract(root: Path) -> dict:
    contract = unseal(root / CONTRACT)
    units = contract.get("units", ())
    if (
        contract.get("schema_version") != "t4_dynamic_v0_full_suite_delta06_v1"
        or contract.get("delta") != 0.6
        or contract.get("search_replicates") != 1
        or contract.get("calls_per_unit") != 1000
        or contract.get("search_call_ceiling") != 15000
        or contract.get("confirmation_call_ceiling") != 0
        or contract.get("total_call_ceiling") != 15000
        or contract.get("container_limit") != 15
        or contract.get("plateau_stop") != {"enabled": False}
        or contract.get("library_path") != EMPTY_LIBRARY
        or contract.get("library_programs") != 0
        or len(contract.get("cells", {})) != 15
        or len(units) != 15
        or {row.get("replicate") for row in units} != {0}
        or {row.get("controller_seed") for row in units} != {20260913}
        or {row.get("docking_seed") for row in units} != {1701}
        or sum(row.get("budget", 0) for row in units) != 15000
    ):
        raise ValueError("Dynamic-v0 delta-0.6 contract differs from the authorized scope")
    if json.loads((root / EMPTY_LIBRARY).read_text()) != []:
        raise ValueError("Dynamic-v0 initial program bank must be empty")
    for path, digest in contract["inputs"].items():
        verify_file(root / path, digest)
    source = unseal(root / SOURCE_CONTRACT)
    if canonical_bytes(contract["controller"]) != canonical_bytes(source["controller"]):
        raise ValueError("historical Dynamic-v0 base controller changed")
    return contract


@contextmanager
def dynamic_v0_namespace():
    original = (
        benchmark.KIND,
        benchmark.CONTRACT,
        benchmark.PREFLIGHT,
        benchmark.load_contract,
        benchmark.load_library,
        benchmark.configured,
        benchmark.initial_program_batch,
        benchmark.ProgramOptimizer,
        benchmark.competitive_plateau,
    )
    benchmark.KIND = KIND
    benchmark.CONTRACT = CONTRACT
    benchmark.PREFLIGHT = PREFLIGHT
    benchmark.load_contract = load_contract
    benchmark.load_library = lambda _root, _contract: ()
    benchmark.configured = configured
    benchmark.initial_program_batch = initial_dynamic_program_batch
    benchmark.ProgramOptimizer = DynamicProgramOptimizer
    benchmark.competitive_plateau = _no_plateau
    try:
        yield
    finally:
        (
            benchmark.KIND,
            benchmark.CONTRACT,
            benchmark.PREFLIGHT,
            benchmark.load_contract,
            benchmark.load_library,
            benchmark.configured,
            benchmark.initial_program_batch,
            benchmark.ProgramOptimizer,
            benchmark.competitive_plateau,
        ) = original


def validate_launch(task: dict, root: Path, validate_revision):
    with dynamic_v0_namespace():
        return benchmark.validate_launch(task, root, validate_revision)


def run_unit(task: dict, root: Path, artifacts: Path, volume, validate_revision, dock):
    contract = load_contract(root)
    if task.get("unit_id") not in {row["unit_id"] for row in contract["units"]}:
        raise ValueError("unit is outside the frozen 15-cell Dynamic-v0 census")
    with dynamic_v0_namespace():
        return benchmark.run_unit(task, root, artifacts, volume, validate_revision, dock)


def material_hashes(root: Path) -> dict[str, str]:
    return {path: sha256_file(root / path) for path in load_contract(root)["inputs"]}
