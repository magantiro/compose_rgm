"""Matched T4 complete-route ablation and dynamic-synthesis diagnostic.

The scientific search loop is the frozen full-suite implementation. This module
selects either the mechanically derived 69-program library or a route-free
dynamic proposer, gives each arm an independent artifact namespace, and
restricts launch to three predeclared replicate-0 units.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    FRESH_SYNTHESIS_PROBABILITY,
    GENERIC_MODULES,
    MAX_SEGMENT_LENGTH,
    NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES,
    NEAR_CAPACITY_MODULE_WEIGHTS,
    ONLINE_COMPOSITION_PROBABILITY,
    DynamicProgramOptimizer,
    initial_dynamic_program_batch,
)
from compose_v4.control.edit_program import EditProgram
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

KIND = "t4_no_complete_routes_diagnostic"
ARMS = ("69_only", "dynamic_only")
ARM_KINDS = {arm: f"{KIND}_{arm}" for arm in ARMS}
CONTRACT = "configs/t4_no_complete_routes_diagnostic_v1.json"
PREFLIGHT = "diagnostics/t4_no_complete_routes/attempt_1/preflight.json"
APP = "modal_apps/t4_no_complete_routes_app.py"
APP_NAME = "compose-t4-no-complete-routes"
SOURCE_CONTRACT = "configs/t4_frozen_program_benchmark_v2.json"
SOURCE_LIBRARY = (
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
)
LIBRARY = "diagnostics/t4_no_complete_routes/attempt_1/library_69.json"
SELECTED_UNITS = ("5ht1b_0_r0", "braf_1_r0", "jak2_1_r0")
PREFLIGHT_PACKAGING_FAILURE = (
    "diagnostics/t4_no_complete_routes/attempt_1/prequery_failure_0001.json"
)
PREFLIGHT_PACKAGING_SOURCE = {
    "source_commit": "ddf92515c8371dc4537075a34ca4e0494075b8eb",
    "source_contract_sha256": (
        "7af070e626575b8e09303a8d84f5e2396ee04830cd841ae720afad53822c7a91"
    ),
    "source_input_sha256": {
        APP: "147a7769fdbe1cb522854e57dc30ae633c2da5270b35912a7eddd37b07248e19",
        "docs/T4_NO_COMPLETE_ROUTES_DIAGNOSTIC.md": (
            "8c151e046fd1629ac4bf47a3a0e13346032869ad3cc81b247b6759e077dfcc65"
        ),
        "src/compose_v4/experiments/t4_no_complete_routes.py": (
            "fb63196b23bb015a69fc159733b528bc60b3231694662dc14b6d264958933ae0"
        ),
        "tools/t4_no_complete_routes.py": (
            "b0759625f64d1226ce0d9eb0ce480ec8343ee905ce46cf40772d9f73f8d4fad2"
        ),
    },
}
COMPATIBILITY_REPINS = {
    "src/compose_v4/control/adaptive_program_optimizer.py": {
        "source_sha256": "587a222c2f517cadf43c4d8e01a9239576992634f133329a073eccfa89b94bcb",
        "diagnostic_sha256": "3d185c06d4b12055119ce30e0cd373c84aa8c3dcbecf1184e361b46c22658e21",
        "decision_equivalence": (
            "adds an unused cold_start_retrieval_candidates=0 default and accepts "
            "an interruption tombstone status that fresh no-retry units cannot emit"
        ),
    },
    "src/compose_v4/control/program_transfer.py": {
        "source_sha256": "cd1a570f42f84297bef7b330b72ca30f6fa108ab785b086cacecb0a5f85126b5",
        "diagnostic_sha256": "309a749e4f16aa2ede79d261ce5ccf5128ffe784d3b3529e8f3235272fdfc32c",
        "decision_equivalence": (
            "adds direct retrieval only when cold_start_retrieval_candidates is "
            "positive; the cloned frozen controller omits it and therefore uses zero"
        ),
    },
    "tools/t4_frozen_program_benchmark.py": {
        "source_sha256": "32e6490fac79d0a3f2ecd2e06785cc69ff46062b0c57078c296994315ac0875e",
        "diagnostic_sha256": "8802bac38f60630aef05303f8b18e1d81dceee09e6bdbec02415a931e8b0b098",
        "decision_equivalence": (
            "changes only the local read-only status command; no worker imports it"
        ),
    },
}


def is_complete_route(row: dict) -> bool:
    """The prospectively frozen mechanical removal predicate."""
    blocks = row.get("program", {}).get("blocks")
    return (
        isinstance(blocks, list)
        and len(blocks) == 1
        and blocks[0].get("label") == "compiled_complete_transformation"
    )


def partition_library(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return retained and removed rows without changing their relative order."""
    retained = [row for row in rows if not is_complete_route(row)]
    removed = [row for row in rows if is_complete_route(row)]
    return retained, removed


def program_ids(rows: list[dict]) -> list[str]:
    return [EditProgram.from_payload(row["program"]).program_id for row in rows]


def recovered_improvement(source_score, full_score, ablated_score):
    """Descriptive lower-is-better fraction of full improvement recovered."""
    if any(value is None for value in (source_score, full_score, ablated_score)):
        return None
    denominator = float(source_score) - float(full_score)
    if denominator == 0.0:
        return None
    return (float(source_score) - float(ablated_score)) / denominator


def validate_ablation_contract(root: Path, contract: dict) -> None:
    diagnostic = contract.get("complete_route_ablation")
    if not isinstance(diagnostic, dict):
        raise TypeError("complete-route diagnostic metadata is absent")
    if (
        diagnostic.get("schema_version") != "t4_complete_route_ablation_v1"
        or tuple(diagnostic.get("selected_units", ())) != SELECTED_UNITS
        or diagnostic.get("new_call_ceiling") != 6000
        or diagnostic.get("max_concurrent_workers") != 6
        or diagnostic.get("automatic_retries") != 0
        or diagnostic.get("confirmation_calls") != 0
    ):
        raise ValueError("complete-route diagnostic scope changed")
    dynamic = diagnostic.get("dynamic_only")
    if (
        not isinstance(dynamic, dict)
        or dynamic.get("initial_route_archive") != []
        or dynamic.get("source_library_rows_loaded") != 0
        or tuple(dynamic.get("generic_modules", ())) != GENERIC_MODULES
        or dynamic.get("max_modules") != 3
        or dynamic.get("max_segment_length") != MAX_SEGMENT_LENGTH
        or dynamic.get("capacity_aware_threshold") != CAPACITY_AWARE_THRESHOLD
        or dynamic.get("capacity_bootstrap_pair_attempts")
        != CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS
        or dynamic.get("capacity_bootstrap_single_attempts")
        != CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
        or tuple(dynamic.get("near_capacity_module_count_probabilities", ()))
        != NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES
        or dynamic.get("near_capacity_module_weights") != NEAR_CAPACITY_MODULE_WEIGHTS
        or dynamic.get("fresh_synthesis_probability") != FRESH_SYNTHESIS_PROBABILITY
        or dynamic.get("online_composition_probability")
        != ONLINE_COMPOSITION_PROBABILITY
        or dynamic.get("intermediate_task_evaluations") != 0
    ):
        raise ValueError("dynamic-only proposal contract changed")
    if diagnostic.get("runtime_compatibility_repins") != COMPATIBILITY_REPINS:
        raise ValueError("frozen-source compatibility repin ledger changed")
    repairs = diagnostic.get("preflight_packaging_repairs")
    if repairs is not None and (
        len(repairs) != 1
        or repairs[0].get("source") != PREFLIGHT_PACKAGING_SOURCE
        or repairs[0].get("failure_path") != PREFLIGHT_PACKAGING_FAILURE
        or repairs[0].get("failure_sha256")
        != sha256_file(root / PREFLIGHT_PACKAGING_FAILURE)
        or repairs[0].get("added_remote_material") != SOURCE_LIBRARY
        or repairs[0].get("oracle_calls") != 0
        or repairs[0].get("scientific_policy_changed") is not False
    ):
        raise ValueError("preflight packaging repair lineage changed")
    if contract["library_path"] != LIBRARY or contract["library_programs"] != 69:
        raise ValueError("diagnostic did not select the exact 69-program library")

    source = unseal(root / SOURCE_CONTRACT)
    for path, record in COMPATIBILITY_REPINS.items():
        if (
            source["inputs"].get(path) != record["source_sha256"]
            or contract["inputs"].get(path) != record["diagnostic_sha256"]
            or sha256_file(root / path) != record["diagnostic_sha256"]
        ):
            raise ValueError(
                f"compatibility repin does not match exact lineage: {path}"
            )
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
            raise ValueError(
                f"complete-route diagnostic changed protected field {field}"
            )

    source_rows = __import__("json").loads((root / SOURCE_LIBRARY).read_text())
    derived, removed = partition_library(source_rows)
    actual = __import__("json").loads((root / LIBRARY).read_text())
    if len(source_rows) != 146 or len(derived) != 69 or len(removed) != 77:
        raise ValueError("complete-route partition does not equal 146 = 69 + 77")
    if canonical_bytes(actual) != canonical_bytes(derived):
        raise ValueError("69-program library is not the exact mechanical ablation")
    if program_ids(actual) != diagnostic.get("retained_program_ids"):
        raise ValueError("retained program identity ledger changed")
    if program_ids(removed) != diagnostic.get("removed_program_ids"):
        raise ValueError("removed program identity ledger changed")
    if sha256_file(root / SOURCE_LIBRARY) != diagnostic.get("source_library_sha256"):
        raise ValueError("source 146-program library changed")
    if sha256_file(root / LIBRARY) != diagnostic.get("derived_library_sha256"):
        raise ValueError("derived 69-program library changed")


@contextmanager
def _diagnostic_namespace(arm=None):
    original = (
        benchmark.KIND,
        benchmark.CONTRACT,
        benchmark.PREFLIGHT,
        benchmark.load_library,
        benchmark.configured,
        benchmark.initial_program_batch,
        benchmark.ProgramOptimizer,
    )
    benchmark.KIND = KIND if arm is None else ARM_KINDS[arm]
    benchmark.CONTRACT, benchmark.PREFLIGHT = CONTRACT, PREFLIGHT
    if arm == "dynamic_only":
        original_configured = benchmark.configured

        def dynamic_configured(contract, unit):
            return replace(
                original_configured(contract, unit),
                current_state_edit_probability=0.0,
                composition_probability=ONLINE_COMPOSITION_PROBABILITY,
                max_composed_programs=3,
            )

        benchmark.load_library = lambda _root, _contract: ()
        benchmark.configured = dynamic_configured
        benchmark.initial_program_batch = initial_dynamic_program_batch
        benchmark.ProgramOptimizer = DynamicProgramOptimizer
    try:
        yield
    finally:
        (
            benchmark.KIND,
            benchmark.CONTRACT,
            benchmark.PREFLIGHT,
            benchmark.load_library,
            benchmark.configured,
            benchmark.initial_program_batch,
            benchmark.ProgramOptimizer,
        ) = original


def load_contract(root: Path) -> dict:
    with _diagnostic_namespace():
        contract = benchmark.load_contract(root)
    validate_ablation_contract(root, contract)
    return contract


def validate_launch(task, root, validate_revision):
    with _diagnostic_namespace():
        contract = benchmark.validate_launch(task, root, validate_revision)
    validate_ablation_contract(root, contract)
    return contract


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    arm = task.get("arm")
    if arm not in ARMS:
        raise ValueError("task lacks a sealed diagnostic arm")
    if task.get("unit_id") not in SELECTED_UNITS:
        raise ValueError("unit is outside the sealed three-cell diagnostic")
    load_contract(root)
    benchmark_task = {key: value for key, value in task.items() if key != "arm"}
    with _diagnostic_namespace(arm):
        return benchmark.run_unit(
            benchmark_task, root, artifacts, volume, validate_revision, dock
        )
