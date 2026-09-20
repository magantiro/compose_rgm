"""Five-cell scored qualification for the frozen route-distilled controller."""

from __future__ import annotations

import functools
import gzip
import json
from contextlib import contextmanager
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_dynamic_optimizer import (
    SHALLOW_PROPOSAL_FLOOR,
    RouteDistilledDynamicOptimizer,
    initial_route_distilled_program_batch,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

KIND = "t4_route_distilled_qualification"
CONTRACT = "configs/t4_route_distilled_qualification_v1.json"
PREFLIGHT = "diagnostics/t4_route_distilled_qualification/preflight.json"
APP = "modal_apps/t4_route_distilled_app.py"
APP_NAME = "compose-t4-route-distilled"
SOURCE_CONTRACT = "configs/t4_frozen_program_benchmark_v2.json"
EMPTY_LIBRARY = "diagnostics/t4_dynamic_v1/empty_library.json"
POLICY_SELECTION = "diagnostics/t4_route_policy_selection/attempt_1/result.json"
ACTOR = "diagnostics/t4_route_distillation/attempt_3/actor.json.gz"
SELECTED_UNITS = (
    "5ht1b_0_r0",
    "braf_1_r0",
    "fa7_0_r0",
    "jak2_1_r0",
    "parp1_0_r0",
)


def qualification_payload() -> dict:
    return {
        "schema_version": "t4_route_distilled_qualification_policy_v1",
        "development_units": list(SELECTED_UNITS),
        "controller_seed": 20260913,
        "docking_seed": 1701,
        "calls_per_unit": 1000,
        "new_call_ceiling": 5000,
        "max_concurrent_workers": 5,
        "automatic_retries": 0,
        "confirmation_calls": 0,
        "plateau_stopping": False,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "shallow_proposal_floor": SHALLOW_PROPOSAL_FLOOR,
        "policy_selection": POLICY_SELECTION,
        "policy_checkpoint": ACTOR,
        "runtime_teacher_lookup": False,
        "full_benchmark_conditional_on_qualification": True,
        "qualification_gate": {
            "durable_units": 5,
            "minimum_scored_endpoints_per_cell": 1,
            "minimum_cells_at_or_better_than_ivg_mean": 4,
        },
    }


def _actor_checkpoint(root: Path) -> dict:
    with gzip.open(root / ACTOR, "rt") as handle:
        envelope = json.load(handle)
    if (
        set(envelope) != {"payload", "payload_sha256"}
        or identity(envelope["payload"]) != envelope["payload_sha256"]
    ):
        raise ValueError("route-distilled actor checkpoint changed")
    return envelope["payload"]


def _no_plateau(curve, ivg_mean, policy):
    del ivg_mean, policy
    return {
        "stop": False,
        "reason": "disabled_by_route_distilled_qualification_contract",
        "calls": len(curve),
    }


@contextmanager
def _namespace(root: Path):
    checkpoint = _actor_checkpoint(root)
    prior_policy = RouteDistilledDynamicOptimizer._configured_policy
    prior_identity = RouteDistilledDynamicOptimizer._configured_checkpoint_identity
    RouteDistilledDynamicOptimizer.configure_policy(checkpoint)
    original = (
        benchmark.KIND,
        benchmark.CONTRACT,
        benchmark.PREFLIGHT,
        benchmark.load_library,
        benchmark.initial_program_batch,
        benchmark.ProgramOptimizer,
        benchmark.competitive_plateau,
    )
    benchmark.KIND = KIND
    benchmark.CONTRACT = CONTRACT
    benchmark.PREFLIGHT = PREFLIGHT
    benchmark.load_library = lambda _root, _contract: ()
    benchmark.initial_program_batch = functools.partial(
        initial_route_distilled_program_batch,
        policy=RouteDistilledDynamicOptimizer._configured_policy,
    )
    benchmark.ProgramOptimizer = RouteDistilledDynamicOptimizer
    benchmark.competitive_plateau = _no_plateau
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
            benchmark.competitive_plateau,
        ) = original
        RouteDistilledDynamicOptimizer._configured_policy = prior_policy
        RouteDistilledDynamicOptimizer._configured_checkpoint_identity = prior_identity


def validate_contract(root: Path, contract: dict) -> None:
    if contract.get("route_distilled_qualification") != qualification_payload():
        raise ValueError("route-distilled qualification policy changed")
    if (
        contract.get("library_path") != EMPTY_LIBRARY
        or contract.get("library_programs") != 0
    ):
        raise ValueError("route-distilled qualification must load no stored route")
    if (root / EMPTY_LIBRARY).read_text() != "[]\n":
        raise ValueError("route-distilled empty library changed")
    selection = json.loads((root / POLICY_SELECTION).read_text())
    if (
        selection.get("selected_policy") != "context_module_prototype"
        or selection.get("costs") != {"oracle_calls": 0, "docking_calls": 0}
        or not all(selection.get("selection_gate", {}).values())
    ):
        raise ValueError("route-distilled policy-selection gate did not pass")
    selected = selection.get("selected_all_route_checkpoint") or {}
    if selected.get("path") != ACTOR or selected.get("sha256") != sha256_file(
        root / ACTOR
    ):
        raise ValueError("selected route-distilled checkpoint identity changed")
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
        "controller",
        "runtime_input_sha256",
        "ivg_reported",
        "upstream_protocol",
    )
    for field in protected:
        if canonical_bytes(contract[field]) != canonical_bytes(source[field]):
            raise ValueError(f"route-distilled qualification changed {field}")
    if contract["delta"] != 0.4 or contract["calls_per_unit"] != 1000:
        raise ValueError("route-distilled qualification endpoint protocol changed")


def load_contract(root: Path) -> dict:
    with _namespace(root):
        contract = benchmark.load_contract(root)
    validate_contract(root, contract)
    return contract


def validate_launch(task, root, validate_revision):
    with _namespace(root):
        contract = benchmark.validate_launch(task, root, validate_revision)
    validate_contract(root, contract)
    return contract


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    if task.get("unit_id") not in SELECTED_UNITS:
        raise ValueError("unit is outside the sealed route-distilled qualification")
    load_contract(root)
    with _namespace(root):
        return benchmark.run_unit(
            task, root, artifacts, volume, validate_revision, dock
        )
