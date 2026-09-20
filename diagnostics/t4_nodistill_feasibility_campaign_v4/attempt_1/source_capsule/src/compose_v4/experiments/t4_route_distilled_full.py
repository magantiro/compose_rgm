"""Conditional full T4 delta-0.4 and delta-0.6 route-distilled benchmark."""

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
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_distilled_qualification import (
    ACTOR,
    EMPTY_LIBRARY,
    POLICY_SELECTION,
    SOURCE_CONTRACT,
)

KIND = {0.4: "t4_route_distilled_full_delta04", 0.6: "t4_route_distilled_full_delta06"}
CONTRACT = {
    0.4: "configs/t4_route_distilled_full_delta04_v1.json",
    0.6: "configs/t4_route_distilled_full_delta06_v1.json",
}
PREFLIGHT = {
    0.4: "diagnostics/t4_route_distilled_full/preflight_delta04.json",
    0.6: "diagnostics/t4_route_distilled_full/preflight_delta06.json",
}
APP = "modal_apps/t4_route_distilled_app.py"
QUALIFICATION = "diagnostics/t4_route_distilled_qualification/result.json"
SEED_PAIRS = ((20260913, 1701), (20260914, 1702), (20260915, 1703))


def full_payload(delta: float) -> dict:
    if delta not in CONTRACT:
        raise ValueError("route-distilled full benchmark delta must be 0.4 or 0.6")
    return {
        "schema_version": "t4_route_distilled_full_policy_v1",
        "delta": delta,
        "cells": 15,
        "search_replicates": 3,
        "units": 45,
        "calls_per_unit": 1000,
        "search_call_ceiling": 45000,
        "combined_dual_threshold_call_ceiling": 90000,
        "max_concurrent_workers_across_thresholds": 15,
        "automatic_retries": 0,
        "confirmation_calls": 0,
        "plateau_stopping": False,
        "controller_docking_seed_pairs": [list(row) for row in SEED_PAIRS],
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "shallow_proposal_floor": SHALLOW_PROPOSAL_FLOOR,
        "policy_selection": POLICY_SELECTION,
        "policy_checkpoint": ACTOR,
        "runtime_teacher_lookup": False,
        "qualification": QUALIFICATION,
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
        "reason": "disabled_by_route_distilled_full_contract",
        "calls": len(curve),
    }


def _contract_path(delta: float) -> str:
    try:
        return CONTRACT[float(delta)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("unknown route-distilled full threshold") from error


def validate_contract(root: Path, contract: dict, *, delta: float) -> None:
    if contract.get("schema_version") != "t4_route_distilled_full_benchmark_v1":
        raise ValueError("route-distilled full contract schema changed")
    if contract.get("route_distilled_full") != full_payload(delta):
        raise ValueError("route-distilled full policy changed")
    if (
        contract.get("delta") != delta
        or contract.get("calls_per_unit") != 1000
        or contract.get("search_replicates") != 3
        or contract.get("search_call_ceiling") != 45000
        or contract.get("confirmation_call_ceiling") != 0
        or contract.get("total_call_ceiling") != 45000
        or contract.get("container_limit") != 15
        or contract.get("plateau_stop") is not None
        or len(contract.get("cells", ())) != 15
        or len(contract.get("units", ())) != 45
    ):
        raise ValueError("route-distilled full benchmark ceiling changed")
    if (
        contract.get("library_path") != EMPTY_LIBRARY
        or contract.get("library_programs") != 0
    ):
        raise ValueError("route-distilled full benchmark loaded stored routes")
    if (root / EMPTY_LIBRARY).read_text() != "[]\n":
        raise ValueError("route-distilled full empty library changed")
    if sorted(
        {
            (row["replicate"], row["controller_seed"], row["docking_seed"])
            for row in contract["units"]
        }
    ) != [(index, *pair) for index, pair in enumerate(SEED_PAIRS)]:
        raise ValueError("route-distilled full paired seeds changed")
    if sum(row["budget"] for row in contract["units"]) != 45000:
        raise ValueError("route-distilled full unit budgets changed")
    qualification = unseal(root / QUALIFICATION)
    if qualification.get("passed") is not True or qualification.get("decision") != (
        "full_dual_threshold_authorized"
    ):
        raise ValueError(
            "route-distilled qualification did not authorize the full wave"
        )
    selection = json.loads((root / POLICY_SELECTION).read_text())
    if selection.get("selected_policy") != "context_module_prototype" or not all(
        selection.get("selection_gate", {}).values()
    ):
        raise ValueError("route-distilled policy-selection gate did not pass")
    selected = selection.get("selected_all_route_checkpoint") or {}
    if selected.get("path") != ACTOR or selected.get("sha256") != sha256_file(
        root / ACTOR
    ):
        raise ValueError("route-distilled full checkpoint identity changed")
    source = unseal(root / SOURCE_CONTRACT)
    if canonical_bytes(contract["cells"]) != canonical_bytes(source["cells"]):
        raise ValueError("route-distilled full source molecules changed")
    if canonical_bytes(contract["controller"]) != canonical_bytes(source["controller"]):
        raise ValueError("route-distilled full controller recipe changed")
    for unit in contract["units"]:
        reference = next(
            row for row in source["units"] if row["unit_id"] == unit["unit_id"]
        )
        for field in (
            "budget",
            "cell",
            "controller_seed",
            "docking_seed",
            "original_seed",
            "replicate",
            "source_idx",
            "target",
            "unit_id",
        ):
            if unit[field] != reference[field]:
                raise ValueError(f"route-distilled full unit changed {field}")
        domain = unit["oracle_domain"]
        if domain["strict_filters"] != {
            "qed_gt": 0.6,
            "sa_lt": 4.0,
            "similarity_gt": delta,
        } or unit["oracle_protocol"] != identity(domain):
            raise ValueError("route-distilled full oracle identity changed")
    for path, digest in contract["inputs"].items():
        verify_file(root / path, digest)


def load_contract(root: Path, *, delta: float) -> dict:
    contract = unseal(root / _contract_path(delta))
    validate_contract(root, contract, delta=delta)
    return contract


@contextmanager
def _namespace(root: Path, *, delta: float):
    checkpoint = _actor_checkpoint(root)
    prior_policy = RouteDistilledDynamicOptimizer._configured_policy
    prior_identity = RouteDistilledDynamicOptimizer._configured_checkpoint_identity
    RouteDistilledDynamicOptimizer.configure_policy(checkpoint)
    original = (
        benchmark.KIND,
        benchmark.CONTRACT,
        benchmark.PREFLIGHT,
        benchmark.load_contract,
        benchmark.load_library,
        benchmark.initial_program_batch,
        benchmark.ProgramOptimizer,
        benchmark.competitive_plateau,
    )
    benchmark.KIND = KIND[delta]
    benchmark.CONTRACT = CONTRACT[delta]
    benchmark.PREFLIGHT = PREFLIGHT[delta]
    benchmark.load_contract = functools.partial(load_contract, delta=delta)
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
            benchmark.load_contract,
            benchmark.load_library,
            benchmark.initial_program_batch,
            benchmark.ProgramOptimizer,
            benchmark.competitive_plateau,
        ) = original
        RouteDistilledDynamicOptimizer._configured_policy = prior_policy
        RouteDistilledDynamicOptimizer._configured_checkpoint_identity = prior_identity


def validate_launch(task, root, validate_revision):
    delta = float(task.get("delta"))
    with _namespace(root, delta=delta):
        contract = benchmark.validate_launch(task, root, validate_revision)
    validate_contract(root, contract, delta=delta)
    return contract


def run_unit(task, root, artifacts, volume, validate_revision, dock):
    delta = float(task.get("delta"))
    load_contract(root, delta=delta)
    with _namespace(root, delta=delta):
        return benchmark.run_unit(
            task, root, artifacts, volume, validate_revision, dock
        )
