"""Audit and fit the frozen zero-oracle T4 measured-utility comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control.target_conditioned_utility_selector import (
    TARGETS,
    MeasuredEndpoint,
    evaluate_scores,
    fit_utility_ranker,
    fixed_structural_score,
    rows_by_fold,
    stable_identity,
    strict_pairs,
    target_pair_counts,
)
from compose_v4.data.scaffold_partition import murcko_scaffold
from compose_v4.rewrite.trace_shard import decode_state

CONTRACT = Path("configs/t4_target_conditioned_utility_selector_v3.json")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(content)


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _load_bound_json(
    root: Path, spec: Mapping[str, Any], *, compressed: bool = False
) -> dict:
    raw = _git_bytes(root, str(spec["git_revision"]), str(spec["path"]))
    observed = sha256_bytes(raw)
    if observed != spec["sha256"]:
        raise RuntimeError(
            f"bound input hash mismatch for {spec['git_revision']}:{spec['path']}: "
            f"expected {spec['sha256']}, observed {observed}"
        )
    decoded = gzip.decompress(raw) if compressed else raw
    payload = json.loads(decoded)
    expected_payload = spec.get("payload_sha256")
    if expected_payload is not None:
        if payload.get("payload_sha256") != expected_payload:
            raise RuntimeError(f"payload identity mismatch for {spec['path']}")
        if canonical_json_hash(payload["payload"]) != expected_payload:
            raise RuntimeError(f"payload seal does not verify for {spec['path']}")
    return payload


def _resolve_contract(
    root: Path, contract_document: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the v4 add-only extension onto the immutable v3 payload."""

    payload = contract_document["payload"]
    if canonical_json_hash(payload) != contract_document["contract_sha256"]:
        raise RuntimeError("utility-selector contract self-hash mismatch")
    if (
        payload.get("schema_version")
        != "t4_target_conditioned_utility_selector_contract_v4"
    ):
        return dict(payload), {"mode": "standalone_contract"}

    base_spec = payload["base_contract"]
    raw = _git_bytes(root, base_spec["git_revision"], base_spec["path"])
    observed = sha256_bytes(raw)
    if observed != base_spec["sha256"]:
        raise RuntimeError(
            "v4 base-contract physical hash mismatch: "
            f"expected {base_spec['sha256']}, observed {observed}"
        )
    base_document = json.loads(raw)
    if (
        base_document.get("contract_sha256") != base_spec["payload_sha256"]
        or canonical_json_hash(base_document["payload"]) != base_spec["payload_sha256"]
    ):
        raise RuntimeError("v4 base-contract payload seal does not verify")
    effective = json.loads(json.dumps(base_document["payload"]))
    if "utility_acquisition" in effective["measured_inputs"]:
        raise RuntimeError("v4 base contract already contains utility acquisition")
    effective["measured_inputs"]["utility_acquisition"] = json.loads(
        json.dumps(payload["added_measured_input"])
    )
    effective["outputs"] = json.loads(json.dumps(payload["outputs"]))
    return effective, {
        "mode": "add_only_v4_extension",
        "base_contract": dict(base_spec),
        "frozen_unchanged": list(payload["frozen_unchanged"]),
        "added_artifact": payload["added_measured_input"]["artifact_id"],
    }


def _canonical_smiles(state: Mapping[str, Any]) -> str:
    smiles = molecular_graph_to_smiles(decode_state(dict(state)))
    if smiles is None:
        raise ValueError("exact state does not serialize to a supported molecule")
    return smiles


def _state_hash(state: Mapping[str, Any]) -> str:
    return stable_identity({"canonical_smiles": _canonical_smiles(state)})


def _cell_parts(cell: str) -> tuple[str, int]:
    target, suffix = cell.rsplit("_", 1)
    source_idx = int(suffix)
    if target not in TARGETS or source_idx not in (0, 1, 2):
        raise ValueError(f"invalid T4 cell: {cell}")
    return target, source_idx


def _make_row(
    *,
    row_id: str,
    artifact: str,
    target: str,
    cell: str,
    delta: float,
    oracle_protocol: str,
    docking_seed: int,
    docking_score: float,
    source_state: Mapping[str, Any],
    endpoint_state: Mapping[str, Any],
    lineage_ids: list[str],
    provenance: Mapping[str, Any],
) -> MeasuredEndpoint:
    parsed_target, source_idx = _cell_parts(cell)
    if parsed_target != target:
        raise ValueError(f"target/cell mismatch for {row_id}")
    source_smiles = _canonical_smiles(source_state)
    endpoint_smiles = _canonical_smiles(endpoint_state)
    scaffold = murcko_scaffold(endpoint_smiles)
    if scaffold is None:
        raise ValueError(f"endpoint scaffold is unavailable for {row_id}")
    return MeasuredEndpoint(
        row_id=row_id,
        artifact=artifact,
        target=target,
        cell=cell,
        source_idx=source_idx,
        delta=float(delta),
        oracle_protocol=oracle_protocol,
        docking_seed=int(docking_seed),
        docking_score=float(docking_score),
        source_smiles=source_smiles,
        endpoint_smiles=endpoint_smiles,
        source_state_sha256=_state_hash(source_state),
        endpoint_state_sha256=_state_hash(endpoint_state),
        scaffold=scaffold,
        lineage_ids=tuple(sorted(set(lineage_ids))),
        provenance=dict(provenance),
    )


def _load_second_generation(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[MeasuredEndpoint], list[dict]]:
    spec = config["measured_inputs"]["second_generation"]
    common_revision = spec["git_revision"]
    result_spec = {**spec["result"], "git_revision": common_revision}
    result = _load_bound_json(root, result_spec)
    result_rows = result["payload"]["rows"]
    result_hash = spec["result"]["sha256"]
    archives = []
    for archive_spec in spec["archives"]:
        payload = _load_bound_json(
            root, {**archive_spec, "git_revision": common_revision}
        )
        if payload.get("result_sha256") != result_hash:
            raise RuntimeError(
                f"second-generation archive does not bind result: {archive_spec['path']}"
            )
        archives.append((archive_spec["path"], payload))
    domains_by_cell = {
        f"{archive['oracle_domain']['target']}_{archive['oracle_domain']['source_idx']}": archive[
            "oracle_domain"
        ]
        for _, archive in archives
    }
    selection_spec = spec.get("selection_lock")
    if selection_spec is None:
        raise RuntimeError(
            "second-generation exact-state admission requires a sealed selection lock"
        )
    selection_document = _load_bound_json(root, selection_spec)
    selection = selection_document["payload"]
    selection_revision = str(selection_spec["git_revision"])
    sealed_inputs: dict[str, str] = selection["inputs_sha256"]
    for path, expected_sha256 in sorted(sealed_inputs.items()):
        observed_sha256 = sha256_bytes(_git_bytes(root, selection_revision, path))
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"second-generation selection input mismatch for {path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
    batch_cache: dict[str, dict] = {}
    for membership in selection["memberships"]:
        path = membership["batch_path"]
        if path not in batch_cache:
            batch_cache[path] = json.loads(_git_bytes(root, selection_revision, path))
    memberships_by_query: dict[str, list[dict]] = defaultdict(list)
    for membership in selection["memberships"]:
        memberships_by_query[membership["query_id"]].append(membership)
    exclusions: list[dict] = []
    admitted: list[MeasuredEndpoint] = []
    for measured in sorted(result_rows, key=lambda row: row["candidate_id"]):
        memberships = memberships_by_query.get(measured["candidate_id"], [])
        if not memberships:
            exclusions.append(
                {
                    "artifact": "second_generation",
                    "identity": measured["candidate_id"],
                    "identity_level": "physical_measurement",
                    "reason": "missing_exact_selection_join",
                }
            )
            continue
        try:
            signatures = set()
            lineage_ids = []
            paths = []
            exact_candidates = []
            for membership in memberships:
                if membership["cell"] != measured["cell"]:
                    raise ValueError("selection membership/result cell mismatch")
                batch = batch_cache[membership["batch_path"]]
                candidates = [
                    candidate
                    for candidate in batch["candidates"]
                    if candidate["candidate_id"] == membership["candidate_id"]
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "selection membership does not resolve one candidate"
                    )
                entry = candidates[0]
                if entry["oracle_protocol"] != measured["oracle_protocol"]:
                    raise ValueError(
                        "selection candidate/result oracle protocol mismatch"
                    )
                if not entry["trace"]["complete"]:
                    raise ValueError("selection candidate trace is incomplete")
                endpoint_state = entry["trace"]["states"][-1]
                if _canonical_smiles(endpoint_state) != measured["smiles"]:
                    raise ValueError(
                        "measured SMILES does not equal exact selected endpoint"
                    )
                signatures.add(
                    (
                        _state_hash(entry["source_state"]),
                        _state_hash(endpoint_state),
                        measured["target"],
                        int(_cell_parts(measured["cell"])[1]),
                        float(domains_by_cell[measured["cell"]]["delta"]),
                    )
                )
                lineage_ids.append(f"program:{entry['trace']['program_id']}")
                paths.append(membership["batch_path"])
                exact_candidates.append(entry)
            if len(signatures) != 1:
                raise ValueError("duplicate arm joins disagree on exact graph identity")
            signature = next(iter(signatures))
            entry = exact_candidates[0]
            admitted.append(
                _make_row(
                    row_id=f"second_generation:{measured['candidate_id']}",
                    artifact="second_generation",
                    target=measured["target"],
                    cell=measured["cell"],
                    delta=float(signature[4]),
                    oracle_protocol=measured["oracle_protocol"],
                    docking_seed=int(measured["docking_seed"]),
                    docking_score=float(measured["ds"]),
                    source_state=entry["source_state"],
                    endpoint_state=entry["trace"]["states"][-1],
                    lineage_ids=lineage_ids,
                    provenance={
                        "candidate_id": measured["candidate_id"],
                        "batch_paths": sorted(set(paths)),
                        "selection_memberships": len(memberships),
                        "pose_sha256": measured["pose_sha256"],
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            exclusions.append(
                {
                    "artifact": "second_generation",
                    "identity": measured["candidate_id"],
                    "identity_level": "physical_measurement",
                    "reason": "invalid_exact_selection_join",
                    "detail": str(error),
                }
            )
    return admitted, exclusions


def _load_program_pool(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[MeasuredEndpoint], list[dict]]:
    spec = config["measured_inputs"]["program_pool"]
    revision = spec["git_revision"]
    remote = _load_bound_json(
        root, {**spec["charged_result"], "git_revision": revision}
    )
    replay = _load_bound_json(
        root, {**spec["exact_replay_join"], "git_revision": revision}
    )
    review = _load_bound_json(root, {**spec["review"], "git_revision": revision})
    source_registry_spec = config["measured_inputs"]["source_registry"]
    source_registry = _load_bound_json(root, source_registry_spec)
    parp1_sources = sorted(
        (row for row in source_registry if row["target"] == "parp1"),
        key=lambda row: row["idx"],
    )
    source_molecule = Chem.MolFromSmiles(parp1_sources[0]["smiles"])
    if source_molecule is None:
        raise RuntimeError("authoritative PARP1-0 source SMILES is invalid")
    authoritative_source_smiles = Chem.MolToSmiles(source_molecule, canonical=True)
    docked_controls = _load_bound_json(
        root, {**spec["docked_controls"], "git_revision": revision}
    )
    winner_protocol_result = _load_bound_json(
        root, {**spec["winner_protocol_result"], "git_revision": revision}
    )
    reused_control = winner_protocol_result["configuration"]["reuse_winner_controls"]
    if (
        reused_control["path"] != spec["docked_controls"]["path"]
        or reused_control["sha256"] != spec["docked_controls"]["sha256"]
    ):
        raise RuntimeError("winner protocol result does not bind docked controls")
    replay_protocol = replay["oracle_protocol"]
    result_protocol = winner_protocol_result["configuration"]
    if (
        replay_protocol["task"]["target"] != "parp1"
        or float(replay_protocol["task"]["delta"]) != 0.4
        or replay_protocol["docking"] != result_protocol["docking"]
        or replay_protocol["task"]["expected_input_sha256"]
        != result_protocol["expected_input_sha256"]
    ):
        raise RuntimeError("winner control protocol differs from replay protocol")
    remote_rows = {row["smiles"]: row for row in remote["payload"]["rows"]}
    winner = review["payload"]["winner"]
    by_endpoint: dict[str, list[dict]] = defaultdict(list)
    for record in replay["records"]:
        by_endpoint[record["endpoint"]].append(record)
    admitted, exclusions = [], []
    protocol = stable_identity(replay["oracle_protocol"])
    for endpoint in sorted(by_endpoint):
        all_records = by_endpoint[endpoint]
        records = [
            record
            for record in all_records
            if _canonical_smiles(record["source_state"]) == authoritative_source_smiles
        ]
        for record in all_records:
            if record not in records:
                exclusions.append(
                    {
                        "artifact": "program_pool",
                        "identity": record["example_id"],
                        "identity_level": "program_representation",
                        "reason": "program_representation_source_mismatch",
                        "endpoint": endpoint,
                    }
                )
        if not records:
            exclusions.append(
                {
                    "artifact": "program_pool",
                    "identity": endpoint,
                    "identity_level": "physical_endpoint_label",
                    "reason": "endpoint_lacks_authoritative_cell_source_representation",
                }
            )
            continue
        try:
            signatures = {
                (
                    _state_hash(record["source_state"]),
                    _state_hash(record["executed_trace"]["states"][-1]),
                    float(record["oracle_label"]["docking_score"]),
                    int(record["oracle_label"]["docking_seed"]),
                )
                for record in records
            }
            if len(signatures) != 1:
                raise ValueError(
                    "program representations disagree on exact state or label"
                )
            sample = records[0]
            score = float(sample["oracle_label"]["docking_score"])
            reused = bool(sample["oracle_label"]["reused_winner_control"])
            if reused:
                controls = [
                    row
                    for row in docked_controls["payload"]
                    if row["role"] == "winner"
                    and row["smiles"] == endpoint
                    and float(row["ds"]) == score
                    and int(row["docking_seed"])
                    == int(sample["oracle_label"]["docking_seed"])
                    and _state_hash(row["state"])
                    == _state_hash(sample["executed_trace"]["states"][-1])
                ]
                if (
                    winner["smiles"] != endpoint
                    or score not in winner["scores"]
                    or len(controls) != 1
                ):
                    raise ValueError(
                        "winner control lacks one exact immutable measured receipt"
                    )
            else:
                charged = remote_rows.get(endpoint)
                if charged is None or float(charged["ds"]) != score:
                    raise ValueError(
                        "program endpoint lacks matching charged result row"
                    )
                if _state_hash(charged["state"]) != _state_hash(
                    sample["executed_trace"]["states"][-1]
                ):
                    raise ValueError("charged endpoint state differs from exact replay")
            lineage_ids = [
                f"program:{record['executed_trace']['program_id']}"
                for record in records
            ]
            admitted.append(
                _make_row(
                    row_id=f"program_pool:{stable_identity([endpoint, protocol])}",
                    artifact="program_pool",
                    target="parp1",
                    cell="parp1_0",
                    delta=0.4,
                    oracle_protocol=protocol,
                    docking_seed=int(sample["oracle_label"]["docking_seed"]),
                    docking_score=score,
                    source_state=sample["source_state"],
                    endpoint_state=sample["executed_trace"]["states"][-1],
                    lineage_ids=lineage_ids,
                    provenance={
                        "representations": len(records),
                        "reused_measured_winner_control": reused,
                        "example_ids": sorted(
                            record["example_id"] for record in records
                        ),
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            exclusions.append(
                {
                    "artifact": "program_pool",
                    "identity": endpoint,
                    "identity_level": "physical_endpoint_label",
                    "reason": "invalid_charged_replay_join",
                    "detail": str(error),
                }
            )
    return admitted, exclusions


def _load_delta06(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[MeasuredEndpoint], list[dict]]:
    spec = config["measured_inputs"]["delta06_prospective"]
    result = _load_bound_json(root, spec["result"])
    request_lock = _load_bound_json(root, spec["request_lock"])["payload"]
    requests = {row["request_id"]: row for row in request_lock["requests"]}
    fold_locks = {}
    for fold_spec in spec["fold_candidate_locks"]:
        fold = int(fold_spec["fold"])
        fold_locks[fold] = _load_bound_json(root, fold_spec, compressed=True)["payload"]
    attempts: dict[tuple[int, str, str], tuple[dict, dict]] = {}
    for fold, payload in fold_locks.items():
        for case in payload["source_cases"]:
            for policy in case["policies"]:
                for attempt in policy["attempts"]:
                    attempts[(fold, case["source_case_id"], attempt["attempt_id"])] = (
                        case,
                        attempt,
                    )
    admitted, exclusions = [], []
    for scored in sorted(
        result["payload"]["requests"], key=lambda row: row["request_id"]
    ):
        if scored["status"] != "complete" or scored.get("score") is None:
            exclusions.append(
                {
                    "artifact": "delta06_prospective",
                    "identity": scored["request_id"],
                    "identity_level": "physical_request",
                    "reason": "failed_or_null_score",
                }
            )
            continue
        try:
            locked = requests[scored["request_id"]]
            if locked["canonical_smiles"] != scored["canonical_smiles"]:
                raise ValueError("scored/request-lock SMILES mismatch")
            joined = []
            for membership in locked["memberships"]:
                key = (
                    int(membership["fold"]),
                    membership["source_case_id"],
                    membership["attempt_id"],
                )
                case, attempt = attempts[key]
                if attempt["status"] != "complete" or attempt["endpoint_state"] is None:
                    raise ValueError("selected delta06 attempt is not complete")
                if (
                    _canonical_smiles(attempt["endpoint_state"])
                    != locked["canonical_smiles"]
                ):
                    raise ValueError(
                        "delta06 exact endpoint differs from locked SMILES"
                    )
                joined.append((membership, case, attempt))
            signatures = {
                (
                    membership["cell"],
                    _state_hash(case["source_state"]),
                    _state_hash(attempt["endpoint_state"]),
                )
                for membership, case, attempt in joined
            }
            if len(signatures) != 1:
                raise ValueError(
                    "delta06 policy memberships disagree on source or endpoint"
                )
            membership, case, attempt = joined[0]
            lineage_ids = []
            for _, _, joined_attempt in joined:
                template_ids = joined_attempt["delta_template_ids"]
                lineage_ids.extend(f"macro:{value}" for value in template_ids)
                if template_ids:
                    lineage_ids.append("macro_program:" + ":".join(template_ids))
            if not lineage_ids:
                lineage_ids = [f"attempt:{membership['attempt_id']}"]
            admitted.append(
                _make_row(
                    row_id=f"delta06:{scored['request_id']}",
                    artifact="delta06_prospective",
                    target=scored["target"],
                    cell=membership["cell"],
                    delta=0.6,
                    oracle_protocol=scored["evaluator"],
                    docking_seed=int(scored["docking_seed"]),
                    docking_score=float(scored["score"]),
                    source_state=case["source_state"],
                    endpoint_state=attempt["endpoint_state"],
                    lineage_ids=lineage_ids,
                    provenance={
                        "request_id": scored["request_id"],
                        "membership_ids": sorted(scored["membership_ids"]),
                        "attempt_ids": sorted(row[0]["attempt_id"] for row in joined),
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            exclusions.append(
                {
                    "artifact": "delta06_prospective",
                    "identity": scored["request_id"],
                    "identity_level": "physical_request",
                    "reason": "invalid_exact_candidate_join",
                    "detail": str(error),
                }
            )
    return admitted, exclusions


def _load_delta04(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[MeasuredEndpoint], list[dict]]:
    spec = config["measured_inputs"]["delta04_prospective"]
    result = _load_bound_json(root, spec["result"])
    lock = _load_bound_json(root, spec["candidate_lock"])["payload"]
    by_request: dict[str, list[dict]] = defaultdict(list)
    for membership in lock["memberships"]:
        by_request[membership["request_id"]].append(membership)
    admitted, exclusions = [], []
    for scored in sorted(
        result["payload"]["requests"], key=lambda row: row["request_id"]
    ):
        if scored["status"] != "complete" or scored.get("score") is None:
            exclusions.append(
                {
                    "artifact": "delta04_prospective",
                    "identity": scored["request_id"],
                    "identity_level": "physical_request",
                    "reason": "failed_or_null_score",
                    "error_type": scored.get("error_type"),
                }
            )
            continue
        try:
            memberships = by_request[scored["request_id"]]
            if not memberships:
                raise ValueError("scored request absent from exact candidate lock")
            signatures = {
                (
                    row["cell"],
                    _state_hash(row["source_state"]),
                    _state_hash(row["endpoint_state"]),
                    row["canonical_smiles"],
                )
                for row in memberships
            }
            if len(signatures) != 1:
                raise ValueError(
                    "delta04 arm memberships disagree on source or endpoint"
                )
            sample = memberships[0]
            if (
                _canonical_smiles(sample["endpoint_state"])
                != sample["canonical_smiles"]
            ):
                raise ValueError("delta04 exact endpoint differs from locked SMILES")
            lineage_ids = []
            for row in memberships:
                lineage_ids.extend(f"macro:{value}" for value in row["patch_ids"])
                lineage_ids.append("macro_program:" + ":".join(row["patch_ids"]))
            admitted.append(
                _make_row(
                    row_id=f"delta04:{scored['request_id']}",
                    artifact="delta04_prospective",
                    target=scored["target"],
                    cell=sample["cell"],
                    delta=0.4,
                    oracle_protocol=sample["evaluator"],
                    docking_seed=int(sample["docking_seed"]),
                    docking_score=float(scored["score"]),
                    source_state=sample["source_state"],
                    endpoint_state=sample["endpoint_state"],
                    lineage_ids=lineage_ids,
                    provenance={
                        "request_id": scored["request_id"],
                        "membership_ids": sorted(
                            row["membership_id"] for row in memberships
                        ),
                        "candidate_ids": sorted(
                            row["candidate_id"] for row in memberships
                        ),
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            exclusions.append(
                {
                    "artifact": "delta04_prospective",
                    "identity": scored["request_id"],
                    "identity_level": "physical_request",
                    "reason": "invalid_exact_candidate_join",
                    "detail": str(error),
                }
            )
    return admitted, exclusions


def _load_utility_acquisition(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[MeasuredEndpoint], list[dict]]:
    """Load only the four sealed one-shot utility-acquisition outcomes."""

    spec = config["measured_inputs"]["utility_acquisition"]
    revision = spec["git_revision"]
    candidate_document = _load_bound_json(
        root, {**spec["candidate_lock"], "git_revision": revision}
    )
    request_document = _load_bound_json(
        root, {**spec["request_lock"], "git_revision": revision}
    )
    result_document = _load_bound_json(
        root, {**spec["result"], "git_revision": revision}
    )
    review_document = _load_bound_json(
        root, {**spec["review"], "git_revision": revision}
    )
    candidates = {
        row["candidate_id"]: row for row in candidate_document["payload"]["candidates"]
    }
    requests = {
        row["request_id"]: row for row in request_document["payload"]["requests"]
    }
    result = result_document["payload"]
    review = review_document["payload"]
    expected = int(spec["expected_request_count"])
    if (
        len(candidates) != expected
        or len(requests) != expected
        or len(result["requests"]) != expected
        or result["request_count"] != expected
        or result["successful_scores"] != expected
        or result["failed_scores"] != 0
        or result["automatic_retries"] != 0
        or result["replacement_or_backfill"] is not False
    ):
        raise RuntimeError("utility-acquisition four-call census is not exact")
    if (
        result["request_lock_physical_sha256"] != spec["request_lock"]["sha256"]
        or result["request_lock_payload_sha256"]
        != spec["request_lock"]["payload_sha256"]
        or request_document["payload"]["candidate_lock_payload_sha256"]
        != spec["candidate_lock"]["payload_sha256"]
    ):
        raise RuntimeError("utility-acquisition result does not bind the exact locks")
    review_inputs = review["inputs"]
    for key, input_key in (
        ("candidate_lock", "candidate_lock"),
        ("request_lock", "request_lock"),
        ("scored_result", "result"),
    ):
        bound = review_inputs[key]
        expected_spec = spec[input_key]
        if (
            bound["physical_sha256"] != expected_spec["sha256"]
            or bound["payload_sha256"] != expected_spec["payload_sha256"]
        ):
            raise RuntimeError(f"utility-acquisition review mismatch: {key}")
    if (
        review["execution"]["first_score_calls_charged"] != expected
        or review["execution"]["successful_scores"] != expected
        or review["execution"]["failed_scores"] != 0
        or review["execution"]["automatic_retries"] != 0
        or review["execution"]["replacements_or_backfill"] != 0
        or review["summary"]["conditional_data_gate_ready"] is not True
        or any(
            value["finite_non_tied_pair"] is not True
            for value in review["cell_summary"].values()
        )
    ):
        raise RuntimeError("utility-acquisition review did not pass its frozen gate")

    admitted, exclusions = [], []
    observed_cells = set()
    for scored in sorted(result["requests"], key=lambda row: row["request_id"]):
        identity = scored["request_id"]
        try:
            if scored["status"] != "complete" or not math.isfinite(
                float(scored["score"])
            ):
                raise ValueError("request is not a finite completed score")
            locked = requests[identity]
            candidate = candidates[locked["candidate_id"]]
            if (
                scored["candidate_id"] != locked["candidate_id"]
                or scored["candidate_id"] != candidate["candidate_id"]
                or scored["target"] != locked["target"]
                or scored["target"] != candidate["target"]
                or scored["cell"] != locked["cell"]
                or scored["cell"] != candidate["cell"]
                or scored["canonical_smiles"] != locked["canonical_smiles"]
                or scored["canonical_smiles"] != candidate["canonical_smiles"]
                or scored["evaluator"] != locked["evaluator"]
                or int(scored["docking_seed"]) != int(locked["docking_seed"])
                or _canonical_smiles(candidate["endpoint_state"])
                != candidate["canonical_smiles"]
                or _canonical_smiles(candidate["source_state"])
                != candidate["source_smiles"]
            ):
                raise ValueError("result/request/candidate identity mismatch")
            observed_cells.add(scored["cell"])
            admitted.append(
                _make_row(
                    row_id=f"utility_acquisition:{identity}",
                    artifact="utility_acquisition",
                    target=scored["target"],
                    cell=scored["cell"],
                    delta=float(spec["delta"]),
                    oracle_protocol=scored["evaluator"],
                    docking_seed=int(scored["docking_seed"]),
                    docking_score=float(scored["score"]),
                    source_state=candidate["source_state"],
                    endpoint_state=candidate["endpoint_state"],
                    lineage_ids=list(candidate["lineage_ids"]),
                    provenance={
                        "request_id": identity,
                        "candidate_id": candidate["candidate_id"],
                        "training_data_acquisition_only": True,
                        "source_artifact": candidate["source_artifact"],
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            exclusions.append(
                {
                    "artifact": "utility_acquisition",
                    "identity": identity,
                    "identity_level": "physical_request",
                    "reason": "invalid_exact_candidate_join",
                    "detail": str(error),
                }
            )
    if observed_cells != set(spec["expected_cells"]):
        raise RuntimeError(
            "utility-acquisition cells differ from frozen expectation: "
            f"{sorted(observed_cells)}"
        )
    return admitted, exclusions


def _leakage_filter(
    rows: list[MeasuredEndpoint], exclusions: list[dict]
) -> tuple[list[MeasuredEndpoint], list[dict], dict[str, Any]]:
    def groups_for(row: MeasuredEndpoint) -> set[str]:
        return {
            f"request:{row.row_id}",
            f"endpoint:{row.endpoint_smiles}",
            f"scaffold:{row.scaffold}",
            f"source:{row.cell}:{row.source_state_sha256}",
            *(f"lineage:{value}" for value in row.lineage_ids),
        }

    grouping: dict[str, list[str]] = defaultdict(list)
    row_groups: dict[str, set[str]] = {}
    for row in rows:
        groups = groups_for(row)
        row_groups[row.row_id] = groups
        for group in groups:
            grouping[group].append(row.row_id)

    parent = {row.row_id: row.row_id for row in rows}

    def find(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for identities in grouping.values():
        for identity in identities[1:]:
            union(identities[0], identity)
    components: dict[str, list[MeasuredEndpoint]] = defaultdict(list)
    for row in rows:
        components[find(row.row_id)].append(row)
    conflicting_components = {
        component
        for component, members in components.items()
        if len({row.source_idx for row in members}) > 1
    }
    directly_conflicting_groups = {
        group
        for group, identities in grouping.items()
        if len(
            {
                next(row.source_idx for row in rows if row.row_id == identity)
                for identity in identities
            }
        )
        > 1
    }
    kept = []
    for row in rows:
        component = find(row.row_id)
        if component in conflicting_components:
            exclusions.append(
                {
                    "artifact": row.artifact,
                    "identity": row.row_id,
                    "identity_level": "admitted_label",
                    "reason": "cross_fold_group_conflict",
                    "component_id": stable_identity(
                        sorted(member.row_id for member in components[component])
                    ),
                    "component_rows": len(components[component]),
                    "directly_conflicting_groups": sorted(
                        set().union(
                            *(
                                row_groups[member.row_id]
                                for member in components[component]
                            )
                        )
                        & directly_conflicting_groups
                    ),
                }
            )
        else:
            kept.append(row)
    post_grouping: dict[str, set[int]] = defaultdict(set)
    for row in kept:
        for group in groups_for(row):
            post_grouping[group].add(row.source_idx)
    post_conflicts = {group for group, folds in post_grouping.items() if len(folds) > 1}
    return (
        kept,
        exclusions,
        {
            "group_count": len(grouping),
            "pre_exclusion_conflicting_group_count": len(directly_conflicting_groups),
            "pre_exclusion_conflicting_groups": sorted(directly_conflicting_groups),
            "pre_exclusion_connected_components": len(components),
            "conflicting_connected_components": len(conflicting_components),
            "excluded_rows": len(rows) - len(kept),
            "post_exclusion_group_count": len(post_grouping),
            "post_exclusion_conflicting_group_count": len(post_conflicts),
            "post_exclusion_conflicting_groups": sorted(post_conflicts),
        },
    )


def _census(rows: list[MeasuredEndpoint]) -> dict[str, Any]:
    by_artifact = Counter(row.artifact for row in rows)
    by_target = Counter(row.target for row in rows)
    by_cell = Counter(row.cell for row in rows)
    by_fold = Counter(row.source_idx for row in rows)
    strata = defaultdict(list)
    for row in rows:
        strata[row.stratum].append(row)
    return {
        "rows": len(rows),
        "artifacts": dict(sorted(by_artifact.items())),
        "targets": dict(sorted(by_target.items())),
        "cells": dict(sorted(by_cell.items())),
        "folds": {str(key): value for key, value in sorted(by_fold.items())},
        "unique_endpoints": len({(row.target, row.endpoint_smiles) for row in rows}),
        "strata": len(strata),
        "comparable_strata": sum(
            len(group) >= 2 and bool(strict_pairs(group)) for group in strata.values()
        ),
        "strict_pairs": len(strict_pairs(rows)),
    }


def _input_reconciliation(
    root: Path,
    config: Mapping[str, Any],
    admitted_before_leakage: list[MeasuredEndpoint],
    admitted_after_leakage: list[MeasuredEndpoint],
    exclusions: list[dict],
) -> dict[str, Any]:
    measured = config["measured_inputs"]
    second = _load_bound_json(
        root,
        {
            **measured["second_generation"]["result"],
            "git_revision": measured["second_generation"]["git_revision"],
        },
    )["payload"]["rows"]
    replay = _load_bound_json(
        root,
        {
            **measured["program_pool"]["exact_replay_join"],
            "git_revision": measured["program_pool"]["git_revision"],
        },
    )["records"]
    program_endpoints = {row["endpoint"] for row in replay}
    delta06 = _load_bound_json(root, measured["delta06_prospective"]["result"])[
        "payload"
    ]["requests"]
    delta04 = _load_bound_json(root, measured["delta04_prospective"]["result"])[
        "payload"
    ]["requests"]
    observed = {
        "second_generation": (len(second), len(second), "physical_request"),
        "program_pool": (
            len(program_endpoints),
            len(program_endpoints),
            "physical_endpoint",
        ),
        "delta06_prospective": (
            len(delta06),
            sum(
                row["status"] == "complete"
                and row.get("score") is not None
                and math.isfinite(float(row["score"]))
                for row in delta06
            ),
            "physical_request",
        ),
        "delta04_prospective": (
            len(delta04),
            sum(
                row["status"] == "complete"
                and row.get("score") is not None
                and math.isfinite(float(row["score"]))
                for row in delta04
            ),
            "physical_request",
        ),
    }
    acquisition_spec = measured.get("utility_acquisition")
    if acquisition_spec is not None:
        acquisition = _load_bound_json(
            root,
            {
                **acquisition_spec["result"],
                "git_revision": acquisition_spec["git_revision"],
            },
        )["payload"]["requests"]
        observed["utility_acquisition"] = (
            len(acquisition),
            sum(
                row["status"] == "complete"
                and row.get("score") is not None
                and math.isfinite(float(row["score"]))
                for row in acquisition
            ),
            "physical_request",
        )
    pre_counts = Counter(row.artifact for row in admitted_before_leakage)
    final_counts = Counter(row.artifact for row in admitted_after_leakage)
    physical_exclusions = Counter()
    representation_exclusions = Counter()
    group_exclusions = Counter()
    failed_or_null = Counter()
    for row in exclusions:
        artifact = row["artifact"]
        if row["reason"] == "cross_fold_group_conflict":
            group_exclusions[artifact] += 1
        elif row["reason"] == "failed_or_null_score":
            failed_or_null[artifact] += 1
        elif row.get("identity_level") == "program_representation":
            representation_exclusions[artifact] += 1
        elif row.get("identity_level") in {
            "physical_measurement",
            "physical_endpoint_label",
            "physical_request",
        }:
            physical_exclusions[artifact] += 1
    artifact_rows = {}
    for artifact, (observed_count, finite_count, identity_level) in observed.items():
        artifact_rows[artifact] = {
            "physical_identity_level": identity_level,
            "observed_physical_units": observed_count,
            "finite_measured_labels": finite_count,
            "failed_or_null_physical_units": failed_or_null[artifact],
            "invalid_or_out_of_cell_physical_labels": physical_exclusions[artifact],
            "admitted_labels_before_group_filter": pre_counts[artifact],
            "cross_fold_group_removals": group_exclusions[artifact],
            "final_labels": final_counts[artifact],
        }
    return {
        "schema_version": "t4_target_conditioned_utility_reconciliation_v1",
        "artifacts": artifact_rows,
        "totals": {
            "observed_physical_units": sum(value[0] for value in observed.values()),
            "finite_measured_labels": sum(value[1] for value in observed.values()),
            "failed_or_null_physical_units": sum(failed_or_null.values()),
            "invalid_or_out_of_cell_physical_labels": sum(physical_exclusions.values()),
            "admitted_labels_before_group_filter": len(admitted_before_leakage),
            "cross_fold_group_removals": sum(group_exclusions.values()),
            "final_labels": len(admitted_after_leakage),
        },
        "representation_diagnostics": {
            "program_pool_observed_representations": len(replay),
            "program_pool_source_mismatch_representations": representation_exclusions[
                "program_pool"
            ],
            "note": "Representation exclusions are provenance diagnostics and are not physical label denominators.",
        },
    }


def _support_plan(rows: list[MeasuredEndpoint]) -> dict[str, Any]:
    folds = []
    supported_targets = set()
    supported_strata = 0
    for fold in range(3):
        train, test = rows_by_fold(rows, fold)
        train_counts = target_pair_counts(train)
        by_stratum = defaultdict(list)
        for row in test:
            by_stratum[row.stratum].append(row)
        supported, abstentions = [], []
        for stratum in sorted(by_stratum):
            group = by_stratum[stratum]
            reason = None
            if len(group) < 2 or not strict_pairs(group):
                reason = "held_stratum_lacks_strict_pair"
            elif train_counts[stratum[0]] < 1:
                reason = "target_absent_from_training_pairs"
            if reason:
                abstentions.append(
                    {
                        "target": stratum[0],
                        "cell": stratum[1],
                        "reason": reason,
                        "rows": len(group),
                    }
                )
            else:
                supported.append(
                    {
                        "target": stratum[0],
                        "cell": stratum[1],
                        "stratum": list(stratum),
                        "rows": len(group),
                    }
                )
                supported_targets.add(stratum[0])
                supported_strata += 1
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "training_pair_counts_by_target": train_counts,
                "supported_test_strata": supported,
                "abstentions": abstentions,
            }
        )
    return {
        "folds": folds,
        "supported_targets": sorted(supported_targets),
        "supported_target_count": len(supported_targets),
        "supported_strata": supported_strata,
    }


def _aggregate_fold_metrics(folds: list[dict], arm: str) -> dict[str, Any]:
    results = [fold["arms"][arm] for fold in folds if arm in fold["arms"]]
    if not results:
        return {"status": "abstain_no_supported_fold"}
    per_stratum = [row for result in results for row in result["per_stratum"]]
    pairs = sum(result["strict_pairs"] for result in results)
    predicted = sum(
        round(result["pair_prediction_coverage"] * result["strict_pairs"])
        for result in results
    )
    correct = sum(
        (
            round(
                result["pairwise_precision"]
                * result["pair_prediction_coverage"]
                * result["strict_pairs"]
            )
            if result["pairwise_precision"] is not None
            else 0
        )
        for result in results
    )
    means = {}
    for key in (
        "ndcg",
        "top1_regret",
        "top3_regret",
        "best_recall_at_1",
        "best_recall_at_3",
        "best_selection_precision_at_1",
        "best_selection_precision_at_3",
    ):
        means[f"source_balanced_{key}"] = float(
            np.mean([row[key] for row in per_stratum])
        )
    return {
        "status": "measured",
        "folds": len(results),
        "candidate_rows": sum(result["candidate_rows"] for result in results),
        "strata": len(per_stratum),
        "strict_pairs": pairs,
        "pair_prediction_coverage": predicted / pairs,
        "pairwise_precision": correct / predicted if predicted else None,
        **means,
        "per_stratum": per_stratum,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(root: Path, output: Path, contract_path: Path = CONTRACT) -> dict[str, Any]:
    contract_document = json.loads((root / contract_path).read_text())
    contract, contract_resolution = _resolve_contract(root, contract_document)
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    producer = {
        "code_revision": code_revision,
        "implementation_sha256": {
            "src/compose_v4/control/target_conditioned_utility_selector.py": sha256_file(
                root / "src/compose_v4/control/target_conditioned_utility_selector.py"
            ),
            "tools/t4_target_conditioned_utility_selector.py": sha256_file(
                root / "tools/t4_target_conditioned_utility_selector.py"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "device": "cpu",
        "precision_if_fit": "float32",
    }
    rows, exclusions = [], []
    for loader in (
        _load_second_generation,
        _load_program_pool,
        _load_delta06,
        _load_delta04,
    ):
        accepted, rejected = loader(root, contract)
        rows.extend(accepted)
        exclusions.extend(rejected)
    if "utility_acquisition" in contract["measured_inputs"]:
        accepted, rejected = _load_utility_acquisition(root, contract)
        rows.extend(accepted)
        exclusions.extend(rejected)
    if len({row.row_id for row in rows}) != len(rows):
        raise RuntimeError("physical measurement identities are not unique")
    duplicate_endpoint = Counter((row.target, row.endpoint_smiles) for row in rows)
    duplicates = [key for key, count in duplicate_endpoint.items() if count > 1]
    if duplicates:
        raise RuntimeError(
            f"cross-artifact target/endpoint duplicates remain: {duplicates}"
        )
    pre_leakage = _census(rows)
    admitted_before_leakage = list(rows)
    rows, exclusions, leakage = _leakage_filter(rows, exclusions)
    post_leakage = _census(rows)
    reconciliation = _input_reconciliation(
        root, contract, admitted_before_leakage, rows, exclusions
    )
    support = _support_plan(rows)
    gate_checks = {
        "all_labels_have_exact_receipts": not any(
            row["reason"].startswith("invalid_") or row["reason"].startswith("missing_")
            for row in exclusions
        ),
        "zero_cross_fold_group_leakage": (
            leakage["post_exclusion_conflicting_group_count"] == 0
        ),
        "at_least_three_supported_targets": support["supported_target_count"] >= 3,
        "at_least_five_supported_strata": support["supported_strata"] >= 5,
        "every_evaluated_target_has_training_pair": all(
            fold["training_pair_counts_by_target"][row["target"]] >= 1
            for fold in support["folds"]
            for row in fold["supported_test_strata"]
        ),
    }
    gate_passed = all(gate_checks.values())
    audit = {
        "schema_version": "t4_target_conditioned_utility_data_audit_v1",
        "contract_sha256": contract_document["contract_sha256"],
        "input_contract_path": str(contract_path),
        "input_contract_physical_sha256": sha256_file(root / contract_path),
        "producer": producer,
        "contract_resolution": contract_resolution,
        "pre_leakage_census": pre_leakage,
        "post_leakage_census": post_leakage,
        "input_reconciliation": reconciliation,
        "leakage": leakage,
        "support": support,
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
        "held_target_transfer": "unsupported",
        "calibration": "unsupported_due_to_sparse_grouped_data",
    }
    split_manifest = {
        "schema_version": "t4_target_conditioned_utility_split_manifest_v1",
        "contract_sha256": contract_document["contract_sha256"],
        "rows": [
            {
                "row_id": row.row_id,
                "fold": row.source_idx,
                "target": row.target,
                "cell": row.cell,
                "stratum": list(row.stratum),
                "source_state_sha256": row.source_state_sha256,
                "endpoint_state_sha256": row.endpoint_state_sha256,
                "scaffold": row.scaffold,
                "lineage_ids": list(row.lineage_ids),
            }
            for row in sorted(rows, key=lambda item: item.row_id)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="t4-utility-selector-", dir=output.parent))
    try:
        _write_json(temp / "data_audit.json", audit)
        _write_json(
            temp / "exclusion_ledger.json",
            {
                "schema_version": "t4_target_conditioned_utility_exclusions_v1",
                "contract_sha256": contract_document["contract_sha256"],
                "count": len(exclusions),
                "reasons": dict(
                    sorted(Counter(row["reason"] for row in exclusions).items())
                ),
                "rows": sorted(
                    exclusions, key=lambda row: (row["artifact"], row["identity"])
                ),
            },
        )
        _write_json(temp / "split_manifest.json", split_manifest)
        if not gate_passed:
            result = {
                "schema_version": "t4_target_conditioned_utility_selector_result_v1",
                "status": "abstain_data_gate_failed",
                "contract_sha256": contract_document["contract_sha256"],
                "data_audit_sha256": sha256_file(temp / "data_audit.json"),
                "exclusion_ledger_sha256": sha256_file(temp / "exclusion_ledger.json"),
                "split_manifest_sha256": sha256_file(temp / "split_manifest.json"),
                "new_oracle_calls": 0,
                "new_docking_calls": 0,
                "modal_launches": 0,
                "fit_performed": False,
                "producer": producer,
            }
        else:
            fold_results = []
            model_config = contract["models"]["optimization"]
            for fold in range(3):
                train, test = rows_by_fold(rows, fold)
                blind, blind_fit = fit_utility_ranker(
                    train,
                    conditioned=False,
                    updates=model_config["updates"],
                    learning_rate=model_config["learning_rate"],
                    base_l2=model_config["base_l2"],
                    target_interaction_l2=model_config["target_interaction_l2"],
                    seed=model_config["seed"] + fold,
                )
                conditioned, conditioned_fit = fit_utility_ranker(
                    train,
                    conditioned=True,
                    updates=model_config["updates"],
                    learning_rate=model_config["learning_rate"],
                    base_l2=model_config["base_l2"],
                    target_interaction_l2=model_config["target_interaction_l2"],
                    seed=model_config["seed"] + fold,
                )
                supported = []
                by_stratum = defaultdict(list)
                for row in test:
                    by_stratum[row.stratum].append(row)
                for stratum, group in by_stratum.items():
                    if (
                        len(group) >= 2
                        and strict_pairs(group)
                        and conditioned.supports(stratum[0])
                    ):
                        supported.extend(group)
                supported = sorted(supported, key=lambda row: row.row_id)
                arms = {}
                if supported:
                    arms["fixed_target_blind_structural"] = evaluate_scores(
                        supported,
                        {row.row_id: fixed_structural_score(row) for row in supported},
                    )
                    arms["target_blind_utility"] = evaluate_scores(
                        supported, {row.row_id: blind.score(row) for row in supported}
                    )
                    arms["target_conditioned_utility"] = evaluate_scores(
                        supported,
                        {row.row_id: conditioned.score(row) for row in supported},
                    )
                checkpoint = {
                    "schema_version": "t4_target_conditioned_utility_fold_checkpoint_v1",
                    "contract_sha256": contract_document["contract_sha256"],
                    "fold": fold,
                    "target_blind": blind.checkpoint(),
                    "target_conditioned": conditioned.checkpoint(),
                    "training_row_ids_sha256": stable_identity(
                        sorted(row.row_id for row in train)
                    ),
                    "runtime_training_rows": 0,
                }
                _write_json(temp / f"fold_{fold}_checkpoint.json", checkpoint)
                fold_results.append(
                    {
                        "fold": fold,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "supported_test_rows": len(supported),
                        "supported_targets": sorted({row.target for row in supported}),
                        "target_blind_fit": blind_fit,
                        "target_conditioned_fit": conditioned_fit,
                        "checkpoint_sha256": sha256_file(
                            temp / f"fold_{fold}_checkpoint.json"
                        ),
                        "arms": arms,
                    }
                )
            aggregate = {
                arm: _aggregate_fold_metrics(fold_results, arm)
                for arm in (
                    "fixed_target_blind_structural",
                    "target_blind_utility",
                    "target_conditioned_utility",
                )
            }
            blind_result = aggregate["target_blind_utility"]
            target_result = aggregate["target_conditioned_utility"]
            improved = (
                target_result.get("status") == "measured"
                and blind_result.get("status") == "measured"
                and target_result["source_balanced_top1_regret"]
                < blind_result["source_balanced_top1_regret"]
                and target_result["source_balanced_ndcg"]
                > blind_result["source_balanced_ndcg"]
            )
            result = {
                "schema_version": "t4_target_conditioned_utility_selector_result_v1",
                "status": "complete_retrospective_within_known_target_diagnostic",
                "contract_sha256": contract_document["contract_sha256"],
                "data_audit_sha256": sha256_file(temp / "data_audit.json"),
                "exclusion_ledger_sha256": sha256_file(temp / "exclusion_ledger.json"),
                "split_manifest_sha256": sha256_file(temp / "split_manifest.json"),
                "folds": fold_results,
                "aggregate": aggregate,
                "target_conditioning_improved_primary_metrics": improved,
                "prospective_lock_authorized": False,
                "held_target_transfer": "unsupported",
                "calibration": "unsupported_due_to_sparse_grouped_data",
                "new_oracle_calls": 0,
                "new_docking_calls": 0,
                "modal_launches": 0,
                "fit_performed": True,
                "producer": producer,
            }
        _write_json(temp / "result.json", result)
        result_sha256 = sha256_file(temp / "result.json")
        (temp / "README.md").write_text(
            "# T4 target-conditioned utility selector result\n\n"
            f"Status: `{result['status']}`.\n\n"
            f"Contract payload SHA-256: `{contract_document['contract_sha256']}`.\n\n"
            f"Result physical SHA-256: `{result_sha256}`.\n\n"
            "This is a retrospective zero-oracle within-known-target ranking diagnostic. "
            "It is not held-target transfer, calibrated utility, prospective optimization or IVG evidence.\n"
        )
        if output.exists():
            raise FileExistsError(
                f"refusing to replace existing result directory: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = args.contract
    if not contract_path.is_absolute():
        contract_path = root / contract_path
    contract_document = json.loads(contract_path.read_text())
    output = args.output or Path(contract_document["payload"]["outputs"]["directory"])
    if not output.is_absolute():
        output = root / output
    result = run(root, output, contract_path.relative_to(root))
    print(
        json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
