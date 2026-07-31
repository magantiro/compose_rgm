#!/usr/bin/env python3
"""Build a train-only non-Cartesian Active8 semantic census."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from compose_v4.data.active8_trace_inventory import (
    ACTIVE8_FAMILIES,
    load_active8_trace_admission,
)
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.data.packed_charge_policy_audit import resolve_unified_manifest_shards
from compose_v4.data.packed_trace_store import read_frozen_source_addressed_packed_shard
from compose_v4.experiments.editing_active8_semantic_census_v2 import (
    build_semantic_census,
    derive_semantic_observation,
    file_sha256,
    implementation_identity,
    load_semantic_census_contract,
    semantic_census_bytes,
    validate_parent_t1_successor_gate,
)

DEFAULT_CONTRACT = ROOT / "configs/editing_active8_within_family_semantic_census_v2.json"
PARENT_T1_GATE = ROOT / "configs/editing_t1_successor_gate_v8.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count independent within-family semantic marginals over exact, "
            "whole-trace-admitted Active8 training rows. This command does not train."
        )
    )
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--mmp-root", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _git_revision() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree_changes = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot bind semantic census to a git revision") from exc
    if tree_changes:
        raise RuntimeError(
            "semantic census requires a clean scientific tree, including no "
            "untracked inputs; commit or isolate changes before producing "
            "authoritative numbers"
        )
    return {"commit": commit, "tree_dirty": False}


def _observations(*, shards, admission):
    observed_lanes: list[tuple[str, str, str]] = []
    for shard in shards:
        lane = (shard.envelope_layer, shard.partition, shard.path.name)
        expected_digest = admission.expected_source_digest(
            packed_shard_name=shard.path.name,
            layer=shard.envelope_layer,
            partition=shard.partition,
        )
        metadata = admission.shard_metadata_by_digest[expected_digest]
        observed_entries = 0
        observed_digest = None
        for addressed in read_frozen_source_addressed_packed_shard(
            shard.path,
            expected_shard_sha256=expected_digest,
            expected_manifest_sha256=str(metadata["packed_manifest_sha256"]),
            expected_overlay_sha256=metadata["packed_provenance_overlay_sha256"],
        ):
            observed_entries += 1
            observed_digest = addressed.address.packed_shard_content_sha256
            if not admission.is_accepted(addressed.address):
                continue
            if len(addressed.trace.steps) != addressed.address.path_length:
                raise RuntimeError("packed trace path length disagrees with its exact address")
            for step_index, step in enumerate(addressed.trace.steps):
                yield derive_semantic_observation(
                    addressed.path.state_at(step_index),
                    addressed.path.state_at(step_index + 1),
                    executor_rule=step.rule_name,
                    action=step.action,
                    data_lane=addressed.address.layer,
                    step_index=step_index,
                    path_length=addressed.address.path_length,
                )
        admission.assert_complete_source_shard(
            packed_shard_name=shard.path.name,
            layer=shard.envelope_layer,
            partition=shard.partition,
            observed_digest=observed_digest,
            observed_entries=observed_entries,
        )
        observed_lanes.append(lane)
    admission.assert_partition_shards("train", observed_lanes)


def main() -> None:
    args = _parser().parse_args()
    code_revision = _git_revision()
    contract = load_semantic_census_contract(args.contract)
    validate_parent_t1_successor_gate(contract, path=PARENT_T1_GATE)
    parent_identity = contract["parent_active8_identity"]
    admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=parent_identity["inventory_manifest_file_sha256"],
        expected_inventory_sha256=parent_identity["inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=parent_identity[
            "effective_source_corpus_cache_sha256"
        ],
        expected_support_contract_sha256=parent_identity["support_contract_sha256"],
    )
    if (
        admission.unified_packed_manifest_sha256
        != parent_identity["unified_packed_manifest_sha256"]
    ):
        raise RuntimeError(
            "Active8 admission unified-manifest identity disagrees with the pinned V8 parent"
        )
    unified_sha256 = file_sha256(args.unified_manifest)
    if unified_sha256 != admission.unified_packed_manifest_sha256:
        raise RuntimeError(
            "unified packed manifest bytes disagree with the Active8 admission identity"
        )
    _manifest, declared = resolve_unified_manifest_shards(
        args.unified_manifest,
        audit_root=args.audit_root,
        mmp_root=args.mmp_root,
    )
    shards = tuple(shard for shard in declared if shard.partition == "train")
    if not shards:
        raise RuntimeError("unified manifest has no train shards")
    source_bindings = []
    expected_family_rows = {family: 0 for family in ACTIVE8_FAMILIES}
    for shard in shards:
        digest = admission.expected_source_digest(
            packed_shard_name=shard.path.name,
            layer=shard.envelope_layer,
            partition=shard.partition,
        )
        metadata = admission.shard_metadata_by_digest[digest]
        source_bindings.append(
            {
                "layer": shard.envelope_layer,
                "partition": shard.partition,
                "relative_path": shard.relative_path,
                "packed_shard_name": shard.path.name,
                "packed_shard_content_sha256": digest,
                "packed_manifest_sha256": metadata["packed_manifest_sha256"],
                "packed_provenance_overlay_sha256": metadata["packed_provenance_overlay_sha256"],
            }
        )
        for family, count in metadata["accepted_nonterminal_rows_by_family"].items():
            expected_family_rows[family] += int(count)

    provenance = {
        "active8_inventory_path": str(args.active8_inventory),
        "active8_inventory_file_sha256": admission.manifest_file_sha256,
        "active8_inventory_sha256": admission.inventory_sha256,
        "effective_source_corpus_cache_sha256": (admission.effective_source_corpus_cache_sha256),
        "support_contract_sha256": admission.support_contract_sha256,
        "unified_packed_manifest_path": str(args.unified_manifest),
        "unified_packed_manifest_sha256": unified_sha256,
        "semantic_census_contract_path": str(args.contract),
        "semantic_census_contract_sha256": file_sha256(args.contract),
        "semantic_census_contract_logical_sha256": contract["contract_sha256"],
        "parent_t1_successor_gate_path": str(PARENT_T1_GATE),
        "parent_t1_successor_gate_file_sha256": contract["parent_t1_successor_gate"]["file_sha256"],
        "parent_t1_successor_gate_contract_sha256": contract["parent_t1_successor_gate"][
            "contract_sha256"
        ],
        "source_shards": source_bindings,
        "implementation": implementation_identity(repo_root=ROOT),
        "code_revision": code_revision,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "determinism": {
            "seed": None,
            "reason": "integer graph labels and exact streaming counts use no random sampling",
            "hardware_and_precision": "not_applicable_to_integer_graph_census",
            "timestamp": "omitted_for_byte_stability_and_operationally_irrelevant",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output.parent,
        prefix=f".{args.output.name}.uniqueness.",
    ) as temporary_directory:
        report = build_semantic_census(
            _observations(shards=shards, admission=admission),
            contract=contract,
            provenance=provenance,
            uniqueness_database=Path(temporary_directory) / "exact_states.sqlite",
        )
    if report["counts"]["rows_by_family"] != expected_family_rows:
        raise RuntimeError(
            "semantic census family counts disagree with the Active8 inventory: "
            f"observed={report['counts']['rows_by_family']}, expected={expected_family_rows}"
        )
    report["artifact_sha256_basis"] = (
        "canonical JSON of this object before artifact_sha256 was added"
    )
    report["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    created = write_bytes_if_absent(args.output, semantic_census_bytes(report))
    print(
        json.dumps(
            {
                "artifact_sha256": report["artifact_sha256"],
                "created": created,
                "output": str(args.output),
                "rows": report["counts"]["rows"],
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
