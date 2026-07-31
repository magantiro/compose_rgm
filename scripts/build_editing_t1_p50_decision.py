#!/usr/bin/env python3
"""Build the bounded-P50 decision from complete physical T1 evidence.

This command performs no optimization and does not alter the prospective T1
thresholds. It validates the exact V4 runtime authority, Active8 admission,
Gate0 structural evidence, and complete physical arm-results manifest before
delegating the scientific decision to the production prerequisite builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.active8_trace_inventory import (  # noqa: E402
    Active8TraceInventoryError,
    load_active8_trace_admission,
)
from compose_v4.experiments.editing_p50_prerequisites import (  # noqa: E402
    EditingP50PrerequisiteError,
    build_t1_p50_decision,
    validate_gate_zero_structural_evidence,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    ACTIVE8_T1_IDENTITY_FIELDS,
    active8_t1_identity,
)
from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    EditingT1RuntimeError,
    load_editing_t1_runtime_contract,
)


class EditingT1DecisionCliError(ValueError):
    """The physical T1 decision inputs are absent, stale, or inconsistent."""


def _load_json_object(
    path: Path,
    *,
    name: str,
    max_file_bytes: int = 1 << 24,
) -> tuple[Mapping[str, Any], str]:
    source = Path(path)
    try:
        content = source.read_bytes()
    except OSError as error:
        raise EditingT1DecisionCliError(f"{name} is absent: {source}") from error
    if not source.is_file() or len(content) > max_file_bytes:
        raise EditingT1DecisionCliError(f"{name} is absent or exceeds its size bound: {source}")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1DecisionCliError(f"{name} is not valid JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise EditingT1DecisionCliError(f"{name} must contain one JSON object")
    return payload, hashlib.sha256(content).hexdigest()


def _atomic_immutable_json_write(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    try:
        content = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingT1DecisionCliError("T1 decision is not finite serializable JSON") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise EditingT1DecisionCliError(
                    f"cannot inspect existing T1 decision artifact: {path}"
                ) from error
            if existing != content:
                raise EditingT1DecisionCliError(
                    f"immutable T1 decision artifact already differs: {path}"
                )
        except OSError as error:
            raise EditingT1DecisionCliError(
                f"cannot atomically publish T1 decision artifact: {path}"
            ) from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def build_editing_t1_p50_decision_artifact(
    *,
    runtime_contract_path: Path,
    arm_results_manifest_path: Path,
    active8_inventory_path: Path,
    active8_inventory_file_sha256: str,
    gate_zero_evidence_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate exact physical parents, recompute the decision, and publish it."""

    runtime_path = Path(runtime_contract_path).resolve()
    manifest_path = Path(arm_results_manifest_path).resolve()
    inventory_path = Path(active8_inventory_path).resolve()
    gate_zero_path = Path(gate_zero_evidence_path).resolve()
    output = Path(output_path).resolve()
    try:
        runtime_contract = load_editing_t1_runtime_contract(runtime_path)
    except EditingT1RuntimeError as error:
        raise EditingT1DecisionCliError(
            f"T1 V4 runtime authority is invalid: {runtime_path}"
        ) from error

    expected_inventory_file_sha256 = str(
        runtime_contract.payload["active8_inventory_manifest_file_sha256"]
    )
    if active8_inventory_file_sha256 != expected_inventory_file_sha256:
        raise EditingT1DecisionCliError(
            "provided Active8 inventory file SHA-256 disagrees with the frozen "
            "T1 V4 runtime authority"
        )
    try:
        admission = load_active8_trace_admission(
            inventory_path,
            expected_manifest_file_sha256=expected_inventory_file_sha256,
            expected_inventory_sha256=str(runtime_contract.payload["active8_inventory_sha256"]),
            expected_effective_source_corpus_cache_sha256=str(
                runtime_contract.payload["active8_effective_source_corpus_cache_sha256"]
            ),
            expected_support_contract_sha256=str(
                runtime_contract.payload["active8_support_contract_sha256"]
            ),
        )
    except Active8TraceInventoryError as error:
        raise EditingT1DecisionCliError(
            f"exact Active8 admission is invalid: {inventory_path}"
        ) from error
    observed_active8_identity = active8_t1_identity(admission)
    expected_active8_identity = {
        field: runtime_contract.payload[field] for field in ACTIVE8_T1_IDENTITY_FIELDS
    }
    if observed_active8_identity != expected_active8_identity:
        raise EditingT1DecisionCliError(
            "Active8 admission identities disagree with the frozen T1 V4 authority"
        )

    gate_zero_payload, gate_zero_file_sha256 = _load_json_object(
        gate_zero_path,
        name="Gate0 structural evidence",
    )
    try:
        validate_gate_zero_structural_evidence(
            gate_zero_payload,
            source_admission=admission,
        )
    except EditingP50PrerequisiteError as error:
        raise EditingT1DecisionCliError(
            f"Gate0 structural evidence is invalid: {gate_zero_path}"
        ) from error
    try:
        decision = build_t1_p50_decision(
            runtime_contract_path=runtime_path,
            arm_results_manifest_path=manifest_path,
            decision_directory=output.parent,
            source_admission=admission,
            source_inventory_file_sha256=expected_inventory_file_sha256,
            gate_zero_evidence_file_sha256=gate_zero_file_sha256,
        )
    except EditingP50PrerequisiteError as error:
        raise EditingT1DecisionCliError(
            "T1-to-P50 decision recomputation rejected the physical evidence"
        ) from error
    _atomic_immutable_json_write(output, decision)
    return decision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        required=True,
        help="Frozen V4 T1 runtime authority inside the decision directory.",
    )
    parser.add_argument(
        "--arm-results-manifest",
        type=Path,
        required=True,
        help="Complete physical T1 arm-results manifest inside the decision directory.",
    )
    parser.add_argument(
        "--active8-inventory",
        type=Path,
        required=True,
        help="Exact whole-trace Active8 admission inventory.",
    )
    parser.add_argument(
        "--active8-inventory-file-sha256",
        required=True,
        help="Expected physical SHA-256 frozen in the V4 T1 authority.",
    )
    parser.add_argument(
        "--gate-zero-evidence",
        type=Path,
        required=True,
        help="Self-hashed Gate0 structural evidence bound to the Active8 admission.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Immutable self-hashed T1 decision artifact to publish atomically.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision = build_editing_t1_p50_decision_artifact(
            runtime_contract_path=args.runtime_contract,
            arm_results_manifest_path=args.arm_results_manifest,
            active8_inventory_path=args.active8_inventory,
            active8_inventory_file_sha256=args.active8_inventory_file_sha256,
            gate_zero_evidence_path=args.gate_zero_evidence,
            output_path=args.output,
        )
    except EditingT1DecisionCliError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": decision["status"],
                "bounded_p50_authorized": decision["bounded_p50_authorized"],
                "full_training_authorized": False,
                "decision_sha256": decision["decision_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
