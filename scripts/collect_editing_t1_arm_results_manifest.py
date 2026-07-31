#!/usr/bin/env python3
"""Collect independently retained Modal receipts into one physical T1 manifest.

The collector never derives receipt identities from durable result payloads.
It requires the local mirrors of the immutable Modal outputs plus one or more
locally retained orchestrator receipts, validates their exact union against the
pre-result runtime contract, and delegates result validation to the production
T1 arm-results manifest builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modal_apps.run_editing_t1_successor_gate import (  # noqa: E402
    load_t1_modal_launch_receipt,
)
from compose_v4.experiments.editing_p50_prerequisites import (  # noqa: E402
    EditingP50PrerequisiteError,
    build_t1_arm_results_manifest,
    planned_t1_arms,
    validate_t1_arm_results_manifest,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    ACTIVE8_T1_IDENTITY_FIELDS,
)
from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    EditingT1RuntimeError,
    EditingT1RuntimeContract,
    load_editing_t1_runtime_contract,
)


class EditingT1ReceiptCollectionError(ValueError):
    """A physical T1 launch receipt set is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise EditingT1ReceiptCollectionError(
            f"cannot hash required T1 artifact: {path}"
        ) from error
    return digest.hexdigest()


def _arm_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        key = (
            value["family"],
            value["panel_kind"],
            value["scope"],
        )
    except KeyError as error:
        raise EditingT1ReceiptCollectionError(
            f"T1 receipt arm is missing identity field {error.args[0]}"
        ) from error
    if not all(isinstance(item, str) and item for item in key):
        raise EditingT1ReceiptCollectionError(
            "T1 receipt arm identity fields must be nonempty strings"
        )
    return key


def _safe_local_result_path(
    result_directory: Path,
    relative_path: object,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise EditingT1ReceiptCollectionError(
            "T1 receipt result_relative_path must be a nonempty string"
        )
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise EditingT1ReceiptCollectionError(
            "T1 receipt result path escapes the local result directory"
        )
    candidate = result_directory.joinpath(*pure.parts).resolve()
    if not candidate.is_relative_to(result_directory):
        raise EditingT1ReceiptCollectionError(
            "T1 receipt result path escapes the local result directory"
        )
    if not candidate.is_file():
        raise EditingT1ReceiptCollectionError(
            f"T1 receipt result is not locally materialized: {candidate}"
        )
    return candidate


def _expected_contract_identities(
    contract: EditingT1RuntimeContract,
) -> Mapping[str, str]:
    return {
        "contract_sha256": contract.sha256,
        "numeric_thresholds_sha256": contract.numeric_thresholds_sha256,
        "panel_artifact_sha256": str(contract.payload["panel_artifact_sha256"]),
        "panel_selection_sha256": str(contract.payload["panel_selection_sha256"]),
        "panel_census_sha256": str(contract.payload["panel_census_sha256"]),
        "panel_capacity_strata_sha256": str(contract.payload["panel_capacity_strata_sha256"]),
        **{field: str(contract.payload[field]) for field in ACTIVE8_T1_IDENTITY_FIELDS},
    }


def _atomic_immutable_json_write(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
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
            if path.read_bytes() != content:
                raise EditingT1ReceiptCollectionError(
                    f"immutable T1 arm-results manifest already differs: {path}"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def collect_editing_t1_arm_results_manifest(
    *,
    runtime_contract_path: Path,
    result_directory: Path,
    launch_receipt_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Validate a complete independent receipt union and build the manifest."""

    if not launch_receipt_paths:
        raise EditingT1ReceiptCollectionError("at least one retained T1 launch receipt is required")
    receipt_paths = tuple(Path(path).resolve() for path in launch_receipt_paths)
    if len(receipt_paths) != len(set(receipt_paths)):
        raise EditingT1ReceiptCollectionError("T1 launch receipt paths contain duplicates")
    try:
        result_root = Path(result_directory).resolve(strict=True)
    except OSError as error:
        raise EditingT1ReceiptCollectionError(
            f"T1 local result directory is absent: {result_directory}"
        ) from error
    if not result_root.is_dir():
        raise EditingT1ReceiptCollectionError(
            f"T1 local result directory is not a directory: {result_root}"
        )
    output = Path(output_path).resolve()
    if output.parent != result_root:
        raise EditingT1ReceiptCollectionError(
            "T1 arm-results manifest must be written directly inside the local result directory"
        )

    contract_path = Path(runtime_contract_path).resolve()
    try:
        contract = load_editing_t1_runtime_contract(contract_path)
    except (OSError, EditingT1RuntimeError, ValueError) as error:
        raise EditingT1ReceiptCollectionError(
            f"T1 pre-result runtime contract is invalid: {contract_path}"
        ) from error
    planned = planned_t1_arms(contract)
    planned_keys = {_arm_key(arm) for arm in planned}
    expected_identities = _expected_contract_identities(contract)
    result_paths: dict[tuple[str, str, str], Path] = {}
    retained_receipts: dict[
        tuple[str, str, str],
        dict[str, object],
    ] = {}
    source_commits: set[str] = set()

    for receipt_path in sorted(receipt_paths):
        try:
            receipt = load_t1_modal_launch_receipt(receipt_path)
        except (OSError, ValueError) as error:
            raise EditingT1ReceiptCollectionError(
                f"T1 launch receipt is dirty or incomplete: {receipt_path}: {error}"
            ) from error
        source_commits.add(str(receipt["source_commit"]))
        arms = receipt["arms"]
        assert isinstance(arms, list)
        for arm in arms:
            assert isinstance(arm, Mapping)
            key = _arm_key(arm)
            if key not in planned_keys:
                raise EditingT1ReceiptCollectionError(
                    f"T1 launch receipt contains an arm absent from the runtime contract: {key}"
                )
            if key in result_paths:
                raise EditingT1ReceiptCollectionError(
                    f"T1 launch receipts duplicate or conflict for arm {key}"
                )
            for field, expected in expected_identities.items():
                if arm[field] != expected:
                    raise EditingT1ReceiptCollectionError(
                        f"T1 launch receipt is stale for {key}: {field}"
                    )
            local_result = _safe_local_result_path(
                result_root,
                arm["result_relative_path"],
            )
            result_paths[key] = local_result
            retained_receipts[key] = {
                "file_sha256": arm["output_sha256"],
                "result_sha256": arm["result_sha256"],
                "cache_receipts": arm["cache_receipts"],
            }

    if len(source_commits) != 1:
        raise EditingT1ReceiptCollectionError(
            "T1 launch receipts were produced from different source commits"
        )
    missing = sorted(planned_keys - set(result_paths))
    if missing:
        raise EditingT1ReceiptCollectionError(
            f"T1 launch receipts do not cover the complete planned arm matrix: {missing}"
        )
    extra = sorted(set(result_paths) - planned_keys)
    if extra:
        raise EditingT1ReceiptCollectionError(f"T1 launch receipts contain unplanned arms: {extra}")

    try:
        manifest = build_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_paths=result_paths,
            retained_result_receipts=retained_receipts,
            manifest_directory=result_root,
        )
        validate_t1_arm_results_manifest(
            manifest,
            manifest_path=output,
            runtime_contract=contract,
            runtime_contract_file_sha256=_sha256_file(contract_path),
        )
    except EditingP50PrerequisiteError as error:
        raise EditingT1ReceiptCollectionError(
            f"T1 physical arm-results manifest validation failed: {error}"
        ) from error
    _atomic_immutable_json_write(output, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect independent local T1 Modal receipts into the complete "
            "physical arm-results manifest"
        )
    )
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument(
        "--launch-receipt",
        type=Path,
        action="append",
        required=True,
        help="repeat for every independently retained Modal invocation receipt",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = collect_editing_t1_arm_results_manifest(
        runtime_contract_path=args.runtime_contract,
        result_directory=args.result_directory,
        launch_receipt_paths=args.launch_receipt,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "result_count": len(manifest["results"]),
                "output": str(args.output),
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
