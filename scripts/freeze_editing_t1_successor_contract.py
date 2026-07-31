#!/usr/bin/env python3
"""Freeze the result-independent current editing T1 runtime contract.

This command performs no optimization and cannot authorize training. Numeric
thresholds and optimizer settings must be supplied prospectively as explicit
JSON objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.active8_trace_inventory import (  # noqa: E402
    load_active8_trace_admission,
)
from compose_v4.experiments.editing_gate_zero_runtime import (  # noqa: E402
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    active8_t1_identity,
)
from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    EDITING_T1_V4_PANEL_RELATIVE_PATH,
    EDITING_T1_V7_CONTRACT_RELATIVE_PATH,
    EditingT1RuntimeError,
    build_editing_t1_runtime_contract,
    write_editing_t1_runtime_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(
    path: Path,
    *,
    name: str,
    max_file_bytes: int = 1 << 24,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > max_file_bytes:
        raise EditingT1RuntimeError(
            f"{name} is absent or exceeds its size bound: {source}"
        )
    try:
        payload = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1RuntimeError(f"{name} is invalid JSON: {source}") from error
    if not isinstance(payload, dict):
        raise EditingT1RuntimeError(f"{name} must contain one JSON object")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-zero-contract",
        type=Path,
        default=ROOT / "configs" / "editing_gate_zero_runtime_v2.json",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=ROOT / EDITING_T1_V4_PANEL_RELATIVE_PATH,
    )
    parser.add_argument(
        "--capacity-census",
        type=Path,
        default=ROOT / EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    )
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument(
        "--active8-inventory-file-sha256",
        required=True,
        help="Expected SHA-256 of the physical Active8 inventory manifest.",
    )
    parser.add_argument(
        "--optimization-json",
        type=Path,
        required=True,
        help="Path to an exact T1 optimization object chosen before results.",
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        required=True,
        help="Path to exact finite T1 thresholds chosen before results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / EDITING_T1_V7_CONTRACT_RELATIVE_PATH,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    gate_zero_contract = load_gate_zero_runtime_contract(args.gate_zero_contract)
    active8_admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=args.active8_inventory_file_sha256,
        expected_support_contract_sha256=gate_zero_contract.sha256,
    )
    gate_zero_unified_manifest = str(
        gate_zero_contract.sidecar["unified_packed_manifest_sha256"]
    )
    if active8_admission.unified_packed_manifest_sha256 != gate_zero_unified_manifest:
        raise EditingT1RuntimeError(
            "Gate0 validation source and Active8 inventory name different "
            "unified packed manifests"
        )
    panel = _load_json_object(args.panel, name="T1 V4 panel")
    capacity_census = _load_json_object(
        args.capacity_census,
        name="T1 capacity census",
    )
    contract = build_editing_t1_runtime_contract(
        gate_zero_runtime_contract_sha256=gate_zero_contract.sha256,
        panel=panel,
        capacity_census=capacity_census,
        capacity_census_file_sha256=_sha256(args.capacity_census),
        active8_identity=active8_t1_identity(active8_admission),
        optimization=_load_json_object(
            args.optimization_json,
            name="T1 prospective optimization settings",
            max_file_bytes=1 << 20,
        ),
        thresholds=_load_json_object(
            args.thresholds_json,
            name="T1 prospective numeric thresholds",
            max_file_bytes=1 << 20,
        ),
    )
    write_editing_t1_runtime_contract(contract, args.output)
    print(
        json.dumps(
            {
                "contract_sha256": contract.sha256,
                "numeric_thresholds_sha256": contract.numeric_thresholds_sha256,
                "output": str(args.output),
                "panel_artifact_sha256": contract.payload["panel_artifact_sha256"],
                "capacity_census_file_sha256": contract.payload[
                    "capacity_census_file_sha256"
                ],
                "capacity_census_sha256": contract.payload["capacity_census_sha256"],
                "active8_inventory_manifest_file_sha256": contract.payload[
                    "active8_inventory_manifest_file_sha256"
                ],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
