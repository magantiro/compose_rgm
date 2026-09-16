"""Split-first pre-fit gates for complete-region patch policy attempt 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.control.complete_region_patch_policy import (
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
    patch_stream_support,
)
from compose_v4.control.complete_region_where_policy import (
    declared_where_support_size,
    decode_where_mask,
    encode_where_mask,
    source_component_count,
)
from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from tools.t4_complete_region_patch_policy import _teacher_corpus

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_complete_region_patch_policy_v2.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_complete_region_patch_policy/attempt_2/support.json"
SUPPORT_SCHEMA = "t4_complete_region_patch_policy_v2_support_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite attempt-2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def load_contract(path: Path = CONTRACT) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(payload):
        raise ValueError(f"attempt-2 contract is not self-hashed: {path}")
    if any(payload["costs"].values()):
        raise ValueError("attempt-2 contract authorizes external cost")
    for row in payload["inputs"].values():
        input_path = ROOT / row["path"]
        if sha256_file(input_path) != row["sha256"]:
            raise ValueError(f"attempt-2 frozen input hash changed: {input_path}")
    return payload


def _require_clean() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("attempt-2 authoritative gate requires clean source")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def support_gate(output: Path, *, contract_path: Path = CONTRACT) -> dict:
    contract = load_contract(contract_path)
    revision = _require_clean()
    teacher_rows, metadata = _teacher_corpus()
    rows = []
    for row in teacher_rows:
        decision = encode_where_mask(row["source"], row["binding"])
        decoded = decode_where_mask(row["source"], decision)
        where_exact = set(decoded) == set(row["binding"])
        where_supported = bool(decoded) and where_exact
        patch_supported, patch_failure = patch_stream_support(row["patch"])
        patch_tokens = encode_patch_stream(row["patch"])
        patch_decoded = decode_patch_stream(
            SourceRegionContext.from_subgoal(row["patch"]), patch_tokens
        )
        rows.append(
            {
                "teacher_program_id": row["route_id"],
                "source_group": row["source_group"],
                "region_index": row["region_index"],
                "live_roles": len(decision.selected),
                "selected_roles": sum(decision.selected),
                "source_components": source_component_count(row["source"], row["binding"]),
                "declared_full_where_support_size": declared_where_support_size(row["source"]),
                "where_declared_supported_without_injection": where_supported,
                "where_mask_roundtrip_exact": where_exact,
                "teacher_injected": False,
                "patch_token_count": len(patch_tokens),
                "patch_supported": patch_supported,
                "patch_roundtrip_exact": patch_decoded == row["patch"],
                "failure_reason": patch_failure,
            }
        )
    if len(rows) != 147:
        raise RuntimeError(f"attempt-2 support census changed: {len(rows)}")
    folds = []
    for split in predeclared_source_folds(metadata):
        held_sources = set(split["test_sources"])
        held = [row for row in rows if row["source_group"] in held_sources]
        where_supported = sum(row["where_declared_supported_without_injection"] for row in held)
        patch_supported = sum(row["patch_supported"] for row in held)
        patch_exact = sum(row["patch_roundtrip_exact"] for row in held)
        folds.append(
            {
                "fold": split["fold"],
                "held_source_hashes": sorted(identity(value) for value in held_sources),
                "held_regions": len(held),
                "where_declared_supported_without_injection": where_supported,
                "where_coverage": where_supported / len(held),
                "patch_supported": patch_supported,
                "patch_roundtrip_exact": patch_exact,
                "patch_precision": patch_exact / max(1, patch_supported),
            }
        )
    where_supported = sum(row["where_declared_supported_without_injection"] for row in rows)
    patch_supported = sum(row["patch_supported"] for row in rows)
    patch_exact = sum(row["patch_roundtrip_exact"] for row in rows)
    payload = {
        "schema_version": SUPPORT_SCHEMA,
        "evidence": "computed split-first zero-oracle attempt-2 pre-fit support audit",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation_revision": revision,
        "where_support": {
            "representation": "implicit nonempty canonical current-role mask",
            "declared_cardinality": "2^n_live - 1",
            "teacher_injection": False,
            "covered": where_supported,
            "denominator": len(rows),
            "coverage": where_supported / len(rows),
            "live_role_distribution": dict(
                sorted(Counter(row["live_roles"] for row in rows).items())
            ),
            "selected_role_distribution": dict(
                sorted(Counter(row["selected_roles"] for row in rows).items())
            ),
            "source_component_distribution": dict(
                sorted(Counter(row["source_components"] for row in rows).items())
            ),
            "minimum_full_support_size": min(
                row["declared_full_where_support_size"] for row in rows
            ),
            "maximum_full_support_size": max(
                row["declared_full_where_support_size"] for row in rows
            ),
            "beam_budget_is_not_full_support": contract["where"]["beam_budget"],
        },
        "patch_support": {
            "covered": patch_supported,
            "denominator": len(rows),
            "coverage": patch_supported / len(rows),
            "roundtrip_exact": patch_exact,
            "precision": patch_exact / max(1, patch_supported),
        },
        "folds": folds,
        "rows": rows,
        "gate": {
            "where_passed": where_supported == len(rows) == 147,
            "patch_passed": patch_supported == patch_exact == len(rows) == 147,
            "passed": where_supported == patch_supported == patch_exact == len(rows) == 147,
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    payload["decision"] = (
        "prefit_support_passed_fitting_authorized"
        if payload["gate"]["passed"]
        else "prefit_support_failed_stop_before_fit"
    )
    _publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = support_gate(arguments.output.resolve(), contract_path=arguments.contract.resolve())
    print(
        json.dumps(
            {
                "where_support": result["where_support"],
                "patch_support": result["patch_support"],
                "gate": result["gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
