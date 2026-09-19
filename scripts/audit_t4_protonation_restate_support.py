#!/usr/bin/env python3
"""Zero-oracle exact-support probe for the sealed 5HT1B-2 strict endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rdkit

from compose_v4.control.protonation_restate_program import (
    compile_protonation_supported_target,
)
from compose_v4.rewrite.action_codec_v5 import codec_implementation_hash

SCHEMA = "t4_atom_protonation_restate_support_probe_v1"
DEFAULT_CONTRACT = Path("configs/t4_atom_protonation_restate_support_v1.json")
DEFAULT_AUDIT = Path(
    "diagnostics/t4_shared_retained_fiber_5ht1b_d06_v1/audit_20260919/result.json"
)
DEFAULT_OUTPUT = Path(
    "diagnostics/t4_atom_protonation_restate_support_v1/attempt_1/result.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_self_hashed(path: Path) -> tuple[dict, str]:
    document = json.loads(path.read_text())
    if set(document) != {"payload", "payload_sha256"}:
        raise ValueError(f"self-hashed document has unexpected fields: {path}")
    digest = hashlib.sha256(_canonical(document["payload"])).hexdigest()
    if digest != document["payload_sha256"]:
        raise ValueError(f"self-hash mismatch: {path}")
    return document["payload"], digest


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_probe(*, contract_path: Path, audit_path: Path) -> dict:
    contract, contract_payload_hash = _load_self_hashed(contract_path)
    audit, audit_payload_hash = _load_self_hashed(audit_path)
    expected_audit_hash = contract["governing_evidence"]["artifact_payload_sha256"]
    if audit_payload_hash != expected_audit_hash:
        raise ValueError("governing 5HT1B-2 audit identity disagrees with the contract")
    source = audit["root"]
    endpoints = [
        row["endpoint"]
        for row in audit["known_answer_diagnostics"]
        if row["evidence_role"] == "reported_delta06_IVG_endpoint_known_answer_only"
    ]
    if len(endpoints) != 3 or len(set(endpoints)) != 3:
        raise ValueError(
            "sealed audit must contain exactly three unique strict endpoints"
        )

    programs = []
    for endpoint in endpoints:
        result = compile_protonation_supported_target(source, endpoint)
        programs.append(
            {
                "target": endpoint,
                "status": result["status"],
                "exact_endpoint": bool(result.get("exact_endpoint", False)),
                "primitive_edits": result.get("primitive_edits"),
                "protonation_restate_count": result.get("protonation_restate_count", 0),
                "active8_tail_primitives": result.get("active8_tail_primitives"),
                "exact_execution_precision_numerator": result.get(
                    "exact_execution_precision_numerator",
                    0,
                ),
                "exact_execution_precision_denominator": result.get(
                    "exact_execution_precision_denominator",
                    0,
                ),
                "actions": result.get("actions", []),
                "residual_blocker": result.get("residual_blocker"),
            }
        )
    supported = sum(row["status"] == "supported" for row in programs)
    exact_numerator = sum(
        int(row["exact_execution_precision_numerator"]) for row in programs
    )
    exact_denominator = sum(
        int(row["exact_execution_precision_denominator"]) for row in programs
    )
    payload = {
        "schema_version": SCHEMA,
        "scientific_problem": contract["scientific_problem"],
        "primary_output": "teacher-forced complete-program support and exact replay receipts",
        "evidence": "computed zero-oracle answer-known support diagnostic",
        "claim_boundary": contract["teacher_forced_claim_boundary"],
        "costs": {"docking_calls": 0, "modal_launches": 0, "oracle_calls": 0},
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": _sha256_file(contract_path),
                "payload_sha256": contract_payload_hash,
            },
            "sealed_exhaustion_audit": {
                "path": str(audit_path),
                "sha256": _sha256_file(audit_path),
                "payload_sha256": audit_payload_hash,
            },
        },
        "implementation": {
            "code_revision": _git_revision(),
            "codec_implementation_hash": codec_implementation_hash(),
            "source_files_sha256": {
                path: _sha256_file(Path(path))
                for path in (
                    "src/compose_v4/rewrite/operators.py",
                    "src/compose_v4/rewrite/action_codec_v5.py",
                    "src/compose_v4/rewrite/kernel.py",
                    "src/compose_v4/control/protonation_restate_program.py",
                    "scripts/audit_t4_protonation_restate_support.py",
                )
            },
        },
        "software": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "platform": platform.platform(),
        },
        "source": source,
        "strict_endpoint_count": len(endpoints),
        "programs": programs,
        "aggregate": {
            "complete_program_support_numerator": supported,
            "complete_program_support_denominator": len(programs),
            "complete_program_support": supported / len(programs),
            "exact_execution_precision_numerator": exact_numerator,
            "exact_execution_precision_denominator": exact_denominator,
            "exact_execution_precision": (
                exact_numerator / exact_denominator if exact_denominator else None
            ),
            "primitive_counts": [row["primitive_edits"] for row in programs],
            "all_within_32_primitive_support": all(
                type(row["primitive_edits"]) is int and row["primitive_edits"] <= 32
                for row in programs
            ),
            "autonomous_proposal_recovery_evaluated": False,
        },
        "residual_blockers": [
            {"target": row["target"], "reason": row["residual_blocker"]}
            for row in programs
            if row["status"] != "supported"
        ],
    }
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = run_probe(contract_path=args.contract, audit_path=args.audit)
    _atomic_json(args.output, document)
    print(json.dumps(document["payload"]["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
