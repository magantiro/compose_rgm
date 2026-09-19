"""Run the frozen 50-prompt, zero-oracle fragment proposal panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import rdBase

from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_constrained_runner import (
    ProposalLimits,
    prompt_id,
    propose_prompt,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

DEFAULT_CONTRACT = Path("configs/fragment_constrained_proposal_panel_v1.json")
DEFAULT_OUTPUT = Path("diagnostics/fragment_constrained_proposal_panel_v1/result.json")
SCHEMA = "compose_fragment_prompt_panel_v1"


def _identity(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"contract is not a sealed payload envelope: {path}")
    payload = envelope["payload"]
    if not isinstance(payload, dict) or envelope["payload_sha256"] != _identity(
        payload
    ):
        raise ValueError(f"contract self-hash mismatch: {path}")
    return payload, str(envelope["payload_sha256"])


def _validate_frozen_inputs(contract: dict[str, Any]) -> None:
    inputs = contract["inputs"]
    material = {
        inputs["prompt_manifest"]["path"]: inputs["prompt_manifest"]["sha256"],
        **contract["material_code_sha256"],
    }
    for raw_path, expected in material.items():
        path = Path(raw_path)
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"frozen input hash mismatch for {path}: {observed} != {expected}"
            )


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def _audit_complete(prompt: FragmentPrompt, row: dict[str, Any]) -> dict[str, Any]:
    endpoint = str(row["endpoint"])
    constraint = check_fragment_constraint(prompt, endpoint)
    receipt = row["receipt"]
    states = tuple(decode_state(value) for value in receipt["states"])
    primitive_edits = int(receipt["primitive_edits"])
    checks = {
        "prompt_identity_matches": row["prompt_id"] == prompt_id(prompt),
        "receipt_endpoint_matches": receipt["endpoint"] == endpoint,
        "final_state_matches_endpoint": bool(states)
        and canonical_state_key(states[-1]) == endpoint,
        "source_state_matches_receipt": bool(states)
        and canonical_state_key(states[0]) == row["source"]["canonical_smiles"],
        "action_count_matches": len(receipt["actions"]) == primitive_edits,
        "state_count_matches": len(states) == primitive_edits + 1,
        "all_committed_states_chemically_valid": bool(states)
        and all(is_valid_state(state) for state in states),
        "all_committed_states_connected_or_null": bool(states)
        and all(is_connected_or_null(state) for state in states),
        "fragment_constraint_satisfied": constraint.satisfied,
        "runner_validation_agrees": all(
            (
                row["validation"]["chemical_valid"],
                row["validation"]["connected_or_null_all_committed_states"],
                row["validation"]["fragment_constraint_satisfied"],
            )
        ),
    }
    return {
        **checks,
        "constraint_reason": constraint.reason,
        "exact_valid_execution": all(checks.values()),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(rows)
    complete = [row for row in rows if row["status"] == "complete"]
    abstained = [row for row in rows if row["status"] == "abstained"]
    failed = [row for row in rows if row["status"] == "failed"]
    constraint_valid = [
        row
        for row in complete
        if row["independent_audit"]["fragment_constraint_satisfied"]
    ]
    exact_valid = [
        row for row in complete if row["independent_audit"]["exact_valid_execution"]
    ]
    endpoints = [row["endpoint"] for row in complete]
    return {
        "attempted_prompts": attempted,
        "completed_proposals": len(complete),
        "abstained_prompts": len(abstained),
        "unexpected_failures": len(failed),
        "proposal_coverage": _ratio(len(complete), attempted),
        "constraint_precision": _ratio(len(constraint_valid), len(complete)),
        "exact_valid_execution_yield": _ratio(len(exact_valid), attempted),
        "unique_endpoints": {
            "count": len(set(endpoints)),
            "completed_endpoint_count": len(endpoints),
            "fraction": len(set(endpoints)) / len(endpoints) if endpoints else None,
        },
        "failure_reasons": {
            "abstentions": dict(
                sorted(Counter(row["reason_code"] for row in abstained).items())
            ),
            "unexpected_failures": dict(
                sorted(Counter(row["error_type"] for row in failed).items())
            ),
            "constraint_failures": dict(
                sorted(
                    Counter(
                        str(row["independent_audit"]["constraint_reason"])
                        for row in complete
                        if not row["independent_audit"]["fragment_constraint_satisfied"]
                    ).items()
                )
            ),
        },
    }


def build_artifact(*, contract_path: Path) -> dict[str, Any]:
    contract, contract_payload_sha256 = _load_contract(contract_path)
    _validate_frozen_inputs(contract)
    asset = Path(contract["inputs"]["prompt_manifest"]["path"])
    prompts = load_genmol_prompts(asset)
    expected_prompt_count = int(contract["selection"]["expected_prompt_count"])
    if len(prompts) != expected_prompt_count:
        raise ValueError(
            f"prompt count drift: {len(prompts)} != {expected_prompt_count}"
        )

    proposal = contract["proposal_law"]
    limits = ProposalLimits(**proposal["limits"])
    variant_index = int(proposal["variant_index"])
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        try:
            row = propose_prompt(prompt, variant_index=variant_index, limits=limits)
            if row["status"] == "complete":
                row["independent_audit"] = _audit_complete(prompt, row)
        # The panel must preserve every prompt outcome. Unexpected exceptions are
        # recorded verbatim and make the command fail after all prompts are tried.
        except Exception as error:  # noqa: BLE001
            row = {
                "schema_version": "compose_fragment_prompt_failure_v1",
                "prompt_id": prompt_id(prompt),
                "drug_name": prompt.drug_name,
                "task": prompt.task.value,
                "variant_index": variant_index,
                "status": "failed",
                "error_type": type(error).__name__,
                "detail": str(error),
                "oracle_calls": 0,
                "scoring_calls": 0,
            }
        rows.append(row)

    overall = _summarize(rows)
    by_task = {
        task.value: _summarize([row for row in rows if row["task"] == task.value])
        for task in FragmentTask
    }
    return {
        "schema_version": SCHEMA,
        "evidence_class": "computed_zero_oracle_full_prompt_panel",
        "claim_boundary": contract["claim_boundary"],
        "contract": {
            "path": str(contract_path),
            "physical_sha256": _sha256(contract_path),
            "payload_sha256": contract_payload_sha256,
        },
        "selection": contract["selection"],
        "configuration": proposal,
        "inputs": contract["inputs"],
        "overall": overall,
        "by_task": by_task,
        "proposals": rows,
        "provenance": {
            "code_revision": _revision(),
            "code_revision_role": (
                "pre-commit base revision; frozen material file hashes bind this panel"
            ),
            "material_code_sha256": contract["material_code_sha256"],
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "deterministic": True,
            "cpu_only": True,
            "random_seed": None,
            "oracle_calls": 0,
            "scoring_calls": 0,
            "modal_launches": 0,
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = build_artifact(contract_path=args.contract)
    _atomic_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall": artifact["overall"],
                "by_task": artifact["by_task"],
            },
            sort_keys=True,
        )
    )
    if artifact["overall"]["unexpected_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
