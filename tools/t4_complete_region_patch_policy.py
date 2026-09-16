"""Split-first support and policy gates for complete T4 region patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.control.complete_region_patch_policy import (
    ATOM_TYPES,
    BOND_ORDERS,
    DEGREES,
    FORMAL_CHARGES,
    HYDROGEN_COUNTS,
    PatchToken,
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
    patch_stream_support,
)
from compose_v4.control.complete_region_program import CompleteRegionProgram
from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_structural_subgoal_audit import teacher_traces

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_complete_region_patch_policy_v1.json"
SUPPORT = ROOT / "diagnostics/t4_complete_region_runtime/attempt_1/support.json"
BENCHMARK = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_complete_region_patch_policy/attempt_1"
SUPPORT_SCHEMA = "t4_complete_region_patch_policy_support_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite complete-region policy artifact: {path}")
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
        raise ValueError(f"complete-region policy contract is not self-hashed: {path}")
    if any(value for value in payload["costs"].values()):
        raise ValueError("complete-region policy contract authorizes external cost")
    for row in payload["inputs"].values():
        input_path = ROOT / row["path"]
        if sha256_file(input_path) != row["sha256"]:
            raise ValueError(f"complete-region policy input hash changed: {input_path}")
    return payload


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _require_clean() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("authoritative complete-region policy gate requires clean source")
    return _revision()


def _token_key(token: PatchToken) -> tuple[str, int, str]:
    return token.kind, token.value, token.factor


def support_gate(output: Path, *, contract_path: Path = CONTRACT) -> dict:
    contract = load_contract(contract_path)
    revision = _require_clean()
    support = unseal(SUPPORT)
    teacher = teacher_traces()
    by_program = {row["program_id"]: row for row in teacher}
    if len(by_program) != 77:
        raise RuntimeError("teacher route census changed")
    metadata = source_group_map(unseal(BENCHMARK), json.loads(SEEDS.read_text()))
    folds = predeclared_source_folds(metadata)
    rows = []
    for support_row in support["rows"]:
        program_id = support_row["teacher_program_id"]
        teacher_row = by_program.get(program_id)
        if teacher_row is None:
            raise ValueError(f"support route lacks exact teacher provenance: {program_id}")
        program = CompleteRegionProgram.from_payload(support_row["runtime_program"])
        for region_index, decision in enumerate(program.decisions):
            tokens = encode_patch_stream(decision.patch)
            covered, reason = patch_stream_support(decision.patch)
            decoded = (
                decode_patch_stream(SourceRegionContext.from_subgoal(decision.patch), tokens)
                if covered
                else None
            )
            rows.append(
                {
                    "teacher_program_id": program_id,
                    "source_group": teacher_row["source_group"],
                    "region_index": region_index,
                    "subgoal_id": decision.patch.subgoal_id,
                    "token_count": len(tokens),
                    "factor_token_counts": dict(
                        sorted(Counter(token.factor for token in tokens).items())
                    ),
                    "supported": covered,
                    "roundtrip_exact": decoded == decision.patch,
                    "failure_reason": reason,
                    "cross_region_created_role_references": sum(
                        value is not None for value in decision.input_provenance
                    ),
                    "tokens": [token.payload() for token in tokens],
                }
            )
    if len(rows) != 147:
        raise RuntimeError(f"complete-region support census changed: {len(rows)}")
    fold_rows = []
    for split in folds:
        train_sources = set(split["train_sources"])
        held_sources = set(split["test_sources"])
        train = [row for row in rows if row["source_group"] in train_sources]
        held = [row for row in rows if row["source_group"] in held_sources]
        train_tokens = {
            _token_key(PatchToken.from_payload(token)) for row in train for token in row["tokens"]
        }
        held_token_rows = [
            _token_key(PatchToken.from_payload(token)) for row in held for token in row["tokens"]
        ]
        supported = sum(row["supported"] for row in held)
        exact = sum(row["roundtrip_exact"] for row in held)
        fold_rows.append(
            {
                "fold": split["fold"],
                "train_sources": sorted(train_sources),
                "held_sources": sorted(held_sources),
                "train_regions": len(train),
                "held_regions": len(held),
                "held_supported": supported,
                "held_support_coverage": supported / len(held),
                "held_roundtrip_exact": exact,
                "held_roundtrip_precision": exact / max(1, supported),
                "training_observed_token_vocabulary_size": len(train_tokens),
                "held_token_instances": len(held_token_rows),
                "held_token_instances_not_observed_in_train": sum(
                    token not in train_tokens for token in held_token_rows
                ),
                "held_unique_tokens_not_observed_in_train": len(
                    set(held_token_rows) - train_tokens
                ),
                "support_note": "Unseen empirical tokens remain representable only through the frozen platform value grammar; they are not leaked into training-fold learned statistics.",
            }
        )
    supported = sum(row["supported"] for row in rows)
    exact = sum(row["roundtrip_exact"] for row in rows)
    payload = {
        "schema_version": SUPPORT_SCHEMA,
        "evidence": "computed split-first zero-oracle complete-patch grammar support",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation_revision": revision,
        "fixed_platform_domains": {
            "atom_types": list(ATOM_TYPES),
            "formal_charges": list(FORMAL_CHARGES),
            "implicit_hydrogens": list(HYDROGEN_COUNTS),
            "degrees": list(DEGREES),
            "bond_orders": list(BOND_ORDERS),
        },
        "census": {
            "routes": 77,
            "regions": len(rows),
            "cross_region_created_role_references": sum(
                row["cross_region_created_role_references"] for row in rows
            ),
            "factor_token_counts": dict(
                sorted(Counter(token["factor"] for row in rows for token in row["tokens"]).items())
            ),
        },
        "gate": {
            "teacher_patch_grammar_support": {
                "covered": supported,
                "denominator": len(rows),
                "coverage": supported / len(rows),
            },
            "roundtrip_precision": {
                "exact": exact,
                "admitted": supported,
                "precision": exact / max(1, supported),
            },
            "passed": supported == exact == 147,
        },
        "folds": fold_rows,
        "rows": rows,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    if not payload["gate"]["passed"]:
        payload["decision"] = "stop_before_fit_missing_complete_patch_grammar_support"
    else:
        payload["decision"] = "complete_patch_grammar_support_gate_passed_fit_authorized"
    _publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "support.json")
    arguments = parser.parse_args()
    result = support_gate(arguments.output.resolve(), contract_path=arguments.contract.resolve())
    print(json.dumps({"census": result["census"], "gate": result["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
