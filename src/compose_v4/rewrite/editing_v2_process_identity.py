"""Content identity for the complete frozen Editing-V2 semantic process."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from compose_v4.rewrite.action_codec_v4 import (
    ACTIVE8_EXECUTOR_RULES,
    codec_implementation_hash,
)
from compose_v4.rewrite.action_codec_v4 import (
    SCHEMA_VERSION as ACTION_CODEC_SCHEMA_VERSION,
)

PROCESS_IDENTITY_SCHEMA = "compose.editing.semantic_process_identity"
PROCESS_IDENTITY_SCHEMA_VERSION = 1
PROCESS_SEMANTICS = "semantic_editing_v2_v1"
_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v1.json"
_IMPLEMENTATION_RELATIVE_PATHS = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/data/charge_policy.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/experiments/reference_successor_kernel.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/aromatic_kekule.py",
    "src/compose_v4/rewrite/editing_v2_process_identity.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/ring_restate_semantics.py",
    "src/compose_v4/rewrite/semantic_atom_restate.py",
    "src/compose_v4/rewrite/semantic_cycle_close.py",
    "src/compose_v4/rewrite/semantic_cycle_open.py",
    "src/compose_v4/rewrite/semantic_trace.py",
    "src/compose_v4/rewrite/trace_shard_v3.py",
    "src/compose_v4/rewrite/tracelet_fiber.py",
)


class EditingV2ProcessIdentityError(ValueError):
    """The frozen process contract or implementation boundary is incomplete."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )


def _contract_self_hash(contract: dict[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )


@lru_cache(maxsize=1)
def editing_v2_process_identity() -> dict[str, object]:
    """Return a full-SHA identity whose change invalidates semantic artifacts."""

    root = _repository_root()
    contract_path = root / _CONTRACT_RELATIVE_PATH
    try:
        contract_bytes = contract_path.read_bytes()
        contract = json.loads(contract_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise EditingV2ProcessIdentityError(
            f"cannot read frozen semantic process contract {contract_path}"
        ) from error
    if not isinstance(contract, dict):
        raise EditingV2ProcessIdentityError(
            "semantic process contract must be an object"
        )
    expected_contract_sha256 = _contract_self_hash(contract)
    if contract.get("contract_sha256") != expected_contract_sha256:
        raise EditingV2ProcessIdentityError(
            "semantic process contract self-hash does not match its contents"
        )
    declared_rules = tuple(contract["action_codec"]["public_editing_v2_rules"])
    if frozenset(declared_rules) != frozenset(ACTIVE8_EXECUTOR_RULES):
        raise EditingV2ProcessIdentityError(
            "semantic process contract and ActionCodecV4 Active8 rules disagree"
        )

    source_sha256: dict[str, str] = {}
    for relative_path in _IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative_path
        try:
            source_sha256[relative_path] = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise EditingV2ProcessIdentityError(
                f"semantic process identity source is missing: {relative_path}"
            ) from error
    body: dict[str, object] = {
        "schema": PROCESS_IDENTITY_SCHEMA,
        "schema_version": PROCESS_IDENTITY_SCHEMA_VERSION,
        "process_semantics": PROCESS_SEMANTICS,
        "contract_relative_path": _CONTRACT_RELATIVE_PATH,
        "contract_sha256": expected_contract_sha256,
        "contract_physical_sha256": _sha256_bytes(contract_bytes),
        "action_codec_schema_version": ACTION_CODEC_SCHEMA_VERSION,
        "action_codec_implementation_hash": codec_implementation_hash(),
        "active_executor_rules": list(ACTIVE8_EXECUTOR_RULES),
        "implementation_source_sha256": source_sha256,
    }
    return {**body, "process_identity_sha256": _canonical_sha256(body)}


def require_editing_v2_process_identity(expected_sha256: str) -> dict[str, object]:
    identity = editing_v2_process_identity()
    if identity["process_identity_sha256"] != expected_sha256:
        raise EditingV2ProcessIdentityError(
            "Editing-V2 process identity differs from the bound artifact identity"
        )
    return identity


__all__ = [
    "PROCESS_IDENTITY_SCHEMA",
    "PROCESS_IDENTITY_SCHEMA_VERSION",
    "PROCESS_SEMANTICS",
    "EditingV2ProcessIdentityError",
    "editing_v2_process_identity",
    "require_editing_v2_process_identity",
]
