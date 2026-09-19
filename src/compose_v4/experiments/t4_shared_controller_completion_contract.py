"""Preparation-only contract checks for the nine-cell T4 completion campaign.

This module deliberately contains no launch function.  The preparation contract
must first acquire a sealed zero-oracle support artifact and a separate, exact
payload-bound authorization in a later revision.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "t4_shared_controller_completion_preparation_v1"
RUNTIME_SCHEMA_VERSION = "t4_shared_controller_completion_runtime_cell_v1"
CAPSULE_MANIFEST_SCHEMA_VERSION = "t4_exact_commit_source_capsule_manifest_v1"
PREPARED_STATUS = "PREPARED_PENDING_ZERO_ORACLE_GATE"
CONTRACT_RELATIVE_PATH = "configs/t4_shared_controller_completion_v1.json"

EXPECTED_EXPERTS = (
    "shallow",
    "anchored_replacement",
    "route_complete_region",
)
EXPECTED_CONTROLLER = {
    "batch": 8,
    "parents": 4,
    "parent_explore": 0.3,
    "exploration": 2,
    "expert_floor_rounds": 2,
    "route_scale_floor_rounds": 2,
    "value_penalty": 1.0,
    "docking_seed": 20260919,
    "proposal": {
        "shallow": {"draws": 480, "horizon": 3},
        "anchored_replacement": {"draws": 512, "horizon": 3},
        "route_complete_region": {
            "pool_size": 192,
            "realization_limit": 96,
            "beam_width": 48,
            "expansion_width": 48,
            "max_bindings_per_template": 4,
            "maximum_expansions": 4000,
            "scale_balanced": True,
            "training_scope": (
                "one shared task-independent expert fit on all 77 locked routes"
            ),
        },
    },
}

PARP_EVALUATOR = {
    "qvina02_sha256": "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0",
    "receptor_sha256": "8d0891ddf915f51cf3108f39dd9dc01dfcf86be342c566021956afdd79eb4ad9",
    "docking_box": [[26.413, 11.282, 27.238], [18.521, 17.479, 19.995]],
}
BRAF_EVALUATOR = {
    "qvina02_sha256": "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0",
    "receptor_sha256": "707b21bfb654321cb14d33ea07e9accab4cb9a884378285cd562c2324a31213a",
    "docking_box": [[84.194, 6.949, -7.081], [22.032, 19.211, 14.106]],
}

_SOURCES = {
    "parp1_0": (0, "CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3"),
    "parp1_1": (1, "COc1[nH]c3cccc2C(=O)NCCc1c23"),
    "parp1_2": (2, "O/N=C/c1cn3CCNC(=O)c2cccc1c23"),
    "braf_0": (
        9,
        "CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34",
    ),
    "braf_1": (
        10,
        "FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)NCCN1CCOCC1)c2)cc3)ccc4Cl",
    ),
    "braf_2": (
        11,
        "FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)Nc1cccnc1)c2)cc3)ccc4Cl",
    ),
}


def payload_identity(payload: dict[str, Any]) -> str:
    """Return the canonical payload identity without importing model code."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_cells() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source_cell in ("parp1_0", "parp1_1", "parp1_2"):
        index, smiles = _SOURCES[source_cell]
        rows.append(
            {
                "cell_key": f"{source_cell}_d04",
                "source_cell": source_cell,
                "target": "parp1",
                "source_global_index": index,
                "source_smiles": smiles,
                "delta": 0.4,
                "controller_seed": 2026091900 + index,
                "volume": f"compose-t4-shared-completion-v1-{source_cell.replace('_', '-')}-d04",
                "evaluator": PARP_EVALUATOR,
            }
        )
    for delta, tag in ((0.4, "d04"), (0.6, "d06")):
        for source_cell in ("braf_0", "braf_1", "braf_2"):
            index, smiles = _SOURCES[source_cell]
            rows.append(
                {
                    "cell_key": f"{source_cell}_{tag}",
                    "source_cell": source_cell,
                    "target": "braf",
                    "source_global_index": index,
                    "source_smiles": smiles,
                    "delta": delta,
                    "controller_seed": 2026091900 + index,
                    "volume": (
                        "compose-t4-shared-completion-v1-"
                        f"{source_cell.replace('_', '-')}-{tag}"
                    ),
                    "evaluator": BRAF_EVALUATOR,
                }
            )
    return tuple(rows)


EXPECTED_CELLS = _expected_cells()

FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "authorization_payload_sha256",
        "cell_to_winner",
        "comparator",
        "comparator_value",
        "known_endpoint",
        "launch_authorization",
        "legacy_resume",
        "prior_outcome",
        "prior_score",
        "promotion_criteria",
        "reported_ivg_delta_0_4",
        "reported_ivg_delta_0_6",
        "required_scored_authorization",
        "required_scored_authorization_template",
        "route_id",
        "source_registry",
        "target_to_program",
        "teacher_route",
        "winner_smiles",
    }
)


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_walk_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_walk_keys(item))
        return keys
    return set()


def _assert_runtime_sanitized(value: Any) -> None:
    leaked = sorted(FORBIDDEN_RUNTIME_KEYS.intersection(_walk_keys(value)))
    if leaked:
        raise ValueError(f"forbidden runtime fields: {leaked}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _validate_checkpoint(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    binding = contract.get("shared_route_checkpoint")
    if not isinstance(binding, dict):
        raise TypeError("shared checkpoint binding is missing")
    if (
        binding.get("path")
        != "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
    ):
        raise ValueError("shared checkpoint path drift")
    checkpoint_path = root / binding["path"]
    if sha256_file(checkpoint_path) != binding.get("sha256"):
        raise ValueError("shared checkpoint physical hash mismatch")
    envelope = _load_json_object(checkpoint_path)
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload_identity(payload) != envelope.get(
        "payload_sha256"
    ):
        raise ValueError("shared checkpoint envelope is invalid")
    if envelope["payload_sha256"] != binding.get("payload_sha256"):
        raise ValueError("shared checkpoint payload identity drift")
    expected = {
        "schema_version": "t4_shared_retained_rewrite_checkpoint_v1",
        "training_routes": 77,
        "training_regions": 147,
        "training_scope": "all_locked_t4_routes_shared_task_independent",
        "runtime_target_conditioning": False,
        "new_oracle_calls": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"shared checkpoint {key} drift")
    if len(payload.get("expert", {}).get("templates", ())) != 137:
        raise ValueError("shared checkpoint template census drift")
    _assert_runtime_sanitized(payload)
    return payload


def cell_runtime_payload(contract: dict[str, Any], cell_key: str) -> dict[str, Any]:
    """Return the only payload later runtime containers may receive."""

    cells = {row["cell_key"]: row for row in contract["cells"]}
    if cell_key not in cells:
        raise ValueError(f"unknown prepared cell {cell_key!r}")
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "preparation_status": contract["status"],
        "scored_calls_authorized": contract["scored_calls_authorized"],
        "cell": cells[cell_key],
        "controller": contract["controller"],
        "support": contract["support"],
        "budget": {
            "charged_call_ceiling": contract["charged_calls_per_cell"],
            "batch": contract["controller"]["batch"],
            "automatic_retries": 0,
            "replacement": False,
            "backfill": False,
        },
        "shared_route_checkpoint": contract["shared_route_checkpoint"],
        "receipt_policy": contract["receipt_policy"],
        "source_capsule": contract["source_capsule"],
    }
    _assert_runtime_sanitized(payload)
    return payload


def validate_preparation_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    """Validate a deliberately unlaunchable nine-cell preparation contract."""

    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = _load_json_object(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected completion preparation schema")
    if contract.get("status") != PREPARED_STATUS:
        raise ValueError("completion campaign is not in its prepared pending state")
    if contract.get("scored_calls_authorized") != 0:
        raise ValueError("prepared campaign must authorize zero scored calls")
    if contract.get("final_contract_payload_sha256") is not None:
        raise ValueError(
            "final contract identity must remain unset before support passes"
        )
    if contract.get("authorization") != {
        "state": "BLOCKED_PENDING_ZERO_ORACLE_GATE",
        "payload_sha256": None,
    }:
        raise ValueError("prepared authorization state drift")
    if contract.get("zero_oracle_support_gate") != {
        "state": "PENDING",
        "artifact": None,
        "artifact_sha256": None,
        "payload_sha256": None,
    }:
        raise ValueError("zero-oracle support gate is not pending and unbound")
    if contract.get("support") != "compose_valid":
        raise ValueError("endpoint support drift")
    if contract.get("charged_calls_per_cell") != 49:
        raise ValueError("per-cell charged-call ceiling drift")
    if contract.get("total_charged_call_ceiling") != 441:
        raise ValueError("campaign charged-call ceiling drift")
    if contract.get("automatic_retries") != 0:
        raise ValueError("automatic retry must remain disabled")
    if (
        contract.get("replacement") is not False
        or contract.get("backfill") is not False
    ):
        raise ValueError("replacement and backfill must remain disabled")
    if contract.get("controller") != EXPECTED_CONTROLLER:
        raise ValueError("shared three-expert controller settings drift")
    if tuple(contract["controller"]["proposal"]) != EXPECTED_EXPERTS:
        raise ValueError("proposal expert order drift")
    if contract.get("cells") != list(EXPECTED_CELLS):
        raise ValueError("nine-cell source/fiber matrix drift")
    keys = [row["cell_key"] for row in contract["cells"]]
    volumes = [row["volume"] for row in contract["cells"]]
    if len(keys) != len(set(keys)) or len(volumes) != len(set(volumes)):
        raise ValueError("cell keys and durable volumes must be one-to-one")
    if contract.get("receipt_policy") != {
        "publish_query_lock_before_docking": True,
        "one_volume_per_cell": True,
        "one_driver_per_cell": True,
        "missing_locked_query": "charge_once_after_deadline_without_resubmission",
        "continuation": "reserved_running_terminal_generation",
        "candidate_exhaustion": "terminal_without_replacement_or_backfill",
    }:
        raise ValueError("receipt or continuation policy drift")
    if contract.get("source_capsule") != {
        "state": "PENDING_EXACT_COMMIT_MANIFEST",
        "manifest_schema": CAPSULE_MANIFEST_SCHEMA_VERSION,
        "manifest_payload_sha256": None,
        "working_tree_packaging_allowed": False,
        "complete_file_census_required": True,
    }:
        raise ValueError("source capsule preparation state drift")
    checkpoint = _validate_checkpoint(root, contract)
    runtime_payloads = [cell_runtime_payload(contract, key) for key in keys]
    return {
        "schema_version": "t4_shared_controller_completion_preflight_v1",
        "status": PREPARED_STATUS,
        "ready_for_scored_launch": False,
        "scored_calls_authorized": 0,
        "contract_file_sha256": sha256_file(contract_path),
        "cell_keys": keys,
        "volumes": volumes,
        "charged_call_ceiling_per_cell": 49,
        "total_charged_call_ceiling": 441,
        "proposal_experts": list(EXPECTED_EXPERTS),
        "checkpoint_sha256": contract["shared_route_checkpoint"]["sha256"],
        "checkpoint_payload_sha256": contract["shared_route_checkpoint"][
            "payload_sha256"
        ],
        "checkpoint_training_routes": checkpoint["training_routes"],
        "checkpoint_training_regions": checkpoint["training_regions"],
        "runtime_payload_sha256": {
            row["cell"]["cell_key"]: payload_identity(row) for row in runtime_payloads
        },
        "blocking_requirements": [
            "sealed_nine_cell_zero_oracle_support_artifact",
            "clean_exact_commit_source_capsule_manifest",
            "final_self_hashed_scored_contract",
            "exact_payload_bound_user_authorization",
            "stale_braf_v4_terminal_receipt_reconciliation",
        ],
    }


def assert_scored_launch_blocked(root: Path, contract_path: Path) -> None:
    """Fail closed for every scored launch attempt under the preparation contract."""

    report = validate_preparation_contract(root, contract_path)
    if report["ready_for_scored_launch"] is not False:
        raise AssertionError("preparation validator unexpectedly enabled launch")
    raise RuntimeError(
        "scored launch is unavailable while the campaign is "
        "PREPARED_PENDING_ZERO_ORACLE_GATE"
    )


def _git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _safe_capsule_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe capsule path: {relative!r}")
    if str(path) != relative:
        raise ValueError(f"non-canonical capsule path: {relative!r}")
    return path


def verify_exact_commit_capsule(
    capsule_root: Path, manifest_path: Path
) -> dict[str, Any]:
    """Verify the complete file census and Git blob identity of a source capsule."""

    capsule_root = capsule_root.resolve()
    envelope = _load_json_object(manifest_path.resolve())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload_identity(payload) != envelope.get(
        "payload_sha256"
    ):
        raise ValueError("source capsule manifest envelope is invalid")
    if payload.get("schema_version") != CAPSULE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected source capsule manifest schema")
    revision = str(payload.get("code_revision", ""))
    tree = str(payload.get("git_tree", ""))
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("source capsule revision is not a full object identity")
    if not re.fullmatch(r"[0-9a-f]{40,64}", tree):
        raise ValueError("source capsule tree is not a full object identity")
    files = payload.get("files")
    if not isinstance(files, dict) or not files or list(files) != sorted(files):
        raise ValueError("source capsule file manifest must be nonempty and sorted")
    declared: set[str] = set()
    for relative, expected in files.items():
        path = _safe_capsule_path(str(relative))
        declared.add(str(path))
        if not isinstance(expected, dict):
            raise TypeError(f"invalid source capsule entry: {relative}")
        full = capsule_root.joinpath(*path.parts)
        if not full.is_file() or full.is_symlink():
            raise ValueError(
                f"source capsule file is absent or not regular: {relative}"
            )
        data = full.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected.get("sha256"):
            raise ValueError(f"source capsule SHA-256 mismatch: {relative}")
        if _git_blob_oid(data) != expected.get("git_blob_oid"):
            raise ValueError(f"source capsule Git blob mismatch: {relative}")
    symlinks = sorted(
        str(path.relative_to(capsule_root).as_posix())
        for path in capsule_root.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(f"source capsule may not contain symlinks: {symlinks}")
    observed = {
        str(path.relative_to(capsule_root).as_posix())
        for path in capsule_root.rglob("*")
        if path.is_file()
    }
    if observed != declared:
        raise ValueError(
            "source capsule file census mismatch: "
            f"missing={sorted(declared - observed)} extra={sorted(observed - declared)}"
        )
    return {
        "schema_version": "t4_exact_commit_source_capsule_verification_v1",
        "code_revision": revision,
        "git_tree": tree,
        "file_count": len(files),
        "manifest_payload_sha256": envelope["payload_sha256"],
        "verified": True,
    }


def _git_output(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, stderr=subprocess.STDOUT
    )


def build_exact_commit_capsule(
    *,
    repository_root: Path,
    revision: str,
    include: tuple[str, ...],
    capsule_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Materialize a complete capsule from Git blobs at one exact revision.

    ``include`` contains repository-relative files or directory prefixes.  The
    working tree is never read, so dirty and untracked files cannot enter the
    capsule accidentally.
    """

    repository_root = repository_root.resolve()
    capsule_root = capsule_root.resolve()
    manifest_path = manifest_path.resolve()
    if capsule_root.exists() or manifest_path.exists():
        raise FileExistsError("capsule and manifest outputs must not already exist")
    if not include:
        raise ValueError("source capsule include set must be nonempty")
    requested = tuple(sorted({str(_safe_capsule_path(item)) for item in include}))
    full_revision = _git_output(repository_root, "rev-parse", f"{revision}^{{commit}}")
    full_revision_text = full_revision.decode().strip()
    tree = (
        _git_output(repository_root, "rev-parse", f"{full_revision_text}^{{tree}}")
        .decode()
        .strip()
    )
    listing = _git_output(
        repository_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        full_revision_text,
        "--",
        *requested,
    )
    entries: list[tuple[str, str]] = []
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        relative = raw_path.decode()
        _safe_capsule_path(relative)
        if mode == "120000":
            raise ValueError(f"source capsule may not contain symlinks: {relative}")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(
                f"unsupported Git object in source capsule: {mode} {kind} {relative}"
            )
        entries.append((relative, oid))
    if not entries:
        raise ValueError("source capsule include set selected no committed files")
    if len(entries) != len({relative for relative, _ in entries}):
        raise ValueError("source capsule Git listing contains duplicate paths")

    capsule_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{capsule_root.name}.tmp-", dir=capsule_root.parent)
    )
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        files: dict[str, dict[str, str]] = {}
        for relative, oid in sorted(entries):
            data = _git_output(repository_root, "cat-file", "blob", oid)
            target = staging.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[relative] = {
                "git_blob_oid": oid,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        payload = {
            "schema_version": CAPSULE_MANIFEST_SCHEMA_VERSION,
            "code_revision": full_revision_text,
            "git_tree": tree,
            "include": list(requested),
            "files": files,
        }
        envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
        temporary_manifest.write_text(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
        )
        verification = verify_exact_commit_capsule(staging, temporary_manifest)
        staging.replace(capsule_root)
        temporary_manifest.replace(manifest_path)
        return verification
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        temporary_manifest.unlink(missing_ok=True)
        raise


__all__ = [
    "CAPSULE_MANIFEST_SCHEMA_VERSION",
    "CONTRACT_RELATIVE_PATH",
    "EXPECTED_CELLS",
    "EXPECTED_CONTROLLER",
    "EXPECTED_EXPERTS",
    "FORBIDDEN_RUNTIME_KEYS",
    "PREPARED_STATUS",
    "RUNTIME_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "assert_scored_launch_blocked",
    "build_exact_commit_capsule",
    "cell_runtime_payload",
    "payload_identity",
    "validate_preparation_contract",
    "verify_exact_commit_capsule",
]
