"""Filtered Modal surface for the bounded editing T1 capacity diagnostic.

Nothing launches on import.  A caller must provide a fresh run label:

    MODAL_PROFILE=nitya modal run \
      modal_apps/run_editing_t1_successor_gate.py \
      --run-label editing-t1-<commit>-v1 \
      --active8-inventory active8_trace_inventory_v1/ACTIVE8_TRACE_INVENTORY.json \
      --active8-inventory-file-sha256 <sha256> \
      --families cycle_attach

That command launches exactly one ``unique_state`` / ``heads_only`` arm on an
L4.  Families must always be selected explicitly.  Panel kind and scope default
to the cheapest first rung; ``--all-scopes`` is the explicit capacity-ladder
opt-in.  Every worker starts from the exact same scratch initialization and
writes one immutable diagnostic JSON.

After every successful local invocation, the orchestrator atomically retains
the independent worker-return identities under
``results/_editing_t1_launch_receipts/<run-label>.json``.  These receipts, not
the durable result payloads, are the physical inputs to the arm-manifest
collector.

The primary stochastic-law diagnostic is selected explicitly with
``--families all_families --panel-kinds
global_repeated_state_distribution``.  Within-family repeated panels use the
separate ``within_family_repeated_state_distribution`` name and remain
secondary diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    EDITING_T1_LOCAL_ADAPTER_FAMILIES,
    EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    EDITING_T1_V4_CONTRACT_RELATIVE_PATH,
    EDITING_T1_V4_PANEL_RELATIVE_PATH,
    EditingT1LaunchAuthority,
    EditingT1RuntimeError,
    load_editing_t1_launch_authority,
    require_editing_t1_family_scope_applicable,
    validate_editing_t1_launch_authority,
    validate_t1_cache_shard_receipt,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    ACTIVE8_T1_IDENTITY_FIELDS,
    EditingT1PanelError,
    within_family_repeated_panel_families,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
RUN_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
)
SCOPES = ("heads_only", "heads_plus_local_adapter", "all")
LOCAL_ADAPTER_FAMILIES = EDITING_T1_LOCAL_ADAPTER_FAMILIES
PANEL_KINDS = (
    "unique_state",
    "global_repeated_state_distribution",
    "within_family_repeated_state_distribution",
)
GPU_CLASSES = ("L4", "A10", "A100")
GLOBAL_FAMILY_SELECTOR = "all_families"
T1_CONTRACT_RELATIVE_PATH = EDITING_T1_V4_CONTRACT_RELATIVE_PATH
T1_PANEL_RELATIVE_PATH = EDITING_T1_V4_PANEL_RELATIVE_PATH
T1_CAPACITY_CENSUS_RELATIVE_PATH = EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH
SERIALIZED_TREE_PATHS = (
    "src",
    "scripts",
    "modal_apps",
    "configs",
    "diagnostics/coherence",
)
T1_MODAL_LAUNCH_RECEIPT_SCHEMA = "compose.editing.t1_modal_launch_receipt"
T1_MODAL_LAUNCH_RECEIPT_VERSION = 1
T1_MODAL_LAUNCH_RECEIPT_STATUS = "COMPLETE_INDEPENDENT_T1_MODAL_RESULTS"
T1_MODAL_LAUNCH_RECEIPT_ROOT = ROOT / "results" / "_editing_t1_launch_receipts"
_T1_MODAL_LAUNCH_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "gate_decision",
    "run_label",
    "source_commit",
    "requested_gpu_class",
    "active8_inventory",
    "active8_inventory_file_sha256",
    "arms",
    "receipt_sha256",
}
_T1_MODAL_ARM_RECEIPT_FIELDS = {
    "family",
    "panel_kind",
    "scope",
    "result_relative_path",
    "output_sha256",
    "output_bytes",
    "result_sha256",
    "contract_sha256",
    "numeric_thresholds_sha256",
    "panel_artifact_sha256",
    "panel_selection_sha256",
    "panel_census_sha256",
    "panel_capacity_strata_sha256",
    *ACTIVE8_T1_IDENTITY_FIELDS,
    "cache_receipts",
    "requested_gpu_class",
    "observed_gpu_name",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "scripts",
        str(REMOTE_ROOT / "scripts"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "modal_apps",
        str(REMOTE_ROOT / "modal_apps"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "configs",
        str(REMOTE_ROOT / "configs"),
        copy=True,
    )
    .add_local_dir(
        ROOT / "diagnostics" / "coherence",
        str(REMOTE_ROOT / "diagnostics" / "coherence"),
        copy=True,
        ignore=("**/__pycache__/**",),
    )
)

app = modal.App("compose-v4-editing-t1-successor-gate")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _validate_run_label(run_label: str) -> None:
    if Path(run_label).name != run_label or RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError("run_label must be a safe 1-128 character artifact basename")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("T1 Modal launch receipt is not finite canonical JSON") from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_mapping(
    value: object,
    fields: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, Mapping) else sorted(fields)
        extra = sorted(set(value) - fields) if isinstance(value, Mapping) else []
        raise ValueError(f"{name} fields mismatch; missing={missing}, extra={extra}")
    return value


def _canonical_cache_receipts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("T1 Modal arm cache_receipts must be a nonempty sequence")
    receipts = []
    try:
        for raw in value:
            if not isinstance(raw, Mapping):
                raise EditingT1RuntimeError("T1 cache receipt must be a mapping")
            receipts.append(validate_t1_cache_shard_receipt(raw))
    except EditingT1RuntimeError as error:
        raise ValueError(f"T1 Modal arm has an invalid cache receipt: {error}") from error
    shard_digests = tuple(receipt.packed_shard_content_sha256 for receipt in receipts)
    if shard_digests != tuple(sorted(shard_digests)) or len(shard_digests) != len(
        set(shard_digests)
    ):
        raise ValueError("T1 Modal arm cache receipts must be unique and in canonical shard order")
    return [asdict(receipt) for receipt in receipts]


def _expected_result_relative_path(
    run_label: str,
    family: str,
    panel_kind: str,
    scope: str,
) -> str:
    return (PurePosixPath(run_label) / f"{panel_kind}.{family}.{scope}.json").as_posix()


def validate_t1_modal_launch_receipt(
    value: object,
) -> Mapping[str, object]:
    """Validate one independently retained local Modal invocation receipt."""

    receipt = _require_exact_mapping(
        value,
        _T1_MODAL_LAUNCH_RECEIPT_FIELDS,
        name="T1 Modal launch receipt",
    )
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        receipt["schema"] != T1_MODAL_LAUNCH_RECEIPT_SCHEMA
        or receipt["schema_version"] != T1_MODAL_LAUNCH_RECEIPT_VERSION
        or receipt["status"] != T1_MODAL_LAUNCH_RECEIPT_STATUS
        or receipt["training_authorized"] is not False
        or receipt["gate_decision"] is not None
        or not _is_sha256(receipt["receipt_sha256"])
        or receipt["receipt_sha256"] != _stable_sha256(body)
    ):
        raise ValueError("T1 Modal launch receipt identity, status, or self-hash is invalid")
    run_label = receipt["run_label"]
    if not isinstance(run_label, str):
        raise ValueError("T1 Modal launch receipt run_label must be a string")
    _validate_run_label(run_label)
    if (
        not isinstance(receipt["source_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", receipt["source_commit"]) is None
        or receipt["requested_gpu_class"] not in GPU_CLASSES
        or not isinstance(receipt["active8_inventory"], str)
        or not receipt["active8_inventory"]
        or not _is_sha256(receipt["active8_inventory_file_sha256"])
    ):
        raise ValueError("T1 Modal launch receipt source, GPU, or Active8 identity is invalid")
    arms = receipt["arms"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("T1 Modal launch receipt must retain at least one returned arm")
    observed_keys: set[tuple[str, str, str]] = set()
    for raw_arm in arms:
        arm = _require_exact_mapping(
            raw_arm,
            _T1_MODAL_ARM_RECEIPT_FIELDS,
            name="T1 Modal arm receipt",
        )
        family = arm["family"]
        panel_kind = arm["panel_kind"]
        scope = arm["scope"]
        if (
            not isinstance(family, str)
            or not isinstance(panel_kind, str)
            or not isinstance(scope, str)
            or family not in (*FAMILIES, GLOBAL_FAMILY_SELECTOR)
            or panel_kind not in PANEL_KINDS
            or scope not in SCOPES
        ):
            raise ValueError("T1 Modal arm receipt identity is off-contract")
        try:
            require_editing_t1_family_scope_applicable(family, scope)
        except EditingT1RuntimeError as error:
            raise ValueError(str(error)) from error
        if (family == GLOBAL_FAMILY_SELECTOR) != (
            panel_kind == "global_repeated_state_distribution"
        ):
            raise ValueError("T1 Modal arm receipt has an invalid family/panel relationship")
        key = (family, panel_kind, scope)
        if key in observed_keys:
            raise ValueError("T1 Modal launch receipt contains a duplicate arm")
        observed_keys.add(key)
        expected_relative = _expected_result_relative_path(
            run_label,
            family,
            panel_kind,
            scope,
        )
        if arm["result_relative_path"] != expected_relative:
            raise ValueError(
                "T1 Modal arm result path is noncanonical or escapes its run directory"
            )
        for field in (
            "output_sha256",
            "result_sha256",
            "contract_sha256",
            "numeric_thresholds_sha256",
            "panel_artifact_sha256",
            "panel_selection_sha256",
            "panel_census_sha256",
            "panel_capacity_strata_sha256",
            *ACTIVE8_T1_IDENTITY_FIELDS,
        ):
            if not _is_sha256(arm[field]):
                raise ValueError(f"T1 Modal arm receipt {field} must be a SHA-256")
        if (
            type(arm["output_bytes"]) is not int
            or arm["output_bytes"] <= 0
            or arm["requested_gpu_class"] != receipt["requested_gpu_class"]
            or not isinstance(arm["observed_gpu_name"], str)
            or not arm["observed_gpu_name"]
            or arm["active8_inventory_manifest_file_sha256"]
            != receipt["active8_inventory_file_sha256"]
        ):
            raise ValueError(
                "T1 Modal arm receipt physical output, GPU, or Active8 binding is invalid"
            )
        canonical_receipts = _canonical_cache_receipts(arm["cache_receipts"])
        if arm["cache_receipts"] != canonical_receipts:
            raise ValueError("T1 Modal arm cache receipts are not canonically serialized")
    return receipt


def build_t1_modal_launch_receipt(
    *,
    run_label: str,
    source_commit: str,
    requested_gpu_class: str,
    tasks: Sequence[Sequence[str]],
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Normalize returned worker identities without reading result payload files."""

    _validate_run_label(run_label)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("T1 source_commit must be a full lowercase Git commit")
    if requested_gpu_class not in GPU_CLASSES:
        raise ValueError("unknown T1 requested GPU class")
    if not tasks or len(tasks) != len(results):
        raise ValueError("T1 Modal invocation must retain one returned result for every task")
    normalized_arms: list[dict[str, object]] = []
    active8_inventory: str | None = None
    active8_file_sha256: str | None = None
    for task, result in zip(tasks, results, strict=True):
        if len(task) != 6 or not isinstance(result, Mapping):
            raise ValueError("T1 Modal task/result receipt shape is invalid")
        (
            task_run_label,
            family,
            panel_kind,
            scope,
            task_active8_inventory,
            task_active8_file_sha256,
        ) = task
        key = (family, panel_kind, scope)
        if task_run_label != run_label or (
            result.get("run_label"),
            result.get("family"),
            result.get("panel_kind"),
            result.get("scope"),
        ) != (run_label, *key):
            raise ValueError("T1 Modal returned arm identity disagrees with its task")
        if result.get("requested_gpu_class") != requested_gpu_class:
            raise ValueError("T1 Modal returned GPU class disagrees with its invocation")
        if result.get("active8_inventory_manifest_file_sha256") != (task_active8_file_sha256):
            raise ValueError("T1 Modal returned Active8 identity disagrees with its task")
        expected_output = (
            ARTIFACT_ROOT
            / "_editing_t1_successor"
            / run_label
            / f"{panel_kind}.{family}.{scope}.json"
        )
        if result.get("output") != str(expected_output):
            raise ValueError("T1 Modal returned result path is noncanonical")
        if active8_inventory is None:
            active8_inventory = task_active8_inventory
            active8_file_sha256 = task_active8_file_sha256
        elif (
            task_active8_inventory != active8_inventory
            or task_active8_file_sha256 != active8_file_sha256
        ):
            raise ValueError("T1 Modal invocation mixed Active8 source identities")
        normalized_arms.append(
            {
                "family": family,
                "panel_kind": panel_kind,
                "scope": scope,
                "result_relative_path": _expected_result_relative_path(
                    run_label,
                    family,
                    panel_kind,
                    scope,
                ),
                "output_sha256": result.get("output_sha256"),
                "output_bytes": result.get("output_bytes"),
                "result_sha256": result.get("result_sha256"),
                "contract_sha256": result.get("contract_sha256"),
                "numeric_thresholds_sha256": result.get("numeric_thresholds_sha256"),
                "panel_artifact_sha256": result.get("panel_artifact_sha256"),
                "panel_selection_sha256": result.get("panel_selection_sha256"),
                "panel_census_sha256": result.get("panel_census_sha256"),
                "panel_capacity_strata_sha256": result.get("panel_capacity_strata_sha256"),
                **{field: result.get(field) for field in ACTIVE8_T1_IDENTITY_FIELDS},
                "cache_receipts": result.get("cache_receipts"),
                "requested_gpu_class": requested_gpu_class,
                "observed_gpu_name": result.get("observed_gpu_name"),
            }
        )
    assert active8_inventory is not None
    assert active8_file_sha256 is not None
    body: dict[str, object] = {
        "schema": T1_MODAL_LAUNCH_RECEIPT_SCHEMA,
        "schema_version": T1_MODAL_LAUNCH_RECEIPT_VERSION,
        "status": T1_MODAL_LAUNCH_RECEIPT_STATUS,
        "training_authorized": False,
        "gate_decision": None,
        "run_label": run_label,
        "source_commit": source_commit,
        "requested_gpu_class": requested_gpu_class,
        "active8_inventory": active8_inventory,
        "active8_inventory_file_sha256": active8_file_sha256,
        "arms": normalized_arms,
    }
    sealed = {**body, "receipt_sha256": _stable_sha256(body)}
    validate_t1_modal_launch_receipt(sealed)
    return sealed


def write_t1_modal_launch_receipt(
    receipt: Mapping[str, object],
    path: Path,
) -> None:
    """Publish one immutable receipt atomically after every successful invocation."""

    validate_t1_modal_launch_receipt(receipt)
    content = (
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"immutable T1 launch receipt already differs: {destination}"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_t1_modal_launch_receipt(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"T1 Modal launch receipt is unreadable: {path}") from error
    return validate_t1_modal_launch_receipt(payload)


def t1_modal_launch_receipt_path(run_label: str) -> Path:
    _validate_run_label(run_label)
    return T1_MODAL_LAUNCH_RECEIPT_ROOT / f"{run_label}.json"


def _require_clean_serialized_tree() -> str:
    """Return the exact source revision or reject a dirty Modal image input."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--",
                *SERIALIZED_TREE_PATHS,
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify the T1 Modal source revision") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("T1 Modal source revision is not a full Git commit")
    if dirty:
        raise RuntimeError(f"refusing to launch T1 from a dirty serialized-code tree: {dirty[:12]}")
    return commit


def _resolve_active8_launch_binding(
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> tuple[str, str]:
    if len(active8_inventory_file_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in active8_inventory_file_sha256
    ):
        raise ValueError("active8_inventory_file_sha256 must be a lowercase SHA-256")
    if not active8_inventory.strip():
        raise ValueError("active8_inventory must be selected explicitly")
    candidate = Path(active8_inventory)
    if not candidate.is_absolute():
        candidate = ARTIFACT_ROOT / candidate
    normalized = candidate.resolve(strict=False)
    try:
        normalized.relative_to(ARTIFACT_ROOT)
    except ValueError:
        raise ValueError(
            "active8_inventory must resolve inside the mounted artifact volume"
        ) from None
    return str(normalized), active8_inventory_file_sha256


def _editing_t1_worker_command(
    *,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    family: str,
    panel_kind: str,
    scope: str,
    output: Path,
) -> list[str]:
    """Name every serialized V4 authority artifact explicitly for the worker."""

    return [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "run_editing_t1_successor_gate.py"),
        "--transfer-root",
        str(ARTIFACT_ROOT),
        "--t1-contract",
        str(REMOTE_ROOT / T1_CONTRACT_RELATIVE_PATH),
        "--panel",
        str(REMOTE_ROOT / T1_PANEL_RELATIVE_PATH),
        "--capacity-census",
        str(REMOTE_ROOT / T1_CAPACITY_CENSUS_RELATIVE_PATH),
        "--active8-inventory",
        active8_inventory,
        "--active8-inventory-file-sha256",
        active8_inventory_file_sha256,
        "--family",
        family,
        "--panel-kind",
        panel_kind,
        "--scope",
        scope,
        "--device",
        "cuda",
        "--output",
        str(output),
    ]


def _resolve_selection(
    raw: str,
    allowed: tuple[str, ...],
    *,
    label: str,
    default: str | None,
    select_all: bool,
    all_selection: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if select_all:
        if raw.strip():
            raise ValueError(f"{label} cannot be combined with --all-{label.replace('_', '-')}")
        return allowed if all_selection is None else all_selection
    if not raw.strip():
        if default is None:
            raise ValueError(f"{label} must be selected explicitly")
        return (default,)
    selected = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not selected:
        raise ValueError(f"{label} selection is empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label} selection contains duplicates")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return selected


def build_task_matrix(
    run_label: str,
    *,
    families: str,
    panel_kinds: str = "",
    scopes: str = "",
    all_families: bool = False,
    all_scopes: bool = False,
    active8_inventory: str = "",
    active8_inventory_file_sha256: str = "",
    launch_authority: EditingT1LaunchAuthority | None = None,
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    """Build an explicit valid arm matrix without silently broadening it."""

    _validate_run_label(run_label)
    active8_path, active8_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory,
        active8_inventory_file_sha256,
    )
    selected_families = _resolve_selection(
        families,
        (*FAMILIES, GLOBAL_FAMILY_SELECTOR),
        label="families",
        default=None,
        select_all=all_families,
        all_selection=FAMILIES,
    )
    selected_panel_kinds = _resolve_selection(
        panel_kinds,
        PANEL_KINDS,
        label="panel_kinds",
        default="unique_state",
        select_all=False,
    )
    selected_scopes = _resolve_selection(
        scopes,
        SCOPES,
        label="scopes",
        default="heads_only",
        select_all=all_scopes,
    )
    if launch_authority is None:
        try:
            authority = load_editing_t1_launch_authority(
                contract_path=ROOT / T1_CONTRACT_RELATIVE_PATH,
                panel_path=ROOT / T1_PANEL_RELATIVE_PATH,
                capacity_census_path=ROOT / T1_CAPACITY_CENSUS_RELATIVE_PATH,
                expected_active8_inventory_manifest_file_sha256=(
                    active8_file_sha256
                ),
            )
        except EditingT1RuntimeError as error:
            raise ValueError(str(error)) from error
    else:
        try:
            authority = validate_editing_t1_launch_authority(
                contract=launch_authority.contract,
                panel=launch_authority.panel,
                capacity_census=launch_authority.capacity_census,
                capacity_census_file_sha256=(
                    launch_authority.capacity_census_file_sha256
                ),
                expected_active8_inventory_manifest_file_sha256=(
                    active8_file_sha256
                ),
            )
        except EditingT1RuntimeError as error:
            raise ValueError(str(error)) from error
    try:
        repeated_families = within_family_repeated_panel_families(authority.panel)
    except EditingT1PanelError as error:
        raise ValueError(str(error)) from error
    global_requested = "global_repeated_state_distribution" in selected_panel_kinds
    global_selected = GLOBAL_FAMILY_SELECTOR in selected_families
    if global_requested or global_selected:
        if selected_panel_kinds != ("global_repeated_state_distribution",) or (
            selected_families != (GLOBAL_FAMILY_SELECTOR,)
        ):
            raise ValueError(
                "the primary global repeated-state panel requires exactly "
                "--families all_families --panel-kinds "
                "global_repeated_state_distribution"
            )
    if "within_family_repeated_state_distribution" in selected_panel_kinds:
        unsupported = sorted(set(selected_families) - set(repeated_families))
        if unsupported:
            raise ValueError(
                "no empirical within-family repeated multi-successor panel exists for "
                f"selected families: {unsupported}"
            )
    tasks: list[tuple[str, str, str, str, str, str]] = []
    for family in selected_families:
        for panel_kind in selected_panel_kinds:
            for scope in selected_scopes:
                try:
                    require_editing_t1_family_scope_applicable(
                        family,
                        scope,
                    )
                except EditingT1RuntimeError as error:
                    if (
                        all_scopes
                        and scope == "heads_plus_local_adapter"
                        and family != GLOBAL_FAMILY_SELECTOR
                        and family not in LOCAL_ADAPTER_FAMILIES
                    ):
                        continue
                    raise ValueError(str(error)) from error
                tasks.append(
                    (
                        run_label,
                        family,
                        panel_kind,
                        scope,
                        active8_path,
                        active8_file_sha256,
                    )
                )
    return tuple(tasks)


def _run_arm_impl(
    run_label: str,
    family: str,
    panel_kind: str,
    scope: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    *,
    requested_gpu_class: str,
) -> dict[str, object]:
    """Execute one immutable T1 arm inside an already provisioned worker."""

    _validate_run_label(run_label)
    active8_inventory, active8_inventory_file_sha256 = _resolve_active8_launch_binding(
        active8_inventory,
        active8_inventory_file_sha256,
    )
    try:
        require_editing_t1_family_scope_applicable(family, scope)
    except EditingT1RuntimeError as error:
        raise ValueError(str(error)) from error
    if panel_kind not in PANEL_KINDS:
        raise ValueError("unknown T1 panel kind")
    if requested_gpu_class not in GPU_CLASSES:
        raise ValueError("unknown T1 GPU class")
    try:
        launch_authority = load_editing_t1_launch_authority(
            contract_path=REMOTE_ROOT / T1_CONTRACT_RELATIVE_PATH,
            panel_path=REMOTE_ROOT / T1_PANEL_RELATIVE_PATH,
            capacity_census_path=(
                REMOTE_ROOT / T1_CAPACITY_CENSUS_RELATIVE_PATH
            ),
            expected_active8_inventory_manifest_file_sha256=(
                active8_inventory_file_sha256
            ),
        )
    except EditingT1RuntimeError as error:
        raise RuntimeError(str(error)) from error
    if (family == GLOBAL_FAMILY_SELECTOR) != (panel_kind == "global_repeated_state_distribution"):
        raise ValueError("all_families is reserved for the primary global repeated-state panel")
    if panel_kind == "within_family_repeated_state_distribution" and (
        family
        not in within_family_repeated_panel_families(launch_authority.panel)
    ):
        raise ValueError(
            f"no empirical within-family repeated multi-successor panel exists for {family}"
        )
    output_directory = ARTIFACT_ROOT / "_editing_t1_successor" / run_label
    output = output_directory / f"{panel_kind}.{family}.{scope}.json"
    if output.exists():
        raise FileExistsError(
            f"T1 output already exists; choose a fresh run label instead of reusing {output}"
        )
    command = _editing_t1_worker_command(
        active8_inventory=active8_inventory,
        active8_inventory_file_sha256=active8_inventory_file_sha256,
        family=family,
        panel_kind=panel_kind,
        scope=scope,
        output=output,
    )
    completed = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    encoded = output.read_bytes()
    payload = json.loads(encoded)
    expected_result_bindings = {
        "contract_sha256": launch_authority.contract.sha256,
        "numeric_thresholds_sha256": (
            launch_authority.contract.numeric_thresholds_sha256
        ),
        "panel_artifact_sha256": launch_authority.contract.payload[
            "panel_artifact_sha256"
        ],
        "panel_selection_sha256": launch_authority.contract.payload[
            "panel_selection_sha256"
        ],
        "panel_census_sha256": launch_authority.contract.payload[
            "panel_census_sha256"
        ],
        "panel_capacity_strata_sha256": launch_authority.contract.payload[
            "panel_capacity_strata_sha256"
        ],
        **dict(launch_authority.active8_identity),
    }
    if (
        payload.get("training_authorized") is not False
        or payload.get("gate_decision") is not None
        or payload.get("numeric_thresholds_frozen") is not True
        or not isinstance(payload.get("numeric_thresholds_sha256"), str)
        or not isinstance(payload.get("result_sha256"), str)
        or not isinstance(payload.get("cache_receipts"), list)
        or not payload["cache_receipts"]
        or any(
            payload.get(field) != expected
            for field, expected in expected_result_bindings.items()
        )
    ):
        raise RuntimeError("T1 worker output is not a sealed decision-free durable result")
    artifact_volume.commit()
    return {
        "run_label": run_label,
        "family": family,
        "panel_kind": panel_kind,
        "scope": scope,
        "requested_gpu_class": requested_gpu_class,
        "observed_gpu_name": gpu_name,
        "output": str(output),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "result_sha256": payload["result_sha256"],
        "contract_sha256": payload["contract_sha256"],
        "numeric_thresholds_sha256": payload["numeric_thresholds_sha256"],
        "panel_artifact_sha256": payload["panel_artifact_sha256"],
        "panel_selection_sha256": payload["panel_selection_sha256"],
        "panel_census_sha256": payload["panel_census_sha256"],
        "panel_capacity_strata_sha256": payload["panel_capacity_strata_sha256"],
        "active8_inventory_manifest_file_sha256": (
            payload["active8_inventory_manifest_file_sha256"]
        ),
        "active8_inventory_sha256": payload["active8_inventory_sha256"],
        "active8_effective_source_corpus_cache_sha256": (
            payload["active8_effective_source_corpus_cache_sha256"]
        ),
        "active8_unified_packed_manifest_sha256": (
            payload["active8_unified_packed_manifest_sha256"]
        ),
        "active8_support_contract_sha256": (payload["active8_support_contract_sha256"]),
        "cache_receipts": payload["cache_receipts"],
        "worker_stdout_tail": completed.stdout[-1000:],
        "training_authorized": False,
        "gate_decision": None,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=7200,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_arm_l4(
    run_label: str,
    family: str,
    panel_kind: str,
    scope: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        panel_kind,
        scope,
        active8_inventory,
        active8_inventory_file_sha256,
        requested_gpu_class="L4",
    )


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=32768,
    timeout=7200,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_arm_a10(
    run_label: str,
    family: str,
    panel_kind: str,
    scope: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        panel_kind,
        scope,
        active8_inventory,
        active8_inventory_file_sha256,
        requested_gpu_class="A10",
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=4,
    memory=32768,
    timeout=7200,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_arm_a100(
    run_label: str,
    family: str,
    panel_kind: str,
    scope: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        panel_kind,
        scope,
        active8_inventory,
        active8_inventory_file_sha256,
        requested_gpu_class="A100",
    )


def _runner_for_gpu(gpu_class: str) -> modal.Function:
    try:
        return {
            "L4": run_arm_l4,
            "A10": run_arm_a10,
            "A100": run_arm_a100,
        }[gpu_class]
    except KeyError as error:
        raise ValueError(
            f"gpu_class must be one of {list(GPU_CLASSES)}, got {gpu_class!r}"
        ) from error


@app.local_entrypoint()
def main(
    run_label: str,
    active8_inventory: str,
    active8_inventory_file_sha256: str,
    families: str = "",
    panel_kinds: str = "",
    scopes: str = "",
    all_families: bool = False,
    all_scopes: bool = False,
    gpu_class: str = "L4",
) -> None:
    source_commit = _require_clean_serialized_tree()
    tasks = build_task_matrix(
        run_label,
        families=families,
        panel_kinds=panel_kinds,
        scopes=scopes,
        all_families=all_families,
        all_scopes=all_scopes,
        active8_inventory=active8_inventory,
        active8_inventory_file_sha256=active8_inventory_file_sha256,
    )
    runner = _runner_for_gpu(gpu_class)
    results = list(runner.starmap(tasks))
    launch_receipt = build_t1_modal_launch_receipt(
        run_label=run_label,
        source_commit=source_commit,
        requested_gpu_class=gpu_class,
        tasks=tasks,
        results=results,
    )
    launch_receipt_path = t1_modal_launch_receipt_path(run_label)
    write_t1_modal_launch_receipt(
        launch_receipt,
        launch_receipt_path,
    )
    print(
        json.dumps(
            {
                "run_label": run_label,
                "source_commit": source_commit,
                "arm_count": len(results),
                "gpu_class": gpu_class,
                "active8_inventory": active8_inventory,
                "active8_inventory_file_sha256": (active8_inventory_file_sha256),
                "tasks": [
                    {
                        "family": family,
                        "panel_kind": panel_kind,
                        "scope": scope,
                    }
                    for (
                        _,
                        family,
                        panel_kind,
                        scope,
                        _,
                        _,
                    ) in tasks
                ],
                "results": results,
                "launch_receipt": str(launch_receipt_path),
                "launch_receipt_sha256": launch_receipt["receipt_sha256"],
                "training_authorized": False,
                "gate_decision": None,
            },
            indent=2,
            sort_keys=True,
        )
    )
