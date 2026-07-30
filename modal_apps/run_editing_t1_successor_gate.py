"""Filtered Modal surface for the bounded editing T1 capacity diagnostic.

Nothing launches on import.  A caller must provide a fresh run label:

    MODAL_PROFILE=nitya modal run \
      modal_apps/run_editing_t1_successor_gate.py \
      --run-label editing-t1-<commit>-v1 \
      --families cycle_attach

That command launches exactly one ``unique_state`` / ``heads_only`` arm on an
L4.  Families must always be selected explicitly.  Panel kind and scope default
to the cheapest first rung; ``--all-scopes`` is the explicit capacity-ladder
opt-in.  Every worker starts from the exact same scratch initialization and
writes one immutable diagnostic JSON.

The primary stochastic-law diagnostic is selected explicitly with
``--families all_families --panel-kinds
global_repeated_state_distribution``.  Within-family repeated panels use the
separate ``within_family_repeated_state_distribution`` name and remain
secondary diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
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
SCOPES = ("heads_only", "heads_plus_pair_projection", "all")
PANEL_KINDS = (
    "unique_state",
    "global_repeated_state_distribution",
    "within_family_repeated_state_distribution",
)
GPU_CLASSES = ("L4", "A10", "A100")
GLOBAL_FAMILY_SELECTOR = "all_families"
WITHIN_FAMILY_REPEATED_FAMILIES = (
    "atom_insert",
    "atom_restate",
    "bond_reroute",
    "cycle_attach",
    "ring_system_restate",
)

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
) -> tuple[tuple[str, str, str, str], ...]:
    """Build an explicit valid arm matrix without silently broadening it."""

    _validate_run_label(run_label)
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
        unsupported = sorted(set(selected_families) - set(WITHIN_FAMILY_REPEATED_FAMILIES))
        if unsupported:
            raise ValueError(
                "no empirical within-family repeated multi-successor panel exists for "
                f"selected families: {unsupported}"
            )
    return tuple(
        (run_label, family, panel_kind, scope)
        for family in selected_families
        for panel_kind in selected_panel_kinds
        for scope in selected_scopes
    )


def _run_arm_impl(
    run_label: str,
    family: str,
    panel_kind: str,
    scope: str,
    *,
    requested_gpu_class: str,
) -> dict[str, object]:
    """Execute one immutable T1 arm inside an already provisioned worker."""

    _validate_run_label(run_label)
    if family not in (*FAMILIES, GLOBAL_FAMILY_SELECTOR) or scope not in SCOPES:
        raise ValueError("unknown T1 family or scope")
    if panel_kind not in PANEL_KINDS:
        raise ValueError("unknown T1 panel kind")
    if requested_gpu_class not in GPU_CLASSES:
        raise ValueError("unknown T1 GPU class")
    if (family == GLOBAL_FAMILY_SELECTOR) != (panel_kind == "global_repeated_state_distribution"):
        raise ValueError("all_families is reserved for the primary global repeated-state panel")
    if panel_kind == "within_family_repeated_state_distribution" and (
        family not in WITHIN_FAMILY_REPEATED_FAMILIES
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
    command = [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "run_editing_t1_successor_gate.py"),
        "--transfer-root",
        str(ARTIFACT_ROOT),
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
    if (
        payload.get("training_authorized") is not False
        or payload.get("gate_decision") is not None
        or not isinstance(payload.get("result_sha256"), str)
        or not isinstance(payload.get("cache_receipts"), list)
        or not payload["cache_receipts"]
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
        "panel_artifact_sha256": payload["panel_artifact_sha256"],
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
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        panel_kind,
        scope,
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
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        panel_kind,
        scope,
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
) -> dict[str, object]:
    return _run_arm_impl(
        run_label,
        family,
        panel_kind,
        scope,
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
    families: str = "",
    panel_kinds: str = "",
    scopes: str = "",
    all_families: bool = False,
    all_scopes: bool = False,
    gpu_class: str = "L4",
) -> None:
    tasks = build_task_matrix(
        run_label,
        families=families,
        panel_kinds=panel_kinds,
        scopes=scopes,
        all_families=all_families,
        all_scopes=all_scopes,
    )
    runner = _runner_for_gpu(gpu_class)
    results = list(runner.starmap(tasks))
    print(
        json.dumps(
            {
                "run_label": run_label,
                "arm_count": len(results),
                "gpu_class": gpu_class,
                "tasks": [
                    {
                        "family": family,
                        "panel_kind": panel_kind,
                        "scope": scope,
                    }
                    for _, family, panel_kind, scope in tasks
                ],
                "results": results,
                "training_authorized": False,
                "gate_decision": None,
            },
            indent=2,
            sort_keys=True,
        )
    )
