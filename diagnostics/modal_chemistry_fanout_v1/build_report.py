"""Aggregate the measured evidence files into diagnostics/modal_chemistry_fanout_v1.json.

Every number is READ from an evidence artifact written by the harness or the local
driver; nothing is typed in by hand except the prose and the Modal price rates, which
carry their own provenance field.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/rmaganti/compose_rgm_git/.worktrees/modal-chemistry-fanout-v1")
EV = ROOT / ".evidence"
sys.path.insert(0, str(ROOT))
from modal_apps import zero_oracle_chemistry_workloads as w  # noqa: E402

#: Modal published on-demand rates. Recorded so the arithmetic is checkable; confirm
#: against the workspace billing page before quoting the dollar figure externally.
CPU_CORE_SECOND_USD = 0.0000131
GIB_SECOND_USD = 0.00000222
UNIT_CPU_CORES = 1.0
UNIT_MEMORY_GIB = 4.0
CONTAINER_SECOND_USD = (
    UNIT_CPU_CORES * CPU_CORE_SECOND_USD + UNIT_MEMORY_GIB * GIB_SECOND_USD
)


def load(name):
    path = EV / name
    return json.loads(path.read_text()) if path.exists() else None


def artifact(payload):
    if payload is None:
        return None
    return payload.get("artifact", payload) if isinstance(payload, dict) else payload


def commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, check=False)
        return r.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def main(out_path: str) -> None:
    drifted_base = load("local_baseline_b600.json")
    drifted_dec = load("local_decompose_b600.json")
    pinned_base = load("local_baseline_pinned_b600.json")
    modal_arm = load("modal_merged_b600.json")
    modal_cell = load("modal_merged_cellgran_b600.json")
    modal_arm_manifest = load("modal_merged_b600.manifest.json")
    modal_cell_manifest = load("modal_merged_cellgran_b600.manifest.json")
    local_parity = load("local_parity.json")
    modal_parity = load("modal_parity.json")
    local_trace = load("local_trace.json")
    modal_trace = load("modal_trace.json")
    grid_manifest = load("modal_grid_b6000.manifest.json")
    local_unit = load("local_unit_b6000.json")
    grid_artifact = load("modal_grid_b6000.json")
    sweep = load("saturation_sweep.json")

    sha = w.canonical_sha256
    strip = w.WORKLOADS["t4_v2_feasibility"]["strip_timing"]

    def t4_sha(payload):
        a = artifact(payload)
        return sha(strip(a)) if a is not None else None

    equivalence = {
        "method": (
            "Canonical sorted-key JSON SHA-256 of the merged artifact with the "
            "machine-dependent fields removed. The reference arm calls the PRODUCTION "
            "function scripts/t4_v2_feasibility_gate.run_cell, not a transcription of "
            "it, so a change in run_cell changes the reference."
        ),
        "timing_fields_excluded": list(w.T4_V2_TIMING_FIELDS),
        "slice": "cells fa7_0 + braf_2, budget 600, seed 1, delta 0.6, all 4 arms + 4 freezes + attribution",
        "identities": {
            "local_drifted_kernel_run_cell": t4_sha(drifted_base),
            "local_drifted_kernel_harness_decomposition": t4_sha(drifted_dec),
            "local_pinned_kernel_run_cell": t4_sha(pinned_base),
            "modal_harness_decomposition_arm_granularity": t4_sha(modal_arm),
            "modal_harness_run_cell_cell_granularity": t4_sha(modal_cell),
        },
    }
    ids = equivalence["identities"]
    equivalence["claims"] = {
        "harness_exact_under_drifted_kernel": (
            ids["local_drifted_kernel_run_cell"]
            == ids["local_drifted_kernel_harness_decomposition"]
        ),
        "harness_exact_under_pinned_kernel_on_modal": (
            ids["modal_harness_run_cell_cell_granularity"]
            == ids["modal_harness_decomposition_arm_granularity"]
        ),
        "laptop_equals_modal_under_matched_kernels": (
            ids["local_pinned_kernel_run_cell"]
            == ids["modal_harness_decomposition_arm_granularity"]
        ),
        "laptop_default_venv_equals_modal": (
            ids["local_drifted_kernel_run_cell"]
            == ids["modal_harness_decomposition_arm_granularity"]
        ),
    }

    parity = {
        "local_parity_sha256": (artifact(local_parity) or {}).get("parity_sha256"),
        "modal_parity_sha256": (artifact(modal_parity) or {}).get("parity_sha256"),
        "molecules_compared": (artifact(local_parity) or {}).get("molecule_total"),
    }
    parity["identical"] = (
        parity["local_parity_sha256"] is not None
        and parity["local_parity_sha256"] == parity["modal_parity_sha256"]
    )

    payload = {
        "schema_version": "modal_chemistry_fanout_v1",
        "source_commit": commit(),
        "oracle_calls": 0,
        "docking_calls": 0,
        "equivalence": equivalence,
        "chemistry_kernel_parity": parity,
        "cost_model": {
            "cpu_core_second_usd": CPU_CORE_SECOND_USD,
            "gib_second_usd": GIB_SECOND_USD,
            "unit_cpu_cores": UNIT_CPU_CORES,
            "unit_memory_gib": UNIT_MEMORY_GIB,
            "container_second_usd": round(CONTAINER_SECOND_USD, 9),
            "provenance": (
                "Modal published on-demand CPU and memory rates; the arithmetic is "
                "recorded so the figure can be re-derived against the workspace's own "
                "billing page."
            ),
        },
        "raw": {
            "modal_arm_manifest": modal_arm_manifest,
            "modal_cell_manifest": modal_cell_manifest,
            "grid_manifest": grid_manifest,
            "local_timings": {
                "drifted_baseline": (drifted_base or {}).get("timing"),
                "drifted_decomposition": (drifted_dec or {}).get("timing"),
                "pinned_baseline": (pinned_base or {}).get("timing"),
                "pinned_single_unit_budget_6000": (local_unit or {}).get("timing"),
            },
            "saturation_sweep": sweep,
            "grid_artifact_complete": (grid_artifact or {}).get("complete"),
            "grid_artifact_missing_units": (grid_artifact or {}).get("missing_units"),
            "trajectory_divergence": None,
        },
    }

    if local_trace and modal_trace:
        L = artifact(local_trace)["cells"]["fa7_0"]["v2_moveset_only"]["trace"]
        R = artifact(modal_trace)["cells"]["fa7_0"]["v2_moveset_only"]["trace"]
        first = next((i for i in range(min(len(L), len(R))) if L[i] != R[i]), None)
        payload["raw"]["trajectory_divergence"] = {
            "arm": "fa7_0 / v2_moveset_only, budget 600, seed 1",
            "identical_prefix_length": first,
            "total_evaluations": min(len(L), len(R)),
            "local_smiles": L[first]["smiles"] if first is not None else None,
            "modal_smiles": R[first]["smiles"] if first is not None else None,
            "local_sa": L[first]["sa"] if first is not None else None,
            "modal_sa": R[first]["sa"] if first is not None else None,
        }

    # Production-budget equivalence: the ONE unit measured locally at budget 6000
    # against the same unit inside the Modal grid.
    units_dir = Path(EV / "grid_v3_units")
    if units_dir.is_dir() and local_unit:
        rows_by = {}
        for path in units_dir.rglob("*.json"):
            row = json.loads(path.read_text())
            # component MUST be in the key: freeze units all carry arm=None, so a
            # (cell, arm, kind) key silently collapsed 48 freezes onto 12 and
            # under-reported the grid by a third.
            key = (row.get("cell"), row.get("arm"), row.get("component"), row.get("kind"))
            rows_by[key] = row
        modal_row = rows_by.get(("fa7_0", "v2_full", None, "arm"))
        local_arm = (artifact(local_unit) or {}).get("cells", {}).get("fa7_0", {}).get(
            "arms", {}).get("v2_full")
        if modal_row and local_arm and modal_row.get("payload"):
            modal_arm_payload = modal_row["payload"]["summary"]
            payload_ok = sha(local_arm) == sha(modal_arm_payload)
            payload_block = {
                "unit": "fa7_0 / v2_full, budget 6000, seed 1",
                "local_pinned_sha256": sha(local_arm),
                "modal_sha256": sha(modal_arm_payload),
                "identical": payload_ok,
                "local_wall_seconds": (local_unit or {}).get("timing", {}).get(
                    "wall_seconds"),
                "modal_wall_seconds": modal_row.get("wall_seconds"),
            }
        else:
            payload_block = {"note": "unit not found in the downloaded grid rows"}
        wall = [float(r["wall_seconds"]) for r in rows_by.values() if r.get("ok", True)]
        payload_block["grid_unit_seconds_total"] = round(sum(
            float(r["wall_seconds"]) for r in rows_by.values()), 1)
        payload_block["grid_units"] = len(rows_by)
        payload_block["grid_containers"] = len({
            r.get("container_uuid") for r in rows_by.values() if r.get("container_uuid")})
        payload_block["grid_unit_seconds_mean"] = (
            round(sum(wall) / len(wall), 1) if wall else None)
        payload_block["grid_unit_seconds_max"] = round(max(wall), 1) if wall else None
        local_w = (local_unit or {}).get("timing", {}).get("wall_seconds")
        modal_w = modal_row.get("wall_seconds") if modal_row else None
        if local_w and modal_w:
            ratio = float(local_w) / float(modal_w)
            payload_block["laptop_over_modal_per_unit_ratio"] = round(ratio, 2)
            payload_block["projected_local_serial_seconds"] = round(
                payload_block["grid_unit_seconds_total"] * ratio, 0)
            payload_block["projection_basis"] = (
                "One unit measured on both sides (fa7_0 / v2_full, budget 6000): "
                f"{local_w}s on the laptop under the pinned kernel at load ~10-12, "
                f"{modal_w}s in a Modal container. The serial-local figure is that "
                "ratio applied to the measured grid container-seconds -- a PROJECTION "
                "from one paired unit, not a measured serial run, and the laptop "
                "carried a load of 200-265 for most of this session, where it would be "
                "far worse."
            )
        payload_block["grid_failed_units"] = [
            {"cell": r.get("cell"), "arm": r.get("arm"), "component": r.get("component"),
             "kind": r.get("kind"), "error_type": r.get("error_type"),
             "error": (r.get("error") or "")[:300]}
            for r in rows_by.values() if not r.get("ok", True)
        ]
        payload["production_budget_equivalence"] = payload_block

    extra = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
    payload.update(extra)
    Path(out_path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload["equivalence"]["claims"], indent=1, sort_keys=True))
    print("parity:", json.dumps(parity, sort_keys=True))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
