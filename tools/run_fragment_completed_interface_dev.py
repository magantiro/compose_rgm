"""Matched, development-only assay of completed-interface redirection.

The fixed protocol and reject rule are in
docs/FRAGMENT_COMPLETED_INTERFACE_REDIRECTION_DEV_2026-09-24.md.
Each prompt/arm unit is atomic and independently resumable; no held prompt is
used to choose this controller mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import networkx
import numpy as np
import pandas
import scipy
import torch
from fetch_official_fragment_evaluator import verify_only
from rdkit import rdBase
from run_fragment_constrained_suite import (
    MANIFEST,
    _kernel_provenance,
    frozen_attachment_identity,
    frozen_sampler_identity,
    run_task,
)

from compose_v4.benchmark.fragment_attachment_control import (
    REDIRECT_COMPLETED_INTERFACES,
    AttachmentControlConfig,
)
from compose_v4.benchmark.fragment_conditioned_sampler import SamplerConfig
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
OUTPUT = Path("diagnostics/fragment_completed_interface_redirection_dev_v1")
CONTRACT = Path("docs/FRAGMENT_COMPLETED_INTERFACE_REDIRECTION_DEV_2026-09-24.md")
EXPECTED_INPUT_SHA256 = {
    CHECKPOINT: "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4",
    MANIFEST: "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9",
}
DECORATION_DRUGS = ("BARICITINIB", "ERLOTINIB", "LIOTHYRONINE", "MARIBAVIR")
CONTROL_PROMPTS = (
    (FragmentTask.MOTIF_EXTENSION, "BARICITINIB"),
    (FragmentTask.SUPERSTRUCTURE_GENERATION, "LOVASTATIN"),
)
PROMPTS = (
    tuple((FragmentTask.SCAFFOLD_DECORATION, drug) for drug in DECORATION_DRUGS) + CONTROL_PROMPTS
)
ARMS = {
    "historical": AttachmentControlConfig(enabled=True),
    "completed_site": AttachmentControlConfig(
        enabled=True, redirect_attachment=REDIRECT_COMPLETED_INTERFACES
    ),
}
SAMPLER = SamplerConfig()
SAMPLES = 20
SEED = 0
IMPLEMENTATION_FILES = (
    "src/compose_v4/benchmark/fragment_attachment_control.py",
    "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
    "src/compose_v4/benchmark/fragment_constrained.py",
    "src/compose_v4/benchmark/fragment_official_metrics.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "tools/run_fragment_constrained_suite.py",
    "tools/run_fragment_completed_interface_dev.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _identity() -> dict:
    inputs = {}
    for path, expected in EXPECTED_INPUT_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"input hash mismatch: {path}: {actual} != {expected}")
        inputs[str(path)] = actual
    return {
        "schema_version": "fragment_completed_interface_dev_v1",
        "contract_sha256": _sha256(CONTRACT),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "inputs_sha256": inputs,
        "implementation_sha256": {name: _sha256(ROOT / name) for name in IMPLEMENTATION_FILES},
        "official_ivg_blobs_sha256": verify_only(),
        "software": {
            **_kernel_provenance(),
            "networkx": networkx.__version__,
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "hardware": {"device": "cpu", "precision": "float32"},
        "attempts_per_prompt": SAMPLES,
        "seed": SEED,
        "sampler": frozen_sampler_identity(SAMPLER),
        "prompts": [[task.value, drug] for task, drug in PROMPTS],
    }


def _unit_path(arm: str, task: FragmentTask, drug: str) -> Path:
    return OUTPUT / "units" / f"{arm}__{task.value}__{drug.lower()}.json"


def _unit_row(unit: dict, drug: str) -> dict:
    rows = unit["result"]["per_drug"].get(drug, ())
    if len(rows) != 1:
        raise RuntimeError(f"missing one result row for {drug}")
    row = rows[0]
    if row["attempts"] != SAMPLES or len(row["attempt_records"]) != SAMPLES:
        raise RuntimeError(f"attempt loss for {drug}")
    for attempt in row["attempt_records"]:
        if len(attempt["accepted_actions"]) != attempt["events"]:
            raise RuntimeError(f"missing accepted action provenance for {drug}")
    return row


def _arm_summary(units: dict, arm: str) -> dict:
    rows = [
        _unit_row(units[(arm, FragmentTask.SCAFFOLD_DECORATION.value, drug)], drug)
        for drug in DECORATION_DRUGS
    ]
    return {
        "attempts": sum(row["attempts"] for row in rows),
        "committed": sum(row["committed_endpoints"] for row in rows),
        "valid_committed": sum(row["committed_chemically_valid"] for row in rows),
        "fragment_preserving_committed": sum(row["committed_fragment_preserving"] for row in rows),
        "full_task_success": sum(row["emitted_nonempty"] for row in rows),
        "official_quality_mean": float(np.mean([row["official"]["quality"] for row in rows])),
        "official_uniqueness_mean": float(np.mean([row["official"]["uniqueness"] for row in rows])),
        "official_diversity_mean": float(np.mean([row["official"]["diversity"] for row in rows])),
        "staging_rejections": sum(row["staging_rejections"] for row in rows),
        "interface_rejections": sum(row["interface_rejections"] for row in rows),
        "lock_rejections": sum(row["lock_rejections"] for row in rows),
        "completed_site_redirect_offers": sum(
            row["completed_site_redirect_offers"] for row in rows
        ),
        "completed_site_redirects": sum(row["completed_site_redirects"] for row in rows),
        "completed_site_redirect_commits": sum(
            row["completed_site_redirect_commits"] for row in rows
        ),
        "seconds": sum(row["seconds"] for row in rows),
        "per_prompt": {
            drug: {
                "committed": row["committed_endpoints"],
                "task_success": row["emitted_nonempty"],
                "official": row["official"],
                "staging_rejections": row["staging_rejections"],
                "completed_site_redirect_commits": row["completed_site_redirect_commits"],
            }
            for drug, row in zip(DECORATION_DRUGS, rows, strict=True)
        },
    }


def main() -> None:
    identity = _identity()
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, metadata = load_factorized_rollout_checkpoint(CHECKPOINT)
    model.eval()
    system = de_novo_rewrite_system()
    prompts = load_genmol_prompts(MANIFEST)
    units = {}
    for arm, control in ARMS.items():
        for task, drug in PROMPTS:
            path = _unit_path(arm, task, drug)
            if path.exists():
                unit = json.loads(path.read_text())
                if unit.get("identity") != identity:
                    raise RuntimeError(f"existing unit identity mismatch: {path}")
            else:
                result = run_task(
                    model,
                    system,
                    prompts,
                    task,
                    seeds=1,
                    samples=SAMPLES,
                    config=SAMPLER,
                    control=control,
                    verbose=False,
                    seed_list=[SEED],
                    drugs=[drug],
                    linker_bridge_atoms=0,
                )
                unit = {
                    "identity": identity,
                    "arm": arm,
                    "attachment_control": frozen_attachment_identity(control),
                    "checkpoint_metadata": {
                        "completed_steps": metadata.get("completed_steps"),
                        "corpus_scope_hash": metadata.get("corpus_scope_hash"),
                    },
                    "task": task.value,
                    "drug": drug,
                    "result": result,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _unit_row(unit, drug)
                _atomic_json(path, unit)
                print(f"saved {path}", flush=True)
            units[(arm, task.value, drug)] = unit

    control_identity = {}
    for task, drug in CONTROL_PROMPTS:
        old = _unit_row(units[("historical", task.value, drug)], drug)
        new = _unit_row(units[("completed_site", task.value, drug)], drug)
        control_identity[f"{task.value}/{drug}"] = old["attempt_records"] == new["attempt_records"]
    old = _arm_summary(units, "historical")
    new = _arm_summary(units, "completed_site")
    checks = {
        "negative_controls_identical": all(control_identity.values()),
        "all_committed_chemically_valid": all(
            row["committed"] == row["valid_committed"] for row in (old, new)
        ),
        "mechanism_accepted_at_least_five": new["completed_site_redirect_commits"] >= 5,
        "staging_rejections_down_at_least_ten_percent": (
            new["staging_rejections"] <= 0.9 * old["staging_rejections"]
        ),
        "full_success_lost_at_most_two": (new["full_task_success"] >= old["full_task_success"] - 2),
        "quality_drop_at_most_five_points": (
            new["official_quality_mean"] >= old["official_quality_mean"] - 0.05
        ),
        "diversity_drop_at_most_point_zero_three": (
            new["official_diversity_mean"] >= old["official_diversity_mean"] - 0.03
        ),
    }
    summary = {
        "schema_version": "fragment_completed_interface_dev_summary_v1",
        "identity": identity,
        "control_identity": control_identity,
        "arms": {"historical": old, "completed_site": new},
        "checks": checks,
        "qualified_for_held_test": all(checks.values()),
        "units_sha256": {
            str(_unit_path(arm, task, drug)): _sha256(_unit_path(arm, task, drug))
            for arm in ARMS
            for task, drug in PROMPTS
        },
    }
    _atomic_json(OUTPUT / "summary.json", summary)
    print(
        json.dumps(
            {
                "checks": checks,
                "arms": {
                    k: {x: v for x, v in row.items() if x != "per_prompt"}
                    for k, row in summary["arms"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
