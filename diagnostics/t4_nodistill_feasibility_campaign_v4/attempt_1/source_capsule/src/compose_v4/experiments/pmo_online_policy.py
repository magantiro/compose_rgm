"""Source-bound online policy intervention using the existing complete-option loop."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.branch_policy import RECIPE
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import load_package
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import worker_identity
from compose_v4.experiments.pmo_inference_speed import law_values
from compose_v4.experiments.production_successor_kernel import _default_rewrite_system
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_online_policy"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
APP = f"modal_apps/{KIND}_app.py"
APP_NAME = "compose-pmo-online-policy"
ARMS = ["balanced", "frozen", "learned"]


def prepare(root: Path) -> dict:
    paths = [
        "diagnostics/pmo_branch_policy/prepared.json",
        "diagnostics/pmo_branch_policy/result_sealed.json",
        "diagnostics/pmo_inference_speed/result_sealed.json",
        "diagnostics/pmo_inference_speed/package_manifest.json",
        "diagnostics/pmo_inference_speed/reference_law_sealed.json",
    ]
    inputs = {p: sha256_file(root / p) for p in paths}
    prior = json.loads((root / paths[0]).read_text())
    completed = unseal(root / paths[1])
    observed, archive = dict(prior["observed"]), {r["smiles"]: r for r in prior["archive"]}
    observations = {r["id"]: r for r in prior["observations"]}
    for arm in completed["arms"].values():
        for row in arm["archive"]:
            s, score = row["smiles"], row["score"]
            if s in observed and observed[s] != score:
                raise ValueError("historical deterministic PMO scores disagree")
            observed[s] = score
            archive.setdefault(s, row)
            if "parent_smiles" in row:
                observations.setdefault(
                    row["id"],
                    {k: row[k] for k in ("id", "parent_smiles", "parent_score", "smiles", "score")},
                )
    data = {
        "schema_version": "pmo_online_prepared_v1",
        "historical_calls": prior["historical_calls"] + completed["new_oracle_calls"],
        "observed": observed,
        "archive": [archive[s] for s in sorted(archive)],
        "observations": [observations[k] for k in sorted(observations)],
        "calibration_parents": [],
        "input_hashes": inputs,
    }
    publish_json(root / PREPARED, data)
    old = json.loads((root / "configs/pmo_branch_policy.json").read_text())
    proof = unseal(root / paths[2])
    contract = {
        "schema_version": "pmo_online_contract_v1",
        "authorization": "2026-09-11 user requests principled controller changes and bounded autonomous development",
        "task": "perindopril_mpo",
        "seed": 20260917,
        "new_oracle_limit": 96,
        "evaluation_rounds": 8,
        "draws_per_bundle": 4,
        "lineages": 4,
        "arms": ARMS,
        "online_updates": True,
        "policy_recipe": RECIPE,
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {
            "path": "docs/PMO_ONLINE_POLICY.md",
            "sha256": sha256_file(root / "docs/PMO_ONLINE_POLICY.md"),
        },
        "inputs": inputs,
        "expected_input_sha256": old["expected_input_sha256"],
        "law_caches": [],
        "package": proof["export"],
        "qualification": {"path": paths[2], "sha256": inputs[paths[2]]},
        "reference_law": {"path": paths[4], "sha256": inputs[paths[4]]},
        "extra_model_sources": {
            p: sha256_file(root / p)
            for p in (
                "src/compose_v4/experiments/editing_v2_process_v2_t1_runtime.py",
                "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_runner.py",
            )
        },
        "portability": {"max_absolute_error": 1e-7, "max_half_l1": 1e-6},
        "reference_training_authorized": False,
        "docking_authorized": False,
        "winner_inputs": False,
        "prescreen": False,
        "compute": {
            "max_workers": 12,
            "worker_timeout": 1200,
            "driver_timeout": 2700,
            "retries": 0,
            "cpu": 1,
            "memory_mib": 8192,
            "cost_bound_usd": 5,
        },
    }
    contract["contract_sha256"] = identity(contract)
    publish_json(root / CONTRACT, contract)
    return contract


def load_contract(root: Path) -> dict:
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("online policy contract hash mismatch")
    if (
        c["task"],
        c["new_oracle_limit"],
        c["evaluation_rounds"],
        c["draws_per_bundle"],
        c["lineages"],
        c["arms"],
        c["online_updates"],
    ) != ("perindopril_mpo", 96, 8, 4, 4, ARMS, True):
        raise ValueError("online policy outside frozen scope")
    if c["policy_recipe"] != RECIPE or c["portability"] != {
        "max_absolute_error": 1e-7,
        "max_half_l1": 1e-6,
    }:
        raise ValueError("online policy recipe or portability criterion changed")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    for p, h in {**c["inputs"], **c["extra_model_sources"]}.items():
        verify_file(root / p, h)
    return c


def compare_laws(actual, reference, limits):
    actions = [r[0] for r in actual]
    if actions != reference["marks"]:
        raise ValueError("runtime action support/order differs from recorded production law")
    a, b = np.asarray([r[1] for r in actual]), np.asarray(reference["probabilities"])
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("runtime probability row malformed")
    error = np.abs(a - b)
    metrics = {
        "exact": actual == list(zip(reference["marks"], reference["probabilities"])),
        "marks": len(a),
        "max_absolute_error": float(error.max(initial=0)),
        "half_l1": float(error.sum() / 2),
        "reference_digest": identity(list(zip(reference["marks"], reference["probabilities"]))),
        "actual_digest": identity(actual),
    }
    if (
        metrics["max_absolute_error"] > limits["max_absolute_error"]
        or metrics["half_l1"] > limits["max_half_l1"]
    ):
        raise ValueError(f"runtime failed prospective numerical admission: {metrics}")
    return metrics


@lru_cache(maxsize=1)
def runtime(root: Path, artifact_root: Path, contract_sha256: str):
    c = load_contract(root)
    if c["contract_sha256"] != contract_sha256:
        raise ValueError("warm model cache belongs to another run contract")
    package = c["package"]
    directory = Path(package["package_path"])
    if directory.parent != artifact_root / "inference_packages":
        raise ValueError("inference package outside approved artifact root")
    start = perf_counter()
    model, _ = load_package(
        directory,
        manifest_sha256=package["manifest_sha256"],
        repo_root=root,
        qualified_source_receipt=(root / c["qualification"]["path"], c["qualification"]["sha256"]),
    )
    load_seconds = perf_counter() - start
    reference = unseal(root / c["reference_law"]["path"])
    _, actual = law_values(model, decode_state(reference["source"]))
    comparison = compare_laws(actual, reference, c["portability"])
    comparison.update(
        package_load_seconds=load_seconds,
        total_validation_seconds=perf_counter() - start,
        manifest_sha256=package["manifest_sha256"],
        actual=actual,
    )
    return {
        "model": model,
        "system": _default_rewrite_system(model),
        "model_checkpoint": "/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt",
        "run_paths": "/artifacts/editing_v2/r_theta_run/run_inputs/RUN_PATHS.json",
        "validation": comparison,
        "primed_law": {
            "source": reference["source"],
            "marks": [v[0] for v in actual],
            "probabilities": [v[1] for v in actual],
        },
    }


@contextmanager
def session(task, root, artifact_root, volume, validate_revision):
    validate_revision(task["image_revision"])
    verify_file(root / APP, task["app_sha256"])
    c = load_contract(root)
    if c["contract_sha256"] != task["contract_sha256"]:
        raise ValueError("online deployment contract mismatch")
    body = {k: task[k] for k in ("contract_sha256", "image_revision", "app_sha256")}
    if identity(body) != task["run_id"]:
        raise ValueError("online deployment run identity mismatch")
    output = artifact_root / KIND / task["run_id"]
    if "worker_id" in task:
        if (
            not 1 <= task["phase"] <= 8
            or not 0 <= task["slot"] < 4
            or task["worker_id"] != worker_identity(task["phase"], task["slot"], task["parent"])
        ):
            raise ValueError("online proposal outside bound parent census")
        output = output / "workers" / task["worker_id"]
    volume.reload()
    lock, stop, progress = (
        threading.RLock(),
        threading.Event(),
        {"phase": "initialization", "oracle_calls": 0},
    )

    def commit():
        with lock:
            volume.commit()

    store = Store(output, commit)
    start = perf_counter()

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "at": _stamp(), "seconds": perf_counter() - start},
            )
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield c, store, progress
    except Exception as error:
        store.save("failure", {"error": repr(error), "progress": dict(progress), "at": _stamp()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)
