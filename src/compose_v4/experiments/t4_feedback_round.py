"""One authorized feedback round after the audited partial docking batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import verify_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import (
    advance_archive,
    payload_hash,
    run_rounds,
)

CONTRACT_PATH = "configs/t4_feedback_round.json"
KIND = "t4_feedback_round"


def feedback_archive(root: Path, contract: dict) -> dict:
    """Import every evaluated endpoint, without inventing an eighth parent.

    This is a new feedback episode, not a bit-identical resume of the failed
    optimizer round. Archive event 2 remains explicitly a partial diagnostic.
    """
    assets = {}
    for name, item in contract["source"].items():
        path = root / item["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("feedback source path escapes artifact root")
        verify_file(path, item["sha256"])
        assets[name] = unseal(path)
    warm, lock, docking = (assets[k] for k in ("warm", "lock", "docking"))
    source_task = lock["source_task"]
    for key, value in contract["task"].items():
        if key == "executor_calls_per_parent":
            if value != 2500:
                raise ValueError("only the approved 2500-call parent share is authorized")
            continue
        if source_task.get(key) != value:
            raise ValueError(f"feedback changed frozen controller field: {key}")
    if source_task["expected_input_sha256"] != contract["expected_input_sha256"]:
        raise ValueError("feedback changed frozen scientific inputs")
    if (
        warm["oracle_attempts"] != 20
        or warm["round"] != 1
        or lock["schema_version"] != "t4_partial_candidate_lock_v1"
        or docking["optimizer_round_completed"] is not False
        or docking["new_oracle_attempts"] != 13
        or len(lock["take"]) != 13
        or lock["source_warm_start_sha256"] != payload_hash(warm)
        or docking["candidate_lock_sha256"] != contract["source"]["lock"]["sha256"]
    ):
        raise ValueError("feedback source is not the audited 20+13-call history")
    # Seed derivation uses immutable source identities, never docking-score
    # thresholds or a search over seeds. Record the fresh-episode boundary.
    seed_identity = {"sources": contract["source"], "seed_rng": contract["task"]["seed_rng"]}
    seed = int(payload_hash(seed_identity)[:16], 16)
    rng_state = np.random.default_rng(seed).bit_generator.state
    adapted_lock = {**lock, "round": 2, "rng_state": rng_state}
    archive = advance_archive(warm, adapted_lock, docking)
    if len(archive["archive"]) != 34 or archive["oracle_attempts"] != 33:
        raise ValueError("feedback archive lost an evaluated molecule or seed")
    archive["feedback_origin"] = {
        "source": contract["source"],
        "event_2": "partial_seven_parent_diagnostic_not_complete_optimizer_round",
        "rng_policy": "fresh_episode_source_hash_seed_v1",
        "derived_seed": seed,
        "failed_parent_replayed": False,
    }
    return archive


def run_remote(
    task, repo_root, artifact_root, volume, runtime_factory, validate_revision, prepare, dock
):
    from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote

    def runner(actual_task, prepare, dock, output, **kwargs):
        contract = json.loads((repo_root / CONTRACT_PATH).read_text())
        if (
            contract["additional_rounds"] != 1
            or contract["compute"]["oracle_call_limit"] != 20
            or actual_task["budget"] != 20
            or actual_task["per_round"] != 20
            or actual_task.get("executor_calls_per_parent") != 2500
            or actual_task["max_executor_applications"] != 20000
            or actual_task["lineages"] != 8
            or actual_task["workers"] != 1
        ):
            raise ValueError("only one new twenty-call feedback round is authorized")
        if rdBase.rdkitVersion != contract["required_rdkit"]:
            raise ValueError("feedback round requires pinned RDKit 2024.03.5")
        warm = feedback_archive(artifact_root, contract)
        result = run_rounds(
            actual_task,
            warm,
            prepare,
            dock,
            output,
            additional_rounds=1,
            endpoint_policy=contract["endpoint_selection_policy"],
            source_identity=contract["source"],
            **kwargs,
        )
        result.update(
            schema_version="t4_feedback_round_v1",
            remaining_new_call_allowance=27 - result["new_oracle_attempts"],
            automatic_continuation=False,
        )
        return result

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        prepare,
        dock,
        contract_path=CONTRACT_PATH,
        run_kind=KIND,
        runner=runner,
    )
