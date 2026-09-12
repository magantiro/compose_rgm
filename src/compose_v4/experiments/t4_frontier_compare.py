"""One paired, warm-start oracle round with resumable independent preparation."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from compose_v4.control.docking_value import snapshot_equivalence
from compose_v4.control.frontier_search import payload_hash
from compose_v4.experiments.continuation_profile import (
    encode_action,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.t4_frontier_audit import verify_preparation
from compose_v4.experiments.t4_frontier_search import FrontierConfig, prepare_slice
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "t4_frontier_compare"
CONTRACT_PATH = "configs/t4_frontier_compare.json"
ARMS = ("post_hoc", "in_loop")


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes root: {relative!r}")
    return path


def normalize_warm_metadata(warm, source_sha256):
    """Migrate spelling only after proving equality to every exact saved graph."""
    archive = []
    corrections = []
    for index, record in enumerate(warm["archive"]):
        exact = canonical_state_key(decode_state(record["state"]))
        molecule = Chem.MolFromSmiles(record["smiles"])
        metadata = None if molecule is None else Chem.MolToSmiles(molecule)
        if metadata != exact:
            raise ValueError(
                "archive molecular metadata differs from exact saved state"
            )
        if record["smiles"] != exact:
            corrections.append(
                {
                    "index": index,
                    "original_smiles": record["smiles"],
                    "canonical_smiles": exact,
                    "exact_state_sha256": payload_hash(record["state"]),
                }
            )
        archive.append({**record, "smiles": exact})
    return {
        **warm,
        "archive": archive,
        "source_metadata_normalization": {
            "schema_version": "t4_exact_archive_metadata_normalization_v1",
            "source_sha256": source_sha256,
            "row_count": len(archive),
            "corrections": corrections,
            "rule": "canonicalize SMILES only after equality to decoded exact persistent-slot graph",
        },
    }


def load_contract(repo_root):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    reuse = contract.get("preparation_reuse")
    preparation_compute_is_valid = (
        contract["compute"]["preparation_containers"] == 0
        and contract["compute"]["maximum_concurrent_containers"] == 1
        if reuse
        else contract["compute"]["preparation_containers"] == 16
        and contract["compute"]["maximum_concurrent_containers"] <= 20
    )
    if (
        payload_hash({k: v for k, v in contract.items() if k != "contract_sha256"})
        != contract["contract_sha256"]
        or contract["additional_rounds"] != 1
        or contract["compute"]["oracle_call_limit"] != 40
        or contract["compute"]["oracle_calls_per_arm"] != 20
        or contract["events_per_parent"] != 48
        or contract["training_authorized"] is not False
        or contract.get("launch_authorization", {}).get("status")
        != "explicit_user_approved"
        or not preparation_compute_is_valid
        or contract["kappa"] != 1
        or set(contract["arms"]) != set(ARMS)
        or contract["arms"]["post_hoc"]
        != {
            **contract["arms"]["in_loop"],
            "guidance": "post_hoc",
            "planning_transitions": 0,
        }
        or rdBase.rdkitVersion != contract["required_rdkit"]
    ):
        raise ValueError("frontier comparison contract or pinned chemistry mismatch")
    for arm in ARMS:
        FrontierConfig(**contract["arms"][arm])
    if reuse and (
        reuse.get("schema_version") != "t4_frontier_preparation_reuse_v1"
        or reuse.get("source_failure_phase") != "lineage_reduction"
        or len(reuse.get("source_run_id", "")) != 64
        or any(c not in "0123456789abcdef" for c in reuse.get("source_run_id", ""))
        or len(reuse.get("source_code_revision", "")) != 40
        or any(
            c not in "0123456789abcdef" for c in reuse.get("source_code_revision", "")
        )
        or set(reuse.get("preparations", {})) != set(ARMS)
        or any(len(reuse["preparations"][arm]) != 8 for arm in ARMS)
    ):
        raise ValueError("frontier preparation reuse contract is incomplete")
    return contract


def load_reused_preparations(artifact_root: Path, contract: dict) -> tuple[dict, dict]:
    """Authenticate a complete pre-oracle source run and its 16 partitions."""
    reuse = contract["preparation_reuse"]
    source_run = reuse["source_run_id"]
    root = artifact_root / KIND / source_run
    launch_path = _inside(artifact_root, reuse["source_launch"]["path"])
    failure_path = _inside(artifact_root, reuse["source_failure"]["path"])
    if launch_path != root / "launch.json" or failure_path != root / "failure.json":
        raise ValueError(
            "source launch or failure path changes the registered source run"
        )
    verify_file(launch_path, reuse["source_launch"]["sha256"])
    verify_file(failure_path, reuse["source_failure"]["sha256"])
    launch = json.loads(launch_path.read_text())
    failure = json.loads(failure_path.read_text())
    if (
        launch.get("run_id") != source_run
        or launch.get("contract_sha256") != reuse["source_contract_file_sha256"]
        or launch.get("image_revision", {}).get("commit")
        != reuse["source_code_revision"]
        or launch.get("image_revision", {})
        .get("serialized_sources", {})
        .get(CONTRACT_PATH)
        != reuse["source_contract_file_sha256"]
        or failure.get("run_id") != source_run
        or failure.get("phase") != reuse["source_failure_phase"]
        or failure.get("error_type") != "ValueError"
        or len(failure.get("preparation_calls", {})) != 16
    ):
        raise ValueError("source frontier run provenance differs from reuse contract")
    if (
        (root / "oracle_barrier.json").exists()
        or (root / "result.json").exists()
        or any(root.glob("*/oracle/*"))
    ):
        raise ValueError("source frontier run crossed the oracle barrier")

    paths, receipts = {}, {}
    for arm in ARMS:
        paths[arm], receipts[arm] = [], []
        for lineage, registered in enumerate(reuse["preparations"][arm]):
            expected = (
                artifact_root
                / KIND
                / arm
                / f"lineage_{lineage:02d}"
                / source_run
                / "preparation.json"
            )
            path = _inside(artifact_root, registered["path"])
            if path != expected:
                raise ValueError(
                    "reused preparation path changes arm, lineage or source run"
                )
            verify_file(path, registered["sha256"])
            part = unseal(path)
            saved = part["checkpoint"]
            if (
                saved["code_revision"] != reuse["source_code_revision"]
                or saved["source_sha256"] != contract["source"]["sha256"]
                or saved["config"] != contract["arms"][arm]
                or saved.get("partition_lineages") != [lineage]
                or part.get("worker", {}).get("arm") != arm
                or part.get("worker", {}).get("lineage_index") != lineage
                or any(
                    saved["input_sha256"].get(key) != value
                    for key, value in contract["expected_input_sha256"].items()
                )
            ):
                raise ValueError(
                    "reused preparation differs from current scientific contract"
                )
            paths[arm].append(path)
            receipts[arm].append(
                {
                    "lineage_index": lineage,
                    "path": str(path),
                    "sha256": registered["sha256"],
                    "checkpoint_sha256": saved["checkpoint_sha256"],
                }
            )
    return paths, {
        "schema_version": "t4_frontier_preparation_reuse_audit_v1",
        "source_run_id": source_run,
        "source_code_revision": reuse["source_code_revision"],
        "source_launch_sha256": reuse["source_launch"]["sha256"],
        "source_failure_sha256": reuse["source_failure"]["sha256"],
        "source_failure_phase": failure["phase"],
        "source_oracle_barrier_absent": True,
        "source_result_absent": True,
        "source_oracle_attempt_artifacts": 0,
        "preparations": receipts,
    }


def load_warm(artifact_root, contract):
    path = artifact_root / contract["source"]["path"]
    if not path.resolve().is_relative_to(artifact_root.resolve()):
        raise ValueError("warm archive escapes artifact root")
    verify_file(path, contract["source"]["sha256"])
    warm = unseal(path)
    if (
        warm["schema_version"] != "t4_exact_archive_v1"
        or warm["oracle_attempts"] != 51
        or warm["round"] != 4
        or len(warm["archive"]) != 52
    ):
        raise ValueError("comparison requires the complete exact 51-call archive")
    return normalize_warm_metadata(warm, contract["source"]["sha256"])


def prepare_worker(
    task, repo_root, artifact_root, volume, runtime_factory, validate_revision
):
    """Separate CPU worker; no oracle code or authority in this entrypoint."""
    contract = load_contract(repo_root)
    arm = task["arm"]
    if arm not in ARMS:
        raise ValueError("unknown comparison arm")
    lineage = task["lineage_index"]
    if type(lineage) is not int or not 0 <= lineage < contract["arms"][arm]["lineages"]:
        raise ValueError("unknown comparison lineage")

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        from compose_v4.experiments.production_successor_kernel import (
            enumerate_factorized_marked_law,
        )

        warm = load_warm(artifact_root, contract)
        seal(
            output / "source_metadata_normalization.json",
            warm["source_metadata_normalization"],
        )
        prepared_path = output / "preparation.json"
        if prepared_path.exists():
            prepared = unseal(prepared_path)
            return {
                "preparation_path": str(prepared_path),
                "checkpoint_sha256": prepared["checkpoint"]["checkpoint_sha256"],
                "new_oracle_calls": 0,
            }
        runtime = runtime_factory()
        # Bind the deployed scientific source tree, not unrelated docs/launchers.
        # This first run has no cross-revision cache-reuse claim.
        implementation = {
            str(p.relative_to(repo_root)): sha256_file(p)
            for base in (
                "src/compose_v4/control",
                "src/compose_v4/rewrite",
                "src/compose_v4/model",
                "src/compose_v4/chem",
                "src/compose_v4/data",
            )
            for p in sorted((repo_root / base).rglob("*.py"))
        }
        implementation["src/compose_v4/experiments/production_successor_kernel.py"] = (
            sha256_file(
                repo_root / "src/compose_v4/experiments/production_successor_kernel.py"
            )
        )
        seal(output / "implementation.json", implementation)
        inputs = {
            **actual_task["expected_input_sha256"],
            "implementation": payload_hash(implementation),
        }
        checkpoint_path = output / "checkpoint.json"
        checkpoint = unseal(checkpoint_path) if checkpoint_path.exists() else None
        memory_cache = {}
        progress.update(
            phase="preparation",
            arm=arm,
            lineage_index=lineage,
            law_evaluations=0,
            law_cache_hits=0,
        )

        def law(graph):
            exact = encode_state(graph)
            key = payload_hash({"state": exact, "inputs": inputs, "time_point": 0.5})
            if key in memory_cache:
                progress["law_cache_hits"] += 1
                return memory_cache[key]
            path = output / "laws" / f"{key}.json"
            if path.exists():
                cached = unseal(path)
                if cached["state"] != exact or cached["inputs"] != inputs:
                    raise ValueError("exact reference cache identity mismatch")
                pairs = [
                    (
                        action_codec_v4 if a["schema_version"] == 4 else action_codec
                    ).decode_action(a)
                    for a in cached["marks"]
                ]
                result = (
                    tuple(r for r, _ in pairs),
                    tuple(a for _, a in pairs),
                    tuple(cached["probabilities"]),
                )
                progress["law_cache_hits"] += 1
            else:
                progress.update(phase="reference_enumeration", latest_law=key)
                started = time.perf_counter()
                row = enumerate_factorized_marked_law(runtime["model"], graph, 0.5)
                result = (
                    tuple(m.executor_rule_name for m in row.marks),
                    tuple(m.action for m in row.marks),
                    tuple(m.probability for m in row.marks),
                )
                seal(
                    path,
                    {
                        "state": exact,
                        "inputs": inputs,
                        "marks": [
                            encode_action(r, a)
                            for r, a in zip(result[0], result[1], strict=True)
                        ],
                        "probabilities": list(result[2]),
                        "seconds": time.perf_counter() - started,
                    },
                )
                progress["law_evaluations"] += 1
                commit()
            memory_cache[key] = result
            progress.update(phase="option_expansion")
            return result

        def checkpoint_progress(saved):
            seal(checkpoint_path, saved)
            progress.update(
                phase="search",
                arm=arm,
                executor_calls=saved["executor_calls"],
                completed_lineages=sum(
                    u["current"]["budget"] == 0 or u["status"] == "no_admissible_action"
                    for u in saved["frontier"]
                ),
                committed_primitives=sum(
                    u["root"]["budget"] - u["current"]["budget"]
                    for u in saved["frontier"]
                ),
                completed_options=sum(len(u["candidates"]) for u in saved["frontier"]),
                lineage_status=[
                    {
                        "lineage": u["lineage_index"],
                        "stage": u["current"]["stage"],
                        "remaining_primitives": u["current"]["budget"],
                        "status": u["status"],
                    }
                    for u in saved["frontier"]
                ],
            )
            commit()

        started = time.perf_counter()
        prepared = prepare_slice(
            warm,
            source_sha256=contract["source"]["sha256"],
            input_sha256=inputs,
            code_revision=actual_task["code_revision"],
            enumerate_law=law,
            system=runtime["system"],
            config=FrontierConfig(**contract["arms"][arm]),
            checkpoint=checkpoint,
            events_per_parent=contract["events_per_parent"],
            progress=checkpoint_progress,
            lineage_indices=(lineage,),
        )
        prepared["worker"] = {
            "arm": arm,
            "law_evaluations": progress["law_evaluations"],
            "law_cache_hits": progress["law_cache_hits"],
            "proposal_seconds_this_invocation": time.perf_counter() - started,
            "lineage_index": lineage,
            "planner_work_by_lineage": [
                prepared["checkpoint"]["frontier"][lineage]["planner"]["work"]
            ],
        }
        seal(prepared_path, prepared)
        commit()
        return {
            "preparation_path": str(prepared_path),
            "checkpoint_sha256": prepared["checkpoint"]["checkpoint_sha256"],
            "new_oracle_calls": 0,
            **prepared["worker"],
        }

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=f"{KIND}/{arm}/lineage_{lineage:02d}",
        runner=runner,
    )


def merge_preparations(parts, warm, system):
    """Deterministic reduction of every lineage; no molecular re-enumeration."""
    if not parts:
        raise ValueError("cannot merge an empty lineage census")
    first = parts[0]["checkpoint"]
    config = FrontierConfig(**first["config"])
    units = {}
    identities = (
        "schema_version",
        "reference_id",
        "round",
        "source_sha256",
        "input_sha256",
        "code_revision",
        "config",
        "archive_prefix",
        "prior_oracle_attempts",
    )
    probe_smiles = tuple(c["smiles"] for part in parts for c in part["pool"])
    snapshots = {
        part["checkpoint"]["value_snapshot"]["snapshot_sha256"]: part["checkpoint"][
            "value_snapshot"
        ]
        for part in parts
    }
    snapshot_ids = sorted(snapshots)
    baseline_snapshot = snapshots[snapshot_ids[0]]
    equivalence = [
        snapshot_equivalence(
            baseline_snapshot, snapshots[key], probe_smiles=probe_smiles
        )
        for key in snapshot_ids
    ]
    for part in parts:
        saved = part["checkpoint"]
        if payload_hash(
            {k: v for k, v in saved.items() if k != "checkpoint_sha256"}
        ) != saved["checkpoint_sha256"] or any(
            saved[k] != first[k] for k in identities
        ):
            raise ValueError(
                "lineage partitions have inconsistent identities or hashes"
            )
        indices = saved.get("partition_lineages", [])
        if (
            len(indices) != 1
            or saved["pending_slice"] is not None
            or saved["retired_lineages"]
        ):
            raise ValueError("expected a complete single-lineage preparation")
        index = indices[0]
        if type(index) is not int or not 0 <= index < config.lineages or index in units:
            raise ValueError("duplicate or out-of-range lineage partition")
        unit = saved["frontier"][index]
        if unit["lineage_index"] != index or (
            unit["current"]["budget"] and unit["status"] != "no_admissible_action"
        ):
            raise ValueError(
                "lineage partition did not complete its scientific horizon"
            )
        units[index] = unit
    if set(units) != set(range(config.lineages)):
        raise ValueError("missing lineage partition; partial pools cannot dock")
    merged = {
        **first,
        "frontier": [units[i] for i in range(config.lineages)],
        "partition_lineages": list(range(config.lineages)),
        "executor_calls": sum(p["checkpoint"]["executor_calls"] for p in parts),
        "new_executor_calls": 0,
    }
    merged.pop("checkpoint_sha256")
    merged["checkpoint_sha256"] = payload_hash(merged)

    def forbidden(_):
        raise AssertionError("lineage reduction must not enumerate molecular laws")

    result = prepare_slice(
        warm,
        source_sha256=first["source_sha256"],
        input_sha256=first["input_sha256"],
        code_revision=first["code_revision"],
        enumerate_law=forbidden,
        system=system,
        config=config,
        checkpoint=merged,
        events_per_parent=0,
    )
    result["source_value_snapshots"] = [snapshots[key] for key in sorted(snapshots)]
    result["source_value_snapshot_equivalence"] = {
        "schema_version": "t4_value_snapshot_equivalence_v1",
        "comparisons": equivalence,
        "unique_snapshot_count": len(snapshots),
        "probe_count": len(set(probe_smiles)),
    }
    result["seconds"] = max(p["seconds"] for p in parts)
    result["worker"] = {
        "parallel_lineages": config.lineages,
        "sum_proposal_seconds": sum(p["seconds"] for p in parts),
        "slowest_lineage_proposal_seconds": max(p["seconds"] for p in parts),
        "law_evaluations": sum(
            p.get("worker", {}).get("law_evaluations", 0) for p in parts
        ),
        "law_cache_hits": sum(
            p.get("worker", {}).get("law_cache_hits", 0) for p in parts
        ),
        "planner_work_by_lineage": [
            units[i]["planner"]["work"] for i in range(config.lineages)
        ],
    }
    return result


def dock_locked_pair(locks, warm, output, dock, *, commit=lambda: None, progress=None):
    """Durable per-attempt ledger. An unknown interrupted attempt is never retried."""
    progress = {} if progress is None else progress
    if set(locks) != set(ARMS):
        raise ValueError("both audited arm locks are required before any oracle call")
    for arm, lock in locks.items():
        if (
            lock["schema_version"] != "t4_frontier_oracle_lock_v1"
            or lock["round"] != warm["round"] + 1
            or len(lock["take"]) > 20
            or lock["prior_oracle_attempts"] != warm["oracle_attempts"]
            or lock["config"]["guidance"] != arm
        ):
            raise ValueError("invalid paired oracle lock or call budget")
    barrier = {
        "lock_sha256": {arm: payload_hash(locks[arm]) for arm in ARMS},
        "prior_attempts_per_arm": warm["oracle_attempts"],
        "maximum_new_attempts": 40,
    }
    barrier_path = output / "oracle_barrier.json"
    if barrier_path.exists():
        if unseal(barrier_path) != barrier:
            raise ValueError("paired oracle locks changed after the barrier")
    else:
        seal(barrier_path, barrier)
        commit()
    results = {}
    prior_feasible_scores = [
        r["ds"] for r in warm["archive"] if r["ds"] is not None and r.get("v") == 0
    ]
    for arm in ARMS:
        lock, docked = locks[arm], []
        for index, candidate in enumerate(lock["take"]):
            item = output / arm / "oracle" / f"{index:03d}"
            identity = {
                "barrier_sha256": payload_hash(barrier),
                "arm": arm,
                "index": index,
                "candidate_sha256": payload_hash(candidate),
            }
            finished, started = item / "result.json", item / "started.json"
            if finished.exists():
                record = unseal(finished)
                if record["identity"] != identity:
                    raise ValueError("cached docking identity mismatch")
            else:
                if started.exists():
                    raise RuntimeError(
                        f"unknown interrupted oracle attempt: {started}; review accounting before any retry"
                    )
                seal(
                    started,
                    {**identity, "started_at_utc": _stamp(), "charged_attempts": 1},
                )
                commit()
                progress.update(
                    phase="docking",
                    arm=arm,
                    oracle_index=index,
                    new_attempts_in_arm=index + 1,
                )
                score = dock(candidate["smiles"], f"{arm}_{index}")
                if score is not None and (
                    isinstance(score, bool) or not math.isfinite(score)
                ):
                    raise ValueError(
                        "nonfinite/invalid docking outcome; attempt is already charged"
                    )
                record = {
                    "identity": identity,
                    "candidate": candidate,
                    "ds": score,
                    "completed_at_utc": _stamp(),
                }
                seal(finished, record)
                commit()
            if record["candidate"] != candidate:
                raise ValueError("cached docking candidate mismatch")
            docked.append({**candidate, "ds": record["ds"], "round": lock["round"]})
            progress.update(
                last_docking_score=record["ds"],
                completed_dockings_in_arm=len(docked),
                best_new_score=min(
                    (r["ds"] for r in docked if r["ds"] is not None), default=None
                ),
                best_feasible_score=min(
                    [
                        *prior_feasible_scores,
                        *(r["ds"] for r in docked if r["ds"] is not None),
                    ],
                    default=None,
                ),
            )
        archive = {
            **warm,
            "round": lock["round"],
            "archive": [*warm["archive"], *docked],
            "oracle_attempts": warm["oracle_attempts"] + len(docked),
        }
        seal(output / arm / "archive.json", archive)
        results[arm] = {
            "new_oracle_attempts": len(docked),
            "total_oracle_attempts": archive["oracle_attempts"],
            "failed_dockings": sum(r["ds"] is None for r in docked),
            "best_new_score": min(
                (r["ds"] for r in docked if r["ds"] is not None), default=None
            ),
            "best_feasible_score": min(
                [
                    *prior_feasible_scores,
                    *(r["ds"] for r in docked if r["ds"] is not None),
                ],
                default=None,
            ),
            "docked": docked,
            "audit": lock.get("audit", {}),
            "selected_options": dict(
                sorted(Counter(c["option"] for c in docked).items())
            ),
            "distinct_docked_bundles": len({c["allocated_bundle_id"] for c in docked}),
        }
        seal(output / arm / "docking.json", results[arm])
        commit()
    return {
        "schema_version": "t4_frontier_comparison_result_v1",
        "status": "complete",
        "arms": results,
        "new_oracle_attempts": sum(r["new_oracle_attempts"] for r in results.values()),
        "automatic_next_round": False,
        "scope": "one unreplicated warm-start development round; unmatched internal compute",
    }


def run_remote(
    task,
    repo_root,
    artifact_root,
    volume,
    runtime_factory,
    validate_revision,
    prepare_pair,
    dock,
):
    contract = load_contract(repo_root)

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        warm = load_warm(artifact_root, contract)
        seal(
            output / "source_metadata_normalization.json",
            warm["source_metadata_normalization"],
        )
        if contract.get("preparation_reuse"):
            progress.update(phase="preparation_reuse_validation")
            preparation_paths, reuse_audit = load_reused_preparations(
                artifact_root, contract
            )
            seal(output / "preparation_reuse_audit.json", reuse_audit)
            generation_revision = contract["preparation_reuse"]["source_code_revision"]
            commit()
        else:
            preparation_paths = prepare_pair(task, output, commit, progress)
            reuse_audit = None
            generation_revision = actual_task["code_revision"]
        volume.reload()
        locks, preparation = {}, {}
        for arm in ARMS:
            paths = [Path(p) for p in preparation_paths[arm]]
            expected_run = (
                contract["preparation_reuse"]["source_run_id"]
                if contract.get("preparation_reuse")
                else task["run_id"]
            )
            expected = [
                artifact_root
                / KIND
                / arm
                / f"lineage_{i:02d}"
                / expected_run
                / "preparation.json"
                for i in range(contract["arms"][arm]["lineages"])
            ]
            if paths != expected:
                raise ValueError("preparation came from another run, arm or lineage")
            progress.update(phase="lineage_reduction", arm=arm)
            prepared = merge_preparations(
                [unseal(p) for p in paths], warm, runtime_factory()["system"]
            )
            path = output / arm / "preparation.json"
            seal(path, prepared)
            if (
                prepared["checkpoint"]["config"] != contract["arms"][arm]
                or prepared["checkpoint"]["source_sha256"]
                != contract["source"]["sha256"]
                or prepared["checkpoint"]["code_revision"] != generation_revision
            ):
                raise ValueError("prepared arm differs from frozen comparison")
            progress.update(phase="selected_path_audit", arm=arm)
            lock = verify_preparation(prepared, warm, runtime_factory()["system"])
            seal(output / arm / "candidate_lock.json", lock)
            locks[arm] = lock
            preparation[arm] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "seconds": prepared["seconds"],
                "executor_calls": prepared["checkpoint"]["executor_calls"],
                **prepared["worker"],
                "partition_sha256": {str(p): sha256_file(p) for p in paths},
            }
            commit()
        result = dock_locked_pair(
            locks, warm, output, dock, commit=commit, progress=progress
        )
        result["preparation"] = preparation
        result["preparation_reuse"] = reuse_audit
        return result

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=KIND,
        runner=runner,
    )
