"""Modal boundary for one bounded macro-feedback episode and reusable workers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.control.molecular_task_search import MolecularHierarchy
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.continuation_profile import ExecutorMeter, publish_json, verify_file
from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
from compose_v4.experiments.t4_macro_beam import BeamConfig, run_search
from compose_v4.experiments.t4_macro_feedback import (
    CONTRACT_PATH,
    KIND,
    extend_candidate,
    initial_archive,
    run_episode,
)
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_partial_docking import dock_saved_row


def contract_at(repo_root):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("feedback requires pinned RDKit")
    if (
        contract["search"]["rounds"] != 10
        or contract["search"]["lineages"] != 8
        or contract["search"]["branches"] != 2
        or contract["compute"]["oracle_call_limit"] != 40
    ):
        raise ValueError("feedback does not match the authorized 10x8x2 / 40-call scope")
    return contract


def worker_remote(task, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    volume.reload()
    contract = contract_at(repo_root)
    number, index = task["round"], task["worker_index"]
    if (
        type(number) is not int
        or not 0 <= number < 10
        or type(index) is not int
        or not 0 <= index < 8
    ):
        raise ValueError("feedback worker outside declared round/parent census")
    prefix = f"rounds/{number:02}/workers/{index:02}"
    episode = artifact_root / KIND / task["run_id"]
    before_path = episode / f"rounds/{number:02}/before.json"
    verify_file(before_path, task["before_sha256"])
    parent = unseal(before_path)["parents"][index]
    if identity(parent) != task["parent_sha256"]:
        raise ValueError("worker parent differs from locked round exact state")

    def runner(_actual_task, _prepare, _dock, output, *, commit, progress):
        def read(name):
            path = output / f"{name}.json"
            return unseal(path) if path.exists() else None

        def save(name, payload):
            seal(output / f"{name}.json", payload)
            commit()

        saved = read("worker_complete")
        if saved is not None:
            if saved["parent_sha256"] != task["parent_sha256"]:
                raise ValueError("saved worker completion belongs to another parent")
            return saved
        progress.update(phase="model_initialization", worker=index, round=number + 1)
        started = perf_counter()
        runtime = runtime_factory()
        paths = {
            "r_theta_checkpoint": Path(runtime["model_checkpoint"]),
            "r_theta_run_paths": Path(runtime["run_paths"]),
            "committor": Path("/artifacts/region_committor/committor_bellman_v1.pt"),
        }
        for name, path in paths.items():
            verify_file(path, contract["expected_input_sha256"][name])
        initialization = perf_counter() - started
        save(
            "generator_gate",
            {
                "input_sha256": {k: contract["expected_input_sha256"][k] for k in paths},
                "initialization_seconds": initialization,
            },
        )
        cache = episode / "caches" / f"{index:02}"

        def cache_save(name, payload):
            seal(cache / f"{name}.json", payload)
            commit()

        def cache_read(name):
            path = cache / f"{name}.json"
            return unseal(path) if path.exists() else None

        law = SavedMarkedLaw(
            runtime["model"],
            cache,
            cache_save,
            cache_read,
            repo_root=repo_root,
            artifact_root=artifact_root,
            contract=contract,
            progress=progress,
        )
        meter = ExecutorMeter(None)
        seed = int(
            np.random.SeedSequence([contract["search"]["seed"], number, index]).generate_state(1)[0]
        )

        def no_task_score(_smiles):
            raise AssertionError("proposal worker must not consult the docking guide")

        with meter.instrument():
            kernel = OptionContinuationKernel(
                law,
                runtime["system"],
                max_executor_applications=None,
                product_gate=EXECUTABLE_PRODUCT_GATE,
            )
            hierarchy = MolecularHierarchy(
                kernel, lazy_applicability=True, include_carbonyl_options=True
            )
            try:
                lock = run_search(
                    decode_search_state(parent["node"]),
                    hierarchy,
                    config=BeamConfig(
                        arm="post_hoc",
                        depth=1,
                        width=3,
                        branches=2,
                        primitive_budget=110,
                        seed=seed,
                    ),
                    score=no_task_score,
                    save=save,
                    read=read,
                    meter=meter,
                    progress=progress,
                )
            finally:
                save("executor_attempts_this_invocation", meter.attempts)
        result = {
            "status": "complete",
            "worker_index": index,
            "round": number,
            "parent_sha256": task["parent_sha256"],
            "candidates": [
                extend_candidate(parent, a["candidate"], prefix)
                for a in lock["attempts"]
                if a["status"] == "complete"
            ],
            "attempts": len(lock["attempts"]),
            "options": dict(
                Counter(a["bundle"]["option"] for a in lock["attempts"] if a["bundle"])
            ),
            "outcomes": dict(Counter(a["status"] for a in lock["attempts"])),
            "proposal_seconds": lock["proposal_seconds_this_invocation"],
            "initialization_seconds": initialization,
            "law_work": law.counts,
            "kernel_work": asdict(kernel.work),
            "executor_calls_this_invocation": meter.calls,
            "replay_verified": True,
            "generator_input_sha256": {k: contract["expected_input_sha256"][k] for k in paths},
            "oracle_calls": 0,
        }
        save("worker_complete", result)
        return result

    # Shared wrapper checks code/contract identity and publishes 30-second heartbeat.
    # Generator inputs are additionally verified in the worker, not loaded in driver.
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        None,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=f"{KIND}/{task['run_id']}/{prefix}",
        runner=runner,
    )


def driver_remote(task, repo_root, artifact_root, volume, validate_revision, parallel, dock):
    volume.reload()
    contract = contract_at(repo_root)

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        archive = initial_archive(contract, artifact_root)
        if archive[0]["smiles"] != actual_task["smiles"]:
            from compose_v4.experiments.t4_macro_beam import canonical_smiles

            if archive[0]["smiles"] != canonical_smiles(actual_task["smiles"]):
                raise ValueError("feedback source differs from PARP1 seed0")

        def propose(number, parents, before_hash):
            tasks = [
                {
                    **task,
                    "round": number,
                    "worker_index": i,
                    "before_sha256": before_hash,
                    "parent_sha256": identity(parent),
                }
                for i, parent in enumerate(parents)
            ]
            progress.update(phase="proposals", workers_complete=0, workers_total=len(tasks))
            started, results = perf_counter(), {}
            for result in parallel(tasks):
                i = result["worker_index"]
                if (
                    i in results
                    or not 0 <= i < len(tasks)
                    or result["round"] != number
                    or result["parent_sha256"] != tasks[i]["parent_sha256"]
                    or not result["replay_verified"]
                ):
                    raise ValueError("proposal worker identity or replay disagrees with round lock")
                results[i] = result
                progress.update(workers_complete=len(results))
                publish_json(output / "progress.json", progress)
                commit()
                print(
                    f"round {number + 1}: proposal workers {len(results)}/{len(tasks)}", flush=True
                )
            if set(results) != set(range(len(tasks))):
                raise ValueError("missing feedback proposal workers")
            volume.reload()
            ordered = [results[i] for i in sorted(results)]
            options = Counter()
            for row in ordered:
                options.update(row["options"])
            return {
                "candidates": [c for r in ordered for c in r["candidates"]],
                "attempts": sum(r["attempts"] for r in ordered),
                "options": dict(sorted(options.items())),
                "workers": ordered,
                "proposal_seconds": perf_counter() - started,
            }

        def execute(round_id, index, digest):
            return dock_saved_row(
                {**task, "run_id": round_id, "index": index, "candidate_lock_sha256": digest},
                repo_root,
                artifact_root,
                volume,
                validate_revision,
                dock,
                run_kind=f"{KIND}/{task['run_id']}/docking",
                batch_limit=4,
            )

        return run_episode(
            contract,
            actual_task,
            archive,
            output,
            propose,
            execute,
            commit=commit,
            progress=progress,
        )

    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        None,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=KIND,
        runner=runner,
    )
