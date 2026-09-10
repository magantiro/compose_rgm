"""One bounded policy update and two fresh, paired PMO search rounds."""

from __future__ import annotations

import json
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.archive_allocation import ArchiveCredit, parent_distribution, top10_sum
from compose_v4.control.branch_policy import RECIPE, BranchPolicy
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store, execute_option
from compose_v4.experiments.pmo_chronological import prepare_stream
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal

KIND = "pmo_branch_policy"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
ARMS = ("balanced", "learned")


def prepare(root: Path) -> dict:
    precedent = root / "diagnostics/pmo_chronological/result.json"
    verify_file(precedent, "6e164790e344eaeb41dff735fcb14ef17f1928211bf5a8c07f2b7118dc665873")
    prior = json.loads(precedent.read_text())
    inputs = {str(precedent.relative_to(root)): sha256_file(precedent)}
    observed, archive, observations = {}, {}, []
    for arm in ("balanced", "adaptive"):
        relative = f"diagnostics/pmo_archive_pilot_100/{arm}.json"
        inputs[relative] = verify_file(root / relative, prior["input_hashes"][relative])
        result = json.loads((root / relative).read_text())
        roots = []
        for i in range(4):
            path = f"diagnostics/pmo_chronological/roots/{arm}/{i:04}.json"
            inputs[path] = verify_file(root / path, prior["input_hashes"][path])
            roots.append(unseal(root / path))
        stream = prepare_stream(result, roots)
        for smiles, score in zip(stream["smiles"], stream["scores"], strict=True):
            if smiles in observed and abs(observed[smiles] - score) > 1e-12:
                raise ValueError("historical arms disagree on deterministic PMO label")
            observed[smiles] = score
        for r in stream["rows"]:
            observations.append(
                {
                    "id": f"history/{arm}/{r['id']}",
                    "parent_smiles": stream["smiles"][r["parent_index"]],
                    "smiles": stream["smiles"][r["product_index"]],
                    "parent_score": r["parent_score"],
                    "score": r["score"],
                }
            )
        for r in result["archive"]:
            archive.setdefault(
                r["smiles"],
                {
                    **r,
                    "id": f"history/{arm}/{r['id']}",
                    "chain": [f"history/{arm}/{x}" for x in r["chain"]],
                },
            )
    bank = [{**archive[s], "score": observed[s]} for s in sorted(archive)]
    remaining, selected = list(bank), []
    rng = np.random.default_rng(20260915)
    for _ in range(4):
        index = int(rng.choice(len(remaining), p=parent_distribution(remaining, observed)))
        selected.append(remaining.pop(index))
    prepared = {
        "schema_version": "pmo_branch_prepared_v1",
        "historical_calls": 200,
        "input_hashes": inputs,
        "archive": bank,
        "observed": observed,
        "observations": observations,
        "calibration_parents": selected,
    }
    publish_json(root / PREPARED, prepared)
    old = json.loads((root / "configs/pmo_archive_pilot.json").read_text())
    contract = {
        "schema_version": "pmo_branch_policy_contract_v1",
        "authorization": "2026-09-10 user requests deploying/testing actual conditional-policy learning intervention",
        "task": "perindopril_mpo",
        "seed": 20260915,
        "new_oracle_limit": 32,
        "calibration_bundles": 4,
        "draws_per_bundle": 4,
        "evaluation_rounds": 2,
        "lineages": 4,
        "arms": list(ARMS),
        "policy_recipe": RECIPE,
        "parent_exploration": 0.2,
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {
            "path": "docs/PMO_BRANCH_POLICY.md",
            "sha256": sha256_file(root / "docs/PMO_BRANCH_POLICY.md"),
        },
        "expected_input_sha256": old["expected_input_sha256"],
        "law_caches": old["law_caches"],
        "required_rdkit": "2024.03.5",
        "reference_training_authorized": False,
        "policy_training_authorized": True,
        "prescreen": False,
        "winner_inputs": False,
        "docking_authorized": False,
        "compute": {
            "driver_timeout": 3600,
            "worker_timeout": 1800,
            "max_workers": 8,
            "worker_cpu": 1,
            "worker_memory_mib": 8192,
            "heartbeat_seconds": 30,
            "max_option_draws_before_reuse": 80,
            "expected_minutes": [8, 20],
            "estimated_cost_usd": [1, 5],
            "reserved_cpu_hour_bound": 5,
            "retries": 0,
        },
    }
    contract["contract_sha256"] = identity(contract)
    publish_json(root / CONTRACT, contract)
    return contract


def load_contract(root: Path) -> dict:
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("branch policy contract hash mismatch")
    if (
        c["task"],
        c["new_oracle_limit"],
        c["calibration_bundles"],
        c["draws_per_bundle"],
        c["evaluation_rounds"],
        c["lineages"],
        c["arms"],
    ) != ("perindopril_mpo", 32, 4, 4, 2, 4, list(ARMS)) or c["policy_recipe"] != RECIPE:
        raise ValueError("branch policy exceeds declared scope or changes recipe")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


def worker_identity(phase: int, slot: int, parent: dict) -> str:
    return identity({"phase": phase, "slot": slot, "parent": parent})


@contextmanager
def session(task, root, artifact_root, volume, validate_revision):
    from rdkit import rdBase

    validate_revision(task["image_revision"])
    verify_file(root / "modal_apps/pmo_branch_policy_app.py", task["app_sha256"])
    contract = load_contract(root)
    body = {k: task[k] for k in ("contract_sha256", "app_sha256", "image_revision")}
    if (
        identity(body) != task["run_id"]
        or contract["contract_sha256"] != task["contract_sha256"]
        or rdBase.rdkitVersion != contract["required_rdkit"]
    ):
        raise ValueError("branch run/input/runtime identity mismatch")
    output = artifact_root / KIND / task["run_id"]
    if "worker_id" in task:
        if (
            task["worker_id"] != worker_identity(task["phase"], task["slot"], task["parent"])
            or task["phase"] not in (0, 1, 2)
            or not 0 <= task["slot"] < 4
        ):
            raise ValueError("worker outside declared exact-parent census")
        output = output / "workers" / task["worker_id"]
    volume.reload()
    lock, stop = threading.RLock(), threading.Event()

    def commit():
        with lock:
            volume.commit()

    store = Store(output, commit)
    bound = {k: task[k] for k in ("run_id", "contract_sha256")}
    previous = store.read("identity")
    if previous is not None and previous != bound:
        raise ValueError("restart changed run identity")
    store.save("identity", bound)
    progress, started = {"phase": "initialization", "oracle_calls": 0}, perf_counter()

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "updated_at": _stamp(), "seconds": perf_counter() - started},
            )
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield contract, store, progress
    except Exception as error:
        store.save("failure", {"error": repr(error), "progress": dict(progress), "at": _stamp()})
        store.flush(force=True)
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)


def propose_remote(task, root, artifact_root, volume, validate_revision, runtime_factory):
    from compose_v4.control.molecular_task_search import MolecularHierarchy
    from compose_v4.control.option_continuation import (
        EXECUTABLE_PRODUCT_GATE,
        OptionContinuationKernel,
    )
    from compose_v4.experiments.continuation_profile import ExecutorMeter
    from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
    from compose_v4.experiments.t4_macro_beam import WitnessIndex

    with session(task, root, artifact_root, volume, validate_revision) as (
        contract,
        store,
        progress,
    ):
        previous = store.read("complete")
        if previous is not None:
            return previous
        start = perf_counter()
        runtime = runtime_factory()
        for key, field in (
            ("r_theta_checkpoint", "model_checkpoint"),
            ("r_theta_run_paths", "run_paths"),
        ):
            verify_file(Path(runtime[field]), contract["expected_input_sha256"][key])
        initialization = perf_counter() - start
        store.save(
            "runtime_gate",
            {
                "input_sha256": contract["expected_input_sha256"],
                "initialization_seconds": initialization,
            },
        )
        law = SavedMarkedLaw(
            runtime["model"],
            store.output,
            store.save,
            store.read,
            repo_root=root,
            artifact_root=artifact_root,
            contract=contract,
            progress=progress,
        )
        meter, attempts, candidates = ExecutorMeter(None), [], []
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
            witnesses = WitnessIndex(meter)
            for draw in range(contract["draws_per_bundle"]):
                rng = np.random.default_rng(
                    np.random.SeedSequence([contract["seed"], task["phase"], task["slot"], draw])
                )
                progress.update(draw=draw, draws=4)
                record = execute_option(
                    task["parent"],
                    hierarchy,
                    ArchiveCredit(),
                    False,
                    rng,
                    f"draws/{draw:02}",
                    store,
                    witnesses,
                    progress,
                )
                attempts.append(
                    {
                        "draw": draw,
                        "status": record["status"],
                        "bundle": record["bundle"],
                        "proposal_seconds": record["proposal_seconds"],
                    }
                )
                if record["status"] == "complete":
                    candidate = dict(record["candidate"])
                    candidate.update(
                        id=f"workers/{task['worker_id']}/draws/{draw:02}",
                        parent_smiles=task["parent"]["smiles"],
                        parent_score=task["parent"]["score"],
                    )
                    candidate["chain"] = task["parent"]["chain"] + [candidate["id"]]
                    candidates.append(candidate)
            store.save("executor", meter.attempts)
        result = {
            "worker_id": task["worker_id"],
            "phase": task["phase"],
            "slot": task["slot"],
            "candidates": candidates,
            "attempts": attempts,
            "replay_verified": True,
            "seconds": perf_counter() - start,
            "initialization_seconds": initialization,
            "law_work": law.counts,
            "executor_calls": meter.calls,
            "oracle_calls": 0,
            "code_revision": task["image_revision"]["commit"],
        }
        store.save("complete", result)
        return result


def candidate_pool(
    candidates: list[dict], known: dict
) -> tuple[list[dict], list[float], list[dict]]:
    groups, excluded = {}, []
    for candidate in candidates:
        if candidate["smiles"] in known:
            excluded.append({"id": candidate["id"], "reason": "already_in_arm_scored_archive"})
            continue
        groups.setdefault(candidate["smiles"], []).append(candidate)
    ordered = [groups[s] for s in sorted(groups)]
    count = sum(len(g) for g in ordered)
    return (
        [{**g[0], "proposal_origins": [r["id"] for r in g]} for g in ordered],
        [len(g) / count for g in ordered],
        excluded,
    )


def _frozen_save(store, name, payload):
    previous = store.read(name)
    if previous is not None and previous != payload:
        raise ValueError(f"restart changed locked {name}")
    store.save(name, payload)
    store.flush(force=True)


def driver_remote(task, root, artifact_root, volume, validate_revision, parallel):
    import importlib.metadata
    import platform
    import resource

    with session(task, root, artifact_root, volume, validate_revision) as (
        contract,
        store,
        progress,
    ):
        previous = store.read("result")
        if previous is not None:
            return previous
        started = perf_counter()
        data = json.loads((root / contract["prepared"]["path"]).read_text())
        history = dict(data["observed"])
        scores = DurableScores(
            store.output,
            make_oracle(contract["task"], root, contract),
            contract["new_oracle_limit"],
            lambda: store.flush(force=True),
            progress,
        )

        def fit(name, observations):
            observation_hash = identity(observations)
            saved = store.read(name)
            if saved is None:
                progress.update(phase="policy_fit", training_observations=len(observations))
                fit_start = perf_counter()
                policy = BranchPolicy.fit(observations, source_sha256=observation_hash)
                store.save(name, policy.payload)
                store.save(f"{name}_time", {"seconds": perf_counter() - fit_start})
            else:
                policy = BranchPolicy(saved)
                if policy.payload["observations_sha256"] != observation_hash:
                    raise ValueError("resumed policy training data changed")
            return policy

        def proposals(phase, parents):
            tasks, ids = {}, []
            for i, parent in enumerate(parents):
                slot = i % 4
                key = worker_identity(phase, slot, parent)
                tasks[key] = {
                    **task,
                    "worker_id": key,
                    "phase": phase,
                    "slot": slot,
                    "parent": parent,
                }
                ids.append(key)
            _frozen_save(store, f"phase/{phase}/parents", {"parents": parents, "worker_ids": ids})
            progress.update(
                phase="proposals", round=phase, workers_complete=0, workers_total=len(tasks)
            )
            results = {}
            for result in parallel(list(tasks.values())):
                key = result["worker_id"]
                if key not in tasks or key in results or not result["replay_verified"]:
                    raise ValueError("proposal completion identity mismatch")
                results[key] = result
                progress.update(workers_complete=len(results))
                print(f"[branch] phase={phase} proposals={len(results)}/{len(tasks)}", flush=True)
            if set(results) != set(tasks):
                raise ValueError("missing proposal workers")
            store.save(f"phase/{phase}/proposals", results)
            return [results[key] for key in ids]

        def score(candidate, lock, role):
            if candidate["smiles"] in history:
                return history[candidate["smiles"]]
            scores.context = {"role": role, "lock_path": str(store.output / f"{lock}.json")}
            return scores.score(candidate["smiles"])["desirability"]

        initial_policy = fit("policies/initial", data["observations"])
        calibration = proposals(0, data["calibration_parents"])
        training_candidates = [c for worker in calibration for c in worker["candidates"]]
        prior_utilities = (
            initial_policy.utilities(training_candidates).tolist() if training_candidates else []
        )
        _frozen_save(
            store,
            "phase/0/candidates",
            {
                "candidates": training_candidates,
                "prior_utilities": prior_utilities,
                "policy_sha256": initial_policy.payload["model_sha256"],
            },
        )
        new_observations, common_archive = [], list(data["archive"])
        common_observed = dict(history)
        for c in training_candidates:
            value = score(c, "phase/0/candidates", "calibration")
            common_observed[c["smiles"]] = value
            if not any(r["smiles"] == c["smiles"] for r in common_archive):
                common_archive.append({**c, "score": value})
            new_observations.append(
                {
                    "id": c["id"],
                    "parent_smiles": c["parent_smiles"],
                    "smiles": c["smiles"],
                    "parent_score": c["parent_score"],
                    "score": value,
                }
            )
        _frozen_save(store, "phase/0/observations", {"observations": new_observations})
        policy = fit("policies/updated", data["observations"] + new_observations)
        states = {
            arm: {
                "archive": list(common_archive),
                "observed": dict(common_observed),
                "selections": [],
                "curve": [],
            }
            for arm in ARMS
        }
        initial = {"best": max(common_observed.values()), "top10_sum": top10_sum(common_observed)}
        for number in range(1, contract["evaluation_rounds"] + 1):
            parents = []
            for arm in ARMS:
                state = states[arm]
                probabilities = parent_distribution(state["archive"], state["observed"])
                for slot in range(4):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([contract["seed"], number, slot, 100])
                    )
                    parents.append(
                        state["archive"][int(rng.choice(len(probabilities), p=probabilities))]
                    )
            outputs = proposals(number, parents)
            choices = []
            for i, result in enumerate(outputs):
                arm, slot = ARMS[i // 4], i % 4
                pool, reference, excluded = candidate_pool(
                    result["candidates"], states[arm]["observed"]
                )
                row = {
                    "arm": arm,
                    "slot": slot,
                    "parent": parents[i],
                    "pool": pool,
                    "excluded": excluded,
                    "worker_id": result["worker_id"],
                    "selected": None,
                }
                if pool:
                    q, audit = policy.distribution(pool, reference, guided=arm == "learned")
                    rng = np.random.default_rng(
                        np.random.SeedSequence([contract["seed"], number, slot, 101])
                    )
                    row.update(selected=int(rng.choice(len(q), p=q)), allocation=audit)
                choices.append(row)
            lock_name = f"phase/{number}/choices"
            _frozen_save(
                store,
                lock_name,
                {
                    "choices": choices,
                    "policy_sha256": policy.payload["model_sha256"],
                    "training_observations_sha256": policy.payload["observations_sha256"],
                },
            )
            # All choices are locked before any fresh label in this round is read.
            evaluated = []
            for choice in choices:
                arm, selected = choice["arm"], choice["selected"]
                state = states[arm]
                if selected is None:
                    event = {
                        "round": number,
                        "slot": choice["slot"],
                        "status": "no_unqueried_candidate",
                        "parent": choice["parent"]["id"],
                    }
                else:
                    candidate = choice["pool"][selected]
                    value = score(candidate, lock_name, f"evaluation/{arm}/{number}")
                    state["observed"][candidate["smiles"]] = value
                    if not any(r["smiles"] == candidate["smiles"] for r in state["archive"]):
                        state["archive"].append({**candidate, "score": value})
                    event = {
                        "round": number,
                        "slot": choice["slot"],
                        "status": "scored",
                        "candidate": candidate,
                        "score": value,
                        "parent_delta": value - candidate["parent_score"],
                        "allocation": choice["allocation"],
                    }
                state["selections"].append(event)
                state["curve"].append(
                    {
                        "request": len(state["selections"]),
                        "best": max(state["observed"].values()),
                        "top10_sum": top10_sum(state["observed"]),
                    }
                )
                evaluated.append({"arm": arm, **event})
            _frozen_save(store, f"phase/{number}/outcomes", {"outcomes": evaluated})
            progress.update(
                phase="evaluation_feedback",
                round=number,
                best_by_arm={a: max(states[a]["observed"].values()) for a in ARMS},
            )
            print(
                f"[branch] round={number} calls={scores.meter.spent}/32 best={progress['best_by_arm']}",
                flush=True,
            )
        arms = {}
        for arm, state in states.items():
            scored = [r for r in state["selections"] if r["status"] == "scored"]
            arms[arm] = {
                "initial": initial,
                "best": max(state["observed"].values()),
                "top10_sum": top10_sum(state["observed"]),
                "mean_selected_score": float(np.mean([r["score"] for r in scored]))
                if scored
                else None,
                "mean_parent_delta": float(np.mean([r["parent_delta"] for r in scored]))
                if scored
                else None,
                "logical_query_requests": len(scored),
                "options": dict(Counter(r["candidate"]["bundle"]["option"] for r in scored)),
                **state,
            }
        result = {
            "schema_version": "pmo_branch_policy_result_v1",
            "status": "complete_bounded_development",
            "historical_calls": data["historical_calls"],
            "historical_unique_labels": len(history),
            "new_oracle_calls": scores.meter.spent,
            "new_physical_calls_by_role": dict(Counter(r["role"] for r in scores.rows)),
            "oracle_limit": 32,
            "calibration_candidates": len(training_candidates),
            "evaluation_policy_frozen": True,
            "initial_policy_sha256": initial_policy.payload["model_sha256"],
            "updated_policy_sha256": policy.payload["model_sha256"],
            "training": policy.payload["fit"],
            "fit_times": {
                name: store.read(f"policies/{name}_time") for name in ("initial", "updated")
            },
            "software": {
                "python": platform.python_version(),
                **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "rdkit", "torch")},
            },
            "hardware": {
                "device": "cpu",
                "policy_precision": "float64",
                "reference_precision": "float32",
                "platform": platform.platform(),
                "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "arms": arms,
            "seconds": perf_counter() - started,
            "run_id": task["run_id"],
            "contract_sha256": contract["contract_sha256"],
            "code_revision": task["image_revision"]["commit"],
            "oracle_ledger": [
                sha256_file(p) for p in sorted((store.output / "oracle").glob("*/result.json"))
            ],
            "interpretation": "warm-start developmental policy intervention; 200 historical calls plus explicit new-call ledger; no official PMO, unseen-task, exact-control or IVG-superiority claim",
        }
        if scores.meter.spent > 32 or policy.payload["observations_sha256"] != identity(
            data["observations"] + new_observations
        ):
            raise ValueError("oracle cap or frozen evaluation training boundary violated")
        store.save("result", result)
        return result
