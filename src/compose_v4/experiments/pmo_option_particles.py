"""Small warm-development complete-option SMC comparison, no new training."""

from __future__ import annotations

import json
import threading
from collections import Counter
from contextlib import contextmanager
from time import perf_counter

import numpy as np

from compose_v4.control.archive_allocation import sample_archive_parents
from compose_v4.control.docking_value import identity
from compose_v4.control.option_particles import advance, log_potentials
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save, worker_identity
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp

KIND = "pmo_option_particles"
APP_NAME = "compose-pmo-option-particles"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_OPTION_PARTICLES.md"
ARMS = ("reference", "immediate", "future")
SEED = 20260921
PRIOR_HASH = "647aeee64661a743a3c0fa5f35613da132a790f6ca2d2a7e542dc18122caeabf"


def prepare(root, prior_path):
    verify_file(prior_path, PRIOR_HASH)
    prior = json.loads(prior_path.read_text())
    old_path = root / "diagnostics/pmo_trajectory_value/prepared.json"
    verify_file(old_path, prior["configuration"]["prepared"]["sha256"])
    old = json.loads(old_path.read_text())
    observed = dict(old["observed"])
    for row in prior["oracle_rows"]:
        if row["smiles"] in observed and observed[row["smiles"]] != row["score"]:
            raise ValueError("historical PMO labels disagree")
        observed[row["smiles"]] = row["score"]
    parents = prior["parents"][:5]
    if [p["development_source"] for p in parents] != [
        "current_best",
        *(f"original_root_{i}" for i in range(4)),
    ]:
        raise ValueError("development start census changed")
    data = {
        "schema_version": "option_particles_prepared_v1",
        "parents": parents,
        "observed": observed,
        "prior_result": {"path": str(prior_path), "sha256": PRIOR_HASH, "run_id": prior["run_id"]},
        "prior_prepared": {
            "path": str(old_path.relative_to(root)),
            "sha256": sha256_file(old_path),
        },
        "prior_new_oracle_calls": prior["new_oracle_calls"],
        "historical_unique_labels": len(observed),
    }
    publish_json(root / PREPARED, data)
    runtime = json.loads((root / "configs/pmo_online_policy.json").read_text())
    c = {
        "schema_version": "option_particles_contract_v1",
        "authorization": "2026-09-11 continuing PMO controller goal; bounded uncommitted snapshot exception approved",
        "task": "perindopril_mpo",
        "seed": SEED,
        "particles": 8,
        "boundaries": 6,
        "arms": list(ARMS),
        "beta": 10.0,
        "draws_per_bundle": 1,
        "new_oracle_limit": 144,
        "initial_indices": [i % 5 for i in range(8)],
        "reference_training_authorized": False,
        "value_training_authorized": False,
        "docking_authorized": False,
        "winner_informed_value": True,
        "runtime_contract_sha256": runtime["contract_sha256"],
        "expected_input_sha256": runtime["expected_input_sha256"],
        "law_caches": [],
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        "snapshot_authorization": {
            "path": "docs/PMO_ROUTE_SUPPORT_DEV_SNAPSHOT.md",
            "sha256": sha256_file(root / "docs/PMO_ROUTE_SUPPORT_DEV_SNAPSHOT.md"),
        },
        "value_checkpoint": {
            "path": f"pmo_trajectory_value/{prior['run_id']}/value.pt",
            "sha256": prior["model_sha256"],
        },
        "compute": {
            "max_workers": 24,
            "driver_containers": 1,
            "worker_tasks_max": 144,
            "worker_timeout": 180,
            "driver_timeout": 900,
            "cpu": 1,
            "memory_mib": 8192,
            "retries": 0,
            "heartbeat_seconds": 30,
            "expected_minutes": [4, 10],
            "estimated_usd": [1, 8],
            "reserved_cpu_hour_bound": 7.45,
        },
    }
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("option particles contract hash mismatch")
    if (
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["draws_per_bundle"],
        c["beta"],
        c["seed"],
    ) != (8, 6, 144, 1, 10.0, SEED) or c["arms"] != list(ARMS):
        raise ValueError("particle experiment exceeds locked recipe")
    if any(
        c[k]
        for k in (
            "reference_training_authorized",
            "value_training_authorized",
            "docking_authorized",
        )
    ):
        raise ValueError("no training/docking authorized in particle probe")
    for key in ("prepared", "protocol", "snapshot_authorization"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


@contextmanager
def session(
    task,
    root,
    artifact_root,
    volume,
    validate_revision,
    *,
    contract_loader=load_contract,
    app_path=APP,
    kind=KIND,
):
    validate_revision(task["image_revision"])
    c = contract_loader(root)
    verify_file(root / app_path, task["app_sha256"])
    body = {k: task[k] for k in ("contract_sha256", "image_revision", "app_sha256")}
    if identity(body) != task["run_id"] or c["contract_sha256"] != task["contract_sha256"]:
        raise ValueError("particle deployment identity mismatch")
    output = artifact_root / kind / task["run_id"]
    if "worker_id" in task:
        if not 1 <= task["phase"] <= c["boundaries"] or not 0 <= task["slot"] < c["particles"]:
            raise ValueError("particle worker outside declared census")
        if c.get("donor_memory_modes"):
            from compose_v4.control.donor_memory import proposal_identity

            expected = proposal_identity(task["donor_memory_id"])
            if task.get("local_selector"):
                from compose_v4.control.local_endpoint_selector import active, policy_identity

                if not c.get("local_selector_arms") or not active(
                    c["seed"], task["phase"], task["slot"]
                ):
                    raise ValueError("local selector task is not authorized by the mixture")
                expected = policy_identity(task["donor_memory_id"], c["endpoint_model_sha256"])
            if task.get("proposal_id") != expected:
                raise ValueError("worker donor memory identity is not authorized")
        elif task.get("proposal_id") not in c.get("proposal_ids", {"reference": None}).values():
            raise ValueError("worker proposal identity is not authorized")
        if task["worker_id"] != worker_identity(
            task["phase"], task["slot"], task["parent"], task.get("proposal_id")
        ):
            raise ValueError("particle task parent/seed identity changed")
        output /= f"workers/{task['worker_id']}"
    volume.reload()
    lock, stop = threading.RLock(), threading.Event()

    def commit():
        with lock:
            volume.commit()

    store, started = Store(output, commit), perf_counter()
    _frozen_save(store, "identity", {k: v for k, v in task.items() if k != "image_revision"})
    progress = {"phase": "initialization", "oracle_calls": 0}

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "at": _stamp(), "seconds": perf_counter() - started},
            )
            store.flush()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield c, store, progress
    except Exception as exc:
        store.save("failure", {"error": repr(exc), "progress": dict(progress), "at": _stamp()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)


def archive_metrics(archive):
    values = sorted(archive.values(), reverse=True)
    return {
        "best": max(values, default=0.0),
        "top10_mean": float(np.mean(values[:10])) if values else 0.0,
        "unique_queried": len(values),
        "top10_denominator": min(10, len(values)),
    }


def driver_remote(
    task, root, artifact_root, volume, validate_revision, parallel, *, run_session=None
):
    import resource

    with (run_session or session)(task, root, artifact_root, volume, validate_revision) as (
        c,
        store,
        progress,
    ):
        previous = store.read("result")
        if previous is not None:
            return previous
        started, at = perf_counter(), _stamp()
        data = json.loads((root / c.get("prepared", {"path": PREPARED})["path"]).read_text())
        history = dict(data["observed"])
        arms = tuple(c.get("arms", ARMS))
        if "future" in arms:
            import torch

            from compose_v4.control.trajectory_value import predict
            from compose_v4.control.winner_imitation import ImitationRanker

            model_path = artifact_root / c["value_checkpoint"]["path"]
            verify_file(model_path, c["value_checkpoint"]["sha256"])
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            model = ImitationRanker(518, hidden=128)
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=False)["model"]
            )
            model.eval()
        scores = DurableScores(
            store.output,
            make_oracle(c["task"], root, c),
            c["new_oracle_limit"],
            lambda: store.flush(force=True),
            progress,
        )
        history.update({r["smiles"]: r["score"] for r in scores.rows})
        initial = [data["parents"][i] for i in c["initial_indices"]]
        for p in initial:
            if history[p["smiles"]] != p["score"]:
                raise ValueError("initial particle score mismatch")
        n = c["particles"]
        state = {
            arm: {
                "particles": initial,
                "log_weights": [-float(np.log(n))] * n,
                "log_potential": [0.0] * n,
                "archive": {p["smiles"]: p["score"] for p in initial},
            }
            for arm in arms
        }
        for arm in arms:
            if c.get("parent_selection_modes", {}).get(arm) == "archive":
                state[arm]["archive_nodes"] = {p["smiles"]: p for p in initial}
        rounds, work = [], []
        for step in range(1, c["boundaries"] + 1):
            done = store.read(f"round/{step}/complete")
            if done is not None:
                state = done["state"]
                rounds.append(done["audit"])
                work.extend(done["workers"])
                continue
            tasks, assignments = {}, {}
            for arm in arms:
                memory_context = {}
                if c.get("donor_memory_modes"):
                    from compose_v4.control.donor_memory import build_memory, proposal_identity

                    memory = build_memory(
                        data["initial_donor_memory"],
                        state[arm]["archive_nodes"],
                        mode=c["donor_memory_modes"][arm],
                    )
                    memory_id = memory["memory_id"]
                    _frozen_save(store, f"memory/{memory_id}", memory)
                    memory_context = {
                        "donor_memory_id": memory_id,
                        "donor_memory_sha256": sha256_file(
                            store.output / f"memory/{memory_id}.json"
                        ),
                    }
                assignments[arm] = []
                for slot, parent in enumerate(state[arm]["particles"]):
                    proposal_id = (
                        proposal_identity(memory_context["donor_memory_id"])
                        if memory_context
                        else c.get("proposal_ids", {}).get(arm)
                    )
                    local_context = {}
                    if arm in c.get("local_selector_arms", []):
                        from compose_v4.control.local_endpoint_selector import (
                            active,
                            policy_identity,
                        )

                        if active(c["seed"], step, slot):
                            proposal_id = policy_identity(
                                memory_context["donor_memory_id"], c["endpoint_model_sha256"]
                            )
                            local_context = {
                                "local_selector": True,
                                "local_exclusions": sorted(
                                    set(data["observed"]) | set(state[arm]["archive"])
                                ),
                            }
                    key = (
                        None if parent is None else worker_identity(step, slot, parent, proposal_id)
                    )
                    assignments[arm].append(key)
                    if key is not None:
                        tasks.setdefault(
                            key,
                            {
                                **task,
                                "phase": step,
                                "slot": slot,
                                "parent": parent,
                                "worker_id": key,
                                **memory_context,
                                **local_context,
                                **({"proposal_id": proposal_id} if proposal_id is not None else {}),
                            },
                        )
            _frozen_save(
                store, f"round/{step}/parents", {"tasks": tasks, "assignments": assignments}
            )
            progress.update(
                phase="proposals", boundary=step, workers_complete=0, workers_total=len(tasks)
            )
            results, proposal_started = {}, perf_counter()
            for result in parallel(list(tasks.values())) if tasks else ():
                key = result["worker_id"]
                if key not in tasks or key in results or not result["replay_verified"]:
                    raise ValueError("unexpected or unverified particle proposal")
                if result.get("proposal_policy_sha256") != tasks[key].get("proposal_id"):
                    raise ValueError("particle proposal used the wrong policy")
                if len(result["attempts"]) != 1 or len(result["candidates"]) > 1:
                    raise ValueError(
                        "particle must use one reference option draw, not candidate reranking"
                    )
                results[key] = result
                store.save(f"round/{step}/workers/{key}", result)
                progress.update(workers_complete=len(results))
                print(
                    f"[particles] boundary={step} proposals={len(results)}/{len(tasks)}", flush=True
                )
            if set(results) != set(tasks):
                raise ValueError("incomplete particle worker census")
            proposal_seconds = perf_counter() - proposal_started
            _frozen_save(store, f"round/{step}/candidate_lock", {"results": results})
            candidates = {
                k: (r["candidates"][0] if r["candidates"] else None) for k, r in results.items()
            }
            requested = sorted({p["smiles"] for p in candidates.values() if p is not None})
            novel = [s for s in requested if s not in history]
            scores.context = {
                "role": f"boundary_{step}",
                "lock_path": str(store.output / f"round/{step}/candidate_lock.json"),
            }
            progress.update(
                phase="oracle", locked_unique=len(requested), new_batch_calls=len(novel)
            )
            for s, value in zip(novel, scores.score_many(novel), strict=True):
                history[s] = value["desirability"]
            for p in candidates.values():
                if p is not None:
                    p["score"] = history[p["smiles"]]
            terminal = step == c["boundaries"]
            future = (
                dict(
                    zip(
                        requested,
                        map(float, predict(model, requested, (c["boundaries"] - step) * 11)),
                        strict=True,
                    )
                )
                if "future" in arms and requested and not terminal
                else {}
            )
            audits = {}
            for arm in arms:
                previous_arm = state[arm]
                proposed = [candidates.get(k) for k in assignments[arm]]
                for p in proposed:
                    if p is not None:
                        previous_arm["archive"][p["smiles"]] = p["score"]
                measured = [None if p is None else p["score"] for p in proposed]
                expected = [
                    None if p is None else future.get(p["smiles"], p["score"]) for p in proposed
                ]
                selection_mode = c.get("selection_modes", {}).get(arm, arm)
                rng = np.random.default_rng(np.random.SeedSequence([c["seed"], step, 999]))
                if c.get("parent_selection_modes", {}).get(arm) == "archive":
                    nodes = previous_arm["archive_nodes"]
                    for p in proposed:
                        if p is not None:
                            nodes.setdefault(p["smiles"], p)
                    chosen, selection = sample_archive_parents(
                        nodes, n, rng, exploration=c["archive_exploration"]
                    )
                    update = {"status": "live", "archive_selection": selection}
                    state[arm] = {
                        "particles": chosen,
                        "archive": previous_arm["archive"],
                        "archive_nodes": nodes,
                    }
                else:
                    psi = log_potentials(
                        selection_mode, measured, expected, terminal=terminal, beta=c["beta"]
                    )
                    update = advance(
                        previous_arm["log_weights"],
                        previous_arm["log_potential"],
                        psi,
                        [p is not None for p in proposed],
                        rng,
                        resample=selection_mode != "reference" and not terminal,
                    )
                    chosen = [proposed[i] for i in update["indices"]]
                    state[arm] = {
                        "particles": chosen,
                        "archive": previous_arm["archive"],
                        "log_weights": update["log_weights"],
                        "log_potential": update["log_potential"],
                    }
                metrics = archive_metrics(state[arm]["archive"])
                audits[arm] = {
                    **metrics,
                    **update,
                    "completed_options": sum(p is not None for p in proposed),
                    "unique_products": len({p["smiles"] for p in proposed if p is not None}),
                    "unique_surviving_products": len(
                        {p["smiles"] for p in chosen if p is not None}
                    ),
                    "option_counts": dict(
                        Counter(p["bundle"]["option"] for p in proposed if p is not None)
                    ),
                    "proposals": proposed,
                    "scores": measured,
                    "future_values": expected,
                }
                progress.update(**{f"best_{arm}": metrics["best"]})
                detail = (
                    "archive branching"
                    if "archive_selection" in update
                    else f"ESS={update['ess']:.2f} resampled={update['resampled']}"
                )
                print(
                    f"[particles] boundary={step} arm={arm} best={metrics['best']:.6f} top10={metrics['top10_mean']:.6f} {detail}",
                    flush=True,
                )
            audit = {
                "boundary": step,
                "arms": audits,
                "unique_requested": len(requested),
                "new_oracle_calls": len(novel),
                "cumulative_new_oracle_calls": scores.meter.spent,
                "proposal_seconds": proposal_seconds,
                "at": _stamp(),
            }
            rows = [{k: v for k, v in r.items() if k != "candidates"} for r in results.values()]
            store.save(
                f"round/{step}/complete",
                {"state": state, "audit": audit, "workers": rows},
                durable=True,
            )
            rounds.append(audit)
            work.extend(rows)
        terminal_outputs = {}
        for arm in arms:
            a = rounds[-1]["arms"][arm]
            if c.get("parent_selection_modes", {}).get(arm) == "archive":
                terminal_outputs[arm] = state[arm]["particles"][0]
                continue
            rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 1000]))
            index = None if a["status"] == "extinct" else int(rng.choice(n, p=a["weights"]))
            terminal_outputs[arm] = None if index is None else state[arm]["particles"][index]
        result = {
            "schema_version": "option_population_result_v2"
            if c.get("parent_selection_modes")
            else "option_particles_result_v1",
            "status": "complete_development",
            "configuration": c,
            "run_id": task["run_id"],
            "image_revision": task["image_revision"],
            "initial_parents": initial,
            "initial_metrics": archive_metrics({p["smiles"]: p["score"] for p in initial}),
            "rounds": rounds,
            "arms": {a: archive_metrics(state[a]["archive"]) for a in arms},
            "archives": {a: state[a]["archive"] for a in arms},
            "terminal_draws": terminal_outputs,
            **(
                {
                    "returned_molecule_rules": {
                        a: "best_archive_molecule"
                        if c["parent_selection_modes"][a] == "archive"
                        else "terminal_particle_draw"
                        for a in arms
                    }
                }
                if c.get("parent_selection_modes")
                else {}
            ),
            "new_oracle_calls": scores.meter.spent,
            "oracle_rows": scores.rows,
            "historical_unique_labels": len(data["observed"]),
            "workers": work,
            "io_timings": dict(store.timings),
            "seconds": perf_counter() - started,
            "started_at": at,
            "finished_at": _stamp(),
            "software": software(),
            "hardware": {
                "device": "cpu",
                "threads": 1,
                "neural_dtype": "float32",
                "particle_dtype": "float64",
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "interpretation": c.get(
                "interpretation",
                "winner-informed warm development, not a matched PMO AUC benchmark; full queried archive differs from terminal weighted draw",
            ),
        }
        store.save("result", result)
        return result
