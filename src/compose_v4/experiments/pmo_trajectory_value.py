"""A small target-free, achieved-trajectory-value option comparison."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.control.trajectory_value import (
    BUDGETS,
    SEED,
    achieved_returns,
    choices,
    fit,
    predict,
)
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save, worker_identity
from compose_v4.experiments.pmo_continuation_choice import draw_slots
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "pmo_trajectory_value"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
APP = f"modal_apps/{KIND}_app.py"
APP_NAME = "compose-pmo-trajectory-value"


def prepare(root):
    from tools.pmo_winner_imitation import sources

    routes, route_hashes = sources()
    prior_path = root / "diagnostics/pmo_continuation_choice/prepared.json"
    prior = json.loads(prior_path.read_text())
    history = dict(prior["observed"])
    inputs = {str(Path(p).relative_to(root)): h for p, h in route_hashes.items()}
    inputs[str(prior_path.relative_to(root))] = sha256_file(prior_path)

    def collect(value):
        if isinstance(value, list):
            for v in value:
                collect(v)
        elif isinstance(value, dict):
            if "smiles" in value and "score" in value:
                s, score = value["smiles"], float(value["score"])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("bad historical objective")
                if s in history and abs(history[s] - score) > 1e-12:
                    raise ValueError("historical oracle labels disagree")
                history[s] = score
            for v in value.values():
                collect(v)

    for p in [
        *(f"diagnostics/pmo_continuation_choice/phase{i}_scored.json" for i in (1, 2, 3)),
        "diagnostics/pmo_public_winner_recovery/scores.json",
    ]:
        inputs[p] = sha256_file(root / p)
        collect(json.loads((root / p).read_text()))
    training, audits = [], []
    for r in routes:
        route = r["route"]
        smiles = [canonical_state_key(decode_state(s)) for s in route["states"]]
        training.append({"id": r["id"], "smiles": smiles})
        audits.append(
            {
                "id": r["id"],
                "points": [
                    {"index": i, "state": route["states"][i], "teacher": route["states"][i + 1]}
                    for i in (0, len(route["actions"]) - 8)
                ],
                "perturb": r["id"] in ("original_root_0", "original_root_1"),
            }
        )
    data = {
        "schema_version": "trajectory_value_prepared_v1",
        "parents": prior["parents"],
        "training_routes": training,
        "audits": audits,
        "observed": history,
        "training_identities": sorted({s for r in training for s in r["smiles"]}),
        "input_hashes": inputs,
    }
    publish_json(root / PREPARED, data)
    runtime = json.loads((root / "configs/pmo_online_policy.json").read_text())
    c = {
        "schema_version": "trajectory_value_contract_v1",
        "task": "perindopril_mpo",
        "seed": SEED,
        "authorization": "2026-09-11 user approves target-free ordinary-proposal trajectory-value probe",
        "new_oracle_limit": 400,
        "draws_per_bundle": 4,
        "arms": ["uniform", "immediate", "future"],
        "reference_training_authorized": False,
        "docking_authorized": False,
        "runtime_contract_sha256": runtime["contract_sha256"],
        "expected_input_sha256": runtime["expected_input_sha256"],
        "law_caches": [],
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {
            "path": "docs/PMO_TRAJECTORY_VALUE.md",
            "sha256": sha256_file(root / "docs/PMO_TRAJECTORY_VALUE.md"),
        },
        "inputs": inputs,
        "compute": {
            "max_workers": 20,
            "worker_tasks_max": 33,
            "worker_timeout": 1200,
            "driver_timeout": 1800,
            "cpu": 1,
            "memory_mib": 8192,
            "expected_minutes": [5, 15],
            "estimated_usd": [0.5, 5],
            "retries": 0,
        },
    }
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("trajectory contract hash mismatch")
    if (c["task"], c["new_oracle_limit"], c["draws_per_bundle"], c["seed"]) != (
        "perindopril_mpo",
        400,
        4,
        SEED,
    ):
        raise ValueError("trajectory probe scope changed")
    if (
        c["reference_training_authorized"]
        or c["docking_authorized"]
        or c["compute"]["max_workers"] != 20
    ):
        raise ValueError("trajectory probe authority changed")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    for p, h in c["inputs"].items():
        verify_file(root / p, h)
    return c


@contextmanager
def session(task, root, artifact_root, volume, validate_revision):
    validate_revision(task["image_revision"])
    c = load_contract(root)
    verify_file(root / APP, task["app_sha256"])
    body = {k: task[k] for k in ("contract_sha256", "image_revision", "app_sha256")}
    if identity(body) != task["run_id"] or task["contract_sha256"] != c["contract_sha256"]:
        raise ValueError("trajectory deployment identity mismatch")
    output = artifact_root / KIND / task["run_id"]
    if "worker_id" in task:
        limit = {0: 5, 1: 7, 2: 21}.get(task["phase"], 0)
        if not 0 <= task["slot"] < limit or task["worker_id"] != worker_identity(
            task["phase"], task["slot"], task["parent"]
        ):
            raise ValueError("worker outside trajectory census")
        output = output / "workers" / task["worker_id"]
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
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield c, store, progress
    except Exception as e:
        store.save("failure", {"error": repr(e), "progress": dict(progress), "at": _stamp()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)


def coverage_remote(task, root, artifact_root, volume, validate_revision):
    from compose_v4.control.option_selector import conditioned_action_distribution
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control.region_rewrite import admissible_indices
    from compose_v4.control.region_selector import region_distribution
    from compose_v4.experiments.pmo_online_policy import runtime
    from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
    from compose_v4.experiments.t4_warm_continuation import exact_context
    from compose_v4.rewrite.action_codec_v4 import encode_action

    with session(task, root, artifact_root, volume, validate_revision) as (c, store, progress):
        prior = store.read("complete")
        if prior is not None:
            return prior
        started = perf_counter()
        rt = runtime(root, artifact_root, c["runtime_contract_sha256"])
        law = SavedMarkedLaw(
            rt["model"],
            store.output,
            store.save,
            store.read,
            repo_root=root,
            artifact_root=artifact_root,
            contract=c,
            progress=progress,
        )
        data = json.loads((root / PREPARED).read_text())
        excluded = set(data["training_identities"])
        rows, perturbation = [], None
        for point in task["parent"]["points"]:
            graph = decode_state(point["state"])
            teacher = canonical_state_key(decode_state(point["teacher"]))
            families, actions, probabilities = law(graph)
            products, failures = {}, 0
            source = canonical_state_key(graph)
            for i, (family, action, probability) in enumerate(
                zip(families, actions, probabilities, strict=True)
            ):
                if probability <= 0:
                    continue
                try:
                    child = rt["system"].apply(graph, family, action)
                except InvalidRewrite:
                    failures += 1
                    continue
                key = canonical_state_key(child)
                if 1 <= child.n_real_atoms <= 40 and key != source:
                    products[i] = (key, child)
            matching = {i for i, (key, _) in products.items() if key == teacher}
            regions, _, _ = region_distribution(
                [r for r in enumerate_regions(source) if 1 <= r.size <= 24]
            )
            generic = []
            for r in regions:
                context = exact_context(graph, source, r)
                indices, _ = admissible_indices(families, actions, context)
                row = conditioned_action_distribution(
                    families, np.asarray(probabilities), indices, "generic"
                )
                if matching & set(row.indices):
                    generic.append({"region": repr(r.key()), "release": r.released_fraction})
            rows.append(
                {
                    "source": task["parent"]["id"],
                    "step": point["index"],
                    "teacher_successor": teacher,
                    "marks": len(actions),
                    "canonical_products": len({s for s, _ in products.values()}),
                    "teacher_present": bool(matching),
                    "teacher_mark_mass": sum(probabilities[i] for i in matching),
                    "productive_mass": sum(probabilities[i] for i in products),
                    "regions": len(regions),
                    "generic_exposure_regions": generic,
                    "invalid_mark_executions": failures,
                }
            )
            if task["parent"]["perturb"] and point["index"] > 0:
                eligible = [i for i, (s, _) in products.items() if s not in excluded]
                if not eligible:
                    raise ValueError("no new ordinary perturbation; no target-directed replacement")
                p = np.array([probabilities[i] for i in eligible])
                p /= p.sum()
                rng = np.random.default_rng(np.random.SeedSequence([SEED, task["slot"], 900]))
                i = eligible[int(rng.choice(len(eligible), p=p))]
                s, child = products[i]
                node = MolecularSearchState.start(
                    child, budget=11, root_id=identity(encode_state(child))
                )
                perturbation = {
                    "id": f"perturbed_{task['parent']['id']}",
                    "development_source": f"perturbed_{task['parent']['id']}",
                    "node": encode_search_state(node),
                    "smiles": s,
                    "chain": [],
                    "primitives": 0,
                    "perturbation": {
                        "source": point["state"],
                        "mark": encode_action(families[i], actions[i]),
                        "selection": "reference-weighted outside training canonical identities",
                    },
                }
        result = {
            "worker_id": task["worker_id"],
            "rows": rows,
            "perturbation": perturbation,
            "seconds": perf_counter() - started,
            "law_work": law.counts,
            "oracle_calls": 0,
            "code_revision": task["image_revision"]["commit"],
        }
        store.save("complete", result)
        return result


def driver_remote(task, root, artifact_root, volume, validate_revision, parallel):
    import importlib.metadata
    import platform
    import resource

    import torch

    with session(task, root, artifact_root, volume, validate_revision) as (c, store, progress):
        prior = store.read("result")
        if prior is not None:
            return prior
        started, at = perf_counter(), _stamp()
        data = json.loads((root / PREPARED).read_text())
        history = dict(data["observed"])
        scores = DurableScores(
            store.output,
            make_oracle(c["task"], root, c),
            c["new_oracle_limit"],
            lambda: store.flush(force=True),
            progress,
        )
        worker_log = []

        def workers(phase, parents):
            tasks = [
                {
                    **task,
                    "phase": phase,
                    "slot": i,
                    "parent": p,
                    "worker_id": worker_identity(phase, i, p),
                }
                for i, p in enumerate(parents)
            ]
            _frozen_save(store, f"phase/{phase}/parents", {"tasks": tasks})
            wanted, results = {t["worker_id"] for t in tasks}, {}
            progress.update(
                phase=f"proposals_{phase}", workers_complete=0, workers_total=len(tasks)
            )
            for result in parallel(phase, tasks) if tasks else ():
                key = result["worker_id"]
                if key not in wanted or key in results:
                    raise ValueError("duplicate/unexpected worker")
                if phase and not result["replay_verified"]:
                    raise ValueError("unverified option")
                results[key] = result
                store.save(f"phase/{phase}/workers/{key}", result)
                progress.update(workers_complete=len(results))
                print(f"[trajectory] phase={phase} workers={len(results)}/{len(tasks)}", flush=True)
            if set(results) != wanted:
                raise ValueError("incomplete worker census")
            ordered = [results[t["worker_id"]] for t in tasks]
            worker_log.extend(ordered)
            return ordered

        def score_list(smiles, name):
            _frozen_save(store, name, {"smiles": sorted(set(smiles))})
            scores.context = {"role": name, "lock_path": str(store.output / f"{name}.json")}
            for s in sorted(set(smiles)):
                if s not in history:
                    history[s] = scores.score(s)["desirability"]

        coverage = workers(0, data["audits"])
        parents = [
            *data["parents"],
            *[w["perturbation"] for w in coverage if w["perturbation"] is not None],
        ]
        if len(parents) != 7:
            raise ValueError("expected five original and two new nearby development starts")
        _frozen_save(store, "probe_starts", {"parents": parents})
        score_list(data["training_identities"], "training_state_labels")
        training_calls = scores.meter.spent
        rows = [
            {"smiles": s, "budget": 0, "return": value, "route": None}
            for s, value in sorted(history.items())
        ]
        for route in data["training_routes"]:
            returns = achieved_returns([history[s] for s in route["smiles"]])
            for s, values in zip(route["smiles"], returns, strict=True):
                rows.extend(
                    {"smiles": s, "budget": b, "return": v, "route": route["id"]}
                    for b, v in zip(BUDGETS, values, strict=True)
                )
        panel = {
            "rows": rows,
            "prepared_sha256": c["prepared"]["sha256"],
            "seed": SEED,
            "role": "answer-informed training",
        }
        _frozen_save(store, "training_panel", panel)
        store.flush(force=True)
        progress.update(phase="fit_trajectory_value")
        fit_start = perf_counter()
        model, checkpoint, fit_metrics = fit(panel)
        fit_metrics["seconds"] = perf_counter() - fit_start
        checkpoint.update(
            configuration=c, panel_sha256=sha256_file(store.output / "training_panel.json")
        )
        temporary = store.output / "value.partial.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(store.output / "value.pt")
        model_hash = sha256_file(store.output / "value.pt")
        _frozen_save(store, "fit", {**fit_metrics, "model_sha256": model_hash})
        store.flush(force=True)
        print(f"[trajectory] fitted {len(rows)} rows, training calls={training_calls}", flush=True)
        score_list([p["smiles"] for p in parents], "probe_start_labels")
        for p in parents:
            p["score"] = history[p["smiles"]]

        def propose_score(phase, nodes):
            result = workers(phase, nodes)
            _frozen_save(store, f"phase/{phase}/candidate_lock", {"workers": result})
            slots = [draw_slots(w) for w in result]
            score_list(
                [r["smiles"] for row in slots for r in row if r is not None],
                f"phase_{phase}_labels",
            )
            for row in slots:
                for r in row:
                    if r is not None:
                        r["score"] = history[r["smiles"]]
            _frozen_save(store, f"phase/{phase}/scored", {"slots": slots})
            return slots

        first = propose_score(1, parents)
        decisions, selected = [], {}
        for i, (parent, row) in enumerate(zip(parents, first, strict=True)):
            live = [r for r in row if r is not None]
            value = (
                dict(
                    zip(
                        [r["smiles"] for r in live], predict(model, [r["smiles"] for r in live], 11)
                    )
                )
                if live
                else {}
            )
            decision = choices(
                row, [None if r is None else value[r["smiles"]] for r in row], seed=SEED + i
            )
            for d in decision.values():
                candidate = d["candidate"]
                d["selected_key"] = None if candidate is None else identity(candidate["node"])
                if candidate is not None:
                    selected.setdefault(d["selected_key"], candidate)
            decisions.append(
                {
                    "source": parent["development_source"],
                    "start_score": parent["score"],
                    "decisions": decision,
                }
            )
        _frozen_save(store, "decisions", {"rows": decisions, "model_sha256": model_hash})
        keys = sorted(selected)
        second = propose_score(2, [selected[k] for k in keys])
        continuations = dict(zip(keys, second, strict=True))
        for row in decisions:
            for d in row["decisions"].values():
                incumbent = d["candidate"]
                candidates = (
                    []
                    if incumbent is None
                    else [
                        incumbent,
                        *[r for r in continuations[d["selected_key"]] if r is not None],
                    ]
                )
                best = (
                    max(candidates, key=lambda r: (r["score"], r["smiles"])) if candidates else None
                )
                d["retained_best"] = best
                d["best_score_with_initial_archive"] = max(
                    row["start_score"], best["score"] if best else 0.0
                )
                d["continuations"] = [] if incumbent is None else continuations[d["selected_key"]]
        arms = {
            a: {
                "scores": [r["decisions"][a]["best_score_with_initial_archive"] for r in decisions],
                "abstentions": sum(r["decisions"][a]["status"] == "abstained" for r in decisions),
            }
            for a in c["arms"]
        }
        for a in arms.values():
            a.update(mean=float(np.mean(a["scores"])), best=max(a["scores"]))
        result = {
            "schema_version": "trajectory_value_result_v1",
            "status": "complete_bounded_development",
            "configuration": c,
            "coverage": coverage,
            "parents": parents,
            "decisions": decisions,
            "arms": arms,
            "fit": fit_metrics,
            "new_oracle_calls": scores.meter.spent,
            "training_calls": training_calls,
            "evaluation_calls": scores.meter.spent - training_calls,
            "historical_unique_labels": len(data["observed"]),
            "oracle_rows": scores.rows,
            "model_sha256": model_hash,
            "seconds": perf_counter() - started,
            "started_at": at,
            "finished_at": _stamp(),
            "workers": len(worker_log),
            "worker_summary": [
                {k: v for k, v in w.items() if k not in ("candidates", "perturbation")}
                for w in worker_log
            ],
            "code_revision": task["image_revision"]["commit"],
            "run_id": task["run_id"],
            "software": {
                "python": platform.python_version(),
                **{
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "scipy", "rdkit", "torch", "PyTDC")
                },
            },
            "hardware": {
                "cpu_threads": 1,
                "device": "cpu",
                "precision": "float32 value, float64 selection",
                "platform": platform.platform(),
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "interpretation": "target-free inference after answer-informed training; seven starts, two-option prefix; not exact h, full-path recovery, calibrated optimal value, or benchmark superiority",
        }
        store.save("result", result)
        return result
