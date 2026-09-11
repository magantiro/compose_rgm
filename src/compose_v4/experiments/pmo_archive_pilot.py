"""Persistent paired PMO pilot over unchanged executable option chemistry."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from compose_v4.control.archive_allocation import (
    ArchiveCredit,
    parent_distribution,
    scale_cell,
    top10_sum,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_macro_beam import WitnessIndex, distance, exact_archive_graph, replay
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.kernel import canonical_state_key

KIND = "pmo_archive_pilot"
CONTRACT = f"configs/{KIND}.json"
ARMS = ("balanced", "adaptive")


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("archive pilot contract self-hash mismatch")
    if (c["task"], c["arms"], c["replicates"], c["oracle_budget"]) != (
        "perindopril_mpo",
        list(ARMS),
        [0],
        100,
    ) or len(c["roots"]) != 4:
        raise ValueError("archive pilot exceeds approved task/arm/query scope")
    for item in c["inputs"].values():
        verify_file(root / item["path"], item["sha256"])
    for row in c["roots"]:
        exact_archive_graph(row)
    if c["search"] != {
        "option_budget": 11,
        "max_attempts": 300,
        "parent_exploration": 0.2,
        "credit_discount": 0.9,
        "credit_pseudocount": 2.0,
    }:
        raise ValueError("archive pilot allocation recipe changed")
    return c


def prepare_contract(root, prior_launch):
    previous = json.loads((root / "configs/pmo_macro_probe.json").read_text())
    launch = json.loads(prior_launch.read_text())
    c = {
        "schema_version": "pmo_archive_pilot_contract_v1",
        "authorization": "2026-09-10 user approved paired persistent perindopril pilot, 100 queries per arm",
        "task": "perindopril_mpo",
        "arms": list(ARMS),
        "replicates": [0],
        "seed": 20260913,
        "oracle_budget": 100,
        "roots": previous["roots"],
        "root_rule": previous["root_rule"],
        "inputs": {
            "root_contract": {
                "path": "configs/pmo_macro_probe.json",
                "sha256": sha256_file(root / "configs/pmo_macro_probe.json"),
            },
            "cohort": previous["inputs"]["cohort"],
        },
        "expected_input_sha256": previous["expected_input_sha256"],
        "law_caches": [
            {
                "path": f"pmo_macro_probe/{launch['run_id']}/perindopril_mpo__post_hoc__0",
                "launch_sha256": sha256_file(prior_launch),
            }
        ],
        "search": {
            "option_budget": 11,
            "max_attempts": 300,
            "parent_exploration": 0.2,
            "credit_discount": 0.9,
            "credit_pseudocount": 2.0,
        },
        "required_rdkit": "2024.03.5",
        "training_authorized": False,
        "prescreen": False,
        "winner_inputs": False,
        "docking_authorized": False,
        "compute": {
            "cases": 2,
            "max_containers": 2,
            "cpu_per_case": 1,
            "memory_mib": 8192,
            "timeout_seconds": 4500,
            "retries": 0,
            "profile_first_attempts": 4,
            "heartbeat_seconds": 30,
            "previous_seconds_per_attempt": [28, 59],
            "unoptimized_minutes_estimate": [45, 100],
            "expected_cost_usd": [0.5, 2],
            "maximum_reserved_cpu_hours": 2.5,
        },
        "interpretation": "one development pair; fixed 100-query zero-padded top10 AUC; no IVG/official PMO/generalization claim",
    }
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


class Store:
    """Atomic local receipts, periodic remote flush, forced oracle barriers."""

    def __init__(self, output, commit, interval=30):
        self.output, self._commit, self.interval = output, commit, interval
        self.last_flush = perf_counter()
        self._flush_lock = RLock()
        self.timings = {"json_seconds": 0.0, "commit_seconds": 0.0, "commits": 0, "writes": 0}

    def flush(self, force=False):
        # Recheck the interval after acquiring the lock: a heartbeat may have
        # waited behind a barrier that already published its local receipt.
        with self._flush_lock:
            if force or perf_counter() - self.last_flush >= self.interval:
                start = perf_counter()
                self._commit()
                self.timings["commit_seconds"] += perf_counter() - start
                self.timings["commits"] += 1
                self.last_flush = perf_counter()

    def save(self, name, value, *, durable=False):
        start = perf_counter()
        seal(self.output / f"{name}.json", value)
        self.timings["json_seconds"] += perf_counter() - start
        self.timings["writes"] += 1
        # One barrier, not an overdue periodic flush followed by a forced one.
        self.flush(force=durable)

    def read(self, name):
        path = self.output / f"{name}.json"
        return unseal(path) if path.exists() else None


def execute_option(
    parent,
    hierarchy,
    credit,
    adaptive,
    rng,
    name,
    store,
    witnesses,
    progress,
    *,
    primitive_budget=11,
):
    """One registered option, no beam or oracle lookahead. Resumable at each primitive."""
    origin = decode_search_state(parent["node"])
    if type(primitive_budget) is not int or primitive_budget < 1:
        raise ValueError("primitive_budget must be a positive integer")
    node = replace(origin, budget=primitive_budget)  # local clock, NOT a query budget
    record = store.read(name)
    if record is not None:
        if record["parent_id"] != parent["id"] or record["source"] != encode_search_state(node):
            raise ValueError("option resume changed its exact parent")
        if record["status"] != "running":
            return record
        rng.bit_generator.state = record["rng_state"]
        node = decode_search_state(record["node"])
        events, bundle = record["events"], record["bundle"]
    else:
        events, bundle = [], None
    source = encode_search_state(replace(origin, budget=primitive_budget))
    started = perf_counter()
    stage_seconds = {"where": 0.0, "what": 0.0, "how": 0.0, "replay": 0.0}
    status = "running"
    while status == "running":
        if node.stage == "where" and events:
            status = "complete"
            break
        progress.update(
            phase="option_proposal",
            stage=node.stage,
            primitive=primitive_budget - node.budget,
            option=None if bundle is None else bundle["option"],
        )
        event = {"source": encode_search_state(node)}
        start = perf_counter()
        if node.stage in ("where", "what"):
            row = hierarchy.row(node)
            if not row.successors:
                stage_seconds[node.stage] += perf_counter() - start
                status = "support_dead_end"
                break
            cells = (
                [scale_cell(n.region.released_fraction) for n in row.successors]
                if node.stage == "where"
                else list(row.labels)
            )
            q, audit = credit.distribution(row.reference, node.stage, cells, adaptive=adaptive)
            selected = int(rng.choice(len(q), p=q))
            following = row.successors[selected]
            event.update(
                stage=node.stage, labels=list(row.labels), selected=selected, allocation=audit
            )
            if node.stage == "what":
                bundle = {
                    "option": following.active.option,
                    "bundle_id": following.active.bundle_id,
                    "r_release": node.region.released_fraction,
                    "region": encode_search_state(node)["region"],
                }
        else:
            following = hierarchy.sample_reference(node, rng)
            if following is None:
                stage_seconds[node.stage] += perf_counter() - start
                status = "support_dead_end"
                break
            event["mark"] = witnesses.find(node.graph, following.graph)
        stage_seconds[node.stage] += perf_counter() - start
        event["product"] = encode_search_state(following)
        events.append(event)
        node = following
        record = {
            "parent_id": parent["id"],
            "source": source,
            "node": encode_search_state(node),
            "events": events,
            "bundle": bundle,
            "status": "running",
            "rng_state": rng.bit_generator.state,
        }
        store.save(name, record)
    record = {
        "parent_id": parent["id"],
        "source": source,
        "node": encode_search_state(node),
        "events": events,
        "bundle": bundle,
        "status": status,
        "rng_state": rng.bit_generator.state,
    }
    if status == "complete":
        if not events or bundle is None or node.stage != "where":
            raise ValueError("incomplete program cannot enter scored archive")
        start = perf_counter()
        primitives = replay(events, hierarchy.kernel.system)
        stage_seconds["replay"] = perf_counter() - start
        record["candidate"] = {
            "id": name,
            "node": encode_search_state(node),
            "smiles": canonical_state_key(node.graph),
            "chain": parent["chain"] + [name],
            "primitives": parent["primitives"] + primitives,
            "primitive_count": primitives,
            "bundle": bundle,
            "structural_change": structural_displacement(
                origin.graph, node.graph, origin.lineage, node.lineage
            ),
            "topology": topology(node.graph),
        }
    record.update(stage_seconds=stage_seconds, proposal_seconds=perf_counter() - started)
    record = json.loads(json.dumps(record, allow_nan=False))
    store.save(name, record)
    return record


def run_case(contract, case, hierarchy, store, scores, *, meter, progress):
    import cProfile

    c = contract["search"]
    signature = {"contract_sha256": contract["contract_sha256"], "case": case}
    previous = store.read("search_identity")
    if previous is not None and previous != signature:
        raise ValueError("archive resume changed experiment")
    store.save("search_identity", signature)
    store.save("roots", {"roots": contract["roots"]})
    store.flush(force=True)
    scores.context = {"role": "initial", "lock_path": str(store.output / "roots.json")}
    observed, archive = {}, []
    for index, root in enumerate(contract["roots"]):
        observed[root["smiles"]] = scores.score(root["smiles"])["desirability"]
        node = MolecularSearchState.start(
            exact_archive_graph(root), budget=c["option_budget"], root_id=identity(root)
        )
        archive.append(
            {
                "id": f"root_{index}",
                "node": encode_search_state(node),
                "smiles": root["smiles"],
                "chain": [],
                "primitives": 0,
            }
        )
    initial_best = max(observed.values())
    archive_keys = {identity(r["node"]) for r in archive}
    credit = ArchiveCredit(discount=c["credit_discount"], pseudocount=c["credit_pseudocount"])
    witnesses = WitnessIndex(meter)
    attempts, choices, profile = [], [], cProfile.Profile()
    for attempt in range(c["max_attempts"]):
        if len(observed) >= contract["oracle_budget"]:
            break
        name = f"attempts/{attempt:04}"
        progress.update(attempt=attempt, archive_states=len(archive), oracle_calls=len(observed))
        rng = np.random.default_rng(
            np.random.SeedSequence([contract["seed"], case["replicate"], attempt, 0])
        )
        p = parent_distribution(archive, observed, c["parent_exploration"])
        selected = int(rng.choice(len(p), p=p))
        parent = archive[selected]
        choice = {
            "parents": [r["id"] for r in archive],
            "probabilities": p.tolist(),
            "selected": selected,
        }
        saved = store.read(f"parents/{attempt:04}")
        if saved is not None and saved != choice:
            raise ValueError("resumed parent allocation changed")
        store.save(f"parents/{attempt:04}", choice)
        rng = np.random.default_rng(
            np.random.SeedSequence([contract["seed"], case["replicate"], attempt, 1])
        )
        profiled = attempt < contract["compute"]["profile_first_attempts"]
        if profiled:
            profile.enable()
        try:
            record = execute_option(
                parent,
                hierarchy,
                credit,
                case["arm"] == "adaptive",
                rng,
                name,
                store,
                witnesses,
                progress,
            )
        finally:
            if profiled:
                profile.disable()
                profile.dump_stats(str(store.output / "proposal_profile.pstats"))
        before = top10_sum(observed)
        novel = False
        if record["status"] == "complete":
            candidate = record["candidate"]
            smiles = candidate["smiles"]
            novel = smiles not in observed
            scores.context = {
                "role": "completed_option",
                "lock_path": str(store.output / f"{name}.json"),
            }
            observed[smiles] = scores.score(smiles)["desirability"]
            # Molecular mass is canonical; retain distinct exact state/lineage variants separately.
            state_key = identity(
                encode_search_state(replace(decode_search_state(candidate["node"]), budget=11))
            )
            if state_key not in archive_keys:
                archive.append(candidate)
                archive_keys.add(state_key)
        reward = max(0.0, top10_sum(observed) - before)
        where = next((e for e in record["events"] if e.get("stage") == "where"), None)
        scale = where["allocation"]["cells"][where["selected"]] if where else None
        option = record["bundle"]["option"] if record["bundle"] else None
        credit.observe(
            name, parent_chain=parent["chain"], scale=scale, option=option, reward=reward
        )
        summary = {
            "attempt": attempt,
            "status": record["status"],
            "parent": parent["id"],
            "option": option,
            "scale": scale,
            "new_canonical": novel,
            "reward": reward,
            "score": observed[record["candidate"]["smiles"]]
            if record["status"] == "complete"
            else None,
            "calls": len(observed),
            "best": max(observed.values()),
            "top10_mean": top10_sum(observed) / 10,
            "proposal_seconds": record["proposal_seconds"],
            "stage_seconds": record["stage_seconds"],
        }
        store.save(f"feedback/{attempt:04}", summary)
        attempts.append(summary)
        choices.extend(e["allocation"] for e in record["events"] if "allocation" in e)
        progress.update(phase="feedback", **summary)
        store.save(
            "progress_result",
            {
                **summary,
                "credit": credit.edges,
                "persistence": store.timings,
                "archive_states": len(archive),
            },
        )
        print(
            f"[archive] {case['arm']} attempt={attempt + 1} calls={len(observed)}/100 "
            f"best={summary['best']:.6f} top10={summary['top10_mean']:.6f} "
            f"option={option} depth={len(parent['chain']) + 1} proposal={record['proposal_seconds']:.1f}s",
            flush=True,
        )
        # Preserve raw attempted execution evidence without retaining quadratic arrays in RAM forever.
        if store.read(f"executor/{attempt:04}") is None:
            store.save(f"executor/{attempt:04}", meter.attempts)
        # Preserve cached-product witnesses even when their full receipts leave RAM.
        for row in meter.attempts[witnesses.cursor :]:
            if row["status"] == "executed":
                witnesses.marks.setdefault(
                    (identity(row["source"]), identity(row["product"])), row["mark"]
                )
        meter.attempts.clear()
        witnesses.cursor = 0
    curve, prefix = [], {}
    for row in scores.rows:
        prefix[row["smiles"]] = row["score"]
        curve.append(
            {
                "calls": len(prefix),
                "best": max(prefix.values()),
                "top10_mean": top10_sum(prefix) / 10,
            }
        )
    if len(observed) != scores.meter.spent or scores.meter.n_primed != 0:
        raise ValueError("archive and charged oracle ledger diverged")
    unique = sorted(observed)
    distances = [distance(a, b) for i, a in enumerate(unique) for b in unique[i + 1 :]]
    return {
        "schema_version": "pmo_archive_pilot_result_v1",
        "case": case,
        "status": "complete"
        if len(observed) == contract["oracle_budget"]
        else "attempt_limit_incomplete",
        "oracle_calls": scores.meter.spent,
        "primed": scores.meter.n_primed,
        "initial_best": initial_best,
        "best": max(observed.values()),
        "top10_mean": top10_sum(observed) / 10,
        "auc_top10": sum(r["top10_mean"] for r in curve) / contract["oracle_budget"]
        if len(curve) == contract["oracle_budget"]
        else None,
        "curve": curve,
        "attempts": attempts,
        "archive": archive,
        "credit": credit.edges,
        "options": dict(Counter(r["option"] for r in attempts if r["option"] is not None)),
        "outcomes": dict(Counter(r["status"] for r in attempts)),
        "allocation_changed_rows": sum(r["total_variation"] > 1e-10 for r in choices),
        "allocation_mean_tv": float(np.mean([r["total_variation"] for r in choices]))
        if choices
        else 0.0,
        "max_option_depth": max(len(r["chain"]) for r in archive),
        "max_primitive_depth": max(r["primitives"] for r in archive),
        "pairwise_distance_mean": float(np.mean(distances)) if distances else None,
        "top10": [
            {"smiles": s, "score": v}
            for s, v in sorted(observed.items(), key=lambda r: (-r[1], r[0]))[:10]
        ],
        "interpretation": contract["interpretation"],
    }


def run_remote(task, root, artifact_root, volume, runtime_factory, validate_revision):
    import importlib.metadata
    import pstats
    import resource
    import threading

    from rdkit import rdBase

    from compose_v4.control.molecular_task_search import MolecularHierarchy
    from compose_v4.control.option_continuation import (
        EXECUTABLE_PRODUCT_GATE,
        OptionContinuationKernel,
    )
    from compose_v4.experiments.continuation_profile import ExecutorMeter
    from compose_v4.experiments.saved_marked_law import SavedMarkedLaw

    validate_revision(task["image_revision"])
    verify_file(root / "modal_apps/pmo_archive_pilot_app.py", task["app_sha256"])
    contract = load_contract(root)
    expected = identity({k: task[k] for k in ("contract_sha256", "app_sha256", "image_revision")})
    if (
        task["run_id"] != expected
        or task["contract_sha256"] != contract["contract_sha256"]
        or task["case"] not in cases(contract)
    ):
        raise ValueError("archive pilot launch identity differs")
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("archive pilot requires pinned executor RDKit")
    output = artifact_root / KIND / task["run_id"] / case_name(task["case"])
    if (output / "launch.json").exists() and json.loads(
        (output / "launch.json").read_text()
    ) != task:
        raise ValueError("archive pilot resume changed launch")
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    publish_json(output / "launch.json", task)
    volume.commit()
    mutex, stop = threading.RLock(), threading.Event()

    def commit():
        with mutex:
            volume.commit()

    store = Store(output, commit)
    progress = {"case": task["case"], "phase": "initialization", "oracle_calls": 0}
    start = perf_counter()

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {
                    **progress,
                    "updated_at": _stamp(),
                    "seconds": perf_counter() - start,
                    "persistence": dict(store.timings),
                },
            )
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        runtime = runtime_factory()
        for key, path in {
            "r_theta_checkpoint": runtime["model_checkpoint"],
            "r_theta_run_paths": runtime["run_paths"],
        }.items():
            verify_file(Path(path), contract["expected_input_sha256"][key])
        gate = {
            "input_sha256": contract["expected_input_sha256"],
            "software": {
                p: importlib.metadata.version(p) for p in ("numpy", "rdkit", "torch", "PyTDC")
            },
            "hardware": {
                "cpu": 1,
                "memory_mib": 8192,
                "accelerator": None,
                "model_precision": "float32",
            },
            "initialization_seconds": perf_counter() - start,
        }
        publish_json(output / "runtime_gate.json", gate)
        store.flush(force=True)
        scores = DurableScores(
            output,
            make_oracle(contract["task"], root, contract),
            contract["oracle_budget"],
            lambda: store.flush(force=True),
            progress,
        )
        law = SavedMarkedLaw(
            runtime["model"],
            output,
            store.save,
            store.read,
            repo_root=root,
            artifact_root=artifact_root,
            contract=contract,
            progress=progress,
        )
        meter = ExecutorMeter(None)
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
            result = run_case(
                contract, task["case"], hierarchy, store, scores, meter=meter, progress=progress
            )
        stats = pstats.Stats(str(output / "proposal_profile.pstats"))
        profile_rows = [
            {
                "file": key[0],
                "line": key[1],
                "function": key[2],
                "primitive_calls": val[0],
                "calls": val[1],
                "self_seconds": val[2],
                "cumulative_seconds": val[3],
            }
            for key, val in sorted(stats.stats.items(), key=lambda item: -item[1][3])[:50]
        ]
        store.save(
            "profile_summary",
            {
                "profiled_first_attempts": 4,
                "rows": profile_rows,
                "persistence": dict(store.timings),
                "runtime": gate,
            },
        )
        result.update(
            seconds=perf_counter() - start,
            runtime=gate,
            law_counts=law.counts,
            executor_calls=meter.calls,
            executor_seconds=meter.seconds,
            cost_scope="law/executor/persistence counters describe this invocation; prior attempt receipts are retained",
            persistence=dict(store.timings),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            run_id=task["run_id"],
            contract_sha256=contract["contract_sha256"],
            code_revision=task["image_revision"]["commit"],
            oracle_ledger=[
                sha256_file(p) for p in sorted((output / "oracle").glob("*/result.json"))
            ],
        )
        store.save("result", result)
        store.flush(force=True)
        return result
    except Exception as error:
        publish_json(
            output / "failure.json", {"error": repr(error), "progress": progress, "at": _stamp()}
        )
        store.flush(force=True)
        raise
    finally:
        stop.set()
        thread.join(timeout=2)


def cases(contract):
    return [
        {"task": contract["task"], "arm": arm, "replicate": replicate}
        for arm in contract["arms"]
        for replicate in contract["replicates"]
    ]


def case_name(case):
    return f"{case['task']}__{case['arm']}__{case['replicate']}"
