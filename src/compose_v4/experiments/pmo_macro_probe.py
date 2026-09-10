"""Small PMO transfer probe over the existing complete-option beam controller.

No docking, surrogate fitting, prescreen labels, new chemistry, or full PMO claim.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.benchmark.molleo_task3 import OracleMeter
from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
from compose_v4.experiments.t4_macro_beam import (
    BeamConfig,
    canonical_smiles,
    distance,
    exact_archive_graph,
    run_search,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

KIND = "pmo_macro_probe"
CONTRACT = f"configs/{KIND}.json"
TASKS = ("albuterol_similarity", "perindopril_mpo", "jnk3")
ARMS = ("post_hoc", "guided")


def load_contract(root):
    value = json.loads((root / CONTRACT).read_text())
    if (
        identity({k: v for k, v in value.items() if k != "contract_sha256"})
        != value["contract_sha256"]
    ):
        raise ValueError("PMO probe contract self-hash mismatch")
    if value["tasks"] != list(TASKS) or value["arms"] != list(ARMS):
        raise ValueError("PMO probe task/arm selection changed")
    if value["replicates"] != [0, 1] or len(value["roots"]) != 4 or value["oracle_budget"] != 44:
        raise ValueError("PMO probe exceeds the four-root, two-replicate, 44-call scope")
    if value["search"] != {"depth": 3, "width": 2, "branches": 2, "primitive_budget": 110}:
        raise ValueError("PMO probe search recipe changed")
    for item in value["inputs"].values():
        verify_file(root / item["path"], item["sha256"])
    for row in value["roots"]:
        exact_archive_graph(row)
    return value


def prepare_contract(root):
    """Choose diverse supported initial states, without evaluating any objective."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = root / "docs/MOLLEO_DEV_COHORT.json"
    cohort = json.loads(source.read_text())
    admitted, excluded = {}, []
    for index, smiles in enumerate(cohort["init"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"bad development source row {index}: {smiles}")
        if not 1 <= mol.GetNumHeavyAtoms() <= 40 or any(
            a.GetFormalCharge() for a in mol.GetAtoms()
        ):
            excluded.append(
                {"index": index, "smiles": smiles, "reason": "neutral_1_to_40_root_panel"}
            )
            continue
        Chem.RemoveStereochemistry(mol)
        achiral = Chem.MolToSmiles(mol)
        graph = pad_molecular_graph(smiles_to_molecular_graph(achiral), 48)
        if canonical_state_key(graph) != achiral:
            raise ValueError(f"initial-state conversion changed row {index}")
        admitted.setdefault(
            achiral, {"smiles": achiral, "state": encode_state(graph), "source_index": index}
        )
    pool = sorted(admitted.values(), key=lambda r: identity(r["smiles"]))
    selected = [pool.pop(0)]
    while len(selected) < 4:
        row = max(
            pool,
            key=lambda r: (min(distance(r["smiles"], s["smiles"]) for s in selected), r["smiles"]),
        )
        selected.append(row)
        pool.remove(row)
    previous = json.loads((root / "configs/t4_macro_lookahead.json").read_text())
    value = {
        "schema_version": "pmo_macro_probe_contract_v1",
        "authorization": "2026-09-10 user: do it and test only a few PMO tasks first",
        "role": "new bounded development transfer probe; old frozen PMO pilot unchanged",
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "replicates": [0, 1],
        "seed": 20260912,
        "oracle_budget": 44,
        "search": {"depth": 3, "width": 2, "branches": 2, "primitive_budget": 110},
        "root_rule": "existing development cohort; neutral 1..40-heavy-atom roots; declared achiral projection; first SHA then fingerprint max-min; no labels",
        "roots": selected,
        "root_exclusions": excluded,
        "root_population_count": len(cohort["init"]),
        "root_admitted_unique": len(admitted),
        "inputs": {
            name: {"path": path, "sha256": sha256_file(root / path)}
            for name, path in {
                "cohort": "docs/MOLLEO_DEV_COHORT.json",
                "jnk3": "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                "jnk3_parity": "diagnostics/pmo_legacy_oracle_parity.json",
            }.items()
        },
        "expected_input_sha256": {
            k: previous["expected_input_sha256"][k]
            for k in ("r_theta_checkpoint", "r_theta_run_paths")
        },
        "law_caches": [],
        "required_rdkit": "2024.03.5",
        "training_authorized": False,
        "prescreen": False,
        "winner_inputs": False,
        "docking_authorized": False,
        "compute": {
            "cases": 12,
            "max_containers": 12,
            "cpu_per_case": 1,
            "memory_mib": 8192,
            "timeout_seconds": 2400,
            "retries": 0,
            "expected_minutes": [15, 30],
            "historical_seconds_per_option": [22, 29],
            "max_option_attempts_per_case": 40,
            "expected_cost_usd": [1, 4],
            "maximum_reserved_cpu_hours": 8,
        },
        "interpretation": "short-prefix top-10 curves and paired option-beam effects only; no 10k AUC extrapolation, no official PMO or unseen-task claim",
    }
    value["contract_sha256"] = identity(value)
    publish_json(root / CONTRACT, value)
    return value


def make_oracle(name, root, contract):
    if name not in TASKS:
        raise ValueError(f"undeclared PMO oracle {name}")
    if name == "jnk3":
        from compose_v4.benchmark.oracles.forest import FrozenForest

        path = root / contract["inputs"]["jnk3"]["path"]
        forest = FrozenForest(path)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        def evaluate(smiles):
            fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
            array = np.zeros(2048, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            return float(forest.probabilities(array.astype(np.float64)[None, :])[0])

        return evaluate
    import sys
    import types

    import rdkit

    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda d, **kwargs: iter(d.items())
    shim.itervalues = lambda d, **kwargs: iter(d.values())
    shim.iterkeys = lambda d, **kwargs: iter(d.keys())
    shim.string_types = (str,)
    sys.modules["rdkit.six"] = rdkit.six = shim
    from tdc import Oracle

    oracle = Oracle(name=name)
    return lambda smiles: float(oracle(smiles))


class DurableScores:
    """Charge all source, branch-ranking and endpoint labels; never prime."""

    def __init__(self, output, oracle, budget, commit, progress):
        self.output, self.oracle, self.commit, self.progress = output, oracle, commit, progress
        self.rows = []
        self.context = {"role": "initial", "lock_path": None}
        for start in sorted((output / "oracle").glob("*/started.json")):
            result = start.with_name("result.json")
            if not result.exists():
                raise RuntimeError(f"unresolved oracle attempt, no implicit retry: {start}")
            row = unseal(result)
            if row["status"] != "complete":
                raise RuntimeError(f"failed oracle attempt remains charged: {result}")
            self.rows.append(row)
        self.meter = OracleMeter(
            self._raw, budget=budget, n_objectives=1, canonicalize=canonical_smiles
        )
        self.meter.restore({r["smiles"]: (r["score"],) for r in self.rows})

    def _raw(self, smiles):
        lock = self.context["lock_path"]
        if not lock or not Path(lock).exists():
            raise ValueError("PMO evaluation requires a durable candidate lock")
        number = len(self.rows)
        row = {
            "schema_version": "pmo_probe_oracle_v1",
            "index": number,
            "smiles": smiles,
            **self.context,
            "lock_sha256": sha256_file(Path(lock)),
            "started_at": _stamp(),
        }
        folder = self.output / "oracle" / f"{number:04}"
        seal(folder / "started.json", row)
        self.commit()
        start = perf_counter()
        try:
            value = float(self.oracle(smiles))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"PMO oracle returned {value!r} for {smiles}")
        except Exception as error:
            seal(folder / "result.json", {**row, "status": "failed", "error": repr(error)})
            self.commit()
            raise
        row.update(score=value, status="complete", oracle_seconds=perf_counter() - start)
        seal(folder / "result.json", row)
        self.commit()
        self.rows.append(row)
        self.progress.update(oracle_calls=len(self.rows), best=max(r["score"] for r in self.rows))
        print(
            f"[pmo] call={len(self.rows)} score={value:.6f} best={self.progress['best']:.6f}",
            flush=True,
        )
        return (value,)

    def score(self, smiles):
        return {"desirability": self.meter(smiles)[0]}


def run_case(contract, case, hierarchy, output, scores, *, meter, commit, progress):
    """Same full option support and execution work; task value only changes retention."""
    summaries, all_candidates, all_attempts = [], [], []
    root_lock = output / "roots.json"
    seal(root_lock, {"roots": contract["roots"]})
    commit()
    scores.context = {"role": "initial", "lock_path": str(root_lock)}
    for root in contract["roots"]:
        scores.score(root["smiles"])
    initial = {r["smiles"]: scores.meter.evaluated()[r["smiles"]][0] for r in contract["roots"]}
    for index, root in enumerate(contract["roots"]):
        prefix = output / "roots" / f"{index:02}"
        progress.update(root=index, phase="search")

        def save(name, value, prefix=prefix):
            seal(prefix / f"{name}.json", value)
            commit()

        def read(name, prefix=prefix):
            path = prefix / f"{name}.json"
            return unseal(path) if path.exists() else None

        def score(smiles, prefix=prefix):
            if case["arm"] != "guided":
                raise AssertionError("post-hoc baseline consulted task value during generation")
            matches = [
                p
                for p in sorted(prefix.glob("levels/*/attempt_*.json"))
                if unseal(p).get("candidate", {}).get("smiles") == smiles
            ]
            if not matches:
                raise ValueError("retention score lacks a completed option witness")
            scores.context = {"role": "in_loop_completed_option", "lock_path": str(matches[0])}
            return scores.score(smiles)

        seed = int(
            np.random.SeedSequence([contract["seed"], case["replicate"], index]).generate_state(1)[
                0
            ]
        )
        config = BeamConfig(arm=case["arm"], seed=seed, **contract["search"])
        node = MolecularSearchState.start(
            exact_archive_graph(root), budget=config.primitive_budget, root_id=identity(root)
        )
        lock = run_search(
            node,
            hierarchy,
            config=config,
            score=score,
            save=save,
            read=read,
            meter=meter,
            progress=progress,
        )
        candidates = [a["candidate"] for a in lock["attempts"] if a["status"] == "complete"]
        scores.context = {"role": "post_lock", "lock_path": str(prefix / "generation_lock.json")}
        for candidate in candidates:
            scores.score(candidate["smiles"])
        all_candidates.extend(candidates)
        all_attempts.extend(lock["attempts"])
        summaries.append(
            {
                "root": index,
                "smiles": root["smiles"],
                "config": asdict(config),
                "initial_score": initial[root["smiles"]],
                "best_descendant": max(
                    (scores.meter.evaluated()[c["smiles"]][0] for c in candidates), default=None
                ),
                "complete": len(candidates),
                "attempts": len(lock["attempts"]),
                "oracle_calls": scores.meter.spent,
            }
        )
        seal(
            output / "progress_result.json",
            {"roots": summaries, "oracle_calls": scores.meter.spent},
        )
        commit()
    curve = []
    for count in range(1, len(scores.rows) + 1):
        ordered = sorted((r["score"] for r in scores.rows[:count]), reverse=True)
        curve.append(
            {"calls": count, "best": ordered[0], "top10_mean": float(np.mean(ordered[:10]))}
        )
    values = scores.meter.evaluated()
    unique = sorted({c["smiles"] for c in all_candidates})
    diversity = [distance(a, b) for i, a in enumerate(unique) for b in unique[i + 1 :]]
    return {
        "schema_version": "pmo_macro_probe_result_v1",
        "status": "complete",
        "case": case,
        "roots": summaries,
        "oracle_calls": scores.meter.spent,
        "primed": scores.meter.n_primed,
        "initial_best": max(initial.values()),
        "best": curve[-1]["best"],
        "top10_mean": curve[-1]["top10_mean"],
        "best_improvement": curve[-1]["best"] - max(initial.values()),
        "curve": curve,
        "options": dict(Counter(a["bundle"]["option"] for a in all_attempts if a.get("bundle"))),
        "outcomes": dict(Counter(a["status"] for a in all_attempts)),
        "unique_candidates": len(unique),
        "candidate_pairwise_distance_mean": float(np.mean(diversity)) if diversity else None,
        "top10": [
            {"smiles": s, "score": v[0]}
            for s, v in sorted(values.items(), key=lambda x: (-x[1][0], x[0]))[:10]
        ],
        "candidates": all_candidates,
        "interpretation": contract["interpretation"],
    }


def run_remote(task, root, artifact_root, volume, runtime_factory, validate_revision):
    import importlib.metadata
    import resource
    import threading

    validate_revision(task["image_revision"])
    verify_file(root / "modal_apps/pmo_macro_probe_app.py", task["app_sha256"])
    contract = load_contract(root)
    if contract["contract_sha256"] != task["contract_sha256"]:
        raise ValueError("deployed PMO contract differs from launch")
    expected_run = identity(
        {k: task[k] for k in ("contract_sha256", "app_sha256", "image_revision")}
    )
    case = task["case"]
    if task["run_id"] != expected_run or case not in cases(contract):
        raise ValueError("undeclared PMO run/case")
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("PMO requires pinned executor RDKit")
    output = artifact_root / KIND / task["run_id"] / case_name(case)
    if (output / "result.json").exists():
        if json.loads((output / "launch.json").read_text()) != task:
            raise ValueError("existing PMO case launch differs")
        return unseal(output / "result.json")
    publish_json(output / "launch.json", task)
    volume.commit()
    progress, stop, mutex = (
        {"case": case, "phase": "initialization"},
        threading.Event(),
        threading.RLock(),
    )
    started = perf_counter()

    def commit():
        with mutex:
            volume.commit()

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "updated_at": _stamp(), "seconds": perf_counter() - started},
            )
            commit()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        runtime = runtime_factory()
        paths = {
            "r_theta_checkpoint": runtime["model_checkpoint"],
            "r_theta_run_paths": runtime["run_paths"],
        }
        for name, path in paths.items():
            verify_file(Path(path), contract["expected_input_sha256"][name])
        gate = {
            "input_sha256": contract["expected_input_sha256"],
            "software": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "rdkit", "torch", "PyTDC")
            },
            "hardware": {
                "cpu": 1,
                "memory_mib": 8192,
                "accelerator": None,
                "model_precision": "float32",
            },
        }
        publish_json(output / "runtime_gate.json", gate)
        commit()
        scores = DurableScores(
            output,
            make_oracle(case["task"], root, contract),
            contract["oracle_budget"],
            commit,
            progress,
        )

        def save(name, value):
            seal(output / f"{name}.json", value)
            commit()

        def read(name):
            path = output / f"{name}.json"
            return unseal(path) if path.exists() else None

        law = SavedMarkedLaw(
            runtime["model"],
            output,
            save,
            read,
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
                contract,
                case,
                hierarchy,
                output,
                scores,
                meter=meter,
                commit=commit,
                progress=progress,
            )
        save("executor_attempts", meter.attempts)
        result.update(
            seconds=perf_counter() - started,
            law_counts=law.counts,
            executor_calls=meter.calls,
            executor_seconds=meter.seconds,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            run_id=task["run_id"],
            contract_sha256=contract["contract_sha256"],
            code_revision=task["image_revision"]["commit"],
            oracle_ledger=[
                sha256_file(p) for p in sorted((output / "oracle").glob("*/result.json"))
            ],
            runtime=gate,
        )
        save("result", result)
        return result
    except Exception as error:
        publish_json(
            output / "failure.json", {"error": repr(error), "progress": progress, "at": _stamp()}
        )
        commit()
        raise
    finally:
        stop.set()
        thread.join(timeout=2)


def cases(contract):
    return [
        {"task": task, "arm": arm, "replicate": replicate}
        for task in contract["tasks"]
        for arm in contract["arms"]
        for replicate in contract["replicates"]
    ]


def case_name(case):
    return f"{case['task']}__{case['arm']}__{case['replicate']}"
