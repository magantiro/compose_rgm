"""PHASE A: can COMPOSE find the rare JNK3-active basin from a random-120 start?

DEVELOPMENT-ONLY. The initialization follows the OFFICIAL PROTOCOL -- 120 random
ZINC-250k molecules, no seeded actives -- but uses DEVELOPMENT seeds. The five
official initialization sets stay sealed behind their latch; spending them here
would spend the official experiment.

THE OBJECTIVE IS DISCOVERY, NOT HYPERVOLUME. From a random-120 start JNK3
actives are roughly 2% of draws and a surrogate fit on the initial population is
blind, which is why every earlier experiment used a seeded fixture and deferred
this. The primary readings are therefore evaluations-to-first-active, actives
found, and distinct active scaffolds. Early hypervolume is recorded but is NOT
the criterion: a policy that finds actives faster with worse early HV is the
right thing at this stage, because the whole later pipeline is gated on having
actives at all.

    Arm A  fixed-scalarization   the control, and the best proven selection rule
    Arm B  novelty-exploration   structural coverage, no surrogate

THE FIBER CACHE IS ENABLED HERE, AND IT IS NOW EARNED. Offline measurement shows
the control re-expands the SAME state on 98% of its decisions (1 distinct start
in 40), so a deterministic re-expansion at 12.5 s would be pure waste. The
determinism gate has passed on three separate seeds -- exact fiber equality on
canonical successors AND R_theta probabilities -- so the cache returns what a
recomputation would return. Both arms share it, so it cannot bias the
comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_dir(ROOT / "artifacts" / "oracles" / "molleo_task3_v1",
                   str(REMOTE_ROOT / "artifacts" / "oracles" / "molleo_task3_v1"),
                   copy=True)
    .add_local_dir(ROOT / "artifacts" / "oracles" / "drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts" / "oracles" / "drd2_svm_v1"),
                   copy=True)
    .add_local_dir(ROOT / "artifacts" / "benchmarks" / "molleo_task3_init_v1",
                   str(REMOTE_ROOT / "artifacts" / "benchmarks"
                       / "molleo_task3_init_v1"), copy=True)
)
app = modal.App("compose-v4-task3-phase-a-discovery")

OUTPUT_ROOT = "/artifacts/editing_v2/task3_phase_a"
#: JNK3 activity thresholds. The oracle is a probability, so "active" is a
#: choice; both are reported rather than one being privileged.
ACTIVE_THRESHOLDS = (0.3, 0.5)


@app.function(
    image=image,
    cpu=2.0,
    # 6 GiB and 3 hours: the archive, surrogate matrix and fiber cache all grow
    # with a 2,500-call budget, and the measured 3.9 GiB peak was taken with a
    # small archive. An OOM or timeout two hours into a run costs the whole run.
    memory=6 * 1024,
    timeout=3 * 60 * 60,
    max_containers=6,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_arm(spec: tuple[str, int, int, int]) -> dict[str, Any]:
    arm_name, seed, budget, candidates = spec
    import numpy as np

    from compose_v4.benchmark.init_sets import development_init_set
    from compose_v4.benchmark.molleo_task3 import hypervolume_qmc
    from compose_v4.benchmark.oracles.task3 import navigation_lockout
    from compose_v4.benchmark.task3_run import Task3Run
    from compose_v4.experiments.task3_rtheta_runtime import build_r_theta, expand_fiber
    from compose_v4.policy.task3.archive import ParetoArchive
    from compose_v4.policy.task3.steering import (
        FixedScalarization,
        NoveltyExploration,
    )
    from compose_v4.policy.task3.surrogate import TanimotoKNN

    started = time.perf_counter()
    steering = {"fixed-scalarization": FixedScalarization,
                "novelty-exploration": NoveltyExploration}[arm_name]()
    model = build_r_theta(ARTIFACT_ROOT, REMOTE_ROOT)
    print({"phase": "runtime_built", "arm": arm_name, "seed": seed,
           "seconds": time.perf_counter() - started}, flush=True)

    root = Path(OUTPUT_ROOT) / f"{arm_name}_seed{seed}"
    run = Task3Run.open(root, seed=seed, budget=budget, policy=arm_name)
    archive = ParetoArchive()
    surrogate = TanimotoKNN()
    if run.resumed is not None:
        already = run.meter.evaluated()
        archive.add_many(already)
        surrogate.update(list(already), list(already.values()))

    # The official protocol's initialization, charged exactly as the benchmark
    # charges it: 120 molecules receiving the objective vector is 120 units.
    if not archive.values:
        initial = development_init_set(seed)
        values = run.evaluate(initial)
        for smiles, value in zip(initial, values):
            archive.add(smiles, value)
        surrogate.update(initial, list(values))
        run.step = 1
        run.checkpoint(policy_state={"arm": arm_name, "phase": "initialized"})
        print({"initialized": run.spent, "arm": arm_name, "seed": seed}, flush=True)

    fiber_cache: dict[str, list[str]] = {}
    cache_hits = expansions = 0
    exhausted: set[str] = set()
    try:
        while run.remaining > 0:
            target = steering.target(archive, run.rng)
            if target is None:
                break
            with navigation_lockout():
                proposed: list[str] = []
                start = None
                for _ in range(12):
                    with archive.restricted_to(set(archive.values) - exhausted):
                        start = steering.start(archive, target, run.rng)
                    if start is None:
                        break
                    if start in fiber_cache:
                        children = fiber_cache[start]
                        cache_hits += 1
                    else:
                        try:
                            children = [row[0] for row in expand_fiber(model, start)]
                        except Exception as error:  # noqa: BLE001
                            print({"expansion_failed": start[:50],
                                   "error": type(error).__name__}, flush=True)
                            exhausted.add(start)
                            continue
                        expansions += 1
                        fiber_cache[start] = children
                    proposed = [s for s in dict.fromkeys(children)
                                if s not in archive.values]
                    if proposed:
                        break
                    exhausted.add(start)
                if not proposed:
                    print({"no_eligible_state": True, "arm": arm_name,
                           "seed": seed, "charged": run.spent}, flush=True)
                    break
                predicted, spread = surrogate.predict_with_spread(proposed)
                try:
                    ranks = steering.rank(predicted, target, spread, proposed)
                except TypeError:
                    try:
                        ranks = steering.rank(predicted, target, spread)
                    except TypeError:
                        ranks = steering.rank(predicted, target)
                order = np.argsort(-ranks)
            chosen = run.affordable([proposed[int(i)] for i in order[:candidates]])
            if not chosen:
                break
            values = run.evaluate(chosen)
            for smiles, value in zip(chosen, values):
                archive.add(smiles, value)
            surrogate.update(chosen, list(values))
            run.step += 1
            if run.step % 10 == 0:
                run.archive = [s for s, _ in archive.front()]
                run.checkpoint(policy_state={"arm": arm_name,
                                             "expansions": expansions})
                artifact_volume.commit()
                print({"arm": arm_name, "seed": seed, "step": run.step,
                       "charged": run.spent, "expansions": expansions,
                       "cache_hits": cache_hits,
                       "elapsed_s": round(time.perf_counter() - started, 1)},
                      flush=True)
        run.checkpoint(policy_state={"arm": arm_name, "done": True})

        # ---- DISCOVERY metrics, in ledger order --------------------------
        from compose_v4.benchmark.run_store import iter_ledger
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        order_rows = list(iter_ledger(run.store.ledger_path))
        jnk3 = [row["v"][1] for row in order_rows]
        report: dict[str, Any] = {
            "arm": arm_name, "seed": seed, "charged": run.spent,
            "expansions": expansions, "cache_hits": cache_hits,
            "distinct_starts_expanded": len(fiber_cache),
            "seconds": time.perf_counter() - started,
        }
        for threshold in ACTIVE_THRESHOLDS:
            hits = [i for i, value in enumerate(jnk3) if value >= threshold]
            scaffolds = set()
            for i in hits:
                mol = Chem.MolFromSmiles(order_rows[i]["smiles"])
                if mol is not None:
                    try:
                        scaffolds.add(Chem.MolToSmiles(
                            MurckoScaffold.GetScaffoldForMol(mol)))
                    except Exception:  # noqa: BLE001
                        pass
            key = str(threshold).replace(".", "")
            report[f"evals_to_first_active_{key}"] = (hits[0] + 1) if hits else None
            report[f"actives_{key}"] = len(hits)
            report[f"active_scaffolds_{key}"] = len(scaffolds)
        points = np.asarray([row["v"] for row in order_rows])
        report["early_hv"] = hypervolume_qmc(points, log2_samples=17)
        report["best_jnk3"] = float(max(jnk3)) if jnk3 else 0.0
        print(json.dumps(report), flush=True)
        artifact_volume.commit()
        return report
    finally:
        run.close()


@app.local_entrypoint()
def main(budget: int = 300, candidates: int = 4, seeds: str = "100,101,102") -> None:
    chosen = [int(s) for s in seeds.split(",")]
    specs = [(arm, seed, budget, candidates)
             for seed in chosen
             for arm in ("fixed-scalarization", "novelty-exploration")]
    print(json.dumps({"phase": "launching_phase_a", "runs": len(specs),
                      "budget_per_arm": budget,
                      "note": ("DEVELOPMENT-ONLY. Official PROTOCOL (random-120) "
                               "on DEVELOPMENT seeds; the official sets stay "
                               "sealed.")}, indent=1))
    results = list(run_arm.map(specs))
    by_arm: dict[str, list[dict]] = {}
    for row in results:
        by_arm.setdefault(row["arm"], []).append(row)
    print("\n--- PHASE A: DISCOVERY (development-only; HV is NOT the criterion) ---")
    for arm, rows in sorted(by_arm.items()):
        rows.sort(key=lambda r: r["seed"])
        first = [r["evals_to_first_active_03"] for r in rows]
        print(f"\n{arm}")
        print(f"  evals to first active (jnk3>=0.3): {first}")
        print(f"  actives   >=0.3: {[r['actives_03'] for r in rows]}   "
              f">=0.5: {[r['actives_05'] for r in rows]}")
        print(f"  scaffolds >=0.3: {[r['active_scaffolds_03'] for r in rows]}")
        print(f"  early HV (not the criterion): "
              f"{[round(r['early_hv'], 4) for r in rows]}")
        print(f"  expansions {[r['expansions'] for r in rows]}  "
              f"cache hits {[r['cache_hits'] for r in rows]}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/task3_phase_a_discovery.json").write_text(
        json.dumps({"STATUS": "DEVELOPMENT-ONLY -- official protocol, development "
                              "seeds; official sets remain sealed",
                    "objective": "DISCOVERY, not hypervolume",
                    "results": results}, indent=1) + "\n")
