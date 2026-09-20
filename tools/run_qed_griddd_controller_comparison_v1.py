"""QED / GrIDDD-Jin controller comparison, arm B: structural program + FiberControl.

NEW FILE. Nothing under ``src/compose_v4/control``, ``src/compose_v4/experiments``,
``modal_apps`` or an existing ``configs`` entry is touched; every frozen module is
imported, never edited.

WHAT THIS RUNS
--------------
Arm A (paper-era COMPOSE, region ``h_phi`` + twisted SMC) is NOT re-run here: its
798 per-source records are banked on the Modal volume at
``editing_v2/r_theta_run/hphi_official800_k8`` and are scored read-only by
``tools/score_qed_griddd_arm_a_v1.py``.  Re-running arm A would require a Modal
launch, which is barred.

Arm B is the current controller: ``run_program_campaign`` driving the structural
program synthesiser over the exact executor, with the QED objective injected
through the ``evaluate`` callable of ``ProgramQueryLedger``.

THE OBJECTIVE IS THE BENCHMARK CONJUNCTION, NOT A BLEND
-------------------------------------------------------
``docs/GRIDDD_JIN_PROTOCOL.md`` bars inventing a combined scalar
(``QED - lambda*(1-sim)``).  The injected scorer is

    score(y) = QED(y) * 1[ Tanimoto(y, x0) >= 0.4 ]

an indicator times a reward -- the FiberControl shape (feasibility constrains the
support, reward decides preference inside it) -- and introduces no tunable
parameter.  It is in ``[0, 1]``, which ``ProgramTask.utility`` requires for a
``pmo``-kind task, and the direction is ``maximize``.

WHY ``kind="pmo"`` AND NOT ``kind="t4"``
----------------------------------------
``ProgramTask(kind="t4")`` carries ``original_seed``/``delta`` and would give the
similarity floor as a genuine support constraint, which is the better shape.  It
is NOT used because ``ProgramTask.endpoint_evaluator`` then applies the T4
medicinal-chemistry gate with ``qed_min=0.6`` and ``sa_max=4.0`` HARDCODED
(``src/compose_v4/control/program_task.py:76-77``).  ``sa_max=4.0`` is not part of
the GrIDDD/Jin task: a candidate with QED 0.95, similarity 0.5 and SA 4.3 solves
the benchmark but would be refused as an ineligible endpoint.  Importing that gate
would silently change the task, so the similarity floor is carried in the scorer
instead and the deviation is recorded rather than absorbed.

SLOT SEMANTICS
--------------
Sources are built with ``production_state_from_smiles(smiles, 48)``.  A TIGHT graph
from ``smiles_to_molecular_graph`` deletes the whole ``atom_insert`` family from
the legal support.  48 -- not the 40 of ``PRODUCTION_MAX_ATOMS`` -- because
``whole_ring_plan.execute_program`` requires an exact supported 48-slot source and
refuses anything else; the paper-era app uses ``CANONICAL_SLOTS = 48`` as well.
All 800 sources are at most 33 heavy atoms, so every one keeps free slots.

SEEDS
-----
Per source, ``uint64(sha256(protocol | source | index)[:8])``, mirroring the frozen
seed manifest of the protocol.  Never Python's process-salted ``hash()``.

This tool charges no paid oracle: QED and Tanimoto are local RDKit calls.  It
performs no Modal operation of any kind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

PROTOCOL = "qed-griddd-controller-comparison-v1"
REGION_QED = 0.90
REGION_SIM = 0.40
CANONICAL_SLOTS = 48


# ---- Seeds ----


def source_seed(source: str, index: int) -> int:
    payload = f"{PROTOCOL}|{source}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# ---- One source ----


def run_one(job: dict) -> dict:
    """Run arm B on a single source. Returns a record; never raises to the pool."""

    index = int(job["index"])
    source = job["source"]
    out_root = Path(job["out_root"])
    record_path = out_root / f"{index:03d}.json"
    if record_path.exists():
        try:
            return json.loads(record_path.read_text())
        except Exception:  # noqa: BLE001 - a truncated record is simply recomputed
            pass

    started = time.perf_counter()
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import QED, rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
        from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
        from compose_v4.control.dynamic_program_synthesis_v21 import (
            initial_dynamic_program_batch_v21,
        )
        from compose_v4.control.program_campaign import (
            ProgramQueryLedger,
            run_program_campaign,
        )
        from compose_v4.control.program_task import ProgramTask, initialization_lock
        from compose_v4.experiments.editing_v2_evaluation_semantics import (
            production_state_from_smiles,
        )
        from compose_v4.rewrite.trace_shard import encode_state

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        source_fp = generator.GetFingerprint(Chem.MolFromSmiles(source))
        observed: list[dict] = []

        def evaluate(smiles: str) -> float:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                observed.append({"smiles": smiles, "qed": 0.0, "sim": 0.0, "valid": False})
                return 0.0
            qed = float(QED.qed(molecule))
            similarity = float(
                DataStructs.TanimotoSimilarity(source_fp, generator.GetFingerprint(molecule))
            )
            observed.append(
                {"smiles": smiles, "qed": qed, "sim": similarity, "valid": True}
            )
            # Indicator times reward. Similarity constrains the support; QED ranks it.
            return qed if similarity >= REGION_SIM else 0.0

        seed = source_seed(source, index)
        state = production_state_from_smiles(source, CANONICAL_SLOTS)
        initialization = initialization_lock(
            [{"state": encode_state(state), "source_id": f"jin{index:03d}"}],
            count=1,
            seed=seed % (2**31),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        )
        task = ProgramTask(f"qed_jin_{index:03d}", "local:qed-free-rdkit-v1", "pmo")
        budget = int(job["budget"])
        ledger = ProgramQueryLedger(
            out_root / "oracle" / f"{index:03d}", task, evaluate, budget=budget
        )
        config = replace(
            ProgramSearchConfig.program_only_recipe(
                seed=seed % (2**31), score_direction="maximize"
            ),
            attempts_per_batch=int(job["attempts_per_batch"]),
            candidates_per_batch=int(job["queries_per_round"]),
            wall_seconds=float(job["wall_seconds"]),
            proposal_cache_entries=128,
            require_broad_runtime=False,
        )
        campaign = run_program_campaign(
            output=out_root / "campaign" / f"{index:03d}",
            task=task,
            config=config,
            initialization=initialization,
            library=(),
            ledger=ledger,
            rounds=int(job["rounds"]),
            queries_per_round=int(job["queries_per_round"]),
            fit_model=None,
            bootstrap_rounds=1,
            initialization_mode="all_scored_pool",
            initial_parent_fraction=0.2,
            initial_batch_fn=initial_dynamic_program_batch_v21,
        )
        rows = [r for r in ledger.rows if r.get("status") == "complete"]
        scored = []
        by_smiles = {o["smiles"]: o for o in observed}
        for order, row in enumerate(rows):
            detail = by_smiles.get(row["endpoint"], {})
            qed = float(detail.get("qed", 0.0))
            similarity = float(detail.get("sim", 0.0))
            scored.append(
                {
                    "order": order,
                    "smiles": row["endpoint"],
                    "score": float(row["score"]),
                    "qed": qed,
                    "sim": similarity,
                    "role": row.get("role"),
                    "success": bool(qed >= REGION_QED and similarity >= REGION_SIM),
                }
            )
        record = {
            "schema_version": "qed_griddd_arm_b_source_v1",
            "protocol": PROTOCOL,
            "index": index,
            "source": source,
            "seed": seed,
            "region": {"qed_min": REGION_QED, "tanimoto_min": REGION_SIM},
            "budget": budget,
            "rounds": int(job["rounds"]),
            "queries_per_round": int(job["queries_per_round"]),
            "charged_scored_endpoints": len(rows),
            "proposed_evaluations": len(observed),
            "scored": scored,
            "campaign_oracle_calls": campaign.get("oracle_calls"),
            "seconds": time.perf_counter() - started,
            "status": "complete",
        }
    except Exception as error:  # noqa: BLE001 - one bad source must not kill the panel
        record = {
            "schema_version": "qed_griddd_arm_b_source_v1",
            "protocol": PROTOCOL,
            "index": index,
            "source": source,
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc()[-2000:],
            "seconds": time.perf_counter() - started,
        }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


# ---- Entry point ----


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/jin/qed_test.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=800)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--queries-per-round", type=int, required=True)
    parser.add_argument("--attempts-per-batch", type=int, default=64)
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    smiles = [ln.strip() for ln in Path(args.sources).read_text().split("\n") if ln.strip()]
    if len(smiles) != 800:
        raise SystemExit(f"expected the 800-source Jin panel, found {len(smiles)}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "index": i,
            "source": smiles[i],
            "out_root": str(out_root),
            "budget": args.budget,
            "rounds": args.rounds,
            "queries_per_round": args.queries_per_round,
            "attempts_per_batch": args.attempts_per_batch,
            "wall_seconds": args.wall_seconds,
        }
        for i in range(args.first, args.last)
    ]
    print(f"arm B: {len(jobs)} sources, budget {args.budget}, {args.workers} workers",
          flush=True)
    done = 0
    began = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        for record in pool.imap_unordered(run_one, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                rate = (time.perf_counter() - began) / done
                print(f"  {done}/{len(jobs)}  {rate:.1f}s/source wall"
                      f"  last={record['index']}:{record['status']}", flush=True)
    print("arm B complete", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
