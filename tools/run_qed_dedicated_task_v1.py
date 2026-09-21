"""Constrained-QED editing run driven by the DEDICATED task (not a T4 label).

NEW FILE. Nothing under src/compose_v4/control, src/compose_v4/experiments,
modal_apps or an existing configs entry is modified.

WHAT CHANGED VERSUS tools/run_qed_griddd_controller_comparison_v1.py
---------------------------------------------------------------------
* The similarity floor moved from the REWARD into the SUPPORT.  The task's
  endpoint evaluator sets oracle_eligible, and propose_batch drops ineligible
  endpoints before they can become candidates, so they never spend budget.  The
  reward is QED itself -- no indicator, no blended scalar, no tunable lambda.
* Thresholds come from configs/qed_dedicated_task_v1.json.  The dataclass has no
  defaults for them, so an incomplete contract is a TypeError.
* The score direction is RESOLVED at construction by interrogating the frozen
  guard in parent_edit_search, not assumed.
* Allocated graph slots (48) and the active heavy-atom ceiling (40) are separate
  named quantities.

This tool charges no paid oracle and performs no Modal operation.  QED and
Morgan-Tanimoto are local RDKit calls; the currency is CPU time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path

PROTOCOL = "qed-dedicated-task-v1"
CONTRACT = "configs/qed_dedicated_task_v1.json"

# Per-process work ledger, populated by the recording task below.
_WORK: dict = {"evaluated": []}


def source_seed(source: str, index: int) -> int:
    payload = f"{PROTOCOL}|{source}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")



def entry_structure(entries, observations) -> list[dict]:
    """Per-entry program structure from a campaign snapshot.

    ``trace["primitive_edits"]`` is an integer COUNT, not a list -- reading it as
    a list silently yields a null depth for every entry.
    """
    score_by_endpoint = {
        row["endpoint"]: row["score"]
        for row in observations.values()
        if isinstance(row, dict) and "endpoint" in row and "score" in row
    }
    structure = []
    if not isinstance(entries, dict):
        return structure
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        program = entry.get("program") or {}
        blocks = program.get("blocks") if isinstance(program, dict) else None
        trace = entry.get("trace") or {}
        provenance = entry.get("provenance") or {}
        size = provenance.get("program_size") or {}
        edits, states = trace.get("primitive_edits"), trace.get("states")
        depth = (
            edits
            if isinstance(edits, int) and not isinstance(edits, bool)
            else (len(states) - 1 if isinstance(states, list) else None)
        )
        endpoint = entry.get("endpoint")
        child, parent = score_by_endpoint.get(endpoint), provenance.get("parent_measured_score")
        structure.append(
            {
                "endpoint": endpoint,
                # Block LABELS, not "name": the program payload keys them as "label".
                "blocks": [b.get("label") for b in blocks] if isinstance(blocks, list) else [],
                "n_blocks": len(blocks) if isinstance(blocks, list) else 0,
                "primitive_depth": depth,
                "channel": provenance.get("channel"),
                "delta_heavy_atoms": size.get("delta_heavy_atoms"),
                "parent_measured_score": parent,
                "child_score": child,
                "parent_to_child_improvement": (
                    float(child) - float(parent)
                    if isinstance(child, (int, float)) and isinstance(parent, (int, float))
                    else None
                ),
            }
        )
    return structure


def _recording_task_class():
    from compose_v4.tasks.qed_edit_task import QedEditTask

    @dataclass(frozen=True)
    class RecordingQedEditTask(QedEditTask):
        """Identical task (same fields, same task_id); records the work it sees.

        Every endpoint the proposal path executes is evaluated exactly once here,
        which is what makes 'distinct molecules property-evaluated' measurable
        rather than inferred.
        """

        def endpoint_evaluator(self):
            inner = super().endpoint_evaluator()

            def evaluate(row):
                out = inner(row)
                _WORK["evaluated"].append(
                    (out.get("smiles"), out["qed"], out["sim"], out["oracle_eligible"])
                )
                return out

            return evaluate

    return RecordingQedEditTask


def run_one(job: dict) -> dict:
    index = int(job["index"])
    source = job["source"]
    out_root = Path(job["out_root"])
    record_path = out_root / f"{index:03d}.json"
    if record_path.exists() and not job.get("force"):
        try:
            return json.loads(record_path.read_text())
        except Exception:  # noqa: BLE001, S110 - a truncated record is recomputed
            pass

    _WORK["evaluated"] = []
    started = time.perf_counter()
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
        from compose_v4.control.dynamic_program_synthesis_v21 import (
            initial_dynamic_program_batch_v21,
        )
        from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
        from compose_v4.control.program_task import initialization_lock
        from compose_v4.rewrite.trace_shard import encode_state
        from compose_v4.tasks.qed_edit_task import (
            load_qed_edit_contract,
            production_source_state,
            resolve_score_direction,
        )

        contract = load_qed_edit_contract(job["contract"])
        task_cls = _recording_task_class()
        task = task_cls(
            name=f"qed_jin_{index:03d}",
            oracle_protocol=job["oracle_protocol"],
            original_source=source,
            qed_target=float(contract["region"]["qed_target"]),
            similarity_floor=float(contract["region"]["similarity_floor"]),
            fingerprint_radius=int(contract["similarity"]["radius"]),
            fingerprint_bits=int(contract["similarity"]["bits"]),
            archive_top_k=int(contract["archive_top_k"]),
        )
        direction = resolve_score_direction(task)

        seed = source_seed(source, index)
        state = production_source_state(source)
        initialization = initialization_lock(
            [{"state": encode_state(state), "source_id": f"jin{index:03d}"}],
            count=1,
            seed=seed % (2**31),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        )
        budget = int(job["budget"])
        reward = task.reward()
        ledger = ProgramQueryLedger(
            out_root / "oracle" / f"{index:03d}", task, reward, budget=budget
        )
        config = replace(
            ProgramSearchConfig.program_only_recipe(seed=seed % (2**31), score_direction=direction),
            attempts_per_batch=int(job["attempts_per_batch"]),
            candidates_per_batch=int(job["candidates_per_batch"]),
            wall_seconds=float(job["wall_seconds"]),
            proposal_cache_entries=128,
            require_broad_runtime=False,
        )
        rounds_seen: list[dict] = []
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
            progress=lambda row: rounds_seen.append(
                {
                    k: row.get(k)
                    for k in ("phase", "round", "proposal_attempts", "pool_size",
                              "queried_candidates", "top_k_utility", "best_new_utility")
                }
            ),
        )

        evaluated = _WORK["evaluated"]
        distinct = {}
        for smiles, qed, sim, eligible in evaluated:
            distinct.setdefault(smiles, (qed, sim, eligible))
        admitted = sum(1 for _, _, _, e in evaluated if e)
        rows = [r for r in ledger.rows if r.get("status") == "complete"]
        scored = []
        for order, row in enumerate(rows):
            detail = distinct.get(row["endpoint"])
            if detail is None:
                # The initialization endpoint is charged through the ledger but is
                # never proposed, so it never reaches the endpoint evaluator.
                # Defaulting it to 0.0/0.0 would silently misreport it.
                measured = task.measure(row["endpoint"])
                qed, sim = measured["qed"], measured["sim"]
            else:
                qed, sim, _eligible = detail
            scored.append(
                {
                    "order": order,
                    "smiles": row["endpoint"],
                    "score": float(row["score"]),
                    "qed": float(qed),
                    "sim": float(sim),
                    "role": row.get("role"),
                    "success": bool(qed >= task.qed_target and sim >= task.similarity_floor),
                }
            )
        snapshot = campaign.get("snapshot", {})
        structure = entry_structure(
            snapshot.get("entries", {}), snapshot.get("observations", {}) or {}
        )

        record = {
            "schema_version": "qed_dedicated_task_source_v1",
            "protocol": PROTOCOL,
            "index": index,
            "source": source,
            "seed": seed,
            "task_id": task.task_id,
            "score_direction_resolved": direction,
            "region": {
                "qed_target": task.qed_target,
                "similarity_floor": task.similarity_floor,
            },
            "budget": budget,
            "rounds": int(job["rounds"]),
            "queries_per_round": int(job["queries_per_round"]),
            "candidates_per_batch": int(job["candidates_per_batch"]),
            "work": {
                "proposal_attempts": sum(
                    int(r.get("proposal_attempts") or 0) for r in rounds_seen
                ),
                "executed_endpoint_evaluations": len(evaluated),
                "distinct_endpoints_evaluated": len(distinct),
                "admitted_endpoint_evaluations": admitted,
                "admitted_fraction": (admitted / len(evaluated)) if evaluated else None,
                "charged_scored_endpoints": len(rows),
                "zero_scored_charged": sum(1 for s in scored if s["score"] <= 0.0),
                "seconds": time.perf_counter() - started,
            },
            "scored": scored,
            # Ordered inspection stream: [qed, sim, eligible] per executed endpoint.
            # This is what makes success-versus-INSPECTED-WORK computable.
            "evaluation_stream": [
                [round(float(q), 4), round(float(m), 4), int(bool(e))]
                for _s, q, m, e in evaluated
            ],
            "best_evaluated": sorted(
                (
                    {"smiles": s, "qed": q, "sim": m, "eligible": e}
                    for s, (q, m, e) in distinct.items()
                ),
                key=lambda r: -r["qed"],
            )[:50],
            "structure": structure[:400],
            "rounds_seen": rounds_seen,
            "campaign_oracle_calls": campaign.get("oracle_calls"),
            "status": "complete",
        }
    except Exception as error:  # noqa: BLE001 - one bad source must not kill the panel
        record = {
            "schema_version": "qed_dedicated_task_source_v1",
            "protocol": PROTOCOL,
            "index": index,
            "source": source,
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc()[-2500:],
            "work": {"seconds": time.perf_counter() - started},
        }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/jin/qed_test.txt")
    parser.add_argument("--contract", default=CONTRACT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--indices", default="", help="comma-separated explicit source indices")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=800)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--queries-per-round", type=int, required=True)
    parser.add_argument("--attempts-per-batch", type=int, default=64)
    parser.add_argument(
        "--candidates-per-batch",
        type=int,
        default=0,
        help="proposal pool size per round; 0 means equal to --queries-per-round. "
             "Raising it increases INSPECTED work at a fixed charged budget, which is "
             "the knob the work-matched comparison needs.",
    )
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--oracle-protocol", default="local:qed-free-rdkit-v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = Path(args.sources).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    smiles = [ln.strip() for ln in raw.decode().split("\n") if ln.strip()]
    if len(smiles) != 800:
        raise SystemExit(f"expected the 800-source Jin panel, found {len(smiles)}")
    print(f"panel sha256 {digest}", flush=True)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "panel.json").write_text(
        json.dumps({"sha256": digest, "count": len(smiles), "path": args.sources})
    )
    indices = (
        [int(x) for x in args.indices.split(",") if x.strip()]
        if args.indices
        else list(range(args.first, args.last))
    )
    jobs = [
        {
            "index": i,
            "source": smiles[i],
            "out_root": str(out_root),
            "contract": args.contract,
            "oracle_protocol": args.oracle_protocol,
            "budget": args.budget,
            "rounds": args.rounds,
            "queries_per_round": args.queries_per_round,
            "candidates_per_batch": args.candidates_per_batch or args.queries_per_round,
            "attempts_per_batch": args.attempts_per_batch,
            "wall_seconds": args.wall_seconds,
            "force": args.force,
        }
        for i in indices
    ]
    print(f"dedicated QED task: {len(jobs)} sources, budget {args.budget}, "
          f"{args.workers} workers", flush=True)
    done, began = 0, time.perf_counter()
    if args.workers <= 1:
        for job in jobs:
            record = run_one(job)
            done += 1
            print(f"  {done}/{len(jobs)} idx={record['index']} {record['status']}", flush=True)
    else:
        with Pool(processes=args.workers) as pool:
            for record in pool.imap_unordered(run_one, jobs, chunksize=1):
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    rate = (time.perf_counter() - began) / done
                    print(f"  {done}/{len(jobs)} {rate:.1f}s/source wall "
                          f"last={record['index']}:{record['status']}", flush=True)
    print("dedicated QED task run complete", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
