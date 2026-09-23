"""Matched Celecoxib canary: production PMO FiberControl vs contextual Q_post acquisition.

ARM A  the production controller exactly as it ships (`_credit_allocate`).
ARM B  the SAME production proposal generation, the SAME ledger, budget and initialization, with
       oracle acquisition replaced by the contextual `Q_post` model plus an audit quota.

SCOPE, DECLARED BEFORE THE RUN.  This canary tests `Q_post` -- the stage whose held-out signal is
strong (0.849 pairwise, +0.857 Spearman on unseen start lineages).  `Q_pre`'s influence on
GENERATION is deliberately NOT in this arm: its measured pre-execution signal is 0.690 and it
shifts parent allocation by only 0.028, so including it would add a weak, unmeasured factor to a
bounded test and make a null impossible to attribute.  If B wins, the honest claim is query
enrichment, not proposal composition.

EVERYTHING ELSE IS HELD FIXED: same contract, same initialization, same seed, same charged budget,
same proposal path, same oracle. The arms differ in exactly one substitution.

Exploration in B is the AUDIT QUOTA, not the model's spread. With a shared constant spread the
expected archive gain is monotone in the predicted mean, so it reorders nothing on its own -- the
audit fraction is doing the real exploration and is reported as such.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import types
from pathlib import Path

import numpy as np

OUT = "diagnostics/pmo_contextual_canary_v1"


def _oracle():
    stub = types.ModuleType("rdkit.six")
    stub.string_types = (str,)
    stub.iteritems = lambda d: iter(d.items())
    import rdkit

    sys.modules["rdkit.six"] = stub
    rdkit.six = stub
    from tdc import Oracle

    return Oracle(name="celecoxib_rediscovery")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = int(args[0]) if args else 250
    rounds = int(args[1]) if len(args) > 1 else 24
    queries = int(args[2]) if len(args) > 2 else 16
    pool_target = int(args[3]) if len(args) > 3 else 128

    from compose_v4.control.docking_value import identity
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.control.pmo_two_stage_control import TwoStageControl
    from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
    from compose_v4.control.program_task import ProgramTask, archive_top_k
    from compose_v4.experiments import pmo_population_v1 as production
    from compose_v4.experiments.pmo_dynamic_v21 import initial_dynamic_program_batch_v21

    root, score = Path("."), _oracle()
    contract = production.load_contract(root)
    initialized = production._load_initialization(
        root, {"initialization": contract["initialization"]}
    )
    checkpoint = production._load_checkpoint(root, contract)
    task_name = "celecoxib_rediscovery"

    def build(arm, folder):
        from dataclasses import replace as _replace

        config = production.configuration(contract["controller"]["seed"])
        # DECOUPLE generation from querying. `propose_batch` breaks as soon as it holds
        # `candidates_per_batch` candidates, so the pool has always been exactly the query
        # batch -- the controller generated precisely what it intended to buy and no selector
        # could express anything. Raising the generation target while capping the QUERY at
        # `queries` is what gives acquisition something to reject.
        if arm != "A_production":
            config = _replace(config, candidates_per_batch=pool_target)
        task = ProgramTask(
            task_name, identity(production._runtime_protocol(contract, task_name)), "pmo"
        )
        ledger = ProgramQueryLedger(
            folder / "oracle", task, lambda s: float(score(s)), budget=budget
        )
        telemetry = {"rounds": [], "arm": arm}

        class ContextualAcquisition(PmoPopulationController):
            """Production proposals; contextual Q_post decides which are worth an oracle call."""

            control = TwoStageControl(audit_fraction=0.15, lineage_floor=0.10)
            control._seen: set = set()  # noqa: RUF012

            def _allocate(self, candidates):
                if arm == "A_production" or not candidates:
                    return super()._allocate(candidates)
                room = min(queries, len(candidates))
                observed = [
                    (o["endpoint"], float(o["score"])) for o in self.observations.values()
                ]
                rows = [
                    {
                        "parent": c["provenance"].get("parent_smiles", c["endpoint"]),
                        "endpoint": c["endpoint"],
                        "smiles": c["endpoint"],
                        "parent_score": float(
                            c["provenance"].get("parent_measured_score") or 0.0
                        ),
                        "families": list(
                            (c.get("program_rule_histogram") or {}).keys()
                        ) or [str(c["provenance"].get("planner_channel", ""))],
                        "requested_modules": len(c.get("program", {}).get("blocks", ()) or ()),
                        "module_count": len(c.get("program", {}).get("blocks", ()) or ()),
                        "primitives": len(c.get("trace", {}).get("actions", ()) or ()),
                        "depth": 1,
                        "generation": self.batches,
                        "capacity_aware": False,
                    }
                    for c in candidates
                ]
                if arm == "R_broad_random":
                    picked = self.rng.choice(len(rows), min(room, len(rows)), replace=False)
                    chosen = [int(i) for i in np.atleast_1d(picked)]
                    detail = [{"index": i, "reason": "random", "propensity": room / len(rows)}
                              for i in chosen]
                else:
                    chosen, detail = self.control.acquire(
                        rows, observed, batch=room, rng=self.rng
                    )
                truth_pool = None
                telemetry["rounds"].append(
                    {
                        "batch": self.batches,
                        "pool": len(candidates),
                        "selected": len(chosen),
                        "fitted": self.control.fitted,
                        "audit_slots": sum(1 for d in detail if d["reason"] == "audit"),
                        "observations": len(self.control.observations),
                        "query_fraction": len(chosen) / max(len(candidates), 1),
                    }
                )
                del truth_pool
                return [candidates[i] for i in chosen], {
                    "mode": "random_broad" if arm == "R_broad_random" else "contextual_q_post",
                    "selected_ids": [candidates[i]["candidate_id"] for i in chosen],
                }

            def observe_batch(self, batch_id, outcomes):
                """Every CHARGED outcome updates Q_post; this is what makes it online."""
                result = super().observe_batch(batch_id, outcomes)
                if arm != "A_production":
                    for observation in self.observations.values():
                        key = observation["endpoint"]
                        if key in self.control._seen:
                            continue
                        self.control._seen.add(key)
                        self.control.observe(
                            {
                                "parent": key, "endpoint": key, "smiles": key,
                                "parent_score": 0.0, "families": [], "requested_modules": 0,
                                "module_count": 0, "primitives": 0, "depth": 1,
                                "generation": self.batches, "capacity_aware": False,
                            },
                            float(observation["score"]),
                        )
                return result

        return task, ledger, config, ContextualAcquisition, telemetry

    results = {}
    for arm in ("R_broad_random", "B_contextual"):
        folder = pathlib.Path(OUT) / arm
        folder.mkdir(parents=True, exist_ok=True)
        task, ledger, config, optimizer, telemetry = build(arm, folder)
        started = time.time()
        print(f"\n=== {arm} : budget {budget} ===", flush=True)
        run_program_campaign(
            output=folder / "campaign", task=task, config=config,
            initialization=initialized, library=(), ledger=ledger,
            rounds=rounds, queries_per_round=queries, hierarchy=None, fit_model=None,
            stagnation_rounds=None, bootstrap_rounds=1,
            initialization_mode="all_scored_pool", initial_parent_fraction=0.2,
            progress=lambda row: None,
            optimizer_type=optimizer, optimizer_kwargs={"jump_checkpoint": checkpoint},
            initial_batch_fn=initial_dynamic_program_batch_v21,
        )
        values = [r["score"] for r in ledger.rows]
        curve = []
        for i in range(1, len(values) + 1):
            observed = [(r["endpoint"], r["score"]) for r in ledger.rows[:i]]
            curve.append(
                {"calls": i, "best": max(values[:i]),
                 "top10": archive_top_k(observed, k=10)}
            )
        results[arm] = {
            "charged_calls": len(ledger.rows),
            "best_score": max(values) if values else None,
            "final_top10": curve[-1]["top10"] if curve else None,
            "auc_top10": float(np.mean([c["top10"] for c in curve])) if curve else None,
            "improving_rate": float(np.mean([v > 0 for v in values])) if values else None,
            "curve": curve,
            "telemetry": telemetry,
            "seconds": round(time.time() - started, 1),
        }
        r = results[arm]
        print(f"{arm}: charged {r['charged_calls']} best {r['best_score']:.4f} "
              f"top10 {r['final_top10']:.4f} auc {r['auc_top10']:.4f}", flush=True)
        with open(pathlib.Path(OUT) / "canary_v1.json", "w") as handle:
            json.dump({"budget": budget, "queries_per_round": queries, "pool_target": pool_target, "arms": results}, handle, indent=1)
    print("WROTE", pathlib.Path(OUT) / "canary_v1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
