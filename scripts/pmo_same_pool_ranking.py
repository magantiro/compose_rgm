"""Given the SAME candidates, does FiberControl rank as well as an oracle-informed top-k?

The blind beam established that a trivial greedy selector over the production proposal stack climbs
to ~0.44 and then plateaus.  That says nothing about FiberControl, because the beam exposes neither
the channel structure nor the retention features `population_features` is built around.  This run
closes that gap without changing anything about the controller.

DESIGN.  The PRODUCTION controller runs normally, so every pool carries genuine channel labels,
retention features, program metadata and parent identity -- all sixteen feature dimensions are live
and none is degenerate.  At each allocation the pool is FROZEN and every candidate in it is scored.
Two selections are then recorded over those identical candidates:

    level 1   oracle top-k        picks the k best by TRUE objective value
    level 2   FiberControl        the real `_fit_value -> acquisition` / `_credit_allocate` path

ONLY FiberControl's selection drives the campaign forward, so neither selector alters future
proposal allocation and the pool sequence is a property of the controller, not of the comparison.
Sequential state is therefore exactly what the controller accumulated -- this is not a stateless
call to `acquisition` on an isolated pool.

EVALUATION ACCOUNTING.  Scoring a whole pool costs objective evaluations that the controller does
not charge.  They are MEASUREMENT, not selection, and they are counted and reported separately from
the ledger's charged calls.  Neither figure may reach an AUC or a scored ledger: this is a labelled
development diagnostic.

SIGN CONVENTION.  Untouched.  The controller negates the archive and the parent offset while
leaving fitted improvements in raw maximize form; this harness never reimplements that, it only
observes what the controller chose.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import types
from pathlib import Path

OUT = "diagnostics/pmo_same_pool_ranking_v1/same_pool_ranking_v1.json"

#: Optimism strengths to replay, in ABSOLUTE units so they can be read against the measured
#: mean positive improvement of ~0.0077 per productive child. 0.25 is what ships.
OPTIMISM_SWEEP = (0.25, 0.05, 0.02, 0.0077, 0.002, 0.0)


def _credit_key_of(candidates, endpoint):
    """The credit cell a chosen endpoint belongs to, via the production key function."""
    from compose_v4.control.pmo_credit import credit_key_from_candidate

    for row in candidates:
        if row["endpoint"] == endpoint:
            return credit_key_from_candidate(row)
    raise KeyError(endpoint)


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
    rounds = int(args[0]) if args else 12
    queries = int(args[1]) if len(args) > 1 else 16
    budget = int(args[2]) if len(args) > 2 else 200

    import numpy as np

    from compose_v4.control.docking_value import identity
    from compose_v4.control.fiber_control import acquisition
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
    from compose_v4.control.program_task import ProgramTask
    from compose_v4.experiments import pmo_population_v1 as production
    from compose_v4.experiments.pmo_dynamic_v21 import (
        initial_dynamic_program_batch_v21,
    )

    root = Path(".")
    score = _oracle()
    contract = production.load_contract(root)
    initialized = production._load_initialization(
        root, {"initialization": contract["initialization"]}
    )
    checkpoint = production._load_checkpoint(root, contract)
    config = production.configuration(contract["controller"]["seed"])
    task_name = "celecoxib_rediscovery"
    task = ProgramTask(
        task_name, identity(production._runtime_protocol(contract, task_name)), "pmo"
    )

    comparisons, diagnostic_evaluations = [], 0

    class Observed(PmoPopulationController):
        """The production controller, observed at the allocation seam and never altered."""

        def _allocate(self, candidates):
            nonlocal diagnostic_evaluations
            selected, detail = super()._allocate(candidates)
            if candidates:
                # ARM 2, measured at the same seam and never allowed to drive: the SHARED
                # FiberControl path alone -- the controller's own fitted value and accumulated
                # SearchState, ranked by `acquisition` over the whole pool with NO pmo_credit,
                # no niches and no channel quotas. A separate RNG keeps the live stream intact.
                shared_value, shared_state = self._fit_value()
                prepared = [
                    {
                        "features": np.asarray(row["fiber_features"], dtype=float),
                        "fingerprint": set(row["fiber_fingerprint"]),
                        "parent_score": -float(
                            row["provenance"].get("parent_measured_score", 0.0)
                        ),
                    }
                    for row in candidates
                ]
                shared_idx = acquisition(
                    prepared,
                    shared_value,
                    shared_state,
                    np.random.default_rng(20260923),
                    batch=len(selected),
                    diversity=0.5,
                    exploration=min(2, len(selected)),
                )
                shared_rows = [candidates[i] for i in shared_idx]

                # ZERO-NEW-ORACLE optimism sweep: the SAME pool and the SAME accumulated credit
                # evidence, replayed with only `prior_weight` varying. The live arm keeps the
                # shipped value; each replay restores it, so the campaign is never perturbed.
                sweep = {}
                if self.credit.cells:
                    shipped = self.credit.prior_weight
                    try:
                        for weight in OPTIMISM_SWEEP:
                            self.credit.prior_weight = float(weight)
                            rows, _detail = self._credit_allocate(
                                candidates, shared_value, len(selected)
                            )
                            sweep[str(weight)] = [r["endpoint"] for r in rows]
                    finally:
                        self.credit.prior_weight = shipped
                truth = {}
                for row in candidates:
                    endpoint = row["endpoint"]
                    if endpoint not in truth:
                        truth[endpoint] = float(score(endpoint))
                        diagnostic_evaluations += 1
                room = len(selected)
                ordered = sorted(candidates, key=lambda r: -truth[r["endpoint"]])
                oracle_pick = ordered[:room]
                fiber_values = [truth[r["endpoint"]] for r in selected]
                shared_values = [truth[r["endpoint"]] for r in shared_rows]
                oracle_values = [truth[r["endpoint"]] for r in oracle_pick]
                pool_values = [truth[r["endpoint"]] for r in candidates]
                ranks = sorted(range(len(ordered)), key=lambda i: 0)
                chosen = {id(r) for r in selected}
                fiber_ranks = [i + 1 for i, r in enumerate(ordered) if id(r) in chosen]
                comparisons.append(
                    {
                        "round": len(comparisons) + 1,
                        "pool_size": len(candidates),
                        "selected": room,
                        "allocation_mode": detail.get("mode") or detail.get("role"),
                        "fiber_mean": sum(fiber_values) / max(len(fiber_values), 1),
                        "shared_acquisition_mean": (
                            sum(shared_values) / max(len(shared_values), 1)
                        ),
                        "shared_acquisition_max": max(shared_values) if shared_values else None,
                        "shared_value_is_fitted": shared_value.weights is not None,
                        "shared_value_observations": shared_value.n,
                        "optimism_sweep_selected_means": {
                            w: sum(truth[e] for e in eps) / max(len(eps), 1)
                            for w, eps in sweep.items()
                        },
                        "optimism_sweep_untried_fraction": {
                            w: sum(
                                1
                                for e in eps
                                if self.credit.cell(
                                    _credit_key_of(candidates, e)
                                ).trials == 0
                            )
                            / max(len(eps), 1)
                            for w, eps in sweep.items()
                        },
                        "fiber_max": max(fiber_values) if fiber_values else None,
                        "oracle_topk_mean": sum(oracle_values) / max(len(oracle_values), 1),
                        "oracle_topk_max": max(oracle_values) if oracle_values else None,
                        "pool_mean": sum(pool_values) / max(len(pool_values), 1),
                        "pool_max": max(pool_values) if pool_values else None,
                        "fiber_selection_ranks": sorted(fiber_ranks),
                        "regret_mean": (
                            sum(oracle_values) / max(len(oracle_values), 1)
                            - sum(fiber_values) / max(len(fiber_values), 1)
                        ),
                    }
                )
                del ranks
            return selected, detail

    folder = pathlib.Path("diagnostics/pmo_same_pool_ranking_v1/run")
    folder.mkdir(parents=True, exist_ok=True)
    ledger = ProgramQueryLedger(folder / "oracle", task, lambda s: float(score(s)), budget=budget)
    started = time.time()
    run_program_campaign(
        output=folder / "campaign",
        task=task,
        config=config,
        initialization=initialized,
        library=(),
        ledger=ledger,
        rounds=rounds,
        queries_per_round=queries,
        hierarchy=None,
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        initial_parent_fraction=0.2,
        progress=lambda row: print(
            f"  round {row.get('round')} charged={row.get('charged_calls')} "
            f"best={row.get('best_score')}",
            flush=True,
        ),
        optimizer_type=Observed,
        optimizer_kwargs={"jump_checkpoint": checkpoint},
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    report = {
        "schema_version": "pmo_same_pool_ranking_v1",
        "evidence_role": "labelled_development_diagnostic",
        "task": task_name,
        "charged_oracle_calls": len(ledger.rows),
        "diagnostic_evaluations_for_measurement": diagnostic_evaluations,
        "note": (
            "Pool scoring is measurement, not selection. Only FiberControl's picks drove the "
            "campaign. Neither count may enter an AUC or a scored ledger."
        ),
        "rounds_observed": len(comparisons),
        "comparisons": comparisons,
        "seconds": round(time.time() - started, 1),
    }
    if comparisons:
        regrets = [c["regret_mean"] for c in comparisons]
        report["mean_regret_vs_oracle_topk"] = sum(regrets) / len(regrets)
        report["rounds_fiber_matched_oracle"] = sum(1 for r in regrets if r <= 1e-9)
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print("WROTE", OUT)
    if comparisons:
        print(
            f"mean regret vs oracle top-k: {report['mean_regret_vs_oracle_topk']:+.4f} "
            f"over {len(comparisons)} allocations; "
            f"matched on {report['rounds_fiber_matched_oracle']}"
        )


if __name__ == "__main__":
    sys.exit(main())
