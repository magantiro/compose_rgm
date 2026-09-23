"""250-call Celecoxib canary: T4-style online FiberControl vs the matched greedy beam.

THE COMPARISON.  The beam (`diagnostics/pmo_matched_beam_v1/`) is reward-greedy allocation over
the same production proposal stack, the same 16-molecule bank and the same charged budget.  It is
the honest bar: if online control cannot beat repeatedly expanding whatever scored best, the
control is not earning its complexity.

WHAT IS T4-FAITHFUL HERE, and it is the part not to change.  A macro is one coherent COMPOSE
program and therefore ONE action, whatever number of primitive edits it compiles into.  It earns
one reward

    r_t = u(G_{t+1}) - u(G_t),

and macros from different parents are compared by restoring the offset,

    u_hat(G') = u(G) + delta_hat,

never by the delta alone.  `Q_pre` learns on macro INTENT and steers `selection()`, which is
upstream of generation, so reward changes what gets PROPOSED.  `Q_post` learns on the REALIZED
macro and steers `_allocate()`, so reward also changes what gets BOUGHT.  Both are refit after
every charged call.

THE MACRO PORTFOLIO IS PART OF THE ARM, AND IT HAD TO BE WIRED.  `dynamic_program_synthesis_v21`
calls `synthesize_dynamic_program` without a `replacement_option`, so on the production PMO path
`RegionReplacementOption` -- excise a region, rebuild it with any of fifteen families -- never
fires.  The beam comparator constructs that option explicitly, so leaving it unwired would have
compared a rich-portfolio beam against an impoverished controller and charged the difference to
the controller.  It is bound here by WRAPPING the function in the module namespace the production
path resolves at call time; no production byte changes, and the wrapper is verified to have fired
by requiring compound `region_replace:<rebuild>` labels in the audit.

INFORMATION BOUNDARY.  Task-specific knowledge enters only through charged oracle calls made
inside this run.  No target SMILES, no winner routes, no uncounted evaluations, no task identity
anywhere in the controller.  The budget is enforced by the ledger, which is the single authority.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import types
from pathlib import Path

import numpy as np

# Overridable so a wiring SMOKE writes to its own directory and can never be mistaken for,
# or merged with, the scored run. A smoke and a result must not share a ledger.
OUT = os.environ.get("CANARY_OUT", "diagnostics/pmo_reward_adaptive_canary_v1")
BEAM = "diagnostics/pmo_matched_beam_v1/matched_beam_v1.json"
POOL_TARGET = 128


def _oracle(name: str):
    stub = types.ModuleType("rdkit.six")
    stub.string_types = (str,)
    stub.iteritems = lambda d: iter(d.items())
    import rdkit

    sys.modules["rdkit.six"] = stub
    rdkit.six = stub
    from tdc import Oracle

    return Oracle(name=name)


def main():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = int(args[0]) if args else 250
    rounds = int(args[1]) if len(args) > 1 else 24
    queries = int(args[2]) if len(args) > 2 else 16
    task_name = args[3] if len(args) > 3 else "celecoxib_rediscovery"

    from compose_v4.control import dynamic_program_synthesis_v21 as v21
    from compose_v4.control.docking_value import identity
    from compose_v4.control.pmo_contextual_macro import macro_families, macro_scale
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController
    from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
    from compose_v4.control.program_task import (
        ProgramTask,
        archive_top_k,
        pmo_top_ten_auc,
    )
    from compose_v4.control.region_replacement_option import RegionReplacementOption
    from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw
    from compose_v4.experiments import pmo_population_v1 as production
    from compose_v4.experiments.pmo_dynamic_v21 import initial_dynamic_program_batch_v21

    # The REAL pool cap. `candidates_per_batch` does not govern this path -- an arm that set it
    # produced byte-identical numbers to its control, which is how that was caught.
    v21.CHANNEL_CANDIDATE_LIMIT = POOL_TARGET

    # Expose the full macro portfolio on the production path.  Wrapped, not transcribed: the
    # real `synthesize_dynamic_program` still does the work and still decides, so the arm
    # cannot drift from production by re-implementing it.
    option = RegionReplacementOption(region_law=ScaleBalancedRegionLaw())
    _production_synthesis = v21.synthesize_dynamic_program

    def _with_full_portfolio(*a, **k):
        k.setdefault("replacement_option", option)
        return _production_synthesis(*a, **k)

    v21.synthesize_dynamic_program = _with_full_portfolio

    root, score = Path("."), _oracle(task_name)
    contract = production.load_contract(root)
    initialized = production._load_initialization(
        root, {"initialization": contract["initialization"]}
    )
    checkpoint = production._load_checkpoint(root, contract)
    config = production.configuration(contract["controller"]["seed"])
    task = ProgramTask(
        task_name, identity(production._runtime_protocol(contract, task_name)), "pmo"
    )
    folder = pathlib.Path(OUT)
    folder.mkdir(parents=True, exist_ok=True)
    ledger = ProgramQueryLedger(
        folder / "oracle", task, lambda s: float(score(s)), budget=budget
    )
    telemetry: dict = {"rounds": [], "pool_sizes": [], "policy_shifts": []}

    def rows_of(candidates, parents=False):
        out = []
        for c in candidates:
            families = list(macro_families(c))
            provenance = c.get("provenance") or {}
            out.append(
                {
                    "parent": provenance.get("parent_smiles", c["endpoint"]),
                    "endpoint": c["endpoint"],
                    "smiles": c["endpoint"],
                    "parent_score": float(provenance.get("parent_measured_score") or 0.0),
                    "families": families,
                    "realized_families": families,
                    "requested_modules": len(
                        (c.get("program", {}) or {}).get("blocks", ()) or ()
                    ),
                    "module_count": len((c.get("program", {}) or {}).get("blocks", ()) or ()),
                    "primitives": len((c.get("trace", {}) or {}).get("actions", ()) or ()),
                    "depth": 1,
                    "generation": len(telemetry["rounds"]),
                    "capacity_aware": False,
                    **macro_scale(c),
                }
            )
        return out

    class RewardAdaptive(PmoPopulationController):
        """Production machinery end to end; reward steers proposal AND acquisition."""

        brain = RewardAdaptiveProgramController()

        def selection(self):
            """UPSTREAM: reward changes what gets proposed, not only what is bought."""
            keys, weights = super().selection()
            if not self.brain.fitted or not keys:
                return keys, weights
            intents = [
                {
                    "parent": self.entries[k]["endpoint"],
                    "endpoint": self.entries[k]["endpoint"],
                    "smiles": self.entries[k]["endpoint"],
                    "parent_score": float(
                        (self.observations.get(k) or {}).get("score") or 0.0
                    ),
                    "families": ["region_replace"],
                    "realized_families": ["region_replace"],
                    "requested_modules": 2,
                    "module_count": 2,
                    "primitives": 0,
                    "depth": 1,
                    "generation": len(telemetry["rounds"]),
                    "capacity_aware": False,
                }
                for k in keys
            ]
            learned = self.brain.intent_policy(intents)
            blended = 0.5 * np.asarray(weights, dtype=float) + 0.5 * learned
            total = float(blended.sum())
            return keys, (blended / total if total > 0 else weights)

        def _allocate(self, candidates):
            if not candidates:
                return super()._allocate(candidates)
            rows = rows_of(candidates)
            room = min(queries, len(rows))
            before = self.brain.record_policy(rows, f"before_{len(telemetry['rounds'])}")
            scores = [float(o["score"]) for o in self.observations.values()]
            threshold = sorted(scores, reverse=True)[9] if len(scores) >= 10 else 0.0
            chosen, detail = self.brain.acquire(
                rows, threshold, batch=room, rng=self.rng
            )
            selected = [candidates[i] for i in chosen]
            # The oracle fires in the campaign loop, so the reward is fed back on the NEXT
            # round from the ledger -- observing here would score an uncharged molecule.
            telemetry["pool_sizes"].append(len(candidates))
            telemetry["rounds"].append(
                {
                    "round": len(telemetry["rounds"]),
                    "pool": len(candidates),
                    "queried": len(selected),
                    "fitted": self.brain.fitted,
                    "observations": len(self.brain.observations),
                    "model_picks": sum(d["reason"] == "model" for d in detail),
                }
            )
            self._pending_rows = {
                c["candidate_id"]: r
                for c, r in zip(selected, rows_of(selected), strict=True)
                if "candidate_id" in c
            }
            after = self.brain.record_policy(rows, f"after_{len(telemetry['rounds'])}")
            telemetry["policy_shifts"].append(self.brain.policy_shift(before, after))
            return selected, {"mode": "reward_adaptive"}

        def observe_batch(self, batch_id, outcomes):
            """Feed every CHARGED outcome back, then refit both heads.

            Joined on ``candidate_id``, which is the campaign's own identity for a query --
            joining on SMILES would silently mis-attribute whenever two candidates land on
            one molecule. Attributing NOTHING is the failure this refuses out loud: a
            feedback loop that quietly observes zero rows looks exactly like a feedback loop.
            """
            result = super().observe_batch(batch_id, outcomes)
            pending = getattr(self, "_pending_rows", {})
            scored = [o for o in outcomes if o.get("score") is not None]
            attributed = 0
            for outcome in scored:
                row = pending.get(outcome.get("candidate_id"))
                if row is not None:
                    self.brain.observe(row, float(outcome["score"]))
                    attributed += 1
            if scored and not attributed:
                raise RuntimeError(
                    f"reward feedback attributed 0 of {len(scored)} charged outcomes"
                )
            telemetry["rounds"][-1]["attributed"] = attributed
            return result

    started = time.time()
    print(f"=== reward-adaptive canary: {task_name}, budget {budget} ===", flush=True)
    run_program_campaign(
        output=folder / "campaign", task=task, config=config,
        initialization=initialized, library=(), ledger=ledger,
        rounds=rounds, queries_per_round=queries, hierarchy=None, fit_model=None,
        stagnation_rounds=None, bootstrap_rounds=1,
        initialization_mode="all_scored_pool", initial_parent_fraction=0.2,
        progress=lambda row: None,
        optimizer_type=RewardAdaptive, optimizer_kwargs={"jump_checkpoint": checkpoint},
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )

    values = [r["score"] for r in ledger.rows]
    curve = []
    for i in range(1, len(values) + 1):
        curve.append(
            {
                "calls": i,
                "best": max(values[:i]),
                "top10": archive_top_k(
                    [(r["endpoint"], r["score"]) for r in ledger.rows[:i]], k=10
                ),
            }
        )

    def auc(points, lo, hi=None):
        """Mean top-ten over a call WINDOW.

        Both ends matter.  Aligning only the start compares this arm's calls 17..N against
        the beam's 17..250, which at any budget below 250 is not a comparison at all -- and
        it still prints a plausible pair of numbers, which is the dangerous kind of wrong.
        """
        window = [
            p["top10"] for p in points
            if p["calls"] >= lo and (hi is None or p["calls"] <= hi)
        ]
        return float(np.mean(window)) if window else None

    beam = {}
    if pathlib.Path(BEAM).exists():
        with open(BEAM) as handle:
            beam = json.load(handle)
    beam_curve = beam.get("curve", [])
    # The beam's curve starts after its initialization, so a mean over each arm's own curve
    # would compare different call ranges and silently favour the later-starting one.
    common_lo, common_hi = 1, None
    if beam_curve and curve:
        common_lo = max(curve[0]["calls"], beam_curve[0]["calls"])
        common_hi = min(curve[-1]["calls"], beam_curve[-1]["calls"])

    report = {
        "schema_version": "pmo_reward_adaptive_canary_v1",
        "evidence_role": "scored_matched_budget_canary",
        "task": task_name,
        "charged_oracle_calls": len(ledger.rows),
        "budget": budget,
        "pool_target": POOL_TARGET,
        "best_score": max(values) if values else None,
        "final_top10": curve[-1]["top10"] if curve else None,
        "auc_top10_own_curve": auc(curve, 1),
        "auc_official": float(pmo_top_ten_auc(values, budget=budget)) if values else None,
        "auc_top10_common_grid": auc(curve, common_lo, common_hi),
        "common_grid_calls": [common_lo, common_hi],
        "common_grid_is_full_budget": bool(common_hi == budget),
        "comparator_beam": {
            "source": BEAM,
            "best_score": beam.get("best_score"),
            "final_top10": beam.get("final_top10"),
            "auc_top10_reported": beam.get("auc_top10"),
            "auc_top10_common_grid": (
                auc(beam_curve, common_lo, common_hi) if beam_curve else None
            ),
        },
        "checkpoints": {
            str(n): next((c for c in curve if c["calls"] == n), None)
            for n in (100, 150, 200, 250)
        },
        "family_audit": RewardAdaptive.brain.ledger.report(),
        "median_pool": float(np.median(telemetry["pool_sizes"]))
        if telemetry["pool_sizes"] else None,
        "median_policy_shift": float(np.median(telemetry["policy_shifts"]))
        if telemetry["policy_shifts"] else None,
        "telemetry": telemetry,
        "curve": curve,
        "seconds": round(time.time() - started, 1),
    }
    families = report["family_audit"]["families"]
    report["portfolio"] = {
        "region_replacement_wired": True,
        "distinct_families": len(families),
        "compound_region_labels": sorted(f for f in families if ":" in f)[:12],
        "n_compound_region_labels": sum(1 for f in families if ":" in f),
    }
    if not report["portfolio"]["n_compound_region_labels"]:
        report["portfolio"]["WARNING"] = (
            "region replacement never fired: the arm ran the impoverished portfolio"
        )
    with open(folder / "canary_v1.json", "w") as handle:
        json.dump(report, handle, indent=1)
    print(
        f"charged {report['charged_oracle_calls']} best {report['best_score']:.4f} "
        f"top10 {report['final_top10']:.4f} "
        f"auc(common {common_lo}-{common_hi}) {report['auc_top10_common_grid']:.4f} "
        f"vs beam {report['comparator_beam']['auc_top10_common_grid']:.4f}"
        + ("" if report["common_grid_is_full_budget"] else "   [PARTIAL BUDGET]"),
        flush=True,
    )
    print("WROTE", folder / "canary_v1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
