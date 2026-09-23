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
from collections import Counter
from pathlib import Path

import numpy as np

# Overridable so a wiring SMOKE writes to its own directory and can never be mistaken for,
# or merged with, the scored run. A smoke and a result must not share a ledger.
OUT = os.environ.get("CANARY_OUT", "diagnostics/pmo_reward_adaptive_canary_v1")
BEAM = "diagnostics/pmo_matched_beam_v1/matched_beam_v1.json"
POOL_TARGET = 128
#: Probability the region-replacement macro is offered at each module position.
REPLACEMENT_OPTION_RATE = 0.5


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
    from compose_v4.control.pmo_contextual_macro import (
        macro_families,
        macro_scale,
        region_replace_labels,
    )
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController
    from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
    from compose_v4.control.program_task import (
        ProgramTask,
        archive_top_k,
        pmo_top_ten_auc,
    )
    from compose_v4.control.region_replacement_option import (
        REPLACEMENT_OPTIONS,
        RegionReplacementOption,
    )
    from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw
    from compose_v4.experiments import pmo_population_v1 as production
    from compose_v4.experiments.pmo_dynamic_v21 import initial_dynamic_program_batch_v21

    # The REAL pool cap. `candidates_per_batch` does not govern this path -- an arm that set it
    # produced byte-identical numbers to its control, which is how that was caught.
    v21.CHANNEL_CANDIDATE_LIMIT = POOL_TARGET

    # Expose the full macro portfolio on the production path.  Wrapped, not transcribed: the
    # real `synthesize_dynamic_program` still does the work and still decides, so the arm
    # cannot drift from production by re-implementing it.
    law = ScaleBalancedRegionLaw()
    # A MUTABLE holder, because Q_pre re-weights the rebuild options every round and a
    # closure over a fixed instance would freeze the upstream half of the control loop.
    portfolio = {"option": RegionReplacementOption(region_law=law), "offers": 0}
    _production_synthesis = v21.synthesize_dynamic_program

    def _with_full_portfolio(*a, **k):
        k.setdefault("replacement_option", portfolio["option"])
        # WITHOUT THIS THE OPTION IS INERT. `replacement_option_rate` defaults to 0.0, so
        # passing the option alone hands it over and never offers it -- the wrapper reads
        # as correct and produces zero region replacements. The beam comparator drives this
        # same option for every child it builds, so a rate near zero would compare a
        # controller that never uses the portfolio against a beam that always does; 0.5
        # gives it parity with the thirteen-family lottery rather than replacing it. This is
        # the arm's DECLARED configuration, not a tuned value.
        k.setdefault("replacement_option_rate", REPLACEMENT_OPTION_RATE)
        portfolio["offers"] += 1
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
    telemetry: dict = {
        "rounds": [], "pool_sizes": [], "policy_shifts": [], "option_policy": [],
        "unscored_parents": 0, "out_of_range_predictions": 0, "duplicate_endpoints": 0,
        "unallocated_batches": 0, "unattributed_outcomes": 0, "root_candidates": 0,
    }

    def rows_of(controller, candidates):
        """Feature rows for a batch of production candidates.

        TWO FIELDS THAT LOOK HARMLESS AND ARE NOT.

        `parent` must be the PARENT MOLECULE, resolved through `entry_id`. Production
        provenance carries no `parent_smiles` at all, so defaulting to the candidate's own
        endpoint made every candidate its own lineage: the lineage floor became inert, the
        lineage telemetry could not show collapse, and the `change` block compared a
        molecule with itself for a constant similarity of 1.0. Nothing errors.

        `parent_score` must be a CHARGED observation. A candidate with no parent is a root,
        and `u_hat(G') = u(G) + delta_hat` has no meaning for it -- so it is flagged rather
        than given a zero, and the caller keeps roots out of the value model instead of
        teaching it that unscored parents are worthless.
        """
        out = []
        for candidate in candidates:
            provenance = candidate.get("provenance") or {}
            families = list(macro_families(candidate))
            entry_id = provenance.get("entry_id")
            entry = controller.entries.get(entry_id) if entry_id is not None else None
            measured = provenance.get("parent_measured_score")
            if measured is None and entry is not None:
                try:
                    measured = controller._measured(entry_id)
                except KeyError:
                    measured = None
            out.append(
                {
                    "parent": (entry or {}).get("endpoint", candidate["endpoint"]),
                    "parent_key": entry_id,
                    "has_parent": entry is not None and measured is not None,
                    "endpoint": candidate["endpoint"],
                    "smiles": candidate["endpoint"],
                    "parent_score": float(measured) if measured is not None else 0.0,
                    "families": families,
                    "realized_families": families,
                    "requested_modules": len(
                        (candidate.get("program", {}) or {}).get("blocks", ()) or ()
                    ),
                    "module_count": len(
                        (candidate.get("program", {}) or {}).get("blocks", ()) or ()
                    ),
                    "primitives": len(
                        (candidate.get("trace", {}) or {}).get("actions", ()) or ()
                    ),
                    "depth": 1,
                    "generation": len(telemetry["rounds"]),
                    "capacity_aware": False,
                    **macro_scale(candidate),
                }
            )
        return out

    class RewardAdaptive(PmoPopulationController):
        """Production machinery end to end; reward steers proposal AND acquisition."""

        brain = RewardAdaptiveProgramController()

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pre_cache: dict = {}
            self._pending_rows: dict = {}

        def _measured(self, key) -> float:
            """A parent's own measured score, joined the way production joins it.

            RAISES when absent rather than returning 0.0. `u_hat(G') = u(G) + delta_hat`
            only means what it says when u(G) is a charged observation; defaulting a missing
            one to zero silently rebuilds the very failure the endpoint join fixed, in a
            quieter form -- every unscored parent would look equally, confidently bad.
            """
            endpoint = self.entries[key]["endpoint"]
            scores = [
                float(r["score"])
                for r in self.observations.values()
                if r["endpoint"] == endpoint
            ]
            if not scores:
                raise KeyError(f"parent {key} has no charged measured score")
            return float(np.mean(scores))

        def _pre_intents(self, keys):
            """One intent per (parent, rebuild option).

            The rebuild option is chosen UPSTREAM and recorded as intent, so reward can
            raise `region_replace:fuse_ring` specifically. Conditioning on the bare
            `region_replace` would let the controller learn after the fact that a fused-ring
            rebuild was good and leave it unable to ask for one.
            """
            intents, index = [], []
            for position, key in enumerate(keys):
                try:
                    measured = self._measured(key)
                except KeyError:
                    # An unscored entry is not an expandable parent for a cross-parent
                    # comparison. Excluded explicitly rather than imputed.
                    telemetry["unscored_parents"] += 1
                    continue
                endpoint = self.entries[key]["endpoint"]
                for option in REPLACEMENT_OPTIONS:
                    intents.append(
                        {
                            "parent": endpoint,
                            "endpoint": endpoint,
                            "smiles": endpoint,
                            "parent_score": measured,
                            "families": region_replace_labels(option),
                            "realized_families": region_replace_labels(option),
                            "requested_modules": 2,
                            "module_count": 2,
                            "primitives": 0,
                            "depth": 1,
                            "generation": len(telemetry["rounds"]),
                            "capacity_aware": False,
                        }
                    )
                    index.append((position, option))
            return intents, index

        def selection(self):
            """UPSTREAM: reward changes what gets proposed, not only what is bought."""
            keys, weights = super().selection()
            if not self.brain.fitted or not keys:
                return keys, weights
            cache_key = (len(telemetry["rounds"]), len(self.brain.observations))
            if self._pre_cache.get("key") != cache_key:
                intents, index = self._pre_intents(keys)
                if not intents:
                    self._pre_cache = {"key": cache_key, "parents": None}
                else:
                    learned = self.brain.intent_policy(intents)
                    parents = np.zeros(len(keys), dtype=float)
                    options = dict.fromkeys(REPLACEMENT_OPTIONS, 0.0)
                    for (position, option), mass in zip(index, learned, strict=True):
                        parents[position] += float(mass)
                        options[option] += float(mass)
                    # Feed the option marginal back into the generator: this is the hop that
                    # makes reward change WHAT gets built, not merely which parent is picked.
                    total = sum(options.values()) or 1.0
                    portfolio["option"] = RegionReplacementOption(
                        options=REPLACEMENT_OPTIONS,
                        weights=tuple(
                            max(options[o] / total, 1e-3) for o in REPLACEMENT_OPTIONS
                        ),
                        region_law=law,
                    )
                    telemetry["option_policy"].append(
                        {
                            "round": len(telemetry["rounds"]),
                            "weights": {o: options[o] / total for o in REPLACEMENT_OPTIONS},
                        }
                    )
                    self._pre_cache = {"key": cache_key, "parents": parents}
            parents = self._pre_cache.get("parents")
            if parents is None:
                return keys, weights
            blended = 0.5 * np.asarray(weights, dtype=float) + 0.5 * parents
            total = float(blended.sum())
            return keys, (blended / total if total > 0 else weights)

        def _allocate(self, candidates):
            if not candidates:
                return super()._allocate(candidates)
            rows = rows_of(self, candidates)
            room = min(queries, len(rows))
            before = self.brain.record_policy(rows, f"before_{len(telemetry['rounds'])}")
            scores = [float(o["score"]) for o in self.observations.values()]
            threshold = sorted(scores, reverse=True)[9] if len(scores) >= 10 else 0.0
            # A root has no u(G), so the T4 correction cannot score it. Ranking it with an
            # imputed zero parent would teach the model that unscored parents are worthless;
            # it goes to the exploration quota instead, which is what an unvalued candidate
            # is for.
            parented = [i for i, r in enumerate(rows) if r["has_parent"]]
            roots = [i for i, r in enumerate(rows) if not r["has_parent"]]
            telemetry["root_candidates"] += len(roots)
            chosen, detail = self.brain.acquire(
                [rows[i] for i in parented], threshold, batch=room, rng=self.rng
            )
            chosen = [parented[i] for i in chosen]
            for entry in detail:
                entry["index"] = parented[entry["index"]]
            if len(chosen) < room and roots:
                extra = list(self.rng.permutation(len(roots))[: room - len(chosen)])
                for raw in extra:
                    index = roots[int(raw)]
                    chosen.append(index)
                    detail.append({"index": index, "reason": "root", "propensity": 0.0})
            # A pool of N programs is not N molecules: two macros can reach the same
            # canonical endpoint, and charging both spends two oracle calls for one answer
            # while making the query fraction look better than it is. Keep the first, keep
            # every provenance record for credit, charge once.
            seen: dict = {}
            unique, unique_detail = [], []
            for index, entry in zip(chosen, detail, strict=True):
                endpoint = rows[index]["endpoint"]
                if endpoint in seen:
                    seen[endpoint].append(index)
                    telemetry["duplicate_endpoints"] += 1
                    continue
                seen[endpoint] = [index]
                unique.append(index)
                unique_detail.append(entry)
            chosen, detail = unique, unique_detail
            selected = [candidates[i] for i in chosen]
            # The oracle fires in the campaign loop, so the reward is fed back on the NEXT
            # round from the ledger -- observing here would score an uncharged molecule.
            # Lineage concentration, measured every round rather than assumed from a floor.
            # A previous PMO run had 2 of 16 lineages produce any scored descendant with one
            # owning ~99%; a floor in code does not prove the executor yields anything from
            # the lineages it protects.
            lineage = Counter(r["parent_key"] for r in rows if r["parent_key"] is not None)
            share = np.asarray([v / max(len(rows), 1) for v in lineage.values()])
            effective = float(1.0 / np.sum(share**2)) if len(share) else 0.0
            predicted = [
                d["predicted_endpoint"] for d in detail if "predicted_endpoint" in d
            ]
            outside = [v for v in predicted if not 0.0 <= v <= 1.0]
            telemetry["out_of_range_predictions"] += len(outside)
            telemetry["pool_sizes"].append(len(candidates))
            telemetry["rounds"].append(
                {
                    "round": len(telemetry["rounds"]),
                    "pool": len(candidates),
                    "unique_endpoints": len({r["endpoint"] for r in rows}),
                    "queried": len(selected),
                    "fitted": self.brain.fitted,
                    "observations": len(self.brain.observations),
                    "model_picks": sum(d["reason"] == "model" for d in detail),
                    "active_lineages": len(lineage),
                    "effective_lineages": effective,
                    "max_lineage_share": float(share.max()) if len(share) else 0.0,
                    "predictions_outside_unit_range": len(outside),
                }
            )
            self._pending_rows = {
                c["candidate_id"]: r
                for c, r in zip(selected, rows_of(self, selected), strict=True)
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
            if not pending:
                # The INITIALIZATION batch is not allocated by this controller, so there is
                # nothing to attribute and equality would fire spuriously. Counted, not
                # skipped silently -- an unexplained unattributed batch is the failure this
                # check exists to catch, so the exemption has to be narrow and visible.
                telemetry["unallocated_batches"] += 1
                telemetry["unattributed_outcomes"] += len(scored)
                return result
            attributed = roots = 0
            for outcome in scored:
                row = pending.get(outcome.get("candidate_id"))
                if row is None:
                    continue
                if row["has_parent"]:
                    self.brain.observe(row, float(outcome["score"]))
                    attributed += 1
                else:
                    # A root's reward is real and is charged; it simply carries no
                    # (parent, macro) transition to learn from. Counted, not dropped.
                    roots += 1
            if attributed + roots != len(scored):
                # Losing even one reward label per round biases exactly the macro and
                # lineage learning this run exists to measure, and "most of them arrived"
                # is indistinguishable from "all of them arrived" in every aggregate.
                raise RuntimeError(
                    f"reward feedback accounted for {attributed} learned + {roots} root "
                    f"of {len(scored)} charged outcomes; online learning requires all"
                )
            telemetry["rounds"][-1]["attributed"] = attributed
            telemetry["rounds"][-1]["root_outcomes"] = roots
            # The ledger truncates the final batch, so a round can charge fewer calls than
            # it selected. Record what was CHARGED; comparing against the selected count
            # reports a correct round as incomplete.
            telemetry["rounds"][-1]["charged"] = len(scored)
            self._pending_rows = {}
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
        "unique_endpoint_ratio": (
            float(np.mean([r["unique_endpoints"] / max(r["pool"], 1)
                           for r in telemetry["rounds"]]))
            if telemetry["rounds"] else None
        ),
        "min_effective_lineages": (
            float(min(r["effective_lineages"] for r in telemetry["rounds"]))
            if telemetry["rounds"] else None
        ),
        "max_lineage_share": (
            float(max(r["max_lineage_share"] for r in telemetry["rounds"]))
            if telemetry["rounds"] else None
        ),
        "unscored_parents_excluded": telemetry["unscored_parents"],
        "root_candidates_seen": telemetry["root_candidates"],
        "unallocated_batches": telemetry["unallocated_batches"],
        "unattributed_outcomes": telemetry["unattributed_outcomes"],
        "rounds_fully_attributed": sum(
            1
            for r in telemetry["rounds"]
            if r.get("attributed", 0) + r.get("root_outcomes", 0) == r.get("charged", 0)
        ),
        "duplicate_endpoints_avoided": telemetry["duplicate_endpoints"],
        "predictions_outside_unit_range": telemetry["out_of_range_predictions"],
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
        "compound_region_labels": sorted(
            f for f in families if f.startswith("region_replace:")
        )[:12],
        "n_compound_region_labels": sum(
            1 for f in families if f.startswith("region_replace:")
        ),
        "option_policy_rounds": len(telemetry["option_policy"]),
        "synthesis_calls_wrapped": portfolio["offers"],
        "replacement_option_rate": REPLACEMENT_OPTION_RATE,
    }
    if not report["portfolio"]["n_compound_region_labels"]:
        # RAISE, not warn. The whole point of this arm is that it carries the same rich
        # portfolio as the beam; a run that quietly produced none of it would be reported
        # as a controller result when it is a wiring result.
        report["portfolio"]["FAILED"] = (
            "region replacement never fired: the arm ran the impoverished portfolio"
        )
        with open(folder / "canary_v1.json", "w") as handle:
            json.dump(report, handle, indent=1)
        raise RuntimeError(
            "region replacement produced no realized macro; refusing to report a "
            f"portfolio result (wrapped syntheses={portfolio['offers']})"
        )
    with open(folder / "canary_v1.json", "w") as handle:
        json.dump(report, handle, indent=1)
    rounds = telemetry["rounds"]
    if rounds:
        print(
            f"lineages: effective min {report['min_effective_lineages']:.2f} "
            f"max share {report['max_lineage_share']:.2f} | "
            f"unique endpoints {report['unique_endpoint_ratio']:.2f} of pool | "
            f"dupes avoided {report['duplicate_endpoints_avoided']} | "
            f"out-of-range {report['predictions_outside_unit_range']}",
            flush=True,
        )
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
