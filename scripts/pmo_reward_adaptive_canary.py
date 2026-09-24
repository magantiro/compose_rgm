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

import contextlib
import dataclasses
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
#: Activate the repo's SHIPPED context-local serialization memo. It is keyed on exact array
#: bytes, so it returns what the uncached call returns and changes no decision, no RNG draw
#: and no candidate -- measured 1.74x on the real synthesis path with an element-wise
#: identical endpoint sequence. Size is immaterial above a few hundred (512 == 8192 ==
#: 65536 measured), so the smallest sufficient value is used. Env-toggleable ONLY so the
#: equivalence harness can run both arms through identical code.
SERIALIZATION_CACHE_ENTRIES = int(os.environ.get("CANARY_SMILES_CACHE") or "512")

#: PROPOSAL BREADTH IS A COUNT, NOT A DURATION.
#:
#: Both proposal loops stop on `len(candidates) >= CHANNEL_CANDIDATE_LIMIT OR elapsed >=
#: wall_seconds`. With the wall term live, machine speed is an input to the search: a faster
#: process fits more attempts inside the same 45 s and builds a different pool from the same
#: seed. That is why a memo returning byte-identical SMILES still moved the trajectory, and
#: it would also mean two Modal containers of differing speed ran different searches.
#:
#: Raising the wall bound past any achievable runtime makes the COUNT the sole normal stop,
#: so the proposal envelope is identical on a laptop, a fast container and a slow one. The
#: matched beam was already count-based (`for _ in range(8)`), so this makes the two arms
#: agree rather than changing one of them.
PROPOSAL_WALL_SECONDS = float(os.environ.get("CANARY_PROPOSAL_WALL") or "1e9")

#: Wall clock survives ONLY as a fail-loud safety bound. A round that exceeds it aborts the
#: run; it never silently returns a smaller pool and continues, which is exactly the failure
#: the count-based envelope exists to remove.
HARD_ROUND_TIMEOUT_SECONDS = float(os.environ.get("CANARY_HARD_TIMEOUT") or "3600")

#: Controller seed. Absent, the contract seed is used, so every existing run is
#: unchanged. Supplied, it is the ONLY thing that differs between replicates -- the
#: initialization bank and every controller constant stay frozen.
SEED_OVERRIDE = os.environ.get("CANARY_SEED")
#: Directory of per-task prescreen initialization banks.  Absent -> the frozen
#: task-independent bank, so the no-prescreen runtime is byte-identical.
PRESCREEN_INIT = os.environ.get("PMO_PRESCREEN_INIT")
#: Charged initialization molecules reach the search ONLY through the bootstrap draw --
#: they are never archive ENTRIES, because an entry carries an `EditProgram` and a molecule
#: handed to the campaign has no edit history.  Drawing that one path uniformly throws away
#: their measured score.  Absent -> "uniform" -> byte-identical to every run so far.
PARENT_WEIGHTING = os.environ.get("PMO_PARENT_WEIGHTING") or "uniform"
#: Arm B of the memory portability gate. ABSENT is the byte-identical OFF: the controller
#: constructs no memory object, so `region_law()` returns None and the unlawed draw keeps
#: its own RNG stream. A "uniform memory" would NOT be a no-op.
ONLINE_MEMORY = (os.environ.get("PMO_ONLINE_MEMORY") or "").lower() in ("1", "true", "yes")
#: Where PyTDC's relative `oracle/<name>.pkl` resolves FROM for the three asset-backed
#: tasks. The Modal image bakes this same path.
ORACLE_ASSET_ROOT = (os.environ.get("PMO_ORACLE_ASSET_ROOT")
                     or "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets")
BEAM = "diagnostics/pmo_matched_beam_v1/matched_beam_v1.json"
BEAM_SOURCE = "scripts/pmo_matched_beam_control.py"
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

    from compose_v4.experiments.pmo_oracle_assets import (
        AssetPinnedOracle,
        assert_positive_control,
        requires_positive_control,
    )

    oracle = Oracle(name=name)
    if not requires_positive_control(name):
        # 20 of the 23 PMO tasks are pure RDKit: no asset file, no cwd dependency.
        return oracle
    # drd2/gsk3b/jnk3 open `oracle/<name>.pkl` by a RELATIVE path, and gsk3b/drd2 do it
    # LAZILY on the first call, inside tdc.Oracle.__call__'s bare `except` -- so a wrong
    # working directory does not raise, it returns default_property 0.0 for the entire
    # budget and writes a ledger nothing can distinguish from a real result. Pin the
    # directory for the oracle's LIFETIME, then refuse to proceed unless known actives
    # actually score. The control runs here, BEFORE the first charged call.
    oracle = AssetPinnedOracle(oracle, Path(ORACLE_ASSET_ROOT), name=name)
    report = assert_positive_control(oracle, name)
    print(
        f"ORACLE POSITIVE CONTROL {name}: {report['n_agreeing']}/{report['n_references']} "
        f"agree, max_abs_delta {report['max_abs_delta']:.3g}",
        flush=True,
    )
    return oracle


def main():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = int(args[0]) if args else 250
    rounds = int(args[1]) if len(args) > 1 else 24
    queries = int(args[2]) if len(args) > 2 else 16
    task_name = args[3] if len(args) > 3 else "celecoxib_rediscovery"

    from compose_v4.chem.molecular_graph import molecular_serialization_cache
    from compose_v4.control import dynamic_program_synthesis as dps
    from compose_v4.control import dynamic_program_synthesis_v21 as v21
    from compose_v4.control.docking_value import identity
    from compose_v4.control.pmo_contextual_macro import (
        macro_families,
        macro_intent_families,
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

    # THE SECOND WALL-CLOCK STOP. The jump lane's realizer is capped by
    # `PRODUCTION_SECONDS_CAP = 20.0` as well as by `PRODUCTION_NODE_BUDGET = 64`, so a
    # faster run completes a different number of realizations and builds a different pool.
    # That is what made two IDENTICAL uncached arms disagree on parent probabilities by
    # 6.3e-4 while every charged decision still matched -- the jump lane contributes few
    # charged candidates, so it perturbed the pool without yet moving the trajectory.
    #
    # The node budget is the deterministic bound and is the one that actually governs
    # successes: measured, every realization needs at most 55 of the 64 nodes and finishes
    # within 16.9 s, while only the unproductive searches run 18-64 s. Raising the seconds
    # cap therefore preserves every success and merely lets hopeless searches reach their
    # node limit. Patched in EVERY namespace holding the constant, because a module that
    # imported the value bound it at import time.
    _realizer_cap = float(os.environ.get("CANARY_REALIZER_SECONDS") or "1e9")
    _patched_caps = []
    for _name, _module in list(sys.modules.items()):
        if not _name.startswith("compose_v4"):
            continue
        if getattr(_module, "PRODUCTION_SECONDS_CAP", None) is not None:
            _module.PRODUCTION_SECONDS_CAP = _realizer_cap
            _patched_caps.append(_name)
    if not _patched_caps:
        raise RuntimeError("realizer seconds cap not found; proposal breadth stays timed")

    # Expose the full macro portfolio on the production path.  Wrapped, not transcribed: the
    # real `synthesize_dynamic_program` still does the work and still decides, so the arm
    # cannot drift from production by re-implementing it.
    law = ScaleBalancedRegionLaw()
    # A MUTABLE holder, because Q_pre re-weights the rebuild options every round and a
    # closure over a fixed instance would freeze the upstream half of the control loop.
    portfolio = {
        "option": RegionReplacementOption(region_law=law),
        "offers": 0,
        "by_module": {},
    }
    _production_synthesis = dps.synthesize_dynamic_program

    def _with_full_portfolio(*a, **k):
        k.setdefault("replacement_option", portfolio["option"])
        # WITHOUT THIS THE OPTION IS INERT. `replacement_option_rate` defaults to 0.0, so
        # passing the option alone hands it over and never offers it. The beam drives this
        # same option for every child it builds, so a rate near zero would compare a
        # controller that never uses the portfolio against a beam that always does.
        k.setdefault("replacement_option_rate", REPLACEMENT_OPTION_RATE)
        # The beam passes this too; without it the thirteen-family lottery draws regions
        # under a different law than the beam's.
        k.setdefault("region_law", law)
        portfolio["offers"] += 1
        return _production_synthesis(*a, **k)

    # PATCH EVERY NAMESPACE THAT HOLDS THE SYMBOL, not just v21's. A module that did
    # `from ... import synthesize_dynamic_program` bound the function OBJECT at import time,
    # so rebinding one module's attribute leaves every other caller on the original. That is
    # why region replacements appeared almost only in the initialization batch: the
    # controller's own lanes never went through the wrapper. The holder list is DERIVED at
    # runtime, so a new caller cannot silently fall outside it.
    patched = []
    for _name, _module in list(sys.modules.items()):
        if not _name.startswith("compose_v4"):
            continue
        if getattr(_module, "synthesize_dynamic_program", None) is _production_synthesis:
            _module.synthesize_dynamic_program = _with_full_portfolio
            patched.append(_name)
    if len(patched) < 2:
        raise RuntimeError(f"portfolio wrapper reached only {patched}; expected every holder")

    root, score = Path("."), _oracle(task_name)
    contract = production.load_contract(root)
    if PRESCREEN_INIT:
        # PRESCREEN ARM. The ONLY change from the frozen runtime is which molecules
        # initialize the population: a per-task bank drawn from the ZINC250k prescreen
        # table by that task's own official oracle. The controller, its constants, the
        # population size and the accounting are untouched.
        #
        # The contract cannot pin this file (it is per task), so the checks the
        # production loader would have applied are applied HERE instead, including the
        # refusal of any candidate row carrying task information.
        init_path = pathlib.Path(PRESCREEN_INIT) / f"{task_name}.json"
        initialized = json.loads(init_path.read_text())
        body = {k: v for k, v in initialized.items() if k != "lock_sha256"}
        if identity(body) != initialized.get("lock_sha256"):
            raise ValueError(f"prescreen initialization lock changed: {init_path}")
        if initialized.get("count") != production.INIT_COUNT:
            raise ValueError("prescreen initialization must keep the frozen population size")
        if len(initialized.get("candidates", ())) != production.INIT_COUNT:
            raise ValueError("prescreen initialization candidate count mismatch")
        if initialized.get("accounting") != (
            "all initialization scores count against each run's oracle budget"
        ):
            raise ValueError("prescreen initialization must charge its scores")
        if any("score" in row or "task" in row for row in initialized["candidates"]):
            raise ValueError("prescreen initialization leaks task information")
        print(f"PRESCREEN initialization: {init_path} "
              f"({initialized['count']} seeds, lock {initialized['lock_sha256'][:16]})",
              flush=True)
    else:
        initialized = production._load_initialization(
            root, {"initialization": contract["initialization"]}
        )
    checkpoint = production._load_checkpoint(root, contract)
    seed = int(SEED_OVERRIDE) if SEED_OVERRIDE else int(contract["controller"]["seed"])
    config = production.configuration(seed)
    config = dataclasses.replace(config, wall_seconds=PROPOSAL_WALL_SECONDS)
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
        "selected_endpoints": [], "rng_state": None,
        "region_replace_with_rebuild": 0, "region_replace_bare": 0,
        "intent_differs_from_realized": 0,
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
            intent = list(macro_intent_families(candidate))
            if intent != families:
                telemetry["intent_differs_from_realized"] += 1
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
                    # Q_pre is credited for what it REQUESTED, Q_post for what EXECUTED.
                    # Keying both on the realized option would flow reward to whichever
                    # rebuild survived executor refusal rather than to the one asked for.
                    "families": intent,
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

    _state_path = folder / "controller_state.json"
    if _state_path.exists():
        with open(_state_path) as handle:
            _restored = RewardAdaptiveProgramController.restore(json.load(handle))
        print(
            f"RESUMED controller state: {len(_restored.observations)} observations, "
            f"fitted={_restored.fitted}",
            flush=True,
        )
    else:
        _restored = RewardAdaptiveProgramController()

    class RewardAdaptive(PmoPopulationController):
        """Production machinery end to end; reward steers proposal AND acquisition."""

        # Restored when a previous attempt left state, fresh otherwise. A container that
        # restarts after preemption resumes the SAME trajectory; without this the campaign
        # archive would continue while the value model silently restarted cold, which is
        # indistinguishable from a working resume in every artifact.
        brain = _restored

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pre_cache: dict = {}
            self._pending_rows: dict = {}
            self._last_rows: list = []
            self._last_policy: dict = {}

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
            self._round_began = getattr(self, "_round_began", time.time())
            if not candidates:
                return super()._allocate(candidates)
            rows = rows_of(self, candidates)
            for row in rows:
                labels = row["families"]
                if any(f.startswith("region_replace:") for f in labels):
                    telemetry["region_replace_with_rebuild"] += 1
                elif "region_replace" in labels:
                    telemetry["region_replace_bare"] += 1
            room = min(queries, len(rows))
            # THE SHIFT MUST STRADDLE THE REWARD UPDATE. Recording `before` and `after`
            # around `acquire` measures a window that contains no refit -- `observe` moved
            # into `observe_batch` -- so it can only ever read 0.0. Compare instead the
            # PREVIOUS round's rows re-scored now, against the policy recorded on those same
            # rows last round: identical inputs, one reward update in between.
            if self._last_rows:
                now = self.brain.record_policy(self._last_rows, f"after_update_{len(telemetry['rounds'])}")
                telemetry["policy_shifts"].append(
                    self.brain.policy_shift(self._last_policy, now)
                )
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
            round_seconds = time.time() - self._round_began
            if round_seconds > HARD_ROUND_TIMEOUT_SECONDS:
                raise RuntimeError(
                    f"round {len(telemetry['rounds'])} took {round_seconds:.0f}s against a "
                    f"{HARD_ROUND_TIMEOUT_SECONDS:.0f}s safety bound; aborting rather than "
                    "continuing on a degraded pool"
                )
            channels = {
                name: dict(counts)
                for name, counts in (self.population_state.get("channels") or {}).items()
            }
            telemetry["pool_sizes"].append(len(candidates))
            telemetry["rounds"].append(
                {
                    "round": len(telemetry["rounds"]),
                    "pool": len(candidates),
                    # The compute contract, auditable per round: how many proposals were
                    # attempted and how many became candidates.
                    "proposal_attempts": sum(
                        int(c.get("proposals", 0)) for c in channels.values()
                    ),
                    "eligible_candidates": sum(
                        int(c.get("eligible_novel", 0)) for c in channels.values()
                    ),
                    "channel_counters": channels,
                    "round_seconds": round(round_seconds, 2),
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
            # PURE I/O, no RNG and no state mutation: a run that is meant to be stopped
            # mid-flight must leave a complete artifact at every round, or killing it
            # discards every in-memory series (option policy, lineages, family audit) while
            # only the oracle ledger survives.
            _checkpoint()
            # Logged from observe_batch instead: attribution is only known once the
            # outcomes come back, and logging here printed `attr None/None` every round.
            self._log_args = (rows, selected, detail)
            telemetry["selected_endpoints"].append([c["endpoint"] for c in selected])
            telemetry["rng_state"] = json.loads(
                json.dumps(self.rng.bit_generator.state, default=str)
            )
            self._round_began = time.time()
            self._last_rows = rows
            self._last_policy = self.brain.record_policy(
                rows, f"round_{len(telemetry['rounds'])}"
            )
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
            if getattr(self, "_log_args", None):
                _round_log(self, *self._log_args)
                self._log_args = None
            # COMMIT THE CONTROLLER STATE HERE, not in _allocate. The campaign commits its
            # round after the outcomes return; writing the brain mid-round left the two
            # checkpoints out of sync by part of a round, so a resumed run restored a brain
            # one step behind its campaign and diverged from call 80 onward -- measured.
            _state = folder / "controller_state.json.tmp"
            with open(_state, "w") as handle:
                json.dump(RewardAdaptive.brain.state(), handle)
            _state.replace(folder / "controller_state.json")
            self._pending_rows = {}
            return result

    def _option_tv() -> float:
        """Total variation between the first and latest rebuild-option policy."""
        history = telemetry["option_policy"]
        if len(history) < 2:
            return 0.0
        first, last = history[0]["weights"], history[-1]["weights"]
        return 0.5 * sum(
            abs(last.get(k, 0.0) - first.get(k, 0.0)) for k in set(first) | set(last)
        )

    def _round_log(controller, rows, selected, detail):
        """One concise status block per round, plus append-only machine-readable rows.

        Observational ONLY: it reads state that already exists and consumes no RNG. A
        logger that drew a random number, or that re-derived a decision, would change the
        trajectory it is supposed to be reporting.
        """
        values = [r["score"] for r in ledger.rows]
        if not values:
            return
        rounds = telemetry["rounds"]
        last = rounds[-1] if rounds else {}
        top10 = archive_top_k([(r["endpoint"], r["score"]) for r in ledger.rows], k=10)
        best = max(values)
        best_row = max(ledger.rows, key=lambda r: r["score"])
        recent = [x.get("round_seconds", 0.0) for x in rounds[-3:]]
        pace = sum(recent) / max(len(recent), 1)
        remaining = max(budget - len(values), 0)
        eta_hours = (remaining / max(queries, 1)) * pace / 3600.0

        # the reward-bearing transitions charged this round
        fresh = [
            r for r in ledger.rows[-len(selected):]
        ] if selected else []
        by_score = sorted(fresh, key=lambda r: -r["score"])[:3]
        gains = []
        for row in rows:
            match = next(
                (q for q in fresh if q["endpoint"] == row["endpoint"]), None
            )
            if match is not None and row.get("has_parent"):
                gains.append((match["score"] - row["parent_score"], row, match))
        gains.sort(key=lambda g: -g[0])

        families = RewardAdaptive.brain.ledger.report()["families"]
        by_mass = sorted(
            families.items(), key=lambda kv: -(kv[1].get("proposal_mass_last") or 0.0)
        )[:3]
        by_delta = sorted(
            (kv for kv in families.items() if kv[1].get("expected_delta_u") is not None),
            key=lambda kv: -kv[1]["expected_delta_u"],
        )[:3]

        print(
            f"[{task_name}] round {last.get('round', '?')} | "
            f"{len(values)}/{budget} charged | best {best:.4f} | top10 {top10:.4f} | "
            f"pool {last.get('pool')} attempts {last.get('proposal_attempts')} "
            f"queried {len(selected)} | lineages {last.get('effective_lineages', 0):.1f} "
            f"maxshare {last.get('max_lineage_share', 0):.2f} | "
            f"shift {(telemetry['policy_shifts'][-1] if telemetry['policy_shifts'] else 0):.3f} "
            f"optTV {_option_tv():.4f} | "
            f"roots {telemetry['root_candidates']} attr {last.get('attributed')}/"
            f"{last.get('charged')} | {last.get('round_seconds', 0):.0f}s "
            f"ETA {eta_hours:.1f}h",
            flush=True,
        )
        print(f"    best so far  {best:.4f}  {best_row['endpoint']}", flush=True)
        for row in by_score:
            print(f"    queried      {row['score']:.4f}  {row['endpoint']}", flush=True)
        if gains and gains[0][0] > 0:
            delta, parent_row, child = gains[0]
            print(
                f"    best gain    +{delta:.4f}  "
                f"{parent_row['parent_score']:.4f} {parent_row['parent']}\n"
                f"                 -> intent {[f for f in parent_row['families'] if ':' in f] or parent_row['families'][:2]}\n"
                f"                 -> realized {[f for f in parent_row['realized_families'] if ':' in f] or parent_row['realized_families'][:2]}\n"
                f"                 -> {child['score']:.4f} {child['endpoint']}",
                flush=True,
            )
        if by_mass:
            print(
                "    families by mass: "
                + ", ".join(f"{k}={v['proposal_mass_last']:.3f}" for k, v in by_mass),
                flush=True,
            )
        if by_delta:
            print(
                "    families by E[du]: "
                + ", ".join(
                    f"{k}={v['expected_delta_u']:+.4f}(n={v['n_queried']})"
                    for k, v in by_delta
                ),
                flush=True,
            )

        with open(folder / "trajectory.jsonl", "a") as handle:
            handle.write(json.dumps({
                "round": last.get("round"), "charged": len(values), "best": best,
                "top10": top10, **{k: last.get(k) for k in (
                    "pool", "proposal_attempts", "queried", "attributed",
                    "effective_lineages", "max_lineage_share", "round_seconds")},
            }) + "\n")
        with open(folder / "queries.jsonl", "a") as handle:
            for entry in detail:
                row = rows[entry["index"]]
                match = next(
                    (q for q in fresh if q["endpoint"] == row["endpoint"]), None
                )
                handle.write(json.dumps({
                    "round": last.get("round"),
                    "parent": row["parent"], "parent_score": row["parent_score"],
                    "parent_key": row["parent_key"], "has_parent": row["has_parent"],
                    "intent_families": row["families"],
                    "realized_families": row["realized_families"],
                    "endpoint": row["endpoint"],
                    "score": match["score"] if match else None,
                    "delta_u": (match["score"] - row["parent_score"]) if match else None,
                    "reason": entry.get("reason"),
                    "predicted_endpoint": entry.get("predicted_endpoint"),
                    "decision_utility": entry.get("decision_utility"),
                }) + "\n")
        if best > getattr(_round_log, "_best", float("-inf")):
            _round_log._best = best
            with open(folder / "best_molecules.jsonl", "a") as handle:
                handle.write(json.dumps({
                    "charged": len(values), "score": best,
                    "smiles": best_row["endpoint"], "task": task_name,
                }) + "\n")
            print(f"    *** NEW GLOBAL BEST {best:.4f} {best_row['endpoint']}", flush=True)

    def _checkpoint():
        values = [r["score"] for r in ledger.rows]
        if not values:
            return
        snapshot = {
            "schema_version": "pmo_reward_adaptive_canary_progress_v1",
            "status": "IN_PROGRESS",
            "task": task_name,
            "budget": budget,
            "charged_oracle_calls": len(values),
            "best_score": max(values),
            "final_top10": archive_top_k(
                [(r["endpoint"], r["score"]) for r in ledger.rows], k=10
            ),
            "option_policy": telemetry["option_policy"],
            "rounds": telemetry["rounds"],
            "policy_shifts": telemetry["policy_shifts"],
            "family_audit": RewardAdaptive.brain.ledger.report(),
            "unscored_parents": telemetry["unscored_parents"],
            "root_candidates": telemetry["root_candidates"],
            "duplicate_endpoints": telemetry["duplicate_endpoints"],
            "region_replace_with_rebuild": telemetry["region_replace_with_rebuild"],
            "region_replace_bare": telemetry["region_replace_bare"],
            "out_of_range_predictions": telemetry["out_of_range_predictions"],
            "elapsed_seconds": round(time.time() - started, 1),
        }
        temporary = folder / "progress.json.tmp"
        with open(temporary, "w") as handle:
            json.dump(snapshot, handle, indent=1)
        temporary.replace(folder / "progress.json")

    started = time.time()
    print(f"=== reward-adaptive canary: {task_name}, budget {budget} ===", flush=True)
    cache_scope = (
        molecular_serialization_cache(max_entries=SERIALIZATION_CACHE_ENTRIES)
        if SERIALIZATION_CACHE_ENTRIES > 0
        else contextlib.nullcontext()
    )
    with cache_scope:
        run_program_campaign(
            output=folder / "campaign", task=task, config=config,
            initialization=initialized, library=(), ledger=ledger,
            rounds=rounds, queries_per_round=queries, hierarchy=None, fit_model=None,
            stagnation_rounds=None, bootstrap_rounds=1,
            initialization_mode="all_scored_pool", initial_parent_fraction=0.2,
            initial_parent_weighting=PARENT_WEIGHTING,
            progress=lambda row: None,
            optimizer_type=RewardAdaptive,
            optimizer_kwargs={"jump_checkpoint": checkpoint,
                              "enable_online_memory": ONLINE_MEMORY},
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
        # After initialization a large root count means provenance is being lost rather
        # than roots being generated, so the FRACTION is what to read, not the count.
        "root_candidate_fraction": (
            telemetry["root_candidates"] / max(sum(telemetry["pool_sizes"]), 1)
        ),
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
        # Counted from the candidate stream at allocation time. The family AUDIT is a
        # different aggregation and reported zero compounds while the candidates carried
        # them, so the witness has to come from the same place the claim does.
        "n_compound_region_labels": telemetry["region_replace_with_rebuild"],
        "region_replace_bare": telemetry["region_replace_bare"],
        "intent_differs_from_realized": telemetry["intent_differs_from_realized"],
        "option_policy_rounds": len(telemetry["option_policy"]),
        "synthesis_calls_wrapped": portfolio["offers"],
        "patched_namespaces": patched,
        "replacement_option_rate": REPLACEMENT_OPTION_RATE,
    }
    # The comparison contract, recorded so nobody has to reconstruct it later. Read from
    # the beam's own source: scripts/pmo_matched_beam_control.py passes
    # replacement_option_rate=0.5 and region_law=ScaleBalancedRegionLaw() to the same
    # synthesis function. Base proposal opportunity is therefore IDENTICAL; the arms differ
    # only in that Q_pre re-weights the rebuild-option marginal while the beam holds the
    # declared default weights fixed.
    # Did reward actually MOVE the option policy, or merely run? A policy that is recorded
    # every round and never departs from uniform would satisfy "option_policy_rounds > 0"
    # while controlling nothing.
    option_policy = telemetry["option_policy"]
    uniform = 1.0 / max(len(REPLACEMENT_OPTIONS), 1)
    report["option_policy_moved"] = {
        "rounds": len(option_policy),
        "max_deviation_from_uniform": (
            max(
                0.5 * sum(abs(w - uniform) for w in snap["weights"].values())
                for snap in option_policy
            )
            if option_policy else 0.0
        ),
        "first_to_last_total_variation": (
            0.5
            * sum(
                abs(
                    option_policy[-1]["weights"].get(o, 0.0)
                    - option_policy[0]["weights"].get(o, 0.0)
                )
                for o in REPLACEMENT_OPTIONS
            )
            if len(option_policy) > 1 else 0.0
        ),
    }
    report["proposal_compute_contract"] = {
        "breadth_is": "fixed attempt count, not elapsed time",
        "proposal_wall_seconds": PROPOSAL_WALL_SECONDS,
        "hard_round_timeout_seconds": HARD_ROUND_TIMEOUT_SECONDS,
        "attempts_per_batch": config.attempts_per_batch,
        "channel_candidate_limit": POOL_TARGET,
        "beam_breadth": "8 attempts per frontier parent, already count-based",
        "serialization_cache_entries": SERIALIZATION_CACHE_ENTRIES,
    }
    report["proposal_portfolio_contract"] = {
        "controller_replacement_option_rate": REPLACEMENT_OPTION_RATE,
        "beam_replacement_option_rate": 0.5,
        "controller_region_law": "ScaleBalancedRegionLaw",
        "beam_region_law": "ScaleBalancedRegionLaw",
        "rates_match": REPLACEMENT_OPTION_RATE == 0.5,
        "difference": "rebuild-option weights are adaptive in the controller, fixed in the beam",
        "beam_source": BEAM_SOURCE,
    }
    if not (
        report["portfolio"]["n_compound_region_labels"]
        or report["portfolio"]["region_replace_bare"]
    ):
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
    equivalence = {
        "endpoint_sequence": [(r["endpoint"], r["score"]) for r in ledger.rows],
        "selected_endpoints_per_round": telemetry["selected_endpoints"],
        "parent_policy_history": [
            {"label": h["label"], "parent_mass": h["parent_mass"]}
            for h in RewardAdaptive.brain.policy_history
        ],
        "option_policy": telemetry["option_policy"],
        "observations": [
            {"endpoint": o["endpoint"], "score": o["score"], "delta_raw": o["delta_raw"]}
            for o in RewardAdaptive.brain.observations
        ],
        "rng_state": telemetry["rng_state"],
        "serialization_cache_entries": SERIALIZATION_CACHE_ENTRIES,
    }
    with open(folder / "equivalence.json", "w") as handle:
        json.dump(equivalence, handle, indent=1, sort_keys=True, default=str)
    with open(folder / "canary_v1.json", "w") as handle:
        json.dump(report, handle, indent=1)
    moved = report["option_policy_moved"]
    print(
        f"option policy: rounds {moved['rounds']} "
        f"dev-from-uniform {moved['max_deviation_from_uniform']:.4f} "
        f"first->last TV {moved['first_to_last_total_variation']:.4f} | "
        f"region_replace compound {report['portfolio']['n_compound_region_labels']} "
        f"bare {report['portfolio']['region_replace_bare']}",
        flush=True,
    )
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
    beam_auc = report["comparator_beam"]["auc_top10_common_grid"]
    # The beam artifact is CELECOXIB-ONLY, so four of five targets have no comparator by
    # construction. A missing comparator is a fact to report, not a reason to fail a run
    # that already produced every number it was asked for.
    comparator = f"vs beam {beam_auc:.4f}" if beam_auc is not None else "no matched beam"
    print(
        f"charged {report['charged_oracle_calls']} "
        f"best {report['best_score']:.4f} top10 {report['final_top10']:.4f} "
        f"auc(common {common_lo}-{common_hi}) "
        f"{(report['auc_top10_common_grid'] or 0.0):.4f} {comparator}"
        + ("" if report["common_grid_is_full_budget"] else "   [PARTIAL BUDGET]"),
        flush=True,
    )
    print("WROTE", folder / "canary_v1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
