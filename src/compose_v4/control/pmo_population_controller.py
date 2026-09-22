"""PMO population control over local, structured and complete-jump programs.

The optimizer shares COMPOSE's exact executor and archive with the existing
Dynamic controller.  Its PMO-specific policy is population level: preserve
several structural basins, allocate an explicit early exploration floor, and
use observed reward only after complete candidates have been generated.  The
complete-jump checkpoint is task blind and contains address-free action roles,
never a task-to-route lookup or executable teacher program.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.bootstrap_pool_continuity import BootstrapPoolContinuity
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNEL_CANDIDATE_LIMIT,
    DynamicV21ProgramOptimizer,
    _candidate,
    _context_key,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
    program_size_profile,
)
from compose_v4.control.fiber_control import ProgramValue, SearchState, acquisition
from compose_v4.control.pmo_credit import (
    PopulationCredit,
    basin_label,
    credit_key_from_candidate,
    improvement,
)
from compose_v4.control.pmo_joint_dependency_jump import (
    CHECKPOINT_SCHEMA,
)
from compose_v4.control.pmo_online_memory import (
    OnlineProposalMemory,
    memory_channel_proposal,
)
from compose_v4.control.pmo_realization import (
    PRODUCTION_MAX_REALIZATIONS,
    PRODUCTION_NODE_BUDGET,
    PRODUCTION_SECONDS_CAP,
    PRODUCTION_SPECIFICATION,
    realize_plan_binding,
)
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_population_controller_v1"
JUMP_CHANNEL = "joint_dependency_region_jump"
CHANNELS = (SHALLOW_CHANNEL, STRUCTURED_CHANNEL, JUMP_CHANNEL)

# Plateau excursion. `PLATEAU_ROUNDS` non-improving rounds arm a bounded
# `ESCAPE_ROUNDS` excursion, and arming restarts the counter so the excursion drains
# before it can be armed again. A search setting, not a calibrated schedule.
PLATEAU_ROUNDS = 3
ESCAPE_ROUNDS = 2

# Novelty bonus for an untried parent, expressed in MULTIPLES of the run's own measured
# mean upside per child rather than in absolute oracle units. Dimensionless on purpose:
# one value covers every oracle because the scale it multiplies is counted, not assumed.
NOVELTY_PRIOR_TRIALS = 1.0
MODE_BY_CHANNEL = {
    SHALLOW_CHANNEL: "refine_elite",
    STRUCTURED_CHANNEL: "global_explore",
    JUMP_CHANNEL: "jump_from_elite",
}
FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _blank_channel() -> dict[str, int | float]:
    return {
        "proposals": 0,
        "exact_executions": 0,
        "eligible_novel": 0,
        "duplicates": 0,
        "ineligible": 0,
        "execution_rejections": 0,
        "selected_for_oracle": 0,
        "charged_outcomes": 0,
        "scored_outcomes": 0,
        "parent_improvements": 0,
        "positive_improvement_sum": 0.0,
    }


def initial_population_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "channels": {channel: _blank_channel() for channel in CHANNELS},
        "best_score": None,
        "rounds_without_improvement": 0,
        "escape_rounds_remaining": 0,
        "allocation_decisions": 0,
        "parent_outcomes": {},
        "credit_key_failures": 0,
    }


def advance_plateau_state(state: dict[str, Any], *, improved: bool) -> dict[str, Any]:
    """Advance the plateau counter and arm a bounded escape excursion. Mutates `state`.

    Arming RESTARTS the non-improvement counter. The previous form re-set the escape
    budget on every non-improving round while the allocator drained only one unit per
    round, so on any plateau longer than the threshold the budget was replenished faster
    than it was spent and the escape never released. Restarting the counter makes the
    excursion bounded by construction: after arming it takes `PLATEAU_ROUNDS` further
    non-improving rounds to re-arm, which is strictly longer than the `ESCAPE_ROUNDS` it
    takes to drain.
    """
    if improved:
        state["rounds_without_improvement"] = 0
        return state
    state["rounds_without_improvement"] = int(state["rounds_without_improvement"]) + 1
    if state["rounds_without_improvement"] >= PLATEAU_ROUNDS:
        state["escape_rounds_remaining"] = ESCAPE_ROUNDS
        state["rounds_without_improvement"] = 0
    return state


def _retained_fraction(source, endpoint) -> float:
    original = np.flatnonzero(is_element(source.atom_types))
    if not len(original):
        return 0.0
    retained = sum(bool(is_element(endpoint.atom_types)[int(slot)]) for slot in original)
    return float(retained / len(original))


def _fingerprint(endpoint: str) -> set[int]:
    molecule = Chem.MolFromSmiles(endpoint)
    if molecule is None:
        raise ValueError("PMO population candidate is not an RDKit-valid molecule")
    return set(map(int, FINGERPRINT.GetFingerprint(molecule).GetOnBits()))


def _program_rule_histogram(candidate: dict[str, Any]) -> dict[str, int]:
    rules = [str(row["executor_rule"]) for row in candidate["trace"]["actions"]]
    return dict(sorted(Counter(rules).items()))


def population_features(candidate: dict[str, Any], *, incumbent_reward: float | None) -> np.ndarray:
    """Task-free complete-program features for PMO reward allocation."""

    provenance = candidate["provenance"]
    parent = provenance.get("parent_measured_score")
    parent_reward = float(parent) if parent is not None else 0.0
    incumbent = parent_reward if incumbent_reward is None else float(incumbent_reward)
    size = candidate.get("program_size", provenance.get("program_size", {}))
    rules = _program_rule_histogram(candidate)
    channel = str(provenance.get("planner_channel", ""))
    return np.asarray(
        [
            -parent_reward,
            incumbent - parent_reward,
            float(size.get("primitive_edits", len(candidate["trace"]["actions"]))) / 32,
            float(size.get("block_count", len(candidate["program"].get("blocks", ())))) / 8,
            float(size.get("delta_from_measured_parent", 0)) / 16,
            float(provenance.get("retention_realized", 1.0)),
            float(provenance.get("retention_target", 1.0)),
            float(channel == SHALLOW_CHANNEL),
            float(channel == STRUCTURED_CHANNEL),
            float(channel == JUMP_CHANNEL),
            rules.get("atom_insert", 0) / 16,
            rules.get("atom_delete", 0) / 16,
            rules.get("cycle_close", 0) / 4,
            rules.get("cycle_open", 0) / 4,
            rules.get("ring_system_restate", 0) / 4,
            1.0,
        ],
        dtype=float,
    )


def _niche_order(candidates: list[dict[str, Any]], *, seed: int) -> list[int]:
    """Deterministic farthest-first order inside one proposal family."""

    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    identities = [row["candidate_id"] for row in candidates]
    first = int(rng.integers(len(candidates)))
    chosen = [first]
    while len(chosen) < len(candidates):
        best = None
        for index, row in enumerate(candidates):
            if index in chosen:
                continue
            fingerprint = set(row["fiber_fingerprint"])
            similarity = max(
                len(fingerprint & set(candidates[prior]["fiber_fingerprint"]))
                / max(
                    1,
                    len(fingerprint | set(candidates[prior]["fiber_fingerprint"])),
                )
                for prior in chosen
            )
            key = (similarity, identities[index])
            if best is None or key < best[0]:
                best = (key, index)
        chosen.append(int(best[1]))
    return chosen


class PmoPopulationController(DynamicV21ProgramOptimizer):
    """Live PMO archive with local, global and complete-jump proposal modes."""

    def __init__(
        self,
        *args,
        jump_checkpoint: dict[str, Any],
        enable_online_memory: bool = False,
        construction_prior=None,
        construction_prior_spec=None,
        **kwargs,
    ):
        if jump_checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("PMO population controller needs a sanitized joint checkpoint")
        super().__init__(*args, **kwargs)
        self.jump_checkpoint = json.loads(json.dumps(jump_checkpoint))
        self.jump_checkpoint_id = identity(self.jump_checkpoint)
        self.jump_rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 311, 2]))
        self.population_state = initial_population_state()
        self._population_bootstrap_pool_id = None
        self._pool_continuity = BootstrapPoolContinuity()
        self.credit = PopulationCredit()
        # Arm B of the matched comparison. OFF by default, so the unflagged controller
        # is byte-identical to arm A: a cold or absent memory makes `region_law()`
        # return None, and an unlawed draw consumes `rng.permutation` while ANY law
        # object consumes `rng.random` -- so a 'uniform law' would not be a no-op.
        self.online_memory = OnlineProposalMemory() if enable_online_memory else None
        self.online_memory_attribution_failures = 0
        self.online_memory_reconstruction: dict[str, Any] | None = None
        # The task-independent chemical prior over the CONSTRUCTION draw.  It is an
        # already-built object rather than a checkpoint path, so the controller never
        # imports torch and a test can drive this seam with a stub.  `None` is the
        # only byte-identical off: `DynamicProgramOptimizer._mutate` reads this
        # attribute and `synthesize_dynamic_program` keeps v1's exact RNG stream when
        # it is absent.  It reads no oracle, no task identity and no score.
        #
        # `construction_prior_spec` is the PRODUCTION form of the same seam: a
        # JSON-serializable declaration that survives `identity(optimizer_kwargs)` and
        # `restore`, where a live model object would raise on the first and be dropped
        # by the second.  Exactly one of the two may be given; supplying both would
        # leave the arm's declared identity and its actual law free to disagree.
        if construction_prior is not None and construction_prior_spec is not None:
            raise ValueError(
                "supply the construction prior as an object OR as a spec, not both: "
                "two sources for one law let the declared arm identity and the law "
                "actually consulted drift apart"
            )
        self.construction_prior_spec = (
            None
            if construction_prior_spec is None
            else json.loads(json.dumps(construction_prior_spec))
        )
        if construction_prior_spec is not None:
            from compose_v4.control.pmo_construction_prior import (
                ConstructionPriorSpec,
                build_construction_prior,
            )

            construction_prior = build_construction_prior(
                ConstructionPriorSpec.from_payload(construction_prior_spec)
            )
        self.construction_prior = construction_prior

    def _channel_proposal(self, channel, entry):
        """The one integration point for the online structural memory.

        Delegates to the unmodified production implementation whenever the memory is
        absent, cold, or the lane is not the shallow one -- so this is a no-op in arm A.
        """

        return memory_channel_proposal(
            self, channel, entry, super()._channel_proposal, self.online_memory
        )

    def observed_upside_scale(self) -> float:
        """Mean positive parent improvement per child, counted from this run.

        This is the reward SCALE the exploration bonus is denominated in. It is derived
        from observations the controller already holds -- never from a per-task
        constant -- so one mechanism covers every oracle. Before any child has been
        scored it returns 0.0, which leaves every parent's novelty bonus identical and
        therefore leaves the allocation exactly where it already was.
        """
        children = 0
        positive = 0.0
        for stats in self.population_state["parent_outcomes"].values():
            children += int(stats.get("children", 0))
            positive += float(stats.get("positive_improvement_sum", 0.0))
        return positive / children if children else 0.0

    def selection(self):
        """Preserve structural niches, then tilt toward parents with measured upside."""

        keys, probability = ProgramOptimizer.selection(self)
        # The novelty bonus is expressed in MEASURED improvement units. A fixed 0.25
        # was sized for a docking score; against PMO's observed mean upside it needed
        # ~1000 trials on one parent before earned upside could overtake it, which made
        # the tilt a pure novelty term that no evidence could outvote.
        scale = self.observed_upside_scale()
        factors = []
        for key in keys:
            stats = self.population_state["parent_outcomes"].get(key, {})
            trials = int(stats.get("children", 0))
            positive = float(stats.get("positive_improvement_sum", 0.0))
            upside = positive / max(1, trials)
            uncertainty = NOVELTY_PRIOR_TRIALS / math.sqrt(trials + 1)
            # The tilt is a RATIO in measured-reward units, not an additive nudge on a
            # dimensionless 1.0. Added to 1.0, both evidence terms sit ~0.007 against a
            # base of 1, so 64 consecutive unproductive children moved a parent's mass by
            # 0.05-0.27% -- the allocator could not tell a working parent from a stalled
            # one. Dividing the earned upside by the run's own scale puts "this parent
            # delivers about one typical improvement per child" at the same weight as
            # "this parent has never been tried", and drops a parent with many children
            # and no upside to roughly half of either.
            factors.append(
                1.0 + (upside / scale if scale > 0.0 else 0.0) + uncertainty
            )
        weighted = probability * np.asarray(factors, dtype=float)
        return keys, weighted / weighted.sum()

    def _jump_plan_order(self) -> list[dict[str, Any]]:
        bands: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for plan in self.jump_checkpoint["plan_latents"]:
            count = int(plan["primitive_count"])
            band = "small" if count <= 7 else "medium" if count <= 15 else "large"
            bands[band].append(plan)
        queues = []
        for band_index, band in enumerate(("small", "medium", "large")):
            rows = bands[band]
            if not rows:
                continue
            mass = np.asarray([max(float(row["mass"]), 1e-12) for row in rows])
            mass /= mass.sum()
            stream = np.random.default_rng(
                np.random.SeedSequence([self.config.seed, 719, band_index])
            )
            permutation = stream.choice(len(rows), len(rows), replace=False, p=mass)
            queues.append([rows[int(index)] for index in permutation])
        # Round-robin queues keep every available scale represented prospectively.
        order = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.pop(0))
        return order

    @staticmethod
    def _joint_program(source, bound: dict[str, Any]):
        stage = {
            "name": "joint_dependency_region_jump",
            "actions": bound["actions"],
            "states": bound["states"],
            "endpoint": bound["endpoint_key"],
        }
        program, binding = extract_program(source, [stage])
        _, trace = execute_program_graph(
            source,
            compile_program_graph(program),
            binding,
            max_primitives=32,
            max_blocks=8,
        )
        if trace["states"] != bound["states"] or trace["endpoint"] != bound["endpoint_key"]:
            raise RuntimeError("joint jump changed during EditProgram extraction")
        return program, binding, trace

    def _generate_jump_pool(self, eligibility, archive_seen, parent_schedule):
        began, attempts, candidates = perf_counter(), [], []
        seen = set(archive_seen)
        plans = self._jump_plan_order()
        if not plans:
            return attempts, candidates, 0.0
        plan_attempts = min(self.config.attempts_per_batch, 32, len(plans))
        plan_offset = (self.batches * plan_attempts) % len(plans)
        for attempt in range(plan_attempts):
            if (
                len(candidates) >= CHANNEL_CANDIDATE_LIMIT
                or perf_counter() - began >= self.config.wall_seconds
            ):
                break
            entry, parent = parent_schedule[attempt % len(parent_schedule)]
            plan = plans[(plan_offset + attempt) % len(plans)]
            source = decode_state(entry["trace"]["states"][-1])
            retention_target = float(self.jump_rng.random())
            try:
                # The jump lane's own wall budget binds this attempt as well, so one
                # hopeless plan cannot consume the lane the way an unbounded complete
                # search would.  The smaller of the two caps applies.
                realized = realize_plan_binding(
                    source,
                    plan,
                    specification=PRODUCTION_SPECIFICATION,
                    node_budget=PRODUCTION_NODE_BUDGET,
                    seconds_cap=min(
                        PRODUCTION_SECONDS_CAP,
                        max(self.config.wall_seconds - (perf_counter() - began), 0.0),
                    ),
                    max_realizations=PRODUCTION_MAX_REALIZATIONS,
                )
                bound = realized["bindings"]
                if not bound:
                    # The outcome rides along: a plan PROVEN incompatible with this
                    # parent and one whose search ran out of budget are different
                    # findings, and one string for both is what made the previous
                    # binder failure unreadable.
                    raise ValueError(
                        "joint plan has no legal binding on this parent "
                        f"({realized['outcome']})"
                    )
                ranked = []
                for row in bound:
                    endpoint_state = decode_state(row["endpoint_state"])
                    retention = _retained_fraction(source, endpoint_state)
                    ranked.append((abs(retention - retention_target), row["endpoint_key"], row))
                _, _, selected = min(ranked, key=lambda row: (row[0], row[1]))
                program, binding, trace = self._joint_program(source, selected)
                graph = compile_program_graph(program)
                size = program_size_profile(graph, source.n_real_atoms)
                size["measured_parent_heavy_atoms"] = source.n_real_atoms
                size["delta_from_measured_parent"] = size["final_heavy_atoms"] - source.n_real_atoms
                retention = _retained_fraction(source, decode_state(trace["states"][-1]))
            except (ValueError, RuntimeError) as error:
                self.population_state["channels"][JUMP_CHANNEL]["proposals"] += 1
                self.population_state["channels"][JUMP_CHANNEL]["execution_rejections"] += 1
                attempts.append(
                    {
                        "attempt": attempt,
                        **parent,
                        "planner_channel": JUMP_CHANNEL,
                        "mode": MODE_BY_CHANNEL[JUMP_CHANNEL],
                        "plan_id": plan["plan_id"],
                        "status": "execution_rejected",
                        "reason": str(error),
                    }
                )
                continue
            endpoint = trace["endpoint"]
            properties = eligibility({"smiles": endpoint})
            if type(properties.get("oracle_eligible")) is not bool:
                raise ValueError("endpoint evaluator must return explicit eligibility")
            status = (
                "duplicate"
                if endpoint in seen
                else "eligible"
                if properties["oracle_eligible"]
                else "ineligible"
            )
            counts = self.population_state["channels"][JUMP_CHANNEL]
            counts["proposals"] += 1
            counts["exact_executions"] += 1
            counts[
                {
                    "eligible": "eligible_novel",
                    "duplicate": "duplicates",
                    "ineligible": "ineligible",
                }[status]
            ] += 1
            metadata = {
                "joint_dependency_region_jump": {
                    "plan_id": plan["plan_id"],
                    "plan_mass": float(plan["mass"]),
                    "checkpoint_id": self.jump_checkpoint_id,
                    "retention_target": retention_target,
                    "retention_realized": retention,
                    "primitive_count": selected["primitive_count"],
                    "component_count": selected["component_count"],
                    "created_dependency_edges": selected["created_dependency_edges"],
                },
                **self._continuation_lineage(entry),
            }
            attempt_record = {
                "attempt": attempt,
                **parent,
                "planner_channel": JUMP_CHANNEL,
                "mode": MODE_BY_CHANNEL[JUMP_CHANNEL],
                "planner_context": _context_key(source),
                "plan_id": plan["plan_id"],
                "status": status,
                "endpoint": endpoint,
                "metadata": metadata,
                "program_size": size,
                "properties": properties,
                "actual_changes": trace["actual_changes"],
                "retention_target": retention_target,
                "retention_realized": retention,
            }
            attempts.append(attempt_record)
            if status == "duplicate":
                self.duplicate_counts[entry["entry_id"]] = (
                    self.duplicate_counts.get(entry["entry_id"], 0) + 1
                )
            if status != "eligible":
                continue
            seen.add(endpoint)
            candidates.append(
                _candidate(
                    optimizer=self,
                    entry=entry,
                    parent=parent,
                    channel=JUMP_CHANNEL,
                    attempt=attempt,
                    source=source,
                    program=program,
                    binding=binding,
                    trace=trace,
                    metadata=metadata,
                    size=size,
                    properties=properties,
                )
            )
            candidates[-1]["provenance"].update(
                {
                    "mode": MODE_BY_CHANNEL[JUMP_CHANNEL],
                    "retention_target": retention_target,
                    "retention_realized": retention,
                }
            )
        return attempts, candidates, perf_counter() - began

    def _augment(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate = json.loads(json.dumps(candidate))
        channel = str(candidate["provenance"].get("planner_channel", SHALLOW_CHANNEL))
        candidate["provenance"].setdefault("mode", MODE_BY_CHANNEL.get(channel, "global_explore"))
        candidate["fiber_fingerprint"] = sorted(_fingerprint(candidate["endpoint"]))
        # Basin is the canonical Bemis-Murcko scaffold, deliberately coarse: analogues that
        # share a scaffold must share a basin or the joint credit cell degenerates into a
        # second parent id and nothing can accumulate evidence.  Acyclic endpoints fall back
        # to an explicit sentinel rather than an empty-string bucket.
        candidate["provenance"]["basin_id"] = basin_label(candidate["endpoint"])
        incumbent = self.population_state["best_score"]
        candidate["fiber_features"] = population_features(
            candidate, incumbent_reward=incumbent
        ).tolist()
        candidate["candidate_id"] = identity(
            {key: value for key, value in candidate.items() if key != "candidate_id"}
        )
        return candidate

    def _fit_value(self) -> tuple[ProgramValue, SearchState]:
        scores: dict[str, list[float]] = defaultdict(list)
        for observation in self.observations.values():
            scores[observation["endpoint"]].append(float(observation["score"]))
        features, improvements = [], []
        for entry in self.entries.values():
            parent = entry.get("provenance", {}).get("parent_measured_score")
            row_features = entry.get("fiber_features")
            if parent is None or row_features is None or entry["endpoint"] not in scores:
                continue
            reward = float(np.mean(scores[entry["endpoint"]]))
            features.append(row_features)
            improvements.append(reward - float(parent))
        value = ProgramValue(penalty=1.0)
        if features:
            value.fit(features, improvements)
        state = SearchState(
            archive={endpoint: -float(np.mean(values)) for endpoint, values in scores.items()},
            budget=0,
            rounds=self.batches,
            history=[],
        )
        return value, state

    def _allocate(self, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
        limit = min(self.config.candidates_per_batch, len(candidates))
        self.population_state["allocation_decisions"] += 1
        if not limit:
            return [], {"selected_ids": [], "mode": "empty_pool"}
        by_channel = {
            channel: [row for row in candidates if row["provenance"]["planner_channel"] == channel]
            for channel in CHANNELS
        }
        escape = self.population_state["escape_rounds_remaining"] > 0
        early = self.batches < 4
        # The jump lane holds NO fixed floor. It is not deleted: its candidates stay in
        # `remaining`, where `PopulationCredit.allocate` guarantees every live cell at
        # least `eps / n` of the budget, so the lane recovers on measured evidence
        # rather than on a reserved quota it was not earning.
        quota = (
            {SHALLOW_CHANNEL: 1, STRUCTURED_CHANNEL: 5, JUMP_CHANNEL: 0}
            if escape
            else {SHALLOW_CHANNEL: 2, STRUCTURED_CHANNEL: 2, JUMP_CHANNEL: 0}
            if early
            else {SHALLOW_CHANNEL: 1, STRUCTURED_CHANNEL: 1, JUMP_CHANNEL: 0}
        )
        chosen: list[dict[str, Any]] = []
        allocation_roles: dict[str, str] = {}
        for channel_index, channel in enumerate(CHANNELS):
            rows = by_channel[channel]
            order = _niche_order(
                rows,
                seed=int(
                    identity(
                        {
                            "seed": self.config.seed,
                            "batch": self.batches,
                            "channel": channel_index,
                        }
                    )[:16],
                    16,
                ),
            )
            floor = [rows[index] for index in order[: min(quota[channel], len(rows))]]
            chosen.extend(floor)
            allocation_roles.update({row["candidate_id"]: "family_niche_floor" for row in floor})
        chosen_ids = {row["candidate_id"] for row in chosen}
        remaining = [row for row in candidates if row["candidate_id"] not in chosen_ids]
        value, state = self._fit_value()
        room = limit - len(chosen)
        credit_detail: dict[str, Any] = {"mode": "floor_only"}
        if room > 0 and remaining:
            prepared = [
                {
                    **row,
                    "features": np.asarray(row["fiber_features"], dtype=float),
                    "fingerprint": set(row["fiber_fingerprint"]),
                    "parent_score": -float(row["provenance"].get("parent_measured_score", 0.0)),
                }
                for row in remaining
            ]
            if self.credit.cells:
                selected_rows, credit_detail = self._credit_allocate(remaining, value, room)
                allocation_role = "joint_credit"
            else:
                # Cold start: no credit evidence exists yet, so allocation is unchanged.
                selected = acquisition(
                    prepared,
                    value,
                    state,
                    self.rng,
                    batch=room,
                    diversity=0.5,
                    exploration=min(2, room),
                )
                selected_rows = [remaining[index] for index in selected]
                credit_detail = {"mode": "cold_start_acquisition"}
                # With an unfitted value function acquisition is deliberately random.
                allocation_role = "fiber_control" if value.weights is not None else "exploration"
            chosen.extend(selected_rows)
            allocation_roles.update({row["candidate_id"]: allocation_role for row in selected_rows})
        chosen = chosen[:limit]
        if escape:
            self.population_state["escape_rounds_remaining"] -= 1
        predicted = (
            []
            if not candidates
            else list(
                map(
                    float,
                    value.predict(
                        np.asarray([row["fiber_features"] for row in candidates], dtype=float)
                    ),
                )
            )
        )
        return chosen, {
            "schema_version": "pmo_population_allocation_v1",
            "mode": "fiber_control_with_family_and_niche_floor",
            "early_quota_active": early,
            "plateau_escape_active": escape,
            "available_by_channel": {key: len(value) for key, value in by_channel.items()},
            "quota_by_channel": quota,
            "selected_by_channel": {
                channel: sum(row["provenance"]["planner_channel"] == channel for row in chosen)
                for channel in CHANNELS
            },
            "selected_ids": [row["candidate_id"] for row in chosen],
            "selection_role_by_candidate": {
                row["candidate_id"]: allocation_roles[row["candidate_id"]] for row in chosen
            },
            "predicted_parent_improvements": dict(
                zip((row["candidate_id"] for row in candidates), predicted, strict=True)
            ),
            "program_value_observations": value.n,
            "credit_allocation": credit_detail,
            "credit_summary": self.credit_report(),
        }

    def _credit_allocate(self, remaining, value, room):
        """Spend `room` oracle slots across joint credit cells, then rank inside each cell.

        The cell share comes from `PopulationCredit.allocate`, which mixes measured credit
        with a uniform floor, so a cell that has never won can still be drawn and reward
        feedback cannot starve global exploration.  The credit decides WHERE to spend; the
        fitted program value decides WHICH candidate inside the drawn cell.
        """
        by_cell: dict[Any, list[dict[str, Any]]] = {}
        for row in remaining:
            try:
                key = credit_key_from_candidate(row)
            except (ValueError, KeyError):
                self.population_state["credit_key_failures"] += 1
                continue
            by_cell.setdefault(key, []).append(row)
        if not by_cell:
            return [], {"mode": "no_credit_keys"}
        cells = sorted(by_cell, key=lambda key: (key.basin, key.parent, key.family, key.scale))
        shares = self.credit.allocate(cells)
        predicted: dict[str, float] = {}
        if value.weights is not None:
            for rows in by_cell.values():
                features = np.asarray([row["fiber_features"] for row in rows], dtype=float)
                scores = map(float, value.predict(features))
                for row, score in zip(rows, scores, strict=True):
                    predicted[row["candidate_id"]] = score
        for rows in by_cell.values():
            rows.sort(key=lambda row: (-predicted.get(row["candidate_id"], 0.0), row["candidate_id"]))
        chosen: list[dict[str, Any]] = []
        draws: list[dict[str, str]] = []
        while len(chosen) < room:
            live = [index for index, key in enumerate(cells) if by_cell[key]]
            if not live:
                break
            weights = np.asarray([shares[index] for index in live], dtype=float)
            weights = weights / weights.sum()
            pick = int(self.rng.choice(live, p=weights))
            key = cells[pick]
            chosen.append(by_cell[key].pop(0))
            draws.append(key.payload())
        return chosen, {
            "mode": "joint_credit_cell_allocation",
            "exploration_floor": self.credit.exploration_floor,
            "cells_available": len(cells),
            "minimum_share": float(min(shares)) if len(shares) else 0.0,
            "drawn_cells": draws,
        }

    def credit_report(self) -> dict[str, Any]:
        """Diagnostic view of the joint credit: what is earning, and where budget goes."""

        cells = self.credit.cells
        earning = [
            {**key.payload(), "trials": cell.trials, "improvements": cell.improvements,
             "positive_improvement_sum": round(cell.positive_improvement_sum, 6)}
            for key, cell in cells.items()
            if cell.improvements > 0
        ]
        earning.sort(key=lambda row: -row["positive_improvement_sum"])
        scale_trials: dict[str, int] = {}
        for key, cell in cells.items():
            scale_trials[key.scale] = scale_trials.get(key.scale, 0) + cell.trials
        total = sum(scale_trials.values())
        productive_parents = {
            key.parent for key, cell in cells.items() if cell.improvements > 0
        }
        return {
            "active_basins": len({key.basin for key in cells}),
            "active_cells": len(cells),
            "cells_with_positive_credit": len(earning),
            "trial_fraction_by_scale": (
                {scale: round(count / total, 4) for scale, count in sorted(scale_trials.items())}
                if total
                else {}
            ),
            "productive_parents": len(productive_parents),
            "top_earning_cells": earning[:10],
            "credit_key_failures": self.population_state.get("credit_key_failures", 0),
        }

    def propose_batch(self, eligibility):
        if self.pending is not None:
            raise ValueError("resolve the pending PMO population batch first")
        archive_seen = {
            row["endpoint"] for row in self.observations.values()
        } | self.failed_endpoints
        schedules = {
            channel: [self._parent() for _ in range(self.config.attempts_per_batch)]
            for channel in CHANNELS
        }
        attempts, pools, seconds = [], {}, {}
        for channel in (SHALLOW_CHANNEL, STRUCTURED_CHANNEL):
            lane_attempts, rows, elapsed = self._generate_channel_pool(
                channel, eligibility, archive_seen, schedules[channel]
            )
            for row in rows:
                row["provenance"]["mode"] = MODE_BY_CHANNEL[channel]
            counts = self.population_state["channels"][channel]
            for attempt in lane_attempts:
                status = attempt["status"]
                counts["proposals"] += 1
                if status != "execution_rejected":
                    counts["exact_executions"] += 1
                key = {
                    "eligible": "eligible_novel",
                    "duplicate": "duplicates",
                    "ineligible": "ineligible",
                    "execution_rejected": "execution_rejections",
                }[status]
                counts[key] += 1
            attempts.extend(lane_attempts)
            pools[channel], seconds[channel] = rows, elapsed
        jump_attempts, jump_rows, jump_seconds = self._generate_jump_pool(
            eligibility, archive_seen, schedules[JUMP_CHANNEL]
        )
        attempts.extend(jump_attempts)
        pools[JUMP_CHANNEL], seconds[JUMP_CHANNEL] = jump_rows, jump_seconds
        merged, seen = [], set()
        for channel in CHANNELS:
            for candidate in pools[channel]:
                if candidate["endpoint"] in seen:
                    attempts.append(
                        {
                            "planner_channel": channel,
                            "status": "cross_channel_duplicate",
                            "endpoint": candidate["endpoint"],
                        }
                    )
                    continue
                seen.add(candidate["endpoint"])
                merged.append(self._augment(candidate))
        selected, allocation = self._allocate(merged)
        for candidate in selected:
            self.population_state["channels"][candidate["provenance"]["planner_channel"]][
                "selected_for_oracle"
            ] += 1
        body = {
            "schema_version": "pmo_population_batch_v1",
            "batch_index": self.batches,
            "configuration": asdict(self.config),
            "candidates": selected,
            "attempts": attempts,
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
            "eligible_pool": {
                "pool_id": identity([row["candidate_id"] for row in merged]),
                "candidates": merged,
            },
            "allocation": allocation,
            "population_state_after_proposal": json.loads(json.dumps(self.population_state)),
        }
        self.pending = {**body, "batch_id": identity(body)}
        return json.loads(
            json.dumps(
                {
                    **self.pending,
                    "proposal_seconds": sum(seconds.values()),
                    "work_cache": {
                        **self.work_cache.report(),
                        "proposal_seconds_by_channel": seconds,
                    },
                    "new_oracle_calls": 0,
                }
            )
        )

    def add_measured_program(self, record, *, receipt_id, score, static_score=None):
        if "fiber_features" not in record:
            record = self._augment(record)
        # A recursive campaign creates a new physical cold-start pool whenever it revisits
        # an initialization parent, so requiring one pool identity forever refuses the
        # surviving-descendant behaviour this optimizer exists to test.  Absorb each
        # physical pool under one continuity identity instead of refusing it.
        record, new_generation_pool = self._pool_continuity.adapt(record)
        bootstrap = record.get("provenance", {}).get("dynamic_v21_bootstrap")
        if bootstrap is not None and new_generation_pool:
            # A fresh pool carries its own sampler state; keeping the previous pool's
            # would desynchronise the proposal streams from the candidates they produced.
            self.shallow_rng.bit_generator.state = bootstrap["shallow_rng"]
            self.structured_rng.bit_generator.state = bootstrap["structured_rng"]
        self._population_bootstrap_pool_id = self._pool_continuity.continuity_pool_id
        return ProgramOptimizer.add_measured_program(
            self,
            record,
            receipt_id=receipt_id,
            score=score,
            static_score=static_score,
        )

    def _observe_into_online_memory(self, candidate: dict[str, Any], score: float) -> None:
        """Record one counted oracle observation and its transition into the memory.

        Everything here comes from the candidate the controller already holds plus the
        score just charged -- no oracle internals, no task identity, no off-ledger
        property evaluation.
        """

        trace = candidate.get("trace") or {}
        endpoint = candidate.get("endpoint") or trace.get("endpoint")
        if not endpoint:
            return
        child = decode_state(trace["states"][-1]) if trace.get("states") else None
        self.online_memory.observe_scored_molecule(
            endpoint=endpoint, score=score, graph=child
        )
        provenance = candidate.get("provenance") or {}
        parent_score = provenance.get("parent_measured_score")
        source = candidate.get("source_state")
        if source is None or not trace.get("actions"):
            return
        parent_graph = decode_state(source)
        families = [a.get("model_family") for a in trace["actions"] if a.get("model_family")]
        touched = tuple(
            int(x) for x in (provenance.get("actual_changes") or {}).get(
                "changed_original_slots", ()
            )
        )
        self.online_memory.observe_transition(
            parent_graph=parent_graph,
            parent_endpoint=provenance.get("parent_endpoint") or "",
            parent_score=None if parent_score is None else float(parent_score),
            child_endpoint=endpoint,
            child_score=score,
            child_heavy=int(child.n_real_atoms) if child is not None else 0,
            family=families[0] if families else "unknown",
            touched_slots=touched,
        )

    def _reconstruct_online_memory(self) -> dict[str, Any]:
        """Rebuild the online memory from the archive's own counted observations.

        A snapshot taken before the memory was persistable carries none, so a plain
        resume would restart arm B's memory COLD -- quietly turning a 1000-call
        memory experiment into a 250-call one followed by a reset, with nothing in
        the artifact to show it.

        The learned content is reconstructible because every quantity the memory
        holds is a sum or a bounded max over counted observations, and all of those
        are durable in the archive.  Only the audit ordinal stamped on each donor
        row depends on arrival order, so a replay reproduces the learned state
        without needing the live ordering.

        This is a RECONSTRUCTION from durable evidence, not a restored in-process
        object: the live memory was never serialised, so byte-equality with it is
        unavailable and is not claimed.  Replay goes through the SAME observation
        path the live run used, so the two cannot drift apart.
        """

        scores: dict[str, float] = {}
        for observation in self.observations.values():
            endpoint, score = observation.get("endpoint"), observation.get("score")
            if endpoint and score is not None:
                scores[endpoint] = float(score)
        replayed = skipped = failed = 0
        for _, entry in sorted(self.entries.items()):
            score = scores.get(entry.get("endpoint"))
            if score is None:
                skipped += 1
                continue
            try:
                self._observe_into_online_memory(entry, score)
            except Exception:  # noqa: BLE001 - mirrors the live path's best-effort attribution
                failed += 1
            else:
                replayed += 1
        if self.entries and not replayed:
            # An archive full of scored entries that reconstructs nothing means the
            # join or the entry shape moved; resuming on an empty memory here would
            # be the silent arm change this whole path exists to prevent.
            raise ValueError(
                f"online-memory reconstruction replayed 0 of {len(self.entries)} "
                "archived entries; refusing to resume arm B on an empty memory"
            )
        return {
            "source": "archived_counted_observations",
            "entries": len(self.entries),
            "replayed": replayed,
            "skipped_unscored": skipped,
            "attribution_failures": failed,
            "byte_equality_with_live_memory_claimed": False,
        }

    def observe_batch(self, batch_id, outcomes):
        if self.pending is None or self.pending.get("batch_id") != batch_id:
            raise ValueError("PMO population outcome lacks its pending batch")
        candidates = {row["candidate_id"]: row for row in self.pending["candidates"]}
        prior_best = self.population_state["best_score"]
        scored = []
        for outcome in outcomes:
            candidate = candidates.get(outcome.get("candidate_id"))
            if candidate is None:
                continue
            channel = candidate["provenance"]["planner_channel"]
            counts = self.population_state["channels"][channel]
            counts["charged_outcomes"] += 1
            score = outcome.get("score")
            if score is None:
                continue
            score = float(score)
            scored.append(score)
            counts["scored_outcomes"] += 1
            if self.online_memory is not None:
                # Feed the memory only COUNTED observations, at the moment the ledger
                # records them. A failure here must neither kill a scored run nor pass
                # silently, so it is counted and surfaced in the snapshot.
                try:
                    self._observe_into_online_memory(candidate, score)
                except Exception:  # noqa: BLE001 - attribution is best-effort, never fatal
                    self.online_memory_attribution_failures += 1
            parent_score = candidate["provenance"].get("parent_measured_score")
            if parent_score is not None:
                gain = score - float(parent_score)
                parent_id = candidate["provenance"]["entry_id"]
                stats = self.population_state["parent_outcomes"].setdefault(
                    parent_id,
                    {"children": 0, "positive_improvement_sum": 0.0},
                )
                stats["children"] += 1
                if gain > 0:
                    counts["parent_improvements"] += 1
                    counts["positive_improvement_sum"] += gain
                    stats["positive_improvement_sum"] += gain
                # Joint credit is the allocation authority; the marginals above stay as
                # telemetry.  PMO maximizes, so the direction is named explicitly rather
                # than inferred from the sign of the gain.
                try:
                    key = credit_key_from_candidate(candidate)
                except (ValueError, KeyError):
                    self.population_state["credit_key_failures"] += 1
                else:
                    self.credit.observe(
                        key,
                        improvement(score, float(parent_score), direction="maximize"),
                    )
        ProgramOptimizer.observe_batch(self, batch_id, outcomes)
        current = max(scored, default=prior_best if prior_best is not None else -math.inf)
        improved = prior_best is None or current > prior_best
        if improved:
            self.population_state["best_score"] = current
        advance_plateau_state(self.population_state, improved=improved)

    def snapshot(self, *, include_history=True):
        snapshot = ProgramOptimizer.snapshot(self, include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["pmo_population"] = {
            "jump_checkpoint_id": self.jump_checkpoint_id,
            "jump_rng": self.jump_rng.bit_generator.state,
            "shallow_rng": self.shallow_rng.bit_generator.state,
            "structured_rng": self.structured_rng.bit_generator.state,
            "population_state": self.population_state,
            "bootstrap_pool_id": self._population_bootstrap_pool_id,
            # The joint credit IS the allocation authority, and the continuity adapter
            # decides which measured programs the archive admits.  Omitting them made a
            # resumed campaign silently revert to the cold-start acquisition path and
            # re-learn from nothing, which is measurably worse than the credit path.
            "credit": self.credit.payload(),
            "pool_continuity": self._pool_continuity.payload(),
            # Same reason as credit and pool_continuity above: without this a
            # resumed arm B silently restarts its memory COLD and drifts back
            # toward arm A, invisibly, for every remaining round.
            "online_memory": (
                None if self.online_memory is None else self.online_memory.payload()
            ),
            # Records that this arm's memory was REBUILT from archived observations
            # rather than restored from a persisted payload, so the revision boundary
            # is legible from the run's own artifact instead of from the commit log.
            "online_memory_reconstruction": self.online_memory_reconstruction,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(
        cls,
        snapshot,
        *,
        hierarchy=None,
        jump_checkpoint=None,
        enable_online_memory: bool = False,
        construction_prior=None,
        construction_prior_spec=None,
    ):
        # `run_program_campaign` passes optimizer_kwargs to BOTH the constructor and
        # this classmethod, so the arm flag has to be accepted here too -- and it has
        # to reach the constructor, or a resumed arm B would rebuild as arm A and then
        # refuse its own snapshot's memory payload.  The same applies to the
        # construction prior: a resumed run that silently dropped it would continue as
        # the uniform arm with nothing in the artifact to show it.
        if jump_checkpoint is None:
            raise ValueError("PMO population restore requires its sanitized jump checkpoint")
        result = ProgramOptimizer.restore.__func__(
            cls,
            snapshot,
            hierarchy=hierarchy,
            constructor_kwargs={
                "jump_checkpoint": jump_checkpoint,
                "enable_online_memory": enable_online_memory,
                "construction_prior": construction_prior,
                "construction_prior_spec": construction_prior_spec,
            },
        )
        state = snapshot.get("pmo_population")
        if (
            not isinstance(state, dict)
            or state.get("jump_checkpoint_id") != result.jump_checkpoint_id
        ):
            raise ValueError("PMO population snapshot/checkpoint identity changed")
        result.jump_rng.bit_generator.state = state["jump_rng"]
        result.shallow_rng.bit_generator.state = state["shallow_rng"]
        result.structured_rng.bit_generator.state = state["structured_rng"]
        result.population_state = json.loads(json.dumps(state["population_state"]))
        memory_payload = state.get("online_memory")
        if memory_payload is not None:
            # A snapshot carrying memory state may only be restored into an arm that
            # HAS a memory: silently discarding it would resume arm B as arm A.
            if result.online_memory is None:
                raise ValueError(
                    "snapshot carries online-memory state but this controller was "
                    "constructed without it; resuming would silently change the arm"
                )
            result.online_memory.restore_payload(memory_payload)
        elif result.online_memory is not None:
            if result.online_memory.ordinal:
                raise ValueError(
                    "controller has a warm online memory but the snapshot carries none"
                )
            result.online_memory_reconstruction = result._reconstruct_online_memory()
        result._population_bootstrap_pool_id = state["bootstrap_pool_id"]
        # Absent keys restore empty objects so a snapshot taken before these were
        # persisted still loads, rather than failing a resume outright.
        result.credit = PopulationCredit.restore(state["credit"]) if state.get("credit") else PopulationCredit()
        result._pool_continuity = BootstrapPoolContinuity.restore(state.get("pool_continuity"))
        return result


__all__ = [
    "CHANNELS",
    "JUMP_CHANNEL",
    "MODE_BY_CHANNEL",
    "SCHEMA",
    "PmoPopulationController",
    "advance_plateau_state",
    "initial_population_state",
    "population_features",
]
