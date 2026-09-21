"""Measured-score adaptation over variable coordinated constructions.

This is a bounded optimizer, not importance sampling or a learned future value.
Its archive retains exact construction provenance; observations arrive only for
locked endpoints under an identified oracle protocol. The legacy broad channel
is dispatched before WHERE and never silently replaced by program retrieval.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from compose_v4.control.current_state_edits import current_state_program
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_learning_data import evidence_niches, exploration_niches
from compose_v4.control.edit_program import EditProgram, attachment_bindings, extract_program
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
    program_size_profile,
)
from compose_v4.control.molecular_task_search import (
    MolecularSearchState,
    dispatch_complete_proposal,
)
from compose_v4.control.program_decomposition import verified_branches
from compose_v4.control.program_mutation import (
    PARAMETER_MOVES,
    attachment_mutation_choices,
    branch_components,
    mutate_attachment,
    mutate_parameter,
    parameter_choices,
    select_branch,
)
from compose_v4.control.program_work_cache import ProgramWorkCache
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state


@dataclass(frozen=True)
class ProgramSearchConfig:
    seed: int = 20260913
    adaptive: bool = True
    channel_probabilities: tuple[float, float, float] = (0.7, 0.2, 0.1)
    max_primitives: int = 32
    max_blocks: int = 8
    attempts_per_batch: int = 128
    candidates_per_batch: int = 16
    wall_seconds: float = 60.0
    max_bindings: int = 64
    exploration: float = 0.2
    double_mutation_probability: float = 0.25
    require_broad_runtime: bool = True
    parent_allocation: str = "score_rank"
    continuation_root: str = "ancestral_constructor"
    current_state_edit_probability: float = 0.0
    decompose_programs: bool = False
    score_direction: str = "minimize"
    proposal_mode: str = "mixed"
    proposal_cache_entries: int = 0
    mutation_sampling: str = "random"
    mutation_context_limit: int = 128
    composition_probability: float = 0.0
    max_composed_programs: int = 2
    composition_stop_probability: float = 0.5
    composition_donor_trials: int = 8
    composition_execution_trials: int = 16
    cold_start_retrieval_candidates: int = 0

    @classmethod
    def parent_edit_recipe(cls, *, seed=20260912, score_direction="minimize"):
        """Explicit new development recipe; legacy constructors keep old defaults.

        The matched learned/control arms use the same score-blind parent law.
        Niche allocation remains a separate opt-in ablation.
        """
        return cls(
            seed=seed,
            score_direction=score_direction,
            parent_allocation="score_blind",
            continuation_root="exact_current_state",
            current_state_edit_probability=0.25,
            decompose_programs=True,
        )

    @classmethod
    def program_only_recipe(cls, *, seed=20260912, score_direction="minimize"):
        """Named optimization proposal, not a sample from the frozen reference.

        Retain the repaired program engine and renormalize its 70:20 allocation.
        Existing-atom edits still execute through the production enumerators.
        """
        from dataclasses import replace

        return replace(
            cls.parent_edit_recipe(seed=seed, score_direction=score_direction),
            proposal_mode="program_only",
            channel_probabilities=(7 / 9, 2 / 9, 0.0),
            require_broad_runtime=False,
        )

    @classmethod
    def two_program_recipe(cls, *, seed=20260912, score_direction="minimize"):
        """Fast program search with a protected composition fixed to two.

        The architecture supports a bounded variable number of transformations,
        but this named first experiment fixes K=2.  Every later transformation
        is rebound on the actual predecessor, and only the completed composition
        reaches the task evaluator.
        """
        from dataclasses import replace

        return replace(
            cls.program_only_recipe(seed=seed, score_direction=score_direction),
            composition_probability=0.25,
            max_composed_programs=2,
        )

    def __post_init__(self):
        if self.mutation_sampling not in ("random", "untried"):
            raise ValueError("unknown mutation sampling policy")
        if type(self.mutation_context_limit) is not int or self.mutation_context_limit < 1:
            raise ValueError("mutation context limit must be a positive integer")
        if type(self.max_composed_programs) is not int or self.max_composed_programs < 2:
            raise ValueError("composition needs a maximum of at least two programs")
        if type(self.composition_donor_trials) is not int or self.composition_donor_trials < 1:
            raise ValueError("composition donor trials must be a positive integer")
        if (
            type(self.composition_execution_trials) is not int
            or self.composition_execution_trials < 1
        ):
            raise ValueError("composition execution trials must be a positive integer")
        if type(self.proposal_cache_entries) is not int or self.proposal_cache_entries < 0:
            raise ValueError("proposal cache capacity must be a nonnegative integer")
        if (
            type(self.cold_start_retrieval_candidates) is not int
            or self.cold_start_retrieval_candidates < 0
            or self.cold_start_retrieval_candidates > self.candidates_per_batch
        ):
            raise ValueError("cold-start retrieval must fit the candidate batch")
        if self.proposal_mode not in ("mixed", "program_only"):
            raise ValueError("unknown explicit proposal mode")
        if self.proposal_mode == "program_only" and self.require_broad_runtime:
            raise ValueError("program-only mode cannot require a reference runtime")
        if self.continuation_root not in ("ancestral_constructor", "exact_current_state"):
            raise ValueError("unknown continuation-root contract")
        if self.score_direction not in ("minimize", "maximize"):
            raise ValueError("score direction must be minimize or maximize")
        if (
            not 0 <= self.current_state_edit_probability <= 1
            or type(self.decompose_programs) is not bool
        ):
            raise ValueError("invalid current-state/decomposition configuration")
        if self.parent_allocation not in (
            "score_rank",
            "score_blind",
            "niche_score",
            "niche_evidence",
        ):
            raise ValueError(
                "parent allocation must be score_rank, score_blind, niche_score or niche_evidence"
            )
        if any(
            type(v) is not int or v < 1
            for v in (
                self.max_primitives,
                self.max_blocks,
                self.attempts_per_batch,
                self.candidates_per_batch,
                self.max_bindings,
            )
        ):
            raise ValueError("program work/candidate caps must be positive integers")
        if (
            type(self.seed) is not int
            or self.seed < 0
            or type(self.adaptive) is not bool
            or type(self.require_broad_runtime) is not bool
        ):
            raise ValueError("invalid explicit seed/adaptation/broad-runtime setting")
        if (
            not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0
            or not 0 < self.exploration < 1
            or not 0 <= self.double_mutation_probability <= 1
            or not 0 <= self.composition_probability <= 1
            or not 0 <= self.composition_stop_probability <= 1
        ):
            raise ValueError("invalid program search wall limit or exploration")
        weights = self.channel_probabilities
        if (
            len(weights) != 3
            or any(not math.isfinite(v) or v < 0 for v in weights)
            or any(v <= 0 for v in weights[:2])
            or not math.isclose(sum(weights), 1)
            or (self.proposal_mode == "mixed" and weights[2] <= 0)
            or (self.proposal_mode == "program_only" and weights[2] != 0)
        ):
            raise ValueError("channel probabilities must sum to one and match the proposal mode")


class ProgramOptimizer:
    def __init__(
        self,
        config: ProgramSearchConfig,
        *,
        source_group: str,
        oracle_protocol: str,
        hierarchy=None,
    ):
        if not source_group or not oracle_protocol:
            raise ValueError("program search requires source-group and oracle-protocol identities")
        if config.require_broad_runtime and hierarchy is None:
            raise ValueError("production program optimizer requires its broad hierarchy")
        self.config, self.source_group, self.oracle_protocol, self.hierarchy = (
            config,
            source_group,
            oracle_protocol,
            hierarchy,
        )
        self.rng = np.random.default_rng(config.seed)
        self.entries, self.observations, self.duplicate_counts = {}, {}, {}
        self.failed_endpoints, self.history, self.pending = set(), [], None
        self.batches = 0
        self._program_cache, self._source_cache, self._branch_cache = {}, {}, {}
        self._niche_cache = None
        self.work_cache = ProgramWorkCache(config.proposal_cache_entries)
        self.mutation_choices_seen = {}

    def _program(self, entry):
        key = identity(entry["program"])
        if key not in self._program_cache:
            self._program_cache[key] = EditProgram.from_payload(entry["program"])
        return self._program_cache[key]

    def _source(self, entry):
        key = identity(entry["source_state"])
        if key not in self._source_cache:
            self._source_cache[key] = decode_state(entry["source_state"])
        return self._source_cache[key]

    def add_measured_program(self, record, *, receipt_id: str, score: float, static_score=None):
        """Register a verified construction and genuine endpoint observation.

        Caller-supplied observations must bind the configured protocol. This
        boundary replays a newly admitted construction once, then caches it.
        """
        if (
            record["source_group"] != self.source_group
            or record["oracle_protocol"] != self.oracle_protocol
        ):
            raise ValueError("measured program belongs to another source or oracle domain")
        if not receipt_id or not math.isfinite(score):
            raise ValueError("measured program needs an identified finite observation")
        if static_score is not None and not math.isfinite(static_score):
            raise ValueError("static-control inheritance must be a finite recorded preference")
        source, program = self._source(record), self._program(record)
        # Admission verifies stored construction work, not the next proposal's
        # horizon. Only the explicit fresh-state recipe separates these clocks.
        fresh = self.config.continuation_root == "exact_current_state"
        _, replay = execute_program_graph(
            source,
            compile_program_graph(program),
            tuple(record["assignment"]),
            max_primitives=max(self.config.max_primitives, len(program.marks))
            if fresh
            else self.config.max_primitives,
            max_blocks=max(self.config.max_blocks, len(program.blocks))
            if fresh
            else self.config.max_blocks,
        )
        if (
            replay["endpoint"] != record["endpoint"]
            or replay["states"] != record["trace"]["states"]
        ):
            raise ValueError("measured program fails exact replay or endpoint identity")
        key = identity(
            {
                "source": record["source_state"],
                "program": program.payload(),
                "assignment": record["assignment"],
            }
        )
        endpoint = record["endpoint"]
        prior = self.observations.get(receipt_id)
        observation = {
            "endpoint": endpoint,
            "score": float(score),
            "oracle_protocol": self.oracle_protocol,
        }
        if prior is not None and prior != observation:
            raise ValueError("observation receipt was rebound to another endpoint/score")
        self.observations[receipt_id] = observation
        if key not in self.entries:
            self.entries[key] = json.loads(
                json.dumps(
                    {
                        **record,
                        "program": program.payload(),
                        "entry_id": key,
                        "static_score": float(score if static_score is None else static_score),
                    }
                )
            )
        return key

    def selection(self):
        """Rank measured endpoints, not repeated decompositions; preserve exploration."""
        groups, labels = defaultdict(list), defaultdict(list)
        for receipt in self.observations.values():
            labels[receipt["endpoint"]].append(receipt["score"])
        for key, entry in sorted(self.entries.items()):
            groups[entry["endpoint"]].append(key)
        endpoints = sorted(groups)
        if not endpoints:
            raise ValueError("program search archive is empty")
        scores = np.asarray(
            [
                np.mean(labels[s])
                if self.config.adaptive
                else np.mean([self.entries[k]["static_score"] for k in groups[s]])
                for s in endpoints
            ]
        )
        costs = scores if self.config.score_direction == "minimize" else -scores
        ranks = np.asarray([1 + np.sum(costs < v) for v in costs], dtype=float)
        quality = (
            np.ones(len(endpoints)) if self.config.parent_allocation == "score_blind" else 1 / ranks
        )
        if self.config.parent_allocation in ("niche_score", "niche_evidence"):
            partition = (
                exploration_niches
                if self.config.parent_allocation == "niche_score"
                else evidence_niches
            )
            key = (tuple(endpoints), tuple(costs), partition.__name__)
            if self._niche_cache is None or self._niche_cache[0] != key:
                self._niche_cache = (key, partition(endpoints, -costs))
            niches = self._niche_cache[1]
            selected = [set(group) for group in niches["selected"]]
            if self.config.parent_allocation == "niche_score":
                # Legacy: mass per NICHE is uniform, so a structurally isolated
                # endpoint draws a full niche's exploitation mass however badly it
                # scores, and score only breaks ties inside a niche.
                quality = np.asarray(
                    [sum(1 / len(group) for group in selected if s in group) for s in endpoints]
                )
            else:
                # Evidence weighting. Niching decides WHICH endpoints may draw
                # exploitation mass -- it caps how many members of one structural
                # basin compete, which is what niching is for -- and the measured
                # score decides HOW MUCH each of them draws. An endpoint that is
                # alone in its basin is admitted, never endowed: its mass is the
                # same rank reciprocal `score_rank` would have given it.
                eligible = set().union(*selected) if selected else set()
                quality = np.asarray(
                    [
                        (1 / rank) if endpoint in eligible else 0.0
                        for endpoint, rank in zip(endpoints, ranks, strict=True)
                    ]
                )
                if not quality.sum():
                    raise ValueError("niche evidence allocation admitted no scored endpoint")
        # Exhaustion tempers exploitation, not the explicit exploration floor.
        quality /= np.asarray(
            [1 + np.mean([self.duplicate_counts.get(k, 0) for k in groups[s]]) for s in endpoints]
        )
        quality = (
            self.config.exploration / len(quality)
            + (1 - self.config.exploration) * quality / quality.sum()
        )
        keys, weights = [], []
        for endpoint, mass in zip(endpoints, quality, strict=True):
            representatives = groups[endpoint]
            unexhausted = np.asarray(
                [1 / (1 + self.duplicate_counts.get(k, 0)) for k in representatives]
            )
            for key, weight in zip(representatives, unexhausted / unexhausted.sum(), strict=True):
                keys.append(key)
                weights.append(mass * weight)
        probabilities = np.asarray(weights) / sum(weights)
        return keys, probabilities

    def _parent(self):
        keys, weights = self.selection()
        index = int(self.rng.choice(len(keys), p=weights))
        return self.entries[keys[index]], {
            "entry_id": keys[index],
            "parent_probability": float(weights[index]),
            "parent_measured_score": float(
                np.mean(
                    [
                        r["score"]
                        for r in self.observations.values()
                        if r["endpoint"] == self.entries[keys[index]]["endpoint"]
                    ]
                )
            ),
        }

    def _mutate(self, entry):
        if (
            self.config.current_state_edit_probability
            and self.rng.random() < self.config.current_state_edit_probability
        ):
            source = decode_state(entry["trace"]["states"][-1])
            program, binding, detail = current_state_program(
                source, self.rng, work_cache=self.work_cache if self.work_cache.capacity else None
            )
            return (
                source,
                program,
                binding,
                {
                    "current_state_edit": detail,
                    **self._continuation_lineage(entry),
                },
            )
        source, program, binding = (
            self._source(entry),
            self._program(entry),
            tuple(entry["assignment"]),
        )
        edits = []
        number = 1 + int(self.rng.random() < self.config.double_mutation_probability)
        for _ in range(number):
            parameters = {m: parameter_choices(program, m) for m in PARAMETER_MOVES}
            attachments = self.work_cache.get(
                "mutation_bindings",
                (
                    identity(encode_state(source)),
                    program.program_id,
                    binding,
                    self.config.max_bindings,
                ),
                lambda program=program, binding=binding: attachment_mutation_choices(
                    source, program, binding, max_bindings=self.config.max_bindings
                ),
            )
            available = (["attachment"] if attachments[1] else []) + [
                m for m in PARAMETER_MOVES if parameters[m]
            ]
            context, untried = None, None
            if self.config.mutation_sampling == "untried":
                # Conditional one-decision menus, never an attachment x payload
                # Cartesian frontier. Later decisions see the modified program.
                context = identity(
                    {
                        "source": encode_state(source),
                        "program": program.payload(),
                        "binding": binding,
                        "max_bindings": self.config.max_bindings,
                    }
                )
                prior = self.mutation_choices_seen.get(context, set())
                menus = {**parameters, "attachment": tuple(a for a, _ in attachments[1])}
                untried = {
                    move: tuple(c for c in menus[move] if identity((move, c)) not in prior)
                    for move in available
                }
                available = [m for m in available if untried[m]]
            if not available:
                raise ValueError(
                    "bounded mutation context exhausted"
                    if context
                    else "program has no available bounded mutation"
                )
            move = available[int(self.rng.integers(len(available)))]
            if move == "attachment":
                prepared = attachments
                if untried is not None:
                    permitted = set(untried[move])
                    prepared = (
                        attachments[0],
                        [(a, d) for a, d in attachments[1] if a in permitted],
                    )
                binding, census = mutate_attachment(
                    source,
                    program,
                    binding,
                    self.rng,
                    max_bindings=self.config.max_bindings,
                    prepared=prepared,
                )
                choice = binding
                edits.append({"kind": move, "available_kinds": available, **census})
            else:
                choices = parameters[move] if untried is None else untried[move]
                choice = choices[int(self.rng.integers(len(choices)))]
                program = mutate_parameter(program, move, choice)
                edits.append(
                    {
                        "kind": move,
                        "available_kinds": available,
                        "choice": choice,
                        "choices": len(choices),
                    }
                )
            if context is not None:
                if (
                    context not in self.mutation_choices_seen
                    and len(self.mutation_choices_seen) >= self.config.mutation_context_limit
                ):
                    del self.mutation_choices_seen[next(iter(self.mutation_choices_seen))]
                self.mutation_choices_seen.setdefault(context, set()).add(identity((move, choice)))
                edits[-1]["sampling"] = "untried_conditional_choice"
                edits[-1]["context"] = context
        return source, program, binding, {"mutations": edits}

    def _branches(self, entry):
        program, binding = self._program(entry), tuple(entry["assignment"])
        if not self.config.decompose_programs:
            return program, binding, {"status": "legacy_blocks"}
        key = entry["entry_id"]
        if key not in self._branch_cache:
            self._branch_cache[key] = verified_branches(
                self._source(entry),
                program,
                binding,
                max_primitives=self.config.max_primitives,
            )
        return self._branch_cache[key]

    def _recombine(self, entry):
        source, program, binding = (
            self._source(entry),
            self._program(entry),
            tuple(entry["assignment"]),
        )
        program, binding, decomposition = self._branches(entry)
        components = branch_components(program)
        count = min(
            len(components), 1 + int(self.rng.random() < self.config.double_mutation_probability)
        )
        removed = sorted(
            int(i) for i in self.rng.choice(len(components), size=count, replace=False)
        )
        retained = sorted(
            set(range(len(program.blocks))) - {i for c in removed for i in components[c]}
        )
        pieces, donors = [], []
        if retained:
            keep, roots = select_branch(program, retained)
            pieces.append((keep, tuple(binding[i] for i in roots)))
        for _ in removed:
            donor, receipt = self._parent()
            donor_program, _, donor_decomposition = self._branches(donor)
            branches = branch_components(donor_program)
            branch = branches[int(self.rng.integers(len(branches)))]
            donor_program, _ = select_branch(donor_program, branch)
            census = self.work_cache.get(
                "donor_bindings",
                (
                    identity(encode_state(source)),
                    donor_program.program_id,
                    self.config.max_bindings,
                ),
                lambda donor_program=donor_program: attachment_bindings(
                    donor_program, source, max_bindings=self.config.max_bindings
                ),
            )
            if not census.assignments:
                raise ValueError("no context-compatible donor branch attachment")
            distances = np.asarray(census.context_distances, dtype=float)
            weights = np.exp(-(distances - distances.min()))
            weights = 0.1 / len(weights) + 0.9 * weights / weights.sum()
            at = int(self.rng.choice(len(weights), p=weights))
            pieces.append((donor_program, census.assignments[at]))
            donors.append(
                {
                    **receipt,
                    "blocks": branch,
                    "bindings_truncated": census.truncated,
                    "binding_probability": float(weights[at]),
                    "decomposition": donor_decomposition,
                }
            )
        if len(pieces) == 1:
            combined, anchors = pieces[0]
        else:
            combined, anchors = combine_bound_programs(
                source,
                tuple(pieces),
                max_primitives=self.config.max_primitives,
                max_blocks=self.config.max_blocks,
            )
        return (
            source,
            combined,
            anchors,
            {
                "removed_branches": removed,
                "donors": donors,
                "decomposition": decomposition,
            },
        )

    @staticmethod
    def _receipt_stages(receipt, *, label_prefix):
        """Expose verified block boundaries for exact program re-extraction."""
        stages, start = [], 0
        for index, block in enumerate(receipt["blocks"]):
            stop = block["stop"]
            stages.append(
                {
                    "name": f"{label_prefix}:{index}:{block['label']}",
                    "actions": receipt["actions"][start:stop],
                    "states": receipt["states"][start : stop + 1],
                    "endpoint": canonical_state_key(decode_state(receipt["states"][stop])),
                }
            )
            start = stop
        return stages

    def _bound_composition_component(self, state, *, max_primitives, max_blocks):
        """Draw one bounded archived branch and bind it to the supplied state.

        Candidate donor entries are sampled by the configured parent law.  The
        task oracle is never consulted.  Context distance only allocates mass
        inside the finite, executor-supported binding pool.
        """
        keys, weights = self.selection()
        count = min(self.config.composition_donor_trials, len(keys))
        selected = self.rng.choice(len(keys), size=count, replace=False, p=weights)
        choices = []
        for position in selected:
            entry_id = keys[int(position)]
            entry = self.entries[entry_id]
            program, _, decomposition = self._branches(entry)
            components = branch_components(program)
            for component in components:
                branch, _ = select_branch(program, component)
                if len(branch.marks) > max_primitives or len(branch.blocks) > max_blocks:
                    continue
                census = self.work_cache.get(
                    "composition_bindings",
                    (
                        identity(encode_state(state)),
                        branch.program_id,
                        self.config.max_bindings,
                    ),
                    lambda branch=branch: attachment_bindings(
                        branch, state, max_bindings=self.config.max_bindings
                    ),
                )
                for assignment, distance in zip(
                    census.assignments, census.context_distances, strict=True
                ):
                    choices.append(
                        {
                            "entry_id": entry_id,
                            "program": branch,
                            "assignment": assignment,
                            "blocks": component,
                            "context_distance": distance,
                            "parent_probability": float(weights[int(position)]),
                            "bindings": len(census.assignments),
                            "donor_components": len(components),
                            "bindings_truncated": census.truncated,
                            "decomposition": decomposition,
                        }
                    )
        if not choices:
            raise ValueError("no reusable component binds within remaining composition work")
        base = np.asarray(
            [
                choice["parent_probability"] / (choice["donor_components"] * choice["bindings"])
                for choice in choices
            ],
            dtype=float,
        )
        distance = np.asarray([choice["context_distance"] for choice in choices], dtype=float)
        contextual = base * np.exp(-(distance - distance.min()))
        proposal = 0.1 * base / base.sum() + 0.9 * contextual / contextual.sum()
        order = self.rng.choice(
            len(choices),
            size=min(self.config.composition_execution_trials, len(choices)),
            replace=False,
            p=proposal,
        )
        failures = []
        for raw_index in order:
            at = int(raw_index)
            choice = choices[at]

            def execute(choice=choice):
                try:
                    product, receipt = execute_program_graph(
                        state,
                        compile_program_graph(choice["program"]),
                        choice["assignment"],
                        max_primitives=max_primitives,
                        max_blocks=max_blocks,
                    )
                except ValueError as error:
                    return {"error": str(error)}
                return {"product": encode_state(product), "receipt": receipt}

            execution_key = identity(
                {
                    "state": encode_state(state),
                    "program": choice["program"].payload(),
                    "assignment": choice["assignment"],
                    "max_primitives": max_primitives,
                    "max_blocks": max_blocks,
                }
            )
            result = self.work_cache.get("composition_execution", execution_key, execute)
            if "error" in result:
                failures.append(result["error"])
                continue
            return (
                choice,
                float(proposal[at]),
                decode_state(result["product"]),
                result["receipt"],
                {
                    "choices_considered": len(choices),
                    "exact_executions_attempted": len(failures) + 1,
                    "failed_execution_reasons": dict(Counter(failures)),
                },
            )
        summary = dict(Counter(failures))
        raise ValueError(
            f"no exact-executable component in {len(order)} bounded choices: {summary}"
        )

    def _compose_programs(self, entry):
        """Complete a bounded variable-size composition before task evaluation."""
        source = decode_state(entry["trace"]["states"][-1])
        current, stages, components, intermediates = source, [], [], []
        active_created_slots = set()
        remaining_primitives, remaining_blocks = (
            self.config.max_primitives,
            self.config.max_blocks,
        )
        stop_reason = None
        for index in range(self.config.max_composed_programs):
            # The first choice must leave at least one primitive and block for
            # the required second component. Additional components are optional.
            reserve = int(index == 0)
            component, probability, product, receipt, compatibility = (
                self._bound_composition_component(
                    current,
                    max_primitives=remaining_primitives - reserve,
                    max_blocks=remaining_blocks - reserve,
                )
            )
            created_inputs = sorted(set(component["assignment"]) & active_created_slots)
            stages.extend(self._receipt_stages(receipt, label_prefix=f"composition_{index + 1}"))
            components.append(
                {
                    **{k: v for k, v in component.items() if k not in ("program", "assignment")},
                    "program_id": component["program"].program_id,
                    "assignment": list(component["assignment"]),
                    "proposal_probability": probability,
                    "input_slots_created_by_earlier_components": created_inputs,
                    "primitive_edits": len(receipt["actions"]),
                    "block_count": len(receipt["blocks"]),
                    "compatibility": compatibility,
                }
            )
            for action in receipt["actions"]:
                payload = action["payload"]
                if action["executor_rule"] == "atom_insert":
                    active_created_slots.add(payload["slot"])
                elif action["executor_rule"] == "atom_delete":
                    active_created_slots.discard(payload["v"])
            current = product
            intermediates.append(receipt["endpoint"])
            remaining_primitives -= len(receipt["actions"])
            remaining_blocks -= len(receipt["blocks"])
            completed = index + 1
            if completed < 2:
                if remaining_primitives < 1 or remaining_blocks < 1:
                    raise ValueError("first composition component leaves no work for the second")
                continue
            if completed == self.config.max_composed_programs:
                stop_reason = "maximum_components"
                break
            if remaining_primitives < 1 or remaining_blocks < 1:
                stop_reason = "work_exhausted"
                break
            if self.rng.random() < self.config.composition_stop_probability:
                stop_reason = "sampled_stop"
                break
        program, assignment = extract_program(source, stages)
        if (
            len(program.marks) > self.config.max_primitives
            or len(program.blocks) > self.config.max_blocks
        ):
            raise RuntimeError("protected pair escaped its complete proposal work limits")
        return (
            source,
            program,
            assignment,
            {
                "program_composition": {
                    "components": components,
                    "component_count": len(components),
                    "intermediate_endpoints": intermediates[:-1],
                    "intermediate_task_evaluations": 0,
                    "primitive_edits": len(program.marks),
                    "blocks": len(program.blocks),
                    "stop_reason": stop_reason,
                },
                **self._continuation_lineage(entry),
            },
        )

    @staticmethod
    def _continuation_lineage(entry):
        ancestors = [
            *entry.get("construction_ancestry", []),
            {
                "entry_id": entry["entry_id"],
                "source_state_sha256": identity(entry["source_state"]),
                "endpoint": entry["endpoint"],
                "primitive_edits": len(entry["program"]["marks"]),
            },
        ]
        return {
            "construction_ancestry": ancestors,
            "original_seed_state": entry.get("original_seed_state", entry["source_state"]),
            "ancestral_primitive_edits": sum(a["primitive_edits"] for a in ancestors),
            "accounting": "new proposal work only; archive/history retain all earlier attempts and queries",
        }

    def _broad(self, entry):
        if self.hierarchy is None:
            raise ValueError("broad_runtime_unavailable_in_local_development")
        program = self._program(entry)
        fresh = self.config.continuation_root == "exact_current_state"
        remaining = self.config.max_primitives - (0 if fresh else len(program.marks))
        blocks = self.config.max_blocks - (0 if fresh else len(program.blocks))
        if remaining < 1 or blocks < 1:
            raise ValueError("no remaining work for a complete broad continuation")
        state = decode_state(entry["trace"]["states"][-1])
        node = MolecularSearchState.start(state, budget=remaining, root_id=entry["entry_id"])
        tail, anchors, info = self.hierarchy.complete_reference_program(
            node, self.rng, max_options=min(3, blocks)
        )
        _, tail_trace = execute_program_graph(
            state, compile_program_graph(tail), anchors, max_primitives=remaining, max_blocks=blocks
        )
        if fresh:
            return (
                state,
                tail,
                anchors,
                {
                    "broad": info,
                    "reused_exact_prefix": True,
                    **self._continuation_lineage(entry),
                },
            )
        # Reuse the exact admitted prefix, then compile the complete constructor.
        stages, start = [], 0
        for trace in (entry["trace"], tail_trace):
            start = 0
            for block in trace["blocks"]:
                stop = block["stop"]
                stages.append(
                    {
                        "name": block["label"],
                        "actions": trace["actions"][start:stop],
                        "states": trace["states"][start : stop + 1],
                        "endpoint": canonical_state_key(decode_state(trace["states"][stop])),
                    }
                )
                start = stop
        source = self._source(entry)
        combined, binding = extract_program(source, stages)
        return source, combined, binding, {"broad": info, "reused_exact_prefix": True}

    def propose_batch(self, eligibility):
        if self.pending is not None:
            raise ValueError("resolve the locked pending batch before generating another")
        started, attempts, candidates = perf_counter(), [], []
        seen = {r["endpoint"] for r in self.observations.values()} | self.failed_endpoints
        for attempt in range(self.config.attempts_per_batch):
            if (
                len(candidates) >= self.config.candidates_per_batch
                or perf_counter() - started >= self.config.wall_seconds
            ):
                break
            entry, parent = self._parent()
            chosen = {}

            def program_sampler(channel, chosen=chosen, entry=entry):
                chosen["channel"] = channel
                return self._mutate(entry) if channel == "mutation" else self._recombine(entry)

            def broad_sampler(chosen=chosen, entry=entry):
                chosen["channel"] = "broad"
                return self._broad(entry)

            try:
                if self.config.composition_probability and (
                    self.rng.random() < self.config.composition_probability
                ):
                    channel = chosen["channel"] = "program_composition"
                    source, program, binding, metadata = self._compose_programs(entry)
                else:
                    channel, (source, program, binding, metadata) = dispatch_complete_proposal(
                        self.rng,
                        program_sampler=program_sampler,
                        broad_sampler=broad_sampler,
                        probabilities=self.config.channel_probabilities,
                        proposal_mode=self.config.proposal_mode,
                    )
                graph = compile_program_graph(program)
                size = program_size_profile(graph, source.n_real_atoms)
                size["measured_parent_heavy_atoms"] = decode_state(
                    entry["trace"]["states"][-1]
                ).n_real_atoms
                size["delta_from_measured_parent"] = (
                    size["final_heavy_atoms"] - size["measured_parent_heavy_atoms"]
                )
                chosen["program_size"] = size

                def execute(source=source, graph=graph, binding=binding):
                    try:
                        _, receipt = execute_program_graph(
                            source,
                            graph,
                            binding,
                            max_primitives=self.config.max_primitives,
                            max_blocks=self.config.max_blocks,
                        )
                    except ValueError as error:
                        return {"error": str(error)}
                    return {"trace": receipt}

                execution_key = identity(
                    {
                        "source": encode_state(source),
                        "program": program.payload(),
                        "binding": binding,
                        "max_primitives": self.config.max_primitives,
                        "max_blocks": self.config.max_blocks,
                    }
                )
                chosen["execution_key"] = execution_key
                executed = self.work_cache.get(
                    "exact_execution",
                    execution_key,
                    execute,
                )
                if "error" in executed:
                    raise ValueError(executed["error"])
                trace = executed["trace"]
                if (
                    decode_state(trace["states"][-1]).n_real_atoms != size["final_heavy_atoms"]
                    or trace["capacity_timeline"] != size["capacity_timeline"]
                ):
                    raise RuntimeError("compiled size accounting differs from exact execution")
            except ValueError as error:
                attempts.append(
                    {
                        "attempt": attempt,
                        **parent,
                        **chosen,
                        "status": "execution_rejected",
                        "reason": str(error),
                    }
                )
                continue
            endpoint = trace["endpoint"]
            properties = eligibility({"smiles": endpoint})
            if type(properties.get("oracle_eligible")) is not bool:
                raise ValueError("endpoint evaluator must return explicit boolean eligibility")
            outcome = (
                "duplicate"
                if endpoint in seen
                else "eligible"
                if properties["oracle_eligible"]
                else "ineligible"
            )
            record = {
                "attempt": attempt,
                **parent,
                "channel": channel,
                "endpoint": endpoint,
                "status": outcome,
                "metadata": metadata,
                "program_size": size,
                "properties": properties,
                "execution_key": execution_key,
                "actual_changes": trace["actual_changes"],
            }
            attempts.append(record)
            if outcome == "duplicate":
                self.duplicate_counts[entry["entry_id"]] = (
                    self.duplicate_counts.get(entry["entry_id"], 0) + 1
                )
            if outcome != "eligible":
                continue
            seen.add(endpoint)
            candidate = {
                "source_group": self.source_group,
                "oracle_protocol": self.oracle_protocol,
                "source_state": trace["states"][0],
                "program": program.payload(),
                "assignment": list(binding),
                "endpoint": endpoint,
                "trace": trace,
                "provenance": record,
                "inherited_static_score": entry["static_score"],
            }
            if "construction_ancestry" in metadata:
                candidate.update(
                    {
                        k: metadata[k]
                        for k in (
                            "construction_ancestry",
                            "original_seed_state",
                            "ancestral_primitive_edits",
                        )
                    }
                )
            elif "construction_ancestry" in entry:
                candidate.update(
                    {
                        k: entry[k]
                        for k in (
                            "construction_ancestry",
                            "original_seed_state",
                            "ancestral_primitive_edits",
                        )
                    }
                )
            candidate["candidate_id"] = identity(candidate)
            candidates.append(candidate)
        body = {
            "schema_version": "adaptive_program_batch_v1",
            "batch_index": self.batches,
            "configuration": asdict(self.config),
            "candidates": candidates,
            "attempts": attempts,
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
        }
        self.pending = {**body, "batch_id": identity(body)}
        return json.loads(
            json.dumps(
                {
                    **self.pending,
                    "proposal_seconds": perf_counter() - started,
                    "work_cache": self.work_cache.report(),
                    "new_oracle_calls": 0,
                }
            )
        )

    def lock_query_subset(self, pool_id, selected_ids, selection_receipt):
        """Separate a generated pool lock from its pre-oracle query allocation."""
        if (
            self.pending is None
            or self.pending["batch_id"] != pool_id
            or "proposal_pool" in self.pending
        ):
            raise ValueError("query subset needs one unresolved, unselected proposal pool")
        available = {c["candidate_id"]: c for c in self.pending["candidates"]}
        if (
            not selected_ids
            or len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids) <= set(available)
        ):
            raise ValueError(
                "query subset contains duplicate, missing or unknown candidate identities"
            )
        if selection_receipt.get("selected_ids") != list(selected_ids):
            raise ValueError("selection receipt disagrees with query candidate order")
        if identity({k: v for k, v in self.pending.items() if k != "batch_id"}) != pool_id:
            raise ValueError("proposal pool lock was modified")
        body = {k: v for k, v in self.pending.items() if k != "batch_id"}
        body["proposal_pool"] = {"pool_id": pool_id, "candidates": body["candidates"]}
        body["candidates"] = [available[k] for k in selected_ids]
        body["selection"] = selection_receipt
        self.pending = json.loads(json.dumps({**body, "batch_id": identity(body)}))
        return json.loads(json.dumps(self.pending))

    def observe_batch(self, batch_id, outcomes):
        """Consume every locked outcome once; failures are not low-score labels."""
        if self.pending is None or batch_id != self.pending["batch_id"]:
            raise ValueError("score update has no matching pending candidate lock")
        if identity({k: v for k, v in self.pending.items() if k != "batch_id"}) != batch_id:
            raise ValueError("pending candidate lock was mutated")
        candidates = {c["candidate_id"]: c for c in self.pending["candidates"]}
        if len(outcomes) != len(candidates) or {o["candidate_id"] for o in outcomes} != set(
            candidates
        ):
            raise ValueError("score update requires exactly one outcome per locked candidate")
        receipts = set()
        for row in outcomes:
            if row["oracle_protocol"] != self.oracle_protocol or not row["receipt_id"]:
                raise ValueError("score outcome lacks its actual oracle protocol/receipt")
            if row["receipt_id"] in receipts:
                raise ValueError("different locked endpoints cannot share one oracle receipt")
            receipts.add(row["receipt_id"])
            prior = self.observations.get(row["receipt_id"])
            if prior is not None and prior != {
                "endpoint": candidates[row["candidate_id"]]["endpoint"],
                "score": row["score"],
                "oracle_protocol": self.oracle_protocol,
            }:
                raise ValueError("score receipt conflicts with the existing archive")
            if row["score"] is not None and not math.isfinite(row["score"]):
                raise ValueError("nonfinite measured score")
            if row["score"] is None and row.get("failure") not in (
                "oracle_failed",
                "cache_miss_not_evaluated",
                "ambiguous_charged_query_unobserved",
            ):
                raise ValueError("missing score requires an explicit failure or unqueried status")
        for row in outcomes:
            candidate = candidates[row["candidate_id"]]
            if row["score"] is None:
                if row["failure"] == "oracle_failed":
                    self.failed_endpoints.add(candidate["endpoint"])
                continue
            self.add_measured_program(
                candidate,
                receipt_id=row["receipt_id"],
                score=row["score"],
                static_score=candidate["inherited_static_score"],
            )
        self.history.append(json.loads(json.dumps({"batch": self.pending, "outcomes": outcomes})))
        self.pending = None
        self.batches += 1

    def snapshot(self, *, include_history=True):
        """Serialize resumable search state.

        Long benchmark runners persist each completed batch separately and may
        omit the in-memory history to avoid quadratic checkpoint growth. This
        changes storage only: entries, observations, RNG state, duplicate
        accounting and pending work remain complete.
        """
        if type(include_history) is not bool:
            raise ValueError("snapshot history flag must be boolean")
        body = {
            "schema_version": "adaptive_program_optimizer_v1",
            "configuration": asdict(self.config),
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
            "rng": self.rng.bit_generator.state,
            "entries": self.entries,
            "observations": self.observations,
            "duplicate_counts": self.duplicate_counts,
            "failed_endpoints": sorted(self.failed_endpoints),
            "history": self.history if include_history else [],
            "pending": self.pending,
            "batches": self.batches,
            "mutation_choices_seen": [
                [key, sorted(values)] for key, values in self.mutation_choices_seen.items()
            ],
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None, constructor_kwargs=None):
        body = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        if (
            snapshot.get("schema_version") != "adaptive_program_optimizer_v1"
            or identity(body) != snapshot["snapshot_id"]
        ):
            raise ValueError("corrupt or incompatible optimizer snapshot")
        config = dict(snapshot["configuration"])
        config["channel_probabilities"] = tuple(config["channel_probabilities"])
        constructor_kwargs = {} if constructor_kwargs is None else dict(constructor_kwargs)
        result = cls(
            ProgramSearchConfig(**config),
            source_group=snapshot["source_group"],
            oracle_protocol=snapshot["oracle_protocol"],
            hierarchy=hierarchy,
            **constructor_kwargs,
        )
        result.rng.bit_generator.state = snapshot["rng"]
        for key in ("entries", "observations", "duplicate_counts", "history", "pending", "batches"):
            setattr(result, key, json.loads(json.dumps(snapshot[key])))
        result.failed_endpoints = set(snapshot["failed_endpoints"])
        result.mutation_choices_seen = {
            key: set(values) for key, values in snapshot.get("mutation_choices_seen", [])
        }
        return result
