"""Measured-score adaptation over variable coordinated constructions.

This is a bounded optimizer, not importance sampling or a learned future value.
Its archive retains exact construction provenance; observations arrive only for
locked endpoints under an identified oracle protocol. The legacy broad channel
is dispatched before WHERE and never silently replaced by program retrieval.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram, attachment_bindings, extract_program
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.molecular_task_search import (
    MolecularSearchState,
    dispatch_complete_proposal,
)
from compose_v4.control.program_mutation import (
    PARAMETER_MOVES,
    branch_components,
    mutate_attachment,
    mutate_parameter,
    parameter_choices,
    select_branch,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


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

    def __post_init__(self):
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
        ):
            raise ValueError("invalid program search wall limit or exploration")
        weights = self.channel_probabilities
        if (
            len(weights) != 3
            or any(not math.isfinite(v) or v <= 0 for v in weights)
            or not math.isclose(sum(weights), 1)
        ):
            raise ValueError("three positive channel probabilities must sum to one")


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
        self._program_cache, self._source_cache = {}, {}

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
        _, replay = execute_program_graph(
            source,
            compile_program_graph(program),
            tuple(record["assignment"]),
            max_primitives=self.config.max_primitives,
            max_blocks=self.config.max_blocks,
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
        ranks = np.asarray([1 + np.sum(scores < v) for v in scores], dtype=float)
        quality = 1 / ranks
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
            # Exhausted endpoints lose allocation too, not only one representation.
            mass /= 1 + sum(self.duplicate_counts.get(k, 0) for k in representatives) / len(
                representatives
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
        }

    def _mutate(self, entry):
        source, program, binding = (
            self._source(entry),
            self._program(entry),
            tuple(entry["assignment"]),
        )
        edits = []
        number = 1 + int(self.rng.random() < self.config.double_mutation_probability)
        for _ in range(number):
            move = ("attachment", *PARAMETER_MOVES)[int(self.rng.integers(4))]
            if move == "attachment":
                binding, census = mutate_attachment(
                    source, program, binding, self.rng, max_bindings=self.config.max_bindings
                )
                edits.append({"kind": move, **census})
            else:
                choices = parameter_choices(program, move)
                if not choices:
                    raise ValueError(f"no conditional choices for {move}")
                choice = choices[int(self.rng.integers(len(choices)))]
                program = mutate_parameter(program, move, choice)
                edits.append({"kind": move, "choice": choice, "choices": len(choices)})
        return source, program, binding, {"mutations": edits}

    def _recombine(self, entry):
        source, program, binding = (
            self._source(entry),
            self._program(entry),
            tuple(entry["assignment"]),
        )
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
            donor_program = self._program(donor)
            branches = branch_components(donor_program)
            branch = branches[int(self.rng.integers(len(branches)))]
            donor_program, _ = select_branch(donor_program, branch)
            census = attachment_bindings(
                donor_program, source, max_bindings=self.config.max_bindings
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
        return source, combined, anchors, {"removed_branches": removed, "donors": donors}

    def _broad(self, entry):
        if self.hierarchy is None:
            raise ValueError("broad_runtime_unavailable_in_local_development")
        program = self._program(entry)
        remaining = self.config.max_primitives - len(program.marks)
        blocks = self.config.max_blocks - len(program.blocks)
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
                channel, (source, program, binding, metadata) = dispatch_complete_proposal(
                    self.rng,
                    program_sampler=program_sampler,
                    broad_sampler=broad_sampler,
                    probabilities=self.config.channel_probabilities,
                )
                _, trace = execute_program_graph(
                    source,
                    compile_program_graph(program),
                    binding,
                    max_primitives=self.config.max_primitives,
                    max_blocks=self.config.max_blocks,
                )
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
                "properties": properties,
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
                    "new_oracle_calls": 0,
                }
            )
        )

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

    def snapshot(self):
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
            "history": self.history,
            "pending": self.pending,
            "batches": self.batches,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None):
        body = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        if (
            snapshot.get("schema_version") != "adaptive_program_optimizer_v1"
            or identity(body) != snapshot["snapshot_id"]
        ):
            raise ValueError("corrupt or incompatible optimizer snapshot")
        config = dict(snapshot["configuration"])
        config["channel_probabilities"] = tuple(config["channel_probabilities"])
        result = cls(
            ProgramSearchConfig(**config),
            source_group=snapshot["source_group"],
            oracle_protocol=snapshot["oracle_protocol"],
            hierarchy=hierarchy,
        )
        result.rng.bit_generator.state = snapshot["rng"]
        for key in ("entries", "observations", "duplicate_counts", "history", "pending", "batches"):
            setattr(result, key, json.loads(json.dumps(snapshot[key])))
        result.failed_endpoints = set(snapshot["failed_endpoints"])
        return result
