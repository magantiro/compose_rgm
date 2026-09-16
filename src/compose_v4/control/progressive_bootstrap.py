"""Persistent, resumable zero-route bootstrap with protected compensation."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    synthesize_named_module_sequence,
)
from compose_v4.control.progressive_structured_sampler import synthesize_lane
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state


class ProgressiveBootstrap:
    """Own RNG state across empty rounds; no scored or teacher initialization."""

    def __init__(self, source, config: ProgramSearchConfig, *, source_group, oracle_protocol):
        self.source, self.config = source, config
        self.source_group, self.oracle_protocol = source_group, oracle_protocol
        self.rngs = {
            "shallow": np.random.default_rng(np.random.SeedSequence([config.seed, 71])),
            "structured": np.random.default_rng(np.random.SeedSequence([config.seed, 260916, 1])),
        }
        self.cursor = 0
        self.round = 0
        self.seen = {canonical_state_key(source)}

    def next_batch(self, eligibility):
        began = perf_counter()
        attempts, candidates = [], []
        for _ in range(self.config.attempts_per_batch):
            if (
                len(candidates) >= self.config.candidates_per_batch
                or perf_counter() - began >= self.config.wall_seconds
            ):
                break
            index = self.cursor
            self.cursor += 1
            capacity_prefix = (
                self.source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
                and index < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS + CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS
            )
            lane = "shallow" if capacity_prefix or index % 2 == 0 else "structured"
            try:
                if capacity_prefix:
                    families = (
                        ("substituent_delete",)
                        if index < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
                        else ("substituent_delete", "substituent_delete")
                    )
                    _, program, binding, trace, metadata = synthesize_named_module_sequence(
                        self.source,
                        self.rngs[lane],
                        families,
                        max_primitives=self.config.max_primitives,
                        max_blocks=self.config.max_blocks,
                    )
                else:
                    _, program, binding, trace, metadata = synthesize_lane(
                        self.source, self.rngs[lane], lane, self.config
                    )
            except ValueError as error:
                attempts.append(
                    {
                        "attempt": index,
                        "planner_channel": lane,
                        "status": "execution_rejected",
                        "reason": str(error),
                    }
                )
                continue
            endpoint = trace["endpoint"]
            properties = eligibility({"smiles": endpoint})
            if type(properties.get("oracle_eligible")) is not bool:
                raise ValueError("endpoint evaluator must return boolean eligibility")
            status = (
                "duplicate"
                if endpoint in self.seen
                else ("eligible" if properties.get("oracle_eligible") is True else "ineligible")
            )
            self.seen.add(endpoint)
            attempt = {
                "attempt": index,
                "planner_channel": lane,
                "endpoint": endpoint,
                "status": status,
                "properties": properties,
                "metadata": metadata,
                "actual_changes": trace["actual_changes"],
            }
            attempts.append(attempt)
            if status == "eligible":
                candidate = {
                    "source_group": self.source_group,
                    "oracle_protocol": self.oracle_protocol,
                    "source_state": encode_state(self.source),
                    "program": program.payload(),
                    "assignment": list(binding),
                    "trace": trace,
                    "endpoint": endpoint,
                    "provenance": {**attempt, "channel": "progressive_bootstrap"},
                    "score": None,
                }
                candidates.append({**candidate, "candidate_id": identity(candidate)})
        body = {
            "schema_version": "progressive_bootstrap_batch_v1",
            "round": self.round,
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
            "initial_route_archive": [],
            "source_library_rows_loaded": 0,
            "candidates": candidates,
            "attempts": attempts,
            "cursor_after": self.cursor,
        }
        self.round += 1
        return {
            **body,
            "batch_id": identity(body),
            "proposal_seconds": perf_counter() - began,
            "new_oracle_calls": 0,
        }

    def snapshot(self):
        body = {
            "schema_version": "progressive_bootstrap_state_v1",
            "configuration": asdict(self.config),
            "source": encode_state(self.source),
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
            "cursor": self.cursor,
            "round": self.round,
            "seen": sorted(self.seen),
            "rngs": {k: v.bit_generator.state for k, v in self.rngs.items()},
        }
        return {**body, "snapshot_id": identity(body)}

    @classmethod
    def restore(cls, snapshot):
        body = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        if snapshot.get("schema_version") != "progressive_bootstrap_state_v1" or identity(
            body
        ) != snapshot.get("snapshot_id"):
            raise ValueError("corrupt or incompatible progressive bootstrap snapshot")
        config = dict(snapshot["configuration"])
        config["channel_probabilities"] = tuple(config["channel_probabilities"])
        result = cls(
            decode_state(snapshot["source"]),
            ProgramSearchConfig(**config),
            source_group=snapshot["source_group"],
            oracle_protocol=snapshot["oracle_protocol"],
        )
        result.cursor, result.round, result.seen = (
            snapshot["cursor"],
            snapshot["round"],
            set(snapshot["seen"]),
        )
        for lane, state in snapshot["rngs"].items():
            result.rngs[lane].bit_generator.state = state
        return result
