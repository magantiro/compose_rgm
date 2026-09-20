"""Dynamic COMPOSE driven by a frozen target-free route-distilled actor.

The actor proposes only generic module families and relative structural roles.
Every proposal is rebound to the current exact graph and executed by the same
program executor as the route-free controller.  A fixed shallow lane preserves
the original generic synthesizer, and the task-specific complete-route archive
is empty when a search starts.
"""

from __future__ import annotations

import json
from dataclasses import replace
from time import perf_counter

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    FRESH_SYNTHESIS_PROBABILITY,
    DynamicProgramOptimizer,
    initial_dynamic_program_batch,
    synthesize_dynamic_program,
)
from compose_v4.control.route_distilled_program_policy import (
    RouteDistilledProgramPolicy,
    synthesize_route_distilled_program,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "route_distilled_dynamic_optimizer_v1"
SNAPSHOT_SCHEMA = "route_distilled_dynamic_optimizer_snapshot_v1"
SHALLOW_PROPOSAL_FLOOR = 0.25
DISTILLED_SEED_LANE = 83
LANE_SELECTION_SEED_LANE = 89


def _candidate(source, program, binding, trace, attempt, source_group, oracle_protocol):
    body = {
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "source_state": encode_state(source),
        "program": program.payload(),
        "assignment": list(binding),
        "trace": trace,
        "endpoint": trace["endpoint"],
        "provenance": attempt,
        "score": None,
    }
    return {**body, "candidate_id": identity(body)}


def _distilled_initial_batch(
    source,
    config,
    policy,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    candidate_limit,
    wall_seconds,
):
    began = perf_counter()
    rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, DISTILLED_SEED_LANE])
    )
    attempts, candidates = [], []
    seen = {canonical_state_key(source)}
    panel_cache = {}
    for index in range(config.attempts_per_batch):
        if len(candidates) >= candidate_limit or perf_counter() - began >= wall_seconds:
            break
        try:
            _, program, binding, trace, metadata = synthesize_route_distilled_program(
                source,
                rng,
                policy,
                max_modules=3,
                max_primitives=config.max_primitives,
                max_blocks=config.max_blocks,
                panel_cache=panel_cache,
            )
        except ValueError as error:
            attempts.append(
                {
                    "attempt": index,
                    "channel": "route_distilled_program",
                    "status": "execution_rejected",
                    "reason": str(error),
                }
            )
            continue
        endpoint = trace["endpoint"]
        properties = eligibility({"smiles": endpoint})
        status = (
            "duplicate"
            if endpoint in seen
            else (
                "eligible"
                if properties.get("oracle_eligible") is True
                else "ineligible"
            )
        )
        attempt = {
            "attempt": index,
            "channel": "route_distilled_program",
            "endpoint": endpoint,
            "status": status,
            "properties": properties,
            "metadata": metadata,
            "actual_changes": trace["actual_changes"],
        }
        attempts.append(attempt)
        if status != "eligible":
            continue
        seen.add(endpoint)
        candidates.append(
            _candidate(
                source,
                program,
                binding,
                trace,
                attempt,
                source_group,
                oracle_protocol,
            )
        )
    return {
        "candidates": candidates,
        "attempts": attempts,
        "rng_state": rng.bit_generator.state,
        "proposal_seconds": perf_counter() - began,
    }


def initial_route_distilled_program_batch(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    policy,
    broad_sampler=None,
):
    """Generate a post-filter union of shallow and route-distilled proposals."""

    if entries:
        raise ValueError(
            "route-distilled initialization requires an empty route archive"
        )
    if broad_sampler is not None:
        raise ValueError(
            "route-distilled initialization does not use reference inference"
        )
    if not isinstance(policy, RouteDistilledProgramPolicy):
        raise TypeError("route-distilled initialization requires a frozen policy")
    began = perf_counter()
    shallow_limit = max(1, round(config.candidates_per_batch * SHALLOW_PROPOSAL_FLOOR))
    distilled_limit = config.candidates_per_batch - shallow_limit
    shallow = initial_dynamic_program_batch(
        source,
        (),
        replace(
            config,
            candidates_per_batch=shallow_limit,
            wall_seconds=max(1.0, config.wall_seconds * SHALLOW_PROPOSAL_FLOOR),
        ),
        source_group=source_group,
        oracle_protocol=oracle_protocol,
        eligibility=eligibility,
    )
    distilled = _distilled_initial_batch(
        source,
        config,
        policy,
        source_group=source_group,
        oracle_protocol=oracle_protocol,
        eligibility=eligibility,
        candidate_limit=distilled_limit,
        wall_seconds=max(1.0, config.wall_seconds * (1 - SHALLOW_PROPOSAL_FLOOR)),
    )
    seen, candidates = set(), []
    for row in (*shallow["candidates"], *distilled["candidates"]):
        if row["endpoint"] in seen:
            continue
        seen.add(row["endpoint"])
        candidates.append(row)
    candidates = candidates[: config.candidates_per_batch]
    body = {
        "schema_version": "initial_route_distilled_program_batch_v1",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "policy_training_identity": policy.training_identity,
        "shallow_proposal_floor": SHALLOW_PROPOSAL_FLOOR,
        "candidates": candidates,
        "attempts": [*shallow["attempts"], *distilled["attempts"]],
        "rng_state_after_preparation": {
            "shallow": shallow["rng_state_after_preparation"],
            "distilled": distilled["rng_state"],
        },
    }
    return {
        **body,
        "batch_id": identity(body),
        "proposal_seconds": perf_counter() - began,
        "proposal_seconds_by_lane": {
            "shallow": shallow["proposal_seconds"],
            "distilled": distilled["proposal_seconds"],
        },
        "new_oracle_calls": 0,
    }


class RouteDistilledDynamicOptimizer(DynamicProgramOptimizer):
    """Reuse measured routes while mixing two target-free fresh proposers."""

    _configured_policy: RouteDistilledProgramPolicy | None = None
    _configured_checkpoint_identity: str | None = None

    @classmethod
    def configure_policy(cls, checkpoint: dict) -> None:
        cls._configured_policy = RouteDistilledProgramPolicy.from_checkpoint(checkpoint)
        cls._configured_checkpoint_identity = identity(checkpoint)

    def __init__(self, *args, **kwargs):
        if (
            self._configured_policy is None
            or self._configured_checkpoint_identity is None
        ):
            raise ValueError("route-distilled optimizer policy is not configured")
        super().__init__(*args, **kwargs)
        self._route_distilled_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, DISTILLED_SEED_LANE])
        )
        self._route_distilled_lane_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, LANE_SELECTION_SEED_LANE])
        )
        self._route_distilled_panel_cache = {}

    def _mutate(self, entry):
        if self.rng.random() >= FRESH_SYNTHESIS_PROBABILITY:
            return ProgramOptimizer._mutate(self, entry)
        source = decode_state(entry["trace"]["states"][-1])
        if self._route_distilled_lane_rng.random() < SHALLOW_PROPOSAL_FLOOR:
            _, program, binding, _, metadata = synthesize_dynamic_program(
                source,
                self.rng,
                max_modules=3,
                max_primitives=self.config.max_primitives,
                max_blocks=self.config.max_blocks,
            )
            channel = "shallow_generic_program"
            detail = {"dynamic_generic_composition": metadata}
        else:
            _, program, binding, _, metadata = synthesize_route_distilled_program(
                source,
                self._route_distilled_rng,
                self._configured_policy,
                max_modules=3,
                max_primitives=self.config.max_primitives,
                max_blocks=self.config.max_blocks,
                panel_cache=self._route_distilled_panel_cache,
            )
            channel = "route_distilled_program"
            detail = {"route_distilled_composition": metadata}
        return (
            source,
            program,
            binding,
            {
                "route_distilled_channel": channel,
                **detail,
                **self._continuation_lineage(entry),
            },
        )

    def snapshot(self, *, include_history=True):
        base = super().snapshot(include_history=include_history)
        body = {
            "schema_version": SNAPSHOT_SCHEMA,
            "base": base,
            "policy_checkpoint_identity": self._configured_checkpoint_identity,
            "distilled_rng": self._route_distilled_rng.bit_generator.state,
            "lane_rng": self._route_distilled_lane_rng.bit_generator.state,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None):
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        if (
            snapshot.get("schema_version") != SNAPSHOT_SCHEMA
            or identity(body) != snapshot.get("snapshot_id")
            or snapshot.get("policy_checkpoint_identity")
            != cls._configured_checkpoint_identity
        ):
            raise ValueError(
                "corrupt or policy-incompatible distilled optimizer snapshot"
            )
        result = super().restore(snapshot["base"], hierarchy=hierarchy)
        result._route_distilled_rng.bit_generator.state = snapshot["distilled_rng"]
        result._route_distilled_lane_rng.bit_generator.state = snapshot["lane_rng"]
        return result
