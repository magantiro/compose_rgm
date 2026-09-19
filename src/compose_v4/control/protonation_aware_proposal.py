"""Task-blind protected programs beginning with an admitted protonation rewrite.

The expert enumerates the complete narrow Editing-V3 protonation fiber on the
current parent and then invokes existing generic structural proposal mechanisms
on the exact rewritten state.  Every complete candidate is replayed from the
original parent under Editing-V3.  Benchmark properties and task reward are not
inputs to proposal generation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
)
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.protonation_restate_program import execute_protonation_program
from compose_v4.control.retained_core_pruning import enumerate_retained_core_prunes
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    propose_route_expert_candidates,
)
from compose_v4.rewrite import action_codec_v4 as v4
from compose_v4.rewrite.action_codec_v5 import encode_action
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v3_protonation_rewrite_system,
)
from compose_v4.rewrite.operators import enumerate_atom_protonation_restates
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA_VERSION = "protonation_aware_retained_subgraph_proposal_v1"
LANES = ("charge_only", "shallow_local", "retained_core_prune", "route_complete_region")


@dataclass(frozen=True)
class ProtonationAwareProposalConfig:
    """Frozen zero-oracle allocation for one task-independent proposal call."""

    seed: int = 20260919
    shallow_draws: int = 256
    maximum_primitives: int = 32
    maximum_active_atoms: int = 40
    persistent_slots: int = 48
    retained_maximum_fragment_atoms: int = 16
    retained_maximum_stages: int = 2
    retained_maximum_prefixes: int = 4096
    route_pool_size: int = 192
    route_realization_limit: int = 96
    route_beam_width: int = 48
    route_expansion_width: int = 48
    route_max_bindings_per_template: int = 4
    route_maximum_expansions: int = 4000
    route_candidate_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        integer_fields = (
            "seed",
            "shallow_draws",
            "maximum_primitives",
            "maximum_active_atoms",
            "persistent_slots",
            "retained_maximum_fragment_atoms",
            "retained_maximum_stages",
            "retained_maximum_prefixes",
            "route_pool_size",
            "route_realization_limit",
            "route_beam_width",
            "route_expansion_width",
            "route_max_bindings_per_template",
            "route_maximum_expansions",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 1
            for name in integer_fields
        ):
            raise ValueError("protonation-aware proposal allocations must be positive integers")
        if (
            self.maximum_primitives != 32
            or self.maximum_active_atoms != 40
            or self.persistent_slots != 48
        ):
            raise ValueError("protonation-aware proposal preserves 32/40/48 support")
        if self.route_realization_limit > self.route_pool_size:
            raise ValueError("route realization limit exceeds route pool size")
        timeout = self.route_candidate_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("route candidate timeout must be positive")


def _lift_v4(record: dict) -> dict:
    rule, action = v4.decode_action(record)
    return encode_action(rule, action)


def _action_seed(seed: int, source_key: str, action_record: dict) -> int:
    payload = json.dumps(
        {"seed": seed, "source": source_key, "action": action_record},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _charge_transition(
    source: MolecularGraph, rewritten: MolecularGraph, endpoint: MolecularGraph
) -> dict:
    return {
        "source_total_formal_charge": int(source.formal_charges.sum()),
        "after_first_primitive_total_formal_charge": int(rewritten.formal_charges.sum()),
        "endpoint_total_formal_charge": int(endpoint.formal_charges.sum()),
    }


def propose_protonation_aware_candidates(
    source: MolecularGraph,
    route_expert: RouteDistilledGoalExpert,
    *,
    config: ProtonationAwareProposalConfig | None = None,
) -> tuple[list[dict], dict]:
    """Generate exact charge-only and charge-plus-structural complete programs.

    Similarity, QED, SA, target identity and reward never enter this function.
    The returned action lists are exact Editing-V3 replay receipts and begin
    with exactly one enumerated ``atom_protonation_restate`` primitive.
    """

    config = ProtonationAwareProposalConfig() if config is None else config
    if source.n_atoms != config.persistent_slots or not (
        1 <= source.n_real_atoms <= config.maximum_active_atoms
    ):
        raise ValueError("expected an exact supported 48-slot source")

    source_key = canonical_state_key(source)
    runtime = editing_v3_protonation_rewrite_system()
    protonation_actions = enumerate_atom_protonation_restates(source)
    abstentions: Counter[str] = Counter()
    lane_generation_attempts: Counter[str] = Counter()
    lane_attempts: Counter[str] = Counter()
    lane_exact: Counter[str] = Counter()
    exact_attempts = 0
    exact_successes = 0
    candidates: dict[str, dict] = {}
    route_telemetry: list[dict[str, Any]] = []

    def commit(
        *,
        protonation_record: dict,
        rewritten: MolecularGraph,
        tail_actions: tuple[dict, ...],
        lane: str,
        provenance: dict,
        expected_key: str,
    ) -> None:
        nonlocal exact_attempts, exact_successes
        lane_attempts[lane] += 1
        combined = (protonation_record, *tail_actions)
        if len(combined) > config.maximum_primitives:
            abstentions[f"{lane}:combined_primitive_limit"] += 1
            return
        exact_attempts += 1
        try:
            endpoint, receipt = execute_protonation_program(source, combined)
        except (ValueError, KeyError) as error:
            abstentions[f"{lane}:exact_replay:{type(error).__name__}"] += 1
            return
        if canonical_state_key(endpoint) != expected_key:
            raise RuntimeError("protonation-aware exact replay changed its endpoint")
        exact_successes += 1
        lane_exact[lane] += 1
        endpoint_key = canonical_state_key(endpoint)
        first_rule = combined[0].get("executor_rule")
        if first_rule != "atom_protonation_restate":
            raise RuntimeError("protonation-aware program does not begin with protonation")
        record = {
            "schema_version": SCHEMA_VERSION,
            "smiles": molecular_graph_to_smiles(endpoint),
            "endpoint_key": endpoint_key,
            "proposal_lane": "protonation_aware_retained_subgraph",
            "program_kind": "charge_only" if not tail_actions else "charge_plus_structural",
            "structural_lane": lane,
            "program_families": ["atom_protonation_restate", lane],
            "actions": list(combined),
            "primitive_edits": int(receipt["primitive_edits"]),
            "tail_primitive_edits": len(tail_actions),
            "charge_transition": _charge_transition(source, rewritten, endpoint),
            "provenance": provenance,
            "exact_execution": True,
            "partial_endpoint_evaluations": 0,
        }
        incumbent = candidates.get(endpoint_key)
        candidate_order = (
            record["primitive_edits"],
            LANES.index(lane),
            json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        )
        incumbent_order = None
        if incumbent is not None:
            incumbent_order = (
                incumbent["primitive_edits"],
                LANES.index(incumbent["structural_lane"]),
                json.dumps(incumbent["provenance"], sort_keys=True, separators=(",", ":")),
            )
        if incumbent is None or candidate_order < incumbent_order:
            candidates[endpoint_key] = record

    for action_index, action in enumerate(protonation_actions):
        protonation_record = encode_action("atom_protonation_restate", action)
        rewritten = runtime.apply(source, "atom_protonation_restate", action)
        action_provenance = {
            "enumeration_index": action_index,
            "vertex": int(action.v),
            "target_state": str(action.target_state),
        }
        commit(
            protonation_record=protonation_record,
            rewritten=rewritten,
            tail_actions=(),
            lane="charge_only",
            provenance=action_provenance,
            expected_key=canonical_state_key(rewritten),
        )
        lane_generation_attempts["charge_only"] += 1

        rng = np.random.default_rng(_action_seed(config.seed, source_key, protonation_record))
        for draw in range(config.shallow_draws):
            lane_generation_attempts["shallow_local"] += 1
            try:
                _, _, _, trace, metadata = synthesize_dynamic_program(
                    rewritten,
                    rng,
                    max_modules=1,
                    max_primitives=config.maximum_primitives - 1,
                    max_blocks=8,
                )
                tail = tuple(_lift_v4(record) for record in trace["actions"])
                expected = decode_state(trace["states"][-1])
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError) as error:
                abstentions[f"shallow_local:generation:{type(error).__name__}"] += 1
                continue
            commit(
                protonation_record=protonation_record,
                rewritten=rewritten,
                tail_actions=tail,
                lane="shallow_local",
                provenance={
                    **action_provenance,
                    "draw": draw,
                    "modules": metadata.get("modules", []),
                },
                expected_key=canonical_state_key(expected),
            )

        retained = enumerate_retained_core_prunes(
            rewritten,
            maximum_fragment_atoms=config.retained_maximum_fragment_atoms,
            maximum_stages=config.retained_maximum_stages,
            maximum_primitives=config.maximum_primitives - 1,
            maximum_prefixes=config.retained_maximum_prefixes,
        )
        for rank, proposal in enumerate(retained, 1):
            lane_generation_attempts["retained_core_prune"] += 1
            commit(
                protonation_record=protonation_record,
                rewritten=rewritten,
                tail_actions=tuple(_lift_v4(record) for record in proposal.actions),
                lane="retained_core_prune",
                provenance={
                    **action_provenance,
                    "proposal_rank": rank,
                    "stages": list(proposal.stages),
                },
                expected_key=canonical_state_key(proposal.product),
            )

        route_rows, telemetry = propose_route_expert_candidates(
            rewritten,
            route_expert,
            pool_size=config.route_pool_size,
            realization_limit=config.route_realization_limit,
            beam_width=config.route_beam_width,
            expansion_width=config.route_expansion_width,
            max_bindings_per_template=config.route_max_bindings_per_template,
            maximum_expansions=config.route_maximum_expansions,
            scale_balanced=True,
            per_candidate_timeout_seconds=config.route_candidate_timeout_seconds,
            include_realized_actions=True,
        )
        route_telemetry.append({**action_provenance, "telemetry": telemetry})
        for row in route_rows:
            lane_generation_attempts["route_complete_region"] += 1
            tail = tuple(_lift_v4(record) for record in row["realized_actions"])
            commit(
                protonation_record=protonation_record,
                rewritten=rewritten,
                tail_actions=tail,
                lane="route_complete_region",
                provenance={
                    **action_provenance,
                    "route_proposal_rank": int(row["route_proposal_rank"]),
                    "route_program_id": str(row["route_program_id"]),
                    "rewrite_scale": str(row["rewrite_scale"]),
                    "realized_primitive_band": str(row["realized_primitive_band"]),
                },
                expected_key=str(row["realized_endpoint_key"]),
            )

    ordered = [candidates[key] for key in sorted(candidates)]
    telemetry = {
        "schema_version": SCHEMA_VERSION,
        "protonation_actions_enumerated": len(protonation_actions),
        "protonation_abstention": not protonation_actions,
        "structural_lanes": list(LANES),
        "lane_generation_attempts": dict(sorted(lane_generation_attempts.items())),
        "lane_attempts": dict(sorted(lane_attempts.items())),
        "lane_exact_programs": dict(sorted(lane_exact.items())),
        "exact_execution_precision_numerator": exact_successes,
        "exact_execution_precision_denominator": exact_attempts,
        "unique_exact_endpoints": len(ordered),
        "abstentions": dict(sorted(abstentions.items())),
        "route_telemetry": route_telemetry,
        "task_or_target_input_used": False,
        "endpoint_or_teacher_input_used": False,
    }
    return ordered, telemetry


__all__ = [
    "LANES",
    "SCHEMA_VERSION",
    "ProtonationAwareProposalConfig",
    "propose_protonation_aware_candidates",
]
