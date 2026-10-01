"""Local route-template proposals with exact execution and T4 endpoint checks.

The route expert constructs candidates. It is separate from the frozen neural
reference used by the downstream selector. This module performs no docking.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    propose_route_expert_candidates,
)
from compose_v4.experiments.t4_fiber_campaign import PROPOSAL_SLOTS, Fiber
from compose_v4.rewrite.kernel import canonical_state_key


@dataclass(frozen=True)
class RouteProposalConfig:
    pool_size: int = 64
    realization_limit: int = 32
    beam_width: int = 32
    expansion_width: int = 24
    max_bindings_per_template: int = 4
    maximum_expansions: int = 4000
    scale_balanced: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "scale_balanced":
                if type(value) is not bool:
                    raise TypeError("scale_balanced must be a Boolean")
            elif type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.realization_limit > self.pool_size:
            raise ValueError("realization_limit must not exceed pool_size")


def load_route_expert(path: Path, *, expected_sha256: str) -> RouteDistilledGoalExpert:
    """Load a pinned T4 expert envelope without fitting or selecting a checkpoint."""
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("route expert requires an explicit lowercase SHA-256")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"route expert SHA-256 mismatch: {path}, expected {expected_sha256}")
    envelope = json.loads(content)
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid route expert envelope: {path}")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise TypeError(f"route expert payload must be an object: {path}")
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"route expert payload hash mismatch: {path}")
    if payload.get("schema_version") != "t4_route_complete_region_expert_checkpoint_v1":
        raise ValueError(f"unsupported route expert schema: {path}")
    if not isinstance(payload.get("expert"), dict):
        raise TypeError(f"route expert payload.expert must be an object: {path}")
    return RouteDistilledGoalExpert.from_checkpoint(payload["expert"])


def expand_route(
    parent: str,
    parent_score: float,
    fiber: Fiber,
    expert: RouteDistilledGoalExpert,
    *,
    config: RouteProposalConfig | None = None,
    include_realized_actions: bool = False,
) -> tuple[list[dict], dict]:
    """Run the route proposal lane and gate endpoints against the original lead.

    The parent score is carried to downstream controller features, never used by
    the route constructor. Receipt capture adds no proposals or random choices.
    """
    if not math.isfinite(parent_score):
        raise ValueError("route parent_score must be finite")
    if type(include_realized_actions) is not bool:
        raise TypeError("include_realized_actions must be a Boolean")
    config = RouteProposalConfig() if config is None else config
    source = pad_molecular_graph(smiles_to_molecular_graph(parent), PROPOSAL_SLOTS)
    parent_key = canonical_state_key(source)
    proposed, telemetry = propose_route_expert_candidates(
        source, expert, **asdict(config), include_realized_actions=include_realized_actions
    )
    records = []
    self_endpoints = ineligible = 0
    for candidate in proposed:
        properties = fiber.check(candidate["smiles"])
        if properties is None:
            ineligible += 1
            continue
        if properties["smiles"] == parent_key:
            self_endpoints += 1
            continue
        records.append(
            {
                **candidate,
                **properties,
                "parent": parent,
                "parent_score": parent_score,
                "delta": fiber.delta,
            }
        )
    return records, {
        **telemetry,
        "eligible_endpoints": len(records),
        "ineligible_endpoints": ineligible,
        "self_endpoints": self_endpoints,
    }
