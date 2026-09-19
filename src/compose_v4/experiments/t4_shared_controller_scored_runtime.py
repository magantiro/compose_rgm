"""Pure helpers for the scored nine-cell shared-controller launcher.

The Modal app owns network and volume operations.  This module keeps launch
identity, parent preview, and the two additive ``route_complete_region``
producers deterministic and testable without Modal or an evaluator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from rdkit import Chem

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.generic_legal_action_policy import enumerate_rule_successors
from compose_v4.control.retained_core_pruning import enumerate_retained_core_prunes
from compose_v4.control.route_complete_region_particle_receipts import (
    make_complete_combination_parent_manifest,
)
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    RUNNING,
    restore_checkpoint,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)
from compose_v4.experiments.t4_shared_controller_scored_contract import (
    AUTHORIZATION_SCHEMA_VERSION,
)
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

AUTHORIZATION_SCHEMA = AUTHORIZATION_SCHEMA_VERSION
LAUNCH_TASK_SCHEMA = "t4_shared_controller_scored_launch_v1"
PROPOSAL_MANIFEST_SCHEMA = "t4_shared_controller_proposal_manifest_v1"
RETAINED_CORE_CONFIG = {
    "maximum_fragment_atoms": 16,
    "maximum_stages": 2,
    "maximum_primitives": 32,
    "maximum_prefixes": 4096,
    "program_family": "retained_core_prune",
    "proposal_expert": "route_complete_region",
}

# One exact, non-cardinality Active8 refinement after a retained-core prune.  The
# prune handles the large feasibility-changing move; the refinement creates a new
# queryable endpoint without undoing that cardinality decision.  Ring-system
# restatement is deliberately not part of this *local* refinement family, and
# atom insertion/deletion remain available through the other production experts.
RETAINED_CORE_REFINE_CONFIG = {
    **RETAINED_CORE_CONFIG,
    "program_family": "retained_core_prune_then_refine",
    "refinement_rules": (
        "atom_restate_semantic",
        "bond_reorder",
        "bond_reroute",
        "cycle_open",
        "cycle_close",
    ),
    "refinement_steps": 1,
}

_HEX = frozenset("0123456789abcdef")


def _hash(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_authorization_receipt(
    receipt: Mapping[str, Any],
    *,
    contract_payload_sha256: str,
    total_charged_call_ceiling: int,
) -> dict[str, Any]:
    """Validate the separate payload-bound scored-call authorization.

    The receipt is intentionally not inferred from the scored contract.  A
    nonzero contract is necessary but cannot substitute for the separate user
    authorization artifact.
    """

    if not isinstance(receipt, Mapping):
        raise TypeError("scored authorization receipt must be a mapping")
    normalized = json.loads(
        json.dumps(dict(receipt), sort_keys=True, separators=(",", ":"))
    )
    required = {
        "schema_version",
        "contract_payload_sha256",
        "authorized_scored_calls",
        "user_statement",
    }
    if set(normalized) != required:
        raise ValueError("scored authorization receipt field drift")
    if normalized["schema_version"] != AUTHORIZATION_SCHEMA:
        raise ValueError("scored authorization receipt schema drift")
    expected_identity = _hash(
        contract_payload_sha256, label="contract payload identity"
    )
    if normalized["contract_payload_sha256"] != expected_identity:
        raise ValueError("authorization receipt binds another scored contract")
    ceiling = _positive_integer(
        total_charged_call_ceiling, label="total charged-call ceiling"
    )
    if normalized["authorized_scored_calls"] != ceiling:
        raise RuntimeError("authorization receipt does not authorize the full ceiling")
    if (
        not isinstance(normalized["user_statement"], str)
        or not normalized["user_statement"]
    ):
        raise ValueError("authorization receipt omitted the exact user statement")
    return normalized


def make_launch_task(
    *,
    contract: Mapping[str, Any],
    contract_payload_sha256: str,
    contract_file_sha256: str,
    authorization_receipt: Mapping[str, Any],
    authorization_receipt_sha256: str,
    code_revision: str,
    source_capsule_payload_sha256: str,
) -> dict[str, Any]:
    """Create the immutable campaign launch task after every authority gate."""

    total = _positive_integer(
        contract.get("total_charged_call_ceiling"),
        label="contract total charged-call ceiling",
    )
    requested = contract.get("scored_calls_requested")
    if requested != total or requested <= 0:
        raise RuntimeError("scored contract has no nonzero authorized payload")
    validate_authorization_receipt(
        authorization_receipt,
        contract_payload_sha256=contract_payload_sha256,
        total_charged_call_ceiling=total,
    )
    revision = str(code_revision)
    if len(revision) not in {40, 64} or any(
        character not in _HEX for character in revision
    ):
        raise ValueError("launch code revision must be a full object identity")
    body = {
        "schema_version": LAUNCH_TASK_SCHEMA,
        "contract_payload_sha256": _hash(
            contract_payload_sha256, label="contract payload identity"
        ),
        "contract_file_sha256": _hash(
            contract_file_sha256, label="contract file identity"
        ),
        "authorization_receipt_sha256": _hash(
            authorization_receipt_sha256, label="authorization receipt identity"
        ),
        "source_capsule_payload_sha256": _hash(
            source_capsule_payload_sha256, label="source capsule identity"
        ),
        "code_revision": revision,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "charged_call_ceiling_per_cell": contract.get("charged_calls_per_cell"),
        "total_charged_call_ceiling": total,
        "cell_keys": [row["cell_key"] for row in contract["cells"]],
    }
    expected_cell_count = int(contract.get("campaign_cell_count", 9))
    if (
        body["charged_call_ceiling_per_cell"] != 49
        or len(body["cell_keys"]) != expected_cell_count
        or total != 49 * expected_cell_count
    ):
        raise ValueError("scored launch matrix or per-cell ceiling drift")
    return {**body, "run_id": payload_identity(body)}


def validate_launch_task(
    task: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_payload_sha256: str,
    contract_file_sha256: str,
    authorization_receipt: Mapping[str, Any],
    authorization_receipt_sha256: str,
    source_capsule_payload_sha256: str,
) -> dict[str, Any]:
    """Reconstruct and compare a task at every remote trust boundary."""

    expected = make_launch_task(
        contract=contract,
        contract_payload_sha256=contract_payload_sha256,
        contract_file_sha256=contract_file_sha256,
        authorization_receipt=authorization_receipt,
        authorization_receipt_sha256=authorization_receipt_sha256,
        code_revision=str(task.get("code_revision", "")),
        source_capsule_payload_sha256=source_capsule_payload_sha256,
    )
    if dict(task) != expected:
        raise ValueError("remote launch task identity drift")
    return expected


def checkpoint_controller_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete controller law bound into every cell checkpoint."""

    controller = contract.get("controller")
    if not isinstance(controller, Mapping):
        raise TypeError("scored contract omitted its controller")
    return copy.deepcopy(dict(controller))


def preview_selected_parents(
    checkpoint: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    """Preview the pure engine's deterministic parent set without advancing it."""

    config = checkpoint_controller_config(contract)
    restored = restore_checkpoint(
        checkpoint,
        cell_key=str(cell["cell_key"]),
        source_smiles=str(cell["source_smiles"]),
        delta=float(cell["delta"]),
        budget_ceiling=int(contract["charged_calls_per_cell"]),
        controller_config=config,
    )
    if restored.checkpoint["status"] != RUNNING:
        raise RuntimeError("terminal checkpoint has no proposal parents")
    return restored.state.parents(
        limit=int(config["parents"]),
        rng=restored.rng,
        explore=float(config["parent_explore"]),
    )


def proposal_seed(
    *, controller_seed: int, round_index: int, parent_index: int, expert_index: int
) -> int:
    """Derive the unchanged v2 expert stream for one parent and round."""

    values = (controller_seed, round_index, parent_index, expert_index)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("proposal seed coordinates must be nonnegative integers")
    return int(
        controller_seed
        + 1_000_003 * round_index
        + 10_007 * parent_index
        + 101 * expert_index
    )


def retained_core_route_records(
    *,
    parent: str,
    parent_score: float,
    original_seed: str,
    delta: float,
    support: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate exact retained-core programs inside the route expert lane."""

    score = float(parent_score)
    if not math.isfinite(score):
        raise ValueError("retained-core parent score must be finite")
    source = pad_molecular_graph(smiles_to_molecular_graph(parent), 48)
    proposals = enumerate_retained_core_prunes(
        source,
        maximum_fragment_atoms=RETAINED_CORE_CONFIG["maximum_fragment_atoms"],
        maximum_stages=RETAINED_CORE_CONFIG["maximum_stages"],
        maximum_primitives=RETAINED_CORE_CONFIG["maximum_primitives"],
        maximum_prefixes=RETAINED_CORE_CONFIG["maximum_prefixes"],
    )
    fiber = Fiber(original_seed, float(delta), support=support)
    by_smiles: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        endpoint = fiber.check(molecular_graph_to_smiles(proposal.product))
        if endpoint is None or endpoint["smiles"] == parent:
            continue
        primitives = len(proposal.actions)
        band = "small" if primitives <= 3 else "medium" if primitives <= 11 else "large"
        row = {
            **endpoint,
            "proposal_lane": "route_complete_region",
            "proposal_experts": ["route_complete_region"],
            "families": ["retained_core_prune"],
            "program_families": ["retained_core_prune"],
            "parent": parent,
            "parent_score": score,
            "delta": float(delta),
            "regions": len(proposal.stages),
            "created": 0,
            "deleted": sum(int(stage["deleted_atoms"]) for stage in proposal.stages),
            "realized_primitives": primitives,
            "realized_primitive_band": band,
            "route_generation_modes": ["retained_core_prune"],
        }
        previous = by_smiles.get(endpoint["smiles"])
        if previous is None or primitives < int(previous["realized_primitives"]):
            by_smiles[endpoint["smiles"]] = row
    records = [by_smiles[key] for key in sorted(by_smiles)]
    for rank, row in enumerate(records, 1):
        row["route_proposal_rank"] = rank
    return records, {
        "program_family": "retained_core_prune",
        "proposal_expert": "route_complete_region",
        "exact_programs": len(proposals),
        "eligible_unique": len(records),
        "configuration": copy.deepcopy(RETAINED_CORE_CONFIG),
    }


def retained_core_prune_refine_records(
    *,
    parent: str,
    parent_score: float,
    original_seed: str,
    delta: float,
    support: str,
    excluded_canonical_smiles_sha256: Iterable[str] = (),
    refinement_rules: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compose one exact local refinement after a complete retained-core prune.

    The intermediate prune is generated from the current parent and may be used as
    a free planning state.  It is never queried or admitted to the archive here.
    Only the completed prune-plus-refinement endpoint is returned.  Every accepted
    result is replayed once from the original parent through the ordinary executor,
    so the function cannot mistake two separately valid steps for a valid protected
    composition.

    ``excluded_canonical_smiles_sha256`` is a query-history denylist, not a reward
    input.  Applying it inside this gate makes its support definition identical to
    deployment rather than counting already charged endpoints as new support.
    """

    score = float(parent_score)
    if not math.isfinite(score):
        raise ValueError("retained-core refinement parent score must be finite")
    rules = (
        RETAINED_CORE_REFINE_CONFIG["refinement_rules"]
        if refinement_rules is None
        else tuple(refinement_rules)
    )
    supported_rules = set(RETAINED_CORE_REFINE_CONFIG["refinement_rules"])
    if not rules or len(set(rules)) != len(rules) or not set(rules) <= supported_rules:
        raise ValueError(
            "retained-core refinement rules are empty, repeated, or unsupported"
        )
    excluded = set(excluded_canonical_smiles_sha256)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
        for value in excluded
    ):
        raise ValueError(
            "retained-core refinement exclusions must be SHA-256 identities"
        )

    source = pad_molecular_graph(smiles_to_molecular_graph(parent), 48)
    prune_programs = enumerate_retained_core_prunes(
        source,
        maximum_fragment_atoms=RETAINED_CORE_REFINE_CONFIG["maximum_fragment_atoms"],
        maximum_stages=RETAINED_CORE_REFINE_CONFIG["maximum_stages"],
        maximum_primitives=RETAINED_CORE_REFINE_CONFIG["maximum_primitives"],
        maximum_prefixes=RETAINED_CORE_REFINE_CONFIG["maximum_prefixes"],
    )
    fiber = Fiber(original_seed, float(delta), support=support)
    counters: Counter[str] = Counter()
    successor_counts: Counter[str] = Counter()
    accepted_by_rule: Counter[str] = Counter()
    by_smiles: dict[str, dict[str, Any]] = {}
    for prune in prune_programs:
        counters["prune_programs"] += 1
        intermediate_smiles = molecular_graph_to_smiles(prune.product)
        if fiber.check(intermediate_smiles) is None:
            counters["prune_intermediate_outside_endpoint_fiber"] += 1
            continue
        counters["prune_intermediate_inside_endpoint_fiber"] += 1
        if len(prune.actions) >= RETAINED_CORE_REFINE_CONFIG["maximum_primitives"]:
            counters["prune_programs_without_refinement_room"] += 1
            continue
        for rule in rules:
            successors = enumerate_rule_successors(prune.product, rule)
            successor_counts[rule] += len(successors)
            for successor in successors:
                counters["refinement_successors"] += 1
                endpoint = fiber.check(molecular_graph_to_smiles(successor.successor))
                if endpoint is None or endpoint["smiles"] == parent:
                    counters["refinement_endpoint_outside_fiber"] += 1
                    continue
                endpoint_identity = hashlib.sha256(
                    endpoint["smiles"].encode()
                ).hexdigest()
                if endpoint_identity in excluded:
                    counters["stale_query_exclusions"] += 1
                    continue
                actions = (*prune.actions, successor.action_record)
                if len(actions) > RETAINED_CORE_REFINE_CONFIG["maximum_primitives"]:
                    counters["cumulative_primitive_budget_abstentions"] += 1
                    continue
                replayed, _ = execute_program(source, list(actions))
                if canonical_state_key(replayed) != successor.successor_key:
                    raise RuntimeError(
                        "retained-core prune/refine composition changed on exact replay"
                    )
                counters["exact_composed_replays"] += 1
                primitives = len(actions)
                band = (
                    "small"
                    if primitives <= 3
                    else "medium" if primitives <= 11 else "large"
                )
                row = {
                    **endpoint,
                    "proposal_lane": "route_complete_region",
                    "proposal_experts": ["route_complete_region"],
                    "families": ["retained_core_prune", f"active8_refine:{rule}"],
                    "program_families": ["retained_core_prune_then_refine"],
                    "parent": parent,
                    "parent_score": score,
                    "delta": float(delta),
                    "regions": len(prune.stages) + 1,
                    "created": 0,
                    "deleted": sum(
                        int(stage["deleted_atoms"]) for stage in prune.stages
                    ),
                    "realized_primitives": primitives,
                    "realized_primitive_band": band,
                    "route_generation_modes": ["retained_core_prune_then_refine"],
                    "refinement_rule": rule,
                    "protected_program_actions": list(actions),
                    "protected_program_sha256": payload_identity(list(actions)),
                    "intermediate_endpoint_queried": False,
                }
                previous = by_smiles.get(endpoint["smiles"])
                rank = (primitives, rule, row["protected_program_sha256"])
                previous_rank = (
                    (
                        int(previous["realized_primitives"]),
                        str(previous["refinement_rule"]),
                        str(previous["protected_program_sha256"]),
                    )
                    if previous is not None
                    else None
                )
                if previous_rank is None or rank < previous_rank:
                    by_smiles[endpoint["smiles"]] = row
                accepted_by_rule[rule] += 1
    records = [by_smiles[key] for key in sorted(by_smiles)]
    for rank, row in enumerate(records, 1):
        row["route_proposal_rank"] = rank
    return records, {
        "program_family": "retained_core_prune_then_refine",
        "proposal_expert": "route_complete_region",
        "eligible_novel_unique": len(records),
        "configuration": {
            **copy.deepcopy(RETAINED_CORE_REFINE_CONFIG),
            "refinement_rules": list(rules),
        },
        "counts": dict(sorted(counters.items())),
        "successors_by_rule": dict(sorted(successor_counts.items())),
        "accepted_occurrences_by_rule": dict(sorted(accepted_by_rule.items())),
        "excluded_query_identity_count": len(excluded),
        "prior_scores_or_receipts_read": False,
    }


def attach_endpoint_fingerprints(
    records: list[dict[str, Any]], *, original_seed: str, delta: float, support: str
) -> list[dict[str, Any]]:
    """Attach deterministic Morgan bit sets required by FiberControl."""

    fiber = Fiber(original_seed, float(delta), support=support)
    prepared = []
    for source in records:
        row = copy.deepcopy(source)
        molecule = Chem.MolFromSmiles(str(row.get("smiles") or ""))
        if molecule is None:
            continue
        row["fingerprint"] = sorted(
            fiber.generator.GetFingerprint(molecule).GetOnBits()
        )
        prepared.append(row)
    return prepared


def validate_stale_braf_exclusions(
    contract: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Validate the hash-only BRAF denylist without opening prior outcomes."""

    binding = contract.get("stale_braf_v4_reconciliation")
    if not isinstance(binding, Mapping):
        raise TypeError("scored contract omitted stale BRAF reconciliation")
    raw = binding.get("excluded_canonical_smiles_sha256")
    if not isinstance(raw, Mapping) or set(raw) != {"braf_0", "braf_1"}:
        raise ValueError("stale BRAF exclusion source census drift")
    expected_counts = {"braf_0": 20, "braf_1": 23}
    result: dict[str, tuple[str, ...]] = {}
    for source_cell, expected_count in expected_counts.items():
        values = raw.get(source_cell)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or values != sorted(set(values))
        ):
            raise ValueError(
                f"stale BRAF exclusion count or ordering drift for {source_cell}"
            )
        result[source_cell] = tuple(
            _hash(value, label=f"stale BRAF exclusion for {source_cell}")
            for value in values
        )
    return result


def filter_stale_braf_candidates(
    proposal_pools: Mapping[str, list[dict[str, Any]]],
    *,
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Remove hash-bound stale BRAF queries before selection or query locking."""

    source_cell = str(cell.get("source_cell", ""))
    binding = contract.get("stale_braf_v4_reconciliation")
    if binding is None:
        if cell.get("target") == "braf":
            raise ValueError("BRAF campaign omitted its stale-query reconciliation")
        exclusions: dict[str, tuple[str, ...]] = {}
    else:
        exclusions = validate_stale_braf_exclusions(contract)
    deny = set(exclusions.get(source_cell, ()))
    if cell.get("target") == "braf" and source_cell not in {
        "braf_0",
        "braf_1",
        "braf_2",
    }:
        raise ValueError("unknown BRAF source cell for stale-query filtering")
    filtered: dict[str, list[dict[str, Any]]] = {}
    excluded_by_expert: dict[str, int] = {}
    for expert, source_rows in proposal_pools.items():
        kept: list[dict[str, Any]] = []
        excluded = 0
        for source in source_rows:
            row = copy.deepcopy(source)
            smiles = row.get("smiles")
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is None:
                raise ValueError("proposal pool contains an invalid molecule")
            canonical = Chem.MolToSmiles(molecule)
            if canonical != smiles:
                raise ValueError("proposal pool contains noncanonical SMILES")
            identity = hashlib.sha256(canonical.encode()).hexdigest()
            if identity in deny:
                excluded += 1
                continue
            kept.append(row)
        filtered[str(expert)] = kept
        excluded_by_expert[str(expert)] = excluded
    return filtered, {
        "source_cell": source_cell,
        "denylist_size": len(deny),
        "excluded_by_expert": dict(sorted(excluded_by_expert.items())),
        "excluded_total": sum(excluded_by_expert.values()),
        "prior_scores_or_receipts_read": False,
    }


def particle_parent_manifest(
    *,
    parent: str,
    shared_checkpoint_sha256: str,
    expert: RouteDistilledGoalExpert,
    code_identities: Mapping[str, str],
    config_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Bind one parent's fixed 28-job particle schedule before dispatch."""

    source = pad_molecular_graph(smiles_to_molecular_graph(parent), 48)
    return make_complete_combination_parent_manifest(
        source_state_sha256=payload_identity(encode_state(source)),
        shared_checkpoint_sha256=shared_checkpoint_sha256,
        expert_training_identity_sha256=expert.training_identity,
        code_identities=code_identities,
        config_identities=config_identities,
    )


def make_proposal_manifest(
    *,
    checkpoint: Mapping[str, Any],
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
    round_index: int,
    deadline: float,
    particle_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the exact parent/expert/particle proposal census for one round."""

    parents = preview_selected_parents(checkpoint, cell=cell, contract=contract)
    trajectory_distillation_enabled = bool(
        contract.get("trajectory_distillation", {"enabled": True})["enabled"]
    )
    if trajectory_distillation_enabled:
        if set(parents) != set(particle_manifests) or len(parents) != len(
            particle_manifests
        ):
            raise ValueError("particle manifest census differs from selected parents")
    elif particle_manifests:
        raise ValueError("COMPOSE-NoDistill may not schedule route-template particles")
    if not math.isfinite(float(deadline)):
        raise ValueError("proposal deadline must be finite")
    experts = ("shallow", "anchored_replacement", "route_complete_region")
    requests = []
    for parent_index, parent in enumerate(parents):
        for expert_index, expert_name in enumerate(experts):
            requests.append(
                {
                    "parent_index": parent_index,
                    "parent": parent,
                    "parent_score": checkpoint["archive"][parent],
                    "expert": expert_name,
                    "proposal_seed": proposal_seed(
                        controller_seed=int(cell["controller_seed"]),
                        round_index=round_index,
                        parent_index=parent_index,
                        expert_index=expert_index,
                    ),
                }
            )
    payload = {
        "schema_version": PROPOSAL_MANIFEST_SCHEMA,
        "base_checkpoint_payload_sha256": payload_identity(dict(checkpoint)),
        "cell_key": cell["cell_key"],
        "round": int(round_index),
        "deadline": float(deadline),
        "parents": parents,
        "requests": requests,
        "trajectory_distillation_enabled": trajectory_distillation_enabled,
        "particle_parent_manifest_sha256": (
            {parent: particle_manifests[parent]["payload_sha256"] for parent in parents}
            if trajectory_distillation_enabled
            else {}
        ),
        "retained_core": copy.deepcopy(RETAINED_CORE_CONFIG),
        "proposal_experts": list(experts),
    }
    return payload


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "LAUNCH_TASK_SCHEMA",
    "PROPOSAL_MANIFEST_SCHEMA",
    "RETAINED_CORE_CONFIG",
    "RETAINED_CORE_REFINE_CONFIG",
    "attach_endpoint_fingerprints",
    "checkpoint_controller_config",
    "filter_stale_braf_candidates",
    "make_launch_task",
    "make_proposal_manifest",
    "particle_parent_manifest",
    "preview_selected_parents",
    "proposal_seed",
    "retained_core_prune_refine_records",
    "retained_core_route_records",
    "validate_authorization_receipt",
    "validate_launch_task",
    "validate_stale_braf_exclusions",
]
