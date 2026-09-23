"""The ONE per-expert proposal unit the unified T4 controller runs.

WHY THIS IS A MODULE AND NOT A CLOSURE IN THE APP
--------------------------------------------------
Until now the same five proposal lanes were written twice: once inside the Modal
campaign app's ``proposal_worker``, and once inside a zero-oracle gate harness
that reproduced the app's call shape by hand.  Two copies of a proposal law are
two proposal laws.  The gate then measures a controller that resembles the one
the campaign runs, and the resemblance is maintained by reading rather than by
construction -- which is exactly the class of defect this repository keeps
paying for, in the form of a transcribed reference that cannot fail usefully.

Everything that decides WHICH MOLECULES a lane proposes lives here.  The Modal
worker supplies a remote fan-out and a baked checkpoint path; the gate supplies a
serial loop and a repository path.  Neither supplies chemistry.

THE FIVE LANES
--------------
``shallow``                              the primary draw lane, and the only one
                                         that can consume a region law.
``anchored_replacement``                 the second primary draw lane.
``route_complete_region``                a beam over a fitted route expert; it
                                         has no draw parameter, which is why the
                                         escalation ladder cannot escalate it.
``zero_support_fallback``                rung 0 for a routed ``region`` kernel.
``protonation_aware_retained_subgraph``  rung 0 for a routed ``state_aware``
                                         kernel.

The first three are the frozen round-one vocabulary
(``t4_integrated_route_fiber.EXPERTS``).  The last two are reachable ONLY from a
support-expansion event: this module refuses to run them without a contract that
authorizes the stage, so a rung-0 lane cannot be invoked from an ordinary round
by naming it in a request.

WHY A RUNG-0 RECORD CARRIES NO ``proposal_lane``
-------------------------------------------------
``t4_integrated_route_fiber._experts`` validates a record's lane against the
frozen expert vocabulary and RAISES on an unknown one, so a record carrying
``"zero_support_fallback"`` would kill ``attach_features`` at precisely the
moment a rung-0 stage first succeeded.  The true stage is moved to its own key,
``support_expansion_stage``, and ``proposal_experts`` is set to the empty list --
which ``normalize_expansion_records`` then deliberately leaves alone.

WHAT IT SPENDS
--------------
CPU.  No lane here reads a docking score, a target identity or an archive value:
``parent_score`` is carried through onto the record and is never consulted.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.route_distilled_goal_expert import (
    propose_route_expert_candidates,
)
from compose_v4.control.t4_unified_routing import PROPOSAL_SLOTS
from compose_v4.control.zero_support_fallback import fallback_candidates
from compose_v4.experiments.t4_fiber_campaign import expand
from compose_v4.experiments.t4_support_expansion import (
    ESCALATABLE_LANES,
    SupportExpansionContractError,
)
from compose_v4.experiments.t4_unified_controller import (
    ROUTED_RUNG_ZERO_LANE,
    resolve_state_routing,
)

SCHEMA_VERSION = "t4_unified_proposal_v1"

#: The frozen round-one vocabulary. Mirrors `t4_integrated_route_fiber.EXPERTS`;
#: a divergence is caught by `tests/test_t4_unified_proposal.py`, which reads
#: that module rather than restating the tuple.
ROUND_ONE_LANES = (*ESCALATABLE_LANES, "route_complete_region")

#: The rung-0 lanes, reachable only from a support-expansion event.
RUNG_ZERO_LANES = tuple(sorted(ROUTED_RUNG_ZERO_LANE.values()))

#: Every lane this unit can run.
LANES = (*ROUND_ONE_LANES, *RUNG_ZERO_LANES)

#: The contract block the state-aware rung-0 lane reads its settings from. The
#: settings are READ from the contract, never invented here, exactly as every
#: other lane's are.
STATE_AWARE_LANE = ROUTED_RUNG_ZERO_LANE["state_aware"]


class ProposalLaneError(ValueError):
    """A proposal lane was requested that this unit cannot honour as asked."""


def _padded_source(smiles: str):
    """The T4 proposal source: 48 SLOTS, which is not the 40 heavy-atom ceiling.

    ``PROPOSAL_SLOTS`` is the capacity of the padded array, and
    ``REPRESENTABLE_HEAVY_ATOMS = 40`` is the separate ceiling the fiber applies
    to ENDPOINTS; both hold at once. A tight graph silently deletes the whole
    insertion family from the legal support, so padding here is load-bearing.
    """

    return pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)


def _authorize_rung_zero(contract: Mapping, lane: str) -> None:
    """Refuse a rung-0 lane whose stage the contract does not authorize.

    Both rung-0 lanes exist to serve a routed expansion. Running one from an
    ordinary round -- by naming it in a proposal request -- would add a lane the
    contract never declared, so the authorization is checked on the worker that
    would otherwise have run it.
    """

    from compose_v4.control.frozen_proposal_escalation import resolve_frozen_escalation

    ladder = resolve_frozen_escalation(contract)
    if not ladder.zero_support_fallback:
        raise SupportExpansionContractError(
            f"lane {lane!r} is a rung-0 stage and the contract's ladder does not "
            "authorize one"
        )
    # The routing must also be resolvable, because the only legitimate way to
    # reach a rung-0 lane is to have been routed to it.
    resolve_state_routing(contract)


def proposal_unit(
    contract: Mapping,
    task: Mapping,
    *,
    fiber,
    route_expert: Any = None,
    region_law: Any = None,
) -> dict:
    """One expert on one parent. Returns eligible records and work counters.

    ``task`` carries ``expert``, ``parent``, ``parent_score``, ``original_seed``
    and ``proposal_seed``.  ``fiber`` is the production gate, already built
    against the cell's own declared similarity reference and delta.
    ``route_expert`` is the fitted expert the two model-backed lanes need; the
    caller loads it, because only the caller knows whether the checkpoint sits in
    a baked image or in a repository checkout.

    ``region_law`` is passed through to ``expand`` for the shallow lane and is
    resolved and PROVEN CONSUMED by ``region_law_contract`` in the caller, which
    is where the contract field lives.  ``expand`` itself fails closed on a law
    handed to any other lane, so a misrouted law raises rather than being
    silently dropped.
    """

    expert = task["expert"]
    if expert not in LANES:
        raise ProposalLaneError(
            f"unknown proposal lane {expert!r}; this unit runs {list(LANES)}"
        )
    parent = task["parent"]
    seed = int(task["proposal_seed"])
    started = time.time()

    if expert in ESCALATABLE_LANES:
        settings = contract["proposal"][expert]
        records = expand(
            parent,
            task["parent_score"],
            fiber,
            np.random.default_rng(seed),
            draws=int(task.get("draws") or settings["draws"]),
            multi_region=True,
            horizon=int(settings["horizon"]),
            proposal_lane=expert,
            region_law=region_law,
        )
        telemetry: dict = {"raw_draws": int(task.get("draws") or settings["draws"])}

    elif expert == "route_complete_region":
        settings = contract["proposal"][expert]
        if route_expert is None:
            raise ProposalLaneError(
                "the route_complete_region lane needs a fitted route expert; the "
                "caller must load the checkpoint its contract pins"
            )
        proposed, raw = propose_route_expert_candidates(
            _padded_source(parent),
            route_expert,
            pool_size=settings["pool_size"],
            realization_limit=settings["realization_limit"],
            beam_width=settings["beam_width"],
            expansion_width=settings["expansion_width"],
            max_bindings_per_template=settings["max_bindings_per_template"],
            maximum_expansions=settings["maximum_expansions"],
        )
        records = []
        for candidate in proposed:
            properties = fiber.check(candidate["smiles"])
            if properties is None or properties["smiles"] == parent:
                continue
            records.append(
                {
                    **candidate,
                    **properties,
                    "parent": parent,
                    "parent_score": task["parent_score"],
                    "delta": contract["delta"],
                }
            )
        telemetry = {
            key: value
            for key, value in raw.items()
            if not isinstance(value, (dict, list)) or key.endswith("counts")
        }

    elif expert == ROUTED_RUNG_ZERO_LANE["region"]:
        _authorize_rung_zero(contract, expert)
        produced, work = fallback_candidates(
            parent,
            np.random.default_rng(seed),
            check=fiber.check,
            reference_smiles=task["original_seed"],
            delta=float(contract["delta"]),
        )
        records = [
            {
                **record,
                "proposal_lane": None,
                "proposal_experts": [],
                "support_expansion_stage": expert,
                "parent": parent,
                "parent_score": task["parent_score"],
                "families": ("atom_delete",),
                "program_families": ("atom_delete",),
                "regions": 1,
                "created": int(record.get("inserted_atoms", 0)),
                "deleted": int(record.get("deleted_atoms", 0)),
                "delta": contract["delta"],
            }
            for record in produced
            if record["smiles"] != parent
        ]
        telemetry = {"zero_support_fallback_work": work.as_dict()}

    elif expert == STATE_AWARE_LANE:
        _authorize_rung_zero(contract, expert)
        from compose_v4.control.protonation_aware_proposal import (
            ProtonationAwareProposalConfig,
            propose_protonation_aware_candidates,
        )

        if route_expert is None:
            raise ProposalLaneError(
                f"the {STATE_AWARE_LANE} lane needs the shared retained route expert; "
                "the caller must load the checkpoint its contract pins"
            )
        settings = contract["proposal"][expert]
        proposed, raw = propose_protonation_aware_candidates(
            _padded_source(parent),
            route_expert,
            config=ProtonationAwareProposalConfig(
                seed=seed,
                shallow_draws=settings["shallow_draws"],
                retained_maximum_fragment_atoms=settings[
                    "retained_maximum_fragment_atoms"
                ],
                retained_maximum_stages=settings["retained_maximum_stages"],
                retained_maximum_prefixes=settings["retained_maximum_prefixes"],
                route_pool_size=settings["route_pool_size"],
                route_realization_limit=settings["route_realization_limit"],
                route_beam_width=settings["route_beam_width"],
                route_expansion_width=settings["route_expansion_width"],
                route_max_bindings_per_template=settings[
                    "route_max_bindings_per_template"
                ],
                route_maximum_expansions=settings["route_maximum_expansions"],
                route_candidate_timeout_seconds=settings[
                    "route_candidate_timeout_seconds"
                ],
            ),
        )
        records = []
        for candidate in proposed:
            properties = fiber.check(candidate["smiles"])
            if properties is None or properties["smiles"] == parent:
                continue
            records.append(
                {
                    **candidate,
                    **properties,
                    "proposal_lane": None,
                    "proposal_experts": [],
                    "support_expansion_stage": expert,
                    "families": sorted(set(candidate.get("program_families") or [])),
                    "parent": parent,
                    "parent_score": task["parent_score"],
                    "delta": contract["delta"],
                }
            )
        telemetry = {
            "protonation_aware_work": {
                key: raw.get(key)
                for key in (
                    "protonation_actions_enumerated",
                    "unique_exact_endpoints",
                    "lane_exact_programs",
                    "exact_execution_precision_numerator",
                    "exact_execution_precision_denominator",
                )
            },
            "distinct_eligible_endpoints": len(records),
        }

    else:  # pragma: no cover - LANES is exhaustive above
        raise ProposalLaneError(f"unhandled proposal lane {expert!r}")

    return {
        "schema_version": SCHEMA_VERSION,
        "expert": expert,
        "parent": parent,
        "proposal_seed": seed,
        "records": records,
        "eligible_records_returned": len(records),
        "telemetry": telemetry,
        "elapsed_seconds": round(time.time() - started, 2),
    }
