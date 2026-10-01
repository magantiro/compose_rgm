"""PMO population control with one explicit frozen-reference exploration slot."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import initial_dynamic_program_batch_v21
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.control.reference_programs import (
    FrozenProgramReference,
    ProgramPanelGuidance,
    pmo_program_input,
)
from compose_v4.control.reference_selection import draw_exploration
from compose_v4.control.region_replacement_option import RegionReplacementOption
from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw
from compose_v4.model.reference_checkpoint import load_frozen_reference


@dataclass(frozen=True)
class PmoProposalConfig:
    candidate_pool_limit: int = 16
    replacement_option_rate: float = 0.0
    realization_seconds_cap: float = 20.0

    def __post_init__(self):
        if type(self.candidate_pool_limit) is not int or self.candidate_pool_limit < 1:
            raise ValueError("PMO candidate pool limit must be a positive integer")
        if (
            not math.isfinite(self.replacement_option_rate)
            or not 0 <= self.replacement_option_rate <= 1
        ):
            raise ValueError("PMO region replacement rate must be a probability")
        if not math.isfinite(self.realization_seconds_cap) or self.realization_seconds_cap <= 0:
            raise ValueError("PMO realization seconds cap must be finite and positive")


def _proposal_tools(spec: PmoProposalConfig) -> dict:
    if spec.replacement_option_rate == 0:
        return {}
    law = ScaleBalancedRegionLaw()
    return {
        "region_law": law,
        "replacement_option": RegionReplacementOption(region_law=law),
        "replacement_option_rate": spec.replacement_option_rate,
    }


def initial_pmo_program_batch(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    broad_sampler=None,
    proposal_spec=None,
):
    """Build a PMO cold-start panel with the same declared proposal policy."""
    proposal = PmoProposalConfig(**(proposal_spec or {}))
    return initial_dynamic_program_batch_v21(
        source,
        entries,
        config,
        source_group=source_group,
        oracle_protocol=oracle_protocol,
        eligibility=eligibility,
        broad_sampler=broad_sampler,
        candidate_pool_limit=proposal.candidate_pool_limit,
        dynamic_synthesis_kwargs=_proposal_tools(proposal),
    )


class PmoReferenceController(PmoPopulationController):
    """Keep the PMO proposer and online allocator, reserving one query for exploration.

    Off mode draws that slot uniformly. Shadow mode scores the same panel without
    changing the draw. Active mode reweights only that slot. All modes use the
    same candidate pool, online observations and charged query allowance.
    """

    def __init__(
        self,
        *args,
        jump_checkpoint: dict,
        reference_spec: dict,
        proposal_spec: dict | None = None,
        **kwargs,
    ):
        if not isinstance(reference_spec, dict) or set(reference_spec) != {"guidance", "asset"}:
            raise ValueError("PMO reference specification needs guidance and asset fields")
        guidance = GuidanceConfig(**reference_spec["guidance"])
        asset = reference_spec["asset"]
        if guidance.mode == "off":
            if asset is not None:
                raise ValueError("off mode cannot carry an active reference asset")
            reference = None
        else:
            if not isinstance(asset, dict) or set(asset) != {
                "path",
                "sha256",
                "catalog_fingerprint",
                "catalog_path",
                "catalog_sha256",
            }:
                raise ValueError("shadow and active PMO modes need a pinned reference asset")
            reference = FrozenProgramReference(
                load_frozen_reference(
                    Path(asset["path"]),
                    expected_sha256=asset["sha256"],
                    expected_catalog_fingerprint=asset["catalog_fingerprint"],
                    catalog_path=Path(asset["catalog_path"]),
                    expected_catalog_sha256=asset["catalog_sha256"],
                )
            )
        super().__init__(*args, jump_checkpoint=jump_checkpoint, **kwargs)
        self.reference_spec = json.loads(json.dumps(reference_spec, sort_keys=True))
        self.guidance = guidance
        self.reference = reference
        proposal = PmoProposalConfig(**(proposal_spec or {}))
        self.proposal_spec = asdict(proposal)
        self.candidate_pool_limit = proposal.candidate_pool_limit
        self.realization_seconds_cap = proposal.realization_seconds_cap
        self.dynamic_synthesis_kwargs = _proposal_tools(proposal)

    def _allocate(self, candidates):
        chosen, detail = super()._allocate(candidates, reserve=1)
        selected_ids = {row["candidate_id"] for row in chosen}
        remaining = [row for row in candidates if row["candidate_id"] not in selected_ids]
        if not remaining:
            detail["reference_slot"] = {"status": "no_remaining_candidate"}
            return chosen, detail
        receipts: list[dict] = []
        guide = None
        if self.guidance.mode != "off":
            programs = tuple(pmo_program_input(row) for row in remaining)
            guide = ProgramPanelGuidance(programs, self.reference, self.guidance)
        index = draw_exploration(
            remaining,
            1,
            self.rng,
            guide=guide,
            receipts=receipts if guide is not None else None,
        )[0]
        candidate = remaining[index]
        chosen.append(candidate)
        candidate_id = candidate["candidate_id"]
        channel = candidate["provenance"]["planner_channel"]
        detail.setdefault("selected_ids", [])
        detail.setdefault(
            "selected_by_channel", {name: 0 for name in self.population_state["channels"]}
        )
        detail.setdefault("selection_role_by_candidate", {})
        detail["selected_ids"].append(candidate_id)
        detail["selected_by_channel"][channel] += 1
        detail["selection_role_by_candidate"][candidate_id] = "reference_exploration"
        detail["reference_slot"] = {
            "status": "selected",
            "candidate_id": candidate_id,
            "guidance": receipts[0] if receipts else {"outcome": "off"},
        }
        return chosen, detail

    def snapshot(self, *, include_history=True):
        snapshot = super().snapshot(include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["pmo_reference"] = {
            "spec_sha256": identity(self.reference_spec),
            "proposal_sha256": identity(self.proposal_spec),
            "guidance": asdict(self.guidance),
            "reference": None if self.reference is None else self.reference.identity(),
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(
        cls,
        snapshot,
        *,
        hierarchy=None,
        jump_checkpoint=None,
        reference_spec=None,
        proposal_spec=None,
    ):
        state = snapshot.get("pmo_reference", {})
        if reference_spec is None or state.get("spec_sha256") != identity(reference_spec):
            raise ValueError("PMO reference specification changed on resume")
        if state.get("proposal_sha256") != identity(
            asdict(PmoProposalConfig(**(proposal_spec or {})))
        ):
            raise ValueError("PMO proposal specification changed on resume")
        return PmoPopulationController.restore.__func__(
            cls,
            snapshot,
            hierarchy=hierarchy,
            jump_checkpoint=jump_checkpoint,
            reference_spec=reference_spec,
            proposal_spec=proposal_spec,
        )
