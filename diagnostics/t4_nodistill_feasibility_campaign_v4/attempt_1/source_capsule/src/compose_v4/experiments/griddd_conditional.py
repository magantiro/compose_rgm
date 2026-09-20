"""Budget-exact, lead-constrained conditional molecular optimization.

This module defines the engineering contract required before a full GrIDDD-
style benchmark can run.  It deliberately separates three arms:

``direct``
    QED-target-conditioned rates with a beta-zero shadow proposal mechanism.
``controller``
    Classifier-free/base rates with lead-aware QED/similarity control.
``combined``
    QED-target-conditioned rates plus the same controller.

Every arm receives the same number of candidate slots and the same exact QED
oracle-call allowance.  A guidance scope cannot overspend.  If a trajectory
ends before using its allowance, the remaining calls are explicitly recorded
as non-selecting padding calls on its terminal molecule.  Padding is visible
in the report and never masquerades as productive control.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator, rdMolDescriptors

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
)
from compose_v4.chem.source_prior import FixedMolecularStatePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.guided_rewrite_sampling import RewriteMarkSampler
from compose_v4.experiments.calibrated_rewrite_sampling import (
    ThinnedRateCalibrationSampler,
)
from compose_v4.experiments.canonical_successor_distillation import (
    CanonicalHistoryThinningSampler,
    FrozenQEDResidualAdapter,
)
from compose_v4.experiments.molecular_property_conditioning import (
    PropertyConditionNormalizer,
)
from compose_v4.experiments.property_conditioned_sampling import (
    PropertyConditionedRewriteSampler,
)
from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    SampledRewriteMark,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system


ArmName = Literal["direct", "controller", "combined"]
RingStratum = Literal["acyclic", "isolated_ring", "fused_or_bridged"]
StateScorer = Callable[[MolecularGraph], float]


BACKBONE_QUALIFICATION_FORMAT = (
    "compose_v4_canonical_successor_backbone_qualification_v1"
)
ANALYTIC_BACKBONE_QUALIFICATION_FORMAT = (
    "compose_v4_analytic_pancake_quotient_backbone_qualification_v1"
)
RETAINED_PANCAKE_CHECKPOINT_SHA256 = (
    "47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf"
)


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class UnconditionalBackboneQualification:
    """Validated handoff from the unconditional canonical-successor lane.

    The schema is intentionally narrow and fail-closed.  A one-state
    distillation smoke cannot be reinterpreted as a qualification manifest,
    and a conditional child checkpoint must cryptographically name this
    manifest and its qualified unconditional parent.
    """

    manifest_path: str
    manifest_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    weight_source: Literal["pancake_derived", "canonical_successor_native"]
    source_checkpoint_sha256: str
    property_condition_dim: int
    multi_state_panel_states: int
    qualification_passed: bool
    canonical_successor_execution: bool
    molecular_self_transitions_virtualized: bool
    empirical_mark_prior_mode: str
    ring_family_mass_mode: str
    p1_p2_imported: bool
    execution_kind: str
    execution_adapter_source_path: str | None
    execution_adapter_source_sha256: str | None
    atom_delete_log_rate_adjustment: float | None
    small_ring_log_rate_adjustment: float | None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        manifest_path: str,
        manifest_sha256: str,
    ) -> "UnconditionalBackboneQualification":
        artifact_format = str(payload.get("format", ""))
        if artifact_format not in {
            BACKBONE_QUALIFICATION_FORMAT,
            ANALYTIC_BACKBONE_QUALIFICATION_FORMAT,
        }:
            raise ValueError(
                "unconditional artifact is not a canonical-successor backbone "
                "qualification manifest"
            )
        checkpoint = payload.get("checkpoint")
        decision = payload.get("decision")
        execution = payload.get("execution")
        modes = payload.get("model_modes")
        evidence = payload.get("evidence")
        if not all(
            isinstance(value, Mapping)
            for value in (checkpoint, decision, execution, modes, evidence)
        ):
            raise ValueError("qualification manifest is missing required sections")
        assert isinstance(checkpoint, Mapping)
        assert isinstance(decision, Mapping)
        assert isinstance(execution, Mapping)
        assert isinstance(modes, Mapping)
        assert isinstance(evidence, Mapping)
        panel = evidence.get("multi_state_panel")
        heldout = evidence.get("heldout_rate_gate")
        rollout = evidence.get("rollout_gate")
        if not all(isinstance(value, Mapping) for value in (panel, heldout, rollout)):
            raise ValueError("qualification manifest lacks the three required evidence gates")
        assert isinstance(panel, Mapping)
        assert isinstance(heldout, Mapping)
        assert isinstance(rollout, Mapping)

        qualification_passed = bool(decision.get("qualification_passed", False))
        evidence_passed = all(
            bool(value.get("passed", False)) for value in (panel, heldout, rollout)
        )
        panel_states = int(panel.get("state_count", 0))
        if not qualification_passed or not evidence_passed or panel_states < 2:
            raise ValueError(
                "unconditional canonical-successor multi-state qualification has not passed"
            )

        weight_source = str(checkpoint.get("weight_source", ""))
        if weight_source not in {"pancake_derived", "canonical_successor_native"}:
            raise ValueError("qualified checkpoint has an unsupported weight lineage")
        checkpoint_sha256 = str(checkpoint.get("sha256", "")).lower()
        source_sha256 = str(checkpoint.get("source_checkpoint_sha256", "")).lower()
        if not _is_sha256(checkpoint_sha256) or not _is_sha256(source_sha256):
            raise ValueError("qualification checkpoint provenance lacks valid SHA-256 values")
        if (
            weight_source == "pancake_derived"
            and source_sha256 != RETAINED_PANCAKE_CHECKPOINT_SHA256
        ):
            raise ValueError("pancake-derived qualification does not name the retained pancake")

        canonical = bool(execution.get("canonical_successor_execution", False))
        virtual_self = bool(
            execution.get("molecular_self_transitions_virtualized", False)
        )
        empirical_mode = str(modes.get("empirical_mark_prior_mode", ""))
        ring_mode = str(modes.get("ring_family_mass_mode", ""))
        p1_p2_imported = bool(modes.get("p1_p2_imported", True))
        if not canonical:
            raise ValueError("qualified backbone does not use canonical successors")
        if not virtual_self:
            raise ValueError("qualified backbone does not virtualize molecular self events")
        if empirical_mode != "none" or p1_p2_imported:
            raise ValueError("qualified backbone imports prohibited P1 behavior")
        if ring_mode != "boolean":
            raise ValueError("qualified backbone imports prohibited P2 behavior")

        execution_kind = str(
            execution.get(
                "backbone_execution_kind",
                "canonical_successor_checkpoint_v1",
            )
        )
        adapter_source_path: str | None = None
        adapter_source_sha256: str | None = None
        atom_delete_adjustment: float | None = None
        small_ring_adjustment: float | None = None
        if artifact_format == ANALYTIC_BACKBONE_QUALIFICATION_FORMAT:
            if execution_kind != "analytic_pancake_quotient_adapter_v1":
                raise ValueError("analytic qualification has the wrong execution kind")
            if checkpoint_sha256 != RETAINED_PANCAKE_CHECKPOINT_SHA256:
                raise ValueError("analytic qualification does not bind the retained pancake")
            analytic_adapter = execution.get("analytic_adapter")
            calibration = execution.get("calibration")
            if not isinstance(analytic_adapter, Mapping) or not isinstance(
                calibration, Mapping
            ):
                raise ValueError(
                    "analytic qualification lacks adapter and calibration provenance"
                )
            adapter_source_path = str(analytic_adapter.get("source_path", ""))
            adapter_source_sha256 = str(
                analytic_adapter.get("source_sha256", "")
            ).lower()
            if not adapter_source_path or not _is_sha256(adapter_source_sha256):
                raise ValueError("analytic adapter source provenance is invalid")
            atom_delete_adjustment = float(
                calibration.get("atom_delete_log_rate_adjustment", float("nan"))
            )
            small_ring_adjustment = float(
                calibration.get("small_ring_log_rate_adjustment", float("nan"))
            )
            if atom_delete_adjustment != -0.5 or small_ring_adjustment != -1.5:
                raise ValueError("analytic qualification changed the passed calibration")
            if int(calibration.get("small_ring_maximum_size", 0)) != 4:
                raise ValueError("analytic qualification changed the small-ring boundary")
            if (
                str(execution.get("history_safety", ""))
                != "canonical_history_exact_thinning_v1"
            ):
                raise ValueError("analytic qualification lacks canonical-history safety")

        return cls(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256.lower(),
            checkpoint_path=str(checkpoint.get("path", "")),
            checkpoint_sha256=checkpoint_sha256,
            weight_source=weight_source,  # type: ignore[arg-type]
            source_checkpoint_sha256=source_sha256,
            property_condition_dim=int(checkpoint.get("property_condition_dim", 0)),
            multi_state_panel_states=panel_states,
            qualification_passed=True,
            canonical_successor_execution=canonical,
            molecular_self_transitions_virtualized=virtual_self,
            empirical_mark_prior_mode=empirical_mode,
            ring_family_mass_mode=ring_mode,
            p1_p2_imported=p1_p2_imported,
            execution_kind=execution_kind,
            execution_adapter_source_path=adapter_source_path,
            execution_adapter_source_sha256=adapter_source_sha256,
            atom_delete_log_rate_adjustment=atom_delete_adjustment,
            small_ring_log_rate_adjustment=small_ring_adjustment,
        )

    def assert_checkpoint_file(self, path: Path) -> None:
        observed = file_sha256(path)
        if observed != self.checkpoint_sha256:
            raise ValueError(
                "qualified unconditional checkpoint hash mismatch: "
                f"expected {self.checkpoint_sha256}, observed {observed}"
            )

    def assert_conditional_child(
        self,
        payload: Mapping[str, object],
        *,
        checkpoint_sha256: str,
    ) -> None:
        """Require a conditioned child to bind this exact qualified parent."""

        if int(payload.get("property_condition_dim", 0)) != 1:
            raise ValueError("three-arm smoke requires a QED-conditioned child checkpoint")
        conditioning = payload.get("property_conditioning")
        if not isinstance(conditioning, Mapping) or tuple(
            str(value) for value in conditioning.get("names", ())
        ) != ("qed",):
            raise ValueError("conditional child lacks frozen QED conditioning metadata")
        if (
            str(
                payload.get(
                    "unconditional_backbone_qualification_manifest_sha256", ""
                )
            ).lower()
            != self.manifest_sha256
        ):
            raise ValueError("conditional child is not bound to this qualification manifest")
        if (
            str(payload.get("unconditional_backbone_checkpoint_sha256", "")).lower()
            != self.checkpoint_sha256
        ):
            raise ValueError("conditional child is not bound to the qualified checkpoint")
        if str(payload.get("weight_source", "")) != self.weight_source:
            raise ValueError("conditional child checkpoint lineage disagrees with its parent")
        if not bool(payload.get("canonical_successor_execution", False)):
            raise ValueError("conditional child does not assert canonical-successor execution")
        if not bool(payload.get("molecular_self_transitions_virtualized", False)):
            raise ValueError("conditional child does not assert virtualized molecular self events")
        if str(payload.get("empirical_mark_prior_mode", "none")) != "none":
            raise ValueError("conditional child imports prohibited P1 behavior")
        if str(payload.get("ring_family_mass_mode", "boolean")) != "boolean":
            raise ValueError("conditional child imports prohibited P2 behavior")
        if bool(payload.get("p1_p2_imported", False)):
            raise ValueError("conditional child reports prohibited P1/P2 imports")
        if not _is_sha256(checkpoint_sha256):
            raise ValueError("conditional child checkpoint SHA-256 is invalid")
        if self.execution_kind == "analytic_pancake_quotient_adapter_v1":
            if (
                str(payload.get("unconditional_backbone_execution_kind", ""))
                != self.execution_kind
            ):
                raise ValueError("conditional child does not bind analytic execution kind")
            sidecar = payload.get("conditional_adapter")
            if not isinstance(sidecar, Mapping):
                raise ValueError("conditional child lacks frozen residual sidecar metadata")
            required = {
                "kind": "frozen_valid_state_qed_residual_v1",
                "base_parameters_frozen": True,
                "base_weights_in_sidecar_state_dict": False,
                "family_hazard_residual": True,
                "within_family_residual": True,
                "independent_total_hazard_residual": False,
                "missing_condition_identity": True,
            }
            if any(sidecar.get(key) != value for key, value in required.items()):
                raise ValueError("conditional child violates the frozen residual contract")

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "weight_source": self.weight_source,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "property_condition_dim": self.property_condition_dim,
            "multi_state_panel_states": self.multi_state_panel_states,
            "qualification_passed": self.qualification_passed,
            "canonical_successor_execution": self.canonical_successor_execution,
            "molecular_self_transitions_virtualized": (
                self.molecular_self_transitions_virtualized
            ),
            "empirical_mark_prior_mode": self.empirical_mark_prior_mode,
            "ring_family_mass_mode": self.ring_family_mass_mode,
            "p1_p2_imported": self.p1_p2_imported,
            "execution_kind": self.execution_kind,
            "execution_adapter_source_path": self.execution_adapter_source_path,
            "execution_adapter_source_sha256": (
                self.execution_adapter_source_sha256
            ),
            "atom_delete_log_rate_adjustment": (
                self.atom_delete_log_rate_adjustment
            ),
            "small_ring_log_rate_adjustment": (
                self.small_ring_log_rate_adjustment
            ),
        }


def load_unconditional_backbone_qualification(
    path: Path,
) -> UnconditionalBackboneQualification:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("qualification manifest must contain a JSON object")
    return UnconditionalBackboneQualification.from_payload(
        raw,
        manifest_path=str(path),
        manifest_sha256=file_sha256(path),
    )


@dataclass(frozen=True)
class CorrectedBackboneContract:
    """Fail-closed interface for a canonical-successor conditional backbone.

    ``weight_source`` records lineage; it does not silently convert an old
    checkpoint into a corrected one.  Qualification is an explicit external
    gate.  P1/P2 modes are rejected because they have not produced a validated
    checkpoint and are outside this conditional system's evidence boundary.
    """

    checkpoint_sha256: str
    weight_source: Literal["pancake_derived", "canonical_successor_native"]
    canonical_successor_execution: bool
    molecular_self_transitions_virtualized: bool
    unconditional_qualification_passed: bool
    empirical_mark_prior_mode: str = "none"
    ring_family_mass_mode: str = "boolean"
    unconditional_execution_kind: str = "canonical_successor_checkpoint_v1"

    def __post_init__(self) -> None:
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.checkpoint_sha256.lower()
        ):
            raise ValueError("checkpoint SHA-256 must be a 64-character hex digest")
        if not self.canonical_successor_execution:
            raise ValueError("conditional backbone requires canonical-successor execution")
        if not self.molecular_self_transitions_virtualized:
            raise ValueError("conditional backbone must virtualize molecular self-transitions")
        if self.empirical_mark_prior_mode != "none":
            raise ValueError("unvalidated P1 empirical mark priors are prohibited")
        if self.ring_family_mass_mode != "boolean":
            raise ValueError("unvalidated P2 topology-mass modes are prohibited")
        if self.unconditional_execution_kind not in {
            "canonical_successor_checkpoint_v1",
            "analytic_pancake_quotient_adapter_v1",
        }:
            raise ValueError("unsupported unconditional execution kind")

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
        *,
        checkpoint_sha256: str,
        weight_source: Literal["pancake_derived", "canonical_successor_native"],
        unconditional_qualification_passed: bool,
        canonical_successor_execution: bool = True,
        molecular_self_transitions_virtualized: bool = True,
    ) -> "CorrectedBackboneContract":
        return cls(
            checkpoint_sha256=checkpoint_sha256,
            weight_source=weight_source,
            canonical_successor_execution=canonical_successor_execution,
            molecular_self_transitions_virtualized=(
                molecular_self_transitions_virtualized
            ),
            unconditional_qualification_passed=unconditional_qualification_passed,
            empirical_mark_prior_mode=str(
                payload.get("empirical_mark_prior_mode", "none")
            ),
            ring_family_mass_mode=str(payload.get("ring_family_mass_mode", "boolean")),
            unconditional_execution_kind=str(
                payload.get(
                    "unconditional_backbone_execution_kind",
                    "canonical_successor_checkpoint_v1",
                )
            ),
        )

    @classmethod
    def from_qualification_manifest(
        cls,
        qualification: UnconditionalBackboneQualification,
        conditional_checkpoint_payload: Mapping[str, object],
        *,
        conditional_checkpoint_sha256: str,
    ) -> "CorrectedBackboneContract":
        qualification.assert_conditional_child(
            conditional_checkpoint_payload,
            checkpoint_sha256=conditional_checkpoint_sha256,
        )
        return cls(
            checkpoint_sha256=conditional_checkpoint_sha256,
            weight_source=qualification.weight_source,
            canonical_successor_execution=(
                qualification.canonical_successor_execution
            ),
            molecular_self_transitions_virtualized=(
                qualification.molecular_self_transitions_virtualized
            ),
            unconditional_qualification_passed=(
                qualification.qualification_passed
            ),
            empirical_mark_prior_mode=str(
                conditional_checkpoint_payload.get(
                    "empirical_mark_prior_mode", "none"
                )
            ),
            ring_family_mass_mode=str(
                conditional_checkpoint_payload.get(
                    "ring_family_mass_mode", "boolean"
                )
            ),
            unconditional_execution_kind=qualification.execution_kind,
        )

    def require_qualified(self, *, allow_unqualified_smoke: bool = False) -> None:
        if not self.unconditional_qualification_passed and not allow_unqualified_smoke:
            raise RuntimeError(
                "corrected backbone has not passed the unconditional qualification gate"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "weight_source": self.weight_source,
            "canonical_successor_execution": self.canonical_successor_execution,
            "molecular_self_transitions_virtualized": (
                self.molecular_self_transitions_virtualized
            ),
            "unconditional_qualification_passed": (
                self.unconditional_qualification_passed
            ),
            "empirical_mark_prior_mode": self.empirical_mark_prior_mode,
            "ring_family_mass_mode": self.ring_family_mass_mode,
            "p1_p2_imported": False,
            "unconditional_execution_kind": self.unconditional_execution_kind,
        }


@dataclass(frozen=True)
class FactorizedConditionalBackbone:
    """Bind a validated execution contract to factorized rewrite weights."""

    model: FactorizedTraceletRateModel
    contract: CorrectedBackboneContract
    qed_normalizer: PropertyConditionNormalizer | None = None
    allow_unqualified_smoke: bool = False

    def __post_init__(self) -> None:
        self.contract.require_qualified(
            allow_unqualified_smoke=self.allow_unqualified_smoke
        )
        if self.qed_normalizer is not None and self.qed_normalizer.names != ("qed",):
            raise ValueError("direct conditional backbone requires a QED normalizer")
        if int(self.model.property_condition_dim) not in {0, 1}:
            raise ValueError("conditional backbone supports zero or one QED condition")
        if int(self.model.property_condition_dim) == 1 and self.qed_normalizer is None:
            raise ValueError("conditioned weights require their frozen QED normalizer")
        if self.model.empirical_mark_prior_mode != "none":
            raise ValueError("conditional backbone model unexpectedly enables P1")
        if self.model.ring_family_mass_mode != "boolean":
            raise ValueError("conditional backbone model unexpectedly enables P2")
        if self.contract.weight_source == "pancake_derived" and not bool(
            getattr(self.model, "virtualize_legacy_self_grafts", False)
        ):
            raise ValueError(
                "pancake-derived weights require runtime self-transition virtualization"
            )

    def sampler(self, *, target_qed: float, direct_conditioning: bool) -> RewriteMarkSampler:
        dimension = int(self.model.property_condition_dim)
        if direct_conditioning:
            if dimension != 1 or self.qed_normalizer is None:
                raise RuntimeError("direct QED conditioning weights are not available")
            return PropertyConditionedRewriteSampler(
                self.model,
                property_values=self.qed_normalizer.transform((target_qed,)),
            )
        if dimension == 0:
            return self.model
        return PropertyConditionedRewriteSampler(
            self.model,
            property_values=(0.0,),
            property_mask=(False,),
        )


@dataclass(frozen=True)
class _TargetBoundQEDResidualSampler:
    adapter: FrozenQEDResidualAdapter
    target_qed: float | None

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        return self.adapter.sample_rewrite_mark_conditioned(
            state,
            time,
            rng,
            target_qed=self.target_qed,
        )


@dataclass
class _RingFamilyAuditSampler:
    """Inference-only ring-family availability audit around one sampler."""

    base_sampler: RewriteMarkSampler
    availability_probe: Callable[[MolecularGraph, float], Mapping[str, bool]]
    _observations: int = field(default=0, init=False, repr=False)
    _available_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def begin_rollout_audit(self) -> None:
        self._observations = 0
        self._available_counts = {
            "ring_system_grow": 0,
            "ring_system_delete": 0,
            "ring_system_restate": 0,
        }
        begin = getattr(self.base_sampler, "begin_rollout_audit", None)
        if callable(begin):
            begin()

    def end_rollout_audit(self) -> dict[str, object]:
        end = getattr(self.base_sampler, "end_rollout_audit", None)
        base = end() if callable(end) else None
        return {
            "method": "ring_family_availability_audit_v1",
            "availability_observations": self._observations,
            "available_observation_counts": dict(self._available_counts),
            "available_at_any_observation": {
                name: count > 0 for name, count in self._available_counts.items()
            },
            "base_sampler_diagnostics": base,
        }

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        available = self.availability_probe(state, time)
        self._observations += 1
        for name in self._available_counts:
            self._available_counts[name] += int(bool(available.get(name, False)))
        return self.base_sampler.sample_rewrite_mark(state, time, rng)


@dataclass(frozen=True)
class FrozenResidualConditionalBackbone:
    """Qualified analytic quotient base plus a trainable QED-only sidecar."""

    adapter: FrozenQEDResidualAdapter
    contract: CorrectedBackboneContract
    small_ring_log_rate_adjustment: float = -1.5
    small_ring_maximum_size: int = 4
    allow_unqualified_smoke: bool = False

    def __post_init__(self) -> None:
        self.contract.require_qualified(
            allow_unqualified_smoke=self.allow_unqualified_smoke
        )
        sidecar = self.adapter.sidecar_contract()
        if not bool(sidecar["base_parameters_frozen"]):
            raise ValueError("conditional sidecar does not freeze the base parameters")
        if bool(sidecar["base_weights_in_sidecar_state_dict"]):
            raise ValueError("conditional sidecar checkpoint contains base weights")
        if bool(sidecar["independent_total_hazard_residual"]):
            raise ValueError("conditional sidecar adds a prohibited total-hazard residual")
        if not bool(sidecar["missing_condition_identity"]):
            raise ValueError("conditional sidecar lacks exact missing-condition identity")

    def sampler(
        self,
        *,
        target_qed: float,
        direct_conditioning: bool,
    ) -> RewriteMarkSampler:
        condition = float(target_qed) if direct_conditioning else None
        raw = _TargetBoundQEDResidualSampler(self.adapter, condition)
        calibrated = ThinnedRateCalibrationSampler(
            raw,
            small_ring_log_rate_adjustment=float(
                self.small_ring_log_rate_adjustment
            ),
            small_ring_maximum_size=int(self.small_ring_maximum_size),
        )
        history_safe = CanonicalHistoryThinningSampler(calibrated)

        def availability(state: MolecularGraph, time: float) -> Mapping[str, bool]:
            table = self.adapter.rate_table(
                state,
                time,
                target_qed=condition,
            )
            return {
                name: bool(table.enabled_families[index])
                for index, name in enumerate(
                    (
                        "atom_insert",
                        "atom_delete",
                        "atom_restate",
                        "bond_reorder",
                        "bond_reroute",
                        "cycle_insert",
                        "cycle_attach",
                        "ring_system_grow",
                        "ring_system_delete",
                        "ring_system_restate",
                    )
                )
                if name.startswith("ring_system_")
            }

        return _RingFamilyAuditSampler(history_safe, availability)


@dataclass(frozen=True)
class GridDDProtocol:
    """Frozen GrIDDD-style task and budget definition."""

    starting_qed_minimum: float = 0.70
    starting_qed_maximum: float = 0.80
    target_qed: float = 0.90
    minimum_tanimoto_similarity: float = 0.40
    candidates_per_start: int = 20
    guidance_oracle_calls_per_candidate: int = 48
    proposals_per_controlled_event: int = 4
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 20260722

    def __post_init__(self) -> None:
        values = (
            self.starting_qed_minimum,
            self.starting_qed_maximum,
            self.target_qed,
            self.minimum_tanimoto_similarity,
        )
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("QED and similarity thresholds must lie in [0, 1]")
        if self.starting_qed_minimum > self.starting_qed_maximum:
            raise ValueError("starting QED bounds are reversed")
        if self.target_qed <= self.starting_qed_maximum:
            raise ValueError("target QED must exceed the starting-QED window")
        if self.candidates_per_start <= 0:
            raise ValueError("candidate count must be positive")
        if self.guidance_oracle_calls_per_candidate < 0:
            raise ValueError("guidance oracle-call budget must be nonnegative")
        if self.proposals_per_controlled_event <= 0:
            raise ValueError("proposal count must be positive")
        if (
            self.guidance_oracle_calls_per_candidate
            and self.guidance_oracle_calls_per_candidate
            < self.proposals_per_controlled_event
        ):
            raise ValueError("guidance budget must accommodate one proposal set")
        if self.fingerprint_radius <= 0 or self.fingerprint_bits <= 0:
            raise ValueError("fingerprint parameters must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap replicate count must be positive")

    @property
    def oracle_calls_per_start_seed_arm(self) -> int:
        return 1 + self.candidates_per_start * (
            self.guidance_oracle_calls_per_candidate + 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fairness_role": "protocol_b_compose_internal_matched_control",
            "query_matched_griddd_claim_authorized": False,
            "starting_qed_interval": [
                self.starting_qed_minimum,
                self.starting_qed_maximum,
            ],
            "target_qed": self.target_qed,
            "minimum_tanimoto_similarity": self.minimum_tanimoto_similarity,
            "candidates_per_start": self.candidates_per_start,
            "guidance_oracle_calls_per_candidate": (
                self.guidance_oracle_calls_per_candidate
            ),
            "final_candidate_oracle_calls_per_candidate": 1,
            "lead_eligibility_oracle_calls": 1,
            "oracle_calls_per_start_seed_arm": (
                self.oracle_calls_per_start_seed_arm
            ),
            "proposals_per_controlled_event": (
                self.proposals_per_controlled_event
            ),
            "fingerprint": {
                "type": "Morgan bit vector",
                "radius": self.fingerprint_radius,
                "bits": self.fingerprint_bits,
            },
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
        }


@dataclass(frozen=True)
class GridDDBenchmarkFairnessContract:
    """Keep native GrIDDD comparison separate from COMPOSE control studies."""

    official_griddd_lead_set_id: str = "griddd_release_qed_800_exact_unresolved"
    exact_griddd_leads_available: bool = False
    candidates_per_start: int = 20
    internal_guidance_oracle_calls_per_candidate: int = 48

    def __post_init__(self) -> None:
        if not self.official_griddd_lead_set_id:
            raise ValueError("the benchmark lead-set identifier must be non-empty")
        if self.candidates_per_start != 20:
            raise ValueError("the frozen GrIDDD comparison requires 20 candidates")
        if self.internal_guidance_oracle_calls_per_candidate <= 0:
            raise ValueError("internal control budget must be positive")
        if self.exact_griddd_leads_available and self.official_griddd_lead_set_id.endswith(
            "unresolved"
        ):
            raise ValueError("an unresolved lead set cannot be marked exact and available")

    @property
    def internal_oracle_calls_per_start_seed_arm(self) -> int:
        return 1 + self.candidates_per_start * (
            self.internal_guidance_oracle_calls_per_candidate + 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "compose_v4_griddd_two_protocol_fairness_v1",
            "shared_task": {
                "starting_qed_interval": [0.70, 0.80],
                "target_qed_interval": [0.90, 1.0],
                "minimum_tanimoto_similarity": 0.40,
                "fingerprint": {
                    "type": "Morgan bit vector",
                    "radius": 2,
                    "bits": 2048,
                    "use_chirality": False,
                },
                "candidates_per_start": self.candidates_per_start,
                "all_start_denominator": True,
            },
            "protocol_a_griddd_comparable": {
                "lead_set_id": self.official_griddd_lead_set_id,
                "exact_leads_available": self.exact_griddd_leads_available,
                "arm": "direct_conditioned_native_sampling",
                "candidate_selection": "native_model_sampling_no_oracle_tilt",
                "oracle_accounting": (
                    "report only benchmark evaluation calls; native sampling cost "
                    "is reported separately"
                ),
                "beta_zero_shadow_rescoring": False,
                "padding_calls": False,
                "results_table": "griddd_comparable_primary_only",
            },
            "protocol_b_compose_internal": {
                "arms": ["direct", "controller", "combined"],
                "candidate_rng_streams_matched_across_arms": True,
                "guidance_oracle_calls_per_candidate": (
                    self.internal_guidance_oracle_calls_per_candidate
                ),
                "exact_oracle_calls_per_start_seed_arm": (
                    self.internal_oracle_calls_per_start_seed_arm
                ),
                "selection_influencing_and_padding_calls_reported_separately": True,
                "results_table": "compose_control_mechanism_only",
            },
            "claim_boundary": {
                "tables_must_remain_separate": True,
                "protocol_b_is_query_matched_to_griddd": False,
                "query_matched_claim_requires_external_baseline_at_same_budget": True,
            },
        }


@dataclass(frozen=True)
class ConditionalArm:
    name: ArmName
    direct_conditioning: bool
    controller: bool
    beta: float

    def __post_init__(self) -> None:
        expected = {
            "direct": (True, False),
            "controller": (False, True),
            "combined": (True, True),
        }
        if self.name not in expected:
            raise ValueError("unknown conditional arm")
        if expected[self.name] != (self.direct_conditioning, self.controller):
            raise ValueError("arm name and conditioning/controller flags disagree")
        if not isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("controller beta must be finite and nonnegative")
        if self.controller and self.beta <= 0.0:
            raise ValueError("controller arms require positive beta")
        if not self.controller and self.beta != 0.0:
            raise ValueError("direct arm must use beta zero")


def standard_conditional_arms(*, beta: float = 8.0) -> tuple[ConditionalArm, ...]:
    return (
        ConditionalArm("direct", True, False, 0.0),
        ConditionalArm("controller", False, True, beta),
        ConditionalArm("combined", True, True, beta),
    )


@dataclass(frozen=True)
class OracleInvocation:
    index: int
    purpose: str
    canonical_state_key: str
    value: float
    influences_selection: bool
    padding: bool
    wall_time_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "purpose": self.purpose,
            "canonical_state_key": self.canonical_state_key,
            "value": self.value,
            "influences_selection": self.influences_selection,
            "padding": self.padding,
            "wall_time_seconds": self.wall_time_seconds,
        }


@dataclass
class ExactOracleBudgetLedger:
    """Count actual deterministic oracle invocations without a hidden cache."""

    scorer: StateScorer
    budget: int
    invocations: list[OracleInvocation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("oracle-call budget must be positive")

    @property
    def used(self) -> int:
        return len(self.invocations)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def score(
        self,
        state: MolecularGraph,
        *,
        purpose: str,
        influences_selection: bool,
        padding: bool = False,
    ) -> float:
        if self.remaining <= 0:
            raise RuntimeError("oracle-call budget exhausted")
        if not purpose:
            raise ValueError("oracle purpose must be non-empty")
        if not is_valid_state(state) or not is_connected_or_null(state):
            raise ValueError("oracle inputs must be valid connected molecular states")
        started = perf_counter()
        value = float(self.scorer(state))
        elapsed = perf_counter() - started
        if not isfinite(value):
            raise ValueError("oracle returned a non-finite value")
        self.invocations.append(
            OracleInvocation(
                index=self.used,
                purpose=purpose,
                canonical_state_key=canonical_state_key(state),
                value=value,
                influences_selection=bool(influences_selection),
                padding=bool(padding),
                wall_time_seconds=float(elapsed),
            )
        )
        return value

    def scope(self, allowance: int, *, label: str) -> "OracleBudgetScope":
        if allowance < 0 or allowance > self.remaining:
            raise ValueError("invalid scoped oracle allowance")
        return OracleBudgetScope(self, allowance=allowance, label=label)

    def assert_exact(self) -> None:
        if self.used != self.budget:
            raise RuntimeError(
                f"oracle budget underfilled: used {self.used} of {self.budget}"
            )

    def summary(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "used": self.used,
            "exact": self.used == self.budget,
            "selection_calls": sum(
                invocation.influences_selection for invocation in self.invocations
            ),
            "padding_calls": sum(invocation.padding for invocation in self.invocations),
            "unique_canonical_states": len(
                {invocation.canonical_state_key for invocation in self.invocations}
            ),
            "oracle_wall_time_seconds": sum(
                invocation.wall_time_seconds for invocation in self.invocations
            ),
        }


@dataclass
class OracleBudgetScope:
    parent: ExactOracleBudgetLedger
    allowance: int
    label: str
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.allowance - self.used

    def score(
        self,
        state: MolecularGraph,
        *,
        purpose: str,
        influences_selection: bool,
        padding: bool = False,
    ) -> float:
        if self.remaining <= 0:
            raise RuntimeError("scoped oracle allowance exhausted")
        value = self.parent.score(
            state,
            purpose=f"{self.label}:{purpose}",
            influences_selection=influences_selection,
            padding=padding,
        )
        self.used += 1
        return value

    def fill(self, state: MolecularGraph) -> None:
        while self.remaining:
            self.score(
                state,
                purpose="unused_guidance_padding",
                influences_selection=False,
                padding=True,
            )

    def assert_exact(self) -> None:
        if self.used != self.allowance:
            raise RuntimeError("scoped oracle allowance underfilled")


def qed_state_oracle(state: MolecularGraph) -> float:
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise ValueError("QED oracle does not score the formal null state")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("QED oracle could not parse a valid molecular state")
    return float(QED.qed(molecule))


@dataclass(frozen=True)
class LeadFingerprint:
    state_key: str
    fingerprint: object


def _morgan_fingerprint(
    state: MolecularGraph,
    *,
    radius: int,
    bits: int,
) -> object:
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise ValueError("fingerprint requires a non-null molecular state")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("fingerprint conversion failed")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    return generator.GetFingerprint(molecule)


def tanimoto_similarity(
    left: MolecularGraph,
    right: MolecularGraph,
    *,
    radius: int = 2,
    bits: int = 2048,
) -> float:
    left_fp = _morgan_fingerprint(left, radius=radius, bits=bits)
    right_fp = _morgan_fingerprint(right, radius=radius, bits=bits)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


@dataclass(frozen=True)
class LeadAwareProposal:
    mark: SampledRewriteMark
    successor: MolecularGraph
    successor_key: str
    qed: float
    similarity: float


@dataclass
class ExactBudgetLeadGuidedSampler:
    """Canonical-successor proposal control with an exact oracle allowance."""

    base_sampler: RewriteMarkSampler
    oracle_scope: OracleBudgetScope
    lead_state: MolecularGraph
    target_qed: float
    minimum_similarity: float
    proposals_per_event: int = 4
    beta: float = 0.0
    similarity_penalty: float = 4.0
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    _audit_events: list[dict[str, object]] = field(default_factory=list, init=False)
    _runtime: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.proposals_per_event <= 0:
            raise ValueError("proposals per event must be positive")
        if not isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")
        if not isfinite(self.similarity_penalty) or self.similarity_penalty <= 0.0:
            raise ValueError("similarity penalty must be finite and positive")
        if self.fingerprint_radius <= 0 or self.fingerprint_bits <= 0:
            raise ValueError("fingerprint parameters must be positive")
        self._runtime = de_novo_rewrite_system()

    def begin_rollout_audit(self) -> None:
        self._audit_events.clear()
        begin = getattr(self.base_sampler, "begin_rollout_audit", None)
        if callable(begin):
            begin()

    def sample_hazard_probe(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        return self.base_sampler.sample_rewrite_mark(state, time, rng)

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        initial = self.sample_hazard_probe(state, time, rng)
        return self.resample_rewrite_mark_after_event(
            state,
            time,
            rng,
            initial,
        )

    def resample_rewrite_mark_after_event(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        initial: SampledRewriteMark,
    ) -> SampledRewriteMark:
        if float(initial.total_hazard) <= 1e-12:
            return initial
        # Reserving K calls before an event guarantees that canonical grouping
        # can never overspend.  Duplicate successors leave calls for later
        # events or explicit terminal padding.
        if self.oracle_scope.remaining < self.proposals_per_event:
            return initial
        proposals = self._draw_proposals(state, time, rng, initial)
        if not proposals:
            return initial
        utilities = np.asarray(
            [
                -abs(proposal.qed - self.target_qed)
                - self.similarity_penalty
                * max(self.minimum_similarity - proposal.similarity, 0.0)
                for proposal in proposals
            ],
            dtype=np.float64,
        )
        log_weights = self.beta * utilities
        log_weights -= float(log_weights.max())
        probabilities = np.exp(log_weights)
        probabilities /= probabilities.sum()
        selected = int(rng.choice(len(proposals), p=probabilities))
        self._audit_events.append(
            {
                "source_key": canonical_state_key(state),
                "model_time": float(time),
                "beta": self.beta,
                "proposals": tuple(
                    {
                        "successor_key": proposal.successor_key,
                        "rule_name": proposal.mark.rule_name,
                        "qed": proposal.qed,
                        "similarity": proposal.similarity,
                        "selection_probability": float(probabilities[index]),
                    }
                    for index, proposal in enumerate(proposals)
                ),
                "selected_index": selected,
            }
        )
        return proposals[selected].mark

    def _draw_proposals(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        initial: SampledRewriteMark,
    ) -> tuple[LeadAwareProposal, ...]:
        marks = [initial]
        reference_hazard = float(initial.total_hazard)
        for _ in range(self.proposals_per_event - 1):
            sampled = self.base_sampler.sample_rewrite_mark(state, time, rng)
            if not np.isclose(
                float(sampled.total_hazard),
                reference_hazard,
                rtol=1e-5,
                atol=1e-8,
            ):
                raise RuntimeError("base sampler changed the state hazard")
            marks.append(sampled)
        unscored: list[tuple[SampledRewriteMark, MolecularGraph, str]] = []
        for mark in marks:
            successor = (
                state
                if mark.action is None
                else self._runtime.apply(state, mark.rule_name, mark.action)
            )
            unscored.append((mark, successor, canonical_state_key(successor)))
        scored: dict[str, tuple[float, float]] = {}
        for _mark, successor, key in unscored:
            if key in scored:
                continue
            qed = self.oracle_scope.score(
                successor,
                purpose="canonical_successor_qed",
                influences_selection=self.beta > 0.0,
            )
            similarity = tanimoto_similarity(
                self.lead_state,
                successor,
                radius=self.fingerprint_radius,
                bits=self.fingerprint_bits,
            )
            scored[key] = (qed, similarity)
        return tuple(
            LeadAwareProposal(mark, successor, key, *scored[key])
            for mark, successor, key in unscored
        )

    def finalize_rollout_audit(self, final_state: MolecularGraph) -> dict[str, object]:
        self.oracle_scope.fill(final_state)
        self.oracle_scope.assert_exact()
        end = getattr(self.base_sampler, "end_rollout_audit", None)
        base_diagnostics = end() if callable(end) else None
        return {
            "oracle_allowance": self.oracle_scope.allowance,
            "oracle_calls": self.oracle_scope.used,
            "padding_calls": sum(
                invocation.padding
                for invocation in self.oracle_scope.parent.invocations
                if invocation.purpose.startswith(f"{self.oracle_scope.label}:")
            ),
            "proposal_events": len(self._audit_events),
            "raw_proposal_marks": self.proposals_per_event * len(self._audit_events),
            "events": tuple(self._audit_events),
            "base_sampler_diagnostics": base_diagnostics,
        }


@dataclass(frozen=True)
class CandidateGenerationResult:
    final_state: MolecularGraph | None
    trajectory_states_audited: int
    all_trajectory_states_valid: bool
    all_trajectory_states_connected: bool
    events: int
    exhausted_event_budget: bool
    stalled: bool = False
    error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class LeadCandidateGenerator(Protocol):
    def generate(
        self,
        *,
        lead_state: MolecularGraph,
        target_qed: float,
        arm: ConditionalArm,
        rng: np.random.Generator,
        oracle_scope: OracleBudgetScope,
    ) -> CandidateGenerationResult: ...


class NativeDirectLeadCandidateGenerator(Protocol):
    def generate_native(
        self,
        *,
        lead_state: MolecularGraph,
        target_qed: float,
        rng: np.random.Generator,
    ) -> CandidateGenerationResult: ...


@dataclass(frozen=True)
class RewriteLeadCandidateGenerator:
    """Generate one edited candidate from a fixed lead molecule."""

    backbone: FactorizedConditionalBackbone | FrozenResidualConditionalBackbone
    n_slots: int = 40
    operational_horizon: float = 16.0
    time_step: float = 0.1
    max_events: int = 128
    proposals_per_event: int = 4
    minimum_similarity: float = 0.4
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048

    def _rollout_result(self, rollout: object) -> CandidateGenerationResult:
        diagnostics = rollout.diagnostics  # type: ignore[attr-defined]
        assert diagnostics is not None
        event_rules = tuple(rollout.event_rules)  # type: ignore[attr-defined]
        ring_names = (
            "ring_system_grow",
            "ring_system_delete",
            "ring_system_restate",
        )
        return CandidateGenerationResult(
            final_state=rollout.final_state,  # type: ignore[attr-defined]
            trajectory_states_audited=len(diagnostics.canonical_state_keys),
            all_trajectory_states_valid=all(diagnostics.state_valid),
            all_trajectory_states_connected=all(
                diagnostics.state_connected_or_null
            ),
            events=len(event_rules),
            exhausted_event_budget=rollout.exhausted_event_budget,  # type: ignore[attr-defined]
            stalled=len(event_rules) == 0,
            metadata={
                "control_diagnostics": rollout.control_diagnostics,  # type: ignore[attr-defined]
                "ring_rewrite_use_counts": {
                    name: event_rules.count(name) for name in ring_names
                },
            },
        )

    def generate_native(
        self,
        *,
        lead_state: MolecularGraph,
        target_qed: float,
        rng: np.random.Generator,
    ) -> CandidateGenerationResult:
        """Draw one direct candidate with no QED oracle in the sampling loop."""

        sampler = self.backbone.sampler(
            target_qed=target_qed,
            direct_conditioning=True,
        )
        rollout = sample_tracelet_ancestral(
            sampler,
            rng=rng,
            n_slots=self.n_slots,
            operational_horizon=self.operational_horizon,
            time_step=self.time_step,
            max_events=self.max_events,
            source_prior=FixedMolecularStatePrior(lead_state),
        )
        return self._rollout_result(rollout)

    def generate(
        self,
        *,
        lead_state: MolecularGraph,
        target_qed: float,
        arm: ConditionalArm,
        rng: np.random.Generator,
        oracle_scope: OracleBudgetScope,
    ) -> CandidateGenerationResult:
        base_sampler = self.backbone.sampler(
            target_qed=target_qed,
            direct_conditioning=arm.direct_conditioning,
        )
        sampler = ExactBudgetLeadGuidedSampler(
            base_sampler=base_sampler,
            oracle_scope=oracle_scope,
            lead_state=lead_state,
            target_qed=target_qed,
            minimum_similarity=self.minimum_similarity,
            proposals_per_event=self.proposals_per_event,
            beta=arm.beta,
            fingerprint_radius=self.fingerprint_radius,
            fingerprint_bits=self.fingerprint_bits,
        )
        rollout = sample_tracelet_ancestral(
            sampler,
            rng=rng,
            n_slots=self.n_slots,
            operational_horizon=self.operational_horizon,
            time_step=self.time_step,
            max_events=self.max_events,
            source_prior=FixedMolecularStatePrior(lead_state),
        )
        return self._rollout_result(rollout)


@dataclass(frozen=True)
class LeadRecord:
    lead_id: str
    state: MolecularGraph


def molecular_ring_stratum(state: MolecularGraph) -> RingStratum:
    """Assign the frozen three-way lead/candidate ring diagnostic stratum.

    ``isolated_ring`` means ring-containing without a fused ring pair or an
    RDKit bridgehead.  It therefore includes spiro systems; the label is a
    mutually exclusive reporting bucket, not a full ring-topology taxonomy.
    """

    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise ValueError("ring stratum requires a non-null molecular state")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("ring stratum conversion failed")
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    if not rings:
        return "acyclic"
    fused = any(
        len(left & right) >= 2
        for index, left in enumerate(rings)
        for right in rings[index + 1 :]
    )
    bridged = rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0
    return "fused_or_bridged" if fused or bridged else "isolated_ring"


def _ring_reference_statistics(states: Sequence[MolecularGraph]) -> dict[str, object]:
    if not states:
        raise ValueError("ring reference statistics require at least one state")
    strata = [molecular_ring_stratum(state) for state in states]
    ring_counts: list[int] = []
    fused: list[bool] = []
    bridged: list[bool] = []
    spiro: list[bool] = []
    for state in states:
        smiles = molecular_graph_to_smiles(state)
        molecule = None if smiles is None else Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("ring reference conversion failed")
        rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
        ring_counts.append(len(rings))
        fused.append(
            any(
                len(left & right) >= 2
                for index, left in enumerate(rings)
                for right in rings[index + 1 :]
            )
        )
        bridged.append(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0)
        spiro.append(rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0)
    counts = {
        name: strata.count(name)
        for name in ("acyclic", "isolated_ring", "fused_or_bridged")
    }
    return {
        "molecules": len(states),
        "stratum_counts": counts,
        "stratum_fractions": {
            name: count / len(states) for name, count in counts.items()
        },
        "mean_ring_count": float(np.mean(ring_counts)),
        "fused_fraction": float(np.mean(fused)),
        "bridged_fraction": float(np.mean(bridged)),
        "spiro_fraction": float(np.mean(spiro)),
    }


def compare_conditional_ring_references(
    unconditional_states: Sequence[MolecularGraph],
    target_pool_states: Sequence[MolecularGraph],
    *,
    target_qed_minimum: float = 0.90,
    scorer: StateScorer = qed_state_oracle,
) -> dict[str, object]:
    """Compare the unconditional corpus to the high-QED target subset."""

    if not isfinite(target_qed_minimum) or not 0.0 <= target_qed_minimum <= 1.0:
        raise ValueError("target QED threshold must lie in [0, 1]")
    target_subset = tuple(
        state
        for state in target_pool_states
        if float(scorer(state)) >= target_qed_minimum
    )
    if not target_subset:
        raise ValueError("target pool contains no molecules in the high-QED subset")
    unconditional = _ring_reference_statistics(unconditional_states)
    high_qed = _ring_reference_statistics(target_subset)
    return {
        "format": "compose_v4_qed_ring_reference_comparison_v1",
        "target_qed_minimum": float(target_qed_minimum),
        "target_pool_molecules": len(target_pool_states),
        "unconditional_corpus": unconditional,
        "high_qed_target_subset": high_qed,
        "high_qed_minus_unconditional": {
            "stratum_fraction_delta": {
                name: float(high_qed["stratum_fractions"][name])  # type: ignore[index]
                - float(unconditional["stratum_fractions"][name])  # type: ignore[index]
                for name in ("acyclic", "isolated_ring", "fused_or_bridged")
            },
            "mean_ring_count_delta": float(high_qed["mean_ring_count"])
            - float(unconditional["mean_ring_count"]),
            "fused_fraction_delta": float(high_qed["fused_fraction"])
            - float(unconditional["fused_fraction"]),
            "bridged_fraction_delta": float(high_qed["bridged_fraction"])
            - float(unconditional["bridged_fraction"]),
        },
        "interpretation_boundary": (
            "The high-QED subset is the conditional reference; the unconditional "
            "global fused frequency is reported only as a comparator."
        ),
    }


def _ring_availability_from_metadata(
    value: object,
) -> Mapping[str, bool] | None:
    if isinstance(value, Mapping):
        available = value.get("available_at_any_observation")
        if isinstance(available, Mapping):
            return {str(name): bool(flag) for name, flag in available.items()}
        for child in value.values():
            found = _ring_availability_from_metadata(child)
            if found is not None:
                return found
    return None


def _ring_sensitivity_report(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate outcome and ring-family diagnostics by lead stratum."""

    report: dict[str, object] = {}
    ring_names = (
        "ring_system_grow",
        "ring_system_delete",
        "ring_system_restate",
    )
    for stratum in ("acyclic", "isolated_ring", "fused_or_bridged"):
        selected = [row for row in rows if row.get("lead_ring_stratum") == stratum]
        candidates = [
            candidate
            for row in selected
            for candidate in row.get("candidates", ())  # type: ignore[union-attr]
            if isinstance(candidate, Mapping)
        ]
        valid = [candidate for candidate in candidates if bool(candidate.get("valid"))]
        qeds = [
            float(candidate["qed"])
            for candidate in valid
            if candidate.get("qed") is not None
        ]
        similarities = [
            float(candidate["similarity_to_lead"])
            for candidate in valid
            if candidate.get("similarity_to_lead") is not None
        ]
        lead_qed_by_row = {
            id(candidate): float(row["lead_qed"])
            for row in selected
            for candidate in row.get("candidates", ())  # type: ignore[union-attr]
            if isinstance(candidate, Mapping)
        }
        qed_deltas = [
            float(candidate["qed"]) - lead_qed_by_row[id(candidate)]
            for candidate in valid
            if candidate.get("qed") is not None
        ]
        availability_counts = {name: 0 for name in ring_names}
        use_counts = {name: 0 for name in ring_names}
        final_strata = {
            "acyclic": 0,
            "isolated_ring": 0,
            "fused_or_bridged": 0,
        }
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            available = _ring_availability_from_metadata(metadata)
            if available is not None:
                for name in ring_names:
                    availability_counts[name] += int(bool(available.get(name, False)))
            if isinstance(metadata, Mapping):
                use = metadata.get("ring_rewrite_use_counts", {})
                if isinstance(use, Mapping):
                    for name in ring_names:
                        use_counts[name] += int(use.get(name, 0))
            final_stratum = candidate.get("final_ring_stratum")
            if final_stratum in final_strata:
                final_strata[str(final_stratum)] += 1
        report[stratum] = {
            "start_seed_attempts": len(selected),
            "successful_start_seeds": sum(bool(row.get("success")) for row in selected),
            "start_seed_success_fraction": (
                None
                if not selected
                else sum(bool(row.get("success")) for row in selected) / len(selected)
            ),
            "candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "valid_candidate_fraction": (
                None if not candidates else len(valid) / len(candidates)
            ),
            "successful_candidate_count": sum(
                bool(candidate.get("success")) for candidate in candidates
            ),
            "successful_candidate_fraction": (
                None
                if not candidates
                else sum(bool(candidate.get("success")) for candidate in candidates)
                / len(candidates)
            ),
            "mean_qed": None if not qeds else float(np.mean(qeds)),
            "mean_qed_delta_from_lead": (
                None if not qed_deltas else float(np.mean(qed_deltas))
            ),
            "mean_similarity_to_lead": (
                None if not similarities else float(np.mean(similarities))
            ),
            "ring_family_available_candidate_counts": availability_counts,
            "ring_rewrite_use_counts": use_counts,
            "valid_final_candidate_ring_strata": final_strata,
        }
    return {
        "stratum_definition": {
            "acyclic": "no RDKit atom rings",
            "isolated_ring": (
                "ring-containing without fused ring pairs or RDKit bridgeheads; "
                "includes spiro systems"
            ),
            "fused_or_bridged": (
                "at least one ring pair sharing >=2 atoms or >=1 RDKit bridgehead"
            ),
        },
        "availability_scope": "available at any sampled trajectory state",
        "strata": report,
        "reference_boundary": (
            "Interpret conditional ring marginals against a high-QED target subset "
            "and the unconditional corpus; global fused frequency alone is not a target."
        ),
    }


def _candidate_diversity(
    states: Sequence[MolecularGraph],
    *,
    radius: int,
    bits: int,
) -> dict[str, float | int | None]:
    if not states:
        return {
            "valid_candidates": 0,
            "unique_candidates": 0,
            "unique_fraction": None,
            "mean_pairwise_tanimoto": None,
            "mean_pairwise_distance": None,
        }
    keys = [canonical_state_key(state) for state in states]
    fingerprints = [
        _morgan_fingerprint(state, radius=radius, bits=bits) for state in states
    ]
    similarities: list[float] = []
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities.extend(
            float(value)
            for value in DataStructs.BulkTanimotoSimilarity(
                fingerprint,
                fingerprints[index + 1 :],
            )
        )
    mean_similarity = float(np.mean(similarities)) if similarities else None
    return {
        "valid_candidates": len(states),
        "unique_candidates": len(set(keys)),
        "unique_fraction": len(set(keys)) / len(states),
        "mean_pairwise_tanimoto": mean_similarity,
        "mean_pairwise_distance": (
            None if mean_similarity is None else 1.0 - mean_similarity
        ),
    }


def _bootstrap_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap input must be a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


@dataclass
class GridDDNativeDirectEvaluator:
    """Protocol A: 20 native direct samples with benchmark-only QED calls."""

    protocol: GridDDProtocol
    generator: NativeDirectLeadCandidateGenerator
    backbone_contract: CorrectedBackboneContract
    scorer: StateScorer = qed_state_oracle
    allow_unqualified_smoke: bool = False

    def __post_init__(self) -> None:
        self.backbone_contract.require_qualified(
            allow_unqualified_smoke=self.allow_unqualified_smoke
        )
        if self.protocol.candidates_per_start != 20:
            raise ValueError("Protocol A requires exactly 20 native candidates")

    def evaluate(
        self,
        leads: Sequence[LeadRecord],
        *,
        seeds: Sequence[int],
    ) -> dict[str, object]:
        if not leads or not seeds:
            raise ValueError("at least one lead and seed are required")
        lead_qed: dict[str, float] = {}
        for lead in leads:
            value = float(self.scorer(lead.state))
            if not (
                self.protocol.starting_qed_minimum
                <= value
                <= self.protocol.starting_qed_maximum
            ):
                raise ValueError(
                    f"lead {lead.lead_id!r} QED {value:.6f} lies outside the "
                    "frozen starting interval"
                )
            lead_qed[lead.lead_id] = value
        rows = [
            self._evaluate_one(
                lead,
                lead_qed=lead_qed[lead.lead_id],
                seed=int(seed),
            )
            for lead in leads
            for seed in seeds
        ]
        evaluation_calls = len(leads) + sum(
            int(row["final_candidate_qed_calls"]) for row in rows
        )
        return {
            "format": "compose_v4_griddd_protocol_a_native_direct_v1",
            "complete": True,
            "smoke_only": bool(self.allow_unqualified_smoke),
            "protocol_role": "griddd_comparable_primary",
            "candidate_generation": "direct_conditioned_native_sampling",
            "candidates_per_start": 20,
            "oracle_tilt_or_rescoring_during_sampling": False,
            "padding_calls": 0,
            "native_sampling_oracle_calls": 0,
            "benchmark_evaluation_qed_calls": evaluation_calls,
            "lead_screening_qed_calls": len(leads),
            "final_candidate_qed_calls": sum(
                int(row["final_candidate_qed_calls"]) for row in rows
            ),
            "all_start_denominator": len(rows),
            "successful_starts": sum(bool(row["success"]) for row in rows),
            "success_fraction_all_starts": (
                sum(bool(row["success"]) for row in rows) / len(rows)
            ),
            "ring_sensitivity": _ring_sensitivity_report(rows),
            "conditional_ring_reference_requirement": {
                "compare_to_unconditional_corpus": True,
                "compare_to_high_qed_target_subset": True,
                "global_fused_frequency_is_target_marginal": False,
            },
            "records": rows,
        }

    def _evaluate_one(
        self,
        lead: LeadRecord,
        *,
        lead_qed: float,
        seed: int,
    ) -> dict[str, object]:
        started = perf_counter()
        lead_seed_code = int.from_bytes(
            hashlib.sha256(lead.lead_id.encode("utf-8")).digest()[:4],
            byteorder="little",
            signed=False,
        )
        child_seeds = np.random.SeedSequence([seed, lead_seed_code]).spawn(20)
        candidates: list[dict[str, object]] = []
        first_success: int | None = None
        best_feasible_qed = lead_qed
        for candidate_index, child_seed in enumerate(child_seeds):
            try:
                result = self.generator.generate_native(
                    lead_state=lead.state,
                    target_qed=self.protocol.target_qed,
                    rng=np.random.default_rng(child_seed),
                )
            except Exception as error:
                result = CandidateGenerationResult(
                    final_state=None,
                    trajectory_states_audited=0,
                    all_trajectory_states_valid=False,
                    all_trajectory_states_connected=False,
                    events=0,
                    exhausted_event_budget=False,
                    stalled=True,
                    error=f"{type(error).__name__}: {error}",
                )
            final = result.final_state
            valid = bool(final is not None and is_valid_state(final))
            connected = bool(final is not None and is_connected_or_null(final))
            candidate_qed: float | None = None
            similarity: float | None = None
            success = False
            if valid and connected and final is not None:
                candidate_qed = float(self.scorer(final))
                similarity = tanimoto_similarity(
                    lead.state,
                    final,
                    radius=self.protocol.fingerprint_radius,
                    bits=self.protocol.fingerprint_bits,
                )
                success = bool(
                    candidate_qed >= self.protocol.target_qed
                    and similarity >= self.protocol.minimum_tanimoto_similarity
                )
                if similarity >= self.protocol.minimum_tanimoto_similarity:
                    best_feasible_qed = max(best_feasible_qed, candidate_qed)
            if success and first_success is None:
                first_success = candidate_index + 1
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "valid": valid,
                    "connected": connected,
                    "qed": candidate_qed,
                    "similarity_to_lead": similarity,
                    "success": success,
                    "events": result.events,
                    "trajectory_states_audited": result.trajectory_states_audited,
                    "all_trajectory_states_valid": result.all_trajectory_states_valid,
                    "all_trajectory_states_connected": (
                        result.all_trajectory_states_connected
                    ),
                    "exhausted_event_budget": result.exhausted_event_budget,
                    "stalled": result.stalled,
                    "error": result.error,
                    "metadata": dict(result.metadata),
                    "final_ring_stratum": (
                        molecular_ring_stratum(final)
                        if valid and connected and final is not None
                        else None
                    ),
                }
            )
        return {
            "lead_id": lead.lead_id,
            "seed": seed,
            "arm": "direct",
            "lead_qed": lead_qed,
            "lead_ring_stratum": molecular_ring_stratum(lead.state),
            "success": first_success is not None,
            "first_success_candidate": first_success,
            "best_feasible_qed": best_feasible_qed,
            "best_feasible_qed_improvement": best_feasible_qed - lead_qed,
            "candidate_attempts": len(candidates),
            "valid_candidates": sum(bool(item["valid"]) for item in candidates),
            "final_candidate_qed_calls": sum(
                item["qed"] is not None for item in candidates
            ),
            "native_sampling_cost": {
                "wall_time_seconds": perf_counter() - started,
                "trajectory_states_audited": sum(
                    int(item["trajectory_states_audited"]) for item in candidates
                ),
                "events": sum(int(item["events"]) for item in candidates),
            },
            "candidates": candidates,
        }


@dataclass
class GridDDConditionalEvaluator:
    protocol: GridDDProtocol
    generator: LeadCandidateGenerator
    backbone_contract: CorrectedBackboneContract
    scorer: StateScorer = qed_state_oracle
    allow_unqualified_smoke: bool = False

    def __post_init__(self) -> None:
        self.backbone_contract.require_qualified(
            allow_unqualified_smoke=self.allow_unqualified_smoke
        )
        if isinstance(self.generator, RewriteLeadCandidateGenerator):
            declared = (
                self.generator.proposals_per_event,
                self.generator.minimum_similarity,
                self.generator.fingerprint_radius,
                self.generator.fingerprint_bits,
            )
            frozen = (
                self.protocol.proposals_per_controlled_event,
                self.protocol.minimum_tanimoto_similarity,
                self.protocol.fingerprint_radius,
                self.protocol.fingerprint_bits,
            )
            if declared != frozen:
                raise ValueError(
                    "rewrite candidate generator does not match the frozen protocol"
                )

    def evaluate(
        self,
        leads: Sequence[LeadRecord],
        *,
        seeds: Sequence[int],
        arms: Sequence[ConditionalArm] | None = None,
    ) -> dict[str, object]:
        if not leads or not seeds:
            raise ValueError("at least one lead and seed are required")
        selected_arms = tuple(arms or standard_conditional_arms())
        if len(selected_arms) != 3 or {arm.name for arm in selected_arms} != {
            "direct",
            "controller",
            "combined",
        }:
            raise ValueError("evaluation requires direct, controller, and combined arms")
        lead_qed: dict[str, float] = {}
        for lead in leads:
            value = float(self.scorer(lead.state))
            if not (
                self.protocol.starting_qed_minimum
                <= value
                <= self.protocol.starting_qed_maximum
            ):
                raise ValueError(
                    f"lead {lead.lead_id!r} QED {value:.6f} lies outside the "
                    "frozen starting interval"
                )
            lead_qed[lead.lead_id] = value

        arm_rows: dict[str, list[dict[str, object]]] = {
            arm.name: [] for arm in selected_arms
        }
        for arm in selected_arms:
            for lead in leads:
                for seed in seeds:
                    arm_rows[arm.name].append(
                        self._evaluate_one(
                            lead,
                            lead_qed=lead_qed[lead.lead_id],
                            seed=int(seed),
                            arm=arm,
                        )
                    )
        arm_reports = {
            arm.name: {
                **self._aggregate_arm(arm_rows[arm.name]),
                "ring_sensitivity": _ring_sensitivity_report(
                    arm_rows[arm.name]
                ),
            }
            for arm in selected_arms
        }
        comparisons = self._paired_comparisons(arm_rows)
        return {
            "format": "compose_v4_griddd_conditional_evaluation_v1",
            "complete": True,
            "smoke_only": bool(self.allow_unqualified_smoke),
            "training_launched": False,
            "protocol": self.protocol.to_dict(),
            "backbone": self.backbone_contract.to_dict(),
            "pre_run_lead_screening": {
                "oracle_calls": len(leads),
                "shared_across_arms": True,
                "excluded_from_matched_per_arm_budget": True,
            },
            "seed_stream_identical_across_arms": True,
            "arms": arm_reports,
            "paired_arm_comparisons": comparisons,
            "all_attempt_denominator": len(leads) * len(seeds),
            "records": arm_rows,
            "large_benchmark_authorized": False,
            "conditional_ring_reference_requirement": {
                "compare_to_unconditional_corpus": True,
                "compare_to_high_qed_target_subset": True,
                "global_fused_frequency_is_target_marginal": False,
            },
        }

    def _evaluate_one(
        self,
        lead: LeadRecord,
        *,
        lead_qed: float,
        seed: int,
        arm: ConditionalArm,
    ) -> dict[str, object]:
        started = perf_counter()
        ledger = ExactOracleBudgetLedger(
            self.scorer,
            budget=self.protocol.oracle_calls_per_start_seed_arm,
        )
        measured_lead_qed = ledger.score(
            lead.state,
            purpose="lead_eligibility",
            influences_selection=False,
        )
        if not np.isclose(measured_lead_qed, lead_qed, rtol=0.0, atol=1e-12):
            raise RuntimeError("deterministic lead oracle changed between calls")
        lead_seed_code = int.from_bytes(
            hashlib.sha256(lead.lead_id.encode("utf-8")).digest()[:4],
            byteorder="little",
            signed=False,
        )
        # Every arm starts from the identical candidate-level random streams.
        # Arm semantics may make those streams diverge after a different mark
        # is selected, but arm identity itself never perturbs seed derivation.
        sequence = np.random.SeedSequence([seed, lead_seed_code])
        child_seeds = sequence.spawn(self.protocol.candidates_per_start)
        candidate_rows: list[dict[str, object]] = []
        valid_states: list[MolecularGraph] = []
        best_feasible_qed = lead_qed
        first_success_candidate: int | None = None
        anytime: list[dict[str, object]] = []
        all_states_valid = True
        all_states_connected = True
        for candidate_index, child_seed in enumerate(child_seeds):
            scope = ledger.scope(
                self.protocol.guidance_oracle_calls_per_candidate,
                label=f"candidate_{candidate_index}:guidance",
            )
            result: CandidateGenerationResult
            try:
                result = self.generator.generate(
                    lead_state=lead.state,
                    target_qed=self.protocol.target_qed,
                    arm=arm,
                    rng=np.random.default_rng(child_seed),
                    oracle_scope=scope,
                )
            except Exception as error:  # all failures remain in the denominator
                result = CandidateGenerationResult(
                    final_state=None,
                    trajectory_states_audited=0,
                    all_trajectory_states_valid=False,
                    all_trajectory_states_connected=False,
                    events=0,
                    exhausted_event_budget=False,
                    stalled=True,
                    error=f"{type(error).__name__}: {error}",
                )
            padding_state = (
                result.final_state
                if result.final_state is not None
                and is_valid_state(result.final_state)
                and is_connected_or_null(result.final_state)
                else lead.state
            )
            scope.fill(padding_state)
            scope.assert_exact()
            final_state = result.final_state
            final_valid = bool(final_state is not None and is_valid_state(final_state))
            final_connected = bool(
                final_state is not None and is_connected_or_null(final_state)
            )
            candidate_qed: float | None = None
            similarity: float | None = None
            success = False
            if final_valid and final_connected and final_state is not None:
                candidate_qed = ledger.score(
                    final_state,
                    purpose=f"candidate_{candidate_index}:final_qed",
                    influences_selection=False,
                )
                similarity = tanimoto_similarity(
                    lead.state,
                    final_state,
                    radius=self.protocol.fingerprint_radius,
                    bits=self.protocol.fingerprint_bits,
                )
                success = bool(
                    candidate_qed >= self.protocol.target_qed
                    and similarity >= self.protocol.minimum_tanimoto_similarity
                )
                valid_states.append(final_state)
                if similarity >= self.protocol.minimum_tanimoto_similarity:
                    best_feasible_qed = max(best_feasible_qed, candidate_qed)
            else:
                ledger.score(
                    lead.state,
                    purpose=f"candidate_{candidate_index}:invalid_final_padding",
                    influences_selection=False,
                    padding=True,
                )
            if success and first_success_candidate is None:
                first_success_candidate = candidate_index + 1
            all_states_valid = all_states_valid and bool(
                result.all_trajectory_states_valid
            )
            all_states_connected = all_states_connected and bool(
                result.all_trajectory_states_connected
            )
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "valid": final_valid,
                    "connected": final_connected,
                    "qed": candidate_qed,
                    "similarity_to_lead": similarity,
                    "success": success,
                    "events": result.events,
                    "trajectory_states_audited": result.trajectory_states_audited,
                    "all_trajectory_states_valid": (
                        result.all_trajectory_states_valid
                    ),
                    "all_trajectory_states_connected": (
                        result.all_trajectory_states_connected
                    ),
                    "exhausted_event_budget": result.exhausted_event_budget,
                    "stalled": result.stalled,
                    "error": result.error,
                    "guidance_oracle_calls": scope.used,
                    "metadata": dict(result.metadata),
                    "final_ring_stratum": (
                        molecular_ring_stratum(final_state)
                        if final_valid and final_connected and final_state is not None
                        else None
                    ),
                }
            )
            anytime.append(
                {
                    "candidates_attempted": candidate_index + 1,
                    "oracle_calls_used": ledger.used,
                    "success_observed": first_success_candidate is not None,
                    "best_feasible_qed": best_feasible_qed,
                    "wall_time_seconds": perf_counter() - started,
                }
            )
        ledger.assert_exact()
        success = first_success_candidate is not None
        return {
            "lead_id": lead.lead_id,
            "seed": seed,
            "arm": arm.name,
            "lead_qed": lead_qed,
            "lead_ring_stratum": molecular_ring_stratum(lead.state),
            "success": success,
            "first_success_candidate": first_success_candidate,
            "best_feasible_qed": best_feasible_qed,
            "best_feasible_qed_improvement": best_feasible_qed - lead_qed,
            "valid_candidates": sum(row["valid"] for row in candidate_rows),
            "connected_candidates": sum(row["connected"] for row in candidate_rows),
            "candidate_attempts": len(candidate_rows),
            "all_trajectory_states_valid": all_states_valid,
            "all_trajectory_states_connected": all_states_connected,
            "event_budget_exhaustions": sum(
                row["exhausted_event_budget"] for row in candidate_rows
            ),
            "stalled_candidates": sum(row["stalled"] for row in candidate_rows),
            "failed_candidates": sum(row["error"] is not None for row in candidate_rows),
            "diversity": _candidate_diversity(
                valid_states,
                radius=self.protocol.fingerprint_radius,
                bits=self.protocol.fingerprint_bits,
            ),
            "oracle": ledger.summary(),
            "wall_time_seconds": perf_counter() - started,
            "anytime": anytime,
            "candidates": candidate_rows,
        }

    def _aggregate_arm(self, rows: list[dict[str, object]]) -> dict[str, object]:
        denominator = len(rows)
        successes = sum(bool(row["success"]) for row in rows)
        return {
            "start_seed_attempts": denominator,
            "successful_start_seeds": successes,
            "success_fraction_all_attempts": successes / denominator,
            "candidate_attempts": sum(int(row["candidate_attempts"]) for row in rows),
            "valid_candidates": sum(int(row["valid_candidates"]) for row in rows),
            "connected_candidates": sum(
                int(row["connected_candidates"]) for row in rows
            ),
            "all_trajectory_states_valid": all(
                bool(row["all_trajectory_states_valid"]) for row in rows
            ),
            "all_trajectory_states_connected": all(
                bool(row["all_trajectory_states_connected"]) for row in rows
            ),
            "exact_oracle_budget_every_start_seed": all(
                bool(row["oracle"]["exact"]) for row in rows  # type: ignore[index]
            ),
            "total_oracle_calls": sum(
                int(row["oracle"]["used"]) for row in rows  # type: ignore[index]
            ),
            "padding_oracle_calls": sum(
                int(row["oracle"]["padding_calls"]) for row in rows  # type: ignore[index]
            ),
            "mean_best_feasible_qed_improvement": float(
                np.mean([float(row["best_feasible_qed_improvement"]) for row in rows])
            ),
            "mean_wall_time_seconds_per_start_seed": float(
                np.mean([float(row["wall_time_seconds"]) for row in rows])
            ),
            "anytime_success_fraction_by_candidate": [
                sum(
                    bool(row["anytime"][index]["success_observed"])  # type: ignore[index]
                    for row in rows
                )
                / denominator
                for index in range(self.protocol.candidates_per_start)
            ],
        }

    def _paired_comparisons(
        self,
        rows: dict[str, list[dict[str, object]]],
    ) -> dict[str, object]:
        keyed = {
            arm: {(str(row["lead_id"]), int(row["seed"])): row for row in arm_rows}
            for arm, arm_rows in rows.items()
        }
        comparisons: dict[str, object] = {}
        for left, right in (
            ("combined", "direct"),
            ("combined", "controller"),
            ("controller", "direct"),
        ):
            keys = sorted(set(keyed[left]) & set(keyed[right]))
            success_delta = np.asarray(
                [
                    float(bool(keyed[left][key]["success"]))
                    - float(bool(keyed[right][key]["success"]))
                    for key in keys
                ],
                dtype=np.float64,
            )
            improvement_delta = np.asarray(
                [
                    float(keyed[left][key]["best_feasible_qed_improvement"])
                    - float(keyed[right][key]["best_feasible_qed_improvement"])
                    for key in keys
                ],
                dtype=np.float64,
            )
            comparisons[f"{left}_minus_{right}"] = {
                "paired_attempts": len(keys),
                "success_fraction_delta": float(success_delta.mean()),
                "success_fraction_delta_bootstrap_95_percent_interval": (
                    _bootstrap_interval(
                        success_delta,
                        replicates=self.protocol.bootstrap_replicates,
                        seed=self.protocol.bootstrap_seed,
                    )
                ),
                "mean_best_feasible_qed_improvement_delta": float(
                    improvement_delta.mean()
                ),
                "improvement_delta_bootstrap_95_percent_interval": (
                    _bootstrap_interval(
                        improvement_delta,
                        replicates=self.protocol.bootstrap_replicates,
                        seed=self.protocol.bootstrap_seed,
                    )
                ),
            }
        return comparisons


__all__ = [
    "ANALYTIC_BACKBONE_QUALIFICATION_FORMAT",
    "BACKBONE_QUALIFICATION_FORMAT",
    "CandidateGenerationResult",
    "ConditionalArm",
    "CorrectedBackboneContract",
    "ExactBudgetLeadGuidedSampler",
    "ExactOracleBudgetLedger",
    "FactorizedConditionalBackbone",
    "FrozenResidualConditionalBackbone",
    "GridDDBenchmarkFairnessContract",
    "GridDDConditionalEvaluator",
    "GridDDNativeDirectEvaluator",
    "GridDDProtocol",
    "LeadCandidateGenerator",
    "NativeDirectLeadCandidateGenerator",
    "LeadRecord",
    "OracleBudgetScope",
    "RETAINED_PANCAKE_CHECKPOINT_SHA256",
    "RewriteLeadCandidateGenerator",
    "UnconditionalBackboneQualification",
    "compare_conditional_ring_references",
    "file_sha256",
    "load_unconditional_backbone_qualification",
    "molecular_ring_stratum",
    "qed_state_oracle",
    "standard_conditional_arms",
    "tanimoto_similarity",
]
