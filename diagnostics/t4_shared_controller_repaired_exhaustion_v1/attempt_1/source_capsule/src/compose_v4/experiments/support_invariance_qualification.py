"""Fail-closed bounded qualification of the scored production support path.

The production enumerator obtains chemistry-derived candidate masks while also
computing model probabilities.  The state-centric compiler intentionally
retains only candidate identities and executor groupings.  This module checks,
on an explicitly bounded deterministic panel, that those support objects do
not drift across scoring times or independently initialized parameter sets.

Passing this gate is not a corpus-wide proof and does not turn the current
scored enumerator into a support-only implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.factorized_successor_training import (
    CompiledStateSuccessorMap,
    SuccessorTrainingError,
    TeacherSuccessorAlias,
    compile_state_successor_map,
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    enumerate_factorized_marked_law,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ACTIVE_RINGCORE_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_delete",
    "ring_system_restate",
)


class SupportInvarianceQualificationError(RuntimeError):
    """The bounded support qualification is invalid or found support drift."""


@dataclass(frozen=True)
class QualifiedModel:
    """One independently initialized model participating in the gate."""

    name: str
    model: FactorizedTraceletRateModel

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("qualified model name must be nonempty")


@dataclass(frozen=True)
class SupportPanelState:
    """One deterministic exact state and its human-readable stratum."""

    name: str
    state: MolecularGraph

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("support-panel state name must be nonempty")


@dataclass(frozen=True, order=True)
class CandidateSupportIdentity:
    """Score-free identity and exact grouping of one enumerated mark."""

    family_name: str
    table_name: str
    coordinate: tuple[int, ...]
    action_sha256: str
    target_key: str
    successor_state_sha256: str
    is_virtual: bool


@dataclass(frozen=True)
class SupportStateSummary:
    name: str
    source_key: str
    source_state_sha256: str
    support_signature_sha256: str
    raw_mark_count: int
    virtual_mark_count: int
    canonical_successor_count: int
    marks_by_family: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ProbabilityComparison:
    """A score comparison aligned on an already-proven support signature."""

    state_name: str
    model_name: str
    time: float
    l1_from_reference: float
    max_abs_from_reference: float


@dataclass(frozen=True)
class SupportInvarianceQualification:
    """Machine-readable passing evidence for one bounded panel."""

    model_parameter_sha256s: tuple[tuple[str, str], ...]
    scoring_times: tuple[float, ...]
    required_families: tuple[str, ...]
    covered_families: tuple[str, ...]
    states: tuple[SupportStateSummary, ...]
    probability_comparisons: tuple[ProbabilityComparison, ...]
    maximum_probability_l1: float
    support_invariant: bool
    probability_variation_observed: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "compose.support_invariance_qualification",
            "schema_version": 1,
            "status": "pass",
            "scope": "deterministic_stratified_development_panel",
            "corpus_wide_proof": False,
            "support_path": (
                "production scored marked-law enumerator -> executor -> "
                "canonical/exact successor grouping"
            ),
            "support_signature_excludes": [
                "mark_probability",
                "family_probability",
                "total_hazard",
                "logits",
                "scoring_time",
                "model_parameters",
            ],
            "action_identity": (
                "full SHA-256 of canonical RewriteActionCodecV2 JSON"
            ),
            "model_parameter_sha256s": dict(self.model_parameter_sha256s),
            "scoring_times": list(self.scoring_times),
            "required_families": list(self.required_families),
            "covered_families": list(self.covered_families),
            "states": [asdict(state) for state in self.states],
            "probability_comparisons": [
                asdict(comparison)
                for comparison in self.probability_comparisons
            ],
            "maximum_probability_l1": self.maximum_probability_l1,
            "support_invariant": self.support_invariant,
            "probability_variation_observed": (
                self.probability_variation_observed
            ),
            "limitations": [
                (
                    "This is a deterministic stratified development panel, "
                    "not a corpus-wide support proof."
                ),
                (
                    "Unseen molecular states, slot capacities, catalogs, and "
                    "future capability configurations are outside this result."
                ),
                (
                    "The production enumerator remains operationally "
                    "score-coupled; this gate detects support drift but does "
                    "not remove its neural forward pass."
                ),
                (
                    "SHA-256 collision resistance is assumed; the digest is "
                    "not a mathematical proof of action equality."
                ),
            ],
        }


def _model_parameter_sha256(model: FactorizedTraceletRateModel) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(tuple(int(item) for item in tensor.shape)).encode("ascii")
        )
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _candidate_support(
    compiled: CompiledStateSuccessorMap,
) -> tuple[CandidateSupportIdentity, ...]:
    candidates = []
    for group in compiled.successor_groups:
        for mark in group.marks:
            candidates.append(
                CandidateSupportIdentity(
                    family_name=mark.alias.family_name,
                    table_name=mark.alias.table_name,
                    coordinate=mark.alias.coordinate,
                    action_sha256=mark.action_sha256,
                    target_key=group.target_key,
                    successor_state_sha256=mark.successor_state_sha256,
                    is_virtual=False,
                )
            )
    for mark in compiled.virtual_marks:
        candidates.append(
            CandidateSupportIdentity(
                family_name=mark.alias.family_name,
                table_name=mark.alias.table_name,
                coordinate=mark.alias.coordinate,
                action_sha256=mark.action_sha256,
                target_key=compiled.source_key,
                successor_state_sha256=mark.successor_state_sha256,
                is_virtual=True,
            )
        )
    ordered = tuple(sorted(candidates))
    if len(ordered) != len(set(ordered)):
        raise SupportInvarianceQualificationError(
            "compiled support contains duplicate candidate identities"
        )
    return ordered


def _support_signature_sha256(
    compiled: CompiledStateSuccessorMap,
    support: tuple[CandidateSupportIdentity, ...],
) -> str:
    payload = {
        "source_key": compiled.source_key,
        "source_state_sha256": compiled.source_state_sha256,
        "candidates": [asdict(candidate) for candidate in support],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probability_vector(
    model: FactorizedTraceletRateModel,
    panel_state: SupportPanelState,
    *,
    time: float,
    compiled_support: tuple[CandidateSupportIdentity, ...],
) -> tuple[float, ...]:
    """Align scored probabilities to a separate score-free support signature."""

    try:
        law = enumerate_factorized_marked_law(
            model,
            panel_state.state,
            time,
        )
    except ProductionSuccessorKernelError as error:
        raise SupportInvarianceQualificationError(
            "scored marked-law enumeration failed during qualification"
        ) from error

    by_alias = {
        TeacherSuccessorAlias(
            family_name=candidate.family_name,
            table_name=candidate.table_name,
            coordinate=candidate.coordinate,
        ): candidate
        for candidate in compiled_support
    }
    if len(by_alias) != len(compiled_support):
        raise SupportInvarianceQualificationError(
            "score-free support repeats a factorized alias coordinate"
        )

    scored: dict[CandidateSupportIdentity, float] = {}
    for mark in law.marks:
        alias = TeacherSuccessorAlias(
            family_name=mark.family_name,
            table_name=mark.table_name,
            coordinate=mark.coordinate,
        )
        candidate = by_alias.get(alias)
        if candidate is None:
            raise SupportInvarianceQualificationError(
                "scored production law and compiled candidate support disagree"
            )
        if rewrite_action_codec_sha256(
            mark.executor_rule_name,
            mark.action,
        ) != candidate.action_sha256:
            raise SupportInvarianceQualificationError(
                "production action identity drifted between support enumerations"
            )
        if candidate in scored:
            raise SupportInvarianceQualificationError(
                "scored production law repeats a candidate identity"
            )
        probability = mark.probability
        if not isfinite(probability) or probability < 0.0:
            raise SupportInvarianceQualificationError(
                "scored production law has an invalid candidate probability"
            )
        scored[candidate] = probability
    if set(scored) != set(compiled_support):
        raise SupportInvarianceQualificationError(
            "scored production law and compiled candidate support disagree"
        )
    vector = tuple(scored[candidate] for candidate in compiled_support)
    if vector and abs(sum(vector) - 1.0) > 1e-6:
        raise SupportInvarianceQualificationError(
            "scored production law is not normalized on qualified support"
        )
    return vector


def _validate_qualification_inputs(
    models: tuple[QualifiedModel, ...],
    panel: tuple[SupportPanelState, ...],
    scoring_times: tuple[float, ...],
    required_families: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if len(models) < 2 or len({id(item.model) for item in models}) < 2:
        raise SupportInvarianceQualificationError(
            "qualification requires at least two distinct model objects"
        )
    if len({item.name for item in models}) != len(models):
        raise SupportInvarianceQualificationError(
            "qualified model names must be unique"
        )
    if any(item.model.training for item in models):
        raise SupportInvarianceQualificationError(
            "qualification models must be in deterministic eval mode"
        )
    capabilities = tuple(item.model.operator_capabilities for item in models)
    if len(set(capabilities)) != 1:
        raise SupportInvarianceQualificationError(
            "qualification models have different operator capabilities"
        )
    if any(
        item.model.enable_ring_grow_macro or not item.model.enable_cycle_ops
        for item in models
    ):
        raise SupportInvarianceQualificationError(
            "qualification requires the RingCore-V1 production capability contract"
        )

    parameter_hashes = tuple(
        (item.name, _model_parameter_sha256(item.model))
        for item in models
    )
    if len({digest for _, digest in parameter_hashes}) < 2:
        raise SupportInvarianceQualificationError(
            "qualification models do not have distinct parameter initializations"
        )
    if not panel or len({item.name for item in panel}) != len(panel):
        raise SupportInvarianceQualificationError(
            "support panel must be nonempty with unique stratum names"
        )
    state_digests = tuple(
        persistent_slot_state_sha256(item.state)
        for item in panel
    )
    if len(set(state_digests)) != len(state_digests):
        raise SupportInvarianceQualificationError(
            "support panel repeats an exact persistent-slot state"
        )
    if (
        len(scoring_times) < 2
        or tuple(sorted(set(scoring_times))) != scoring_times
        or any(not 0.0 < time < 1.0 for time in scoring_times)
    ):
        raise SupportInvarianceQualificationError(
            "scoring times must be at least two sorted unique values in (0,1)"
        )
    if (
        not required_families
        or tuple(sorted(set(required_families))) != required_families
        or not set(required_families).issubset(ACTIVE_RINGCORE_FAMILIES)
    ):
        raise SupportInvarianceQualificationError(
            "required families must be sorted unique active RingCore families"
        )
    return parameter_hashes


def qualify_stratified_support_invariance(
    models: Iterable[QualifiedModel],
    panel: Iterable[SupportPanelState],
    *,
    scoring_times: tuple[float, ...] = (0.13, 0.79),
    required_families: tuple[str, ...] = tuple(
        sorted(ACTIVE_RINGCORE_FAMILIES)
    ),
    probability_variation_tolerance: float = 1e-8,
    system: RewriteSystem | None = None,
) -> SupportInvarianceQualification:
    """Run the bounded support/action/grouping invariance gate."""

    resolved_models = tuple(models)
    resolved_panel = tuple(panel)
    parameter_hashes = _validate_qualification_inputs(
        resolved_models,
        resolved_panel,
        scoring_times,
        required_families,
    )
    if (
        isinstance(probability_variation_tolerance, bool)
        or not isinstance(probability_variation_tolerance, (int, float))
        or not isfinite(float(probability_variation_tolerance))
        or probability_variation_tolerance < 0.0
    ):
        raise ValueError(
            "probability variation tolerance must be finite and nonnegative"
        )
    runtime = system or de_novo_rewrite_system()

    reference_support: dict[str, tuple[CandidateSupportIdentity, ...]] = {}
    reference_probabilities: dict[str, tuple[float, ...]] = {}
    state_summaries: dict[str, SupportStateSummary] = {}
    probability_comparisons = []
    family_coverage: Counter[str] = Counter()

    for model_index, qualified_model in enumerate(resolved_models):
        for time_index, time in enumerate(scoring_times):
            for panel_state in resolved_panel:
                try:
                    compiled = compile_state_successor_map(
                        qualified_model.model,
                        panel_state.state,
                        time=time,
                        system=runtime,
                    )
                except SuccessorTrainingError as error:
                    raise SupportInvarianceQualificationError(
                        "state-centric support compilation failed for "
                        f"{qualified_model.name}/{time}/{panel_state.name}"
                    ) from error
                support = _candidate_support(compiled)
                probabilities = _probability_vector(
                    qualified_model.model,
                    panel_state,
                    time=time,
                    compiled_support=support,
                )

                is_reference = model_index == 0 and time_index == 0
                if is_reference:
                    reference_support[panel_state.name] = support
                    reference_probabilities[panel_state.name] = probabilities
                    counts = Counter(
                        candidate.family_name for candidate in support
                    )
                    family_coverage.update(counts)
                    state_summaries[panel_state.name] = SupportStateSummary(
                        name=panel_state.name,
                        source_key=compiled.source_key,
                        source_state_sha256=compiled.source_state_sha256,
                        support_signature_sha256=_support_signature_sha256(
                            compiled,
                            support,
                        ),
                        raw_mark_count=len(support),
                        virtual_mark_count=sum(
                            int(candidate.is_virtual)
                            for candidate in support
                        ),
                        canonical_successor_count=len(
                            compiled.successor_groups
                        ),
                        marks_by_family=tuple(sorted(counts.items())),
                    )
                    continue

                reference = reference_support[panel_state.name]
                if support != reference:
                    raise SupportInvarianceQualificationError(
                        "candidate/action/exact-successor support drift for "
                        f"{qualified_model.name}/{time}/{panel_state.name}"
                    )
                reference_scores = reference_probabilities[panel_state.name]
                deltas = tuple(
                    abs(observed - expected)
                    for observed, expected in zip(
                        probabilities,
                        reference_scores,
                        strict=True,
                    )
                )
                probability_comparisons.append(
                    ProbabilityComparison(
                        state_name=panel_state.name,
                        model_name=qualified_model.name,
                        time=float(time),
                        l1_from_reference=float(sum(deltas)),
                        max_abs_from_reference=float(max(deltas, default=0.0)),
                    )
                )

    covered_families = tuple(sorted(family_coverage))
    missing_families = sorted(set(required_families) - set(covered_families))
    if missing_families:
        raise SupportInvarianceQualificationError(
            "stratified panel does not cover required active families: "
            + ", ".join(missing_families)
        )
    maximum_probability_l1 = max(
        (
            comparison.l1_from_reference
            for comparison in probability_comparisons
        ),
        default=0.0,
    )
    probability_variation_observed = (
        maximum_probability_l1 > probability_variation_tolerance
    )
    if not probability_variation_observed:
        raise SupportInvarianceQualificationError(
            "qualification did not observe probability variation across "
            "distinct scores; support and probabilities were not empirically separated"
        )

    return SupportInvarianceQualification(
        model_parameter_sha256s=parameter_hashes,
        scoring_times=scoring_times,
        required_families=required_families,
        covered_families=covered_families,
        states=tuple(
            state_summaries[item.name]
            for item in resolved_panel
        ),
        probability_comparisons=tuple(probability_comparisons),
        maximum_probability_l1=maximum_probability_l1,
        support_invariant=True,
        probability_variation_observed=True,
    )


def build_default_ringcore_support_fixture(
    *,
    model_seeds: tuple[int, int] = (101, 202),
    n_slots: int = 8,
) -> tuple[tuple[QualifiedModel, ...], tuple[SupportPanelState, ...]]:
    """Build the deterministic small development fixture used by the gate."""

    def state(smiles: str) -> MolecularGraph:
        return pad_molecular_graph(
            smiles_to_molecular_graph(smiles),
            n_slots,
        )

    ring = state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(
        sizes=(ring.n_real_atoms,)
    ).sample(np.random.default_rng(314159), n_slots=n_slots)
    trace = compile_carbon_tree_to_target(
        source,
        ring,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))

    qualified_models = []
    for seed in model_seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = FactorizedTraceletRateModel(
                catalog,
                hidden_dim=8,
                message_passing_steps=1,
                enable_ring_restates=True,
                enable_cyclic_graft=True,
                enable_heteroatom_scan=True,
                enable_ring_opening=True,
                enable_cycle_ops=True,
                enable_ring_grow_macro=False,
                atom_vocabulary=ORGANIC_VOCABULARY,
            ).eval()
        qualified_models.append(
            QualifiedModel(name=f"independent_seed_{seed}", model=model)
        )

    panel = (
        SupportPanelState(
            name="null_root_birth",
            state=empty_molecular_graph(n_slots),
        ),
        SupportPanelState(
            name="acyclic_general_edit",
            state=state("CCO"),
        ),
        SupportPanelState(
            name="ring_topology_and_electronics",
            state=ring,
        ),
    )
    return tuple(qualified_models), panel


__all__ = [
    "ACTIVE_RINGCORE_FAMILIES",
    "CandidateSupportIdentity",
    "ProbabilityComparison",
    "QualifiedModel",
    "SupportInvarianceQualification",
    "SupportInvarianceQualificationError",
    "SupportPanelState",
    "SupportStateSummary",
    "build_default_ringcore_support_fixture",
    "qualify_stratified_support_invariance",
]
