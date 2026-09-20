"""Bounded exact enumerability probe for the legacy ring-growth macro.

This module is intentionally diagnostic infrastructure, not a production
successor evaluator.  It recovers the complete conditional law

``p(RingSystemGrow action | ring_system_grow family, state, time)``

from the existing macro-only factorization:

``template -> placement -> electronic labels``.

Every branch is enumerated exactly under explicit resource bounds.  Equivalent
internal encodings are summed into complete :class:`RingSystemGrow` actions,
and the resulting law is checked against the model's resonance-invariant
teacher scorer.  The probe never substitutes template probability for a
complete-action probability and never enables the guarded hybrid regime.
"""

from __future__ import annotations

import math
import time as wall_time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.ring_system_fiber import (
    SemanticRingSystemDecoder,
    instantiate_semantic_ring_system_grow,
    ring_system_grow_electronic_key,
    semantic_ring_next_category_mask,
)
from compose_v4.rewrite.tracelets import (
    RingSystemGrow,
    is_valid_ring_system_grow,
    lower_ring_system_grow,
)

_SUPPORTED_ELECTRONIC_MODES = frozenset(
    {"catalog_exact", "factorized_local", "factorized_contextual"}
)


class RingMacroEnumerabilityError(RuntimeError):
    """Exact bounded recovery of the macro action law was unavailable."""


@dataclass(frozen=True)
class RingMacroEnumerationLimits:
    """Fail-closed resource limits for one exact state-level probe."""

    max_legal_templates: int = 512
    max_raw_placements: int = 32_768
    max_supported_placements: int = 16_384
    max_prefix_nodes: int = 1_000_000
    max_complete_encodings: int = 250_000
    max_distinct_actions: int = 250_000
    max_wall_seconds: float = 60.0
    normalization_tolerance: float = 2e-5
    scorer_log_tolerance: float = 3e-5

    def validate(self) -> None:
        integer_limits = {
            "max_legal_templates": self.max_legal_templates,
            "max_raw_placements": self.max_raw_placements,
            "max_supported_placements": self.max_supported_placements,
            "max_prefix_nodes": self.max_prefix_nodes,
            "max_complete_encodings": self.max_complete_encodings,
            "max_distinct_actions": self.max_distinct_actions,
        }
        for name, value in integer_limits.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        real_limits = {
            "max_wall_seconds": self.max_wall_seconds,
            "normalization_tolerance": self.normalization_tolerance,
            "scorer_log_tolerance": self.scorer_log_tolerance,
        }
        for name, value in real_limits.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, order=True)
class RingMacroEncodingAddress:
    """One complete path and its raw mass before action aggregation."""

    template_index: int
    conditional_log_probability: float
    placement_index: int | None = None
    electronic_categories: tuple[int, ...] | None = None
    catalog_candidate_index: int | None = None


@dataclass(frozen=True)
class EnumeratedRingMacroAction:
    """One complete executable action after encoding-alias aggregation."""

    action: RingSystemGrow
    conditional_log_probability: float
    conditional_probability: float
    encoding_count: int
    encoding_addresses: tuple[RingMacroEncodingAddress, ...]
    successor_key: str
    primitive_lowering_length: int


@dataclass(frozen=True)
class RingMacroEnumerabilityResult:
    """Certificate for one exact within-family ring-macro law."""

    source_key: str
    time: float
    electronic_mode: str
    normalization_applicable: bool
    legal_template_count: int
    raw_placement_count: int
    supported_placement_count: int
    prefix_node_count: int
    complete_encoding_count: int
    distinct_complete_action_count: int
    resonance_invariant_action_count: int
    duplicate_encoding_count: int
    encoding_total_probability: float
    action_total_probability: float
    max_teacher_scorer_log_error: float
    elapsed_seconds: float
    actions: tuple[EnumeratedRingMacroAction, ...]


@dataclass
class _MutableActionMass:
    log_probability: float
    addresses: list[RingMacroEncodingAddress]


def _logaddexp(left: float, right: float) -> float:
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    high = max(left, right)
    low = min(left, right)
    return high + math.log1p(math.exp(low - high))


def _logsumexp(values: Iterable[float]) -> float:
    result = float("-inf")
    for value in values:
        result = _logaddexp(result, float(value))
    return result


def _assert_probability_normalized(
    log_probabilities: Iterable[float],
    *,
    tolerance: float,
    context: str,
) -> float:
    values = tuple(float(value) for value in log_probabilities)
    if not values:
        raise RingMacroEnumerabilityError(
            f"{context} advertised support but has no probability entries"
        )
    if any(not math.isfinite(value) for value in values):
        raise RingMacroEnumerabilityError(f"{context} contains a non-finite log probability")
    total = math.exp(_logsumexp(values))
    if not math.isfinite(total) or abs(total - 1.0) > tolerance:
        raise RingMacroEnumerabilityError(
            f"{context} is not normalized: total_probability={total:.9g}"
        )
    return total


def _check_elapsed(
    started_at: float,
    limits: RingMacroEnumerationLimits,
    *,
    context: str,
) -> None:
    elapsed = wall_time.monotonic() - started_at
    if elapsed > limits.max_wall_seconds:
        raise RingMacroEnumerabilityError(
            f"exact macro enumeration exceeded max_wall_seconds "
            f"during {context}: {elapsed:.3f} > {limits.max_wall_seconds:.3f}"
        )


def _check_count(
    *,
    name: str,
    value: int,
    maximum: int,
) -> None:
    if value > maximum:
        raise RingMacroEnumerabilityError(
            f"exact macro enumeration exceeded {name}: {value} > {maximum}"
        )


def _enumerate_semantic_sequence_law(
    model: FactorizedTraceletRateModel,
    decoder: SemanticRingSystemDecoder,
    category_logits: Tensor,
    *,
    limits: RingMacroEnumerationLimits,
    started_at: float,
    context: str,
    prefix_node_count: int,
) -> tuple[tuple[tuple[tuple[int, ...], float], ...], int]:
    """Enumerate one normalized electronic-label conditional under global bounds."""

    leaves: list[tuple[tuple[int, ...], float]] = []

    def visit(
        prefix: tuple[int, ...],
        label_log_probability: Tensor,
    ) -> None:
        nonlocal prefix_node_count
        prefix_node_count += 1
        _check_count(
            name="max_prefix_nodes",
            value=prefix_node_count,
            maximum=limits.max_prefix_nodes,
        )
        _check_elapsed(
            started_at,
            limits,
            context=f"{context}, electronic prefix",
        )
        if len(prefix) == decoder.span:
            leaves.append(
                (
                    prefix,
                    float(label_log_probability.detach().cpu()),
                )
            )
            return

        category_mask = torch.tensor(
            semantic_ring_next_category_mask(decoder, prefix),
            dtype=torch.bool,
            device=model.device,
        )
        contextual_logits = model._ring_semantic_contextual_logits(
            decoder,
            category_logits,
            position=len(prefix),
            prefix=prefix,
        )
        if category_mask.shape != contextual_logits.shape:
            raise RingMacroEnumerabilityError(
                "semantic category mask and logits do not align"
            )
        if not bool(category_mask.any()):
            raise RingMacroEnumerabilityError(
                "completion-aware semantic decoder reached a dead prefix"
            )
        next_log_probabilities = torch.log_softmax(
            contextual_logits.masked_fill(
                ~category_mask,
                float("-inf"),
            ),
            dim=0,
        )
        for category in (
            torch.nonzero(
                category_mask,
                as_tuple=False,
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        ):
            visit(
                (*prefix, int(category)),
                label_log_probability + next_log_probabilities[int(category)],
            )

    visit((), category_logits.new_zeros(()))
    _assert_probability_normalized(
        (log_probability for _prefix, log_probability in leaves),
        tolerance=limits.normalization_tolerance,
        context=f"electronic sequence conditional for {context}",
    )
    return tuple(leaves), prefix_node_count


def _verify_complete_action(
    state: MolecularGraph,
    action: RingSystemGrow,
) -> tuple[str, int]:
    """Verify macro execution and its deterministic validity-closed lowering."""

    if not is_valid_ring_system_grow(state, action):
        raise RingMacroEnumerabilityError(
            "enumerated RingSystemGrow action fails its authoritative validator"
        )
    try:
        lowering = lower_ring_system_grow(state, action)
    except Exception as exc:
        raise RingMacroEnumerabilityError(
            "enumerated RingSystemGrow action has no primitive lowering"
        ) from exc
    if not lowering:
        raise RingMacroEnumerabilityError(
            "enumerated RingSystemGrow action has an empty primitive lowering"
        )

    runtime = de_novo_rewrite_system()
    current = state
    try:
        for rule_name, primitive_action in lowering:
            current = runtime.apply(current, rule_name, primitive_action)
            if not is_valid_state(current) or not is_connected_or_null(current):
                raise RingMacroEnumerabilityError(
                    "macro lowering committed an invalid or disconnected intermediate"
                )
        macro_successor = runtime.apply(state, "ring_system_grow", action)
        lowered_key = canonical_state_key(current)
        macro_key = canonical_state_key(macro_successor)
    except RingMacroEnumerabilityError:
        raise
    except Exception as exc:
        raise RingMacroEnumerabilityError(
            "enumerated RingSystemGrow action failed executor verification"
        ) from exc
    if lowered_key != macro_key:
        raise RingMacroEnumerabilityError(
            "macro successor disagrees with its deterministic primitive lowering"
        )
    return macro_key, len(lowering)


def _record_encoding(
    action_masses: dict[RingSystemGrow, _MutableActionMass],
    *,
    action: RingSystemGrow,
    log_probability: float,
    address: RingMacroEncodingAddress,
    complete_encoding_count: int,
    raw_encoding_log_total: float,
    limits: RingMacroEnumerationLimits,
) -> tuple[int, float]:
    if not math.isfinite(log_probability):
        raise RingMacroEnumerabilityError("complete macro encoding has non-finite log probability")
    if abs(float(address.conditional_log_probability) - log_probability) > 1e-12:
        raise RingMacroEnumerabilityError(
            "encoding address mass disagrees with the raw complete-path mass"
        )
    complete_encoding_count += 1
    raw_encoding_log_total = _logaddexp(
        raw_encoding_log_total,
        float(log_probability),
    )
    _check_count(
        name="max_complete_encodings",
        value=complete_encoding_count,
        maximum=limits.max_complete_encodings,
    )
    try:
        incumbent = action_masses.get(action)
    except TypeError as exc:
        raise RingMacroEnumerabilityError(
            "RingSystemGrow actions do not provide hashable complete-action identity"
        ) from exc
    if incumbent is None:
        action_masses[action] = _MutableActionMass(
            log_probability=float(log_probability),
            addresses=[address],
        )
        _check_count(
            name="max_distinct_actions",
            value=len(action_masses),
            maximum=limits.max_distinct_actions,
        )
    else:
        incumbent.log_probability = _logaddexp(
            incumbent.log_probability,
            float(log_probability),
        )
        incumbent.addresses.append(address)
    return complete_encoding_count, raw_encoding_log_total


def enumerate_ring_macro_action_law(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    *,
    time: float = 0.5,
    limits: RingMacroEnumerationLimits | None = None,
) -> RingMacroEnumerabilityResult:
    """Recover and certify the exact bounded macro-only complete-action law.

    The function fails closed if the model is not in the existing safe
    macro-only profile, if any supported template lacks an enumerable
    placement/electronic law, or if an explicit resource bound is exceeded.
    It never falls back to template-only mass.
    """

    resolved_limits = limits or RingMacroEnumerationLimits()
    resolved_limits.validate()
    if not isinstance(model, FactorizedTraceletRateModel):
        raise TypeError("model must be a FactorizedTraceletRateModel")
    if model.enable_cycle_ops or not model.enable_ring_grow_macro:
        raise RingMacroEnumerabilityError(
            "probe requires the guarded macro-only profile "
            "(enable_cycle_ops=False, enable_ring_grow_macro=True)"
        )
    if model.ring_electronic_mode not in _SUPPORTED_ELECTRONIC_MODES:
        raise RingMacroEnumerabilityError(
            f"exact enumeration is unavailable for electronic mode {model.ring_electronic_mode!r}"
        )
    if tuple(int(index) for index in getattr(model, "excluded_sampling_ring_template_indices", ())):
        raise RingMacroEnumerabilityError(
            "sampling-time ring-template exclusions define a different law; "
            "the exact probe requires the unmodified macro family"
        )
    if "ring_system_grow" in {
        str(name) for name in getattr(model, "disabled_sampling_rule_names", ())
    }:
        raise RingMacroEnumerabilityError(
            "ring_system_grow is disabled at sampling time; no active macro law exists"
        )
    if not math.isfinite(float(time)) or not 0.0 <= float(time) <= 1.0:
        raise ValueError("time must be finite and lie in [0, 1]")
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise RingMacroEnumerabilityError(
            "macro probe source must be a valid connected molecule or null"
        )

    started_at = wall_time.monotonic()
    try:
        return _enumerate_ring_macro_action_law(
            model,
            state,
            time=float(time),
            limits=resolved_limits,
            started_at=started_at,
        )
    except RingMacroEnumerabilityError:
        raise
    except Exception as exc:
        raise RingMacroEnumerabilityError(
            "exact complete-action enumeration failed; no approximation was used"
        ) from exc


def _enumerate_ring_macro_action_law(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    *,
    time: float,
    limits: RingMacroEnumerationLimits,
    started_at: float,
) -> RingMacroEnumerabilityResult:
    batch = prepare_factorized_mark_batch(
        (state,),
        (time,),
        (None,),
        (None,),
        (0.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=True,
        compute_ring_restates=model.enable_ring_restates,
        compute_cyclic_graft=model.enable_cyclic_graft,
        compute_ring_opening=model.enable_ring_opening,
        compute_ring_system_delete=model.enable_ring_system_delete,
    ).to(model.device)

    action_masses: dict[RingSystemGrow, _MutableActionMass] = {}
    raw_placement_count = 0
    supported_placement_count = 0
    prefix_node_count = 0
    complete_encoding_count = 0
    raw_encoding_log_total = float("-inf")

    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch)
        masks, logits, action_log_z = model._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=True,
        )
        family_index = MARK_RULE_TO_INDEX["ring_system_grow"]
        template_mask = masks["ring_system_grow"][0]
        template_logits = logits["ring_system_grow"][0]
        family_log_normalizer = action_log_z[0, family_index]
        legal_template_indices = tuple(
            int(index)
            for index in torch.nonzero(template_mask, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        _check_count(
            name="max_legal_templates",
            value=len(legal_template_indices),
            maximum=limits.max_legal_templates,
        )
        _check_elapsed(started_at, limits, context="template support")

        if not legal_template_indices:
            if bool(torch.isfinite(family_log_normalizer)):
                raise RingMacroEnumerabilityError(
                    "empty ring macro support has a finite family normalizer"
                )
            elapsed = wall_time.monotonic() - started_at
            return RingMacroEnumerabilityResult(
                source_key=canonical_state_key(state),
                time=time,
                electronic_mode=model.ring_electronic_mode,
                normalization_applicable=False,
                legal_template_count=0,
                raw_placement_count=0,
                supported_placement_count=0,
                prefix_node_count=0,
                complete_encoding_count=0,
                distinct_complete_action_count=0,
                resonance_invariant_action_count=0,
                duplicate_encoding_count=0,
                encoding_total_probability=0.0,
                action_total_probability=0.0,
                max_teacher_scorer_log_error=0.0,
                elapsed_seconds=elapsed,
                actions=(),
            )
        if not bool(torch.isfinite(family_log_normalizer)):
            raise RingMacroEnumerabilityError(
                "nonempty ring macro support has a non-finite family normalizer"
            )

        template_log_probabilities = {
            template_index: float(
                (template_logits[template_index] - family_log_normalizer).detach().cpu()
            )
            for template_index in legal_template_indices
        }
        _assert_probability_normalized(
            template_log_probabilities.values(),
            tolerance=limits.normalization_tolerance,
            context="ring template conditional",
        )

        for template_index in legal_template_indices:
            _check_elapsed(
                started_at,
                limits,
                context=f"template {template_index}",
            )
            template_log_probability = template_log_probabilities[template_index]

            if model.ring_electronic_mode == "catalog_exact":
                candidates, candidate_log_probabilities = (
                    model._ring_exact_candidate_log_probabilities(
                        state,
                        template_index,
                        node[0],
                        global_state[0],
                        pair[0],
                    )
                )
                if not candidates:
                    raise RingMacroEnumerabilityError(
                        "exact ring support advertised an empty catalog candidate "
                        f"table for template {template_index}"
                    )
                candidate_logs = tuple(
                    float(value) for value in candidate_log_probabilities.detach().cpu().tolist()
                )
                if len(candidate_logs) != len(candidates):
                    raise RingMacroEnumerabilityError(
                        "catalog candidate probabilities do not align with actions"
                    )
                catalog_placements = tuple(
                    dict.fromkeys(candidate.placement for candidate in candidates)
                )
                raw_placement_count += len(catalog_placements)
                supported_placement_count += len(catalog_placements)
                _check_count(
                    name="max_raw_placements",
                    value=raw_placement_count,
                    maximum=limits.max_raw_placements,
                )
                _check_count(
                    name="max_supported_placements",
                    value=supported_placement_count,
                    maximum=limits.max_supported_placements,
                )
                _assert_probability_normalized(
                    candidate_logs,
                    tolerance=limits.normalization_tolerance,
                    context=f"catalog electronic conditional for template {template_index}",
                )
                for candidate_index, (candidate, candidate_log_probability) in enumerate(
                    zip(candidates, candidate_logs)
                ):
                    encoding_log_probability = template_log_probability + candidate_log_probability
                    complete_encoding_count, raw_encoding_log_total = _record_encoding(
                        action_masses,
                        action=candidate.action,
                        log_probability=encoding_log_probability,
                        address=RingMacroEncodingAddress(
                            template_index=template_index,
                            conditional_log_probability=encoding_log_probability,
                            catalog_candidate_index=candidate_index,
                        ),
                        complete_encoding_count=complete_encoding_count,
                        raw_encoding_log_total=raw_encoding_log_total,
                        limits=limits,
                    )
                continue

            placement_groups = model._ring_template_placement_groups(
                state,
                template_index,
            )
            raw_placement_count += len(placement_groups)
            _check_count(
                name="max_raw_placements",
                value=raw_placement_count,
                maximum=limits.max_raw_placements,
            )
            placement_logits, semantic_tables, placement_mask = (
                model._ring_supported_placement_tables(
                    state,
                    placement_groups,
                    node[0],
                    global_state[0],
                    pair[0],
                )
            )
            if (
                len(semantic_tables) != len(placement_groups)
                or placement_mask.numel() != len(placement_groups)
                or placement_logits.numel() != len(placement_groups)
            ):
                raise RingMacroEnumerabilityError(
                    "semantic placement factors have inconsistent dimensions"
                )
            supported_indices = tuple(
                int(index)
                for index in torch.nonzero(placement_mask, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            if not supported_indices:
                raise RingMacroEnumerabilityError(
                    "semantic ring support advertised an empty placement table "
                    f"for template {template_index}; exact law unavailable"
                )
            supported_placement_count += len(supported_indices)
            _check_count(
                name="max_supported_placements",
                value=supported_placement_count,
                maximum=limits.max_supported_placements,
            )
            placement_log_probability_tensor = torch.log_softmax(
                placement_logits.masked_fill(~placement_mask, float("-inf")),
                dim=0,
            )
            placement_log_probabilities = {
                index: float(placement_log_probability_tensor[index].detach().cpu())
                for index in supported_indices
            }
            _assert_probability_normalized(
                placement_log_probabilities.values(),
                tolerance=limits.normalization_tolerance,
                context=f"placement conditional for template {template_index}",
            )

            for placement_index in supported_indices:
                context = f"template {template_index}, placement {placement_index}"
                _check_elapsed(
                    started_at,
                    limits,
                    context=context,
                )
                decoder, category_logits = semantic_tables[placement_index]
                placement_log_probability = placement_log_probabilities[placement_index]
                label_law, prefix_node_count = _enumerate_semantic_sequence_law(
                    model,
                    decoder,
                    category_logits,
                    limits=limits,
                    started_at=started_at,
                    context=context,
                    prefix_node_count=prefix_node_count,
                )
                for prefix, label_log_probability in label_law:
                    action = instantiate_semantic_ring_system_grow(
                        state,
                        decoder,
                        prefix,
                    )
                    encoding_log_probability = (
                        template_log_probability
                        + placement_log_probability
                        + label_log_probability
                    )
                    complete_encoding_count, raw_encoding_log_total = _record_encoding(
                        action_masses,
                        action=action,
                        log_probability=encoding_log_probability,
                        address=RingMacroEncodingAddress(
                            template_index=template_index,
                            conditional_log_probability=encoding_log_probability,
                            placement_index=placement_index,
                            electronic_categories=prefix,
                        ),
                        complete_encoding_count=complete_encoding_count,
                        raw_encoding_log_total=raw_encoding_log_total,
                        limits=limits,
                    )

        _check_elapsed(started_at, limits, context="complete encoding aggregation")
        if not action_masses:
            raise RingMacroEnumerabilityError(
                "nonempty ring macro support produced no complete actions"
            )

        if not math.isfinite(raw_encoding_log_total):
            raise RingMacroEnumerabilityError(
                "nonempty ring macro support has no finite raw encoding mass"
            )
        encoding_total_probability = math.exp(raw_encoding_log_total)
        if (
            not math.isfinite(encoding_total_probability)
            or abs(encoding_total_probability - 1.0) > limits.normalization_tolerance
        ):
            raise RingMacroEnumerabilityError(
                "raw complete-encoding law is not normalized: "
                f"total_probability={encoding_total_probability:.9g}"
            )
        action_total_probability = _assert_probability_normalized(
            (mutable.log_probability for mutable in action_masses.values()),
            tolerance=limits.normalization_tolerance,
            context="aggregated complete RingSystemGrow action law",
        )

        # The teacher scorer deliberately quotients resonance-equivalent
        # complete actions.  Check at exactly that level while retaining exact
        # dataclass action identity in the result above.
        resonance_groups: dict[tuple, list[RingSystemGrow]] = {}
        for action in action_masses:
            resonance_groups.setdefault(
                ring_system_grow_electronic_key(action),
                [],
            ).append(action)
        max_scorer_log_error = 0.0
        for actions in resonance_groups.values():
            expected_log_probability = _logsumexp(
                action_masses[action].log_probability for action in actions
            )
            representative = min(actions, key=repr)
            action_score, legal = model._teacher_action_score(
                0,
                representative,
                "ring_system_grow",
                batch,
                node,
                global_state,
                pair,
                masks,
                logits,
            )
            if not bool(legal) or not bool(torch.isfinite(action_score)):
                raise RingMacroEnumerabilityError(
                    "enumerated action is absent from the model's teacher scorer"
                )
            observed_log_probability = float((action_score - family_log_normalizer).detach().cpu())
            error = abs(observed_log_probability - expected_log_probability)
            max_scorer_log_error = max(max_scorer_log_error, error)
            if error > limits.scorer_log_tolerance:
                raise RingMacroEnumerabilityError(
                    "enumerated action mass disagrees with the model teacher "
                    f"scorer: log_error={error:.9g}"
                )

    verified_actions = []
    for action, mutable in sorted(action_masses.items(), key=lambda item: repr(item[0])):
        _check_elapsed(started_at, limits, context="executor/lowering verification")
        successor_key, lowering_length = _verify_complete_action(state, action)
        probability = math.exp(mutable.log_probability)
        verified_actions.append(
            EnumeratedRingMacroAction(
                action=action,
                conditional_log_probability=mutable.log_probability,
                conditional_probability=probability,
                encoding_count=len(mutable.addresses),
                encoding_addresses=tuple(sorted(mutable.addresses)),
                successor_key=successor_key,
                primitive_lowering_length=lowering_length,
            )
        )

    elapsed = wall_time.monotonic() - started_at
    _check_elapsed(started_at, limits, context="final certificate")
    if abs(encoding_total_probability - action_total_probability) > limits.normalization_tolerance:
        raise RingMacroEnumerabilityError(
            "raw encoding and aggregated action probability totals disagree"
        )
    return RingMacroEnumerabilityResult(
        source_key=canonical_state_key(state),
        time=time,
        electronic_mode=model.ring_electronic_mode,
        normalization_applicable=True,
        legal_template_count=len(legal_template_indices),
        raw_placement_count=raw_placement_count,
        supported_placement_count=supported_placement_count,
        prefix_node_count=prefix_node_count,
        complete_encoding_count=complete_encoding_count,
        distinct_complete_action_count=len(verified_actions),
        resonance_invariant_action_count=len(resonance_groups),
        duplicate_encoding_count=complete_encoding_count - len(verified_actions),
        encoding_total_probability=encoding_total_probability,
        action_total_probability=action_total_probability,
        max_teacher_scorer_log_error=max_scorer_log_error,
        elapsed_seconds=elapsed,
        actions=tuple(verified_actions),
    )


__all__ = [
    "EnumeratedRingMacroAction",
    "RingMacroEncodingAddress",
    "RingMacroEnumerabilityError",
    "RingMacroEnumerabilityResult",
    "RingMacroEnumerationLimits",
    "enumerate_ring_macro_action_law",
]
