"""Score-free access to the production factorized action support.

The scientific process remains defined by the frozen model action tables and
production coordinate decoder.  Static training-cache preparation needs those
legal coordinates, but it must not spend an encoder pass evaluating scores that
are discarded.  This adapter supplies zero-valued feature tensors, reads the
unchanged production masks, and decodes only caller-selected families.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.factorized_mark_conditional import (
    operator_capability_batch_kwargs,
)
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    _coordinate_action,
    _TABLE_FAMILIES,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key


@dataclass(frozen=True)
class LegalRewriteMark:
    """One legal production coordinate without a model score."""

    family_name: str
    table_name: str
    executor_rule_name: str
    action: Any
    coordinate: tuple[int, ...]


@dataclass(frozen=True)
class FactorizedLegalSupport:
    """Selected legal marks plus the exact all-family legal-mark census."""

    source_key: str
    marks: tuple[LegalRewriteMark, ...]
    raw_mark_count: int
    marks_by_family: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.source_key:
            raise ValueError("legal support source key must be nonempty")
        if type(self.raw_mark_count) is not int or self.raw_mark_count < 0:
            raise ValueError("legal support raw_mark_count must be nonnegative")
        if tuple(sorted(dict(self.marks_by_family).items())) != self.marks_by_family:
            raise ValueError("legal support family census must be sorted and unique")
        if sum(count for _family, count in self.marks_by_family) != self.raw_mark_count:
            raise ValueError("legal support family census does not match raw_mark_count")


def enumerate_factorized_legal_support_many(
    model: FactorizedTraceletRateModel,
    states: Sequence[MolecularGraph],
    times: Sequence[float],
    *,
    included_families: Sequence[frozenset[str]] | None = None,
    chemistry_feature_cache: MutableMapping[Any, Any] | None = None,
    chemistry_feature_cache_limit: int = 2048,
) -> tuple[FactorizedLegalSupport, ...]:
    """Decode selected legal marks without evaluating the neural scorer.

    ``included_families`` changes work, not support: ``raw_mark_count`` and
    ``marks_by_family`` always describe every legal production mark.
    """

    if not isinstance(model, FactorizedTraceletRateModel):
        raise TypeError("legal support enumeration requires FactorizedTraceletRateModel")
    selected_states = tuple(states)
    selected_times = tuple(float(value) for value in times)
    if not selected_states or len(selected_states) != len(selected_times):
        raise ValueError("legal support states and times must be nonempty and aligned")
    if any(not 0.0 < value < 1.0 for value in selected_times):
        raise ValueError("legal support times must lie in the open unit interval")
    if model.enable_ring_grow_macro:
        raise ProductionSuccessorKernelError(
            "the legacy ring_system_grow macro is enabled during legal-support enumeration"
        )
    if not model.enable_cycle_ops:
        raise ProductionSuccessorKernelError(
            "compositional cycle close/open are disabled during legal-support enumeration"
        )

    known_families = frozenset(family for family, _table in _TABLE_FAMILIES)
    if included_families is None:
        requested = tuple(known_families for _state in selected_states)
    else:
        requested = tuple(frozenset(row) for row in included_families)
        if len(requested) != len(selected_states):
            raise ValueError("included legal-support families must align with states")
        unknown = frozenset().union(*requested) - known_families
        if unknown:
            raise ValueError(f"unknown legal-support families requested: {sorted(unknown)}")

    batch = prepare_factorized_mark_batch(
        selected_states,
        selected_times,
        (None,) * len(selected_states),
        (None,) * len(selected_states),
        (0.0,) * len(selected_states),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        chemistry_feature_cache=chemistry_feature_cache,
        chemistry_feature_cache_limit=chemistry_feature_cache_limit,
        **operator_capability_batch_kwargs(model.operator_capabilities),
    ).to(model.device)
    dtype = next(model.parameters()).dtype
    node = torch.zeros(
        (batch.batch_size, batch.n_slots, model.hidden_dim),
        dtype=dtype,
        device=model.device,
    )
    global_state = torch.zeros(
        (batch.batch_size, model.hidden_dim),
        dtype=dtype,
        device=model.device,
    )
    pair = torch.zeros(
        (batch.batch_size, batch.n_slots, batch.n_slots, model.hidden_dim),
        dtype=dtype,
        device=model.device,
    )
    with torch.inference_mode():
        masks, _logits, _action_log_z = model._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
    if bool(masks["ring_system_grow"].any()):
        raise ProductionSuccessorKernelError(
            "ring_system_grow has legal support despite the production macro being disabled"
        )
    cpu_masks = {name: value.detach().cpu() for name, value in masks.items()}
    rows: list[FactorizedLegalSupport] = []
    for batch_index, (state, families) in enumerate(zip(selected_states, requested, strict=True)):
        one = batch.subbatch(batch_index, batch_index + 1)
        counts: dict[str, int] = {}
        decoded: list[LegalRewriteMark] = []
        for family_name, table_name in _TABLE_FAMILIES:
            row_mask = cpu_masks[table_name][batch_index]
            count = int(row_mask.sum().item())
            counts[family_name] = counts.get(family_name, 0) + count
            if family_name not in families:
                continue
            for coordinate_row in torch.nonzero(row_mask, as_tuple=False):
                coordinate = tuple(int(value) for value in coordinate_row)
                executor_rule_name, action = _coordinate_action(
                    model,
                    state,
                    one,
                    family_name=family_name,
                    table_name=table_name,
                    coordinate=coordinate,
                )
                decoded.append(
                    LegalRewriteMark(
                        family_name=family_name,
                        table_name=table_name,
                        executor_rule_name=executor_rule_name,
                        action=action,
                        coordinate=coordinate,
                    )
                )
        family_census = tuple(sorted(counts.items()))
        rows.append(
            FactorizedLegalSupport(
                source_key=canonical_state_key(state),
                marks=tuple(decoded),
                raw_mark_count=sum(counts.values()),
                marks_by_family=family_census,
            )
        )
    return tuple(rows)


__all__ = [
    "FactorizedLegalSupport",
    "LegalRewriteMark",
    "enumerate_factorized_legal_support_many",
]
