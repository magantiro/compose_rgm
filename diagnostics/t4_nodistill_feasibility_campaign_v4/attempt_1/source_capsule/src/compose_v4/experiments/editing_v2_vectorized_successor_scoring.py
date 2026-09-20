"""Batched teacher-successor scoring, replacing per-alias Python loops.

WHY
---
MEASURED on an A10G, batch 32: backward is 503.6 ms of a 634.3 ms step -- 79%,
and ~15.7 ms per example for a 6.18M-parameter model, roughly a hundred times
what the arithmetic justifies. It is not FLOPs. ``forward_teacher_successor_
batch`` walks

    for batch_index in range(batch_size):
        for alias in fiber.aliases:
            value = _compiled_alias_log_probability(...)   # one index per alias
        selected[batch_index] = logsumexp(stack(values))   # one write per row

so a step builds thousands of tiny autograd nodes and backward pays per-node
launch and engine overhead. Raising the batch makes it worse, not better:
MEASURED 32 -> 128 scales backward 14.3x for 4x the work and drops throughput
from 50.45 to 16.49 examples/s.

WHAT THIS DOES INSTEAD
----------------------
Every alias in the batch is gathered in one operation PER TABLE -- there are
eight tables, not thousands of aliases -- and the per-row aggregation becomes a
single segmented logsumexp. The node count stops scaling with aliases.

The arithmetic is unchanged and deliberately so:

    log p(alias) = family_log_probability[b, f] + logits[t][b, *coord]
                                                - action_log_z[b, f]

WHY IT IS NOT ASSUMED EQUIVALENT
--------------------------------
A segmented reduction reassociates the sum, so bitwise equality with the
stacked ``logsumexp`` is NOT guaranteed even though the mathematics is
identical. This is the numerically load-bearing core of the objective, so the
parity gate compares per-alias values, per-row aggregates, the loss AND the
gradients against the original implementation, and the tolerance is stated
rather than discovered.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from compose_v4.experiments.factorized_successor_training import (
    FactorizedSuccessorPrediction,
    SuccessorTrainingError,
    TeacherSuccessorAlias,
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    _factorized_action_probability_tables,
    _log1mexp,
)
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_TO_INDEX


def segment_logsumexp(
    values: Tensor,
    segment_ids: Tensor,
    *,
    segments: int,
) -> Tensor:
    """``logsumexp`` within each segment, in a handful of ops.

    Segments with no members return ``-inf``, matching what an empty stack
    would mean. The maximum is detached before shifting: that is the standard
    stable form, and detaching it keeps the gradient exactly softmax rather
    than double-counting the argmax element through the shift.
    """

    if values.numel() == 0:
        return torch.full((segments,), float("-inf"), device=values.device, dtype=values.dtype)
    maxima = torch.full(
        (segments,), float("-inf"), device=values.device, dtype=values.dtype
    ).scatter_reduce(0, segment_ids, values.detach(), reduce="amax", include_self=True)
    shifted = (values - maxima[segment_ids]).exp()
    totals = torch.zeros(segments, device=values.device, dtype=values.dtype).index_add(
        0, segment_ids, shifted
    )
    # An empty segment keeps -inf; log(0) + -inf would be nan, so guard it.
    return torch.where(
        totals > 0,
        maxima + torch.log(totals.clamp_min(torch.finfo(values.dtype).tiny)),
        torch.full_like(maxima, float("-inf")),
    )


def _flatten_aliases(
    fibers: Sequence[Any],
    nonterminal: Sequence[bool],
) -> tuple[list[int], list[TeacherSuccessorAlias]]:
    rows: list[int] = []
    flat: list[TeacherSuccessorAlias] = []
    for index, fiber in enumerate(fibers):
        if not nonterminal[index] or fiber is None:
            continue
        for alias in fiber.aliases:
            rows.append(index)
            flat.append(alias)
    return rows, flat


def gather_alias_log_probabilities(
    rows: Sequence[int],
    aliases: Sequence[TeacherSuccessorAlias],
    *,
    masks: Mapping[str, Tensor],
    logits: Mapping[str, Tensor],
    action_log_z: Tensor,
    family_log_probability: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Score every alias in the batch with one gather per action table.

    Returns ``(log_probabilities, legality, family_index)`` aligned with
    ``aliases``. One gather per table rather than per alias is the entire
    point: there are eight tables and can be thousands of aliases.
    """

    device = action_log_z.device
    count = len(aliases)
    values = torch.zeros(count, device=device, dtype=action_log_z.dtype)
    legal = torch.zeros(count, device=device, dtype=torch.bool)
    families = torch.zeros(count, dtype=torch.long, device=device)

    by_table: dict[str, list[int]] = {}
    for position, alias in enumerate(aliases):
        family_index = MARK_RULE_TO_INDEX.get(alias.family_name)
        if family_index is None:
            raise SuccessorTrainingError(
                f"compiled successor references unknown family {alias.family_name!r}"
            )
        families[position] = family_index
        by_table.setdefault(alias.table_name, []).append(position)

    row_tensor = torch.as_tensor(list(rows), dtype=torch.long, device=device)
    for table_name, positions in by_table.items():
        table_logits = logits.get(table_name)
        table_mask = masks.get(table_name)
        if table_logits is None or table_mask is None:
            raise SuccessorTrainingError(
                f"compiled successor references unknown table {table_name!r}"
            )
        index = torch.as_tensor(positions, dtype=torch.long, device=device)
        arity = table_logits.dim() - 1
        coordinates = [alias.coordinate for alias in (aliases[p] for p in positions)]
        if any(len(c) != arity for c in coordinates):
            raise SuccessorTrainingError(
                f"table {table_name!r} expects {arity} coordinates per alias"
            )
        selectors: list[Tensor] = [row_tensor[index]]
        for axis in range(arity):
            axis_index = torch.as_tensor(
                [c[axis] for c in coordinates], dtype=torch.long, device=device
            )
            bound = int(table_logits.shape[axis + 1])
            if bool((axis_index < 0).any()) or bool((axis_index >= bound).any()):
                raise SuccessorTrainingError(
                    "compiled successor coordinate lies outside the current action table"
                )
            selectors.append(axis_index)
        values = values.index_copy(0, index, table_logits[tuple(selectors)])
        legal = legal.index_copy(0, index, table_mask[tuple(selectors)])

    family_terms = (
        family_log_probability[row_tensor, families] - action_log_z[row_tensor, families]
    )
    return values + family_terms, legal, families


def forward_teacher_successor_batch_vectorized(
    model: Any,
    batch: Any,
    fibers: Sequence[Any],
) -> FactorizedSuccessorPrediction:
    """Batched equivalent of ``forward_teacher_successor_batch``.

    ``state_supports`` is not accepted. The corpus has ZERO virtual aliases
    across all 151,082 compiled entries, so the productive normalizer is
    ``log1mexp(-inf) = 0`` throughout; supporting the general case here without
    a corpus that exercises it would be untested code on the critical path.
    A fiber carrying virtual aliases raises rather than being silently ignored.
    """

    if len(fibers) != batch.batch_size:
        raise ValueError("teacher successor fibers must align with the batch")
    nonterminal = [float(v) > 0.0 for v in batch.teacher_rates.detach().cpu()]
    (
        batch,
        global_state,
        masks,
        logits,
        action_log_z,
        enabled,
        family_log_probability,
    ) = _factorized_action_probability_tables(model, batch)
    has_legal_mark = enabled.any(dim=-1)
    size = batch.batch_size
    device = global_state.device

    for index, fiber in enumerate(fibers):
        if not nonterminal[index]:
            if fiber is not None:
                raise SuccessorTrainingError(
                    "terminal training row carries a teacher-successor fiber"
                )
            continue
        if fiber is None:
            raise SuccessorTrainingError("nonterminal training row lacks its teacher fiber")
        if fiber.state_support.virtual_aliases:
            raise SuccessorTrainingError(
                "vectorized scoring does not implement virtual aliases; the frozen "
                "corpus has none, and silently ignoring them would change the "
                "productive normalizer"
            )

    rows, aliases = _flatten_aliases(fibers, nonterminal)
    if not aliases:
        raise SuccessorTrainingError("batch carries no teacher successor alias")

    values, legal, families = gather_alias_log_probabilities(
        rows,
        aliases,
        masks=masks,
        logits=logits,
        action_log_z=action_log_z,
        family_log_probability=family_log_probability,
    )
    if not bool(legal.all()):
        raise SuccessorTrainingError(
            "precompiled teacher alias is no longer legal in the current batch"
        )

    segment_ids = torch.as_tensor(list(rows), dtype=torch.long, device=device)
    selected = segment_logsumexp(values, segment_ids, segments=size)
    # No virtual aliases: log1mexp(-inf) = 0 for every row, computed rather
    # than hardcoded so the identity is visible in the graph.
    virtual = torch.full((size,), float("-inf"), device=device, dtype=global_state.dtype)
    productive_log_probability = _log1mexp(virtual)

    teacher_family = torch.zeros(size, dtype=torch.long, device=device)
    for index, fiber in enumerate(fibers):
        if not nonterminal[index]:
            continue
        name = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(
            batch.teacher_rule_names[index], batch.teacher_rule_names[index]
        )
        if name is None:
            alias_families = {alias.family_name for alias in fiber.aliases}
            if len(alias_families) != 1:
                raise SuccessorTrainingError(
                    "teacher family is absent and cannot be inferred uniquely "
                    "from a cross-family successor fiber"
                )
            name = next(iter(alias_families))
        family_index = MARK_RULE_TO_INDEX.get(name)
        if family_index is None:
            raise SuccessorTrainingError(f"teacher row references unknown family {name!r}")
        teacher_family[index] = family_index

    within = families == teacher_family[segment_ids]
    if not bool(within.any()):
        raise SuccessorTrainingError(
            "teacher successor fiber has no alias in the recorded teacher family"
        )
    within_totals = segment_logsumexp(
        values[within], segment_ids[within], segments=size
    )
    teacher_family_log_probability = torch.where(
        torch.as_tensor(nonterminal, device=device),
        family_log_probability[torch.arange(size, device=device), teacher_family],
        torch.zeros(size, device=device, dtype=global_state.dtype),
    )

    live = torch.as_tensor(nonterminal, device=device)
    zeros = torch.zeros(size, device=device, dtype=global_state.dtype)
    selected = torch.where(live, selected, zeros)
    selected_productive = torch.where(live, selected - productive_log_probability, zeros)
    selected_within = torch.where(
        live, within_totals - teacher_family_log_probability, zeros
    )

    total_hazard = F.softplus(
        model.total_hazard_head(global_state).squeeze(-1)
    ) * has_legal_mark.to(global_state.dtype)
    return FactorizedSuccessorPrediction(
        total_hazard=total_hazard,
        productive_hazard=total_hazard * productive_log_probability.exp(),
        selected_successor_log_probability=selected,
        selected_productive_successor_log_probability=selected_productive,
        productive_log_probability=productive_log_probability,
        teacher_family_log_probability=teacher_family_log_probability,
        selected_within_teacher_family_log_probability=selected_within,
        family_log_probabilities=family_log_probability,
        enabled_families=enabled,
    )


__all__ = [
    "forward_teacher_successor_batch_vectorized",
    "gather_alias_log_probabilities",
    "segment_logsumexp",
]
