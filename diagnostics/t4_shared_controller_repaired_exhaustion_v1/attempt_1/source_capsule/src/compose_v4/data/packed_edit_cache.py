"""Packed training cache: independently addressable examples that PRESERVE the trace-first law.

The scientific sampling law is, and must remain:

    layer  ->  trace  ->  one progress state  ->  one jump target OR the terminal (no-jump) target

Flattening every supervised transition into one urn and drawing uniformly is a DIFFERENT objective: a
trace's draw probability is independent of its length, so a step of a K-step trace carries roughly 1/K of
it. Uniform-over-transitions would upweight long traces in proportion to K (MMP paths average ~5.9 steps),
materially changing the optimized expectation. This module therefore makes rows *addressable*, not
*equiprobable*: the sampler still draws layer -> trace -> progress, and the row is a RETRIEVAL target.

What makes that fast without changing the law: the two quantities that used to require walking the trace
on every draw are closed-form given three integers stored per row.

  * the progress marginal is exactly Binomial(path_length, alpha(t)) (``TraceProgressCTMC``: N_t ~
    Bin(K, alpha(t))), so it depends on the path LENGTH and the time -- never on the trace contents;
  * the family-stratified proposal is ``1 / (n_families_on_trace * size_of_this_family_group)``.

So a row carries ``path_length``, ``n_stratification_families`` and ``family_group_size``, and the
sampler reproduces ``_sample_tracelet_progress`` exactly -- including its importance weight -- without
materializing the trace. Storing a scalar probability instead would be WRONG: both quantities are
functions of the per-draw time.

Layout (one row per independently addressable example; a K-step trace contributes K+1 rows, the last
being its terminal/no-jump example):

    layer_id, trace_id, progress_index, path_length, is_terminal, family_id,
    state_id, fiber_id, target_action_index, target_successor_group,
    n_stratification_families, family_group_size

``state_id`` / ``fiber_id`` are content-addressed into deduped stores: several rows referencing the same
exact state must not each carry a copy of a potentially large legal-candidate fiber.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

TERMINAL_FAMILY = "<TERMINAL>"

# Row schema. Bumping this invalidates a built cache.
PACKED_CACHE_SCHEMA_VERSION = 1

_ROW_FIELDS = (
    "layer_id",
    "trace_id",
    "progress_index",
    "path_length",
    "is_terminal",
    "family_id",
    "state_id",
    "fiber_id",
    "target_action_index",
    "target_successor_group",
    "n_stratification_families",
    "family_group_size",
)


@dataclass(frozen=True)
class PackedExampleRow:
    """One independently addressable training example."""

    layer_id: int
    trace_id: int
    progress_index: int
    path_length: int
    is_terminal: bool
    family_id: int
    state_id: int
    fiber_id: int
    target_action_index: int
    target_successor_group: int
    n_stratification_families: int
    family_group_size: int


def stratification_structure(rule_names: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    """Reproduce the grouping ``_sample_tracelet_progress`` builds, without the trace.

    Returns ``(n_families, group_size_per_position)`` where position ``K`` (the terminal) is included as
    its own family -- exactly as the live sampler does (``groups["<TERMINAL>"] = [path_length]``).
    """
    groups: dict[str, int] = {}
    for name in rule_names:
        groups[name] = groups.get(name, 0) + 1
    n_families = len(groups) + 1  # + <TERMINAL>
    sizes = tuple(groups[name] for name in rule_names) + (1,)
    return n_families, sizes


def progress_marginal(path_length: int, alpha: float) -> np.ndarray:
    """Binomial(path_length, alpha) over progress 0..path_length -- the exact GM progress law.

    Mirrors ``TraceProgressCTMC.marginal`` including its degenerate alpha endpoints.
    """
    if path_length == 0:
        return np.ones(1, dtype=np.float64)
    out = np.zeros(path_length + 1, dtype=np.float64)
    if alpha <= 0.0:
        out[0] = 1.0
        return out
    if alpha >= 1.0:
        out[-1] = 1.0
        return out
    k = np.arange(path_length + 1)
    log_binom = gammaln(path_length + 1) - gammaln(k + 1) - gammaln(path_length - k + 1)
    return np.exp(log_binom + k * np.log(alpha) + (path_length - k) * np.log1p(-alpha))


def sample_progress_packed(
    *,
    path_length: int,
    group_sizes: tuple[int, ...],
    n_families: int,
    alpha: float,
    rng,
    stratification_fraction: float,
) -> tuple[int, float]:
    """Exact packed-path equivalent of ``_sample_tracelet_progress``.

    Draws a progress index and returns its importance weight, using only the stored integers. The
    returned pair must match the trace-walking implementation in distribution AND in weight.
    """
    marginal = progress_marginal(path_length, alpha)
    fraction = float(stratification_fraction)
    if fraction == 0.0:
        progress = int(rng.choice(len(marginal), p=marginal))
        return progress, 1.0

    if rng.random() < fraction:
        # family-uniform, then position-uniform within that family: identical to drawing a family index
        # then one of its positions. Reproduced here by sampling a position with probability
        # 1/(n_families * group_size), which is exactly the live proposal.
        weights = np.array(
            [1.0 / (n_families * group_sizes[i]) for i in range(path_length + 1)], dtype=np.float64
        )
        progress = int(rng.choice(len(weights), p=weights / weights.sum()))
    else:
        progress = int(rng.choice(len(marginal), p=marginal))

    stratified = 1.0 / (n_families * group_sizes[progress])
    proposal = (1.0 - fraction) * float(marginal[progress]) + fraction * stratified
    if proposal <= 0.0:
        raise RuntimeError("progress proposal assigned zero probability")
    return progress, float(marginal[progress]) / proposal


def terminal_probability(path_length: int, n_families: int, expected_alpha_power) -> float:
    """Closed-form P(a draw lands on the terminal position) for one trace.

    ``expected_alpha_power(K)`` supplies ``E_t[alpha(t)**K]`` under the production time law, so this needs
    no Monte Carlo. Used by the exposure audit to convert trace draws into transition exposure.
    """
    stratified = 1.0 / n_families
    return 0.5 * float(expected_alpha_power(path_length)) + 0.5 * stratified


def build_rows_for_trace(
    *,
    layer_id: int,
    trace_id: int,
    rule_names: tuple[str, ...],
    family_ids: tuple[int, ...],
    state_ids: tuple[int, ...],
    fiber_ids: tuple[int, ...],
    target_action_indices: tuple[int, ...],
    target_successor_groups: tuple[int, ...],
) -> tuple[PackedExampleRow, ...]:
    """Expand one trace into its K+1 addressable rows (K jumps + 1 terminal).

    ``state_ids`` has length K+1 (a state per progress position, including the endpoint); the jump-only
    tuples have length K.
    """
    path_length = len(rule_names)
    if len(state_ids) != path_length + 1:
        raise ValueError("state_ids must cover every progress position including the endpoint")
    for name, seq in (
        ("fiber_ids", fiber_ids),
        ("target_action_indices", target_action_indices),
        ("target_successor_groups", target_successor_groups),
        ("family_ids", family_ids),
    ):
        if len(seq) != path_length:
            raise ValueError(f"{name} must have one entry per jump step")
    n_families, group_sizes = stratification_structure(rule_names)
    rows = []
    for progress in range(path_length + 1):
        terminal = progress == path_length
        rows.append(
            PackedExampleRow(
                layer_id=layer_id,
                trace_id=trace_id,
                progress_index=progress,
                path_length=path_length,
                is_terminal=terminal,
                family_id=-1 if terminal else family_ids[progress],
                state_id=state_ids[progress],
                fiber_id=-1 if terminal else fiber_ids[progress],
                target_action_index=-1 if terminal else target_action_indices[progress],
                target_successor_group=-1 if terminal else target_successor_groups[progress],
                n_stratification_families=n_families,
                family_group_size=group_sizes[progress],
            )
        )
    return tuple(rows)


def rows_to_arrays(rows) -> dict[str, np.ndarray]:
    """Columnar view for O(1) lookup and GPU-friendly gathering."""
    return {
        field: np.asarray([getattr(row, field) for row in rows], dtype=np.int64)
        for field in _ROW_FIELDS
    }


def row_offsets_by_trace(rows) -> dict[tuple[int, int], int]:
    """(layer_id, trace_id) -> index of that trace's progress-0 row.

    A trace's row block is contiguous, so ``offset + progress_index`` addresses any example in O(1) --
    the retrieval primitive that lets the sampler stay trace-first while the cache stays flat.
    """
    offsets: dict[tuple[int, int], int] = {}
    for i, row in enumerate(rows):
        key = (row.layer_id, row.trace_id)
        if key not in offsets:
            offsets[key] = i
        elif i - offsets[key] != row.progress_index:
            raise ValueError(f"non-contiguous row block for trace {key}")
    return offsets
