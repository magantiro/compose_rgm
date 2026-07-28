"""The packed cache must reproduce the trace-first sampling LAW, not merely the per-example loss.

Two distinct failure modes are covered:

  1. per-example: given the same packed row, the loss/gradient semantics must match (that panel lives in
     tests/test_segmented_successor_equivalence.py);
  2. DISTRIBUTIONAL: over many draws, the packed path must reproduce the trusted path's progress
     distribution, terminal share, family frequencies and importance-weight distribution.

(2) is the safeguard against the specific regression the owner called out: making every transition
independently addressable must NOT silently become uniform-over-transitions sampling. A discriminating
test proves the panel can actually fail -- a uniform-over-transitions sampler is checked to be REJECTED.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.data.packed_edit_cache import (  # noqa: E402
    build_rows_for_trace,
    progress_marginal,
    row_offsets_by_trace,
    sample_progress_packed,
    stratification_structure,
)

FRACTION = 0.5  # production progress_stratification_fraction
DRAWS = 60000
TOL = 0.02


# ---- reference: the trusted trace-walking law, transcribed from _sample_tracelet_progress ------------


def _reference_sample(rule_names, alpha, rng, fraction=FRACTION):
    """Structurally identical to compose_v4.experiments.tracelet_conditional._sample_tracelet_progress,
    but driven by an explicit alpha so no molecules need replaying."""
    path_length = len(rule_names)
    marginal = progress_marginal(path_length, alpha)
    groups: dict[str, list[int]] = {}
    for progress, name in enumerate(rule_names):
        groups.setdefault(name, []).append(progress)
    groups["<TERMINAL>"] = [path_length]
    family_names = tuple(groups)

    if rng.random() < fraction:
        family = family_names[int(rng.integers(0, len(family_names)))]
        positions = groups[family]
        progress = int(positions[int(rng.integers(0, len(positions)))])
    else:
        progress = int(rng.choice(len(marginal), p=marginal))

    family = rule_names[progress] if progress < path_length else "<TERMINAL>"
    stratified = 1.0 / (len(family_names) * len(groups[family]))
    proposal = (1.0 - fraction) * float(marginal[progress]) + fraction * stratified
    return progress, float(marginal[progress]) / proposal


TRACES = [
    ("single_step_cycle", ("bond_insert",)),
    ("two_step_same_family", ("atom_delete", "atom_delete")),
    ("mmp_like_delete_then_insert", tuple(["atom_delete"] * 5 + ["atom_insert"] * 3)),
    ("corruption_mixed", ("atom_restate", "bond_reorder", "atom_delete", "atom_restate")),
    ("long_skewed", tuple(["atom_delete"] * 11 + ["ring_system_restate"])),
    ("all_distinct", ("atom_insert", "bond_reroute", "ring_system_delete")),
]
ALPHAS = [0.15, 0.5, 0.9]


@pytest.mark.parametrize("name,rule_names", TRACES, ids=[t[0] for t in TRACES])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_progress_distribution_matches(name, rule_names, alpha):
    """Progress-index distribution conditional on path length."""
    n_families, sizes = stratification_structure(rule_names)
    ref, pak = Counter(), Counter()
    r1, r2 = np.random.default_rng(0), np.random.default_rng(1)
    for _ in range(DRAWS):
        ref[_reference_sample(rule_names, alpha, r1)[0]] += 1
        pak[
            sample_progress_packed(
                path_length=len(rule_names), group_sizes=sizes, n_families=n_families,
                alpha=alpha, rng=r2, stratification_fraction=FRACTION,
            )[0]
        ] += 1
    for progress in range(len(rule_names) + 1):
        a, b = ref[progress] / DRAWS, pak[progress] / DRAWS
        assert abs(a - b) < TOL, f"{name} alpha={alpha} progress={progress}: {a:.4f} vs {b:.4f}"


@pytest.mark.parametrize("name,rule_names", TRACES, ids=[t[0] for t in TRACES])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_terminal_share_and_importance_weights_match(name, rule_names, alpha):
    """Terminal (no-jump) share and the importance-weight distribution."""
    path_length = len(rule_names)
    n_families, sizes = stratification_structure(rule_names)
    r1, r2 = np.random.default_rng(2), np.random.default_rng(3)
    ref_term = pak_term = 0
    ref_w, pak_w = [], []
    for _ in range(DRAWS):
        p, w = _reference_sample(rule_names, alpha, r1)
        ref_term += p == path_length
        ref_w.append(w)
        p, w = sample_progress_packed(
            path_length=path_length, group_sizes=sizes, n_families=n_families,
            alpha=alpha, rng=r2, stratification_fraction=FRACTION,
        )
        pak_term += p == path_length
        pak_w.append(w)
    assert abs(ref_term / DRAWS - pak_term / DRAWS) < TOL, f"{name}: terminal share differs"
    assert abs(np.mean(ref_w) - np.mean(pak_w)) < TOL, f"{name}: mean importance weight differs"
    assert abs(np.std(ref_w) - np.std(pak_w)) < 0.05, f"{name}: weight spread differs"


@pytest.mark.parametrize("name,rule_names", TRACES, ids=[t[0] for t in TRACES])
def test_jump_family_frequencies_match(name, rule_names):
    """Per-family landing frequencies (weighted by the importance weight = expected gradient mass)."""
    path_length = len(rule_names)
    n_families, sizes = stratification_structure(rule_names)
    r1, r2 = np.random.default_rng(4), np.random.default_rng(5)
    ref_mass, pak_mass = Counter(), Counter()
    for _ in range(DRAWS):
        p, w = _reference_sample(rule_names, 0.5, r1)
        ref_mass[rule_names[p] if p < path_length else "<TERMINAL>"] += w
        p, w = sample_progress_packed(
            path_length=path_length, group_sizes=sizes, n_families=n_families,
            alpha=0.5, rng=r2, stratification_fraction=FRACTION,
        )
        pak_mass[rule_names[p] if p < path_length else "<TERMINAL>"] += w
    ref_total, pak_total = sum(ref_mass.values()), sum(pak_mass.values())
    for family in set(ref_mass) | set(pak_mass):
        a, b = ref_mass[family] / ref_total, pak_mass[family] / pak_total
        assert abs(a - b) < TOL, f"{name} family {family}: gradient mass {a:.4f} vs {b:.4f}"


def test_importance_weight_is_unbiased_against_the_marginal():
    """The weighted landing distribution must recover the GM marginal exactly -- that is what the
    importance weight is FOR. If it did not, stratification would silently change the objective."""
    rule_names = ("atom_delete", "atom_delete", "atom_restate", "bond_reorder")
    alpha = 0.4
    n_families, sizes = stratification_structure(rule_names)
    marginal = progress_marginal(len(rule_names), alpha)
    rng = np.random.default_rng(6)
    mass = np.zeros(len(rule_names) + 1)
    n = 200000
    for _ in range(n):
        p, w = sample_progress_packed(
            path_length=len(rule_names), group_sizes=sizes, n_families=n_families,
            alpha=alpha, rng=rng, stratification_fraction=FRACTION,
        )
        mass[p] += w
    mass /= n
    assert np.allclose(mass, marginal, atol=0.01), f"weighted mass {mass} vs marginal {marginal}"


def test_panel_rejects_uniform_over_transitions_sampling():
    """Proof the panel is DISCRIMINATING.

    The tempting "improvement" once rows are addressable is to draw a row uniformly. That upweights long
    traces by ~K and drops the terminal share. This asserts the distributional panel REJECTS it -- so the
    tests above are load-bearing, not decorative.
    """
    rule_names = tuple(["atom_delete"] * 5 + ["atom_insert"] * 3)  # K=8, MMP-like
    path_length = len(rule_names)
    alpha = 0.5
    rng_ref, rng_bad = np.random.default_rng(7), np.random.default_rng(8)
    ref_term = bad_term = 0
    for _ in range(DRAWS):
        ref_term += _reference_sample(rule_names, alpha, rng_ref)[0] == path_length
        # BUG: uniform over the trace's K+1 addressable rows
        bad_term += int(rng_bad.integers(0, path_length + 1)) == path_length
    ref_share, bad_share = ref_term / DRAWS, bad_term / DRAWS
    assert abs(ref_share - bad_share) > TOL, (
        "uniform-over-transitions must DISAGREE with the trace-first law here; if it agrees, this panel "
        f"cannot catch the objective change (ref {ref_share:.4f} vs uniform {bad_share:.4f})"
    )


def test_rows_are_contiguous_and_addressable():
    """offset + progress_index must address any example in O(1), and a K-step trace yields K+1 rows."""
    rule_names = ("atom_delete", "atom_restate", "bond_reorder")
    k = len(rule_names)
    rows = build_rows_for_trace(
        layer_id=0, trace_id=42, rule_names=rule_names,
        family_ids=(1, 2, 3), state_ids=(10, 11, 12, 13), fiber_ids=(20, 21, 22),
        target_action_indices=(0, 1, 2), target_successor_groups=(5, 6, 7),
    )
    assert len(rows) == k + 1
    assert rows[-1].is_terminal and rows[-1].family_id == -1 and rows[-1].fiber_id == -1
    assert not any(r.is_terminal for r in rows[:-1])
    assert rows[-1].state_id == 13, "terminal row must reference the endpoint state"
    offsets = row_offsets_by_trace(rows)
    base = offsets[(0, 42)]
    for progress in range(k + 1):
        assert rows[base + progress].progress_index == progress
    n_families, sizes = stratification_structure(rule_names)
    assert all(r.n_stratification_families == n_families for r in rows)
    assert tuple(r.family_group_size for r in rows) == sizes


def test_terminal_row_carries_no_jump_target():
    """Terminal examples stay explicit -- they train the exit-rate term and must not be silently dropped."""
    rows = build_rows_for_trace(
        layer_id=1, trace_id=0, rule_names=("bond_insert",), family_ids=(4,),
        state_ids=(1, 2), fiber_ids=(3,), target_action_indices=(0,), target_successor_groups=(9,),
    )
    terminal = rows[-1]
    assert terminal.is_terminal
    assert terminal.target_action_index == -1 and terminal.target_successor_group == -1
    assert terminal.n_stratification_families == 2  # one jump family + <TERMINAL>
