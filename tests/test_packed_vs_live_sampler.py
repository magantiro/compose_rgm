"""The packed progress sampler must match the LIVE production sampler, not a transcription of it.

tests/test_packed_sampler_law_equivalence.py compares against a local reimplementation of
``_sample_tracelet_progress``. That is useful for covering trace shapes cheaply, but it is a COPY: if the
production function changes, that panel stays green while training silently diverges. This module closes
the gap by driving the real ``_sample_tracelet_progress`` over real ``TraceProgressCTMC`` records built
from real molecules, and comparing it to the packed path.

It also pins the alpha<->time relationship. The packed path is parameterised by ``alpha``, the live one by
``time``; they coincide only because ``PowerSurvivalScheduler.power == 1``. If that default ever changes,
the packed sampler would silently sample a different progress law, so the coupling is asserted rather
than assumed.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compose_v4.data.packed_edit_cache import (  # noqa: E402
    progress_marginal,
    sample_progress_packed,
    stratification_structure,
)
from compose_v4.experiments.analogue_prior import rewrite_trace_from_record  # noqa: E402
from compose_v4.experiments.tracelet_conditional import (  # noqa: E402
    _sample_tracelet_progress,
)
from compose_v4.rewrite.progress import PowerSurvivalScheduler, TraceProgressCTMC  # noqa: E402

FRACTION = 0.5
POOL = REPO / "diagnostics/composition/analogue_trace_pool.jsonl"


def test_scheduler_power_is_one_so_alpha_equals_time():
    """The packed path substitutes alpha for time. That is only valid at power == 1."""
    scheduler = PowerSurvivalScheduler()
    assert scheduler.power == 1.0, (
        "packed_edit_cache parameterises progress by alpha and the audits treat alpha == t; a non-unit "
        "scheduler power breaks that identity and the packed sampler must be re-derived"
    )
    for t in (0.0, 0.13, 0.5, 0.87, 1.0):
        assert scheduler.alpha(t) == pytest.approx(t, abs=1e-12)


def _real_paths(limit=12):
    if not POOL.exists():
        pytest.skip("local analogue pool sample unavailable")
    import json

    paths = []
    for line in POOL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            trace = rewrite_trace_from_record(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
        paths.append(TraceProgressCTMC(trace))
        if len(paths) >= limit:
            break
    if not paths:
        pytest.skip("no rebuildable pool records")
    return paths


def test_packed_marginal_matches_the_live_marginal_exactly():
    """progress_marginal must reproduce TraceProgressCTMC.marginal for real traces, not approximately."""
    for path in _real_paths():
        for t in (0.01, 0.25, 0.5, 0.75, 0.99):
            live = path.marginal(t)
            packed = progress_marginal(path.path_length, PowerSurvivalScheduler().alpha(t))
            assert np.allclose(live, packed, atol=1e-12, rtol=1e-12), (
                f"K={path.path_length} t={t}: live {live} vs packed {packed}"
            )


def test_packed_stratification_structure_matches_the_live_grouping():
    """The stored (n_families, group_sizes) must equal what the live sampler derives from the trace."""
    for path in _real_paths():
        rule_names = tuple(step.rule_name for step in path.trace.steps)
        n_families, sizes = stratification_structure(rule_names)

        groups: dict[str, list[int]] = {}
        for progress, name in enumerate(rule_names):
            groups.setdefault(name, []).append(progress)
        groups["<TERMINAL>"] = [path.path_length]

        assert n_families == len(groups)
        assert len(sizes) == path.path_length + 1
        for progress in range(path.path_length):
            assert sizes[progress] == len(groups[rule_names[progress]])
        assert sizes[path.path_length] == 1


@pytest.mark.parametrize("time_point", [0.15, 0.5, 0.85])
def test_packed_and_live_progress_distributions_agree_on_real_traces(time_point):
    """The decisive check: same progress distribution and same terminal share, live vs packed."""
    draws = 20000
    alpha = PowerSurvivalScheduler().alpha(time_point)
    for path in _real_paths(limit=6):
        rule_names = tuple(step.rule_name for step in path.trace.steps)
        n_families, sizes = stratification_structure(rule_names)

        live_rng, packed_rng = np.random.default_rng(0), np.random.default_rng(1)
        live, packed = Counter(), Counter()
        for _ in range(draws):
            live[_sample_tracelet_progress(
                path, time=time_point, rng=live_rng, stratification_fraction=FRACTION
            )[0]] += 1
            packed[sample_progress_packed(
                path_length=path.path_length, group_sizes=sizes, n_families=n_families,
                alpha=alpha, rng=packed_rng, stratification_fraction=FRACTION,
            )[0]] += 1
        for progress in range(path.path_length + 1):
            a, b = live[progress] / draws, packed[progress] / draws
            assert abs(a - b) < 0.02, (
                f"K={path.path_length} t={time_point} progress={progress}: live {a:.4f} vs packed {b:.4f}"
            )


@pytest.mark.parametrize("time_point", [0.15, 0.5, 0.85])
def test_packed_and_live_importance_weights_agree_on_real_traces(time_point):
    """A matching progress distribution with mismatched weights would still change the gradient."""
    draws = 20000
    alpha = PowerSurvivalScheduler().alpha(time_point)
    for path in _real_paths(limit=6):
        rule_names = tuple(step.rule_name for step in path.trace.steps)
        n_families, sizes = stratification_structure(rule_names)

        live_rng, packed_rng = np.random.default_rng(2), np.random.default_rng(3)
        live_mass, packed_mass = np.zeros(path.path_length + 1), np.zeros(path.path_length + 1)
        for _ in range(draws):
            p, w = _sample_tracelet_progress(
                path, time=time_point, rng=live_rng, stratification_fraction=FRACTION
            )
            live_mass[p] += w
            p, w = sample_progress_packed(
                path_length=path.path_length, group_sizes=sizes, n_families=n_families,
                alpha=alpha, rng=packed_rng, stratification_fraction=FRACTION,
            )
            packed_mass[p] += w
        live_mass /= draws
        packed_mass /= draws
        assert np.allclose(live_mass, packed_mass, atol=0.02), (
            f"K={path.path_length} t={time_point}: weighted mass {live_mass} vs {packed_mass}"
        )
        # and both must recover the GM marginal -- the property the weight exists to preserve
        assert np.allclose(live_mass, path.marginal(time_point), atol=0.02)


def test_zero_stratification_reduces_to_the_plain_marginal_in_both_paths():
    """fraction == 0 is the unstratified GM law; both paths must degenerate to it identically."""
    draws = 20000
    for path in _real_paths(limit=4):
        rule_names = tuple(step.rule_name for step in path.trace.steps)
        n_families, sizes = stratification_structure(rule_names)
        live_rng, packed_rng = np.random.default_rng(4), np.random.default_rng(5)
        live, packed = Counter(), Counter()
        for _ in range(draws):
            p, w = _sample_tracelet_progress(
                path, time=0.5, rng=live_rng, stratification_fraction=0.0
            )
            assert w == 1.0, "unstratified draws must carry unit weight"
            live[p] += 1
            p, w = sample_progress_packed(
                path_length=path.path_length, group_sizes=sizes, n_families=n_families,
                alpha=0.5, rng=packed_rng, stratification_fraction=0.0,
            )
            assert w == 1.0
            packed[p] += 1
        for progress in range(path.path_length + 1):
            assert abs(live[progress] / draws - packed[progress] / draws) < 0.02
