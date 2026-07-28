"""The production training mixture must expose THREE independent layers with configured weights.

Cycle supervision defines RingCore. If cycle records are folded into the corruption layer:
  * their draw probability cannot be controlled independently;
  * artifact volume silently determines training mass;
  * layer exposure cannot be audited.

So the layer weights are an authoritative CONFIG value, never inferred from record counts, and the
sampler must receive three separate record groups.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.experiments.hierarchical_sampler import (  # noqa: E402
    CYCLE_OPS,
    GENERAL_CORRUPTION,
    MMP_ANALOGUE,
    PRODUCTION_LAYERS,
    build_layered_sampler,
)
from ring_core_identity import (  # noqa: E402
    LAYER_WEIGHTS_HASH,
    PRODUCTION_LAYER_WEIGHTS,
    layer_weights_hash_from_mapping,
    recompute_layer_weights_hash,
)


def test_production_layers_are_three_and_named():
    assert PRODUCTION_LAYERS == (GENERAL_CORRUPTION, CYCLE_OPS, MMP_ANALOGUE)
    assert len(set(PRODUCTION_LAYERS)) == 3


def test_owner_locked_layer_weights():
    """The first scaled run's weights are owner-locked; drift must break the hash."""
    assert PRODUCTION_LAYER_WEIGHTS == {
        GENERAL_CORRUPTION: 0.40,
        CYCLE_OPS: 0.25,
        MMP_ANALOGUE: 0.35,
    }
    assert abs(sum(PRODUCTION_LAYER_WEIGHTS.values()) - 1.0) < 1e-12


def test_layer_weights_hash_is_stable_and_sensitive():
    assert recompute_layer_weights_hash() == LAYER_WEIGHTS_HASH
    drifted = dict(PRODUCTION_LAYER_WEIGHTS)
    drifted[CYCLE_OPS] = 0.30
    assert layer_weights_hash_from_mapping(drifted) != LAYER_WEIGHTS_HASH


def _records(n, path_length):
    """Minimal duck-typed records: the sampler only reads ``path.path_length``."""
    class _P:
        def __init__(self, length):
            self.path_length = length

    class _R:
        def __init__(self, length):
            self.path = _P(length)

    return tuple(_R(path_length) for _ in range(n))


def test_cycle_records_are_not_folded_into_corruption():
    """Draw frequency must track the CONFIGURED weight, not the record-count ratio.

    Cycle records are deliberately the most numerous layer here (as in the real corpus); if they were
    folded into corruption, or if weights were inferred from size, the realized cycle share would jump
    far above its configured 0.25.
    """
    by_layer = {
        GENERAL_CORRUPTION: _records(50, 3),
        CYCLE_OPS: _records(900, 1),
        MMP_ANALOGUE: _records(120, 6),
    }
    sampler = build_layered_sampler(
        by_layer, layer_weights=PRODUCTION_LAYER_WEIGHTS, path_length_bins=(5, 9, 13), seed=7
    )
    offsets, start = {}, 0
    for layer, records in by_layer.items():
        offsets[layer] = (start, start + len(records))
        start += len(records)

    rng = np.random.default_rng(0)
    counts = dict.fromkeys(by_layer, 0)
    draws = 40000
    for _ in range(draws):
        i = sampler.draw(rng)
        for layer, (lo, hi) in offsets.items():
            if lo <= i < hi:
                counts[layer] += 1
                break
    for layer, weight in PRODUCTION_LAYER_WEIGHTS.items():
        realized = counts[layer] / draws
        assert abs(realized - weight) < 0.02, f"{layer}: configured {weight}, realized {realized:.4f}"

    # and the count ratio it must NOT follow
    cycle_count_share = len(by_layer[CYCLE_OPS]) / sum(len(v) for v in by_layer.values())
    assert cycle_count_share > 0.80
    assert abs(counts[CYCLE_OPS] / draws - cycle_count_share) > 0.5


def test_weights_are_never_inferred_from_artifact_size():
    """Two corpora with very different layer sizes must yield the same realized layer mixture."""
    def realized(sizes):
        by_layer = {
            GENERAL_CORRUPTION: _records(sizes[0], 3),
            CYCLE_OPS: _records(sizes[1], 1),
            MMP_ANALOGUE: _records(sizes[2], 6),
        }
        sampler = build_layered_sampler(
            by_layer, layer_weights=PRODUCTION_LAYER_WEIGHTS, path_length_bins=(5, 9, 13), seed=3
        )
        bounds, start = {}, 0
        for layer, records in by_layer.items():
            bounds[layer] = (start, start + len(records))
            start += len(records)
        rng = np.random.default_rng(11)
        counts = dict.fromkeys(by_layer, 0)
        for _ in range(20000):
            i = sampler.draw(rng)
            for layer, (lo, hi) in bounds.items():
                if lo <= i < hi:
                    counts[layer] += 1
                    break
        return {k: v / 20000 for k, v in counts.items()}

    a = realized((50, 900, 120))
    b = realized((900, 50, 800))
    for layer in PRODUCTION_LAYER_WEIGHTS:
        assert abs(a[layer] - b[layer]) < 0.03, f"{layer} moved with corpus size: {a} vs {b}"


def test_missing_layer_is_rejected_not_silently_dropped():
    """A production mixture missing a positively-weighted layer must fail loudly."""
    with pytest.raises(ValueError):
        build_layered_sampler(
            {GENERAL_CORRUPTION: _records(0, 3)},
            layer_weights=PRODUCTION_LAYER_WEIGHTS,
            path_length_bins=(5, 9, 13),
            seed=1,
        )
