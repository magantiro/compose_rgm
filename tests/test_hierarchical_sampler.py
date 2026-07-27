"""Hierarchical training sampler: layer weighting, cold-element coverage floors, determinism.

The sampler must (a) draw layers by CONFIGURED weight (not the raw count ratio), (b) guarantee measured
minimum coverage for cold non-CNOF elements so the widened heads are supervised, (c) reduce to the uniform
baseline when weights track counts and no floor is set, and (d) be deterministic given a seed.
"""
from __future__ import annotations

from collections import Counter

from compose_v4.experiments.hierarchical_sampler import (
    CORRUPTION,
    MMP,
    HierarchicalMarkSampler,
    RecordTag,
)


def _corpus():
    # 90 corruption records (CNOF-heavy), 10 MMP records; sulfur appears in only a few corruption records.
    tags = []
    for i in range(90):
        elems = frozenset({"C", "N", "O"}) if i % 15 else frozenset({"C", "S"})
        tags.append(RecordTag(layer=CORRUPTION, path_length=(i % 5) + 1, target_elements=elems))
    for i in range(10):
        tags.append(RecordTag(layer=MMP, path_length=(i % 3) + 1, target_elements=frozenset({"C", "O"})))
    return tags


def test_layer_weights_override_the_count_ratio() -> None:
    tags = _corpus()  # counts: 90% corruption / 10% mmp
    sampler = HierarchicalMarkSampler(tags, layer_weights={CORRUPTION: 0.5, MMP: 0.5},
                                      samples_per_epoch=4000, seed=0)
    layers = Counter(tags[i].layer for i in sampler)
    frac_mmp = layers[MMP] / sum(layers.values())
    # requested 50/50 -> realized ~0.5 MMP, far above the 0.10 count ratio.
    assert 0.42 < frac_mmp < 0.58, frac_mmp
    assert sampler.realized_layer_fractions()[MMP] == 0.10


def test_cold_element_floor_guarantees_sulfur_coverage() -> None:
    tags = _corpus()
    # without a floor, MMP-weighted sampling barely touches the few sulfur records.
    no_floor = HierarchicalMarkSampler(tags, layer_weights={CORRUPTION: 0.3, MMP: 0.7},
                                       cold_element_floor=0.0, samples_per_epoch=2000, seed=1)
    s_no = sum("S" in tags[i].target_elements for i in no_floor) / 2000
    with_floor = HierarchicalMarkSampler(tags, layer_weights={CORRUPTION: 0.3, MMP: 0.7},
                                         cold_element_floor=0.15, samples_per_epoch=2000, seed=1)
    s_yes = sum("S" in tags[i].target_elements for i in with_floor) / 2000
    assert s_yes >= 0.14, s_yes            # floor met
    assert s_yes > s_no                    # and strictly more than without it


def test_uniform_reproduction_when_weights_track_counts() -> None:
    tags = _corpus()
    sampler = HierarchicalMarkSampler(tags, layer_weights={CORRUPTION: 0.9, MMP: 0.1},
                                      cold_element_floor=0.0, samples_per_epoch=5000, seed=2)
    layers = Counter(tags[i].layer for i in sampler)
    # weights == count ratio -> realized layer mix ~ the count ratio (0.90 / 0.10).
    assert abs(layers[CORRUPTION] / 5000 - 0.90) < 0.05


def test_deterministic_given_seed() -> None:
    tags = _corpus()
    kw = dict(layer_weights={CORRUPTION: 0.5, MMP: 0.5}, cold_element_floor=0.1,
              samples_per_epoch=500, seed=7)
    assert list(HierarchicalMarkSampler(tags, **kw)) == list(HierarchicalMarkSampler(tags, **kw))
    assert len(list(HierarchicalMarkSampler(tags, **kw))) == 500


def test_curriculum_bins_partition_by_path_length() -> None:
    tags = [RecordTag(layer=CORRUPTION, path_length=pl, target_elements=frozenset({"C"}))
            for pl in (1, 1, 2, 3, 5, 8)]
    sampler = HierarchicalMarkSampler(tags, layer_weights={CORRUPTION: 1.0},
                                      path_length_bins=(1, 2, 4), samples_per_epoch=1, seed=0)
    # edges (1,2,4) -> bins {<=1},{<=2},{<=4},{>4}: path lengths 1,1->bin0; 2->bin1; 3->bin2; 5,8->bin3.
    bins = {b for (ly, b) in sampler._groups}
    assert bins == {0, 1, 2, 3}
