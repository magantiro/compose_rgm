"""Invariants for the global (primitive-free) graph delta between two molecules."""

from __future__ import annotations

import subprocess
import sys

from compose_v4.experiments.global_delta_census import (
    TEMPLATES,
    assign_template,
    compute_global_delta,
    stable_seed,
)

_BENZENE = "c1ccccc1"
_TOLUENE = "Cc1ccccc1"
_IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_ANION = "CC(=O)[O-]"
_ACID = "CC(=O)O"
_PARA_XYLENE = "Cc1ccc(C)cc1"


def _pairs():
    return [
        (_BENZENE, _BENZENE),
        (_BENZENE, _TOLUENE),
        (_TOLUENE, _BENZENE),
        (_IBUPROFEN, _ASPIRIN),
        (_ACID, _ANION),
        (_BENZENE, _PARA_XYLENE),
        ("CCCCCC", "C1CCCCC1"),
    ]


def test_atom_conservation_holds_on_every_pair() -> None:
    for source, target in _pairs():
        delta = compute_global_delta(source, target)
        assert delta.status == "ok", (source, target, delta.status)
        assert delta.retained_atoms + delta.excised_atoms == delta.source_heavy
        assert delta.retained_atoms + delta.installed_atoms == delta.target_heavy
        assert delta.delta_heavy == delta.target_heavy - delta.source_heavy
        assert delta.delta_heavy == delta.installed_atoms - delta.excised_atoms


def test_region_sizes_sum_to_their_totals() -> None:
    for source, target in _pairs():
        delta = compute_global_delta(source, target)
        assert sum(delta.excised_region_sizes) == delta.excised_atoms
        assert sum(delta.installed_region_sizes) == delta.installed_atoms
        assert len(delta.excised_region_sizes) == delta.n_excised_regions
        assert len(delta.installed_region_sizes) == delta.n_installed_regions
        assert all(size > 0 for size in delta.excised_region_sizes)
        assert all(size > 0 for size in delta.installed_region_sizes)


def test_identity_pair_retains_everything() -> None:
    delta = compute_global_delta(_IBUPROFEN, _IBUPROFEN)
    assert delta.excised_atoms == 0
    assert delta.installed_atoms == 0
    assert delta.retained_fraction_source == 1.0
    assert delta.operation_family == "identity"
    assert assign_template(delta) == "T0_identity"


def test_pure_growth_and_pure_deletion_are_distinguished() -> None:
    grow = compute_global_delta(_BENZENE, _TOLUENE)
    assert grow.excised_atoms == 0 and grow.installed_atoms == 1
    assert grow.operation_family == "grow"
    shrink = compute_global_delta(_TOLUENE, _BENZENE)
    assert shrink.installed_atoms == 0 and shrink.excised_atoms == 1
    assert shrink.operation_family == "delete"


def test_charge_change_is_reported_and_is_not_an_excision() -> None:
    delta = compute_global_delta(_ACID, _ANION)
    assert delta.charge_source == 0
    assert delta.charge_target == -1
    assert delta.charge_delta == -1
    assert delta.excised_atoms == 0 and delta.installed_atoms == 0


def test_ring_closure_is_seen_as_a_ring_change_not_an_atom_change() -> None:
    delta = compute_global_delta("CCCCCC", "C1CCCCC1")
    assert delta.ring_count_source == 0
    assert delta.ring_count_target == 1
    assert delta.ring_count_delta == 1
    assert delta.ring_topology_changed is True
    assert assign_template(delta) in {"T4_ring_rewire", "T0_core_restate"}


def test_two_separated_substituents_read_as_multi_region() -> None:
    delta = compute_global_delta(_BENZENE, _PARA_XYLENE)
    assert delta.installed_atoms == 2
    assert delta.n_installed_regions == 2
    assert delta.multi_region is True
    assert delta.max_anchor_separation >= 2
    assert assign_template(delta) == "T5_multi_region_edit"


def test_anchors_never_exceed_the_boundary_edge_count() -> None:
    for source, target in _pairs():
        delta = compute_global_delta(source, target)
        assert delta.n_anchor_atoms <= delta.n_attachment_boundaries or (
            delta.n_attachment_boundaries == 0 and delta.n_anchor_atoms == 0
        )


def test_every_pair_receives_exactly_one_known_template() -> None:
    for source, target in _pairs():
        delta = compute_global_delta(source, target)
        assert assign_template(delta) in TEMPLATES


def test_unparseable_input_is_refused_not_silently_scored() -> None:
    delta = compute_global_delta("not_a_molecule", _BENZENE)
    assert delta.status == "unparseable"
    assert assign_template(delta) == "unclassified"


def test_stable_seed_is_stable_across_processes() -> None:
    """``hash()`` is PYTHONHASHSEED-salted; the digest seed must not be."""
    expected = stable_seed("pmo", "global_delta", "v1")
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from compose_v4.experiments.global_delta_census import stable_seed;"
        "print(stable_seed('pmo', 'global_delta', 'v1'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
    )
    assert int(out.stdout.strip()) == expected


def test_installed_content_axes_separate_a_chain_from_a_ring() -> None:
    """``_grow_actions`` can only build a linear single-bonded C/N/O chain."""
    chain = compute_global_delta(_BENZENE, "CCCc1ccccc1")
    assert chain.installed_atoms == 3
    assert chain.installed_region_has_ring is False
    assert chain.largest_installed_is_linear_chain is True
    assert chain.installed_expressible_as_grow_chain is True

    ring = compute_global_delta(_BENZENE, "c1ccc(-c2ccccc2)cc1")
    assert ring.installed_atoms == 6
    assert ring.installed_region_has_ring is True
    assert ring.largest_installed_is_linear_chain is False
    assert ring.installed_expressible_as_grow_chain is False


def test_a_branched_installed_region_is_not_a_grow_chain() -> None:
    branched = compute_global_delta(_BENZENE, "CC(C)Cc1ccccc1")
    assert branched.installed_branch_points >= 1
    assert branched.installed_expressible_as_grow_chain is False


def test_an_installed_chain_longer_than_the_cap_is_not_expressible() -> None:
    long_chain = compute_global_delta(_BENZENE, "CCCCCCCCCc1ccccc1")
    assert long_chain.installed_atoms == 9
    assert long_chain.largest_installed_is_linear_chain is True
    assert long_chain.installed_expressible_as_grow_chain is False


def test_anchor_groups_counts_merged_sites_not_raw_regions() -> None:
    both = compute_global_delta(_BENZENE, _PARA_XYLENE)
    assert both.n_anchor_groups == 2
    single = compute_global_delta(_BENZENE, _TOLUENE)
    assert single.n_anchor_groups == 1
