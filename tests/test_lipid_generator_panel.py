"""Lipid unconditional-generator eval panel (P2-G3 / L1).

Validates the panel computes the lipid marginals and behaves as a self-check:
two samples of the same corpus distribution have near-identical architecture
marginals (low JS), and the reportable fields (degradable-linker rate, region
shares, lipid-scaffold novelty) are well-formed.
"""

from __future__ import annotations

from scripts.analyze_lipid_generator_panel import (
    lipid_marginals,
    lipid_scaffold_novelty,
    _js_marginals,
)

# small hand set: two amino-ester lipids (tails + ester + amine head) + one amide
LIPIDS_A = [
    "CCCCCCCCCCCCOC(=O)CCN(C)CCN(C)CCC(=O)OCCCCCCCCCCCC",
    "CCCCCCCCOC(=O)CCN(C)CCC(=O)OCCCCCCCC",
    "CCCCCCCCCCCCCCNC(=O)CCN(C)CCCN(C)C",
]
LIPIDS_B = [
    "CCCCCCCCCCOC(=O)CCN(C)CCN(C)CCC(=O)OCCCCCCCCCC",
    "CCCCCCCCCCOC(=O)CCN(C)CCC(=O)OCCCCCCCCCC",
    "CCCCCCCCCCCCNC(=O)CCN(C)CCCN(C)C",
]


def test_marginals_are_well_formed() -> None:
    m = lipid_marginals(LIPIDS_A)
    assert m["n"] == 3
    arch = m["architecture"]
    for axis in ("n_tails", "tail_length", "branching", "unsaturation", "basic_nitrogen_count"):
        assert abs(sum(arch[axis].values()) - 1.0) < 1e-6
    share = arch["region_atom_share"]
    assert abs(sum(share.values()) - 1.0) < 1e-6
    assert share["tail"] > share["linker"]          # lipids are tail-heavy
    assert 0.0 <= m["degradable_linker_rate"] <= 1.0
    assert m["degradable_linker_rate"] > 0.5         # ester/amide lipids are degradable


def test_self_check_same_distribution_has_low_js() -> None:
    js = _js_marginals(lipid_marginals(LIPIDS_A), lipid_marginals(LIPIDS_B))
    # two closely-matched lipid sets => small architecture divergence on every axis
    assert all(v < 0.3 for v in js.values()), js


def test_scaffold_novelty_bounds() -> None:
    nov = lipid_scaffold_novelty(LIPIDS_A, LIPIDS_A)
    assert nov["lipid_scaffold_novelty"] == 0.0       # identical set => nothing novel
    nov2 = lipid_scaffold_novelty(LIPIDS_A, [])
    assert nov2["lipid_scaffold_novelty"] == 1.0      # no training cores => all novel
