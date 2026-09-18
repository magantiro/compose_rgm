"""The secondary chemistry screen must be narrower than our taste and wider than the task.

Two failure directions, both already paid for in this project.

TOO WIDE handicaps the benchmark comparison. The existing `queryable_fiber.instability`
bundle rejects 7.0% of the benchmark's own Jin QED lead set and 13.8% of held-out
GuacaMol, chiefly through a plain-enone rule that fires on 9.0% of GuacaMol. Search run
under that bundle is searching a strictly smaller fiber than T4 defines, so its scores are
not comparable to a published number obtained on the full one.

TOO NARROW lets the optimiser win on chemistry nobody can use: an acyl-nitroso topped a
32-call experiment at -10.6 and was retracted, and four of 118 autonomous endpoints in the
delta=0.6 lane carried a free primary imine.

So every rule faces an admission test against a reference population, and the test is run
here rather than asserted in prose.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from compose_v4.control.chemistry_screen import (
    FALSE_POSITIVE_CEILING,
    SCREEN_MOTIFS,
    false_positive_rates,
    screen,
)

REFERENCE = Path("configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv")


def _reference_leads() -> list[str]:
    with REFERENCE.open() as handle:
        return [
            row["canonical_nonisomeric_smiles"]
            for row in csv.DictReader(handle)
            if row.get("canonical_nonisomeric_smiles")
        ]


def test_every_rule_carries_a_stated_mechanism():
    """A rule without a mechanism is a preference, and preferences are not stability."""
    for name, smarts, mechanism in SCREEN_MOTIFS:
        assert smarts.strip(), name
        assert len(mechanism) > 40, f"{name} needs a mechanism, not a label"


def test_no_rule_may_reject_the_benchmarks_own_leads():
    """The admission test. A rule firing on real drug-like chemistry describes chemistry."""
    leads = _reference_leads()
    assert len(leads) > 500, "the reference population must be large enough to bound a rate"
    rates = false_positive_rates(leads)
    offenders = {name: rate for name, rate in rates.items() if rate > FALSE_POSITIVE_CEILING}
    assert not offenders, (
        f"these rules fire on more than {FALSE_POSITIVE_CEILING:.1%} of the benchmark's own "
        f"leads and are therefore describing ordinary chemistry, not instability: {offenders}"
    )


def test_the_enone_rule_is_excluded_and_would_fail_admission():
    """Kept as a live negative control on the admission test itself.

    `michael_acceptor_ketone` is in the search-time bundle and is the single largest
    contributor to its false-positive rate. If the admission test ever stops rejecting it,
    the test has broken.
    """
    from rdkit import Chem

    assert "michael_acceptor_ketone" not in {name for name, _, _ in SCREEN_MOTIFS}
    enone = Chem.MolFromSmarts("[CX3]=[CX3][CX3]=[OX1]")
    leads = [Chem.MolFromSmiles(s) for s in _reference_leads()]
    leads = [m for m in leads if m is not None]
    rate = sum(1 for m in leads if m.HasSubstructMatch(enone)) / len(leads)
    assert rate > FALSE_POSITIVE_CEILING, (
        f"the enone rule fires on {rate:.1%} of real leads; it must fail admission"
    )


def test_the_official_t4_seeds_all_pass():
    """The screen may never refuse a molecule the benchmark itself starts from."""
    import json

    payload = json.loads(Path("configs/t4_frozen_program_benchmark_v2.json").read_text())
    for cell, record in payload["payload"]["cells"].items():
        assert screen(record["original_seed"]) == [], cell


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # The retracted acyl-nitroso that topped a 32-call experiment at -10.6.
        ("CC(=O)ONOC(=O)CC1Nc2ccccc2-c2ccnc3c2c1c1n3C(=O)CCC1", "acyl_nitroso_or_N_O_acyl"),
        # An autonomous delta=0.6 endpoint at -10.10 carrying a free primary ketimine.
        (
            "CN(C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23)C1CC(=N)C(=O)C1",
            "primary_ketimine_or_aldimine",
        ),
    ],
)
def test_the_motifs_that_motivated_the_screen_are_caught(smiles, expected):
    assert expected in screen(smiles)


@pytest.mark.parametrize(
    "smiles, why",
    [
        ("NC(=N)c1ccccc1", "benzamidine: the N lone pair stabilises the C=N"),
        ("CC(=NO)C", "an oxime, ordinary drug chemistry"),
        ("CC(C)=NNc1ccccc1", "a hydrazone, ordinary drug chemistry"),
        ("COC(=N)c1ccccc1", "an imidate"),
        ("O=C1C=CC(=O)C=C1", "a quinone -- an enone, deliberately NOT screened"),
    ],
)
def test_the_imine_rule_is_the_narrow_form(smiles, why):
    """It must not become a ban on C=N. Each of these carries a C=N and must pass."""
    assert "primary_ketimine_or_aldimine" not in screen(smiles), why


def test_the_screen_is_not_the_task_gate():
    """A molecule may fail the screen and still be a valid T4 endpoint, and vice versa.

    This is the distinction the reporting depends on: the official number is computed on
    the benchmark's own criterion, and the vetted number is reported beside it, never
    instead of it.
    """
    autonomous_best = "COC(=O)CC1Nc2c(ccc3c2OC(=N)C3=O)-c2ccnc3[nH]cc1c23"
    from compose_v4.experiments.t4_fiber_campaign import Fiber

    root = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
    assert Fiber(root, 0.6).check(autonomous_best) is not None, "it is a valid T4 endpoint"
