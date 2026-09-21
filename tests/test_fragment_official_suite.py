"""Guards for the fragment-constrained benchmark protocol.

Each test here exists because a specific way of inflating a reported number is
cheap and invisible in the artifact: counting a failed attempt as no attempt,
re-drawing until a sample lands, tuning the sampler per instance, deriving a
"reproducible" seed from a salted hash, or auditing fragment preservation with
a check that cannot fail.  Every test below is paired with a negative control:
it asserts the guard FIRES on the bad input, not merely that it passes on the
good one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    assert_emission_invariants,
    official_unique_valid,
)

MANIFEST = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"


def _prompts():
    return load_genmol_prompts(MANIFEST)


def _run_suite():
    import run_fragment_constrained_suite as suite

    return suite


# ---- Denominator: a failed attempt is an attempt ----


def test_failure_placeholder_is_counted_invalid_not_dropped():
    """An attempt that produced nothing must lower validity, not vanish.

    ``FAILED_SAMPLE_PLACEHOLDER`` is the empty string, and an empty SMILES
    PARSES to a zero-atom Mol rather than returning None.  The official metric
    survives that only because it tests the truthiness of the canonical string.
    If that ever changed, every failed attempt would silently score as valid,
    so pin the behaviour rather than trusting it.
    """
    mol = Chem.MolFromSmiles(FAILED_SAMPLE_PLACEHOLDER)
    assert mol is not None, "an empty SMILES is expected to parse to a 0-atom Mol"
    assert not Chem.MolToSmiles(mol, canonical=True)

    real = "c1ccccc1"
    samples = [real] * 70 + [FAILED_SAMPLE_PLACEHOLDER] * 30
    assert len(official_unique_valid(samples)) == 1
    # The valid count the official validity ratio is built from.
    valid = [s for s in samples if s and Chem.MolToSmiles(Chem.MolFromSmiles(s))]
    assert len(valid) == 70
    assert len(valid) / len(samples) * 100 == pytest.approx(70.0)


def test_official_metrics_refuse_a_short_sample_list():
    """Scoring 40 survivors as if 40 were requested is the cherry-pick to block."""
    from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics

    with pytest.raises(ValueError, match="expected exactly 100"):
        official_prompt_metrics(["c1ccccc1"] * 40, expected_samples=100)


def test_emission_invariants_reject_noncanonical_and_disconnected():
    """Both failures inflate a reported number, so both must raise."""
    assert_emission_invariants(["c1ccccc1", FAILED_SAMPLE_PLACEHOLDER])
    with pytest.raises(ValueError, match="not canonical"):
        assert_emission_invariants(["C1=CC=CC=C1"])
    with pytest.raises(ValueError, match="disconnected"):
        assert_emission_invariants(["c1ccccc1.CO"])


# ---- Reproducibility ----


def test_prompt_rng_seed_is_stable_across_processes():
    """The published seed must name one stream, not one process's stream."""
    suite = _run_suite()
    assert suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0) == (
        suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0)
    )
    # Pinned so a future refactor of the derivation is visible as a diff.
    assert suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0) == 722072583
    assert suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 1) != (
        suite.prompt_rng_seed("BARICITINIB", "superstructure_generation", 0)
    )


def test_frozen_sampler_identity_moves_when_any_knob_moves():
    """A per-instance tune must be visible in the artifact's hash."""
    from compose_v4.benchmark.fragment_conditioned_sampler import SamplerConfig

    suite = _run_suite()
    base = suite.frozen_sampler_identity(SamplerConfig())
    assert base["config_sha256"] == suite.frozen_sampler_identity(SamplerConfig())["config_sha256"]
    for field, value in (
        ("max_events", 64),
        ("mark_attempts_per_event", 48),
        ("operational_horizon", 32.0),
        ("n_slots", 56),
    ):
        tuned = suite.frozen_sampler_identity(SamplerConfig(**{field: value}))
        assert tuned["config_sha256"] != base["config_sha256"], field


# ---- Fragment preservation, with a negative control ----


def test_preservation_audit_accepts_a_superstructure_and_rejects_a_near_miss():
    """The audit must FAIL on a molecule that lost the core.

    A containment check that returns True for everything would report a
    preservation rate of 100% no matter what the sampler did.
    """
    suite = _run_suite()
    prompt = next(
        p
        for p in _prompts()
        if p.drug_name == "BARICITINIB" and p.task is FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    queries = suite.audit_queries(prompt)
    core = prompt.fragments[0]

    # A genuine superstructure: the core plus a methyl.
    grown = "Cc1nc(-c2cnn(C3CNC3)c2)c2cc[nH]c2n1"
    assert suite.contains_all_fragments(grown, queries)

    # Negative controls.
    assert not suite.contains_all_fragments("c1ccccc1", queries)
    assert not suite.contains_all_fragments("", queries)
    # Disconnected: the core is present but the molecule is not one piece.
    assert not suite.contains_all_fragments(f"{core}.CO", queries)
    # A ring deletion inside the core must not pass.
    broken = "Cc1nc(-c2cnn(CCCN)c2)c2cc[nH]c2n1"
    assert not suite.contains_all_fragments(broken, queries)


def test_linker_audit_requires_two_disjoint_embeddings():
    """Two fragments matching the SAME atoms is not two fragments."""
    suite = _run_suite()
    prompt = next(
        p
        for p in _prompts()
        if p.drug_name == "BARICITINIB" and p.task is FragmentTask.LINKER_DESIGN
    )
    queries = suite.audit_queries(prompt)
    assert len(queries) == 2
    assert suite.contains_all_fragments(prompt.original_smiles, queries)
    # One of the two cores alone must not satisfy the pair.
    alone = Chem.MolToSmiles(queries[1])
    assert not suite.contains_all_fragments(alone, queries)


def test_distance_reference_is_the_dummy_stripped_prompt():
    """Distance depends on the reference, so pin which molecule it is."""
    suite = _run_suite()
    prompt = next(
        p
        for p in _prompts()
        if p.drug_name == "BARICITINIB" and p.task is FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    reference = suite.prompt_reference_smiles(prompt)
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(prompt.fragments[0]), canonical=True)
    assert reference == expected
    assert "*" not in reference


def test_distance_reference_parses_for_every_released_prompt():
    """An aromatic-N attachment point makes the dummy-stripped SMILES unparseable.

    Upstream's ``calculate_average_tanimoto`` raises ``Invalid prompt SMILES
    string`` on such a reference, which killed three shards mid-sweep.  Capping
    the dummy with hydrogen fixes it; this pins that EVERY released prompt
    yields a parseable reference, and that the fix is confined to the
    pathological cases -- for any fragment whose stripped form already parsed,
    the reference must be unchanged, or previously reported distances would
    have silently moved.
    """
    suite = _run_suite()
    changed = []
    for prompt in _prompts():
        stripped = ".".join(
            Chem.MolToSmiles(q, canonical=True) for q in suite.audit_queries(prompt)
        )
        reference = suite.prompt_reference_smiles(prompt)
        assert Chem.MolFromSmiles(reference) is not None, (
            prompt.drug_name,
            prompt.task.value,
            reference,
        )
        if stripped != reference:
            changed.append((prompt.drug_name, prompt.task.value))
            # It may only differ where the old form could not be used at all.
            assert Chem.MolFromSmiles(stripped) is None, (
                prompt.drug_name,
                prompt.task.value,
            )
    # LESINURAD (motif/linker/morphing) and MARIBAVIR (linker/morphing) attach at
    # an aromatic nitrogen. Nothing else may move.
    assert sorted(changed) == sorted(
        [
            ("LESINURAD", "motif_extension"),
            ("LESINURAD", "linker_design"),
            ("LESINURAD", "scaffold_morphing"),
            ("MARIBAVIR", "linker_design"),
            ("MARIBAVIR", "scaffold_morphing"),
        ]
    ), changed
