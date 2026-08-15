"""The initialization contract: sealed, disjoint, and the same every time."""

from __future__ import annotations

import hashlib
import json

import pytest

from compose_v4.benchmark.init_sets import (
    DEFAULT_INIT_DIR,
    DEVELOPMENT_SEEDS,
    OFFICIAL_SEEDS,
    POPULATION,
    OfficialRunNotAuthorised,
    development_init_set,
    init_provenance,
    official_init_set,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_INIT_DIR / "official_init_sets.json").exists(),
    reason="init sets not frozen; run freeze_molleo_task3_init_sets.py")


def test_reaching_for_the_official_sets_by_accident_fails():
    """The official five-seed run is spent once. Make it impossible to spend by
    typing the wrong function name."""
    with pytest.raises(OfficialRunNotAuthorised, match="sealed until the policy"):
        official_init_set(0)


def test_every_set_is_the_benchmark_size():
    for seed in DEVELOPMENT_SEEDS:
        assert len(development_init_set(seed)) == POPULATION
    for seed in OFFICIAL_SEEDS:
        assert len(official_init_set(seed, official=True)) == POPULATION


def test_no_development_molecule_appears_in_an_official_set():
    """This is what makes development data disjoint. A policy tuned on
    development never saw an official starting molecule."""
    official = {smiles for seed in OFFICIAL_SEEDS
                for smiles in official_init_set(seed, official=True)}
    development = {smiles for seed in DEVELOPMENT_SEEDS
                   for smiles in development_init_set(seed)}
    assert not (official & development)


def test_the_molecules_within_a_set_are_distinct():
    for seed in DEVELOPMENT_SEEDS:
        molecules = development_init_set(seed)
        assert len(set(molecules)) == len(molecules)


def test_the_official_sets_still_hash_to_what_was_sealed():
    """A redraw after seeing a result would change these digests."""
    data = json.loads((DEFAULT_INIT_DIR / "official_init_sets.json").read_text())
    for seed, molecules in data["sets"].items():
        digest = hashlib.sha256("\n".join(molecules).encode()).hexdigest()
        assert digest == data["sets_sha256"][seed]


def test_every_molecule_parses_and_is_already_canonical():
    """The pool is canonicalised at draw time so that later membership checks
    are string comparisons that cannot disagree with the oracle's canonicaliser."""
    from compose_v4.benchmark.oracles import canonical

    for seed in (DEVELOPMENT_SEEDS[0], OFFICIAL_SEEDS[0]):
        molecules = (development_init_set(seed) if seed in DEVELOPMENT_SEEDS
                     else official_init_set(seed, official=True))
        for smiles in molecules[:40]:
            assert canonical(smiles) == smiles


def test_provenance_names_the_pool_and_says_the_official_sets_are_sealed():
    provenance = init_provenance()
    assert provenance["population"] == POPULATION
    assert "SEALED" in provenance["official_status"]
    assert len(provenance["pool_sha256"]) == 64
