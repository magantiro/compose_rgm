from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from compose_v4.experiments.graph_ga_adapter import (
    UPSTREAM_DIR,
    UPSTREAM_ZINC_SIZE_PRIOR,
    expected_scoring_calls,
    run_graph_ga,
    size_prior_from_sources,
)

pytest.importorskip("rdkit")

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SOURCES = [
    "Cc1ccccc1-c1noc(-c2cc3ccccc3oc2=O)n1",
    "COc1ccc(-c2nc(CS(=O)(=O)CC(=O)NCc3ccccc3Br)c(C)o2)cc1",
    "C=CC=C(C=C)CCC(=O)N1C(=O)C(C(C)O)C1OC(C)=O",
]

#: Byte-identical to upstream commit 4b49f182. If these change, the vendored
#: algorithm is no longer upstream and PROVENANCE.md is a false claim.
UPSTREAM_SHA256 = {
    "GB_GA.py": "fd4adcd40523263abb3be38a7ee70e0766be5326048c812f157cabd949f7a71e",
    "crossover.py": "6f1e84da39ce270d2a8981041eb6227ac1557d029f6b0290b1c95d8cda7d91ea",
    "mutate.py": "0839dd392ae0c38022b3df0b742037ae76a704d9c72a058a33bd1e3f9fe17e54",
}


def _logp(smiles: str) -> float:
    from rdkit.Chem import Crippen

    return float(Crippen.MolLogP(Chem.MolFromSmiles(smiles)))


def test_vendored_upstream_is_unmodified() -> None:
    for name, expected in UPSTREAM_SHA256.items():
        digest = hashlib.sha256((UPSTREAM_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, f"{name} no longer matches upstream 4b49f182"


def test_nonnegative_transform_must_be_declared() -> None:
    """GB-GA's selection cannot accept a negative objective; the fix is a deviation."""
    with pytest.raises(ValueError, match="nonnegative_transform"):
        run_graph_ga(
            sources=SOURCES,
            objective=_logp,
            seed_path=Path("/tmp/gbga_test_seed.smi"),
            budget=50,
            budget_counter="unique_valid_canonical_evaluations",
            nonnegative_transform="whatever",
        )


def test_constant_shift_requires_a_positive_constant() -> None:
    with pytest.raises(ValueError, match="constant_shift"):
        run_graph_ga(
            sources=SOURCES,
            objective=_logp,
            seed_path=Path("/tmp/gbga_test_seed.smi"),
            budget=50,
            budget_counter="unique_valid_canonical_evaluations",
            nonnegative_transform="constant_shift",
        )


def test_size_prior_defaults_to_the_panel_not_upstream_zinc() -> None:
    """The size prior is a fairness parameter, so it must track the panel."""
    average, stdev = size_prior_from_sources(SOURCES)
    assert 15.0 < average < 40.0
    assert average != UPSTREAM_ZINC_SIZE_PRIOR[0]
    assert stdev > 0.0


def test_real_upstream_run_reproduces_the_published_accounting_identity() -> None:
    population_size, generations = 10, 2
    result = run_graph_ga(
        sources=SOURCES,
        objective=_logp,
        seed_path=Path("/tmp/gbga_test_seed.smi"),
        budget=10_000,
        budget_counter="unique_valid_canonical_evaluations",
        population_size=population_size,
        generations=generations,
        seed=7,
        nonnegative_transform="upstream_clamp",
    )
    # The paper: "The population size is 20 and 50 generations are used
    # (i.e. 1000 J(m) evaluations per run)" -- initial population plus each
    # generation's offspring.
    assert result.counters["oracle_requests"] == expected_scoring_calls(
        population_size, result.generations_run
    )
    assert all(result.accountant_manifest["invariants"].values())
    assert not result.wall_clock_exceeded


def test_budget_stops_the_real_algorithm_cleanly() -> None:
    result = run_graph_ga(
        sources=SOURCES,
        objective=_logp,
        seed_path=Path("/tmp/gbga_test_seed.smi"),
        budget=5,
        budget_counter="unique_valid_canonical_evaluations",
        population_size=10,
        generations=5,
        seed=7,
        nonnegative_transform="upstream_clamp",
    )
    # Upstream has no budget concept, so exhaustion unwinds out of GA(); the
    # adapter must surface that as a clean flag rather than an exception.
    assert result.budget_exhausted
    assert result.counters["unique_valid_canonical_evaluations"] <= 5
