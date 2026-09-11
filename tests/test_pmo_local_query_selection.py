import numpy as np
import pytest

from tools.pmo_local_query_selection import partition, query_arms


def test_outcome_free_split_and_permutation_invariant_query_allocation():
    smiles = [f"fixture-{i}" for i in range(100)]
    split = partition(smiles)
    assert split == partition(smiles[::-1])
    assert set(split["train"]).isdisjoint(split["calibration"])
    assert set(split["train"]) | set(split["calibration"]) == set(smiles)
    prediction = np.arange(100)
    arms = query_arms(smiles, prediction)
    assert arms == query_arms(smiles[::-1], prediction[::-1])
    assert arms["predicted"] == smiles[-16:][::-1]
    assert len(set(arms["uniform"])) == 16
    with pytest.raises(ValueError, match="duplicate"):
        partition(smiles + smiles[:1])
    with pytest.raises(ValueError, match="nonfinite"):
        query_arms(smiles, np.full(100, np.nan))
