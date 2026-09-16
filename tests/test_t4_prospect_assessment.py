"""Guards for free-label assessment of a prospected pool.

The dangerous failure here is silent: if the pool's endpoint identity disagrees with
the corpus's, every lookup misses, zero rediscoveries are reported, and that reads as
"the pool found nothing known" rather than "the join is broken".
"""

from __future__ import annotations

import pytest

from compose_v4.experiments.t4_proposal_prior_dataset import _text_sha256
from compose_v4.experiments.t4_prospect_assessment import (
    endpoint_sha256,
    free_labels,
    shape_comparison,
)


def _record(endpoint, score, cell="parp1_0", primitives=20):
    return {
        "cell": cell,
        "endpoint_sha256": endpoint_sha256(endpoint),
        "score_mean": score,
        "primitive_count": primitives,
        "changed_slot_count": 6,
        "net_created": 14,
    }


def _row(endpoint, primitives=20):
    return {
        "endpoint": endpoint,
        "primitives": primitives,
        "families": ["append_ring"],
        "changed_originals": 6,
        "created": 14,
    }


def test_the_pool_and_the_corpus_agree_on_endpoint_identity():
    """A silent join mismatch would report zero rediscoveries and look like a result."""
    for smiles in ("CCO", "c1ccccc1O", "CC(=O)Nc1ccccc1"):
        assert endpoint_sha256(smiles) == _text_sha256(smiles)


def test_a_rediscovered_endpoint_carries_its_historical_score():
    result = free_labels([_row("CCO")], [_record("CCO", -13.2)], cell="parp1_0")
    assert result["rediscovered"] == 1
    assert result["best_rediscovered_score"] == -13.2
    assert result["unscored_members"] == 0
    assert result["new_oracle_calls"] == 0


def test_the_best_of_several_historical_scores_for_one_molecule_is_used():
    records = [_record("CCO", -9.0), _record("CCO", -13.2), _record("CCO", -11.0)]
    assert free_labels([_row("CCO")], records, cell="parp1_0")["best_rediscovered_score"] == -13.2


def test_a_score_from_another_cell_is_never_borrowed():
    result = free_labels([_row("CCO")], [_record("CCO", -13.2, cell="jak2_1")], cell="parp1_0")
    assert result["rediscovered"] == 0 and result["best_rediscovered_score"] is None
    assert result["unscored_members"] == 1


def test_an_unscored_pool_reports_no_label_rather_than_a_zero():
    result = free_labels([_row("CCN"), _row("CCC")], [_record("CCO", -13.2)], cell="parp1_0")
    assert result["rediscovered"] == 0
    assert result["median_rediscovered_score"] is None
    assert result["rediscovery_rate"] == 0.0


def test_an_empty_pool_does_not_divide_by_zero():
    result = free_labels([], [_record("CCO", -13.2)], cell="parp1_0")
    assert result["pool_size"] == 0 and result["rediscovery_rate"] == 0.0


def test_shape_comparison_abstains_without_a_reference():
    assert shape_comparison([_row("CCO")], [], cell="parp1_0")["comparable"] is False
    assert shape_comparison([], [_record("CCO", -13.2)], cell="parp1_0")["comparable"] is False


def test_shape_comparison_reports_both_sides_and_calls_itself_shape():
    result = shape_comparison([_row("CCO", primitives=21)], [_record("CCN", -14.3)], cell="parp1_0")
    assert result["comparable"] is True
    assert result["pool"]["median_primitives"] == pytest.approx(21)
    assert result["historical_best"]["median_primitives"] == pytest.approx(20)
    assert result["reference_best_score"] == -14.3
    assert "not evidence of scoring well" in result["interpretation"]
