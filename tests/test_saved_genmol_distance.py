import pytest

from tools.score_saved_fragment_genmol_distance import evaluate, official_function


def test_distance_uses_unique_outputs_but_retains_attempt_counts():
    def metric(reference, frame):
        assert reference == "CCO"
        assert frame["smiles"].tolist() == ["CCO", "CCC"]
        return 0.25

    assert evaluate(["CCO", "CCO", "", "CCC"], "CCO", metric) == {
        "attempts": 4,
        "outputs": 3,
        "unique_valid": 2,
        "distance": 0.25,
    }


def test_empty_prompt_is_explicit_not_zero():
    assert evaluate(["", ""], "CCO", None)["distance"] is None


def test_unpinned_source_cannot_execute(tmp_path):
    source = tmp_path / "metric.py"
    source.write_text("raise AssertionError('must never execute')")
    with pytest.raises(ValueError, match="pinned"):
        official_function(source)
