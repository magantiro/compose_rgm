import pytest

from compose_v4.experiments.pmo_online_policy import compare_laws


def test_runtime_admission_measures_roundoff_but_rejects_changed_support_or_mass():
    reference = {"marks": [{"action": "a"}, {"action": "b"}], "probabilities": [0.3, 0.7]}
    limits = {"max_absolute_error": 1e-7, "max_half_l1": 1e-6}
    actual = list(zip(reference["marks"], reference["probabilities"]))
    assert compare_laws(actual, reference, limits)["exact"]
    near = [(actual[0][0], 0.3 + 1e-9), (actual[1][0], 0.7 - 1e-9)]
    report = compare_laws(near, reference, limits)
    assert not report["exact"] and report["max_absolute_error"] < 1e-8
    with pytest.raises(ValueError, match="support/order"):
        compare_laws(list(reversed(actual)), reference, limits)
    with pytest.raises(ValueError, match="failed prospective"):
        compare_laws([(actual[0][0], 0.5), (actual[1][0], 0.5)], reference, limits)
