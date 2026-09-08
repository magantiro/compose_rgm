import pytest

from tools.t4_ring_feasibility import Properties, fingerprint_accounting


def test_fingerprint_loss_separates_shared_loss_from_union_growth():
    result = fingerprint_accounting({1, 2, 3}, {1, 2, 4}, {2, 4, 5})
    assert result["similarity_before"] == 0.5
    assert result["similarity_after"] == 0.2
    assert result["lost_shared_bits"] == [1]
    assert result["new_nonseed_bits"] == [5]
    with pytest.raises(ValueError, match="nonempty"):
        fingerprint_accounting(set(), {1}, {2})


def test_terminal_properties_use_original_seed_and_existing_t4_violation():
    props = Properties("c1ccccc1")
    before, after = props("c1ccccc1"), props("C1CCNCC1")
    assert before["sim"] == 1
    accounting = fingerprint_accounting(
        props.seed_fp.GetOnBits(), props.bits("c1ccccc1"), props.bits("C1CCNCC1")
    )
    assert accounting["similarity_after"] == after["sim"]
    assert after["v"] > 0
    assert props("C1CCNCC1") is after
