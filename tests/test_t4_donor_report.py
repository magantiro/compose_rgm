import pytest

from tools.t4_donor_report import validate_locked_row


def test_only_property_roundoff_is_tolerated_not_selection_or_molecules():
    expected = {"qed": 0.6023412239033229, "oracle_eligible": True, "smiles": "CC"}
    actual = {**expected, "qed": 0.6023412239033228}
    validate_locked_row(expected, actual)
    for changed in ({"qed": 0.59}, {"oracle_eligible": False}, {"smiles": "CCC"}):
        with pytest.raises(ValueError):
            validate_locked_row(expected, {**actual, **changed})
