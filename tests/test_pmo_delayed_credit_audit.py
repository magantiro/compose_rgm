from copy import deepcopy

from compose_v4.experiments.pmo_delayed_credit_audit import (
    attach_returns,
    leakage_groups,
    option_family,
)


def _row(identifier, boundary, parent_score, score, chain, root="root-a"):
    return {
        "run_id": "run",
        "arm": "arm",
        "boundary": boundary,
        "id": identifier,
        "root_id": root,
        "parent_smiles": "CC",
        "product_smiles": "CCC",
        "parent_score": parent_score,
        "score": score,
        "chain": chain,
    }


def test_returns_credit_a_temporary_loss_only_after_recovery():
    rows = [
        _row("one", 1, 0.5, 0.3, ["one"]),
        _row("two", 2, 0.3, 0.4, ["one", "two"]),
        _row("three", 3, 0.4, 0.7, ["one", "two", "three"]),
    ]
    attach_returns(rows, [1, 2])
    assert rows[0]["returns"] == {"1": 0.5, "2": 0.7}
    assert rows[1]["returns"] == {"1": 0.7, "2": 0.7}


def test_returns_never_import_another_arm_lineage():
    rows = [
        _row("one", 1, 0.5, 0.3, ["one"]),
        {**_row("two", 2, 0.3, 0.9, ["one", "two"]), "arm": "other"},
    ]
    attach_returns(rows, [2])
    assert rows[0]["returns"]["2"] == 0.5


def test_leakage_groups_join_roots_that_share_any_state():
    left = _row("one", 1, 0.5, 0.4, ["one"], root="left")
    right = deepcopy(left)
    right.update(root_id="right", id="two", chain=["two"], product_smiles="CCCC")
    separate = deepcopy(left)
    separate.update(
        root_id="separate",
        id="three",
        chain=["three"],
        parent_smiles="N",
        product_smiles="NN",
    )
    groups = leakage_groups([left, right, separate])
    assert groups["left"] == groups["right"]
    assert groups["left"] != groups["separate"]


def test_option_family_preserves_channel_and_ring_attachment_class():
    assert option_family("construct:pendant:6:5,1,0:aromatic:0") == "construct_pendant"
    assert (
        option_family("replace_region:construct:fused:5:4,1,0:nonaromatic:0")
        == "replace_region_construct_fused"
    )
    assert option_family("generic") == "generic"
