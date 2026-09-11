from tools.pmo_persistent_lookahead import best_first, continuation_tasks


def test_worse_first_moves_survive_lookahead_and_matching_tasks_share_streams():
    root = {"smiles": "root", "score": 0.7}
    first = [{"smiles": "lower", "score": 0.5}, None, {"smiles": "higher", "score": 0.8}, None]
    selected = best_first(root, first)
    assert selected == (2, first[2])
    plans = continuation_tasks(0, first, selected)
    assert len(plans["immediate"]) == len(plans["lookahead"]) == 8
    assert plans["lookahead"][0][3] == first[0]
    assert plans["lookahead"][2][3] is None
    assert plans["immediate"][:2] == plans["lookahead"][4:6]
    assert best_first(root, [first[0], None]) == (-1, root)
