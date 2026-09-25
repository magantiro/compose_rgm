"""The QED ablation launcher changes only the declared size-family restriction."""

from scripts.qed_h40_fixed_size_launch import build_tasks, load_contract


def test_contract_is_self_hashed_and_local_inputs_match():
    contract = load_contract()
    assert contract["source_panel"]["distinct_sources"] == 800
    assert contract["sampler"]["terminal_returns_per_source"] == 8


def test_only_fixed_tasks_receive_size_restriction():
    contract = load_contract()
    full_tasks = build_tasks("recover", contract)
    fixed_tasks = build_tasks("fixed", contract)
    assert [task["index"] for task in full_tasks] == [135, 408]
    assert all("size_fixed" not in task for task in full_tasks)
    assert len(fixed_tasks) == 800
    assert [task["index"] for task in fixed_tasks] == list(range(800))
    for task in fixed_tasks:
        assert task["size_fixed"] is True
        assert task["arm"] == "restart"
        assert task["horizon"] == 40
        assert task["budget_max"] == 24
        assert task["head_dir"] == "hphi_v2"
        assert task["k_start"] == 0 and task["k_end"] == 8
