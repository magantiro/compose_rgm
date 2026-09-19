from compose_v4.experiments.t4_integrated_route_fiber_v2 import (
    EXPERTS,
    expert_census,
    merge_expert_pools,
)


def _row(smiles, lane, score=-8.0):
    return {
        "smiles": smiles,
        "proposal_lane": lane,
        "parent": "CC",
        "parent_score": score,
        "families": [lane],
        "program_families": [lane],
    }


def test_v2_union_keeps_retained_core_provenance():
    pools = {name: [] for name in EXPERTS}
    pools["shallow"] = [_row("CCO", "shallow")]
    pools["retained_core_prune"] = [_row("CCO", "retained_core_prune", -9.0)]
    merged = merge_expert_pools(pools)
    assert len(merged) == 1
    assert merged[0]["proposal_experts"] == ["retained_core_prune", "shallow"]
    census = expert_census(merged)
    assert census["retained_core_prune"] == 1
    assert census["multi_expert"] == 1
