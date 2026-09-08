import pytest

from tools.t4_ring_feasibility import Properties, fingerprint_accounting, refinement_start


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


def test_refinement_continues_exact_saved_birth_lineage():
    from pathlib import Path

    from compose_v4.control.ring_program import RingProgress
    from compose_v4.experiments.t4_matched_pilot import unseal
    from tools.t4_ring_feasibility import saved_bundle

    root = Path(__file__).resolve().parents[1] / "diagnostics/t4_ring_program_round/attempt_1/run"
    if not (root / "warm_start.json").exists():
        pytest.skip("vendored 2026-09-08 exact T4 ring audit is absent")
    warm = unseal(root / "warm_start.json")
    lock = unseal(root / "round_4/candidate_lock.json")
    bundle = next(b for b in lock["bundles"] if b["option"].startswith("construct:"))
    initial, _ = saved_bundle(lock, warm, bundle, Properties(warm["archive"][0]["smiles"]))
    endpoint = next(
        t
        for w in lock["work"]
        for t in w["sampled_transitions"]
        if t["bundle_id"] == initial.bundle_id and t["step"] == initial.horizon
    )
    progress = RingProgress.from_payload(endpoint["ring_progress"])
    node = refinement_start(initial, endpoint["product"], progress)
    assert node.remaining == 2
    assert node.bundle_id == initial.bundle_id
    assert node.context.frozen == initial.context.frozen
    assert node.graph.n_real_atoms == initial.graph.n_real_atoms + len(progress.path)
    assert all(node.lineage.id_of[s] == i for s, i in initial.lineage.id_of.items())
    assert [node.lineage.id_of[s] for s in progress.path] == list(
        range(initial.lineage.next_id, initial.lineage.next_id + len(progress.path))
    )
