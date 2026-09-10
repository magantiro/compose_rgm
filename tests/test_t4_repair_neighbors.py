"""Small repair-census regressions; synthetic reference is not scientific evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_carbonyl_option import engineering_law, initial

from compose_v4.experiments.t4_repair_neighbors import (
    enumerate_products,
    recovery_counts,
    select_panel,
)
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.trace_shard import encode_state


def test_panel_is_deterministic_ineligible_and_canonical_unique():
    rows = [
        {"attempt_id": str(i), "smiles": s, "v": v}
        for i, (s, v) in enumerate((("C", 0.2), ("CC", 0.1), ("CCC", 0), ("CC", 0.1)))
    ]
    assert [r["smiles"] for r in select_panel(rows[::-1], 2)] == ["CC", "C"]
    with pytest.raises(ValueError, match="insufficient"):
        select_panel(rows, 3)
    with pytest.raises(ValueError, match="invalid constraint"):
        select_panel([{"v": float("nan"), "smiles": "C", "attempt_id": "0"}], 1)


def test_full_positive_mark_support_and_exact_witnesses_without_task_gate():
    graph, system = initial("CC").graph, editing_v2_semantic_rewrite_system()
    families, actions, _ = engineering_law(graph)
    valid = []
    for f, a in zip(families, actions, strict=True):
        try:
            system.apply(graph, f, a)
        except InvalidRewrite:
            continue
        valid.append((f, a))
    valid.append(valid[0])  # A syntactic duplicate must not count as a new molecule.
    row = tuple(f for f, a in valid), tuple(a for f, a in valid), (1 / len(valid),) * len(valid)
    result = enumerate_products(graph, lambda _: row, system)
    assert result["counts"]["positive_mass_marks"] == len(valid)
    assert len(result["products"]) < len(valid)
    assert len({r["smiles"] for r in result["products"]}) == len(result["products"])
    for product in result["products"]:
        for witness in product["witnesses"]:
            codec = action_codec_v4 if witness["schema_version"] == 4 else action_codec
            f, a = codec.decode_action(witness)
            replayed = system.apply(graph, f, a)
            assert canonical_state_key(replayed) == product["smiles"]
        witness = product["witnesses"][0]
        codec = action_codec_v4 if witness["schema_version"] == 4 else action_codec
        assert encode_state(system.apply(graph, *codec.decode_action(witness))) == product["state"]
    zero = enumerate_products(
        graph,
        lambda _: (row[0], row[1], (0.0,) + (1 / (len(valid) - 1),) * (len(valid) - 1)),
        system,
    )
    assert zero["counts"]["zero_mass_marks"] == 1
    assert enumerate_products(graph, lambda _: ((), (), ()), system)["products"] == []


def test_recovery_accounting_does_not_call_incumbent_novel():
    rows = [
        {"smiles": s, "v": v, "oracle_eligible": eligible}
        for s, v, eligible in (
            ("C", 0, True),
            ("CC", 0, True),
            ("CCC", 0, False),
            ("CCCC", 0.1, False),
        )
    ]
    counts = recovery_counts(rows, {"C"}, "C")
    assert counts == {
        "unique_products": 4,
        "benchmark_feasible": 3,
        "eligible": 2,
        "new_eligible": 1,
        "known_eligible": 1,
        "returns_to_incumbent": 1,
        "benchmark_failures": 1,
        "medchem_only_failures": 1,
    }


def test_fixed_contract_and_single_zero_oracle_launcher(monkeypatch, tmp_path):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    contract = json.loads(Path("configs/t4_repair_neighbors.json").read_text())
    digest = contract.pop("contract_sha256")
    assert (
        hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        == digest
    )
    calls = []

    def lookup(app, name):
        assert (app, name) == ("genmol-t4-opt", "t4_repair_neighbors")

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id="repair-call")

        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda _: "a" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **_: {"commit": "b" * 40, "image_revision_sha256": "c" * 64},
    )
    t4_launch.launch_continuation_profile({"commit": "b" * 40}, repair_neighbors=True)
    receipt = json.loads((tmp_path / "diagnostics/t4_repair_neighbors_spawn.json").read_text())
    assert len(calls) == 1 and receipt["oracle_call_limit"] == receipt["oracle_calls"] == 0
