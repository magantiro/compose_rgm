"""Zero-network boundary and exact-path tests for the production audit driver."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
from test_fused_option import fixture_law, kernel, source

from compose_v4.control.option_continuation import sample_option_trajectory
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.fused_reference_profile import (
    CONTRACT_PATH,
    initial_state,
    load_contract,
    run_reference_profile,
)
from compose_v4.rewrite.kernel import editing_v2_rewrite_system

ROOT = Path(__file__).resolve().parents[1]


def profile(path, budget=32, law=fixture_law):
    return run_reference_profile(
        source(),
        law,
        editing_v2_rewrite_system(),
        {"seed": 0, "max_executor_applications": budget},
        path,
        snapshot_id="fixture-not-Rtheta",
        commit_volume=lambda: None,
        progress={},
    )


def test_reference_driver_matches_the_existing_reference_sampler(tmp_path):
    recorded = profile(tmp_path)

    def forbidden(_):
        pytest.fail("reference sampler evaluated a guidance callback")

    baseline = sample_option_trajectory(
        source(),
        kernel(),
        forbidden,
        forbidden,
        seed=0,
        snapshot_id="fixture-not-Rtheta",
        max_expansions=0,
        max_terminal_evaluations=0,
        estimator="reference",
    )
    assert recorded["status"] == baseline["status"] == "complete"
    assert recorded["endpoint"] == baseline["endpoint"]
    assert [s["probability"] for s in recorded["trace"]] == [
        s["probability"] for s in baseline["trace"]
    ]
    assert [s["successor"]["graph"] for s in recorded["trace"]] == [
        s["exact_state"] for s in baseline["trace"]
    ]
    assert recorded["conditional_augmented_path_log_probability"] == pytest.approx(
        baseline["conditional_path_logq"]
    )
    assert recorded["independent_topology_witness"]["passed"]
    assert recorded["oracle_calls"] == recorded["terminal_objective_calls"] == 0


def test_complete_rows_paths_and_rng_are_durable(tmp_path):
    result = profile(tmp_path)
    for receipt in result["rows"] + result["laws"]:
        assert sha256_file(tmp_path / receipt["path"]) == receipt["sha256"]
    for step in result["trace"]:
        row = json.loads((tmp_path / step["row_path"]).read_text())
        assert row["source"] == json.loads(json.dumps(step["source"]))
        assert row["successors"][step["selected_index"]] == json.loads(
            json.dumps(step["successor"])
        )
        assert sum(row["probabilities"]) == pytest.approx(1.0)
        assert step["kl"] == 0
    path = json.loads((tmp_path / "sampled_path.json").read_text())
    assert path["sampling_complete"]
    assert path["rng_state"]["bit_generator"] == "PCG64"
    assert path["current_state"]["step"] == 5
    attempts = [
        a
        for p in (tmp_path / "attempts").glob("*.json")
        for a in json.loads(p.read_text())["attempts"]
    ]
    assert sorted(a["call_index"] for a in attempts) == list(
        range(result["total_public_executor_calls"])
    )


@pytest.mark.parametrize("budget", [0, 1, 6, 7])
def test_budget_failure_preserves_complete_units_not_a_partial_law(tmp_path, budget):
    result = profile(tmp_path, budget)
    assert result["status"] == "executor_budget_exhausted"
    assert result["endpoint"] is None
    assert result["total_public_executor_calls"] == budget
    if budget < 6:
        assert result["rows"] == []
    else:
        path = json.loads((tmp_path / "sampled_path.json").read_text())
        assert not path["sampling_complete"]
        assert len(path["trace"]) == budget - 5


def test_invalid_law_is_not_published(tmp_path):
    def invalid(graph):
        families, actions, _ = fixture_law(graph)
        return families, actions, [float("nan")] * len(actions)

    with pytest.raises(ValueError, match="malformed"):
        profile(tmp_path, law=invalid)
    assert not (tmp_path / "laws").exists()


def test_frozen_contract_and_source_use_applicability_not_docking():
    contract = load_contract(ROOT / CONTRACT_PATH)
    seeds = json.loads((ROOT / contract["source_manifest"]).read_text())
    node, selection = initial_state(contract, seeds)
    assert node.graph.n_real_atoms == 19
    assert node.remaining == 5
    assert node.option == "build_fused_ring"
    assert selection["is_qm_sample"] is False
    assert selection["eligible_oriented_edges"]
    assert selection["intended_r_release"] == len(node.context.locus) / 19
    assert all(set(edge) <= node.context.locus for edge in selection["eligible_oriented_edges"])
    poisoned_scores = [{**s, "published_ds": -9999, "seed_qed": 0, "seed_sa": 999} for s in seeds]
    other, other_selection = initial_state(contract, poisoned_scores)
    assert other.key() == node.key()
    assert other_selection == selection


def test_contract_rejects_bad_self_hash_and_resealed_docking(tmp_path):
    value = load_contract(ROOT / CONTRACT_PATH)
    path = tmp_path / "contract.json"
    value["oracle_calls"] = 20
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="self-hash"):
        load_contract(path)
    body = {k: v for k, v in value.items() if k != "contract_sha256"}
    value["contract_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="oracle_calls"):
        load_contract(path)


def test_launch_surface_is_one_cpu_task_with_no_oracle_driver():
    tree = ast.parse((ROOT / "modal_apps/genmol_t4_opt_app.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_reference_profile"
    )
    decorator = fn.decorator_list[0]
    keywords = {k.arg: k.value for k in decorator.keywords}
    assert ast.literal_eval(keywords["cpu"]) == (1.0, 1.0)
    assert ast.literal_eval(keywords["memory"]) == 6144
    assert ast.literal_eval(keywords["max_containers"]) == 1
    assert ast.literal_eval(keywords["retries"]) == 0
    assert ast.literal_eval(keywords["timeout"]) == 900
    calls = {
        n.func.id
        for statement in fn.body
        for n in ast.walk(statement)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert calls == {"run_remote_task"}


def test_runtime_gate_failure_stops_before_model_enumeration(tmp_path):
    from compose_v4.experiments.fused_reference_profile import run_remote_task

    repo, volume_root = tmp_path / "repo", tmp_path / "volume"
    contract = load_contract(ROOT / CONTRACT_PATH)
    (repo / "configs").mkdir(parents=True)
    (repo / "modal_apps").mkdir()
    (repo / "docs").mkdir()
    (repo / contract["source_manifest"]).write_bytes(
        (ROOT / contract["source_manifest"]).read_bytes()
    )
    app = repo / "modal_apps/genmol_t4_opt_app.py"
    app.write_text("# synthetic boundary fixture, not a deployed app\n")
    for key in ("run_paths", "checkpoint"):
        path = tmp_path / key
        path.write_text("synthetic boundary fixture")
        contract[key], contract[f"{key}_sha256"] = str(path), sha256_file(path)
    body = {k: v for k, v in contract.items() if k != "contract_sha256"}
    contract["contract_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    contract_path = repo / CONTRACT_PATH
    contract_path.write_text(json.dumps(contract))
    revision = {"commit": "a" * 40, "image_revision_sha256": "b" * 64}
    identity = {
        "app_sha256": sha256_file(app),
        "contract_sha256": sha256_file(contract_path),
        "image_revision_sha256": revision["image_revision_sha256"],
    }
    run_id = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    task = {
        "image_revision": revision,
        "run_id": run_id,
        "app_sha256": identity["app_sha256"],
        "contract_sha256": identity["contract_sha256"],
    }
    events = []

    class Volume:
        def commit(self):
            events.append("commit")

    def refused_runtime():
        events.append("runtime")
        raise ValueError("frozen catalog mismatch, deliberately not bypassed")

    with pytest.raises(ValueError, match="frozen catalog mismatch"):
        run_remote_task(
            task,
            repo_root=repo,
            artifact_root=volume_root,
            volume=Volume(),
            runtime_factory=refused_runtime,
            validate_revision=lambda _: events.append("identity"),
        )
    output = volume_root / "fused_reference_profile" / run_id
    assert json.loads((output / "failure.json").read_text())["error_type"] == "ValueError"
    assert not (output / "laws").exists()
    assert not (output / "result.json").exists()
    assert events[0] == "identity"


@pytest.mark.parametrize("fused", [False, True])
def test_launcher_spawns_only_the_selected_profile(tmp_path, monkeypatch, fused):
    import sys
    from types import SimpleNamespace

    from tools import t4_launch

    kind = "fused_reference_profile" if fused else "continuation_profile"
    (tmp_path / "configs").mkdir()
    (tmp_path / "modal_apps").mkdir()
    (tmp_path / "diagnostics").mkdir()
    config = CONTRACT_PATH if fused else "configs/continuation_profile_v1.json"
    (tmp_path / config).write_bytes((ROOT / config).read_bytes())
    (tmp_path / "modal_apps/genmol_t4_opt_app.py").write_text("# launch boundary fixture\n")
    revision = {"commit": "a" * 40, "image_revision_sha256": "b" * 64}
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "modal_apps.run_process_v2_p50_app",
        SimpleNamespace(local_image_revision=lambda **_: revision),
    )
    calls = []

    def lookup(app, name):
        calls.append((app, name))

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id="fc-fixture")

        return SimpleNamespace(spawn=spawn)

    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    t4_launch.launch_continuation_profile({"commit": revision["commit"]}, fused=fused)
    assert len(calls) == 2
    assert calls[0] == ("genmol-t4-opt", kind)
    receipt = json.loads((tmp_path / f"diagnostics/{kind}_spawn.json").read_text())
    assert receipt["task"] == calls[1]
    assert receipt["oracle_calls"] == 0
    assert receipt["call_id"] == "fc-fixture"
    assert receipt["volume_path"] == f"/{kind}/{calls[1]['run_id']}"
