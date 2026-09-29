"""Offline paired-law, candidate-lock and real-executor integration checks."""

import ast
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_fused_option import fixture_law

from compose_v4.control import option_selector, region_rewrite
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.experiments.continuation_profile import canonical_bytes
from compose_v4.experiments.t4_matched_pilot import (
    ARMS,
    PLAN_FIELDS,
    primitive_distribution,
    run_pair,
    seal,
    unseal,
    verify_pair,
)
from compose_v4.rewrite.kernel import editing_v2_rewrite_system

ROOT = Path(__file__).resolve().parents[1]


def test_reference_and_guided_laws_match_qualified_arithmetic():
    p, h = np.array([0.7, 0.29, 0.01]), np.array([0.1, 0.4, 0.9])
    assert np.array_equal(primitive_distribution(p, None, "reference"), p)
    expected, *_ = region_rewrite.kl_tilt(p, h, kappa=1)
    expected = 0.9 * expected + 0.1 * p
    expected /= expected.sum()
    q = primitive_distribution(p, h, "committor")
    assert np.array_equal(q, expected)
    assert np.sum(q * np.log(q / p)) <= 1 + 1e-12
    assert np.all(q >= 0.1 * p - 1e-12)
    assert not np.array_equal(p, q)


@pytest.mark.parametrize(
    "p,h,arm,kappa",
    [
        ([0.5], [1], "committor", 1),
        ([1], [1], "typo", 1),
        ([1], [1], "reference", 2),
        ([1], [float("nan")], "committor", 1),
    ],
)
def test_invalid_law_fails(p, h, arm, kappa):
    with pytest.raises(ValueError):
        primitive_distribution(p, h, arm, kappa)


def fixture_lock(task):
    bundle = dict.fromkeys(PLAN_FIELDS, None)
    bundle.update(bundle_id="fixture-bundle", option="generic")
    return {
        "task": task,
        "round": 1,
        "oracle_calls": 0,
        "input_sha256": {"fixture": "not-a-model"},
        "bundles": [bundle],
        "take": [{"smiles": "CCC", "v": 0, "option": "generic", "program_complete": True}],
    }


def test_both_locks_precede_oracle_and_completed_result_is_reused(tmp_path):
    task = {"smiles": "CC", "max_executor_applications": 20}
    calls = []

    def prepare(t, checkpoint, cached):
        calls.append(("prepare", t["arm"]))
        return fixture_lock(t)

    def dock(smiles, arm):
        assert all((tmp_path / a / "candidate_lock.json").exists() for a in ARMS)
        assert (tmp_path / "oracle_barrier.json").exists()
        calls.append(("dock", arm))
        return [-8.0]

    result = run_pair(task, prepare, dock, tmp_path)
    assert calls == [("prepare", a) for a in ARMS] + [("dock", a) for a in ARMS]
    assert result["arms"]["committor"]["best"]["ds"] == -8
    calls.clear()
    assert run_pair(task, prepare, dock, tmp_path) == result
    assert calls == []


def test_plan_mismatch_never_spends_oracle(tmp_path):
    def prepare(t, *_):
        lock = fixture_lock(t)
        lock["bundles"][0]["q_option"] = 0.1 if t["arm"] == "reference" else 0.2
        return lock

    with pytest.raises(ValueError, match="identical region/option"):
        run_pair(
            {"max_executor_applications": 20},
            prepare,
            lambda *_: pytest.fail("oracle called before paired validation"),
            tmp_path,
        )


@pytest.mark.parametrize("defect", ["duplicates", "incomplete", "second_round", "input_hash"])
def test_candidate_lock_gates_fail(defect):
    locks = {a: fixture_lock({}) for a in ARMS}
    bad = locks["committor"]
    if defect == "duplicates":
        bad["take"] *= 2
    elif defect == "incomplete":
        bad["take"][0].update(option="build_fused_ring", program_complete=False)
    elif defect == "second_round":
        bad["round"] = 2
    else:
        bad["input_sha256"] = {"fixture": "different"}
    with pytest.raises(ValueError):
        verify_pair(locks)


def test_failed_prepare_abstains_and_is_not_retried(tmp_path):
    calls = []

    def fail(*_):
        calls.append("prepare")
        raise ContinuationBudgetExceeded("fixture exhausted")

    with pytest.raises(ContinuationBudgetExceeded):
        run_pair(
            {"max_executor_applications": 0},
            fail,
            lambda *_: pytest.fail("docking after failed preparation"),
            tmp_path,
        )
    with pytest.raises(RuntimeError, match="audit before retry"):
        run_pair({"max_executor_applications": 0}, fail, None, tmp_path)
    assert calls == ["prepare"]


def test_corrupt_complete_unit_is_not_recomputed(tmp_path):
    path = tmp_path / "unit.json"
    seal(path, {"a": 1})
    content = json.loads(path.read_text())
    content["payload"]["a"] = 2
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="corrupt"):
        unseal(path)


def test_frozen_config_self_hash_and_scope():
    config = json.loads((ROOT / "configs/t4_matched_pilot.json").read_text())
    identity = config.pop("contract_sha256")
    assert hashlib.sha256(canonical_bytes(config)).hexdigest() == identity
    assert config["training_authorized"] is False
    assert config["task"]["budget"] == 20
    assert config["task"]["kappa"] == 1
    assert config["task"]["workers"] == 1
    assert config["compute"]["oracle_call_limit"] == 40


def population_fixture(monkeypatch, tmp_path):
    """Execute the real app body; replace only remote I/O, model law and draws.

    No copied proposal logic. This explicitly synthetic benzene fixture forces
    a fused draw to cover a path a small random pilot might never select.
    """
    source = ROOT / "modal_apps/genmol_t4_opt_app.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "t4_population_cell"
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    # Hash I/O is tested at the paired boundary; no production asset substitution.
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "sha256_file":
            child.body = [ast.Return(ast.Constant("explicit-fixture-hash"))]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = {
        "Any": object,
        "Path": Path,
        "json": json,
        "time": time,
        "REMOTE_ROOT": tmp_path,
        "MAX_ACTIVE_ATOMS": 40,
        "CANONICAL_SLOTS": 48,
        "QED_MIN": 0.6,
        "SA_MAX": 4,
        "TIME_POINT": 0.5,
        "artifact_volume": SimpleNamespace(reload=lambda: None),
        "_runtime": lambda: {
            "model": None,
            "system": editing_v2_rewrite_system(),
            "model_checkpoint": "fixture",
            "run_paths": "fixture",
        },
        "_dock_many": lambda *_: pytest.fail("prepare-only called docking"),
    }
    exec(compile(module, str(source), "exec"), ns)  # noqa: S102 - trusted repository AST only
    from compose_v4.experiments import production_successor_kernel as prod

    def law(_model, graph, _time):
        f, a, p = fixture_law(graph)
        return SimpleNamespace(
            marks=[
                SimpleNamespace(executor_rule_name=fam, action=act, probability=pr)
                for fam, act, pr in zip(f, a, p)
            ]
        )

    monkeypatch.setattr(prod, "enumerate_factorized_marked_law", law)
    n_features = len(region_rewrite.STRUCTURAL_FEATURES)
    net = torch.nn.Sequential(
        torch.nn.Linear(n_features, 48), torch.nn.ReLU(), torch.nn.Linear(48, 1)
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {"n_features": n_features, "state_dict": net.state_dict()},
    )

    def forced_fused(options, _rng, **_kwargs):
        assert "generic" in options
        assert "build_fused_ring" in options
        return option_selector.OptionChoice(
            "build_fused_ring", tuple(options), tuple(1 / len(options) for _ in options)
        )

    monkeypatch.setattr(option_selector, "sample_option", forced_fused)

    def isolated_prepare(*args, **kwargs):
        # The app runs in a dedicated inference container, where disabling
        # autograd process-wide is safe. This fixture executes that same body
        # inside pytest, so restore the caller's gradient mode afterward.
        grad_enabled = torch.is_grad_enabled()
        try:
            return ns["t4_population_cell"](*args, **kwargs)
        finally:
            torch.set_grad_enabled(grad_enabled)

    return isolated_prepare


@torch.enable_grad()
def test_real_population_preparation_carries_fused_progress_and_emits_only_endpoint(
    monkeypatch, tmp_path
):
    prepare = population_fixture(monkeypatch, tmp_path)
    task = {
        "smiles": "c1ccccc1",
        "delta": 0.4,
        "target": "parp1",
        "budget": 20,
        "seed_rng": 1000,
        "lineages": 1,
        "regions_per_lineage": 1,
        "workers": 1,
        "particles_per_region": 1,
        "include_fused": True,
        "prepare_only": True,
        "primitive_guidance": "reference",
    }
    # Select whole-molecule region, not the production Q(M); applicability fixture.
    from compose_v4.control import region, region_selector

    regions = region.enumerate_regions(task["smiles"])
    largest = max(regions, key=lambda r: r.size)
    monkeypatch.setattr(region_selector, "sample_region", lambda *_a, **_kw: (largest, None))
    units = []
    reference = prepare(task, units.append)
    guided = prepare({**task, "primitive_guidance": "committor"})
    verify_pair({"reference": reference, "committor": guided})
    assert len(units) == 1
    assert len(reference["take"]) == 1
    c = reference["take"][0]
    assert c["step"] == 5 and c["program_complete"]
    assert c["d_cycle_rank"] == 1 and c["d_heavy"] == 4 and c["d_ring_systems"] == 0
    assert reference["oracle_calls"] == 0
    # Reuse a complete parent without rerunning molecular work.
    resumed = prepare(task, parent_cache={0: units[0]})
    assert resumed["take"] == reference["take"]
    assert resumed["bundles"] == reference["bundles"]


@torch.enable_grad()
def test_real_population_accepts_exact_slot_pendant_at_applicability_and_execution(
    monkeypatch, tmp_path
):
    from compose_v4.control import region, region_selector
    from compose_v4.experiments import production_successor_kernel as prod
    from compose_v4.rewrite.operators import BondInsert

    prepare = population_fixture(monkeypatch, tmp_path)

    def law(_model, _graph, _time):
        return SimpleNamespace(
            marks=[
                SimpleNamespace(
                    executor_rule_name="bond_insert", action=BondInsert(6, 11, 1), probability=1.0
                )
            ]
        )

    monkeypatch.setattr(prod, "enumerate_factorized_marked_law", law)

    def forced_append(options, _rng, **_kwargs):
        # Fails if first-product applicability still uses the SMILES predicate.
        assert "generic" in options and "append_system" in options
        return option_selector.OptionChoice(
            "append_system", tuple(options), tuple(1 / len(options) for _ in options)
        )

    monkeypatch.setattr(option_selector, "sample_option", forced_append)
    seed = "c1ccccc1CCCCCC"
    largest = max(
        (r for r in region.enumerate_regions(seed) if {6, 11} <= r.atoms), key=lambda r: r.size
    )
    monkeypatch.setattr(region_selector, "sample_region", lambda *_a, **_kw: (largest, None))
    result = prepare(
        {
            "smiles": seed,
            "delta": 0.4,
            "target": "parp1",
            "budget": 20,
            "seed_rng": 1000,
            "lineages": 1,
            "regions_per_lineage": 1,
            "workers": 1,
            "particles_per_region": 1,
            "prepare_only": True,
            "primitive_guidance": "reference",
        }
    )
    assert result["oracle_calls"] == 0
    assert len(result["take"]) == 1
    candidate = result["take"][0]
    assert candidate["option"] == "append_system" and candidate["step"] == 1
    assert candidate["d_cycle_rank"] == candidate["d_ring_systems"] == 1
    assert candidate["d_heavy"] == 0
