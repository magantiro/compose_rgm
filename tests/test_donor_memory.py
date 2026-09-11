"""Focused memory law, arm isolation, and actual worker handoff regressions."""

import json
from contextlib import contextmanager

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.donor_memory import build_memory, proposal_identity, validate_memory
from compose_v4.control.molecular_search_codec import encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments import pmo_donor_comparison as worker_module
from compose_v4.experiments import pmo_option_particles as driver_module
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_archive_pilot import Store


def node(smiles, score):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    return {
        "id": smiles,
        "smiles": smiles,
        "score": score,
        "node": encode_search_state(MolecularSearchState.start(graph, budget=64, root_id=smiles)),
        "chain": [],
        "primitives": 0,
    }


def donor(row):
    return {
        "smiles": row["smiles"],
        "score": row["score"],
        "state": row["node"]["graph"],
        "origin": row["id"],
    }


def test_memory_updates_exact_states_without_canonical_multiplicity():
    a, b = node("CC", 0.4), node("CCC", 0.6)
    fixed = build_memory([donor(a)], {}, mode="fixed")
    same = build_memory([donor(a)], {"CC": a}, mode="evolving")
    assert fixed == same
    evolving = build_memory([donor(a)], {"CC": a, "CCC": b}, mode="evolving")
    assert [r["smiles"] for r in evolving["rows"]] == ["CC", "CCC"]
    assert evolving["rows"][1]["state"] == b["node"]["graph"]
    assert evolving["probabilities"][1] > evolving["probabilities"][0] > 0
    validate_memory(evolving, evolving["memory_id"])
    assert build_memory([donor(a)], {"CCC": b}, mode="fixed") == fixed
    with pytest.raises(ValueError, match="content identity"):
        validate_memory({**evolving, "probabilities": [0, 1]}, evolving["memory_id"])


def test_round_locked_memory_shared_initial_work_and_no_cross_arm_import(tmp_path, monkeypatch):
    parent = node("CC", 0.4)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {"parents": [parent], "observed": {"CC": 0.4}, "initial_donor_memory": [donor(parent)]}
        )
    )
    c = {
        "prepared": {"path": "prepared.json"},
        "particles": 2,
        "boundaries": 3,
        "initial_indices": [0, 0],
        "arms": ["fixed", "evolving"],
        "seed": 12,
        "donor_memory_modes": {"fixed": "fixed", "evolving": "evolving"},
        "parent_selection_modes": {"fixed": "archive", "evolving": "archive"},
        "archive_exploration": 0.2,
        "task": "perindopril_mpo",
        "new_oracle_limit": 12,
    }
    store = Store(tmp_path / "out", lambda: None)

    @contextmanager
    def session(*args):
        yield c, store, {}

    monkeypatch.setattr(driver_module, "make_oracle", lambda *args: lambda s: min(0.9, len(s) / 10))
    batches = []

    def parallel(tasks):
        batches.append(tasks)
        if tasks[0]["phase"] == 3 and len(batches) == 3:
            raise RuntimeError("simulated interruption before final proposals")
        for t in tasks:
            memory = store.read(f"memory/{t['donor_memory_id']}")
            validate_memory(memory, t["donor_memory_id"])
            assert t["proposal_id"] == proposal_identity(memory["memory_id"])
            keys = {r["smiles"] for r in memory["rows"]}
            if t["phase"] == 1:
                assert keys == {"CC"}
                smiles = "CCC"
            elif t["phase"] == 2:
                smiles = "CCCC" if len(keys) > 1 else "CCO"
            else:
                if len(keys) > 1:
                    assert "CCCC" in keys and "CCO" not in keys
                else:
                    assert keys == {"CC"}
                smiles = "CCCCC"
            child = {**node(smiles, 0), "bundle": {"option": "generic"}}
            yield {
                "worker_id": t["worker_id"],
                "replay_verified": True,
                "proposal_policy_sha256": t["proposal_id"],
                "attempts": [{"status": "complete"}],
                "candidates": [child],
            }

    task = {"run_id": "fixture", "image_revision": {"commit": "fixture"}}
    with pytest.raises(RuntimeError, match="simulated interruption"):
        driver_module.driver_remote(
            task, tmp_path, tmp_path, None, None, parallel, run_session=session
        )
    first = driver_module.driver_remote(
        task, tmp_path, tmp_path, None, None, parallel, run_session=session
    )
    assert [len(b) for b in batches] == [2, 4, 4, 4]
    assert batches[2] == batches[3]  # exact memory/parent/RNG task identity survives restart
    assert "CCO" not in first["archives"]["evolving"]
    assert "CCCC" not in first["archives"]["fixed"]
    assert first["new_oracle_calls"] == 4
    assert (
        driver_module.driver_remote(
            task, tmp_path, tmp_path, None, None, parallel, run_session=session
        )
        == first
    )
    assert len(batches) == 4


def test_memory_donor_worker_and_reference_identity(tmp_path, monkeypatch):
    parent = node("CCO", 0.2)
    memory = build_memory([donor(node("CCCC", 0.7))], {}, mode="fixed")
    artifact = tmp_path / "artifact"
    run_store = Store(artifact / "memory_test" / "fixture", lambda: None)
    run_store.save(f"memory/{memory['memory_id']}", memory)
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    c = {
        "seed": 1,
        "donor_memory_modes": {"evolving": "evolving"},
        "artifact_kind": "memory_test",
        "prepared": {"path": "prepared.json"},
    }
    rng_seed = next(
        s
        for s in range(30)
        if np.random.default_rng(np.random.SeedSequence([s, 1, 0, 776])).random() < 0.5
    )
    c["seed"] = rng_seed
    store = Store(tmp_path / "worker", lambda: None)

    @contextmanager
    def session(*args):
        yield c, store, {}

    task = {
        "run_id": "fixture",
        "phase": 1,
        "slot": 0,
        "parent": parent,
        "worker_id": "fixture_worker",
        "image_revision": {"commit": "fixture"},
        "proposal_id": proposal_identity(memory["memory_id"]),
        "donor_memory_id": memory["memory_id"],
        "donor_memory_sha256": sha256_file(run_store.output / f"memory/{memory['memory_id']}.json"),
    }
    result = worker_module.worker_remote(
        task,
        tmp_path,
        artifact,
        None,
        None,
        None,
        contract_loader=lambda root: c,
        run_session=session,
    )
    assert result["proposal_policy_sha256"] == task["proposal_id"]
    assert result["attempts"][0]["component"] == "donor"
    for candidate in result["candidates"]:
        assert candidate["bundle"]["donor_smiles"] == "CCCC"
        assert candidate["bundle"]["donor_memory_id"] == memory["memory_id"]
    policy = worker_module.ReferenceComponent(task["proposal_id"])
    assert policy.payload["model_sha256"] == task["proposal_id"]
