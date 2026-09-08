"""Offline exact-state and round-synchronous continuation regressions."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_t4_matched_pilot import population_fixture

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import context_from_region
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_warm_continuation import (
    advance_archive,
    canonical_slots,
    endpoint,
    exact_context,
    initial_archive,
    payload_hash,
    run_continuation,
    verify_round,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/t4_matched_repaired/attempt_1/committor"


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


@pytest.mark.parametrize("smiles", ["CCC(O)N", "c1ccncc1", "CC(=O)[O-]", "c1cc[nH]c1"])
def test_region_mapping_preserves_exact_sparse_slots(smiles):
    dense = graph(smiles)
    permutation = np.roll(np.arange(48)[::-1], 7)
    sparse = replace(
        dense,
        atom_types=dense.atom_types[permutation],
        bonds=dense.bonds[np.ix_(permutation, permutation)],
        formal_charges=dense.formal_charges[permutation],
        implicit_h_counts=dense.implicit_h_counts[permutation],
    )
    original = encode_state(sparse)
    mapping = canonical_slots(sparse, smiles)
    for region in enumerate_regions(smiles):
        a, b = context_from_region(region), exact_context(sparse, smiles, region)
        assert b.locus == frozenset(mapping[i] for i in a.locus)
        assert b.frozen == frozenset(mapping[i] for i in a.frozen)
        assert b.terminals == tuple((mapping[i], mapping[j], v) for i, j, v in a.terminals)
        assert (a.phase, a.interface, a.k_components) == (b.phase, b.interface, b.k_components)
    assert encode_state(sparse) == original
    with pytest.raises(ValueError, match="metadata differs"):
        canonical_slots(sparse, "Cl")


def test_saved_archive_retains_every_score_and_exact_endpoint():
    # Committed result asset, not downloaded test data.
    lock, docking = unseal(SOURCE / "candidate_lock.json"), unseal(SOURCE / "docking.json")
    warm = initial_archive(lock, docking)
    assert warm["oracle_attempts"] == 20 and len(warm["archive"]) == 21
    assert warm["rng_state"] == lock["rng_state"]
    for saved, result in zip(warm["archive"][1:], docking["docked"], strict=True):
        assert saved["ds"] == result["ds"]
        assert saved["state"] == endpoint(lock, result)
        assert saved["ancestors"] == [lock["task"]["smiles"]]
        assert canonical_slots(decode_state(saved["state"]), saved["smiles"])
    bad = copy.deepcopy(lock)
    bad["take"] = bad["take"][:-1]
    with pytest.raises(ValueError, match="all 20"):
        initial_archive(bad, docking)


def test_endpoint_matches_step_not_later_canonical_recurrence():
    lock = unseal(SOURCE / "candidate_lock.json")
    candidate = next(c for c in lock["take"] if c["option"] == "build_ring_system")
    assert candidate["step"] == 11
    expected = endpoint(lock, candidate)
    for unit in lock["work"]:
        for row in unit["sampled_transitions"]:
            if row["bundle_id"] == candidate["bundle_id"] and row["step"] == 9:
                row["product"] = {"not": "the selected step"}
    assert endpoint(lock, candidate) == expected


def fixture_source(tmp_path):
    """Synthetic graph receipts for orchestration only, not chemical evidence."""
    task = {"budget": 20, "per_round": 20, "max_executor_applications": 100}
    candidates, transitions = [], []
    source = encode_state(graph("C"))
    for n in range(2, 22):
        smiles = "C" * n
        c = {
            "smiles": smiles,
            "parent": "C",
            "bundle_id": str(n),
            "step": 1,
            "option": "generic",
            "program_complete": True,
            "v": 0,
        }
        candidates.append(c)
        transitions.append(
            {
                **c,
                "source": source,
                "product": encode_state(graph(smiles)),
                "canonical_product": smiles,
            }
        )
    lock = {
        "task": {**task, "smiles": "C", "primitive_guidance": "committor"},
        "round": 1,
        "take": candidates,
        "work": [{"sampled_transitions": transitions}],
        "input_sha256": {},
        "rng_state": np.random.default_rng(1000).bit_generator.state,
    }
    source_dir = tmp_path / "source"
    digest = seal(source_dir / "candidate_lock.json", lock)
    docking = {
        "candidate_lock_sha256": digest,
        "docked": [{**c, "ds": -8 if i else None} for i, c in enumerate(candidates)],
    }
    dock_digest = seal(source_dir / "docking.json", docking)
    contract = {
        "additional_rounds": 2,
        "compute": {"oracle_call_limit": 40},
        "task": task,
        "expected_input_sha256": {},
        "source": {"candidate_lock_sha256": digest, "docking_sha256": dock_digest},
    }
    return source_dir, contract, {**task, "expected_input_sha256": {}}


def synthetic_prepare(task, checkpoint, cached, warm):
    parent = warm["archive"][-1]
    smiles = parent["smiles"] + "C"
    candidate = {
        "smiles": smiles,
        "parent": parent["smiles"],
        "bundle_id": "bundle",
        "step": 1,
        "option": "generic",
        "program_complete": True,
        "v": 0,
        "state": encode_state(graph(smiles)),
    }
    return {
        "task": task,
        "round": warm["round"] + 1,
        "oracle_calls": 0,
        "input_sha256": {},
        "rng_state": warm["rng_state"],
        "take": [candidate],
        "bundles": [candidate],
        "work": [
            {
                "sampled_transitions": [
                    {
                        **candidate,
                        "source": parent["state"],
                        "product": candidate["state"],
                        "canonical_product": smiles,
                    }
                ]
            }
        ],
    }


def test_two_batches_lock_before_docking_and_resume_without_recomputation(tmp_path):
    source, contract, task = fixture_source(tmp_path)
    output, calls = tmp_path / "out", []

    def dock(smiles, name):
        assert (output / name / "candidate_lock.json").exists()
        assert (output / name / "docking_started.json").exists()
        if name == "round_3":
            assert (output / "round_2/archive.json").exists()
        calls.append(name)
        return [-9.0] * len(smiles)

    result = run_continuation(
        task, synthetic_prepare, dock, output, source=source, contract=contract
    )
    assert calls == ["round_2", "round_3"]
    assert result["new_oracle_attempts"] == 2  # no backfill for a short batch
    assert result["cumulative_oracle_attempts"] == 22
    warm = unseal(output / "round_3/archive.json")
    assert warm["archive"][1]["ds"] is None
    assert warm["archive"][-1]["ancestors"] == ["C", "C" * 21, "C" * 22]
    resumed = run_continuation(
        task,
        lambda *_: pytest.fail("recomputed"),
        lambda *_: pytest.fail("redocked"),
        output,
        source=source,
        contract=contract,
    )
    assert resumed == result


def test_failed_oracle_never_retries_implicitly(tmp_path):
    source, contract, task = fixture_source(tmp_path)

    def fail(*_):
        raise RuntimeError("synthetic oracle interruption")

    with pytest.raises(RuntimeError, match="synthetic oracle"):
        run_continuation(
            task, synthetic_prepare, fail, tmp_path / "out", source=source, contract=contract
        )
    with pytest.raises(RuntimeError, match="no automatic re-docking"):
        run_continuation(
            task,
            lambda *_: pytest.fail("recomputed"),
            fail,
            tmp_path / "out",
            source=source,
            contract=contract,
        )


@pytest.mark.parametrize("defect", ["hash", "cap", "task", "prepare"])
def test_source_and_budget_fail_closed(tmp_path, defect):
    source, contract, task = fixture_source(tmp_path)
    prepare = synthetic_prepare
    if defect == "hash":
        contract["source"]["docking_sha256"] = "wrong"
    elif defect == "cap":
        contract["additional_rounds"] = 3
    elif defect == "task":
        contract["task"]["per_round"] = 21
    else:

        def prepare(*_):
            raise RuntimeError("synthetic preparation failure")

    with pytest.raises((ValueError, RuntimeError)):
        run_continuation(
            task,
            prepare,
            lambda *_: pytest.fail("oracle called"),
            tmp_path / "out",
            source=source,
            contract=contract,
        )
    if defect == "prepare":
        with pytest.raises(RuntimeError, match="audit before retry"):
            run_continuation(
                task, synthetic_prepare, None, tmp_path / "out", source=source, contract=contract
            )


@pytest.mark.parametrize("defect", ["incomplete", "round", "duplicate", "state", "rows"])
def test_round_and_archive_invariants(tmp_path, defect):
    source, _contract, task = fixture_source(tmp_path)
    warm = initial_archive(unseal(source / "candidate_lock.json"), unseal(source / "docking.json"))
    lock = synthetic_prepare(task, None, None, warm)
    if defect == "incomplete":
        lock["take"][0].update(option="build_ring_system", program_complete=False)
    elif defect == "round":
        lock["round"] = 1
    elif defect == "duplicate":
        lock["take"] *= 2
    elif defect == "state":
        lock["take"][0]["state"] = warm["archive"][0]["state"]
    else:
        with pytest.raises(ValueError, match="full locked batch"):
            advance_archive(warm, lock, {"docked": []})
        return
    with pytest.raises(ValueError):
        verify_round(lock, warm, task)


@torch.enable_grad()
def test_real_warm_preparation_reuses_states_and_original_seed(monkeypatch, tmp_path):
    from compose_v4.chem import molecular_graph
    from compose_v4.control import option_selector
    from compose_v4.experiments import production_successor_kernel as prod

    prepare = population_fixture(monkeypatch, tmp_path)
    lock, docking = unseal(SOURCE / "candidate_lock.json"), unseal(SOURCE / "docking.json")
    warm = initial_archive(lock, docking)
    task = {**lock["task"], "warm_start_sha256": payload_hash(warm), "expected_input_sha256": {}}
    monkeypatch.setattr(
        prod, "enumerate_factorized_marked_law", lambda *_: SimpleNamespace(marks=[])
    )
    monkeypatch.setattr(
        molecular_graph,
        "smiles_to_molecular_graph",
        lambda *_: pytest.fail("reconstructed executable parent from SMILES"),
    )
    monkeypatch.setattr(
        option_selector,
        "sample_option",
        lambda options, *_a, **_kw: option_selector.OptionChoice(
            "generic", tuple(options), tuple(1 / len(options) for _ in options)
        ),
    )
    result = prepare(task, warm_start=warm)
    assert result["round"] == 2 and result["task"]["smiles"] == lock["task"]["smiles"]
    assert len(result["bundles"]) == 24
    ranked = sorted(
        warm["archive"][1:],
        key=lambda a: (
            0 if a["v"] <= 0 and a["ds"] is not None else 1,
            a["ds"] if a["ds"] is not None else 0,
            a["v"],
        ),
    )
    assert {b["parent"] for b in result["bundles"]} == {r["smiles"] for r in ranked[:8]}
    assert result["take"] == []
    with pytest.raises(ValueError, match="archive identity"):
        prepare({**task, "warm_start_sha256": "bad"}, warm_start=warm)


def test_contract_inherits_frozen_task_and_hashes():
    config = json.loads((ROOT / "configs/t4_warm_continuation.json").read_text())
    identity = config.pop("contract_sha256")
    assert payload_hash(config) == identity
    old = json.loads((ROOT / "configs/t4_matched_pilot.json").read_text())
    assert config["task"] == old["task"]
    assert config["expected_input_sha256"] == old["expected_input_sha256"]
    assert config["source"]["candidate_lock_sha256"] == sha256_file(SOURCE / "candidate_lock.json")
    assert config["source"]["docking_sha256"] == sha256_file(SOURCE / "docking.json")


@torch.enable_grad()
def test_warm_product_similarity_stays_relative_to_original_seed(monkeypatch, tmp_path):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from compose_v4.control import option_selector, region_selector
    from compose_v4.control.region import Region
    from compose_v4.experiments import production_successor_kernel as prod
    from compose_v4.rewrite.operators import BondInsert

    prepare = population_fixture(monkeypatch, tmp_path)
    seed, parent = "c1ccccc1", "c1ccccc1CCCCCC"
    warm = {
        "round": 1,
        "rng_state": np.random.default_rng(1000).bit_generator.state,
        "archive": [
            {"smiles": seed, "state": encode_state(graph(seed)), "ds": None},
            {"smiles": parent, "state": encode_state(graph(parent)), "ds": -8, "v": 0},
        ],
    }
    monkeypatch.setattr(
        prod,
        "enumerate_factorized_marked_law",
        lambda *_: SimpleNamespace(
            marks=[
                SimpleNamespace(
                    executor_rule_name="bond_insert", action=BondInsert(6, 11, 1), probability=1.0
                )
            ]
        ),
    )
    monkeypatch.setattr(
        option_selector,
        "sample_option",
        lambda options, *_a, **_kw: option_selector.OptionChoice(
            "append_system", tuple(options), tuple(1 / len(options) for _ in options)
        ),
    )
    region = Region(frozenset(range(12)), (), "fixture", "fixture", 12, 0, "multi", 0, True)
    monkeypatch.setattr(region_selector, "sample_region", lambda *_a, **_kw: (region, None))
    task = {
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
        "warm_start_sha256": payload_hash(warm),
    }
    result = prepare(task, warm_start=warm)
    assert len(result["take"]) == 1
    c = result["take"][0]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = lambda s: gen.GetFingerprint(Chem.MolFromSmiles(s))
    assert c["sim"] == DataStructs.TanimotoSimilarity(fp(seed), fp(c["smiles"]))
    assert c["sim"] != DataStructs.TanimotoSimilarity(fp(parent), fp(c["smiles"]))
    assert c["d_cycle_rank"] == 1 and c["d_ring_systems"] == 1
    assert c["state"] == endpoint(result, c)
