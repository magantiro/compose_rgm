"""Focused protocol checks for the two-source QED program-controller diagnostic."""

import json

import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from run_qed_pmo_program_two_source_v1 import (
    digest,
    load_contract,
    properties,
    source_rows,
    source_seed,
)


def _payload():
    return {
        "schema": "qed_pmo_program_two_source_v1",
        "source_indices": [0, 1],
        "arms": ["dynamic_v21", "pmo_population"],
        "charged_qed_calls_per_source_arm": 25,
        "rounds": 3,
        "queries_per_round": 8,
        "attempts_per_batch": 64,
        "proposal_wall_seconds": 20.0,
        "similarity_floor": 0.4,
        "qed_success_floor": 0.9,
        "oracle": "local_rdkit_qed_only",
        "workers": 1,
    }


def test_contract_hash_and_fixed_envelope(tmp_path):
    payload = _payload()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"payload": payload, "payload_sha256": digest(payload)}))
    assert load_contract(path)[0] == payload
    payload["charged_qed_calls_per_source_arm"] = 26
    path.write_text(json.dumps({"payload": payload, "payload_sha256": digest(payload)}))
    with pytest.raises(ValueError, match="scientific envelope"):
        load_contract(path)
    path.write_text(json.dumps({"payload": payload, "payload_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_contract(path)


def test_v2_requires_oversubscribed_proposal_pool(tmp_path):
    payload = _payload()
    payload["schema"] = "qed_pmo_program_two_source_v2"
    payload["proposal_pool_size"] = 16
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"payload": payload, "payload_sha256": digest(payload)}))
    assert load_contract(path)[0] == payload
    payload["proposal_pool_size"] = 8
    path.write_text(json.dumps({"payload": payload, "payload_sha256": digest(payload)}))
    with pytest.raises(ValueError, match="scientific envelope"):
        load_contract(path)


def test_first_two_exact_jin_sources_are_supported_and_deterministic():
    sources = source_rows(_payload())
    assert [row["index"] for row in sources] == [0, 1]
    assert all(0.7 <= row["qed"] <= 0.8 for row in sources)
    assert all(len(row["encoded_state"]["atom_types"]) == 48 for row in sources)
    assert source_seed(sources[0]["smiles"], 0) == source_seed(sources[0]["smiles"], 0)
    assert source_seed(sources[0]["smiles"], 0) != source_seed(sources[1]["smiles"], 1)


def test_qed_score_uses_declared_source_relative_similarity():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    source = "CCO"
    fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(source))
    same = properties(source, fingerprint)
    other = properties("c1ccccc1", fingerprint)
    assert same["similarity"] == 1.0
    assert same["qed"] > 0.0
    assert other["similarity"] < 0.4
