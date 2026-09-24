"""Small behavior tests for the post-ring molecule-level diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from denovo_postring_quality_audit import (
    _group_summary,
    audit,
    characterize,
    sa_components,
)
from denovo_restate_prior_pilot import canonical_payload_hash, verify_contract
from denovo_source_size_audit import carbon_branchpoints
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer


def _record(smiles: str, *, valid_state: bool = True) -> dict:
    return {
        "index": 0,
        "trajectory_seed": 17,
        "smiles": smiles,
        "events": 0,
        "event_rules": [],
        "valid_state": valid_state,
        "connected": True,
    }


def test_ring_and_element_descriptors_distinguish_aromatic_unsaturation() -> None:
    benzene = characterize(_record("c1ccccc1"), "test", Path("source.json"))
    cyclohexene = characterize(_record("C1=CCCCC1"), "test", Path("source.json"))
    difluoro = characterize(_record("CC(F)F"), "test", Path("source.json"))

    assert benzene["aromatic_rings"] == 1
    assert benzene["nonaromatic_unsaturated_5_6_rings"] == 0
    assert cyclohexene["aromatic_rings"] == 0
    assert cyclohexene["nonaromatic_unsaturated_5_6_rings"] == 1
    summary = _group_summary([benzene, cyclohexene, difluoro])
    assert summary["mean_F_atoms"] == pytest.approx(2 / 3)
    assert summary["fraction_with_F"] == pytest.approx(1 / 3)


def test_sa_component_reconstruction_matches_installed_scorer() -> None:
    molecule = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    components = sa_components(molecule)
    assert components["score"] == pytest.approx(sascorer.calculateScore(molecule), abs=1e-10)


def test_saved_committed_invalid_state_fails_before_metric_parity(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "bad.json").write_text(
        json.dumps(
            {
                "arm": "C1",
                "design": "unit",
                "start": 0,
                "stop": 1,
                "records": [_record("CCO", valid_state=False)],
            }
        )
    )
    official = tmp_path / "official.json"
    official.write_text("{}")
    with pytest.raises(ValueError, match="exact-validity invariant"):
        audit([shards], tmp_path / "result", official)


def test_pilot_contract_fails_closed_on_payload_change(tmp_path: Path) -> None:
    payload = {"inputs": {}, "implementation_sha256": {"pilot_script": "unused"}}
    contract = {
        "payload": payload,
        "payload_sha256": canonical_payload_hash(payload),
    }
    path = tmp_path / "contract.json"
    contract["payload"]["unsealed_change"] = 1
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="contract payload SHA-256 mismatch"):
        verify_contract(path)


def test_carbon_branchpoints_excludes_aromatic_atoms_and_rejects_bad_smiles() -> None:
    assert carbon_branchpoints("CC(C)C") == 1
    assert carbon_branchpoints("c1ccccc1") == 0
    with pytest.raises(ValueError, match="invalid saved/source SMILES"):
        carbon_branchpoints("not-a-smiles")
