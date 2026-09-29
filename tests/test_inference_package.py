"""Inference reload must preserve the model and reject untrusted bytes/source drift."""

from pathlib import Path

import pytest
import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.inference_package import load_package, write_package
from compose_v4.experiments.pmo_inference_speed import law_values
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/compose_v4/model/factorized_tracelet_rate_model.py"


@pytest.fixture
def package(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    write_package(
        tmp_path, model, provenance={"dependency_sources": {SOURCE: sha256_file(ROOT / SOURCE)}}
    )
    return tmp_path, model, sha256_file(tmp_path / "manifest.json")


def test_roundtrip_has_exact_production_law(package):
    path, original, digest = package
    with torch.enable_grad():
        loaded, manifest = load_package(path, manifest_sha256=digest, repo_root=ROOT)
        assert torch.is_grad_enabled()
    graph = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 12)
    assert law_values(original, graph)[1] == law_values(loaded, graph)[1]
    assert manifest["software"]["torch"] == torch.__version__
    with pytest.raises(ValueError, match="already exists"):
        write_package(path, loaded, provenance={})


def test_corrupt_model_rejected_before_unpickle(package, monkeypatch):
    path, _, digest = package
    (path / "model.pt").write_bytes(b"not a trusted model")
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("unpickle before authentication")
    )
    with pytest.raises(ValueError):
        load_package(path, manifest_sha256=digest, repo_root=ROOT)


def test_unauthorized_source_change_rejected_before_unpickle(package, monkeypatch):
    path, _, digest = package
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("unpickle before source validation")
    )
    with pytest.raises(ValueError, match="unauthorized"):
        load_package(
            path, manifest_sha256=digest, repo_root=ROOT, probe_source_changes={SOURCE: "0" * 64}
        )


def test_qualified_receipt_binds_same_package_before_loading(package):
    path, _, digest = package
    manifest = unseal(path / "manifest.json")
    receipt = {
        "status": "pass",
        "oracle_calls": 0,
        "export": {"manifest_sha256": digest},
        "package_tensor_sha256": manifest["tensor_sha256"],
        "rows": [{"parity": True}] * 3,
        "cache_sources": {},
    }
    proof = path / "proof.json"
    seal(proof, receipt)
    load_package(
        path,
        manifest_sha256=digest,
        repo_root=ROOT,
        qualified_source_receipt=(proof, sha256_file(proof)),
    )
    receipt["package_tensor_sha256"] = "wrong"
    seal(proof, receipt)
    with pytest.raises(ValueError, match="qualification"):
        load_package(
            path,
            manifest_sha256=digest,
            repo_root=ROOT,
            qualified_source_receipt=(proof, sha256_file(proof)),
        )
