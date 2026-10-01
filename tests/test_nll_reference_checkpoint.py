"""Identity and support checks for the shared canonical-successor reference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.production_successor_kernel import FactorizedCanonicalSuccessorKernel
from compose_v4.model.reference_checkpoint import load_frozen_reference

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "experiments/reference/model.json").read_text())
CHECKPOINT = ROOT / "local_assets/fragments/r_theta_nll.pt"
CATALOG = ROOT / "local_assets/fragments/catalog.json"


def _load(*, checkpoint_sha: str | None = None, catalog_sha: str | None = None):
    return load_frozen_reference(
        CHECKPOINT,
        expected_sha256=checkpoint_sha or MANIFEST["checkpoint"]["sha256"],
        expected_catalog_fingerprint=MANIFEST["catalog"]["fingerprint"],
        catalog_path=CATALOG,
        expected_catalog_sha256=catalog_sha or MANIFEST["catalog"]["sha256"],
    )


@pytest.mark.external_artifact
def test_nll_reference_loads_strictly_and_normalizes_canonical_successors():
    if not CHECKPOINT.is_file():
        pytest.skip("fetch the NLL checkpoint through Git LFS")
    reference = _load()
    assert reference.checkpoint_sha256 == MANIFEST["checkpoint"]["sha256"]
    assert reference.catalog_sha256 == MANIFEST["catalog"]["sha256"]
    assert reference.max_active_atoms == 40
    assert reference.model.editing_process_semantics == "semantic_editing_v2_v2"
    assert not reference.model.training
    source = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 48)
    successors = (
        FactorizedCanonicalSuccessorKernel(reference.model, time=0.5).successors(source).successors
    )
    assert len(successors) > 1
    assert sum(row.probability for row in successors) == pytest.approx(1.0, abs=2e-5)


@pytest.mark.external_artifact
def test_nll_reference_rejects_wrong_asset_hashes():
    if not CHECKPOINT.is_file():
        pytest.skip("fetch the NLL checkpoint through Git LFS")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _load(checkpoint_sha="0" * 64)
    with pytest.raises(ValueError, match="ring catalog asset SHA-256 mismatch"):
        _load(catalog_sha="0" * 64)
