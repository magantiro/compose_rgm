import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.qed_shared_reference import (
    QEDSharedReference,
    SharedReferenceConfig,
)
from compose_v4.experiments.successor_kernel import CanonicalSuccessor
from compose_v4.rewrite.kernel import canonical_state_key

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = json.loads((ROOT / "experiments/reference/model.json").read_text())
CHECKPOINT = ROOT / "local_assets/fragments/r_theta_nll.pt"
SHA256 = REFERENCE["checkpoint"]["sha256"]
CATALOG = REFERENCE["catalog"]["fingerprint"]
CATALOG_PATH = ROOT / "local_assets/fragments/catalog.json"
CATALOG_SHA256 = REFERENCE["catalog"]["sha256"]
OTHER_SHA256 = "0" * 64


def test_shared_reference_config_rejects_invalid_time() -> None:
    with pytest.raises(ValueError, match="time"):
        SharedReferenceConfig(CHECKPOINT, SHA256, CATALOG, float("nan"))


def test_qed_reference_conditions_on_active_atom_limit() -> None:
    source = smiles_to_molecular_graph("C")
    larger = smiles_to_molecular_graph("CC")
    candidates = (
        CanonicalSuccessor("one", source, 0.4, 1),
        CanonicalSuccessor("two", larger, 0.6, 1),
    )
    process = object.__new__(QEDSharedReference)
    process.config = SimpleNamespace(persistent_slots=1)
    process.reference = SimpleNamespace(max_active_atoms=1)
    process.kernel = SimpleNamespace(successors=lambda _: SimpleNamespace(successors=candidates))
    accepted = process.successors(source)
    assert len(accepted) == 1
    assert accepted[0].key == "one"
    assert accepted[0].probability == 1.0


@pytest.mark.external_artifact
def test_qed_adapter_uses_nll_checkpoint_for_canonical_transitions() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip("shared reference asset is absent. See experiments/fragments/assets.json")
    config = SharedReferenceConfig(
        CHECKPOINT,
        SHA256,
        CATALOG,
        time=0.5,
        catalog_path=CATALOG_PATH,
        catalog_sha256=CATALOG_SHA256,
    )
    process = QEDSharedReference.load(config)
    source = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 48)
    embedding = process.encode(source)
    assert embedding.shape == (256,)
    assert np.isfinite(embedding).all()
    successors = process.successors(source)
    assert len(successors) > 1
    assert sum(item.probability for item in successors) == pytest.approx(1.0, abs=2e-5)
    result = process.sample(source, np.random.default_rng(71))
    assert result is not None
    assert canonical_state_key(result) in {item.key for item in successors}
    assert process.identity()["checkpoint_sha256"] == SHA256
    assert process.identity()["catalog_sha256"] == CATALOG_SHA256


@pytest.mark.external_artifact
def test_qed_adapter_rejects_head_fitted_to_another_reference() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip("shared reference asset is absent. See experiments/fragments/assets.json")
    process = QEDSharedReference.load(
        SharedReferenceConfig(
            CHECKPOINT,
            SHA256,
            CATALOG,
            time=0.5,
            catalog_path=CATALOG_PATH,
            catalog_sha256=CATALOG_SHA256,
        )
    )
    with pytest.raises(ValueError, match="different reference checkpoint"):
        process.require_matching_value_head({"r_theta_sha256": OTHER_SHA256})
    process.require_matching_value_head({"r_theta_sha256": SHA256})
