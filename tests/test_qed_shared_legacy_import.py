"""The H24 corpus can supply terminal labels without importing its fitted head."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.experiments.qed_shared_legacy_import import imported_train_rollouts
from compose_v4.experiments.qed_shared_sources import QEDSourceRoles
from compose_v4.experiments.qed_shared_training import value_examples_with_bellman


class _Reference:
    config = type("Config", (), {"persistent_slots": 48})()

    def identity(self) -> dict:
        return {"max_active_atoms": 40}

    def encode(self, _state) -> np.ndarray:
        return np.zeros(256, dtype=np.float32)


def _row(index: int, source: str) -> dict:
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    original = Chem.MolFromSmiles(source)
    source_fp = fingerprint.GetFingerprint(original)
    path = [source] * 24 + ["CC"]
    qualities = [round(float(QED.qed(Chem.MolFromSmiles(smiles))), 6) for smiles in path]
    similarities = [
        round(
            float(
                DataStructs.TanimotoSimilarity(
                    source_fp, fingerprint.GetFingerprint(Chem.MolFromSmiles(smiles))
                )
            ),
            6,
        )
        for smiles in path
    ]
    canonical = Chem.MolToSmiles(original)
    trajectories = []
    for replicate in range(2):
        payload = f"hphi-corpus-v1|{canonical}|{replicate}".encode()
        trajectories.append(
            {
                "replicate": replicate,
                "seed": int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"),
                "termination": {"kind": "complete", "edits": 24, "reason": None},
                "path": path,
                "qed": qualities,
                "similarity_to_source": similarities,
            }
        )
    return {
        "index": index,
        "source": source,
        "canonical_source": canonical,
        "status": "OK",
        "horizon": 24,
        "trajectories": trajectories,
    }


def _fixture() -> tuple[dict, QEDSourceRoles, tuple[str, ...]]:
    input_sources = ("CCO", "CCN", "CCC")
    roles = QEDSourceRoles(
        train=("CCO", "CCC"),
        train_input_indices=(0, 2),
        validation=(),
        test=("CCN",),
        excluded_train_indices=(1,),
        manifest_sha256="frozen-split",
    )
    corpus = {
        "schema": "compose.hphi.rollout_corpus",
        "corpus_version": "hphi-corpus-v1",
        "n": 3,
        "results": [_row(i, source) for i, source in enumerate(input_sources)],
    }
    return corpus, roles, input_sources


def test_import_omits_test_overlap_and_uses_terminal_labels() -> None:
    corpus, roles, input_sources = _fixture()
    reference = _Reference()
    imported = imported_train_rollouts(corpus, roles, input_sources, reference.identity())
    assert [row["source_input_row_index"] for row in imported] == [0, 2]
    assert all(row["source_role"] == "train" for row in imported)
    values = value_examples_with_bellman(
        reference, imported[0], budget_max=24, regions=((0.4, 0.8),)
    )
    assert len(values.labels) == 48
    assert not np.any(values.labels)
    assert imported[0]["trajectories"][0]["path"][0]["qed"] >= 0.4


def test_import_rejects_seed_or_source_drift() -> None:
    corpus, roles, input_sources = _fixture()
    corpus["results"][0]["trajectories"][0]["seed"] += 1
    with pytest.raises(ValueError, match="incomplete"):
        imported_train_rollouts(corpus, roles, input_sources, {"max_active_atoms": 40})
    corpus, roles, input_sources = _fixture()
    corpus["results"][2]["source"] = "CC"
    with pytest.raises(ValueError, match="differs"):
        imported_train_rollouts(corpus, roles, input_sources, {"max_active_atoms": 40})


def test_import_rejects_metric_drift_and_validation_role() -> None:
    corpus, roles, input_sources = _fixture()
    corpus["results"][0]["trajectories"][0]["qed"][4] = 0.0
    with pytest.raises(ValueError, match="metric drift"):
        imported_train_rollouts(corpus, roles, input_sources, {"max_active_atoms": 40})
    corpus, roles, input_sources = _fixture()
    reference = _Reference()
    rollout = imported_train_rollouts(corpus, roles, input_sources, reference.identity())[0]
    rollout["source_role"] = "validation"
    with pytest.raises(ValueError, match="provenance or role"):
        value_examples_with_bellman(reference, rollout, budget_max=24)
