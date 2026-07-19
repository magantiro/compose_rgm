from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.train_tracelet_cnof_gate import (
    _atomic_shared_torch_save,
    _evaluation_batch_cache_path,
    _evaluation_batch_cache_signature,
    _load_evaluation_batch_cache,
)


def _path_signature() -> dict[str, object]:
    return {
        "format_version": 7,
        "seed": 17,
        "max_atoms": 40,
        "train_smiles": ("CC", "CCC"),
        "validation_smiles": ("CO",),
        "test_smiles": ("CN",),
    }


def _evaluation_signature(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "training_backend": "factorized_marks",
        "seed": 17,
        "validation_examples": 8,
        "test_examples": 16,
        "late_time_fraction": 0.5,
        "operational_horizon": 16.0,
        "progress_stratification_fraction": 0.5,
        "bond_representation": "aromatic",
        "ring_electronic_mode": "factorized_local",
    }
    arguments.update(overrides)
    return _evaluation_batch_cache_signature(_path_signature(), **arguments)


def test_evaluation_cache_is_content_addressed_by_scientific_configuration(
    tmp_path,
) -> None:
    baseline = _evaluation_signature()
    changed = _evaluation_signature(progress_stratification_fraction=0.0)
    assert baseline != changed
    assert _evaluation_batch_cache_path(tmp_path, baseline) != (
        _evaluation_batch_cache_path(tmp_path, changed)
    )


def test_evaluation_cache_round_trip_validates_signature_and_partition_sizes(
    tmp_path,
) -> None:
    signature = _evaluation_signature()
    path = _evaluation_batch_cache_path(tmp_path, signature)
    _atomic_shared_torch_save(
        {
            "signature": signature,
            "validation_batch": SimpleNamespace(batch_size=8),
            "test_batch": SimpleNamespace(batch_size=16),
        },
        path,
    )

    validation, test = _load_evaluation_batch_cache(
        path,
        signature=signature,
        validation_examples=8,
        test_examples=16,
    )
    assert validation.batch_size == 8
    assert test.batch_size == 16

    with pytest.raises(ValueError, match="cache/config mismatch"):
        _load_evaluation_batch_cache(
            path,
            signature=_evaluation_signature(seed=18),
            validation_examples=8,
            test_examples=16,
        )

    with pytest.raises(ValueError, match="invalid partition sizes"):
        _load_evaluation_batch_cache(
            path,
            signature=signature,
            validation_examples=7,
            test_examples=16,
        )
