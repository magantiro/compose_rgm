"""The materialized-scorer loader must authenticate bytes, not just read them.

This path exists so a host that cannot re-derive the frozen scorer state from
its seed can still compile against the exact frozen weights.  Its whole value
rests on refusing anything that is not those weights: if it degraded to "load
whatever is on disk", it would become the split-brain corpus the frozen
constant exists to prevent, and would look identical while doing it.

The negative cases matter more than the positive one, so they are the tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    _SCORE_REVISION_PREDECESSOR,
    ProcessV2T1RuntimeError,
    load_materialized_scorer_state,
)

FROZEN = _SCORE_REVISION_PREDECESSOR["initial_model_state_sha256"]
RESIDUAL = "graft_relation_head.weight"


def _write(directory: Path, state: dict, *, receipt_base_sha256: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / "predecessor_state.pt")
    torch.save(state, directory / "revised_state.pt")
    (directory / "MATERIALIZATION_RECEIPT.json").write_text(
        json.dumps(
            {
                "base_state_sha256": receipt_base_sha256,
                "software": {"system": "Linux", "machine": "x86_64"},
            }
        )
    )
    return directory


def _decoy_state() -> dict:
    return {
        "encoder.weight": torch.ones(4, 3),
        RESIDUAL: torch.zeros(2, 2),
    }


def test_receipt_naming_another_base_state_is_refused(tmp_path: Path) -> None:
    """A receipt that does not name the frozen constant is not our scorer."""

    directory = _write(tmp_path / "m", _decoy_state(), receipt_base_sha256="0" * 64)
    with pytest.raises(ProcessV2T1RuntimeError, match="not the frozen"):
        load_materialized_scorer_state(directory)


def test_bytes_disagreeing_with_their_own_receipt_are_refused(tmp_path: Path) -> None:
    """The receipt is a claim; the bytes are the evidence. Hash the evidence.

    A receipt that merely QUOTES the frozen constant must not be enough, or the
    check would authenticate a JSON file rather than the weights.
    """

    directory = _write(tmp_path / "m", _decoy_state(), receipt_base_sha256=FROZEN)
    with pytest.raises(ProcessV2T1RuntimeError, match="materialized scorer bytes hash"):
        load_materialized_scorer_state(directory)


def test_the_residual_parameter_is_excluded_from_the_base_hash(tmp_path: Path) -> None:
    """Base state is the predecessor MINUS its zero-initialized residual.

    Pinning this keeps the loader's notion of "base" identical to the builder's;
    if they ever diverged, the loader would authenticate a hash the builder
    never checks and the seal would silently stop meaning anything.
    """

    state = _decoy_state()
    base_only = {name: value for name, value in state.items() if name != RESIDUAL}
    directory = _write(
        tmp_path / "m", state, receipt_base_sha256=state_dict_semantic_sha256(base_only)
    )
    # The receipt now matches the bytes, so it fails on the FROZEN comparison
    # instead -- which proves the residual was excluded exactly as intended.
    with pytest.raises(ProcessV2T1RuntimeError, match="not the frozen"):
        load_materialized_scorer_state(directory)
