"""Shard reuse after an outage must be exact, or it is worse than recomputing.

The main lane lost a launch to a client-side DNS failure today. Three layers
answer that: server-side fan-out, ``modal run --detach``, and per-source shard
commits that a relaunch can skip. Only the third is testable without Modal, and
it is the one that can silently corrupt a result -- reusing a shard produced
under a different panel, horizon or seed set would mix two measurements into
one table.

``_reusable_shard`` is pure, so the reuse predicate is pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from modal_apps.claim2_trajectory_characterization_app import (  # noqa: E402
    SHARD_IDENTITY_FIELDS,
    _reusable_shard,
)

TASK = {
    "index": 3,
    "source": "CNC(=O)c1ccccc1",
    "horizon": 6,
    "seeds": [0, 1],
    "kernel_budget": 40,
    "panel_sha256": "a" * 64,
    "family_law_sha256": "b" * 64,
    "out_dir": "claim2_trajectory_smoke",
}


def write_shard(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema": "compose.claim2.trajectory_shard",
        "source": TASK["source"],
        "horizon": TASK["horizon"],
        "seeds": list(TASK["seeds"]),
        "kernel_budget": TASK["kernel_budget"],
        "panel_sha256": TASK["panel_sha256"],
        "family_law_sha256": TASK["family_law_sha256"],
        "budget_exhausted": False,
    }
    payload.update(overrides)
    path = tmp_path / "source-0003.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_matching_completed_shard_is_reused(tmp_path):
    assert _reusable_shard(write_shard(tmp_path), TASK)


def test_a_shard_from_a_different_panel_is_not_reused(tmp_path):
    """Two panels are two experiments; mixing them would be undetectable later."""
    assert not _reusable_shard(write_shard(tmp_path, panel_sha256="c" * 64), TASK)


def test_a_shard_from_a_different_frozen_family_law_is_not_reused(tmp_path):
    assert not _reusable_shard(write_shard(tmp_path, family_law_sha256="d" * 64), TASK)


def test_a_shard_at_a_different_horizon_is_not_reused(tmp_path):
    """H=6 and H=8 trajectories are not interchangeable measurements."""
    assert not _reusable_shard(write_shard(tmp_path, horizon=8), TASK)


def test_a_shard_with_a_different_seed_set_is_not_reused(tmp_path):
    assert not _reusable_shard(write_shard(tmp_path, seeds=[0, 1, 2]), TASK)


def test_a_shard_for_a_different_source_is_not_reused(tmp_path):
    """Index collision across panels must not silently import a foreign molecule."""
    assert not _reusable_shard(write_shard(tmp_path, source="CCO"), TASK)


def test_a_shard_run_under_a_different_budget_is_not_reused(tmp_path):
    assert not _reusable_shard(write_shard(tmp_path, kernel_budget=20), TASK)


def test_a_budget_exhausted_shard_is_not_treated_as_complete(tmp_path):
    """Truncated trajectories are not a completed measurement of that source."""
    assert not _reusable_shard(write_shard(tmp_path, budget_exhausted=True), TASK)


def test_a_truncated_or_unparseable_shard_is_simply_redone(tmp_path):
    path = tmp_path / "source-0003.json"
    path.write_text('{"schema": "compose.claim2.trajec')
    assert not _reusable_shard(path, TASK)


def test_a_foreign_schema_is_not_reused(tmp_path):
    assert not _reusable_shard(write_shard(tmp_path, schema="compose.something.else"), TASK)


def test_identity_fields_cover_everything_that_defines_the_measurement():
    """A field added to the task without being checked here is a silent-reuse bug."""
    assert set(SHARD_IDENTITY_FIELDS) == {
        "source",
        "horizon",
        "panel_sha256",
        "family_law_sha256",
    }


def test_the_local_entrypoint_never_imports_rdkit():
    """``modal run`` uses a different interpreter that has no rdkit.

    Molecule work belongs in the committed panel, computed by a normal python3
    script and mounted into the image, not recomputed at launch.
    """
    source = (
        REPO / "modal_apps" / "claim2_trajectory_characterization_app.py"
    ).read_text()
    entrypoint = source.split("@app.local_entrypoint()")[1]
    for forbidden in ("rdkit", "compose_v4", "descriptor_vector", "MurckoScaffold"):
        assert forbidden not in entrypoint, (
            f"the local entrypoint references {forbidden!r}; it runs under modal's "
            "interpreter, which lacks rdkit"
        )


def test_module_scope_imports_stay_light_enough_for_modals_interpreter():
    source = (
        REPO / "modal_apps" / "claim2_trajectory_characterization_app.py"
    ).read_text()
    module_scope = source.split("def _runtime()")[0]
    assert "rdkit" not in module_scope
    assert "from compose_v4" not in module_scope
