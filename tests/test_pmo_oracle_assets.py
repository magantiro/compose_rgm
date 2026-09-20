"""The asset-backed PMO oracles must resolve their assets at CALL time, and be proven
to by a call against pinned reference values.

These tests reproduce the production defect directly: a lazily-loaded relative asset
path plus a bare ``except`` that returns a constant.  That combination charged a full
250-call PMO gsk3b budget and recorded exact zeros on 250 distinct molecules, and
every gate in place at the time passed, because each one either constructed the oracle
without calling it or asked only whether a score was nonzero.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_oracle_assets import (
    ASSET_BACKED_PMO_TASKS,
    POSITIVE_CONTROL_STATUS,
    POSITIVE_CONTROLS,
    AssetPinnedOracle,
    OraclePositiveControlFailure,
    assert_positive_control,
    pinned_working_directory,
    run_positive_control,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "artifacts/oracles/molleo_task3_v1/molleo_task3_oracle_manifest.json"


def _panel_scores() -> dict[str, dict[str, float]]:
    rows = json.loads(PANEL.read_text())["reference_scores"]
    return {row["smiles"]: row for row in rows}


# ---- the pinned constants are not transcriptions to be trusted -------------


def test_every_reference_value_is_the_frozen_panel_value():
    """Re-derive each constant from the frozen artifact, never from the module."""
    panel = _panel_scores()
    for task, references in POSITIVE_CONTROLS.items():
        for reference in references:
            assert reference.smiles in panel, f"{task}: {reference.smiles} is not on the panel"
            expected = panel[reference.smiles][task]
            assert reference.expected == pytest.approx(expected, abs=1e-12), (
                f"{task}: pinned {reference.expected} != panel {expected} for {reference.smiles}"
            )


def test_each_panel_spans_high_intermediate_and_inactive():
    """A binary 'is it nonzero' check misses a wrong-asset load; graded rows do not."""
    for task, references in POSITIVE_CONTROLS.items():
        roles = {reference.role.split("/")[-1] for reference in references}
        assert "high" in roles, f"{task}: no high-scoring reference"
        assert "intermediate" in roles, f"{task}: no intermediate reference"
        assert "inactive" in roles, f"{task}: no inactive reference"
        values = sorted(reference.expected for reference in references)
        assert values[0] < 0.1 < values[-1], f"{task}: references do not span the range"


def test_every_gated_task_has_a_positive_control():
    """A task may not be gated on a control that does not exist."""
    assert ASSET_BACKED_PMO_TASKS <= set(POSITIVE_CONTROLS)
    assert set(POSITIVE_CONTROLS) <= set(POSITIVE_CONTROL_STATUS)
    assert POSITIVE_CONTROL_STATUS["gsk3b"] == "MEASURED"


# ---- the defect itself -----------------------------------------------------


class _LazyRelativeAssetOracle:
    """A faithful miniature of PyTDC's failure mode.

    The asset is loaded lazily on the first call, from a relative path, cached in an
    attribute that is only set on success, and any failure is swallowed into a
    constant -- exactly ``tdc.Oracle.__call__``'s bare ``except`` returning
    ``default_property``.
    """

    default_property = 0.0

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self._table: dict[str, float] | None = None
        self.swallowed = 0

    def __call__(self, smiles: str) -> float:
        try:
            if self._table is None:
                # relative, resolved against cwd AT CALL TIME
                self._table = json.loads(Path("oracle/table.json").read_text())
            return float(self._table[smiles])
        except Exception:  # noqa: BLE001 - this is the defect being reproduced
            self.swallowed += 1
            return self.default_property


@pytest.fixture()
def asset_root(tmp_path: Path) -> Path:
    references = POSITIVE_CONTROLS["gsk3b"]
    table = {reference.smiles: reference.expected for reference in references}
    (tmp_path / "oracle").mkdir()
    (tmp_path / "oracle" / "table.json").write_text(json.dumps(table))
    return tmp_path


@pytest.fixture()
def elsewhere(tmp_path_factory) -> Path:
    """A working directory from which the relative asset path does not resolve."""
    return tmp_path_factory.mktemp("elsewhere")


def test_the_production_pattern_fails_and_is_silent(asset_root: Path, elsewhere: Path):
    """Construct inside the asset dir, restore cwd, call: every score is the constant.

    This is the run that was recorded as complete.  It must fail here, or the fix
    below is not being tested against anything.
    """
    oracle = _LazyRelativeAssetOracle({})
    with pinned_working_directory(asset_root):
        pass  # construction-only pinning, exactly as the defective worker did it
    with pinned_working_directory(elsewhere):
        report = run_positive_control(oracle, "gsk3b")
    assert report["all_zero"] is True
    assert report["passed"] is False
    assert oracle.swallowed == report["n_references"], "the failure must be silent"


def test_call_time_pinning_resolves_the_asset_from_any_cwd(asset_root: Path, elsewhere: Path):
    pinned = AssetPinnedOracle(_LazyRelativeAssetOracle({}), asset_root, name="gsk3b")
    with pinned_working_directory(elsewhere):
        report = assert_positive_control(pinned, "gsk3b")
    assert report["passed"] is True
    assert report["n_agreeing"] == report["n_references"]
    assert report["distinct_observed_values"] >= 3


def test_pinning_survives_a_cwd_change_made_after_construction(asset_root: Path, elsewhere: Path):
    """The invariant is over the oracle's LIFETIME, not the moment it was built."""
    pinned = AssetPinnedOracle(_LazyRelativeAssetOracle({}), asset_root, name="gsk3b")
    reference = POSITIVE_CONTROLS["gsk3b"][0]
    assert pinned(reference.smiles) == pytest.approx(reference.expected)
    with pinned_working_directory(elsewhere):
        assert pinned(reference.smiles) == pytest.approx(reference.expected)
    os.chdir(elsewhere)
    try:
        assert pinned(reference.smiles) == pytest.approx(reference.expected)
    finally:
        os.chdir(ROOT)


def test_priming_populates_the_cache_so_later_calls_need_no_filesystem(asset_root: Path, elsewhere: Path):
    pinned = AssetPinnedOracle(_LazyRelativeAssetOracle({}), asset_root, name="gsk3b")
    reference = POSITIVE_CONTROLS["gsk3b"][0]
    pinned.prime(reference.smiles)
    (asset_root / "oracle" / "table.json").unlink()
    with pinned_working_directory(elsewhere):
        assert pinned(reference.smiles) == pytest.approx(reference.expected)


def test_the_working_directory_is_restored_even_when_the_call_raises(asset_root: Path):
    def boom(_: str) -> float:
        raise ValueError("scoring failed")

    pinned = AssetPinnedOracle(boom, asset_root, name="gsk3b")
    before = Path.cwd()
    with pytest.raises(ValueError):
        pinned("C")
    assert Path.cwd() == before


# ---- the gate refuses the failure signatures -------------------------------


def test_a_constant_zero_oracle_is_refused():
    with pytest.raises(OraclePositiveControlFailure) as failure:
        assert_positive_control(lambda _: 0.0, "gsk3b")
    assert "positive control" in str(failure.value)


def test_a_nonzero_but_wrong_oracle_is_refused():
    """`score > 0` passes here; the pinned expectation does not.

    This is why the control carries expected values and tolerances rather than a
    nonzero check -- a wrong or stale asset scores nonzero on actives.
    """
    report = run_positive_control(lambda _: 0.87, "gsk3b")
    assert all(row["observed"] > 0 for row in report["rows"])
    assert report["passed"] is False
    with pytest.raises(OraclePositiveControlFailure):
        assert_positive_control(lambda _: 0.87, "gsk3b")


def test_an_oracle_right_only_on_the_extremes_is_refused():
    """Half-credit is not credit: the intermediates are the discriminating rows."""
    table = {r.smiles: r.expected for r in POSITIVE_CONTROLS["gsk3b"]}

    def thresholded(smiles: str) -> float:
        return 1.0 if table[smiles] >= 0.9 else 0.0

    with pytest.raises(OraclePositiveControlFailure) as failure:
        assert_positive_control(thresholded, "gsk3b")
    assert "intermediate" in str(failure.value)


def test_a_raising_oracle_is_reported_not_swallowed():
    def boom(_: str) -> float:
        raise FileNotFoundError("oracle/gsk3b_current.pkl")

    report = run_positive_control(boom, "gsk3b")
    assert report["passed"] is False
    assert all(row["error"] for row in report["rows"])
    assert "FileNotFoundError" in report["rows"][0]["error"]


# ---- the audit constants must match the measurement, not a memory ----------

AUDIT = ROOT / "diagnostics/pmo_oracle_asset_audit_v1.json"


def _audit() -> dict:
    return json.loads(AUDIT.read_text())


def test_the_gated_task_set_is_the_measured_one():
    """`ASSET_BACKED_PMO_TASKS` is a claim about PyTDC's bytes; hold it to the audit."""
    audit = _audit()
    assert set(audit["asset_backed_tasks"]) == set(ASSET_BACKED_PMO_TASKS)
    assert len(audit["tasks"]) == 23, "the audit must cover the whole PMO suite"
    unresolved = [t for t, row in audit["tasks"].items()
                  if row.get("classification") == "UNRESOLVED"]
    assert not unresolved, f"unclassified tasks leave the audit incomplete: {unresolved}"


def test_the_recorded_resolution_mode_matches_the_audit():
    """Lazy and eager are different exposures and must not be conflated.

    A lazy load fails into `default_property` and produces a plausible ledger; an
    eager one raises at construction.  The audit measured gsk3b and drd2 lazy and
    jnk3 eager, which also means the pre-diagnosis guess that jnk3 shared gsk3b's
    silent failure mode was wrong.
    """
    from compose_v4.experiments.pmo_oracle_assets import ASSET_RESOLUTION

    audit = _audit()
    expected = {
        "ASSET_BACKED_LAZY_RELATIVE": "LAZY_RELATIVE_AT_CALL",
        "ASSET_BACKED_EAGER_RELATIVE": "EAGER_RELATIVE_AT_CONSTRUCTION",
    }
    for task in ASSET_BACKED_PMO_TASKS:
        measured = audit["tasks"][task]["classification"]
        assert ASSET_RESOLUTION[task] == expected[measured], task


def test_every_asset_backed_oracle_was_actually_called_and_agreed():
    """Construction success is not evidence. Require a CALL with matching values."""
    audit = _audit()
    for task in ASSET_BACKED_PMO_TASKS:
        row = audit["dynamic"][task]
        assert row["status"] == "POSITIVE_CONTROL_PASSED", f"{task}: {row['status']}"
        control = row["positive_control"]
        assert control["passed"] is True
        assert control["n_agreeing"] == control["n_references"] == len(POSITIVE_CONTROLS[task])
        assert control["max_abs_delta"] <= 1e-6
        assert control["all_zero"] is False


def test_the_cwd_counterfactual_is_recorded_for_the_lazy_oracles():
    """The defect must be demonstrated per task, not asserted from a shared idiom."""
    audit = _audit()
    for task in ("gsk3b", "drd2"):
        row = audit["dynamic"][task]
        assert row["cwd_dependent"] is True, task
        assert row["score_from_unpinned_cwd"] == 0.0, task
        assert row["score_with_cwd_pinned_at_call"] > 0.0, task
        # A stray `oracle/` beside the unpinned cwd would make the comparison vacuous.
        assert row["elsewhere_has_oracle_dir"] is False, task


def test_the_swallow_that_made_the_defect_silent_is_recorded():
    audit = _audit()
    assert audit["swallow"]["oracle_call_has_bare_except"] is True
    assert audit["swallow"]["default_property_returned_on_failure"] is True
