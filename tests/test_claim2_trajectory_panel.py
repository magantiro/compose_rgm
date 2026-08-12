"""Claim-2 panel selection, with the held-out boundary as the main subject.

The reserve pool is the confirmatory panel.  Materializing it is the act that
opens it, so the gate is tested here rather than trusted to a convention in a
document.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "claim2_select_trajectory_panel", REPO / "scripts" / "claim2_select_trajectory_panel.py"
)
panel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(panel)

COMMITTED_PANEL = REPO / "diagnostics" / "claim2_trajectory_development_panel.json"


# ---- the held-out boundary ------------------------------------------------


def test_reserve_pool_refuses_to_run_without_written_authorization():
    """--pool reserve must fail BEFORE reading a single held-out molecule."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "claim2_select_trajectory_panel.py"),
            "--pool", "reserve",
            "--out", "/dev/null",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert result.returncode != 0
    assert "HELD-OUT" in result.stderr
    assert "authorized" in result.stderr


def test_training_pool_refuses_a_meaningless_authorization():
    """An authorization string on the held-in pool signals a confused invocation."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "claim2_select_trajectory_panel.py"),
            "--pool", "training",
            "--i-am-authorized-to-open-the-matched-reserve", "someone",
            "--out", "/dev/null",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert result.returncode != 0
    assert "meaningless" in result.stderr


# ---- banding --------------------------------------------------------------


def test_support_bands_match_the_matched_validation_carve():
    assert panel.SUPPORT_BANDS == ((0, 0), (1, 4), (5, 24), (25, 10**9))
    assert panel.SUPPORT_BAND_ORDER == ("0", "1-4", "5-24", "25+")
    assert panel.band_label(0) == "0"
    assert panel.band_label(1) == "1-4"
    assert panel.band_label(4) == "1-4"
    assert panel.band_label(5) == "5-24"
    assert panel.band_label(24) == "5-24"
    assert panel.band_label(25) == "25+"
    assert panel.band_label(10_000) == "25+"


def test_size_bands_tile_the_heavy_atom_window_without_gaps():
    bounds = [(low, high) for _label, low, high in panel.SIZE_BANDS]
    assert bounds[0][0] == panel.MIN_HEAVY_ATOMS
    assert bounds[-1][1] == panel.MAX_HEAVY_ATOMS
    for (_low, high), (next_low, _next_high) in zip(bounds, bounds[1:]):
        assert next_low == high + 1
    assert panel.size_band(panel.MIN_HEAVY_ATOMS - 1) is None
    assert panel.size_band(panel.MAX_HEAVY_ATOMS + 1) is None


def test_heavy_atom_ceiling_leaves_horizon_headroom_below_the_process_bound():
    """A 38-atom source would hit the 40-atom process ceiling inside six edits.

    Suppressed growth at the ceiling is indistinguishable from a learned
    preference against growing, which would corrupt the mobility measurement.
    """
    process_ceiling = 40
    assert process_ceiling - panel.MAX_HEAVY_ATOMS >= 6


# ---- burned-endpoint exclusion --------------------------------------------


def test_burned_endpoints_include_both_sources_and_targets(tmp_path):
    """The sealed67 amendment dropped pairs whose TARGET was another panel's source."""
    seal = tmp_path / "diagnostics" / "editing_v2_controller_panel_seal.json"
    seal.parent.mkdir(parents=True)
    seal.write_text(
        json.dumps(
            {
                "development": [{"source": "CCO", "target": "CCC"}],
                "sealed": [{"source": "CCN", "target": "CCO"}],
            }
        )
    )
    burned = panel.load_burned_endpoints(tmp_path)
    assert burned == {"CCO", "CCC", "CCN"}


def test_burned_endpoints_accept_a_plain_string_list(tmp_path):
    probe = tmp_path / "diagnostics" / "editing_v2_experiment_c0_planning_signal.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps({"per_source": [{"source": "c1ccccc1"}]}))
    assert panel.load_burned_endpoints(tmp_path) == {"c1ccccc1"}


def test_missing_burn_file_is_not_silently_fatal(tmp_path):
    assert panel.load_burned_endpoints(tmp_path) == set()


# ---- the committed held-in panel ------------------------------------------


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(COMMITTED_PANEL.read_text())


def test_committed_panel_is_held_in_and_says_so(committed):
    assert committed["held_out_opened"] is False
    assert committed["authorization"] is None
    assert committed["status"] == "SMOKE_HELD_IN"


def test_committed_panel_covers_every_support_band_and_size_band(committed):
    composition = committed["composition"]
    for band in panel.SUPPORT_BAND_ORDER:
        for size, _low, _high in panel.SIZE_BANDS:
            assert composition.get(f"{band}|{size}", 0) > 0, f"{band}|{size} is empty"


def test_committed_panel_sources_are_inside_the_declared_window(committed):
    for row in committed["sources"]:
        assert panel.MIN_HEAVY_ATOMS <= row["heavy_atoms"] <= panel.MAX_HEAVY_ATOMS
        assert row["support_band"] == panel.band_label(row["scaffold_support"])
        assert row["size_band"] == panel.size_band(row["heavy_atoms"])


def test_committed_panel_sources_are_distinct_and_hash_bound(committed):
    import hashlib

    sources = [row["source"] for row in committed["sources"]]
    assert len(set(sources)) == len(sources)
    digest = hashlib.sha256(json.dumps(sorted(sources), sort_keys=True).encode()).hexdigest()
    assert digest == committed["panel_sha256"]


def test_committed_panel_is_drawn_from_the_held_in_universe(committed):
    import gzip

    reserve = json.load(
        gzip.open(REPO / "diagnostics" / "editing_v2_matched_validation_reserve_ids.json.gz", "rt")
    )
    held_in = set(reserve["training_source_keys"])
    held_out = set(reserve["reserve_source_keys"])
    for row in committed["sources"]:
        assert row["source"] in held_in
        assert row["source"] not in held_out
