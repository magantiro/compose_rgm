"""Guards for the actual-sampler support probe.

The probe exists to decide three things about the production proposal path, so the
tests that matter are the ones that fail when a support exclusion is missed, when a
repeated proposal is mistaken for exploration, or when a supplied program's exact
realization stops being checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control.dynamic_program_synthesis import GENERIC_MODULES as V0_MODULES
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_proposal_access_probe import (
    SCHEMA_VERSION,
    _metadata_payloads,
    attempt_signature,
    declared_families,
    family_census,
    probe_context,
    similarity_reporter,
    status_census,
    support_decision,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_proposal_access_probe_v1.json"
ARTIFACT = ROOT / "diagnostics/t4_proposal_prior/access_probe_v1"

# The two constructions the strategy report names as v1-only, and therefore the ones
# a probe over the v0 registry alone would silently fail to look for.
V1_ONLY = ("ring_path_remodel", "construct_substituted_ring")


def _contract():
    payload = unseal(CONTRACT)
    assert payload["schema_version"] == SCHEMA_VERSION
    return payload


def _attempt(*, families=(), failures=None, endpoint="CCO", status="ineligible", lane="shallow"):
    return {
        "planner_channel": lane,
        "endpoint": endpoint,
        "status": status,
        "metadata": {
            "modules": [
                {"family": family, "parameters": {"slot": index}}
                for index, family in enumerate(families)
            ],
            "module_failure_counts": dict(failures or {}),
        },
    }


# ---- Registry ----


def test_the_declared_families_cover_v0_and_the_v1_only_constructions():
    families = declared_families()
    assert set(V0_MODULES) <= set(families)
    assert all(family in families for family in V1_ONLY)


# ---- Census ----


def test_a_family_that_is_tried_and_refused_counts_as_attempted_not_realized():
    census = family_census(
        [
            _attempt(families=("substituent_delete",)),
            _attempt(failures={"construct_substituted_ring:work_limit": 3}),
        ]
    )
    assert census["realized"] == {"substituent_delete": 1}
    assert census["attempted"] == {"construct_substituted_ring": 3, "substituent_delete": 1}
    assert census["rejection_reasons"] == {"construct_substituted_ring:work_limit": 3}


def test_lane_metadata_is_found_through_its_wrapper():
    payload = {"modules": [{"family": "append_ring"}], "module_failure_counts": {}}
    assert _metadata_payloads(payload) == [payload]
    assert _metadata_payloads({"dynamic_generic_composition": payload, "accounting": 3}) == [
        payload
    ]
    assert _metadata_payloads(None) == []


def test_status_census_counts_statuses_and_lanes():
    census = status_census(
        [_attempt(status="eligible"), _attempt(status="duplicate", lane="structured")]
    )
    assert census["attempts"] == 2
    assert census["by_status"] == {"duplicate": 1, "eligible": 1}
    assert census["by_lane"] == {"shallow": 1, "structured": 1}


# ---- Proposal identity ----


def test_two_proposals_reaching_one_molecule_are_not_one_proposal():
    first = _attempt(families=("substituent_delete",), endpoint="CCO")
    second = _attempt(families=("append_ring",), endpoint="CCO")
    assert attempt_signature(first) != attempt_signature(second)


def test_the_same_proposal_has_the_same_signature():
    assert attempt_signature(_attempt(families=("cycle_close",))) == attempt_signature(
        _attempt(families=("cycle_close",))
    )


def test_changing_only_a_module_parameter_changes_the_signature():
    first = _attempt(families=("functionalize",))
    second = json.loads(json.dumps(first))
    second["metadata"]["modules"][0]["parameters"] = {"slot": 99}
    assert attempt_signature(first) != attempt_signature(second)


# ---- Gate ----


def _row(cell="braf_1", *, admission=True, explores=True, realized=None, attempted=None):
    families = {
        "realized": dict(realized or {"substituent_delete": 4}),
        "attempted": dict(attempted or {"substituent_delete": 4}),
        "rejection_reasons": {},
    }
    return {
        "cell": cell,
        "bootstrap": {"explores_after_first_round": explores, "families": families},
        "archive": {"archive_exact_admission": admission, "families": families},
    }


def test_a_clean_probe_passes():
    decision = support_decision([_row()], _contract())
    assert decision["decision"] == "PASS" and decision["failures"] == []


def test_a_failed_exact_realization_fails_the_probe():
    decision = support_decision([_row(admission=False)], _contract())
    assert decision["decision"] == "FAIL"
    assert any("exact archive admission" in failure for failure in decision["failures"])


def test_a_round_that_drew_no_new_proposal_fails_the_probe():
    decision = support_decision([_row(explores=False)], _contract())
    assert decision["decision"] == "FAIL"
    assert any("no new proposal" in failure for failure in decision["failures"])


def test_a_family_attempted_often_and_never_realized_is_a_support_exclusion():
    decision = support_decision(
        [
            _row(
                realized={"substituent_delete": 4},
                attempted={"substituent_delete": 4, "fuse_ring": 9},
            )
        ],
        _contract(),
    )
    assert decision["decision"] == "FAIL"
    assert decision["support_excluded"] == ["fuse_ring"]
    assert any("fuse_ring" in failure for failure in decision["failures"])


def test_a_family_barely_attempted_is_reported_not_failed():
    decision = support_decision(
        [
            _row(
                realized={"substituent_delete": 4},
                attempted={"substituent_delete": 4, "fuse_ring": 2},
            )
        ],
        _contract(),
    )
    assert decision["decision"] == "PASS"
    assert decision["support_excluded"] == []
    assert "fuse_ring" in decision["never_realized_anywhere"]


def test_the_exclusion_threshold_comes_from_the_contract():
    contract = _contract()
    contract["gate"]["minimum_attempts_for_exclusion"] = 2
    decision = support_decision(
        [
            _row(
                realized={"substituent_delete": 4},
                attempted={"substituent_delete": 4, "fuse_ring": 2},
            )
        ],
        contract,
    )
    assert decision["support_excluded"] == ["fuse_ring"]


# ---- Cross-artifact consistency ----


def test_a_probe_context_whose_root_disagrees_with_its_history_is_refused():
    contract = _contract()
    unit = {"cell": "braf_1", "original_seed": "CCO", "oracle_protocol": "p"}
    result = {
        "unit": {"cell": "braf_1", "original_seed": "CCN", "oracle_protocol": "p"},
        "champion": {"candidate": {}, "receipt_id": "r", "score": -1.0},
    }
    with pytest.raises(ValueError, match="original_seed"):
        probe_context(result, unit, contract, index=0)


def test_every_probe_cell_is_bound_by_the_declared_source_registry():
    contract = _contract()
    units = {unit["cell"] for unit in unseal(ROOT / contract["source_registry"]["path"])["units"]}
    assert {context["cell"] for context in contract["probe_contexts"]} <= units


# ---- Historical proximity ----


def test_the_historical_score_stays_on_the_historical_molecule():
    report = similarity_reporter({"c1ccccc1O": -12.5, "CCO": -3.0})
    result = report(["c1ccccc1O"])
    assert result["exact_historical_endpoint_hits"] == 1
    assert result["best_historical_score_among_exact_hits"] == -12.5
    assert result["maximum_nearest_similarity"] == pytest.approx(1.0)
    assert result["best_historical_score_available_in_this_cell"] == -12.5
    assert "no generated molecule" in result["interpretation"]


def test_a_similarity_tie_is_broken_toward_the_better_historical_score():
    # Two historical molecules, same structure, different recorded scores: the probe
    # must report the informative member of the tie, not the worst one.
    report = similarity_reporter({"CCO": -3.0, "OCC": -11.0})
    result = report(["CCO"])
    assert result["maximum_nearest_similarity"] == pytest.approx(1.0)
    assert result["best_historical_score_among_nearest_molecules"] == -11.0


def test_a_generated_molecule_never_receives_a_score():
    result = similarity_reporter({"c1ccccc1O": -12.5})(["CCOCC"])
    assert "score" not in {key for key in result if not key.startswith("best_historical")}
    assert result["exact_historical_endpoint_hits"] == 0
    assert result["best_historical_score_among_exact_hits"] is None


def test_proximity_abstains_without_history():
    assert similarity_reporter({})(["CCO"]) == {"historical_endpoints": 0, "abstained": True}


# ---- Published artifact ----


def _published():
    return unseal(ARTIFACT / "result.json")


def test_the_published_probe_made_no_oracle_call():
    payload = _published()
    assert payload["new_oracle_calls"] == 0 and payload["new_labels"] == 0
    assert all(row["new_oracle_calls"] == 0 for row in payload["contexts"])


def test_the_published_probe_covers_every_declared_context():
    payload = _published()
    contract = _contract()
    assert [row["cell"] for row in payload["contexts"]] == [
        context["cell"] for context in contract["probe_contexts"]
    ]


def test_the_published_decision_is_reproducible_from_its_own_contexts():
    payload = _published()
    replayed = support_decision(payload["contexts"], _contract())
    assert replayed["decision"] == payload["gate"]["decision"]
    assert replayed["failures"] == payload["gate"]["failures"]


def test_the_published_probe_honoured_its_declared_attempt_budget():
    payload = _published()
    lanes = _contract()["lanes"]
    for row in payload["contexts"]:
        assert row["bootstrap"]["attempts"] <= lanes["bootstrap"]["declared_attempt_budget"]
        assert row["archive"]["attempts"] <= lanes["archive"]["declared_attempt_budget"]
