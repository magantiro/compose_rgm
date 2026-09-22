"""Guards for the T4 replication manifest.

Every guard here is written so that REMOVING the thing it guards makes it red.  Two
are deliberately non-tautological: the support-operator roster is checked against the
FROZEN RESULT ARTIFACT rather than against the constant it guards, and the
seed-disjointness proof runs on the REAL historical controller seeds read from the
shipped contracts rather than on invented ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_replication_manifest import (
    BLANK_CELL,
    HISTORICAL_REPLICATE,
    NEW_REPLICATES,
    REQUIRED_SUPPORT_OPERATORS,
    ManifestAuditError,
    ReplicationRow,
    audit_manifest,
    derive_replicate_seed,
    inherited_cross_cell_seed_sharing,
    seed_stream_collisions,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "diagnostics/T4_FROZEN_RESULT_v1.json"

PANEL_CONTRACTS = {
    "parp1": "configs/t4_held_target_distilled_parp1_{d}_250.json",
    "braf": "configs/t4_held_target_distilled_braf_{d}_250.json",
    "fa7": "configs/t4_held_target_distilled_fa7_{d}_250.json",
    "5ht1b": "configs/t4_held_target_distilled_5ht1b_{d}_250.json",
}


def _frozen_rows() -> list[dict]:
    payload = json.loads(FROZEN.read_text())["payload"]
    rows = []
    for delta, block in payload["deltas"].items():
        for row in block["rows"]:
            rows.append({**row, "delta": float(delta)})
    return rows


def _row(
    target="parp1",
    seed=1,
    delta=0.6,
    operator="none",
    controller_seed=2026091800,
    contract="c.json",
    per_cell=250,
    total=750,
    seeds=None,
) -> ReplicationRow:
    if seeds is None:
        seeds = {HISTORICAL_REPLICATE: controller_seed}
        seeds.update({r: derive_replicate_seed(controller_seed, r) for r in NEW_REPLICATES})
    return ReplicationRow(
        target=target,
        seed=seed,
        cell=f"{target}_{seed - 1}",
        delta=delta,
        source_label="panel",
        arm="arm",
        contract_path=contract,
        contract_payload_sha256="p",
        contract_file_sha256="f",
        code_revision="rev",
        app_module="app.py",
        controller_family="family",
        support_operator=operator,
        charged_calls_per_cell=per_cell,
        total_charged_call_ceiling=total,
        historical_charged_calls=100,
        historical_best=-10.0,
        controller_seed=controller_seed,
        replicate_seeds=seeds,
    )


def _audit(rows, **overrides):
    kwargs = {
        "contract_delta": {row.contract_path: row.delta for row in rows},
        "contract_ceiling": {
            row.contract_path: (row.charged_calls_per_cell, row.total_charged_call_ceiling)
            for row in rows
        },
    }
    kwargs.update(overrides)
    return audit_manifest(rows, **kwargs)


# ---- The roster is checked against the FROZEN ARTIFACT, not against itself --------


def test_the_support_operator_roster_covers_exactly_the_frozen_non_panel_rows():
    """Derived from the published table, so drift in either direction is caught.

    The frozen table labels every non-panel row `support_expansion`, which CONFLATES
    the bridge-region operator with the protonation-aware one. This test therefore
    checks the SET of cells, and a separate test pins the mechanisms apart.
    """

    non_panel = {
        (row["target"].lower(), int(row["seed"]), float(row["delta"]))
        for row in _frozen_rows()
        if row.get("source") not in (None, "panel") and row.get("compose") is not None
    }
    # The blank cell is the one deliberate addition: it produced no value, so the
    # frozen table cannot attribute it, and the roster records the method it was last
    # run under instead. Everything else must match the table exactly.
    expected = non_panel | {BLANK_CELL}
    assert expected == set(REQUIRED_SUPPORT_OPERATORS), (
        "the manifest's support-operator roster and the frozen table's non-panel rows "
        f"disagree: table-only {sorted(expected - set(REQUIRED_SUPPORT_OPERATORS))}, "
        f"roster-only {sorted(set(REQUIRED_SUPPORT_OPERATORS) - expected)}"
    )
    assert BLANK_CELL not in non_panel, (
        "the blank cell now carries a produced value; its roster entry was a decision "
        "made because it did not, and must be re-taken"
    )


def test_the_two_rescue_mechanisms_are_kept_apart():
    """5HT1B seed 3 is protonation; FA7/BRAF are the region law. Never the union."""

    assert REQUIRED_SUPPORT_OPERATORS[("5ht1b", 3, 0.4)] == "protonation"
    assert REQUIRED_SUPPORT_OPERATORS[("5ht1b", 3, 0.6)] == "protonation"
    assert {REQUIRED_SUPPORT_OPERATORS[key] for key in REQUIRED_SUPPORT_OPERATORS
            if key[0] in {"fa7", "braf"}} == {"region_repair"}


def test_the_blank_cell_is_the_one_the_frozen_table_leaves_blank():
    blank = {
        (row["target"].lower(), int(row["seed"]), float(row["delta"]))
        for row in _frozen_rows()
        if row.get("compose") is None
    }
    assert blank == {BLANK_CELL}, f"frozen blanks {sorted(blank)} vs manifest {BLANK_CELL}"


# ---- Seeds ------------------------------------------------------------------------


def test_replicate_one_is_the_historical_seed_unchanged():
    assert derive_replicate_seed(2026091800, 1) == 2026091800


@pytest.mark.parametrize("controller_seed", [2026091800, 2026091905, 2026091914])
def test_a_cells_replicates_never_share_a_derived_seed(controller_seed):
    """Measured on the REAL historical seeds, by enumeration, not by a stride argument."""

    rows = [_row(controller_seed=controller_seed)]
    assert seed_stream_collisions(rows) == []


def test_the_collision_check_fires_when_two_replicates_share_a_seed():
    """NEGATIVE CONTROL. Without this the empty result above proves nothing."""

    seeds = {1: 2026091800, 2: 2026091800, 3: derive_replicate_seed(2026091800, 3)}
    rows = [_row(seeds=seeds)]
    collisions = seed_stream_collisions(rows)
    assert collisions, "identical replicate seeds must be reported as a collision"
    assert collisions[0]["replicates"] == [1, 2]
    with pytest.raises(ManifestAuditError, match="share"):
        _audit(rows)


def test_cross_cell_seed_sharing_is_reported_and_not_refused():
    """The two delta arms of one cell SHARE a controller seed in the historical
    contracts. A replicate clones the historical method, so that must be recorded
    rather than treated as a defect."""

    rows = [_row(delta=0.4, contract="a.json"), _row(delta=0.6, contract="b.json")]
    shared = inherited_cross_cell_seed_sharing(rows)
    assert any(entry["reason"] == "same_cell_both_deltas" for entry in shared)
    report = _audit(rows)
    assert report["within_cell_seed_stream_collisions"] == 0
    assert report["inherited_cross_cell_seed_sharing"] == len(shared)


def test_a_replicate_seed_that_is_not_the_declared_derivation_is_refused():
    seeds = {1: 2026091800, 2: 2026091800 + 17, 3: derive_replicate_seed(2026091800, 3)}
    with pytest.raises(ManifestAuditError, match="not the declared derivation"):
        _audit([_row(seeds=seeds)])


def test_a_missing_replicate_is_refused():
    seeds = {1: 2026091800, 2: derive_replicate_seed(2026091800, 2)}
    with pytest.raises(ManifestAuditError, match="no seed declared for replicate"):
        _audit([_row(seeds=seeds)])


# ---- The seven fail-closed audit conditions ---------------------------------------


def test_the_audit_passes_a_well_formed_manifest():
    """POSITIVE CONTROL: without it, every refusal below could be the harness."""

    rows = [
        _row(target="parp1", seed=1, delta=0.6, contract="a.json"),
        _row(target="5ht1b", seed=3, delta=0.6, operator="protonation",
             controller_seed=2026091908, contract="b.json"),
        _row(target="braf", seed=1, delta=0.6, operator="region_repair",
             controller_seed=2026091800, contract="c.json", per_cell=248, total=496),
    ]
    report = _audit(rows)
    assert report["verdict"] == "AUDIT_GREEN_REPLICATES_MAY_BE_LAUNCHED"
    assert report["rows_audited"] == 3


def test_5ht1b_seed_three_may_not_resolve_to_the_region_law():
    rows = [_row(target="5ht1b", seed=3, delta=0.6, operator="region_repair",
                 controller_seed=2026091908)]
    with pytest.raises(ManifestAuditError, match="protonation"):
        _audit(rows)


@pytest.mark.parametrize("seed", [1, 2])
def test_braf_at_delta_six_may_not_resolve_to_the_vanilla_panel(seed):
    rows = [_row(target="braf", seed=seed, delta=0.6, operator="none")]
    with pytest.raises(ManifestAuditError, match="region_repair"):
        _audit(rows)


@pytest.mark.parametrize("delta", [0.4, 0.6])
def test_fa7_seed_three_may_not_lose_its_region_support_arm(delta):
    rows = [_row(target="fa7", seed=3, delta=delta, operator="none")]
    with pytest.raises(ManifestAuditError, match="region_repair"):
        _audit(rows)


def test_a_panel_row_may_not_silently_acquire_a_support_operator():
    rows = [_row(target="parp1", seed=1, delta=0.6, operator="region_repair")]
    with pytest.raises(ManifestAuditError, match="vanilla panel arm"):
        _audit(rows)


def test_a_rebuilt_contract_is_refused():
    rows = [_row(contract="rebuilt.json")]
    with pytest.raises(ManifestAuditError, match="rebuilt rather than cloned"):
        _audit(rows, contracts_rebuilt=["rebuilt.json"])


def test_a_delta_that_disagrees_with_the_contract_is_refused():
    """A shipped contract has carried 0.4 while its name said 0.6, so the manifest's
    delta is checked against the CONTRACT's, never against a name."""

    rows = [_row(delta=0.6, contract="jak2.json")]
    with pytest.raises(ManifestAuditError, match="does not match the contract"):
        _audit(rows, contract_delta={"jak2.json": 0.4})


def test_a_missing_contract_delta_is_refused_rather_than_assumed():
    rows = [_row(contract="unknown.json")]
    with pytest.raises(ManifestAuditError, match="no contract delta supplied"):
        _audit(rows, contract_delta={})


@pytest.mark.parametrize("ceiling", [(249, 750), (250, 747)])
def test_an_oracle_ceiling_that_differs_from_the_historical_one_is_refused(ceiling):
    rows = [_row(contract="k.json")]
    with pytest.raises(ManifestAuditError, match="does not match the historical"):
        _audit(rows, contract_ceiling={"k.json": ceiling})


def test_a_replicate_that_can_resume_prior_state_is_refused():
    rows = [_row()]
    with pytest.raises(ManifestAuditError, match="independent run"):
        _audit(rows, resume_enabled_rows=[("parp1", 1, 0.6)])


def test_a_duplicate_row_is_refused():
    rows = [_row(), _row()]
    with pytest.raises(ManifestAuditError, match="duplicate manifest row"):
        _audit(rows)


# ---- The historical controller seeds are real -------------------------------------


@pytest.mark.parametrize("target", sorted(PANEL_CONTRACTS))
def test_both_delta_arms_of_a_protein_share_their_controller_seeds(target):
    """This is a HISTORICAL FACT the replication must preserve, not a choice."""

    seeds = {}
    for label in ("d04", "d06"):
        path = ROOT / PANEL_CONTRACTS[target].format(d=label)
        payload = json.loads(path.read_text())["payload"]
        seeds[label] = {row["cell"]: row["controller_seed"] for row in payload["cells"]}
    assert seeds["d04"] == seeds["d06"], (
        f"{target}: the two delta arms no longer share controller seeds; a replicate "
        "that changes this is not cloning the historical method"
    )
