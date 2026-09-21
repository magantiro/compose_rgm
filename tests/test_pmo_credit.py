"""Invariants of the PMO joint credit object and its floored allocation.

These tests make zero oracle calls. Improvements are supplied directly, which is the
point: the allocation law has to be correct before any budget is spent against it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_credit import (
    ACYCLIC_BASIN,
    CREDIT_AXES,
    DEFAULT_EXPLORATION_FLOOR,
    SCHEMA_VERSION,
    CreditKey,
    PopulationCredit,
    basin_label,
    credit_key_from_candidate,
    improvement,
    program_family,
    scale_for_channel,
)

# One Bemis-Murcko scaffold, five substituent variations around it.
_ANALOGUE_SERIES = (
    "c1ccc(cc1)C(=O)NC1CCNCC1",
    "c1ccc(cc1)C(=O)NC1CCN(C)CC1",
    "Cc1ccc(cc1)C(=O)NC1CCNCC1",
    "Clc1ccc(cc1)C(=O)NC1CCNCC1",
    "CCc1ccc(cc1)C(=O)NC1CCNCC1",
)

_FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _frozen_basin_id(smiles: str) -> str:
    """The identifier the frozen controller records, reproduced for comparison."""
    molecule = Chem.MolFromSmiles(smiles)
    bits = sorted(set(map(int, _FINGERPRINT.GetFingerprint(molecule).GetOnBits())))
    return identity(
        {"fingerprint_prefix": bits[:64], "heavy_atoms": molecule.GetNumHeavyAtoms()}
    )


def _key(basin="core", parent="p0", family="atom_insert", scale="refine") -> CreditKey:
    return CreditKey(basin=basin, parent=parent, family=family, scale=scale)


def _candidate(endpoint: str, *, parent: str, channel: str, rules) -> dict:
    return {
        "endpoint": endpoint,
        "provenance": {"entry_id": parent, "planner_channel": channel},
        "trace": {"actions": [{"executor_rule": rule} for rule in rules]},
    }


# ---- Basin labelling ----


def test_scaffold_basin_groups_the_analogue_series_the_frozen_identifier_splits():
    """The recorded `basin_id` is a per-molecule key; the scaffold label is a basin."""
    frozen = {_frozen_basin_id(smiles) for smiles in _ANALOGUE_SERIES}
    scaffold = {basin_label(smiles) for smiles in _ANALOGUE_SERIES}
    assert len(frozen) == len(_ANALOGUE_SERIES)
    assert len(scaffold) == 1


def test_acyclic_molecules_get_an_explicit_sentinel_not_the_empty_scaffold():
    assert basin_label("CCCCO") == ACYCLIC_BASIN
    assert basin_label("c1ccccc1") != ACYCLIC_BASIN


def test_basin_label_refuses_unparseable_and_empty_input():
    for bad in ("", "not-a-molecule"):
        with pytest.raises(ValueError):
            basin_label(bad)


# ---- Family and scale labelling ----


def test_program_family_is_order_free_and_multiplicity_free():
    assert program_family(["cycle_close", "atom_insert"]) == "atom_insert+cycle_close"
    assert program_family(["atom_insert", "atom_insert", "cycle_close"]) == (
        program_family(["cycle_close", "atom_insert"])
    )
    with pytest.raises(ValueError):
        program_family([])


def test_channel_names_translate_onto_the_three_scales():
    assert scale_for_channel("shallow_program_channel") == "refine"
    assert scale_for_channel("structured_program_channel") == "medium"
    assert scale_for_channel("joint_dependency_region_jump") == "jump"
    with pytest.raises(ValueError):
        scale_for_channel("some_other_channel")


def test_credit_key_rejects_a_scale_outside_the_vocabulary():
    with pytest.raises(ValueError):
        CreditKey(basin="core", parent="p0", family="atom_insert", scale="teleport")
    with pytest.raises(ValueError):
        CreditKey(basin="", parent="p0", family="atom_insert", scale="refine")


def test_credit_key_from_candidate_reads_the_controller_record_shape():
    candidate = _candidate(
        _ANALOGUE_SERIES[0],
        parent="entry-7",
        channel="joint_dependency_region_jump",
        rules=["cycle_close", "atom_insert", "atom_insert"],
    )
    key = credit_key_from_candidate(candidate)
    assert key.parent == "entry-7"
    assert key.scale == "jump"
    assert key.family == "atom_insert+cycle_close"
    assert key.basin == basin_label(_ANALOGUE_SERIES[0])
    candidate["provenance"].pop("entry_id")
    with pytest.raises(ValueError):
        credit_key_from_candidate(candidate)


# ---- Improvement sign convention ----


def test_improvement_follows_the_declared_direction():
    assert improvement(0.7, 0.5, direction="maximize") == pytest.approx(0.2)
    assert improvement(-9.5, -9.0, direction="minimize") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        improvement(0.7, 0.5, direction="descending")
    with pytest.raises(ValueError):
        improvement(float("inf"), 0.5, direction="maximize")


# ---- Joint evidence and derived marginals ----


def test_trials_count_observations_and_upside_ignores_negative_outcomes():
    credit = PopulationCredit()
    key = _key()
    credit.observe(key, 0.4)
    credit.observe(key, -0.9)
    credit.observe(key, 0.2)
    cell = credit.cell(key)
    assert cell.trials == 3
    assert cell.improvements == 2
    assert cell.positive_improvement_sum == pytest.approx(0.6)
    assert cell.best_improvement == pytest.approx(0.4)


def test_an_untried_cell_reads_as_empty_rather_than_missing():
    credit = PopulationCredit()
    cell = credit.cell(_key(parent="never-tried"))
    assert cell.trials == 0
    assert cell.best_improvement == -math.inf


def test_marginal_equals_the_joint_summed_over_that_axis():
    """The defect this module exists to fix: marginals must not drift from the joint."""
    credit = PopulationCredit()
    observations = [
        (_key(basin="a", parent="p0", scale="refine"), 0.5),
        (_key(basin="a", parent="p0", scale="jump"), 1.5),
        (_key(basin="b", parent="p1", scale="jump"), -0.3),
        (_key(basin="b", parent="p1", scale="jump"), 0.7),
    ]
    for key, value in observations:
        credit.observe(key, value)
    for axis in CREDIT_AXES:
        marginal = credit.marginal(axis)
        for label, bucket in marginal.items():
            expected_trials = sum(
                cell.trials for key, cell in credit.cells.items() if getattr(key, axis) == label
            )
            expected_sum = sum(
                cell.positive_improvement_sum
                for key, cell in credit.cells.items()
                if getattr(key, axis) == label
            )
            assert bucket.trials == expected_trials
            assert bucket.positive_improvement_sum == pytest.approx(expected_sum)
        assert sum(bucket.trials for bucket in marginal.values()) == len(observations)
    assert credit.marginal("scale")["jump"].trials == 3
    assert credit.marginal("basin")["a"].positive_improvement_sum == pytest.approx(2.0)


def test_marginal_rejects_an_axis_outside_the_credit_key():
    with pytest.raises(ValueError):
        PopulationCredit().marginal("channel")


# ---- Value ----


def test_value_is_upside_per_trial_plus_a_shrinking_optimism_bonus():
    """`prior_weight` is a MULTIPLE of `observed_scale()`, not an absolute bonus.

    It was an absolute constant sized for a docking score, which made an untried cell
    strictly outrank a cell that had already delivered a measured PMO improvement.
    """
    credit = PopulationCredit(prior_weight=0.25)
    key = _key()
    # No trials anywhere: the scale is undefined, so the bonus is zero rather than a
    # constant. Every cell is equal here, so the allocation is uniform either way.
    assert credit.value(key) == pytest.approx(0.0)
    credit.observe(key, 1.0)
    assert credit.observed_scale() == pytest.approx(1.0)
    assert credit.value(key) == pytest.approx(1.0 + 0.25 * 1.0 / math.sqrt(2))
    credit.observe(key, -1.0)
    # the negative outcome halves the per-trial upside; it never goes below zero
    assert credit.observed_scale() == pytest.approx(0.5)
    assert credit.value(key) == pytest.approx(0.5 + 0.25 * 0.5 / math.sqrt(3))


def test_the_optimism_bonus_shrinks_monotonically_with_evidence():
    """Zero-improvement observations isolate the bonus: upside stays 0, so value IS it.

    The scale is held fixed by a separate productive cell, because the bonus is now
    denominated in it and an archive with no improvement anywhere has no scale to use.
    """
    credit = PopulationCredit()
    scale_source = _key(parent="scale_source")
    credit.observe(scale_source, 1.0)
    key = _key()
    bonuses = []
    for _ in range(5):
        assert credit.cell(key).positive_improvement_sum == 0.0
        bonuses.append(credit.value(key) / credit.observed_scale())
        credit.observe(key, 0.0)
    assert bonuses == sorted(bonuses, reverse=True)
    assert bonuses[0] > bonuses[-1]


# ---- Allocation ----


def test_allocation_is_a_distribution_over_distinct_cells():
    credit = PopulationCredit()
    keys = [_key(parent=f"p{index}") for index in range(4)]
    shares = credit.allocate(keys)
    assert shares.shape == (4,)
    assert float(shares.sum()) == pytest.approx(1.0)
    # with no evidence anywhere the allocation is uniform
    assert shares == pytest.approx(np.full(4, 0.25))
    assert credit.allocate([]).shape == (0,)
    with pytest.raises(ValueError):
        credit.allocate([keys[0], keys[0]])


def test_every_live_cell_keeps_the_floor_under_an_adversarial_early_winner():
    """Budget cannot collapse onto one lineage, however good that lineage looks."""
    credit = PopulationCredit(exploration_floor=DEFAULT_EXPLORATION_FLOOR)
    keys = [_key(parent=f"p{index}") for index in range(8)]
    for _ in range(500):
        credit.observe(keys[0], 1_000.0)
    shares = credit.allocate(keys)
    floor = DEFAULT_EXPLORATION_FLOOR / len(keys)
    assert float(shares.sum()) == pytest.approx(1.0)
    assert float(shares.min()) >= floor - 1e-12
    assert float(shares[0]) > float(shares[1])
    # the winner cannot take the whole budget, whatever it scored
    assert float(shares[0]) <= 1.0 - DEFAULT_EXPLORATION_FLOOR * (len(keys) - 1) / len(keys)


def test_a_higher_floor_moves_budget_back_toward_the_neglected_cells():
    keys = [_key(parent=f"p{index}") for index in range(4)]
    loose, tight = PopulationCredit(exploration_floor=0.1), PopulationCredit(
        exploration_floor=0.25
    )
    for credit in (loose, tight):
        for _ in range(50):
            credit.observe(keys[0], 5.0)
    assert float(tight.allocate(keys).min()) > float(loose.allocate(keys).min())
    assert float(tight.allocate(keys)[0]) < float(loose.allocate(keys)[0])


def test_exploration_floor_must_lie_strictly_inside_the_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            PopulationCredit(exploration_floor=bad)


def test_allocation_report_records_the_floor_it_guarantees():
    credit = PopulationCredit()
    keys = [_key(parent="p0"), _key(parent="p1", scale="jump")]
    credit.observe(keys[1], 2.0)
    report = credit.allocation_report(keys)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["minimum_share"] == pytest.approx(DEFAULT_EXPLORATION_FLOOR / 2)
    assert [row["share"] for row in report["cells"]] == pytest.approx(
        list(credit.allocate(keys))
    )
    assert all(row["share"] >= report["minimum_share"] - 1e-12 for row in report["cells"])
    assert report["cells"][1]["trials"] == 1


# ---- Serialization ----


def test_snapshot_round_trips_and_is_deterministic():
    credit = PopulationCredit(exploration_floor=0.15, prior_weight=0.4)
    credit.observe(_key(basin="b", parent="p1", scale="jump"), 0.9)
    credit.observe(_key(basin="a", parent="p0"), -0.2)
    credit.observe(_key(basin="a", parent="p0"), 0.3)
    payload = credit.payload()
    restored = PopulationCredit.restore(payload)
    assert restored.payload() == payload
    assert restored.exploration_floor == pytest.approx(0.15)
    assert restored.prior_weight == pytest.approx(0.4)
    assert restored.cells == credit.cells
    keys = sorted(credit.cells)
    assert restored.allocate(keys) == pytest.approx(credit.allocate(keys))
    # cells are emitted in sorted key order, so the snapshot is byte-stable
    assert [row["basin"] for row in payload["cells"]] == ["a", "b"]


def test_restore_rejects_a_foreign_schema_and_a_repeated_cell():
    credit = PopulationCredit()
    credit.observe(_key(), 1.0)
    payload = credit.payload()
    with pytest.raises(ValueError):
        PopulationCredit.restore({**payload, "schema_version": "something_else"})
    with pytest.raises(ValueError):
        PopulationCredit.restore({**payload, "cells": payload["cells"] * 2})


def test_an_unobserved_credit_object_still_serializes():
    payload = PopulationCredit().payload()
    assert payload["cells"] == []
    assert PopulationCredit.restore(payload).payload() == payload
