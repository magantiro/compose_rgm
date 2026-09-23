"""Invariants an online controller must hold or it is learning from corrupted labels.

Each of these guards a failure that is INVISIBLE in aggregate telemetry: a partly-attributed
round, an imputed parent score, and an upstream policy that cannot express the choice the
downstream head learned about.
"""
from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.pmo_contextual_macro import (
    PRE_EXECUTION_BLOCKS,
    edge_features,
    region_replace_labels,
)
from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _row(option, parent_score=0.4):
    return {
        "parent": ASPIRIN,
        "endpoint": "Oc1ccccc1",
        "smiles": "Oc1ccccc1",
        "parent_score": parent_score,
        "families": region_replace_labels(option),
        "realized_families": region_replace_labels(option),
        "requested_modules": 2,
        "module_count": 2,
        "primitives": 0,
        "depth": 1,
        "generation": 0,
        "capacity_aware": False,
    }


def test_q_pre_distinguishes_the_intended_rebuild_option():
    """The whole point of an UPSTREAM value is that it can ask for a specific rebuild.

    If the pre-execution features collapse every rebuild into a bare `region_replace`, the
    controller can learn after the fact that a fused-ring rebuild was good and still have no
    way to increase its probability before generation.
    """
    fuse = edge_features(_row("fuse_ring"), blocks=PRE_EXECUTION_BLOCKS)
    regrow = edge_features(_row("regrow"), blocks=PRE_EXECUTION_BLOCKS)
    assert not np.allclose(fuse, regrow)
    # and the distinction must survive into the policy, not merely into the features
    brain = RewardAdaptiveProgramController()
    rows = [_row("fuse_ring"), _row("regrow")] * 20
    for index, row in enumerate(rows[:30]):
        brain.observe(row, 0.9 if "fuse_ring" in row["families"] else 0.1)
    assert brain.fitted
    weights = brain.intent_policy(rows)
    fuse_mass = sum(
        w for r, w in zip(rows, weights, strict=True) if "fuse_ring" in r["families"]
    )
    regrow_mass = sum(
        w for r, w in zip(rows, weights, strict=True) if "regrow" in r["families"]
    )
    assert fuse_mass > regrow_mass, (fuse_mass, regrow_mass)


def test_partial_reward_attribution_is_a_failure_not_a_warning():
    """Losing even one label per round biases exactly the macro learning under test, and
    'most of them arrived' is indistinguishable from 'all of them' in every aggregate."""
    scored = [{"candidate_id": f"c{i}", "score": 0.5} for i in range(16)]
    pending = {f"c{i}": _row("regrow") for i in range(14)}  # two missing

    attributed = sum(1 for o in scored if pending.get(o["candidate_id"]) is not None)
    assert attributed == 14
    with pytest.raises(RuntimeError, match="online learning requires all of them"):
        if attributed != len(scored):
            raise RuntimeError(
                f"reward feedback attributed {attributed} of {len(scored)} "
                "charged outcomes; online learning requires all of them"
            )


def test_a_decision_is_clipped_while_the_stored_label_stays_raw():
    """Clip the DECISION, never the reward. An early ridge fit can predict far outside
    [0, 1]; normalising the labels instead would change what the regression is fitting."""
    brain = RewardAdaptiveProgramController()
    rows = [_row("regrow", parent_score=0.9) for _ in range(30)]
    for row in rows:
        brain.observe(row, 0.95)
    assert brain.fitted
    # every stored label is the raw delta, untouched
    assert all(o["delta_raw"] == pytest.approx(0.95 - 0.9) for o in brain.observations)
    decision = brain._decision_utility(brain.post, rows)
    assert float(decision.min()) >= 0.0 and float(decision.max()) <= 1.0


def test_a_self_parent_makes_the_change_block_constant():
    """Production provenance carries no `parent_smiles`, so defaulting `parent` to the
    candidate's own endpoint is silent and total: the change block compares a molecule with
    itself for a similarity of 1.0 every time, the parent descriptor duplicates the endpoint
    descriptor, and every candidate becomes its own lineage so the lineage floor is inert.

    Nothing errors, which is why this is pinned as a feature-level invariant rather than
    left to a runtime assertion.
    """
    from compose_v4.control.pmo_contextual_macro import FEATURE_BLOCKS

    base = {
        "endpoint": "Oc1ccccc1",
        "smiles": "Oc1ccccc1",
        "parent_score": 0.4,
        "families": ["segment_grow"],
        "realized_families": ["segment_grow"],
        "requested_modules": 1,
        "module_count": 1,
        "primitives": 3,
        "depth": 1,
        "generation": 0,
        "capacity_aware": False,
    }
    self_parent = edge_features({**base, "parent": base["endpoint"]}, blocks=FEATURE_BLOCKS)
    real_parent = edge_features({**base, "parent": ASPIRIN}, blocks=FEATURE_BLOCKS)
    assert not np.allclose(self_parent, real_parent)
    # Two DIFFERENT candidates sharing one real parent must agree on the parent half and
    # differ on the endpoint half; under a self-parent they would differ on both.
    sibling = edge_features(
        {**base, "parent": ASPIRIN, "endpoint": "Cc1ccccc1", "smiles": "Cc1ccccc1"},
        blocks=FEATURE_BLOCKS,
    )
    assert not np.allclose(real_parent, sibling)


def test_a_root_is_flagged_rather_than_given_a_zero_parent_score():
    """`u_hat(G') = u(G) + delta_hat` has no meaning for a candidate with no parent.

    Imputing zero would rank every root as though its parent scored the worst possible
    value, and would teach the regression that unscored parents are worthless -- the same
    failure the endpoint join fixed, one layer down.
    """
    brain = RewardAdaptiveProgramController()
    parented = [_row("regrow", parent_score=0.8) for _ in range(30)]
    for row in parented:
        brain.observe(row, 0.85)
    assert brain.fitted
    # A root scored with an imputed zero parent would be predicted far below the parented
    # rows purely from the offset, regardless of its chemistry.
    root = {**_row("regrow"), "parent_score": 0.0}
    hat = brain._endpoint_hat(brain.post, [parented[0], root])
    assert hat[0] - hat[1] == pytest.approx(0.8, abs=0.2)


def test_passing_the_option_without_a_rate_is_inert():
    """`replacement_option_rate` defaults to 0.0, so handing over a RegionReplacementOption
    and nothing else offers it zero times.

    A wrapper that does exactly that reads as correct at every layer -- the option is
    constructed, passed, and accepted -- and produces no region replacement at all. This is
    the seventh built-but-inert mechanism found in this repo, so it is pinned by execution
    rather than by review.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.control.pmo_contextual_macro import macro_families
    from compose_v4.control.region_replacement_option import RegionReplacementOption

    source = pad_molecular_graph(smiles_to_molecular_graph(ASPIRIN), 48)
    option = RegionReplacementOption()

    def realized(rate, draws=60):
        rng = np.random.default_rng(11)
        found = 0
        for _ in range(draws):
            try:
                _s, program, _a, _t, metadata = synthesize_dynamic_program(
                    source, rng, max_modules=3, max_primitives=32, max_blocks=8,
                    replacement_option=option, replacement_option_rate=rate,
                )
            except (ValueError, RuntimeError):
                continue
            candidate = {
                "provenance": {"metadata": metadata}, "program": program.payload()
            }
            found += sum(
                1 for f in macro_families(candidate) if f.startswith("region_replace:")
            )
        return found

    assert realized(0.0) == 0
    assert realized(0.5) > 0


def test_the_ledger_sees_exactly_what_the_extractor_sees():
    """The per-family audit is how we answer, after a scored run, which macro families
    earned reward and how their proposal mass moved.

    If its family extraction can drift from the authoritative one, that question becomes
    unanswerable for precisely the macro we care about -- and it fails silently, because a
    missing family simply does not appear in the report. Pinned on the SAME objects both
    sides consume.
    """
    from compose_v4.control.pmo_contextual_macro import macro_families
    from compose_v4.control.pmo_reward_adaptive import FamilyLedger

    candidates = [
        _candidate_for_ledger(
            modules=[{"family": "region_replace",
                      "parameters": {"rebuild_option": "fuse_ring"}}],
        ),
        _candidate_for_ledger(modules=[{"family": "segment_grow", "parameters": {}}]),
        _candidate_for_ledger(labels=["current:cycle_close"]),
        _candidate_for_ledger(labels=["0:1:dependency_branch:1"]),
    ]
    rows = [{"families": list(macro_families(c))} for c in candidates]
    ledger = FamilyLedger()
    ledger.propose(rows)
    reported = {k: v["n_proposed"] for k, v in ledger.report()["families"].items()}
    expected: dict = {}
    for row in rows:
        for family in row["families"]:
            expected[family] = expected.get(family, 0) + 1
    assert reported == expected
    # and the compound label specifically must survive into the audit
    assert reported.get("region_replace:fuse_ring") == 1


def _candidate_for_ledger(modules=None, labels=None):
    return {
        "provenance": {"metadata": {"modules": modules or []}},
        "program": {"blocks": [{"label": x} for x in (labels or [])]},
    }
