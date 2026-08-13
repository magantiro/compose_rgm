"""Adversarial fixtures for the hard-scaffold constraint metrics.

Two project rules bind this file.

**Every new analysis metric ships with a fixture in which its intended
conclusion is FALSE, and a test asserting the metric says so.** Seven
sign-fixed statistics have been caught in this project, the seventh in new code
written by someone who already knew the rule. Knowing it is not sufficient.

**A sign guarantee belongs to an estimand, not to an arm pair.** Every
`GUARANTEED_SIGN` declaration carries a written mathematical reason naming its
estimand, AND an adversarial fixture showing a NEIGHBOURING metric on the SAME
arm pair can go the opposite direction. A green test must establish the SCOPE of
a guarantee, not merely perpetuate the declaration.

Runs on CPU with no checkpoint, no kernel and no Gate-0 chain.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from compose_v4.experiments.constraints_hard_mask import (
    apply_hard_mask,
    retention_statistics,
)

# --------------------------------------------------------------------------
# The guaranteed estimand, and its exact scope.
# --------------------------------------------------------------------------


def test_constraint_satisfaction_is_one_by_construction() -> None:
    """GUARANTEED_SIGN, and this test states the estimand it applies to.

    Estimand: "fraction of committed states satisfying C" on the arm
    `hard_scaffold_mask_verified`.

    Mathematical reason: the commit set is F_C(x) = {y in F(x) : C(y) = 1} by
    construction, so every committed y satisfies C(y) = 1. The estimand is
    identically 1 and has NO falsifying range.

    Therefore it is a CONSTRUCTION CHECK, recorded without a denominator, and it
    is never reported as an empirical success. The tests below establish that
    this guarantee does NOT extend to the neighbouring metrics.
    """
    fiber = apply_hard_mask(
        source_key="CCO",
        successor_keys=("CCC", "CCN", "CCS"),
        successor_probability=(0.5, 0.3, 0.2),
        predicate=lambda key: key in {"CCC", "CCN"},
    )
    assert all(key in {"CCC", "CCN"} for key in fiber.retained_keys)
    assert fiber.retained_width == 2


# --------------------------------------------------------------------------
# ADVERSARIAL FIXTURES: neighbouring metrics on the SAME arm that CAN go the
# other way. If any of these ever becomes sign-fixed, the guarantee has silently
# widened and this file must fail loudly.
# --------------------------------------------------------------------------


def test_support_retention_can_collapse_on_the_hard_arm() -> None:
    """The mask strangles the fiber, and the metric says so.

    Intended conclusion of the experiment ("the constraint leaves room to act")
    is FALSE in this fixture. Retention is 0.05, far below the reused V4a floor
    of 0.10, and the statistic reports it rather than being pinned at 1.
    """
    keys = tuple(f"K{i}" for i in range(20))
    fiber = apply_hard_mask(
        source_key="SRC",
        successor_keys=keys,
        successor_probability=tuple(0.05 for _ in keys),
        predicate=lambda key: key == "K0",
    )
    assert fiber.support_retention == pytest.approx(0.05)
    assert fiber.support_retention < 0.10, "V4a floor must be able to FAIL"
    assert not fiber.is_mask_empty


def test_mask_empty_is_reachable_and_reported() -> None:
    """Mask-empty is a real outcome, not an error path.

    The V4b ceiling is 0.05. A metric that could never exceed it would not be a
    measurement.
    """
    keys = ("A", "B", "C", "D")
    fibers = [
        apply_hard_mask(
            source_key="SRC",
            successor_keys=keys,
            successor_probability=(0.25, 0.25, 0.25, 0.25),
            predicate=lambda key: False,
        )
        for _ in range(3)
    ]
    stats = retention_statistics(fibers)
    assert all(fiber.is_mask_empty for fiber in fibers)
    assert stats["mask_empty_rate"] == pytest.approx(1.0)
    assert stats["mask_empty_rate"] > 0.05, "V4b ceiling must be able to FAIL"
    assert stats["support_retention_median"] == pytest.approx(0.0)


def test_retention_statistics_span_the_full_range() -> None:
    """Across states the retention statistic must not be pinned at either end."""
    everything = apply_hard_mask(
        "SRC", ("A", "B"), (0.5, 0.5), predicate=lambda key: True
    )
    nothing = apply_hard_mask(
        "SRC", ("A", "B"), (0.5, 0.5), predicate=lambda key: False
    )
    half = apply_hard_mask(
        "SRC", ("A", "B"), (0.5, 0.5), predicate=lambda key: key == "A"
    )
    stats = retention_statistics([everything, nothing, half])
    assert stats["support_retention_min"] == pytest.approx(0.0)
    assert stats["support_retention_max"] == pytest.approx(1.0)
    assert stats["support_retention_median"] == pytest.approx(0.5)


def test_vacuous_mask_is_distinguishable_from_a_useful_one() -> None:
    """A mask that removes nothing is a FAILURE mode, and must be visible.

    "The constraint is nonvacuous" must be falsifiable, or the census cannot
    stop the lane when it should.
    """
    fiber = apply_hard_mask(
        "SRC", ("A", "B", "C"), (0.4, 0.4, 0.2), predicate=lambda key: True
    )
    assert fiber.support_retention == pytest.approx(1.0)
    assert fiber.removed_reference_mass == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Renormalization: required by validate_successor_batch.
# --------------------------------------------------------------------------


def test_retained_probability_is_renormalized_to_one() -> None:
    fiber = apply_hard_mask(
        "SRC",
        ("A", "B", "C", "D"),
        (0.1, 0.2, 0.3, 0.4),
        predicate=lambda key: key in {"B", "D"},
    )
    assert sum(fiber.retained_probability) == pytest.approx(1.0)
    # relative ordering under the reference law is preserved
    assert fiber.retained_probability[0] == pytest.approx(0.2 / 0.6)
    assert fiber.retained_probability[1] == pytest.approx(0.4 / 0.6)
    assert fiber.removed_reference_mass == pytest.approx(0.4)


def test_alias_duplicates_are_rejected_not_silently_merged() -> None:
    """The kernel guarantees distinct keys; a duplicate means an upstream bug."""
    with pytest.raises(ValueError, match="aliases were not merged"):
        apply_hard_mask("SRC", ("A", "A"), (0.5, 0.5), predicate=lambda key: True)


def test_source_key_in_successors_is_rejected() -> None:
    with pytest.raises(ValueError, match="self-transition"):
        apply_hard_mask("SRC", ("SRC",), (1.0,), predicate=lambda key: True)


# --------------------------------------------------------------------------
# Protected-object regression: the bug the cross-check caught.
# --------------------------------------------------------------------------


def _murcko_atom_indices(mol):
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from constraints_hard_scaffold_census import murcko_atom_indices

    return murcko_atom_indices(mol)


@pytest.mark.parametrize(
    "smiles",
    [
        "O=C1c2cc(O)c(O)c(O)c2C(=O)c2cccc(O)c21",
        "Cc1ccccc1-c1noc(-c2cc3ccccc3oc2=O)n1",
        "CC(=NNC(=O)c1ccc(O)nn1)c1cccnc1",
        "O=C1OC2(CCN(C(=O)Nc3ncc(-c4cccc(C(F)F)c4)cn3)CC2)c2ccccc21",
    ],
)
def test_murcko_core_keeps_exocyclic_double_bonded_atoms(smiles: str) -> None:
    """Bemis-Murcko keeps ring carbonyl oxygens; pruning-only drops them.

    The first implementation of `murcko_atom_indices` disagreed with RDKit on 16
    of 30 held-in sources for exactly this reason. This is the regression guard.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    ours = len(_murcko_atom_indices(mol))
    theirs = MurckoScaffold.GetScaffoldForMol(mol).GetNumHeavyAtoms()
    assert ours == theirs, (
        f"protected-object rule disagrees with RDKit on {smiles}: "
        f"{ours} vs {theirs}"
    )


def test_acyclic_source_has_an_empty_protected_core() -> None:
    """An acyclic source is INELIGIBLE, not silently given a whole-molecule core."""
    mol = Chem.MolFromSmiles("CCCCCCO")
    assert len(_murcko_atom_indices(mol)) == 0


# --------------------------------------------------------------------------
# Provenance: every digest in handoff.json must recompute.
#
# "Agents fabricate provenance far more readily than they fabricate numbers...
# a wrong number gets re-measured; a fake citation gets trusted." The
# countermeasure is to keep the manifest machine-checkable.
# --------------------------------------------------------------------------


def _manifest():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / (
        "docs/workstreams/constraints-hard/handoff.json"
    )
    if not path.is_file():
        pytest.skip("handoff.json not present")
    return Path(__file__).resolve().parents[1], json.loads(path.read_text())


def test_handoff_manifest_digests_recompute() -> None:
    import hashlib

    root, manifest = _manifest()

    recorded = dict(manifest["artifact_sha256"])
    recorded.update(
        {
            manifest["frozen_input_paths"][key]: digest
            for key, digest in manifest["frozen_input_sha256"].items()
        }
    )

    mismatched: list[str] = []
    for relative, digest in recorded.items():
        path = root / relative
        if not path.is_file():
            mismatched.append(f"{relative}: MISSING on disk")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            mismatched.append(
                f"{relative}: recorded {digest[:12]}, actual {actual[:12]}"
            )

    assert not mismatched, "handoff.json digests do not recompute:\n" + "\n".join(
        mismatched
    )


def test_handoff_manifest_declares_no_claim_bearing_compute() -> None:
    """The lane's operational constraints are asserted, not merely written down."""
    _root, manifest = _manifest()

    assert manifest["held_out_opened"] is False
    assert manifest["claim_bearing_compute_run"] is False
    assert manifest["modal_launched"] is False
    assert manifest["gpu_used"] is False
    assert manifest["external_dependencies_installed"] == []
    assert manifest["identity_invariant_provable"] is False
    assert (
        manifest["executor_semantics_verdict"] == "LABELED_SUBGRAPH_PRESENCE_INVARIANT"
    )
    assert manifest["files_modified_belonging_to_other_lanes"] == []
