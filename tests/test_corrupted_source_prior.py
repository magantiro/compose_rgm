"""Corrupted-source-prior records are well-formed PathRecords the GM trainer consumes (both directions).

Complements test_source_corruption (which checks the traces replay): this checks the ``PathRecord``
wrapper the training loop iterates over is well-formed for the corrupted-molecule source prior.
"""

from __future__ import annotations

from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records

_MOLS = (
    "COc1cc(C(=O)Nc2ccccc2Oc2ccccc2)on1",
    "C=CCNC(=O)Nc1ccc(F)c(NC(=O)OC)c1",
    "COC(=O)c1ccc(CNc2ccn(C)n2)[nH]1",
)


def test_build_records_both_directions_wellformed() -> None:
    records, attempted = build_corrupted_prior_records(_MOLS, n_slots=40, depth_max=5, seed=0)
    assert attempted == len(_MOLS)
    assert len(records) >= 1  # at least one molecule yields a usable corrupted-prior record
    for record in records:
        assert record.path.path_length >= 1  # non-empty source -> target teacher trace
