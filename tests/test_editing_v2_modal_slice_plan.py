"""The slice planner must never re-request work that is already published.

This exists because the parser shipped with a real bug: it required the
listing line to contain "/chunks/", but `modal volume ls` emits paths rooted at
the prefix ("chunks/<task>/..."), so it matched NOTHING. The planner then
reported 21,778 records missing when the true figure was 11,461 -- it would
have re-bought 10,317 records already compiled and paid for, and nothing in
the output looked wrong.

A silent zero is the dangerous failure here, so it is the first test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from editing_v2_modal_slice_plan import (  # noqa: E402
    _covered_from_listing,
    _parse_slice_id,
    _covered_from_local,
    _runs,
)

TASK = "a" * 64


def _write_listing(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "listing.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_relative_listing_paths_are_counted(tmp_path: Path) -> None:
    """`modal volume ls` output is rooted at the prefix, with no leading slash."""

    listing = _write_listing(tmp_path, [
        f"chunks/{TASK}/000000000-000000015/RECEIPT.json",
        f"chunks/{TASK}/000000015-000000015/RECEIPT.json",
    ])
    covered = _covered_from_listing(listing)
    assert covered[TASK] == set(range(30))


def test_absolute_listing_paths_are_also_counted(tmp_path: Path) -> None:
    listing = _write_listing(tmp_path, [
        f"/artifacts/editing_v2/corpus/chunks/{TASK}/000000100-000000025/RECEIPT.json",
    ])
    assert _covered_from_listing(listing)[TASK] == set(range(100, 125))


def test_receipts_that_parse_into_nothing_abort(tmp_path: Path) -> None:
    """The bug's signature: receipts present, coverage empty. Must not proceed."""

    listing = _write_listing(tmp_path, [
        "some/unexpected/layout/RECEIPT.json",
        "another/odd/one/RECEIPT.json",
    ])
    with pytest.raises(SystemExit, match="REFUSING TO PLAN"):
        _covered_from_listing(listing)


def test_a_genuinely_empty_listing_is_not_an_error(tmp_path: Path) -> None:
    assert _covered_from_listing(_write_listing(tmp_path, ["chunks/"])) == {}


def test_slice_dirs_without_a_receipt_are_not_covered(tmp_path: Path) -> None:
    """An interrupted write must be recompiled, not assumed done."""

    root = tmp_path / "corpus" / "chunks" / TASK
    (root / "000000000-000000010").mkdir(parents=True)
    (root / "000000000-000000010" / "RECEIPT.json").write_text("{}")
    (root / "000000010-000000010").mkdir(parents=True)  # no receipt
    assert _covered_from_local(tmp_path / "corpus")[TASK] == set(range(10))


def test_the_two_publishers_use_different_grids_and_still_union(tmp_path: Path) -> None:
    """250-entry local slices and 15-entry Modal receipts describe one corpus.

    Comparing directory NAMES would treat these as unrelated; comparing record
    intervals is what makes the union correct.
    """

    root = tmp_path / "corpus" / "chunks" / TASK
    (root / "000000000-000000250").mkdir(parents=True)
    (root / "000000000-000000250" / "RECEIPT.json").write_text("{}")
    local = _covered_from_local(tmp_path / "corpus")
    listing = _covered_from_listing(_write_listing(tmp_path, [
        f"chunks/{TASK}/000000250-000000015/RECEIPT.json",
    ]))
    union = local[TASK] | listing[TASK]
    assert union == set(range(265))


@pytest.mark.parametrize(
    "missing,limit,expected",
    [
        ([], 5, []),
        ([0, 1, 2], 5, [(0, 3)]),
        ([0, 1, 5, 6], 5, [(0, 2), (5, 2)]),           # a gap splits the run
        ([0, 1, 2, 3, 4, 5], 3, [(0, 3), (3, 3)]),     # limit splits the run
    ],
)
def test_runs_chunks_contiguous_gaps(missing, limit, expected) -> None:
    assert _runs(missing, limit) == expected


def test_subset_slice_ids_are_parsed_not_dropped() -> None:
    """A subset slice carries a third field; two-field parsing DROPS it.

    Dropping it means the planner treats published work as missing and
    re-requests it -- money already spent, spent again. The digest is in the
    address precisely because under a subset the offset indexes the filtered
    stream, so the two-field name alone is ambiguous.
    """

    assert _parse_slice_id("000000140-000000015") == (140, 15)
    assert _parse_slice_id("000000140-000000015-c27c9502789f1093") == (140, 15)


def test_non_numeric_slice_ids_are_rejected() -> None:
    assert _parse_slice_id("chunks") is None
    assert _parse_slice_id("abc-def") is None
    assert _parse_slice_id("000000140") is None


def test_subset_and_whole_task_slices_both_count_as_covered(tmp_path: Path) -> None:
    """The same range compiled whole and under a subset are different artifacts.

    Both are real publications and both cover their records, so coverage must
    union them rather than recognise only the historical two-field name.
    """

    root = tmp_path / "corpus" / "chunks" / TASK
    for name in ("000000000-000000010", "000000010-000000010-c27c9502789f1093"):
        (root / name).mkdir(parents=True)
        (root / name / "RECEIPT.json").write_text("{}")
    assert _covered_from_local(tmp_path / "corpus")[TASK] == set(range(20))
