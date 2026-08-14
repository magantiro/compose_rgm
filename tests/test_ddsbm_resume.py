"""DDSBM driver resume: LOSSLESS, and safe against a torn partial.

The previous driver was WRITE-ONLY -- it committed a partial every 250 results
but nothing read it back, so a driver loss preserved the data and threw away
the work. That is the same defect that cost 76 minutes on pathwise Stage B.
These tests are the bar: resume must be lossless, and a torn checkpoint must
degrade to a full run rather than corrupt the benchmark payload.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from modal_apps.ddsbm_endpoint_competence_app import resume_from_partial


def _tasks(n: int) -> list[dict]:
    return [{"index": i, "smiles": f"C{'C' * (i % 5)}"} for i in range(n)]


def _write(partial: Path, results: list[dict]) -> None:
    partial.write_bytes(gzip.compress(json.dumps({"results": results}).encode()))


def test_no_partial_means_everything_is_pending(tmp_path):
    done, pending = resume_from_partial(_tasks(20), tmp_path / "absent.json.gz")
    assert done == []
    assert len(pending) == 20


def test_resume_is_LOSSLESS_not_merely_safe(tmp_path):
    """Every completed source is recovered AND never recomputed."""
    tasks = _tasks(500)
    finished = [{"index": i, "status": "OK", "endpoint": f"E{i}"} for i in range(317)]
    p = tmp_path / "full.partial.json.gz"
    _write(p, finished)

    done, pending = resume_from_partial(tasks, p)

    assert len(done) == 317, "recovered work must not be lost"
    assert len(pending) == 183, "completed sources must not be recomputed"
    assert len(done) + len(pending) == len(tasks), "no source may vanish"
    # The union is exactly the original index set -- nothing dropped, nothing dup.
    assert {r["index"] for r in done} | {t["index"] for t in pending} == set(range(500))
    assert not ({r["index"] for r in done} & {t["index"] for t in pending})


def test_resume_preserves_the_recovered_payloads_verbatim(tmp_path):
    tasks = _tasks(10)
    finished = [{"index": 3, "status": "OK", "endpoint": "X", "kernel_calls": 6}]
    p = tmp_path / "x.partial.json.gz"
    _write(p, finished)
    done, _ = resume_from_partial(tasks, p)
    assert done == finished, "resumed records must survive byte-for-byte"


def test_out_of_order_indices_resume_correctly(tmp_path):
    """`order_outputs=False` means results arrive shuffled."""
    tasks = _tasks(100)
    finished = [{"index": i, "status": "OK"} for i in (97, 3, 55, 0, 12)]
    p = tmp_path / "o.partial.json.gz"
    _write(p, finished)
    done, pending = resume_from_partial(tasks, p)
    assert len(done) == 5
    assert {t["index"] for t in pending} == set(range(100)) - {97, 3, 55, 0, 12}


def test_duplicate_indices_in_a_partial_are_collapsed(tmp_path):
    tasks = _tasks(10)
    p = tmp_path / "d.partial.json.gz"
    _write(p, [{"index": 1, "status": "OK"}, {"index": 1, "status": "OK"}])
    done, pending = resume_from_partial(tasks, p)
    assert len(done) == 1, "a doubly-written partial must not inflate n"
    assert len(done) + len(pending) == 10


# --------------------------------------------------------------------------
# Torn writes -- the bar raised after Stage B
# --------------------------------------------------------------------------

def test_TORN_gzip_degrades_to_a_full_run_rather_than_crashing(tmp_path):
    """A half-written checkpoint must cost time, never correctness."""
    p = tmp_path / "torn.partial.json.gz"
    good = gzip.compress(json.dumps({"results": [{"index": 0}]}).encode())
    p.write_bytes(good[: len(good) // 2])          # truncated mid-stream
    done, pending = resume_from_partial(_tasks(50), p)
    assert done == []
    assert len(pending) == 50, "a torn partial must not silently skip sources"


def test_TORN_json_inside_valid_gzip_degrades_to_a_full_run(tmp_path):
    p = tmp_path / "badjson.partial.json.gz"
    p.write_bytes(gzip.compress(b'{"results": [{"index": 0}, {"ind'))
    done, pending = resume_from_partial(_tasks(50), p)
    assert done == []
    assert len(pending) == 50


def test_empty_file_degrades_to_a_full_run(tmp_path):
    p = tmp_path / "empty.partial.json.gz"
    p.write_bytes(b"")
    done, pending = resume_from_partial(_tasks(7), p)
    assert done == []
    assert len(pending) == 7


def test_records_without_an_integer_index_are_not_counted_as_done(tmp_path):
    """A partial written mid-append must not mask a source as complete."""
    tasks = _tasks(10)
    p = tmp_path / "part.partial.json.gz"
    _write(p, [{"index": 0, "status": "OK"}, {"status": "OK"}, {"index": "2"}])
    done, pending = resume_from_partial(tasks, p)
    assert len(done) == 1
    assert {t["index"] for t in pending} == set(range(10)) - {0}
