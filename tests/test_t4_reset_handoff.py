"""Additional cross-lane, durable-resume and lossless-handoff invariants."""

import gzip
import json
import sys

import pytest

from compose_v4.experiments import t4_objective_reset as reset
from compose_v4.experiments.t4_matched_pilot import unseal
from tests.test_objective_program_search import eligible, fixture_search
from tests.test_t4_objective_reset import Volume, unit  # noqa: F401 - pytest fixture
from tools import t4_reset_data_pack
from tools.t4_objective_reset_status import historical_at
from tools.t4_reset_verification import junit


def test_both_channels_enter_one_measured_archive():
    search = fixture_search(rich=8)
    batch = search.propose_batch(eligible)
    assert {search.lane(c) for c in batch["candidates"]} == {"shallow", "structured"}
    outcomes = [
        {
            "candidate_id": c["candidate_id"],
            "receipt_id": f"synthetic-joint:{i}",
            "score": -3.0 - i,
            "failure": None,
            "oracle_protocol": search.base.oracle_protocol,
        }
        for i, c in enumerate(batch["candidates"])
    ]
    search.observe_batch(batch["batch_id"], outcomes)
    endpoints = {e["endpoint"] for e in search.base.entries.values()}
    assert all(c["endpoint"] in endpoints for c in batch["candidates"])
    assert search.channel_stats["structured"]["charged"] > 0


def test_completed_receipt_recovers_without_redocking(unit, tmp_path):  # noqa: F811 - fixture
    calls = []

    class InterruptedVolume(Volume):
        commits = 0

        def commit(self):
            self.commits += 1
            if self.commits == 5:
                raise InterruptedError("synthetic interruption after result persistence")

    volume = InterruptedVolume()

    def fake_dock(smiles, *_):
        calls.append(smiles)
        return -1.0

    kwargs = {
        "arm": "support_control",
        "run_id": "synthetic-resume",
        "volume": volume,
        "dock": fake_dock,
        "calls": 2,
    }
    with pytest.raises(InterruptedError):
        reset.run_scored_unit(unit, tmp_path, **kwargs)
    result = reset.run_scored_unit(unit, tmp_path, **kwargs)
    assert len(calls) == result["query_count"] == 2


def test_evidence_pack_is_lossless_and_refuses_overwrite(tmp_path, monkeypatch):
    source, output, audit = (tmp_path / p for p in ("data.jsonl", "data.jsonl.gz", "audit.json"))
    source.write_text(json.dumps({"arm": "synthetic_fixture", "score": -1}) + "\n")
    audit.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["pack", str(source), str(output), "--audit", str(audit)])
    t4_reset_data_pack.main()
    assert gzip.decompress(output.read_bytes()) == source.read_bytes()
    manifest = unseal(output.with_suffix(".manifest.json"))
    assert manifest["rows"] == 1 and manifest["new_labels"] == 0
    with pytest.raises(FileExistsError):
        t4_reset_data_pack.main()


def test_verification_does_not_promote_failure_or_absence(tmp_path):
    path = tmp_path / "suite.xml"
    assert junit(path)["status"] == "not_finished_or_not_run"
    path.write_text(
        '<testsuites><testsuite tests="1" failures="1" errors="0"><testcase classname="fixture" name="negative"><failure>failure</failure></testcase></testsuite></testsuites>'
    )
    assert junit(path)["status"] == "failed"
    assert junit(path)["failed_nodes"] == ["fixture::negative"]


def test_matched_calls_never_import_future_scores_or_other_protocols():
    units = [
        {
            "arm": "v0",
            "cell": "fixture",
            "replicate": 0,
            "query_count": 10,
            "query_mapping_verified": True,
            "rounds": 2,
            "rounds_recovered": 2,
        }
    ]
    rows = [
        {
            "arm": "v0",
            "cell": "fixture",
            "replicate": 0,
            "query_verified": True,
            "query": q,
            "score": score,
        }
        for q, score in [(1, -1), (5, -2), (10, -3)]
    ]
    rows.append({**rows[0], "arm": "v0d06", "score": -99})
    assert historical_at("fixture", 5, units, rows)["v0"] == -2
    assert historical_at("fixture", 11, units, rows)["v0"] is None
    assert historical_at("fixture", 0, units, rows)["v0"] is None
