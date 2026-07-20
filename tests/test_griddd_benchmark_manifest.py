from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/benchmarks/griddd_qed_lead_manifest_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exact_jin_qed_leads_are_frozen_and_hash_bound() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    exact = payload["jin_iclr19_exact"]
    rows = _rows(exact["table"])
    assert exact["status"] == "exact_public_artifact"
    assert exact["source_sha256"] == (
        "704103777e8050eb59f4d15d9997ca6070ba05a6b878b8e18738bfb1e706a090"
    )
    assert len(rows) == exact["lead_count"] == 800
    assert len({row["canonical_isomeric_smiles"] for row in rows}) == 800
    assert all(0.70 <= float(row["qed"]) <= 0.80 for row in rows)
    assert _sha256(ROOT / exact["table"]) == exact["table_sha256"]


def test_griddd_reconstruction_is_never_labeled_as_the_exact_release() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    decision = payload["decision"]
    reconstruction = payload["griddd_code_reconstruction"]
    rows = _rows(reconstruction["table"])
    assert decision["griddd_release_exact_list_available"] is False
    assert decision["griddd_primary_protocol_launch_authorized"] is False
    assert decision["no_substitute_labeled_exact"] is True
    assert reconstruction["status"] == "reproducible_reconstruction_not_exact"
    assert reconstruction["must_not_be_labeled"] == "official GrIDDD 800"
    assert "reconstruction" in Path(reconstruction["table"]).name
    assert len(rows) == reconstruction["lead_count"] == 800
    assert _sha256(ROOT / reconstruction["table"]) == reconstruction["table_sha256"]


def test_metric_and_two_protocol_fairness_boundaries_are_frozen() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metric = payload["metric_contract"]
    fairness = payload["fairness_contract"]
    primary = fairness["protocol_a_griddd_comparable"]
    internal = fairness["protocol_b_compose_internal"]
    boundary = fairness["claim_boundary"]
    assert metric["fingerprint"]["equivalent_on_all_frozen_rows"] is True
    assert primary["arm"] == "direct_conditioned_native_sampling"
    assert primary["beta_zero_shadow_rescoring"] is False
    assert primary["padding_calls"] is False
    assert internal["exact_oracle_calls_per_start_seed_arm"] == 981
    assert internal["selection_influencing_and_padding_calls_reported_separately"] is True
    assert boundary["tables_must_remain_separate"] is True
    assert boundary["protocol_b_is_query_matched_to_griddd"] is False

