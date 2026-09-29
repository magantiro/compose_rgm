import json
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_property_program_wave import (
    build_curriculum,
    load_contract,
    verified_adapters,
)
from compose_v4.experiments.pmo_target_program_wave import score_values

ROOT = Path(__file__).resolve().parents[1]


def test_contract_freezes_three_tasks_and_forty_eight_queries():
    contract = load_contract(ROOT)

    assert tuple(sorted(contract["tasks"])) == ("gsk3b", "jnk3", "qed")
    assert contract["oracle"]["total_query_ceiling"] == 48
    assert contract["information_regime"]["candidate_injection"] is False
    assert contract["information_regime"]["held_out_claim"] is False
    assert (
        contract["excluded"]["drd2"]["downloaded_pickle_sha256"]
        != contract["excluded"]["drd2"]["local_manifest_pickle_sha256"]
    )


def test_frozen_forest_assets_match_the_official_pickle_lineage():
    contract = load_contract(ROOT)
    manifest = json.loads((ROOT / contract["oracle"]["forest_manifest_path"]).read_text())

    for task in ("gsk3b", "jnk3"):
        assert (
            manifest[task]["provenance"]["pickle_sha256"]
            == contract["tasks"][task]["official_pickle_sha256"]
        )
        assert (
            manifest[task]["parameters_npz_sha256"] == contract["tasks"][task]["parameters_sha256"]
        )


@pytest.mark.alternate_kernel
def test_compatibility_is_installed_before_pinned_tdc_source_import(require_software_versions):
    require_software_versions({"PyTDC": "0.3.6"}, purpose="the frozen PMO property-program oracle")
    adapters = verified_adapters(ROOT, load_contract(ROOT))

    assert tuple(sorted(adapters)) == ("gsk3b", "jnk3", "qed")


def test_real_property_programs_replay_from_the_unrelated_root():
    curriculum = build_curriculum(ROOT, load_contract(ROOT))

    assert curriculum["structural_gate"]["passed"] is True
    assert curriculum["structural_gate"]["programs_replayed"] == 45
    assert all(len(row["programs"]) == 15 for row in curriculum["tasks"].values())


def test_official_metric_finishes_with_the_last_top_ten_value():
    result = score_values([0.1] + [1.0] * 15, load_contract(ROOT))

    assert result["oracle_calls"] == 16
    assert result["final_top10"] == 1.0
    assert result["auc_top10_official_10k"] == pytest.approx(0.9992)
