from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

pytest.importorskip("joblib")
pd = pytest.importorskip("pandas")
pytest.importorskip("rdkit")

from compose_v4.oracles.pan_lung_filtering import (  # noqa: E402
    fit_lut_applicability_domain,
    fit_molecular_applicability_domain,
    lut_admission,
    molecular_admission,
)
from compose_v4.oracles.reward_guard import (  # noqa: E402
    FilteringRewardGuard,
    RewardFineTuningDisabledError,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_MANIFEST = REPO_ROOT / "artifacts/oracles/pan_lung_filtering_v1/manifest.json"


def test_molecular_admission_rejects_unparseable_and_accepts_training_structure() -> None:
    structures = (
        "CCCCCCCCN(C)CCCCCCCC",
        "CCCCCCCCN(CC)CCCCCCCC",
        "CCCCCCCCN(CCC)CCCCCCCC",
    )
    domain = fit_molecular_applicability_domain(structures)
    assert molecular_admission(domain, structures[0])["admitted"]
    rejected = molecular_admission(domain, "not-a-smiles")
    assert not rejected["admitted"]
    assert rejected["reasons"]


def test_lut_gate_allows_novel_pair_of_known_components_only() -> None:
    frame = pd.DataFrame(
        {
            "head": ["1A1", "1A2"],
            "tail": ["B1", "B2"],
            "round": [1, 2],
        }
    )
    domain = fit_lut_applicability_domain(frame)
    novel_pair = lut_admission(domain, "1A1", "B2", 2)
    assert novel_pair["admitted"]
    assert novel_pair["novel_combination_of_known_components"]
    assert not lut_admission(domain, "9A999", "B2", 2)["admitted"]
    assert not lut_admission(domain, "1A1", "B99", 2)["admitted"]
    assert not lut_admission(domain, "1A1", "B2", 3)["admitted"]


def test_reward_guard_is_filtering_only_and_ood_returns_no_reward() -> None:
    if not BUNDLE_MANIFEST.exists():
        pytest.skip("serialized filtering bundle is not present")
    audit = json.loads(
        (REPO_ROOT / "diagnostics/pan_lung_filtering_admission_test.json").read_text()
    )
    known = audit["cases"]["a549_in_domain"]["admission"]["canonical_smiles"]
    guard = FilteringRewardGuard(BUNDLE_MANIFEST)
    filtering = guard.filter_molecular("a549", known)
    assert filtering["admitted"]
    assert filtering["filtering_score"] is not None
    assert filtering["reward"] is None
    ood = guard.reward_molecular("a549", "C")
    assert not ood["admitted"]
    assert ood["filtering_score"] is None
    assert ood["reward"] is None
    with pytest.raises(RewardFineTuningDisabledError):
        guard.reward_molecular("a549", known)


def test_reward_authorization_requires_manifest_and_frozen_preregistration(
    tmp_path: Path,
) -> None:
    if not BUNDLE_MANIFEST.exists():
        pytest.skip("serialized filtering bundle is not present")
    manifest = json.loads(BUNDLE_MANIFEST.read_text())
    manifest["reward_fine_tuning_enabled"] = True
    authorized_manifest = tmp_path / "manifest.json"
    authorized_manifest.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_hash = sha256(authorized_manifest.read_bytes()).hexdigest()
    registration = {
        "format": "compose_pan_lung_reward_preregistration_v1",
        "enable_reward_fine_tuning": True,
        "frozen_before_optimization": True,
        "bundle_manifest_sha256": manifest_hash,
        "registration_id": "unit-test-registration",
        "registered_at_utc": "2026-07-20T00:00:00Z",
        "candidate_budget": 10,
        "minimum_pessimistic_scores": {"a549": 0.0},
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration, sort_keys=True) + "\n")
    assert FilteringRewardGuard(
        authorized_manifest, registration_path
    ).reward_fine_tuning_authorized
    registration["enable_reward_fine_tuning"] = False
    registration_path.write_text(json.dumps(registration, sort_keys=True) + "\n")
    assert not FilteringRewardGuard(
        authorized_manifest, registration_path
    ).reward_fine_tuning_authorized
