"""The v2 protonation-rescue preflight must REFUSE, not merely describe.

Every guard here is exercised by mutating a staged copy of a real contract and
requiring a refusal.  A guard that only appears in a docstring is documentation; a
guard with a failing negative control is a check.

The staged root symlinks the real ``src``/``modal_apps``/``docs``/``diagnostics`` so the
hash pins and the gate/receipt evidence are the production ones, and holds a REAL
``configs`` directory whose contract can be mutated per test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments import t4_protonation_rescue_contract as v1
from compose_v4.experiments.t4_protonation_rescue_contract_v2 import (
    ARMS,
    AUTHORIZATION_METADATA_KEYS,
    PINNED_GATE,
    PREPARED_STATUS,
    SCHEMA_VERSION,
    assert_payload_carries_no_authorization,
    reconstruct_superseded_payload,
    validate_rescue_preflight_v2,
)

REPO = Path(__file__).resolve().parents[1]
LINKED = ("src", "modal_apps", "docs")


def _stage(tmp_path: Path, *, omit: tuple[str, ...] = ()) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    for name in LINKED:
        (root / name).symlink_to(REPO / name)
    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    for entry in (REPO / "diagnostics").iterdir():
        relative = f"diagnostics/{entry.name}"
        if relative in omit:
            continue
        (diagnostics / entry.name).symlink_to(entry)
    configs = root / "configs"
    configs.mkdir()
    for spec in ARMS.values():
        name = Path(spec["contract"]).name
        (configs / name).write_text((REPO / spec["contract"]).read_text())
    return root


def _write(root: Path, arm: str, payload: dict) -> Path:
    path = root / ARMS[arm]["contract"]
    path.write_text(
        json.dumps(
            {"payload": payload, "payload_sha256": identity(payload)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _payload(root: Path, arm: str) -> dict:
    return json.loads((root / ARMS[arm]["contract"]).read_text())["payload"]


def _check(root: Path, arm: str, **kwargs) -> dict:
    return validate_rescue_preflight_v2(
        root, root / ARMS[arm]["contract"], arm=arm, require_clean_runtime=False, **kwargs
    )


# ---- Positive control ----


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_both_arms_pass_and_read_their_own_delta_and_ceiling(tmp_path, arm):
    root = _stage(tmp_path)
    report = _check(root, arm)
    assert report["delta"] == ARMS[arm]["delta"]
    assert report["charged_call_ceiling"] == ARMS[arm]["ceiling"]
    assert report["charged_calls_per_cell"] == ARMS[arm]["ceiling"]
    assert report["status"] == "PREPARED_NO_SCORED_AUTHORIZATION"
    assert report["automatic_retries"] == 0
    assert report["cell"] == "5ht1b_2"
    assert report["source_global_index"] == 8


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_supplying_the_contract_identity_binds_launch_authority(tmp_path, arm):
    root = _stage(tmp_path)
    contract = _payload(root, arm)
    report = _check(root, arm, authorization_payload_sha256=identity(contract))
    assert report["status"] == "LAUNCH_AUTHORIZATION_BOUND"


def test_the_two_arms_do_not_share_a_delta_or_a_ceiling():
    assert ARMS["d06"]["delta"] != ARMS["d04"]["delta"]
    assert ARMS["d06"]["ceiling"] != ARMS["d04"]["ceiling"]


# ---- Refusals the brief names explicitly ----


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_wrong_authorization_hash(tmp_path, arm):
    root = _stage(tmp_path)
    with pytest.raises(ValueError, match="does not bind the contract payload"):
        _check(root, arm, authorization_payload_sha256="0" * 64)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_an_altered_delta(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["delta"] = 0.5
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="requires delta"):
        _check(root, arm)


@pytest.mark.parametrize(
    "field", ["charged_calls_per_cell", "total_charged_call_ceiling"]
)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_wrong_ceiling(tmp_path, arm, field):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload[field] = 49
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="requires"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_missing_zero_oracle_gate(tmp_path, arm):
    root = _stage(tmp_path, omit=(PINNED_GATE,))
    assert not (root / PINNED_GATE).exists()
    with pytest.raises(ValueError, match="pinned zero-oracle feasibility gate is absent"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_an_unauthorized_extra_field_mutation(tmp_path, arm):
    """Only the recorded launch-path pins may move; anything else breaks continuity."""
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["quietly_added_field"] = "not authorized by anybody"
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="does not reconstruct"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_payload_the_authorization_receipt_does_not_name(tmp_path, arm):
    """With no revision record to prove continuity, the receipt must name the payload."""
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload.pop("launch_path_revision", None)
    payload["quietly_added_field"] = "not authorized by anybody"
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="authorization receipt authorizes"):
        _check(root, arm)


# ---- The design rule: a scientific payload is not authorization metadata ----


@pytest.mark.parametrize("key", sorted(AUTHORIZATION_METADATA_KEYS))
def test_refuses_authorization_metadata_anywhere_in_the_payload(key):
    with pytest.raises(ValueError, match="authorization metadata is present"):
        assert_payload_carries_no_authorization({"status": PREPARED_STATUS, key: {}})


def test_refuses_authorization_metadata_nested_below_the_top_level():
    payload = {"status": PREPARED_STATUS, "evidence": [{"authorization": {}}]}
    with pytest.raises(ValueError, match=r"evidence\[0\]\.authorization"):
        assert_payload_carries_no_authorization(payload)


def test_refuses_a_status_that_asserts_in_payload_authorization():
    payload = {"status": "AUTHORIZED_BY_OWNER_FOR_PROTONATION_RESCUE"}
    with pytest.raises(ValueError, match="asserts in-payload authorization"):
        assert_payload_carries_no_authorization(payload)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_contract_that_absorbed_its_own_authorization(tmp_path, arm):
    """The exact regression: consent written into the payload moves its hash."""
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    before = identity(payload)
    payload["status"] = "AUTHORIZED_BY_OWNER_FOR_PROTONATION_RESCUE"
    payload["authorization"] = {"authorized_payload_sha256": before}
    _write(root, arm, payload)
    assert identity(_payload(root, arm)) != before
    with pytest.raises(ValueError, match="authorization metadata is present"):
        _check(root, arm)


# ---- Identity, mechanism and budget guards ----


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_foreign_cell(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["cells"][0]["cell"] = "braf_0"
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="cell identity drift"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_drifted_source_index_or_smiles(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["cells"][0]["source_global_index"] = 7
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="source index drift"):
        _check(root, arm)
    payload = _payload(root, arm)
    payload["cells"][0]["source_global_index"] = 8
    payload["cells"][0]["smiles"] = "c1ccccc1"
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="source SMILES drift"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_two_cells(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["cells"].append(dict(payload["cells"][0]))
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="exactly one"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_missing_protonation_expert(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    del payload["proposal"]["protonation_aware_retained_subgraph"]
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="protonation-aware expert is missing"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_automatic_retries(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["automatic_retries"] = 1
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="no automatic retries"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_contract_that_does_not_pin_its_own_wrapper(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    del payload["runtime_inputs_sha256"][ARMS[arm]["app"]]
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="does not pin its own wrapper"):
        _check(root, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_drifted_runtime_input(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    payload["runtime_inputs_sha256"]["src/compose_v4/rewrite/kernel.py"] = "0" * 64
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="runtime input mismatch"):
        _check(root, arm)


def test_refuses_the_other_arms_contract(tmp_path):
    """A wrapper must not validate the sibling arm and inherit its delta and budget."""
    root = _stage(tmp_path)
    with pytest.raises(ValueError, match="expects configs/"):
        validate_rescue_preflight_v2(
            root,
            root / ARMS["d04"]["contract"],
            arm="d06",
            require_clean_runtime=False,
        )


def test_refuses_an_unknown_arm(tmp_path):
    root = _stage(tmp_path)
    with pytest.raises(ValueError, match="unknown protonation rescue arm"):
        validate_rescue_preflight_v2(
            root, root / ARMS["d06"]["contract"], arm="d05", require_clean_runtime=False
        )


# ---- Launch-path revision continuity ----


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_reconstruction_recovers_the_authorized_payload(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    revision = payload.get("launch_path_revision")
    if revision is None:
        pytest.skip("contract carries no launch-path revision")
    rebuilt = identity(reconstruct_superseded_payload(payload))
    assert rebuilt == revision["supersedes_payload_sha256"]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_refuses_a_revision_record_that_does_not_reconstruct(tmp_path, arm):
    root = _stage(tmp_path)
    payload = _payload(root, arm)
    if "launch_path_revision" not in payload:
        pytest.skip("contract carries no launch-path revision")
    payload["launch_path_revision"]["supersedes_payload_sha256"] = "0" * 64
    _write(root, arm, payload)
    with pytest.raises(ValueError, match="does not reconstruct"):
        _check(root, arm)


# ---- The v1 module must stay fit for the older 49-call experiment ----


def test_v1_module_is_untouched_for_the_superseded_experiment():
    assert v1.SCHEMA_VERSION == "t4_5ht1b2_protonation_rescue_contract_v1"
    assert v1.PREPARED_STATUS == "PREPARED_AWAITING_SEPARATE_SCORED_AUTHORIZATION"
    source = (
        REPO / "src/compose_v4/experiments/t4_protonation_rescue_contract.py"
    ).read_text()
    assert 'contract.get("delta") != 0.6' in source
    assert 'contract.get("charged_calls_per_cell") != 49' in source


def test_v2_module_does_not_hardcode_a_single_delta_or_ceiling():
    source = (
        REPO / "src/compose_v4/experiments/t4_protonation_rescue_contract_v2.py"
    ).read_text()
    assert 'contract.get("delta") != 0.6' not in source
    assert "!= 49" not in source
    assert SCHEMA_VERSION.endswith("_v2")
