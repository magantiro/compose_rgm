"""The Process-V2 contract chain must mirror V1 exactly and never mix with it.

Two failures are possible here and both are silent, so both get an explicit test.

1. **Redesign disguised as a re-pin.** A chain that re-pins hashes while quietly
   moving a threshold, a cell definition, or a partition count is not a mirror; it
   is a new policy with an old name. Every mirrored block is therefore read back
   out of the FROZEN V1 config on disk and compared, so an edited constant in the
   builder fails here rather than shipping.
2. **A mixed V1/V2 artifact.** The V1 chain and the V2 chain describe different
   processes and their hashes are not interchangeable. Each mixing defect -- a V1
   identity pin, a ``_v1`` parent path, a moved Active8 order, a granted authority
   flag, a stale self-hash, a changed field set -- is asserted to raise with a
   message naming THAT defect, because a chain that fails only at the self-hash
   tells a reader nothing about which binding drifted.

Expectations are literals or values read from the V1/parent files themselves, never
recomputations of the builder under test.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    ACTIVE8_DECISION_RUNTIME,
    AUTHORITY_FIELDS,
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
    P50_RECIPE_POLICY,
    PROCESS_V2_CHAIN_ARTIFACTS,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
    ProcessV2ChainError,
    build_process_v2_chain_artifact,
    load_process_v2_chain_artifact,
    process_v2_chain_self_hash,
    serialize_process_v2_chain_artifact,
    validate_process_v2_chain_artifact,
    write_process_v2_chain,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)

_ROOT = Path(__file__).resolve().parents[1]

# The commit that froze the V1 chain.  Nothing under a ``_v1`` name may differ from
# its bytes at this revision.
_FROZEN_V1_BASE_REVISION = "d5cfcaf"

# The four frozen non-chain files the chain binds.  None of them is V1-named.
_EXTERNAL_PARENTS: tuple[str, ...] = (
    "configs/editing_corpus_v2_contract.json",
    "configs/editing_gate_zero_semantic_model_process_v2.json",
    "configs/editing_v2_semantic_process_v2.json",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
)

# The V1 artifact each chain member mirrors.
_V1_COUNTERPART: dict[str, str] = {
    ACTIVE8_DECISION_RUNTIME: "configs/editing_v2_semantic_active8_decision_runtime_v1.json",
    CAPABILITY_CELLS: "configs/editing_v2_semantic_capability_cells_v1.json",
    DEVELOPMENT_CELL_ROLES: "configs/editing_v2_semantic_development_cell_roles_v1.json",
    GATE_ZERO_STRUCTURAL: "configs/editing_v2_semantic_gate_zero_structural_v1.json",
    T1_PANEL_POLICY: "configs/editing_v2_semantic_t1_panel_policy_v1.json",
    T1_CAPACITY_POLICY: "configs/editing_v2_semantic_t1_capacity_policy_v1.json",
    P50_RECIPE_POLICY: "configs/editing_v2_semantic_p50_recipe_policy_v1.json",
}


# ---- Helpers ----


def _load(relative_path: str) -> dict:
    return json.loads((_ROOT / relative_path).read_bytes())


def _physical(relative_path: str) -> str:
    return hashlib.sha256((_ROOT / relative_path).read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def _isolated_repo() -> Iterator[Path]:
    """A throwaway repo root holding only the frozen external parents.

    The chain is written there rather than into the working tree so that a
    byte-stability test can never rewrite a committed artifact as a side effect.
    """

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for relative_path in _EXTERNAL_PARENTS:
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / relative_path, target)
        yield root


def _git_show(revision: str, relative_path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "-C", str(_ROOT), "show", f"{revision}:{relative_path}"],
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def _revision_available(revision: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _v1_config_paths() -> list[str]:
    return sorted(
        str(path.relative_to(_ROOT))
        for path in (_ROOT / "configs").glob("*.json")
        if "_v1" in path.stem
    )


# ---- (a) build -> validate -> load round trip ----


def test_every_artifact_round_trips_build_validate_load() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        built = build_process_v2_chain_artifact(name, repo_root=_ROOT)
        validated = validate_process_v2_chain_artifact(built, name=name, repo_root=_ROOT)
        assert validated == built
        loaded = load_process_v2_chain_artifact(name, repo_root=_ROOT)
        assert loaded == built, f"{name} on disk differs from its deterministic rebuild"
        assert (_ROOT / name).read_bytes() == serialize_process_v2_chain_artifact(built)


def test_chain_has_seven_artifacts_in_dependency_order() -> None:
    assert PROCESS_V2_CHAIN_ARTIFACTS == (
        ACTIVE8_DECISION_RUNTIME,
        CAPABILITY_CELLS,
        DEVELOPMENT_CELL_ROLES,
        GATE_ZERO_STRUCTURAL,
        T1_PANEL_POLICY,
        T1_CAPACITY_POLICY,
        P50_RECIPE_POLICY,
    )
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        assert name.startswith("configs/editing_v2_process_v2_")
        assert "_v1" not in Path(name).stem


def test_each_artifact_carries_exactly_one_self_hash_field() -> None:
    """The chain verifier DISCOVERS a self-hash by its equation, so there must be one."""

    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = _load(name)
        satisfying = [
            key
            for key, value in payload.items()
            if key.endswith("_sha256")
            and isinstance(value, str)
            and value == _canonical({k: v for k, v in payload.items() if k != key})
        ]
        assert satisfying == ["contract_sha256"], f"{name} self-hash fields: {satisfying}"
        assert payload["contract_sha256"] == process_v2_chain_self_hash(payload)


# ---- (b) byte stability ----


def test_write_process_v2_chain_is_byte_stable_across_runs() -> None:
    with _isolated_repo() as root:
        first = write_process_v2_chain(root)
        first_bytes = {n: (root / n).read_bytes() for n in PROCESS_V2_CHAIN_ARTIFACTS}
        second = write_process_v2_chain(root)
        second_bytes = {n: (root / n).read_bytes() for n in PROCESS_V2_CHAIN_ARTIFACTS}
        assert first == second
        assert first_bytes == second_bytes
        assert list(first) == list(PROCESS_V2_CHAIN_ARTIFACTS)
        for name in PROCESS_V2_CHAIN_ARTIFACTS:
            assert first_bytes[name] == (_ROOT / name).read_bytes(), (
                f"the committed {name} is not what the builder produces"
            )
            assert first_bytes[name].endswith(b"\n")


def test_write_process_v2_chain_seals_parents_before_children() -> None:
    """A child pins its parent's FINAL bytes, so an unwritten parent must be an error."""

    with _isolated_repo() as root:
        with pytest.raises(ProcessV2ChainError, match="parents-first"):
            build_process_v2_chain_artifact(GATE_ZERO_STRUCTURAL, repo_root=root)
        sealed = write_process_v2_chain(root)
        structural = json.loads((root / GATE_ZERO_STRUCTURAL).read_bytes())
        for role, parent in (
            ("decision_runtime", ACTIVE8_DECISION_RUNTIME),
            ("capability_cell_registry", CAPABILITY_CELLS),
            ("development_cell_roles", DEVELOPMENT_CELL_ROLES),
        ):
            assert structural["parents"][role]["semantic_sha256"] == sealed[parent]


# ---- (c) parent pins resolve to real physical and semantic hashes ----


def test_parent_pins_resolve_to_the_real_physical_and_semantic_hashes() -> None:
    # ``semantic_sha256`` expectations are read out of the target itself, never
    # recomputed with the builder's own helper.
    corpus = "configs/editing_corpus_v2_contract.json"
    classifier = "src/compose_v4/data/editing_v2_semantic_capability_cells.py"
    expected_semantic: dict[str, str] = {
        "configs/editing_gate_zero_semantic_model_process_v2.json": _load(
            "configs/editing_gate_zero_semantic_model_process_v2.json"
        )["contract_sha256"],
        "configs/editing_v2_semantic_process_v2.json": _load(
            "configs/editing_v2_semantic_process_v2.json"
        )["contract_sha256"],
        corpus: _canonical(_load(corpus)),
        classifier: _physical(classifier),
    }
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        expected_semantic[name] = _load(name)["contract_sha256"]

    seen_targets: set[str] = set()
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        parents = _load(name)["parents"]
        assert parents, f"{name} pins no parent"
        for role, pin in parents.items():
            assert set(pin) == {"path", "file_sha256", "semantic_sha256"}, (
                f"{name}:{role} is not the Gate-0 canonical triple"
            )
            target = pin["path"]
            seen_targets.add(target)
            assert (_ROOT / target).is_file()
            assert pin["file_sha256"] == _physical(target), f"{name}:{role} physical hash"
            assert pin["semantic_sha256"] == expected_semantic[target], (
                f"{name}:{role} semantic hash"
            )

    assert seen_targets - set(PROCESS_V2_CHAIN_ARTIFACTS) == set(_EXTERNAL_PARENTS)


def test_every_artifact_binds_the_live_process_v2_identity() -> None:
    live_v2 = str(editing_process_v2_identity()["process_identity_sha256"])
    live_v1 = str(editing_v2_process_identity()["process_identity_sha256"])
    assert live_v1 != live_v2
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        identity = _load(name)["process_identity"]
        assert identity["process_identity_sha256"] == live_v2
        assert identity["process_semantics"] == "semantic_editing_v2_v2"
        assert identity["identity_schema"] == "compose.editing.semantic_process_v2_identity"
        assert identity["provider"] == "editing_process_v2_identity"


def test_every_artifact_declares_an_admitted_source_slot() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        block = _load(name)["admitted_source"]
        assert block == {
            "schema": "compose.data.process_v2_admitted_source",
            "completion_sha256": None,
            "run_identity_sha256": None,
        }


# ---- (d) a V1 identity pin is rejected ----


def test_v1_identity_pin_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(GATE_ZERO_STRUCTURAL, repo_root=_ROOT)
    payload["process_identity"] = dict(payload["process_identity"])
    payload["process_identity"]["process_identity_sha256"] = str(
        editing_v2_process_identity()["process_identity_sha256"]
    )
    with pytest.raises(ProcessV2ChainError, match="binds the V1 semantic process identity"):
        validate_process_v2_chain_artifact(
            payload, name=GATE_ZERO_STRUCTURAL, repo_root=_ROOT
        )


def test_an_unknown_identity_pin_is_rejected_distinctly() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    payload["process_identity"] = dict(payload["process_identity"])
    payload["process_identity"]["process_identity_sha256"] = "0" * 64
    with pytest.raises(ProcessV2ChainError) as raised:
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)
    message = str(raised.value)
    assert "which is not the live Process-V2 identity" in message
    assert "V1 semantic process identity" not in message


# ---- (e) a _v1 parent path is rejected ----


def test_v1_parent_path_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    v1_roles = _load("configs/editing_v2_semantic_development_cell_roles_v1.json")
    payload["parents"] = {
        "development_cell_roles": {
            "path": "configs/editing_v2_semantic_development_cell_roles_v1.json",
            "file_sha256": _physical(
                "configs/editing_v2_semantic_development_cell_roles_v1.json"
            ),
            "semantic_sha256": v1_roles["policy_sha256"],
        }
    }
    with pytest.raises(ProcessV2ChainError, match="must never bind a _v1 config"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)


def test_a_stale_parent_pin_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=_ROOT)
    payload["parents"] = json.loads(json.dumps(payload["parents"]))
    payload["parents"]["t1_panel_policy"]["semantic_sha256"] = "1" * 64
    with pytest.raises(ProcessV2ChainError, match="but the live value is"):
        validate_process_v2_chain_artifact(
            payload, name=T1_CAPACITY_POLICY, repo_root=_ROOT
        )


# ---- (f) authority flags ----


def test_every_authority_flag_is_false_in_all_seven_artifacts() -> None:
    assert AUTHORITY_FIELDS == (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
    )
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = _load(name)
        for field in AUTHORITY_FIELDS:
            assert payload[field] is False, f"{name}:{field}"
        assert payload["status"].endswith("_NO_DOWNSTREAM_AUTHORITY"), name


def test_a_granted_authority_flag_is_rejected() -> None:
    for field in AUTHORITY_FIELDS:
        payload = build_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=_ROOT)
        payload[field] = True
        with pytest.raises(ProcessV2ChainError, match="not exactly False"):
            validate_process_v2_chain_artifact(
                payload, name=P50_RECIPE_POLICY, repo_root=_ROOT
            )


def test_gate_zero_structural_grants_no_authority_on_pass() -> None:
    decision = _load(GATE_ZERO_STRUCTURAL)["decision_policy"]
    granted = {key: value for key, value in decision.items() if key.startswith("pass_grants_")}
    assert granted and all(value is False for value in granted.values())


# ---- (g) Active8 order ----


def test_active8_order_matches_the_corpus_contract_constant() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        assert tuple(_load(name)["active_families"]) == tuple(ACTIVE8_FAMILIES), name
    assert tuple(_load(T1_CAPACITY_POLICY)["required_families"]) == tuple(ACTIVE8_FAMILIES)


def test_a_reordered_active8_list_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["active_families"] = list(reversed(ACTIVE8_FAMILIES))
    with pytest.raises(ProcessV2ChainError, match="differs from the Active8"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


# ---- Remaining loud-failure modes ----


def test_a_disagreeing_self_hash_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(DEVELOPMENT_CELL_ROLES, repo_root=_ROOT)
    payload["contract_sha256"] = "2" * 64
    with pytest.raises(ProcessV2ChainError, match="disagrees"):
        validate_process_v2_chain_artifact(
            payload, name=DEVELOPMENT_CELL_ROLES, repo_root=_ROOT
        )


def test_a_changed_field_set_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    payload["extra_policy_knob"] = 1
    with pytest.raises(ProcessV2ChainError, match="field set differs"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)

    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    del payload["support_time_hex"]
    with pytest.raises(ProcessV2ChainError, match="field set differs"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)


def test_a_silently_moved_policy_value_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=_ROOT)
    payload["thresholds"] = dict(payload["thresholds"])
    payload["thresholds"]["minimum_unique_state_teacher_successor_probability"] = 0.5
    with pytest.raises(ProcessV2ChainError, match="mirrored V1 policy values are frozen"):
        validate_process_v2_chain_artifact(
            payload, name=T1_CAPACITY_POLICY, repo_root=_ROOT
        )


def test_an_unknown_artifact_name_is_rejected() -> None:
    with pytest.raises(ProcessV2ChainError, match="is not a Process-V2 chain artifact"):
        build_process_v2_chain_artifact("configs/nope.json", repo_root=_ROOT)


def test_a_non_object_artifact_is_rejected() -> None:
    with pytest.raises(ProcessV2ChainError, match="must be a JSON object"):
        validate_process_v2_chain_artifact([], name=T1_PANEL_POLICY, repo_root=_ROOT)


# ---- (h) the frozen V1 chain is untouched ----


def test_no_v1_config_changed_since_the_frozen_base_revision() -> None:
    if not _revision_available(_FROZEN_V1_BASE_REVISION):
        pytest.skip(f"{_FROZEN_V1_BASE_REVISION} is not reachable from this checkout")
    paths = _v1_config_paths()
    assert len(paths) >= 7, "the V1 chain configs must be present to be checked"
    for relative_path in paths:
        frozen = _git_show(_FROZEN_V1_BASE_REVISION, relative_path)
        assert frozen is not None, f"{relative_path} is absent at the frozen base"
        assert (_ROOT / relative_path).read_bytes() == frozen, (
            f"{relative_path} is a frozen V1 artifact and must not be modified"
        )


def test_every_mirrored_v1_counterpart_is_present_and_distinct() -> None:
    for name, counterpart in _V1_COUNTERPART.items():
        assert (_ROOT / counterpart).is_file()
        assert _physical(name) != _physical(counterpart)


# ---- (i) the mirror really is a mirror ----


def test_capability_cell_definitions_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[CAPABILITY_CELLS])
    v2 = _load(CAPABILITY_CELLS)
    assert v2["family_contexts"] == v1["family_contexts"]
    assert v2["cell_identity_policy"] == v1["cell_identity_policy"]
    assert v2["cell_identity_policy"]["namespace"] == "editing_v2_active8_v1"
    assert v2["exact_evidence_strata"] == v1["exact_evidence_strata"]
    assert v2["fail_closed_policy"] == v1["fail_closed_policy"]
    assert v2["scope"] == v1["scope"]
    assert v2["bindings"]["data_lanes"] == v1["bindings"]["data_lanes"]
    assert v2["bindings"]["partition_roles"] == v1["bindings"]["partition_roles"]
    assert (
        v2["bindings"]["action_codec_schema_version"]
        == v1["bindings"]["action_codec_schema_version"]
    )


def test_development_cell_roles_and_counts_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[DEVELOPMENT_CELL_ROLES])
    v2 = _load(DEVELOPMENT_CELL_ROLES)
    for field in (
        "required_cell_ids",
        "conditional_cell_ids",
        "separate_lane_cell_ids",
        "conditional_policy",
        "separate_lane_policy",
        "partition_policy",
        "scope",
    ):
        assert v2[field] == v1[field], field
    assert v2["partition_policy"]["required_cell_count"] == 17
    assert v2["partition_policy"]["conditional_cell_count"] == 3
    assert v2["partition_policy"]["separate_lane_cell_count"] == 2
    assert len(v2["required_cell_ids"]) == 17
    assert len(v2["conditional_cell_ids"]) == 3
    assert len(v2["separate_lane_cell_ids"]) == 2


def test_decision_runtime_model_and_software_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[ACTIVE8_DECISION_RUNTIME])
    v2 = _load(ACTIVE8_DECISION_RUNTIME)
    assert v2["software"] == v1["software"]
    # The ONE intended difference: the V2 operator capability expectation, read
    # from the Gate-0 Process-V2 model/process contract this runtime pins.
    assert set(v2["model"]) - set(v1["model"]) == {"operator_capability_fingerprint"}
    assert {k: v for k, v in v2["model"].items() if k in v1["model"]} == v1["model"]
    assert v2["model"]["operator_capability_fingerprint"] == (
        _load("configs/editing_gate_zero_semantic_model_process_v2.json")["model_identity"][
            "operator_capability_fingerprint"
        ]
    )
    assert v2["model"]["operator_capability_fingerprint"] == "d79ffe8ef65f3fb3"


def test_gate_zero_structural_blocks_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[GATE_ZERO_STRUCTURAL])
    v2 = _load(GATE_ZERO_STRUCTURAL)
    for field in ("required_architecture", "structural_checks", "decision_policy"):
        assert v2[field] == v1[field], field


def test_t1_panel_policy_thresholds_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[T1_PANEL_POLICY])
    v2 = _load(T1_PANEL_POLICY)
    for field in (
        "maximum_entries_by_family",
        "minimum_entries_by_family",
        "objective_unit",
        "panel_kind",
        "support_time_hex",
        "cache_handoff",
        "hazard_included",
        "gate_thresholds_included",
        "optimizer_policy_included",
        "p50_policy_included",
        "repeated_state_panel_included",
        "successor_fiber_cache_compiled",
        "empirical_multiplicity_receipts_included",
    ):
        assert v2[field] == v1[field], field


def test_t1_capacity_policy_blocks_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[T1_CAPACITY_POLICY])
    v2 = _load(T1_CAPACITY_POLICY)
    for field in (
        "thresholds",
        "panel_cardinality",
        "sampling_law",
        "optimization",
        "required_families",
        "empirical_repeated_state_gate",
        "objective_unit",
        "panel_kind",
        "support_time_hex",
        "hazard_included",
    ):
        assert v2[field] == v1[field], field


def test_p50_recipe_policy_blocks_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[P50_RECIPE_POLICY])
    v2 = _load(P50_RECIPE_POLICY)
    for field in (
        "optimization",
        "objective",
        "time_derivation",
        "sampling",
        "cache",
        "thresholds",
        "active_families",
        "required_physical_binding_purposes",
        "scientific_scope",
        "p500_authorized",
    ):
        assert v2[field] == v1[field], field
