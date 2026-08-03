"""A resolved evidence binding must pin measured provenance and grant nothing.

THE FIXTURE IS REAL, AND THAT IS THE POINT
------------------------------------------
Version 1 of this module drove every test through a hand-written admitted-source
descriptor.  That descriptor carried ``physical_inventory_sha256``,
``process_identity_sha256`` and ``semantic_evidence_sha256`` -- three names no
version of the adapter has ever emitted -- and the binding required all three.  So
``build_resolved_evidence_binding`` could not have been called on a real
``ProcessV2AdmittedSource.identity()`` at all, and all thirty-seven tests passed,
because the fixture had been shaped to the code rather than to the artifact and
there was no production caller to disagree.

Every test below therefore runs against a descriptor produced by the production
writers over real molecules and proved by the production rebind.  A hand-written
descriptor is used exactly once more, in
``test_a_hand_written_descriptor_cannot_substitute_for_the_real_one``, to prove
that it is now refused rather than accepted.

Two failure modes are the reason the artifact exists at all, and both are silent.

1. **Measured evidence smuggled into a frozen contract.** The seven prospective
   contracts used to advertise a fillable ``admitted_source`` slot that could never
   be filled. Measured provenance lives here instead and the contract is cited,
   never mutated.
2. **Authority acquired by accident.** A binding proves identity; it never permits
   training or a launch. Since admitted-source schema 3 there is one authority
   vocabulary rather than two, so the descriptor's block is checked at its own
   depth rather than translated.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_process_v2_admitted_source import (
    ADMITTED_SOURCE_SCHEMA,
    ADMITTED_SOURCE_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_policy_registry import FROZEN_POLICY_SOURCES
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    RESOLVED_EVIDENCE_BINDING_SCHEMA,
    RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
    RESOLVED_EVIDENCE_BINDING_STATUS,
    IdentityRole,
    PointerKind,
    canonical_sha256,
    typed_pointer,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    EDITING_CORPUS_V2_CONTRACT,
    GATE_ZERO_STRUCTURAL,
    P50_RECIPE_POLICY,
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
)
from compose_v4.experiments.editing_v2_process_v2_evidence_binding import (
    BINDING_SELF_HASH_FIELD,
    EVIDENCE_BINDING_STAGES,
    RESOLVED_EVIDENCE_BINDING_FIELDS,
    ProcessV2EvidenceBindingError,
    build_resolved_evidence_binding,
    load_resolved_evidence_binding,
    resolved_evidence_binding_self_hash,
    serialize_resolved_evidence_binding,
    validate_resolved_evidence_binding,
    write_resolved_evidence_binding,
)

_ROOT = Path(__file__).resolve().parents[1]

# The V1 payload fixture builders are not importable as a package.  pytest already
# puts ``tests`` on ``sys.path`` under the default prepend import mode; the entry
# is added explicitly so this module also imports under ``importmode=importlib``.
if str(_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tests"))

import test_editing_process_v2_admitted_source as adapter_fixture  # noqa: E402

# That module installs the rebind's own independent oracle as the effective-mask
# authority when the production symbol is absent from the worktree.  Rebinding its
# autouse fixture keeps this module under the same authority.
_effective_mask_authority = adapter_fixture._effective_mask_authority


# ---- The real admitted source ----


@pytest.fixture(scope="module")
def real_identity(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One real published overlay, resolved and described. Built once.

    Module-scoped because building and proving it runs the production rebind, and
    every test below wants the same descriptor rather than a fresh one.
    """

    root = tmp_path_factory.mktemp("evidence_binding_payload")
    fixture = adapter_fixture._build_and_prove(
        root / "artifacts", adapter_fixture._TASKS_WITH_EXCLUSION
    )
    return adapter_fixture._resolve(fixture).identity()


def _prerequisites(**overrides: Any) -> dict[str, dict[str, Any]]:
    prerequisites = {
        "chunk_cache_completion": typed_pointer(
            kind=PointerKind.REMOTE_ARTIFACT,
            provider="process_v2_chunk_cache",
            target="/editing_v2/fixture_chunk_cache/CACHE_COMPLETE.json",
            target_schema="compose.editing_v2.process_v2.chunk_cache_completion",
            identity_role=IdentityRole.SEMANTIC,
            hash_algorithm="self_hash_field_v1",
            sha256="b1" * 32,
        ),
        "editing_corpus_contract": typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="frozen_repository_artifact",
            target=EDITING_CORPUS_V2_CONTRACT,
            target_schema="compose.editing_corpus_contract",
            identity_role=IdentityRole.PHYSICAL,
            sha256=hashlib.sha256(
                (_ROOT / EDITING_CORPUS_V2_CONTRACT).read_bytes()
            ).hexdigest(),
        ),
    }
    prerequisites.update(overrides)
    return prerequisites


def _build(identity: dict[str, Any], stage: str = "gate_zero", **kwargs: Any) -> dict[str, Any]:
    return build_resolved_evidence_binding(
        stage=stage,
        admitted_source=kwargs.pop("admitted_source", identity),
        measured_prerequisites=kwargs.pop("measured_prerequisites", _prerequisites()),
        repo_root=kwargs.pop("repo_root", _ROOT),
        **kwargs,
    )


def _reseal_source(binding: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Swap the embedded descriptor and reseal the BINDING around it.

    Used to prove the two self-hashes are independent: a resealed binding does not
    make a tampered descriptor valid.
    """

    body = {
        key: value
        for key, value in binding.items()
        if key != BINDING_SELF_HASH_FIELD
    }
    body["admitted_source"] = source
    return {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}


def _reseal_identity(source: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in sorted(source.items())
        if key != "admitted_source_sha256"
    }
    sealed = {**body, "admitted_source_sha256": canonical_sha256(body)}
    return dict(sorted(sealed.items()))


@contextmanager
def _repo_copy() -> Iterator[Path]:
    """A throwaway root holding the cited contracts and their frozen sources."""

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        sources = [source.relative_path for source in FROZEN_POLICY_SOURCES.values()]
        for relative_path in (
            *PROCESS_V2_CHAIN_ARTIFACTS,
            EDITING_CORPUS_V2_CONTRACT,
            "src/compose_v4/data/editing_process_v2_admitted_source.py",
            *sources,
        ):
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / relative_path, target)
        yield root


# ---- (a) the required end-to-end acceptance ----


def test_a_real_resolved_source_builds_validates_serializes_reloads_and_revalidates(
    real_identity: dict[str, Any],
) -> None:
    """The acceptance the version-1 fixture could not express.

    A real ``ProcessV2AdmittedSource.identity()`` was structurally unbuildable
    under version 1: it omits the three names that version required.
    """

    binding = _build(real_identity)
    assert binding["admitted_source"] == real_identity
    assert validate_resolved_evidence_binding(binding, repo_root=_ROOT) == binding
    with tempfile.TemporaryDirectory() as raw:
        path = write_resolved_evidence_binding(binding, path=Path(raw) / "binding.json")
        reloaded = load_resolved_evidence_binding(path, repo_root=_ROOT)
    assert reloaded == binding
    # The descriptor survived a `sort_keys=True` round trip unchanged, which is
    # what embedding it verbatim requires.
    assert reloaded["admitted_source"] == real_identity
    assert validate_resolved_evidence_binding(reloaded, repo_root=_ROOT) == reloaded


def test_a_hand_written_descriptor_cannot_substitute_for_the_real_one() -> None:
    """The version-1 fixture, verbatim, must now be refused.

    This is the exact object every version-1 test used. It is not merely
    incomplete: three of its six identity fields name nothing the adapter emits,
    which is why a binding built from it proved nothing about the real one.
    """

    version_one_fixture: dict[str, Any] = {
        "adapter_implementation_sha256": "a1" * 32,
        "completion_sha256": "a2" * 32,
        "physical_inventory_sha256": "a3" * 32,
        "process_identity_sha256": "a4" * 32,
        "run_identity_sha256": "a5" * 32,
        "semantic_evidence_sha256": "a6" * 32,
        "counts": {
            "source_entries": 30,
            "admitted_entries": 22,
            "rejected_entries": 8,
        },
        "rejected_traces_by_code": {
            "fixture_reason_alpha": 5,
            "fixture_reason_beta": 3,
        },
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "p50_authorized": False,
    }
    with pytest.raises(
        ProcessV2EvidenceBindingError, match="not one this binding can pin"
    ) as raised:
        _build(version_one_fixture)
    # It is not even shaped like the artifact: it declares no schema at all, which
    # is how far from a real descriptor the version-1 fixture was.
    assert "declares schema None" in str(raised.value)

    # Give it the envelope it lacked, so the refusal has to come from the FIELDS.
    # Now the fabricated names are named as unexpected and the real ones as
    # missing, which is the divergence itself rather than a generic mismatch.
    enveloped = {
        **version_one_fixture,
        "schema": ADMITTED_SOURCE_SCHEMA,
        "schema_version": ADMITTED_SOURCE_SCHEMA_VERSION,
        "status": "V2_ADMISSION_OVERLAY_RESOLVED_NO_TRAINING_AUTHORITY",
    }
    del enveloped["p50_authorized"]
    enveloped.update(dict.fromkeys(AUTHORITY_FIELDS, False))
    with pytest.raises(ProcessV2EvidenceBindingError) as raised:
        _build(_reseal_identity(enveloped))
    message = str(raised.value)
    unexpected = message.split("unexpected=")[1]
    assert "physical_inventory_sha256" in unexpected
    assert "semantic_evidence_sha256" in unexpected
    missing = message.split("missing=")[1].split("unexpected=")[0]
    assert "admitted_evidence_sha256" in missing
    assert "rejection_ledger" in missing


def test_this_module_carries_no_fabricated_admitted_source_descriptor() -> None:
    """A copied descriptor must not creep back in as a convenience.

    The single hand-written object above is the refusal test's subject, so it is
    the only permitted occurrence; anything else would be a second fixture that
    can drift from the artifact exactly as the first one did.
    """

    # `process_identity_sha256` is deliberately NOT listed: it is a real field of
    # the contract's own `process_identity` block, and only its appearance in an
    # admitted-source descriptor was fabricated. These two exist nowhere.
    fabricated = {"physical_inventory_sha256", "semantic_evidence_sha256"}
    tree = ast.parse(Path(__file__).read_text())
    mentions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name != "test_this_module_carries_no_fabricated_admitted_source_descriptor"
        and fabricated & {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
    }
    assert mentions == {
        "test_a_hand_written_descriptor_cannot_substitute_for_the_real_one",
        "test_schema_version_one_is_refused_with_its_reason",
    }, (
        "the fabricated names belong only where they are proven refused; anywhere "
        f"else is a second fixture that can drift from the artifact: {sorted(mentions)}"
    )
    # Nothing at module scope may hold a descriptor either: the version-1 fixture
    # was a module-level constant reused by every test.
    module_constants = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not any(
        name.lstrip("_").lower().startswith(("fixture", "admitted", "identity"))
        for name in module_constants
    ), sorted(module_constants)


def test_building_the_same_inputs_twice_is_byte_identical(
    real_identity: dict[str, Any],
) -> None:
    first = serialize_resolved_evidence_binding(_build(real_identity))
    second = serialize_resolved_evidence_binding(_build(real_identity))
    assert first == second
    assert first.endswith(b"\n")


def test_every_stage_binds_its_own_governing_contract(
    real_identity: dict[str, Any],
) -> None:
    assert dict(EVIDENCE_BINDING_STAGES) == {
        "gate_zero": GATE_ZERO_STRUCTURAL,
        "t1_panel": T1_PANEL_POLICY,
        "t1_capacity": T1_CAPACITY_POLICY,
        "bounded_p50": P50_RECIPE_POLICY,
    }
    for stage, contract in sorted(EVIDENCE_BINDING_STAGES.items()):
        binding = _build(real_identity, stage)
        assert binding["stage"] == stage
        assert binding["prospective_contract"]["physical"]["target"] == contract
        assert binding["prospective_contract"]["semantic"]["target"] == contract


def test_an_unknown_stage_is_refused(real_identity: dict[str, Any]) -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="not a resolved-evidence"):
        _build(real_identity, "p2000")


def test_the_schema_envelope_is_the_frozen_one(real_identity: dict[str, Any]) -> None:
    binding = _build(real_identity)
    assert binding["schema"] == RESOLVED_EVIDENCE_BINDING_SCHEMA
    assert binding["schema_version"] == RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION
    assert binding["status"] == RESOLVED_EVIDENCE_BINDING_STATUS
    assert binding[BINDING_SELF_HASH_FIELD] == resolved_evidence_binding_self_hash(binding)
    assert BINDING_SELF_HASH_FIELD != SELF_HASH_FIELD, (
        "a binding must not reuse a contract's self-hash field name"
    )


def test_a_smuggled_measured_evidence_block_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """Failure mode 1, arriving through the binding rather than the contract.

    A binding is the artifact that carries measured provenance downstream, so an
    unknown block sealed into one is carried by every consumer that reads it. The
    validator had no exact-field-set check at all -- both sibling validators do --
    so this resealed exactly, validated clean, and reloaded from disk unchanged.
    """

    binding = _build(real_identity)
    body = {key: value for key, value in binding.items() if key != BINDING_SELF_HASH_FIELD}
    body["measured_later_evidence"] = {
        "checkpoint_sha256": "9" * 64,
        "selected_step": 16000,
    }
    smuggled = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
    # It is sealed correctly, so no other guard has anything to say about it.
    assert smuggled[BINDING_SELF_HASH_FIELD] == resolved_evidence_binding_self_hash(smuggled)
    with pytest.raises(ProcessV2EvidenceBindingError, match="field set differs") as raised:
        validate_resolved_evidence_binding(smuggled, repo_root=_ROOT)
    assert "measured_later_evidence" in str(raised.value).split("unexpected=")[1]

    # And it does not survive a round trip through the publisher either.
    with tempfile.TemporaryDirectory() as raw:
        path = write_resolved_evidence_binding(smuggled, path=Path(raw) / "binding.json")
        with pytest.raises(ProcessV2EvidenceBindingError, match="field set differs"):
            load_resolved_evidence_binding(path, repo_root=_ROOT)


def test_the_declared_field_set_is_exactly_what_the_builder_emits(
    real_identity: dict[str, Any],
) -> None:
    """The constant may not drift from the builder, in either direction."""

    binding = _build(real_identity)
    assert set(binding) == set(RESOLVED_EVIDENCE_BINDING_FIELDS)
    assert list(RESOLVED_EVIDENCE_BINDING_FIELDS) == sorted(RESOLVED_EVIDENCE_BINDING_FIELDS)
    assert set(AUTHORITY_FIELDS) < set(RESOLVED_EVIDENCE_BINDING_FIELDS)
    assert BINDING_SELF_HASH_FIELD in RESOLVED_EVIDENCE_BINDING_FIELDS


def test_the_prospective_contract_block_carries_exactly_three_fields(
    real_identity: dict[str, Any],
) -> None:
    """The nested key set, which the top-level field set cannot see."""

    binding = _build(real_identity)
    body = {key: value for key, value in binding.items() if key != BINDING_SELF_HASH_FIELD}
    body["prospective_contract"] = {
        **body["prospective_contract"],
        "measured_at_launch_sha256": "7" * 64,
    }
    resealed = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
    with pytest.raises(ProcessV2EvidenceBindingError, match="must carry exactly"):
        validate_resolved_evidence_binding(resealed, repo_root=_ROOT)


def test_a_binding_that_declares_another_status_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """A binding that says it is something else is not this artifact."""

    binding = _build(real_identity)
    body = {key: value for key, value in binding.items() if key != BINDING_SELF_HASH_FIELD}
    body["status"] = "MEASURED_EVIDENCE_AUTHORIZES_THE_BOUNDED_P50_LAUNCH"
    relabelled = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
    with pytest.raises(ProcessV2EvidenceBindingError, match="status is") as raised:
        validate_resolved_evidence_binding(relabelled, repo_root=_ROOT)
    assert RESOLVED_EVIDENCE_BINDING_STATUS in str(raised.value)


def test_a_contract_expecting_another_binding_schema_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """The version is checked; so is the schema NAME, which is the other half."""

    with _repo_copy() as root:
        target = root / GATE_ZERO_STRUCTURAL
        payload = json.loads(target.read_bytes())
        payload["resolved_evidence_binding"]["cited_by_schema"] = (
            "compose.editing_v2.process_v2.measured_evidence_binding"
        )
        target.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        with pytest.raises(ProcessV2EvidenceBindingError, match="expects to be cited by"):
            _build(real_identity, "gate_zero", repo_root=root)


def test_a_binding_naming_another_self_hash_field_for_its_contract_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """Resealed, so only the pinned self-hash field name can refuse it.

    Which field carries a contract's self-hash is what a consumer needs in order
    to check that contract at all; a binding that names the wrong one sends every
    reader to a field that does not exist.
    """

    binding = _build(real_identity)
    body = {key: value for key, value in binding.items() if key != BINDING_SELF_HASH_FIELD}
    body["prospective_contract"] = {
        **body["prospective_contract"],
        "self_hash_field": "evidence_binding_sha256",
    }
    resealed = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
    assert resealed[BINDING_SELF_HASH_FIELD] == resolved_evidence_binding_self_hash(resealed)
    with pytest.raises(
        ProcessV2EvidenceBindingError, match="prospective_contract.self_hash_field"
    ):
        validate_resolved_evidence_binding(resealed, repo_root=_ROOT)


def test_schema_version_one_is_refused_with_its_reason(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    body = {k: v for k, v in binding.items() if k != BINDING_SELF_HASH_FIELD}
    body["schema_version"] = 1
    downgraded = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
    with pytest.raises(ProcessV2EvidenceBindingError, match="no migration path") as raised:
        validate_resolved_evidence_binding(downgraded, repo_root=_ROOT)
    message = str(raised.value)
    for fabricated in (
        "physical_inventory_sha256",
        "process_identity_sha256",
        "semantic_evidence_sha256",
    ):
        assert fabricated in message


# ---- (b) the prospective contract is cited, never mutated ----


def test_the_binding_pins_path_physical_semantic_and_the_self_hash_field(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity, "t1_capacity")
    block = binding["prospective_contract"]
    assert set(block) == {"physical", "self_hash_field", "semantic"}
    assert block["self_hash_field"] == SELF_HASH_FIELD
    raw = (_ROOT / T1_CAPACITY_POLICY).read_bytes()
    assert block["physical"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert block["semantic"]["sha256"] == json.loads(raw)[SELF_HASH_FIELD]
    assert block["physical"]["identity_role"] == IdentityRole.PHYSICAL
    assert block["semantic"]["identity_role"] == IdentityRole.SEMANTIC
    assert block["semantic"]["hash_algorithm"] == "self_hash_field_v1"


def test_a_stale_contract_pin_is_refused(real_identity: dict[str, Any]) -> None:
    binding = _build(real_identity)
    binding["prospective_contract"]["physical"]["sha256"] = "0" * 64
    with pytest.raises(ProcessV2EvidenceBindingError, match="but the live value is"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_a_binding_that_cites_another_stages_contract_is_refused(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity, "gate_zero")
    other = _build(real_identity, "bounded_p50")
    binding["prospective_contract"] = other["prospective_contract"]
    with pytest.raises(ProcessV2EvidenceBindingError, match="is governed by"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_an_absent_contract_makes_the_binding_unbuildable(
    real_identity: dict[str, Any],
) -> None:
    with _repo_copy() as root:
        (root / GATE_ZERO_STRUCTURAL).unlink()
        with pytest.raises(ProcessV2EvidenceBindingError, match="not readable"):
            _build(real_identity, "gate_zero", repo_root=root)


def test_an_edited_contract_makes_an_existing_binding_stale(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity, "t1_panel")
    with _repo_copy() as root:
        target = root / T1_PANEL_POLICY
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(ProcessV2EvidenceBindingError):
            validate_resolved_evidence_binding(binding, repo_root=root)


def test_a_contract_expecting_another_binding_version_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """The contract is the side that says which schema may cite it."""

    with _repo_copy() as root:
        target = root / GATE_ZERO_STRUCTURAL
        payload = json.loads(target.read_bytes())
        payload["resolved_evidence_binding"]["cited_by_schema_version"] = 1
        target.write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        )
        with pytest.raises(
            ProcessV2EvidenceBindingError, match="expects resolved-evidence-binding"
        ):
            _build(real_identity, "gate_zero", repo_root=root)


def test_a_contract_binding_another_process_identity_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """Evidence from one process may not be joined to another's frozen policy."""

    with _repo_copy() as root:
        target = root / GATE_ZERO_STRUCTURAL
        payload = json.loads(target.read_bytes())
        payload["process_identity"]["process_identity_sha256"] = "7" * 64
        target.write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        )
        with pytest.raises(
            ProcessV2EvidenceBindingError, match="may not join evidence from one process"
        ):
            _build(real_identity, "gate_zero", repo_root=root)


# ---- (c) the embedded descriptor is the adapter's own artifact ----


def test_the_descriptor_is_embedded_verbatim_and_nothing_is_projected(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    embedded = binding["admitted_source"]
    assert embedded == real_identity
    assert embedded["schema"] == ADMITTED_SOURCE_SCHEMA
    assert embedded["schema_version"] == ADMITTED_SOURCE_SCHEMA_VERSION
    # Its own authority block travels with it, in the one frozen vocabulary.
    for name in AUTHORITY_FIELDS:
        assert embedded[name] is False, name
        assert binding[name] is False, name


def test_a_disagreeing_admitted_source_schema_is_refused(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    broken = _reseal_source(
        binding,
        _reseal_identity({**real_identity, "schema": "compose.data.process_v2_admitted_source"}),
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="not one this binding can pin"):
        validate_resolved_evidence_binding(broken, repo_root=_ROOT)


def test_the_two_self_hashes_are_independent(real_identity: dict[str, Any]) -> None:
    """Resealing the binding does not make a tampered descriptor valid."""

    binding = _build(real_identity)

    # Tampered descriptor, binding NOT resealed: the binding's own self-hash fails.
    unsealed = dict(binding)
    unsealed["admitted_source"] = {**real_identity, "completion_sha256": "3" * 64}
    with pytest.raises(ProcessV2EvidenceBindingError):
        validate_resolved_evidence_binding(unsealed, repo_root=_ROOT)

    # Tampered descriptor, binding resealed around it: the binding self-hash now
    # agrees, and the descriptor's OWN self-hash is what refuses.
    resealed = _reseal_source(
        binding, {**real_identity, "completion_sha256": "3" * 64}
    )
    assert resealed[BINDING_SELF_HASH_FIELD] == resolved_evidence_binding_self_hash(resealed)
    with pytest.raises(ProcessV2EvidenceBindingError, match="self-hash"):
        validate_resolved_evidence_binding(resealed, repo_root=_ROOT)


def test_a_descriptor_from_a_different_adapter_is_refused(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    broken = _reseal_source(
        binding,
        _reseal_identity({**real_identity, "adapter_implementation_sha256": "0" * 64}),
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="different adapter"):
        validate_resolved_evidence_binding(broken, repo_root=_ROOT)


def test_a_missing_admitted_source_is_refused(real_identity: dict[str, Any]) -> None:
    binding = _build(real_identity)
    del binding["admitted_source"]
    with pytest.raises(ProcessV2EvidenceBindingError, match="must be an object"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


# ---- (d) the census must reconcile, through the owning validator ----


def test_a_census_that_does_not_reconcile_is_refused(
    real_identity: dict[str, Any],
) -> None:
    broken = _reseal_identity(
        {
            **real_identity,
            "counts": dict(
                sorted({**real_identity["counts"], "source_entries": 99}.items())
            ),
        }
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="does not reconcile"):
        _build(real_identity, admitted_source=broken)


def test_a_reason_census_that_disagrees_with_the_count_is_refused(
    real_identity: dict[str, Any],
) -> None:
    assert real_identity["rejected_traces_by_code"], "the fixture must reject something"
    code = next(iter(real_identity["rejected_traces_by_code"]))
    broken = _reseal_identity({**real_identity, "rejected_traces_by_code": {code: 4}})
    with pytest.raises(ProcessV2EvidenceBindingError, match="sums to 4"):
        _build(real_identity, admitted_source=broken)


def test_a_coerced_count_is_refused(real_identity: dict[str, Any]) -> None:
    for value in ("22", 22.0, True, None):
        broken = _reseal_identity(
            {
                **real_identity,
                "counts": dict(
                    sorted({**real_identity["counts"], "admitted_entries": value}.items())
                ),
            }
        )
        with pytest.raises(ProcessV2EvidenceBindingError, match="not an exact int"):
            _build(real_identity, admitted_source=broken)


# ---- (e) authority ----


def test_every_authority_field_is_present_and_false(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    for name in AUTHORITY_FIELDS:
        assert binding[name] is False, name
    assert "bounded_p50_authorized" in AUTHORITY_FIELDS


def test_a_granted_authority_field_is_refused(real_identity: dict[str, Any]) -> None:
    for name in AUTHORITY_FIELDS:
        binding = _build(real_identity)
        body = {k: v for k, v in binding.items() if k != BINDING_SELF_HASH_FIELD}
        body[name] = True
        granted = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
        with pytest.raises(ProcessV2EvidenceBindingError, match="grants authority"):
            validate_resolved_evidence_binding(granted, repo_root=_ROOT)


def test_a_grant_nested_in_the_embedded_descriptor_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """A top-level scan would miss this; the descriptor is nested by construction."""

    binding = _build(real_identity)
    resealed = _reseal_source(
        binding, _reseal_identity({**real_identity, "t1_authorized": True})
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="grants authority") as raised:
        validate_resolved_evidence_binding(resealed, repo_root=_ROOT)
    assert "admitted_source.t1_authorized" in str(raised.value)


def test_a_grant_under_an_unregistered_name_is_refused(
    real_identity: dict[str, Any],
) -> None:
    """The vocabulary-aware guard cannot see a name nobody has registered."""

    binding = _build(real_identity)
    body = {k: v for k, v in binding.items() if k != BINDING_SELF_HASH_FIELD}
    body["p500_authorized"] = True
    granted = {**body, BINDING_SELF_HASH_FIELD: canonical_sha256(body)}
    assert "p500_authorized" not in AUTHORITY_FIELDS
    with pytest.raises(ProcessV2EvidenceBindingError, match="grants authority"):
        validate_resolved_evidence_binding(granted, repo_root=_ROOT)


def test_the_retired_spelling_is_refused_in_the_embedded_descriptor(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    retired = {k: v for k, v in real_identity.items() if k != "bounded_p50_authorized"}
    retired["p50_authorized"] = False
    resealed = _reseal_source(binding, _reseal_identity(retired))
    with pytest.raises(ProcessV2EvidenceBindingError, match="retired authority field"):
        validate_resolved_evidence_binding(resealed, repo_root=_ROOT)


# ---- (f) measured prerequisites ----


def test_a_missing_repository_prerequisite_is_refused(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    binding["measured_prerequisites"]["editing_corpus_contract"]["target"] = (
        "configs/no_such_contract.json"
    )
    with pytest.raises(ProcessV2EvidenceBindingError, match="which is not present"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_a_stale_repository_prerequisite_is_refused(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    binding["measured_prerequisites"]["editing_corpus_contract"]["sha256"] = "1" * 64
    with pytest.raises(ProcessV2EvidenceBindingError, match="the live physical hash is"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_a_remote_prerequisite_is_never_looked_for_locally(
    real_identity: dict[str, Any],
) -> None:
    """A volume path is verified by its stage loader, not by this validator."""

    binding = _build(real_identity)
    remote = binding["measured_prerequisites"]["chunk_cache_completion"]
    assert remote["kind"] == PointerKind.REMOTE_ARTIFACT
    assert not (_ROOT / remote["target"].lstrip("/")).exists()
    assert validate_resolved_evidence_binding(binding, repo_root=_ROOT) == binding


def test_a_malformed_prerequisite_pointer_is_refused(
    real_identity: dict[str, Any],
) -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="not a typed pointer"):
        _build(real_identity, measured_prerequisites={"broken": {"path": "configs/x.json"}})


def test_prerequisites_are_serialized_in_a_stable_order(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    assert list(binding["measured_prerequisites"]) == sorted(
        binding["measured_prerequisites"]
    )


# ---- (g) self-hash and publication ----


def test_a_disagreeing_self_hash_is_refused(real_identity: dict[str, Any]) -> None:
    binding = _build(real_identity)
    binding[BINDING_SELF_HASH_FIELD] = "2" * 64
    with pytest.raises(ProcessV2EvidenceBindingError, match="self-hash"):
        validate_resolved_evidence_binding(binding, repo_root=_ROOT)


def test_publication_leaves_no_partial_file_behind(
    real_identity: dict[str, Any],
) -> None:
    binding = _build(real_identity)
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        path = write_resolved_evidence_binding(binding, path=directory / "binding.json")
        assert path.is_file()
        assert [child.name for child in directory.iterdir()] == ["binding.json"]


def test_an_unparseable_binding_is_refused() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "binding.json"
        path.write_bytes(b"{nope")
        with pytest.raises(ProcessV2EvidenceBindingError, match="not valid JSON"):
            load_resolved_evidence_binding(path, repo_root=_ROOT)


def test_an_absent_binding_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="not readable"):
        load_resolved_evidence_binding(Path("/nonexistent/binding.json"), repo_root=_ROOT)


def test_a_non_object_binding_is_refused() -> None:
    with pytest.raises(ProcessV2EvidenceBindingError, match="must be a JSON object"):
        validate_resolved_evidence_binding([], repo_root=_ROOT)
