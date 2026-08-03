"""A governed root must be proven by its owner, not inferred from its neighbours.

``scripts/verify_process_v2_chain.py`` used to prove only the EDGES between
artifacts.  An edge is a statement about two artifacts' hashes; it says nothing
about either artifact's own body.  So a root could be edited anywhere no other
artifact addressed it and the run still reported ``AGREES`` -- and the graph leaf
``P50_RECIPE_POLICY`` is addressed by no ``repository_config`` pointer at all, so
every value inside it was in that position: policy thresholds, the projected body,
and ``shared_policy_registry.registry_identity_sha256``, which is a real pin with no
adjacent path sibling and therefore invisible to any structural pointer scan.

Every test here drives the production entry point over a mutated copy of the real
graph.  The demonstrations that matter use ``validators={}`` as a control arm: with
no owning validator the run falls back to exactly the edge graph that existed
before, so a mutation that the control arm cannot see and the real run refuses is a
measurement of what root validation adds, not an assertion that it exists.

Nothing here hard-codes a contract hash or a contract schema version.  Every value
a test needs is read from the live artifact it is about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    GATE_ZERO_MODEL_PROCESS_V2,
    P50_RECIPE_POLICY,
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD,
    T1_PANEL_POLICY,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# Mutation helpers are REUSED from the sibling suite: ``_reseal`` seals through the
# production rule, so a mutation cannot fail for the wrong reason by also breaking
# the artifact's self-hash.
from test_verify_process_v2_chain import _graph_copy, _reseal  # noqa: E402
from verify_process_v2_chain import (  # noqa: E402
    AGREES,
    DISAGREES,
    FAIL,
    INCONCLUSIVE,
    UNVERIFIED,
    owning_root_validators,
    verify_process_v2_chain,
)

_EDGE_KEY = "process_identity"
_REGISTRY_PIN = ("shared_policy_registry", "registry_identity_sha256")

#: The residue of the committed graph, measured: every hash literal that no
#: pointer, no identity edge, no self-hash and no deterministic rebuild covers.
#: All of it sits inside the two frozen parents, which are pinned by their
#: children's physical AND semantic hashes -- so their bytes cannot drift -- but a
#: hash INSIDE one of them addresses a third artifact that can move without moving
#: the parent, and this verifier does not resolve those.
_MEASURED_UNCHECKED_PINS: list[tuple[str, str]] = [
    (
        "configs/editing_gate_zero_semantic_model_process_v2.json",
        ".process_identity_sha256",
    ),
    *(
        ("configs/editing_v2_semantic_process_v2.json", location)
        for location in (
            ".action_codec_identity.source_sha256.src/compose_v4/rewrite/action_codec_v4.py",
            ".atom_delete_mask_implementation.source_sha256."
            "src/compose_v4/model/factorized_tracelet_rate_model.py",
            ".atom_delete_mask_implementation.source_sha256."
            "src/compose_v4/rewrite/process_v2_atom_delete.py",
            ".canonicalizer.source_sha256.src/compose_v4/rewrite/kernel.py",
            ".downstream_invalidation.superseded_v1_process_identity_sha256",
            ".inherited_v1_process.contract_physical_sha256",
            ".inherited_v1_process.contract_sha256",
            ".lineage.rejected_pre_run_candidate_identity.process_identity_sha256",
            ".lineage.superseded_v1_payload_identity.process_identity_sha256",
            ".unchanged_executor.source_sha256.src/compose_v4/rewrite/kernel.py",
            ".unchanged_executor.source_sha256.src/compose_v4/rewrite/operators.py",
        )
    ),
]


# ---- Helpers ----


def _categories(report: dict[str, Any], severity: str | None = None) -> list[str]:
    return [
        finding["category"]
        for finding in report["findings"]
        if severity is None or finding["severity"] == severity
    ]


def _root_row(report: dict[str, Any], artifact: str) -> dict[str, Any]:
    return next(row for row in report["roots"] if row["artifact"] == artifact)


def _both_arms(artifact: str, mutate: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same mutated graph, verified with and without owning validators.

    The ``validators={}`` arm is the control: it degrades the run to the edge graph
    alone, which is what the verifier proved before roots were validated.
    """

    with _graph_copy() as root:
        _reseal(root, artifact, mutate)
        owned = verify_process_v2_chain(repo_root=root)
        control = verify_process_v2_chain(repo_root=root, validators={})
    return owned, control


# ---- The committed graph ----


def test_every_governed_root_is_validated_by_its_owning_validator() -> None:
    """The root set is read from production, so an eighth contract cannot slip in.

    ``owning_root_validators`` derives its registry from
    ``PROCESS_V2_CHAIN_ARTIFACTS`` rather than transcribing a list, and this asserts
    the two agree. A contract added to the chain and forgotten here would otherwise
    be a root that nothing proves while the run still says ``AGREES``.
    """

    report = verify_process_v2_chain(repo_root=_ROOT)
    assert report["status"] == AGREES, report["findings"]
    assert [row["artifact"] for row in report["roots"]] == list(PROCESS_V2_CHAIN_ARTIFACTS)
    assert {row["result"] for row in report["roots"]} == {"validated"}
    assert set(owning_root_validators()) == set(PROCESS_V2_CHAIN_ARTIFACTS)


def test_agreement_requires_every_root_to_have_been_validated() -> None:
    """``AGREES`` and an unvalidated root must never coexist, whatever the reason."""

    scenarios: list[dict[str, Any]] = [verify_process_v2_chain(repo_root=_ROOT)]
    with _graph_copy() as root:
        scenarios.append(verify_process_v2_chain(repo_root=root))
        scenarios.append(verify_process_v2_chain(repo_root=root, validators={}))
        _reseal(root, P50_RECIPE_POLICY, lambda payload: payload.pop(_EDGE_KEY))
        scenarios.append(verify_process_v2_chain(repo_root=root))
    for report in scenarios:
        if report["status"] != AGREES:
            continue
        assert {row["result"] for row in report["roots"]} == {"validated"}, report["roots"]


# ---- What root validation adds over the edge graph ----


def test_a_body_edit_no_pointer_addresses_is_caught_only_by_the_owning_validator() -> None:
    """The graph leaf: nothing pins it, so the edge graph is structurally blind here.

    The mutated field is read from the live artifact rather than named, so the test
    keeps working when the projected policy body changes shape.
    """

    live = json.loads((_ROOT / P50_RECIPE_POLICY).read_bytes())
    edited = sorted(
        key
        for key, value in live.items()
        if isinstance(value, str) and key != SELF_HASH_FIELD
    )[0]

    def edit(payload: dict[str, Any]) -> None:
        payload[edited] = f"{payload[edited]} (an edit no other artifact addresses)"

    owned, control = _both_arms(P50_RECIPE_POLICY, edit)

    assert owned["status"] == DISAGREES
    refusals = [
        finding
        for finding in owned["findings"]
        if finding["category"] == "root_artifact_refused_by_its_validator"
    ]
    assert [finding["location"] for finding in refusals] == [P50_RECIPE_POLICY]
    assert _root_row(owned, P50_RECIPE_POLICY)["result"] == "refused"

    # The control arm proves the claim is a measurement: without an owning
    # validator the edit produces no pointer, identity or hash finding at all.
    assert set(_categories(control)) == {"root_validator_unknown"}


def test_a_bare_hash_pin_with_no_adjacent_path_is_proven_by_the_rebuild() -> None:
    """The named gap, closed for governed roots.

    ``shared_policy_registry.registry_identity_sha256`` is a pin: it addresses the
    shared policy registry, and the registry moves whenever a frozen policy source
    or the projection allowlist moves. It carries no adjacent ``path`` sibling, so
    no structural pointer scan can see it and no pointer edge covers it. Before the
    root was rebuilt deterministically it read as checked and was not.
    """

    block, pin = _REGISTRY_PIN
    live = json.loads((_ROOT / P50_RECIPE_POLICY).read_bytes())[block][pin]
    forged = "0" * 64
    assert live != forged

    owned, control = _both_arms(
        P50_RECIPE_POLICY, lambda payload: payload[block].__setitem__(pin, forged)
    )

    assert owned["status"] == DISAGREES
    assert "root_artifact_refused_by_its_validator" in _categories(owned, FAIL)
    assert set(_categories(control)) == {"root_validator_unknown"}


def test_the_self_hash_alone_does_not_make_a_root_sound() -> None:
    """Resealing keeps an artifact self-consistent; it does not make it correct.

    Every mutation in this file is resealed through the production rule, so the
    self-hash always agrees with the body. That is exactly the state in which a
    self-hash proves nothing, and the deterministic rebuild is what still refuses.
    """

    block, pin = _REGISTRY_PIN
    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, lambda payload: payload[block].__setitem__(pin, "0" * 64))
        payload = json.loads((root / P50_RECIPE_POLICY).read_bytes())
        report = verify_process_v2_chain(repo_root=root)
    # The artifact still declares exactly one satisfied self-hash field, and the
    # semantic pointers its parents would use still resolve.
    assert isinstance(payload[SELF_HASH_FIELD], str)
    assert "ambiguous_self_hash" not in _categories(report)
    assert "stale_semantic_pointer" not in _categories(report)
    assert "root_artifact_refused_by_its_validator" in _categories(report, FAIL)


# ---- Roots nothing owns, and roots nothing can validate ----


def test_a_root_with_no_owning_validator_is_unverified_never_assumed_sound() -> None:
    report = verify_process_v2_chain(repo_root=_ROOT, validators={})
    assert report["status"] == INCONCLUSIVE
    assert _categories(report, FAIL) == []
    assert _categories(report, UNVERIFIED) == ["root_validator_unknown"] * len(
        PROCESS_V2_CHAIN_ARTIFACTS
    )
    assert {row["result"] for row in report["roots"]} == {"no_validator"}


# ---- A validator may be withheld, never substituted ----


def _forge_the_registry_pin(payload: dict[str, Any]) -> None:
    """A tamper inside the graph leaf that only its owning validator can see."""

    block, pin = _REGISTRY_PIN
    payload[block][pin] = "0" * 64


def test_a_substituted_validator_is_refused_unrun_rather_than_believed() -> None:
    """``validators={}`` was refused correctly; a mapping of NO-OPS was not.

    That is the hole root validation exists to close, reopened through the seam
    that measures it. Every root reported ``validated``, the run reported
    ``AGREES`` over a tampered governed root, and the leaf's unchecked pins were
    absorbed into the covered set -- because ``proven`` membership followed the
    validator not raising, and a callable that does nothing never raises.

    A validator is EVIDENCE. Withholding it makes a run weaker, which is the
    control arm; supplying one's own would let the caller write the verdict, so it
    is refused without being invoked.
    """

    invoked: list[Path] = []

    def a_validator_of_my_own(repo_root: Path) -> None:
        invoked.append(repo_root)

    substituted = dict.fromkeys(PROCESS_V2_CHAIN_ARTIFACTS, a_validator_of_my_own)
    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, _forge_the_registry_pin)
        owned = verify_process_v2_chain(repo_root=root)
        injected = verify_process_v2_chain(repo_root=root, validators=substituted)

    assert owned["status"] == DISAGREES
    assert "root_artifact_refused_by_its_validator" in _categories(owned, FAIL)

    assert injected["status"] == INCONCLUSIVE
    assert invoked == [], "a substituted validator must never be invoked"
    assert {row["result"] for row in injected["roots"]} == {"substituted_validator"}
    assert _categories(injected, UNVERIFIED) == ["root_validator_not_registered"] * len(
        PROCESS_V2_CHAIN_ARTIFACTS
    )
    assert _categories(injected, FAIL) == []


def test_an_injected_validator_can_only_make_a_run_weaker_never_stronger() -> None:
    """The property, stated over the report rather than over one scenario.

    A substituted validator must land a run exactly where withholding one lands
    it: nothing proven, every pin still listed. It absorbed 6 of the leaf's pins
    before, which is the "looks checked but is not" failure one level up.
    """

    substituted = dict.fromkeys(PROCESS_V2_CHAIN_ARTIFACTS, lambda repo_root: None)
    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, _forge_the_registry_pin)
        injected = verify_process_v2_chain(repo_root=root, validators=substituted)
        withheld = verify_process_v2_chain(repo_root=root, validators={})
        owned = verify_process_v2_chain(repo_root=root)

    assert injected["unchecked_pins"] == withheld["unchecked_pins"]
    assert len(injected["unchecked_pins"]) > len(owned["unchecked_pins"])
    assert [pin for pin in injected["unchecked_pins"] if pin["artifact"] == P50_RECIPE_POLICY]
    assert AGREES not in {injected["status"], withheld["status"]}


def test_a_registered_validator_moved_onto_another_root_is_refused() -> None:
    """The second half of the rule: registered is not enough, it must own THIS root.

    Pointing the leaf at another contract's validator would prove that contract
    and report the leaf as validated, which is a stale binding of exactly the kind
    this verifier refuses everywhere else.
    """

    registry = owning_root_validators()
    swapped = {**registry, P50_RECIPE_POLICY: registry[T1_PANEL_POLICY]}
    report = verify_process_v2_chain(repo_root=_ROOT, validators=swapped)

    assert report["status"] == INCONCLUSIVE
    named = [
        finding
        for finding in report["findings"]
        if finding["category"] == "root_validator_not_registered"
    ]
    assert [finding["location"] for finding in named] == [P50_RECIPE_POLICY]
    assert _root_row(report, P50_RECIPE_POLICY)["result"] == "substituted_validator"
    assert {
        row["result"] for row in report["roots"] if row["artifact"] != P50_RECIPE_POLICY
    } == {"validated"}


def test_withholding_one_validator_is_supported_and_weakens_only_that_root() -> None:
    """The seam stays usable: a partial mapping is a partial control arm."""

    registry = owning_root_validators()
    withheld = {name: registry[name] for name in registry if name != P50_RECIPE_POLICY}
    report = verify_process_v2_chain(repo_root=_ROOT, validators=withheld)

    assert report["status"] == INCONCLUSIVE
    assert _categories(report, UNVERIFIED) == ["root_validator_unknown"]
    assert _root_row(report, P50_RECIPE_POLICY)["result"] == "no_validator"
    assert {
        row["result"] for row in report["roots"] if row["artifact"] != P50_RECIPE_POLICY
    } == {"validated"}
    assert [pin for pin in report["unchecked_pins"] if pin["artifact"] == P50_RECIPE_POLICY]


def test_a_frozen_parent_used_as_a_root_is_not_silently_accepted() -> None:
    """Rooting at a frozen parent proves nothing, and the run must say so.

    ``configs/editing_gate_zero_semantic_model_process_v2.json`` is a legitimate
    graph member and an illegitimate root: it declares no process-identity edge and
    no chain validator owns it.
    """

    report = verify_process_v2_chain(repo_root=_ROOT, roots=(GATE_ZERO_MODEL_PROCESS_V2,))
    assert report["status"] != AGREES
    assert _root_row(report, GATE_ZERO_MODEL_PROCESS_V2)["result"] != "validated"


def test_a_skipped_root_validation_is_recorded_and_still_blocks_agreement() -> None:
    """The one case where the owning validator is deliberately not invoked.

    When the identity-edge rule has already refused an artifact, invoking the
    owning validator can only restate the same defect less specifically ("field set
    differs from the deterministic rebuild" instead of "declares no process-identity
    edge"). This is the same precedence the pointer checks apply, where role
    coherence is checked before the hash so the specific diagnostic is not degraded.

    What must not happen is the skip becoming silent, so it is recorded by name and
    the run can never agree while a root carries it.
    """

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, lambda payload: payload.pop(_EDGE_KEY))
        report = verify_process_v2_chain(repo_root=root)
    row = _root_row(report, P50_RECIPE_POLICY)
    assert row["result"] == "not_validated"
    assert row["detail"]
    assert report["status"] == INCONCLUSIVE
    assert "no_identity_edge_declared" in _categories(report, UNVERIFIED)


def test_an_absent_root_is_never_reported_as_validated() -> None:
    with _graph_copy() as root:
        (root / P50_RECIPE_POLICY).unlink()
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "missing_graph_artifact" in _categories(report, FAIL)
    assert _root_row(report, P50_RECIPE_POLICY)["result"] != "validated"


def test_an_unparseable_root_is_never_reported_as_validated() -> None:
    with _graph_copy() as root:
        (root / P50_RECIPE_POLICY).write_bytes(b"{not json")
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "unparseable_artifact" in _categories(report, FAIL)
    assert _root_row(report, P50_RECIPE_POLICY)["result"] != "validated"


# ---- The residue, stated rather than implied ----


def test_no_governed_root_leaves_an_unchecked_pin() -> None:
    """Every hash literal inside a root is accounted for by the rebuild."""

    report = verify_process_v2_chain(repo_root=_ROOT)
    governed = set(PROCESS_V2_CHAIN_ARTIFACTS)
    assert [pin for pin in report["unchecked_pins"] if pin["artifact"] in governed] == []


def test_the_unchecked_pins_of_the_frozen_parents_are_listed_by_name() -> None:
    """A pin that looks checked but is not is the worst case, so it is named.

    The frozen parents are pinned by their children's physical AND semantic hashes,
    so their bytes cannot drift unnoticed. What is genuinely unverified is a hash
    INSIDE one of them that addresses a third artifact: that third artifact can move
    without changing the parent's bytes, so every pointer in the graph still agrees.
    This verifier does not resolve those, and the report says so rather than leaving
    a reader to infer coverage from an ``AGREES``.
    """

    report = verify_process_v2_chain(repo_root=_ROOT)
    assert report["status"] == AGREES, report["findings"]
    pins = report["unchecked_pins"]
    assert pins, "the inventory is empty, so nothing proves it is being computed"
    assert set(pin["artifact"] for pin in pins) <= set(report["artifacts"])
    for pin in pins:
        payload = json.loads((_ROOT / pin["artifact"]).read_bytes())
        assert pin["sha256"] in json.dumps(payload), pin
        assert len(pin["sha256"]) == 64
    # The SIZE of the residue is the number that matters, and nothing pinned it:
    # a run that started leaving another pin unchecked would report it here and
    # still pass every assertion above. The inventory is measured, so growing it
    # is a deliberate edit to this list, not a silent one.
    assert [(pin["artifact"], pin["location"]) for pin in pins] == _MEASURED_UNCHECKED_PINS
    assert len(pins) == 12


def test_a_root_whose_rebuild_was_refused_is_not_counted_as_covering_its_pins() -> None:
    """Coverage follows what was PROVEN, never what was merely attempted.

    A refused root has no deterministic rebuild behind it, so the bare pins inside
    it are exactly as unverified as they were before the phase existed. Marking them
    covered because a validator ran would be the "looks checked but is not" failure
    the inventory exists to prevent, one level up.
    """

    block, pin = _REGISTRY_PIN
    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, lambda payload: payload[block].__setitem__(pin, "0" * 64))
        report = verify_process_v2_chain(repo_root=root)
    assert _root_row(report, P50_RECIPE_POLICY)["result"] == "refused"
    listed = [entry for entry in report["unchecked_pins"] if entry["artifact"] == P50_RECIPE_POLICY]
    assert [entry["location"] for entry in listed] == [f".{block}.{pin}"], report["unchecked_pins"]


def test_a_root_the_identity_rule_refused_is_not_counted_as_covering_its_pins() -> None:
    """The same rule for the other way a root goes unproven."""

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, lambda payload: payload.pop(_EDGE_KEY))
        report = verify_process_v2_chain(repo_root=root)
    assert _root_row(report, P50_RECIPE_POLICY)["result"] == "not_validated"
    assert [
        entry for entry in report["unchecked_pins"] if entry["artifact"] == P50_RECIPE_POLICY
    ], report["unchecked_pins"]


def test_an_unchecked_pin_becomes_checked_when_its_artifact_is_rebuilt() -> None:
    """The inventory is a function of what was proven, not a static allowlist."""

    owned = verify_process_v2_chain(repo_root=_ROOT)
    control = verify_process_v2_chain(repo_root=_ROOT, validators={})
    governed = set(PROCESS_V2_CHAIN_ARTIFACTS)
    assert [pin for pin in owned["unchecked_pins"] if pin["artifact"] in governed] == []
    assert [pin for pin in control["unchecked_pins"] if pin["artifact"] in governed] != []
