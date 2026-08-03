"""Regression tests for the Process-V2 completion-binding predicate.

Before it will resolve anything, ``resolve_process_v2_admitted_source`` reads the
published run completion and refuses it unless it is a completion of the declared
kind and version, carries the non-authorizing status, claims no training
authority, is sealed by its own hash, and binds *this* plan -- its run, its plan
bytes, its task inventory and its V1 payload binding -- together with the *live*
Process-V2 identity.  Ten clauses implement that refusal.

An adversarial pass over Wave 1 disabled each clause in turn and re-ran the whole
rebind test group.  Only ``process_v2_identity_sha256`` was pinned, by
``test_editing_process_v2_admitted_source.py``'s
``test_a_completion_that_does_not_bind_the_live_identity_refuses``.  Every other
clause could be deleted with the entire group still green, so a completion that
named a different run, a different plan, a different task inventory or a
different payload binding, that declared the wrong schema, version or status,
that claimed training authority, or whose own seal was simply wrong, was accepted
by every test in the repository.  This module pins the nine that were unpinned.

Method
------

Each test tampers with exactly one field of the *published* completion artifact
and then drives the production resolver.  Nothing here reimplements the
predicate, and the tampering is chosen so that only the guard under test can
refuse:

* for the eight fields the resolver compares against something else, the
  completion is **re-sealed**, so the self-hash clause cannot be what refuses;
* for the self-hash clause itself, ``gate_zero_run`` -- a field the resolver
  reads nowhere -- is moved and the seal is left **stale**, so the seal is the
  only thing that can refuse.

That is what makes each test fail when, and only when, its own clause is removed.
``test_the_republished_completion_still_resolves_unchanged`` is the control that
keeps the other nine honest: it proves the tamper harness itself resolves.

The fixture payload, its published overlay and the re-seal helper are the ones
the adapter's own test module already builds through the production writers and
the production rebind, so every artifact under test is a real published artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from compose_v4.data.editing_process_v2_admitted_source import ProcessV2AdmittedSourceError
from compose_v4.data.editing_process_v2_rebind import COMPLETION_FILENAME

# The V1 payload fixture builders are not importable as a package.  pytest already
# puts ``tests`` on ``sys.path`` under the default prepend import mode; the entry
# is added explicitly so this module also imports under ``importmode=importlib``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_process_v2_admitted_source as adapter_fixture  # noqa: E402

# That module installs the rebind's own independent oracle as the effective-mask
# authority when the production symbol is absent from the worktree.  Rebinding its
# autouse fixture here keeps this module under the same authority, so both build
# the same overlay.
_effective_mask_authority = adapter_fixture._effective_mask_authority

_TASKS = adapter_fixture._ADMITTED_ONLY_TASKS


# ---- Tampering with the published completion ----------------------------------


def _published(tmp_path: Path):
    """Build, prove and publish one real Process-V2 rebind run over the fixture."""

    return adapter_fixture._build_and_prove(tmp_path / "artifacts", _TASKS)


def _completion(fixture) -> dict:
    """The completion exactly as the production reducer published it."""

    return json.loads((fixture.run_root / COMPLETION_FILENAME).read_text())


def _foreign_digest(value: object) -> str:
    """A different, well-formed 64-hex digest, so shape is never what refuses.

    Substituting ``"0" * 64`` would leave open that a guard rejected the *shape*
    of the value rather than the binding it expresses.
    """

    return hashlib.sha256(str(value).encode()).hexdigest()


def _republish_completion(fixture, *, reseal: bool, **overrides: object) -> dict:
    """Rewrite the published completion in place, optionally re-sealing it.

    Every override must name a field the completion already carries and must
    change it, so a renamed or dropped completion field fails loudly here instead
    of quietly making a test vacuous.  With ``reseal=True`` the seal is recomputed
    over the tampered body, leaving the forged completion indistinguishable from
    an honest one except in the single field under test.
    """

    path = fixture.run_root / COMPLETION_FILENAME
    completion = _completion(fixture)
    for name, value in overrides.items():
        assert name in completion, f"the completion carries no field {name!r}"
        assert completion[name] != value, f"the completion already carries {name}={value!r}"
        completion[name] = value
    if reseal:
        completion["completion_sha256"] = adapter_fixture._self_hash(
            completion, "completion_sha256"
        )
    path.write_bytes(
        json.dumps(completion, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )
    return completion


# ---- The control --------------------------------------------------------------


def test_the_republished_completion_still_resolves_unchanged(tmp_path: Path) -> None:
    """Without this, every refusal below could be the harness rather than a guard.

    ``_republish_completion`` rewrites and re-seals the published completion.  If
    that alone made the run unresolvable, all nine refusals would pass for the
    wrong reason and would keep passing with every guard deleted.  Resolving to
    the identical identity proves the harness is faithful, so a later refusal is
    attributable to the one field that test moved.
    """

    fixture = _published(tmp_path)
    before = adapter_fixture._resolve(fixture).identity()
    _republish_completion(fixture, reseal=True)
    assert adapter_fixture._resolve(fixture).identity() == before


# ---- The completion must be a sealed, non-authorizing completion --------------


def test_a_completion_declaring_a_foreign_schema_refuses(tmp_path: Path) -> None:
    """Without this, any JSON object under the completion filename is a completion.

    The resolver reads the completion by path, not by kind.  With the schema
    clause deleted, a document of a different kind -- here the rebind plan's own
    schema -- served at that path is accepted and its fields are read as if they
    were a completion's.
    """

    fixture = _published(tmp_path)
    _republish_completion(fixture, reseal=True, schema=str(fixture.plan["schema"]))
    with pytest.raises(ProcessV2AdmittedSourceError, match="completion is malformed"):
        adapter_fixture._resolve(fixture)


def test_a_completion_declaring_a_foreign_schema_version_refuses(tmp_path: Path) -> None:
    """Without this, a completion written by an incompatible reducer resolves.

    The completion schema is versioned because its fields have changed meaning
    before.  With the version clause deleted, a document from another version is
    read field-by-field under this version's assumptions instead of being
    refused.
    """

    fixture = _published(tmp_path)
    published = _completion(fixture)
    _republish_completion(
        fixture, reseal=True, schema_version=int(published["schema_version"]) + 1
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="completion is malformed"):
        adapter_fixture._resolve(fixture)


def test_a_completion_whose_status_claims_authority_refuses(tmp_path: Path) -> None:
    """Without this, a completion can rename its own status into an authorizing one.

    ``COMPLETION_STATUS`` is the single string that says this artifact confers no
    training authority, and downstream readers assert on it.  With the status
    clause deleted, the completion chooses its own status and the resolver
    accepts a run that advertises itself as authorized.
    """

    fixture = _published(tmp_path)
    _republish_completion(
        fixture, reseal=True, status="COMPLETE_V1_COMPATIBILITY_PROOF_TRAINING_AUTHORIZED"
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="completion is malformed"):
        adapter_fixture._resolve(fixture)


def test_a_completion_claiming_training_authority_refuses(tmp_path: Path) -> None:
    """Without this, a completion can grant itself the authority it must not hold.

    The adapter's whole contract is that it resolves a source and confers nothing.
    With this clause deleted, a completion carrying ``training_authorized: true``
    -- correctly sealed, so nothing else can catch it -- resolves, and the one
    check standing between a granted field and a resolved source is gone.
    """

    fixture = _published(tmp_path)
    _republish_completion(fixture, reseal=True, training_authorized=True)
    with pytest.raises(ProcessV2AdmittedSourceError, match="completion is malformed"):
        adapter_fixture._resolve(fixture)


def test_a_completion_whose_own_seal_is_stale_refuses(tmp_path: Path) -> None:
    """Without this, a completion whose own self-hash is wrong is accepted.

    The seal is what makes every completion field the reducer published tamper
    evident, including the fields the resolver never reads individually.  This
    test moves exactly such a field, ``gate_zero_run``, and leaves the seal stale:
    no other clause looks at it, so with the seal clause deleted the forged
    completion resolves and the seal covers nothing at all.
    """

    fixture = _published(tmp_path)
    _republish_completion(fixture, reseal=False, gate_zero_run=True)
    with pytest.raises(ProcessV2AdmittedSourceError, match="completion is malformed"):
        adapter_fixture._resolve(fixture)


# ---- The completion must bind this plan ---------------------------------------


def test_a_completion_naming_a_different_run_refuses(tmp_path: Path) -> None:
    """Without this, a completion proved for one run resolves another run's plan.

    ``run_identity_sha256`` is what ties the completion to the run namespace its
    task results live in.  With this clause deleted, the resolver reads the
    completion of a different run as the completion of this one, and the census
    it publishes describes a run nobody proved.
    """

    fixture = _published(tmp_path)
    _republish_completion(
        fixture,
        reseal=True,
        run_identity_sha256=_foreign_digest(fixture.plan["run_identity_sha256"]),
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="does not bind this plan"):
        adapter_fixture._resolve(fixture)


def test_a_completion_naming_a_different_plan_refuses(tmp_path: Path) -> None:
    """Without this, a completion for a different plan resolves against this one.

    ``plan_sha256`` is the content address of the plan the reduction actually
    reduced.  With this clause deleted the completion no longer has to name the
    plan being resolved, so a reduction of some other plan supplies the census.
    """

    fixture = _published(tmp_path)
    _republish_completion(
        fixture, reseal=True, plan_sha256=_foreign_digest(fixture.plan["plan_sha256"])
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="does not bind this plan"):
        adapter_fixture._resolve(fixture)


def test_a_completion_naming_a_different_task_inventory_refuses(tmp_path: Path) -> None:
    """Without this, a completion over a different set of tasks resolves.

    ``task_inventory_sha256`` is the content address of the exact range-task set
    the plan schedules.  With this clause deleted, a completion reduced over a
    different sharding of the same payload is accepted, and the run's execution
    address and its proof of completeness cease to be the same object.
    """

    fixture = _published(tmp_path)
    _republish_completion(
        fixture,
        reseal=True,
        task_inventory_sha256=_foreign_digest(fixture.plan["task_inventory_sha256"]),
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="does not bind this plan"):
        adapter_fixture._resolve(fixture)


def test_a_completion_naming_a_different_v1_payload_binding_refuses(tmp_path: Path) -> None:
    """Without this, a completion proved over different V1 chemistry resolves.

    ``v1_payload_binding_sha256`` is the content address of the immutable V1
    payload the overlay decided.  It is the last field tying the completion to
    the chemistry, so with this clause deleted a completion proved over another
    corpus is accepted and the resolved source silently changes what corpus it
    is.
    """

    fixture = _published(tmp_path)
    _republish_completion(
        fixture,
        reseal=True,
        v1_payload_binding_sha256=_foreign_digest(fixture.plan["v1_payload_binding_sha256"]),
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="does not bind this plan"):
        adapter_fixture._resolve(fixture)
