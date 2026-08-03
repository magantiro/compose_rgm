"""End-to-end tests for the Process-V2 rebind: plan, execute, reduce, resolve.

``tests/test_editing_process_v2_rebind.py`` proves the rebind *library* one
concern at a time.  This module drives the whole chain the way an operator
does -- freeze a plan, run every range, reduce, then resolve the result into a
:class:`ProcessV2AdmittedSource` -- and pins the properties that only the joined
chain can express.  Its V1 payloads are built by that module's production
fixture builders, so no V1 artifact is hand-written here either.

Invariants under test
---------------------

1. A record that survives the rebind reaches the resolved source carrying its
   **V1 identity and its V2 admission overlay side by side**, never merged and
   never relabelled: the V1 process-identity hash and the live Process-V2
   identity hash are both present and are different values.
2. A teacher outside the frozen Process-V2 fiber rejects **exactly one trace,
   whole**.  The run completes, the trace publishes no proof row, and every
   per-transition counter it would have contributed -- including those of its
   steps the mask *did* admit -- is dropped.
3. An integrity mismatch **publishes nothing**: no task directory, no proof, no
   manifest, no receipt, and the run stays unreducible and unresolvable.
4. Sharding is a scheduling decision.  The proof rows and the semantic census
   are identical row for row across two shardings; only the run *address*
   moves, because ``entries_per_task`` is deliberately part of
   ``run_identity_sha256``.
5. The task namespace is exact: a missing range, a duplicated result and an
   unexpected object all fail, while a killed worker's private staging
   directory does not block the retry.
6. The plan driver refuses an unexpected historical process identity and a
   payload root that mixes migrations.
7. No stage of this chain confers authority.

Reachability boundary: the charge gate is shared, so it is not V2-only
-----------------------------------------------------------------------

The brief for this module asked for a V1 delete teacher that Process V2 refuses
*on charge grounds*.  That set is empty by construction, and
``test_a_charge_violating_delete_can_never_reach_a_v1_payload`` proves it rather
than asserting it: the V1 semantic runtime carries
``editing_charge_policy_constraint`` (``rewrite/kernel.py:379``) as a hard
condition and Process-V2 gate 4 (``rewrite/process_v2_atom_delete.py:341``) is
the *same* ``charge_policy_preserved`` predicate over the same source and
successor.  A charge-violating ``atom_delete`` is therefore rejected by the V1
migration itself (``semantic_action_rejected``) and can never appear in a V1
payload for the rebind to re-decide.

The whole-trace rejection is exercised on a **charged aromatic lead**
(1,2-dimethylpyridinium) through the gate that genuinely is V2-only: an aromatic
connected-nonleaf deletion.  The charged chemistry is real, the exclusion is
measured, and the published outcome asserted below -- run completes, zero proof
rows, ``atom_delete_outside_process_v2_mask``, no partial credit -- is the same
outcome the requested case would have produced.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_process_v2_admitted_source import (
    ADMITTED_SOURCE_SCHEMA,
    ADMITTED_SOURCE_STATUS,
    ProcessV2AdmittedSourceIncomplete,
    resolve_process_v2_admitted_source,
)
from compose_v4.data.editing_v2_process_v2_schema import AUTHORITY_FIELDS
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    PLAN_FILENAME,
    PROOF_FILENAME,
    RECEIPT_FILENAME,
    TASK_DIRNAME,
    ProcessV2RebindError,
    ProcessV2RebindExclusionCode,
    ProcessV2RebindIncomplete,
    ProcessV2RebindIntegrityCode,
    ProcessV2RebindMismatch,
    completed_process_v2_rebind_task_ids,
    execute_process_v2_rebind_task,
    reduce_process_v2_rebind,
    write_process_v2_rebind_plan,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    RECEIPT_FILENAME as V1_RECEIPT_FILENAME,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    PROCESS_SEMANTICS,
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.process_v2_atom_delete import (
    ProcessV2AtomDeleteRejectionCode,
    process_v2_atom_delete_mask,
    resolve_process_v2_atom_delete,
)

# The V1 payload fixture builders and the plan driver are not importable as
# packages.  pytest already puts ``tests`` on ``sys.path`` under the default
# prepend import mode; both entries are added explicitly so this module also
# imports under ``importmode=importlib`` and outside pytest.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (str(_REPO_ROOT / "tests"), str(_REPO_ROOT / "scripts")):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import plan_process_v2_rebind as plan_driver  # noqa: E402
import test_editing_process_v2_rebind as v1_fixture  # noqa: E402

ROOT = v1_fixture.ROOT
SLOTS = v1_fixture.SLOTS
OUTSIDE_MASK = ProcessV2RebindExclusionCode.ATOM_DELETE_OUTSIDE_PROCESS_V2_MASK.value

# ---- Fixture chemistry --------------------------------------------------------

# A charged zwitterion whose only Process-V2-admissible deletion is its carbonyl
# oxygen (slot 6).  It proves a charged, representable lead survives the whole
# chain rather than being quietly dropped.
BETAINE_SMILES = "C[N+](C)(C)CC(=O)[O-]"
BETAINE_ADMITTED_SLOT = 6

# A charged aromatic lead.  Slot 0 is its C2 methyl, which Process V2 admits;
# the successor is 1-methylpyridinium, whose ring atoms are aromatic
# connected-nonleaf slots that Process V2 excludes by frozen decision.  So the
# two-step trace has one admitted teacher followed by one excluded teacher,
# which is what makes "no partial credit" falsifiable.
PYRIDINIUM_SMILES = "Cc1cccc[n+]1C"
PYRIDINIUM_ADMITTED_SLOT = 0
PYRIDINIUM_EXCLUDED_SLOT = 2

TRIM_BETAINE = "trim_betaine_carbonyl"
OPEN_PYRIDINIUM = "demethylate_then_open_pyridinium"
CHARGE_VIOLATING_DEMETHYLATION = "charge_violating_demethylation"

_EXTRA_FIXTURE_TRACES = {
    TRIM_BETAINE: (BETAINE_SMILES, (("atom_delete", AtomDelete(BETAINE_ADMITTED_SLOT)),)),
    OPEN_PYRIDINIUM: (
        PYRIDINIUM_SMILES,
        (
            ("atom_delete", AtomDelete(PYRIDINIUM_ADMITTED_SLOT)),
            ("atom_delete", AtomDelete(PYRIDINIUM_EXCLUDED_SLOT)),
        ),
    ),
    # Deleting a methyl off the quaternary nitrogen. The legacy executor admits
    # it; the V1 semantic runtime and Process V2 both refuse it on charge.
    CHARGE_VIOLATING_DEMETHYLATION: (BETAINE_SMILES, (("atom_delete", AtomDelete(0)),)),
}

# Four intact, Process-V2-compatible traces: 10 states and 6 transitions.
COMPATIBLE_TASKS = (
    (
        REQUIRED_DATA_LANES[0],
        REQUIRED_PARTITION_ROLES[0],
        ("cyclize_hexane", TRIM_BETAINE),
    ),
    (
        REQUIRED_DATA_LANES[1],
        REQUIRED_PARTITION_ROLES[1],
        ("trim_methylcyclohexane", "open_cyclohexane"),
    ),
)
# The same payload plus one trace whose second teacher is outside the fiber.
EXCLUDING_TASKS = (
    COMPATIBLE_TASKS[0],
    (
        REQUIRED_DATA_LANES[1],
        REQUIRED_PARTITION_ROLES[1],
        ("trim_methylcyclohexane", OPEN_PYRIDINIUM, "open_cyclohexane"),
    ),
)

COMPATIBLE_COUNTS = {
    "source_entries": 4,
    "admitted_entries": 4,
    "rejected_entries": 0,
    "admitted_states": 10,
    "admitted_transitions": 6,
}
EXCLUDING_COUNTS = {**COMPATIBLE_COUNTS, "source_entries": 5, "rejected_entries": 1}


@pytest.fixture(autouse=True)
def _register_extra_fixture_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add this module's charged leads to the shared fixture trace registry."""

    for name, definition in _EXTRA_FIXTURE_TRACES.items():
        monkeypatch.setitem(v1_fixture._FIXTURE_TRACES, name, definition)


# ---- Helpers ------------------------------------------------------------------


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)


def _net_formal_charge(state) -> int:
    """Sum charges over real elements only; a slot-stable state has null gaps."""

    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _run(payload, plan) -> Path:
    return v1_fixture._run_root(payload, plan)


def _task_root(payload, plan) -> Path:
    return _run(payload, plan) / TASK_DIRNAME


def _task_output(payload, plan, task: Mapping[str, object]) -> Path:
    return _task_root(payload, plan) / str(task["task_identity_sha256"])


def _manifests(payload, plan) -> list[dict]:
    return [
        json.loads((_task_output(payload, plan, task) / MANIFEST_FILENAME).read_text())
        for task in plan["tasks"]
    ]


def _proof_rows(payload, plan) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(_run(payload, plan).rglob(PROOF_FILENAME)):
        with gzip.open(shard, "rb") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _rejection_rows(payload, plan) -> list[dict]:
    return [row for manifest in _manifests(payload, plan) for row in manifest["rejected_traces"]]


def _completed_run(tmp_path: Path, *, tasks, entries_per_task: int = 1):
    """Build a payload, freeze a plan, execute every range and reduce."""

    payload = v1_fixture._build_v1_payload(tmp_path / "artifacts", tasks=tasks)
    plan = v1_fixture._plan_for(payload, entries_per_task=entries_per_task)
    v1_fixture._execute_all(payload, plan)
    completion = reduce_process_v2_rebind(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    return payload, plan, completion


def _resolve(payload, plan):
    return resolve_process_v2_admitted_source(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )


# ---- 1. An intact compatible trace reaches the admitted source ----------------


def test_an_intact_compatible_trace_resolves_with_both_identities_side_by_side(
    tmp_path: Path,
) -> None:
    """Plan, execute, reduce and resolve, then read the joined record.

    Without this the chain is only proven stage by stage.  A rebind whose
    completion is correct but whose resolver yields a different population, or
    which relabels a V1 row as if it had been produced under Process V2, would
    pass every single-stage test and still redefine the corpus for whatever
    builder consumes it.
    """

    payload, plan, completion = _completed_run(tmp_path, tasks=COMPATIBLE_TASKS)
    assert completion["counts"] == COMPATIBLE_COUNTS
    assert completion["rejected_traces_by_code"] == {}

    source = _resolve(payload, plan)
    assert dict(source.counts) == COMPATIBLE_COUNTS
    assert dict(source.rejected_traces_by_code) == {}
    assert source.identity()["schema"] == ADMITTED_SOURCE_SCHEMA
    assert source.identity()["status"] == ADMITTED_SOURCE_STATUS

    records = list(source.iter_records())
    assert len(records) == COMPATIBLE_COUNTS["admitted_entries"]
    # Iterating twice must not yield a different population.
    assert len(list(source.iter_records())) == len(records)

    pinned_v1 = plan["pinned_process_identity"]["process_identity_sha256"]
    live_v2 = plan["process_v2_identity"]["process_identity_sha256"]
    assert pinned_v1 != live_v2, "the fixture cannot prove separation if the hashes coincide"

    proofs_by_trace = {row["trace_id"]: row for row in _proof_rows(payload, plan)}
    planned_tasks = {str(task["task_identity_sha256"]) for task in plan["tasks"]}
    for record in records:
        v1_identity = record.v1_identity
        overlay = record.v2_admission
        # The V1 side keeps V1 semantics and the superseded V1 identity.
        assert v1_identity.process_semantics == PROCESS_SEMANTICS
        assert v1_identity.process_identity_sha256 == pinned_v1
        assert v1_identity.record_sha256 == record.v1_record["record_sha256"]
        assert v1_identity.trace_id == record.addressed.address.trace_id
        assert v1_identity.path_length == len(record.addressed.path.states) - 1
        # The V2 side is a separate overlay naming the live V2 identity.
        assert overlay.admitted is True
        assert overlay.process_v2_identity_sha256 == live_v2
        assert overlay.run_identity_sha256 == plan["run_identity_sha256"]
        assert overlay.task_identity_sha256 in planned_tasks
        assert overlay.proof_sha256 == proofs_by_trace[v1_identity.trace_id]["proof_sha256"]
        assert overlay.replayed_transitions == v1_identity.path_length
        assert len(overlay.process_v2_atom_delete_candidates) == v1_identity.path_length + 1
        # Both identities are readable from the one record, and they differ.
        assert v1_identity.process_identity_sha256 != overlay.process_v2_identity_sha256

    assert {record.v1_identity.trace_id for record in records} == set(proofs_by_trace)
    # A charged lead really is in the admitted population, not merely claimed.
    charged = [
        record
        for record in records
        if bool((record.addressed.path.states[0].formal_charges != 0).any())
    ]
    assert len(charged) == 1
    assert charged[0].v1_identity.path_length == 1
    # The zwitterion is net-neutral, so only a per-atom check finds it.
    assert _net_formal_charge(charged[0].addressed.path.states[0]) == 0


# ---- 2. A whole-trace support exclusion, and the boundary that forces it ------


def test_a_charge_violating_delete_can_never_reach_a_v1_payload(tmp_path: Path) -> None:
    """The charge gate is shared, so no V1 payload can carry a violating teacher.

    Without this the module's substitution of an aromatic exclusion for the
    requested charge exclusion would be an unchecked claim.  It is instead a
    measured property of two runtimes: the legacy executor admits the deletion,
    the V1 semantic runtime refuses it, Process V2 refuses it for the charge
    reason, and the production V1 migration therefore drops the whole trace
    before the rebind can ever see it.
    """

    state = _state(BETAINE_SMILES)
    methyl_on_the_quaternary_nitrogen = AtomDelete(0)
    # The legacy executor, which has no charge constraint, admits it.
    de_novo_rewrite_system().apply(state, "atom_delete", methyl_on_the_quaternary_nitrogen)
    # Process V2 refuses it, and names charge as the reason.
    resolution = resolve_process_v2_atom_delete(state, methyl_on_the_quaternary_nitrogen)
    assert resolution.admitted is False
    assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED
    # The V1 semantic runtime refuses it too, for the same shared predicate.
    with pytest.raises(ValueError):
        editing_v2_semantic_rewrite_system().apply(
            state,
            "atom_delete",
            methyl_on_the_quaternary_nitrogen,
        )

    # So the production V1 migration rejects the trace instead of recording it.
    tasks = (
        v1_fixture._V1_TASKS[0],
        (
            REQUIRED_DATA_LANES[1],
            REQUIRED_PARTITION_ROLES[1],
            (CHARGE_VIOLATING_DEMETHYLATION,),
        ),
    )
    payload = v1_fixture._build_v1_payload(tmp_path / "artifacts", tasks=tasks)
    receipt = json.loads(
        (
            payload.payload_root
            / TASK_DIRNAME
            / payload.v1_task_identities[1]
            / V1_RECEIPT_FILENAME
        ).read_text()
    )
    assert receipt["counts"] == {"source": 1, "admitted": 0, "rejected": 1}
    assert receipt["rejections_by_code"] == {"semantic_action_rejected": 1}
    assert v1_fixture._read_v1_records(v1_fixture._semantic_dir(payload, 1)) == []


def test_a_teacher_outside_the_process_v2_mask_rejects_its_whole_trace(
    tmp_path: Path,
) -> None:
    """The run completes, the trace publishes nothing, and no step earns credit.

    Two failures would go unnoticed without this.  A rebind that treated an
    expected support exclusion as an integrity mismatch would abort the whole
    run on a payload that is perfectly intact.  A rebind that admitted the
    supported *prefix* of an excluded trace would publish a corpus whose
    transitions came from traces it also reports as rejected -- and the
    difference is only visible because the excluded trace's first teacher is
    inside the mask, so a partial-credit implementation would count it.
    """

    # Premise, measured against the production authority: step 0 is admitted and
    # step 1 is not.  If this ever flips, the test below stops meaning anything.
    source_state = _state(PYRIDINIUM_SMILES)
    successor = de_novo_rewrite_system().apply(
        source_state,
        "atom_delete",
        AtomDelete(PYRIDINIUM_ADMITTED_SLOT),
    )
    assert _net_formal_charge(source_state) == 1, "the excluded trace must be a charged lead"
    assert bool(process_v2_atom_delete_mask(source_state)[PYRIDINIUM_ADMITTED_SLOT]) is True
    assert bool(process_v2_atom_delete_mask(successor)[PYRIDINIUM_EXCLUDED_SLOT]) is False
    assert (
        resolve_process_v2_atom_delete(successor, AtomDelete(PYRIDINIUM_EXCLUDED_SLOT)).admitted
        is False
    )

    payload, plan, completion = _completed_run(tmp_path, tasks=EXCLUDING_TASKS)
    assert plan["expected_entry_count"] == 5
    # The run COMPLETED: reduce published a completion rather than refusing.
    assert (_run(payload, plan) / COMPLETION_FILENAME).is_file()
    assert completion["counts"] == EXCLUDING_COUNTS
    assert (
        completion["counts"]["admitted_entries"] + completion["counts"]["rejected_entries"]
        == completion["counts"]["source_entries"]
    )
    assert completion["rejected_traces_by_code"] == {OUTSIDE_MASK: 1}
    assert completion["unsupported_teacher_steps_by_code"] == {OUTSIDE_MASK: 1}

    rejections = _rejection_rows(payload, plan)
    assert len(rejections) == 1
    rejection = rejections[0]
    assert rejection["exclusion_code"] == OUTSIDE_MASK
    assert rejection["path_length"] == 2
    # The FIRST unsupported teacher is step 1, not step 0: the mask admitted the
    # first deletion of this very trace.
    assert rejection["step_index"] == 1
    assert rejection["unsupported_teacher_steps"] == 1

    # Zero proof rows for the rejected trace.
    proof_rows = _proof_rows(payload, plan)
    assert len(proof_rows) == 4
    assert rejection["trace_id"] not in {row["trace_id"] for row in proof_rows}

    # No partial credit: the range that rejected published no admitted counter
    # at all, even though it evaluated the mask and found candidates.
    rejecting = [manifest for manifest in _manifests(payload, plan) if manifest["rejected_traces"]]
    assert len(rejecting) == 1
    manifest = rejecting[0]
    assert manifest["admitted_entries"] == 0
    assert manifest["admitted_states"] == 0
    assert manifest["admitted_transitions"] == 0
    assert manifest["family_histogram"] == {}
    assert set(manifest["teacher_census"].values()) == {0}
    assert manifest["process_v2_atom_delete_census"]["states_evaluated"] == 3
    assert manifest["process_v2_atom_delete_census"]["candidate_slots"] >= 1

    # And the run-level teacher census counts three atom_delete teachers, not
    # four: the excluded trace's in-mask step 0 teacher is dropped with it.
    assert completion["teacher_census"]["atom_delete_teachers"] == 3

    source = _resolve(payload, plan)
    assert dict(source.counts) == EXCLUDING_COUNTS
    assert dict(source.rejected_traces_by_code) == {OUTSIDE_MASK: 1}
    admitted = list(source.iter_records())
    assert len(admitted) == 4
    assert rejection["trace_id"] not in {record.v1_identity.trace_id for record in admitted}


# ---- 3. Integrity: publish nothing --------------------------------------------


def test_an_integrity_mismatch_publishes_no_object_at_all(tmp_path: Path) -> None:
    """One corrupted V1 record leaves its whole range unpublished.

    Without this a refusal could still leave a manifest, a receipt or a partial
    proof shard behind, and a later reduction -- or a resolver that reads task
    outputs directly -- would treat those bytes as proven evidence.  The
    refusal must be an absence, not a flagged presence.
    """

    payload = v1_fixture._build_v1_payload(tmp_path / "artifacts", tasks=COMPATIBLE_TASKS)
    corrupted: list[str] = []

    def forge_the_first_canonical_key(record: dict) -> None:
        if corrupted:
            return
        corrupted.append(str(record["trace_id"]))
        record["canonical_state_keys"][-1] = "CCCCCC|fixture-forged-key"
        record["steps"][-1]["successor_key"] = "CCCCCC|fixture-forged-key"

    v1_fixture._reseal_v1_task(
        payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0],
        forge_the_first_canonical_key,
    )
    assert len(corrupted) == 1

    plan = v1_fixture._plan_for(payload)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    refused: dict[str, ProcessV2RebindMismatch] = {}
    for task in plan["tasks"]:
        try:
            execute_process_v2_rebind_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=payload.artifact_root,
                repo_root=ROOT,
            )
        except ProcessV2RebindMismatch as error:
            refused[str(task["task_identity_sha256"])] = error
    assert len(refused) == 1, "exactly the range holding the corrupted record must refuse"
    (refused_task, error), = refused.items()
    assert ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES in {
        finding.code for finding in error.findings
    }
    report = error.as_report()
    assert report["published"] is False
    assert report["training_authorized"] is False

    task_root = _task_root(payload, plan)
    # Nothing at all under the refused identity: not the directory, not a
    # staging sibling, not a stray file.
    assert not (task_root / refused_task).exists()
    assert {path.name for path in task_root.iterdir()} == {
        str(task["task_identity_sha256"]) for task in plan["tasks"]
    } - {refused_task}
    published = [path for path in task_root.rglob("*") if path.is_file()]
    assert {path.name for path in published} == {
        PROOF_FILENAME,
        MANIFEST_FILENAME,
        RECEIPT_FILENAME,
    }
    # Three files each, for the three ranges that did publish, and nothing more.
    assert len(published) == 3 * (len(plan["tasks"]) - 1)
    for name in (PROOF_FILENAME, MANIFEST_FILENAME, RECEIPT_FILENAME):
        assert not (task_root / refused_task / name).exists()

    # The run is neither reducible nor resolvable.
    with pytest.raises(ProcessV2RebindIncomplete, match="missing 1"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    assert not (_run(payload, plan) / COMPLETION_FILENAME).exists()
    with pytest.raises(ProcessV2AdmittedSourceIncomplete):
        _resolve(payload, plan)


# ---- 4. Sharding moves the address, never the evidence ------------------------


def test_sharding_changes_the_run_address_but_not_the_admitted_evidence(
    tmp_path: Path,
) -> None:
    """Two shardings of one payload publish identical evidence at two addresses.

    Without this a rebind run on twenty containers would not be the artifact
    that was reviewed on one.  The test has to assert BOTH halves: that the
    proof rows and census are byte-identical, and that the run address really
    does move, because ``entries_per_task`` is deliberately folded into
    ``run_identity_sha256`` and pretending otherwise would hide a genuine
    provenance decision.
    """

    per_entry = _completed_run(tmp_path / "per_entry", tasks=EXCLUDING_TASKS, entries_per_task=1)
    per_source = _completed_run(
        tmp_path / "per_source",
        tasks=EXCLUDING_TASKS,
        entries_per_task=1000,
    )
    wide_payload, wide_plan, wide_completion = per_entry
    coarse_payload, coarse_plan, coarse_completion = per_source

    # The premise: the two plans really do partition the work differently.
    assert len(wide_plan["tasks"]) == 5
    assert len(coarse_plan["tasks"]) == 2

    # The evidence is identical, row for row.
    wide_rows = sorted(json.dumps(row, sort_keys=True) for row in _proof_rows(*per_entry[:2]))
    coarse_rows = sorted(json.dumps(row, sort_keys=True) for row in _proof_rows(*per_source[:2]))
    assert wide_rows, "the wide plan published no proof rows to compare"
    assert wide_rows == coarse_rows
    assert sorted(
        json.dumps(row, sort_keys=True) for row in _rejection_rows(*per_entry[:2])
    ) == sorted(json.dumps(row, sort_keys=True) for row in _rejection_rows(*per_source[:2]))

    # The semantic completion is identical.
    assert sorted(wide_completion) == sorted(coarse_completion)
    for field in (
        "counts",
        "rejected_traces_by_code",
        "unsupported_teacher_steps_by_code",
        "teacher_census",
        "process_v2_atom_delete_census",
        "family_histogram",
        "v1_payload_binding_sha256",
        "process_v2_identity_sha256",
        "pinned_process_identity_sha256",
    ):
        assert wide_completion[field] == coarse_completion[field], field

    # Exactly which completion fields the schedule is allowed to move, as an
    # absolute list, measured on a payload that CONTAINS a support exclusion.
    # `rejected_trace_inventory_sha256` is in this set only because the
    # inventory rows it hashes embed the rebind `task_identity_sha256`
    # (`editing_process_v2_rebind.py:2515`), which is a range address and
    # therefore schedule-dependent -- while the rejection rows themselves,
    # asserted identical above, are not.  A payload with zero rejections hashes
    # an empty inventory and cannot observe this, so pinning the set here is
    # what keeps the choice deliberate.
    assert {
        key for key in wide_completion if wide_completion[key] != coarse_completion[key]
    } == {
        "completion_sha256",
        "plan_file_sha256",
        "plan_sha256",
        "rejected_trace_inventory_sha256",
        "result_inventory",
        "result_inventory_sha256",
        "run_identity_sha256",
        "task_count",
        "task_inventory_sha256",
    }
    assert wide_plan["expected_entry_count"] == coarse_plan["expected_entry_count"] == 5

    # The resolved sources agree on their census and on every record identity.
    wide_source = _resolve(*per_entry[:2])
    coarse_source = _resolve(*per_source[:2])
    assert dict(wide_source.counts) == dict(coarse_source.counts)
    assert dict(wide_source.rejected_traces_by_code) == dict(coarse_source.rejected_traces_by_code)
    assert sorted(
        json.dumps(record.v1_identity.as_payload(), sort_keys=True)
        for record in wide_source.iter_records()
    ) == sorted(
        json.dumps(record.v1_identity.as_payload(), sort_keys=True)
        for record in coarse_source.iter_records()
    )

    # The address DOES move, and `entries_per_task` is what moves it: replanning
    # the coarse payload at the wide sharding reproduces the wide run identity
    # exactly, so nothing else about the two runs differs.
    assert wide_plan["run_artifact_root"] != coarse_plan["run_artifact_root"]
    assert wide_plan["run_identity_sha256"] != coarse_plan["run_identity_sha256"]
    assert wide_plan["entries_per_task"] == 1
    assert coarse_plan["entries_per_task"] == 1000
    replanned = v1_fixture._plan_for(coarse_payload, entries_per_task=1)
    assert replanned["run_identity_sha256"] == wide_plan["run_identity_sha256"]
    assert replanned["run_artifact_root"] == wide_plan["run_artifact_root"]
    assert wide_payload.artifact_root != coarse_payload.artifact_root


# ---- 5. Task namespace: exact, and tolerant of a killed worker ----------------


def test_missing_duplicated_and_unexpected_task_objects_all_fail(tmp_path: Path) -> None:
    """The reducer counts exact ranges, not directories that look plausible.

    Each of these silently corrupts the census if unenforced: a missing range
    under-counts the corpus, a duplicated result double-counts one range while
    dropping another, and an unexpected object is either a stale run bleeding
    into this one or a half-written result. A killed worker's private staging
    directory is none of those and must not block its retry.
    """

    payload, plan, completion = _completed_run(tmp_path, tasks=COMPATIBLE_TASKS)
    task_root = _task_root(payload, plan)
    identities = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    assert completion["task_count"] == len(identities) == 4

    # (a) A killed worker's private staging directory is tolerated.
    staging = task_root / f".{identities[0]}.killed-worker.staging"
    staging.mkdir()
    (staging / PROOF_FILENAME).write_bytes(b"half-written")
    assert completed_process_v2_rebind_task_ids(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    ) == set(identities)
    assert (
        reduce_process_v2_rebind(
            plan,
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
        == completion
    )
    shutil.rmtree(staging)

    # (b) An unexpected object is a hard failure, not an ignored stray.
    unexpected = task_root / ("e" * 64)
    unexpected.mkdir()
    with pytest.raises(ProcessV2RebindError, match="unexpected objects"):
        completed_process_v2_rebind_task_ids(
            plan,
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
    with pytest.raises(ProcessV2RebindError, match="unexpected objects"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    unexpected.rmdir()

    # (c) A duplicated result -- one range's bytes published under another
    # range's identity -- is refused by name.
    victim = task_root / identities[1]
    shutil.rmtree(victim)
    shutil.copytree(task_root / identities[0], victim)
    with pytest.raises(ProcessV2RebindError, match=f"mismatches task {identities[1]}"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    shutil.rmtree(victim)

    # (d) A missing range is incomplete, and nothing downstream may resolve.
    with pytest.raises(ProcessV2RebindIncomplete, match="missing 1"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    with pytest.raises(ProcessV2AdmittedSourceIncomplete, match="missing 1"):
        _resolve(payload, plan)


# ---- 6. The plan driver refuses the wrong payload ------------------------------


def test_the_plan_driver_refuses_an_unexpected_or_mixed_v1_payload(tmp_path: Path) -> None:
    """The driver reads pinned identities from the payload and checks them.

    The pinned identities cannot be recomputed live -- the V1 process identity
    is superseded -- so they are read from the immutable receipts. Two things
    must therefore be proven rather than assumed: that the payload was built
    under the process the operator named, and that the payload root holds one
    migration rather than several. Without the first, a plan silently binds to
    whatever historical process the bytes happen to carry; without the second,
    it binds to whichever task receipt sorted first.
    """

    payload = v1_fixture._build_v1_payload(tmp_path / "artifacts", tasks=COMPATIBLE_TASKS)
    live_v1_sha = editing_v2_process_identity()["process_identity_sha256"]
    expected_process, expected_builder = v1_fixture._pinned_identities()

    process_identity, builder_identity = plan_driver.read_pinned_identities(
        payload.payload_root,
        expected_process_identity_sha256=live_v1_sha,
    )
    assert process_identity == expected_process
    assert builder_identity == expected_builder

    # The driver's own default is the recorded superseded identity, which this
    # fixture payload deliberately does not carry.
    assert live_v1_sha != SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    with pytest.raises(plan_driver.PlanDriverError, match="unexpected historical process"):
        plan_driver.read_pinned_identities(
            payload.payload_root,
            expected_process_identity_sha256=SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
        )
    # `build_plan` inherits the refusal, and reaches it before it ever asks the
    # repository for a source revision.
    with pytest.raises(plan_driver.PlanDriverError, match="unexpected historical process"):
        plan_driver.build_plan(
            artifact_root=payload.artifact_root,
            v1_payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
            expected_process_identity_sha256=SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
            output_artifact_prefix="/artifacts/rebind_fixture",
            entries_per_task=1,
        )

    # A payload root whose receipts disagree is refused as a mixed migration.
    receipt_path = (
        payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[1] / V1_RECEIPT_FILENAME
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["builder_identity"] = {**receipt["builder_identity"], "schema_version": 999}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(plan_driver.PlanDriverError, match="disagree on their pinned identities"):
        plan_driver.read_pinned_identities(
            payload.payload_root,
            expected_process_identity_sha256=live_v1_sha,
        )


# ---- 7. No stage confers authority --------------------------------------------


def test_no_stage_of_the_rebind_chain_emits_training_authority(tmp_path: Path) -> None:
    """Plan, task, completion, driver envelope and resolved source: all false.

    This chain reuses a completed migration and proves compatibility. It is not
    a Gate 0 run, not a T1 run and not a P50 run, and nothing it publishes may
    be read as authorization for one. A single stage that omitted the flag --
    or, worse, defaulted it true -- would let a consumer treat proven-compatible
    data as authorized data.
    """

    payload, plan, completion = _completed_run(tmp_path, tasks=EXCLUDING_TASKS)

    # The plan carries exactly one authority field, and it is false.
    assert plan["training_authorized"] is False
    assert {key for key in plan if key.endswith("_authorized")} == {"training_authorized"}
    assert "NO_TRAINING_AUTHORITY" in plan["status"]

    # Every published task object.
    for task in plan["tasks"]:
        output = _task_output(payload, plan, task)
        for filename in (MANIFEST_FILENAME, RECEIPT_FILENAME):
            document = json.loads((output / filename).read_text())
            assert document["training_authorized"] is False
            assert "NO_TRAINING_AUTHORITY" in document["status"]

    # The run completion.
    assert completion["training_authorized"] is False
    assert completion["gate_zero_run"] is False
    assert "NO_TRAINING_AUTHORITY" in completion["status"]

    # The driver's report, which names the P50 field `bounded_p50_authorized`.
    envelope = plan_driver.plan_envelope(plan, written=None)
    assert {
        envelope["training_authorized"],
        envelope["gate_zero_authorized"],
        envelope["t1_authorized"],
        envelope["bounded_p50_authorized"],
    } == {False}
    assert envelope["plan_artifact_path"] is None

    # The resolved source object and the descriptor a consumer must record.
    # Both now name the P50 field `bounded_p50_authorized`: admitted-source
    # schema 3 removed the adapter's retired spelling, so outside the frozen
    # process contract there is one spelling, and the object publishes the whole
    # frozen vocabulary rather than a four-field subset of it.
    source = _resolve(payload, plan)
    for name in AUTHORITY_FIELDS:
        assert getattr(source, name) is False, name
    assert not hasattr(source, "p50_authorized")
    identity = source.identity()
    assert {
        key: value for key, value in identity.items() if key.endswith("_authorized")
    } == dict.fromkeys(AUTHORITY_FIELDS, False)
    assert identity["status"] == ADMITTED_SOURCE_STATUS

    # The published plan file on the volume says the same thing as the object.
    published_plan = json.loads((_run(payload, plan) / PLAN_FILENAME).read_text())
    assert published_plan["training_authorized"] is False
    assert published_plan["status"] == plan["status"]
