"""The Active8 release sentinel: the only re-enumeration anywhere in the pipeline.

WHY IT EXISTS
-------------
There is a measured residual that no amount of internal consistency can reach.
A per-action count that moves WITHIN its arithmetic bounds -- a
``raw_mark_count`` of 41 published as 40, say -- and is then propagated
consistently through the row totals, the task summary and the run census is
undetectable by construction: every aggregate is derived from the moved number,
so every aggregate agrees with it, and every invariant it has to satisfy still
holds.  Nothing downstream re-derives the evidence, so nothing downstream can
notice either.

The sentinel converts that residual from undetectable to SAMPLED.  It re-enters
the chunk, re-enumerates the candidate fiber through the production evaluator,
and compares the COMPLETE candidate-evidence payload and the cell assignment
against what was published.  It cannot prove the whole corpus; it makes the
undetectable failure a detectable one at a bounded, prospectively frozen rate.

WHEN IT RUNS
------------
After every task has published and BEFORE the authoritative completion.  Any
mismatch blocks completion: the reduction publishes nothing.

SELECTION
---------
Ranking is by SHA-256 over the frozen ``SENTINEL_SALT`` and the pair, so it was
fixed before any result existed and cannot be chosen after seeing one.  At or
below ``SENTINEL_EXHAUSTIVE_THRESHOLD`` unique accepted
``(source_state_sha256, action_sha256)`` pairs the sentinel checks all of them;
above it, the deduplicated union of ``SENTINEL_PER_CELL_EXAMPLES`` per REQUIRED
capability cell and ``SENTINEL_GLOBAL_EXAMPLES`` globally.  The per-cell half
exists because a global sample of a skewed corpus can miss a rare required cell
entirely.

THE DICTIONARY ORACLE
---------------------
``_dictionary_successor_oracle`` regroups the production marked law with an
ordinary Python dictionary and compares the result against the production
canonical quotient.  It reuses the production executor and the production
canonical key -- those DEFINE the process -- and reimplements only the grouping,
which is the step being checked.  It is a BOUNDED TEST INSTRUMENT: it runs on
at most ``SENTINEL_ORACLE_EXAMPLES`` examples, it never decides admission, and
no number it produces other than ``oracle_mismatches`` is ever published.
``tests/test_editing_v2_process_v2_active8_pipeline.py`` drives it against the
repository's own ``reference_successor_kernel`` oracle, which production code is
forbidden to import, so the instrument itself is checked by the real oracle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.editing_process_v2_rebind import mounted_process_v2_artifact_path
from compose_v4.data.editing_v2_process_v2_active8_map import (
    _chunk_target,
    _classification_block,
    _evidence_payload,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    read_process_v2_chunk_target,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    SENTINEL_BLOCKS_COMPLETION,
    SENTINEL_EXHAUSTIVE_THRESHOLD,
    SENTINEL_GLOBAL_EXAMPLES,
    SENTINEL_ORACLE_EXAMPLES,
    SENTINEL_PER_CELL_EXAMPLES,
    SENTINEL_RESULT_FIELDS,
    SENTINEL_SALT,
    SENTINEL_SCHEMA,
    SENTINEL_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256, self_hashed
from compose_v4.data.editing_v2_semantic_active8_admission import (
    ProductionSemanticExactCandidateChecker,
)
from compose_v4.experiments.production_successor_kernel import canonical_successor_result
from compose_v4.rewrite.kernel import canonical_state_key

SENTINEL_PASSED = "SENTINEL_PASSED"
SENTINEL_FAILED = "SENTINEL_FAILED"
SELECTION_EXHAUSTIVE = "exhaustive_all_unique_accepted_pairs"
SELECTION_RANKED = "deduplicated_union_of_per_cell_and_global_ranked_examples"


class ProcessV2Active8SentinelError(RuntimeError):
    """The sentinel cannot be run over the published Active8 evidence."""


def sentinel_rank(source_state_sha256: str, action_sha256: str) -> str:
    """The prospectively frozen rank of one accepted pair."""

    digest = hashlib.sha256()
    digest.update(SENTINEL_SALT.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(source_state_sha256).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(action_sha256).encode("utf-8"))
    return digest.hexdigest()


def _pair(transition: Mapping[str, Any]) -> tuple[str, str]:
    evidence = transition["candidate_evidence"]
    return str(evidence["source_state_sha256"]), str(evidence["action_sha256"])


def select_sentinel_pairs(
    transitions: Sequence[Mapping[str, Any]], *, required_cell_ids: Sequence[str]
) -> dict[str, Any]:
    """Choose the pairs to re-enumerate, deterministically and salt-ranked."""

    occurrences: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    cells: dict[str, set[tuple[str, str]]] = {cell: set() for cell in required_cell_ids}
    for transition in transitions:
        pair = _pair(transition)
        occurrences.setdefault(pair, []).append(transition)
        cell = transition["capability_cell_id"]
        if cell in cells:
            cells[str(cell)].add(pair)
    # One re-enumeration per pair, compared against EVERY transition carrying it.
    # Deduplicating to a representative and checking only that one would leave a
    # moved count in any other occurrence of the same pair undetected, which is
    # precisely the residual this instrument exists to sample.
    for published in occurrences.values():
        published.sort(
            key=lambda transition: (
                str(transition["v1_task_identity_sha256"]),
                int(transition["entry_index"]),
                int(transition["step_index"]),
            )
        )
    representatives = {pair: rows[0] for pair, rows in occurrences.items()}
    unique = sorted(occurrences)
    if len(unique) <= SENTINEL_EXHAUSTIVE_THRESHOLD:
        selected = sorted(unique, key=lambda pair: sentinel_rank(*pair))
        return {
            "selection_mode": SELECTION_EXHAUSTIVE,
            "unique_accepted_pairs": len(unique),
            "per_cell_examples": 0,
            "global_examples": 0,
            "selected": selected,
            "representatives": representatives,
            "occurrences": occurrences,
        }
    per_cell: set[tuple[str, str]] = set()
    for cell in sorted(cells):
        ranked = sorted(cells[cell], key=lambda pair: sentinel_rank(*pair))
        per_cell.update(ranked[:SENTINEL_PER_CELL_EXAMPLES])
    globally = sorted(unique, key=lambda pair: sentinel_rank(*pair))[
        :SENTINEL_GLOBAL_EXAMPLES
    ]
    selected = sorted(per_cell | set(globally), key=lambda pair: sentinel_rank(*pair))
    return {
        "selection_mode": SELECTION_RANKED,
        "unique_accepted_pairs": len(unique),
        "per_cell_examples": len(per_cell),
        "global_examples": len(globally),
        "selected": selected,
        "representatives": representatives,
        "occurrences": occurrences,
    }


def _dictionary_successor_oracle(result: Any, state: Any, *, system: Any) -> dict[str, int]:
    """Regroup the production marked law with an ordinary dictionary.

    BOUNDED TEST INSTRUMENT.  It reuses the production executor and canonical
    key and reimplements only the aggregation, so it can disagree with the
    production quotient exactly where the production grouping is wrong.
    """

    source_key = canonical_state_key(state)
    aliases: dict[str, int] = {}
    for mark in result.marked_law.marks:
        successor = system.apply(state, mark.executor_rule_name, mark.action)
        key = canonical_state_key(successor)
        if key == source_key:
            continue
        aliases[key] = aliases.get(key, 0) + 1
    return aliases


def run_release_sentinel(
    plan: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
    *,
    checker: ProductionSemanticExactCandidateChecker,
    namespace: str,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Re-enumerate the selected accepted transitions and report every mismatch."""

    binding = plan["binding"]
    selection = select_sentinel_pairs(
        transitions, required_cell_ids=list(binding["required_cell_ids"])
    )
    selected: list[tuple[str, str]] = list(selection["selected"])
    representatives: Mapping[tuple[str, str], Mapping[str, Any]] = selection[
        "representatives"
    ]
    occurrences: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] = selection[
        "occurrences"
    ]
    oracle_pairs = set(selected[:SENTINEL_ORACLE_EXAMPLES])

    by_task: dict[str, list[tuple[str, str]]] = {}
    for pair in selected:
        transition = representatives[pair]
        by_task.setdefault(str(transition["task_identity_sha256"]), []).append(pair)

    tasks = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    evidence_mismatches: list[str] = []
    cell_mismatches: list[str] = []
    oracle_mismatches: list[str] = []
    oracle_checked = 0
    for task_identity in sorted(by_task):
        task = tasks.get(task_identity)
        if task is None:
            raise ProcessV2Active8SentinelError(
                f"a selected transition names an unplanned task: {task_identity}"
            )
        wanted: dict[int, list[tuple[str, str]]] = {}
        for pair in by_task[task_identity]:
            wanted.setdefault(int(representatives[pair]["entry_index"]), []).append(pair)
        target = _chunk_target(task)
        source_output = mounted_process_v2_artifact_path(
            target.source_artifact_path,
            artifact_root=Path(artifact_root),
            field="task.cache_source_artifact_path",
        )
        for read in read_process_v2_chunk_target(
            source_output,
            target=target,
            sentinel_replay_entries=0,
            recover_row_errors=False,
            repo_root=Path(repo_root),
        ):
            pairs = wanted.get(int(read.entry_index))
            if not pairs or read.addressed is None:
                continue
            for pair in pairs:
                anchor = representatives[pair]
                step_index = int(anchor["step_index"])
                addressed = read.addressed
                audit = checker.evaluate(addressed, step_index)
                observed = _evidence_payload(audit)
                source = addressed.path.state_at(step_index)
                successor = addressed.path.state_at(step_index + 1)
                block = _classification_block(
                    source,
                    successor,
                    addressed.trace.steps[step_index],
                    family=str(anchor["model_family"]),
                    supported=bool(audit.evidence.supported),
                    step_index=step_index,
                    path_length=int(addressed.address.path_length),
                    namespace=namespace,
                )
                # Every published transition carrying this pair is compared
                # against the one re-enumeration: the exact source state and the
                # exact action determine the fiber, so they must all agree.
                for published in occurrences[pair]:
                    where = (
                        f"{published['v1_task_identity_sha256']}"
                        f"[{published['entry_index']}]:{published['step_index']}"
                    )
                    if observed != dict(published["candidate_evidence"]):
                        evidence_mismatches.append(where)
                    if (
                        block["capability_cell_id"] != published["capability_cell_id"]
                        or block["family_context"] != published["family_context"]
                        or block["audit_axes"] != published["audit_axes"]
                    ):
                        cell_mismatches.append(where)
                if pair in oracle_pairs:
                    oracle_checked += 1
                    if _oracle_disagrees(checker, source):
                        oracle_mismatches.append(
                            f"{task_identity}[{read.entry_index}]:{step_index}"
                        )
    body = {
        "schema": SENTINEL_SCHEMA,
        "schema_version": SENTINEL_SCHEMA_VERSION,
        "status": (
            SENTINEL_PASSED
            if not (evidence_mismatches or cell_mismatches or oracle_mismatches)
            else SENTINEL_FAILED
        ),
        "salt": SENTINEL_SALT,
        "unique_accepted_pairs": int(selection["unique_accepted_pairs"]),
        "selection_mode": str(selection["selection_mode"]),
        "selected_pairs": len(selected),
        "per_cell_examples": int(selection["per_cell_examples"]),
        "global_examples": int(selection["global_examples"]),
        "oracle_examples": oracle_checked,
        "evidence_mismatches": sorted(evidence_mismatches),
        "cell_mismatches": sorted(cell_mismatches),
        "oracle_mismatches": sorted(oracle_mismatches),
    }
    sentinel = self_hashed(body, field="sentinel_sha256")
    if tuple(sentinel) != SENTINEL_RESULT_FIELDS:
        raise ProcessV2Active8SentinelError("the sentinel result field set disagrees")
    return sentinel


def _oracle_disagrees(
    checker: ProductionSemanticExactCandidateChecker, state: Any
) -> bool:
    """Compare the production canonical quotient against the dictionary oracle."""

    result = canonical_successor_result(
        checker.model, state, checker.time, system=checker.system
    )
    produced = {
        str(successor.key): int(successor.alias_count) for successor in result.batch.successors
    }
    return produced != _dictionary_successor_oracle(result, state, system=checker.system)


def require_sentinel_passed(sentinel: Mapping[str, Any]) -> None:
    """Any mismatch blocks completion; there is no partial release."""

    if not SENTINEL_BLOCKS_COMPLETION:  # pragma: no cover - frozen True in the seam
        raise ProcessV2Active8SentinelError("the sentinel must block completion")
    if set(sentinel) != set(SENTINEL_RESULT_FIELDS):
        raise ProcessV2Active8SentinelError("the sentinel result field set disagrees")
    if canonical_sha256(
        {key: value for key, value in sentinel.items() if key != "sentinel_sha256"}
    ) != sentinel["sentinel_sha256"]:
        raise ProcessV2Active8SentinelError("the sentinel self-hash disagrees")
    if sentinel["status"] != SENTINEL_PASSED:
        raise ProcessV2Active8SentinelError(
            "the release sentinel found a mismatch, so the Active8 run publishes no "
            f"completion: evidence={sentinel['evidence_mismatches']} "
            f"cells={sentinel['cell_mismatches']} oracle={sentinel['oracle_mismatches']}"
        )


__all__ = [
    "SELECTION_EXHAUSTIVE",
    "SELECTION_RANKED",
    "SENTINEL_FAILED",
    "SENTINEL_PASSED",
    "ProcessV2Active8SentinelError",
    "require_sentinel_passed",
    "run_release_sentinel",
    "select_sentinel_pairs",
    "sentinel_rank",
]
