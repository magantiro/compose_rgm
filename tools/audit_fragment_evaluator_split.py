#!/usr/bin/env python3
"""Split the fragment evaluator into the measurements it had been conflating.

The suite reported ONE number called "validity" and placed it in a table beside
GenMol's validity column.  They are not the same quantity.  GenMol's validity is
the fraction of generated SMILES that are chemically valid.  Ours was the
fraction of ATTEMPTS that produced a molecule which is chemically valid AND
connected AND contains the prompt fragment AND extends every declared
attachment site -- because the sampler withholds anything failing the task
constraint, emitting an unparseable placeholder in its place, which the official
function then counts as invalid.

That is a strictly stronger condition reported under a weaker condition's name.
This tool takes an emission capture (nothing is generated here) and reports, per
attempt, which of the conditions held and the exact reason the attempt was not
counted:

    produced                 the trajectory committed a molecule at all
    chemical_validity        RDKit parses AND sanitizes -- the GenMol definition
    connected                one fragment, which the official salt filter wants
    graph_validity           re-enters a production MolecularGraph and passes
                             is_valid_state, the invariant the executor enforces
    fragment_containment     every prompt fragment embeds, atom-level, disjoint
    attachment_task_success  every DECLARED attachment site is extended

A sample that is chemically valid, connected, graph-valid and
fragment-containing but fails only the last condition is a failed
motif-extension, not an invalid molecule, and must not be reported as one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from run_fragment_constrained_suite import (
    MANIFEST,
    audit_queries,
    contains_all_fragments,
)

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.chem.state import is_valid_state


def _graph_valid(smiles: str) -> tuple[bool, str | None]:
    """Does this molecule re-enter a production state and satisfy the invariant?"""
    try:
        from compose_v4.experiments.editing_v2_evaluation_semantics import (
            production_state_from_smiles,
        )

        state = production_state_from_smiles(smiles)
    except Exception as exc:  # noqa: BLE001 - any failure is the answer
        return False, f"state_reconstruction_failed: {type(exc).__name__}: {exc}"
    try:
        return bool(is_valid_state(state)), None
    except Exception as exc:  # noqa: BLE001
        return False, f"is_valid_state_raised: {type(exc).__name__}: {exc}"


def classify(smiles: str, prompt, queries) -> dict:
    """The six conditions for one produced molecule, plus the first failure."""
    record: dict = {"smiles": smiles}

    mol = Chem.MolFromSmiles(smiles)
    record["chemical_validity"] = mol is not None
    if mol is None:
        record["rejection_reason"] = "unparseable_smiles"
        return record
    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(smiles, sanitize=False))
        record["sanitizes"] = True
    except Exception as exc:  # noqa: BLE001
        record["sanitizes"] = False
        record["chemical_validity"] = False
        record["rejection_reason"] = f"sanitization_failed: {type(exc).__name__}"
        return record

    record["connected"] = len(Chem.GetMolFrags(mol)) == 1
    graph_ok, graph_note = _graph_valid(smiles)
    record["graph_validity"] = graph_ok
    if graph_note:
        record["graph_note"] = graph_note
    record["fragment_containment"] = contains_all_fragments(smiles, queries)
    result = check_fragment_constraint(prompt, smiles)
    record["attachment_task_success"] = bool(result.satisfied)
    record["constraint_reason"] = result.reason

    if not record["connected"]:
        record["rejection_reason"] = "disconnected"
    elif not record["graph_validity"]:
        record["rejection_reason"] = "graph_invalid"
    elif not record["fragment_containment"]:
        record["rejection_reason"] = "fragment_not_contained"
    elif not record["attachment_task_success"]:
        record["rejection_reason"] = f"task_only: {result.reason}"
    else:
        record["rejection_reason"] = None
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()

    payload = json.loads(args.emissions.read_text())
    task = FragmentTask(payload["task"])
    prompt = next(
        p
        for p in load_genmol_prompts(MANIFEST)
        if p.drug_name == payload["drug"] and p.task is task
    )
    queries = audit_queries(prompt)

    committed = payload["committed_endpoints"]
    emitted = payload["emitted"]
    attempts = payload["attempts"]
    emitted_set = Counter(s for s in emitted if s)

    records = [classify(s, prompt, queries) for s in committed]

    produced = len(committed)
    not_produced = attempts - produced
    counts = {
        "attempts": attempts,
        "produced_a_molecule": produced,
        "produced_nothing": not_produced,
        "chemical_validity": sum(r["chemical_validity"] for r in records),
        "connected": sum(r.get("connected", False) for r in records),
        "graph_validity": sum(r.get("graph_validity", False) for r in records),
        "fragment_containment": sum(r.get("fragment_containment", False) for r in records),
        "attachment_task_success": sum(r.get("attachment_task_success", False) for r in records),
    }
    reasons = Counter(r["rejection_reason"] for r in records if r["rejection_reason"])
    task_only = sum(
        1 for r in records if (r["rejection_reason"] or "").startswith("task_only")
    )

    rates = {
        # Denominator = ATTEMPTS, which is the benchmark denominator.
        "chemical_validity_pct_of_attempts": 100.0 * counts["chemical_validity"] / attempts,
        "graph_validity_pct_of_attempts": 100.0 * counts["graph_validity"] / attempts,
        "fragment_containment_pct_of_attempts": 100.0
        * counts["fragment_containment"]
        / attempts,
        "attachment_task_success_pct_of_attempts": 100.0
        * counts["attachment_task_success"]
        / attempts,
        # Denominator = molecules actually produced. This is the ratio that tests
        # the by-construction claim; it is NOT a benchmark validity.
        "chemical_validity_pct_of_produced": (
            100.0 * counts["chemical_validity"] / produced if produced else float("nan")
        ),
        "graph_validity_pct_of_produced": (
            100.0 * counts["graph_validity"] / produced if produced else float("nan")
        ),
    }

    result = {
        "schema": "compose_fragment_evaluator_split_v1",
        "row": f"{payload['task']}/{payload['drug']}/seed{payload['seed']}",
        "rng_seed": payload["rng_seed"],
        "generating_rdkit": payload.get("generating_rdkit"),
        "counts": counts,
        "rates": rates,
        "rejection_reasons": dict(reasons),
        "failed_on_task_semantics_only": task_only,
        "reported_strict_validity_pct": 100.0 * sum(1 for s in emitted if s) / attempts,
        "emitted_equals_a_committed_endpoint": all(
            s in Counter(committed) or s in emitted_set for s in emitted if s
        ),
        "examples_failing_task_only": [
            {k: r[k] for k in ("smiles", "constraint_reason")}
            for r in records
            if (r["rejection_reason"] or "").startswith("task_only")
        ][: args.examples],
        "examples_failing_containment": [
            {k: r[k] for k in ("smiles", "constraint_reason")}
            for r in records
            if r["rejection_reason"] == "fragment_not_contained"
        ][: args.examples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**result, "per_sample": records}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
