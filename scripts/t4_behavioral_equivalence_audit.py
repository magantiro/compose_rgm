"""Branch-level equivalence between the historical T4 panel and the unified policy.

THE QUESTION
------------
Not "was every published row launched from the same app or contract hash" -- that
is known to be false and is a fact about how the mechanisms were discovered, not
about the molecular process.  The question is whether the unified state-adaptive
controller, given the SAME molecular state, selects and executes the SAME
proposal mechanism that produced each historical row.

WHAT MAKES THE ANSWER MEANINGFUL
--------------------------------
`t4_unified_routing.molecular_applicability` takes a `MolecularGraph` and
nothing else -- no target name, no cell id, no score, no provenance.  So the
branch choice CANNOT depend on target identity or historical outcome; that is a
structural property of the signature, not a claim this audit has to test
statistically.  What this audit tests is the remaining question: does the
chemistry-only rule land on the branch the historical arm actually used.

VACUITY GUARD
-------------
A row whose historical arm never reached a fallback branch cannot demonstrate
branch agreement, and counting it as agreement would inflate the verdict.  Those
rows are reported as PRIMARY_PATH and are excluded from the agreement fraction,
which is computed only over rows that actually fired a fallback.  If no row
fires, the audit reports UNEVALUATED rather than a vacuous pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.t4_unified_routing import (
    PROPOSAL_SLOTS,
    applicable_kernel_if_exhausted,
    molecular_applicability,
)

MANIFEST = ROOT / "diagnostics/t4_replication_manifest/manifest_v1.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"

#: The historical rescue operators, expressed as the unified kernel each one IS.
#: `region_repair` is the conditioned region law on the shallow lane;
#: `protonation` is the state-aware lane.  The mapping is a statement about what
#: the historical code executed, and each entry is justified in the audit output.
OPERATOR_TO_KERNEL = {
    "region_repair": "region",
    "protonation": "state_aware",
}


def _payload(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data.get("payload", data)


def _unified_policy() -> dict:
    """The single controller configuration, read from the sealed contracts.

    Every arm must agree on every controller field; a disagreement is reported
    rather than silently reduced to one arm's value.
    """

    arms = {}
    for path in sorted(ROOT.glob("configs/t4_unified_controller_*_d06_v1.json")):
        payload = _payload(path)
        arms[path.name] = {
            "region_law": payload["proposal"]["shallow"].get("region_law"),
            "shallow_draws": payload["proposal"]["shallow"]["draws"],
            "anchored_draws": payload["proposal"]["anchored_replacement"]["draws"],
            "state_aware_lane": "protonation_aware_retained_subgraph"
            in payload["proposal"],
            "state_routing": payload["state_routing"]["policy"],
            "support_expansion": payload["support_expansion"]["policy"],
            "support": payload["support"],
            "charged_calls_per_cell": payload["charged_calls_per_cell"],
            "batch": payload["batch"],
            "parents": payload["parents"],
            "value_penalty": payload["value_penalty"],
        }
    reference = next(iter(arms.values()))
    divergent = {
        key: {name: arm[key] for name, arm in arms.items()}
        for key in reference
        if len({json.dumps(arm[key], sort_keys=True) for arm in arms.values()}) > 1
    }
    return {"arms_checked": sorted(arms), "policy": reference, "divergent": divergent}


def main() -> int:
    manifest = _payload(MANIFEST)
    policy = _unified_policy()

    seed_index = _seed_index()
    applicability_cache: dict[str, dict] = {}
    rows = []
    for row in manifest["rows"]:
        entry = _resolve(row, seed_index)
        smiles = entry["smiles"]
        if smiles not in applicability_cache:
            applicability = molecular_applicability(
                pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
            )
            applicability_cache[smiles] = {
                "record": applicability.as_record(),
                "kernel_if_exhausted": applicable_kernel_if_exhausted(applicability),
            }
        resolved = applicability_cache[smiles]

        operator = row["support_operator"]
        fired = operator != "none"
        historical_branch = OPERATOR_TO_KERNEL.get(operator, "primary")
        unified_branch = resolved["kernel_if_exhausted"]
        agrees = (historical_branch == unified_branch) if fired else None

        rows.append(
            {
                "cell": row["cell"],
                "target": row["target"],
                "starting_molecule_index": entry["idx"],
                "start_smiles": smiles,
                "delta": row["delta"],
                # (1) historical execution branch
                "historical_execution_branch": historical_branch,
                "historical_app_module": row["app_module"],
                "historical_support_operator": operator,
                # (2) the state/support condition that caused the branch
                "state_support_condition": {
                    "fallback_fired_historically": fired,
                    "trigger": "terminal candidate_exhaustion" if fired else None,
                    "molecular_state": resolved["record"],
                },
                # (3) branch the unified router selects from the same state
                "unified_router_branch_if_exhausted": unified_branch,
                # (4) proposal kernel / (5) region law / (6) state-aware machinery
                "unified_proposal_kernel": policy["policy"]["shallow_draws"],
                "unified_region_law": policy["policy"]["region_law"],
                "unified_state_aware_lane": policy["policy"]["state_aware_lane"],
                # (7) support / exhaustion trigger
                "unified_trigger": "terminal candidate_exhaustion only",
                # (8) search widths / ladder
                "unified_support_expansion": policy["policy"]["support_expansion"],
                "unified_state_routing": policy["policy"]["state_routing"],
                # (9) executor and feasibility semantics
                "feasibility": {
                    "delta": row["delta"],
                    "qed_min": 0.6,
                    "sa_max": 4.0,
                    "support": policy["policy"]["support"],
                },
                # (10) branch choice cannot read identity
                "branch_depends_on_state_not_identity": True,
                "branch_identity_basis": (
                    "molecular_applicability() accepts a MolecularGraph and no "
                    "name, id, score or provenance, so target identity and "
                    "historical outcome are not in scope of the decision"
                ),
                "historical_charged_calls": row["historical_charged_calls"],
                "historical_best": row["historical_best"],
                "replicate_seeds": row["replicate_seeds"],
                "BEHAVIORALLY_EQUIVALENT_TO_UNIFIED_POLICY": (
                    agrees if fired else "PRIMARY_PATH"
                ),
                "agrees_on_fired_branch": agrees,
            }
        )

    fired_rows = [r for r in rows if r["agrees_on_fired_branch"] is not None]
    agreeing = [r for r in fired_rows if r["agrees_on_fired_branch"]]
    primary_rows = [r for r in rows if r["agrees_on_fired_branch"] is None]

    if not fired_rows:
        verdict = "UNEVALUATED_NO_FALLBACK_ROW"
    elif len(agreeing) == len(fired_rows):
        verdict = "ALL_FIRED_ROWS_BEHAVIORALLY_EQUIVALENT"
    else:
        verdict = "MISMATCH"

    report = {
        "schema_version": "t4_behavioral_equivalence_audit_v1",
        "new_oracle_calls": 0,
        "docking_calls": 0,
        "kernel": "rdkit 2024.03.5 (T4 production)",
        "question": (
            "does the unified state-adaptive controller select and execute the "
            "same molecular process that produced each historical row, given the "
            "same molecular state?"
        ),
        "equivalence_criterion": (
            "branch-level: historical execution branch == branch the unified "
            "router selects from the same state. Wrapper module, contract hash, "
            "controller fingerprint and code revision are NOT criteria."
        ),
        "rows_audited": len(rows),
        "fired_rows": len(fired_rows),
        "fired_rows_agreeing": len(agreeing),
        "primary_path_rows": len(primary_rows),
        "agreement_on_fired_rows": f"{len(agreeing)}/{len(fired_rows)}",
        "scope_limit": {
            "distinct_source_molecules_among_fired_rows": None,  # filled below
            "distinct_branch_values_observed": None,  # filled below
            "warning": (
                "7 fired ROWS rest on fewer distinct MOLECULES, because a source "
                "that exhausted at both thresholds contributes two rows sharing "
                "one molecular state. The operative rule is close to binary -- "
                "net formal charge != 0 selects state_aware, else region -- so "
                "this is a small test that could not have failed in many ways. "
                "It establishes that the unified router lands on the branch each "
                "historical arm used; it does NOT establish that formal charge "
                "universally determines the correct molecular-search operator."
            ),
        },
        "vacuity_guard": (
            "rows whose historical arm never reached a fallback branch are "
            "reported PRIMARY_PATH and EXCLUDED from the agreement fraction"
        ),
        "unified_policy": policy,
        "verdict": verdict,
        "rows": rows,
    }
    report["scope_limit"]["distinct_source_molecules_among_fired_rows"] = len(
        {r["start_smiles"] for r in fired_rows}
    )
    report["scope_limit"]["distinct_branch_values_observed"] = sorted(
        {r["unified_router_branch_if_exhausted"] for r in fired_rows}
    )
    destination = ROOT / "diagnostics/t4_behavioral_equivalence/audit_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"verdict: {verdict}")
    print(f"rows audited: {len(rows)}  fired: {len(fired_rows)}  primary: {len(primary_rows)}")
    print(f"agreement on fired rows: {len(agreeing)}/{len(fired_rows)}")
    print(f"unified policy divergent fields: {policy['divergent'] or 'NONE'}")
    for row in fired_rows:
        mark = "OK " if row["agrees_on_fired_branch"] else "BAD"
        print(
            f"  {mark} {row['cell']:9s} d{row['delta']}  "
            f"historical={row['historical_execution_branch']:11s} "
            f"unified={row['unified_router_branch_if_exhausted']:11s} "
            f"({row['historical_support_operator']})"
        )
    print(
        "scope limit: "
        f"{report['scope_limit']['distinct_source_molecules_among_fired_rows']} distinct "
        f"molecules, branches {report['scope_limit']['distinct_branch_values_observed']}"
    )
    print(f"written: {destination.relative_to(ROOT)}")
    return 0 if verdict.startswith("ALL_FIRED") else 1


def _seed_index() -> dict[tuple[str, int], dict]:
    """(target, per-target position) -> the seeds-file entry, DERIVED from the file.

    An earlier version of this audit hardcoded the target order and had braf and
    jak2 transposed, which silently paired every braf row with a jak2 molecule.
    It still "passed", because both targets are neutral and therefore route to the
    same kernel -- a mapping bug that the verdict could not detect. The order is
    now read from the file, and `_resolve` additionally asserts the resolved entry
    carries the row's own target, so a transposition fails loudly instead.
    """

    entries = json.loads(SEEDS.read_text())
    position: dict[str, int] = {}
    index: dict[tuple[str, int], dict] = {}
    for entry in entries:
        target = entry["target"]
        index[(target, position.get(target, 0))] = entry
        position[target] = position.get(target, 0) + 1
    return index


def _resolve(row: dict, index: dict[tuple[str, int], dict]) -> dict:
    per_target = int(row["cell"].rsplit("_", 1)[1])
    entry = index[(row["target"], per_target)]
    if entry["target"] != row["target"]:
        raise ValueError(
            f"seed resolution crossed targets for {row['cell']}: "
            f"row={row['target']} entry={entry['target']}"
        )
    return entry


if __name__ == "__main__":
    raise SystemExit(main())
