"""Run the zero-oracle actual-sampler support probe under its frozen contract.

Resumable: each probe context is sealed as it completes, and a rerun reuses a
completed context instead of resampling it. Zero oracle calls, no model fit, no
change to the sampler, compiler, executor or eligibility.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_proposal_access_probe import (
    SCHEMA_VERSION,
    declared_families,
    probe_context,
    similarity_reporter,
    support_decision,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "diagnostics/t4_strategy_reset/20260916/scored_rows.jsonl.gz"


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def historical_by_cell() -> dict[str, dict[str, float]]:
    """Best observed score per historical endpoint, grouped by benchmark cell."""
    best: dict[str, dict[str, float]] = defaultdict(dict)
    with gzip.open(PACK, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            current = best[row["cell"]].get(row["endpoint"])
            if current is None or row["score"] < current:
                best[row["cell"]][row["endpoint"]] = row["score"]
    return dict(best)


def root_eligibility(units, contract) -> dict:
    """What the frozen endpoint gate says about each cell's own starting molecule.

    Eligibility is checked only at completed endpoints, so the root's own properties
    are not a gate. They are recorded because they bound what a bounded edit can do:
    a proposal must stay within the similarity threshold of this molecule while
    clearing the QED and SA thresholds, and those two demands pull against each other
    when the root itself is far below the QED gate.
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    result = {}
    for context in contract["probe_contexts"]:
        unit = units[context["cell"]]
        molecule = Chem.MolFromSmiles(unit["original_seed"])
        evaluate = strict_endpoint_scorer(unit["original_seed"], delta=contract["delta"])
        properties = evaluate({"smiles": unit["original_seed"]})
        result[context["cell"]] = {
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "root_qed": float(QED.qed(molecule)),
            "root_passes_its_own_endpoint_gate": bool(properties.get("oracle_eligible")),
            "root_exclusion_reasons": properties.get("endpoint_exclusion_reasons"),
        }
    return result


def _report(payload: dict) -> str:
    gate = payload["gate"]
    lines = [
        "# T4 actual-sampler proposal-access probe",
        "",
        f"Decision: **{gate['decision']}**. Contract `{payload['contract_sha256'][:16]}`.",
        "Zero oracle calls, no model fit, no sampler or compiler change.",
        "",
        "## Per-context yield",
        "",
        (
            "| Cell | Bootstrap attempts | Distinct proposals | Unique endpoints | Eligible "
            "| New proposal each round | Archive attempts | Archive pool |"
        ),
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in payload["contexts"]:
        boot, arc = row["bootstrap"], row["archive"]
        lines.append(
            f"| {row['cell']} | {boot['attempts']} | {boot['distinct_proposals']} "
            f"| {boot['unique_endpoints']} | {boot['eligible']} "
            f"| {'yes' if boot['explores_after_first_round'] else 'NO'} "
            f"| {arc['attempts']} | {arc['pool']} |"
        )
    lines += [
        "",
        "A distinct proposal is a distinct (lane, module families and parameters, endpoint,",
        "status) draw. Two different proposals can legitimately reach the same molecule, so",
        "unique endpoints are reported but the repetition repair is judged on proposals.",
        "",
        "Only the bootstrap lane is teacher-free. The archive lane injects one historical",
        "champion for diagnostics, so its numbers are conditional support around a known",
        "good endpoint, never autonomous recovery.",
        "",
        "## Declared family support, pooled over the five contexts",
        "",
        "| Family | Realized | Attempted |",
        "| --- | ---: | ---: |",
    ]
    for family in declared_families():
        lines.append(
            f"| {family} | {gate['pooled_realized'].get(family, 0)} "
            f"| {gate['pooled_attempted'].get(family, 0)} |"
        )
    lines += ["", "## Gate", ""]
    if gate["failures"]:
        lines.append("Failed criteria (preserved, not widened):")
        lines += [f"- {failure}" for failure in gate["failures"]]
    else:
        lines.append("Exact realization, continued exploration and family support all held.")
    if gate["never_realized_anywhere"]:
        lines += [
            "",
            "Declared families realized nowhere in this probe: "
            + ", ".join(gate["never_realized_anywhere"])
            + ".",
        ]
    lines += ["", "## Nearest historically scored molecule", ""]
    lines += [
        (
            "| Cell | Lane | Generated | Median nearest | Exact hits | Best score among "
            "exact hits | Best score reachable in cell |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["contexts"]:
        for lane, near in sorted(row.get("historical_proximity", {}).items()):
            if "median_nearest_similarity" not in near:
                continue
            hit = near["best_historical_score_among_exact_hits"]
            lines.append(
                f"| {row['cell']} | {lane} | {near['generated_scored']} "
                f"| {near['median_nearest_similarity']:.3f} "
                f"| {near['exact_historical_endpoint_hits']} "
                f"| {'none' if hit is None else f'{hit:.1f}'} "
                f"| {near['best_historical_score_available_in_this_cell']:.1f} |"
            )
    lines += [
        "",
        "Similarity is to historically scored endpoints of the same cell. Their docking",
        "scores stay attached to those historical molecules; no generated molecule is given",
        "a value here, and section 7 of the strategy report measured that structural",
        "proximity is not a dependable utility label.",
        "",
    ]
    roots = payload.get("root_eligibility") or {}
    if roots:
        lines += [
            "## The starting molecule each cell must stay similar to",
            "",
            (
                "| Cell | Heavy atoms | Root QED | Root passes the endpoint gate "
                "| Bootstrap eligible |"
            ),
            "| --- | ---: | ---: | --- | ---: |",
        ]
        eligible = {row["cell"]: row["bootstrap"]["eligible"] for row in payload["contexts"]}
        for cell, row in sorted(roots.items(), key=lambda item: -item[1]["root_qed"]):
            lines.append(
                f"| {cell} | {row['heavy_atoms']} | {row['root_qed']:.3f} "
                f"| {'yes' if row['root_passes_its_own_endpoint_gate'] else 'no'} "
                f"| {eligible.get(cell, 0)} |"
            )
        lines += [
            "",
            "Eligible yield is ordered by the root's own QED across all five cells. The gate",
            "requires QED above 0.6 at the completed endpoint while similarity to this same",
            "root stays above the threshold, so a root far below the gate constrains how much",
            "a bounded edit can move. This is a measured association over five cells, not a",
            "proof that no supported program clears the gate.",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- Failing to sample an exact historical endpoint inside this budget is not a",
        "  failure criterion and is never a proof of zero support.",
        "- Local RDKit is newer than the pinned benchmark image. This is a support probe,",
        "  not a substitute for a pinned-image result.",
        "- The wall-clock stopping budget is lifted so the attempt count binds; realized",
        "  attempts and elapsed seconds are recorded.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=ROOT / "configs/t4_proposal_access_probe_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "diagnostics/t4_proposal_prior/access_probe_v1"
    )
    args = parser.parse_args()

    contract = unseal(args.contract)
    if contract.get("schema_version") != SCHEMA_VERSION:
        parser.error(f"not a {SCHEMA_VERSION} contract")
    registry_path = ROOT / contract["source_registry"]["path"]
    verify_file(registry_path, contract["source_registry"]["sha256"])
    units = {unit["cell"]: unit for unit in unseal(registry_path)["units"]}
    historical = historical_by_cell()
    args.output.mkdir(parents=True, exist_ok=True)
    began, rows = perf_counter(), []
    for index, context in enumerate(contract["probe_contexts"]):
        completed = args.output / "contexts" / f"{context['run']}.json"
        if completed.exists():
            rows.append(unseal(completed))
            print({"cell": context["cell"], "phase": "reused_completed_context"}, flush=True)
            continue
        path = Path(context["result_path"])
        if not path.exists():
            raise SystemExit(
                f"probe input is missing: {path}\n"
                "The raw checkpoint mirror is not tracked in git. Recover it with "
                "diagnostics/t4_strategy_reset/20260916/fetch_history.py into a new "
                "directory and verify against the recovery manifests."
            )
        verify_file(path, context["result_sha256"])
        result = unseal(path)
        row = probe_context(
            result,
            units[context["cell"]],
            contract,
            index=index,
            similarity=similarity_reporter(historical.get(context["cell"], {})),
        )
        row["run"] = context["run"]
        seal(completed, row)
        rows.append(row)
        print(
            {
                "cell": row["cell"],
                "bootstrap_attempts": row["bootstrap"]["attempts"],
                "bootstrap_eligible": row["bootstrap"]["eligible"],
                "archive_pool": row["archive"]["pool"],
                "elapsed_seconds": perf_counter() - began,
            },
            flush=True,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": sha256_file(args.contract),
        "contexts": rows,
        "gate": support_decision(rows, contract),
        "root_eligibility": root_eligibility(units, contract),
        "declared_families": list(declared_families()),
        "new_oracle_calls": 0,
        "new_labels": 0,
        "elapsed_seconds": perf_counter() - began,
        "code_revision": _revision(),
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "completed_at_utc": _stamp(),
    }
    seal(args.output / "result.json", payload)
    (args.output / "REPORT.md").write_text(_report(payload))
    print(
        json.dumps(
            {"decision": payload["gate"]["decision"], "failures": payload["gate"]["failures"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
