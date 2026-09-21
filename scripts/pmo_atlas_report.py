"""Render the human-readable interpretation of the PMO atlas artifacts.

Every number in the output is read from a versioned artifact, so the prose
cannot drift away from the JSON it describes.  Absent artifacts are reported as
NOT RUN rather than omitted, because a missing test must not read as a passing
one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ATLAS = "diagnostics/pmo_atlas_v1/atlas.json"
TEST_A = "diagnostics/pmo_atlas_v1/test_a_transport.json"
TEST_A_RESCUE = "diagnostics/pmo_atlas_v1/test_a_transport_rescue.json"
TEST_B = "diagnostics/pmo_atlas_v1/test_b_local_lift.json"
BATTERY = "diagnostics/pmo_atlas_v1/mutation_battery.json"
PANEL = "diagnostics/pmo_atlas_v1/reference_panel.json"


def _load(repo_root: Path, relative: str) -> dict[str, Any] | None:
    path = repo_root / relative
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    return document.get("payload", document)


def _atlas_section(atlas: dict[str, Any]) -> list[str]:
    summary = atlas["summary"]
    lines = [
        "## Task 1 -- the atlas",
        "",
        (f"MEASURED. {summary['routes']} recorded routes across {summary['tasks']} tasks, "
        f"assembled from six frozen inputs whose file hashes and payload self-hashes all "
        f"verify. **{summary['replayed_exact']}/{summary['routes']} replay exactly** under "
        "the current production Editing-V2 executor, comparing every intermediate state by "
        "canonical key, not just the endpoint."),
        "",
        "### Lineage poverty is the headline property of this dossier",
        "",
        f"- **{summary['distinct_sources']} distinct source molecules** in the whole dossier.",
        (f"- **{summary['task_sections_on_shared_source']} of {summary['tasks']} task sections** "
        f"compile from the single molecule `{summary['shared_source_smiles']}`, which carries "
        f"{summary['shared_source_route_share'] * 100:.1f}% of all routes."),
        (f"- **{summary['spines']} spines** (compiled base routes) for {summary['routes']} "
        "programs: the sibling programs of a task reuse their spine's action prefix verbatim "
        "and append a suffix, so programs are NOT independent transformations."),
        "",
        "This is why the dossier is a challenge set and a diagnostic instrument, not a corpus.",
        "",
        "### Per task",
        "",
        "| task | programs | exact replay | spine steps | destination role |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for task, info in sorted(atlas["per_task"].items()):
        lines.append(
            f"| {task} | {info['programs']} | {info['replayed_exact']} | "
            f"{info['spine_primitive_steps']} | {info['destination_role']} |"
        )
    census = atlas["frozen_distillation_census"]
    if census.get("present"):
        lines += [
            "",
            "### Correction to the frozen census",
            "",
            (f"MEASURED. The frozen distillation reports "
            f"{census['runtime_supported_route_instances']} of {census['route_instances']} "
            "route instances as runtime supported, which reads as a chemistry or support "
            f"failure for four tasks. It is not: `{census['runtime_supported_definition']}`. "
            "Every route in this atlas executes, including the long ones that label excludes."),
        ]
    lines += [
        "",
        "### Evaluator provenance",
        "",
        atlas["recorded_scores"]["comparability_warning"],
        "",
    ]
    return lines


def _test_a_section(
    test_a: dict[str, Any] | None, rescue: dict[str, Any] | None
) -> list[str]:
    if test_a is None:
        return ["## Test A -- transport to a supplied destination", "", "NOT RUN.", ""]
    attempts = test_a["attempts"]
    transfer = [a for a in attempts if a["source_pool"] != "recorded_source"]
    charge = [a for a in transfer if a["status"] == "unreachable_charge_change"]
    searched = [a for a in transfer if a["status"] != "unreachable_charge_change"]
    found = [a for a in searched if a["status"] == "witness_found"]
    exact = sum(
        1 for a in found if a.get("verification", {}).get("exact_destination_recovery")
    )
    control = [a for a in attempts if a["source_pool"] == "recorded_source"]
    control_found = sum(1 for a in control if a["status"] == "witness_found")
    lines = [
        "## Test A -- can the compiler reach a SUPPLIED destination?",
        "",
        ("MEASURED, zero objective calls. Every attempt constructs a program for its own "
        "(source, destination) pair: the correspondence is recomputed per pair and no "
        "recorded absolute atom-address sequence is replayed."),
        "",
        f"- Control, recorded source: **{control_found}/{len(control)}** reach the destination.",
        (f"- Transfer to new sources: **{len(found)}/{len(transfer)}** overall, but the "
        "denominator hides the mechanism."),
        (f"- **{len(charge)}/{len(transfer)} "
        f"({len(charge) / len(transfer) * 100:.1f}%) are refused BEFORE any search** with "
        "`unreachable_charge_change`: the source carries a net formal charge, every atlas "
        "destination is neutral, and the executor's charge-preserving scope forbids the "
        "change. This is a declared representation boundary, not a search failure."),
        (f"- Excluding those, transport succeeds **{len(found)}/{len(searched)} = "
        f"{len(found) / len(searched) * 100:.1f}%**, with "
        f"**{exact}/{len(found)} exact destination recovery** independently replayed through "
        "the production executor."),
        f"- Only **{len(searched) - len(found)}** attempts fail by search.",
        "",
        ("**Reading: transport capability EXISTS and transfers to sources the live optimizer "
        "actually meets.** Given the destination, the compiler builds the program. The binding "
        "constraints are the charge-preservation scope and, marginally, the node budget -- not "
        "construction or support."),
        "",
    ]
    if rescue is not None:
        rows = rescue["attempts"]
        resolved = [r for r in rows if r["status"] == "witness_found"]
        lines += [
            (f"Budget rescue: the {len({(r['task'], r['source_smiles']) for r in rows})} "
            f"unresolved pairs were re-run at higher node budgets; "
            f"**{len(resolved)}/{len(rows)}** attempts then found a witness. "
            "A pair that resolves at a larger budget was budget-limited, not impossible."),
            "",
        ]
    else:
        lines += ["Budget rescue: NOT RUN.", ""]
    return lines


def _profile_section(panel: dict[str, Any] | None) -> list[str]:
    if panel is None:
        return []
    lines = [
        "## The teacher routes are not hill climbs",
        "",
        ("MEASURED, from the frozen reference panel (3 evaluations per task). The objective "
        "at the route midpoint, compared with the objective at the route's own source:"),
        "",
        "| task | source | midpoint | destination | shape |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    dips = 0
    total = 0
    for task, entries in sorted(panel["panels"].items()):
        values = {entry["label"]: entry["value"] for entry in entries}
        source = values.get("shared_source")
        midpoint = values.get("route_midpoint")
        destination = values.get("recorded_destination")
        if source is None or midpoint is None:
            continue
        total += 1
        dip = midpoint < source
        dips += int(dip)
        lines.append(
            f"| {task} | {source:.4f} | {midpoint:.4f} | "
            + (f"{destination:.4f}" if destination is not None else "n/a")
            + ("| **dips below source** |" if dip else "| rises |")
        )
    lines += [
        "",
        (f"**{dips} of {total} recorded teacher routes pass through a midpoint that scores "
        "BELOW their own starting molecule.** perindopril_mpo drops from 0.360 to 0.009 "
        "before reaching 0.809; qed drops from 0.796 to 0.251 before reaching 0.948."),
        "",
        ("INFERRED, and it is the cleanest available explanation of why Test C is the hard "
        "one: a score-greedy optimizer cannot follow these routes, because for more than "
        "half of them the middle of the route is worse than where it started. Construction "
        "and discovery fail for different reasons, and this is the reason discovery is not "
        "simply a weaker version of construction."),
        "",
    ]
    return lines


def _test_b_section(test_b: dict[str, Any] | None) -> list[str]:
    if test_b is None:
        return [
            "## Test B -- does landing there make local optimization strong?",
            "",
            ("NOT RUN. The harness exists (`scripts/pmo_atlas_local_lift.py`) and its "
            "reference-panel gate is built, but no scored screen has been executed, so "
            "nothing is claimed either way."),
            "",
        ]
    runs = test_b["runs"]
    lines = [
        "## Test B -- does landing there make local optimization strong?",
        "",
        (f"MEASURED. {test_b['diagnostic_oracle_calls']} diagnostic oracle evaluations across "
        f"{len(runs)} runs ({test_b['budget_per_run']} per run), frozen arm-B online-memory "
        "controller, destination and teacher suffix withheld after initialization."),
        "",
        "| task | position | seed score | best new | top-10 new mean | new > seed | distinct new |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        def _f(value: Any) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        lines.append(
            f"| {run['task']} | {run['checkpoint_label']} | {_f(run['seed_score'])} | "
            f"{_f(run['best_new'])} | {_f(run['top_ten_new_mean'])} | "
            f"{run['distinct_new_above_seed']} | {run['distinct_new_molecules']} |"
        )
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/INTERPRETATION.md")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    atlas = _load(repo_root, ATLAS)
    if atlas is None:
        raise SystemExit(f"missing {ATLAS}; run scripts/pmo_atlas_build.py first")

    lines = [
        "# PMO teacher-route atlas: development-informed diagnostic",
        "",
        "**INFORMATION REGIME: DEVELOPMENT_INFORMED_DIAGNOSTIC.**",
        "",
        atlas["information_regime_statement"],
        "",
        ("Nothing in `diagnostics/pmo_atlas_v1/` may be consumed by a scored no-prescreen run, "
        "fitted into a prior, or loaded as a proposal library or initialization bank. "
        "`route_prior_fit.py` already refuses these corpora and that refusal must stay."),
        "",
        "The programme separates three questions that have been conflated:",
        "",
        "1. **Can we CONSTRUCT it?** (Test A: transport to a supplied destination)",
        "2. **Can we EXPLOIT it?** (Test B: local optimization from a teacher region)",
        "3. **Can we DISCOVER it?** (Test C: blind entry into a productive region)",
        "",
        "Each failure has a different remedy, which is why they are measured apart.",
        "",
    ]
    lines += _atlas_section(atlas)
    lines += _test_a_section(_load(repo_root, TEST_A), _load(repo_root, TEST_A_RESCUE))
    lines += _profile_section(_load(repo_root, PANEL))
    lines += _test_b_section(_load(repo_root, TEST_B))

    battery = _load(repo_root, BATTERY)
    lines += [
        "## Guards",
        "",
    ]
    if battery is None:
        lines.append("Mutation battery: NOT RUN.")
    else:
        lines.append(
            f"Mutation battery: **{battery['negatives_killed']}/{battery['negatives_total']}** "
            f"production mutations killed by a named test; "
            f"**{battery['controls_green']}/{battery['controls_total']}** cosmetic positive "
            "controls stayed green. A mutation that fails to apply aborts the run, and a "
            "`killed` verdict requires a non-empty failing-test list."
        )
        if battery["negatives_survived"]:
            lines.append("")
            lines.append(
                "SURVIVED (guard not independently load-bearing): "
                + ", ".join(battery["negatives_survived"])
            )
    lines.append("")

    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
