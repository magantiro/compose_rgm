"""Render the external-baseline qualification registry to markdown.

The JSON registry is the single source of truth; this script produces the
human-readable matrix so the two cannot drift.  ``--check`` re-renders and
compares, which is what the test asserts.

    python scripts/render_comparator_registry.py            # write
    python scripts/render_comparator_registry.py --check     # verify in sync
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.baseline_qualification import (  # noqa: E402
    SUBCAPABILITIES,
    load_qualification_registry,
)

QUALIFICATION_DIR = ROOT / "docs" / "workstreams" / "baseline-qualification"
REGISTRY_PATH = QUALIFICATION_DIR / "comparator_registry_v3.json"
OUTPUT_PATH = QUALIFICATION_DIR / "COMPARATOR_MATRIX.md"

_VERDICT_GLYPH = {"YES": "yes", "NO": "no", "PARTIAL": "partial", "UNVERIFIED": "**UNVERIFIED**"}

_AXIS_HEADERS = [
    ("source_conditioning", "source conditioning"),
    ("objective_specific_retraining", "objective-specific retraining"),
    ("intermediate_states_exposed", "intermediate states exposed"),
    ("pathwise_constraints_native", "pathwise constraints native"),
    ("dynamic_goal_switching_native", "mid-trajectory goal switch native"),
    ("restart_at_supplied_state_under_new_objective", "restart at supplied state, new objective"),
]


def _escape(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _capability_table(methods: list[dict]) -> list[str]:
    header = ["method"] + [label for _, label in _AXIS_HEADERS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method in methods:
        cells = [f"**{method['id']}**"]
        for axis, _ in _AXIS_HEADERS:
            cell = method["capabilities"][axis]
            cells.append(_VERDICT_GLYPH[cell["verdict"]])
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _classification_table(methods: list[dict]) -> list[str]:
    header = ["method"] + list(SUBCAPABILITIES) + ["verdict"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method in methods:
        cells = [f"**{method['id']}**"]
        cells += [method["classification"][subcap] for subcap in SUBCAPABILITIES]
        cells.append(f"**{method['method_verdict']}**")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(registry: dict) -> str:
    methods = registry["methods"]
    lines: list[str] = []
    add = lines.append

    add("<!-- GENERATED FILE. Edit comparator_registry_v3.json and re-run")
    add("     scripts/render_comparator_registry.py. Do not hand-edit. -->")
    add("")
    add("# COMPOSE comparator registry v3 — external-baseline qualification")
    add("")
    add(f"**Registry id:** `{registry['registry_id']}`  ")
    add(f"**Artifact status:** `{registry['artifact_status']}`  ")
    add(f"**Held-out data opened:** {'yes' if registry['held_out_data_opened'] else 'no'}  ")
    add(f"**Smoke executed:** {'yes' if registry['smoke_plan']['executed'] else 'no'}")
    add("")
    add(registry["purpose"])
    add("")

    escalations = registry.get("compose_claims_a_baseline_does_natively") or []
    add("## Highest-value finding: COMPOSE claims a baseline does natively")
    add("")
    if not escalations:
        add(
            "**NONE.** No verified evidence was found that any qualified baseline "
            "natively supports same-prefix dynamic retargeting or pathwise "
            "constraints at every intermediate state. See the near-miss table "
            "below before treating this as settled."
        )
    else:
        add(
            "Read the third and fourth columns together. Neither column is the "
            "finding on its own: the baseline genuinely does the thing, **and** "
            "the COMPOSE claim is still distinguishable — but only if it is "
            "stated as the conjunction, never as the part."
        )
        add("")
        for index, row in enumerate(escalations, start=1):
            add(f"**{index}. {_escape(row['method'])} — `{_escape(row['subcapability'])}`**")
            add("")
            add(f"- *What it natively does.* {_escape(row['what'])}")
            add(f"- *What COMPOSE still has.* {_escape(row['what_compose_still_has'])}")
            add(f"- *Evidence.* {_escape(row['evidence'])}")
            add("")
    add("")

    near = registry.get("near_misses") or []
    if near:
        add("### Near misses — the cells a reviewer will push on")
        add("")
        add("| method | axis | why it is not a native match | evidence |")
        add("|---|---|---|---|")
        for row in near:
            add(
                f"| {_escape(row['method'])} | `{_escape(row['axis'])}` | "
                f"{_escape(row['why_not'])} | {_escape(row['evidence'])} |"
            )
        add("")

    add("## Capability matrix")
    add("")
    lines.extend(_capability_table(methods))
    add("")
    add(
        "`**UNVERIFIED**` means no primary source was found; it is an honest gap "
        "and must not be used to support either a COMPOSE novelty claim or a "
        "baseline exclusion."
    )
    add("")

    add("## Classification against COMPOSE Claim-4 sub-capabilities")
    add("")
    for key, text in registry["compose_subcapabilities"].items():
        add(f"- `{key}` — {text}")
    add("")
    lines.extend(_classification_table(methods))
    add("")

    add("## Per-method records")
    add("")
    for method in methods:
        add(f"### {method['id']} (`{method['citation_key']}`)")
        add("")
        add(f"**Native task.** {method['native_task']}")
        add("")
        prov = method["provenance"]
        add("| provenance | value |")
        add("|---|---|")
        for field in (
            "paper_url",
            "code_repository",
            "code_reference",
            "license",
            "checkpoints",
            "cpu_feasible",
        ):
            add(f"| {field} | {_escape(prov[field])} |")
        add("")
        add("| capability | verdict | evidence |")
        add("|---|---|---|")
        for axis, label in _AXIS_HEADERS:
            cell = method["capabilities"][axis]
            add(
                f"| {label} | {_VERDICT_GLYPH[cell['verdict']]} | "
                f"{_escape(cell['evidence'])} |"
            )
        add("")
        oracle = method["oracle_accounting"]
        add(f"**Oracle accounting.** {oracle['definition']} "
            f"Rejected proposals counted: {oracle['counts_rejected_proposals']}. "
            f"Training-phase oracle calls: {oracle['training_phase_oracle_calls']}. "
            f"Default total budget: {oracle['default_total_budget']}.")
        add("")
        add(f"**Edit / generation budget.** {method['edit_budget']}")
        add("")
        adapter = method["adapter"]
        add(f"**Adapter status.** `{adapter['status']}` — {adapter.get('note', '')}")
        if adapter.get("work_items"):
            add("")
            for item in adapter["work_items"]:
                add(f"- {item}")
        add("")
        contract = method["fairness_contract"]
        add(f"**Fairness contract.** Matched quantity: {contract.get('matched_quantity', 'N/A')}. "
            f"Not fair because: {contract.get('not_fair_because', 'N/A')}")
        add("")
        smoke = method.get("smoke_cost") or {}
        if smoke:
            add(f"**Costed smoke.** {smoke.get('plan', '')} "
                f"Estimated CPU cost: {smoke.get('estimate', 'UNKNOWN')}. "
                f"Confidence: {smoke.get('confidence', 'UNKNOWN')}.")
            add("")

    add("## Fairness rules that bind every method")
    add("")
    for rule in registry["global_fairness_rules"]:
        add(f"- {rule}")
    add("")
    add("## Smoke plan")
    add("")
    add(registry["smoke_plan"]["definition"])
    add("")
    add(f"- sources: {registry['smoke_plan']['sources']}")
    add(f"- executed: {registry['smoke_plan']['executed']}")
    add(f"- total estimated cost: {registry['smoke_plan']['total_estimate']}")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the rendering is current")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    registry = load_qualification_registry(args.registry)
    rendered = render(registry)
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            print(f"{args.output} is out of date; re-run without --check", file=sys.stderr)
            return 1
        print(f"{args.output} is current")
        return 0
    args.output.write_text(rendered)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
