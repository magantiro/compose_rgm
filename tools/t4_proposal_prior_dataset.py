"""Build the split-clean T4 proposal-prior corpus under its frozen contract.

Zero oracle calls, zero model fitting, zero runtime controller inputs. The group
split and admission rules are read from the sealed contract; this tool only
executes them and publishes the leakage audit, coverage census and GO/ABSTAIN
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_proposal_prior_dataset import (
    build,
    load_contract,
    read_scored_pack,
    verify_contract_inputs,
    write_records,
)

ROOT = Path(__file__).resolve().parents[1]


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _report(audit: dict) -> str:
    gate = audit["gate"]
    lines = [
        "# T4 proposal-prior corpus, split-clean preparation",
        "",
        f"Decision: **{gate['decision']}**. Contract `{audit['contract_sha256'][:16]}`.",
        "Zero oracle calls, zero new labels, no model fit.",
        "",
        "## Admission",
        "",
        f"- input rows: {audit['input_rows']}",
        f"- admitted rows: {audit['admitted_rows']}",
        f"- labelled constructions after collapsing repeat receipts: {audit['records']}",
        f"- excluded rows: {audit['excluded_rows']} {audit['exclusion_counts']}",
        "",
        "The recovered pack was already filtered by the audit that produced it, so no",
        "admission rule rejected anything here. The rules are therefore exercised only by",
        "their tests, not by this data; that is a property of this input, not evidence that",
        "the rules are unnecessary.",
        "",
        "## Group disjointness",
        "",
        "| Identity | Distinct values | Values crossing a cell |",
        "| --- | ---: | ---: |",
    ]
    for field, evidence in sorted(audit["disjointness_evidence"].items()):
        lines.append(
            f"| {field} | {evidence['distinct_values']} | {evidence['values_crossing_groups']} |"
        )
    lines += [
        "",
        "## Per-fold census",
        "",
        "| Fold | Train records | Train cells | Held records | Held sources | Held lineages |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold, sides in sorted(audit["census"].items()):
        train, held = sides["training"], sides["heldout"]
        lines.append(
            f"| {fold} | {train['records']} | {len(train['cells'])} | {held['records']} "
            f"| {held['source_contexts']} | {held['lineage_components']} |"
        )
    lines += [
        "",
        "## Construction-family support",
        "",
        "Tier is the minimum training-side count across all five folds.",
        "",
        "| Family | Minimum training records | Tier |",
        "| --- | ---: | --- |",
    ]
    for family, tier in sorted(
        gate["family_tiers"].items(),
        key=lambda item: -item[1]["minimum_training_records_across_folds"],
    ):
        lines.append(
            f"| {family} | {tier['minimum_training_records_across_folds']} | {tier['tier']} |"
        )
    power = audit["heldout_power"]
    weights = audit["weight_summary"]
    sizes = audit["lineage_component_sizes"]
    lines += [
        "",
        "## Statistical power of a held-out fold",
        "",
        "Adaptive search descends from one bootstrap root, so a cell's scored",
        "constructions collapse into very few genealogies. Record counts overstate power.",
        "",
        "| Fold | Held records | Held lineages | Held sources | Weighted effective n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for fold, row in sorted(power.items()):
        lines.append(
            f"| {fold} | {row['records']} | {row['lineage_components']} "
            f"| {row['source_contexts']} | {row['weighted_effective_sample_size']:.1f} |"
        )
    lines += [
        "",
        f"All {audit['records']} records fall in {sizes['components']} lineage components",
        f"(median {sizes['median']}, maximum {sizes['maximum']}, {sizes['singletons']} singletons).",
        "",
        "Under the frozen target/cell/lineage/record hierarchy the heaviest single record",
        (
            f"carries {weights['maximum_record_weight']:.4f} of corpus mass "
            f"({weights['maximum_over_uniform']:.0f}x uniform), the heaviest 100 records carry"
        ),
        f"{weights['mass_in_heaviest_100_records']:.3f}, and the whole corpus has effective",
        f"n = {weights['effective_sample_size']:.0f} against {weights['record_count']} records.",
        "Equalizing across lineage components over-corrects here, because a cell holds one",
        "very large genealogy beside a few singletons. This is a measured property of the",
        "corpus, reported as a finding; the hierarchy was frozen before it was computed and",
        "was not changed afterwards. A capped or size-damped variant is a decision for the",
        "separate fitting contract, taken with this number in hand.",
        "",
        "## Gate",
        "",
    ]
    if gate["failures"]:
        lines.append("Failed criteria (preserved, not widened):")
        lines += [f"- {failure}" for failure in gate["failures"]]
    else:
        lines.append("Every frozen hard criterion and required family held.")
    lines += [
        "",
        "## Limits",
        "",
        "- These are adaptive historical search observations, not independent samples.",
        "- Held-target folds measure generic transfer of this corpus. They are not a",
        "  held-out benchmark result, and a checkpoint fit on all cells stays trained-on-T4.",
        "- Program payloads are bound by checkpoint hash and entry id, not copied. A later",
        "  fitting step must resolve them inside its own training fold.",
        "- Repeat dockings of one construction were collapsed; their range is preserved per",
        "  record because REPORT section 7 measured non-trivial docking repeat variation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=ROOT / "configs/t4_proposal_prior_dataset_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "diagnostics/t4_proposal_prior/dataset_v1"
    )
    args = parser.parse_args()

    contract = load_contract(args.contract)
    inputs = verify_contract_inputs(contract, ROOT)
    manifest = unseal(ROOT / contract["inputs"]["checkpoint_manifest"]["path"])
    rows = read_scored_pack(ROOT / contract["inputs"]["scored_pack"]["path"])

    audit, records = build(contract, rows, manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "records.jsonl.gz"
    body_sha256 = write_records(records_path, records)
    audit = {
        **audit,
        "contract_path": str(Path(contract["outputs"]["records"]).parent),
        "verified_input_sha256": inputs,
        "checkpoint_manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        "records_artifact": {
            "path": str(records_path.relative_to(ROOT)),
            "decompressed_sha256": body_sha256,
            "gzip_sha256": sha256_file(records_path),
        },
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "completed_at_utc": _stamp(),
    }
    seal(args.output / "audit.json", audit)
    (args.output / "REPORT.md").write_text(_report(audit))
    summary = {
        "decision": audit["gate"]["decision"],
        "records": audit["records"],
        "excluded": audit["exclusion_counts"],
        "failures": audit["gate"]["failures"],
        "supported_families": audit["gate"]["supported_families"],
        "sparse_families": audit["gate"]["sparse_families"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
