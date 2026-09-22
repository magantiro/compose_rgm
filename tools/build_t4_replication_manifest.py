"""Join the provenance and run-census artifacts into the 30-row replication manifest.

The manifest is the GROUND TRUTH for two separate things and must serve both:

* Plan A -- replicate the historical per-cell method twice more for like-for-like
  3-run statistics against the comparators, which both report a multi-run mean.
* Plan B -- the unified state-routed controller, whose blind routing decisions are
  checked against the historical kernel each cell actually used.

It is BUILT, never hand-written: every field comes from `provenance_v1.json` (traced
from the frozen table back to contracts, receipts and commits) or `run_census_v1.json`
(read from the Modal volumes), and the replicate seeds come from the declared
derivation in `compose_v4.experiments.t4_replication_manifest`.  The audit then
re-reads the CONTRACTS ON DISK rather than trusting the manifest, so a comparison
whose expectation is recomputed from the thing under test cannot pass vacuously.

Nothing here docks, launches or writes to a volume.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_replication_manifest import (
    HISTORICAL_REPLICATE,
    NEW_REPLICATES,
    SCHEMA_VERSION,
    ManifestAuditError,
    ReplicationRow,
    audit_manifest,
    derive_replicate_seed,
    inherited_cross_cell_seed_sharing,
    seed_stream_collisions,
)

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "diagnostics/t4_replication_manifest/provenance_v1.json"
CENSUS = ROOT / "diagnostics/t4_replication_manifest/run_census_v1.json"
OUT = ROOT / "diagnostics/t4_replication_manifest/manifest_v1.json"

#: The provenance artifact names operators in its own vocabulary; the manifest module
#: pins the roster. Mapped explicitly so a new operator name cannot be absorbed silently.
OPERATOR_ALIASES = {
    "none": "none",
    "region_repair": "region_repair",
    "protonation": "protonation",
    "support_expansion": "support_expansion",
}


def _payload(path: Path) -> dict:
    body = json.loads(path.read_text())
    return body.get("payload", body)


def _contract_facts(paths: set[str]) -> tuple[dict[str, float], dict[str, tuple[int, int]]]:
    """Delta and ceiling read from the CONTRACTS, never from a file name or a manifest."""

    deltas: dict[str, float] = {}
    ceilings: dict[str, tuple[int, int]] = {}
    for relative in sorted(paths):
        payload = _payload(ROOT / relative)
        deltas[relative] = float(payload["delta"])
        ceilings[relative] = (
            int(payload["charged_calls_per_cell"]),
            int(payload["total_charged_call_ceiling"]),
        )
    return deltas, ceilings


def build() -> tuple[list[ReplicationRow], dict]:
    provenance = _payload(PROVENANCE)
    census = _payload(CENSUS)
    by_cell = {
        (row["target"].lower(), int(row["seed_index"]), float(row["delta"])): row
        for row in census["per_cell"]
    }

    rows: list[ReplicationRow] = []
    for source in provenance["rows"]:
        target = source["target"].lower()
        seed = int(source["seed"])
        delta = float(source["delta"])
        census_row = by_cell.get((target, seed - 1, delta), {})
        controller_seed = int(source["controller_seed"])
        operator = OPERATOR_ALIASES[source["support_operator"]]
        notes = []
        if source.get("blank_cell_note"):
            notes.append(str(source["blank_cell_note"]))
        if int(source.get("contract_identity_count", 1)) > 1:
            notes.append(
                f"the campaign spanned {source['contract_identity_count']} contract "
                f"identities; it FINISHED under "
                f"{source['contract_payload_the_run_FINISHED_under'][:12]}"
            )
        if census_row.get("selection_verdict") and census_row["selection_verdict"] != "single_run":
            notes.append(
                f"{census_row['attempted_runs']} runs on the volume; "
                f"{census_row.get('selection_verdict_reason', '')}"
            )
        rows.append(
            ReplicationRow(
                target=target,
                seed=seed,
                cell=source["cell"],
                delta=delta,
                source_label=source["frozen_table"]["source_label"],
                arm=source["arm"],
                contract_path=source["contract_path"],
                contract_payload_sha256=source["contract_payload_sha256"],
                contract_file_sha256=source["contract_file_sha256"],
                code_revision=",".join(source["code_revisions"]),
                app_module=source["app_module"],
                controller_family=source["configuration_family"],
                support_operator=operator,
                charged_calls_per_cell=int(source["oracle_ceiling"]),
                total_charged_call_ceiling=int(
                    _payload(ROOT / source["contract_path"])["total_charged_call_ceiling"]
                ),
                historical_charged_calls=source["frozen_table"].get("realized_charged_calls"),
                historical_best=source["frozen_table"].get("compose"),
                controller_seed=controller_seed,
                replicate_seeds={
                    HISTORICAL_REPLICATE: controller_seed,
                    **{r: derive_replicate_seed(controller_seed, r) for r in NEW_REPLICATES},
                },
                completed_runs_today=census_row.get("completed_runs"),
                selection_verdict=census_row.get("selection_verdict"),
                notes=tuple(notes),
            )
        )

    deltas, ceilings = _contract_facts({row.contract_path for row in rows})
    # The ceiling a RESCUE contract declares is a debit of measured prior spend on that
    # cell. An independent replicate has no such prior, so the manifest records the
    # historical ceiling AND flags it rather than silently cloning a short budget.
    report = audit_manifest(rows, contract_delta=deltas, contract_ceiling=ceilings)
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    try:
        rows, report = build()
    except ManifestAuditError as failure:
        print("AUDIT FAILED -- nothing docks.\n", failure)
        return 1

    provenance = _payload(PROVENANCE)
    census = _payload(CENSUS)
    debited = [row for row in rows if row.charged_calls_per_cell != 250]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "new_oracle_calls": 0,
        "built_from": {
            "provenance": str(PROVENANCE.relative_to(ROOT)),
            "run_census": str(CENSUS.relative_to(ROOT)),
        },
        "why_replicates_exist": (
            "Both comparators report a MULTI-RUN MEAN per cell "
            "(docs/genmol_t4_all_methods.json: 'mean docking score of the most "
            "optimized lead over 3 runs'; configs/t4_published_invirtuogen_baseline.json: "
            "'value is InVirtuoGen mean, stderr in parentheses in the source', with "
            "per-cell stderrs of 0.1-0.9 kcal/mol). The frozen COMPOSE table reports a "
            "SINGLE-RUN best, which is not like-for-like."
        ),
        "selection_bias_answer": census["selection_bias_summary"],
        "configuration_families": provenance["configuration_families"],
        "label_audit": provenance["label_audit"],
        "broken_links": provenance["broken_links"],
        "audit": report,
        "seed_derivation": {
            "replicate_1": "the historical controller_seed, unchanged; never re-run",
            "replicate_n": "controller_seed + 8_000_000_029 * (n - 1)",
            "within_cell_stream_collisions": len(seed_stream_collisions(rows)),
            "inherited_cross_cell_seed_sharing": len(inherited_cross_cell_seed_sharing(rows)),
            "inherited_sharing_note": (
                "the two delta arms of one cell carry the SAME controller_seed in the "
                "historical contracts; a replicate clones the historical method, so that "
                "sharing is preserved and recorded rather than treated as a defect"
            ),
        },
        "budget_rule": (
            "clone the CEILING and the STOPPING RULE, never the realized call count. "
            f"{len(debited)} rows carry a sub-250 ceiling because the historical rescue "
            "contract debited measured prior spend on that cell; an independent replicate "
            "starting fresh has no such prior and the owner must decide deliberately "
            "whether to inherit the debit or use the full 250."
        ),
        "rows_with_debited_ceiling": [
            {"cell": row.cell, "delta": row.delta, "ceiling": row.charged_calls_per_cell}
            for row in debited
        ],
        "rows": [row.as_record() for row in rows],
    }
    payload["payload_sha256"] = identity(
        {k: v for k, v in payload.items() if k != "generated_at_utc"}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"{'target':7} {'sd':>2} {'delta':>5} {'arm':28} {'operator':14} "
          f"{'ceil':>4} {'real':>4} {'family':16} {'runs':>4} {'selection':32}")
    for row in sorted(payload["rows"], key=lambda r: (r["delta"], r["target"], r["seed"])):
        print(f"{row['target']:7} {row['seed']:2d} {row['delta']:5.1f} {row['arm']:28} "
              f"{row['support_operator']:14} {row['charged_calls_per_cell']:4d} "
              f"{row['historical_charged_calls']!s:>4} {row['controller_family'][:16]:16} "
              f"{row['completed_runs_today']!s:>4} {row['selection_verdict']!s:32}")
    print()
    print(json.dumps(payload["audit"], indent=2))
    print(json.dumps(payload["selection_bias_answer"], indent=2))
    print(f"manifest payload_sha256 {payload['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
