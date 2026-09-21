"""Does the ring-grow HOST explain the ROLLOUT support, not just training traces?

The established de-novo figure -- uniform mass on 3/4-ring templates over the
exact executable support is 0.4431 at the states where ring events fire -- was
measured on rollout states and is stored, with those states, in the committed
reference audit. That makes it directly attributable: the audit supplies both
the support statistic and the state it was measured on, so the host census can
be computed against the same rows without re-deriving either.

The question this answers is whether the training-side mechanism (a ring
decision is made against an acyclic-carbon host that every committed ring
shrinks) is the same mechanism operating at sampling time, or a different one
that happens to look similar.

Reads the audit, computes nothing the audit already reports, and cross-checks
its own read of the mean against the audit's stored summary before reporting
anything -- so a misread of the artifact cannot be mistaken for a finding.

Trains nothing. Calls no oracle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT / "diagnostics" / "ring_calibration" / "step2500_exact_support_audit_12.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=40)
    args = parser.parse_args()

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.eval.ring_support_host import (
        eligible_host_census,
        host_size_predicts,
    )

    audit = json.loads(args.audit.read_text())
    rows = audit["rows"]

    measured: list[dict] = []
    for row in rows:
        state = pad_molecular_graph(
            smiles_to_molecular_graph(row["before"]), args.max_atoms
        )
        measured.append(
            {
                "row": row["row"],
                "legal_template_count": row["legal_template_count"],
                "legal_small_template_count": row["legal_small_template_count"],
                "small_mass_uniform_support": row["small_mass_uniform_support"],
                **eligible_host_census(state),
            }
        )

    # Cross-check against the audit's OWN stored summary before reporting.
    # A disagreement means this script misread the artifact, which must not be
    # discovered after the attribution has been believed.
    observed_mean = sum(m["small_mass_uniform_support"] for m in measured) / len(measured)
    stored_mean = float(audit["summary"]["mean_masses"]["small_mass_uniform_support"])
    if abs(observed_mean - stored_mean) > 1e-9:
        raise RuntimeError(
            f"read {observed_mean} but the audit stores {stored_mean}; reconcile "
            "before attributing anything"
        )

    trees = [m["largest_host_tree"] for m in measured]
    saturated = [m for m in measured if m["small_mass_uniform_support"] >= 1.0]
    report = {
        "measurement": "denovo_rollout_host_attribution_v1",
        "trains_nothing": True,
        "oracle_calls": 0,
        "audit": str(args.audit.relative_to(ROOT)),
        "states": len(measured),
        "reproduces_audit_mean": True,
        "mean_small_mass_uniform_support": observed_mean,
        "mean_largest_host_tree": sum(trees) / len(trees),
        "r_mass_vs_largest_host_tree": host_size_predicts(
            trees, [m["small_mass_uniform_support"] for m in measured]
        ),
        "r_log_legal_vs_largest_host_tree": host_size_predicts(
            trees, [math.log(m["legal_template_count"] + 1) for m in measured]
        ),
        # The headline the mean hides: states where EVERY legal template is a
        # small ring, so the policy has no choice available to it at all.
        "states_with_no_non_small_option": len(saturated),
        "saturated_rows": [
            {
                "row": m["row"],
                "legal_template_count": m["legal_template_count"],
                "largest_host_tree": m["largest_host_tree"],
                "lost_to_cycles": m["lost_to_cycles"],
            }
            for m in saturated
        ],
        "rows": measured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"rows", "saturated_rows"}},
            indent=1,
            sort_keys=True,
        )
    )
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
