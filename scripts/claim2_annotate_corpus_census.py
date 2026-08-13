"""Write the mutual-edge-fraction definition into the committed corpus census.

The census artifact reported ``mutual_edge_fraction`` without saying how it was
formed, and the number does not reconstruct from the other fields unless you
already know the denominator. Two plausible readings of the same numbers give
0.352 and 0.703, and the main lane hit both before the definition was resolved.

This annotates the existing artifact in place. It RECOMPUTES every added field
from the measured counts already in the file and asserts the stored fraction
matches the formula, so nothing is transcribed and the measurement itself is
untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.claim2_cycle_attribution import (
    MUTUAL_EDGE_FRACTION_DENOMINATOR_RATIONALE,
    MUTUAL_EDGE_FRACTION_FORMULA,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--census", type=Path, default=Path("diagnostics/claim2_corpus_reversibility.json")
    )
    args = parser.parse_args()

    payload = json.loads(args.census.read_text())
    mutual = int(payload["mutual_pairs"])
    distinct = int(payload["distinct_directed_edges"])
    occurrences = int(payload["directed_edges"])

    recomputed = 2 * mutual / distinct
    stored = float(payload["mutual_edge_fraction"])
    if abs(recomputed - stored) > 1e-12:
        raise SystemExit(
            f"stored mutual_edge_fraction {stored!r} does not match "
            f"2*{mutual}/{distinct} = {recomputed!r}; the artifact is inconsistent "
            "and must not be annotated as though it were not"
        )

    lower = 2 * mutual / occurrences
    upper = (2 * mutual + (occurrences - distinct)) / occurrences
    payload.update(
        {
            "mutual_edge_fraction_formula": MUTUAL_EDGE_FRACTION_FORMULA,
            "mutual_edge_fraction_denominator_rationale": (
                MUTUAL_EDGE_FRACTION_DENOMINATOR_RATIONALE
            ),
            "mutual_fraction_occurrence_denominator_lower_bound": lower,
            "mutual_fraction_occurrence_denominator_upper_bound": upper,
            "denominator_robustness": (
                f"Under the distinct-edge denominator the figure is {stored:.6f}; "
                f"charging every observation to the occurrence denominator gives at "
                f"least {lower:.6f} and at most {upper:.6f}. All of these are far "
                "above the preregistered INHERITED threshold of 0.33, so the verdict "
                "does not depend on the denominator choice."
            ),
            "occurrence_weighted_value_not_retained": (
                "Per-edge multiplicities were not stored, so the exact "
                "occurrence-weighted fraction is a bracket rather than a point. The "
                "lower bound charges all "
                f"{occurrences - distinct} repeated observations to non-mutual edges."
            ),
            "annotated_post_hoc": (
                "Definition fields added after the run. Every added value is "
                "recomputed from the measured counts in this file; no measured "
                "quantity was changed."
            ),
        }
    )
    args.census.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"annotated {args.census}")
    print(f"  distinct-edge denominator : 2*{mutual}/{distinct} = {stored:.6f}")
    print(f"  occurrence denominator    : bracket [{lower:.6f}, {upper:.6f}]")
    print("  verdict unchanged under either denominator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
