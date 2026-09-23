"""Where do two IDENTICAL runs first diverge?

A boolean "they differ" cannot be acted on. This walks the six quantities in the order the
controller produces them and reports the FIRST mismatch, because the earliest divergence is
the only one that is a cause -- everything after it is a consequence.

It is also a negative control in its own right: if two runs configured identically do not
reproduce each other, then no comparison between two DIFFERENT configurations means anything,
and any earlier claim of the form "this change was inert" was measuring the instrument.
"""
from __future__ import annotations

import json
import pathlib
import sys

ORDER = (
    ("parent_probabilities", "parent_policy_history"),
    ("rebuild_option_probabilities", "option_policy"),
    ("selected_endpoints_per_round", "selected_endpoints_per_round"),
    ("generated_endpoint_sequence", "endpoint_sequence"),
    ("observations_rewards_updates", "observations"),
    ("rng_state", "rng_state"),
)


def _first_difference(left, right):
    """Index of the first differing element, or None when one is a prefix of the other."""
    if not isinstance(left, list) or not isinstance(right, list):
        return None if left == right else "value"
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            return index
    return None if len(left) == len(right) else "length"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    a, b = pathlib.Path(args[0]), pathlib.Path(args[1])
    with open(a / "equivalence.json") as handle:
        left = json.load(handle)
    with open(b / "equivalence.json") as handle:
        right = json.load(handle)

    findings, first = {}, None
    for name, key in ORDER:
        where = _first_difference(left.get(key), right.get(key))
        size = len(left.get(key) or []) if isinstance(left.get(key), list) else 1
        findings[name] = {"compared": size, "first_difference": where}
        if where is not None and first is None:
            first = name
        print(
            f"  {'SAME' if where is None else 'DIFFERS'}  {name}"
            f"  (compared {size}"
            + (f", first at {where}" if where is not None else "")
            + ")"
        )
        if where is not None and isinstance(where, int) and isinstance(left.get(key), list):
            print(f"      A: {json.dumps(left[key][where])[:150]}")
            print(f"      B: {json.dumps(right[key][where])[:150]}")

    empty = [n for n, f in findings.items() if f["compared"] == 0]
    verdict = (
        "REPRODUCIBLE"
        if first is None and not empty
        else ("VACUOUS_EMPTY_EVIDENCE" if first is None else "NONDETERMINISTIC")
    )
    report = {
        "schema_version": "pmo_determinism_control_v1",
        "evidence_role": "same_configuration_negative_control",
        "arms": [str(a), str(b)],
        "findings": findings,
        "empty_comparisons": empty,
        "first_divergence": first,
        "verdict": verdict,
    }
    out = pathlib.Path("diagnostics/pmo_determinism_control_v1")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "control.json", "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nVERDICT: {verdict}" + (f"   first divergence: {first}" if first else ""))
    if empty:
        print(f"WARNING: compared nothing for {empty}")
    print("WROTE", out / "control.json")
    return 0 if verdict == "REPRODUCIBLE" else 1


if __name__ == "__main__":
    sys.exit(main())
