"""Seal the controller evaluation panel: 91 = 24 development + 67 final held-out.

Run BEFORE any h_phi work exists, so the commit that introduces this file is
itself the proof that the 67 were designated before the value model was
trained. A split announced afterwards proves nothing; a split in git history
before the model exists is checkable by anyone.

WHAT THE SEAL MEANS
-------------------
The 24 development pairs were already used to establish that future-aware
control recovers targets myopic control misses, so h_phi's performance on them
is no longer an honest read -- they informed the override rule and the value
definition.

The 67 sealed pairs must not be used to evaluate h_phi until the value model,
its training law, its checkpoint-selection rule, and its inference rule are all
frozen. Nothing forbids reading them; the discipline is that no h_phi decision
is made on the basis of what they show.

Neither set may be used to TRAIN h_phi. Training pairs come from held-in
molecules, generated separately.

The commitment hashes let anyone verify later that the evaluation set was not
quietly reshaped once results were known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: Must match the stratified selection the pilot ran, or the seal describes a
#: different experiment from the one whose result we are acting on.
DEVELOPMENT_PAIRS = 24


def commitment(rows: list[dict[str, Any]]) -> str:
    """Order-independent digest over (source, target, steps)."""

    items = sorted(f"{r['source']}>>{r['target']}@{r['steps']}" for r in rows)
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def select_development(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Reproduce the pilot's stratified selection exactly."""

    per_band = max(1, count // 3)
    selected: list[dict[str, Any]] = []
    for length in (4, 5, 6):
        band = sorted([r for r in rows if r["steps"] == length],
                      key=lambda r: r["source_heavy_atoms"])
        take = min(per_band, len(band))
        picks = ([round(i * (len(band) - 1) / (take - 1)) for i in range(take)]
                 if take > 1 else [0])
        selected.extend(band[i] for i in dict.fromkeys(picks))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility", required=True, type=Path)
    parser.add_argument("--ran", required=True, type=Path,
                        help="the pilot result, to confirm the seal matches it")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = json.loads(args.feasibility.read_text())["transformations"]
    development = select_development(rows, DEVELOPMENT_PAIRS)
    keys = {(r["source"], r["target"]) for r in development}
    sealed = [r for r in rows if (r["source"], r["target"]) not in keys]
    print(f"{len(rows)} pairs = {len(development)} development + {len(sealed)} sealed")

    # The seal is worthless if it names a different 24 than the pilot ran.
    ran = json.loads(args.ran.read_text())["per_pair"]
    ran_targets = {r["target"] for r in ran}
    # The pilot reports canonical keys; compare on the target it recovered
    # toward, which is stable across the SMILES/canonical distinction for these.
    matched = sum(1 for r in development if r["target"] in ran_targets)
    print(f"  development pairs also present in the pilot result: "
          f"{matched}/{len(development)}")
    if matched != len(development):
        print("  WARNING: the seal does not describe the panel the pilot ran. "
              "Investigate before trusting either.")

    payload = {
        "schema": "compose.editing_v2.controller_panel_seal",
        "status": "SEALED_BEFORE_ANY_H_PHI_WORK_EXISTS",
        "rule": {
            "development": (
                "Used to establish that future-aware control recovers targets "
                "myopic control misses. h_phi performance on these is NOT an "
                "honest read: they informed the override rule and value "
                "definition."),
            "sealed": (
                "Must not be used to evaluate h_phi until the value model, its "
                "training law, its checkpoint-selection rule and its inference "
                "rule are all frozen."),
            "training": (
                "NEITHER set may be used to train h_phi. Training pairs come "
                "from held-in molecules and are generated separately."),
        },
        "counts": {"total": len(rows), "development": len(development),
                   "sealed": len(sealed)},
        "commitment": {
            "development_sha256": commitment(development),
            "sealed_sha256": commitment(sealed),
            "all_sha256": commitment(rows),
        },
        "development_pairs_verified_against_pilot": matched,
        "development": [
            {k: r[k] for k in ("source", "target", "steps", "source_heavy_atoms",
                               "target_heavy_atoms", "slots", "shape")}
            for r in development],
        "sealed": [
            {k: r[k] for k in ("source", "target", "steps", "source_heavy_atoms",
                               "target_heavy_atoms", "slots", "shape")}
            for r in sealed],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  development sha256 {payload['commitment']['development_sha256'][:16]}")
    print(f"  sealed      sha256 {payload['commitment']['sealed_sha256'][:16]}")

    import collections
    print(f"\n  sealed composition by verified steps: "
          f"{dict(sorted(collections.Counter(r['steps'] for r in sealed).items()))}")
    sizes = [r["source_heavy_atoms"] for r in sealed]
    print(f"  sealed source heavy atoms: min {min(sizes)} max {max(sizes)}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
