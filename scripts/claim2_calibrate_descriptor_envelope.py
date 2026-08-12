"""Calibrate the Claim-2 chemical envelope on held-in molecules.

Run BEFORE any trajectory exists.  The order matters more than the arithmetic:
an envelope fitted after seeing rollouts can always be widened until the
trajectories fit inside it, at which point "stayed in distribution" measures
nothing.  Fitting first, committing the hashes, and only then rolling out is
what makes envelope retention a falsifiable quantity.

The envelope is a coordinate-wise [0.5%, 99.5%] box over the nine frozen
descriptors, plus the mean and standard deviation used to standardize drift.
Both quantiles are frozen in ``claim2_trajectory_metrics`` rather than passed
in, so there is no knob to turn here.

Pool: ``training_source_keys`` from the committed matched-validation reserve
ids -- the held-in universe.  The held-out reserve is never read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from compose_v4.experiments.claim2_trajectory_metrics import (
    DESCRIPTOR_NAMES,
    calibrate_envelope,
    descriptor_vector,
)

RESERVE_IDS = Path("diagnostics/editing_v2_matched_validation_reserve_ids.json.gz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap the calibration sample; 0 uses every held-in molecule",
    )
    args = parser.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    reserve = json.load(gzip.open(args.repo_root / RESERVE_IDS, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    if args.limit:
        held_in = held_in[: args.limit]
    print(f"calibrating on {len(held_in):,} held-in molecules")

    vectors, unparseable = [], 0
    for smiles in held_in:
        vector = descriptor_vector(smiles)
        if vector is None:
            unparseable += 1
            continue
        vectors.append(vector)
    print(f"  usable {len(vectors):,}, unparseable {unparseable}")

    envelope = calibrate_envelope(
        vectors,
        calibration_source=(
            f"training_source_keys of {RESERVE_IDS.name} "
            f"({len(vectors)} molecules of {len(held_in)})"
        ),
        status="DESIGN_ONLY_CALIBRATED_ON_HELD_IN_BEFORE_ANY_TRAJECTORY",
    )
    inside = sum(1 for vector in vectors if envelope.contains(vector))
    retention = inside / len(vectors)

    print(f"\n{'descriptor':>18}  {'p0.5':>10} {'p99.5':>10} {'mean':>10} {'sd':>10}")
    for index, name in enumerate(DESCRIPTOR_NAMES):
        print(
            f"{name:>18}  {envelope.lower[index]:>10.3f} {envelope.upper[index]:>10.3f} "
            f"{envelope.mean[index]:>10.3f} {envelope.stdev[index]:>10.3f}"
        )
    print(f"\nheld-in self-retention: {inside:,}/{len(vectors):,} = {retention:.4f}")
    print(
        "  This is the CEILING for every arm. A coordinate-wise box at 0.5%/99.5% "
        "over nine descriptors excludes part of its own calibration sample, so "
        "retention below 1.0 is expected and is not drift."
    )

    payload = envelope.to_json()
    payload.update(
        {
            "schema": "compose.claim2.descriptor_envelope",
            "held_in_self_retention": retention,
            "held_in_usable": len(vectors),
            "held_in_unparseable": unparseable,
            "envelope_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "names": list(envelope.names),
                        "lower": list(envelope.lower),
                        "upper": list(envelope.upper),
                        "mean": list(envelope.mean),
                        "stdev": list(envelope.stdev),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
