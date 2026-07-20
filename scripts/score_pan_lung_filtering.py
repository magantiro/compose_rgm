#!/usr/bin/env python3
"""Score admitted candidates with the frozen filtering-only lung bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from compose_v4.oracles.pan_lung_filtering import (  # noqa: E402
    score_lut_filter,
    score_molecular_filter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/oracles/pan_lung_filtering_v1/manifest.json",
    )
    subparsers = parser.add_subparsers(dest="kind", required=True)
    molecular = subparsers.add_parser("molecular")
    molecular.add_argument("--domain", choices=("a549", "lumi"), required=True)
    molecular.add_argument("--smiles", required=True)
    lut = subparsers.add_parser("lut")
    lut.add_argument("--head", required=True)
    lut.add_argument("--tail", required=True)
    lut.add_argument("--round", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()

    if args.kind == "molecular":
        output = score_molecular_filter(args.manifest, args.domain, args.smiles)
    else:
        output = score_lut_filter(args.manifest, args.head, args.tail, args.round)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
