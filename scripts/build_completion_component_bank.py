#!/usr/bin/env python
"""Compile the objective-blind completion component bank.

The donor list is ``docs/PMO_INIT_BANK.json`` -- 100 SMILES described by their
own provenance as "objective-blind, fixed before any PMO task is run".  Nothing
task-specific, answer-known or oracle-informed enters this artifact: every row
is a pendant component of a fixed molecule list, carried by graph alone.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from compose_v4.control.completion_component_law import (  # noqa: E402
    build_component_bank,
)
from compose_v4.control.docking_value import identity  # noqa: E402

DONOR = "docs/PMO_INIT_BANK.json"
OUT = "configs/pmo_completion_component_bank_v1.json"


def main() -> None:
    donor = json.load(open(DONOR))
    bank, census = build_component_bank(
        donor["smiles"],
        slots=48,
        minimum_size=1,
        maximum_size=32,
        provenance={
            "donor_path": DONOR,
            "donor_rule": donor["rule"],
            "donor_sha256": identity(donor),
            "oracle_calls": 0,
            "task_identities_read": 0,
        },
    )
    payload = bank.as_json()
    payload["build_census"] = census
    payload["bank_identity_sha256"] = bank.identity_sha256
    with open(OUT, "w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=1)
    sizes = [spec.size for spec in bank.components]
    ring = sum(1 for spec in bank.components if spec.has_ring)
    branched = sum(1 for spec in bank.components if spec.branch_points)
    print(f"components retained : {len(bank.components)}")
    print(f"identity            : {bank.identity_sha256}")
    print(f"size median/max     : {sorted(sizes)[len(sizes) // 2]} / {max(sizes)}")
    print(f"ring-bearing        : {ring / len(sizes):.3f}")
    print(f"branched            : {branched / len(sizes):.3f}")
    print(f"size >= 13          : {sum(1 for s in sizes if s >= 13) / len(sizes):.3f}")
    print("census              :", census)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


main()
