"""Zero-oracle transfer decision for the frozen anchored-replacement expert."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "t4_anchored_replacement_transfer_v1"


def compare_transfer_cells(
    summaries: dict[str, dict[str, dict[str, Any]]],
    *,
    minimum_panel_yield: int,
) -> dict[str, Any]:
    """Apply the frozen support gate without using docking or teacher structures."""

    if not summaries:
        raise ValueError("the transfer comparison needs at least one cell")
    cells: dict[str, Any] = {}
    for cell, arms in sorted(summaries.items()):
        if set(arms) != {"shallow", "anchored_replacement"}:
            raise ValueError(f"{cell} does not contain the two frozen proposal lanes")
        shallow = arms["shallow"]
        anchored = arms["anchored_replacement"]
        if shallow.get("new_oracle_calls") != 0 or anchored.get("new_oracle_calls") != 0:
            raise ValueError("a zero-oracle support artifact reports a charged call")
        anchored_yield = int(anchored["unique_eligible_endpoints"])
        shallow_yield = int(shallow["unique_eligible_endpoints"])
        cells[cell] = {
            "shallow_unique_eligible": shallow_yield,
            "anchored_unique_eligible": anchored_yield,
            "anchored_minus_shallow": anchored_yield - shallow_yield,
            "anchored_worker_failures": int(anchored["worker_failures"]),
            "shallow_worker_failures": int(shallow["worker_failures"]),
            "minimum_panel_yield_pass": anchored_yield >= minimum_panel_yield,
            "nonzero_support_pass": anchored_yield > 0,
        }
    engineering = all(
        row["anchored_worker_failures"] == 0 and row["shallow_worker_failures"] == 0
        for row in cells.values()
    )
    nonzero = all(row["nonzero_support_pass"] for row in cells.values())
    panel = all(row["minimum_panel_yield_pass"] for row in cells.values())
    return {
        "cells": cells,
        "engineering_gate": "PASS" if engineering else "FAIL",
        "transfer_support_gate": "PASS" if engineering and nonzero else "FAIL",
        "full_panel_yield_gate": "PASS" if engineering and panel else "FAIL",
        "minimum_panel_yield": minimum_panel_yield,
        "new_oracle_calls": 0,
        "interpretation": (
            "PASS only establishes autonomous feasible proposal support on both new "
            "sources; it is not docking utility or FiberControl evidence"
        ),
    }
