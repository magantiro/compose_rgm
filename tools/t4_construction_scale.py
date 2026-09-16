"""Publish the construction-scale diagnostic under its frozen contract.

Zero oracle calls, no model fit, no production sampler modified. Resumable: each
cell's sampling result is sealed as it completes and reused on a rerun.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_construction_scale import (
    SCHEMA_VERSION,
    builder_families,
    eligibility_by_size,
    family_profile,
    root_construction_density,
    route_decomposition,
    score_by_band,
    upper_bound_95,
)
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

ROOT = Path(__file__).resolve().parents[1]
GAP_CELLS = ("parp1_0", "jak2_1")


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _report(payload: dict) -> str:
    d, curves = payload["route_decomposition"], payload["eligibility_by_size"]
    lines = [
        "# T4 construction scale: the value is where the sampler cannot go",
        "",
        f"Contract `{payload['contract_sha256'][:16]}`. Zero oracle calls, no model fit,",
        "no production sampler modified.",
        "",
        "## The winning routes decompose into a few coordinated regions",
        "",
        (
            f"- {d['routes']} exact teacher routes, median {d['median_regions']:.0f} "
            f"dependency regions, median {d['median_primitives']:.0f} primitives"
        ),
        (
            f"- primitives per region: p25 {d['primitives_per_region']['p25']:.0f}, "
            f"median {d['primitives_per_region']['median']:.0f}, "
            f"p90 {d['primitives_per_region']['p90']:.0f}"
        ),
        (
            f"- {d['routes_reusing_a_created_handle']}/{d['routes']} reuse a handle they "
            f"created (median {d['median_reused_handles']:.0f} of "
            f"{d['median_created_handles']:.0f})"
        ),
        (
            f"- **{d['routes_with_an_ineligible_interior']}/{d['routes']} have an ineligible "
            f"interior**, median {d['median_ineligible_internal_prefixes']:.0f} prefixes"
        ),
        "",
        "That last line is why a chain of small eligible programs cannot get there.",
        "",
        "## What each family does",
        "",
        "| family | primitives | changed originals | created |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, row in sorted(
        payload["family_profile"].items(), key=lambda kv: -kv[1]["median_created"]
    ):
        lines.append(
            f"| {family} | {row['median_primitives']:.0f} "
            f"| {row['median_changed_originals']:.0f} | {row['median_created']:.0f} |"
        )
    lines += [
        "",
        (
            "Builders (create at least as much as they disturb): "
            f"{', '.join(payload['builder_families'])}. Only these move a program toward the"
        ),
        "winning shape; the rest scatter edits and break similarity.",
        "",
        "## Where the value is, and where the sampler is",
        "",
        "| primitives | corpus n | best | p5 | median | sampler eligible (parp1) | (jak2) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for band, row in payload["score_by_band"].items():
        rates = []
        for cell in GAP_CELLS:
            entry = curves.get(cell, {}).get("bands", {}).get(band)
            rates.append("-" if entry is None else f"{entry['rate']:.1%}")
        lines.append(
            f"| {band} | {row['n']} | {row['best']:.1f} | {row['p5']:.1f} "
            f"| {row['median']:.1f} | {rates[0]} | {rates[1]} |"
        )
    base = payload["baseline"]
    lines += [
        "",
        (
            f"**{base['successes']} of {base['trials']}** sampled proposals at >=15 "
            f"primitives were eligible ({base['rate']:.2%}); one-sided 95% upper bound "
            f"{base['upper_bound_95']:.2%}."
        ),
        "",
        "## But the region is densely populated",
        "",
        "| cell | root-applied | >=15 primitives | share | distinct endpoints | best | p5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, row in payload["root_density"].items():
        lines.append(
            f"| {cell} | {row['root_applied']} | {row['at_or_above_threshold']} "
            f"| {row['share']:.1%} | {row['distinct_endpoints']} "
            f"| {row['best']:.1f} | {row['p5']:.1f} |"
        )
    lines += [
        "",
        "Every corpus record is eligible by construction, so these are real eligible",
        "endpoints reachable in one program from the benchmark root. The region is not a",
        "needle; the sampler is aimed away from it.",
        "",
        "## Limits",
        "",
        "- This measures ACCESS, not docking value. Strategy report section 7 measured",
        "  distance-to-best against score-gap at Spearman 0.12 on PARP1, so eligible large",
        "  constructions still need real docking feedback to be aimed.",
        "- The root-applied density describes what the historical search, holding the",
        "  winner bank, chose to try. It proves the region is populated, not that it is",
        "  easy to hit without the bank.",
        "- Size bands, the 15-primitive threshold and the builder set were chosen after",
        "  inspecting this corpus. The prospective lane gate in the contract was not.",
        "- Local RDKit is newer than the pinned benchmark image.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=ROOT / "configs/t4_construction_scale_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "diagnostics/t4_proposal_prior/construction_scale_v1"
    )
    parser.add_argument("--attempts", type=int, default=200)
    args = parser.parse_args()

    contract = unseal(args.contract)
    if contract.get("schema_version") != SCHEMA_VERSION:
        parser.error(f"not a {SCHEMA_VERSION} contract")
    for entry in contract["inputs"].values():
        verify_file(ROOT / entry["path"], entry["sha256"])

    with gzip.open(ROOT / contract["inputs"]["corpus"]["path"], "rt") as handle:
        records = [json.loads(line) for line in handle]
    # probes.json is a plain artifact, not a sealed envelope; its identity is carried
    # by the contract's own hash check above rather than by a payload self-hash.
    probes = json.loads(
        (ROOT / contract["inputs"]["teacher_route_descriptors"]["path"]).read_text()
    )
    units = {
        u["cell"]: u for u in unseal(ROOT / contract["inputs"]["source_registry"]["path"])["units"]
    }
    roots = {cell: identity(unit["source_state"]) for cell, unit in units.items()}

    profile = family_profile(records)
    from compose_v4.control.dynamic_program_synthesis import GENERIC_MODULES

    builders = builder_families(profile, universe=GENERIC_MODULES)

    args.output.mkdir(parents=True, exist_ok=True)
    began, curves = perf_counter(), {}
    for cell in GAP_CELLS:
        completed = args.output / "cells" / f"{cell}.json"
        if completed.exists():
            curves[cell] = unseal(completed)
            print({"cell": cell, "phase": "reused"}, flush=True)
            continue
        from compose_v4.rewrite.trace_shard import decode_state

        unit = units[cell]
        curve = eligibility_by_size(
            decode_state(unit["source_state"]),
            strict_endpoint_scorer(unit["original_seed"], delta=0.4),
            np.random.default_rng(np.random.SeedSequence([20260916, 77, len(cell)])),
            attempts=args.attempts,
            constructive=True,
            builders=builders,
        )
        seal(completed, curve)
        curves[cell] = curve
        print(
            {"cell": cell, "bands": {k: v["rate"] for k, v in curve["bands"].items()}}, flush=True
        )

    large = [
        (band["n"], band["eligible"])
        for curve in curves.values()
        for key, band in curve["bands"].items()
        if key in ("15-22", "23-32")
    ]
    trials = sum(n for n, _ in large)
    successes = sum(e for _, e in large)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": sha256_file(args.contract),
        "route_decomposition": route_decomposition(probes["teacher_route_descriptors"]),
        "family_profile": profile,
        "builder_families": list(builders),
        "score_by_band": score_by_band(records, GAP_CELLS),
        "root_density": root_construction_density(records, roots),
        "eligibility_by_size": curves,
        "baseline": {
            "successes": successes,
            "trials": trials,
            "rate": successes / trials if trials else None,
            "upper_bound_95": upper_bound_95(successes, trials) if trials else None,
            "method": "one-sided 95% Clopper-Pearson upper bound",
            "cells": list(GAP_CELLS),
        },
        "new_oracle_calls": 0,
        "new_labels": 0,
        "elapsed_seconds": perf_counter() - began,
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
    }
    seal(args.output / "result.json", payload)
    (args.output / "REPORT.md").write_text(_report(payload))
    print(json.dumps({"builders": list(builders), "baseline": payload["baseline"]}, indent=2))


if __name__ == "__main__":
    main()
