"""Freeze the V2 dataset: verify predeclared gates, then emit three identities.

This is a PRE-TRAINING GATE, not a report. It recomputes the census, the split
resolution and the sampling law from durable artifacts only, checks each result
against a number declared BEFORE the scan, and refuses to emit anything if a
single check disagrees.

THE EXPECTED VALUES BELOW ARE NOT TO BE EDITED TO MATCH A SCAN.
If the scan disagrees, either an input moved or something is wrong; both are
findings. Updating the constant to match whatever was measured converts a gate
into a rubber stamp, which is precisely how this project once carried a
"145,180 pre-existing rows" figure that described a plan rather than any
artifact.

DURABLE INPUTS ONLY
-------------------
Every root must live outside an OS-reapable path. A frozen hash is a promise
that the bytes it describes still exist; computing one from ``/private/tmp``
would be a promise about bytes the OS deletes on a schedule. A worktree was
reaped mid-session exactly that way.

THREE IDENTITIES, DELIBERATELY SEPARATE
---------------------------------------
    library_sha256       what chemistry physically exists
    split_sha256         which canonical sources belong to which role
    sampling_law_sha256  how the training role is actually drawn

They are separate because they change for different reasons and at different
rates. A resampling ablation moves the third and must not appear to invalidate
the first two; recompiling more chemistry moves the first without touching the
split. Collapsing them into one hash would make every ablation look like a new
corpus.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from editing_v2_available_library_census import library_state_pairs  # noqa: E402
from editing_v2_build_sampling_law import allocate, attribute  # noqa: E402
from editing_v2_resolve_split_precedence import (  # noqa: E402
    PRECEDENCE,
    resolve,
    sources_by_role,
)

from compose_v4.data.durable_path import require_durable_path  # noqa: E402

# ---- Predeclared gates. Declared before the scan. Do not edit to fit a scan. ----
EXPECTED = {
    "canonical_train_pairs": 144_886,
    "families": 8,
    "min_family_share": 0.0465,
    "capability_cells": 19,
    "multi_successor_fraction": 0.294,
    "realized_synthetic_share": 0.252,
    "excluded_train_sources": 97,
    "cross_role_source_overlap": 0,
    # A family-level law can extinguish a capability outright: measured,
    # atom_restate:valence_state_change (55 pairs, synthetic-only) received ZERO
    # mass while every family coefficient read perfectly and drift was +0.00000.
    # This is the only gate that looks below family granularity.
    "capability_cells_with_mass": 19,
}
#: Shares are compared at three decimals, the precision they were declared at.
SHARE_TOLERANCE = 0.0005


def _fail(checks: list[tuple[str, bool, str]]) -> None:
    print("\nGATE RESULTS")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34} {detail}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(
            "\nFREEZE REFUSED: a predeclared gate failed. Do not edit EXPECTED to "
            "match the scan -- either an input moved or something is wrong, and "
            "both are findings."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--corpus-root", action="append", default=[])
    parser.add_argument("--pairs-file", action="append", default=[])
    parser.add_argument("--manifest", required=True,
                        help="Prepared training manifest; its realized per-cell mass "
                             "is gated here.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    active8 = require_durable_path(args.active8_root, role="active8 root")
    corpus_roots = [require_durable_path(r, role="corpus root") for r in args.corpus_root]
    pairs_files = [require_durable_path(p, role="pair-digest file") for p in args.pairs_file]
    print("durable inputs verified\n")

    # ---- split ----
    by_role = sources_by_role(active8)
    kept, excluded = resolve(by_role)
    overlap = sum(
        len(kept[a] & kept[b])
        for i, a in enumerate(PRECEDENCE)
        for b in PRECEDENCE[i + 1:]
    )

    # ---- library + law ----
    state_pairs: set[str] = set()
    for root in corpus_roots:
        state_pairs |= library_state_pairs(root)
    for path in pairs_files:
        state_pairs |= set(json.loads(Path(path).read_text()))
    # The LIBRARY is every compiled train pair; the LAW draws only from those
    # whose source is absent from every held-out partition. Conflating them
    # would report the library as 16 pairs smaller than it is and quietly fold
    # a split decision into a chemistry inventory.
    rows = attribute(active8, state_pairs, set())
    eligible = [r for r in rows if r["source"] not in set(excluded["train"])]
    plan = allocate(eligible, floor=0.05, cap=0.22, synthetic_target=0.18, max_oversample=3.0)
    supply, draw = plan["supply"], plan["draw"]

    # LIBRARY statistics come from every compiled train pair.
    total = len(rows)
    families = collections.Counter(r["family"] for r in rows)
    cells = {r["cell"] for r in rows}
    sources = {r["source"] for r in rows}
    successors_per_source = collections.Counter(r["source"] for r in rows)
    multi = sum(1 for s in sources if successors_per_source[s] > 1)
    multi_fraction = multi / max(len(sources), 1)
    realized_synth = sum(v for (_, syn), v in draw.items() if syn)
    eligible_total = len(eligible)
    min_share = min(v / total for v in families.values())

    manifest = json.loads(Path(args.manifest).read_text())
    realized_cells = manifest.get("capability_cell_realized", {})
    cells_with_mass = sum(1 for v in realized_cells.values() if v > 0)

    checks = [
        ("canonical_train_pairs", total == EXPECTED["canonical_train_pairs"],
         f"{total:,} (expected {EXPECTED['canonical_train_pairs']:,})"),
        ("families", len(families) == EXPECTED["families"],
         f"{len(families)} (expected {EXPECTED['families']})"),
        ("min_family_share", min_share >= EXPECTED["min_family_share"] - SHARE_TOLERANCE,
         f"{100*min_share:.2f}% (floor {100*EXPECTED['min_family_share']:.2f}%)"),
        ("capability_cells", len(cells) == EXPECTED["capability_cells"],
         f"{len(cells)} (expected {EXPECTED['capability_cells']})"),
        ("multi_successor_fraction",
         abs(multi_fraction - EXPECTED["multi_successor_fraction"]) <= SHARE_TOLERANCE,
         f"{100*multi_fraction:.2f}% (expected {100*EXPECTED['multi_successor_fraction']:.1f}%)"),
        ("realized_synthetic_share",
         abs(realized_synth - EXPECTED["realized_synthetic_share"]) <= SHARE_TOLERANCE,
         f"{100*realized_synth:.2f}% (expected {100*EXPECTED['realized_synthetic_share']:.1f}%)"),
        ("excluded_train_sources",
         len(excluded["train"]) == EXPECTED["excluded_train_sources"],
         f"{len(excluded['train'])} (expected {EXPECTED['excluded_train_sources']})"),
        ("cross_role_source_overlap", overlap == EXPECTED["cross_role_source_overlap"],
         f"{overlap} (expected {EXPECTED['cross_role_source_overlap']})"),
        ("capability_cells_with_mass",
         cells_with_mass == EXPECTED["capability_cells_with_mass"],
         f"{cells_with_mass} of {len(realized_cells)} drawn "
         f"(expected {EXPECTED['capability_cells_with_mass']})"),
    ]
    _fail(checks)

    # ---- three identities ----
    library_body = {
        "canonical_train_pairs": total,
        "by_family": {f: families[f] for f in sorted(families)},
        "capability_cells": sorted(cells),
        "distinct_sources": len(sources),
        "multi_successor_sources": multi,
    }
    split_body = {
        "precedence": list(PRECEDENCE),
        "sources_kept": {r: len(kept[r]) for r in PRECEDENCE},
        "excluded_source_keys": {r: sorted(excluded[r]) for r in PRECEDENCE},
    }
    law_body = {
        "constraints": {"family_floor": 0.05, "family_cap": 0.22,
                        "synthetic_target": 0.18, "max_oversample": 3.0,
                        "capability_cell_floor": int(manifest["cell_floor"])},
        "capability_cell_realized": realized_cells,
        "strata": sorted(
            [
                {"family": f, "provenance": "synthetic" if syn else "real",
                 "available_pairs": supply.get((f, syn), 0), "draw_share": round(share, 9)}
                for (f, syn), share in draw.items() if share > 0
            ],
            key=lambda s: (s["family"], s["provenance"]),
        ),
    }

    def _h(body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    frozen = {
        "schema": "compose.editing_v2.v2_dataset_freeze",
        "schema_version": 1,
        "status": "FROZEN_DATASET_IDENTITY",
        "library_sha256": _h(library_body),
        "split_sha256": _h(split_body),
        "sampling_law_sha256": _h(law_body),
        "gates": {name: detail for name, _, detail in checks},
        "census": {
            "canonical_train_pairs": total,
            "eligible_after_exclusions": eligible_total,
            "distinct_sources": len(sources),
            "multi_successor_sources": multi,
            "multi_successor_fraction": round(multi_fraction, 4),
            "capability_cells": len(cells),
            "available_synthetic_share": round(
                sum(1 for r in rows if r["synthetic"]) / max(total, 1), 4),
            "realized_synthetic_share": round(realized_synth, 4),
            "family_available_pairs": {f: families[f] for f in sorted(families)},
            "family_available_share": {
                f: round(families[f] / total, 4) for f in sorted(families)},
            "family_realized_share": {
                f: round(sum(v for (fam, _), v in draw.items() if fam == f), 4)
                for f in sorted(families)},
        },
        "split": {
            "precedence": list(PRECEDENCE),
            "sources_per_role_kept": {r: len(kept[r]) for r in PRECEDENCE},
            "sources_excluded_per_role": {r: len(excluded[r]) for r in PRECEDENCE},
            "residual_cross_role_overlap": overlap,
        },
        "inputs": {
            "active8_root": str(active8),
            "corpus_roots": [str(r) for r in corpus_roots],
            "pair_digest_files": [str(p) for p in pairs_files],
        },
    }
    Path(args.out).write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    print(f"\n  library_sha256      {frozen['library_sha256']}")
    print(f"  split_sha256        {frozen['split_sha256']}")
    print(f"  sampling_law_sha256 {frozen['sampling_law_sha256']}")
    print(f"\nFROZEN -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
