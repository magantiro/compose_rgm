#!/usr/bin/env python3
"""Build the immutable whole-trace active-8 editing-corpus boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.active8_trace_inventory import (
    ACTIVE8_FAMILIES,
    Active8SourceShard,
    ProductionExactCandidateChecker,
    build_active8_trace_inventory,
)
from compose_v4.data.packed_charge_policy_audit import (
    resolve_unified_manifest_shards,
)
from compose_v4.experiments.editing_gate_zero_runtime import (
    build_scratch_ringcore_model,
    load_gate_zero_runtime_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Admit a packed editing trace only when every teacher belongs to "
            "the active-8 production support, then retain every progress row."
        )
    )
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--mmp-root", type=Path, default=None)
    parser.add_argument(
        "--support-contract",
        type=Path,
        required=True,
        help=(
            "Frozen scratch RingCore runtime contract defining the exact "
            "operator capabilities; this is identity input, not Gate-0 evidence."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cache-size", type=int, default=4096)
    return parser


def main() -> None:
    args = _parser().parse_args()
    contract = load_gate_zero_runtime_contract(args.support_contract)
    if contract.required_families != ACTIVE8_FAMILIES:
        raise RuntimeError(
            "support contract active families disagree with the inventory boundary"
        )
    model, _ = build_scratch_ringcore_model(contract)
    source_manifest, declared = resolve_unified_manifest_shards(
        args.unified_manifest,
        audit_root=args.audit_root,
        mmp_root=args.mmp_root,
    )
    shards = tuple(
        Active8SourceShard(
            manifest_layer=shard.manifest_layer,
            envelope_layer=shard.envelope_layer,
            partition=shard.partition,
            relative_path=shard.relative_path,
            path=shard.path,
        )
        for shard in declared
    )
    checker = ProductionExactCandidateChecker(
        model,
        cache_size=args.candidate_cache_size,
    )
    completed = 0

    def progress(report: dict[str, object]) -> None:
        nonlocal completed
        completed += 1
        counts = report["counts"]
        print(
            f"[{completed:03d}/{len(shards):03d}] "
            f"{report['manifest_layer']}/{report['partition']}/"
            f"{report['relative_path']}: "
            f"{counts['accepted_traces']} accepted, "
            f"{counts['excluded_traces']} excluded",
            flush=True,
        )

    manifest = build_active8_trace_inventory(
        shards,
        exact_candidate_checker=checker,
        source_manifest_path=args.unified_manifest,
        source_manifest=source_manifest,
        support_contract_sha256=contract.sha256,
        output_dir=args.output_dir,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "exclusions_by_reason": manifest["exclusions_by_reason"],
                "effective_source_corpus_cache_sha256": manifest[
                    "source_identity"
                ]["effective_source_corpus_cache_sha256"],
                "inventory_sha256": manifest["inventory_sha256"],
                "output": str(
                    args.output_dir / "ACTIVE8_TRACE_INVENTORY.json"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
