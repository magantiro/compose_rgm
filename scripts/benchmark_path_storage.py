"""Measure exact eager versus checkpointed rewrite-path storage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from time import perf_counter

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.tracelet_conditional import (
    build_tracelet_path_records,
    build_tree_transport_path_records,
)
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from compose_v4.rewrite.compiler import TraceCompilationError
from compose_v4.rewrite.kernel import InvalidRewrite
from compose_v4.rewrite.tracelets import TraceletLoweringError


def _read_eligible(path: Path, *, limit: int, max_atoms: int) -> tuple[str, ...]:
    selected = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            text = fields[0]
            try:
                size = smiles_to_molecular_graph(text).n_real_atoms
            except ValueError:
                continue
            if size <= max_atoms:
                selected.append(text)
            if len(selected) >= limit:
                break
    if not selected:
        raise ValueError("no eligible molecules found")
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--couplings", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if min(args.limit, args.max_atoms, args.couplings, args.checkpoint_interval) <= 0:
        raise ValueError("limit, slots, couplings, and checkpoint interval must be positive")

    smiles = _read_eligible(
        args.smiles_file,
        limit=args.limit,
        max_atoms=args.max_atoms,
    )
    proposal_pairs = []
    for text in smiles:
        try:
            record = build_tracelet_path_records(
                (text,),
                n_slots=args.max_atoms,
                typed_ring_payloads=True,
                workers=0,
                checkpoint_interval=args.checkpoint_interval,
            )[0]
        except (TraceCompilationError, TraceletLoweringError, InvalidRewrite):
            continue
        proposal_pairs.append((text, record))
    proposal_records = tuple(record for _, record in proposal_pairs)
    ring_catalog = build_typed_ring_catalog(
        (record.path.trace for record in proposal_records),
        max_cycle_templates=512,
        max_attach_templates=512,
        max_ear_templates=512,
    )
    supported = tuple(
        text
        for text, record in proposal_pairs
        if ring_catalog.supports_trace(record.path.trace)
    )
    sizes = Counter(smiles_to_molecular_graph(text).n_real_atoms for text in supported)
    prior = DegreeBoundedCarbonTreePrior.from_size_counts(dict(sizes))
    started = perf_counter()
    compact_records = []
    failed_transport = 0
    for index, text in enumerate(supported):
        try:
            compact_records.extend(
                build_tree_transport_path_records(
                    (text,),
                    n_slots=args.max_atoms,
                    source_prior=prior,
                    seed=args.seed + index,
                    couplings_per_target=args.couplings,
                    transport_mode="size_matched_graft",
                    typed_ring_payloads=True,
                    ring_catalog=ring_catalog,
                    workers=0,
                    checkpoint_interval=args.checkpoint_interval,
                )
            )
        except (TraceCompilationError, TraceletLoweringError, InvalidRewrite):
            failed_transport += 1
    compact = tuple(compact_records)
    if not compact:
        raise RuntimeError("no transport paths compiled")
    compile_seconds = perf_counter() - started
    eager = tuple(
        PathRecord(record.target_key, TraceProgressCTMC(record.path.trace))
        for record in compact
    )
    compact_bytes = sum(len(pickle.dumps(record, protocol=5)) for record in compact)
    eager_bytes = sum(len(pickle.dumps(record, protocol=5)) for record in eager)
    print(
        json.dumps(
            {
                "targets": len(supported),
                "unsupported_targets": len(smiles) - len(supported),
                "transport_failed_targets": failed_transport,
                "records": len(compact),
                "max_atoms": args.max_atoms,
                "checkpoint_interval": args.checkpoint_interval,
                "compile_seconds": compile_seconds,
                "records_per_second": len(compact) / compile_seconds,
                "eager_bytes": eager_bytes,
                "checkpointed_bytes": compact_bytes,
                "compression_ratio": eager_bytes / compact_bytes,
                "mean_path_length": sum(
                    record.path.path_length for record in compact
                )
                / len(compact),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
