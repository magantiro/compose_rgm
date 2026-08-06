#!/usr/bin/env python3
"""Benchmark compact Process-V2 fiber compilation against the exhaustive oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rdkit
import torch

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    load_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    GATE_ZERO_MODEL_PROCESS_V2,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    validate_process_v2_t1_prepared_inputs,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    build_semantic_scratch_runtime,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    compiled_successor_map_from_payload,
)
from compose_v4.experiments.factorized_successor_training import (
    compile_state_successor_map,
    compile_teacher_successor_fibers_support_only,
    resolve_successor_process_runtime,
    rewrite_action_codec_sha256,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.experiments.score_free_successor_support import (
    enumerate_factorized_legal_support_many,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.trace_shard import decode_state

_SCHEMA = "compose.editing_v2.process_v2_successor_fiber_fastpath_benchmark"
_SCHEMA_VERSION = 1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes | object) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _model(prepared: dict[str, object], *, repo_root: Path):
    binding = prepared["model_binding"]
    if not isinstance(binding, dict) or not isinstance(binding.get("model_runtime"), dict):
        raise ValueError("prepared T1 artifact lacks its model runtime binding")
    runtime = binding["model_runtime"]
    config = SemanticScratchModelConfig(
        initialization_seed=int(runtime["initialization_seed"]),
        max_atoms=int(runtime["max_atoms"]),
        hidden_dim=int(runtime["hidden_dim"]),
        message_passing_steps=int(runtime["message_passing_steps"]),
        mark_dim=int(runtime["mark_dim"]),
        dtype=str(runtime["dtype"]),
        atom_vocabulary_class_count=int(runtime["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(runtime["catalog_fingerprint"]),
    )
    contract = load_gate_zero_semantic_contract(repo_root / GATE_ZERO_MODEL_PROCESS_V2)
    scratch = build_semantic_scratch_runtime(config, contract)
    if scratch.semantic_model_identity["editing_process_semantics"] != "semantic_editing_v2_v2":
        raise ValueError("benchmark model is not the Process-V2 semantic editor")
    return scratch.model.eval()


def _sample(entries: list[dict[str, object]], *, per_family: int) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in sorted(entries, key=lambda row: str(row["panel_entry_sha256"])):
        family = str(entry["model_family"])
        if len(by_family[family]) < per_family:
            by_family[family].append(entry)
    if len(by_family) != 8 or any(len(rows) != per_family for rows in by_family.values()):
        raise ValueError("benchmark input lacks the requested balanced Active8 sample")
    return [entry for family in sorted(by_family) for entry in by_family[family]]


def benchmark(
    input_path: Path,
    *,
    repo_root: Path,
    source_revision: str,
    samples_per_family: int,
) -> dict[str, object]:
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("source revision must be a full lowercase Git commit")
    raw = input_path.read_bytes()
    prepared = validate_process_v2_t1_prepared_inputs(json.loads(raw))
    selected = _sample(prepared["entries"], per_family=samples_per_family)
    model = _model(prepared, repo_root=repo_root)
    sources = tuple(decode_state(entry["exact_state"]) for entry in selected)
    support_start = time.perf_counter()
    supports = enumerate_factorized_legal_support_many(
        model,
        sources,
        (0.5,) * len(sources),
        included_families=tuple(frozenset({str(entry["model_family"])}) for entry in selected),
    )
    support_seconds = time.perf_counter() - support_start
    system = resolve_successor_process_runtime(model).system
    targets = []
    for entry, source, support in zip(selected, sources, supports, strict=True):
        matches = []
        for mark in support.marks:
            successor = system.apply(source, mark.executor_rule_name, mark.action)
            if (
                persistent_slot_state_sha256(successor) == entry["target_state_sha256"]
                and rewrite_action_codec_sha256(
                    mark.executor_rule_name,
                    mark.action,
                    schema_version=action_codec_v4.SCHEMA_VERSION,
                )
                == entry["teacher_action_sha256"]
            ):
                matches.append(successor)
        if len(matches) != 1:
            raise ValueError("the frozen teacher action did not reconstruct exactly once")
        targets.append(matches[0])

    exhaustive_start = time.perf_counter()
    exhaustive = tuple(compile_state_successor_map(model, source, time=0.5) for source in sources)
    exhaustive_seconds = time.perf_counter() - exhaustive_start
    fast_start = time.perf_counter()
    compact = compile_teacher_successor_fibers_support_only(
        model,
        sources,
        tuple(targets),
        teacher_action_sha256s=tuple(str(entry["teacher_action_sha256"]) for entry in selected),
        teacher_families=tuple(str(entry["model_family"]) for entry in selected),
        times=(0.5,) * len(sources),
    )
    fast_seconds = time.perf_counter() - fast_start

    for entry, exhaustive_map, compact_row in zip(selected, exhaustive, compact, strict=True):
        frozen_map = compiled_successor_map_from_payload(entry["successor_partition"])
        frozen_fiber = teacher_successor_fiber_from_exact_digest(
            frozen_map, str(entry["target_state_sha256"])
        )
        current_fiber = teacher_successor_fiber_from_exact_digest(
            exhaustive_map, str(entry["target_state_sha256"])
        )
        if current_fiber != frozen_fiber or compact_row.teacher_fiber != frozen_fiber:
            raise ValueError("compact, current exhaustive, and frozen teacher fibers differ")

    body: dict[str, object] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "status": "MEASURED_BOUNDED_FASTPATH_EQUIVALENCE",
        "source_revision": source_revision,
        "input": {
            "path": str(input_path.resolve()),
            "file_sha256": _sha256(raw),
            "bytes": len(raw),
            "artifact_sha256": prepared["artifact_sha256"],
            "process_identity_sha256": prepared["process_identity_sha256"],
            "panel_sha256": prepared["panel_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "precision": "float32",
            "device": "cpu",
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if sys.platform == "darwin" else 1024),
        },
        "sample": {
            "examples_per_family": samples_per_family,
            "entry_count": len(selected),
            "families": sorted({str(entry["model_family"]) for entry in selected}),
            "panel_entry_sha256s": [str(entry["panel_entry_sha256"]) for entry in selected],
        },
        "result": {
            "teacher_target_reconstruction_seconds": support_seconds,
            "exhaustive_sequential_seconds": exhaustive_seconds,
            "compact_batched_seconds": fast_seconds,
            "speedup": exhaustive_seconds / fast_seconds,
            "fiber_mismatch_count": 0,
            "raw_mark_count_mismatch_count": sum(
                compact_row.raw_mark_count != int(entry["raw_mark_count"])
                for entry, compact_row in zip(selected, compact, strict=True)
            ),
        },
        "claim_boundary": {
            "measured": (
                "The compact compiler exactly matched both the current exhaustive compiler and "
                "the frozen exhaustive T1 fibers on this balanced 40-slot Active8 sample."
            ),
            "not_claimed": (
                "This bounded equivalence and runtime measurement is not a universal proof over "
                "every molecule in declared support."
            ),
        },
    }
    return {**body, "benchmark_sha256": _sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--samples-per-family", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.samples_per_family <= 0:
        parser.error("--samples-per-family must be positive")
    result = benchmark(
        arguments.input,
        repo_root=arguments.repo_root.resolve(),
        source_revision=arguments.source_revision,
        samples_per_family=arguments.samples_per_family,
    )
    _atomic_write(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
