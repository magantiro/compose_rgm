"""Fail-closed V2 publication for compiled molecular-pair edit pools.

The mining work directory is mutable execution state.  This module verifies that every expected
compile shard exists exactly once, that shard assignment covers the global pair indices without gaps or
overlap, and that every emitted row belongs to one frozen analogue-support contract.  Only then does it
copy the reduced pool into a content-addressed, no-overwrite namespace and publish
``MMP_POOL_COMPLETE.json`` for the V2 packer.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from compose_v4.experiments.analogue_prior import ANALOGUE_SUPPORT_CONTRACT

POOL_CONTRACT_SCHEMA = "compose.mmp_analogue_pool.v2"
COMPILER_IDENTITY_SCHEMA = "compose.mmp_compiler_identity.v2"
COMPILE_SHARD_SCHEMA = "compose.mmp_compile_shard.v2"
REDUCTION_SCHEMA = "compose.mmp_mining_reduction.v2"
FREEZER_IDENTITY_SCHEMA = "compose.mmp_pool_freezer_identity.v2"
FROZEN_NAMESPACE_PREFIX = "mmp_pool_v2_"

_EXECUTION_ONLY_CONFIG_FIELDS = frozenset({
    "corpus_path",
    "corpus_sha256",
    "n_shards",
    "shard_index",
    "shard_rule",
    "split_workers",
    "out_dir",
})
_REQUIRED_SCIENTIFIC_CONFIG_FIELDS = frozenset({
    "corpus_id",
    "train_size",
    "validation_size",
    "test_size",
    "max_atoms",
    "split_seed",
    "scope_name",
    "scan_all",
    "max_variable_atoms",
    "max_pairs_per_core",
    "scaffold_k",
    "scaffold_max_per_scaffold",
    "scaffold_max_per_transformation",
    "scaffold_mcs_candidates",
    "scaffold_mcs_timeout",
    "scaffold_max_group_size",
    "max_pairs_per_source",
    "corruption_depth_max",
    "corruption_couplings_per_target",
    "corruption_sample_size",
    "corruption_seed",
    "organic_vocabulary",
})
_HEX64 = frozenset("0123456789abcdef")


class MMPPoolFreezeError(ValueError):
    """The source run is incomplete, internally inconsistent, or already frozen differently."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _HEX64 for character in digest):
        raise MMPPoolFreezeError(f"{field} must be a full lowercase SHA-256")
    return digest


def require_artifact_path(value: object, *, field: str, directory: bool = False) -> str:
    raw = value if isinstance(value, str) else ""
    candidate = PurePosixPath(raw)
    if (
        not candidate.is_absolute()
        or len(candidate.parts) < 3
        or candidate.parts[1] != "artifacts"
        or ".." in candidate.parts
        or str(candidate) != raw
    ):
        kind = "directory" if directory else "file"
        raise MMPPoolFreezeError(f"{field} must be a normalized {kind} below /artifacts")
    if not directory and raw.endswith("/"):
        raise MMPPoolFreezeError(f"{field} must name a file")
    return raw


def mounted_artifact_path(artifact_path: str, artifact_root: str | Path) -> Path:
    normalized = require_artifact_path(artifact_path, field="artifact_path")
    return Path(artifact_root) / PurePosixPath(normalized).relative_to("/artifacts")


def partition_independent_mining_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only execution/fan-out fields; every scientific mining choice remains identity-bearing."""
    if not isinstance(config, Mapping):
        raise MMPPoolFreezeError("mining config must be an object")
    missing = sorted(_REQUIRED_SCIENTIFIC_CONFIG_FIELDS - set(config))
    if missing:
        raise MMPPoolFreezeError(f"mining config lacks scientific fields: {missing}")
    return {
        key: config[key]
        for key in sorted(config)
        if key not in _EXECUTION_ONLY_CONFIG_FIELDS
    }


def build_compiler_identity(
    source_files_sha256: Mapping[str, str],
    *,
    runtime_versions: Mapping[str, str],
    support_contract: str = ANALOGUE_SUPPORT_CONTRACT,
) -> dict[str, Any]:
    if not source_files_sha256:
        raise MMPPoolFreezeError("compiler identity requires source-file hashes")
    if not runtime_versions:
        raise MMPPoolFreezeError("compiler identity requires runtime versions")
    files = {
        str(path): _require_sha256(digest, field=f"compiler source {path}")
        for path, digest in sorted(source_files_sha256.items())
    }
    runtimes = {
        str(name): str(version).strip()
        for name, version in sorted(runtime_versions.items())
    }
    if any(not name or not version for name, version in runtimes.items()):
        raise MMPPoolFreezeError("compiler runtime names and versions must be explicit")
    body = {
        "schema": COMPILER_IDENTITY_SCHEMA,
        "analogue_support_contract": str(support_contract),
        "source_files_sha256": files,
        "runtime_versions": runtimes,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def validate_compiler_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise MMPPoolFreezeError("compiler identity must be an object")
    files = identity.get("source_files_sha256")
    runtime_versions = identity.get("runtime_versions")
    rebuilt = build_compiler_identity(
        files if isinstance(files, Mapping) else {},
        runtime_versions=(
            runtime_versions if isinstance(runtime_versions, Mapping) else {}
        ),
        support_contract=str(identity.get("analogue_support_contract", "")),
    )
    if dict(identity) != rebuilt:
        raise MMPPoolFreezeError("compiler identity fields or identity SHA-256 disagree")
    if rebuilt["analogue_support_contract"] != ANALOGUE_SUPPORT_CONTRACT:
        raise MMPPoolFreezeError(
            "compiler identity uses a stale analogue support contract: "
            f"{rebuilt['analogue_support_contract']!r}"
        )
    return rebuilt


def expected_pair_indices(total_pairs: int, n_shards: int, shard_index: int) -> tuple[int, ...]:
    if (
        isinstance(total_pairs, bool)
        or not isinstance(total_pairs, int)
        or total_pairs < 0
        or isinstance(n_shards, bool)
        or not isinstance(n_shards, int)
        or n_shards <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < n_shards
    ):
        raise MMPPoolFreezeError("invalid global-pair shard assignment")
    return tuple(range(shard_index, total_pairs, n_shards))


def pair_index_assignment_sha256(indices: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(index) for index in indices).encode()).hexdigest()


def _row_support_contract(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    return metadata.get("support_contract") if isinstance(metadata, Mapping) else None


def _scan_jsonl(
    path: Path,
    *,
    require_support_contract: bool,
    require_objects: bool = True,
) -> tuple[list[Any], str, int]:
    digest = hashlib.sha256()
    rows: list[Any] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise MMPPoolFreezeError(
                    f"{path} line {line_number} is not valid JSON"
                ) from exc
            if require_objects and not isinstance(row, dict):
                raise MMPPoolFreezeError(f"{path} line {line_number} is not a JSON object")
            if require_support_contract:
                if not isinstance(row, dict):
                    raise MMPPoolFreezeError(
                        f"{path} line {line_number} is not a JSON object"
                    )
                observed = _row_support_contract(row)
                diagnostics = row.get("diagnostics")
                diagnostic_contract = (
                    diagnostics.get("support_contract")
                    if isinstance(diagnostics, Mapping)
                    else None
                )
                if (
                    observed != ANALOGUE_SUPPORT_CONTRACT
                    or diagnostic_contract != ANALOGUE_SUPPORT_CONTRACT
                ):
                    raise MMPPoolFreezeError(
                        f"{path} line {line_number} has mixed or missing support contracts"
                    )
            rows.append(row)
    return rows, digest.hexdigest(), len(rows)


def build_compile_shard_manifest(
    *,
    shard_path: str | Path,
    shard_artifact_path: str,
    shard_index: int,
    n_shards: int,
    total_pairs: int,
    source_pairs_path: str,
    source_pairs_sha256: str,
    mining_config_sha256: str,
    compiler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(shard_path)
    rows, content_sha256, record_count = _scan_jsonl(
        path,
        require_support_contract=True,
    )
    assignment = expected_pair_indices(total_pairs, n_shards, shard_index)
    allowed = set(assignment)
    seen_keys: set[tuple[int, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        pair_index = row.get("pair_index")
        direction = row.get("direction")
        if (
            isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index not in allowed
        ):
            raise MMPPoolFreezeError(
                f"compile shard {shard_index} row {row_number} has an out-of-assignment pair_index"
            )
        if direction not in ("forward", "reverse"):
            raise MMPPoolFreezeError(
                f"compile shard {shard_index} row {row_number} has invalid direction {direction!r}"
            )
        key = (pair_index, str(direction))
        if key in seen_keys:
            raise MMPPoolFreezeError(
                f"compile shard {shard_index} repeats pair/direction {key}"
            )
        seen_keys.add(key)
    compiler = validate_compiler_identity(compiler_identity)
    return {
        "schema": COMPILE_SHARD_SCHEMA,
        "COMPILE_SHARD_COMPLETE": True,
        "shard_index": shard_index,
        "n_shards": n_shards,
        "shard_path": require_artifact_path(shard_artifact_path, field="shard_path"),
        "content_sha256": content_sha256,
        "records": record_count,
        "assigned_pairs": {
            "rule": "global_pair_index_mod_n_shards",
            "total_pairs": total_pairs,
            "count": len(assignment),
            "indices_sha256": pair_index_assignment_sha256(assignment),
        },
        "source_pairs": {
            "path": require_artifact_path(source_pairs_path, field="source_pairs.path"),
            "sha256": _require_sha256(source_pairs_sha256, field="source_pairs.sha256"),
            "records": total_pairs,
        },
        "mining_config_sha256": _require_sha256(
            mining_config_sha256,
            field="mining_config_sha256",
        ),
        "compiler_identity_sha256": compiler["identity_sha256"],
        "analogue_support_contract": ANALOGUE_SUPPORT_CONTRACT,
    }


def compile_manifest_artifact_path(shard_artifact_path: str) -> str:
    path = PurePosixPath(
        require_artifact_path(shard_artifact_path, field="shard_artifact_path")
    )
    return str(path.with_name(f"{path.name}.manifest.json"))


def compile_manifest_file_path(shard_path: str | Path) -> Path:
    """Return the one local sidecar convention for a compiled JSONL shard."""
    path = Path(shard_path)
    return path.with_name(f"{path.name}.manifest.json")


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(manifest), indent=2, sort_keys=True).encode() + b"\n"


def write_bytes_if_absent(path: str | Path, content: bytes) -> bool:
    """Atomically create one immutable file; identical reuse is allowed, overwrite is not."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
            return True
        except FileExistsError:
            if destination.read_bytes() != content:
                raise MMPPoolFreezeError(
                    f"immutable artifact already exists with different bytes: {destination}"
                ) from None
            return False
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


@dataclass(frozen=True)
class CompileInventory:
    rows: tuple[dict[str, Any], ...]
    shards: tuple[dict[str, Any], ...]
    records_sha256: str

    @property
    def record_count(self) -> int:
        return len(self.rows)


def _compiled_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    pair_index = row.get("pair_index")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
        raise MMPPoolFreezeError("every compiled record requires a nonnegative global pair_index")
    direction_rank = {"forward": 0, "A->B": 0, "reverse": 1, "B->A": 1}
    return (
        pair_index,
        direction_rank.get(str(row.get("direction")), 2),
        str(row.get("direction")),
        str(row.get("source_smiles")),
        str(row.get("target_key")),
    )


def deterministic_dedup_and_cap(
    records: Sequence[Mapping[str, Any]],
    *,
    max_pairs_per_source: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition-independent global ordering followed by the scientific directed-pair cap."""
    if (
        isinstance(max_pairs_per_source, bool)
        or not isinstance(max_pairs_per_source, int)
        or max_pairs_per_source <= 0
    ):
        raise MMPPoolFreezeError("max_pairs_per_source must be a positive integer")
    ordered = sorted(records, key=_compiled_row_sort_key)
    seen: set[tuple[str, str]] = set()
    per_source: Counter[str] = Counter()
    final: list[dict[str, Any]] = []
    dropped_duplicate = 0
    dropped_cap = 0
    family_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    for raw_record in ordered:
        record = dict(raw_record)
        source = str(record.get("source_smiles", ""))
        target = str(record.get("target_key", ""))
        key = (source, target)
        if key in seen:
            dropped_duplicate += 1
            continue
        if per_source[source] >= max_pairs_per_source:
            dropped_cap += 1
            continue
        seen.add(key)
        per_source[source] += 1
        final.append(record)
        histogram = record.get("operator_histogram")
        if not isinstance(histogram, Mapping):
            raise MMPPoolFreezeError("compiled record lacks operator_histogram")
        for family, count in histogram.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise MMPPoolFreezeError("operator_histogram counts must be nonnegative integers")
            family_counts[str(family)] += count
        direction_counts[str(record.get("direction", ""))] += 1
        layer_counts[str(record.get("layer", ""))] += 1
    return final, {
        "records_in": len(ordered),
        "records_out": len(final),
        "dropped_duplicate": dropped_duplicate,
        "dropped_per_source_cap": dropped_cap,
        "family_histogram": dict(family_counts),
        "direction_balance": dict(direction_counts),
        "layer_counts": dict(layer_counts),
    }


def compiled_records_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=_compiled_row_sort_key):
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_compile_inventory(
    *,
    artifact_root: str | Path,
    source_run_root: str,
    expected_shards: int,
    total_pairs: int,
    source_pairs_path: str,
    source_pairs_sha256: str,
    mining_config_sha256: str,
    compiler_identity: Mapping[str, Any],
) -> CompileInventory:
    """Verify exact shard set, immutable manifests and mathematical pair-index coverage."""
    if (
        isinstance(expected_shards, bool)
        or not isinstance(expected_shards, int)
        or expected_shards <= 0
    ):
        raise MMPPoolFreezeError("expected_shards must be a positive integer")
    if (
        isinstance(total_pairs, bool)
        or not isinstance(total_pairs, int)
        or total_pairs < 0
    ):
        raise MMPPoolFreezeError("total_pairs must be a nonnegative integer")
    run_artifact = require_artifact_path(
        source_run_root,
        field="source_run_root",
        directory=True,
    )
    run_path = mounted_artifact_path(run_artifact + "/placeholder", artifact_root).parent
    compiled_dir = run_path / "compiled"
    expected_data = {
        f"shard_{index:04d}.jsonl"
        for index in range(expected_shards)
    }
    observed_data = {path.name for path in compiled_dir.glob("shard_*.jsonl")}
    observed_manifests = {
        path.name
        for path in compiled_dir.glob("shard_*.jsonl.manifest.json")
    }
    expected_manifests = {
        compile_manifest_file_path(name).name
        for name in expected_data
    }
    if observed_data != expected_data or observed_manifests != expected_manifests:
        raise MMPPoolFreezeError(
            "compile shard set is partial or unexpected: "
            f"missing_data={sorted(expected_data - observed_data)}, "
            f"unexpected_data={sorted(observed_data - expected_data)}, "
            f"missing_manifests={sorted(expected_manifests - observed_manifests)}, "
            f"unexpected_manifests={sorted(observed_manifests - expected_manifests)}"
        )

    compiler = validate_compiler_identity(compiler_identity)
    all_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    observed_shard_indices: set[int] = set()
    for shard_index in range(expected_shards):
        name = f"shard_{shard_index:04d}.jsonl"
        data_path = compiled_dir / name
        manifest_path = compile_manifest_file_path(compiled_dir / name)
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MMPPoolFreezeError(f"invalid compile manifest {manifest_path}") from exc
        artifact_path = f"{run_artifact}/compiled/{name}"
        expected_manifest = build_compile_shard_manifest(
            shard_path=data_path,
            shard_artifact_path=artifact_path,
            shard_index=shard_index,
            n_shards=expected_shards,
            total_pairs=total_pairs,
            source_pairs_path=source_pairs_path,
            source_pairs_sha256=source_pairs_sha256,
            mining_config_sha256=mining_config_sha256,
            compiler_identity=compiler,
        )
        if manifest != expected_manifest:
            raise MMPPoolFreezeError(
                f"compile manifest {manifest_path} disagrees with its shard or frozen inputs"
            )
        recorded_index = manifest["shard_index"]
        if recorded_index in observed_shard_indices:
            raise MMPPoolFreezeError(f"overlapping compile shard index {recorded_index}")
        observed_shard_indices.add(recorded_index)
        rows, _, _ = _scan_jsonl(data_path, require_support_contract=True)
        all_rows.extend(rows)
        inventory.append({
            **manifest,
            "manifest_path": compile_manifest_artifact_path(artifact_path),
            "manifest_sha256": file_sha256(manifest_path),
        })

    if observed_shard_indices != set(range(expected_shards)):
        raise MMPPoolFreezeError("compile shard assignment has gaps or overlap")
    return CompileInventory(
        rows=tuple(sorted(all_rows, key=_compiled_row_sort_key)),
        shards=tuple(inventory),
        records_sha256=compiled_records_sha256(all_rows),
    )


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MMPPoolFreezeError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MMPPoolFreezeError(f"{name} must be a JSON object")
    return payload


def _source_input_identity(source_inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source_inputs, Mapping):
        raise MMPPoolFreezeError("source_inputs must be an object")
    corpus = source_inputs.get("corpus")
    pairs = source_inputs.get("pairs")
    mine_meta = source_inputs.get("mine_meta")
    if not all(isinstance(value, Mapping) for value in (corpus, pairs, mine_meta)):
        raise MMPPoolFreezeError("source_inputs must contain corpus, pairs and mine_meta objects")
    pair_records = pairs.get("records")
    if (
        isinstance(pair_records, bool)
        or not isinstance(pair_records, int)
        or pair_records < 0
    ):
        raise MMPPoolFreezeError("pairs.records must be a nonnegative integer")
    normalized = {
        "corpus": {
            "id": str(corpus.get("id", "")),
            "path": str(corpus.get("path", "")),
            "sha256": _require_sha256(corpus.get("sha256"), field="corpus.sha256"),
            "scope_name": str(corpus.get("scope_name", "")),
            "scope_hash": str(corpus.get("scope_hash", "")),
        },
        "pairs": {
            "path": require_artifact_path(pairs.get("path"), field="pairs.path"),
            "sha256": _require_sha256(pairs.get("sha256"), field="pairs.sha256"),
            "records": pair_records,
        },
        "mine_meta": {
            "path": require_artifact_path(mine_meta.get("path"), field="mine_meta.path"),
            "sha256": _require_sha256(mine_meta.get("sha256"), field="mine_meta.sha256"),
        },
    }
    for field in ("id", "path", "scope_name", "scope_hash"):
        if not normalized["corpus"][field]:
            raise MMPPoolFreezeError(f"corpus.{field} must be explicit")
    return normalized


def freeze_reduced_mmp_pool(
    reduction_summary_path: str | Path,
    *,
    artifact_root: str | Path = "/artifacts",
    output_parent_artifact_path: str = "/artifacts/mmp_pool_frozen_v2",
) -> dict[str, Any]:
    """Validate a completed mining run and immutably publish its packer-compatible V2 pool."""
    summary_path = Path(reduction_summary_path)
    summary = _load_json_object(summary_path, name="reduction summary")
    if summary.get("schema") != REDUCTION_SCHEMA or summary.get("REDUCTION_COMPLETE") is not True:
        raise MMPPoolFreezeError("reduction summary is partial or has the wrong V2 schema")

    config = summary.get("config")
    scientific_config = partition_independent_mining_config(
        config if isinstance(config, Mapping) else {}
    )
    if summary.get("partition_independent_mining_config") != scientific_config:
        raise MMPPoolFreezeError("partition-independent mining config disagrees with full config")
    config_sha256 = canonical_sha256(scientific_config)
    if summary.get("mining_config_sha256") != config_sha256:
        raise MMPPoolFreezeError("mining config SHA-256 disagrees")

    compiler = validate_compiler_identity(summary.get("compiler_identity", {}))
    source_inputs = _source_input_identity(summary.get("source_inputs", {}))
    source_run_root = require_artifact_path(
        summary.get("source_run_root"),
        field="source_run_root",
        directory=True,
    )
    expected_summary_artifact = f"{source_run_root}/mining_summary_full.json"
    if summary.get("summary_artifact_path") != expected_summary_artifact:
        raise MMPPoolFreezeError("reduction summary artifact path disagrees with source run root")
    mounted_summary = mounted_artifact_path(expected_summary_artifact, artifact_root)
    if mounted_summary.resolve() != summary_path.resolve():
        raise MMPPoolFreezeError("reduction summary was not read from its declared artifact path")
    if source_inputs["pairs"]["path"] != f"{source_run_root}/pairs.jsonl":
        raise MMPPoolFreezeError("pairs path must belong to the declared source run")
    if source_inputs["mine_meta"]["path"] != f"{source_run_root}/mine_meta.json":
        raise MMPPoolFreezeError("mine_meta path must belong to the declared source run")

    pairs_path = mounted_artifact_path(source_inputs["pairs"]["path"], artifact_root)
    pair_rows, pairs_sha256, pair_count = _scan_jsonl(
        pairs_path,
        require_support_contract=False,
        require_objects=False,
    )
    if (
        pairs_sha256 != source_inputs["pairs"]["sha256"]
        or pair_count != source_inputs["pairs"]["records"]
    ):
        raise MMPPoolFreezeError("source pairs bytes/count disagree with reduction summary")
    for index, pair in enumerate(pair_rows):
        if not isinstance(pair, list) or len(pair) != 3:
            raise MMPPoolFreezeError(f"source pair row {index} must be [source,target,core]")

    mine_meta_path = mounted_artifact_path(source_inputs["mine_meta"]["path"], artifact_root)
    if file_sha256(mine_meta_path) != source_inputs["mine_meta"]["sha256"]:
        raise MMPPoolFreezeError("mine_meta bytes disagree with reduction summary")
    mine_meta = _load_json_object(mine_meta_path, name="mine_meta")
    corpus_from_meta = mine_meta.get("provenance", {})
    if not isinstance(corpus_from_meta, Mapping):
        raise MMPPoolFreezeError("mine_meta provenance must be an object")
    if (
        mine_meta.get("config") != config
        or mine_meta.get("partition_independent_mining_config") != scientific_config
        or mine_meta.get("mining_config_sha256") != config_sha256
        or mine_meta.get("pairs") != source_inputs["pairs"]
        or corpus_from_meta.get("compiler_identity_v2") != compiler
        or source_inputs["corpus"] != {
            "id": str(corpus_from_meta.get("corpus_id", "")),
            "path": str(config.get("corpus_path", "")),
            "sha256": str(corpus_from_meta.get("corpus_sha256", "")),
            "scope_name": str(corpus_from_meta.get("corpus_scope", "")),
            "scope_hash": str(corpus_from_meta.get("corpus_scope_hash", "")),
        }
    ):
        raise MMPPoolFreezeError("mine_meta provenance disagrees with reduction source inputs")

    compile_summary = summary.get("compile")
    if not isinstance(compile_summary, Mapping):
        raise MMPPoolFreezeError("reduction summary lacks compile evidence")
    expected_shards = compile_summary.get("expected_shards")
    if (
        isinstance(expected_shards, bool)
        or not isinstance(expected_shards, int)
        or expected_shards <= 0
    ):
        raise MMPPoolFreezeError("compile.expected_shards must be a positive integer")
    inventory = validate_compile_inventory(
        artifact_root=artifact_root,
        source_run_root=source_run_root,
        expected_shards=expected_shards,
        total_pairs=pair_count,
        source_pairs_path=source_inputs["pairs"]["path"],
        source_pairs_sha256=pairs_sha256,
        mining_config_sha256=config_sha256,
        compiler_identity=compiler,
    )
    if (
        compile_summary.get("records") != inventory.record_count
        or compile_summary.get("records_sha256") != inventory.records_sha256
        or compile_summary.get("shards") != list(inventory.shards)
        or compile_summary.get("coverage") != {
            "rule": "global_pair_index_mod_n_shards",
            "total_pairs": pair_count,
            "unique_assigned_pairs": pair_count,
            "missing_pairs": 0,
            "overlapping_pairs": 0,
        }
    ):
        raise MMPPoolFreezeError("compile evidence is partial, overlapping or changed")

    pool_summary = summary.get("pool")
    if not isinstance(pool_summary, Mapping):
        raise MMPPoolFreezeError("reduction summary lacks pool evidence")
    source_pool_artifact = require_artifact_path(pool_summary.get("path"), field="pool.path")
    if source_pool_artifact != f"{source_run_root}/edit_pool_full.jsonl":
        raise MMPPoolFreezeError("reduced pool path must belong to the declared source run")
    source_pool_path = mounted_artifact_path(source_pool_artifact, artifact_root)
    pool_rows, pool_sha256, pool_count = _scan_jsonl(
        source_pool_path,
        require_support_contract=True,
    )
    max_pairs_per_source = scientific_config.get("max_pairs_per_source")
    expected_pool, expected_reduce = deterministic_dedup_and_cap(
        inventory.rows,
        max_pairs_per_source=max_pairs_per_source,
    )
    if (
        pool_summary.get("sha256") != pool_sha256
        or pool_summary.get("records") != pool_count
        or pool_rows != expected_pool
        or summary.get("global_reduce") != expected_reduce
    ):
        raise MMPPoolFreezeError("reduced pool bytes/content/cap report do not reconcile")
    if pool_count <= 0:
        raise MMPPoolFreezeError("refusing to freeze an empty MMP pool")

    identity_payload = {
        "schema": FREEZER_IDENTITY_SCHEMA,
        "source_run_root": source_run_root,
        "pool_sha256": pool_sha256,
        "pool_records": pool_count,
        "analogue_support_contract": ANALOGUE_SUPPORT_CONTRACT,
        "compiler_identity_sha256": compiler["identity_sha256"],
        "source_inputs": source_inputs,
        "partition_independent_mining_config": scientific_config,
        "mining_config_sha256": config_sha256,
        "compiled_records_sha256": inventory.records_sha256,
        "global_reduce": summary["global_reduce"],
    }
    freeze_identity = canonical_sha256(identity_payload)
    output_parent = require_artifact_path(
        output_parent_artifact_path,
        field="output_parent_artifact_path",
        directory=True,
    )
    if not PurePosixPath(output_parent).name.startswith("mmp_pool_frozen_v2"):
        raise MMPPoolFreezeError("frozen output parent must be a distinct mmp_pool_frozen_v2 namespace")
    namespace = f"{FROZEN_NAMESPACE_PREFIX}{freeze_identity[:20]}"
    frozen_root_artifact = f"{output_parent}/{namespace}"
    frozen_pool_artifact = f"{frozen_root_artifact}/edit_pool_full.jsonl"
    frozen_contract_artifact = f"{frozen_root_artifact}/MMP_POOL_COMPLETE.json"
    frozen_pool_path = mounted_artifact_path(frozen_pool_artifact, artifact_root)
    frozen_contract_path = mounted_artifact_path(frozen_contract_artifact, artifact_root)
    frozen_root_path = frozen_pool_path.parent
    if frozen_root_path.exists():
        unexpected = {
            path.name
            for path in frozen_root_path.iterdir()
            if path.name not in {"edit_pool_full.jsonl", "MMP_POOL_COMPLETE.json"}
        }
        if unexpected:
            raise MMPPoolFreezeError(
                f"frozen namespace contains unexpected artifacts: {sorted(unexpected)}"
            )

    contract = {
        "schema": POOL_CONTRACT_SCHEMA,
        "MMP_POOL_COMPLETE": True,
        "pool_path": frozen_pool_artifact,
        "pool_sha256": pool_sha256,
        "pool_records": pool_count,
        "analogue_support_contract": ANALOGUE_SUPPORT_CONTRACT,
        "freeze_identity": freeze_identity,
        "compiler_identity": compiler,
        "source_inputs": source_inputs,
        "partition_independent_mining_config": scientific_config,
        "mining_config_sha256": config_sha256,
        "reduction": {
            "source_run_root": source_run_root,
            "summary_path": str(summary.get("summary_artifact_path", "")),
            "compiled_records": inventory.record_count,
            "compiled_records_sha256": inventory.records_sha256,
            "expected_compile_shards": expected_shards,
            "global_reduce": summary["global_reduce"],
        },
    }
    pool_bytes = source_pool_path.read_bytes()
    if hashlib.sha256(pool_bytes).hexdigest() != pool_sha256:
        raise MMPPoolFreezeError("source pool changed during freeze")
    pool_created = write_bytes_if_absent(frozen_pool_path, pool_bytes)
    try:
        write_bytes_if_absent(frozen_contract_path, _manifest_bytes(contract))
    except Exception:
        if pool_created:
            frozen_pool_path.unlink(missing_ok=True)
        raise
    return {
        "freeze_identity": freeze_identity,
        "namespace": namespace,
        "pool_path": frozen_pool_artifact,
        "pool_sha256": pool_sha256,
        "pool_records": pool_count,
        "contract_path": frozen_contract_artifact,
        "contract_sha256": file_sha256(frozen_contract_path),
        "contract": contract,
    }


__all__ = [
    "COMPILE_SHARD_SCHEMA",
    "COMPILER_IDENTITY_SCHEMA",
    "CompileInventory",
    "FREEZER_IDENTITY_SCHEMA",
    "FROZEN_NAMESPACE_PREFIX",
    "MMPPoolFreezeError",
    "POOL_CONTRACT_SCHEMA",
    "REDUCTION_SCHEMA",
    "build_compile_shard_manifest",
    "build_compiler_identity",
    "canonical_json_bytes",
    "canonical_sha256",
    "compile_manifest_artifact_path",
    "compile_manifest_file_path",
    "compiled_records_sha256",
    "deterministic_dedup_and_cap",
    "expected_pair_indices",
    "file_sha256",
    "freeze_reduced_mmp_pool",
    "mounted_artifact_path",
    "pair_index_assignment_sha256",
    "partition_independent_mining_config",
    "require_artifact_path",
    "validate_compile_inventory",
    "validate_compiler_identity",
    "write_bytes_if_absent",
]
