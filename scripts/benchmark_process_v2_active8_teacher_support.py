#!/usr/bin/env python
"""Benchmark Process-V2 Active8 teacher support at the frozen CPU shape.

This is a non-authorizing performance and equivalence diagnostic.  It compares
the fast batched teacher-support checker with the slow production canonical
successor quotient on a deterministic panel selected from the fixed Jin et al.
lead benchmark.  Panel construction is outside every timed region.

Each timed configuration runs in a fresh child interpreter.  Consequently its
``ru_maxrss`` is attributable to one model, one panel, and one checker rather
than to a previous configuration's high-water mark.  The output grants no
Gate-0, T1, P50, checkpoint-selection, or training authority.

Usage::

    PYTHONPATH=src python scripts/benchmark_process_v2_active8_teacher_support.py \
        --panel-size 64 \
        --report diagnostics/process_v2_active8_teacher_support_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from compose_v4.chem.molecular_graph import (  # noqa: E402
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (  # noqa: E402
    persistent_slot_state_sha256,
)
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES  # noqa: E402
from compose_v4.data.editing_v2_process_v2_active8_admission import (  # noqa: E402
    ProcessV2SemanticExactCandidateAudit,
    ProcessV2TeacherSupportEvidence,
    ProductionProcessV2BatchedTeacherSupportChecker,
    ProductionProcessV2SemanticExactCandidateChecker,
    build_process_v2_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (  # noqa: E402
    model_runtime_descriptor,
)
from compose_v4.data.editing_v2_process_v2_schema import (  # noqa: E402
    authority_false_block,
    canonical_sha256,
)
from compose_v4.data.packed_trace_store import (  # noqa: E402
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.experiments.editing_gate_zero_semantic_contract import (  # noqa: E402
    load_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (  # noqa: E402
    SemanticScratchModelConfig,
    SemanticScratchRuntime,
    build_semantic_scratch_runtime,
)
from compose_v4.experiments.production_successor_kernel import (  # noqa: E402
    ScoredRewriteMark,
    canonical_successor_result,
)
from compose_v4.rewrite.action_codec_v4 import (  # noqa: E402
    decode_action,
    encode_action,
)
from compose_v4.rewrite.kernel import (  # noqa: E402
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace  # noqa: E402

SCHEMA = "compose.diagnostics.process_v2_active8_teacher_support_benchmark"
SCHEMA_VERSION = 1
STATUS = "COMPUTED_LOCAL_DIAGNOSTIC_NO_DOWNSTREAM_AUTHORITY"
PANEL_SCHEMA = "compose.diagnostics.process_v2_active8_teacher_support_panel"
PANEL_SCHEMA_VERSION = 1

JIN_CSV = Path("configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv")
SEMANTIC_CONTRACT = Path("configs/editing_gate_zero_semantic_model_process_v2.json")
DEFAULT_REPORT = Path(
    "diagnostics/process_v2_active8_teacher_support_benchmark.json"
)
SMILES_COLUMN = "canonical_nonisomeric_smiles"

INITIALIZATION_SEED = 20260730
MAX_SLOTS = 40
HIDDEN_DIM = 256
MESSAGE_PASSING_STEPS = 6
MARK_DIM = 32
DTYPE = "torch.float32"
ATOM_VOCABULARY_CLASS_COUNT = 15
CATALOG_FINGERPRINT = "639ff6078c32d43c"
CANDIDATE_TIME = 0.5
TORCH_THREADS = 1
BATCH_SIZES = (8, 16, 32, 64)
ALLOWED_PANEL_SIZES = (16, 32, 64)

_MATERIAL_PATHS = (
    Path("scripts/benchmark_process_v2_active8_teacher_support.py"),
    JIN_CSV,
    SEMANTIC_CONTRACT,
    Path("src/compose_v4/data/editing_v2_process_v2_active8_admission.py"),
    Path("src/compose_v4/data/editing_v2_process_v2_active8_plan.py"),
    Path("src/compose_v4/experiments/editing_v2_semantic_runtime.py"),
    Path("src/compose_v4/experiments/production_successor_kernel.py"),
    Path("src/compose_v4/model/factorized_tracelet_rate_model.py"),
    Path("src/compose_v4/rewrite/action_codec_v4.py"),
    Path("src/compose_v4/rewrite/kernel.py"),
)


class TeacherSupportBenchmarkError(RuntimeError):
    """The benchmark could not establish its declared panel or parity."""


@dataclass(frozen=True, slots=True)
class PanelEntry:
    """One exact one-step teacher selected outside the timed region."""

    panel_index: int
    lead_index: int
    source_line_number: int
    source_smiles: str
    model_family: str
    executor_rule: str
    action_record: dict[str, Any]
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    source_canonical_key: str
    target_canonical_key: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "PanelEntry":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            raise TeacherSupportBenchmarkError(
                "teacher-support panel entry fields disagree"
            )
        return cls(**dict(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1e6


def _write_json_atomically(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def build_frozen_runtime(*, repo_root: Path = REPO_ROOT) -> SemanticScratchRuntime:
    """Construct the exact frozen 40-slot CPU model used by Gate 0."""

    torch.set_num_threads(TORCH_THREADS)
    contract = load_gate_zero_semantic_contract(repo_root / SEMANTIC_CONTRACT)
    config = SemanticScratchModelConfig(
        initialization_seed=INITIALIZATION_SEED,
        max_atoms=MAX_SLOTS,
        hidden_dim=HIDDEN_DIM,
        message_passing_steps=MESSAGE_PASSING_STEPS,
        mark_dim=MARK_DIM,
        dtype=DTYPE,
        atom_vocabulary_class_count=ATOM_VOCABULARY_CLASS_COUNT,
        catalog_fingerprint=CATALOG_FINGERPRINT,
    )
    runtime = build_semantic_scratch_runtime(config, contract)
    observed = model_runtime_descriptor(runtime)
    exact = {
        "atom_vocabulary_class_count": ATOM_VOCABULARY_CLASS_COUNT,
        "catalog_fingerprint": CATALOG_FINGERPRINT,
        "dtype": DTYPE,
        "hidden_dim": HIDDEN_DIM,
        "initialization_seed": INITIALIZATION_SEED,
        "mark_dim": MARK_DIM,
        "max_atoms": MAX_SLOTS,
        "message_passing_steps": MESSAGE_PASSING_STEPS,
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in exact.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise TeacherSupportBenchmarkError(
            f"constructed runtime differs from the frozen benchmark shape: {mismatches}"
        )
    return runtime


def _source_state(smiles: str):
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), MAX_SLOTS)
    active = int(is_element(state.atom_types).sum())
    if active > MAX_SLOTS or int(state.n_atoms) != MAX_SLOTS:
        raise TeacherSupportBenchmarkError(
            f"benchmark source violates the {MAX_SLOTS}-slot bound"
        )
    return state


def _entry_from_mark(
    *,
    panel_index: int,
    row: Mapping[str, str],
    state: Any,
    mark: ScoredRewriteMark,
) -> PanelEntry | None:
    system = editing_v2_semantic_rewrite_system()
    successor = system.apply(state, mark.executor_rule_name, mark.action)
    source_key = canonical_state_key(state)
    target_key = canonical_state_key(successor)
    if target_key == source_key:
        return None
    record = encode_action(mark.executor_rule_name, mark.action)
    rule, action = decode_action(record)
    if rule != mark.executor_rule_name or action != mark.action:
        raise TeacherSupportBenchmarkError("ActionV4 changed a selected Jin teacher")
    replay = system.apply(state, rule, action)
    target_sha256 = persistent_slot_state_sha256(successor)
    if persistent_slot_state_sha256(replay) != target_sha256:
        raise TeacherSupportBenchmarkError(
            "ActionV4 replay changed a selected Jin successor"
        )
    if str(record["model_family"]) != mark.family_name:
        raise TeacherSupportBenchmarkError(
            "ActionV4 family differs from the production marked law"
        )
    return PanelEntry(
        panel_index=panel_index,
        lead_index=int(row["lead_index"]),
        source_line_number=int(row["source_line_number"]),
        source_smiles=str(row[SMILES_COLUMN]),
        model_family=str(mark.family_name),
        executor_rule=str(rule),
        action_record=dict(record),
        action_sha256=canonical_sha256(record),
        source_state_sha256=persistent_slot_state_sha256(state),
        target_state_sha256=target_sha256,
        source_canonical_key=source_key,
        target_canonical_key=target_key,
    )


def build_jin_panel(
    runtime: SemanticScratchRuntime,
    *,
    csv_path: Path,
    panel_size: int,
) -> tuple[tuple[PanelEntry, ...], dict[str, Any]]:
    """Select a balanced panel from exact production marked laws.

    At most one teacher per family is taken from one lead.  This prevents one
    unusually alias-rich state from supplying an entire family while retaining
    deterministic source and mark order.  The slow quotient used for selection
    is deliberately outside every timed child-process measurement.
    """

    if panel_size not in ALLOWED_PANEL_SIZES:
        raise ValueError(f"panel_size must be one of {ALLOWED_PANEL_SIZES}")
    per_family = panel_size // len(ACTIVE8_FAMILIES)
    if per_family < 2 or per_family * len(ACTIVE8_FAMILIES) != panel_size:
        raise ValueError("panel_size must provide at least two entries per family")

    selected: dict[str, list[PanelEntry]] = {
        family: [] for family in ACTIVE8_FAMILIES
    }
    exclusions: Counter[str] = Counter()
    scanned_rows = 0
    quotient_rows = 0
    system = editing_v2_semantic_rewrite_system()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scanned_rows += 1
            try:
                molecule = smiles_to_molecular_graph(str(row[SMILES_COLUMN]))
            except Exception:
                exclusions["parse_failure"] += 1
                continue
            active = int(is_element(molecule.atom_types).sum())
            if active > MAX_SLOTS:
                exclusions["above_40_active_atoms"] += 1
                continue
            state = pad_molecular_graph(molecule, MAX_SLOTS)
            result = canonical_successor_result(
                runtime.model,
                state,
                CANDIDATE_TIME,
                system=system,
            )
            quotient_rows += 1
            first_productive: dict[str, ScoredRewriteMark] = {}
            for mark in result.marked_law.marks:
                if (
                    mark.family_name not in selected
                    or len(selected[mark.family_name]) >= per_family
                    or mark.family_name in first_productive
                ):
                    continue
                successor = system.apply(state, mark.executor_rule_name, mark.action)
                if canonical_state_key(successor) != result.marked_law.source_key:
                    first_productive[mark.family_name] = mark
            for family in ACTIVE8_FAMILIES:
                mark = first_productive.get(family)
                if mark is None:
                    continue
                entry = _entry_from_mark(
                    panel_index=sum(len(items) for items in selected.values()),
                    row=row,
                    state=state,
                    mark=mark,
                )
                if entry is not None:
                    selected[family].append(entry)
            if all(len(items) == per_family for items in selected.values()):
                break

    missing = {
        family: per_family - len(items)
        for family, items in selected.items()
        if len(items) != per_family
    }
    if missing:
        raise TeacherSupportBenchmarkError(
            "the Jin benchmark cannot supply the balanced teacher panel without "
            f"fabrication: missing={missing}, scanned_rows={scanned_rows}"
        )
    # Family-major ordering makes the balance explicit, then a fixed reverse
    # permutation prevents benchmark success from depending on source order.
    entries = tuple(
        entry
        for family in ACTIVE8_FAMILIES
        for entry in selected[family]
    )[::-1]
    entries = tuple(
        PanelEntry(**{**entry.as_payload(), "panel_index": index})
        for index, entry in enumerate(entries)
    )
    return entries, {
        "selection": "first_productive_mark_per_family_per_jin_lead_then_reverse",
        "panel_size": len(entries),
        "per_family": per_family,
        "family_counts": dict(sorted(Counter(e.model_family for e in entries).items())),
        "unique_source_states": len({e.source_state_sha256 for e in entries}),
        "scanned_csv_rows": scanned_rows,
        "quotient_rows_constructed_outside_timing": quotient_rows,
        "exclusions": dict(sorted(exclusions.items())),
        "active_atom_limitation": None,
    }


def panel_payload(
    entries: Sequence[PanelEntry],
    *,
    csv_sha256: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": PANEL_SCHEMA,
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": "FROZEN_FOR_ONE_LOCAL_BENCHMARK_NO_DOWNSTREAM_AUTHORITY",
        **authority_false_block(),
        "source_csv_sha256": csv_sha256,
        "selection": dict(selection),
        "entries": [entry.as_payload() for entry in entries],
    }
    return {**body, "panel_sha256": canonical_sha256(body)}


def load_panel(path: Path) -> tuple[PanelEntry, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    observed = body.pop("panel_sha256", None)
    if (
        value.get("schema") != PANEL_SCHEMA
        or value.get("schema_version") != PANEL_SCHEMA_VERSION
        or observed != canonical_sha256(body)
        or any(value.get(field) is not False for field in authority_false_block())
    ):
        raise TeacherSupportBenchmarkError("benchmark panel is stale or authorizing")
    entries = tuple(PanelEntry.from_payload(item) for item in value["entries"])
    if tuple(entry.panel_index for entry in entries) != tuple(range(len(entries))):
        raise TeacherSupportBenchmarkError("benchmark panel ordering is not contiguous")
    return entries


def query_from_entry(entry: PanelEntry) -> tuple[AddressedPackedTrace, int]:
    state = _source_state(entry.source_smiles)
    if (
        persistent_slot_state_sha256(state) != entry.source_state_sha256
        or canonical_state_key(state) != entry.source_canonical_key
    ):
        raise TeacherSupportBenchmarkError("Jin panel source no longer reconstructs exactly")
    rule, action = decode_action(entry.action_record)
    if rule != entry.executor_rule or canonical_sha256(entry.action_record) != entry.action_sha256:
        raise TeacherSupportBenchmarkError("Jin panel action identity changed")
    successor = editing_v2_semantic_rewrite_system().apply(state, rule, action)
    if (
        persistent_slot_state_sha256(successor) != entry.target_state_sha256
        or canonical_state_key(successor) != entry.target_canonical_key
    ):
        raise TeacherSupportBenchmarkError("Jin panel successor no longer reconstructs exactly")
    step = RewriteStep(rule, action)
    trace = RewriteTrace(
        source=state,
        target=successor,
        steps=(step,),
        metadata={"benchmark": SCHEMA, "lead_index": entry.lead_index},
    )
    address = PackedTraceAddress(
        packed_shard_content_sha256=hashlib.sha256(b"jin-active8-teacher-panel").hexdigest(),
        packed_shard_name="jin-active8-teacher-panel.json",
        entry_index=entry.panel_index,
        trace_id=f"jin-active8-teacher-{entry.panel_index:04d}",
        layer="teacher_support_performance_diagnostic",
        partition="external_benchmark_not_selection",
        source_key=entry.source_canonical_key,
        target_key=entry.target_canonical_key,
        path_length=1,
    )
    return (
        AddressedPackedTrace(
            address=address,
            trace=trace,
            path=PackedTraceProgress(trace, (state, successor)),
        ),
        0,
    )


def common_evidence(value: Any) -> dict[str, Any]:
    """Project slow and fast evidence onto their exact common contract."""

    if isinstance(value, ProcessV2SemanticExactCandidateAudit):
        evidence = value.evidence
        exact = value.exact_successor_mark_count
    elif isinstance(value, ProcessV2TeacherSupportEvidence):
        evidence = value
        exact = value.exact_successor_mark_count
    else:
        raise TypeError("teacher-support evidence has an unexpected type")
    return {
        "supported": bool(evidence.supported),
        "exclusion_reason": evidence.exclusion_reason,
        "action_sha256": str(evidence.action_sha256),
        "source_state_sha256": str(evidence.source_state_sha256),
        "target_state_sha256": str(evidence.target_state_sha256),
        "source_canonical_key": str(evidence.source_canonical_key),
        "canonical_successor_key": str(evidence.canonical_successor_key),
        "raw_mark_count": int(evidence.raw_mark_count),
        "matching_mark_count": int(evidence.matching_mark_count),
        "exact_successor_mark_count": int(exact),
    }


def measure_worker(
    *,
    method: str,
    batch_size: int | None,
    panel_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    entries = load_panel(panel_path)
    runtime = build_frozen_runtime(repo_root=repo_root)
    queries = tuple(query_from_entry(entry) for entry in entries)
    policy = build_process_v2_semantic_active8_admission_policy()
    if method == "slow_full_quotient":
        checker: Any = ProductionProcessV2SemanticExactCandidateChecker(
            runtime.model,
            policy=policy,
            cache_size=max(len(entries), 1),
            time=CANDIDATE_TIME,
        )
    elif method == "fast_batched_teacher_support":
        if batch_size not in BATCH_SIZES:
            raise TeacherSupportBenchmarkError(f"invalid fast batch size: {batch_size}")
        checker = ProductionProcessV2BatchedTeacherSupportChecker(
            runtime.model,
            policy=policy,
            batch_size=batch_size,
            time=CANDIDATE_TIME,
        )
    else:
        raise TeacherSupportBenchmarkError(f"unknown benchmark method: {method}")

    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    if method == "slow_full_quotient":
        observed = tuple(checker.evaluate(*query) for query in queries)
    else:
        observed = checker.evaluate_many(queries)
    wall_seconds = time.perf_counter() - wall_before
    cpu_seconds = time.process_time() - cpu_before
    evidence = [common_evidence(item) for item in observed]
    if len(evidence) != len(entries) or not all(item["supported"] for item in evidence):
        raise TeacherSupportBenchmarkError(
            "the selected production teachers did not all remain supported"
        )
    return {
        "method": method,
        "batch_size": batch_size,
        "query_count": len(entries),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "queries_per_wall_second": len(entries) / max(wall_seconds, 1e-12),
        "queries_per_cpu_second": len(entries) / max(cpu_seconds, 1e-12),
        "process_peak_rss_mb": _peak_rss_mb(),
        "evidence_sha256": canonical_sha256(evidence),
        "evidence": evidence,
    }


def _run_isolated_worker(
    *,
    method: str,
    batch_size: int | None,
    panel_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-method",
        method,
        "--panel",
        str(panel_path),
    ]
    if batch_size is not None:
        command.extend(("--worker-batch-size", str(batch_size)))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    environment["OMP_NUM_THREADS"] = str(TORCH_THREADS)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise TeacherSupportBenchmarkError(
            f"isolated worker returned invalid JSON; stderr={completed.stderr!r}"
        ) from error


def _git_state(repo_root: Path) -> tuple[str, list[str]]:
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return revision, dirty


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "networkx", "rdkit", "scipy", "torch"):
        module = __import__(package)
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "packages": versions,
    }


def build_report(
    *,
    panel: Mapping[str, Any],
    panel_selection: Mapping[str, Any],
    runtime: SemanticScratchRuntime,
    slow: Mapping[str, Any],
    fast: Sequence[Mapping[str, Any]],
    report_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if any(item["evidence_sha256"] != slow["evidence_sha256"] for item in fast):
        raise TeacherSupportBenchmarkError(
            "fast teacher-support evidence differs from the slow full quotient"
        )
    revision, dirty = _git_state(repo_root)
    try:
        reported_output_path = report_path.relative_to(repo_root).as_posix()
    except ValueError:
        reported_output_path = str(report_path.resolve())
    material = [
        {"path": path.as_posix(), "sha256": _sha256(repo_root / path)}
        for path in _MATERIAL_PATHS
    ]
    fast_rows = []
    for item in fast:
        row = {key: value for key, value in item.items() if key != "evidence"}
        row["wall_speedup_vs_slow"] = float(slow["wall_seconds"]) / max(
            float(item["wall_seconds"]), 1e-12
        )
        row["cpu_speedup_vs_slow"] = float(slow["cpu_seconds"]) / max(
            float(item["cpu_seconds"]), 1e-12
        )
        fast_rows.append(row)
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evidence_class": "computed_diagnostic",
        **authority_false_block(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "command": (
                "PYTHONPATH=src python "
                "scripts/benchmark_process_v2_active8_teacher_support.py"
            ),
            "code_revision": revision,
            "repository_dirty": bool(dirty),
            "repository_dirty_paths": dirty,
            "material_inputs": material,
            "output_path": reported_output_path,
            "panel_sha256": panel["panel_sha256"],
            "panel_source": {
                "path": JIN_CSV.as_posix(),
                "sha256": panel["source_csv_sha256"],
                "role": "fixed_external_lead_benchmark_not_model_selection",
            },
            "split_identity": "external_benchmark_not_selection_or_threshold_setting",
            "sample_count": int(panel_selection["panel_size"]),
            "exclusions": dict(panel_selection["exclusions"]),
            "seed": INITIALIZATION_SEED,
            "seed_derivation": "frozen_semantic_scratch_runtime",
            "environment": _environment(),
        },
        "configuration": {
            "candidate_time": CANDIDATE_TIME,
            "batch_sizes": list(BATCH_SIZES),
            "torch_threads": TORCH_THREADS,
            "model_runtime": model_runtime_descriptor(runtime),
            "precision": DTYPE,
            "device": str(runtime.model.device),
            "panel_selection_outside_timing": True,
            "fresh_child_process_per_measurement": True,
        },
        "panel": dict(panel_selection),
        "result": {
            "exact_common_evidence_parity": True,
            "evidence_sha256": slow["evidence_sha256"],
            "slow_full_quotient": {
                key: value for key, value in slow.items() if key != "evidence"
            },
            "fast_batched_teacher_support": fast_rows,
        },
        "limitations": [
            "local CPU timing does not measure Modal volume I/O or cold-start latency",
            "panel rows share some source states because balance is enforced per family",
            "peak RSS includes model and panel construction before the timed region",
            "parity covers the exact common evidence contract; quotient-only alias and successor counts remain release-sentinel outputs",
        ],
    }
    return report


def run_benchmark(*, panel_size: int, report_path: Path, repo_root: Path) -> dict[str, Any]:
    csv_path = repo_root / JIN_CSV
    runtime = build_frozen_runtime(repo_root=repo_root)
    entries, selection = build_jin_panel(
        runtime,
        csv_path=csv_path,
        panel_size=panel_size,
    )
    panel = panel_payload(entries, csv_sha256=_sha256(csv_path), selection=selection)
    with tempfile.TemporaryDirectory(prefix="compose-active8-benchmark-") as temporary:
        panel_path = Path(temporary) / "panel.json"
        _write_json_atomically(panel_path, panel)
        slow = _run_isolated_worker(
            method="slow_full_quotient",
            batch_size=None,
            panel_path=panel_path,
            repo_root=repo_root,
        )
        fast = tuple(
            _run_isolated_worker(
                method="fast_batched_teacher_support",
                batch_size=batch_size,
                panel_path=panel_path,
                repo_root=repo_root,
            )
            for batch_size in BATCH_SIZES
        )
    report = build_report(
        panel=panel,
        panel_selection=selection,
        runtime=runtime,
        slow=slow,
        fast=fast,
        report_path=report_path,
        repo_root=repo_root,
    )
    _write_json_atomically(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-size", type=int, choices=ALLOWED_PANEL_SIZES, default=64)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--panel", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-method",
        choices=("slow_full_quotient", "fast_batched_teacher_support"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-batch-size", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_method is not None:
        if args.panel is None:
            raise TeacherSupportBenchmarkError("worker mode requires --panel")
        result = measure_worker(
            method=args.worker_method,
            batch_size=args.worker_batch_size,
            panel_path=args.panel,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report = run_benchmark(
        panel_size=args.panel_size,
        report_path=report_path,
        repo_root=REPO_ROOT,
    )
    fastest = max(
        report["result"]["fast_batched_teacher_support"],
        key=lambda item: item["wall_speedup_vs_slow"],
    )
    print(
        f"parity=PASS queries={report['panel']['panel_size']} "
        f"best_batch={fastest['batch_size']} "
        f"wall_speedup={fastest['wall_speedup_vs_slow']:.2f}x"
    )
    print(f"-> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
