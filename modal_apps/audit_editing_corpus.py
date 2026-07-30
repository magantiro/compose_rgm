"""Exact, model-independent census of the frozen RingCore editing corpus.

The audit reads the packed derivative stores directly.  It never replays a
rewrite, changes a training artifact, or inspects a checkpoint.  Besides raw
trace/step counts it computes the *actual expected coefficient* of every
teacher-family log-probability in the production Generator-Matching loss:

    record sampler probability
    x E_t[Pr(N_t = progress)]
    x operational teacher rate (path_length - progress).

This is different from counting selected examples.  In particular, long paths
carry larger teacher rates and the sequential ordering of their operators can
change effective supervision substantially.

Example:

    MODAL_PROFILE=nitya modal run modal_apps/audit_editing_corpus.py \
      --output-name ringcore-v1-editing-corpus-audit-2026-07-29.json
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import modal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
ARTIFACT_ROOT = Path("/artifacts")
DEFAULT_UNIFIED_MANIFEST = ARTIFACT_ROOT / "UNIFIED_PACKED_MANIFEST.json"
DEFAULT_OVERLAY = ARTIFACT_ROOT / "REPRESENTABILITY_OVERLAY.json"
OUTPUT_DIRECTORY = ARTIFACT_ROOT / "_frozen_corpus_audits"
OUTPUT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.json\Z")

PRODUCTION_LAYER_WEIGHTS = {
    "general_corruption": 0.40,
    "cycle_operations": 0.25,
    "mmp_analogue": 0.35,
}
PATH_LENGTH_BIN_EDGES = (5, 9, 13)
LATE_TIME_FRACTION = 0.5
OPERATIONAL_HORIZON = 16.0
NORMALIZED_TIME_LOW = 0.01
NORMALIZED_TIME_HIGH = 0.99
QUADRATURE_ORDER = 128

LAYER_STORAGE = {
    "general_corruption": ("audit_layers", "corruption"),
    "cycle_operations": ("audit_layers", "cycle_ops"),
    "mmp_analogue": ("mmp_layer", ""),
}
PARTITIONS = ("train", "validation", "test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "rdkit==2024.3.5")
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
)
app = modal.App("compose-v4-audit-editing-corpus")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _bin_index(path_length: int, edges: tuple[int, ...] = PATH_LENGTH_BIN_EDGES) -> int:
    for index, edge in enumerate(edges):
        if path_length <= edge:
            return index
    return len(edges)


def _mapped_interval_nodes(
    low: float,
    high: float,
    *,
    order: int = QUADRATURE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    values = low + (nodes + 1.0) * (high - low) / 2.0
    normalized_weights = weights / 2.0
    return values, normalized_weights


def integrated_progress_probabilities(path_length: int) -> np.ndarray:
    """Average ``Binomial(path_length, t)`` under the production time law."""

    if path_length < 0:
        raise ValueError("path_length must be nonnegative")
    if path_length == 0:
        return np.ones(1, dtype=np.float64)

    normalized_t, normalized_weights = _mapped_interval_nodes(
        NORMALIZED_TIME_LOW,
        NORMALIZED_TIME_HIGH,
    )
    operational_s, operational_weights = _mapped_interval_nodes(
        0.0,
        OPERATIONAL_HORIZON,
    )
    operational_t = 1.0 - np.exp(-operational_s)

    positions = np.arange(path_length + 1, dtype=np.int64)
    coefficients = np.asarray(
        [math.comb(path_length, int(position)) for position in positions],
        dtype=np.float64,
    )

    def average_at(times: np.ndarray, weights: np.ndarray) -> np.ndarray:
        probabilities = (
            coefficients[None, :]
            * times[:, None] ** positions[None, :]
            * (1.0 - times[:, None])
            ** (path_length - positions)[None, :]
        )
        return np.sum(probabilities * weights[:, None], axis=0)

    return (
        (1.0 - LATE_TIME_FRACTION)
        * average_at(normalized_t, normalized_weights)
        + LATE_TIME_FRACTION
        * average_at(operational_t, operational_weights)
    )


def expected_mark_coefficients(
    families: tuple[str, ...],
) -> tuple[dict[str, float], float, float]:
    """Return family teacher-rate coefficients, nonterminal mass, terminal mass."""

    path_length = len(families)
    progress = integrated_progress_probabilities(path_length)
    coefficients: dict[str, float] = defaultdict(float)
    for index, family in enumerate(families):
        coefficients[family] += float(progress[index]) * float(path_length - index)
    return dict(coefficients), float(progress[:-1].sum()), float(progress[-1])


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _trace_key_from_wire(trace: dict) -> str:
    # Production packed loading calls ``trace_key(trace)`` without the raw
    # record argument.  The decoded RewriteTrace intentionally carries no
    # record-level trace_id/row_id, even if the wire record has one; therefore
    # the authoritative key is the content digest below.
    digest = hashlib.sha256()
    for step in trace["steps"]:
        action = step["action"]
        digest.update(str(action["executor_rule"]).encode())
        neighbors = ""
        if action["payload_type"] == "AtomInsert":
            neighbors = tuple(
                tuple(int(value) for value in pair)
                for pair in action["payload"]["neighbors"]
            )
        digest.update(str(neighbors).encode())
    return "sha:" + digest.hexdigest()[:24]


def _active_atom_count(state: dict) -> int:
    from compose_v4.chem.molecular_graph import is_element

    return int(is_element(np.asarray(state["atom_types"], dtype=np.int32)).sum())


def _cycle_rank(state: dict) -> int:
    atom_count = _active_atom_count(state)
    if atom_count == 0:
        return 0
    return int(len(state["bonds"]) - atom_count + 1)


def _iter_entries(paths: Iterable[Path]):
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield path, json.loads(line)


def _wire_family(step: dict) -> str:
    action = step["action"]
    family = str(action["model_family"])
    executor = str(action["executor_rule"])
    expected = {
        "bond_insert": "cycle_insert",
        "bond_delete": "cycle_attach",
    }.get(executor, executor)
    if family != expected:
        raise ValueError(
            f"wire ontology mismatch: executor {executor!r} declares {family!r}, "
            f"expected {expected!r}"
        )
    return family


def _paths_for(
    *,
    manifest: dict,
    layer: str,
    partition: str,
) -> tuple[Path, ...]:
    root_key, subdirectory = LAYER_STORAGE[layer]
    root = Path(manifest["roots"][root_key])
    directory = root / subdirectory / partition if subdirectory else root / partition
    paths = tuple(sorted(directory.glob("*.jsonl.gz")))
    declared = tuple(
        Path(relative).name
        for relative in manifest["layers"][layer][partition]
    )
    if tuple(path.name for path in paths) != declared:
        raise ValueError(
            f"packed shard mismatch for {layer}/{partition}: "
            f"found {[path.name for path in paths]}, declared {list(declared)}"
        )
    return paths


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _pair_counter_dict(counter: Counter) -> dict[str, int]:
    return {
        f"{left}|{right}": int(value)
        for (left, right), value in sorted(counter.items())
    }


def _partition_census(
    *,
    manifest: dict,
    overlay: dict,
    layer: str,
    partition: str,
) -> tuple[dict, Counter]:
    allowed_exclusions = {
        (
            str(item["shard"]),
            str(item["trace_key"]),
        )
        for item in overlay["exclusions"]
        if item["layer"] == layer and item["partition"] == partition
    }
    exclusions_seen: set[tuple[str, str]] = set()
    family_steps: Counter = Counter()
    traces_with_family: Counter = Counter()
    sequence_counts: Counter = Counter()
    adjacency: Counter = Counter()
    cooccurrence: Counter = Counter()
    path_lengths: Counter = Counter()
    bin_records: Counter = Counter()
    direction: Counter = Counter()
    first_family: Counter = Counter()
    last_family: Counter = Counter()
    insert_arity: Counter = Counter()
    cycle_bond_orders: Counter = Counter()
    ring_delete_atom_deletions: Counter = Counter()
    ring_delete_bond_deletions: Counter = Counter()
    ring_delete_topology: Counter = Counter()
    ring_restate_change_counts: Counter = Counter()
    atom_count_delta: Counter = Counter()
    cycle_rank_delta: Counter = Counter()
    joint_state_delta: Counter = Counter()
    records = 0

    for shard, entry in _iter_entries(
        _paths_for(manifest=manifest, layer=layer, partition=partition)
    ):
        trace = entry["trace"]
        trace_key = _trace_key_from_wire(trace)
        exclusion_identity = (shard.name, trace_key)
        if exclusion_identity in allowed_exclusions:
            exclusions_seen.add(exclusion_identity)
            continue

        steps = trace["steps"]
        families = tuple(_wire_family(step) for step in steps)
        if not families:
            raise ValueError(f"zero-length trace in {layer}/{partition}/{shard.name}")
        records += 1
        path_lengths[len(families)] += 1
        bin_records[_bin_index(len(families))] += 1
        sequence_counts[families] += 1
        family_steps.update(families)
        traces_with_family.update(set(families))
        first_family[families[0]] += 1
        last_family[families[-1]] += 1
        adjacency.update(zip(families, families[1:]))
        unique = sorted(set(families))
        cooccurrence.update(
            (left, right)
            for left_index, left in enumerate(unique)
            for right in unique[left_index:]
        )
        direction[str((trace.get("metadata") or {}).get("prior", "<missing>"))] += 1

        source_state = entry["states"][0]
        target_state = entry["states"][-1]
        atom_delta = _active_atom_count(target_state) - _active_atom_count(source_state)
        rank_delta = _cycle_rank(target_state) - _cycle_rank(source_state)
        atom_count_delta[atom_delta] += 1
        cycle_rank_delta[rank_delta] += 1
        joint_state_delta[(atom_delta, rank_delta)] += 1

        for step in steps:
            action = step["action"]
            payload = action["payload"]
            payload_type = action["payload_type"]
            if payload_type == "AtomInsert":
                insert_arity[len(payload["neighbors"])] += 1
            elif payload_type == "BondInsert":
                cycle_bond_orders[f"close:{int(payload['order'])}"] += 1
            elif payload_type == "BondDelete":
                cycle_bond_orders["open"] += 1
            elif payload_type == "RingSystemDelete":
                ring_delete_atom_deletions[len(payload["atom_deletions"])] += 1
                ring_delete_bond_deletions[len(payload["bond_deletions"])] += 1
                ring_delete_topology[str(payload["topology_class"])] += 1
            elif payload_type == "RingSystemRestate":
                ring_restate_change_counts[len(payload["changes"])] += 1

    if exclusions_seen != allowed_exclusions:
        raise ValueError(
            f"overlay mismatch for {layer}/{partition}: "
            f"missing {sorted(allowed_exclusions - exclusions_seen)}"
        )

    sequence_summary = Counter(
        {
            " -> ".join(sequence): count
            for sequence, count in sequence_counts.items()
        }
    )
    top_sequences = [
        {"sequence": sequence, "records": int(count)}
        for sequence, count in sequence_summary.most_common(30)
    ]
    census = {
        "records": records,
        "steps": int(sum(family_steps.values())),
        "overlay_exclusions_applied": len(exclusions_seen),
        "path_length_histogram": _counter_dict(path_lengths),
        "records_by_curriculum_bin": _counter_dict(bin_records),
        "family_step_counts": _counter_dict(family_steps),
        "traces_containing_family": _counter_dict(traces_with_family),
        "first_family": _counter_dict(first_family),
        "last_family": _counter_dict(last_family),
        "adjacent_family_pairs": _pair_counter_dict(adjacency),
        "trace_family_cooccurrence": _pair_counter_dict(cooccurrence),
        "direction_metadata": _counter_dict(direction),
        "top_family_sequences": top_sequences,
        "source_to_target_atom_count_delta": _counter_dict(atom_count_delta),
        "source_to_target_cycle_rank_delta": _counter_dict(cycle_rank_delta),
        "source_to_target_joint_delta": {
            f"atoms:{atoms}|cycle_rank:{rank}": int(count)
            for (atoms, rank), count in sorted(joint_state_delta.items())
        },
        "subtypes": {
            "atom_insert_neighbor_arity": _counter_dict(insert_arity),
            "cycle_bond_orders": _counter_dict(cycle_bond_orders),
            "ring_system_delete_atom_deletion_count": _counter_dict(
                ring_delete_atom_deletions
            ),
            "ring_system_delete_bond_deletion_count": _counter_dict(
                ring_delete_bond_deletions
            ),
            "ring_system_delete_topology_class": _counter_dict(
                ring_delete_topology
            ),
            "ring_system_restate_bond_change_count": _counter_dict(
                ring_restate_change_counts
            ),
        },
    }
    return census, sequence_counts


def _expected_training_law(
    *,
    sequence_counts_by_layer: dict[str, Counter],
    census_by_layer: dict[str, dict],
) -> dict:
    family_landing_mass: Counter = Counter()
    family_rate_coefficient: Counter = Counter()
    layer_landing_mass: Counter = Counter()
    layer_rate_coefficient: Counter = Counter()
    terminal_mass = 0.0
    record_mass = 0.0

    for layer, sequence_counts in sequence_counts_by_layer.items():
        layer_weight = PRODUCTION_LAYER_WEIGHTS[layer]
        bin_counts = {
            int(key): int(value)
            for key, value in census_by_layer[layer][
                "records_by_curriculum_bin"
            ].items()
        }
        nonempty_bins = len(bin_counts)
        for families, multiplicity in sequence_counts.items():
            bin_index = _bin_index(len(families))
            record_probability = (
                layer_weight
                / float(nonempty_bins)
                / float(bin_counts[bin_index])
            )
            total_probability = record_probability * float(multiplicity)
            record_mass += total_probability
            progress = integrated_progress_probabilities(len(families))
            terminal_mass += total_probability * float(progress[-1])
            for index, family in enumerate(families):
                landing = total_probability * float(progress[index])
                rate_coefficient = landing * float(len(families) - index)
                family_landing_mass[family] += landing
                family_rate_coefficient[family] += rate_coefficient
                layer_landing_mass[layer] += landing
                layer_rate_coefficient[layer] += rate_coefficient

    if abs(record_mass - 1.0) > 1e-10:
        raise ValueError(f"hierarchical record probabilities sum to {record_mass}, not one")
    nonterminal_mass = float(sum(family_landing_mass.values()))
    if abs(nonterminal_mass + terminal_mass - 1.0) > 1e-9:
        raise ValueError("integrated progress probabilities do not reconstruct one")
    total_rate = float(sum(family_rate_coefficient.values()))

    return {
        "record_probability_mass": record_mass,
        "terminal_landing_mass": terminal_mass,
        "nonterminal_landing_mass": nonterminal_mass,
        "family_landing_mass_over_all_draws": {
            key: float(value)
            for key, value in sorted(family_landing_mass.items())
        },
        "family_landing_share_given_nonterminal": {
            key: float(value) / nonterminal_mass
            for key, value in sorted(family_landing_mass.items())
        },
        "family_teacher_rate_coefficient": {
            key: float(value)
            for key, value in sorted(family_rate_coefficient.items())
        },
        "family_share_of_mark_log_probability_gradient": {
            key: float(value) / total_rate
            for key, value in sorted(family_rate_coefficient.items())
        },
        "layer_nonterminal_landing_mass": {
            key: float(value)
            for key, value in sorted(layer_landing_mass.items())
        },
        "layer_share_of_mark_log_probability_gradient": {
            key: float(value) / total_rate
            for key, value in sorted(layer_rate_coefficient.items())
        },
        "total_expected_teacher_rate_per_draw": total_rate,
        "interpretation": (
            "Importance correction makes progress stratification unbiased. "
            "The mark-log-probability gradient coefficient is the teacher "
            "rate (path_length-progress), so it is not the same as example "
            "counts or nonterminal landing frequency."
        ),
    }


def build_audit(
    *,
    manifest_path: Path = DEFAULT_UNIFIED_MANIFEST,
    overlay_path: Path = DEFAULT_OVERLAY,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    overlay = json.loads(Path(overlay_path).read_text())
    if manifest.get("manifest_checksum") != "5c5c254e1054c081":
        raise ValueError(
            "refusing to audit an unexpected unified manifest: "
            f"{manifest.get('manifest_checksum')!r}"
        )
    if overlay.get("effective_corpus_checksum") != "32372dc5d73139a7":
        raise ValueError(
            "refusing to audit an unexpected representability overlay: "
            f"{overlay.get('effective_corpus_checksum')!r}"
        )

    partitions: dict[str, dict[str, dict]] = {}
    training_sequences: dict[str, Counter] = {}
    for partition in PARTITIONS:
        partitions[partition] = {}
        for layer in PRODUCTION_LAYER_WEIGHTS:
            census, sequences = _partition_census(
                manifest=manifest,
                overlay=overlay,
                layer=layer,
                partition=partition,
            )
            partitions[partition][layer] = census
            if partition == "train":
                training_sequences[layer] = sequences

    expected_training_law = _expected_training_law(
        sequence_counts_by_layer=training_sequences,
        census_by_layer=partitions["train"],
    )
    result = {
        "schema": "compose.diagnostics.editing_corpus_operator_audit",
        "schema_version": 1,
        "source": {
            "unified_manifest_checksum": manifest["manifest_checksum"],
            "representability_overlay_checksum": overlay[
                "effective_corpus_checksum"
            ],
            "roots": manifest["roots"],
            "layer_weights": PRODUCTION_LAYER_WEIGHTS,
            "path_length_bin_edges": list(PATH_LENGTH_BIN_EDGES),
            "time_law": {
                "late_time_fraction": LATE_TIME_FRACTION,
                "operational_horizon": OPERATIONAL_HORIZON,
                "normalized_uniform_interval": [
                    NORMALIZED_TIME_LOW,
                    NORMALIZED_TIME_HIGH,
                ],
                "quadrature_order": QUADRATURE_ORDER,
            },
        },
        "partitions": partitions,
        "expected_training_law": expected_training_law,
    }
    result["content_sha256"] = _stable_hash(result)
    return result


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    timeout=60 * 30,
    cpu=4.0,
    memory=4096,
)
def audit_remote(output_name: str) -> dict:
    if OUTPUT_NAME_PATTERN.fullmatch(output_name) is None:
        raise ValueError("output_name must be a safe JSON basename")
    result = build_audit()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / output_name
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing != result:
            raise FileExistsError(
                f"{output_path} already exists with different content"
            )
    else:
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        artifact_volume.commit()
    return {
        "output_path": str(output_path),
        "content_sha256": result["content_sha256"],
        "train_records": sum(
            layer["records"] for layer in result["partitions"]["train"].values()
        ),
        "train_steps": sum(
            layer["steps"] for layer in result["partitions"]["train"].values()
        ),
        "family_gradient_shares": result["expected_training_law"][
            "family_share_of_mark_log_probability_gradient"
        ],
    }


@app.local_entrypoint()
def main(
    output_name: str = "ringcore-v1-editing-corpus-audit-2026-07-29.json",
) -> None:
    print(json.dumps(audit_remote.remote(output_name), indent=2, sort_keys=True))
