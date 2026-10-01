"""Build finite-horizon value examples from shared-reference QED rollouts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.hphi_region_features import build_features, in_region, input_dim
from compose_v4.experiments.hphi_rollout import registered_regions
from compose_v4.experiments.qed_shared_rollouts import rollout_seed

LEGACY_H24_CORPUS_SHA256 = "647f8265f5143e2b15dc047de42e837203593c42beb7e45e2b94e55e41602581"


def _expected_seed(rollout: dict, source: Chem.Mol, represented: str, replicate: int) -> int:
    if rollout["schema_version"] == "compose.qed.shared_rollout.v1":
        return rollout_seed(represented, replicate)
    if rollout["schema_version"] != "compose.qed.imported_h24_rollout.v1":
        raise ValueError("unsupported QED rollout schema")
    if rollout.get("source_role") != "train" or rollout.get("import_provenance") != {
        "corpus_sha256": LEGACY_H24_CORPUS_SHA256,
        "seed_scheme": "hphi-corpus-v1",
    }:
        raise ValueError("QED imported rollout has unsupported provenance or role")
    canonical_source = Chem.MolToSmiles(source)
    payload = f"hphi-corpus-v1|{canonical_source}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class ValueExamples:
    features: np.ndarray
    labels: np.ndarray
    region_indices: np.ndarray
    next_row_indices: np.ndarray
    next_terminal_targets: np.ndarray


def value_examples(
    reference,
    rollout: dict,
    *,
    budget_max: int,
    regions: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-local features and terminal-region Monte Carlo labels."""
    features, labels, _ = value_examples_with_regions(
        reference, rollout, budget_max=budget_max, regions=regions
    )
    return features, labels


def value_examples_with_regions(
    reference,
    rollout: dict,
    *,
    budget_max: int,
    regions: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return value examples with the goal-grid index of every training row."""
    result = value_examples_with_bellman(reference, rollout, budget_max=budget_max, regions=regions)
    return result.features, result.labels, result.region_indices


def value_examples_with_bellman(
    reference,
    rollout: dict,
    *,
    budget_max: int,
    regions: Sequence[tuple[float, float]] | None = None,
    embedding_lookup: Mapping[str, np.ndarray] | None = None,
) -> ValueExamples:
    """Bind each Bellman row to its successor or exact terminal boundary."""
    if rollout.get("schema_version") not in (
        "compose.qed.shared_rollout.v1",
        "compose.qed.imported_h24_rollout.v1",
    ):
        raise ValueError("unsupported QED rollout schema")
    if rollout.get("reference") != reference.identity():
        raise ValueError("QED rollout belongs to another reference configuration")
    if rollout.get("source_role") not in ("train", "validation"):
        raise ValueError("QED value examples require a training or validation rollout")
    horizon = rollout["configuration"]["horizon"]
    if type(horizon) is not int or horizon < 1 or horizon > budget_max:
        raise ValueError("QED rollout horizon exceeds the value-head budget")
    expected_replicates = rollout["configuration"]["replicates"]
    trajectories = rollout["trajectories"]
    if len(trajectories) != expected_replicates:
        raise ValueError("QED rollout trajectory count differs from configuration")
    goals = tuple(regions if regions is not None else registered_regions())
    if not goals:
        raise ValueError("QED value training requires at least one goal region")
    slots = reference.config.persistent_slots
    source = rollout["source_represented"]
    original = Chem.MolFromSmiles(rollout["source_original"])
    if original is None:
        raise ValueError("QED rollout original source cannot be parsed")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    source_fingerprint = fingerprint.GetFingerprint(original)
    cache: dict[str, np.ndarray] = {}

    def encode(smiles: str) -> np.ndarray:
        cached = cache.get(smiles)
        if cached is None:
            cached = embedding_lookup.get(smiles) if embedding_lookup is not None else None
            if cached is None:
                state = pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)
                cached = reference.encode(state)
            cached = np.asarray(cached, dtype=np.float64)
            if cached.shape != (256,) or not np.isfinite(cached).all():
                raise ValueError(f"QED reference embedding is invalid for {smiles!r}")
            cache[smiles] = cached
        return cached

    source_embedding = encode(source)
    examples = []
    labels = []
    region_indices = []
    row_by_key: dict[tuple[int, int, int], int] = {}
    next_keys: list[tuple[int, int, int] | None] = []
    next_terminal_targets: list[int] = []
    for replicate, trajectory in enumerate(trajectories):
        if trajectory.get("replicate") != replicate:
            raise ValueError("QED rollout replicate order or identity changed")
        if trajectory.get("seed") != _expected_seed(rollout, original, source, replicate):
            raise ValueError("QED rollout seed differs from the declared derivation")
        path = trajectory["path"]
        if not path or path[0]["smiles"] != source or len(path) > horizon + 1:
            raise ValueError("QED rollout path is empty, mismatched, or over horizon")
        if trajectory["status"] == "HORIZON" and len(path) != horizon + 1:
            raise ValueError("QED horizon trajectory ended before its declared horizon")
        if trajectory["status"] not in ("HORIZON", "TERMINAL"):
            raise ValueError("unsupported QED rollout terminal status")
        if trajectory["status"] == "TERMINAL" and len(path) == horizon + 1:
            raise ValueError("QED terminal trajectory cannot fill the declared horizon")
        for step, node in enumerate(path):
            molecule = Chem.MolFromSmiles(node["smiles"])
            if molecule is None:
                raise ValueError(f"QED rollout replicate {replicate} step {step} cannot be parsed")
            quality = float(QED.qed(molecule))
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    source_fingerprint, fingerprint.GetFingerprint(molecule)
                )
            )
            if (
                abs(quality - float(node["qed"])) > 1e-8
                or abs(similarity - float(node["similarity_to_source"])) > 1e-8
            ):
                raise ValueError(f"QED rollout replicate {replicate} step {step} metric drift")
        path_embedding = [encode(node["smiles"]) for node in path]
        completed = trajectory["status"] == "HORIZON"
        for step, node in enumerate(path):
            budget = horizon - step
            if budget == 0:
                continue
            quality = float(node["qed"])
            similarity = float(node["similarity_to_source"])
            if not np.isfinite(quality) or not np.isfinite(similarity):
                raise ValueError("QED rollout has non-finite benchmark properties")
            for region_index, region in enumerate(goals):
                terminal_success = completed and in_region(
                    float(path[-1]["qed"]),
                    float(path[-1]["similarity_to_source"]),
                    region,
                )
                examples.append(
                    build_features(
                        path_embedding[step],
                        source_embedding,
                        quality,
                        similarity,
                        region,
                        budget,
                        budget_max,
                    ).astype(np.float32)
                )
                labels.append(float(terminal_success))
                region_indices.append(region_index)
                row_by_key[(replicate, step, region_index)] = len(examples) - 1
                if step + 1 == len(path):
                    next_keys.append(None)
                    next_terminal_targets.append(0)
                elif step + 1 == horizon:
                    next_keys.append(None)
                    next_terminal_targets.append(int(terminal_success))
                else:
                    next_keys.append((replicate, step + 1, region_index))
                    next_terminal_targets.append(-1)
    if not examples:
        return ValueExamples(
            features=np.empty((0, input_dim(budget_max)), np.float32),
            labels=np.empty((0,), np.float32),
            region_indices=np.empty((0,), np.int16),
            next_row_indices=np.empty((0,), np.int32),
            next_terminal_targets=np.empty((0,), np.int8),
        )
    if any(key is not None and key not in row_by_key for key in next_keys):
        raise ValueError("QED Bellman successor is absent from the source-local feature rows")
    return ValueExamples(
        features=np.stack(examples),
        labels=np.asarray(labels, dtype=np.float32),
        region_indices=np.asarray(region_indices, dtype=np.int16),
        next_row_indices=np.asarray(
            [-1 if key is None else row_by_key[key] for key in next_keys], dtype=np.int32
        ),
        next_terminal_targets=np.asarray(next_terminal_targets, dtype=np.int8),
    )
