"""Executable, hash-verified sampler for layered COMPOSE-Lipid manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


class LayeredSamplerError(ValueError):
    """Raised when a sampler artifact violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LayeredStructureRow:
    layer_id: str
    structure_sha256: str
    canonical_isomeric_smiles: str
    split_group_tokens: frozenset[str]
    within_layer_probability: float


@dataclass(frozen=True)
class _Layer:
    layer_id: str
    probability: float
    rows: tuple[LayeredStructureRow, ...]


class LayerAwareStructureSampler:
    """Draw a layer first and a frozen weighted structure second.

    Split-group exclusions are applied before sampling and re-normalize only
    within each layer. Layer masses therefore remain invariant across folds;
    if a holdout removes an entire layer, sampling fails instead of silently
    moving that probability to another chemistry source.
    """

    def __init__(self, layers: Iterable[_Layer]) -> None:
        self.layers = tuple(layers)
        if not self.layers:
            raise LayeredSamplerError("sampler contains no layers")
        layer_probabilities = np.asarray([layer.probability for layer in self.layers])
        if np.any(layer_probabilities < 0) or not np.isclose(layer_probabilities.sum(), 1.0):
            raise LayeredSamplerError("layer probabilities must be nonnegative and sum to one")
        structure_ids = [
            row.structure_sha256 for layer in self.layers for row in layer.rows
        ]
        if len(structure_ids) != len(set(structure_ids)):
            raise LayeredSamplerError("a canonical structure appears in multiple sampler rows")
        for layer in self.layers:
            probabilities = np.asarray(
                [row.within_layer_probability for row in layer.rows], dtype=float
            )
            if not len(probabilities) or np.any(probabilities < 0):
                raise LayeredSamplerError(f"layer {layer.layer_id!r} has invalid row weights")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
                raise LayeredSamplerError(
                    f"layer {layer.layer_id!r} row weights do not sum to one"
                )

    @classmethod
    def load(cls, manifest_path: str | Path) -> "LayerAwareStructureSampler":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        artifact = manifest["sampler_rows"]
        row_path = Path(artifact["path"])
        if not row_path.is_absolute():
            candidate = path.parents[3] / row_path
            if candidate.is_file():
                row_path = candidate
            else:
                row_path = Path.cwd() / row_path
        row_path = row_path.resolve()
        actual_hash = sha256_file(row_path)
        if actual_hash != artifact["sha256"]:
            raise LayeredSamplerError(
                f"sampler-row hash mismatch: expected {artifact['sha256']}, got {actual_hash}"
            )
        layer_contract = manifest["layers"]
        grouped: dict[str, list[LayeredStructureRow]] = {
            layer_id: [] for layer_id in layer_contract
        }
        with row_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "layer_id",
                "structure_sha256",
                "canonical_isomeric_smiles",
                "split_group_tokens_json",
                "within_layer_probability",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise LayeredSamplerError(f"sampler rows lack fields {sorted(required)}")
            for raw in reader:
                layer_id = raw["layer_id"]
                if layer_id not in grouped:
                    raise LayeredSamplerError(f"row references unknown layer {layer_id!r}")
                tokens = json.loads(raw["split_group_tokens_json"])
                if not isinstance(tokens, list) or not all(
                    isinstance(token, str) for token in tokens
                ):
                    raise LayeredSamplerError("split_group_tokens_json must be a string list")
                grouped[layer_id].append(
                    LayeredStructureRow(
                        layer_id=layer_id,
                        structure_sha256=raw["structure_sha256"],
                        canonical_isomeric_smiles=raw["canonical_isomeric_smiles"],
                        split_group_tokens=frozenset(tokens),
                        within_layer_probability=float(raw["within_layer_probability"]),
                    )
                )
        actual_count = sum(len(rows) for rows in grouped.values())
        if actual_count != int(artifact["row_count"]):
            raise LayeredSamplerError(
                f"sampler row count mismatch: expected {artifact['row_count']}, got {actual_count}"
            )
        layers = [
            _Layer(
                layer_id=layer_id,
                probability=float(layer_contract[layer_id]["layer_probability"]),
                rows=tuple(grouped[layer_id]),
            )
            for layer_id in sorted(layer_contract)
        ]
        return cls(layers)

    def sample(
        self,
        count: int,
        *,
        seed: int,
        excluded_split_group_tokens: Iterable[str] = (),
    ) -> list[LayeredStructureRow]:
        if count < 0:
            raise ValueError("count must be nonnegative")
        if count == 0:
            return []
        excluded = frozenset(excluded_split_group_tokens)
        eligible_rows: list[tuple[LayeredStructureRow, ...]] = []
        cumulative_within: list[np.ndarray] = []
        for layer in self.layers:
            rows = tuple(
                row for row in layer.rows if row.split_group_tokens.isdisjoint(excluded)
            )
            if not rows:
                raise LayeredSamplerError(
                    f"split exclusions removed every row from layer {layer.layer_id!r}"
                )
            probabilities = np.asarray(
                [row.within_layer_probability for row in rows], dtype=float
            )
            probabilities /= probabilities.sum()
            eligible_rows.append(rows)
            cumulative_within.append(np.cumsum(probabilities))
        layer_probabilities = np.asarray([layer.probability for layer in self.layers])
        cumulative_layers = np.cumsum(layer_probabilities)
        generator = np.random.default_rng(seed)
        layer_uniforms = generator.random(count)
        row_uniforms = generator.random(count)
        layer_indices = np.searchsorted(cumulative_layers, layer_uniforms, side="right")
        selected: list[LayeredStructureRow] = []
        for layer_index, row_uniform in zip(layer_indices, row_uniforms, strict=True):
            row_index = int(
                np.searchsorted(cumulative_within[layer_index], row_uniform, side="right")
            )
            selected.append(eligible_rows[layer_index][row_index])
        return selected
