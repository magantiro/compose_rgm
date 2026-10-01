"""Finite-horizon QED value inference bound to one frozen molecular reference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.hphi_region_features import build_features, in_region, input_dim
from compose_v4.experiments.qed_shared_reference import QEDSharedReference
from compose_v4.experiments.qed_source_support import audit_qed_source
from compose_v4.model.factorized_tracelet_rate_model import molecular_state_cache_key


def _verified_bytes(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"QED value asset hash mismatch: {path}; expected {expected_sha256}")
    return data


@dataclass(frozen=True)
class QEDValueAssets:
    head: Path
    head_sha256: str
    normalization: Path
    normalization_sha256: str
    metadata: Path
    metadata_sha256: str

    @classmethod
    def from_directory(cls, directory: Path) -> QEDValueAssets:
        """Read a fitted head's fixed-name, hash-bound asset manifest."""
        directory = Path(directory)
        manifest_path = directory / "assets.json"
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("schema_version") != "compose.qed.shared_value_assets.v1":
            raise ValueError(f"unsupported QED value asset manifest: {manifest_path}")
        for asset_name, filename in (
            ("head", "head.pt"),
            ("normalization", "normalization.json"),
            ("metadata", "metadata.json"),
        ):
            entry = manifest.get(asset_name)
            if (
                not isinstance(entry, dict)
                or entry.get("path") != filename
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in entry["sha256"])
            ):
                raise ValueError(
                    f"QED value asset manifest has invalid {asset_name}: {manifest_path}"
                )
        return cls(
            directory / "head.pt",
            manifest["head"]["sha256"],
            directory / "normalization.json",
            manifest["normalization"]["sha256"],
            directory / "metadata.json",
            manifest["metadata"]["sha256"],
        )


class QEDSharedValueHead:
    def __init__(
        self,
        reference: QEDSharedReference,
        head: torch.nn.Module,
        mean: np.ndarray,
        scale: np.ndarray,
        metadata: dict,
    ) -> None:
        if metadata.get("schema_version") != "compose.qed.shared_value.v1":
            raise ValueError("unsupported QED shared-value metadata")
        if metadata.get("reference") != reference.identity():
            raise ValueError("QED value head was fitted to another reference configuration")
        if metadata.get("feature_schema") != "region_features_v1":
            raise ValueError("unsupported QED value feature schema")
        budget_max = metadata.get("budget_max")
        if type(budget_max) is not int or budget_max < 1:
            raise ValueError("QED value metadata needs a positive budget_max")
        width = input_dim(budget_max)
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError(f"QED value normalization must have width {width}")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("QED value normalization is not finite with positive scales")
        if not metadata.get("source_split_sha256"):
            raise ValueError("QED value head lacks its source split identity")
        self.reference = reference
        self.head = head.eval()
        for parameter in self.head.parameters():
            if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
                raise ValueError("QED value head must use CPU float32 parameters")
            parameter.requires_grad_(False)
        self.mean = mean.astype(np.float64)
        self.scale = scale.astype(np.float64)
        self.metadata = metadata
        self.budget_max = budget_max

    @classmethod
    def load(
        cls,
        reference: QEDSharedReference,
        assets: QEDValueAssets,
        *,
        expected_source_split_sha256: str,
    ) -> QEDSharedValueHead:
        metadata = json.loads(_verified_bytes(assets.metadata, assets.metadata_sha256))
        if metadata.get("source_split_sha256") != expected_source_split_sha256:
            raise ValueError("QED value head belongs to another source split")
        if metadata.get("guidance_target") != {
            "region": [0.9, 0.4],
            "qualified": True,
        }:
            raise ValueError("QED value head did not qualify on the benchmark goal")
        normalization = json.loads(
            _verified_bytes(assets.normalization, assets.normalization_sha256)
        )
        _verified_bytes(assets.head, assets.head_sha256)
        head = torch.jit.load(str(assets.head), map_location="cpu")
        _verified_bytes(assets.head, assets.head_sha256)
        return cls(
            reference,
            head,
            np.asarray(normalization["mean"], dtype=np.float64),
            np.asarray(normalization["scale"], dtype=np.float64),
            metadata,
        )

    def for_source(self, source: str) -> BoundQEDValue:
        support = audit_qed_source(
            source, max_active_atoms=self.reference.reference.max_active_atoms
        )
        if not support.supported or not support.benchmark_equivalent or support.represented is None:
            raise ValueError("QED value source is outside benchmark-compatible support")
        original = Chem.MolFromSmiles(source)
        if original is None:
            raise ValueError("QED value source cannot be parsed")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        source_fingerprint = generator.GetFingerprint(original)
        source_state = pad_molecular_graph(
            smiles_to_molecular_graph(support.represented),
            self.reference.config.persistent_slots,
        )
        embedding = self.reference.encode(source_state)
        return BoundQEDValue(
            owner=self,
            source_fingerprint=source_fingerprint,
            generator=generator,
            source_embedding=embedding,
            embedding_cache={molecular_state_cache_key(source_state): embedding},
        )


@dataclass
class BoundQEDValue:
    owner: QEDSharedValueHead
    source_fingerprint: object
    generator: object
    source_embedding: np.ndarray
    property_cache: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)
    embedding_cache: dict[object, np.ndarray] = field(default_factory=dict, repr=False)
    value_cache: dict[tuple[object, int, tuple[float, float]], float] = field(
        default_factory=dict, repr=False
    )

    def properties(self, state: MolecularGraph) -> tuple[float, float]:
        smiles = molecular_graph_to_smiles(state)
        if smiles is None:
            raise ValueError("QED value state cannot be serialized")
        cached = self.property_cache.get(smiles)
        if cached is not None:
            return cached
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("QED value state cannot be parsed")
        result = (
            float(QED.qed(molecule)),
            float(
                DataStructs.TanimotoSimilarity(
                    self.source_fingerprint, self.generator.GetFingerprint(molecule)
                )
            ),
        )
        self.property_cache[smiles] = result
        return result

    def in_target(self, state: MolecularGraph, region: tuple[float, float]) -> bool:
        return in_region(*self.properties(state), region)

    def value(self, state: MolecularGraph, budget: int, region: tuple[float, float]) -> float:
        if type(budget) is not int or not 0 <= budget <= self.owner.budget_max:
            raise ValueError("QED value budget is outside the fitted range")
        state_key = molecular_state_cache_key(state)
        cache_key = state_key, budget, region
        cached = self.value_cache.get(cache_key)
        if cached is not None:
            return cached
        quality, similarity = self.properties(state)
        if in_region(quality, similarity, region):
            self.value_cache[cache_key] = 1.0
            return 1.0
        if budget == 0:
            self.value_cache[cache_key] = 0.0
            return 0.0
        embedding = self.embedding_cache.get(state_key)
        if embedding is None:
            embedding = self.owner.reference.encode(state)
            self.embedding_cache[state_key] = embedding
        features = build_features(
            embedding,
            self.source_embedding,
            quality,
            similarity,
            region,
            budget,
            self.owner.budget_max,
        )
        normalized = ((features - self.owner.mean) / self.owner.scale).astype(np.float32)
        with torch.no_grad():
            logits = self.owner.head(torch.from_numpy(normalized).unsqueeze(0))
        if logits.shape != (1, 1) or not torch.isfinite(logits).all():
            raise ValueError("QED value head returned invalid logits")
        probability = float(torch.sigmoid(logits[0, 0]))
        if not math.isfinite(probability):
            raise ValueError("QED value head returned a non-finite probability")
        self.value_cache[cache_key] = probability
        return probability
