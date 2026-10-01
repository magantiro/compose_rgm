"""Unguided QED-editing trajectories from the shared molecular reference."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.qed_source_support import audit_qed_source


@dataclass(frozen=True)
class QEDRolloutConfig:
    horizon: int
    replicates: int

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if type(self.replicates) is not int or self.replicates < 1:
            raise ValueError("replicates must be a positive integer")


def rollout_seed(canonical_source: str, replicate: int) -> int:
    payload = f"compose.qed.shared_rollout.v1|{canonical_source}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rollout_source(reference, source: str, source_index: int, config: QEDRolloutConfig) -> dict:
    """Produce complete trajectories without a task score or fitted value head."""
    support = audit_qed_source(source, max_active_atoms=reference.reference.max_active_atoms)
    if not support.supported or not support.benchmark_equivalent:
        raise ValueError(f"QED source {source_index} is outside benchmark-compatible support")
    if support.represented is None:
        raise ValueError("supported QED source has no represented molecule")
    original = Chem.MolFromSmiles(source)
    if original is None:
        raise ValueError(f"QED source {source_index} cannot be parsed")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    source_fingerprint = generator.GetFingerprint(original)
    slots = reference.config.persistent_slots

    def record(smiles: str) -> dict[str, float | str]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"reference produced an undecodable molecule: {smiles!r}")
        return {
            "smiles": smiles,
            "qed": float(QED.qed(molecule)),
            "similarity_to_source": float(
                DataStructs.TanimotoSimilarity(
                    source_fingerprint, generator.GetFingerprint(molecule)
                )
            ),
        }

    trajectories = []
    for replicate in range(config.replicates):
        seed = rollout_seed(support.represented, replicate)
        rng = np.random.default_rng(seed)
        state = pad_molecular_graph(smiles_to_molecular_graph(support.represented), slots)
        path = [record(support.represented)]
        status = "HORIZON"
        for _ in range(config.horizon):
            successor = reference.sample(state, rng)
            if successor is None:
                status = "TERMINAL"
                break
            smiles = molecular_graph_to_smiles(successor)
            if smiles is None:
                raise ValueError("reference produced a state that cannot be serialized")
            state = pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)
            path.append(record(smiles))
        trajectories.append({"replicate": replicate, "seed": seed, "status": status, "path": path})

    return {
        "schema_version": "compose.qed.shared_rollout.v1",
        "source_index": source_index,
        "source_original": source,
        "source_represented": support.represented,
        "reference": reference.identity(),
        "configuration": asdict(config),
        "trajectories": trajectories,
    }
