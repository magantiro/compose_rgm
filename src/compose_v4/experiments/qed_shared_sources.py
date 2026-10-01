"""Load QED source roles with a split check at the graph representation level."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem


@dataclass(frozen=True)
class QEDSourceRoles:
    train: tuple[str, ...]
    train_input_indices: tuple[int, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    excluded_train_indices: tuple[int, ...]
    manifest_sha256: str


def _load_role(root: Path, entry: dict, role: str) -> tuple[str, ...]:
    path = root / entry["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{role} source hash mismatch: {path}")
    rows = tuple(line.strip() for line in data.decode().splitlines() if line.strip())
    if len(rows) != entry["rows"]:
        raise ValueError(f"{role} source count mismatch: {path}")
    return rows


def _group_keys(rows: tuple[str, ...], role: str) -> tuple[str, ...]:
    keys = []
    for index, smiles in enumerate(rows):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"{role} row {index}: cannot parse {smiles!r}")
        keys.append(Chem.MolToSmiles(molecule, isomericSmiles=False))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{role}: repeated nonisomeric graph within source role")
    return tuple(keys)


def load_qed_source_roles(root: Path, manifest_path: Path) -> QEDSourceRoles:
    """Verify frozen input hashes and the declared train-only exclusion."""
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != "compose.qed.shared_sources.v1":
        raise ValueError(f"unsupported QED source manifest: {manifest_path}")
    if manifest.get("group_key") != "canonical_nonisomeric_smiles":
        raise ValueError("QED split must use the graph representation's identity")

    train_all = _load_role(root, manifest["train"], "train")
    validation = _load_role(root, manifest["validation"], "validation")
    test = _load_role(root, manifest["test"], "test")
    train_keys = _group_keys(train_all, "train")
    validation_keys = set(_group_keys(validation, "validation"))
    test_keys = set(_group_keys(test, "test"))
    if validation_keys & test_keys:
        raise ValueError("validation and test share a nonisomeric graph")

    exclusions = manifest["excluded_train_rows"]
    indices = tuple(row["index_zero_based"] for row in exclusions)
    if len(indices) != len(set(indices)) or any(
        type(index) is not int or not 0 <= index < len(train_all) for index in indices
    ):
        raise ValueError("invalid or repeated train exclusion index")
    for row in exclusions:
        if row["reason"] != "same_nonisomeric_graph_as_test_source":
            raise ValueError("unsupported train exclusion reason")
    actual_overlap = {
        index for index, key in enumerate(train_keys) if key in validation_keys | test_keys
    }
    if actual_overlap != set(indices):
        raise ValueError(
            "declared train exclusions do not match the source-role overlap: "
            f"declared={sorted(indices)}, actual={sorted(actual_overlap)}"
        )
    if any(train_keys[index] not in test_keys for index in indices):
        raise ValueError("train exclusion is not a test-graph overlap")

    kept_indices = tuple(index for index in range(len(train_all)) if index not in actual_overlap)
    train = tuple(train_all[index] for index in kept_indices)
    return QEDSourceRoles(
        train,
        kept_indices,
        validation,
        test,
        tuple(sorted(indices)),
        hashlib.sha256(raw).hexdigest(),
    )
