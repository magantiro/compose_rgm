"""Split-first, nested C/N/O/F data for a de novo prior-scale comparison.

This prepares data only. It never fits a model or reads a benchmark score.
The 50k arm is a prefix of one frozen training reservoir, and both arms share
the same validation, IID-test and scaffold-held-out identities.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from hashlib import blake2b, sha256
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import (
    IDX_TO_ELEMENT,
    MolecularGraphError,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null
from compose_v4.data.cnof import CNOFCorpusSplit, murcko_scaffold_key

SCHEMA = "compose.frozen_cnof_prior_split.v1"
_ALLOWED_ELEMENTS = frozenset({"C", "N", "O", "F"})
_PARTITIONS = ("train", "validation", "iid_test", "scaffold_test")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(text: str, *, max_atoms: int) -> tuple[str | None, str | None]:
    """Use the shipped CNOF graph representation; retain a refusal reason."""

    if not text:
        return None, "empty"
    if "." in text:
        return None, "disconnected"
    try:
        graph = smiles_to_molecular_graph(text)
    except (MolecularGraphError, ValueError):
        return None, "unparseable_or_unsupported_state"
    if graph.n_real_atoms == 0:
        return None, "empty"
    if graph.n_real_atoms > max_atoms:
        return None, "too_big"
    if not is_connected_or_null(graph):
        return None, "disconnected"
    real = is_element(graph.atom_types)
    if np.any(graph.formal_charges[real] != 0):
        return None, "charged"
    if any(
        IDX_TO_ELEMENT[int(atom_type)] not in _ALLOWED_ELEMENTS
        for atom_type in graph.atom_types[real]
    ):
        return None, "unsupported_element"
    return molecular_graph_to_smiles(graph), None


def _group_order(seed: int, scaffold: str) -> bytes:
    return blake2b(f"{seed}\0{scaffold}".encode(), digest_size=16).digest()


def _arm_scaffold_diagnostics(
    train: list[str],
    validation: list[str],
    iid_test: list[str],
    *,
    scaffold_by_smiles: dict[str, str],
) -> dict:
    train_counts = Counter(scaffold_by_smiles[smiles] for smiles in train)
    validation_keys = {scaffold_by_smiles[smiles] for smiles in validation}
    iid_test_keys = {scaffold_by_smiles[smiles] for smiles in iid_test}
    return {
        "train_molecules": len(train),
        "train_scaffold_groups": len(train_counts),
        "train_acyclic_molecules": train_counts.get("<ACYCLIC>", 0),
        "train_largest_group_fraction": max(train_counts.values()) / len(train),
        "train_top_scaffold_groups": [
            [key, count]
            for key, count in sorted(train_counts.items(), key=lambda item: (-item[1], item[0]))[
                :10
            ]
        ],
        "iid_scaffold_overlap_count": len(
            (set(train_counts) & validation_keys)
            | (set(train_counts) & iid_test_keys)
            | (validation_keys & iid_test_keys)
        ),
    }


def prepare_frozen_cnof_split(
    source: Path,
    *,
    expected_source_sha256: str,
    source_access_basis: str,
    code_revision: str,
    seed: int,
    max_atoms: int,
    small_train_size: int,
    validation_size: int,
    iid_test_size: int,
    scaffold_test_min: int,
    scaffold_test_max: int,
    expected_eligible_unique: int | None = None,
) -> dict:
    """Return one auditable, deterministic split payload.

    Source row numbers are one-based. A scaffold group is indivisible. A group
    exceeding the declared held-out cap stays outside the scaffold test, with
    that decision recorded, rather than being silently split across roles.
    """

    source = Path(source)
    if not source.is_file():
        raise ValueError(f"missing source corpus: {source}")
    actual_hash = file_sha256(source)
    if actual_hash != expected_source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch for {source}: "
            f"expected {expected_source_sha256}, observed {actual_hash}"
        )
    if not code_revision or not isinstance(code_revision, str):
        raise ValueError("code_revision must name the exact preparation revision")
    if not source_access_basis or not isinstance(source_access_basis, str):
        raise ValueError("source_access_basis must document the corpus access basis")
    sizes = (
        max_atoms,
        small_train_size,
        validation_size,
        iid_test_size,
        scaffold_test_min,
        scaffold_test_max,
    )
    if any(not isinstance(value, int) or value <= 0 for value in sizes):
        raise ValueError("all size and capacity fields must be positive integers")
    if scaffold_test_max < scaffold_test_min:
        raise ValueError("scaffold_test_max must be >= scaffold_test_min")
    if expected_eligible_unique is not None and expected_eligible_unique <= 0:
        raise ValueError("expected_eligible_unique must be positive")

    first_seen: dict[str, int] = {}
    exclusions: list[list[int | str]] = []
    duplicates: list[list[int]] = []
    with source.open() as handle:
        source_rows = 0
        for source_rows, line in enumerate(handle, start=1):
            fields = line.strip().split()
            canonical, reason = _classify(fields[0] if fields else "", max_atoms=max_atoms)
            if reason is not None:
                exclusions.append([source_rows, reason])
            elif canonical in first_seen:
                duplicates.append([source_rows, first_seen[canonical]])
            else:
                if canonical is None:
                    raise RuntimeError(
                        f"CNOF classifier returned no molecule or reason at source row {source_rows}"
                    )
                first_seen[canonical] = source_rows
    if expected_eligible_unique is not None and len(first_seen) != expected_eligible_unique:
        raise ValueError(
            f"eligible CNOF molecule census mismatch for {source}: "
            f"expected {expected_eligible_unique}, observed {len(first_seen)}"
        )

    groups: dict[str, list[str]] = defaultdict(list)
    scaffold_by_smiles: dict[str, str] = {}
    for canonical in first_seen:
        scaffold = murcko_scaffold_key(canonical)
        scaffold_by_smiles[canonical] = scaffold
        groups[scaffold].append(canonical)
    ordered_scaffolds = sorted(groups, key=lambda key: (_group_order(seed, key), key))
    held_scaffolds: set[str] = set()
    skipped_large_groups: list[list[str | int]] = []
    held_count = 0
    for scaffold in ordered_scaffolds:
        if held_count >= scaffold_test_min:
            break
        n_group = len(groups[scaffold])
        if held_count + n_group > scaffold_test_max:
            skipped_large_groups.append([scaffold, n_group])
            continue
        held_scaffolds.add(scaffold)
        held_count += n_group
    if held_count < scaffold_test_min:
        raise ValueError(
            f"cannot make a scaffold-disjoint holdout of at least "
            f"{scaffold_test_min} within cap {scaffold_test_max}; got {held_count}"
        )

    scaffold_test = [
        canonical for canonical in first_seen if scaffold_by_smiles[canonical] in held_scaffolds
    ]
    remaining = [
        canonical for canonical in first_seen if scaffold_by_smiles[canonical] not in held_scaffolds
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    rng.shuffle(scaffold_test)
    needed = small_train_size + validation_size + iid_test_size
    if len(remaining) < needed:
        raise ValueError(
            f"only {len(remaining)} non-held-out CNOF molecules in {source}; need at least {needed}"
        )
    iid_test = remaining[:iid_test_size]
    validation = remaining[iid_test_size : iid_test_size + validation_size]
    train = remaining[iid_test_size + validation_size :]
    arm_diagnostics = {
        "small": _arm_scaffold_diagnostics(
            train[:small_train_size],
            validation,
            iid_test,
            scaffold_by_smiles=scaffold_by_smiles,
        ),
        "large": _arm_scaffold_diagnostics(
            train,
            validation,
            iid_test,
            scaffold_by_smiles=scaffold_by_smiles,
        ),
    }
    partitions = {
        "train": [[first_seen[smiles], smiles] for smiles in train],
        "validation": [[first_seen[smiles], smiles] for smiles in validation],
        "iid_test": [[first_seen[smiles], smiles] for smiles in iid_test],
        "scaffold_test": [[first_seen[smiles], smiles] for smiles in scaffold_test],
    }
    config = {
        "seed": seed,
        "max_atoms": max_atoms,
        "small_train_size": small_train_size,
        "validation_size": validation_size,
        "iid_test_size": iid_test_size,
        "scaffold_test_min": scaffold_test_min,
        "scaffold_test_max": scaffold_test_max,
        "expected_eligible_unique": expected_eligible_unique,
        "group_policy": "murcko_whole_group_hash_order_skip_oversize_v1",
    }
    payload = {
        "schema": SCHEMA,
        "source": {
            "path": str(source),
            "sha256": actual_hash,
            "access_basis": source_access_basis,
        },
        "code_revision": code_revision,
        "python_version": sys.version.split()[0],
        "rdkit_version": rdBase.rdkitVersion,
        "config": config,
        "census": {
            "source_rows": source_rows,
            "eligible_unique": len(first_seen),
            "excluded": len(exclusions),
            "duplicates": len(duplicates),
            "exclusion_reasons": dict(sorted(Counter(reason for _, reason in exclusions).items())),
            "scaffold_groups": len(groups),
            "training_arm_diagnostics": arm_diagnostics,
            "skipped_oversize_scaffold_groups": skipped_large_groups,
        },
        "partitions": partitions,
        "partition_sha256": {name: _digest(rows) for name, rows in partitions.items()},
        "exclusions": exclusions,
        "duplicates": duplicates,
    }
    payload["payload_sha256"] = _digest(payload)
    validate_frozen_cnof_split(payload, deep=True)
    return payload


def validate_frozen_cnof_split(payload: dict, *, deep: bool = False) -> None:
    """Fail closed on corruption, overlap, unsupported rows or altered policy."""

    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported CNOF split schema: {payload.get('schema')!r}")
    claimed_hash = payload.get("payload_sha256")
    if claimed_hash != _digest({k: v for k, v in payload.items() if k != "payload_sha256"}):
        raise ValueError("frozen CNOF split payload hash mismatch")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(_PARTITIONS):
        raise ValueError("frozen CNOF split has missing or extra partitions")
    if payload.get("partition_sha256") != {name: _digest(partitions[name]) for name in _PARTITIONS}:
        raise ValueError("frozen CNOF split partition hash mismatch")
    config = payload["config"]
    if len(partitions["train"]) < config["small_train_size"]:
        raise ValueError("frozen training reservoir is smaller than small arm")
    arm_diagnostics = payload["census"]["training_arm_diagnostics"]
    if set(arm_diagnostics) != {"small", "large"}:
        raise ValueError("frozen training-arm diagnostics are missing")
    if arm_diagnostics["small"]["train_molecules"] != config["small_train_size"]:
        raise ValueError("frozen small-arm count mismatch")
    if arm_diagnostics["large"]["train_molecules"] != len(partitions["train"]):
        raise ValueError("frozen large-arm count mismatch")
    if len(partitions["validation"]) != config["validation_size"]:
        raise ValueError("frozen validation count mismatch")
    if len(partitions["iid_test"]) != config["iid_test_size"]:
        raise ValueError("frozen IID-test count mismatch")
    if (
        not config["scaffold_test_min"]
        <= len(partitions["scaffold_test"])
        <= config["scaffold_test_max"]
    ):
        raise ValueError("frozen scaffold-test count outside declared bounds")
    all_smiles: set[str] = set()
    all_source_rows: set[int] = set()
    for name in _PARTITIONS:
        for row in partitions[name]:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], int)
                or not isinstance(row[1], str)
            ):
                raise ValueError(f"malformed source row in {name}: {row!r}")
    scaffold_by_smiles = (
        {row[1]: murcko_scaffold_key(row[1]) for name in _PARTITIONS for row in partitions[name]}
        if deep
        else {}
    )
    held_scaffolds = (
        {scaffold_by_smiles[row[1]] for row in partitions["scaffold_test"]} if deep else set()
    )
    for name in _PARTITIONS:
        for row in partitions[name]:
            smiles = row[1]
            if not isinstance(smiles, str) or smiles in all_smiles:
                raise ValueError(f"duplicate or malformed canonical molecule in {name}: {smiles!r}")
            all_smiles.add(smiles)
            if row[0] <= 0 or row[0] in all_source_rows:
                raise ValueError(f"duplicate or invalid source row in {name}: {row[0]}")
            all_source_rows.add(row[0])
            if deep and _classify(smiles, max_atoms=config["max_atoms"])[0] != smiles:
                raise ValueError(f"noncanonical or out-of-scope molecule in {name}: {smiles}")
            if deep and name != "scaffold_test" and scaffold_by_smiles[smiles] in held_scaffolds:
                raise ValueError(f"scaffold group leaked into {name}: {smiles}")
    census = payload["census"]
    if len(all_smiles) != census["eligible_unique"]:
        raise ValueError("eligible molecule census does not equal partition union")
    if (
        census["source_rows"]
        != census["eligible_unique"] + census["excluded"] + census["duplicates"]
    ):
        raise ValueError("source-row census does not reconcile")
    excluded_rows = [row[0] for row in payload["exclusions"]]
    duplicate_rows = [row[0] for row in payload["duplicates"]]
    if len(set(excluded_rows + duplicate_rows)) != len(excluded_rows) + len(duplicate_rows):
        raise ValueError("exclusion and duplicate source-row ledgers overlap")
    if all_source_rows.intersection(excluded_rows + duplicate_rows):
        raise ValueError("accepted source row appears in exclusion/duplicate ledger")
    if any(row[1] not in all_source_rows for row in payload["duplicates"]):
        raise ValueError("duplicate ledger refers to a non-accepted first row")
    if any(row[1] >= row[0] for row in payload["duplicates"]):
        raise ValueError("duplicate ledger first row does not precede duplicate")
    reason_counts = dict(sorted(Counter(row[1] for row in payload["exclusions"]).items()))
    if reason_counts != census["exclusion_reasons"]:
        raise ValueError("exclusion reason census does not match row ledger")
    if len(all_source_rows) + len(excluded_rows) + len(duplicate_rows) != census["source_rows"]:
        raise ValueError("source-row ledgers do not cover the input corpus")
    if deep and set(range(1, census["source_rows"] + 1)) != (
        all_source_rows | set(excluded_rows) | set(duplicate_rows)
    ):
        raise ValueError("source-row ledgers contain gaps or out-of-range rows")
    if deep:
        validation = [row[1] for row in partitions["validation"]]
        iid_test = [row[1] for row in partitions["iid_test"]]
        train = [row[1] for row in partitions["train"]]
        expected_diagnostics = {
            "small": _arm_scaffold_diagnostics(
                train[: config["small_train_size"]],
                validation,
                iid_test,
                scaffold_by_smiles=scaffold_by_smiles,
            ),
            "large": _arm_scaffold_diagnostics(
                train,
                validation,
                iid_test,
                scaffold_by_smiles=scaffold_by_smiles,
            ),
        }
        if arm_diagnostics != expected_diagnostics:
            raise ValueError("frozen training-arm scaffold diagnostics mismatch")


def _arm_name(payload: dict, train_size: int) -> str:
    reservoir_size = len(payload["partitions"]["train"])
    if train_size == payload["config"]["small_train_size"]:
        return "small"
    if train_size == reservoir_size:
        return "large"
    raise ValueError(
        f"train_size {train_size} is not either frozen arm "
        f"({payload['config']['small_train_size']}, {reservoir_size})"
    )


def load_frozen_cnof_arm(path: Path, *, train_size: int) -> CNOFCorpusSplit:
    """Load an arm without reading the source corpus or changing held-out rows."""

    payload = json.loads(Path(path).read_text())
    validate_frozen_cnof_split(payload)
    reservoir = payload["partitions"]["train"]
    arm_name = _arm_name(payload, train_size)
    return CNOFCorpusSplit(
        train=tuple(row[1] for row in reservoir[:train_size]),
        validation=tuple(row[1] for row in payload["partitions"]["validation"]),
        test=tuple(row[1] for row in payload["partitions"]["iid_test"]),
        scanned_lines=payload["census"]["source_rows"],
        eligible_molecules=payload["census"]["eligible_unique"],
        split_strategy="frozen_nested_with_scaffold_test_v1",
        scaffold_overlap_count=payload["census"]["training_arm_diagnostics"][arm_name][
            "iid_scaffold_overlap_count"
        ],
    )


def frozen_cnof_arm_identity(path: Path, *, train_size: int) -> dict:
    """Return the exact data identity for a trainer checkpoint/receipt."""

    path = Path(path)
    payload = json.loads(path.read_text())
    validate_frozen_cnof_split(payload)
    reservoir = payload["partitions"]["train"]
    arm_name = _arm_name(payload, train_size)
    return {
        "schema": SCHEMA,
        "source_path": payload["source"]["path"],
        "manifest_file_sha256": file_sha256(path),
        "manifest_payload_sha256": payload["payload_sha256"],
        "source_sha256": payload["source"]["sha256"],
        "train_prefix_sha256": _digest(reservoir[:train_size]),
        "validation_sha256": payload["partition_sha256"]["validation"],
        "iid_test_sha256": payload["partition_sha256"]["iid_test"],
        "scaffold_test_sha256": payload["partition_sha256"]["scaffold_test"],
        "max_atoms": payload["config"]["max_atoms"],
        "validation_size": payload["config"]["validation_size"],
        "iid_test_size": payload["config"]["iid_test_size"],
        "train_size": train_size,
        "train_arm": arm_name,
        "train_scaffold_diagnostics": payload["census"]["training_arm_diagnostics"][arm_name],
    }


def frozen_cnof_source_alias_identity(source: Path, *, expected_sha256: str) -> dict:
    """Bind a runtime mount path to the manifest's exact source bytes.

    The preparation host and an accelerator container can mount one frozen
    corpus at different absolute paths. A path-only equality check rejects
    that valid deployment; accepting any path without checking bytes would
    silently change the training population. This bridge verifies the full
    physical SHA-256 and records both runtime path and observed identity.
    """

    source = Path(source)
    if not source.is_file():
        raise ValueError(f"missing frozen CNOF runtime source: {source}")
    observed = file_sha256(source)
    if observed != expected_sha256:
        raise ValueError(
            f"frozen CNOF source SHA-256 mismatch for {source}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return {"runtime_source_path": str(source), "runtime_source_sha256": observed}
