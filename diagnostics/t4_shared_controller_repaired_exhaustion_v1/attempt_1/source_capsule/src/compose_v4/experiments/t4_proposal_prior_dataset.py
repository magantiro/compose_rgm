"""Split-clean corpus preparation for the T4 joint proposal prior.

The recovered historical table (`scored_rows.jsonl.gz`) is adaptive search output,
not a training set: repeated serialisations, repeated dockings of one molecule and
long productive genealogies are all over-represented, and nothing in it is split.

This module freezes group structure *before* any feature, vocabulary or moment is
derived, as required by the scientific-validity rules. It

1. proves that the declared atomic group (`cell`) is disjoint in every identity
   that could leak between folds, and fails closed otherwise;
2. admits charged observations under explicit reason codes;
3. collapses repeated receipts of one construction into a single labelled unit
   carrying its own dispersion;
4. closes genealogical edges into within-cell lineage components;
5. assigns whole targets to leave-one-target-out folds;
6. equalizes mass through the declared target/cell/lineage/record hierarchy;
7. publishes a per-fold coverage census and a GO/ABSTAIN decision.

It fits nothing, calls no oracle, and writes no runtime controller input. The
program payloads themselves are *bound by identity* (checkpoint file hash plus
entry id) rather than copied, so a later within-fold fitting step resolves them
from the hashed mirror instead of from a second, drifting copy.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_proposal_prior_dataset_v1"

# Identity fields that must never appear under two different cells. A crossing
# would put the same molecule, source state, construction or evaluator on both
# sides of a fold boundary.
DISJOINT_IDENTITIES = ("endpoint", "input_state_sha256", "entry_id", "protocol")

REASON_CODES = (
    "missing_or_nonfinite_score",
    "ambiguous_program_join",
    "undeclared_cell",
    "unbound_checkpoint",
    "missing_identity_field",
)

IDENTITY_FIELDS = ("cell", "target", "entry_id", "protocol", "endpoint", "input_state_sha256")

# The bank macro is legitimate training evidence but is not a generic module, and
# must never be confused with runtime route replay.
BANK_MACRO_FAMILY = "compiled_complete_transformation"

# ---- Contract ----


def load_contract(path: Path) -> dict:
    """Read a sealed contract and refuse a payload whose self-hash does not hold."""
    payload = unseal(Path(path))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a {SCHEMA_VERSION} contract: {path}")
    return payload


def verify_contract_inputs(contract: dict, root: Path) -> dict:
    """Every declared input must still hash to its frozen identity."""
    observed = {}
    for name, entry in sorted(contract["inputs"].items()):
        path = Path(root) / entry["path"]
        if not path.exists():
            raise ValueError(f"declared contract input is missing: {path}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"contract input changed: {path}: {entry['sha256']} -> {digest}")
        observed[name] = digest
    return observed


# ---- Reading ----


def read_scored_pack(path: Path) -> list[dict]:
    with gzip.open(Path(path), "rt") as handle:
        return [json.loads(line) for line in handle]


# ---- Group structure ----


def group_disjointness(rows, *, group_key: str = "cell", identities=DISJOINT_IDENTITIES) -> dict:
    """Evidence that no listed identity appears under two groups.

    Returned rather than asserted so the audit can publish the measurement; the
    builder raises on any violation.
    """
    evidence = {}
    for field in identities:
        groups = defaultdict(set)
        for row in rows:
            value = row.get(field)
            if value is not None:
                groups[value].add(row[group_key])
        crossing = sorted(key for key, seen in groups.items() if len(seen) > 1)
        evidence[field] = {
            "distinct_values": len(groups),
            "values_crossing_groups": len(crossing),
            "examples": crossing[:5],
        }
    return evidence


def assert_group_disjointness(evidence: dict) -> None:
    violated = sorted(f for f, e in evidence.items() if e["values_crossing_groups"])
    if violated:
        raise ValueError(f"declared group is not disjoint in: {', '.join(violated)}")


def fold_assignment(contract: dict) -> dict:
    """Map every declared cell to its leave-one-target-out fold, checking coverage."""
    members = contract["folds"]["members"]
    if len(members) != contract["folds"]["count"]:
        raise ValueError("declared fold count disagrees with declared fold membership")
    assignment: dict[str, str] = {}
    for fold, cells in members.items():
        for cell in cells:
            if cell in assignment:
                raise ValueError(f"cell {cell} is claimed by two folds")
            assignment[cell] = fold
    declared = {cell for cells in contract["grouping"]["declared_cells"].values() for cell in cells}
    if declared != set(assignment):
        raise ValueError("fold membership does not cover exactly the declared cells")
    return assignment


# ---- Admission ----


def _finite_score(row) -> bool:
    score = row.get("score")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)


def admit(rows, contract: dict, checkpoint_manifest: dict) -> tuple[list[dict], list[dict]]:
    """Apply the frozen admission rules; every rejection carries one reason code."""
    assignment = fold_assignment(contract)
    bound = set(checkpoint_manifest["checkpoints"])
    admitted, excluded = [], []

    def reject(row, code):
        excluded.append(
            {
                "reason": code,
                "cell": row.get("cell"),
                "receipt_id": row.get("receipt_id"),
                "entry_id": row.get("entry_id"),
            }
        )

    for row in rows:
        if any(row.get(field) is None for field in IDENTITY_FIELDS):
            reject(row, "missing_identity_field")
        elif row["cell"] not in assignment:
            reject(row, "undeclared_cell")
        elif not _finite_score(row):
            reject(row, "missing_or_nonfinite_score")
        elif row.get("entry_alternatives") != 1:
            reject(row, "ambiguous_program_join")
        elif Path(row.get("checkpoint", "")).name not in bound:
            reject(row, "unbound_checkpoint")
        else:
            admitted.append(row)
    if len(admitted) + len(excluded) != len(rows):
        raise ValueError("admission did not account for every input row")
    return admitted, excluded


# ---- Observation collapse ----


def collapse_observations(admitted) -> list[dict]:
    """One labelled unit per (cell, entry_id, protocol), with its own dispersion.

    Repeated serialisations and repeated dockings of the same construction are not
    independent observations; carrying them as separate rows would give one
    molecule several times its true mass. Measured: 3,732 endpoint/protocol groups
    carry more than one receipt.

    A construction reached in several replicates can carry several recorded
    genealogical parents (measured: 945 of 34,073 units). That is provenance, not a
    conflict, so every parent is retained and every one of them is unioned into the
    record's lineage component. Source state and endpoint are asserted single-valued
    because a unit with two of either would not be one construction.
    """
    grouped = defaultdict(list)
    for row in admitted:
        grouped[(row["cell"], row["entry_id"], row["protocol"])].append(row)
    records = []
    for (cell, entry_id, protocol), group in sorted(grouped.items()):
        scores = sorted(float(row["score"]) for row in group)
        head = group[0]
        for field in ("input_state_sha256", "endpoint"):
            if len({row[field] for row in group}) != 1:
                raise ValueError(f"construction {entry_id} has conflicting {field}")
        parents = sorted(
            {row["genealogical_parent"] for row in group if row.get("genealogical_parent")}
        )
        parent_scores = sorted(
            {
                float(row["parent_score"])
                for row in group
                if isinstance(row.get("parent_score"), (int, float))
                and not isinstance(row.get("parent_score"), bool)
                and math.isfinite(row["parent_score"])
            }
        )
        calls = sorted({row["query"] for row in group if row.get("query") is not None})
        records.append(
            {
                "record_id": _record_id(cell, entry_id, protocol),
                "cell": cell,
                "target": head["target"],
                "entry_id": entry_id,
                "protocol": protocol,
                "arms": sorted({row["arm"] for row in group}),
                "replicates": sorted({row["replicate"] for row in group}),
                "checkpoints": sorted({Path(row["checkpoint"]).name for row in group}),
                "receipt_ids": sorted(row["receipt_id"] for row in group),
                "observation_count": len(group),
                "score_mean": sum(scores) / len(scores),
                "score_min": scores[0],
                "score_max": scores[-1],
                "score_range": scores[-1] - scores[0],
                "parent_entry_ids": parents,
                "parent_scores": parent_scores,
                "input_state_sha256": head["input_state_sha256"],
                "endpoint_sha256": _text_sha256(head["endpoint"]),
                "primitive_count": head.get("primitive_count"),
                "changed_slot_count": head.get("changed_slot_count"),
                "net_created": head.get("net_created"),
                "net_deleted": head.get("net_deleted"),
                "ring_count_change": head.get("ring_count_change"),
                "channel": head.get("channel"),
                "planner_channel": head.get("planner_channel"),
                "construction_families": sorted(
                    set(block_families(head.get("block_labels") or []))
                ),
                "rule_counts": dict(sorted((head.get("rule_counts") or {}).items())),
                "call_indices": calls,
                "call_index_available": bool(calls),
            }
        )
    return records


def _record_id(cell: str, entry_id: str, protocol: str) -> str:
    body = canonical_bytes({"cell": cell, "entry_id": entry_id, "protocol": protocol})
    return hashlib.sha256(body).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def block_families(labels) -> list[str]:
    """Generic module family of each hierarchical block label.

    Labels are paths such as `composition_2:0:1:dependency_branch:1`; the family is
    the last segment that is not a positional index.
    """
    families = []
    for label in labels:
        segments = [s for s in str(label).split(":") if not s.isdigit()]
        families.append(segments[-1] if segments else str(label))
    return families


# ---- Lineage ----


def lineage_components(records) -> dict[str, str]:
    """Close genealogical edges into components, keyed inside one cell.

    Parents are archive entries, which may themselves be unscored and therefore
    absent from the corpus; the edge is still followed so two scored descendants of
    one unscored ancestor stay in the same component.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node):
        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            parent[node], node = root, parent[node]
        return root

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for record in records:
        node = (record["cell"], record["entry_id"])
        parent.setdefault(node, node)
        for parent_entry_id in record["parent_entry_ids"]:
            ancestor = (record["cell"], parent_entry_id)
            parent.setdefault(ancestor, ancestor)
            union(node, ancestor)
    return {
        record["record_id"]: "{}::{}".format(*find((record["cell"], record["entry_id"])))
        for record in records
    }


# ---- Weighting ----


def hierarchical_weights(records, contract: dict) -> dict[str, float]:
    """Equal total mass at every declared level, divided uniformly at the next."""
    hierarchy = contract["weights"]["hierarchy"]
    if hierarchy != ["target", "cell", "lineage_component", "record"]:
        raise ValueError("unsupported declared weight hierarchy")
    components = lineage_components(records)
    by_target = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        by_target[record["target"]][record["cell"]][components[record["record_id"]]].append(
            record["record_id"]
        )
    weights: dict[str, float] = {}
    for cells in by_target.values():
        target_mass = 1.0 / len(by_target)
        for lineages in cells.values():
            cell_mass = target_mass / len(cells)
            for ids in lineages.values():
                lineage_mass = cell_mass / len(lineages)
                for record_id in ids:
                    weights[record_id] = lineage_mass / len(ids)
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"hierarchical weights do not normalize: {total}")
    return weights


# ---- Census and gate ----


def census(records, assignment: dict) -> dict:
    """Per-fold training/held-out counts. Sides are reported separately, never pooled."""
    folds = sorted(set(assignment.values()))
    components = lineage_components(records)
    result = {}
    for fold in folds:
        sides = {"training": [], "heldout": []}
        for record in records:
            side = "heldout" if assignment[record["cell"]] == fold else "training"
            sides[side].append(record)
        result[fold] = {
            side: {
                "records": len(rows),
                "cells": sorted({r["cell"] for r in rows}),
                "targets": sorted({r["target"] for r in rows}),
                "source_contexts": len({r["input_state_sha256"] for r in rows}),
                "lineage_components": len({components[r["record_id"]] for r in rows}),
                "observations": sum(r["observation_count"] for r in rows),
                "bank_macro_records": sum(
                    BANK_MACRO_FAMILY in r["construction_families"] for r in rows
                ),
                "family_counts": dict(
                    sorted(
                        Counter(
                            family for r in rows for family in r["construction_families"]
                        ).items()
                    )
                ),
            }
            for side, rows in sides.items()
        }
    return result


def coverage_decision(fold_census: dict, contract: dict) -> dict:
    """GO only if every frozen hard criterion and every required family holds."""
    gate = contract["coverage_gate"]
    hard = gate["hard"]
    dense = gate["family_tiers"]["dense"]
    sparse = gate["family_tiers"]["sparse"]
    failures = []
    families = sorted(
        {
            family
            for fold in fold_census.values()
            for side in fold.values()
            for family in side["family_counts"]
        }
    )
    tiers = {}
    for family in families:
        minimum = min(f["training"]["family_counts"].get(family, 0) for f in fold_census.values())
        tiers[family] = {
            "minimum_training_records_across_folds": minimum,
            "tier": "dense" if minimum >= dense else ("sparse" if minimum >= sparse else "absent"),
        }
    for fold, sides in sorted(fold_census.items()):
        train, held = sides["training"], sides["heldout"]
        if train["records"] < hard["minimum_training_records_per_fold"]:
            failures.append(f"{fold}: training records {train['records']}")
        if len(train["cells"]) < hard["minimum_training_cells_per_fold"]:
            failures.append(f"{fold}: training cells {len(train['cells'])}")
        if held["records"] < hard["minimum_heldout_records_per_fold"]:
            failures.append(f"{fold}: held-out records {held['records']}")
        if held["source_contexts"] < hard["minimum_heldout_source_contexts_per_fold"]:
            failures.append(f"{fold}: held-out source contexts {held['source_contexts']}")
    for family in gate["required_dense_families"]:
        if tiers.get(family, {}).get("tier") != "dense":
            observed = tiers.get(family, {}).get("minimum_training_records_across_folds", 0)
            failures.append(f"required family {family} is not dense in every fold ({observed})")
    return {
        "decision": "GO" if not failures else "ABSTAIN",
        "failures": failures,
        "family_tiers": tiers,
        "supported_families": sorted(f for f, t in tiers.items() if t["tier"] == "dense"),
        "sparse_families": sorted(f for f, t in tiers.items() if t["tier"] == "sparse"),
    }


# ---- Build ----


def build(contract: dict, rows, checkpoint_manifest: dict) -> dict:
    """Deterministic end-to-end preparation; raises on any leakage violation."""
    assignment = fold_assignment(contract)
    evidence = group_disjointness(rows)
    assert_group_disjointness(evidence)
    admitted, excluded = admit(rows, contract, checkpoint_manifest)
    records = collapse_observations(admitted)
    components = lineage_components(records)
    weights = hierarchical_weights(records, contract)
    for record in records:
        record["fold"] = assignment[record["cell"]]
        record["lineage_component"] = components[record["record_id"]]
        record["weight"] = weights[record["record_id"]]
    # Defence in depth: a component is keyed inside one cell and a cell is assigned
    # whole, so this cannot fire unless one of those two rules is later broken.
    cross = sorted(
        component for component, folds in _folds_per_component(records).items() if len(folds) > 1
    )
    if cross:
        raise ValueError(f"lineage components cross folds: {cross[:5]}")
    fold_census = census(records, assignment)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": _payload_sha256(contract),
        "input_rows": len(rows),
        "admitted_rows": len(admitted),
        "records": len(records),
        "excluded_rows": len(excluded),
        "exclusion_counts": dict(sorted(Counter(e["reason"] for e in excluded).items())),
        "disjointness_evidence": evidence,
        "fold_assignment": assignment,
        "census": fold_census,
        "gate": coverage_decision(fold_census, contract),
        "weight_summary": _weight_summary(records),
        "heldout_power": heldout_power(records),
        "lineage_component_sizes": _component_sizes(records),
        "new_oracle_calls": 0,
        "new_labels": 0,
    }, records


def _component_sizes(records) -> dict:
    sizes = sorted(Counter(record["lineage_component"] for record in records).values())
    return {
        "components": len(sizes),
        "minimum": sizes[0],
        "median": sizes[len(sizes) // 2],
        "maximum": sizes[-1],
        "singletons": sum(1 for size in sizes if size == 1),
    }


def _folds_per_component(records) -> dict[str, set]:
    folds = defaultdict(set)
    for record in records:
        folds[record["lineage_component"]].add(record["fold"])
    return folds


def _weight_summary(records) -> dict:
    by_target = defaultdict(float)
    by_cell = defaultdict(float)
    for record in records:
        by_target[record["target"]] += record["weight"]
        by_cell[record["cell"]] += record["weight"]
    weights = [record["weight"] for record in weight_ordered(records)]
    uniform = 1.0 / len(records)
    return {
        "per_target": dict(sorted(by_target.items())),
        "per_cell": dict(sorted(by_cell.items())),
        "maximum_record_weight": weights[0],
        "minimum_record_weight": weights[-1],
        "maximum_over_uniform": weights[0] / uniform,
        "mass_in_heaviest_100_records": sum(weights[:100]),
        "records_above_100x_uniform": sum(1 for w in weights if w >= 100 * uniform),
        "effective_sample_size": effective_sample_size(weights),
        "record_count": len(records),
        "interpretation": (
            "diagnostic only; the weight hierarchy is frozen by the contract and was not "
            "changed after this measurement"
        ),
    }


def weight_ordered(records) -> list[dict]:
    return sorted(records, key=lambda record: -record["weight"])


def effective_sample_size(weights) -> float:
    """Kish effective sample size of a weight vector."""
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    return 0.0 if squared == 0 else (total * total) / squared


def heldout_power(records) -> dict:
    """How many independent units each fold's held-out side really contains.

    Adaptive search descends from one bootstrap root, so a cell's thousands of
    scored constructions collapse into very few genealogies. Record counts badly
    overstate the statistical power of a held-out fold.
    """
    sides = defaultdict(list)
    for record in records:
        sides[record["fold"]].append(record)
    result = {}
    for fold, rows in sorted(sides.items()):
        weights = [row["weight"] for row in rows]
        result[fold] = {
            "records": len(rows),
            "lineage_components": len({row["lineage_component"] for row in rows}),
            "source_contexts": len({row["input_state_sha256"] for row in rows}),
            "weighted_effective_sample_size": effective_sample_size(weights),
        }
    return result


def _payload_sha256(payload) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_records(path: Path, records) -> str:
    """Byte-stable gzip of the record table, ordered by its content-addressed id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"".join(
        canonical_bytes(record) + b"\n" for record in sorted(records, key=lambda r: r["record_id"])
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(filename="", mode="wb", fileobj=temporary.open("wb"), mtime=0) as handle:
        handle.write(body)
    temporary.replace(path)
    return hashlib.sha256(body).hexdigest()
