from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_packed_charge_policy import _atomic_json_write  # noqa: E402
from compose_v4.chem.molecular_graph import (  # noqa: E402
    BOND_DOUBLE,
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    MolecularGraph,
    NULL_IDX,
)
from compose_v4.data.charge_policy import (  # noqa: E402
    CHARGED_CENTER_BOND_ROW_CHANGED,
    CHARGED_CENTER_ELEMENT_CHANGED,
    CHARGED_CENTER_H_CHANGED,
    FORMAL_CHARGE_ARRAY_CHANGED,
    audit_charge_policy_transition,
    charge_policy_preserved,
)
from compose_v4.data.packed_charge_policy_audit import (  # noqa: E402
    DeclaredPackedShard,
    audit_packed_charge_policy,
    resolve_unified_manifest_shards,
)
from compose_v4.data.packed_trace_store import write_packed_shard  # noqa: E402
from compose_v4.rewrite.action_codec import encode_action  # noqa: E402
from compose_v4.rewrite.operators import (  # noqa: E402
    AtomRestate,
    BondReorder,
)
from compose_v4.rewrite.trace_shard import encode_state  # noqa: E402


def _state(
    *,
    charges=(1, 0, 0, 0),
    atom_types=None,
    hydrogens=(2, 3, 0, 0),
    bond_order=BOND_SINGLE,
) -> MolecularGraph:
    n = 4
    atoms = np.asarray(
        atom_types
        or (
            ELEMENT_TO_IDX["N"],
            ELEMENT_TO_IDX["C"],
            NULL_IDX,
            NULL_IDX,
        ),
        dtype=np.int32,
    )
    bonds = np.zeros((n, n), dtype=np.int32)
    if bond_order:
        bonds[0, 1] = bonds[1, 0] = bond_order
    return MolecularGraph(
        atom_types=atoms,
        formal_charges=np.asarray(charges, dtype=np.int32),
        implicit_h_counts=np.asarray(hydrogens, dtype=np.int32),
        bonds=bonds,
    )


def _entry(
    trace_id: str,
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    rule: str,
    action,
) -> dict:
    return {
        "trace": {
            "trace_id": trace_id,
            "layer": "mmp_analogue",
            "partition": "validation",
            "source_key": f"{trace_id}-source",
            "target_key": f"{trace_id}-target",
            "path_length": 1,
            "steps": [{"action": encode_action(rule, action)}],
            "metadata": {},
        },
        "states": [encode_state(source), encode_state(target)],
    }


def _synthetic_shard(tmp_path: Path) -> Path:
    source = _state()
    clean = _state(
        atom_types=(
            ELEMENT_TO_IDX["N"],
            ELEMENT_TO_IDX["O"],
            NULL_IDX,
            NULL_IDX,
        ),
        hydrogens=(2, 1, 0, 0),
    )
    charge_changed = _state(charges=(0, 1, 0, 0))
    protected_payload_changed = _state(
        atom_types=(
            ELEMENT_TO_IDX["O"],
            ELEMENT_TO_IDX["C"],
            NULL_IDX,
            NULL_IDX,
        ),
        hydrogens=(1, 3, 0, 0),
    )
    protected_bond_changed = _state(bond_order=BOND_DOUBLE)
    entries = [
        _entry(
            "clean",
            source,
            clean,
            rule="atom_restate",
            action=AtomRestate(
                v=1,
                atom_type=ELEMENT_TO_IDX["O"],
                formal_charge=0,
                implicit_h_count=1,
            ),
        ),
        _entry(
            "charge-array",
            source,
            charge_changed,
            rule="atom_restate",
            action=AtomRestate(
                v=0,
                atom_type=ELEMENT_TO_IDX["N"],
                formal_charge=0,
                implicit_h_count=2,
            ),
        ),
        _entry(
            "protected-payload",
            source,
            protected_payload_changed,
            rule="atom_restate",
            action=AtomRestate(
                v=0,
                atom_type=ELEMENT_TO_IDX["O"],
                formal_charge=1,
                implicit_h_count=1,
            ),
        ),
        _entry(
            "protected-bond",
            source,
            protected_bond_changed,
            rule="bond_reorder",
            action=BondReorder(a=0, b=1, new_order=BOND_DOUBLE),
        ),
    ]
    shard = tmp_path / "validation" / "shard_0000.jsonl.gz"
    write_packed_shard(shard, entries, provenance={"capability_hash": "audit-test"})
    return shard


def test_authoritative_predicate_detects_each_exact_policy_component() -> None:
    source = _state()
    assert charge_policy_preserved(
        source,
        _state(
            atom_types=(
                ELEMENT_TO_IDX["N"],
                ELEMENT_TO_IDX["O"],
                NULL_IDX,
                NULL_IDX,
            ),
            hydrogens=(2, 1, 0, 0),
        ),
    )

    charge = audit_charge_policy_transition(source, _state(charges=(0, 1, 0, 0)))
    assert charge.violation_types == (FORMAL_CHARGE_ARRAY_CHANGED,)
    payload = audit_charge_policy_transition(
        source,
        _state(
            atom_types=(
                ELEMENT_TO_IDX["O"],
                ELEMENT_TO_IDX["C"],
                NULL_IDX,
                NULL_IDX,
            ),
            hydrogens=(1, 3, 0, 0),
        ),
    )
    assert payload.violation_types == (
        CHARGED_CENTER_ELEMENT_CHANGED,
        CHARGED_CENTER_H_CHANGED,
    )
    bond = audit_charge_policy_transition(
        source, _state(bond_order=BOND_DOUBLE)
    )
    assert bond.violation_types == (CHARGED_CENTER_BOND_ROW_CHANGED,)


def test_streaming_audit_reports_exact_addresses_and_bounded_examples(
    tmp_path: Path,
) -> None:
    shard = _synthetic_shard(tmp_path)
    declared = DeclaredPackedShard(
        manifest_layer="mmp_analogue",
        envelope_layer="mmp_analogue",
        partition="validation",
        relative_path="validation/shard_0000.jsonl.gz",
        path=shard,
    )
    report = audit_packed_charge_policy((declared,), max_examples=2)

    assert report["status"] == "FAIL_CHARGE_POLICY_VIOLATIONS"
    assert report["counts"] == {
        "shards": 1,
        "entries": 4,
        "states": 8,
        "transitions": 4,
        "traces_with_violations": 3,
        "transitions_with_violations": 3,
        "violation_events": 4,
    }
    assert report["violating_transitions_by_type"] == {
        FORMAL_CHARGE_ARRAY_CHANGED: 1,
        CHARGED_CENTER_ELEMENT_CHANGED: 1,
        CHARGED_CENTER_H_CHANGED: 1,
        CHARGED_CENTER_BOND_ROW_CHANGED: 1,
    }
    assert report["unique_violating_traces_by_type"] == {
        FORMAL_CHARGE_ARRAY_CHANGED: 1,
        CHARGED_CENTER_ELEMENT_CHANGED: 1,
        CHARGED_CENTER_H_CHANGED: 1,
        CHARGED_CENTER_BOND_ROW_CHANGED: 1,
    }
    assert report["violation_groups"]["formal_charge_coordinate_mutation"][
        "transitions"
    ] == 1
    assert report["violation_groups"]["formal_charge_coordinate_mutation"][
        "unique_traces"
    ] == 1
    assert report["violation_groups"]["formal_charge_coordinate_mutation"][
        "transition_subtypes"
    ] == {
        "formal_charge_created": 1,
        "formal_charge_deleted": 1,
        "formal_charge_nonzero_value_changed": 0,
    }
    assert report["violation_groups"]["protected_charged_center_mutation"][
        "transitions"
    ] == 2
    assert report["violation_groups"]["protected_charged_center_mutation"][
        "unique_traces"
    ] == 2
    assert report["shards"][0]["violating_entry_indices"] == [1, 2, 3]
    exclusions = report["exclusion_payload"]
    assert exclusions["excluded_unique_trace_addresses"] == 3
    assert exclusions["shards"][0]["violating_entry_indices"] == [1, 2, 3]
    assert [
        entry["trace_id"] for entry in exclusions["shards"][0]["entries"]
    ] == ["charge-array", "protected-payload", "protected-bond"]
    assert [
        entry["violation_type_mask"]
        for entry in exclusions["shards"][0]["entries"]
    ] == [1, 6, 8]
    payload_body = {
        key: value for key, value in exclusions.items() if key != "payload_sha256"
    }
    assert exclusions["payload_sha256"] == hashlib.sha256(
        json.dumps(
            payload_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert len(report["examples"]) == 2
    first = report["examples"][0]
    assert first["address"]["entry_index"] == 1
    assert first["address"]["trace_id"] == "charge-array"
    assert first["address"]["progress_index"] == 0
    assert len(first["address"]["packed_shard_content_sha256"]) == 64
    assert first["action"]["model_family"] == "atom_restate"
    assert len(first["before_persistent_state_sha256"]) == 64
    assert report["method"] == {
        "state_source": "exact_packed_consecutive_states",
        "canonical_smiles_used": False,
        "rdkit_used": False,
        "executor_replay_used": False,
        "corpus_mutated": False,
    }


def test_unified_manifest_inventory_and_atomic_output(tmp_path: Path) -> None:
    mmp_root = tmp_path / "mmp"
    _synthetic_shard(mmp_root)
    manifest = {
        "artifact": "unified_packed_corpus_manifest",
        "roots": {
            "audit_layers": "/artifact/not-present",
            "mmp_layer": "/artifact/not-present",
        },
        "layers": {
            "mmp_analogue": {
                "train": [],
                "validation": ["validation/shard_0000.jsonl.gz"],
                "test": [],
            }
        },
        "totals": {"shards": 1, "entries": 4, "states": 8},
        "manifest_checksum": "fixture",
    }
    manifest_path = tmp_path / "UNIFIED_PACKED_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    loaded, shards = resolve_unified_manifest_shards(
        manifest_path,
        mmp_root=mmp_root,
    )
    report = audit_packed_charge_policy(
        shards,
        source_manifest_path=manifest_path,
        source_manifest=loaded,
    )
    output = tmp_path / "nested" / "audit.json"
    _atomic_json_write(output, report)

    assert json.loads(output.read_text())["input_inventory_sha256"] == report[
        "input_inventory_sha256"
    ]
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert not list(output.parent.glob("*.tmp"))
