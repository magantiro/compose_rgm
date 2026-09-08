"""Post-hoc analysis of saved closures; no generator, oracle, or contract repair.

Usage: python tools/t4_closure_diagnosis.py AUDIT_DIR LEDGER_DIR
LEDGER_DIR contains reference.json and committor.json, copied from each arm's
executor_attempts_0.json on the volume identified by remote_inventory.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control import region_rewrite as RR
from compose_v4.control.macro_engine import APPEND_MIN_NEW_ATOMS, append_system_closure
from compose_v4.control.region import enumerate_regions
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import ARMS, unseal
from compose_v4.gates.med_chem_gate import is_valid
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def slot_ring_systems(payload: dict) -> list[set[int]]:
    state = decode_state(payload)
    nodes = [int(i) for i in np.flatnonzero(is_element(state.atom_types))]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((i, j) for i in nodes for j in nodes if i < j and state.bonds[i, j])
    graph.remove_edges_from(list(nx.bridges(graph)))
    return [set(c) for c in nx.connected_components(graph) if len(c) > 1]


def region_identity(region) -> str:
    return hashlib.sha256(
        json.dumps(
            [list(region.key()[0]), [list(x) for x in region.key()[1]]],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]


def diagnose(root: Path, ledger_root: Path) -> dict:
    inventory = json.loads((root / "remote_inventory.json").read_text())
    system, replayed, rows, witnesses, hashes = editing_v2_rewrite_system(), {}, {}, [], {}
    for arm in ARMS:
        path = ledger_root / f"{arm}.json"
        hashes[str(path)] = sha256_file(path)
        if hashes[str(path)] != inventory["files"][f"{arm}/executor_attempts_0.json"]["sha256"]:
            raise ValueError(f"ledger hash mismatch: {path}")
        lock_path = root / arm / "candidate_lock.json"
        hashes[str(lock_path)] = sha256_file(lock_path)
        lock = unseal(lock_path)
        trace = [t for w in lock["work"] for t in w["sampled_transitions"]]
        ends = {
            canonical_bytes(t["product"]): t
            for t in trace
            if t["option"] == "build_ring_system" and t["step"] == 8
        }
        rows[arm] = {t["bundle_id"]: [] for t in ends.values()}
        for attempt in json.loads(path.read_text())["attempts"]:
            key = canonical_bytes(attempt["source"])
            if (
                key not in ends
                or attempt["status"] != "executed"
                or attempt["mark"].get("executor_rule") not in ("cycle_close", "bond_insert")
            ):
                continue
            bundle = ends[key]["bundle_id"]
            before, after = (slot_ring_systems(attempt[k]) for k in ("source", "product"))
            old = set().union(*before)
            new = [s for s in after if not s & old]
            exact_accept = len(after) > len(before) and any(
                len(s) >= APPEND_MIN_NEW_ATOMS for s in new
            )
            source, product = (decode_state(attempt[k]) for k in ("source", "product"))
            before_smiles, smiles = canonical_state_key(source), canonical_state_key(product)
            legacy_accept = append_system_closure(before_smiles)(smiles)
            row = {
                "mark": attempt["mark"],
                "new_system_sizes": sorted(len(s) for s in new),
                "exact_slot_contract": exact_accept,
                "smiles_contract": legacy_accept,
                "product": smiles,
            }
            rows[arm][bundle].append(row)
            if not exact_accept or legacy_accept:
                continue
            origin = decode_state(next(t for t in trace if t["bundle_id"] == bundle)["source"])
            receipt = next(b for b in lock["bundles"] if b["bundle_id"] == bundle)
            region = next(
                r
                for r in enumerate_regions(receipt["parent"])
                if region_identity(r) == receipt["region_id"]
            )
            ctx = RR.context_from_region(region)
            for slot in np.flatnonzero(
                is_element(source.atom_types) & ~is_element(origin.atom_types)
            ):
                ctx = ctx.with_locus(int(slot))
            family, mark = decode_action(attempt["mark"])
            identity = hashlib.sha256(
                canonical_bytes([attempt["source"], attempt["mark"]])
            ).hexdigest()
            if identity not in replayed:
                replayed[identity] = encode_state(system.apply(source, family, mark))
            checks = {
                "exact_replay_matches": replayed[identity] == attempt["product"],
                "region_admissible": bool(RR.admissible_indices([family], [mark], ctx)[0]),
                "frozen_context_preserved": RR.context_preserved(
                    origin, product, ctx.frozen, ctx.terminal_context_slots
                ),
                "valid": is_valid(smiles),
                "connected": RR.graph_connected(product),
                "within_active_support": 0 < product.n_real_atoms <= 40,
            }
            if not all(checks.values()):
                raise ValueError(f"false-rejection witness fails another requirement: {checks}")
            witnesses.append({"arm": arm, "bundle_id": bundle, **row, "checks": checks})
    return {
        "schema_version": "t4_saved_closure_diagnosis_v1",
        "rows": rows,
        "witnesses": witnesses,
        "new_model_calls": 0,
        "new_oracle_calls": 0,
        "replayed_unique_executor_calls": len(replayed),
        "minimum_new_system_atoms": APPEND_MIN_NEW_ATOMS,
        "input_sha256": hashes,
        "remote_inventory_sha256": sha256_file(root / "remote_inventory.json"),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "source_revision": json.loads((root / "result.json").read_text())["code_revision"],
        "review_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "networkx": nx.__version__,
        },
        "evidence_scope": "saved counterfactuals at three failed growth endpoints per arm; not an exhaustive reachability or affinity result",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("ledger_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.root, args.ledger_root), sort_keys=True, indent=2))
