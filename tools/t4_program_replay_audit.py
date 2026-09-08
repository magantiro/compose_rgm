"""Replay completed pendant programs from sealed T4 locks and saved executor marks.

This validates sampled state paths, not probabilities or exhaustive reachability.
The transition trace omits the selected mark identity; a matching recorded mark
is an execution witness, not evidence that this exact mark was sampled.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control import region_rewrite as RR
from compose_v4.control.macro_engine import MACRO_FAMILIES, slot_ring_systems, state_contract_for
from compose_v4.control.option_selector import option_horizon, primitive_option_at_step
from compose_v4.control.region import enumerate_regions
from compose_v4.experiments.continuation_profile import canonical_bytes, sha256_file
from compose_v4.experiments.t4_matched_pilot import ARMS, unseal, verify_pair
from compose_v4.gates.med_chem_gate import is_valid
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def verify_transition(transition, attempt, macro, origin, context, system):
    """Fail closed on a mismatched ledger entry, replay, or production predicate."""
    if attempt["status"] != "executed" or any(
        attempt[k] != transition[k] for k in ("source", "product")
    ):
        raise ValueError("ledger entry differs from the sampled exact transition")
    family, action = decode_action(attempt["mark"])
    source, product = (decode_state(transition[k]) for k in ("source", "product"))
    if family not in MACRO_FAMILIES[macro]:
        raise ValueError("execution witness lies outside the active macro family")
    if encode_state(system.apply(source, family, action)) != transition["product"]:
        raise ValueError("exact executor replay differs from the saved product")
    contract = state_contract_for(macro, source)
    key = canonical_state_key(product)
    checks = {
        "canonical_identity": key == transition["canonical_product"],
        "valid": is_valid(key),
        "connected": RR.graph_connected(product),
        "active_support": 0 < product.n_real_atoms <= 40,
        "region_admissible": bool(RR.admissible_indices([family], [action], context)[0]),
        "context_preserved": RR.context_preserved(
            origin, product, context.frozen, context.terminal_context_slots
        ),
        "macro_contract": contract is None or bool(contract(product)),
    }
    if not all(checks.values()):
        raise ValueError(f"sampled path failed production checks: {checks}")
    slot = RR.created_slot(action)
    return context.with_locus(slot) if slot is not None else context


def report(root: Path, ledger_root: Path) -> dict:
    locks = {arm: unseal(root / arm / "candidate_lock.json") for arm in ARMS}
    verify_pair(locks)
    inventory_path = root / "remote_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    hashes = {str(inventory_path): sha256_file(inventory_path)}
    system, rows = editing_v2_rewrite_system(), []
    for arm, lock in locks.items():
        ledger_path = ledger_root / f"{arm}.json"
        for path, remote in (
            (ledger_path, f"{arm}/executor_attempts_0.json"),
            (root / arm / "candidate_lock.json", f"{arm}/candidate_lock.json"),
        ):
            hashes[str(path)] = sha256_file(path)
            if hashes[str(path)] != inventory["files"][remote]["sha256"]:
                raise ValueError(f"remote artifact hash mismatch: {path}")
        attempts = {}
        for attempt in json.loads(ledger_path.read_text())["attempts"]:
            if attempt["status"] == "executed":
                key = canonical_bytes([attempt["source"], attempt["product"]])
                attempts.setdefault(key, []).append(attempt)
        for bundle in lock["bundles"]:
            if not bundle["option"].startswith("build_") or not bundle["program_complete"]:
                continue
            if bundle["option"] != "build_ring_system":
                raise ValueError("this bounded audit does not validate fused progress semantics")
            trace = sorted(
                (
                    t
                    for w in lock["work"]
                    for t in w["sampled_transitions"]
                    if t["bundle_id"] == bundle["bundle_id"]
                ),
                key=lambda t: t["step"],
            )
            horizon = option_horizon(bundle["option"], lock["task"]["max_handoff"])
            if [t["step"] for t in trace] != list(range(1, horizon + 1)):
                raise ValueError("completed single-particle program lacks its full ordered path")
            origin = decode_state(trace[0]["source"])
            regions = [
                r
                for r in enumerate_regions(bundle["parent"])
                if sorted(r.atoms) == bundle["region_atoms"]
            ]
            if len(regions) != 1:
                raise ValueError("saved region does not resolve uniquely")
            context = RR.context_from_region(regions[0])
            previous, steps = trace[0]["source"], []
            for t in trace:
                if t["source"] != previous:
                    raise ValueError("sampled exact-state path is discontinuous")
                matches = attempts.get(canonical_bytes([t["source"], t["product"]]), [])
                macro = primitive_option_at_step(bundle["option"], t["step"] - 1)
                matches = [
                    a for a in matches if decode_action(a["mark"])[0] in MACRO_FAMILIES[macro]
                ]
                if not matches:
                    raise ValueError("sampled transition has no recorded admissible mark witness")
                context = verify_transition(t, matches[0], macro, origin, context, system)
                steps.append(
                    {
                        "step": t["step"],
                        "macro": macro,
                        "mark": matches[0]["mark"],
                        "canonical_product": t["canonical_product"],
                        "distinct_matching_marks": len(
                            {canonical_bytes(a["mark"]) for a in matches}
                        ),
                    }
                )
                previous = t["product"]
            endpoint = decode_state(previous)
            candidates = [c for c in lock["pool"] if c["bundle_id"] == bundle["bundle_id"]]
            if not candidates or any(
                c["smiles"] != canonical_state_key(endpoint) for c in candidates
            ):
                raise ValueError("emitted endpoint differs from the completed exact path")
            old_ring_atoms = frozenset().union(*slot_ring_systems(origin))
            rows.append(
                {
                    "arm": arm,
                    "bundle_id": bundle["bundle_id"],
                    "steps": steps,
                    "new_disjoint_ring_system_sizes": sorted(
                        len(s) for s in slot_ring_systems(endpoint) if s.isdisjoint(old_ring_atoms)
                    ),
                    "docked": any(c["bundle_id"] == bundle["bundle_id"] for c in lock["take"]),
                    "candidate": candidates[0],
                    "all_checks_passed": True,
                }
            )
    return {
        "schema_version": "t4_pendant_program_replay_v1",
        "programs": rows,
        "input_sha256": hashes,
        "configuration": {arm: lock["task"] for arm, lock in locks.items()},
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "machine": platform.machine(),
            "device": "cpu",
            "precision": "integer graph replay",
        },
        "new_model_calls": 0,
        "new_oracle_calls": 0,
        "executor_replays": sum(len(row["steps"]) for row in rows),
        "evidence_scope": "all completed pendant programs in one inspected paired development round; no reachability or affinity generalization",
        "mark_identity_limitation": "a recorded matching mark witnesses each sampled exact-state transition; the trace omits the sampled mark identity",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("ledger_root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(report(args.root, args.ledger_root), sort_keys=True, indent=2, allow_nan=False)
    )
