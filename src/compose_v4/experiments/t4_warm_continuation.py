"""Exact-state warm continuation of the frozen guided T4 development cell."""

from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.region_rewrite import RewriteContext
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    _index_preserving_rdkit_molecule,
)
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    canonical_bytes,
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.production_successor_kernel import canonical_state_key
from compose_v4.experiments.t4_endpoint_selection import (
    LEGACY_RANK_ALL,
    T4_FEASIBLE_ONLY,
    feasible_endpoint,
    validate_policy,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

CONTRACT_PATH = "configs/t4_warm_continuation.json"


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _atom_output_order(molecule) -> list[int]:
    # RDKit 2024 serializes this property as a Python list with a trailing
    # comma; newer releases emit JSON-compatible lists. Both carry the same
    # permutation. Parse literals, never evaluate code or alter the ordering.
    order = ast.literal_eval(molecule.GetProp("_smilesAtomOutputOrder"))
    if (
        not isinstance(order, list)
        or any(type(index) is not int for index in order)
        or sorted(order) != list(range(molecule.GetNumAtoms()))
    ):
        raise ValueError("RDKit _smilesAtomOutputOrder is not a complete atom permutation")
    return order


def canonical_slots(state, smiles: str) -> dict[int, int]:
    """Map SMILES metadata indices to saved slots; never reconstruct a state.

    Sampling still uses the qualified SMILES region order. Canonical output
    orders align two views of the same molecule, including sparse slot layouts.
    """
    molecule, slot_to_rdkit = _index_preserving_rdkit_molecule(state)
    metadata = Chem.MolFromSmiles(smiles)
    if metadata is None or Chem.MolToSmiles(metadata) != Chem.MolToSmiles(molecule):
        raise ValueError("parent SMILES metadata differs from exact saved state")
    inverse = {index: slot for slot, index in slot_to_rdkit.items()}
    saved_order = _atom_output_order(molecule)
    metadata_order = _atom_output_order(metadata)
    mapping = {a: inverse[b] for a, b in zip(metadata_order, saved_order, strict=True)}
    if set(mapping.values()) != set(slot_to_rdkit):
        raise ValueError("region-to-slot mapping is not a complete bijection")
    return mapping


def exact_context(state, smiles: str, region) -> RewriteContext:
    mapping = canonical_slots(state, smiles)
    locus = frozenset(mapping[a] for a in region.atoms)
    real = frozenset(int(i) for i in np.flatnonzero(is_element(state.atom_types)))
    return RewriteContext(
        frozen=real - locus,
        locus=locus,
        terminals=tuple(
            (mapping[out], mapping[inside], order) for inside, out, order in region.boundary
        ),
        interface=region.interface,
        k_components=region.n_context_components,
        phase="grow_new" if region.interface == "splitting" else "prune_old",
    )


def endpoint(lock: dict, candidate: dict) -> dict:
    """Recover the exact selected step, not a later recurrence of its SMILES."""
    matches = [
        row["product"]
        for unit in lock["work"]
        for row in unit["sampled_transitions"]
        if row["bundle_id"] == candidate["bundle_id"]
        and row["step"] == candidate["step"]
        and row["canonical_product"] == candidate["smiles"]
        # Repeated draws of the same bundle can reach the same canonical
        # molecule at the same depth with different persistent slot layouts.
        # New warm candidates carry the exact committed state; use it to
        # disambiguate, never pick an arbitrary canonical match.
        and ("state" not in candidate or row["product"] == candidate["state"])
    ]
    if not matches or any(value != matches[0] for value in matches):
        raise ValueError("selected candidate has missing or ambiguous exact endpoint")
    state = decode_state(matches[0])
    if canonical_state_key(state) != candidate["smiles"]:
        raise ValueError("saved endpoint canonical identity mismatch")
    if not 1 <= state.n_real_atoms <= 40 or len(state.atom_types) != 48:
        raise ValueError("saved endpoint exceeds the frozen editing support")
    if "state" in candidate and candidate["state"] != matches[0]:
        raise ValueError("candidate state differs from its sampled transition")
    return matches[0]


def initial_archive(lock: dict, docking: dict) -> dict:
    if lock["round"] != 1 or lock["task"]["primitive_guidance"] != "committor":
        raise ValueError("warm source must be the guided first-round lock")
    if len(lock["take"]) != 20 or len(docking["docked"]) != 20:
        raise ValueError("warm source must retain all 20 first-round attempts")
    sources = [
        r["source"] for u in lock["work"] for r in u["sampled_transitions"] if r["step"] == 1
    ]
    if not sources or any(s != sources[0] for s in sources):
        raise ValueError("cold-start lock does not have one exact source state")
    seed = lock["task"]["smiles"]
    canonical_slots(decode_state(sources[0]), seed)
    warm = {
        "schema_version": "t4_exact_archive_v1",
        "round": 0,
        "archive": [{"smiles": seed, "state": sources[0], "ds": None, "round": 0, "ancestors": []}],
        "rng_state": lock["rng_state"],
        "oracle_attempts": 0,
    }
    return advance_archive(warm, lock, docking)


def advance_archive(warm: dict, lock: dict, docking: dict) -> dict:
    if lock["round"] != warm["round"] + 1:
        raise ValueError("nonconsecutive warm round")
    if len(lock["take"]) != len(docking["docked"]):
        raise ValueError("docking rows do not cover the full locked batch")
    archive = list(warm["archive"])
    parents = {r["smiles"]: r for r in archive}
    seen = {canonical_state_key(decode_state(r["state"])) for r in archive}
    for candidate, docked in zip(lock["take"], docking["docked"], strict=True):
        if {k: docked[k] for k in candidate} != candidate:
            raise ValueError("docking row differs from its locked candidate")
        if docked["ds"] is not None and not np.isfinite(docked["ds"]):
            raise ValueError("nonfinite docking score in archive")
        if candidate["smiles"] in seen:
            raise ValueError("previously evaluated canonical molecule in new batch")
        parent = parents[candidate["parent"]]
        archive.append(
            {
                **docked,
                "state": endpoint(lock, candidate),
                "round": lock["round"],
                "ancestors": [*parent["ancestors"], parent["smiles"]],
            }
        )
        seen.add(candidate["smiles"])
    return {
        **warm,
        "archive": archive,
        "round": lock["round"],
        "rng_state": lock["rng_state"],
        "oracle_attempts": warm["oracle_attempts"] + len(lock["take"]),
    }


def verify_round(lock: dict, warm: dict, task: dict) -> None:
    policy = task.get("endpoint_selection_policy", LEGACY_RANK_ALL)
    validate_policy(policy)
    if lock["task"] != task or lock["round"] != warm["round"] + 1:
        raise ValueError("round lock task or round mismatch")
    if lock["oracle_calls"] != 0 or len(lock["take"]) > task["per_round"]:
        raise ValueError("round lock violates zero-oracle preparation or batch cap")
    if lock["input_sha256"] != task["expected_input_sha256"]:
        raise ValueError("round lock changed frozen scientific inputs")
    if task.get("executor_calls_per_parent") is not None:
        shares = [unit["parent_budget"] for unit in lock["work"]]
        if len(shares) != 8 or any(
            share["limit"] != 2500 or not 0 <= share["executor_calls"] <= 2500 for share in shares
        ):
            raise ValueError("round must account for all eight bounded parent shares")
        if sum(share["executor_calls"] for share in shares) != lock["total_public_executor_calls"]:
            raise ValueError("parent shares do not reconcile with the public executor ledger")
    previous = {canonical_state_key(decode_state(c["state"])) for c in warm["archive"]}
    identities = [c["smiles"] for c in lock["take"]]
    if len(set(identities)) != len(identities) or previous.intersection(identities):
        raise ValueError("canonical duplicate in round lock")
    parents = {c["smiles"]: c["state"] for c in warm["archive"]}
    for bundle in lock["bundles"]:
        sources = [
            r["source"]
            for u in lock["work"]
            for r in u["sampled_transitions"]
            if r["bundle_id"] == bundle["bundle_id"] and r["step"] == 1
        ]
        if any(s != parents[bundle["parent"]] for s in sources):
            raise ValueError("proposal did not start from the saved exact parent")
    for candidate in lock["take"]:
        if policy == T4_FEASIBLE_ONLY and (
            not feasible_endpoint(candidate) or candidate.get("oracle_eligible") is not True
        ):
            raise ValueError("ineligible endpoint in strict T4 oracle lock")
        if (
            candidate["option"] in ("build_ring_system", "build_fused_ring")
            and not candidate["program_complete"]
        ):
            raise ValueError("incomplete compound program in oracle lock")
        endpoint(lock, candidate)


def prepare_round(task, warm, prepare, output, commit, progress):
    path = output / "candidate_lock.json"
    if path.exists():
        lock = unseal(path)
    else:
        if list(output.glob("executor_attempts_*.json")):
            raise RuntimeError("failed preparation receipt exists; audit before retry")
        cached = {}
        for path_ in sorted((output / "parents").glob("*.json")):
            unit = unseal(path_)
            if unit["task"] != task:
                raise ValueError("cached parent task differs")
            cached[unit["parent_index"]] = unit
        prior = max((u["cumulative_executor_calls"] for u in cached.values()), default=0)
        meter = ExecutorMeter(task["max_executor_applications"] - prior)

        def checkpoint(unit):
            if unit["parent_index"] not in cached:
                seal(
                    output / "parents" / f"{unit['parent_index']:02d}.json",
                    {**unit, "task": task, "cumulative_executor_calls": prior + meter.calls},
                )
                progress.update(
                    parent_complete=unit["parent_index"] + 1, executor_calls=prior + meter.calls
                )
                commit()

        start = time.perf_counter()
        try:
            with meter.instrument():
                lock = prepare(task, checkpoint, cached, warm)
        finally:
            publish_json(
                output / f"executor_attempts_{prior}.json",
                {"attempts": meter.attempts, "prior_calls": prior},
            )
            commit()
        lock.update(
            total_public_executor_calls=prior + meter.calls,
            prepare_wall_seconds=time.perf_counter() - start,
        )
        verify_round(lock, warm, task)
        seal(path, lock)
        commit()
    verify_round(lock, warm, task)
    return lock, sha256_file(path)


def run_continuation(
    task,
    prepare,
    dock,
    output: Path,
    *,
    source: Path,
    contract: dict,
    commit=lambda: None,
    progress=None,
):
    """Two locked batches, full archive reuse, no automatic failed-work retry."""
    if contract["additional_rounds"] != 2 or contract["compute"]["oracle_call_limit"] != 40:
        raise ValueError("only two rounds and 40 new oracle calls are authorized")
    endpoint_policy = contract.get("endpoint_selection_policy", LEGACY_RANK_ALL)
    validate_policy(endpoint_policy)
    progress = {} if progress is None else progress
    lock_path, dock_path = source / "candidate_lock.json", source / "docking.json"
    verify_file(lock_path, contract["source"]["candidate_lock_sha256"])
    verify_file(dock_path, contract["source"]["docking_sha256"])
    source_lock, source_docking = unseal(lock_path), unseal(dock_path)
    if source_docking["candidate_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("source docking is not bound to its candidate lock")
    for key, value in contract["task"].items():
        if source_lock["task"].get(key) != value or task.get(key) != value:
            raise ValueError(f"warm continuation changed frozen task field: {key}")
    if source_lock["input_sha256"] != contract["expected_input_sha256"]:
        raise ValueError("source scientific input identities differ")
    warm = initial_archive(source_lock, source_docking)
    return run_rounds(
        task,
        warm,
        prepare,
        dock,
        output,
        additional_rounds=2,
        endpoint_policy=endpoint_policy,
        source_identity=contract["source"],
        commit=commit,
        progress=progress,
    )


def run_rounds(
    task,
    warm,
    prepare,
    dock,
    output,
    *,
    additional_rounds,
    endpoint_policy,
    source_identity,
    commit=lambda: None,
    progress=None,
):
    """Shared locked-round execution; callers validate prospective authorization."""
    if additional_rounds not in (1, 2):
        raise ValueError("only one or two explicitly authorized rounds are supported")
    validate_policy(endpoint_policy)
    progress = {} if progress is None else progress
    initial_attempts = warm["oracle_attempts"]
    initial_round = warm["round"]
    feasible = [c for c in warm["archive"] if c["ds"] is not None and c["v"] <= 0]
    progress.update(best_score=min((c["ds"] for c in feasible), default=None))
    seal(output / "warm_start.json", warm)
    commit()
    results = []
    for rd in range(initial_round + 1, initial_round + additional_rounds + 1):
        directory = output / f"round_{rd}"
        round_task = {
            **task,
            "arm": "committor",
            "primitive_guidance": "committor",
            "prepare_only": True,
            "warm_start_sha256": payload_hash(warm),
            "endpoint_selection_policy": endpoint_policy,
        }
        progress.update(
            phase="prepare",
            round=rd,
            parent_complete=0,
            executor_calls=0,
            cumulative_oracle_attempts=warm["oracle_attempts"],
        )
        locked, digest = prepare_round(round_task, warm, prepare, directory, commit, progress)
        result_path, attempt_path = directory / "docking.json", directory / "docking_started.json"
        if result_path.exists():
            result = unseal(result_path)
        else:
            if attempt_path.exists():
                raise RuntimeError("interrupted oracle attempt; no automatic re-docking")
            progress.update(phase="docking", oracle_attempts=len(locked["take"]))
            publish_json(
                attempt_path,
                {
                    "candidate_lock_sha256": digest,
                    "started_at_utc": _stamp(),
                    "attempts": len(locked["take"]),
                },
            )
            commit()
            start = time.perf_counter()
            scores = dock([c["smiles"] for c in locked["take"]], f"round_{rd}")
            if len(scores) != len(locked["take"]) or any(
                s is not None and not np.isfinite(s) for s in scores
            ):
                raise ValueError("oracle returned missing rows or nonfinite scores")
            result = {
                "round": rd,
                "candidate_lock_sha256": digest,
                "docked": [{**c, "ds": s} for c, s in zip(locked["take"], scores)],
                "oracle_attempts": len(scores),
                "oracle_failures": scores.count(None),
                "docking_seconds": time.perf_counter() - start,
                "completed_at_utc": _stamp(),
            }
            seal(result_path, result)
            commit()
        if result["candidate_lock_sha256"] != digest:
            raise ValueError("docking result is not bound to its round lock")
        warm = advance_archive(warm, locked, result)
        checkpoint_path = directory / "archive.json"
        if checkpoint_path.exists() and unseal(checkpoint_path) != warm:
            raise ValueError("saved archive differs from deterministic round reduction")
        seal(checkpoint_path, warm)
        commit()
        feasible = [c for c in warm["archive"] if c["ds"] is not None and c["v"] <= 0]
        best = min(feasible, key=lambda c: c["ds"]) if feasible else None
        progress.update(
            phase="round_complete",
            best_score=best["ds"] if best else None,
            cumulative_oracle_attempts=warm["oracle_attempts"],
        )
        results.append({**result, "best_so_far": best})
    return {
        "schema_version": "t4_warm_continuation_v1",
        "status": "complete",
        "task": task,
        "source": source_identity,
        "rounds": results,
        "new_oracle_attempts": warm["oracle_attempts"] - initial_attempts,
        "cumulative_oracle_attempts": warm["oracle_attempts"],
        "interpretation_scope": "guided-only inspected development continuation",
    }


def run_remote(
    task, repo_root, artifact_root, volume, runtime_factory, validate_revision, prepare, dock
):
    from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote

    def runner(actual_task, prepare, dock, output, **kwargs):
        contract = json.loads((repo_root / CONTRACT_PATH).read_text())
        return run_continuation(
            actual_task,
            prepare,
            dock,
            output,
            source=artifact_root / contract["source"]["volume_path"],
            contract=contract,
            **kwargs,
        )

    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        prepare,
        dock,
        contract_path=CONTRACT_PATH,
        run_kind="t4_warm_continuation",
        runner=runner,
    )
