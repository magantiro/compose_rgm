"""Zero-oracle production-law repair of the five saved development routes."""

import json
import subprocess
import threading
from itertools import pairwise
from time import perf_counter

from compose_v4.control.docking_value import identity
from compose_v4.control.supported_route import (
    BridgeConfig,
    exact_key,
    lower_supported,
    transported_target,
)
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import dependency_sources, software
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import worker_identity
from compose_v4.experiments.pmo_online_policy import runtime
from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "pmo_route_support"
APP_NAME = "compose-pmo-route-support"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"


def development_revision(root):
    """Explicitly uncommitted source snapshot, authorized for this repair only."""
    from modal_apps.run_process_v2_p50_app import _serialized_source_paths

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    body = {
        "schema": "compose.route_support_development_snapshot_v1",
        "commit": commit,
        "worktree_clean": False,
        "serialized_sources": {p: sha256_file(root / p) for p in _serialized_source_paths(root)},
    }
    return {**body, "image_revision_sha256": identity(body)}


def validate_development_revision(revision, root):
    from modal_apps.run_process_v2_p50_app import _serialized_source_paths

    body = {k: v for k, v in revision.items() if k != "image_revision_sha256"}
    if (
        revision["schema"] != "compose.route_support_development_snapshot_v1"
        or revision["worktree_clean"] is not False
        or identity(body) != revision["image_revision_sha256"]
        or set(revision["serialized_sources"]) != set(_serialized_source_paths(root))
    ):
        raise ValueError("development source snapshot identity mismatch")
    for p, h in revision["serialized_sources"].items():
        verify_file(root / p, h)


def prepare(root, prior_path):
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "spawned":
        raise ValueError("prior coverage requires a successful spawn receipt")
    c0 = json.loads((root / "configs/pmo_trajectory_value.json").read_text())
    if prior["task"]["contract_sha256"] != c0["contract_sha256"]:
        raise ValueError("prior receipt and coverage contract differ")
    prior_dest = root / f"diagnostics/{KIND}/prior_spawn.json"
    publish_json(prior_dest, prior)
    manifest = root / "diagnostics/pmo_public_winner_recovery/result.json"
    rows = json.loads(manifest.read_text())["paths"]
    inputs = {str(manifest.relative_to(root)): sha256_file(manifest)}
    inputs.update({r["path"]: r["sha256"] for r in rows})
    for p in (
        "configs/pmo_trajectory_value.json",
        "diagnostics/pmo_trajectory_value/prepared.json",
        str(prior_dest.relative_to(root)),
        "docs/PMO_ROUTE_SUPPORT_REPAIR.md",
        "docs/PMO_ROUTE_SUPPORT_DEV_SNAPSHOT.md",
    ):
        inputs[p] = sha256_file(root / p)
    c = {
        "schema_version": "production_route_repair_v1",
        "authorization": "2026-09-11 user requests first proposal-support repair",
        "inputs": inputs,
        "sources": rows,
        "prior_receipt": str(prior_dest.relative_to(root)),
        "runtime_contract_sha256": c0["runtime_contract_sha256"],
        "expected_input_sha256": c0["expected_input_sha256"],
        "law_caches": [],
        "oracle_limit": 0,
        "training": False,
        "max_new_laws_per_source": 160,
        "bridge_expansions": 16,
        "bridge_steps": 8,
        "workers": 5,
        "seed": None,
        "deterministic_compiler": True,
    }
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("route-support contract hash mismatch")
    if c["oracle_limit"] != 0 or c["training"] or c["workers"] != 5:
        raise ValueError("route support has no oracle/training authority")
    for p, h in c["inputs"].items():
        verify_file(root / p, h)
    old = json.loads((root / c["prior_receipt"]).read_text())
    for p, h in dependency_sources(old["task"]["image_revision"]["serialized_sources"]).items():
        verify_file(root / p, h)
    return c, old


class LawLimit(RuntimeError):
    pass


def worker_remote(task, root, artifact_root, volume, validate):
    validate(task["image_revision"])
    c, prior = load_contract(root)
    body = {k: task[k] for k in ("contract_sha256", "image_revision", "app_sha256")}
    if identity(body) != task["run_id"] or c["contract_sha256"] != task["contract_sha256"]:
        raise ValueError("route task identity mismatch")
    verify_file(root / APP, task["app_sha256"])
    row = c["sources"][task["slot"]]
    output = artifact_root / KIND / task["run_id"] / row["source_id"]
    volume.reload()
    lock, stop = threading.RLock(), threading.Event()

    def commit():
        with lock:
            volume.commit()

    store = Store(output, commit)
    previous = store.read("result")
    if previous is not None:
        return previous
    start = perf_counter()
    progress = {
        "phase": "initialization",
        "source": row["source_id"],
        "completed_transitions": 0,
        "oracle_calls": 0,
    }

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "at": _stamp(), "seconds": perf_counter() - start},
            )
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        rt = runtime(root, artifact_root, c["runtime_contract_sha256"])
        saved_law = SavedMarkedLaw(
            rt["model"],
            output,
            store.save,
            store.read,
            repo_root=root,
            artifact_root=artifact_root,
            contract=c,
            progress=progress,
        )
        audits = json.loads((root / "diagnostics/pmo_trajectory_value/prepared.json").read_text())[
            "audits"
        ]
        old_slot = next(i for i, a in enumerate(audits) if a["id"] == row["source_id"])
        old_worker = worker_identity(0, old_slot, audits[old_slot])
        old_dir = artifact_root / prior["prefix"] / "workers" / old_worker

        def law(graph):
            key = exact_key(graph)
            name = f"laws/{key}"
            if key not in saved_law.memory and store.read(name) is None:
                old_path = old_dir / f"{name}.json"
                if old_path.exists():
                    payload = unseal(old_path)
                    if payload["source"] != encode_state(graph):
                        raise ValueError("old law exact state mismatch")
                    store.save(name, payload)
                    store.save(
                        f"reuse/{key}", {"path": str(old_path), "sha256": sha256_file(old_path)}
                    )
                elif saved_law.counts["fresh_laws"] >= c["max_new_laws_per_source"]:
                    raise LawLimit("bounded source law allowance exhausted")
            return saved_law(graph)

        original = json.loads((root / row["path"]).read_text())["result"]
        originals = [decode_state(s) for s in original["states"]]
        current = originals[0]
        order = list(range(current.n_atoms))
        steps, transitions = [], []
        status = "supported_route"
        for i, (before, after) in enumerate(pairwise(originals)):
            goal, new_order = transported_target(before, after, current, order)
            receipt = store.read(f"transitions/{i:03}")
            if receipt is None:
                try:
                    result = lower_supported(
                        current,
                        goal,
                        law,
                        rt["system"],
                        config=BridgeConfig(c["bridge_expansions"], c["bridge_steps"]),
                    )
                except LawLimit as exc:
                    result = {"status": "law_allowance_unresolved", "steps": [], "reason": str(exc)}
                receipt = {
                    "original_step": i,
                    "source_key": exact_key(current),
                    "goal_key": exact_key(goal),
                    "slot_order": new_order,
                    **result,
                }
                store.save(f"transitions/{i:03}", receipt)
            if receipt["source_key"] != exact_key(current) or receipt["goal_key"] != exact_key(
                goal
            ):
                raise ValueError("resumed transition has incompatible exact boundary")
            transitions.append(receipt)
            if receipt["status"] != "supported":
                status = "route_unresolved"
                break
            for step in receipt["steps"]:
                family, action = decode_action(step["action"])
                if (
                    exact_key(current) != step["source_key"]
                    or step["selected_mark_probability"] <= 0
                ):
                    raise ValueError("invalid positive-support witness")
                current = rt["system"].apply(current, family, action)
                if encode_state(current) != step["state"]:
                    raise ValueError("independent supported-path replay mismatch")
                steps.append(step)
            if exact_key(current) != exact_key(goal):
                raise ValueError("lowering did not rejoin exact intended state")
            order = new_order
            progress.update(
                phase="lowering",
                completed_transitions=i + 1,
                total_transitions=len(original["actions"]),
                primitives=len(steps),
            )
            print(
                f"[support] {row['source_id']} {i + 1}/{len(original['actions'])} primitives={len(steps)}",
                flush=True,
            )
        if status == "supported_route" and canonical_state_key(current) != canonical_state_key(
            originals[-1]
        ):
            raise ValueError("repaired route endpoint differs from target")
        result = {
            "schema_version": "supported_route_result_v1",
            "status": status,
            "source": row["source_id"],
            "original_steps": len(original["actions"]),
            "completed_original_steps": progress["completed_transitions"],
            "primitive_steps": len(steps),
            "source_state": original["states"][0],
            "endpoint": canonical_state_key(current),
            "target": canonical_state_key(originals[-1]),
            "steps": steps,
            "transitions": transitions,
            "law_work": saved_law.counts,
            "seconds": perf_counter() - start,
            "started_at": task["started_at"],
            "finished_at": _stamp(),
            "oracle_calls": 0,
            "configuration": c,
            "code_revision": task["image_revision"]["commit"],
            "run_id": task["run_id"],
            "software": software(),
            "hardware": {"device": "cpu", "threads": 1, "dtype": "float32 neural"},
            "model_capabilities": {
                k: getattr(rt["model"], k)
                for k in (
                    "editing_process_semantics",
                    "atom_delete_action_semantics",
                    "cycle_close_action_semantics",
                    "cycle_open_action_semantics",
                )
            },
            "scope": "known-target primitive support certificate, not blind search, macro selection probability, or shortest route",
        }
        store.save("result", result)
        return result
    except Exception as exc:
        store.save("failure", {"error": repr(exc), "progress": progress, "at": _stamp()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)
