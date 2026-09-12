"""Lock and compile unscored products from recorded complete-plan pools."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file

SEED = 20261012
PARENT_SCORE_FLOOR = 0.65
MAX_POOLS = 32
QUEUE_DEPTH = 3
ROLES = ("actor_top", "uniform_hash", "largest_release")


def unseal(path: Path) -> dict:
    value = json.loads(path.read_text())
    if set(value) != {"payload", "payload_sha256"}:
        raise ValueError(f"expected sealed JSON: {path}")
    if identity(value["payload"]) != value["payload_sha256"]:
        raise ValueError(f"sealed JSON identity mismatch: {path}")
    return value["payload"]


@dataclass(frozen=True)
class Pool:
    worker_id: str
    source: str
    parent_score: float
    phase: int
    slot: int
    products: tuple[str, ...]
    probabilities: tuple[float, ...]
    releases: tuple[float, ...]
    plans: dict
    parent: dict


def round_robin_pools(pools: list[Pool], limit: int = MAX_POOLS) -> list[Pool]:
    """Balance pool instances across strong exact parent structures."""
    if type(limit) is not int or limit < 1:
        raise ValueError("pool limit must be a positive integer")
    groups = defaultdict(list)
    scores = {}
    for pool in pools:
        if pool.parent_score < PARENT_SCORE_FLOOR:
            continue
        groups[pool.source].append(pool)
        scores[pool.source] = pool.parent_score
    for rows in groups.values():
        rows.sort(key=lambda row: (row.phase, row.slot, row.worker_id))
    sources = sorted(groups, key=lambda source: (-scores[source], source))
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for source in sources:
            if depth < len(groups[source]):
                selected.append(groups[source][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def role_queues(pool: Pool, known: set[str], used: set[str]) -> dict[str, list[dict]]:
    """Build disjoint deterministic queues without consulting unknown task value."""
    rows = [
        {
            "smiles": smiles,
            "probability": float(probability),
            "release": float(release),
            "plan": pool.plans[smiles],
        }
        for smiles, probability, release in zip(
            pool.products, pool.probabilities, pool.releases, strict=True
        )
        if smiles not in known and smiles not in used
    ]
    orders = {
        "actor_top": lambda row: (-row["probability"], row["smiles"]),
        "uniform_hash": lambda row: (
            identity(["plan-pool-prevalence", SEED, pool.worker_id, row["smiles"]]),
            row["smiles"],
        ),
        "largest_release": lambda row: (-row["release"], row["smiles"]),
    }
    result = {}
    for role in ROLES:
        available = [row for row in rows if row["smiles"] not in used]
        queue = sorted(available, key=orders[role])[:QUEUE_DEPTH]
        for row in queue:
            used.add(row["smiles"])
        result[role] = queue
    return result


def load_pools(result_path: Path, artifact_root: Path) -> tuple[list[Pool], dict]:
    result = json.loads(result_path.read_text())
    run_id = result.get("run_id")
    if result.get("status") != "complete_development" or not run_id:
        raise ValueError("pool lock requires one complete development run")
    workers = {
        row["worker_id"]: row for row in result["workers"] if row.get("plan_decision")
    }
    worker_root = artifact_root / run_id / "workers"
    pools = []
    inputs = {str(result_path): sha256_file(result_path)}
    for worker_id, receipt in sorted(workers.items()):
        directory = worker_root / worker_id
        identity_path, lock_path = (
            directory / "identity.json",
            directory / "plan_lock.json",
        )
        task, lock = unseal(identity_path), unseal(lock_path)
        inputs[str(identity_path)] = sha256_file(identity_path)
        inputs[str(lock_path)] = sha256_file(lock_path)
        decision = lock["decision"]
        if decision != receipt["plan_decision"] or task["worker_id"] != worker_id:
            raise ValueError(f"worker/plan lock mismatch: {worker_id}")
        products = tuple(decision["products"])
        probabilities = tuple(map(float, decision["probabilities"]))
        if (
            not products
            or len(products) != len(probabilities)
            or set(products) != set(lock["plans"])
            or abs(sum(probabilities) - 1) > 1e-9
        ):
            raise ValueError(f"invalid recorded pool: {worker_id}")
        releases = tuple(float(lock["plans"][smiles]["release"]) for smiles in products)
        parent = task["parent"]
        if decision["source"] != parent["smiles"]:
            raise ValueError(f"plan source differs from exact parent: {worker_id}")
        pools.append(
            Pool(
                worker_id=worker_id,
                source=decision["source"],
                parent_score=float(parent["score"]),
                phase=int(task["phase"]),
                slot=int(task["slot"]),
                products=products,
                probabilities=probabilities,
                releases=releases,
                plans=lock["plans"],
                parent=parent,
            )
        )
    if len(pools) != 46:
        raise ValueError(
            f"expected 46 recorded complete-plan pools, found {len(pools)}"
        )
    return pools, {"run_id": run_id, "inputs_sha256": inputs, "result": result}


def build_lock(
    result_path: Path,
    artifact_root: Path,
    prepared_path: Path,
    *,
    max_pools: int = MAX_POOLS,
) -> dict:
    pools, provenance = load_pools(result_path, artifact_root)
    prepared = json.loads(prepared_path.read_text())
    known = set(prepared["observed"])
    known.update(row["smiles"] for row in provenance["result"]["oracle_rows"])
    provenance["inputs_sha256"][str(prepared_path)] = sha256_file(prepared_path)
    used = set()
    selected = []
    for pool in round_robin_pools(pools, max_pools):
        queues = role_queues(pool, known, used)
        selected.append(
            {
                "worker_id": pool.worker_id,
                "source": pool.source,
                "parent_score": pool.parent_score,
                "phase": pool.phase,
                "slot": pool.slot,
                "parent": pool.parent,
                "queues": queues,
            }
        )
    body = {
        "schema_version": "pmo_plan_pool_queue_lock_v1",
        "run_id": provenance["run_id"],
        "seed": SEED,
        "parent_score_floor": PARENT_SCORE_FLOOR,
        "max_pools": max_pools,
        "queue_depth": QUEUE_DEPTH,
        "roles": list(ROLES),
        "known_score_count": len(known),
        "selected_pools": selected,
        "selected_pool_count": len(selected),
        "queued_unique_products": len(used),
        "new_oracle_calls": 0,
        "inputs_sha256": provenance["inputs_sha256"],
    }
    return {**body, "lock_sha256": identity(body)}
