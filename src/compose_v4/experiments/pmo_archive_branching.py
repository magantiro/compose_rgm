"""Matched archive branching versus forward-only SMC from the current champion."""

from __future__ import annotations

import json
import tarfile
from functools import partial

from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_donor_comparison import CHANNEL, CHANNEL_ID
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.rewrite.kernel import canonical_state_key

KIND = "pmo_archive_branching"
APP_NAME = "compose-pmo-archive-branching"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_ARCHIVE_BRANCHING.md"
PRIOR_HASHES = (
    "2072617650562b9a2bac11b6445b5113204505ceffa56be95e058af561ee4b27",
    "92da4b2adf711aa0e0c7146f8a397dde7372fdd66608e1ae9a10b6e71387c4c5",
)


def prepare(root, prior_path, replicate_path):
    observed, nodes, losses, inputs = {}, {}, [], {}
    old_data = None
    calls = 0
    for path, digest in zip((prior_path, replicate_path), PRIOR_HASHES, strict=True):
        verify_file(path, digest)
        result = json.loads(path.read_text())
        if result["status"] != "complete_development":
            raise ValueError("archive branching requires completed donor comparisons")
        inputs[str(path)] = digest
        snapshot = path.with_name("source_snapshot.tar.gz")
        inputs[str(snapshot)] = sha256_file(snapshot)
        with tarfile.open(snapshot, "r:gz") as tar:
            prepared_bytes = tar.extractfile(result["configuration"]["prepared"]["path"]).read()
        import hashlib

        if (
            hashlib.sha256(prepared_bytes).hexdigest()
            != result["configuration"]["prepared"]["sha256"]
        ):
            raise ValueError("prior exact parents/labels differ from original source snapshot")
        data = json.loads(prepared_bytes)
        if old_data is not None and data != old_data:
            raise ValueError("donor comparisons used different original starts or labels")
        old_data = data
        for s, score in [
            *data["observed"].items(),
            *((r["smiles"], r["score"]) for r in result["oracle_rows"]),
        ]:
            if s in observed and observed[s] != score:
                raise ValueError("incompatible deterministic PMO historical labels")
            observed[s] = score
        for row in result["initial_parents"]:
            nodes.setdefault(row["smiles"], row)
        best = max(result["archives"]["hybrid"], key=result["archives"]["hybrid"].get)
        parents = result["initial_parents"]
        trajectory = []
        for rnd in result["rounds"]:
            a = rnd["arms"]["hybrid"]
            selected = [a["proposals"][i] for i in a["indices"]]
            trajectory.append(
                {
                    "boundary": rnd["boundary"],
                    "parent_incumbent_copies": sum(
                        p is not None and p["smiles"] == best for p in parents
                    ),
                    "surviving_incumbent_copies": sum(
                        p is not None and p["smiles"] == best for p in selected
                    ),
                    "active_best": max(
                        (p["score"] for p in selected if p is not None), default=None
                    ),
                    "archive_best": a["best"],
                }
            )
            parents = selected
            for arm in result["configuration"]["arms"]:
                for p in rnd["arms"][arm]["proposals"]:
                    if p is not None:
                        nodes.setdefault(p["smiles"], p)
        losses.append(
            {
                "run_id": result["run_id"],
                "result_sha256": digest,
                "best_smiles": best,
                "rounds": trajectory,
            }
        )
        calls += result["new_oracle_calls"]
    ranked = sorted(nodes.values(), key=lambda p: (-p["score"], p["smiles"]))
    for p in ranked:
        if (
            canonical_state_key(decode_search_state(p["node"]).graph) != p["smiles"]
            or observed[p["smiles"]] != p["score"]
        ):
            raise ValueError("archive parent lacks matching exact state or historical score")
    if ranked[0]["score"] != 0.6030226891555273:
        raise ValueError("archive test must challenge the actual current champion")
    publish_json(
        root / PREPARED,
        {
            "schema_version": "archive_branching_prepared_v1",
            "parents": ranked[:16],
            "donors": old_data["donors"],
            "observed": observed,
            "historical_prescreen_calls": old_data["historical_prescreen_calls"],
            "historical_probe_physical_calls": old_data["prior_physical_validation_and_new_calls"],
            "historical_comparison_physical_calls": calls,
            "inputs": inputs,
            "selection": "top 16 distinct exact molecules from both completed comparisons; no known public winner; same starts in both arms",
        },
    )
    publish_json(
        root / f"diagnostics/{KIND}/retention_evidence.json",
        {
            "inputs": inputs,
            "runs": losses,
            "producer_sha256": sha256_file(
                root / "src/compose_v4/experiments/pmo_archive_branching.py"
            ),
        },
    )
    previous = json.loads((root / "configs/pmo_donor_comparison.json").read_text())
    c = {
        k: previous[k]
        for k in ("task", "expected_input_sha256", "runtime_contract_sha256", "law_caches")
    }
    c.update(
        schema_version="archive_branching_contract_v1",
        authorization="active T4+PMO controller goal authorizes bounded empirical controller changes",
        seed=20260929,
        particles=16,
        boundaries=4,
        draws_per_bundle=1,
        primitive_budget=64,
        arms=["smc", "archive"],
        proposal_ids={"smc": CHANNEL_ID, "archive": CHANNEL_ID},
        channel=CHANNEL,
        selection_modes={"smc": "immediate"},
        parent_selection_modes={"smc": "smc", "archive": "archive"},
        archive_exploration=0.2,
        beta=10.0,
        initial_indices=list(range(16)),
        new_oracle_limit=128,
        include_region_replacement=True,
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        winner_donors=False,
        prepared={"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        compute={
            "max_workers": 29,
            "driver_containers": 1,
            "cpu": 1,
            "memory_mib": 8192,
            "worker_timeout": 180,
            "driver_timeout": 900,
            "retries": 0,
            "worker_tasks_max": 128,
            "expected_minutes": [3, 8],
            "reserved_cpu_hour_bound": 6.65,
            "reserved_usd_cap": 10,
        },
        interpretation="warm exposed development from existing champion; archive optimization versus SMC under the same donor/reference proposal law; not exact original-reference Doob or PMO AUC",
    )
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if c["contract_sha256"] != identity({k: v for k, v in c.items() if k != "contract_sha256"}):
        raise ValueError("archive branching contract self-hash mismatch")
    if (
        c["seed"],
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["primitive_budget"],
        c["archive_exploration"],
        c["beta"],
    ) != (20260929, 16, 4, 128, 64, 0.2, 10.0):
        raise ValueError("archive branching scope changed")
    if (
        c["arms"] != ["smc", "archive"]
        or c["proposal_ids"] != {a: CHANNEL_ID for a in c["arms"]}
        or c["channel"] != CHANNEL
    ):
        raise ValueError("matched proposal law changed")
    if c["parent_selection_modes"] != {"smc": "smc", "archive": "archive"} or c[
        "selection_modes"
    ] != {"smc": "immediate"}:
        raise ValueError("parent selection modes changed")
    if (
        c["task"] != "perindopril_mpo"
        or c["initial_indices"] != list(range(16))
        or c["draws_per_bundle"] != 1
        or not c["include_region_replacement"]
    ):
        raise ValueError("archive branching task/support changed")
    if any(
        c[k]
        for k in (
            "reference_training_authorized",
            "value_training_authorized",
            "docking_authorized",
            "winner_donors",
        )
    ):
        raise ValueError("no training, docking or public winner injection in archive branching")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
