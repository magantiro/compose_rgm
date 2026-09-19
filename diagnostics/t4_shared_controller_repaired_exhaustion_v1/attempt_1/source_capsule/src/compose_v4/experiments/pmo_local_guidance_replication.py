"""One fixed fresh-seed replication; paid-cache reuse cannot change proposals."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import partial

from compose_v4.control.docking_value import identity
from compose_v4.experiments import pmo_local_guidance as first
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import initial_score_history
from compose_v4.experiments.pmo_option_particles import session as particle_session

KIND = "pmo_local_guidance_replication"
APP_NAME = "compose-pmo-local-guidance-replication"
APP, CONTRACT = f"modal_apps/{KIND}_app.py", f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_LOCAL_GUIDANCE_REPLICATION.md"
FIRST_CONTRACT = "0682c9efc6dea9051ef8bee34a6d0d3dca4f95b84eba8cf8285e6a4381bb07aa"
EVIDENCE = "diagnostics/pmo_local_guidance/evidence.json"
DECISION = "positive_best_score_earns_unchanged_replication"


def prepare(root, prior_path):
    original = first.load_contract(root)
    evidence = json.loads((root / EVIDENCE).read_text())
    verify_file(prior_path, evidence["input_sha256"][str(prior_path)])
    prior = json.loads(prior_path.read_text())
    if original["contract_sha256"] != FIRST_CONTRACT or prior["configuration"] != original:
        raise ValueError("replication must start from the declared first comparison")
    if evidence["decision"] != DECISION or prior["status"] != "complete_development":
        raise ValueError("replication requires a completed positive audited first run")
    data = json.loads((root / original["prepared"]["path"]).read_text())
    rows = prior["oracle_rows"]
    if len(rows) != prior["new_oracle_calls"] or any(r["status"] != "complete" for r in rows):
        raise ValueError("first-run oracle ledger is incomplete")
    cache = {r["smiles"]: r["score"] for r in rows}
    if len(cache) != len(rows) or set(cache) & set(data["observed"]):
        raise ValueError("first-run physical query ledger contains duplicate paid labels")
    data["requested_only_scores"] = cache
    data["historical_development_physical_calls"] += len(rows)
    data["replication_source"] = {
        "run_id": prior["run_id"],
        "result_sha256": sha256_file(prior_path),
        "original_prepared_sha256": original["prepared"]["sha256"],
    }
    initial_score_history(data)
    publish_json(root / PREPARED, data)
    c = deepcopy(original)
    metadata = root / f"diagnostics/{KIND}/cache_metadata"
    publish_json(metadata / "launch.json", {"image_revision": prior["image_revision"]})
    publish_json(metadata / "runtime_gate.json", {"input_sha256": c["expected_input_sha256"]})
    c["law_caches"].append(
        {
            "path": f"{first.KIND}/{prior['run_id']}",
            "metadata_path": f"{KIND}/cache_metadata/{prior['run_id']}",
            "source_result_sha256": sha256_file(prior_path),
            "run_id": prior["run_id"],
            "launch_sha256": sha256_file(metadata / "launch.json"),
            "cache_paths": sorted(
                f"workers/{w['worker_id']}"
                for w in prior["workers"]
                if w["law_work"].get("fresh_laws", 0) > 0
            ),
        }
    )
    c.update(
        artifact_kind=KIND,
        seed=20261009,
        prepared={"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        replication={
            **data["replication_source"],
            "evidence": {"path": EVIDENCE, "sha256": sha256_file(root / EVIDENCE)},
            "first_contract_sha256": FIRST_CONTRACT,
            "input_files": {
                p: sha256_file(root / p)
                for p in (first.CONTRACT, first.PREPARED, first.PROTOCOL, EVIDENCE)
            },
            "law_cache": c["law_caches"][-1],
        },
    )
    c["contract_sha256"] = identity({k: v for k, v in c.items() if k != "contract_sha256"})
    publish_json(root / CONTRACT, c)
    load_contract(root)
    return {
        "contract_sha256": c["contract_sha256"],
        "seed": c["seed"],
        "cached_labels": len(cache),
        "original_exclusions": len(data["observed"]),
        "historical_development_calls": data["historical_development_physical_calls"],
    }


def load_contract(root):
    c = first.load_contract(root, contract_path=CONTRACT, seed=20261009)
    original = first.load_contract(root)
    if original["contract_sha256"] != FIRST_CONTRACT:
        raise ValueError("original local-guidance contract changed")
    allowed = {
        "artifact_kind",
        "seed",
        "prepared",
        "protocol",
        "replication",
        "contract_sha256",
        "law_caches",
    }
    if {k: v for k, v in c.items() if k not in allowed} != {
        k: v for k, v in original.items() if k not in allowed
    }:
        raise ValueError("replication changed the original algorithm or resource limits")
    if c["law_caches"] != original["law_caches"] + [c["replication"]["law_cache"]]:
        raise ValueError("replication changed the declared compatible cache sources")
    for path, digest in c["replication"]["input_files"].items():
        verify_file(root / path, digest)
    binding = c["replication"]["evidence"]
    verify_file(root / binding["path"], binding["sha256"])
    evidence = json.loads((root / binding["path"]).read_text())
    if evidence["decision"] != DECISION or evidence["run_id"] != c["replication"]["run_id"]:
        raise ValueError("replication lacks the positive first-run decision")
    data = json.loads((root / c["prepared"]["path"]).read_text())
    old = json.loads((root / original["prepared"]["path"]).read_text())
    added = {"requested_only_scores", "replication_source", "historical_development_physical_calls"}
    if {k: v for k, v in data.items() if k not in added} != {
        k: v for k, v in old.items() if k not in added
    }:
        raise ValueError("replication changed original starts, model, donors or exclusions")
    if data["historical_development_physical_calls"] != (
        old["historical_development_physical_calls"] + len(data["requested_only_scores"])
    ):
        raise ValueError("replication historical call accounting differs from paid cache")
    initial_score_history(data)
    return c


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
worker_remote = partial(first.worker_remote, run_session=session, contract_loader=load_contract)
