"""Two-arm complete-option continuation, actual scores and no learned value head."""

from __future__ import annotations

import json
from functools import partial

from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.rewrite.kernel import canonical_state_key

KIND = "pmo_replacement_continuation"
APP_NAME = "compose-pmo-replacement-continuation"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
PROTOCOL = "docs/PMO_REPLACEMENT_CONTINUATION.md"
PRIOR_HASH = "de54a444fdd286447ca399f391ec1e47c3f1c4950e3b647a041fbee2397273b4"


def prepare(root, prior_path):
    verify_file(prior_path, PRIOR_HASH)
    prior = json.loads(prior_path.read_text())
    previous = prior["configuration"]
    old_path = root / previous["prepared"]["path"]
    verify_file(old_path, previous["prepared"]["sha256"])
    old = json.loads(old_path.read_text())
    products = [
        p for p in prior["candidates"] if p["bundle"]["option"].startswith("replace_region:")
    ]
    if len(old["parents"]) != 5 or len(products) != 4:
        raise ValueError("continuation requires all five roots and four completed replacements")
    observed = dict(old["observed"])
    for row in prior["oracle_rows"]:
        if row["smiles"] in observed and observed[row["smiles"]] != row["score"]:
            raise ValueError("historical labels disagree")
        observed[row["smiles"]] = row["score"]
    parents = old["parents"] + products
    for p in parents:
        if canonical_state_key(decode_search_state(p["node"]).graph) != p["smiles"]:
            raise ValueError("exact continuation state differs from declared identity")
        if observed[p["smiles"]] != p["score"]:
            raise ValueError("continuation score lacks a matching historical observation")
    publish_json(
        root / PREPARED,
        {
            "schema_version": "replacement_continuation_prepared_v1",
            "parents": parents,
            "observed": observed,
            "prior_result": {
                "path": str(prior_path),
                "sha256": PRIOR_HASH,
                "run_id": prior["run_id"],
            },
            "prior_prepared": {
                "path": str(old_path.relative_to(root)),
                "sha256": sha256_file(old_path),
            },
            "selection": "all five original warm roots, then every completed replacement in locked task order",
        },
    )
    c = {k: previous[k] for k in ("task", "expected_input_sha256", "runtime_contract_sha256")}
    c.update(
        schema_version="replacement_continuation_contract_v1",
        authorization="continuing user-requested PMO goal; bounded development with clean committed source",
        seed=20260923,
        particles=9,
        boundaries=4,
        arms=["reference", "immediate"],
        beta=10.0,
        draws_per_bundle=1,
        initial_indices=list(range(9)),
        new_oracle_limit=72,
        primitive_budget=40,
        include_region_replacement=True,
        law_caches=[],
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        winner_input=False,
        prepared={"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        compute={
            "max_workers": 18,
            "driver_containers": 1,
            "worker_tasks_max": 72,
            "worker_timeout": 240,
            "driver_timeout": 1200,
            "cpu": 1,
            "memory_mib": 8192,
            "retries": 0,
            "heartbeat_seconds": 30,
            "expected_minutes": [4, 10],
            "estimated_usd": [1, 8],
            "reserved_cpu_hour_bound": 5.14,
        },
        interpretation="warm winner-informed development roots, no winner/value-head input in this run; not matched PMO AUC; all queried archive differs from terminal weighted draw",
    )
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("replacement continuation contract hash mismatch")
    if (
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["draws_per_bundle"],
        c["seed"],
        c["beta"],
        c["primitive_budget"],
        c["include_region_replacement"],
    ) != (9, 4, 72, 1, 20260923, 10.0, 40, True):
        raise ValueError("replacement continuation exceeds locked recipe")
    if (
        c["arms"] != ["reference", "immediate"]
        or c["initial_indices"] != list(range(9))
        or "value_checkpoint" in c
        or any(
            c[k]
            for k in (
                "reference_training_authorized",
                "value_training_authorized",
                "docking_authorized",
                "winner_input",
            )
        )
    ):
        raise ValueError("continuation changed its arms, starts or no-training/value-head scope")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
