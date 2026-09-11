"""Frozen task-trained proposals versus unchanged proposals, shared score selection."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.learned_proposal import RECIPE, ProposalPolicy
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_option_particles import driver_remote as particle_driver
from compose_v4.experiments.pmo_option_particles import session as particle_session

KIND = "pmo_learned_proposal"
APP_NAME = "compose-pmo-learned-proposal"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
MODEL = f"diagnostics/{KIND}/model.json"
TRAINING = f"diagnostics/{KIND}/training.json"
PROTOCOL = "docs/PMO_LEARNED_PROPOSAL.md"


def prepare(root, prior_path, *, model_path, manifest_path, decisions_path):
    audit = json.loads((root / "diagnostics/pmo_replacement_continuation/report.json").read_text())
    verify_file(prior_path, audit["input_paths_sha256"][str(prior_path)])
    prior = json.loads(prior_path.read_text())
    if prior["run_id"] != "72e39c6079e3c06b10122a57661e8f0e529f876277a83fe16e45ec82a3962bdb":
        raise ValueError("proposal comparison must follow the registered continuation")
    previous = prior["configuration"]
    old_path = root / previous["prepared"]["path"]
    verify_file(old_path, previous["prepared"]["sha256"])
    parents = json.loads(old_path.read_text())["parents"]
    if len(parents) != 9 or prior["initial_parents"] != parents:
        raise ValueError("proposal comparison start census changed")
    model = ProposalPolicy(json.loads(model_path.read_text()))
    verify_file(decisions_path, model.payload["data_sha256"])
    training = json.loads(decisions_path.read_text())
    verify_file(manifest_path, training["input_hashes"][str(manifest_path)])
    manifest = json.loads(manifest_path.read_text())
    observed, inputs = {}, {str(prior_path): sha256_file(prior_path)}
    for run in manifest["runs"]:
        result_path = Path(run["result_path"])
        verify_file(result_path, manifest["inputs"][str(result_path)])
        result = json.loads(result_path.read_text())
        prep = root / run["configuration"]["prepared"]["path"]
        verify_file(prep, run["configuration"]["prepared"]["sha256"])
        labels = list(json.loads(prep.read_text())["observed"].items())
        labels += [(r["smiles"], r["score"]) for r in result["oracle_rows"]]
        for smiles, score in labels:
            if smiles in observed and observed[smiles] != score:
                raise ValueError("historical training/query labels disagree")
            observed[smiles] = score
        inputs[str(result_path)] = sha256_file(result_path)
    publish_json(root / MODEL, model.payload)
    publish_json(
        root / TRAINING,
        {
            **{k: v for k, v in training.items() if k != "examples"},
            "prepared_decisions": {
                "path": str(decisions_path),
                "sha256": sha256_file(decisions_path),
            },
            "download_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "durable_training_data": {
                "workspace": "rahul-94866",
                "volume": "compose-v4-artifacts",
                "prefix": f"pmo_proposal_training/{model.payload['data_sha256']}",
                "decisions_sha256": sha256_file(decisions_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "fit_receipt": json.loads(model_path.with_suffix(".fit.json").read_text()),
        },
    )
    publish_json(
        root / PREPARED,
        {
            "schema_version": "learned_proposal_prepared_v1",
            "parents": parents,
            "observed": observed,
            "input_hashes": inputs,
            "training_unique_labels": len(observed),
            "training_decisions_sha256": sha256_file(decisions_path),
            "interpretation": "warm winner-informed development; historical labels are charged prior data, not free benchmark initialization",
        },
    )
    c = {k: previous[k] for k in ("task", "expected_input_sha256", "runtime_contract_sha256")}
    c.update(
        schema_version="learned_proposal_contract_v1",
        authorization="2026-09-11 explicit user approval to try task-conditioned proposal learning and resume PMO goal",
        seed=20260924,
        particles=9,
        boundaries=4,
        arms=["baseline", "learned"],
        beta=10.0,
        draws_per_bundle=1,
        initial_indices=list(range(9)),
        new_oracle_limit=72,
        primitive_budget=40,
        include_region_replacement=True,
        proposal_ids={"baseline": None, "learned": model.payload["model_sha256"]},
        selection_modes={"baseline": "immediate", "learned": "immediate"},
        policy_recipe=RECIPE,
        law_caches=[],
        proposal_training_authorized=True,
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        prepared={"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        model={"path": MODEL, "sha256": sha256_file(root / MODEL)},
        training={"path": TRAINING, "sha256": sha256_file(root / TRAINING)},
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        compute={
            "max_workers": 18,
            "driver_containers": 1,
            "worker_tasks_max": 72,
            "worker_timeout": 300,
            "driver_timeout": 1200,
            "cpu": 1,
            "memory_mib": 8192,
            "retries": 0,
            "heartbeat_seconds": 30,
            "expected_minutes": [6, 15],
            "estimated_usd": [1, 10],
            "reserved_cpu_hour_bound": 6.34,
        },
        interpretation="frozen proposal learning causal comparison with identical actual-score SMC; changed proposal path references, no exact original-reference Doob claim; warm development, not matched PMO AUC",
    )
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("learned proposal contract hash mismatch")
    if (
        c["seed"],
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["draws_per_bundle"],
        c["beta"],
        c["primitive_budget"],
    ) != (20260924, 9, 4, 72, 1, 10.0, 40):
        raise ValueError("proposal comparison exceeds authorized recipe")
    if c["arms"] != ["baseline", "learned"] or c["selection_modes"] != {
        "baseline": "immediate",
        "learned": "immediate",
    }:
        raise ValueError("proposal comparison must hold actual-score selection fixed")
    if (
        c["policy_recipe"] != RECIPE
        or c["initial_indices"] != list(range(9))
        or not c["include_region_replacement"]
    ):
        raise ValueError("proposal recipe, starts or option support changed")
    if (
        any(
            c[k]
            for k in (
                "reference_training_authorized",
                "value_training_authorized",
                "docking_authorized",
            )
        )
        or "value_checkpoint" in c
    ):
        raise ValueError("reference/value training or docking not authorized")
    for key in ("prepared", "protocol", "training", "model"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    model = ProposalPolicy(json.loads((root / c["model"]["path"]).read_text()))
    if c["proposal_ids"] != {"baseline": None, "learned": model.payload["model_sha256"]}:
        raise ValueError("comparison proposal checkpoint identity mismatch")
    return c


def proposal_factory(root, contract, task):
    key = task.get("proposal_id")
    if key is None:
        return None
    model = ProposalPolicy(json.loads((root / contract["model"]["path"]).read_text()))
    if key != model.payload["model_sha256"]:
        raise ValueError("worker requested an unbound proposal model")
    return model


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)
driver_remote = partial(particle_driver, run_session=session)
