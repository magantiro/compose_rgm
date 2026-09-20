"""Payload-authorized, receipt-sharded nine-cell T4 completion campaign.

Every cell has one private volume and driver. Immutable dispatch intents are
published before workers are spawned, so a lost job is never retried, replaced,
or backfilled. The module is inert until the separately authorized final
contract and its exact-commit source capsule validate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_CAMPAIGN_VARIANT = os.environ.get("COMPOSE_T4_CAMPAIGN_VARIANT", "full")
if (
    _CAMPAIGN_VARIANT == "full"
    and not (ROOT / "configs/t4_shared_controller_completion_v1.json").exists()
):
    if (ROOT / "configs/t4_compose_nodistill_transfer_v1.json").exists():
        _CAMPAIGN_VARIANT = "nodistill_transfer_v1"
    elif (ROOT / "configs/t4_compose_nodistill_parp1_v1.json").exists():
        _CAMPAIGN_VARIANT = "nodistill_parp1_v1"
if _CAMPAIGN_VARIANT == "nodistill_parp1_v1":
    from compose_v4.experiments.t4_compose_nodistill_contract import (
        AUTHORIZATION_RELATIVE_PATH,
        CAPSULE_MANIFEST_RELATIVE_PATH,
        CAPSULE_ROOT_RELATIVE_PATH,
        FINAL_CONTRACT_RELATIVE_PATH,
        validate_scored_contract,
    )
    from compose_v4.experiments.t4_compose_nodistill_contract import (
        PREPARATION_RELATIVE_PATH as CONTRACT_RELATIVE_PATH,
    )
elif _CAMPAIGN_VARIANT == "nodistill_transfer_v1":
    from compose_v4.experiments.t4_compose_nodistill_transfer_contract import (
        AUTHORIZATION_RELATIVE_PATH,
        CAPSULE_MANIFEST_RELATIVE_PATH,
        CAPSULE_ROOT_RELATIVE_PATH,
        FINAL_CONTRACT_RELATIVE_PATH,
        validate_scored_contract,
    )
    from compose_v4.experiments.t4_compose_nodistill_transfer_contract import (
        PREPARATION_RELATIVE_PATH as CONTRACT_RELATIVE_PATH,
    )
elif _CAMPAIGN_VARIANT == "repaired_exhaustion_v1":
    from compose_v4.experiments.t4_repaired_exhaustion_contract import (
        AUTHORIZATION_RELATIVE_PATH,
        CAPSULE_MANIFEST_RELATIVE_PATH,
        CAPSULE_ROOT_RELATIVE_PATH,
        FINAL_CONTRACT_RELATIVE_PATH,
        validate_scored_contract,
    )
    from compose_v4.experiments.t4_repaired_exhaustion_contract import (
        PREPARATION_RELATIVE_PATH as CONTRACT_RELATIVE_PATH,
    )
elif _CAMPAIGN_VARIANT == "nodistill_feasibility_v4":
    from compose_v4.experiments.t4_nodistill_feasibility_campaign_v4_contract import (
        AUTHORIZATION_RELATIVE_PATH,
        CAPSULE_MANIFEST_RELATIVE_PATH,
        CAPSULE_ROOT_RELATIVE_PATH,
        FINAL_CONTRACT_RELATIVE_PATH,
        validate_scored_contract,
    )
    from compose_v4.experiments.t4_nodistill_feasibility_campaign_v4_contract import (
        PREPARATION_RELATIVE_PATH as CONTRACT_RELATIVE_PATH,
    )
else:
    from compose_v4.experiments.t4_shared_controller_completion_contract import (
        CONTRACT_RELATIVE_PATH,
    )
    from compose_v4.experiments.t4_shared_controller_scored_contract import (
        AUTHORIZATION_RELATIVE_PATH,
        CAPSULE_MANIFEST_RELATIVE_PATH,
        CAPSULE_ROOT_RELATIVE_PATH,
        FINAL_CONTRACT_RELATIVE_PATH,
        validate_scored_contract,
    )

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
    sha256_file,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    make_launch_task,
    validate_launch_task,
)

_VARIANT_APPS = {
    "full": (
        "compose-t4-shared-controller-completion-v1",
        "modal_apps/t4_shared_controller_completion_v1_app.py",
    ),
    "nodistill_parp1_v1": (
        "compose-t4-compose-nodistill-parp1-v1",
        "modal_apps/t4_compose_nodistill_parp1_v1_app.py",
    ),
    "nodistill_transfer_v1": (
        "compose-t4-compose-nodistill-transfer-v1",
        "modal_apps/t4_compose_nodistill_transfer_v1_app.py",
    ),
    "repaired_exhaustion_v1": (
        "compose-t4-repaired-exhaustion-v1",
        "modal_apps/t4_shared_controller_repaired_exhaustion_v1_app.py",
    ),
    "nodistill_feasibility_v4": (
        "compose-t4-nodistill-feasibility-v4",
        "modal_apps/t4_nodistill_feasibility_v4_app.py",
    ),
}
try:
    APP_NAME, APP_RELATIVE_PATH = _VARIANT_APPS[_CAMPAIGN_VARIANT]
except KeyError as error:
    raise ValueError(f"unknown T4 campaign variant: {_CAMPAIGN_VARIANT}") from error
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
REMOTE_ROOT = Path("/capsule")
REMOTE_SEALED = Path("/sealed")
CELL_MOUNT = Path("/cell")

FINAL_CONTRACT_PATH = ROOT / FINAL_CONTRACT_RELATIVE_PATH
AUTHORIZATION_PATH = ROOT / AUTHORIZATION_RELATIVE_PATH
CAPSULE_ROOT = ROOT / CAPSULE_ROOT_RELATIVE_PATH
CAPSULE_MANIFEST_PATH = ROOT / CAPSULE_MANIFEST_RELATIVE_PATH
REMOTE_FINAL_CONTRACT = REMOTE_SEALED / "scored_contract.json"
REMOTE_AUTHORIZATION = REMOTE_SEALED / "scored_authorization.json"
REMOTE_CAPSULE_MANIFEST = REMOTE_SEALED / "source_capsule_manifest.json"


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    identity = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_identity(payload) != identity:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload, str(identity)


PREPARATION = json.loads((ROOT / CONTRACT_RELATIVE_PATH).read_text())
CELLS = tuple(PREPARATION["cells"])
_EXPECTED_CELL_COUNTS = {
    "full": 9,
    "nodistill_parp1_v1": 3,
    "nodistill_transfer_v1": 3,
    "repaired_exhaustion_v1": 2,
    "nodistill_feasibility_v4": 3,
}
_EXPECTED_CELL_COUNT = _EXPECTED_CELL_COUNTS[_CAMPAIGN_VARIANT]
if len(CELLS) != _EXPECTED_CELL_COUNT:
    raise ValueError(
        f"{_CAMPAIGN_VARIANT} launcher requires exactly "
        f"{_EXPECTED_CELL_COUNT} cells"
    )
CELL_KEYS = tuple(row["cell_key"] for row in CELLS)
CELL_VOLUMES = {row["cell_key"]: row["volume"] for row in CELLS}


def _sealed_inputs_present() -> bool:
    return all(
        path.exists()
        for path in (
            FINAL_CONTRACT_PATH,
            AUTHORIZATION_PATH,
            CAPSULE_ROOT,
            CAPSULE_MANIFEST_PATH,
        )
    )


def _runtime_image() -> modal.Image:
    result = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.0",
            "numpy==1.26.4",
            "scipy==1.13.1",
            "networkx==3.3",
            "rdkit==2024.3.5",
        )
        .apt_install("openbabel", "curl", "ca-certificates")
        .run_commands(
            "mkdir -p /opt/dock/receptors",
            f"curl --fail -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
            "chmod +x /opt/dock/qvina02",
            f"curl --fail -sSL -o /opt/dock/receptors/parp1.pdbqt {MOOD}/receptors/parp1.pdbqt",
            f"curl --fail -sSL -o /opt/dock/receptors/braf.pdbqt {MOOD}/receptors/braf.pdbqt",
            f"curl --fail -sSL -o /opt/dock/receptors/jak2.pdbqt {MOOD}/receptors/jak2.pdbqt",
            f"curl --fail -sSL -o /opt/dock/receptors/5ht1b.pdbqt {MOOD}/receptors/5ht1b.pdbqt",
        )
        .env(
            {
                "PYTHONPATH": (
                    f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'modal_apps'}"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "COMPOSE_T4_CAMPAIGN_VARIANT": _CAMPAIGN_VARIANT,
            }
        )
    )
    if not _sealed_inputs_present():
        return result
    return (
        result.add_local_dir(CAPSULE_ROOT, str(REMOTE_ROOT), copy=True)
        .add_local_file(FINAL_CONTRACT_PATH, str(REMOTE_FINAL_CONTRACT), copy=True)
        .add_local_file(AUTHORIZATION_PATH, str(REMOTE_AUTHORIZATION), copy=True)
        .add_local_file(CAPSULE_MANIFEST_PATH, str(REMOTE_CAPSULE_MANIFEST), copy=True)
    )


image = _runtime_image()
app = modal.App(APP_NAME)


def local_scored_context() -> dict[str, Any]:
    contract = validate_scored_contract(
        repository_root=ROOT,
        contract_path=FINAL_CONTRACT_PATH,
        authorization_path=AUTHORIZATION_PATH,
        capsule_root=CAPSULE_ROOT,
        capsule_manifest_path=CAPSULE_MANIFEST_PATH,
    )
    contract_payload, contract_identity = _load_envelope(FINAL_CONTRACT_PATH)
    if contract_payload != contract:
        raise ValueError("validated scored contract changed while loading")
    authorization, _ = _load_envelope(AUTHORIZATION_PATH)
    manifest, manifest_identity = _load_envelope(CAPSULE_MANIFEST_PATH)
    entry = manifest.get("files", {}).get(APP_RELATIVE_PATH)
    if not isinstance(entry, dict) or entry.get("sha256") != sha256_file(
        ROOT / APP_RELATIVE_PATH
    ):
        raise ValueError("local launcher differs from the exact source capsule")
    launch = make_launch_task(
        contract=contract,
        contract_payload_sha256=contract_identity,
        contract_file_sha256=sha256_file(FINAL_CONTRACT_PATH),
        authorization_receipt=authorization,
        authorization_receipt_sha256=sha256_file(AUTHORIZATION_PATH),
        code_revision=manifest["code_revision"],
        source_capsule_payload_sha256=manifest_identity,
    )
    return {"contract": contract, "launch": launch}


def scored_preflight_report() -> dict:
    paths = {
        "contract": FINAL_CONTRACT_PATH,
        "authorization": AUTHORIZATION_PATH,
        "capsule_root": CAPSULE_ROOT,
        "capsule_manifest": CAPSULE_MANIFEST_PATH,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {
            "schema_version": "t4_shared_controller_scored_preflight_v1",
            "ready": False,
            "missing": missing,
            "modal_calls_created": 0,
        }
    context = local_scored_context()
    return {
        "schema_version": "t4_shared_controller_scored_preflight_v1",
        "ready": True,
        "run_id": context["launch"]["run_id"],
        "scored_calls_requested": context["contract"]["scored_calls_requested"],
        "cell_keys": list(context["launch"]["cell_keys"]),
        "volumes": [CELL_VOLUMES[key] for key in CELL_KEYS],
        "modal_calls_created": 0,
    }


def _remote_context(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = validate_scored_contract(
        repository_root=REMOTE_ROOT,
        contract_path=REMOTE_FINAL_CONTRACT,
        authorization_path=REMOTE_AUTHORIZATION,
        capsule_root=REMOTE_ROOT,
        capsule_manifest_path=REMOTE_CAPSULE_MANIFEST,
    )
    contract_payload, contract_identity = _load_envelope(REMOTE_FINAL_CONTRACT)
    authorization, _ = _load_envelope(REMOTE_AUTHORIZATION)
    manifest, manifest_identity = _load_envelope(REMOTE_CAPSULE_MANIFEST)
    launch = task.get("launch", task)
    validate_launch_task(
        launch,
        contract=contract_payload,
        contract_payload_sha256=contract_identity,
        contract_file_sha256=sha256_file(REMOTE_FINAL_CONTRACT),
        authorization_receipt=authorization,
        authorization_receipt_sha256=sha256_file(REMOTE_AUTHORIZATION),
        source_capsule_payload_sha256=manifest_identity,
    )
    if (
        contract_payload != contract
        or launch["code_revision"] != manifest["code_revision"]
    ):
        raise ValueError("remote contract or exact source identity drift")
    return contract, launch


def _bound_cell(contract: dict[str, Any], cell_key: str) -> dict[str, Any]:
    rows = [row for row in contract["cells"] if row["cell_key"] == cell_key]
    if len(rows) != 1 or rows[0]["volume"] != CELL_VOLUMES[cell_key]:
        raise ValueError("cell or private-volume binding drift")
    return rows[0]


def _run_root(launch: dict[str, Any]) -> Path:
    return CELL_MOUNT / launch["run_id"]


def _publish_or_assert(store: Any, path: str, payload: dict[str, Any]) -> str:
    try:
        existing = store.read(path)
    except FileNotFoundError:
        try:
            return store.publish_once(path, payload)
        except FileExistsError:
            existing = store.read(path)
    if existing != payload:
        raise ValueError(f"immutable receipt drift: {path}")
    return payload_identity(existing)


def _load_route_expert(contract: dict[str, Any]):
    from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert

    if not contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
        raise RuntimeError("COMPOSE-NoDistill may not load a route checkpoint")
    path = REMOTE_ROOT / contract["shared_route_checkpoint"]["path"]
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload_identity(payload) != envelope.get("payload_sha256")
        or envelope["payload_sha256"]
        != contract["shared_route_checkpoint"]["payload_sha256"]
        or sha256_file(path) != contract["shared_route_checkpoint"]["sha256"]
    ):
        raise ValueError("shared route checkpoint identity drift")
    return RouteDistilledGoalExpert.from_checkpoint(payload["expert"])


def _proposal_answer(
    request: dict[str, Any], cell: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Run one configured generic proposal expert on one measured parent."""

    import time

    import numpy as np

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.route_distilled_goal_expert import (
        propose_route_expert_candidates,
    )
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_shared_controller_scored_runtime import (
        attach_endpoint_fingerprints,
        attach_generic_scale_band,
        feasibility_headroom_v4_records,
        generic_topology_macro_records,
        protonation_aware_records,
        retained_core_route_records,
    )

    started = time.time()
    expert_name = request["expert"]
    controller = contract["controller"]
    fiber = Fiber(cell["source_smiles"], cell["delta"], support=contract["support"])
    if expert_name in {"shallow", "anchored_replacement"}:
        settings = controller["proposal"][expert_name]
        records = expand(
            request["parent"],
            request["parent_score"],
            fiber,
            np.random.default_rng(request["proposal_seed"]),
            draws=settings["draws"],
            multi_region=True,
            horizon=settings["horizon"],
            proposal_lane=expert_name,
        )
        for row in records:
            row["proposal_experts"] = [expert_name]
        telemetry: dict[str, Any] = {"raw_draws": settings["draws"]}
    elif expert_name == "route_complete_region":
        settings = controller["proposal"][expert_name]
        records = []
        route_telemetry: dict[str, Any]
        if contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
            source = pad_molecular_graph(
                smiles_to_molecular_graph(request["parent"]), 48
            )
            proposed, route_telemetry = propose_route_expert_candidates(
                source,
                _load_route_expert(contract),
                pool_size=settings["pool_size"],
                realization_limit=settings["realization_limit"],
                beam_width=settings["beam_width"],
                expansion_width=settings["expansion_width"],
                max_bindings_per_template=settings["max_bindings_per_template"],
                maximum_expansions=settings["maximum_expansions"],
                scale_balanced=settings["scale_balanced"],
            )
            for row in proposed:
                properties = fiber.check(row["smiles"])
                if properties is None or properties["smiles"] == request["parent"]:
                    continue
                records.append(
                    {
                        **row,
                        **properties,
                        "parent": request["parent"],
                        "parent_score": request["parent_score"],
                        "delta": cell["delta"],
                        "route_generation_modes": ["legacy_route_expert"],
                    }
                )
        else:
            route_telemetry = {
                "status": "disabled_by_compose_nodistill",
                "templates_loaded": 0,
                "route_checkpoint_read": False,
            }
        retained, retained_telemetry = retained_core_route_records(
            parent=request["parent"],
            parent_score=request["parent_score"],
            original_seed=cell["source_smiles"],
            delta=cell["delta"],
            support=contract["support"],
        )
        records.extend(retained)
        telemetry = {
            "distilled_route": route_telemetry,
            "retained_core": retained_telemetry,
        }
    elif expert_name == "protonation_aware_retained_subgraph":
        records, telemetry = protonation_aware_records(
            parent=request["parent"],
            parent_score=request["parent_score"],
            original_seed=cell["source_smiles"],
            delta=cell["delta"],
            support=contract["support"],
            proposal_seed_value=request["proposal_seed"],
            route_expert=_load_route_expert(contract),
            settings=controller["proposal"][expert_name],
        )
    elif expert_name == "generic_topology_macro_v2":
        records, telemetry = generic_topology_macro_records(
            parent=request["parent"],
            proposal_seed_value=request["proposal_seed"],
            settings=controller["proposal"][expert_name],
        )
        for row in records:
            row["parent_score"] = request["parent_score"]
            row["delta"] = cell["delta"]
    elif expert_name == "generic_feasibility_headroom_v4":
        records, telemetry = feasibility_headroom_v4_records(
            parent=request["parent"],
            proposal_seed_value=request["proposal_seed"],
            similarity_minimum=cell["delta"],
            settings=controller["proposal"][expert_name],
        )
        for row in records:
            row["parent_score"] = request["parent_score"]
            row["delta"] = cell["delta"]
    else:
        raise ValueError(f"unknown proposal expert: {expert_name}")
    if not contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
        records = [attach_generic_scale_band(row) for row in records]
    records = attach_endpoint_fingerprints(
        records,
        original_seed=cell["source_smiles"],
        delta=cell["delta"],
        support=contract["support"],
    )
    return {
        "expert": expert_name,
        "parent": request["parent"],
        "parent_score": request["parent_score"],
        "proposal_seed": request["proposal_seed"],
        "records": records,
        "telemetry": telemetry,
        "eligible_unique": len({row["smiles"] for row in records}),
        "elapsed_seconds": time.time() - started,
    }


def _failed_particle_receipt(spec: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    from compose_v4.control.route_complete_region_particles import (
        RECEIPT_SCHEMA_VERSION,
    )

    records: list[dict[str, Any]] = []
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "job": spec.payload(),
        "status": "failed",
        "records": records,
        "telemetry": {
            "candidate_count": 0,
            "candidate_set_sha256": payload_identity(records),
            "source_state_sha256": manifest["source_state_sha256"],
            "expert_training_identity_sha256": manifest[
                "expert_training_identity_sha256"
            ],
        },
    }
    return {"payload": payload, "payload_sha256": payload_identity(payload)}


def _fiber_gate_route_rows(
    rows: list[dict[str, Any]],
    *,
    cell: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the bound endpoint gate to every legacy, retained, or particle row."""

    from compose_v4.experiments.t4_fiber_campaign import Fiber

    fiber = Fiber(cell["source_smiles"], cell["delta"], support=contract["support"])
    eligible = []
    for source in rows:
        properties = fiber.check(source.get("smiles", ""))
        if properties is None or properties["smiles"] == source.get("parent"):
            continue
        eligible.append({**source, **properties})
    return eligible


CELL_FUNCTIONS: dict[str, dict[str, Any]] = {}


def _register_cell(cell_binding: dict[str, Any]) -> dict[str, Any]:
    cell_key = cell_binding["cell_key"]
    safe = cell_key.replace("-", "_")
    volume = modal.Volume.from_name(cell_binding["volume"], create_if_missing=True)
    common = {
        "image": image,
        "cpu": (1.0, 1.0),
        "memory": 4096,
        "retries": 0,
        "volumes": {str(CELL_MOUNT): volume},
        "scaledown_window": 20,
        "include_source": False,
        "serialized": True,
    }

    @app.function(
        **common,
        name=f"proposal_{safe}",
        max_containers=16,
        timeout=7_200 if _CAMPAIGN_VARIANT == "nodistill_feasibility_v4" else 3_000,
    )
    def proposal_worker(task: dict[str, Any]) -> dict[str, Any]:
        from compose_v4.experiments.t4_shared_controller_cell_runtime import (
            ReceiptStore,
        )

        contract, launch = _remote_context(task)
        cell = _bound_cell(contract, cell_key)
        volume.reload()
        store = ReceiptStore(_run_root(launch), flush=volume.commit)
        manifest = store.read(task["manifest_path"])
        request = task["request"]
        if request not in manifest["requests"]:
            raise ValueError("proposal request is outside its immutable manifest")
        try:
            return store.read(task["receipt_path"])
        except FileNotFoundError:
            pass
        try:
            answer = _proposal_answer(request, cell, contract)
        except (
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            receipt = {
                "schema_version": "t4_shared_controller_proposal_receipt_v1",
                "status": "failed",
                "request": request,
                "error": repr(error),
            }
        else:
            receipt = {
                "schema_version": "t4_shared_controller_proposal_receipt_v1",
                "status": "complete",
                "request": request,
                "answer": answer,
            }
        try:
            store.publish_once(task["receipt_path"], receipt)
        except FileExistsError:
            return store.read(task["receipt_path"])
        return receipt

    @app.function(**common, name=f"particle_{safe}", max_containers=128, timeout=3000)
    def particle_worker(task: dict[str, Any]) -> dict[str, Any]:
        from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
        from compose_v4.chem.state import pad_molecular_graph
        from compose_v4.control.route_complete_region_particle_receipts import (
            CompleteCombinationParticleReceiptStore,
        )
        from compose_v4.control.route_complete_region_particles import (
            CompleteCombinationJobSpec,
        )
        from compose_v4.experiments.t4_shared_controller_cell_runtime import (
            ReceiptStore,
        )

        contract, launch = _remote_context(task)
        if not contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
            raise RuntimeError("COMPOSE-NoDistill has no route-template particles")
        _bound_cell(contract, cell_key)
        volume.reload()
        run_root = _run_root(launch)
        durable = ReceiptStore(run_root, flush=volume.commit)
        payload = durable.read(task["manifest_path"])
        raw = dict(task["job"])
        raw.pop("job_id", None)
        spec = CompleteCombinationJobSpec(**raw)
        if task["job"] not in payload["jobs"]:
            raise ValueError("particle job is outside its parent manifest")
        store = CompleteCombinationParticleReceiptStore(
            run_root / task["particle_root"],
            {"payload": payload, "payload_sha256": payload_identity(payload)},
            flush=volume.commit,
        )
        source = pad_molecular_graph(smiles_to_molecular_graph(task["parent"]), 48)
        try:
            receipt, _ = store.run_job_once(source, _load_route_expert(contract), spec)
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            failed = _failed_particle_receipt(spec, payload)
            try:
                store.publish_receipt(spec, failed)
            except FileExistsError:
                return store.load_receipt(spec)
            return failed
        return receipt

    @app.function(**common, name=f"dock_{safe}", max_containers=8, timeout=1200)
    def dock_worker(task: dict[str, Any]) -> dict[str, Any]:
        import time

        from compose_v4.experiments.t4_docking_adapter import dock_t4
        from compose_v4.experiments.t4_shared_controller_cell_runtime import (
            ReceiptStore,
            execute_reserved_query_once,
        )

        contract, launch = _remote_context(task)
        cell = _bound_cell(contract, cell_key)
        volume.reload()
        store = ReceiptStore(_run_root(launch), flush=volume.commit)
        receptor = Path(f"/opt/dock/receptors/{cell['target']}.pdbqt")
        physical = {
            "qvina02_sha256": sha256_file(Path("/opt/dock/qvina02")),
            "receptor_sha256": sha256_file(receptor),
        }
        expected = cell["evaluator"]
        if physical != {
            "qvina02_sha256": expected["qvina02_sha256"],
            "receptor_sha256": expected["receptor_sha256"],
        }:
            raise ValueError("docking executable or receptor identity drift")

        def evaluate(smiles: str) -> dict[str, Any]:
            started = time.time()
            score = dock_t4(
                smiles,
                task["query_id"],
                contract["controller"]["docking_seed"],
                box={
                    "coordinates": expected["docking_box"],
                    "receptor": str(receptor),
                },
            )
            return {
                "score": score,
                "failure": None if score is not None else "oracle_no_score",
                "elapsed_seconds": time.time() - started,
                "evaluator_sha256": physical,
            }

        return execute_reserved_query_once(
            store,
            lock_path=task["lock_path"],
            reservation_path=task["reservation_path"],
            receipt_path=task["receipt_path"],
            query_id=task["query_id"],
            evaluate=evaluate,
        )

    @app.function(**common, name=f"driver_{safe}", max_containers=1, timeout=86_400)
    def driver(task: dict[str, Any]) -> dict[str, Any]:
        return _drive_cell(
            task=task,
            cell_key=cell_key,
            volume=volume,
            proposal_worker=proposal_worker,
            particle_worker=particle_worker,
            dock_worker=dock_worker,
        )

    @app.function(**common, name=f"status_{safe}", max_containers=1, timeout=120)
    def status(task: dict[str, Any]) -> dict[str, Any]:
        from compose_v4.experiments.t4_shared_controller_cell_runtime import (
            ReceiptStore,
        )

        contract, launch = _remote_context(task)
        _bound_cell(contract, cell_key)
        volume.reload()
        store = ReceiptStore(_run_root(launch))
        try:
            result = store.read("result.json")
        except FileNotFoundError:
            try:
                checkpoint = store.read("checkpoint.json")
            except FileNotFoundError:
                return {"cell_key": cell_key, "status": "not_started", "calls": 0}
            return {
                "cell_key": cell_key,
                "status": checkpoint["status"],
                "calls": checkpoint["charged_count"],
                "rounds": checkpoint["rounds_completed"],
            }
        return {
            "cell_key": cell_key,
            "status": result["status"],
            "calls": result["charged_calls"],
            "rounds": len(result.get("rounds", ())),
        }

    return {
        "volume": volume,
        "proposal": proposal_worker,
        "particle": particle_worker,
        "dock": dock_worker,
        "driver": driver,
        "status": status,
    }


def _read_optional(store: Any, path: str) -> dict[str, Any] | None:
    try:
        return store.read(path)
    except FileNotFoundError:
        return None


def _particle_store(*, store: Any, run_root: Path, relative_root: str, volume: Any):
    from compose_v4.control.route_complete_region_particle_receipts import (
        CompleteCombinationParticleReceiptStore,
    )

    payload = store.read(f"{relative_root}/manifest.json")
    result = CompleteCombinationParticleReceiptStore(
        run_root / relative_root,
        {"payload": payload, "payload_sha256": payload_identity(payload)},
        flush=volume.commit,
    )
    return result, payload


def _drive_cell(
    *,
    task: dict[str, Any],
    cell_key: str,
    volume: Any,
    proposal_worker: Any,
    particle_worker: Any,
    dock_worker: Any,
) -> dict[str, Any]:
    """Resume one cell only across durable, immutable receipt boundaries."""

    import time

    from compose_v4.control.route_complete_region_particle_receipts import (
        CompleteCombinationParticleReceiptStore,
    )
    from compose_v4.control.route_complete_region_particles import (
        CompleteCombinationJobSpec,
    )
    from compose_v4.experiments.t4_shared_controller_cell_runtime import (
        ReceiptStore,
        make_query_lock,
        settle_query_lock,
    )
    from compose_v4.experiments.t4_shared_controller_checkpoint import (
        RUNNING,
        apply_settled_observations,
        initialize_after_settled_root,
        select_parent_and_batch,
    )
    from compose_v4.experiments.t4_shared_controller_scored_runtime import (
        admit_runtime_proposal_pools,
        checkpoint_controller_config,
        make_proposal_manifest,
        particle_parent_manifest,
        preview_selected_parents,
    )

    contract, launch = _remote_context(task)
    cell = _bound_cell(contract, cell_key)
    settings = contract["operational_settings"]
    controller = checkpoint_controller_config(contract)
    run_root = _run_root(launch)
    volume.reload()
    store = ReceiptStore(run_root, flush=volume.commit)
    _publish_or_assert(store, "launch.json", launch)
    generation = task.get("driver_generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("driver task omitted its nonnegative continuation generation")
    confirmation = task.get("confirmed_prior_call_terminal")
    if confirmation is not (generation > 0):
        raise ValueError("driver generation has invalid prior-terminal confirmation")
    _publish_or_assert(
        store,
        f"driver_generations/generation_{generation:03d}.json",
        {
            "schema_version": "t4_shared_controller_driver_generation_v1",
            "generation": generation,
            "confirmed_prior_call_terminal": confirmation,
            "query_workers_resubmitted": False,
        },
    )
    result = _read_optional(store, "result.json")
    if result is not None:
        return result
    checkpoint = _read_optional(store, "checkpoint.json")

    if checkpoint is None:
        lock_path = "root/query_lock.json"
        lock = _read_optional(store, lock_path)
        if lock is None:
            lock = make_query_lock(
                cell_key=cell_key,
                round_index=0,
                charged_before=0,
                selected_rows=[{"smiles": cell["source_smiles"]}],
                query_deadline=time.time() + settings["query_wait_seconds"],
            )
            store.publish_once(lock_path, lock)
        query = lock["queries"][0]
        dispatch_path = "root/query_dispatch_intent.json"
        if _read_optional(store, dispatch_path) is None:
            store.publish_once(
                dispatch_path,
                {
                    "schema_version": "t4_shared_controller_query_dispatch_v1",
                    "query_lock_payload_sha256": payload_identity(lock),
                    "query_ids": [query["query_id"]],
                    "automatic_retries": 0,
                },
            )
            dock_worker.spawn(
                {
                    "launch": launch,
                    "lock_path": lock_path,
                    "reservation_path": (
                        f"root/query_reservations/{query['query_id']}.json"
                    ),
                    "receipt_path": f"root/query_receipts/{query['query_id']}.json",
                    "query_id": query["query_id"],
                }
            )
        while True:
            time.sleep(settings["driver_poll_seconds"])
            volume.reload()
            receipt = _read_optional(
                store, f"root/query_receipts/{query['query_id']}.json"
            )
            receipts = {} if receipt is None else {query["query_id"]: receipt}
            settlement = settle_query_lock(lock, receipts, now=time.time())
            if settlement["action"] != "wait":
                break
        observation = settlement["observations"][0]
        if observation.get("score") is None:
            result = {
                "schema_version": "t4_shared_controller_scored_cell_result_v1",
                "status": "root_oracle_failure",
                "terminal_reason": observation["failure"],
                "run_id": launch["run_id"],
                "contract_payload_sha256": launch["contract_payload_sha256"],
                "cell_key": cell_key,
                "charged_calls": 1,
                "charged_call_ceiling": 49,
                "no_retry_replacement_or_backfill": True,
            }
            store.publish_once("result.json", result)
            return result
        checkpoint = initialize_after_settled_root(
            cell_key=cell_key,
            source_smiles=cell["source_smiles"],
            delta=cell["delta"],
            budget_ceiling=contract["charged_calls_per_cell"],
            controller_config=controller,
            controller_seed=cell["controller_seed"],
            root_query_lock=lock,
            settled_root=settlement,
        )
        _publish_or_assert(store, "checkpoints/round_000.json", checkpoint)
        store.replace("checkpoint.json", checkpoint)

    while checkpoint["status"] == RUNNING:
        round_index = checkpoint["rounds_completed"] + 1
        round_root = f"rounds/{round_index:03d}"
        manifest_path = f"{round_root}/proposal_manifest.json"
        manifest = _read_optional(store, manifest_path)
        if manifest is None:
            parents = preview_selected_parents(checkpoint, cell=cell, contract=contract)
            parent_manifests = {}
            if contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
                expert = _load_route_expert(contract)
                for parent_index, parent in enumerate(parents):
                    relative_root = f"{round_root}/particles/p{parent_index:02d}"
                    parent_manifest = particle_parent_manifest(
                        parent=parent,
                        shared_checkpoint_sha256=contract["shared_route_checkpoint"][
                            "sha256"
                        ],
                        expert=expert,
                        code_identities={
                            "source_capsule": launch["source_capsule_payload_sha256"]
                        },
                        config_identities={
                            "scored_contract": launch["contract_payload_sha256"],
                            "shared_route_checkpoint": contract[
                                "shared_route_checkpoint"
                            ]["payload_sha256"],
                        },
                    )
                    CompleteCombinationParticleReceiptStore(
                        run_root / relative_root,
                        parent_manifest,
                        flush=volume.commit,
                    )
                    parent_manifests[parent] = parent_manifest
            manifest = make_proposal_manifest(
                checkpoint=checkpoint,
                cell=cell,
                contract=contract,
                round_index=round_index,
                deadline=time.time() + settings["proposal_wait_seconds"],
                particle_manifests=parent_manifests,
            )
            store.publish_once(manifest_path, manifest)

        dispatch_path = f"{round_root}/proposal_dispatch_intent.json"
        if _read_optional(store, dispatch_path) is None:
            store.publish_once(
                dispatch_path,
                {
                    "schema_version": "t4_shared_controller_proposal_dispatch_v1",
                    "proposal_manifest_payload_sha256": payload_identity(manifest),
                    "expert_jobs": len(manifest["requests"]),
                    "particle_jobs": (
                        28 * len(manifest["parents"])
                        if contract.get("trajectory_distillation", {"enabled": True})[
                            "enabled"
                        ]
                        else 0
                    ),
                    "automatic_retries": 0,
                },
            )
            for request in manifest["requests"]:
                proposal_worker.spawn(
                    {
                        "launch": launch,
                        "manifest_path": manifest_path,
                        "receipt_path": (
                            f"{round_root}/proposal_receipts/"
                            f"p{request['parent_index']:02d}_{request['expert']}.json"
                        ),
                        "request": request,
                    }
                )
            if contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
                for parent_index, parent in enumerate(manifest["parents"]):
                    relative_root = f"{round_root}/particles/p{parent_index:02d}"
                    payload = store.read(f"{relative_root}/manifest.json")
                    for job in payload["jobs"]:
                        particle_worker.spawn(
                            {
                                "launch": launch,
                                "parent": parent,
                                "particle_root": relative_root,
                                "manifest_path": f"{relative_root}/manifest.json",
                                "job": job,
                            }
                        )

        while True:
            volume.reload()
            missing = any(
                _read_optional(
                    store,
                    f"{round_root}/proposal_receipts/"
                    f"p{request['parent_index']:02d}_{request['expert']}.json",
                )
                is None
                for request in manifest["requests"]
            )
            if contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
                for parent_index, _ in enumerate(manifest["parents"]):
                    relative_root = f"{round_root}/particles/p{parent_index:02d}"
                    particle_store, payload = _particle_store(
                        store=store,
                        run_root=run_root,
                        relative_root=relative_root,
                        volume=volume,
                    )
                    for job in payload["jobs"]:
                        values = dict(job)
                        values.pop("job_id", None)
                        try:
                            particle_store.load_receipt(
                                CompleteCombinationJobSpec(**values)
                            )
                        except FileNotFoundError:
                            missing = True
            if not missing or time.time() >= manifest["deadline"]:
                break
            time.sleep(settings["driver_poll_seconds"])

        pools: dict[str, list[dict[str, Any]]] = {
            expert: [] for expert in manifest["proposal_experts"]
        }
        route_by_parent = {parent: [] for parent in manifest["parents"]}
        receipt_statuses: dict[str, int] = {}
        for request in manifest["requests"]:
            receipt = _read_optional(
                store,
                f"{round_root}/proposal_receipts/"
                f"p{request['parent_index']:02d}_{request['expert']}.json",
            )
            status = "missing_at_deadline" if receipt is None else receipt.get("status")
            receipt_statuses[status] = receipt_statuses.get(status, 0) + 1
            if status != "complete":
                continue
            rows = receipt["answer"]["records"]
            if request["expert"] == "route_complete_region":
                route_by_parent[request["parent"]].extend(rows)
            else:
                pools[request["expert"]].extend(rows)

        particle_telemetry = []
        if contract.get("trajectory_distillation", {"enabled": True})["enabled"]:
            for parent_index, parent in enumerate(manifest["parents"]):
                relative_root = f"{round_root}/particles/p{parent_index:02d}"
                particle_store, payload = _particle_store(
                    store=store,
                    run_root=run_root,
                    relative_root=relative_root,
                    volume=volume,
                )
                for job in payload["jobs"]:
                    values = dict(job)
                    values.pop("job_id", None)
                    spec = CompleteCombinationJobSpec(**values)
                    try:
                        particle_store.load_receipt(spec)
                    except FileNotFoundError:
                        particle_store.mark_missing_at_deadline(spec)
                combined, telemetry = particle_store.settle(route_by_parent[parent])
                for row in combined:
                    row["parent"] = parent
                    row["parent_score"] = checkpoint["archive"][parent]
                    row["delta"] = cell["delta"]
                pools["route_complete_region"].extend(combined)
                particle_telemetry.append({"parent": parent, **telemetry})
        else:
            for parent in manifest["parents"]:
                pools["route_complete_region"].extend(route_by_parent[parent])
                particle_telemetry.append(
                    {
                        "parent": parent,
                        "status": "disabled_by_compose_nodistill",
                        "scheduled_jobs": 0,
                        "templates_loaded": 0,
                        "route_checkpoint_read": False,
                    }
                )

        pools, admission_ledger = admit_runtime_proposal_pools(
            pools,
            cell=cell,
            contract=contract,
        )
        _publish_or_assert(
            store,
            f"{round_root}/selection_input.json",
            {
                "schema_version": "t4_shared_controller_selection_input_v1",
                "round": round_index,
                "proposal_manifest_payload_sha256": payload_identity(manifest),
                "expert_receipt_statuses": dict(sorted(receipt_statuses.items())),
                "eligible_by_expert": {
                    expert_name: len(rows) for expert_name, rows in pools.items()
                },
                "runtime_admission": admission_ledger,
                "stale_braf_exclusions": admission_ledger["stale_query_exclusions"],
                "particle_telemetry": particle_telemetry,
            },
        )

        plan_path = f"{round_root}/round_plan.json"
        plan = _read_optional(store, plan_path)
        if plan is None:
            plan = select_parent_and_batch(
                checkpoint,
                pools,
                cell_key=cell_key,
                source_smiles=cell["source_smiles"],
                delta=cell["delta"],
                budget_ceiling=contract["charged_calls_per_cell"],
                controller_config=controller,
                query_deadline=time.time() + settings["query_wait_seconds"],
            )
            store.publish_once(plan_path, plan)
        if plan["status"] != "pending_settlement":
            checkpoint = plan["checkpoint"]
            _publish_or_assert(
                store, f"checkpoints/round_{round_index:03d}.json", checkpoint
            )
            store.replace("checkpoint.json", checkpoint)
            break

        lock_path = f"{round_root}/query_lock.json"
        _publish_or_assert(store, lock_path, plan["query_lock"])
        query_dispatch_path = f"{round_root}/query_dispatch_intent.json"
        if _read_optional(store, query_dispatch_path) is None:
            store.publish_once(
                query_dispatch_path,
                {
                    "schema_version": "t4_shared_controller_query_dispatch_v1",
                    "query_lock_payload_sha256": plan["query_lock_payload_sha256"],
                    "query_ids": [
                        row["query_id"] for row in plan["query_lock"]["queries"]
                    ],
                    "automatic_retries": 0,
                },
            )
            for query in plan["query_lock"]["queries"]:
                dock_worker.spawn(
                    {
                        "launch": launch,
                        "lock_path": lock_path,
                        "reservation_path": (
                            f"{round_root}/query_reservations/"
                            f"{query['query_id']}.json"
                        ),
                        "receipt_path": (
                            f"{round_root}/query_receipts/{query['query_id']}.json"
                        ),
                        "query_id": query["query_id"],
                    }
                )

        while True:
            time.sleep(settings["driver_poll_seconds"])
            volume.reload()
            receipts = {}
            for query in plan["query_lock"]["queries"]:
                receipt = _read_optional(
                    store,
                    f"{round_root}/query_receipts/{query['query_id']}.json",
                )
                if receipt is not None:
                    receipts[query["query_id"]] = receipt
            settlement = settle_query_lock(
                plan["query_lock"], receipts, now=time.time()
            )
            if settlement["action"] != "wait":
                break
        checkpoint = apply_settled_observations(
            checkpoint,
            plan,
            settlement,
            cell_key=cell_key,
            source_smiles=cell["source_smiles"],
            delta=cell["delta"],
            budget_ceiling=contract["charged_calls_per_cell"],
            controller_config=controller,
        )
        _publish_or_assert(
            store, f"checkpoints/round_{round_index:03d}.json", checkpoint
        )
        store.replace("checkpoint.json", checkpoint)

    result = {
        "schema_version": "t4_shared_controller_scored_cell_result_v1",
        "status": checkpoint["status"],
        "terminal_reason": checkpoint["terminal_reason"],
        "run_id": launch["run_id"],
        "contract_payload_sha256": launch["contract_payload_sha256"],
        "cell_key": cell_key,
        "charged_calls": checkpoint["charged_count"],
        "charged_call_ceiling": contract["charged_calls_per_cell"],
        "archive": checkpoint["archive"],
        "rounds": checkpoint["rounds"],
        "final_best": min(checkpoint["archive"].values()),
        "no_retry_replacement_or_backfill": True,
        "claim_boundary": contract["claim_boundary"],
    }
    _publish_or_assert(store, "result.json", result)
    return result


for _cell_binding in CELLS:
    CELL_FUNCTIONS[_cell_binding["cell_key"]] = _register_cell(_cell_binding)


def _launch_receipt_path(run_id: str) -> Path:
    if _CAMPAIGN_VARIANT == "nodistill_parp1_v1":
        return (
            ROOT
            / "diagnostics/t4_compose_nodistill_parp1_v1/attempt_4/launches"
            / f"{run_id}.json"
        )
    if _CAMPAIGN_VARIANT == "nodistill_transfer_v1":
        return (
            ROOT
            / "diagnostics/t4_compose_nodistill_transfer_v1/attempt_1/launches"
            / f"{run_id}.json"
        )
    if _CAMPAIGN_VARIANT == "repaired_exhaustion_v1":
        return (
            ROOT
            / "diagnostics/t4_shared_controller_repaired_exhaustion_v1/attempt_1/launches"
            / f"{run_id}.json"
        )
    if _CAMPAIGN_VARIANT == "nodistill_feasibility_v4":
        return (
            ROOT
            / "diagnostics/t4_nodistill_feasibility_campaign_v4/attempt_1/launches"
            / f"{run_id}.json"
        )
    return (
        ROOT
        / "diagnostics/t4_shared_controller_completion_v1/launches"
        / f"{run_id}.json"
    )


def _publish_launch_receipt(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing duplicate scored launch receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _replace_launch_receipt(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"launch receipt does not exist: {path}")
    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _plan_driver_resume(
    receipt: dict[str, Any],
    *,
    terminal_by_cell: dict[str, bool],
    phase_status_by_cell: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Independently resume only cells whose prior call is confirmed terminal."""

    from compose_v4.experiments.t4_shared_controller_cell_runtime import (
        mark_driver_terminal,
        reserve_driver_generation,
    )

    cells = receipt.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(receipt["launch"]["cell_keys"]):
        raise ValueError("launch receipt driver-cell census drift")
    if set(terminal_by_cell) != set(cells) or set(phase_status_by_cell) != set(cells):
        raise ValueError("resume confirmation or status census drift")
    planned = {}
    for cell_key in receipt["launch"]["cell_keys"]:
        state = cells[cell_key]["driver_state"]
        if not terminal_by_cell[cell_key]:
            planned[cell_key] = reserve_driver_generation(
                phase_status=phase_status_by_cell[cell_key],
                existing_state=state,
            )
            continue
        if state["state"] in {"reserved", "running"}:
            state = mark_driver_terminal(state)
        elif state["state"] != "terminal":
            raise ValueError("launch receipt contains an invalid driver state")
        planned[cell_key] = reserve_driver_generation(
            phase_status=phase_status_by_cell[cell_key],
            existing_state=state,
            confirmed_prior_call_terminal=True,
        )
    return planned


def _modal_call_is_terminal(function_call_id: str) -> bool:
    """Ask Modal for authoritative terminal state without launching work."""

    from modal.call_graph import InputStatus

    call = modal.FunctionCall.from_id(function_call_id)
    matches = [
        row for row in call.get_call_graph() if row.function_call_id == function_call_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Modal call graph omitted or duplicated the driver call")
    if matches[0].status == InputStatus.PENDING:
        return False
    if matches[0].status not in {
        InputStatus.SUCCESS,
        InputStatus.FAILURE,
        InputStatus.INIT_FAILURE,
        InputStatus.TERMINATED,
        InputStatus.TIMEOUT,
    }:
        raise RuntimeError("Modal returned an unknown driver call state")
    return True


def _spawn_reserved_drivers(
    *, path: Path, receipt: dict[str, Any], planned: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Publish every reservation before spawning, then record each call identity."""

    from compose_v4.experiments.t4_shared_controller_cell_runtime import (
        mark_driver_running,
    )

    updated = json.loads(json.dumps(receipt))
    for cell_key, decision in planned.items():
        if decision.get("state") is not None:
            updated["cells"][cell_key]["driver_state"] = decision["state"]
    _replace_launch_receipt(path, updated)
    for cell_key in updated["launch"]["cell_keys"]:
        decision = planned[cell_key]
        if decision["action"] != "spawn":
            continue
        generation = decision["state"]["generation"]
        call = CELL_FUNCTIONS[cell_key]["driver"].spawn(
            {
                "launch": updated["launch"],
                "driver_generation": generation,
                "confirmed_prior_call_terminal": generation > 0,
            }
        )
        updated["cells"][cell_key]["driver_state"] = mark_driver_running(
            decision["state"], call.object_id
        )
        _replace_launch_receipt(path, updated)
    return updated


@app.local_entrypoint()
def main(mode: str = "preflight", run_id: str = "") -> None:
    if mode == "preflight":
        print(json.dumps(scored_preflight_report(), indent=2, sort_keys=True))
        return
    if mode == "launch":
        from compose_v4.experiments.t4_shared_controller_cell_runtime import (
            reserve_driver_generation,
        )

        context = local_scored_context()
        launch = context["launch"]
        receipt_path = _launch_receipt_path(launch["run_id"])
        if receipt_path.exists():
            raise FileExistsError("this exact scored payload was already launched")
        planned = {
            cell_key: reserve_driver_generation(
                phase_status="running", existing_state=None
            )
            for cell_key in launch["cell_keys"]
        }
        cells = {
            cell_key: {
                "volume": CELL_VOLUMES[cell_key],
                "driver_state": planned[cell_key]["state"],
            }
            for cell_key in launch["cell_keys"]
        }
        receipt = {
            "schema_version": "t4_shared_controller_scored_launch_receipt_v1",
            "launch": launch,
            "cells": cells,
            "driver_count": len(launch["cell_keys"]),
            "automatic_retries": 0,
            "replacement": False,
            "backfill": False,
        }
        _publish_launch_receipt(receipt_path, receipt)
        receipt = _spawn_reserved_drivers(
            path=receipt_path, receipt=receipt, planned=planned
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    if mode not in {"resume", "status"} or not run_id:
        raise ValueError("use mode=preflight, launch, or resume/status with run_id")
    receipt, _ = _load_envelope(_launch_receipt_path(run_id))
    launch = receipt["launch"]
    if launch["run_id"] != run_id:
        raise ValueError("local launch receipt run identity drift")
    if mode == "resume":
        context = local_scored_context()
        if context["launch"] != launch:
            raise ValueError("resume authority differs from the original launch")
        terminal_by_cell = {}
        for cell_key in launch["cell_keys"]:
            state = receipt["cells"][cell_key]["driver_state"]
            if state["state"] == "terminal":
                terminal_by_cell[cell_key] = True
                continue
            function_call_id = state.get("function_call_id")
            terminal_by_cell[cell_key] = bool(function_call_id) and (
                _modal_call_is_terminal(function_call_id)
            )
        status_rows = {}
        for cell_key in launch["cell_keys"]:
            if terminal_by_cell[cell_key]:
                status_rows[cell_key] = CELL_FUNCTIONS[cell_key]["status"].remote(
                    {"launch": launch}
                )
            else:
                status_rows[cell_key] = {"status": "running"}
        phase_status_by_cell = {
            cell_key: (
                "running"
                if row["status"] in {"not_started", "running"}
                else row["status"]
            )
            for cell_key, row in status_rows.items()
        }
        planned = _plan_driver_resume(
            receipt,
            terminal_by_cell=terminal_by_cell,
            phase_status_by_cell=phase_status_by_cell,
        )
        receipt = _spawn_reserved_drivers(
            path=_launch_receipt_path(run_id), receipt=receipt, planned=planned
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    records = [
        CELL_FUNCTIONS[cell_key]["status"].remote({"launch": launch})
        for cell_key in launch["cell_keys"]
    ]
    print(json.dumps({"run_id": run_id, "cells": records}, indent=2, sort_keys=True))


__all__ = [
    "APP_NAME",
    "CELL_FUNCTIONS",
    "CELL_KEYS",
    "CELL_VOLUMES",
    "app",
    "local_scored_context",
    "main",
    "scored_preflight_report",
]
