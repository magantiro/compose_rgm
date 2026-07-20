"""Bounded Modal pilot for the qualified frozen-backbone QED sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
LOCAL_CHECKPOINT = Path("/private/tmp/pancake_checkpoint/checkpoint.recovery.pt")
LOCAL_PARTITION = Path("/private/tmp/compose_v4_stage3_paths_manifest.pt")
LOCAL_QUALIFICATION = ROOT / "diagnostics/canonical_successor_analytic_backbone_qualification.json"
LOCAL_CONFIG = (
    ROOT
    / "configs/experiments/griddd_qed_frozen_residual_pilot_v5_canonical.json"
)
REMOTE_CHECKPOINT = REMOTE_ROOT / "inputs/checkpoint.recovery.pt"
REMOTE_PARTITION = REMOTE_ROOT / "inputs/compose_v4_stage3_paths_manifest.pt"
REMOTE_QUALIFICATION = REMOTE_ROOT / "inputs/canonical_successor_analytic_backbone_qualification.json"
REMOTE_CONFIG = (
    REMOTE_ROOT
    / "configs/experiments/griddd_qed_frozen_residual_pilot_v5_canonical.json"
)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_file(LOCAL_CHECKPOINT, str(REMOTE_CHECKPOINT), copy=True)
    .add_local_file(LOCAL_PARTITION, str(REMOTE_PARTITION), copy=True)
    .add_local_file(LOCAL_QUALIFICATION, str(REMOTE_QUALIFICATION), copy=True)
)

app = modal.App("compose-v4-qed-frozen-residual")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_torch(path: Path, payload: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    *,
    adapter: object,
    optimizer: object,
    config: dict[str, object],
    qualification_sha256: str,
    step: int,
    best_validation_loss: float,
    evaluations_without_improvement: int,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "format": "compose_v4_frozen_qed_residual_checkpoint_v1",
        "checkpoint_kind": "frozen_conditional_sidecar",
        "step": int(step),
        "property_condition_dim": 1,
        "property_conditioning": {
            "names": ["qed"],
            "means": [float(config["property"]["normalizer_mean"])],  # type: ignore[index]
            "standard_deviations": [
                float(config["property"]["normalizer_standard_deviation"])  # type: ignore[index]
            ],
        },
        "conditional_adapter": adapter.sidecar_contract(),  # type: ignore[attr-defined]
        "sidecar_state_dict": adapter.state_dict(),  # type: ignore[attr-defined]
        "optimizer_state_dict": optimizer.state_dict(),  # type: ignore[attr-defined]
        "unconditional_backbone_qualification_manifest_sha256": qualification_sha256,
        "unconditional_backbone_checkpoint_sha256": config["base"][  # type: ignore[index]
            "checkpoint_sha256"
        ],
        "unconditional_backbone_execution_kind": "analytic_pancake_quotient_adapter_v1",
        "weight_source": "pancake_derived",
        "canonical_successor_execution": True,
        "molecular_self_transitions_virtualized": True,
        "empirical_mark_prior_mode": "none",
        "ring_family_mass_mode": "boolean",
        "p1_p2_imported": False,
        "best_validation_loss": float(best_validation_loss),
        "evaluations_without_improvement": int(evaluations_without_improvement),
        "history": history,
        "training_config": config,
        "training_config_sha256": _sha256(REMOTE_CONFIG),
        "base_weights_in_checkpoint": False,
    }


@app.function(
    image=image,
    gpu="A100",
    cpu=32.0,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={"/artifacts": artifact_volume},
)
def train_stage(run_label: str) -> dict[str, object]:
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import QED

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.canonical_successor_distillation import (
        AnalyticPancakeQuotientSampler,
        FrozenQEDResidualAdapter,
        FrozenQEDResidualConfig,
    )
    from compose_v4.experiments.factorized_mark_conditional import (
        FactorizedMarkDataset,
    )
    from compose_v4.experiments.griddd_conditional import (
        RETAINED_PANCAKE_CHECKPOINT_SHA256,
        load_unconditional_backbone_qualification,
    )
    from compose_v4.experiments.tracelet_conditional import (
        build_tree_transport_path_records,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        MARK_RULE_TO_INDEX,
    )
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.ring_system_fiber import (
        structured_ring_trace_supported,
    )
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    config = json.loads(REMOTE_CONFIG.read_text())
    if config.get("format") != "compose_v4_griddd_qed_frozen_residual_pilot_v1":
        raise ValueError("unsupported frozen-QED pilot config")
    if str(config["run_label"]) != run_label:
        raise ValueError("run label must exactly match the frozen pilot config")
    if int(config["optimization"]["steps"]) != 500:
        raise ValueError("bounded pilot must remain exactly 500 steps")
    if int(config["optimization"]["batch_size"]) != 1:
        raise ValueError("frozen pilot driver currently requires batch size one")
    checkpoint_sha256 = _sha256(REMOTE_CHECKPOINT)
    if checkpoint_sha256 != RETAINED_PANCAKE_CHECKPOINT_SHA256:
        raise ValueError("embedded base checkpoint is not the retained pancake")
    if _sha256(REMOTE_PARTITION) != str(
        config["data"]["partition_manifest_sha256"]
    ):
        raise ValueError("embedded training partition manifest hash mismatch")
    qualification = load_unconditional_backbone_qualification(REMOTE_QUALIFICATION)
    if qualification.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("qualification and embedded checkpoint disagree")
    qualification_sha256 = _sha256(REMOTE_QUALIFICATION)

    run_dir = Path("/artifacts") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "compose_v4_frozen_qed_residual_run_manifest_v1",
        "phase": "preparing_paths",
        "run_label": run_label,
        "config": config,
        "config_sha256": _sha256(REMOTE_CONFIG),
        "checkpoint_sha256": checkpoint_sha256,
        "partition_manifest_sha256": _sha256(REMOTE_PARTITION),
        "qualification_manifest_sha256": qualification_sha256,
        "started_unix_seconds": time.time(),
    }
    _write_json(run_dir / "run_config.json", manifest)
    artifact_volume.commit()

    torch.manual_seed(int(config["optimization"]["seed"]))
    np.random.seed(int(config["optimization"]["seed"]))
    base_model, base_payload = load_factorized_rollout_checkpoint(REMOTE_CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device).eval()
    adapter = FrozenQEDResidualAdapter(
        AnalyticPancakeQuotientSampler(base_model),
        FrozenQEDResidualConfig(
            qed_mean=float(config["property"]["normalizer_mean"]),
            qed_standard_deviation=float(
                config["property"]["normalizer_standard_deviation"]
            ),
            hidden_dim=int(config["adapter"]["hidden_dim"]),
        ),
    ).train()
    if any(parameter.requires_grad for parameter in base_model.parameters()):
        raise RuntimeError("base parameters are not frozen")
    if any("base" in key for key in adapter.state_dict()):
        raise RuntimeError("sidecar state dict unexpectedly contains base weights")

    partition = torch.load(
        REMOTE_PARTITION,
        map_location="cpu",
        weights_only=False,
    )
    signature = partition["signature"]
    train_pool = tuple(signature["train_smiles"])
    validation_pool = tuple(signature["validation_smiles"])
    seed = int(config["optimization"]["seed"])
    rng = np.random.default_rng(seed)
    train_count = int(config["data"]["train_endpoint_subset"])
    validation_count = int(config["data"]["validation_endpoint_subset"])
    train_indices = np.sort(rng.choice(len(train_pool), train_count, replace=False))
    validation_indices = np.sort(
        rng.choice(len(validation_pool), validation_count, replace=False)
    )
    train_smiles = tuple(train_pool[int(index)] for index in train_indices)
    validation_smiles = tuple(
        validation_pool[int(index)] for index in validation_indices
    )
    source_prior = base_payload.get("tree_source_prior")
    if source_prior is None:
        raise ValueError("retained checkpoint lacks its frozen tree source prior")
    train_proposal_records = build_tree_transport_path_records(
        train_smiles,
        n_slots=40,
        source_prior=source_prior,
        seed=seed,
        couplings_per_target=int(config["data"]["path_couplings_per_target"]),
        transport_mode=str(config["data"]["tree_transport"]),
        typed_ring_payloads=True,
        ring_catalog=None,
        workers=int(config["data"]["path_workers"]),
        checkpoint_interval=int(config["data"]["path_checkpoint_interval"]),
    )
    validation_proposal_records = build_tree_transport_path_records(
        validation_smiles,
        n_slots=40,
        source_prior=source_prior,
        seed=seed + 1,
        couplings_per_target=1,
        transport_mode=str(config["data"]["tree_transport"]),
        typed_ring_payloads=True,
        ring_catalog=None,
        workers=int(config["data"]["path_workers"]),
        checkpoint_interval=int(config["data"]["path_checkpoint_interval"]),
    )
    train_records = tuple(
        record
        for record in train_proposal_records
        if structured_ring_trace_supported(
            record.path.trace,
            base_model.ring_catalog,
        )
    )
    validation_records = tuple(
        record
        for record in validation_proposal_records
        if structured_ring_trace_supported(
            record.path.trace,
            base_model.ring_catalog,
        )
    )
    minimum_train = int(config["data"]["minimum_supported_train_records"])
    minimum_validation = int(
        config["data"]["minimum_supported_validation_records"]
    )
    manifest.update(
        {
            "phase": "path_support_filtered",
            "ring_catalog_support_policy": config["data"][
                "ring_catalog_support_policy"
            ],
            "train_path_proposals": len(train_proposal_records),
            "train_path_records": len(train_records),
            "train_paths_rejected_by_frozen_ring_catalog": (
                len(train_proposal_records) - len(train_records)
            ),
            "validation_path_proposals": len(validation_proposal_records),
            "validation_path_records": len(validation_records),
            "validation_paths_rejected_by_frozen_ring_catalog": (
                len(validation_proposal_records) - len(validation_records)
            ),
        }
    )
    _write_json(run_dir / "run_config.json", manifest)
    artifact_volume.commit()
    if len(train_records) < minimum_train:
        raise RuntimeError(
            "frozen ring catalog retained too few QED training paths: "
            f"{len(train_records)} < {minimum_train}"
        )
    if len(validation_records) < minimum_validation:
        raise RuntimeError(
            "frozen ring catalog retained too few QED validation paths: "
            f"{len(validation_records)} < {minimum_validation}"
        )

    def target_conditions(smiles_values: tuple[str, ...]) -> dict[str, tuple[float]]:
        result = {}
        for text in smiles_values:
            molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                raise ValueError("training partition contains an invalid molecule")
            state = pad_molecular_graph(smiles_to_molecular_graph(text), 40)
            result[canonical_state_key(state)] = (float(QED.qed(molecule)),)
        return result

    train_conditions = target_conditions(train_smiles)
    validation_conditions = target_conditions(validation_smiles)
    optimization = config["optimization"]
    train_dataset = FactorizedMarkDataset(
        train_records,
        start_index=0,
        length=int(optimization["steps"]),
        seed=seed,
        late_time_fraction=float(optimization["late_time_fraction"]),
        operational_horizon=float(optimization["operational_horizon"]),
        progress_stratification_fraction=float(
            optimization["progress_stratification_fraction"]
        ),
        target_property_conditions=train_conditions,
        condition_dropout_probability=float(
            config["property"]["condition_dropout_probability"]
        ),
        ring_family_mass_mode="boolean",
    )
    validation_dataset = FactorizedMarkDataset(
        validation_records,
        start_index=0,
        length=int(optimization["validation_examples"]),
        seed=seed + 2,
        late_time_fraction=float(optimization["late_time_fraction"]),
        operational_horizon=float(optimization["operational_horizon"]),
        progress_stratification_fraction=float(
            optimization["progress_stratification_fraction"]
        ),
        target_property_conditions=validation_conditions,
        condition_dropout_probability=0.0,
        ring_family_mass_mode="boolean",
    )
    manifest.update(
        {
            "phase": "training",
            "device": str(device),
            "train_endpoint_indices": [int(value) for value in train_indices],
            "validation_endpoint_indices": [
                int(value) for value in validation_indices
            ],
            "train_path_records": len(train_records),
            "validation_path_records": len(validation_records),
            "train_path_proposals": len(train_proposal_records),
            "validation_path_proposals": len(validation_proposal_records),
            "train_paths_rejected_by_frozen_ring_catalog": (
                len(train_proposal_records) - len(train_records)
            ),
            "validation_paths_rejected_by_frozen_ring_catalog": (
                len(validation_proposal_records) - len(validation_records)
            ),
        }
    )
    _write_json(run_dir / "run_config.json", manifest)
    artifact_volume.commit()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )

    def target_for(example: object) -> float | None:
        values = example.property_condition_values
        mask = example.property_condition_mask
        if values is None or mask is None:
            raise RuntimeError("QED training example lacks its target condition")
        return float(values[0]) if bool(mask[0]) else None

    @torch.no_grad()
    def evaluate(step: int) -> dict[str, object]:
        adapter.eval()
        losses = []
        family_hits = 0
        nonterminal = 0
        mean_delta = []
        identity_error = []
        for index in range(len(validation_dataset)):
            example = validation_dataset[index]
            target_qed = target_for(example)
            prediction, batch = adapter.forward_mark_example(
                example.state,
                example.time,
                target_qed=target_qed,
                teacher_rule_name=example.teacher_rule_name,
                teacher_action=example.teacher_action,
                teacher_rate=example.teacher_rate,
                importance_weight=example.importance_weight,
            )
            from compose_v4.model.factorized_tracelet_rate_model import (
                factorized_mark_bregman_loss,
            )

            losses.append(float(factorized_mark_bregman_loss(prediction, batch)))
            if example.teacher_rule_name is not None:
                teacher_family = MARK_RULE_TO_INDEX[example.teacher_rule_name]
                family_hits += int(
                    int(prediction.family_log_probabilities.argmax(dim=-1)[0])
                    == teacher_family
                )
                nonterminal += 1
            present = adapter.rate_table(
                example.state,
                example.time,
                target_qed=target_qed,
            )
            missing = adapter.rate_table(
                example.state,
                example.time,
                target_qed=None,
            )
            base = adapter.base_sampler.rate_table(example.state, example.time)
            mean_delta.append(
                float(present.family_log_hazard_residuals.abs().mean())
            )
            identity_error.append(
                float(
                    (
                        missing.family_rates - base.productive_family_rates
                    ).abs().max()
                )
            )
        adapter.train()
        return {
            "step": int(step),
            "validation_examples": len(validation_dataset),
            "validation_loss": float(np.mean(losses)),
            "family_top1_accuracy": (
                None if not nonterminal else family_hits / nonterminal
            ),
            "mean_absolute_family_log_hazard_residual": float(
                np.mean(mean_delta)
            ),
            "maximum_missing_condition_identity_error": float(
                max(identity_error, default=0.0)
            ),
        }

    history: list[dict[str, object]] = []
    initial = evaluate(0)
    history.append(initial)
    best_validation_loss = float(initial["validation_loss"])
    best_step = 0
    evaluations_without_improvement = 0
    best_payload = _checkpoint_payload(
        adapter=adapter,
        optimizer=optimizer,
        config=config,
        qualification_sha256=qualification_sha256,
        step=0,
        best_validation_loss=best_validation_loss,
        evaluations_without_improvement=0,
        history=history,
    )
    _save_torch(run_dir / "checkpoint.best.pt", best_payload)
    _write_json(run_dir / "evaluations/step_0000.json", initial)
    artifact_volume.commit()

    stopped_early = False
    train_started = time.perf_counter()
    last_step = 0
    for index in range(int(optimization["steps"])):
        step = index + 1
        example = train_dataset[index]
        optimizer.zero_grad(set_to_none=True)
        loss = adapter.loss_for_mark_example(
            example.state,
            example.time,
            target_qed=target_for(example),
            teacher_rule_name=example.teacher_rule_name,
            teacher_action=example.teacher_action,
            teacher_rate=example.teacher_rate,
            importance_weight=example.importance_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite sidecar loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            max_norm=float(optimization["gradient_clip_norm"]),
        )
        optimizer.step()
        last_step = step
        if step % int(optimization["evaluation_every"]) != 0:
            continue
        metrics = evaluate(step)
        metrics.update(
            {
                "train_loss_at_step": float(loss.detach()),
                "elapsed_training_seconds": time.perf_counter() - train_started,
            }
        )
        history.append(metrics)
        validation_loss = float(metrics["validation_loss"])
        relative = float(optimization["early_stopping_min_relative_improvement"])
        improved = validation_loss < best_validation_loss * (1.0 - relative)
        if improved:
            best_validation_loss = validation_loss
            best_step = step
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        payload = _checkpoint_payload(
            adapter=adapter,
            optimizer=optimizer,
            config=config,
            qualification_sha256=qualification_sha256,
            step=step,
            best_validation_loss=best_validation_loss,
            evaluations_without_improvement=evaluations_without_improvement,
            history=history,
        )
        _save_torch(run_dir / f"checkpoint.step{step:04d}.pt", payload)
        _save_torch(run_dir / "checkpoint.recovery.pt", payload)
        if improved:
            _save_torch(run_dir / "checkpoint.best.pt", payload)
        _write_json(run_dir / f"evaluations/step_{step:04d}.json", metrics)
        status = {
            "format": "compose_v4_frozen_qed_residual_status_v1",
            "phase": "training",
            "run_label": run_label,
            "current_step": step,
            "maximum_steps": int(optimization["steps"]),
            "best_step": best_step,
            "best_validation_loss": best_validation_loss,
            "evaluations_without_improvement": evaluations_without_improvement,
            "latest_metrics": metrics,
        }
        _write_json(run_dir / "status.json", status)
        artifact_volume.commit()
        print(json.dumps(status, sort_keys=True), flush=True)
        if evaluations_without_improvement >= int(
            optimization["early_stopping_patience"]
        ):
            stopped_early = True
            break

    final = {
        "format": "compose_v4_frozen_qed_residual_status_v1",
        "phase": "complete",
        "run_label": run_label,
        "completed_step": last_step,
        "maximum_steps": int(optimization["steps"]),
        "stopped_early": stopped_early,
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "base_parameters_frozen": True,
        "base_weights_in_sidecar_checkpoint": False,
        "protocol_a_launched": False,
        "protocol_b_ready_for_post_training_evaluation": True,
    }
    _write_json(run_dir / "status.json", final)
    _write_json(run_dir / "metrics.json", final)
    artifact_volume.commit()
    return final


@app.local_entrypoint()
def main(
    run_label: str = (
        "compose-v4-griddd-qed-frozen-residual-pilot-20260720-v5-canonical"
    ),
) -> None:
    config = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    frozen_run_label = str(config["run_label"])
    if run_label != frozen_run_label:
        raise ValueError(
            "run label must exactly match the frozen pilot config before spawn: "
            f"expected {frozen_run_label!r}, received {run_label!r}"
        )
    call = train_stage.spawn(run_label)
    print(
        json.dumps(
            {
                "phase": "qed_frozen_residual_training_spawned",
                "run_label": run_label,
                "function_call_id": call.object_id,
                "artifacts": f"compose-v4-artifacts/{run_label}",
            },
            sort_keys=True,
        )
    )
