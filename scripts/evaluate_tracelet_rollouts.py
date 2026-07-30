"""Evaluate target-free ancestral rollouts from a self-contained checkpoint.

This entrypoint deliberately does not load compiled source-to-target teachers.
Selected checkpoints contain the model architecture, typed ring catalog, and
source prior needed for inference.  Teacher-conditioned validation belongs to
the training/evaluation entrypoint; endpoint and trajectory metrics do not.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, molecular_graph_to_smiles
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.eval.ring_calibration import undesirable_small_ring_mask
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
from compose_v4.experiments.cnof_conditional import (
    corpus_rollout_metrics,
    validate_rollout_trajectory_diagnostics,
)
from compose_v4.experiments.calibrated_rewrite_sampling import (
    ThinnedRateCalibrationSampler,
)
from compose_v4.experiments.parallel_tracelet_sampling import (
    sample_tracelet_ancestral_many,
)
from compose_v4.experiments.p50_completion import (
    validate_p50_completion_member,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkEmpiricalPriors,
    FactorizedTraceletRateModel,
    MARK_RULE_NAMES,
)


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_write(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _historical_optional_bool(
    payload: dict[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    """Read optional legacy metadata without accepting truthy strings or null."""

    if key not in payload:
        return default
    value = payload[key]
    if type(value) is not bool:
        raise ValueError(
            f"checkpoint metadata {key!r} must be a literal Boolean when present"
        )
    return value


def load_reusable_rollouts(
    path: Path,
    *,
    signature: dict[str, object],
    samples: int,
    require_trajectory_diagnostics: bool = True,
) -> tuple[object, ...] | None:
    """Load a complete compatible cache or fail closed on stale artifacts."""

    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("rollout cache payload must be a dictionary")
    if payload.get("signature") != signature:
        raise ValueError("rollout cache signature does not match this evaluation")
    raw_rollouts = payload.get("rollouts")
    if not isinstance(raw_rollouts, (tuple, list)):
        raise ValueError("rollout cache lacks a rollout sequence")
    rollouts = tuple(raw_rollouts)
    if len(rollouts) != samples:
        raise ValueError(
            "rollout cache size does not match requested samples: "
            f"{len(rollouts)} != {samples}"
        )
    if require_trajectory_diagnostics and any(
        not _has_complete_trajectory_diagnostics(rollout) for rollout in rollouts
    ):
        raise ValueError("rollout cache lacks complete compact trajectory diagnostics")
    return rollouts


def _has_complete_trajectory_diagnostics(rollout: object) -> bool:
    try:
        validate_rollout_trajectory_diagnostics(rollout)
    except ValueError:
        return False
    return True


def _read_smiles_file(path: Path, *, limit: int) -> tuple[str, ...]:
    smiles: list[str] = []
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            smiles.append(fields[0])
            if len(smiles) >= limit:
                break
    if not smiles:
        raise ValueError(f"no SMILES found in quality reference file: {path}")
    return tuple(smiles)


def load_factorized_rollout_checkpoint(
    path: Path,
    *,
    expected_scope_hash: str | None = None,
) -> tuple[FactorizedTraceletRateModel, dict[str, object]]:
    """Reconstruct a rollout-ready model and validate inference metadata. If ``expected_scope_hash`` is
    given, the checkpoint's persisted ``corpus_scope_hash`` must match it -- loading a model trained under a
    different corpus scope (e.g. CNOF-neutral vs broad-organic) fails loudly rather than sampling silently
    off-scope."""

    raw_payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw_payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    payload = dict(raw_payload)
    validate_p50_completion_member(path, payload)
    if (
        "state_dict" not in payload
        and payload.get("checkpoint_kind") == "exact_training_recovery"
    ):
        missing_recovery = sorted(
            {"best_state_dict", "best_metrics"} - payload.keys()
        )
        if missing_recovery:
            raise ValueError(
                "recovery checkpoint lacks selected-best rollout state: "
                f"{missing_recovery}"
            )
        # A live recovery artifact contains both the last optimizer state and
        # the validation-selected state.  Preview evaluation must sample the
        # latter and must never resume or mutate training.
        payload["state_dict"] = payload["best_state_dict"]
        payload["selected_validation"] = payload["best_metrics"]
        payload["rollout_state_source"] = "recovery_best_state_dict"
    required = {
        "state_dict",
        "ring_catalog",
        "tree_source_prior",
        "hidden_dim",
        "message_passing_steps",
        "training_backend",
        "source_prior",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint lacks rollout metadata: {missing}")
    if expected_scope_hash is not None and payload.get("corpus_scope_hash") != expected_scope_hash:
        raise ValueError(
            f"checkpoint corpus_scope_hash {payload.get('corpus_scope_hash')!r} != expected "
            f"{expected_scope_hash!r}; refusing to load a model trained under a different corpus scope")
    if payload["training_backend"] != "factorized_marks":
        raise ValueError("rollout-only evaluation currently requires factorized_marks")
    if payload["source_prior"] != "carbon_tree":
        raise ValueError("rollout-only evaluation currently requires carbon_tree")
    empirical_prior_payload = payload.get("empirical_mark_priors")
    empirical_mark_priors = (
        None
        if empirical_prior_payload is None
        else FactorizedMarkEmpiricalPriors.from_dict(empirical_prior_payload)
    )
    # B-edit checkpoints carry the wider organic heads + editing families; reconstruct them from the
    # metadata (absent on de-novo B -> CNOF vocab + editing families off, byte-identical to before).
    corrupted_prior_mix = _historical_optional_bool(
        payload,
        "corrupted_prior_mix",
        default=False,
    )
    enable_ring_restates = _historical_optional_bool(
        payload,
        "enable_ring_restates",
        default=corrupted_prior_mix,
    )
    model = FactorizedTraceletRateModel(
        payload["ring_catalog"],
        hidden_dim=int(payload["hidden_dim"]),
        message_passing_steps=int(payload["message_passing_steps"]),
        ring_electronic_mode=str(
            payload.get("ring_electronic_mode", "factorized_local")
        ),
        rate_factorization=str(payload.get("rate_factorization", "hierarchical")),
        property_condition_dim=int(payload.get("property_condition_dim", 0)),
        empirical_mark_prior_mode=str(
            payload.get("empirical_mark_prior_mode", "none")
        ),
        empirical_mark_priors=empirical_mark_priors,
        ring_family_mass_mode=str(payload.get("ring_family_mass_mode", "boolean")),
        ring_template_factorization=str(
            payload.get("ring_template_factorization", "flat")
        ),
        atom_vocabulary=(
            ORGANIC_VOCABULARY if bool(payload.get("organic_vocabulary")) else None
        ),
        enable_ring_restates=enable_ring_restates,
        enable_cyclic_graft=corrupted_prior_mix,
        enable_heteroatom_scan=corrupted_prior_mix,
        enable_ring_opening=corrupted_prior_mix,
        # Compositional ring-op families (cycle_close/cycle_open); absent metadata -> off -> byte-identical.
        enable_cycle_ops=bool(payload.get("enable_cycle_ops")),
        # RING_CORE_V1 disables the legacy whole-ring grow macro; absent metadata -> True -> byte-identical.
        enable_ring_grow_macro=bool(payload.get("enable_ring_grow_macro", True)),
        # Historical checkpoints always exposed whole-ring deletion.  V2 may
        # explicitly disable that accelerator while keeping primitive opening.
        enable_ring_system_delete=_historical_optional_bool(
            payload,
            "enable_ring_system_delete",
            default=True,
        ),
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    legacy_role_keys = {
        name
        for name in model.state_dict()
        if name.startswith("ring_system_role_head.")
    }
    legacy_v0 = (
        not unexpected_keys
        and missing_keys == legacy_role_keys
        and "ring_electronic_mode" not in payload
        and int(
            getattr(
                payload["ring_catalog"],
                "ring_system_electronic_alias_version",
                0,
            )
        )
        == 0
    )
    if legacy_v0:
        # Legacy checkpoints learned only the C/N/O/F factor.  The executable
        # completion predicate supplied the allowable electronic role.  Zero
        # logits reproduce that neutral factor and avoid random untrained role
        # preferences while retaining the current exact validity mask.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in legacy_role_keys:
                    parameter.zero_()
        payload["checkpoint_compatibility"] = (
            "legacy_v0_neutral_ring_role_logits"
        )
        model.virtualize_legacy_self_grafts = True
    elif missing_keys or unexpected_keys:
        raise ValueError(
            "checkpoint/model state mismatch: "
            f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
        )
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-reference-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--corpus-workers", type=int, default=16)
    parser.add_argument("--rollout-samples", type=int, default=2000)
    parser.add_argument("--rollout-workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--operational-horizon", type=float, default=16.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--quality-reference-limit", type=int, default=5000)
    parser.add_argument("--fcd-generated-limit", type=int, default=2000)
    parser.add_argument(
        "--disable-rule",
        action="append",
        default=[],
        help="Disable one production rewrite family during sampling only.",
    )
    parser.add_argument(
        "--exclude-ring-template-cycle-size-at-most",
        type=int,
        default=None,
        help=(
            "Inference-only calibration ablation: exclude complete ring-system "
            "templates containing a cycle no larger than this value."
        ),
    )
    parser.add_argument(
        "--atom-delete-log-rate-adjustment",
        type=float,
        default=0.0,
        help=(
            "Non-positive inference-only log multiplier for atom-delete rates; "
            "implemented by exact CTMC thinning without family renormalization."
        ),
    )
    parser.add_argument(
        "--small-ring-log-rate-adjustment",
        type=float,
        default=0.0,
        help=(
            "Non-positive inference-only log multiplier for whole-ring actions "
            "containing a 3/4-member cycle; implemented by exact CTMC thinning."
        ),
    )
    parser.add_argument(
        "--triple-bond-log-rate-adjustment",
        type=float,
        default=0.0,
        help=(
            "Non-positive inference-only log multiplier for atom-insert/bond-reorder "
            "marks that install a triple bond; implemented by exact CTMC thinning "
            "with positive support preserved (a soft rate reduction, not a ban)."
        ),
    )
    parser.add_argument("--fast-split", action="store_true")
    args = parser.parse_args()

    if args.rollout_samples <= 0 or args.rollout_workers <= 0:
        raise ValueError("rollout samples and workers must be positive")
    if args.max_atoms <= 0 or args.max_events <= 0:
        raise ValueError("max atoms and events must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    model, checkpoint = load_factorized_rollout_checkpoint(args.checkpoint)
    unknown_disabled_rules = sorted(set(args.disable_rule) - set(MARK_RULE_NAMES))
    if unknown_disabled_rules:
        raise ValueError(f"unknown disabled rewrite rules: {unknown_disabled_rules}")
    model.disabled_sampling_rule_names = frozenset(args.disable_rule)
    if (
        args.exclude_ring_template_cycle_size_at_most is not None
        and args.exclude_ring_template_cycle_size_at_most < 3
    ):
        raise ValueError("excluded ring-template cycle size must be at least three")
    excluded_ring_template_indices: tuple[int, ...] = ()
    if args.exclude_ring_template_cycle_size_at_most is not None:
        excluded_mask = undesirable_small_ring_mask(
            model.ring_system_templates,
            maximum_size=args.exclude_ring_template_cycle_size_at_most,
        )
        excluded_ring_template_indices = tuple(
            int(index) for index in torch.nonzero(excluded_mask).flatten().tolist()
        )
    model.excluded_sampling_ring_template_indices = excluded_ring_template_indices
    calibrated_sampler = ThinnedRateCalibrationSampler(
        model,
        family_log_rate_adjustments=(
            ("atom_delete", float(args.atom_delete_log_rate_adjustment)),
        ),
        small_ring_log_rate_adjustment=float(
            args.small_ring_log_rate_adjustment
        ),
        triple_bond_log_rate_adjustment=float(
            args.triple_bond_log_rate_adjustment
        ),
    )
    source_prior = checkpoint["tree_source_prior"]
    if max(source_prior.sizes) > args.max_atoms:
        raise ValueError("checkpoint source prior exceeds --max-atoms")
    print(
        json.dumps(
            {
                "phase": "rollout_checkpoint_loaded",
                "path": str(args.checkpoint),
                "selected_validation": checkpoint.get("selected_validation"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=not args.fast_split,
        workers=args.corpus_workers,
    )
    print(json.dumps({"phase": "rollout_split_loaded"}), flush=True)

    progress_interval = max(args.rollout_samples // 10, 1)

    def report_progress(completed: int, total: int) -> None:
        if completed % progress_interval == 0 or completed == total:
            print(
                json.dumps(
                    {"phase": "sampling", "completed": completed, "total": total},
                    sort_keys=True,
                ),
                flush=True,
            )

    rollout_signature: dict[str, object] = {
        "format": "compose_v4_rollout_cache_v3_compact_trajectories",
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "rollout_samples": args.rollout_samples,
        "sampling_seed": args.seed + 4,
        "max_atoms": args.max_atoms,
        "operational_horizon": args.operational_horizon,
        "time_step": args.time_step,
        "max_events": args.max_events,
        "disabled_rule_names": sorted(model.disabled_sampling_rule_names),
        "excluded_ring_template_indices": excluded_ring_template_indices,
        "rate_calibration": calibrated_sampler.calibration_signature,
        "source_prior": "carbon_tree",
    }
    rollouts = load_reusable_rollouts(
        args.rollout_cache,
        signature=rollout_signature,
        samples=args.rollout_samples,
    )
    if rollouts is None:
        rollouts = sample_tracelet_ancestral_many(
            calibrated_sampler,
            seed=args.seed + 4,
            samples=args.rollout_samples,
            workers=args.rollout_workers,
            n_slots=args.max_atoms,
            operational_horizon=args.operational_horizon,
            time_step=args.time_step,
            max_events=args.max_events,
            torch_threads_per_worker=args.torch_threads,
            progress_callback=report_progress,
            source_prior=source_prior,
        )
        if any(not _has_complete_trajectory_diagnostics(item) for item in rollouts):
            raise RuntimeError("fresh rollouts lack compact trajectory diagnostics")
        _atomic_torch_save(
            {
                "rollouts": rollouts,
                "signature": rollout_signature,
                "checkpoint": str(args.checkpoint),
                "seed": args.seed + 4,
                "n_slots": args.max_atoms,
                "operational_horizon": args.operational_horizon,
                "time_step": args.time_step,
                "max_events": args.max_events,
                "source_prior": "carbon_tree",
                "tree_source_prior": source_prior,
            },
            args.rollout_cache,
        )
        print(
            json.dumps(
                {"phase": "rollouts_saved", "path": str(args.rollout_cache)},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {"phase": "rollouts_reused", "path": str(args.rollout_cache)},
                sort_keys=True,
            ),
            flush=True,
        )

    generated_smiles = tuple(
        text
        for rollout in rollouts
        if (text := molecular_graph_to_smiles(rollout.final_state)) is not None
    )
    event_counts = Counter(rule for rollout in rollouts for rule in rollout.event_rules)
    quality_reference = _read_smiles_file(
        args.quality_reference_file,
        limit=args.quality_reference_limit,
    )
    rollout_metrics = corpus_rollout_metrics(
        rollouts,
        train_smiles=split.train,
        reference_smiles=split.test,
    )
    report: dict[str, object] = {
        "evaluation_kind": "checkpoint_rollout_only",
        "teacher_path_cache_loaded": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": rollout_signature["checkpoint_sha256"],
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "selected_validation": checkpoint.get("selected_validation"),
        "seed": args.seed,
        "source_prior": {
            "kind": "carbon_tree",
            "tree_size_prior": checkpoint.get("tree_size_prior"),
            "sizes": source_prior.sizes,
            "probabilities": source_prior.probabilities,
        },
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "eligible": split.eligible_molecules,
            "scanned_lines": split.scanned_lines,
            "max_atoms": args.max_atoms,
        },
        "generated_nonnull_smiles": len(generated_smiles),
        "generated_smiles": generated_smiles,
        "rollout": rollout_metrics,
        "trajectory_diagnostics_available": bool(
            rollout_metrics.get("trajectory_diagnostics", {}).get("available")
        ),
        "rollout_event_counts": dict(sorted(event_counts.items())),
        "generated_ring_taxonomy": (
            ring_taxonomy_report(generated_smiles) if generated_smiles else None
        ),
        "reference_ring_taxonomy": ring_taxonomy_report(split.test),
        "sampler": {
            "target_available": False,
            "beam_search": False,
            "ancestral_ctmc": True,
            "operational_horizon": args.operational_horizon,
            "time_step": args.time_step,
            "max_events": args.max_events,
            "workers": args.rollout_workers,
            "seed_scheme": "seed_sequence_per_trajectory_v1",
            "disabled_rule_names": sorted(model.disabled_sampling_rule_names),
            "excluded_ring_template_cycle_size_at_most": (
                args.exclude_ring_template_cycle_size_at_most
            ),
            "excluded_ring_template_count": len(excluded_ring_template_indices),
            "rate_calibration": calibrated_sampler.calibration_signature,
        },
    }
    if generated_smiles:
        report["molecular_quality"] = molecular_quality_report(
            generated_smiles,
            reference_smiles=quality_reference,
            train_smiles=split.train,
            include_fcd=True,
            fcd_reference_limit=args.quality_reference_limit,
            fcd_generated_limit=args.fcd_generated_limit,
            fcd_device="cpu",
            seed=args.seed + 5,
        )
    report["molecular_quality_reference"] = {
        "file": str(args.quality_reference_file),
        "source": "external",
        "count": len(quality_reference),
    }
    _atomic_json_write(report, args.output)
    print(
        json.dumps(
            {
                "phase": "rollout_evaluation_complete",
                "generated_nonnull_smiles": len(generated_smiles),
                "valid_fraction": report["rollout"].get("valid_fraction"),
                "fcd": report.get("molecular_quality", {}).get(
                    "frechet_chemnet_distance"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
