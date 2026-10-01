"""Hash-verified, CPU-only inference loading for frozen molecular references.

The NLL-trained Editing-V2 law uses a weights-only checkpoint and a separate
versioned ring catalog. The object-checkpoint compatibility path requires
trusted, independently pinned bytes. Task examples use the NLL law.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.experiments.p50_completion import validate_p50_completion_member
from compose_v4.model.contextual_ring_restate_rate_model import (
    ContextualRingRestateFactorizedTraceletRateModel,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
    LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
    LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
    LEGACY_EDITING_PROCESS_SEMANTICS,
    LEGACY_RING_RESTATE_SCORER_MODE,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedMarkEmpiricalPriors,
    FactorizedTraceletRateModel,
)
from compose_v4.model.ring_catalog_asset import load_ring_catalog_asset
from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint


@dataclass(frozen=True)
class LoadedReference:
    model: FactorizedTraceletRateModel
    checkpoint_sha256: str
    catalog_fingerprint: str
    max_active_atoms: int
    catalog_sha256: str | None = None


NLL_CHECKPOINT_SCHEMA = "compose.editing_v2.r_theta_corpus_checkpoint"
NLL_SELECTED_STATE_SHA256 = "c977ee3fe0cfdcafa204a2add27f3feef59f5dfcf4f3384659d8902a9006167c"


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Recompute the training pipeline's semantic tensor-state digest."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("selected_model_state must be a nonempty tensor dictionary")
    digest = hashlib.sha256(b"compose.current_state_dict.semantic.v1\0")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("selected_model_state has an invalid tensor entry")
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ValueError(f"selected_model_state tensor {name!r} is not dense data")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"selected_model_state tensor {name!r} contains nonfinite values")
        for block in (
            name.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"),
            value.reshape(-1).view(torch.uint8).numpy().tobytes(),
        ):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()


def _reconstruct_nll(path: Path, payload: dict, catalog_path: Path, catalog_fingerprint: str):
    if payload.get("schema") != NLL_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported NLL checkpoint schema: {path}")
    if payload.get("selected_step") != 12500:
        raise ValueError(f"NLL checkpoint selected step differs from the frozen result: {path}")
    state = payload.get("selected_model_state")
    if payload.get("selected_model_state_sha256") != NLL_SELECTED_STATE_SHA256:
        raise ValueError(f"NLL checkpoint state identity differs from the frozen result: {path}")
    if _state_dict_sha256(state) != NLL_SELECTED_STATE_SHA256:
        raise ValueError(f"NLL checkpoint selected tensor state failed its semantic hash: {path}")
    catalog = load_ring_catalog_asset(catalog_path, expected_fingerprint=catalog_fingerprint)
    model = ContextualRingRestateFactorizedTraceletRateModel(
        catalog,
        hidden_dim=256,
        message_passing_steps=6,
        mark_dim=32,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        cycle_open_scorer_mode="pair_linear",
        editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def _boolean(payload: dict, key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"checkpoint metadata {key!r} must be a literal Boolean")
    return value


def _reconstruct(path: Path, payload: dict) -> FactorizedTraceletRateModel:
    validate_p50_completion_member(path, payload)
    if "state_dict" not in payload and payload.get("checkpoint_kind") == "exact_training_recovery":
        if not {"best_state_dict", "best_metrics"} <= payload.keys():
            raise ValueError("recovery checkpoint lacks its selected-best inference state")
        payload = {**payload, "state_dict": payload["best_state_dict"]}
    required = {
        "state_dict",
        "ring_catalog",
        "tree_source_prior",
        "hidden_dim",
        "message_passing_steps",
        "training_backend",
        "source_prior",
    }
    if missing := sorted(required - payload.keys()):
        raise ValueError(f"checkpoint lacks inference metadata: {missing}")
    if (
        payload["training_backend"] != "factorized_marks"
        or payload["source_prior"] != "carbon_tree"
    ):
        raise ValueError("this loader requires the factorized_marks/carbon_tree checkpoint lineage")
    if payload.get("editing_process_semantics") not in (None, "", LEGACY_EDITING_PROCESS_SEMANTICS):
        raise ValueError("this loader cannot reconstruct a non-legacy editing process")
    # The original loader assumes these modes. Reject explicit contradictory
    # metadata rather than silently loading same-shaped tensors under another law.
    semantic_defaults = {
        "atom_restate_action_semantics": LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
        "cycle_close_action_semantics": LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
        "cycle_open_action_semantics": LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
        "atom_delete_action_semantics": LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
        "ring_restate_scorer_mode": LEGACY_RING_RESTATE_SCORER_MODE,
    }
    for key, expected in semantic_defaults.items():
        if key in payload and payload[key] != expected:
            raise ValueError(f"unsupported checkpoint {key}: {payload[key]!r}")
    prior = payload.get("empirical_mark_priors")
    mix = _boolean(payload, "corrupted_prior_mix", False)
    for key in ("enable_cyclic_graft", "enable_heteroatom_scan", "enable_ring_opening"):
        if key in payload and _boolean(payload, key, mix) != mix:
            raise ValueError(f"checkpoint {key} contradicts the legacy corrupted_prior_mix support")
    if any(
        torch.is_floating_point(value) and value.dtype != torch.float32
        for value in payload["state_dict"].values()
    ):
        raise ValueError(
            "checkpoint weights must be float32; implicit precision conversion is refused"
        )
    model = FactorizedTraceletRateModel(
        payload["ring_catalog"],
        hidden_dim=int(payload["hidden_dim"]),
        message_passing_steps=int(payload["message_passing_steps"]),
        ring_electronic_mode=str(payload.get("ring_electronic_mode", "factorized_local")),
        rate_factorization=str(payload.get("rate_factorization", "hierarchical")),
        property_condition_dim=int(payload.get("property_condition_dim", 0)),
        empirical_mark_prior_mode=str(payload.get("empirical_mark_prior_mode", "none")),
        empirical_mark_priors=None
        if prior is None
        else FactorizedMarkEmpiricalPriors.from_dict(prior),
        ring_family_mass_mode=str(payload.get("ring_family_mass_mode", "boolean")),
        ring_template_factorization=str(payload.get("ring_template_factorization", "flat")),
        atom_vocabulary=ORGANIC_VOCABULARY
        if _boolean(payload, "organic_vocabulary", False)
        else None,
        enable_ring_restates=_boolean(payload, "enable_ring_restates", mix),
        enable_cyclic_graft=mix,
        enable_heteroatom_scan=mix,
        enable_ring_opening=mix,
        enable_cycle_ops=_boolean(payload, "enable_cycle_ops", False),
        enable_ring_grow_macro=_boolean(payload, "enable_ring_grow_macro", True),
        enable_ring_system_delete=_boolean(payload, "enable_ring_system_delete", True),
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    missing, unexpected = set(incompatible.missing_keys), set(incompatible.unexpected_keys)
    role_keys = {key for key in model.state_dict() if key.startswith("ring_system_role_head.")}
    legacy_v0 = (
        not unexpected
        and missing == role_keys
        and "ring_electronic_mode" not in payload
        and int(getattr(payload["ring_catalog"], "ring_system_electronic_alias_version", 0)) == 0
    )
    if legacy_v0:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in role_keys:
                    parameter.zero_()
        model.virtualize_legacy_self_grafts = True
    elif missing or unexpected:
        raise ValueError(
            f"checkpoint/model state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if model.property_condition_dim != 0:
        raise ValueError("a goal-independent reference must not require property conditioning")
    model.eval().requires_grad_(False)
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("frozen reference inference requires CPU float32 parameters")
    return model


def load_frozen_reference(
    path: Path,
    *,
    expected_sha256: str,
    expected_catalog_fingerprint: str,
    catalog_path: Path | None = None,
    expected_catalog_sha256: str | None = None,
) -> LoadedReference:
    """Verify model and catalog bytes before reconstructing either process."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be an independently supplied lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{16}", expected_catalog_fingerprint):
        raise ValueError("expected_catalog_fingerprint must be the declared 16-character digest")
    path = Path(path)
    if (catalog_path is None) != (expected_catalog_sha256 is None):
        raise ValueError("catalog_path and expected_catalog_sha256 must be supplied together")
    if catalog_path is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_catalog_sha256):
            raise ValueError("expected_catalog_sha256 must be a lowercase SHA-256")
        if hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() != expected_catalog_sha256:
            raise ValueError(f"ring catalog asset SHA-256 mismatch: {catalog_path}")
    # Verify and deserialize the same open file. Recheck afterwards to detect
    # concurrent mutation; never return a model from a changed artifact.
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"checkpoint SHA-256 mismatch: {path}; expected {expected_sha256}")
        stream.seek(0)
        with torch.random.fork_rng(devices=[]):
            if catalog_path is None:
                payload = torch.load(stream, map_location="cpu", weights_only=False)
            else:
                payload = torch.load(stream, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError(f"checkpoint payload must be a dictionary: {path}")
            if catalog_path is None:
                maximum = payload.get("max_atoms")
                if type(maximum) is not int or maximum < 1:
                    raise ValueError("checkpoint must declare a positive integer max_atoms")
                model = _reconstruct(path, payload)
            else:
                maximum = 40
                model = _reconstruct_nll(
                    path, payload, Path(catalog_path), expected_catalog_fingerprint
                )
        stream.seek(0)
        after = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            after.update(chunk)
        if after.hexdigest() != expected_sha256:
            raise ValueError(f"checkpoint changed during loading: {path}")
    catalog = ring_catalog_fingerprint(model.ring_catalog)
    if catalog != expected_catalog_fingerprint:
        raise ValueError(
            f"reference catalog drift: {catalog}; expected {expected_catalog_fingerprint}"
        )
    return LoadedReference(model, expected_sha256, catalog, maximum, expected_catalog_sha256)
