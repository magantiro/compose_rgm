"""Bounded E5 verification of the canonical molecular-successor quotient.

This module is new experimental logic over the production process definition.  It
does not enumerate chemistry, decode model coordinates, canonicalize molecules,
or independently reconstruct successor probabilities.  Real molecular rows come
only from :func:`canonical_successor_result`, the authoritative segmented
pushforward over the production marked law.

Two intentionally separate layers are checked:

* a bounded panel of real molecular states exercises production enumeration,
  execution, canonical grouping, sampling, and persistent-slot relabeling;
* deterministic algebraic fixtures exercise within-fiber refinement, exact
  successor-level Doob control, and explicit negative mark-level power/top-k
  counterexamples.

The slow dictionary aggregator remains a TEST-ONLY oracle.  It is imported only
by tests, never here or by the result-producing CLI.

Scientific boundary
-------------------
Passing E5 licenses the quotient-level interpretation and successor-level control
of the derived molecular process.  It does not imply that the historical
selected-mark training objective was canonical-successor likelihood.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.production_successor_kernel import (
    SuccessorKernelResult,
    canonical_successor_result,
)
from compose_v4.experiments.registry import experiment, load_registry
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.model.segmented_successor import segmented_successor_logprobs
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

E5_CONTRACT_SCHEMA = "compose.experiments.e5_quotient_invariance_contract"
E5_CONTRACT_VERSION = 1
E5_RESULT_SCHEMA = "compose.experiments.e5_quotient_invariance_foundation"
E5_RESULT_VERSION = 1
E5_RESULT_STATUS = "FOUNDATION_VERIFICATION_COMPLETE_NO_PAPER_CLAIM"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/chem/source_prior.py",
    "src/compose_v4/chem/state.py",
    "src/compose_v4/experiments/registry.py",
    "src/compose_v4/experiments/quotient_invariance.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/experiments/successor_kernel.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/model/segmented_successor.py",
    "src/compose_v4/rewrite/compiler.py",
    "src/compose_v4/rewrite/factorized_fiber.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/trace.py",
    "src/compose_v4/rewrite/tree_transport.py",
    "src/compose_v4/rewrite/typed_ring_catalog.py",
)
_EXPECTED_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "experiment_id",
    "panel_id",
    "paper_claim_authorized",
    "checkpoint_required",
    "model_fixture",
    "real_state_panel",
    "slot_relabeling",
    "within_fiber_refinement",
    "sampling",
    "negative_mark_controls",
    "numeric_tolerances",
    "required_registry_checks",
    "scope_caveat",
    "contract_sha256",
}
_EXPECTED_RESULT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "experiment_id",
    "panel_id",
    "paper_claim_authorized",
    "checkpoint_used",
    "training_performed",
    "optimizer_steps",
    "scope_caveat",
    "provenance",
    "dictionary_oracle_policy",
    "synthetic_fixture",
    "real_state_rows",
    "checks",
    "summary",
    "artifact_sha256",
}


class QuotientInvarianceError(RuntimeError):
    """The E5 contract or a quotient invariant failed."""


@dataclass(frozen=True)
class EncodedMark:
    """One abstract mark assigned to an already-established successor fiber."""

    mark_id: str
    successor_key: str
    mass: float


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QuotientInvarianceError("E5 metadata is not finite canonical JSON") from error


def stable_sha256(value: object) -> str:
    """SHA-256 of canonical JSON."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def contract_self_hash(contract: Mapping[str, Any]) -> str:
    """Hash a contract excluding the field that stores the hash."""

    body = dict(contract)
    body.pop("contract_sha256", None)
    return stable_sha256(body)


def artifact_self_hash(artifact: Mapping[str, Any]) -> str:
    """Hash a result excluding the field that stores the hash."""

    body = dict(artifact)
    body.pop("artifact_sha256", None)
    return stable_sha256(body)


def implementation_sha256(repo_root: str | Path = _REPO_ROOT) -> str:
    """Bind evidence to every production and experimental source it executes."""

    root = Path(repo_root)
    digest = hashlib.sha256()
    for relative in sorted(_IMPLEMENTATION_SOURCES):
        path = root / relative
        if not path.is_file():
            raise QuotientInvarianceError(f"E5 implementation source is absent: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_e5_contract(
    path: str | Path,
    *,
    registry_path: str | Path,
) -> dict[str, Any]:
    """Load a self-hashed development contract and bind it to registry E5."""

    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise QuotientInvarianceError(f"cannot read E5 contract: {contract_path}") from error
    if not isinstance(payload, dict) or set(payload) != _EXPECTED_CONTRACT_FIELDS:
        raise QuotientInvarianceError("E5 contract has missing or unknown top-level fields")
    _validate_contract_binding(payload, registry_path=registry_path)
    return payload


def _validate_contract_binding(
    payload: Mapping[str, Any],
    *,
    registry_path: str | Path,
) -> None:
    if set(payload) != _EXPECTED_CONTRACT_FIELDS:
        raise QuotientInvarianceError("E5 contract has missing or unknown top-level fields")
    if (
        payload["schema"] != E5_CONTRACT_SCHEMA
        or payload["schema_version"] != E5_CONTRACT_VERSION
        or payload["experiment_id"] != "E5"
        or payload["panel_id"] != "QUOTIENT_INVARIANCE"
        or payload["paper_claim_authorized"] is not False
        or payload["checkpoint_required"] is not False
    ):
        raise QuotientInvarianceError("E5 contract identity or scientific status is invalid")
    claimed = payload["contract_sha256"]
    if not isinstance(claimed, str) or claimed != contract_self_hash(payload):
        raise QuotientInvarianceError("E5 contract self-hash does not match")

    registry = load_registry(registry_path)
    registered = experiment(registry, "E5")
    if registered["panel_ids"] != ["QUOTIENT_INVARIANCE"]:
        raise QuotientInvarianceError("registry E5 panel identity does not match the contract")
    if tuple(registered.get("checks", ())) != tuple(payload["required_registry_checks"]):
        raise QuotientInvarianceError("registry E5 checks drifted from the development contract")
    if "WITHIN a successor fiber" not in str(registered.get("scope_caveat", "")):
        raise QuotientInvarianceError("registry E5 no longer states the within-fiber scope caveat")

    _validate_contract_values(payload)


def _validate_contract_values(payload: Mapping[str, Any]) -> None:
    model = payload["model_fixture"]
    if (
        not isinstance(model, dict)
        or int(model["max_atoms"]) <= 0
        or not 0.0 <= float(model["time"]) <= 1.0
        or model["enable_cycle_ops"] is not True
        or model["enable_ring_grow_macro"] is not False
    ):
        raise QuotientInvarianceError("E5 model fixture is outside the RingCore contract")
    panel = payload["real_state_panel"]
    if not isinstance(panel, list) or not panel:
        raise QuotientInvarianceError("E5 real-state panel is empty")
    identifiers = [row.get("state_id") for row in panel if isinstance(row, dict)]
    if len(identifiers) != len(panel) or len(set(identifiers)) != len(identifiers):
        raise QuotientInvarianceError("E5 state identifiers are absent or duplicated")
    weights = tuple(float(value) for value in payload["within_fiber_refinement"]["split_weights"])
    if len(weights) < 2 or any(value <= 0.0 for value in weights):
        raise QuotientInvarianceError("within-fiber refinement requires at least two positive pieces")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise QuotientInvarianceError("within-fiber split weights must sum to one")
    sampling = payload["sampling"]
    if int(sampling["draws_per_state"]) <= 0:
        raise QuotientInvarianceError("sampling draws must be positive")
    alpha = float(sampling["familywise_error_probability"])
    if not 0.0 < alpha < 1.0:
        raise QuotientInvarianceError("sampling familywise error probability must lie in (0,1)")


def build_fixture_model(contract: Mapping[str, Any]) -> FactorizedTraceletRateModel:
    """Construct the deterministic untrained mechanism fixture named by the contract."""

    model_config = contract["model_fixture"]
    n_slots = int(model_config["max_atoms"])

    def trace(smiles: str):
        target = pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(1),
            n_slots=n_slots,
        )
        return compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )

    catalog = build_typed_ring_catalog(
        tuple(trace(smiles) for smiles in model_config["ring_catalog_seed_smiles"])
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(model_config["seed"]))
        model = FactorizedTraceletRateModel(
            catalog,
            hidden_dim=int(model_config["hidden_dim"]),
            message_passing_steps=int(model_config["message_passing_steps"]),
            mark_dim=int(model_config["mark_dim"]),
            enable_ring_restates=bool(model_config["enable_ring_restates"]),
            enable_cyclic_graft=bool(model_config["enable_cyclic_graft"]),
            enable_heteroatom_scan=bool(model_config["enable_heteroatom_scan"]),
            enable_ring_opening=bool(model_config["enable_ring_opening"]),
            enable_cycle_ops=bool(model_config["enable_cycle_ops"]),
            enable_ring_grow_macro=bool(model_config["enable_ring_grow_macro"]),
            enable_ring_system_delete=bool(model_config["enable_ring_system_delete"]),
            atom_vocabulary=ORGANIC_VOCABULARY,
        )
    return model.eval()


def permute_persistent_slots(
    state: MolecularGraph,
    permutation: Sequence[int],
) -> MolecularGraph:
    """Relabel every persistent coordinate by one explicit bijection."""

    order = np.asarray(tuple(int(value) for value in permutation), dtype=np.int64)
    n_slots = len(state.atom_types)
    if order.shape != (n_slots,) or set(order.tolist()) != set(range(n_slots)):
        raise QuotientInvarianceError("slot relabeling must be a bijection over every persistent slot")
    return MolecularGraph(
        atom_types=state.atom_types[order].copy(),
        formal_charges=state.formal_charges[order].copy(),
        implicit_h_counts=state.implicit_h_counts[order].copy(),
        bonds=state.bonds[np.ix_(order, order)].copy(),
    )


def probability_map(result: SuccessorKernelResult) -> dict[str, float]:
    """Canonical successor law supplied by the authoritative evaluator."""

    return {successor.key: float(successor.probability) for successor in result.batch.successors}


def max_probability_residual(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Maximum absolute coordinate residual, treating an absent key as zero."""

    return max(
        (abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in set(left) | set(right)),
        default=0.0,
    )


def quotient_mark_encoding(marks: Sequence[EncodedMark]) -> dict[str, float]:
    """Push an abstract mark encoding through the production segmented reduction."""

    if not marks:
        raise QuotientInvarianceError("a quotient fixture must contain at least one mark")
    if len({mark.mark_id for mark in marks}) != len(marks):
        raise QuotientInvarianceError("encoded mark identifiers must be unique")
    if any(not math.isfinite(mark.mass) or mark.mass <= 0.0 for mark in marks):
        raise QuotientInvarianceError("encoded mark masses must be finite and strictly positive")
    keys = tuple(sorted({mark.successor_key for mark in marks}))
    key_to_index = {key: index for index, key in enumerate(keys)}
    logits = torch.tensor(
        [math.log(mark.mass) for mark in marks],
        dtype=torch.float64,
    )
    log_probabilities = segmented_successor_logprobs(
        logits,
        torch.zeros(len(marks), dtype=torch.long),
        torch.tensor([key_to_index[mark.successor_key] for mark in marks], dtype=torch.long),
        n_examples=1,
        n_successors=len(keys),
    )
    return {
        key: float(log_probabilities[index].exp())
        for index, key in enumerate(keys)
    }


def refine_within_fibers(
    marks: Sequence[EncodedMark],
    split_weights: Sequence[float],
) -> tuple[EncodedMark, ...]:
    """Replace each mark by aliases in the same successor fiber, preserving its mass."""

    weights = tuple(float(value) for value in split_weights)
    if len(weights) < 2 or any(value <= 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise QuotientInvarianceError("refinement weights must be positive and sum to one")
    return tuple(
        EncodedMark(
            mark_id=f"{mark.mark_id}.refined.{index}",
            successor_key=mark.successor_key,
            mass=mark.mass * weight,
        )
        for mark in marks
        for index, weight in enumerate(weights)
    )


def successor_doob_control(
    base_law: Mapping[str, float],
    next_values: Mapping[str, float],
) -> dict[str, float]:
    """One exact finite-horizon Doob row over canonical successors."""

    if set(base_law) != set(next_values):
        raise QuotientInvarianceError("Doob values must be defined exactly on the base successor support")
    unnormalized = {
        key: float(probability) * float(next_values[key])
        for key, probability in base_law.items()
    }
    if any(not math.isfinite(value) or value < 0.0 for value in unnormalized.values()):
        raise QuotientInvarianceError("Doob weights must be finite and nonnegative")
    normalizer = sum(unnormalized.values())
    if normalizer <= 0.0:
        raise QuotientInvarianceError("Doob row is undefined because its backward value is zero")
    return {key: value / normalizer for key, value in sorted(unnormalized.items())}


def _positive_fixture_value(key: str) -> float:
    integer = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    fraction = integer / float((1 << 64) - 1)
    return 0.25 + 1.75 * fraction


def mark_power_then_quotient(
    marks: Sequence[EncodedMark],
    *,
    beta: float,
) -> dict[str, float]:
    """The intentionally non-invariant mark-level power control."""

    if not math.isfinite(beta) or beta <= 0.0:
        raise QuotientInvarianceError("mark power beta must be finite and positive")
    powered = tuple(
        EncodedMark(mark.mark_id, mark.successor_key, mark.mass**beta)
        for mark in marks
    )
    return quotient_mark_encoding(powered)


def mark_top_k_then_quotient(
    marks: Sequence[EncodedMark],
    *,
    k: int,
) -> dict[str, float]:
    """The intentionally non-invariant mark-level top-k control."""

    if not 0 < int(k) <= len(marks):
        raise QuotientInvarianceError("top-k must retain at least one and at most every mark")
    selected = tuple(
        sorted(marks, key=lambda mark: (-mark.mass, mark.mark_id))[: int(k)]
    )
    return quotient_mark_encoding(selected)


def sampling_tolerance(*, support_size: int, draws: int, alpha: float) -> float:
    """Simultaneous per-category Hoeffding threshold with a union bound."""

    if support_size <= 0 or draws <= 0 or not 0.0 < alpha < 1.0:
        raise QuotientInvarianceError("invalid sampling-tolerance inputs")
    return math.sqrt(math.log((2.0 * support_size) / alpha) / (2.0 * draws))


def _synthetic_fixture(contract: Mapping[str, Any]) -> dict[str, Any]:
    base = (
        EncodedMark("y.0", "successor_y", 0.5),
        EncodedMark("z.0", "successor_z", 0.5),
    )
    refined = (
        EncodedMark("y.0", "successor_y", 0.125),
        EncodedMark("y.1", "successor_y", 0.125),
        EncodedMark("y.2", "successor_y", 0.125),
        EncodedMark("y.3", "successor_y", 0.125),
        EncodedMark("z.0", "successor_z", 0.5),
    )
    base_law = quotient_mark_encoding(base)
    refined_law = quotient_mark_encoding(refined)
    values = {"successor_y": 2.0, "successor_z": 1.0}
    base_control = successor_doob_control(base_law, values)
    refined_control = successor_doob_control(refined_law, values)
    beta = float(contract["negative_mark_controls"]["power_beta"])
    top_k = int(contract["negative_mark_controls"]["top_k"])
    base_power = mark_power_then_quotient(base, beta=beta)
    refined_power = mark_power_then_quotient(refined, beta=beta)
    base_top_k = mark_top_k_then_quotient(base, k=top_k)
    refined_top_k = mark_top_k_then_quotient(refined, k=top_k)
    return {
        "base_raw_mark_count": len(base),
        "refined_raw_mark_count": len(refined),
        "base_successor_law": base_law,
        "refined_successor_law": refined_law,
        "successor_mass_invariance_residual": max_probability_residual(base_law, refined_law),
        "base_controlled_law": base_control,
        "refined_controlled_law": refined_control,
        "successor_control_invariance_residual": max_probability_residual(
            base_control,
            refined_control,
        ),
        "mark_power": {
            "beta": beta,
            "base_law": base_power,
            "refined_law": refined_power,
            "divergence": max_probability_residual(base_power, refined_power),
        },
        "mark_top_k": {
            "k": top_k,
            "base_law": base_top_k,
            "refined_law": refined_top_k,
            "divergence": max_probability_residual(base_top_k, refined_top_k),
        },
    }


def _real_state_row(
    *,
    model: FactorizedTraceletRateModel,
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    row_index: int,
) -> dict[str, Any]:
    model_config = contract["model_fixture"]
    state = pad_molecular_graph(
        smiles_to_molecular_graph(str(row["smiles"])),
        int(model_config["max_atoms"]),
    )
    result = canonical_successor_result(model, state, float(model_config["time"]))
    law = probability_map(result)
    if not law:
        raise QuotientInvarianceError(f"E5 panel state {row['state_id']} has no productive successors")

    permutation_rng = np.random.default_rng(
        int(contract["slot_relabeling"]["seed"]) + row_index
    )
    permutation = permutation_rng.permutation(len(state.atom_types))
    if np.array_equal(permutation, np.arange(len(permutation))):
        permutation = np.roll(permutation, 1)
    relabeled = permute_persistent_slots(state, permutation)
    if canonical_state_key(relabeled) != canonical_state_key(state):
        raise QuotientInvarianceError("a persistent-slot relabeling changed molecular identity")
    relabeled_result = canonical_successor_result(
        model,
        relabeled,
        float(model_config["time"]),
    )
    relabeled_law = probability_map(relabeled_result)

    coarse = tuple(
        EncodedMark(f"{key}.coarse", key, mass)
        for key, mass in sorted(law.items())
    )
    refined = refine_within_fibers(
        coarse,
        contract["within_fiber_refinement"]["split_weights"],
    )
    coarse_law = quotient_mark_encoding(coarse)
    refined_law = quotient_mark_encoding(refined)
    values = {key: _positive_fixture_value(key) for key in law}
    coarse_control = successor_doob_control(coarse_law, values)
    refined_control = successor_doob_control(refined_law, values)

    sampling = contract["sampling"]
    draws = int(sampling["draws_per_state"])
    keys = tuple(sorted(law))
    probabilities = np.asarray([law[key] for key in keys], dtype=np.float64)
    probabilities /= probabilities.sum()
    sampling_rng = np.random.default_rng(int(sampling["seed"]) + row_index)
    counts = sampling_rng.multinomial(draws, probabilities)
    frequencies = counts.astype(np.float64) / draws
    sample_residual = float(np.max(np.abs(frequencies - probabilities)))
    sample_tolerance = sampling_tolerance(
        support_size=len(keys),
        draws=draws,
        alpha=float(sampling["familywise_error_probability"]),
    )

    return {
        "state_id": str(row["state_id"]),
        "smiles": str(row["smiles"]),
        "source_key": result.batch.source_key,
        "raw_mark_count": result.diagnostics.raw_mark_count,
        "productive_mark_count": result.diagnostics.productive_mark_count,
        "canonical_successor_count": result.diagnostics.canonical_successor_count,
        "support_compression": (
            result.diagnostics.productive_mark_count
            - result.diagnostics.canonical_successor_count
        ),
        "aliased_successor_count": sum(
            multiplicity > 1
            for multiplicity in result.diagnostics.alias_multiplicities
        ),
        "maximum_alias_multiplicity": max(
            result.diagnostics.alias_multiplicities,
            default=0,
        ),
        "raw_productive_mass": result.diagnostics.raw_productive_mass,
        "virtual_self_mass": result.diagnostics.virtual_self_mass,
        "aggregate_successor_mass": sum(law.values()),
        "slot_permutation": [int(value) for value in permutation],
        "slot_relabeling_support_identical": set(law) == set(relabeled_law),
        "slot_relabeling_mass_residual": max_probability_residual(law, relabeled_law),
        "coarse_raw_mark_count": len(coarse),
        "refined_raw_mark_count": len(refined),
        "within_fiber_refinement_mass_residual": max_probability_residual(
            coarse_law,
            refined_law,
        ),
        "successor_control_invariance_residual": max_probability_residual(
            coarse_control,
            refined_control,
        ),
        "sampling": {
            "draws": draws,
            "seed": int(sampling["seed"]) + row_index,
            "support_size": len(keys),
            "maximum_frequency_residual": sample_residual,
            "simultaneous_tolerance": sample_tolerance,
            "passes": sample_residual <= sample_tolerance,
        },
    }


def run_e5_foundation(
    contract: Mapping[str, Any],
    *,
    registry_path: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> dict[str, Any]:
    """Run bounded E5 mechanism checks and return a strict self-hashed artifact."""

    _validate_contract_binding(contract, registry_path=registry_path)
    model = build_fixture_model(contract)
    rows = [
        _real_state_row(
            model=model,
            row=row,
            contract=contract,
            row_index=index,
        )
        for index, row in enumerate(contract["real_state_panel"])
    ]
    synthetic = _synthetic_fixture(contract)
    tolerances = contract["numeric_tolerances"]
    checks = {
        "raw_mark_multiplicity_differs": (
            synthetic["base_raw_mark_count"] != synthetic["refined_raw_mark_count"]
            and all(row["coarse_raw_mark_count"] != row["refined_raw_mark_count"] for row in rows)
        ),
        "aggregate_canonical_successor_mass_identical": (
            synthetic["successor_mass_invariance_residual"]
            <= float(tolerances["successor_mass_invariance"])
            and all(
                abs(row["aggregate_successor_mass"] - 1.0)
                <= float(tolerances["successor_mass_invariance"])
                for row in rows
            )
        ),
        "sampled_successor_frequency_matches_aggregate_mass": all(
            row["sampling"]["passes"] for row in rows
        ),
        "persistent_slot_relabeling_leaves_kernel_unchanged": all(
            row["slot_relabeling_support_identical"]
            and row["slot_relabeling_mass_residual"] <= float(tolerances["slot_relabeling"])
            for row in rows
        ),
        "within_fiber_action_refinement_leaves_state_law_unchanged": all(
            row["within_fiber_refinement_mass_residual"]
            <= float(tolerances["successor_mass_invariance"])
            for row in rows
        ),
        "exact_control_on_quotient_invariant_to_refinement": (
            synthetic["successor_control_invariance_residual"]
            <= float(tolerances["controlled_law_invariance"])
            and all(
                row["successor_control_invariance_residual"]
                <= float(tolerances["controlled_law_invariance"])
                for row in rows
            )
        ),
        "mark_level_power_divergence_demonstrated": synthetic["mark_power"]["divergence"] > 0.0,
        "mark_level_top_k_divergence_demonstrated": synthetic["mark_top_k"]["divergence"] > 0.0,
        "real_panel_contains_aliasing": any(row["aliased_successor_count"] > 0 for row in rows),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise QuotientInvarianceError(f"E5 foundation checks failed: {failed}")

    root = Path(repo_root)
    registry = Path(registry_path)
    body: dict[str, Any] = {
        "schema": E5_RESULT_SCHEMA,
        "schema_version": E5_RESULT_VERSION,
        "status": E5_RESULT_STATUS,
        "experiment_id": "E5",
        "panel_id": "QUOTIENT_INVARIANCE",
        "paper_claim_authorized": False,
        "checkpoint_used": None,
        "training_performed": False,
        "optimizer_steps": 0,
        "scope_caveat": contract["scope_caveat"],
        "provenance": {
            "contract_sha256": contract["contract_sha256"],
            "registry_path": str(registry),
            "registry_file_sha256": file_sha256(registry),
            "implementation_sha256": implementation_sha256(root),
            "implementation_sources": list(_IMPLEMENTATION_SOURCES),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "dictionary_oracle_policy": {
            "role": "test_only",
            "used_to_produce_reported_metrics": False,
            "verification_required_before_review": True,
        },
        "synthetic_fixture": synthetic,
        "real_state_rows": rows,
        "checks": checks,
        "summary": {
            "state_count": len(rows),
            "total_raw_marks": sum(row["raw_mark_count"] for row in rows),
            "total_productive_marks": sum(row["productive_mark_count"] for row in rows),
            "total_canonical_successors": sum(row["canonical_successor_count"] for row in rows),
            "maximum_alias_multiplicity": max(
                row["maximum_alias_multiplicity"] for row in rows
            ),
            "maximum_slot_relabeling_residual": max(
                row["slot_relabeling_mass_residual"] for row in rows
            ),
            "maximum_refinement_residual": max(
                row["within_fiber_refinement_mass_residual"] for row in rows
            ),
            "maximum_control_residual": max(
                row["successor_control_invariance_residual"] for row in rows
            ),
            "maximum_sampling_residual": max(
                row["sampling"]["maximum_frequency_residual"] for row in rows
            ),
            "mark_level_power_divergence": synthetic["mark_power"]["divergence"],
            "mark_level_top_k_divergence": synthetic["mark_top_k"]["divergence"],
        },
    }
    artifact = {**body, "artifact_sha256": stable_sha256(body)}
    if artifact_self_hash(artifact) != artifact["artifact_sha256"]:
        raise QuotientInvarianceError("internal E5 artifact self-hash failure")
    return artifact


def validate_e5_artifact(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    registry_path: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> None:
    """Validate identity, self-hash, provenance, and non-claiming scope."""

    _validate_contract_binding(contract, registry_path=registry_path)
    if set(artifact) != _EXPECTED_RESULT_FIELDS:
        raise QuotientInvarianceError("E5 artifact has missing or unknown top-level fields")
    if (
        artifact["schema"] != E5_RESULT_SCHEMA
        or artifact["schema_version"] != E5_RESULT_VERSION
        or artifact["status"] != E5_RESULT_STATUS
        or artifact["experiment_id"] != "E5"
        or artifact["panel_id"] != "QUOTIENT_INVARIANCE"
        or artifact["paper_claim_authorized"] is not False
        or artifact["checkpoint_used"] is not None
        or artifact["training_performed"] is not False
        or artifact["optimizer_steps"] != 0
    ):
        raise QuotientInvarianceError("E5 artifact identity or scientific status is invalid")
    if artifact_self_hash(artifact) != artifact["artifact_sha256"]:
        raise QuotientInvarianceError("E5 artifact self-hash does not match")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise QuotientInvarianceError("E5 artifact provenance is absent")
    registry = Path(registry_path)
    expected_provenance = {
        "contract_sha256": contract["contract_sha256"],
        "registry_path": str(registry),
        "registry_file_sha256": file_sha256(registry),
        "implementation_sha256": implementation_sha256(repo_root),
        "implementation_sources": list(_IMPLEMENTATION_SOURCES),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "rdkit_version": rdBase.rdkitVersion,
    }
    if provenance != expected_provenance:
        raise QuotientInvarianceError("E5 artifact provenance does not match the current contract/runtime")
    oracle_policy = artifact.get("dictionary_oracle_policy")
    if oracle_policy != {
        "role": "test_only",
        "used_to_produce_reported_metrics": False,
        "verification_required_before_review": True,
    }:
        raise QuotientInvarianceError("E5 artifact weakens the dictionary-oracle boundary")
    checks = artifact.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise QuotientInvarianceError("E5 artifact contains an absent or failed mechanism check")


def freeze_e5_artifact(
    artifact: Mapping[str, Any],
    output_path: str | Path,
    *,
    contract: Mapping[str, Any],
    registry_path: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> None:
    """Write exact evidence bytes once; an existing different artifact is an error."""

    validate_e5_artifact(
        artifact,
        contract=contract,
        registry_path=registry_path,
        repo_root=repo_root,
    )
    path = Path(output_path)
    encoded = _canonical_json_bytes(artifact) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise QuotientInvarianceError(f"immutable E5 artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = path.open("xb")
    except FileExistsError as error:
        raise QuotientInvarianceError(f"immutable E5 artifact collision: {path}") from error
    with descriptor:
        descriptor.write(encoded)


__all__ = [
    "E5_CONTRACT_SCHEMA",
    "E5_CONTRACT_VERSION",
    "E5_RESULT_SCHEMA",
    "E5_RESULT_STATUS",
    "E5_RESULT_VERSION",
    "EncodedMark",
    "QuotientInvarianceError",
    "artifact_self_hash",
    "build_fixture_model",
    "contract_self_hash",
    "file_sha256",
    "freeze_e5_artifact",
    "implementation_sha256",
    "load_e5_contract",
    "mark_power_then_quotient",
    "mark_top_k_then_quotient",
    "max_probability_residual",
    "permute_persistent_slots",
    "probability_map",
    "quotient_mark_encoding",
    "refine_within_fibers",
    "run_e5_foundation",
    "sampling_tolerance",
    "stable_sha256",
    "successor_doob_control",
    "validate_e5_artifact",
]
