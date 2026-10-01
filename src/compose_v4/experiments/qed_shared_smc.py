"""Finite-horizon QED particle control over the shared reference process."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.hphi_smc import (
    normalized_weights,
    should_resample,
    systematic_resample,
    terminal_output,
)
from compose_v4.experiments.qed_source_support import audit_qed_source


@dataclass(frozen=True)
class QEDSMCConfig:
    horizon: int = 24
    particles: int = 32
    candidates: int = 8
    qed_minimum: float = 0.9
    similarity_minimum: float = 0.4

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.horizon, self.particles, self.candidates)
        ):
            raise ValueError("QED SMC horizon, particles and candidates must be positive integers")
        if (
            not math.isfinite(self.qed_minimum)
            or not math.isfinite(self.similarity_minimum)
            or not 0.0 <= self.qed_minimum <= 1.0
            or not 0.0 <= self.similarity_minimum <= 1.0
        ):
            raise ValueError("QED SMC thresholds must be in [0, 1]")

    @property
    def region(self) -> tuple[float, float]:
        return self.qed_minimum, self.similarity_minimum


def candidate_seed(source_represented: str, candidate_index: int) -> int:
    payload = f"compose.qed.shared_smc.v1|{source_represented}|{candidate_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _canonical_state(state, slots: int):
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise ValueError("QED reference returned an unserializable state")
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def run_candidate(reference, bound_value, source_state, config: QEDSMCConfig, rng):
    """Return one candidate, or an explicit extinct slot, from one particle run."""
    states = [source_state] * config.particles
    stopped = [bound_value.in_target(source_state, config.region)] * config.particles
    if all(stopped):
        return {"status": "OK", "state": source_state, "steps": 0, "resamples": 0, "extinct": False}
    log_weights = np.zeros(config.particles, dtype=np.float64)
    resamples = 0
    steps = 0
    for step in range(config.horizon):
        budget = config.horizon - step
        steps = step + 1
        for index in range(config.particles):
            if stopped[index]:
                continue
            current = states[index]
            current_value = bound_value.value(current, budget, config.region)
            if current_value <= 0.0:
                log_weights[index] = -np.inf
                stopped[index] = True
                continue
            successor = reference.sample(current, rng)
            if successor is None:
                log_weights[index] = -np.inf
                stopped[index] = True
                continue
            successor = _canonical_state(successor, reference.config.persistent_slots)
            next_value = bound_value.value(successor, budget - 1, config.region)
            if next_value <= 0.0:
                log_weights[index] = -np.inf
                stopped[index] = True
                continue
            log_weights[index] += math.log(next_value) - math.log(current_value)
            states[index] = successor
            stopped[index] = bound_value.in_target(successor, config.region)

        if np.all(np.isneginf(log_weights)):
            break
        if should_resample(normalized_weights(log_weights)):
            indices = systematic_resample(normalized_weights(log_weights), rng)
            states = [states[index] for index in indices]
            stopped = [stopped[index] for index in indices]
            log_weights = np.zeros(config.particles, dtype=np.float64)
            resamples += 1
        if all(stopped):
            break

    selected, status = terminal_output(log_weights, rng)
    return {
        "status": status,
        "state": source_state if selected is None else states[selected],
        "steps": steps,
        "resamples": resamples,
        "extinct": selected is None,
    }


def run_source(reference, value_head, source: str, config: QEDSMCConfig) -> dict:
    """Return independent terminal candidates; never rank intermediate particles."""
    support = audit_qed_source(source, max_active_atoms=reference.reference.max_active_atoms)
    if not support.supported or not support.benchmark_equivalent or support.represented is None:
        raise ValueError("QED SMC source is outside benchmark-compatible support")
    if config.horizon > value_head.budget_max:
        raise ValueError("QED SMC horizon exceeds the fitted value-head budget")
    source_state = pad_molecular_graph(
        smiles_to_molecular_graph(support.represented), reference.config.persistent_slots
    )
    bound_value = value_head.for_source(source)
    candidates = []
    for candidate_index in range(config.candidates):
        seed = candidate_seed(support.represented, candidate_index)
        result = run_candidate(
            reference,
            bound_value,
            source_state,
            config,
            np.random.default_rng(seed),
        )
        quality, similarity = bound_value.properties(result["state"])
        candidates.append(
            {
                "index": candidate_index,
                "seed": seed,
                "status": result["status"],
                "smiles": source
                if result["extinct"]
                else molecular_graph_to_smiles(result["state"]),
                "qed": quality,
                "similarity_to_source": similarity,
                "success": not result["extinct"]
                and quality >= config.qed_minimum
                and similarity >= config.similarity_minimum,
                "steps": result["steps"],
                "resamples": result["resamples"],
            }
        )
    return {
        "schema_version": "compose.qed.shared_smc_source.v1",
        "source_original": source,
        "source_represented": support.represented,
        "reference": reference.identity(),
        "value_head_source_split_sha256": value_head.metadata["source_split_sha256"],
        "configuration": asdict(config),
        "success": any(candidate["success"] for candidate in candidates),
        "candidates": candidates,
    }
