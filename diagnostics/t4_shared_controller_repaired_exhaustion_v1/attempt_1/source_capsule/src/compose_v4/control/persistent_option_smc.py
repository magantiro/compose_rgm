"""Restartable SMC at completed-option boundaries with proposal correction.

For a reference option law ``B`` and an actual proposal ``q``, each completed
option contributes

``log B - log q + log h_next - log h_previous``.

This module owns no chemistry.  Callers persist the production exact-state
encoding, option audit and controller snapshot; resampling never reconstructs a
molecule from SMILES.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.hphi_smc import (
    effective_sample_size,
    normalized_weights,
    systematic_resample,
)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("persistent particle payload must be finite JSON") from error


@dataclass(frozen=True)
class PersistentOptionParticle:
    particle_id: str
    exact_state: dict
    history: tuple[dict, ...]
    log_weight: float | None
    log_potential: float | None
    alive: bool
    rng_identity: str
    controller_snapshot: str

    def __post_init__(self) -> None:
        if not self.particle_id or not self.rng_identity or not self.controller_snapshot:
            raise ValueError("particle, RNG and controller identities are required")
        _json_copy(self.exact_state)
        _json_copy(self.history)
        if self.alive:
            if (
                self.log_weight is None
                or self.log_potential is None
                or not np.isfinite([self.log_weight, self.log_potential]).all()
            ):
                raise ValueError("live particles require finite weights and potentials")
        elif self.log_weight is not None or self.log_potential is not None:
            raise ValueError("dead particles must have no serialized weight or potential")

    @property
    def exact_state_id(self) -> str:
        return identity(self.exact_state)

    def to_dict(self) -> dict:
        return _json_copy(
            {
                "particle_id": self.particle_id,
                "exact_state": self.exact_state,
                "exact_state_id": self.exact_state_id,
                "history": list(self.history),
                "log_weight": self.log_weight,
                "log_potential": self.log_potential,
                "alive": self.alive,
                "rng_identity": self.rng_identity,
                "controller_snapshot": self.controller_snapshot,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict) -> PersistentOptionParticle:
        data = _json_copy(payload)
        state = data.pop("exact_state")
        expected = data.pop("exact_state_id")
        if identity(state) != expected:
            raise ValueError("persistent exact-state identity mismatch")
        return cls(
            exact_state=state,
            history=tuple(data.pop("history")),
            **data,
        )


@dataclass(frozen=True)
class PersistentOptionPopulation:
    particles: tuple[PersistentOptionParticle, ...]
    option_boundary: int
    rng_state: dict
    controller_snapshot: str
    control_context: dict | None = None

    def __post_init__(self) -> None:
        if not self.particles or self.option_boundary < 0 or not self.controller_snapshot:
            raise ValueError("population requires particles, a boundary and a controller snapshot")
        if len({particle.particle_id for particle in self.particles}) != len(self.particles):
            raise ValueError("population particle identities must be unique")
        if any(
            particle.controller_snapshot != self.controller_snapshot for particle in self.particles
        ):
            raise ValueError("population mixes controller snapshots")
        _json_copy(self.rng_state)
        if self.control_context is not None:
            _json_copy(self.control_context)

    @classmethod
    def start(
        cls,
        exact_states: Sequence[dict],
        *,
        seed: int,
        controller_snapshot: str,
        log_potentials: Sequence[float] | None = None,
        control_context: dict | None = None,
    ):
        states = tuple(_json_copy(value) for value in exact_states)
        if (
            not states
            or isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("population start requires exact states and a nonnegative seed")
        n = len(states)
        potentials = (
            np.zeros(n, dtype=float)
            if log_potentials is None
            else np.asarray(log_potentials, dtype=float)
        )
        if potentials.shape != (n,) or not np.isfinite(potentials).all():
            raise ValueError("initial log potentials must be finite and state-aligned")
        particles = tuple(
            PersistentOptionParticle(
                particle_id=identity({"root": i, "state": state, "seed": int(seed)})[:24],
                exact_state=state,
                history=(),
                log_weight=-math.log(n),
                log_potential=float(potentials[i]),
                alive=True,
                rng_identity=identity({"stream": i, "seed": int(seed)})[:24],
                controller_snapshot=controller_snapshot,
            )
            for i, state in enumerate(states)
        )
        return cls(
            particles,
            0,
            _json_copy(np.random.default_rng(int(seed)).bit_generator.state),
            controller_snapshot,
            None if control_context is None else _json_copy(control_context),
        )

    def to_dict(self) -> dict:
        body = {
            "schema_version": "persistent_option_population_v2",
            "particles": [particle.to_dict() for particle in self.particles],
            "option_boundary": self.option_boundary,
            "rng_state": self.rng_state,
            "controller_snapshot": self.controller_snapshot,
            "control_context": self.control_context,
        }
        return {**_json_copy(body), "population_id": identity(body)}

    @classmethod
    def from_dict(cls, payload: dict) -> PersistentOptionPopulation:
        data = _json_copy(payload)
        schema = data.pop("schema_version", None)
        if schema not in (
            "persistent_option_population_v1",
            "persistent_option_population_v2",
        ):
            raise ValueError("unsupported persistent option-population schema")
        expected = data.pop("population_id", None)
        body = {"schema_version": schema, **data}
        if identity(body) != expected:
            raise ValueError("persistent option-population identity mismatch")
        particles = tuple(PersistentOptionParticle.from_dict(row) for row in data["particles"])
        return cls(
            particles,
            int(data["option_boundary"]),
            data["rng_state"],
            data["controller_snapshot"],
            data.get("control_context"),
        )


@dataclass(frozen=True)
class OptionTransition:
    source_particle_id: str
    next_exact_state: dict | None
    log_reference_probability: float
    log_proposal_probability: float
    next_log_potential: float | None
    alive: bool
    audit: dict
    operational: dict | None = None

    def __post_init__(self) -> None:
        if not self.source_particle_id:
            raise ValueError("option transition requires a source particle")
        if (
            not math.isfinite(self.log_reference_probability)
            or not math.isfinite(self.log_proposal_probability)
            or self.log_reference_probability > 1e-12
            or self.log_proposal_probability > 1e-12
        ):
            raise ValueError(
                "sampled options require finite log probabilities no greater than zero"
            )
        _json_copy(self.audit)
        if self.operational is not None:
            _json_copy(self.operational)
        if self.alive:
            if self.next_exact_state is None or self.next_log_potential is None:
                raise ValueError("live option transition requires a state and potential")
            _json_copy(self.next_exact_state)
            if not math.isfinite(self.next_log_potential):
                raise ValueError("live option transition potential must be finite")
        elif self.next_exact_state is not None or self.next_log_potential is not None:
            raise ValueError("dead option transitions have no next state or potential")


def advance_population(
    population: PersistentOptionPopulation,
    transitions: Sequence[OptionTransition],
    *,
    resample: bool = True,
) -> tuple[PersistentOptionPopulation, dict]:
    """Advance once, accumulate ratios, then optionally resample at ESS < N/2."""

    rows = tuple(transitions)
    n = len(population.particles)
    if len(rows) != n:
        raise ValueError("one option transition is required per particle slot")
    if [row.source_particle_id for row in rows] != [
        particle.particle_id for particle in population.particles
    ]:
        raise ValueError("option transitions are not aligned with the population")
    updated = np.full(n, -np.inf)
    increments: list[float | None] = []
    for i, (particle, row) in enumerate(zip(population.particles, rows, strict=True)):
        if not particle.alive:
            if row.alive:
                raise ValueError("a dead particle cannot produce a live transition")
            increments.append(None)
            continue
        if not row.alive:
            increments.append(None)
            continue
        increment = (
            row.log_reference_probability
            - row.log_proposal_probability
            + row.next_log_potential
            - particle.log_potential
        )
        updated[i] = particle.log_weight + increment
        increments.append(float(increment))

    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(population.rng_state)
    if np.isneginf(updated).all():
        particles = tuple(
            PersistentOptionParticle(
                particle.particle_id,
                particle.exact_state,
                particle.history + (_json_copy(row.audit),),
                None,
                None,
                False,
                particle.rng_identity,
                population.controller_snapshot,
            )
            for particle, row in zip(population.particles, rows, strict=True)
        )
        following = PersistentOptionPopulation(
            particles,
            population.option_boundary + 1,
            _json_copy(rng.bit_generator.state),
            population.controller_snapshot,
            population.control_context,
        )
        return following, {
            "status": "extinct",
            "weights": [0.0] * n,
            "ess": 0.0,
            "resampled": False,
            "indices": list(range(n)),
            "increments": increments,
        }

    weights = normalized_weights(updated)
    ess = effective_sample_size(weights)
    triggered = bool(resample and ess < n / 2)
    indices = systematic_resample(weights, rng) if triggered else np.arange(n)
    normalized = [None if weight == 0 else float(math.log(weight)) for weight in weights]
    particles = []
    for slot, index in enumerate(indices):
        parent, row = population.particles[int(index)], rows[int(index)]
        if not row.alive:
            if triggered:
                raise RuntimeError("a zero-weight option transition survived resampling")
            particles.append(
                PersistentOptionParticle(
                    parent.particle_id,
                    parent.exact_state,
                    parent.history + (_json_copy(row.audit),),
                    None,
                    None,
                    False,
                    parent.rng_identity,
                    population.controller_snapshot,
                )
            )
            continue
        particle_id = (
            identity(
                {
                    "parent": parent.particle_id,
                    "boundary": population.option_boundary + 1,
                    "slot": slot,
                }
            )[:24]
            if triggered
            else parent.particle_id
        )
        rng_identity = identity(
            {
                "parent_stream": parent.rng_identity,
                "boundary": population.option_boundary + 1,
                "slot": slot,
            }
        )[:24]
        particles.append(
            PersistentOptionParticle(
                particle_id,
                _json_copy(row.next_exact_state),
                parent.history + (_json_copy(row.audit),),
                -math.log(n) if triggered else normalized[int(index)],
                float(row.next_log_potential),
                True,
                rng_identity,
                population.controller_snapshot,
            )
        )
    following = PersistentOptionPopulation(
        tuple(particles),
        population.option_boundary + 1,
        _json_copy(rng.bit_generator.state),
        population.controller_snapshot,
        population.control_context,
    )
    return following, {
        "status": "live",
        "weights": weights.tolist(),
        "ess": float(ess),
        "resampled": triggered,
        "indices": indices.tolist(),
        "increments": increments,
    }
