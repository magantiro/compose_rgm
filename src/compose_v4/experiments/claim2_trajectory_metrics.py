"""Claim 2: the frozen mobility--fidelity metric suite.

Everything here is a pure function of a realized trajectory -- a list of
canonical molecular keys (which are canonical SMILES) plus the operator
families committed at each step.  No model, no checkpoint, no Modal runtime.
That is deliberate: the expensive job emits trajectories, and every number in
the paper is recomputed locally from committed shards.  A metric bug therefore
costs a local rerun, not a container-hour.

THE FRONTIER IS THE RESULT, NOT A PILE OF METRICS
-------------------------------------------------
No single number is allowed to define "good trajectories".  The claim is a
two-dimensional Pareto statement with a pre-declared decision rule:

    mobility  M = median endpoint ECFP4 Tanimoto DISTANCE from the source
    fidelity  F = fraction of committed intermediate states inside the
                  descriptor envelope calibrated on held-in molecules BEFORE
                  any trajectory was looked at

``frontier_verdict`` returns exactly one of ``dominates`` / ``dominated_by`` /
``incomparable``.  All three are real outcomes.  ``incomparable`` -- R_theta
buys mobility by spending fidelity, or the reverse -- is a publishable finding
and must not be resolved by inventing a scalarization after seeing the numbers.

THE FALSIFIABILITY GATE
-----------------------
This project has been bitten three times by statistics whose sign was fixed
before any data existed: an action selected by argmax V_G and then scored by
V_G; a "verified" arm that was secretly greedy; a sign test against a null that
policy improvement makes false.  Every metric declared in ``METRIC_REGISTRY``
must therefore answer one question:

    what value could this take if the hypothesis were false?

``validate_metric_registry`` refuses any metric whose ``falsifying_observation``
is empty.  ``BANNED_STATISTICS`` names the specific comparisons that are
tautological on this instrument and may never be reported as evidence, with the
reason each one is unfalsifiable.  A metric is not a measurement unless a
different world could have produced a different number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, log, sqrt
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

# ---- the frozen descriptor set --------------------------------------------

#: The nine descriptors named by the workstream contract, in frozen order.
#: Simple and reproducible on purpose: this is a distributional sanity check,
#: not a claim about medicinal plausibility.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "molecular_weight",
    "clogp",
    "qed",
    "formal_charge",
    "hbd",
    "hba",
    "ring_count",
    "fraction_csp3",
    "heavy_atoms",
)

#: Coordinate-wise envelope quantiles. Frozen before calibration so the
#: envelope cannot be widened until the trajectories fit inside it.
ENVELOPE_LOWER_QUANTILE = 0.005
ENVELOPE_UPPER_QUANTILE = 0.995

#: ECFP4 == Morgan radius 2. Bit length fixed so similarity is reproducible.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


class MetricError(ValueError):
    """A trajectory metric was asked for something it cannot honestly compute."""


def descriptor_vector(smiles: str) -> tuple[float, ...] | None:
    """The frozen nine-descriptor vector, or ``None`` for an unparseable key.

    Returning ``None`` rather than raising matters: an unparseable committed
    state would itself be a finding about the executor, and it must be counted
    rather than crash the analysis of every other trajectory.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        quantitative_estimate = float(QED.qed(molecule))
    except Exception:  # noqa: BLE001 - QED fails on exotic valences; not fatal
        return None
    try:
        values = (
            float(Descriptors.MolWt(molecule)),
            float(Crippen.MolLogP(molecule)),
            quantitative_estimate,
            float(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
            float(Lipinski.NumHDonors(molecule)),
            float(Lipinski.NumHAcceptors(molecule)),
            float(rdMolDescriptors.CalcNumRings(molecule)),
            float(rdMolDescriptors.CalcFractionCSP3(molecule)),
            float(molecule.GetNumHeavyAtoms()),
        )
    except Exception:  # noqa: BLE001
        return None
    return values if all(isfinite(value) for value in values) else None


@dataclass(frozen=True)
class DescriptorEnvelope:
    """A coordinate-wise box plus standardization, calibrated on held-in molecules.

    Calibrated BEFORE any trajectory is generated.  The status field is carried
    so an analysis cannot silently consume an envelope that was refit after
    seeing results.
    """

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    mean: tuple[float, ...]
    stdev: tuple[float, ...]
    calibration_source: str
    calibration_count: int
    status: str

    def __post_init__(self) -> None:
        width = len(self.names)
        for field_name in ("lower", "upper", "mean", "stdev"):
            if len(getattr(self, field_name)) != width:
                raise MetricError(f"envelope {field_name} is not aligned with names")
        for index, (low, high) in enumerate(zip(self.lower, self.upper)):
            if not low <= high:
                raise MetricError(f"envelope interval for {self.names[index]} is inverted")
        for index, deviation in enumerate(self.stdev):
            if deviation <= 0.0:
                raise MetricError(
                    f"envelope stdev for {self.names[index]} is {deviation}; "
                    "a degenerate coordinate cannot standardize drift"
                )

    def contains(self, vector: Sequence[float]) -> bool:
        """All nine coordinates inside their calibrated interval."""
        return all(
            low <= value <= high
            for value, low, high in zip(vector, self.lower, self.upper)
        )

    def coordinate_membership(self, vector: Sequence[float]) -> dict[str, bool]:
        return {
            name: bool(low <= value <= high)
            for name, value, low, high in zip(self.names, vector, self.lower, self.upper)
        }

    def standardize(self, vector: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            (value - centre) / deviation
            for value, centre, deviation in zip(vector, self.mean, self.stdev)
        )

    def drift(self, source: Sequence[float], state: Sequence[float]) -> float:
        """Standardized L2 descriptor displacement from source to state."""
        left = self.standardize(source)
        right = self.standardize(state)
        return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def to_json(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "mean": list(self.mean),
            "stdev": list(self.stdev),
            "calibration_source": self.calibration_source,
            "calibration_count": self.calibration_count,
            "status": self.status,
            "lower_quantile": ENVELOPE_LOWER_QUANTILE,
            "upper_quantile": ENVELOPE_UPPER_QUANTILE,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "DescriptorEnvelope":
        return cls(
            names=tuple(payload["names"]),
            lower=tuple(float(v) for v in payload["lower"]),
            upper=tuple(float(v) for v in payload["upper"]),
            mean=tuple(float(v) for v in payload["mean"]),
            stdev=tuple(float(v) for v in payload["stdev"]),
            calibration_source=str(payload["calibration_source"]),
            calibration_count=int(payload["calibration_count"]),
            status=str(payload["status"]),
        )


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank quantile. Deterministic and dependency-free."""
    if not sorted_values:
        raise MetricError("cannot take a quantile of an empty sample")
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def calibrate_envelope(
    vectors: Sequence[Sequence[float]],
    *,
    calibration_source: str,
    status: str,
) -> DescriptorEnvelope:
    """Fit the coordinate-wise envelope on a held-in descriptor sample."""
    if len(vectors) < 100:
        raise MetricError(
            f"envelope calibration needs a real sample; received {len(vectors)} vectors"
        )
    columns = list(zip(*vectors))
    lower, upper, mean, stdev = [], [], [], []
    for index, column in enumerate(columns):
        ordered = sorted(column)
        lower.append(_quantile(ordered, ENVELOPE_LOWER_QUANTILE))
        upper.append(_quantile(ordered, ENVELOPE_UPPER_QUANTILE))
        centre = sum(column) / len(column)
        variance = sum((value - centre) ** 2 for value in column) / max(len(column) - 1, 1)
        mean.append(centre)
        # A constant coordinate (formal charge is charge-preserving by process
        # scope) would divide drift by zero. Floor it rather than drop it, so
        # the coordinate still participates in the box test.
        stdev.append(sqrt(variance) if variance > 0.0 else 1.0)
    return DescriptorEnvelope(
        names=DESCRIPTOR_NAMES,
        lower=tuple(lower),
        upper=tuple(upper),
        mean=tuple(mean),
        stdev=tuple(stdev),
        calibration_source=calibration_source,
        calibration_count=len(vectors),
        status=status,
    )


# ---- structural mobility --------------------------------------------------


def morgan_fingerprint(smiles: str):
    """ECFP4 bit vector, or ``None`` when the key does not parse."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    return generator.GetFingerprint(molecule)


def tanimoto_distance(left: str, right: str) -> float | None:
    """1 - ECFP4 Tanimoto. ``None`` when either key does not parse."""
    from rdkit import DataStructs

    left_fingerprint = morgan_fingerprint(left)
    right_fingerprint = morgan_fingerprint(right)
    if left_fingerprint is None or right_fingerprint is None:
        return None
    return 1.0 - float(DataStructs.TanimotoSimilarity(left_fingerprint, right_fingerprint))


def ring_system_count(smiles: str) -> int | None:
    """Number of fused ring SYSTEMS -- rings sharing >=2 atoms are one system.

    Ring count alone cannot distinguish "grew a second isolated ring" from
    "fused a ring onto an existing one", and those are different kinds of
    structural movement.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    rings = [frozenset(ring) for ring in Chem.GetSymmSSSR(molecule)]
    if not rings:
        return 0
    parent = list(range(len(rings)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2:
                parent[find(i)] = find(j)
    return len({find(index) for index in range(len(rings))})


def heavy_atom_count(smiles: str) -> int | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else int(molecule.GetNumHeavyAtoms())


# ---- trajectory health ----------------------------------------------------


@dataclass(frozen=True)
class TrajectoryHealth:
    """Cycling and degeneracy statistics for one realized trajectory.

    Reported, not judged.  Some reversible behaviour may be part of the
    reference law -- an editor that can undo is not thereby pathological.  The
    claim is about the ABSENCE OF PATHOLOGICAL DOMINATION, so the comparison
    across arms carries the argument, not the raw level of any one rate.
    """

    committed_edits: int
    distinct_states: int
    unique_state_fraction: float
    immediate_reversal_count: int
    immediate_reversal_rate: float
    two_cycle_count: int
    two_cycle_rate: float
    revisit_count: int
    revisit_rate: float
    reached_horizon: bool
    dead_end: bool

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def trajectory_health(states: Sequence[str], horizon: int) -> TrajectoryHealth:
    """Cycling statistics of a committed state sequence ``[x0, x1, ..., xT]``.

    Definitions, fixed here so no two analyses disagree:

    * *immediate reversal* -- ``x_{t+1} == x_{t-1}``: the step undid the one
      before it.  The base kernel is state-only and does not forbid this.
    * *two-cycle* -- an ``a -> b -> a`` pattern anywhere; counted as the number
      of immediate reversals, which is the same event named from the other end.
      Kept as a separate field because the workstream contract names both and a
      future history-aware wrapper would separate them.
    * *revisit* -- a committed state equal to ANY earlier committed state,
      whether or not it was the immediately preceding one.
    * *dead end* -- the trajectory stopped before the horizon because the state
      had no productive canonical successor.  This is a property of the
      executable support, so it is shared across arms at a given state and is
      informative only through which states each law reaches.
    """
    if not states:
        raise MetricError("a trajectory must contain at least its source state")
    committed = len(states) - 1
    distinct = len(set(states))
    reversals = sum(
        1 for index in range(1, committed) if states[index + 1] == states[index - 1]
    )
    seen: set[str] = set()
    revisits = 0
    for state in states:
        if state in seen:
            revisits += 1
        seen.add(state)
    return TrajectoryHealth(
        committed_edits=committed,
        distinct_states=distinct,
        unique_state_fraction=distinct / len(states),
        immediate_reversal_count=reversals,
        immediate_reversal_rate=reversals / max(committed - 1, 1) if committed > 1 else 0.0,
        two_cycle_count=reversals,
        two_cycle_rate=reversals / max(committed - 1, 1) if committed > 1 else 0.0,
        revisit_count=revisits,
        revisit_rate=revisits / max(committed, 1),
        reached_horizon=committed >= horizon,
        dead_end=committed < horizon,
    )


def family_entropy(family_sets: Sequence[Sequence[str]]) -> float:
    """Shannon entropy (nats) of committed operator families.

    A successor reachable through several families has genuinely ambiguous
    attribution, so each committed step contributes ``1/|families|`` to every
    family that reaches it.  This rule is fixed BEFORE any result; the raw
    family sets are kept in every shard so an alternative attribution can be
    computed after the fact without a rerun.

    Direction is NOT preregistered.  A learned law concentrating below uniform
    is expected and is not by itself evidence of anything; the falsifiable
    question is whether families are ABANDONED, which ``family_coverage``
    answers.
    """
    weights: Counter[str] = Counter()
    for names in family_sets:
        if not names:
            continue
        share = 1.0 / len(set(names))
        for name in set(names):
            weights[name] += share
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for weight in weights.values():
        fraction = weight / total
        if fraction > 0.0:
            entropy -= fraction * log(fraction)
    return entropy


def family_coverage(family_sets: Sequence[Sequence[str]]) -> set[str]:
    """The set of operator families a trajectory actually committed."""
    covered: set[str] = set()
    for names in family_sets:
        covered.update(names)
    return covered


def multi_family_trajectory(family_sets: Sequence[Sequence[str]]) -> bool:
    """Whether the trajectory used at least two distinct operator families."""
    return len(family_coverage(family_sets)) >= 2


# ---- the mobility-fidelity frontier ---------------------------------------


@dataclass(frozen=True)
class FrontierPoint:
    """One arm's position on the frontier, with its denominators."""

    arm: str
    mobility: float
    fidelity: float
    source_count: int
    trajectory_count: int
    state_count: int

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


DOMINATES = "dominates"
DOMINATED_BY = "dominated_by"
INCOMPARABLE = "incomparable"
UNRESOLVED = "unresolved"


def frontier_verdict(
    reference: FrontierPoint,
    other: FrontierPoint,
    *,
    mobility_resolved: bool,
    fidelity_resolved: bool,
) -> str:
    """Pareto relation of ``reference`` to ``other`` under a pre-declared rule.

    ``mobility_resolved`` / ``fidelity_resolved`` come from the paired bootstrap
    over SOURCES -- the independent statistical unit.  An axis whose confidence
    interval covers zero contributes NO direction, which is how a small panel
    is prevented from manufacturing a verdict out of noise.

    Four outcomes are reachable, and every one of them is a real result:

    * ``dominates``     -- better or equal on both axes, strictly better and
                           resolved on at least one;
    * ``dominated_by``  -- the same statement with the arms exchanged;
    * ``incomparable``  -- resolved movement in opposite directions on the two
                           axes: mobility was bought with fidelity, or fidelity
                           with mobility.  This is a finding, not a failure, and
                           must not be scalarized away after the fact;
    * ``unresolved``    -- neither axis separated.  Report the panel as
                           underpowered; do not read a direction off the point
                           estimates.
    """
    mobility_direction = 0
    if mobility_resolved:
        mobility_direction = 1 if reference.mobility > other.mobility else -1
    fidelity_direction = 0
    if fidelity_resolved:
        fidelity_direction = 1 if reference.fidelity > other.fidelity else -1

    if mobility_direction == 0 and fidelity_direction == 0:
        return UNRESOLVED
    if mobility_direction >= 0 and fidelity_direction >= 0:
        return DOMINATES
    if mobility_direction <= 0 and fidelity_direction <= 0:
        return DOMINATED_BY
    return INCOMPARABLE


# ---- the falsifiability gate ----------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One declared metric and the observation that would falsify the claim."""

    name: str
    definition: str
    unit: str
    #: What the metric would look like in a world where the hypothesis is FALSE.
    #: Empty means the metric is not a measurement and the registry refuses it.
    falsifying_observation: str
    preregistered_direction: str = "none"


#: Comparisons that are tautological on this instrument. Reporting any of these
#: as evidence for Claim 2 is an instrument defect, not a weak result.
BANNED_STATISTICS: dict[str, str] = {
    "reference_log_likelihood_of_own_trajectories": (
        "E_R[log R] >= E_U[log R] is Gibbs' inequality. Sampling from R_theta and "
        "then scoring with R_theta cannot come out the other way, so the "
        "comparison has no false-hypothesis value and is not a measurement."
    ),
    "reference_probability_of_chosen_successor": (
        "The successor was chosen BY that probability. Selector and scorer are "
        "the same object; the statistic is fixed before any data exists."
    ),
    "support_size_by_arm": (
        "The executable support is identical across arms at a given state by "
        "construction. Any per-state difference is an implementation bug, not a "
        "result; it belongs in the arms-comparable assertion, never in a table."
    ),
    "validity_rate_by_arm": (
        "Every committed state is a complete valid molecule because the executor "
        "makes it so. 100% for all arms is structural and carries no information "
        "about what R_theta learned."
    ),
}


METRIC_REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="endpoint_tanimoto_distance",
        definition="1 - ECFP4 Tanimoto between the endpoint and the source molecule.",
        unit="distance in [0, 1]",
        falsifying_observation=(
            "R_theta near 0 while uniform and empirical-family move away from the "
            "source: the learned law would be making cosmetic edits and the "
            "'nontrivial structural movement' half of the claim fails."
        ),
    ),
    MetricSpec(
        name="heavy_atom_change",
        definition="Signed endpoint minus source heavy-atom count.",
        unit="atoms",
        falsifying_observation=(
            "A large one-signed mean for R_theta: monotone growth or erosion is "
            "trivial movement dressed as transport. Distance alone cannot tell "
            "them apart, which is why this is reported beside it."
        ),
    ),
    MetricSpec(
        name="ring_system_change",
        definition="Signed endpoint minus source fused-ring-system count.",
        unit="ring systems",
        falsifying_observation=(
            "Identically zero for R_theta across the panel: the learned law never "
            "changes ring topology and the transport is confined to decoration."
        ),
    ),
    MetricSpec(
        name="unique_state_fraction",
        definition="Distinct committed states divided by committed states.",
        unit="fraction",
        falsifying_observation=(
            "R_theta materially below the unlearned arms: the learned law would be "
            "revisiting more, which is the cycling pathology the claim denies."
        ),
    ),
    MetricSpec(
        name="immediate_reversal_rate",
        definition="Fraction of steps t with x_{t+1} == x_{t-1}.",
        unit="fraction",
        falsifying_observation=(
            "R_theta materially above the unlearned arms. Note the direction is "
            "NOT preregistered as 'lower is better' -- some reversibility may be "
            "part of the reference law; domination by reversals is the failure."
        ),
    ),
    MetricSpec(
        name="any_state_revisit_rate",
        definition="Fraction of committed states equal to some earlier state.",
        unit="fraction",
        falsifying_observation=(
            "R_theta above the unlearned arms: a learned law that concentrates "
            "mass can cycle MORE than uniform, and that outcome is reachable here."
        ),
    ),
    MetricSpec(
        name="operator_family_entropy",
        definition="Shannon entropy of committed families under fractional attribution.",
        unit="nats",
        falsifying_observation=(
            "R_theta collapsing to a single family (entropy 0) is the operator-"
            "collapse stop rule. Merely being below uniform is expected of any "
            "learned law and is reported descriptively, never as support."
        ),
    ),
    MetricSpec(
        name="family_coverage_count",
        definition="Distinct operator families committed across the panel per arm.",
        unit="families",
        falsifying_observation=(
            "R_theta covering strictly fewer families than the unlearned arms and "
            "abandoning a family entirely: operator collapse under the frozen law."
        ),
    ),
    MetricSpec(
        name="multi_family_trajectory_fraction",
        definition="Fraction of trajectories committing at least two families.",
        unit="fraction",
        falsifying_observation=(
            "Near 0 for R_theta: each trajectory would be a single-operator walk, "
            "not composition of the operator basis."
        ),
    ),
    MetricSpec(
        name="endpoint_uniqueness",
        definition="Distinct endpoints divided by seeds, per source, then averaged.",
        unit="fraction",
        falsifying_observation=(
            "R_theta near 1/seeds: a learned law concentrating onto one endpoint "
            "per source would be near-deterministic transport, not a stochastic "
            "process, and source-conditioned diversity would be lost."
        ),
    ),
    MetricSpec(
        name="envelope_retention",
        definition=(
            "Fraction of committed intermediate states inside the held-in "
            "coordinate-wise descriptor envelope."
        ),
        unit="fraction",
        falsifying_observation=(
            "R_theta below the unlearned arms: the learned law would be leaving "
            "the chemical envelope faster than uniform rewriting, which is the "
            "broad-drift stop rule."
        ),
    ),
    MetricSpec(
        name="standardized_descriptor_drift",
        definition="L2 norm of the standardized descriptor displacement from source.",
        unit="standard deviations",
        falsifying_observation=(
            "R_theta drift growing faster in edit count than the unlearned arms."
        ),
    ),
    MetricSpec(
        name="early_dead_end_rate",
        definition="Fraction of trajectories stopping before the horizon.",
        unit="fraction",
        falsifying_observation=(
            "R_theta materially above the unlearned arms: the learned law would be "
            "steering into terminal states of the shared support."
        ),
    ),
    MetricSpec(
        name="arm_divergence_tv",
        definition="Minimum pairwise total variation between the three arms' laws.",
        unit="total variation in [0, 1]",
        falsifying_observation=(
            "Near 0 across the panel: the arms are one process wearing three "
            "labels, every downstream comparison is void, and the run is an "
            "INVALID_INSTRUMENT rather than a null result."
        ),
    ),
)


def validate_metric_registry(
    registry: Iterable[MetricSpec] = METRIC_REGISTRY,
) -> None:
    """Refuse any declared metric that cannot come out the other way.

    This is the gate the workstream brief requires, made executable.  Prose in a
    protocol document does not stop a metric with a fixed sign from reaching a
    results table; an import-time check does.
    """
    seen: set[str] = set()
    for spec in registry:
        if spec.name in seen:
            raise MetricError(f"metric {spec.name!r} is declared twice")
        seen.add(spec.name)
        if not spec.falsifying_observation.strip():
            raise MetricError(
                f"metric {spec.name!r} declares no falsifying observation. State what "
                "value it would take if the hypothesis were false, or remove it: a "
                "statistic whose sign is fixed before any data exists is not a "
                "measurement."
            )
        if spec.name in BANNED_STATISTICS:
            raise MetricError(
                f"metric {spec.name!r} is on the banned list: {BANNED_STATISTICS[spec.name]}"
            )


def assert_not_banned(name: str) -> None:
    """Raise if a caller tries to report a tautological comparison."""
    if name in BANNED_STATISTICS:
        raise MetricError(f"{name} may not be reported as evidence: {BANNED_STATISTICS[name]}")


# ---- paired resampling over sources ---------------------------------------


def paired_bootstrap_interval(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    resamples: int = 2000,
    seed: int = 20260812,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for ``mean(left - right)`` over SOURCES.

    The source molecule is the independent statistical unit; several rollout
    seeds from one source are repeated measures and must be averaged within the
    source before resampling, or the interval is computed against a denominator
    the experiment does not have.  Callers pass already-averaged per-source
    values.

    Returns ``(point_estimate, low, high)``.
    """
    import random

    shared = sorted(set(left) & set(right))
    if not shared:
        raise MetricError("paired bootstrap needs sources measured under both arms")
    differences = [left[source] - right[source] for source in shared]
    point = sum(differences) / len(differences)
    generator = random.Random(seed)
    size = len(differences)
    means: list[float] = []
    for _ in range(resamples):
        sample = [differences[generator.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return point, _quantile(means, tail), _quantile(means, 1.0 - tail)


def resolved(interval: tuple[float, float, float]) -> bool:
    """True when the paired interval excludes zero."""
    _point, low, high = interval
    return low > 0.0 or high < 0.0


__all__ = [
    "BANNED_STATISTICS",
    "DESCRIPTOR_NAMES",
    "DOMINATED_BY",
    "DOMINATES",
    "DescriptorEnvelope",
    "ENVELOPE_LOWER_QUANTILE",
    "ENVELOPE_UPPER_QUANTILE",
    "FrontierPoint",
    "INCOMPARABLE",
    "METRIC_REGISTRY",
    "MORGAN_BITS",
    "MORGAN_RADIUS",
    "MetricError",
    "MetricSpec",
    "TrajectoryHealth",
    "UNRESOLVED",
    "assert_not_banned",
    "calibrate_envelope",
    "descriptor_vector",
    "family_coverage",
    "family_entropy",
    "frontier_verdict",
    "heavy_atom_count",
    "morgan_fingerprint",
    "multi_family_trajectory",
    "paired_bootstrap_interval",
    "resolved",
    "ring_system_count",
    "tanimoto_distance",
    "trajectory_health",
    "validate_metric_registry",
]
