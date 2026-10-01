"""Task semantics shared by complete-program search, not interchangeable oracles."""

import math
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import decode_state


@dataclass(frozen=True)
class ProgramTask:
    name: str
    oracle_protocol: str
    kind: str
    original_seed: str | None = None
    delta: float | None = None

    def __post_init__(self):
        if not self.name or not self.oracle_protocol or self.kind not in ("t4", "pmo"):
            raise ValueError("task needs a name, protocol and explicit t4/pmo kind")
        if self.kind == "t4":
            if (
                not self.original_seed
                or Chem.MolFromSmiles(self.original_seed) is None
                or self.delta not in (0.4, 0.6)
            ):
                raise ValueError("T4 requires its original benchmark seed and similarity threshold")
        elif self.original_seed is not None or self.delta is not None:
            raise ValueError("PMO must not inherit T4 original-seed constraints")

    @property
    def top_k(self):
        return 1 if self.kind == "t4" else 10

    @property
    def task_id(self):
        return identity(asdict(self))

    def utility(self, score):
        if isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("task observations must be finite numbers")
        if self.kind == "pmo" and not 0 <= score <= 1:
            raise ValueError("PMO reward outside declared [0,1] range")
        return -float(score) if self.kind == "t4" else float(score)

    def endpoint_evaluator(self):
        if self.kind == "pmo":
            # Executor support is checked separately. No T4 medicinal-chemistry filter.
            return lambda row: {
                **row,
                "oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None,
            }
        from rdkit.Contrib.SA_Score import sascorer

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(self.original_seed))

        def evaluate(row):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                return {**row, "oracle_eligible": False, "reason": "invalid_molecule"}
            result = {
                **row,
                **calculate_properties(
                    mol,
                    seed_fp=seed_fp,
                    generator=generator,
                    sa_scorer=sascorer.calculateScore,
                    delta=self.delta,
                    qed_min=0.6,
                    sa_max=4.0,
                ),
            }
            return {**result, "oracle_eligible": acceptable_endpoint(result)}

        return evaluate


def archive_top_k(observations, *, k):
    """Mean of best min(k,n) unique-endpoint mean utilities; not official PMO AUC.

    Repeats update an endpoint's mean rather than creating extra top-ten entries.
    Failures must be excluded explicitly by the caller, never imputed as rewards.
    """
    if type(k) is not int or k < 1:
        raise ValueError("positive integer archive k required")
    labels = {}
    for endpoint, utility in observations:
        if not endpoint or not math.isfinite(utility):
            raise ValueError("archive needs identified finite eligible observations")
        labels.setdefault(endpoint, []).append(utility)
    if not labels:
        return None
    return float(np.mean(sorted((np.mean(v) for v in labels.values()), reverse=True)[:k]))


def predicted_archive_gains(predictions, observed_utilities, *, k):
    """Plug-in gain to a fixed-size best-k archive, not posterior expected gain."""
    values = np.asarray(predictions, dtype=float)
    observed = np.asarray(observed_utilities, dtype=float)
    if (
        values.ndim != 1
        or observed.ndim != 1
        or not np.isfinite(values).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("finite one-dimensional prediction/archive arrays required")
    if type(k) is not int or k < 1:
        raise ValueError("positive integer archive k required")
    if len(observed) < k:
        # Before the archive fills, utility ranking is explicit, not a fictional
        # gain against fabricated zero-valued molecules (especially for docking).
        return values.copy()
    threshold = np.sort(observed)[-k]
    return np.maximum(values - threshold, 0) / k


def initialization_lock(records, *, count, seed, source_sha256):
    """Choose exact task-independent starts before seeing any oracle labels.

    Every returned row requires a charged initialization query. This function
    never calls an oracle or treats an initialization state as a non-null edit.
    """
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive initialization count and nonnegative seed required")
    if len(source_sha256) != 64:
        raise ValueError("initialization corpus SHA-256 required")
    rows = {}
    for record in records:
        if set(record) != {"state", "source_id"}:
            raise ValueError("initialization accepts only exact state/source_id, never task labels")
        state = decode_state(record["state"])
        _, trace = execute_program(state, [])
        rows.setdefault(trace["endpoint"], {**record, "endpoint": trace["endpoint"]})
    if count > len(rows):
        raise ValueError("not enough unique supported task-independent initial states")
    keys = sorted(rows)
    selected = np.random.default_rng(seed).choice(len(keys), size=count, replace=False)
    body = {
        "schema_version": "program_initialization_lock_v1",
        "source_sha256": source_sha256,
        "seed": seed,
        "count": count,
        "available_unique": len(keys),
        "candidates": [rows[keys[int(i)]] for i in selected],
        "accounting": "all initialization scores count against each run's oracle budget",
        "new_oracle_calls": 0,
    }
    return {**body, "lock_sha256": identity(body)}


def pmo_top_ten_auc(ordered_rewards, *, budget, frequency=100, finish=False):
    """PMO mol_opt top_auc convention: logged trapezoids and optional flat tail.

    Source inspected 2026-09-12:
    https://github.com/wenhao-gao/mol_opt/blob/main/main/optimizer.py#L27
    Input is one reward per charged unique canonical molecule, including starts.
    Failed or unresolved evaluations are not fabricated zero-valued molecules.
    """
    values = np.asarray(ordered_rewards, dtype=float)
    if type(budget) is not int or budget < 1 or type(frequency) is not int or frequency < 1:
        raise ValueError("positive PMO budget and logging frequency required")
    if (
        values.ndim != 1
        or len(values) > budget
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("invalid unique-query PMO reward sequence")
    if not len(values):
        return None
    previous_x, previous_y, area = 0, 0.0, 0.0
    for count in [*range(frequency, len(values), frequency), len(values)]:
        current = float(np.mean(np.sort(values[:count])[-10:]))
        area += (count - previous_x) * (current + previous_y) / 2
        previous_x, previous_y = count, current
    if finish:
        area += (budget - len(values)) * previous_y
    return area / budget
