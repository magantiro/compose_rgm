"""One durable, resumable, metered Task 3 run -- the object a policy is handed.

A policy gets `evaluate()`, `remaining`, `archive` and `checkpoint()`, and gets
durability and budget enforcement whether or not it thought about them.  It
cannot reach the objectives without going through the meter, and it cannot spend
a call without that call reaching disk.

    run = Task3Run.open(root, seed=0, policy="graph_ga")
    while run.remaining:
        offspring = policy.propose(run.archive, run.rng)
        values = run.evaluate(offspring)          # charged, ledgered, cached
        run.archive = policy.select(...)
        run.checkpoint(policy_state=policy.state())
    run.close()

RESUME IS EXACT, NOT APPROXIMATE
--------------------------------
Reopening the same directory replays the ledger into the meter (so already-paid
molecules cost nothing again), restores the RNG bit-generator state (so the next
random draw is the one the crashed run would have made), and restores the
archive and step.  Because the oracle is batch-invariant and the ledger stores
exact float64, a resumed run continues the run it resumed rather than a similar
one.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from compose_v4.benchmark.molleo_task3 import OracleMeter, hypervolume_qmc
from compose_v4.benchmark.oracles import Task3Objectives, canonical
from compose_v4.benchmark.run_store import ResumeState, RunStore, restore_rng

DEFAULT_BUDGET = 10_000


def _git_commit() -> str:
    """Which tree produced this run. Unknown is recorded as unknown."""

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return "unknown"


@dataclass
class Task3Run:
    """A metered, ledgered, resumable run over the five frozen objectives."""

    store: RunStore
    meter: OracleMeter
    objectives: Task3Objectives
    rng: np.random.Generator
    seed: int
    step: int = 0
    archive: list[str] = field(default_factory=list)
    resumed: ResumeState | None = None

    @classmethod
    def open(cls, root: Path, *, seed: int, budget: int = DEFAULT_BUDGET,
             policy: str = "unnamed", objectives: Task3Objectives | None = None,
             fsync_every: int = 25, resume: bool = True) -> Task3Run:
        objectives = objectives or Task3Objectives()
        metadata = {
            "oracle_bundle": str(objectives.bundle_dir),
            "oracle_manifest_schema": objectives.manifest["schema"],
            "git_commit": _git_commit(),
            "counting_rule": "one novel canonical molecule = one unit",
            "semantics": "strict-10k",
        }
        store, state = RunStore.open(
            Path(root), seed=seed, budget=budget, policy=policy,
            fsync_every=fsync_every, metadata=metadata, resume=resume)
        meter = OracleMeter(objectives, budget=budget, canonicalize=canonical,
                            evaluate_many=objectives.evaluate_many,
                            on_evaluated=store.record)
        # The seed is the RNG's identity on a fresh run; on a resumed run the
        # checkpointed bit-generator state supersedes it, because continuing
        # from a re-seeded generator would silently repeat draws.
        rng = np.random.default_rng(seed)
        step, archive = 0, []
        if state is not None:
            meter.restore(state.evaluations)
            rng = restore_rng(state.rng_state) or rng
            step, archive = state.step, list(state.archive)
        return cls(store=store, meter=meter, objectives=objectives, rng=rng,
                   seed=seed, step=step, archive=archive, resumed=state)

    # ---- what a policy uses ---------------------------------------------

    @property
    def remaining(self) -> int:
        return self.meter.remaining

    @property
    def spent(self) -> int:
        return self.meter.spent

    def evaluate(self, smiles: list[str]) -> list[tuple[float, ...]]:
        """Charge, evaluate, ledger, FSYNC, and only then hand values back.

        The flush is the point. Without it a crash could leave the policy having
        already acted on evaluations that never reached disk -- and on resume
        those molecules would be re-proposed and charged a SECOND time, so the
        run would have consumed more real oracle work than its counter admits.
        That is the exact accounting leak this harness refuses to copy from the
        released benchmark; it would be indefensible to reintroduce it through
        the back door of a lost write.

        The guarantee this buys, stated exactly: NO EVALUATION THAT INFLUENCED
        THE SEARCH IS UNCHARGED. A crash can still lose an in-flight batch, but
        the policy never saw it, so it gained nothing from it.

        One fsync per generation is ~80 fsyncs in a 10,000-call run. It is not
        worth optimising.
        """

        values = self.meter.batch(smiles)
        self.store.flush()
        return values

    def affordable(self, smiles: list[str]) -> list[str]:
        """The longest prefix of `smiles` that fits in what is left.

        Offered so a policy near the end of its budget trims deliberately
        instead of discovering the wall by exception.
        """

        keep: list[str] = []
        novel = 0
        seen: set[str] = set()
        evaluated = self.meter.evaluated()
        for item in smiles:
            key = canonical(item) or item
            if key not in evaluated and key not in seen:
                # Ask the meter rather than reimplementing its arithmetic:
                # under the per-objective counting rule each molecule costs five.
                if not self.meter.can_afford(novel + 1):
                    break
                seen.add(key)
                novel += 1
            keep.append(item)
        return keep

    def checkpoint(self, *, policy_state: dict | None = None,
                   arrays: dict[str, np.ndarray] | None = None) -> None:
        self.store.checkpoint(step=self.step, archive=self.archive, rng=self.rng,
                              policy_state=policy_state, arrays=arrays)

    def hypervolume(self, *, log2_samples: int = 20) -> float:
        """HV of everything evaluated so far, against the origin.

        All five objectives are normalised to [0, 1] with higher better, so the
        reference point is the origin and the value is a fraction of the unit
        box -- comparable across runs without a shared reference set.

        Deterministic (Sobol over the fixed unit box), so two policies are
        measured against the very same sample points and a small real difference
        is not buried under two independent estimation errors.
        """

        points = list(self.meter.evaluated().values())
        return hypervolume_qmc(points, log2_samples=log2_samples)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Task3Run:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
