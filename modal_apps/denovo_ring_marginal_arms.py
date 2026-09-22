"""Three matched de-novo arms, testing whether pinning a GLOBAL ring marginal
repairs the realized ring distribution.

The hypothesis under test
-------------------------
COMPOSE's ring prior is fine globally; its realized ring marginal drifts during
ROLLOUT because the state-dependent executable support distorts it.  The catalog
carries ~3% small-ring mass, but at the states where the sampler actually makes
ring decisions ~44% of the legal support is small rings -- the learned model
pushes that DOWN to ~28% and produces ~28%.  The network is correcting, not
causing.  The mechanism is ``_eligible_grow_host_graph``: every committed ring
permanently removes its atoms from the acyclic-carbon host, so each successive
ring decision is made against a smaller host, and a small host only fits small
rings.

Neither GenMol nor InVirtuoGen owns a ring controller.  Both pin a GLOBAL
marginal before generation -- InVirtuoGen factorizes ``p(x) = p(n) p_theta(x|n)``
with ``p(n)`` fitted empirically on ZINC250k; GenMol draws its mask count from an
empirical training distribution -- and their representations then let ring
content arrive from the data distribution.  This app asks whether the same move
works for the ring skeleton::

    p(G) = p(R) p_theta(G | R)

The arms
--------
``A``  the shipped process, unchanged.
``B``  the ring-system COUNT drawn from the corpus law and realized at ``t = 0``,
       each template chosen freely from the model's own support.
``C``  the full ring-system SIGNATURE drawn from the corpus law and realized at
       ``t = 0``.

``A -> B`` attributes earliness plus the count; ``B -> C`` attributes the pinned
size law.

``B`` IS A SUBSTITUTION and is reported as one.  The brief's arm B was
``exact_early_ring``, which is a TEACHER-TRACE schedule applied at training time:
Lineage B was trained on ``sequential``, so that arm cannot be obtained from this
checkpoint by any inference flag -- only by a retrain.  ``B`` here is the
inference-time analogue of its intent (commit ring transactions at the earliest
state that can carry them) and is the right ablation for separating earliness
from the pinned marginal, but it is not the same object.

Discipline
----------
The latent is drawn BEFORE generation and no endpoint is ever rejected on how
its rings came out; unrealizable plans are recorded, not retried.  No SA or
small-ring reward term exists anywhere in this app -- the diagnosis is that the
defect is upstream of the learned rates.  Every arm shares the ``t = 0`` tree for
a given trajectory seed, because each draws it as the first use of its RNG.

Example::

    modal run --detach modal_apps/denovo_ring_marginal_arms.py::fit_prior_entry
    modal run --detach modal_apps/denovo_ring_marginal_arms.py \
        --samples 200 --shard-size 10
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
if REMOTE_ROOT.is_dir():
    sys.path.insert(0, str(REMOTE_ROOT))
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

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
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT), str(REMOTE_ROOT / "src"))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
)

app = modal.App("compose-denovo-ring-marginal-arms")
volume = modal.Volume.from_name("compose-denovo-eval-v1", create_if_missing=False)

VOL = Path("/vol")
CHECKPOINT = VOL / "lineageB" / "checkpoint.best_so_far.pt"
TRAIN_SMILES = VOL / "guacamol" / "guacamol_subset_500000_seed0.smiles"
BASE = VOL / "ring_marginal"
PRIOR_PATH = BASE / "plan_prior_v1.json"
CENSUS_PATH = BASE / "corpus_ring_census_v1.json"

EXPECTED_SHA256 = {
    "checkpoint": "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c",
    "train": "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791",
}

# Sampling geometry, taken verbatim from ``denovo_official_eval`` so arm A is the
# same process this repository already evaluates, not a re-tuned one.
MAX_ATOMS = 40
OPERATIONAL_HORIZON = 16.0
TIME_STEP = 0.1
MAX_EVENTS = 128
MODEL_LABEL = "step1000"
ARMS = ("A", "B", "C")


# ---- helpers ----------------------------------------------------------------


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_pinned_inputs() -> dict[str, str]:
    observed = {
        "checkpoint": _sha256(CHECKPOINT),
        "train": _sha256(TRAIN_SMILES),
    }
    mismatched = {
        key: (observed[key], EXPECTED_SHA256[key])
        for key in EXPECTED_SHA256
        if observed[key] != EXPECTED_SHA256[key]
    }
    if mismatched:
        raise RuntimeError(f"pinned input sha256 mismatch: {mismatched}")
    return observed


def _runtime_versions() -> dict[str, str]:
    import networkx
    import numpy
    import rdkit
    import scipy
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "rdkit": rdkit.__version__,
    }


def design_tag(total: int, prior_sha256: str, horizon: float = OPERATIONAL_HORIZON) -> str:
    """The FULL identity of a sampling design, used as the shard directory.

    Everything that changes which molecules a shard contains is in the tag:
    the weights, the trajectory-seed family (which depends on ``total``), the
    horizon and the plan prior.  Keying a shard by its seed alone has already
    pooled two designs into one headline number in this repository's
    published-benchmark path.
    """

    return (
        f"m{MODEL_LABEL}_n{int(total)}"
        f"_h{float(horizon):.1f}".replace(".", "p")
        + f"_p{prior_sha256[:12]}"
    )


def shard_path(design: str, arm: str, seed: int, start: int, stop: int) -> Path:
    return BASE / design / arm / f"seed{seed}_{start:06d}_{stop:06d}.json"


def _trajectory_seed_slice(seed: int, total: int, start: int, stop: int):
    from compose_v4.experiments.parallel_tracelet_sampling import _trajectory_seeds

    return _trajectory_seeds(seed, total)[start:stop]


# ---- the corpus law ---------------------------------------------------------


@app.function(image=image, volumes={str(VOL): volume}, timeout=7200, cpu=4, memory=16384)
def fit_prior(limit: int | None = None) -> dict:
    """Fit ``p(R | heavy-atom bin)`` and the reference census from one pass."""

    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from denovo_fit_ring_plan_prior import scan_corpus

    from compose_v4.eval.denovo_ring_marginal import (
        DEFAULT_HEAVY_ATOM_BIN_EDGES,
        RingSystemPlanPrior,
        heavy_atom_bin,
        ring_signature_census,
    )

    _assert_pinned_inputs()
    rows = scan_corpus(TRAIN_SMILES, limit)
    prior = RingSystemPlanPrior.fit(rows, bin_edges=DEFAULT_HEAVY_ATOM_BIN_EDGES)
    census = ring_signature_census(signature for _heavy, signature in rows)
    census["source"] = TRAIN_SMILES.name
    census["source_sha256"] = EXPECTED_SHA256["train"]
    census["heavy_atom_bin_edges"] = list(DEFAULT_HEAVY_ATOM_BIN_EDGES)
    census["by_heavy_atom_bin"] = {
        label: ring_signature_census(
            signature
            for heavy, signature in rows
            if heavy_atom_bin(heavy, edges=DEFAULT_HEAVY_ATOM_BIN_EDGES) == label
        )
        for label in sorted(
            {
                heavy_atom_bin(heavy, edges=DEFAULT_HEAVY_ATOM_BIN_EDGES)
                for heavy, _signature in rows
            }
        )
    }
    BASE.mkdir(parents=True, exist_ok=True)
    prior.write(PRIOR_PATH)
    CENSUS_PATH.write_text(json.dumps(census, indent=1))
    volume.commit()
    return {
        "molecules": len(rows),
        "prior_sha256": _sha256(PRIOR_PATH),
        "prior_bins": sorted(prior.tables),
        "ring_size_fraction": census["ring_size_fraction"],
        "ring_systems_per_molecule": census["ring_systems_per_molecule"],
        "strained_ring_fraction": census["strained_ring_fraction"],
        "fraction_with_strained_ring": census["fraction_with_strained_ring"],
        "versions": _runtime_versions(),
    }


@app.local_entrypoint()
def fit_prior_entry(limit: int = 0) -> None:
    print(json.dumps(fit_prior.remote(limit or None), indent=2, default=str))


# ---- sampling ---------------------------------------------------------------


@app.function(
    image=image,
    volumes={str(VOL): volume},
    timeout=14400,
    cpu=4,
    max_containers=60,
    retries=modal.Retries(max_retries=5),
)
def sample_shard(task: dict) -> dict:
    """Sample one arm's slice of one trajectory-seed family.

    Work units are deliberately SMALL.  A previous 60-shard de-novo fan-out at
    50 trajectories per shard lost 59 of 60 shards because a single shard
    exhausted its retries under preemption and the map raised; the shard is the
    unit that survives a preemption, so it is sized to land between them.
    """

    import time

    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    import torch
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.eval.denovo_ring_marginal import RingSystemPlanPrior
    from compose_v4.experiments.denovo_ring_plan import (
        CatalogSignatureIndex,
        sample_denovo_arm,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    torch.set_num_threads(1)
    volume.reload()

    arm = str(task["arm"])
    seed = int(task["seed"])
    total = int(task["total"])
    start = int(task["start"])
    stop = int(task["stop"])
    horizon = float(task.get("horizon", OPERATIONAL_HORIZON))
    prior_sha = str(task["prior_sha256"])

    design = design_tag(total, prior_sha, horizon)
    out_path = shard_path(design, arm, seed, start, stop)
    if out_path.exists():
        return {"shard": str(out_path.relative_to(BASE)), "reused": True}

    _assert_pinned_inputs()
    observed_prior_sha = _sha256(PRIOR_PATH)
    if observed_prior_sha != prior_sha:
        raise RuntimeError(
            f"plan prior sha256 moved: {observed_prior_sha} != {prior_sha}"
        )
    model, payload = load_factorized_rollout_checkpoint(str(CHECKPOINT))
    model.eval()
    source_prior = payload["tree_source_prior"]
    plan_prior = RingSystemPlanPrior.read(PRIOR_PATH)
    index = CatalogSignatureIndex.build(model.ring_system_templates)
    runtime = de_novo_rewrite_system()

    records = []
    began = time.time()
    fiber_cache: dict = {}
    rate_cache: dict = {}
    for position, trajectory_seed in enumerate(
        _trajectory_seed_slice(seed, total, start, stop), start=start
    ):
        record = sample_denovo_arm(
            model,
            arm=arm,
            rng=np.random.default_rng(trajectory_seed),
            source_prior=source_prior,
            index=index,
            plan_prior=plan_prior,
            n_slots=MAX_ATOMS,
            operational_horizon=horizon,
            time_step=TIME_STEP,
            max_events=MAX_EVENTS,
            fiber_cache=fiber_cache,
            rate_cache=rate_cache,
            runtime=runtime,
        )
        record["index"] = position
        record["trajectory_seed"] = int(trajectory_seed)
        records.append(record)
        print(
            json.dumps(
                {
                    "phase": "sampling",
                    "arm": arm,
                    "done": position - start + 1,
                    "of": stop - start,
                    "elapsed_s": round(time.time() - began, 1),
                }
            ),
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "design": design,
                "seed": seed,
                "total": total,
                "start": start,
                "stop": stop,
                "horizon": horizon,
                "model": MODEL_LABEL,
                "model_checkpoint_sha256": EXPECTED_SHA256["checkpoint"],
                "plan_prior_sha256": prior_sha,
                "elapsed_seconds": time.time() - began,
                "versions": _runtime_versions(),
                "records": records,
            }
        )
    )
    volume.commit()
    print(
        json.dumps(
            {
                "phase": "shard_done",
                "arm": arm,
                "shard": out_path.name,
                "n": len(records),
                "elapsed_s": round(time.time() - began, 1),
            }
        ),
        flush=True,
    )
    return {"shard": str(out_path.relative_to(BASE)), "n": len(records), "reused": False}


# ---- scoring ----------------------------------------------------------------


def _read_arm_records(design: str, arm: str) -> tuple[list[dict], list[str]]:
    """Every record for one arm, refusing shards that span two designs."""

    directory = BASE / design / arm
    if not directory.is_dir():
        return [], []
    paths = sorted(directory.glob("seed*.json"))
    records: list[dict] = []
    totals: dict[int, list[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        totals.setdefault(int(payload.get("total", 0)), []).append(path.name)
        if str(payload.get("arm")) != arm:
            raise RuntimeError(f"shard {path.name} is arm {payload.get('arm')!r}, not {arm!r}")
        records.extend(payload["records"])
    if len(totals) > 1:
        raise RuntimeError(
            f"arm {arm} shards span more than one sampling design: "
            + "; ".join(f"total={t}: {sorted(n)}" for t, n in sorted(totals.items()))
        )
    records.sort(key=lambda item: item["index"])
    return records, [path.name for path in paths]


@app.function(image=image, volumes={str(VOL): volume}, timeout=7200, cpu=4, memory=16384)
def score(total: int, prior_sha256: str, horizon: float = OPERATIONAL_HORIZON) -> dict:
    """Score every arm of one design against each other and against the corpus."""

    import numpy as np
    from rdkit import RDLogger

    from compose_v4.eval.denovo_benchmark import (
        denovo_benchmark_metrics,
        strained_ring_census,
    )
    from compose_v4.eval.denovo_ring_decomposition import ring_decomposition_report
    from compose_v4.eval.denovo_ring_marginal import (
        ring_signature_census,
        ring_size_total_variation,
        ring_system_signature_of_smiles,
    )

    RDLogger.DisableLog("rdApp.*")
    volume.reload()
    design = design_tag(total, prior_sha256, horizon)
    corpus = json.loads(CENSUS_PATH.read_text())

    report: dict = {
        "design": design,
        "model": MODEL_LABEL,
        "total_per_arm": total,
        "horizon": horizon,
        "plan_prior_sha256": prior_sha256,
        "versions": _runtime_versions(),
        "corpus_reference": {
            key: corpus[key]
            for key in (
                "molecules",
                "rings_per_molecule",
                "ring_systems_per_molecule",
                "ring_size_fraction",
                "strained_ring_fraction",
                "fraction_with_strained_ring",
                "ring_system_count_fraction",
                "source",
                "source_sha256",
            )
        },
        "arms": {},
    }
    for arm in ARMS:
        records, shard_names = _read_arm_records(design, arm)
        if not records:
            continue
        generated = [str(item.get("smiles") or "") for item in records]
        metrics = denovo_benchmark_metrics(generated)
        signatures = [
            signature
            for signature in (
                ring_system_signature_of_smiles(text) for text in generated if text
            )
            if signature is not None
        ]
        ring_census = ring_signature_census(signatures) if signatures else None
        distinct = sorted({text for text in generated if text})
        row: dict = {
            "arm": arm,
            "shards": len(shard_names),
            "attempted": len(records),
            "complete": len(records) == total,
            "published_metrics": metrics,
            "strained_ring_census": strained_ring_census(distinct),
            "decomposition": ring_decomposition_report(generated, records),
            "ring_signature_census": ring_census,
            "valid_state_fraction": float(
                np.mean([bool(item.get("valid_state")) for item in records])
            ),
            "connected_fraction": float(
                np.mean([bool(item.get("connected")) for item in records])
            ),
            "exhausted_event_budget_fraction": float(
                np.mean([bool(item.get("exhausted_event_budget")) for item in records])
            ),
            "mean_events": float(np.mean([int(item["events"]) for item in records])),
            "mean_plan_events": float(
                np.mean([int(item.get("plan_events", 0)) for item in records])
            ),
        }
        if ring_census is not None:
            row["ring_size_total_variation_vs_corpus"] = ring_size_total_variation(
                ring_census["ring_size_fraction"], corpus["ring_size_fraction"]
            )
        plans = [item.get("ring_plan") for item in records if item.get("ring_plan")]
        if plans:
            requested = sum(len(plan["requested"]) for plan in plans)
            realized = sum(len(plan["realized"]) for plan in plans)
            reasons: dict[str, int] = {}
            for plan in plans:
                for _system, reason in plan["unrealized"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
            row["plan_realization"] = {
                "plans": len(plans),
                "requested_systems": requested,
                "realized_systems": realized,
                "realization_rate": realized / requested if requested else 0.0,
                "fully_realized_fraction": sum(
                    1 for plan in plans if plan["fully_realized"]
                )
                / len(plans),
                "unrealized_reasons": reasons,
                "bin_fallback_fraction": sum(
                    1
                    for plan in plans
                    if plan.get("bin_provenance", {}).get("bin_fallback")
                )
                / len(plans),
            }
        report["arms"][arm] = row

    out = BASE / design / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    volume.commit()
    return report


@app.local_entrypoint()
def main(
    samples: int = 200,
    shard_size: int = 10,
    seed: int = 20260922,
    horizon: float = OPERATIONAL_HORIZON,
    arms: str = "A,B,C",
) -> None:
    """Fan out every arm's shards, then score whatever landed."""

    prior_sha = prior_digest.remote()
    selected = tuple(part.strip() for part in arms.split(",") if part.strip())
    tasks = [
        {
            "arm": arm,
            "seed": seed,
            "total": samples,
            "start": start,
            "stop": min(start + shard_size, samples),
            "horizon": horizon,
            "prior_sha256": prior_sha,
        }
        for arm in selected
        for start in range(0, samples, shard_size)
    ]
    print(
        json.dumps(
            {
                "design": design_tag(samples, prior_sha, horizon),
                "arms": selected,
                "shards": len(tasks),
                "shard_size": shard_size,
            }
        ),
        flush=True,
    )
    landed = 0
    failed = 0
    # ``order_outputs=False`` because an ordered map makes a healthy fan-out
    # deliver nothing until its slowest shard returns.  Per-shard failures are
    # caught so one exhausted shard cannot destroy its healthy siblings.
    for result in sample_shard.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failed += 1
            print(json.dumps({"phase": "shard_failed", "error": repr(result)}), flush=True)
            continue
        landed += 1
        print(json.dumps({"phase": "landed", **result}), flush=True)
    print(json.dumps({"phase": "fanout_done", "landed": landed, "failed": failed}), flush=True)
    report = score.remote(samples, prior_sha, horizon)
    print(json.dumps({"phase": "scored", "arms": sorted(report["arms"])}, indent=1))


@app.function(image=image, volumes={str(VOL): volume}, timeout=600, cpu=1)
def prior_digest() -> str:
    volume.reload()
    if not PRIOR_PATH.exists():
        raise RuntimeError(
            "the plan prior has not been fitted; run ::fit_prior_entry first"
        )
    return _sha256(PRIOR_PATH)
