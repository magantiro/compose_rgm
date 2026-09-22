"""Unconditional de-novo generation evaluation under the pinned production kernel.

This app answers one question: what does COMPOSE's *target-free* learned process
produce when it is sampled with no target, no source molecule, and no controller?

The process is ancestral CTMC sampling (``sample_tracelet_ancestral``) started
from ``DegreeBoundedCarbonTreePrior`` -- a random degree-capped all-carbon alkane
tree whose size is drawn from a corpus-fitted categorical.  That prior is
structured *noise*: it carries no heteroatom, bond-order, ring or scaffold
information, and it is not a corpus molecule.  It is serialized inside the
checkpoint as ``tree_source_prior``, so the sampled initial law is exactly the
one the model was trained against.

Kernel pinning
--------------
Every number this app produces is computed inside an image pinned to the
versions the checkpoint's own ``manifest.json`` records for its training run
(python 3.11, torch 2.4.0, numpy 1.26.4, scipy 1.13.1, networkx 3.3,
rdkit 2024.3.5).  The de-novo rollout path never calls
``build_production_ringcore_catalog``: the typed ring catalog is deserialized
from the checkpoint payload, which is a strictly stronger guarantee than the
frozen-fingerprint check (it is byte-identical to the catalog used in training).
The rdkit pin still matters, because validity, canonicalization and aromaticity
perception all feed the reported metrics.

Sharding is exact, not approximate
----------------------------------
Per-trajectory seeds come from ``SeedSequence(seed).spawn(samples)``, so shard
``k`` of a run simply takes the slice ``[k*size, (k+1)*size)`` of that tuple and
the union over shards is bit-identical to one unsharded run at the same seed.

Example
-------
    modal run --detach modal_apps/denovo_official_eval.py::probe
    modal run --detach modal_apps/denovo_official_eval.py \
        --samples-per-seed 1000 --seeds 3 --shards-per-seed 20
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

# Pinned to the checkpoint manifest's recorded training runtime:
#   {"cuda_device_name": "NVIDIA A100-SXM4-40GB", "python": "3.11.12",
#    "rdkit": "2024.03.5", "torch": "2.4.0+cu121"}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
        "fcd-torch==1.0.7",
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

app = modal.App("compose-denovo-official-eval")
volume = modal.Volume.from_name("compose-denovo-eval-v1", create_if_missing=False)

VOL = Path("/vol")
CHECKPOINT = VOL / "lineageB" / "checkpoint.best_so_far.pt"
TRAIN_SMILES = VOL / "guacamol" / "guacamol_subset_500000_seed0.smiles"
REFERENCE_SMILES = VOL / "guacamol" / "guacamol_heldout_val_5000_seed0.smiles"

# SHA-256 of the two corpus files and the checkpoint, as recorded in the
# training run's manifest.json.  A mismatch means the evaluation is not being
# run against the artifacts the model was trained with, so it aborts.
EXPECTED_SHA256 = {
    "checkpoint": "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c",
    "train": "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791",
    "reference": "1f8e92f70018dfb94965977f87b7844dea791e4727ea8f7d251946925d1aa99f",
}

# ---- Checkpoint sweep registry ----
#
# The model states of the Lineage B de-novo trajectory that survive as artifacts,
# each pinned by sha256 so a sweep point can never silently sample the wrong
# weights.  Both were trained from the SAME recipe and seed (20260717); the
# step-2500 state comes from the continuation run whose validation history is
# byte-identical to Lineage B's through step 1000, so these are two points on one
# trajectory rather than two independent models.
#
# ``step1000`` deliberately keeps the historical UNNESTED shard directory so every
# shard sampled before this registry existed stays valid and reusable.
DEFAULT_MODEL = "step1000"
MODELS: dict[str, dict[str, str]] = {
    "step1000": {
        "relative_path": "lineageB/checkpoint.best_so_far.pt",
        "sha256": "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c",
        "completed_steps": "1000",
    },
    "step2500": {
        "relative_path": "lineageB_step2500/checkpoint.best_so_far.pt",
        "sha256": "bb53e00237bd4be60436e149c7f92cd75b30ce555ae96d9264164e303150abb1",
        "completed_steps": "2500",
    },
}


def model_checkpoint(model: str) -> Path:
    """The pinned checkpoint path for one sweep point."""

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {sorted(MODELS)}")
    return VOL / MODELS[model]["relative_path"]

# Sampling geometry, taken from the production rollout evaluator's defaults
# (scripts/evaluate_tracelet_rollouts.py) so this is the same process the repo
# already evaluates, not a re-tuned one.
MAX_ATOMS = 40
OPERATIONAL_HORIZON = 16.0
TIME_STEP = 0.1
MAX_EVENTS = 128


# ---- helpers ----------------------------------------------------------------


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_pinned_inputs(model: str = DEFAULT_MODEL) -> dict[str, str]:
    """Fail closed if any input is not the artifact the model was trained with.

    The corpus pins are shared across sweep points; the checkpoint pin is the one
    for ``model``, so sampling the wrong weights aborts rather than producing a
    plausible row attributed to the wrong training step.
    """

    expected = {
        "checkpoint": MODELS[model]["sha256"],
        "train": EXPECTED_SHA256["train"],
        "reference": EXPECTED_SHA256["reference"],
    }
    observed = {
        "checkpoint": _sha256(model_checkpoint(model)),
        "train": _sha256(TRAIN_SMILES),
        "reference": _sha256(REFERENCE_SMILES),
    }
    mismatched = {
        key: (observed[key], expected[key])
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatched:
        raise RuntimeError(f"pinned input sha256 mismatch for model {model!r}: {mismatched}")
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


def _trajectory_seed_slice(seed: int, total: int, start: int, stop: int) -> tuple[int, ...]:
    from compose_v4.experiments.parallel_tracelet_sampling import _trajectory_seeds

    return _trajectory_seeds(seed, total)[start:stop]


def horizon_tag(horizon: float) -> str:
    """Filesystem-safe tag for one sweep point ('16.0' -> 'h16p0')."""

    return "h" + f"{float(horizon):.1f}".replace(".", "p")


def shard_path(
    seed: int, start: int, stop: int, horizon: float, model: str = DEFAULT_MODEL
) -> Path:
    """Shard location, keyed by horizon AND model.

    The model is part of the PATH for the same reason the horizon is: two sweep
    points must never silently reuse each other's shards.  The default model keeps
    the historical unnested layout so shards sampled before the sweep registry
    existed remain addressable.
    """

    base = VOL / "shards" / horizon_tag(horizon)
    if model != DEFAULT_MODEL:
        base = base / model
    return base / f"seed{seed}_{start:06d}_{stop:06d}.json"


def _read_smiles(path: Path, limit: int | None = None) -> tuple[str, ...]:
    values: list[str] = []
    with path.open() as handle:
        for line in handle:
            text = line.strip().split()[0] if line.strip() else ""
            if text:
                values.append(text)
            if limit is not None and len(values) >= limit:
                break
    return tuple(values)


# ---- probe ------------------------------------------------------------------


@app.function(image=image, volumes={str(VOL): volume}, timeout=1800, cpu=2)
def probe() -> dict:
    """Verify the pinned image, the inputs, and that the model samples at all."""

    import numpy as np

    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.chem.state import is_connected_or_null, is_valid_state
    from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral

    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    versions = _runtime_versions()
    shas = _assert_pinned_inputs()
    model, payload = load_factorized_rollout_checkpoint(str(CHECKPOINT))
    prior = payload["tree_source_prior"]

    rng = np.random.default_rng(0)
    sources = [molecular_graph_to_smiles(prior.sample(rng, n_slots=MAX_ATOMS)) for _ in range(3)]

    samples = []
    for trajectory_seed in _trajectory_seed_slice(20260920, 4, 0, 4):
        rollout = sample_tracelet_ancestral(
            model,
            rng=np.random.default_rng(trajectory_seed),
            n_slots=MAX_ATOMS,
            operational_horizon=OPERATIONAL_HORIZON,
            time_step=TIME_STEP,
            max_events=MAX_EVENTS,
            source_prior=prior,
        )
        samples.append(
            {
                "smiles": molecular_graph_to_smiles(rollout.final_state),
                "events": len(rollout.event_times),
                "valid_state": bool(is_valid_state(rollout.final_state)),
                "connected": bool(is_connected_or_null(rollout.final_state)),
            }
        )

    return {
        "versions": versions,
        "input_sha256": shas,
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "completed_steps": payload.get("completed_steps"),
        "training_steps": payload.get("training_steps"),
        "source_prior": payload.get("source_prior"),
        "source_prior_class": type(prior).__name__,
        "atom_vocabulary_size": int(getattr(model, "atom_vocabulary_size", -1) or -1),
        "t0_sources": sources,
        "samples": samples,
    }


@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=4, memory=8192)
def corpus_reference(limit: int = 5000) -> dict:
    """Score the TRAINING CORPUS itself on the published metric.

    This is the calibration a generated-sample number is meaningless without:
    ``QED >= 0.6 AND SA <= 4`` is a strong filter -- aspirin (QED 0.550) and
    caffeine (0.538) both fail it -- so "quality 40%" means nothing until the
    rate real drug-like molecules achieve on the SAME metric is known.  It is
    an upper reference for a generator trained on this corpus, not a target,
    and it is NOT a published baseline.
    """

    from compose_v4.eval.denovo_benchmark import denovo_benchmark_metrics

    _assert_pinned_inputs()
    rows: dict[str, dict] = {}
    for name, path in (("train", TRAIN_SMILES), ("heldout_reference", REFERENCE_SMILES)):
        smiles = _read_smiles(path, limit=limit)
        metrics = denovo_benchmark_metrics(list(smiles))
        metrics["source"] = name
        metrics["limit"] = limit
        rows[name] = metrics
        print(
            json.dumps(
                {
                    "phase": "corpus_reference",
                    "source": name,
                    "n": metrics["attempted"],
                    "validity": metrics["validity"],
                    "uniqueness": metrics["uniqueness"],
                    "quality": metrics["quality"],
                    "quality_given_valid_unique": metrics["quality_given_valid_unique"],
                    "diversity": metrics["diversity"],
                }
            ),
            flush=True,
        )

    out = VOL / "corpus_reference.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    volume.commit()
    return rows


@app.local_entrypoint()
def corpus_reference_entry(limit: int = 5000) -> None:
    print(json.dumps(corpus_reference.remote(limit), indent=2, default=str))


# ---- sampling ---------------------------------------------------------------


@app.function(
    image=image,
    volumes={str(VOL): volume},
    timeout=14400,
    cpu=8,
    max_containers=40,
    retries=modal.Retries(max_retries=3),
)
def sample_shard(task: dict) -> dict:
    """Sample one exact slice of one seed's trajectory-seed tuple."""

    import time

    import numpy as np

    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.chem.state import is_connected_or_null, is_valid_state
    from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral

    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    import torch
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    torch.set_num_threads(1)

    seed = int(task["seed"])
    total = int(task["total"])
    start = int(task["start"])
    stop = int(task["stop"])
    # The exploration parameter swept for the quality-diversity frontier.  It is
    # part of the shard PATH so a sweep point can never silently reuse a shard
    # sampled at a different horizon.
    horizon = float(task.get("horizon", OPERATIONAL_HORIZON))
    model_label = str(task.get("model", DEFAULT_MODEL))
    out_path = shard_path(seed, start, stop, horizon, model_label)
    if out_path.exists():
        volume.reload()
        return {
            "shard": out_path.name,
            "reused": True,
            "seed": seed,
            "horizon": horizon,
            "model": model_label,
        }

    _assert_pinned_inputs(model_label)
    model, payload = load_factorized_rollout_checkpoint(str(model_checkpoint(model_label)))
    prior = payload["tree_source_prior"]
    model.eval()

    records = []
    began = time.time()
    fiber_cache: dict = {}
    rate_cache: dict = {}
    for index, trajectory_seed in enumerate(
        _trajectory_seed_slice(seed, total, start, stop), start=start
    ):
        rollout = sample_tracelet_ancestral(
            model,
            rng=np.random.default_rng(trajectory_seed),
            n_slots=MAX_ATOMS,
            operational_horizon=horizon,
            time_step=TIME_STEP,
            max_events=MAX_EVENTS,
            fiber_cache=fiber_cache,
            rate_cache=rate_cache,
            source_prior=prior,
        )
        records.append(
            {
                "index": index,
                "trajectory_seed": int(trajectory_seed),
                "smiles": molecular_graph_to_smiles(rollout.final_state),
                "events": len(rollout.event_times),
                "valid_state": bool(is_valid_state(rollout.final_state)),
                "connected": bool(is_connected_or_null(rollout.final_state)),
                "exhausted_event_budget": bool(
                    getattr(rollout, "exhausted_event_budget", False)
                ),
                "event_rules": list(rollout.event_rules),
            }
        )
        if (index - start + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "phase": "sampling",
                        "seed": seed,
                        "done": index - start + 1,
                        "of": stop - start,
                        "elapsed_s": round(time.time() - began, 1),
                    }
                ),
                flush=True,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_out = {
        "seed": seed,
        "total": total,
        "start": start,
        "stop": stop,
        "horizon": horizon,
        # Which sweep point produced these molecules, and the exact weights, so a
        # shard is self-describing rather than identified only by its directory.
        "model": model_label,
        "model_checkpoint_sha256": MODELS[model_label]["sha256"],
        "model_completed_steps": MODELS[model_label]["completed_steps"],
        "elapsed_seconds": time.time() - began,
        "versions": _runtime_versions(),
        "records": records,
    }
    out_path.write_text(json.dumps(payload_out))
    volume.commit()
    print(
        json.dumps(
            {
                "phase": "shard_done",
                "shard": out_path.name,
                "n": len(records),
                "elapsed_s": round(time.time() - began, 1),
            }
        ),
        flush=True,
    )
    return {
        "shard": out_path.name,
        "n": len(records),
        "reused": False,
        "seed": seed,
        "horizon": horizon,
    }


# ---- scoring ----------------------------------------------------------------


def _read_seed_shards(shard_dir: Path, seed: int) -> tuple[list[dict], int | None, list[str]]:
    """Records for one seed, refusing shards that span more than one design.

    Per-trajectory seeds are derived from ``(seed, total)``, so two shards with
    the same ``seed`` but different ``total`` are different trajectory families.
    They can coexist in one directory under plausible names whenever an earlier
    run used a different N, and the glob matches both -- pooling them would mix
    two experiments into one headline number while every count still looked
    reasonable.  Fail closed instead.
    """

    paths = sorted(shard_dir.glob(f"seed{seed}_*.json"))
    records: list[dict] = []
    totals: dict[int, list[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        totals.setdefault(int(payload.get("total", 0)), []).append(path.name)
        records.extend(payload["records"])
    if len(totals) > 1:
        raise RuntimeError(
            f"shards for seed {seed} span more than one sampling design in {shard_dir}: "
            + "; ".join(f"total={total}: {sorted(names)}" for total, names in sorted(totals.items()))
        )
    records.sort(key=lambda item: item["index"])
    total_expected = next(iter(totals)) if totals else None
    return records, (total_expected or None), [path.name for path in paths]


@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=4, memory=8192)
def score_available(seed: int, horizon: float = OPERATIONAL_HORIZON) -> dict:
    """Score whatever shards have landed for one seed, WITHOUT requiring the
    full run.

    ``score_seed`` deliberately refuses a partial seed, because the headline
    row must be the complete 1,000.  This variant exists so an in-flight run
    can be reported honestly at its true N: it returns ``attempted`` and
    ``complete`` so a partial number can never be mistaken for the full one.
    """

    from rdkit import Chem, RDLogger

    from compose_v4.eval.denovo_benchmark import (
        denovo_benchmark_metrics,
        strained_ring_census,
    )

    RDLogger.DisableLog("rdApp.*")
    volume.reload()
    _assert_pinned_inputs()

    shard_dir = VOL / "shards" / horizon_tag(horizon)
    records, total_expected, shard_names = _read_seed_shards(shard_dir, seed)

    generated = [item["smiles"] or "" for item in records]
    metrics = denovo_benchmark_metrics(generated)

    import numpy as np

    distinct = sorted(
        {
            Chem.MolToSmiles(Chem.MolFromSmiles(t))
            for t in generated
            if t and Chem.MolFromSmiles(t) is not None
        }
    )
    metrics["strained_ring_census"] = strained_ring_census(distinct)
    metrics["seed"] = seed
    metrics["horizon"] = horizon
    metrics["shards_present"] = len(shard_names)
    metrics["target_total"] = total_expected
    metrics["complete"] = bool(total_expected and len(records) == total_expected)
    metrics["mean_events"] = (
        float(np.mean([item["events"] for item in records])) if records else 0.0
    )
    metrics["connected_fraction"] = (
        float(np.mean([bool(i.get("connected")) for i in records])) if records else 0.0
    )
    metrics["valid_state_fraction"] = (
        float(np.mean([bool(i.get("valid_state")) for i in records])) if records else 0.0
    )
    metrics["mean_heavy_atoms"] = (
        float(
            np.mean(
                [
                    Chem.MolFromSmiles(t).GetNumAtoms()
                    for t in generated
                    if t and Chem.MolFromSmiles(t) is not None
                ]
            )
        )
        if generated
        else 0.0
    )
    metrics["versions"] = _runtime_versions()

    out = VOL / "partial_reports" / horizon_tag(horizon) / f"seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, default=str))
    volume.commit()
    return metrics


@app.local_entrypoint()
def score_available_entry(
    seed: int = 20260920, horizon: float = OPERATIONAL_HORIZON
) -> None:
    print(json.dumps(score_available.remote(seed, horizon), indent=2, default=str))



@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=4, memory=8192)
def score_seed(
    seed: int,
    samples_per_seed: int,
    horizon: float,
    with_novelty: bool = False,
) -> dict:
    """Primary four metrics for ONE seed at ONE horizon.

    Deliberately cheap and FCD-free so a completed seed can be reported the
    moment its shards land, rather than after the whole 3-seed grid.  Novelty
    needs the 500k train corpus canonicalized, so it is opt-in.
    """

    from rdkit import Chem, RDLogger

    from compose_v4.eval.denovo_benchmark import (
        denovo_benchmark_metrics,
        strained_ring_census,
    )

    RDLogger.DisableLog("rdApp.*")
    volume.reload()

    shard_dir = VOL / "shards" / horizon_tag(horizon)
    records, _total, shard_names = _read_seed_shards(shard_dir, seed)

    indices = [item["index"] for item in records]
    if indices != list(range(samples_per_seed)):
        raise RuntimeError(
            f"seed {seed} @ {horizon}: expected contiguous 0..{samples_per_seed - 1}, "
            f"got {len(indices)} records from {len(shard_names)} shards"
        )

    # One entry per ATTEMPT; an empty string is a generation that emitted no
    # molecule and must count against validity rather than be dropped.
    generated = [item["smiles"] or "" for item in records]

    train_canonical = None
    if with_novelty:
        train_canonical = set()
        for text in _read_smiles(TRAIN_SMILES):
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                train_canonical.add(Chem.MolToSmiles(mol))

    metrics = denovo_benchmark_metrics(generated, train_canonical=train_canonical)

    import numpy as np

    metrics["seed"] = seed
    metrics["horizon"] = horizon
    metrics["mean_events"] = float(np.mean([item["events"] for item in records]))
    metrics["budget_exhausted_fraction"] = float(
        np.mean([bool(item.get("exhausted_event_budget")) for item in records])
    )
    metrics["connected_fraction"] = float(
        np.mean([bool(item.get("connected")) for item in records])
    )
    metrics["valid_state_fraction"] = float(
        np.mean([bool(item.get("valid_state")) for item in records])
    )
    metrics["mean_heavy_atoms"] = float(
        np.mean(
            [
                Chem.MolFromSmiles(text).GetNumAtoms()
                for text in generated
                if text and Chem.MolFromSmiles(text) is not None
            ]
        )
    )
    # Census over the distinct valid molecules -- the same set the quality
    # numerator is drawn from -- so the two are directly comparable.
    from rdkit import Chem as _Chem

    distinct = sorted(
        {
            _Chem.MolToSmiles(_Chem.MolFromSmiles(t))
            for t in generated
            if t and _Chem.MolFromSmiles(t) is not None
        }
    )
    metrics["strained_ring_census"] = strained_ring_census(distinct)
    metrics["versions"] = _runtime_versions()
    metrics["input_sha256"] = _assert_pinned_inputs()

    out = VOL / "seed_reports" / horizon_tag(horizon) / f"seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, default=str))
    volume.commit()
    return metrics


@app.function(image=image, volumes={str(VOL): volume}, timeout=7200, cpu=8, memory=16384)
def score(
    run_label: str,
    samples_per_seed: int,
    seeds: list[int],
    horizon: float = OPERATIONAL_HORIZON,
    include_fcd: bool = False,
) -> dict:
    """SECONDARY: pooled distributional report (descriptor Wassersteins, FCD).

    The four headline metrics come from :func:`score_seed`, which is FCD-free so
    it can report a seed immediately.  This reduction is for the distributional
    numbers only and defaults to FCD OFF, because FCD is expensive and is not
    part of the published headline table.
    """

    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    from compose_v4.eval.molecular_quality import molecular_quality_report

    RDLogger.DisableLog("rdApp.*")
    volume.reload()
    _assert_pinned_inputs()

    train_smiles = _read_smiles(TRAIN_SMILES)
    reference_smiles = _read_smiles(REFERENCE_SMILES)
    train_canonical = set()
    for smiles in train_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            train_canonical.add(Chem.MolToSmiles(mol))

    per_seed: dict[str, dict] = {}
    all_generated: list[str] = []
    for seed in seeds:
        shard_dir = VOL / "shards" / horizon_tag(horizon)
        shard_paths = sorted(shard_dir.glob(f"seed{seed}_*.json"))
        records: list[dict] = []
        for path in shard_paths:
            records.extend(json.loads(path.read_text())["records"])
        records.sort(key=lambda item: item["index"])
        indices = [item["index"] for item in records]
        if indices != list(range(samples_per_seed)):
            raise RuntimeError(
                f"seed {seed}: expected contiguous 0..{samples_per_seed - 1}, "
                f"got {len(indices)} records"
            )

        raw = [item["smiles"] for item in records]
        attempted = len(raw)
        # A null/unserializable final state is a generation attempt that
        # produced no molecule; it counts against validity, it is not dropped.
        emitted = [text for text in raw if text]
        canonical: list[str] = []
        for text in emitted:
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol))

        unique = set(canonical)
        validity = len(canonical) / attempted
        uniqueness = len(unique) / len(canonical) if canonical else 0.0
        # Novelty over the UNIQUE valid set (GuacaMol / MOSES convention),
        # reported alongside the repo's multiset convention so the two are
        # never silently conflated.
        novel_unique = [smiles for smiles in unique if smiles not in train_canonical]
        novelty_unique = len(novel_unique) / len(unique) if unique else 0.0
        novelty_multiset = (
            float(np.mean([smiles not in train_canonical for smiles in canonical]))
            if canonical
            else 0.0
        )

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mols = [Chem.MolFromSmiles(smiles) for smiles in sorted(unique)]
        fingerprints = [generator.GetFingerprint(mol) for mol in mols if mol is not None]
        pair_total = 0.0
        pair_count = 0
        for position, fingerprint in enumerate(fingerprints):
            rest = fingerprints[position + 1 :]
            if rest:
                sims = DataStructs.BulkTanimotoSimilarity(fingerprint, rest)
                pair_total += float(np.sum(sims))
                pair_count += len(sims)
        # IntDiv1 over distinct pairs (diagonal excluded).
        diversity_offdiag = 1.0 - (pair_total / pair_count) if pair_count else 0.0
        # MOSES IntDiv1 includes the i==j diagonal in the mean.
        n_fp = len(fingerprints)
        diversity_moses = (
            1.0 - ((2.0 * pair_total + n_fp) / (n_fp * n_fp)) if n_fp else 0.0
        )

        per_seed[str(seed)] = {
            "attempted": attempted,
            "emitted_nonempty": len(emitted),
            "valid": len(canonical),
            "unique": len(unique),
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty_unique_set": novelty_unique,
            "novelty_valid_multiset": novelty_multiset,
            "internal_diversity_offdiagonal": diversity_offdiag,
            "internal_diversity_moses_intdiv1": diversity_moses,
            "mean_events": float(np.mean([item["events"] for item in records])),
            "mean_heavy_atoms": float(
                np.mean(
                    [
                        Chem.MolFromSmiles(smiles).GetNumAtoms()
                        for smiles in canonical
                        if Chem.MolFromSmiles(smiles) is not None
                    ]
                )
            )
            if canonical
            else 0.0,
            "budget_exhausted_fraction": float(
                np.mean([item["exhausted_event_budget"] for item in records])
            ),
        }
        per_seed[str(seed)]["novelty_unique_set"] = (
            len(novel_unique) / len(unique) if unique else 0.0
        )
        all_generated.extend(canonical)

    # Distributional report against the matched held-out reference, computed by
    # the repo's own evaluator so it is directly comparable to the historical
    # lineage numbers.
    quality = molecular_quality_report(
        tuple(all_generated),
        reference_smiles=reference_smiles,
        train_smiles=train_smiles[:50000],
        include_fcd=include_fcd,
        fcd_reference_limit=5000,
        fcd_generated_limit=len(all_generated),
        fcd_device="cpu",
    )

    report = {
        "run_label": run_label,
        "protocol": {
            "samples_per_seed": samples_per_seed,
            "seeds": seeds,
            "max_atoms": MAX_ATOMS,
            "operational_horizon": OPERATIONAL_HORIZON,
            "time_step": TIME_STEP,
            "max_events": MAX_EVENTS,
        },
        "versions": _runtime_versions(),
        "input_sha256": _assert_pinned_inputs(),
        "per_seed": per_seed,
        "pooled_quality_report": {
            key: value
            for key, value in quality.items()
            if key not in ("descriptor_distributions",)
        },
        "pooled_descriptor_distributions": quality.get("descriptor_distributions"),
    }
    out = VOL / f"{run_label}_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    volume.commit()
    return report


@app.local_entrypoint()
def probe_entry() -> None:
    print(json.dumps(probe.remote(), indent=2, default=str))


# PREDECLARED sweep grid for the quality-diversity frontier.  ``operational_
# horizon`` is the CTMC's run length: it is the only global exploration
# parameter the unconditional sampler exposes (the ``temperature`` knobs in the
# repo all live in the target-directed controller, which is switched off here).
# These values are fixed before any of them is run, and every point is reported.
SWEEP_HORIZONS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)


def _shard_tasks(
    seed_values: list[int], samples_per_seed: int, shards_per_seed: int, horizon: float
) -> list[dict]:
    if samples_per_seed % shards_per_seed:
        raise ValueError("samples_per_seed must divide evenly into shards_per_seed")
    size = samples_per_seed // shards_per_seed
    return [
        {
            "seed": seed,
            "total": samples_per_seed,
            "start": shard * size,
            "stop": (shard + 1) * size,
            "horizon": horizon,
        }
        for seed in seed_values
        for shard in range(shards_per_seed)
    ]


def _run_points(
    points: list[tuple[float, list[int]]],
    samples_per_seed: int,
    shards_per_seed: int,
    with_novelty: bool,
) -> list[dict]:
    """Sample every (horizon, seed) point, scoring each seed the moment it lands."""

    tasks: list[dict] = []
    for horizon, seed_values in points:
        tasks.extend(
            _shard_tasks(seed_values, samples_per_seed, shards_per_seed, horizon)
        )

    remaining: dict[tuple[float, int], int] = {}
    for task in tasks:
        key = (float(task["horizon"]), int(task["seed"]))
        remaining[key] = remaining.get(key, 0) + 1

    print(
        json.dumps(
            {
                "phase": "launch",
                "shards": len(tasks),
                "points": [[h, s] for h, s in points],
                "samples_per_seed": samples_per_seed,
            }
        ),
        flush=True,
    )

    reports: list[dict] = []
    for done, result in enumerate(sample_shard.map(tasks, order_outputs=False), start=1):
        print(
            json.dumps(
                {"phase": "shard_complete", "done": done, "of": len(tasks), **result}
            ),
            flush=True,
        )
        key = (float(result["horizon"]), int(result["seed"]))
        remaining[key] -= 1
        if remaining[key] == 0:
            # This seed is complete: score and report it now, without waiting
            # for the rest of the grid.
            horizon, seed = key
            metrics = score_seed.remote(
                seed, samples_per_seed, horizon, with_novelty
            )
            reports.append(metrics)
            print(
                json.dumps(
                    {
                        "phase": "SEED_COMPLETE",
                        "horizon": horizon,
                        "seed": seed,
                        "validity": metrics["validity"],
                        "uniqueness": metrics["uniqueness"],
                        "quality": metrics["quality"],
                        "quality_given_valid_unique": metrics[
                            "quality_given_valid_unique"
                        ],
                        "diversity": metrics["diversity"],
                        "mean_heavy_atoms": metrics["mean_heavy_atoms"],
                        "mean_events": metrics["mean_events"],
                    },
                    indent=2,
                    default=str,
                ),
                flush=True,
            )
    return reports


def _write_summary(run_label: str, reports: list[dict]) -> None:
    from collections import defaultdict

    by_horizon: dict[float, list[dict]] = defaultdict(list)
    for entry in reports:
        by_horizon[float(entry["horizon"])].append(entry)

    print(json.dumps({"phase": "summary", "run_label": run_label}), flush=True)
    for horizon in sorted(by_horizon):
        entries = by_horizon[horizon]
        row = {
            "horizon": horizon,
            "seeds": len(entries),
            "validity": [round(float(e["validity"]), 4) for e in entries],
            "uniqueness": [round(float(e["uniqueness"]), 4) for e in entries],
            "quality": [round(float(e["quality"]), 4) for e in entries],
            "diversity": [round(float(e["diversity"]), 4) for e in entries],
        }
        print(json.dumps(row), flush=True)


@app.local_entrypoint()
def main(
    samples_per_seed: int = 1000,
    seeds: int = 3,
    shards_per_seed: int = 20,
    run_label: str = "denovo_official_v1",
    base_seed: int = 20260920,
    horizon: float = OPERATIONAL_HORIZON,
    with_novelty: bool = False,
) -> None:
    """The primary benchmark: N molecules per seed at ONE horizon."""

    seed_values = [base_seed + offset for offset in range(seeds)]
    reports = _run_points(
        [(float(horizon), seed_values)],
        samples_per_seed,
        shards_per_seed,
        with_novelty,
    )
    _write_summary(run_label, reports)


@app.local_entrypoint()
def sample_model(
    model: str = DEFAULT_MODEL,
    samples_per_seed: int = 200,
    seeds: int = 1,
    shards_per_seed: int = 20,
    base_seed: int = 20260920,
    horizon: float = OPERATIONAL_HORIZON,
) -> None:
    """Sample ONE checkpoint of the training-step sweep, matched to the others.

    Sampling only -- no scoring.  The sweep's diagnostic row (published metrics
    plus the ring/size decomposition) is built from the persisted shards by
    ``scripts/denovo_checkpoint_sweep.py``, so the per-molecule data stays
    available for questions posed after the run.

    Every sweep point must use the SAME base_seed, samples_per_seed and horizon;
    only ``model`` changes.  Shards are idempotent and keyed by model, so a
    relaunch after preemption reuses what landed and redoes nothing.
    """

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {sorted(MODELS)}")
    seed_values = [base_seed + offset for offset in range(seeds)]
    tasks = _shard_tasks(seed_values, samples_per_seed, shards_per_seed, float(horizon))
    for task in tasks:
        task["model"] = model

    print(
        json.dumps(
            {
                "phase": "launch",
                "model": model,
                "completed_steps": MODELS[model]["completed_steps"],
                "checkpoint_sha256": MODELS[model]["sha256"],
                "shards": len(tasks),
                "samples_per_seed": samples_per_seed,
                "seeds": seed_values,
                "horizon": float(horizon),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    completed = 0
    for result in sample_shard.map(tasks, order_outputs=False):
        completed += 1
        print(
            json.dumps(
                {"phase": "shard_complete", "done": completed, "of": len(tasks), **result},
                sort_keys=True,
            ),
            flush=True,
        )
    print(json.dumps({"phase": "model_complete", "model": model, "shards": completed}), flush=True)


@app.local_entrypoint()
def sweep(
    samples_per_seed: int = 1000,
    seeds: int = 3,
    shards_per_seed: int = 20,
    run_label: str = "denovo_frontier_v1",
    base_seed: int = 20260920,
    with_novelty: bool = False,
) -> None:
    """The quality-diversity frontier over the PREDECLARED horizon grid."""

    seed_values = [base_seed + offset for offset in range(seeds)]
    points = [(float(h), seed_values) for h in SWEEP_HORIZONS]
    reports = _run_points(points, samples_per_seed, shards_per_seed, with_novelty)
    _write_summary(run_label, reports)
