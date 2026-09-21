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
from pathlib import Path
import sys

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


def _assert_pinned_inputs() -> dict[str, str]:
    """Fail closed if any input is not the artifact the model was trained with."""

    observed = {
        "checkpoint": _sha256(CHECKPOINT),
        "train": _sha256(TRAIN_SMILES),
        "reference": _sha256(REFERENCE_SMILES),
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
    import numpy
    import rdkit
    import scipy
    import torch
    import networkx

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
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    import torch

    torch.set_num_threads(1)

    seed = int(task["seed"])
    total = int(task["total"])
    start = int(task["start"])
    stop = int(task["stop"])
    out_path = VOL / "shards" / f"seed{seed}_{start:06d}_{stop:06d}.json"
    if out_path.exists():
        volume.reload()
        return {"shard": out_path.name, "reused": True}

    _assert_pinned_inputs()
    model, payload = load_factorized_rollout_checkpoint(str(CHECKPOINT))
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
            operational_horizon=OPERATIONAL_HORIZON,
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
    return {"shard": out_path.name, "n": len(records), "reused": False}


# ---- scoring ----------------------------------------------------------------


@app.function(image=image, volumes={str(VOL): volume}, timeout=7200, cpu=8, memory=16384)
def score(run_label: str, samples_per_seed: int, seeds: list[int]) -> dict:
    """Reduce the shards and compute per-seed unconditional metrics."""

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
        shard_paths = sorted((VOL / "shards").glob(f"seed{seed}_*.json"))
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
        include_fcd=True,
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


@app.local_entrypoint()
def main(
    samples_per_seed: int = 1000,
    seeds: int = 3,
    shards_per_seed: int = 20,
    run_label: str = "denovo_official_v1",
    base_seed: int = 20260920,
) -> None:
    seed_values = [base_seed + offset for offset in range(seeds)]
    if samples_per_seed % shards_per_seed:
        raise ValueError("samples_per_seed must divide evenly into shards_per_seed")
    size = samples_per_seed // shards_per_seed
    tasks = [
        {
            "seed": seed,
            "total": samples_per_seed,
            "start": shard * size,
            "stop": (shard + 1) * size,
        }
        for seed in seed_values
        for shard in range(shards_per_seed)
    ]
    print(json.dumps({"phase": "launch", "shards": len(tasks), "seeds": seed_values}))
    done = 0
    for result in sample_shard.map(tasks, order_outputs=False):
        done += 1
        print(json.dumps({"phase": "shard_complete", "done": done, "of": len(tasks), **result}))
    report = score.remote(run_label, samples_per_seed, seed_values)
    print(json.dumps({"phase": "scored", "per_seed": report["per_seed"]}, indent=2, default=str))
