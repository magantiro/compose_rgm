"""Acceptance measurement for the de-novo ring dependency-block schedule.

The compiler used to emit grafts, then non-ring decoration, then every ring
transaction last, so the ring decision was supervised on a state whose legal
ring-template support had already collapsed.  ``ring_dependency_block`` commits
each ring system at the earliest graft prefix the executor accepts it at.

This measures, on real training molecules drawn uniformly at random from the
recipe's own train partition:

    P_support(3- or 4-ring | at a ring decision point)

under both schedules, with the FULL per-minimum-ring-size histogram beside it,
and it checks endpoint exactness per trace by canonical key AND by array
identity.  Nothing is trained, no checkpoint is written, no oracle exists on
this path.  The ring catalog and the carbon-tree source prior are deserialized
from the checkpoint payload rather than rebuilt, which is strictly stronger
than a fingerprint check.

Arms are matched by construction: both schedules see the same target and the
same source draw, so an arm difference cannot come from a different draw.
"""

from __future__ import annotations

import json
import os
import sys
import time
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

app = modal.App("compose-denovo-ring-dependency-block")
volume = modal.Volume.from_name("compose-denovo-eval-v1", create_if_missing=False)

VOL = Path("/vol")
TRAIN_SMILES = VOL / "guacamol" / "guacamol_subset_500000_seed0.smiles"
CHECKPOINT = VOL / "lineageB" / "checkpoint.best_so_far.pt"
CHECKPOINT_SHA256 = "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c"
TRAIN_SMILES_SHA256 = "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791"

# The recipe Lineage B trained under.
MAX_ATOMS = 40
TRAIN_SIZE = 50_000
VALIDATION_SIZE = 2_000
TEST_SIZE = 2_000
CORPUS_SEED = 20260717
SMALL_RING_MAX = 4

ARMS = ("sequential", "ring_dependency_block")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned() -> dict[str, str]:
    """Refuse to measure against artifacts the model was not trained with."""

    observed = {
        "checkpoint": _sha256(CHECKPOINT),
        "train_smiles": _sha256(TRAIN_SMILES),
    }
    expected = {
        "checkpoint": CHECKPOINT_SHA256,
        "train_smiles": TRAIN_SMILES_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"pinned input mismatch: {observed} != {expected}")
    return observed


def _versions() -> dict[str, str]:
    import networkx
    import numpy
    import rdkit
    import scipy
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "rdkit": rdkit.__version__,
        "torch": torch.__version__,
    }


@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=8, memory=32768)
def select_targets(sample_size: int, seed: int) -> dict:
    """Draw targets uniformly at random from the recipe's own train split.

    Targets are training molecules, not an arbitrary corpus slice: the question
    is about the distribution the ring decision is supervised on.  The split is
    cached on the volume under a key naming the corpus digest AND every split
    parameter, so a cache built under different sizes or seed can never be
    served for this one.
    """

    import numpy as np

    from compose_v4.data.cnof import load_cnof_corpus_split

    _assert_pinned()
    started = time.time()
    key = (
        f"{TRAIN_SMILES_SHA256}_{TRAIN_SIZE}_{VALIDATION_SIZE}_{TEST_SIZE}"
        f"_{MAX_ATOMS}_{CORPUS_SEED}"
    )
    cache_path = VOL / "schedule_probe" / f"train_split_{key}.json"
    reused = cache_path.exists()
    if reused:
        train = tuple(json.loads(cache_path.read_text())["train"])
    else:
        split = load_cnof_corpus_split(
            TRAIN_SMILES,
            train_size=TRAIN_SIZE,
            validation_size=VALIDATION_SIZE,
            test_size=TEST_SIZE,
            max_atoms=MAX_ATOMS,
            seed=CORPUS_SEED,
            scan_all=True,
            workers=8,
        )
        train = tuple(split.train)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"cache_key": key, "train": list(train)}))
        volume.commit()
    rng = np.random.default_rng(seed)
    if sample_size > len(train):
        raise ValueError(f"requested {sample_size} of {len(train)}")
    chosen = sorted(int(i) for i in rng.choice(len(train), sample_size, replace=False))
    return {
        "train_size": len(train),
        "split_cache_reused": reused,
        "split_cache_key": key,
        "sample_size": sample_size,
        "seed": seed,
        "indices": chosen,
        "targets": [train[index] for index in chosen],
        "elapsed_seconds": time.time() - started,
    }


# Preemption is an expected event, not a code failure.  Shards are small and
# idempotent, so a retry redoes at most one short stride rather than a long run.
@app.function(
    image=image,
    volumes={str(VOL): volume},
    timeout=5400,
    cpu=2,
    memory=8192,
    retries=modal.Retries(max_retries=3),
)
def acceptance_shard(payload: dict) -> dict:
    """Compile one stride of molecules under both schedules and measure support."""

    import numpy as np
    import torch

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.eval.denovo_schedule_probe import (
        ring_size_support_histogram,
        small_ring_category_mask,
        small_ring_support_mass,
    )
    from compose_v4.rewrite.commuting_schedule import states_are_array_exact
    from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
    from compose_v4.rewrite.trace import execute_trace
    from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    torch.set_num_threads(1)
    _assert_pinned()
    model, checkpoint = load_factorized_rollout_checkpoint(str(CHECKPOINT))
    prior = checkpoint["tree_source_prior"]
    templates = model.ring_system_templates
    category = small_ring_category_mask(templates, maximum_size=SMALL_RING_MAX)
    runtime = de_novo_rewrite_system()

    started = time.time()
    rows: list[dict] = []
    support_evaluations = 0
    for index, smiles in zip(payload["indices"], payload["targets"]):
        row: dict = {"index": index, "smiles": smiles, "arms": {}}
        try:
            target = pad_molecular_graph(smiles_to_molecular_graph(smiles), MAX_ATOMS)
        except (ValueError, RuntimeError, KeyError) as error:
            row["target_parse_error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            continue
        row["heavy_atoms"] = int(target.n_real_atoms)
        target_key = canonical_state_key(target)
        source = prior.sample(
            np.random.default_rng(CORPUS_SEED + index), n_slots=MAX_ATOMS
        )
        for arm in ARMS:
            entry: dict = {}
            try:
                trace = compile_carbon_tree_to_target(
                    source,
                    target,
                    system=runtime,
                    use_bond_reroute=True,
                    align_source=False,
                    flexible_size=True,
                    typed_ring_payloads=True,
                    ring_catalog=None,
                    event_schedule=arm,
                )
            except Exception as error:  # noqa: BLE001 - recorded, never dropped
                entry["compile_error"] = f"{type(error).__name__}: {error}"
                row["arms"][arm] = entry
                continue
            endpoint, states = execute_trace(
                trace.source, trace.steps, system=runtime, return_states=True
            )
            entry["trace_length"] = len(trace.steps)
            entry["endpoint_canonical_key_matches"] = bool(
                canonical_state_key(endpoint) == target_key
            )
            entry["endpoint_array_exact"] = bool(
                states_are_array_exact(endpoint, target)
            )
            for key in (
                "ring_dependency_block_commits",
                "ring_dependency_block_deferred",
                "ring_dependency_block_fell_back_to_sequential",
            ):
                entry[key] = trace.metadata.get(key)
            events = []
            for position, step in enumerate(trace.steps):
                if step.rule_name != "ring_system_grow":
                    continue
                state = states[position]
                support = model._ring_grow_support(state)
                support_evaluations += 1
                mass, legal, legal_small = small_ring_support_mass(support, category)
                events.append(
                    {
                        "index": position,
                        "fraction_of_trace": position / len(trace.steps),
                        "small_mass_uniform_support": mass,
                        "legal_template_count": legal,
                        "legal_small_template_count": legal_small,
                        "size_histogram": ring_size_support_histogram(
                            support, templates
                        ),
                        "heavy_atoms_at_event": int(state.n_real_atoms),
                    }
                )
            entry["ring_events"] = events
            row["arms"][arm] = entry
        rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "molecule",
                    "shard": payload.get("shard"),
                    "done": len(rows),
                    "of": len(payload["targets"]),
                    "elapsed": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
    return {
        "shard": payload.get("shard"),
        "rows": rows,
        "support_evaluations": support_evaluations,
        "elapsed_seconds": time.time() - started,
        "versions": _versions(),
        "catalog_template_count": len(templates),
        "catalog_small_template_count": int(category.sum()),
    }
