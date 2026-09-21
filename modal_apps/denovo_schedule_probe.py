"""Zero-training probe: is the exact ring-rescheduling repair live on real traces?

Three schedule arms over the SAME real de-novo training traces, measuring whether
moving ring transactions earlier repairs the degenerate ring-formation support:

    A  sequential                        the schedule Lineage B actually trained on
    B  exact_early_ring, shipped barriers atom_insert/atom_delete are barriers
    C  exact_early_ring, barrier relaxed  same exactness test, no rule-name barrier

Nothing is trained, no checkpoint is written, and no oracle exists on this path.
The checkpoint is opened read-only for two things the measurement cannot invent:
the typed ring catalog and the carbon-tree source prior, both deserialized from
the payload rather than rebuilt -- which is strictly stronger than a fingerprint
check, since they are byte-identical to the ones training used.

The instrument check is separate and comes first: it recomputes the committed
reference audit's own rows from its own stored states and requires exact
agreement.  That isolates "does this enumeration agree with the audit's" from
"are training-trace states distributed like rollout states", which are different
questions and must not be allowed to explain each other away.
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

# Pinned to the checkpoint manifest's recorded training runtime.
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

app = modal.App("compose-denovo-schedule-probe")
volume = modal.Volume.from_name("compose-denovo-eval-v1", create_if_missing=False)

VOL = Path("/vol")
TRAIN_SMILES = VOL / "guacamol" / "guacamol_subset_500000_seed0.smiles"

MODELS: dict[str, dict[str, str]] = {
    "step1000": {
        "relative_path": "lineageB/checkpoint.best_so_far.pt",
        "sha256": "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c",
    },
    "step2500": {
        "relative_path": "lineageB_step2500/checkpoint.best_so_far.pt",
        "sha256": "bb53e00237bd4be60436e149c7f92cd75b30ce555ae96d9264164e303150abb1",
    },
}
TRAIN_SMILES_SHA256 = "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791"

# The recipe Lineage B trained under (recipes/tree_fcd_transfer_stage3_flexible_graft.json).
MAX_ATOMS = 40
TRAIN_SIZE = 50_000
VALIDATION_SIZE = 2_000
TEST_SIZE = 2_000
CORPUS_SEED = 20260717
TRANSPORT_MODE = "flexible_size_graft"
SMALL_RING_MAX = 4


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned(model_label: str) -> dict[str, str]:
    """Refuse to measure against artifacts the model was not trained with."""

    checkpoint = VOL / MODELS[model_label]["relative_path"]
    observed = {
        "checkpoint": _sha256(checkpoint),
        "train_smiles": _sha256(TRAIN_SMILES),
    }
    expected = {
        "checkpoint": MODELS[model_label]["sha256"],
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
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "torch": torch.__version__,
    }


def _load_model(model_label: str):
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    checkpoint = VOL / MODELS[model_label]["relative_path"]
    return load_factorized_rollout_checkpoint(str(checkpoint))


# ---- Stage 1: instrument check ----


@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=4, memory=16384)
def instrument_check(reference_rows: list[dict], model_label: str = "step2500") -> dict:
    """Recompute the committed reference audit's own rows and demand agreement.

    The reference states come from the audit artifact, so this compares two
    enumerations of the SAME states.  Any disagreement is an instrument fault and
    must be reconciled before any arm result is reported.
    """

    import torch

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.eval.denovo_schedule_probe import (
        small_ring_category_mask,
        small_ring_support_mass,
    )

    torch.set_num_threads(1)
    shas = _assert_pinned(model_label)
    model, _payload = _load_model(model_label)
    category = small_ring_category_mask(
        model.ring_system_templates, maximum_size=SMALL_RING_MAX
    )

    started = time.time()
    rows: list[dict] = []
    support_evaluations = 0
    for reference in reference_rows:
        state = pad_molecular_graph(
            smiles_to_molecular_graph(reference["before"]), MAX_ATOMS
        )
        support = model._ring_grow_support(state)
        support_evaluations += 1
        mass, legal, legal_small = small_ring_support_mass(support, category)
        rows.append(
            {
                "row": reference["row"],
                "expected_legal_template_count": reference["legal_template_count"],
                "observed_legal_template_count": legal,
                "expected_legal_small_template_count": reference[
                    "legal_small_template_count"
                ],
                "observed_legal_small_template_count": legal_small,
                "expected_small_mass_uniform_support": reference[
                    "small_mass_uniform_support"
                ],
                "observed_small_mass_uniform_support": mass,
                "agrees": (
                    legal == reference["legal_template_count"]
                    and legal_small == reference["legal_small_template_count"]
                    and abs(mass - reference["small_mass_uniform_support"]) < 1e-12
                ),
            }
        )
        print(json.dumps({"phase": "instrument_row", **rows[-1]}), flush=True)

    return {
        "model_label": model_label,
        "input_sha256": shas,
        "versions": _versions(),
        "catalog_template_count": len(model.ring_system_templates),
        "catalog_small_template_count": int(category.sum()),
        "rows": rows,
        "rows_agreeing": sum(1 for row in rows if row["agrees"]),
        "rows_total": len(rows),
        "support_evaluations": support_evaluations,
        "elapsed_seconds": time.time() - started,
    }


@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=4, memory=16384)
def catalog_identity(model_label: str) -> dict:
    """Fingerprint one checkpoint's ring catalog so arms and instrument agree.

    The instrument check replicates rows produced under step-2500 while the arms
    are measured on the Lineage B (step-1000) catalog.  That transfer is only
    legitimate if the two catalogs are the same object, so it is verified rather
    than assumed.
    """

    import hashlib

    from compose_v4.eval.denovo_schedule_probe import small_ring_category_mask

    _assert_pinned(model_label)
    model, payload = _load_model(model_label)
    category = small_ring_category_mask(
        model.ring_system_templates, maximum_size=SMALL_RING_MAX
    )
    digest = hashlib.sha256()
    for template in model.ring_system_templates:
        digest.update(repr(template).encode())
    return {
        "model_label": model_label,
        "template_count": len(model.ring_system_templates),
        "small_template_count": int(category.sum()),
        "template_repr_sha256": digest.hexdigest(),
        "source_prior_class": type(payload["tree_source_prior"]).__name__,
    }


# ---- Stage 2: target selection ----


@app.function(image=image, volumes={str(VOL): volume}, timeout=3600, cpu=8, memory=32768)
def select_targets(sample_size: int, seed: int) -> dict:
    """Draw probe targets from the SAME train split the recipe defines.

    Targets are training molecules, not an arbitrary corpus slice, because the
    question is about the distribution the ring decision is supervised on.
    """

    import numpy as np

    from compose_v4.data.cnof import load_cnof_corpus_split

    _assert_pinned("step1000")
    started = time.time()
    # The split is a pure function of the corpus bytes and the recipe's split
    # parameters, so it is cached on the volume under a key naming BOTH.  A cache
    # that keyed on the corpus alone could serve a split built under different
    # sizes or seed, which is the derived-artifact-cache failure this repo has
    # already paid for once.
    cache_key = (
        f"{TRAIN_SMILES_SHA256}_{TRAIN_SIZE}_{VALIDATION_SIZE}_{TEST_SIZE}"
        f"_{MAX_ATOMS}_{CORPUS_SEED}"
    )
    cache_path = VOL / "schedule_probe" / f"train_split_{cache_key}.json"
    reused = False
    if cache_path.exists():
        train = tuple(json.loads(cache_path.read_text())["train"])
        reused = True
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
        cache_path.write_text(json.dumps({"cache_key": cache_key, "train": list(train)}))
        volume.commit()
    rng = np.random.default_rng(seed)
    if sample_size > len(train):
        raise ValueError(f"requested {sample_size} targets from {len(train)}")
    chosen = rng.choice(len(train), size=sample_size, replace=False)
    return {
        "train_size": len(train),
        "split_cache_reused": reused,
        "split_cache_key": cache_key,
        "sample_size": sample_size,
        "seed": seed,
        "targets": [train[int(index)] for index in sorted(chosen)],
        "elapsed_seconds": time.time() - started,
    }


# ---- Stage 3: the three arms ----


@app.function(image=image, volumes={str(VOL): volume}, timeout=5400, cpu=2, memory=16384)
def probe_shard(payload_in: dict) -> dict:
    """Compile one stride of targets and measure all three arms on each.

    Arms are matched by construction: the base trace is compiled ONCE and the two
    scheduled arms are derived from that same trace, so an arm difference cannot
    come from a different source draw.
    """

    import numpy as np
    import torch

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.eval.denovo_schedule_probe import (
        ARM_A,
        ARM_BARRIERS,
        RING_GROW_RULE,
        apply_schedule_arm,
        observe_ring_events,
        ring_event_blockers,
        ring_event_indices,
        small_ring_category_mask,
        small_ring_support_mass,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.ring_system_fiber import structured_ring_trace_supported
    from compose_v4.rewrite.tracelet_compiler import TraceCompilationError
    from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

    torch.set_num_threads(1)
    targets: list[str] = payload_in["targets"]
    offsets: list[int] = payload_in["offsets"]
    seed = int(payload_in["seed"])
    model_label = payload_in["model_label"]

    _assert_pinned(model_label)
    model, checkpoint_payload = _load_model(model_label)
    catalog = checkpoint_payload["ring_catalog"]
    prior = checkpoint_payload["tree_source_prior"]
    category = small_ring_category_mask(
        model.ring_system_templates, maximum_size=SMALL_RING_MAX
    )
    system = de_novo_rewrite_system()

    started = time.time()
    events: list[dict] = []
    arm_counters: dict[str, dict[str, int]] = {
        name: {
            "traces": 0,
            "dropped": 0,
            "attempted_swaps": 0,
            "accepted_swaps": 0,
            "position_report_disagreements": 0,
        }
        for name in (ARM_A, *ARM_BARRIERS)
    }
    drop_reasons: dict[str, list[str]] = {name: [] for name in ARM_BARRIERS}
    compiled = 0
    compile_failures = 0
    trainer_supported_traces = 0
    blockers: list[dict] = []
    support_evaluations = 0
    support_cache: dict[str, tuple[float, int, int]] = {}

    for offset, smiles in zip(offsets, targets):
        try:
            target = pad_molecular_graph(smiles_to_molecular_graph(smiles), MAX_ATOMS)
        except (ValueError, RuntimeError, KeyError) as error:
            compile_failures += 1
            print(json.dumps({"phase": "target_parse_failed", "offset": offset,
                              "error": f"{type(error).__name__}: {error}"}), flush=True)
            continue
        # One independent source draw per target, matching the prior the trainer
        # samples from; seeding per target keeps every arm on the same draw and
        # makes the whole shard reproducible independently of stride order.
        source = prior.sample(np.random.default_rng(seed + offset), n_slots=MAX_ATOMS)
        try:
            # The trainer compiles its raw tree-transport records with
            # ring_catalog=None and only afterwards KEEPS the records the catalog
            # supports (train_tracelet_cnof_gate.py calls
            # build_tree_transport_path_records with ring_catalog=None, then
            # filters by structured_ring_trace_supported).  Passing the catalog
            # INTO the compiler is a different and far stricter predicate: it
            # refused 12 of 12 real training molecules in the smoke.  Reproduce
            # the trainer's order, then record its keep filter separately.
            base = compile_carbon_tree_to_target(
                source,
                target,
                use_bond_reroute=True,
                align_source=False,
                flexible_size=True,
                typed_ring_payloads=True,
                ring_catalog=None,
                system=system,
            )
        except (TraceCompilationError, ValueError, RuntimeError) as error:
            compile_failures += 1
            print(
                json.dumps({"phase": "compile_failed", "offset": offset,
                            "error": f"{type(error).__name__}: {error}"}),
                flush=True,
            )
            continue
        compiled += 1
        trainer_supported = bool(structured_ring_trace_supported(base, catalog))
        if trainer_supported:
            trainer_supported_traces += 1

        arm_traces = {ARM_A: base}
        for arm, barriers in ARM_BARRIERS.items():
            result = apply_schedule_arm(
                base,
                arm=arm,
                phase_barrier_rule_names=barriers,
                system=system,
            )
            arm_counters[arm]["attempted_swaps"] += result.attempted_swaps
            arm_counters[arm]["accepted_swaps"] += result.accepted_swaps
            if result.dropped or result.trace is None:
                arm_counters[arm]["dropped"] += 1
                if result.drop_reason:
                    drop_reasons[arm].append(result.drop_reason)
                continue
            # Cross-check the scheduler's self-report against an independent
            # recomputation from the returned trace.  A disagreement is recorded,
            # never reconciled in favour of the report.
            if tuple(result.reported_positions_after) != ring_event_indices(
                result.trace
            ):
                arm_counters[arm]["position_report_disagreements"] += 1
            arm_traces[arm] = result.trace

        for arm, trace in arm_traces.items():
            arm_counters[arm]["traces"] += 1
            # What actually stops each ring event moving further left. The
            # scheduler breaks WITHOUT counting an attempt when it meets a
            # barrier, so its own counters cannot distinguish "hit the shipped
            # barrier" from "the exactness test refused" -- and those imply
            # opposite recommendations.
            for blocker in ring_event_blockers(
                trace,
                barrier_rule_names=ARM_BARRIERS[
                    "B_exact_early_ring_default_barriers"
                ],
                system=system,
            ):
                blockers.append({"arm": arm, "offset": offset, **blocker})
            for observation in observe_ring_events(trace, system=system):
                mass = legal = legal_small = None
                if observation.rule_name == RING_GROW_RULE:
                    state_key = _state_key(observation.state)
                    cached = support_cache.get(state_key)
                    if cached is None:
                        support = model._ring_grow_support(observation.state)
                        support_evaluations += 1
                        cached = small_ring_support_mass(support, category)
                        support_cache[state_key] = cached
                    mass, legal, legal_small = cached
                events.append(
                    {
                        "arm": arm,
                        "offset": offset,
                        "trainer_supported": trainer_supported,
                        "rule_name": observation.rule_name,
                        "index": observation.index,
                        "trace_length": observation.trace_length,
                        "fraction_of_trace": observation.fraction_of_trace,
                        "free_slots": observation.free_slots,
                        "occupied_free_slots": observation.occupied_free_slots,
                        "small_mass_uniform_support": mass,
                        "legal_template_count": legal,
                        "legal_small_template_count": legal_small,
                    }
                )
        if compiled % 5 == 0:
            print(
                json.dumps({"phase": "progress", "compiled": compiled,
                            "events": len(events),
                            "support_evaluations": support_evaluations,
                            "elapsed": round(time.time() - started, 1)}),
                flush=True,
            )

    return {
        "shard": payload_in.get("shard"),
        "model_label": model_label,
        "targets_requested": len(targets),
        "compiled": compiled,
        "compile_failures": compile_failures,
        "trainer_supported_traces": trainer_supported_traces,
        "arm_counters": arm_counters,
        "drop_reasons": {arm: sorted(set(reasons))[:5] for arm, reasons in drop_reasons.items()},
        "events": events,
        "blockers": blockers,
        "support_evaluations": support_evaluations,
        "support_cache_size": len(support_cache),
        "elapsed_seconds": time.time() - started,
    }


def _state_key(state) -> str:
    """Content key for one padded state, used only to avoid repeat support work."""

    import hashlib

    digest = hashlib.sha256()
    for array in (
        state.atom_types,
        state.formal_charges,
        state.implicit_h_counts,
        state.bonds,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()
