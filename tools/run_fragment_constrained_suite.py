#!/usr/bin/env python3
"""Run the public fragment-constrained benchmark under the OFFICIAL protocol.

Protocol, taken from the released InVirtuoGen evaluator rather than from a paper
description (``in_virtuo_gen/evaluation/downstream.py``):

* 100 generations per drug per task (``--num_samples_eval``, forced to 100).
* Metrics per drug from ``train_utils.metrics.evaluate_smiles``.
* A task row is the UNWEIGHTED MEAN over that task's 10 drugs.
* Reported as mean +/- standard deviation over ``--num_seeds`` (default 3) runs.
* ``scaffold_morphing`` reuses the linker prompts and the linker results, which
  is what the executable upstream code does.

COMPOSE emits the failure placeholder for an attempt that produced nothing
admissible, so the official function counts it invalid.  Official validity is
therefore COMPOSE's STRICT benchmark validity: chemically valid, connected, AND
prompt-satisfying.  Chemical validity of committed endpoints is recorded
separately, because committed states are valid by construction and that is a
different claim from satisfying the prompt.

Zero oracle calls.  QED and SA are the benchmark's own quality diagnostic and are
computed inside the official evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_attachment_control import (
    SINGLE_INTERFACE_RELEASE,
    AttachmentControlConfig,
    AttachmentController,
    AttachmentSpec,
    interface_coverage_report,
)
from compose_v4.benchmark.fragment_conditioned_sampler import (
    FragmentConditioningError,
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_distance,
    official_prompt_metrics,
    official_unique_valid,
)

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
OFFICIAL_SAMPLES_PER_PROMPT = 100
OFFICIAL_SEEDS = 3


def prompt_rng_seed(drug_name: str, task_value: str, seed: int) -> int:
    """A REPRODUCIBLE per-(drug, task, seed) RNG seed.

    ``hash()`` on a str is salted by ``PYTHONHASHSEED``, so a seed derived from
    it differs in every process and no run using it can be reproduced -- not
    even by re-running the identical command.  BLAKE2b is stable across
    processes, machines and interpreter versions.
    """
    key = f"{drug_name}|{task_value}|{seed}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)


def frozen_attachment_identity(control: AttachmentControlConfig) -> dict:
    """The attachment controller's configuration and its hash, frozen before the sweep.

    ONE controller configuration is used for every drug and every task.  The
    hash is over the canonical JSON of the dataclass, so any per-drug or
    per-task adjustment would move it and be visible in every shard; the
    aggregator refuses to combine shards whose hashes disagree.
    """
    payload = {k: v for k, v in sorted(asdict(control).items())}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "config": payload,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "arm": "attachment_control" if control.enabled else "frozen_sampler_baseline",
    }


def frozen_sampler_identity(config: SamplerConfig) -> dict:
    """The sampler configuration and its hash, recorded BEFORE the sweep.

    One configuration is used for every task, drug and seed.  The hash is over
    the canonical JSON of the dataclass, so any per-instance tuning would move
    it and be visible in the artifact.
    """
    payload = {k: v for k, v in sorted(asdict(config).items())}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "config": payload,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


# ---- Independent fragment-preservation audit ----
#
# This does NOT reuse ``check_fragment_constraint``.  That function decides
# which samples are emitted, so scoring the emitted samples with it again would
# be a comparison whose expectation is recomputed from the code under test and
# could not fail.  Here the query is rebuilt from the prompt string by a
# different route and matched independently, and it is run over the COMMITTED
# endpoints -- every state COMPOSE actually produced, before any filtering --
# which is what separates "preserved by the pathwise lock" from "preserved by
# discarding the failures".

_DUMMY = Chem.MolFromSmarts("[#0]")


def audit_queries(prompt) -> list[Chem.Mol]:
    """Dummy-stripped, independently constructed substructure queries."""
    queries = []
    for fragment in prompt.fragments:
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            raise ValueError(f"audit could not parse fragment {fragment!r}")
        stripped = Chem.DeleteSubstructs(mol, _DUMMY)
        stripped.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(stripped)
        queries.append(stripped)
    return queries


def contains_all_fragments(smiles: str, queries: list[Chem.Mol]) -> bool:
    """True when every query embeds, pairwise NON-OVERLAPPING, in ``smiles``."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return False
    per_query = [
        mol.GetSubstructMatches(q, uniquify=False, useChirality=False, maxMatches=1000)
        for q in queries
    ]
    if any(not matches for matches in per_query):
        return False

    def bind(index: int, used: frozenset) -> bool:
        if index == len(per_query):
            return True
        for match in per_query[index]:
            atoms = frozenset(match)
            if atoms.isdisjoint(used) and bind(index + 1, used | atoms):
                return True
        return False

    return bind(0, frozenset())


def prompt_reference_smiles(prompt) -> str:
    """The dummy-stripped prompt, as the reference MOLECULE for ``distance``.

    This must be a SMILES RDKit can re-parse, which the audit query need not be.
    Deleting a dummy from an attachment-bearing aromatic nitrogen leaves an
    ``n`` with neither a substituent nor a hydrogen, and the resulting string
    does not parse -- upstream's ``calculate_average_tanimoto`` raises
    ``Invalid prompt SMILES string`` on it.  Capping the dummy with HYDROGEN
    instead is both parseable and what the retained core physically is, which is
    the same resolution ``retained_core`` already uses to build the start state.

    For a fragment whose attachment point is not pathological the two routes
    agree exactly, because deleting a dummy leaves an implicit hydrogen in the
    freed valence anyway; ``tests/test_fragment_official_suite.py`` pins that
    agreement across every released prompt so this cannot silently move a
    reported distance.
    """
    parts = []
    for fragment in prompt.fragments:
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            raise ValueError(f"unparseable fragment {fragment!r}")
        editable = Chem.RWMol(mol)
        for atom in editable.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetIsotope(0)
                atom.SetNoImplicit(False)
                atom.SetFormalCharge(0)
        core = editable.GetMol()
        Chem.SanitizeMol(core)
        core = Chem.RemoveHs(core)
        parts.append(Chem.MolToSmiles(core, canonical=True))
    return ".".join(x for x in parts if x)


def _kernel_provenance() -> dict[str, str]:
    import rdkit
    import torch

    return {
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run_task(
    model,
    system,
    prompts,
    task: FragmentTask,
    *,
    seeds: int,
    samples: int,
    config: SamplerConfig,
    control: AttachmentControlConfig,
    verbose: bool = True,
    seed_list: list[int] | None = None,
    drugs: list[str] | None = None,
    linker_bridge_atoms: int = 0,
) -> dict:
    task_prompts = [p for p in prompts if p.task is task]
    if drugs:
        wanted = {d.upper() for d in drugs}
        task_prompts = [p for p in task_prompts if p.drug_name.upper() in wanted]
    per_seed_rows: list[dict[str, float]] = []
    per_drug_detail: dict[str, list[dict]] = defaultdict(list)
    build_failures: list[dict[str, str]] = []

    for seed in (seed_list if seed_list is not None else range(seeds)):
        drug_metrics: list[dict[str, float]] = []
        for prompt in task_prompts:
            try:
                context = build_prompt_context(
                    prompt,
                    config=config,
                    control=control,
                    linker_bridge_atoms=linker_bridge_atoms,
                )
            except FragmentConditioningError as exc:
                build_failures.append(
                    {"drug": prompt.drug_name, "seed": seed, "error": str(exc)}
                )
                continue

            # Measurement only: the same function the program uses as its
            # predicate, so the number that judges the run and the number the
            # run steers by cannot disagree.
            controller_for_report = AttachmentController(
                context.attachment
                or AttachmentSpec((), (), (frozenset(context.locked_slots),)),
                context.locked_slots,
                control,
            )

            rng_seed = prompt_rng_seed(prompt.drug_name, task.value, seed)
            rng = np.random.default_rng(rng_seed)
            receipt = SamplingReceipt()
            emitted: list[str] = []
            attempt_records: list[dict] = []
            started = time.time()
            refusal_fields = (
                "lock_rejections", "executor_refusals", "budget_exhausted",
                "constraint_failures", "interface_rejections", "staging_rejections",
                "redirections", "separation_failures",
            )
            for attempt_index in range(samples):
                before_committed = len(receipt.committed_endpoints)
                before_refusals = {
                    field: getattr(receipt, field) for field in refusal_fields
                }
                before_families = dict(receipt.families)
                out = sample_completion(
                    model,
                    system,
                    context,
                    rng,
                    config=config,
                    receipt=receipt,
                    control=control,
                )
                emitted.append(out if out else FAILED_SAMPLE_PLACEHOLDER)
                new_committed = receipt.committed_endpoints[before_committed:]
                if len(new_committed) > 1:
                    raise AssertionError("one attempt committed more than one endpoint")
                attempt_records.append(
                    {
                        "attempt_index": attempt_index,
                        "committed_smiles": new_committed[0] if new_committed else None,
                        "emitted_smiles": out,
                        "events": receipt.events[-1],
                        "refusal_deltas": {
                            field: getattr(receipt, field) - before_refusals[field]
                            for field in refusal_fields
                        },
                        "family_deltas": {
                            family: count - before_families.get(family, 0)
                            for family, count in receipt.families.items()
                            if count > before_families.get(family, 0)
                        },
                    }
                )
            elapsed = time.time() - started

            metrics = official_prompt_metrics(emitted, expected_samples=samples)

            queries = audit_queries(prompt)
            reference = prompt_reference_smiles(prompt)
            committed = receipt.committed_endpoints
            committed_preserving = sum(
                contains_all_fragments(s, queries) for s in committed
            )
            emitted_real = [s for s in emitted if s]
            emitted_preserving = sum(
                contains_all_fragments(s, queries) for s in emitted_real
            )
            metrics["distance"] = official_distance(emitted, reference)
            metrics["distance_to_original_drug"] = official_distance(
                emitted, prompt.original_smiles
            )
            drug_metrics.append(metrics)
            per_drug_detail[prompt.drug_name].append(
                {
                    "seed": seed,
                    "rng_seed": rng_seed,
                    "official": metrics,
                    "attempts": samples,
                    "validity_denominator": samples,
                    "emitted_nonempty": len(emitted_real),
                    "distance_reference_prompt": reference,
                    "unique_valid_count": len(official_unique_valid(emitted)),
                    "committed_endpoints": len(committed),
                    "committed_chemically_valid": _chemically_valid(committed),
                    "committed_fragment_preserving": committed_preserving,
                    "emitted_fragment_preserving": emitted_preserving,
                    "constraint_failures": receipt.constraint_failures,
                    "lock_rejections": receipt.lock_rejections,
                    "budget_exhausted": receipt.budget_exhausted,
                    "executor_refusals": receipt.executor_refusals,
                    "mean_events": float(np.mean(receipt.events)) if receipt.events else 0.0,
                    "families": dict(receipt.families),
                    "seconds": round(elapsed, 2),
                    # THE MOLECULES, not a sample of them.  Secondary metrics
                    # (uniqueness / quality / diversity / distance) were
                    # previously uncorrectable because only counters and five
                    # example SMILES survived, while the task filter censored
                    # 51% of motif and 96% of decoration endpoints.  Both lists
                    # are retained so any of those metrics can be re-scored
                    # over EITHER denominator without re-running the sampler.
                    "committed_endpoint_smiles": list(committed),
                    "emitted_samples": list(emitted),
                    # Attempt-aligned provenance, including a null committed
                    # endpoint for a genuine no-output trajectory.  The two
                    # legacy lists above remain for backward compatibility.
                    "attempt_records": attempt_records,
                    # ---- Two-interface path accounting ----
                    #
                    # A linker row is only ABOUT designed linkers if the
                    # realized-length distribution is beside it, and that
                    # distribution is undefined without the length the
                    # construction SEEDED. Both are recorded here, per committed
                    # endpoint rather than as a summary: a run that stores
                    # aggregates cannot answer a question posed after the fact,
                    # and that has already cost this workstream a set of
                    # secondary metrics permanently. All zero or empty for a
                    # single-core prompt and with the program off, except
                    # realized_linker_lengths, which is measured either way and
                    # is exactly what makes the two arms comparable.
                    "seeded_linker_length": (
                        controller_for_report.realized_linker_length(
                            context.start_state
                        )
                        if context.attachment
                        else None
                    ),
                    "realized_linker_lengths": list(receipt.linker_lengths),
                    "path_targets": list(receipt.path_targets),
                    "path_transactions": receipt.path_transactions,
                    "path_transaction_refusals": receipt.path_transaction_refusals,
                    "path_rejections": receipt.path_rejections,
                    "interface_rejections": receipt.interface_rejections,
                    "staging_rejections": receipt.staging_rejections,
                    "redirections": receipt.redirections,
                    "separation_failures": receipt.separation_failures,
                    "committed_interfaces_covered": receipt.interface_covered,
                    "declared_interfaces": list(
                        context.attachment.interfaces if context.attachment else ()
                    ),
                    "attachment_spec": (
                        context.attachment.identity_payload()
                        if context.attachment
                        else None
                    ),
                    "start_state_interface_coverage": (
                        interface_coverage_report(
                            context.start_state,
                            context.attachment,
                            context.locked_slots,
                        )
                        if context.attachment
                        else None
                    ),
                }
            )
            if verbose:
                print(
                    f"  [{task.value} seed={seed}] {prompt.drug_name:14s} "
                    f"val={metrics['validity']:6.2f} uniq={metrics['uniqueness']:6.2f} "
                    f"qual={metrics['quality']:6.2f} div={metrics['diversity']:.3f} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

        if drug_metrics:
            per_seed_rows.append(
                {
                    key: float(np.nanmean([m[key] for m in drug_metrics]))
                    for key in _ROW_KEYS
                }
            )

    summary = {
        key: {
            "mean": float(np.mean([r[key] for r in per_seed_rows])),
            "std": float(np.std([r[key] for r in per_seed_rows])),
        }
        for key in _ROW_KEYS
    } if per_seed_rows else {}

    return {
        "task": task.value,
        "prompts_declared": len(task_prompts),
        "prompts_scored": len(task_prompts) - len({f["drug"] for f in build_failures}),
        "samples_per_prompt": samples,
        "seeds": seeds,
        "per_seed": per_seed_rows,
        "summary": summary,
        "build_failures": build_failures,
        "per_drug": {k: v for k, v in per_drug_detail.items()},
    }


_ROW_KEYS = (
    "validity",
    "uniqueness",
    "quality",
    "diversity",
    "distance",
    "distance_to_original_drug",
)


def _chemically_valid(smiles_list) -> int:
    from rdkit import Chem

    total = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is not None and len(Chem.GetMolFrags(mol)) == 1:
            total += 1
    return total


def build_parser() -> argparse.ArgumentParser:
    """The suite's argument parser, factored out so the wiring is testable.

    A knob that is parsed but never reaches SamplerConfig is inert, and inert
    knobs are invisible to every check that reads the recorded configuration --
    the value is written into the artifact either way.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--task", action="append", default=None, help="repeatable; default = all"
    )
    parser.add_argument("--seeds", type=int, default=OFFICIAL_SEEDS)
    parser.add_argument("--seed-list", type=int, action="append", default=None,
                        help="explicit seed ids for sharding; default = range(--seeds)")
    parser.add_argument("--drug", action="append", default=None,
                        help="repeatable drug filter for sharding")
    parser.add_argument("--samples", type=int, default=OFFICIAL_SAMPLES_PER_PROMPT)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--operational-horizon", type=float, default=16.0)
    parser.add_argument(
        "--mark-attempts-per-event",
        type=int,
        default=24,
        help=(
            "bounded rejection budget per event. GLOBAL: one value covers every "
            "drug and every task. Exhausting it stops the trajectory, which "
            "shows up as a lost attempt rather than an invalid molecule, so "
            "raising it trades wall clock for sampler efficiency and cannot "
            "change what is chemically admissible."
        ),
    )
    parser.add_argument(
        "--attachment-control",
        action="store_true",
        help=(
            "steer growth to the prompt's DECLARED attachment interfaces. "
            "Default off, so the frozen-sampler baseline rows reproduce."
        ),
    )
    interface_modes = parser.add_mutually_exclusive_group()
    interface_modes.add_argument(
        "--allow-post-coverage-core-growth",
        action="store_true",
        help=(
            "with --attachment-control, release the permanent ban on growth "
            "from undeclared retained-core atoms after all required attachment "
            "interfaces have been covered. Attachment-first and redirection "
            "remain on. Default off preserves the existing controller."
        ),
    )
    interface_modes.add_argument(
        "--release-single-interface-after-coverage",
        action="store_true",
        help=(
            "derive the post-coverage rule only from the declared constraint: "
            "release extra retained-core growth for exactly one interface; "
            "keep permanent restriction for two or more; leave zero-interface "
            "prompts unchanged. Requires --attachment-control."
        ),
    )
    parser.add_argument(
        "--path-program",
        action="store_true",
        help=(
            "drive the core-to-core path length toward a per-trajectory target "
            "drawn from the declared band. GLOBAL: one band covers every drug "
            "and every task, and the program is vacuous unless the prompt "
            "declares two retained regions, which is a property of the "
            "SPECIFICATION and never of an instance. Default off."
        ),
    )
    parser.add_argument(
        "--linker-bridge-atoms",
        type=int,
        default=0,
        help=(
            "seed this many unlocked atoms between the two declared sites of a "
            "two-core prompt, instead of joining the cores directly. GLOBAL, and "
            "vacuous for a single-core prompt. Default 0 keeps the direct join "
            "so every existing row reproduces. The seeded length is not a "
            "result and is recorded in the artifact so no realized-length "
            "number can be read without it."
        ),
    )
    return parser


def control_from_args(args: argparse.Namespace) -> AttachmentControlConfig:
    """ONE controller configuration for every drug and every task.

    Built here rather than inline in ``main`` so the command-line-to-controller
    mapping is a thing a test can drive. A test that reconstructs the config
    itself is recomputing its expectation from the code under test and cannot
    fail when the real call site stops reading the flag.
    """
    if (
        args.allow_post_coverage_core_growth
        or args.release_single_interface_after_coverage
    ) and not args.attachment_control:
        raise ValueError(
            "interface-release mode requires --attachment-control"
        )
    restriction = (
        SINGLE_INTERFACE_RELEASE
        if args.release_single_interface_after_coverage
        else not bool(args.allow_post_coverage_core_growth)
    )
    return AttachmentControlConfig(
        enabled=bool(args.attachment_control),
        restrict_interfaces=restriction,
        path_program=bool(args.path_program),
    )


def run_task_settings(args: argparse.Namespace) -> dict:
    """Everything ``run_task`` takes from the command line, in one place.

    The args-to-run_task hop is a second place a knob can be dropped, and the
    call-site test for the run_task-to-build_prompt_context hop does not cover
    it -- two hops sharing one sink is exactly how a wiring mutation survives a
    single consultation test.
    """
    return {
        "seeds": args.seeds,
        "samples": args.samples,
        "seed_list": args.seed_list,
        "drugs": args.drug,
        "linker_bridge_atoms": args.linker_bridge_atoms,
    }


def protocol_block(args: argparse.Namespace) -> dict:
    """The protocol the rows were produced under, as the artifact records it.

    The seeded bridge lives here rather than in a config hash because it is a
    property of the START STATE and belongs to no dataclass. An artifact that
    omits it cannot be read: a realized core-to-core length means nothing
    without the length the construction supplied, and a reader with only the
    former would take an inherited seed for a designed linker.
    """
    return {
        "source": "in_virtuo_gen/evaluation/downstream.py @ b50bb3ae",
        "samples_per_prompt": args.samples,
        "seeds": args.seeds,
        "task_row": "unweighted mean over the task's drugs",
        "scaffold_morphing": "upstream copies the linker result",
        "linker_bridge_atoms": args.linker_bridge_atoms,
    }


def sampler_config_from_args(args: argparse.Namespace) -> SamplerConfig:
    """ONE sampler configuration for every drug and every task.

    These are GLOBAL knobs. A per-task or per-drug value would be benchmark
    engineering rather than a general capability, so neither this function nor
    the parser offers any way to express one.
    """
    return SamplerConfig(
        max_events=args.max_events,
        operational_horizon=args.operational_horizon,
        mark_attempts_per_event=args.mark_attempts_per_event,
    )


def main() -> None:
    args = build_parser().parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    prompts = load_genmol_prompts(MANIFEST)
    config = sampler_config_from_args(args)
    # ONE controller configuration for every drug and every task.  Nothing here
    # reads a drug name, a task label or any other instance identity.
    control = control_from_args(args)

    selected = (
        [FragmentTask(t) for t in args.task]
        if args.task
        else list(FragmentTask)
    )

    results = {}
    for task in selected:
        print(f"\n=== {task.value} ===", flush=True)
        results[task.value] = run_task(
            model,
            system,
            prompts,
            task,
            config=config,
            control=control,
            **run_task_settings(args),
        )

    payload = {
        "schema": "compose_fragment_official_suite_v1",
        "protocol": protocol_block(args),
        "kernel": _kernel_provenance(),
        "checkpoint": {
            "path": str(args.checkpoint),
            "completed_steps": meta.get("completed_steps"),
            "corpus_scope_hash": meta.get("corpus_scope_hash"),
            "atom_vocabulary": "ORGANIC" if len(model.atom_vocabulary.classes) == 15 else "CNOF",
        },
        "sampler": frozen_sampler_identity(config),
        "attachment_control": frozen_attachment_identity(control),
        "metric_provenance": {
            "validity_uniqueness_quality_diversity": (
                "official in_virtuo_gen.train_utils.metrics.evaluate_smiles "
                "@ b50bb3ae, already_smiles=True"
            ),
            "distance": (
                "NOT in the released evaluator: evaluate_smiles returns no "
                "distance key. Estimator is the official "
                "in_virtuo_gen.utils.mol.calculate_average_tanimoto(prompt=...); "
                "reference = the dummy-stripped prompt fragment(s)"
            ),
            "fragment_preservation": (
                "independent audit, atom-level non-overlapping substructure "
                "match, run over COMMITTED endpoints as well as emitted samples"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
